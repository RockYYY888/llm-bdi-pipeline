"""
HTN method synthesis for Stage 3.

The synthesizer uses LLM output as the only source of compound tasks and methods.
Primitive action tasks are still injected from the domain so Stage 3 can render
and validate executable AgentSpeak wrappers.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from stage1_interpretation.grounding_map import GroundingMap
from stage3_method_synthesis.htn_prompts import (
	build_htn_system_prompt,
	build_htn_user_prompt,
)
from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
	HTNTargetTaskBinding,
)
from utils.hddl_condition_parser import HDDLConditionParser
from utils.negation_mode_resolver import NegationResolution, resolve_negation_modes


class HTNSynthesisError(RuntimeError):
	"""Raised when Stage 3 cannot produce a valid HTN method library."""

	def __init__(
		self,
		message: str,
		*,
		model: Optional[str],
		llm_prompt: Optional[Dict[str, str]],
		llm_response: Optional[str],
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		super().__init__(message)
		self.model = model
		self.llm_prompt = llm_prompt
		self.llm_response = llm_response
		self.metadata = dict(metadata or {})


class HTNMethodSynthesizer:
	"""Build an HTN method library for the current DFA targets."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		model: Optional[str] = None,
		base_url: Optional[str] = None,
		timeout: float = 60.0,
		max_tokens: int = 8192,
		max_attempts: int = 3,
	) -> None:
		self.api_key = api_key
		self.model = model or "deepseek-chat"
		self.base_url = base_url
		self.timeout = timeout
		self.max_tokens = max_tokens
		self.max_attempts = max(1, max_attempts)
		self.parser = HDDLConditionParser()
		self.client = None

		if api_key:
			from openai import OpenAI

			if base_url:
				self.client = OpenAI(api_key=api_key, base_url=base_url)
			else:
				self.client = OpenAI(api_key=api_key)

	def synthesize(
		self,
		domain: Any,
		grounding_map: GroundingMap | Dict[str, Any] | None,
		dfa_result: Dict[str, Any],
		*,
		query_text: Optional[str] = None,
		negation_hints: Optional[Dict[str, Any]] = None,
	) -> Tuple[HTNMethodLibrary, Dict[str, Any]]:
		"""Create a method library and metadata for logging."""

		normalised_grounding_map = self._normalise_grounding_map(grounding_map)
		target_literals = self.extract_target_literals(normalised_grounding_map, dfa_result)
		negation_resolution = resolve_negation_modes(
			domain,
			target_literals,
			query_text=query_text,
			stage1_hints=negation_hints,
		)
		target_literals = [
			negation_resolution.apply(literal)
			for literal in target_literals
		]
		primitive_tasks = self._build_primitive_tasks(domain)

		metadata: Dict[str, Any] = {
			"used_llm": False,
			"model": self.model if self.client else None,
			"target_literals": [literal.to_signature() for literal in target_literals],
			"negation_resolution": negation_resolution.to_dict(),
			"compound_tasks": 0,
			"primitive_tasks": len(primitive_tasks),
			"methods": 0,
			"failure_stage": None,
			"failure_reason": None,
			"llm_prompt": None,
			"llm_response": None,
			"llm_finish_reason": None,
			"llm_attempts": 0,
			"llm_generation_attempts": 0,
			"pruned_constructive_siblings": 0,
			"generated_target_guard_methods": 0,
		}

		if not self.client:
			raise self._build_synthesis_error(
				metadata,
				"preflight",
				(
					"Stage 3 requires a configured OPENAI_API_KEY. "
					"HTN method synthesis only accepts live LLM output."
				),
			)

		prompt = {
			"system": build_htn_system_prompt(),
			"user": build_htn_user_prompt(
				domain,
				metadata["target_literals"],
				self._schema_hint(),
			),
		}
		metadata["llm_prompt"] = prompt

		retry_feedback: Optional[str] = None
		last_validation_error: Optional[Exception] = None
		llm_only_library: Optional[HTNMethodLibrary] = None
		for generation_attempt in range(1, self.max_attempts + 1):
			metadata["llm_generation_attempts"] = generation_attempt
			llm_library, response_text, finish_reason = self._request_complete_llm_library(
				prompt,
				domain,
				metadata,
				repair_instruction=retry_feedback,
			)

			metadata["llm_response"] = response_text
			metadata["llm_finish_reason"] = finish_reason

			candidate_library = HTNMethodLibrary(
				compound_tasks=list(llm_library.compound_tasks),
				primitive_tasks=primitive_tasks,
				methods=list(llm_library.methods),
				target_literals=list(target_literals),
				target_task_bindings=list(llm_library.target_task_bindings),
			)
			candidate_library = self._apply_negation_resolution_to_library(
				candidate_library,
				negation_resolution,
			)
			candidate_library = self._normalise_target_binding_signatures(candidate_library)
			candidate_library, pruned_count = self._prune_redundant_constructive_siblings(
				candidate_library,
				domain,
			)
			metadata["pruned_constructive_siblings"] = pruned_count
			candidate_library, generated_guard_count = self._complete_missing_target_guard_methods(
				candidate_library,
			)
			metadata["generated_target_guard_methods"] = generated_guard_count
			try:
				self._validate_library(candidate_library, domain)
			except Exception as exc:
				last_validation_error = exc
				if generation_attempt >= self.max_attempts:
					break
				retry_feedback = (
					"Your previous JSON failed strict validation. "
					"Regenerate the full JSON object and fix this exact error: "
					f"{str(exc)[:1200]}"
				)
				continue
			llm_only_library = candidate_library
			last_validation_error = None
			break

		if llm_only_library is None:
			raise self._build_synthesis_error(
				metadata,
				"library_validation",
				"LLM HTN library failed validation: "
				f"{last_validation_error}",
			)

		metadata["used_llm"] = True
		metadata["compound_tasks"] = len(llm_only_library.compound_tasks)
		metadata["primitive_tasks"] = len(llm_only_library.primitive_tasks)
		metadata["methods"] = len(llm_only_library.methods)
		return llm_only_library, metadata

	def extract_target_literals(
		self,
		grounding_map: GroundingMap | None,
		dfa_result: Dict[str, Any],
	) -> List[HTNLiteral]:
		"""Read atomic transition labels from the simplified DFA."""

		if grounding_map is None:
			return []

		transition_specs = self.extract_progressing_transitions(grounding_map, dfa_result)
		seen: set[str] = set()
		literals: List[HTNLiteral] = []

		for spec in transition_specs:
			label = spec["label"]
			if label in seen:
				continue
			seen.add(label)
			literals.append(spec["literal"])

		return literals

	def extract_progressing_transitions(
		self,
		grounding_map: GroundingMap | None,
		dfa_result: Dict[str, Any],
	) -> List[Dict[str, Any]]:
		"""Preserve DFA state-to-state progress edges for later rendering."""

		if grounding_map is None:
			return []

		dfa_dot = dfa_result.get("dfa_dot", "")
		graph = self._parse_dfa_graph(dfa_dot)
		selected_edges = self._select_relevant_edges(graph)
		state_aliases = self._build_state_aliases(graph, selected_edges)
		accepting_state_aliases = sorted(
			state_aliases[state]
			for state in graph.get("accepting", set())
			if state in state_aliases
		)
		seen_edges: set[Tuple[str, str, str]] = set()
		transition_specs: List[Dict[str, Any]] = []

		for source, target, raw_label in selected_edges:
			if raw_label in {"true", "false"}:
				continue

			edge_key = (source, target, raw_label)
			if edge_key in seen_edges:
				continue
			seen_edges.add(edge_key)

			literal = self._literal_from_label(raw_label, grounding_map)
			source_alias = state_aliases[source]
			target_alias = state_aliases[target]
			label_signature = literal.to_signature()
			transition_name = (
				f"dfa_step_{source_alias}_{target_alias}_"
				f"{self._transition_suffix_for_literal(literal)}"
			)
			transition_specs.append(
				{
					"transition_name": transition_name,
					"label": label_signature,
					"raw_label": raw_label,
					"literal": literal,
					"source_state": source_alias,
					"target_state": target_alias,
					"raw_source_state": source,
					"raw_target_state": target,
					"initial_state": state_aliases.get(graph.get("init_state")) if graph.get("init_state") else None,
					"accepting_states": accepting_state_aliases,
				},
			)

		return transition_specs

	def _extract_relevant_dfa_labels(self, dfa_dot: str) -> List[str]:
		"""Prefer labels on goal-progressing edges; fall back when progress is not encoded."""

		graph = self._parse_dfa_graph(dfa_dot)
		return [label for _, _, label in self._select_relevant_edges(graph)]

	def _parse_dfa_graph(self, dfa_dot: str) -> Dict[str, Any]:
		accepting: set[str] = set()
		edges: List[Tuple[str, str, str]] = []
		init_state: Optional[str] = None

		for line in dfa_dot.splitlines():
			stripped = line.strip()
			if not stripped:
				continue

			multi_accepting = re.match(
				r'node\s*\[\s*shape\s*=\s*doublecircle\s*\];\s*([^;]+);',
				stripped,
			)
			if multi_accepting:
				accepting.update(re.findall(r'[A-Za-z0-9_]+', multi_accepting.group(1)))
				continue

			single_accepting = re.match(
				r'([A-Za-z0-9_]+)\s*\[\s*shape\s*=\s*doublecircle\s*\];',
				stripped,
			)
			if single_accepting:
				accepting.add(single_accepting.group(1))
				continue

			init_match = re.match(
				r'init\s*->\s*([A-Za-z0-9_]+)\s*;',
				stripped,
			)
			if init_match:
				init_state = init_match.group(1)
				continue

			edge_match = re.match(
				r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*\[label="([^"]+)"\];',
				stripped,
			)
			if edge_match:
				source, target, label = edge_match.groups()
				edges.append((source, target, label))

		return {
			"accepting": accepting,
			"edges": edges,
			"init_state": init_state,
		}

	def _select_relevant_edges(self, graph: Dict[str, Any]) -> List[Tuple[str, str, str]]:
		edges = graph.get("edges", [])
		accepting = graph.get("accepting", set())
		if not edges:
			return []

		progress_edges = self._extract_progressing_edges(edges, accepting)
		if progress_edges:
			return progress_edges

		accepting_loop_edges = self._extract_accepting_loop_edges(edges, accepting)
		if accepting_loop_edges:
			return accepting_loop_edges

		return edges

	def _extract_progressing_edges(
		self,
		edges: List[Tuple[str, str, str]],
		accepting: set[str],
	) -> List[Tuple[str, str, str]]:
		if not edges or not accepting:
			return []

		distances = self._distance_to_accepting(edges, accepting)
		progress_edges: List[Tuple[str, str, str]] = []
		for source, target, label in edges:
			source_distance = distances.get(source)
			target_distance = distances.get(target)
			if source_distance is None or target_distance is None:
				continue
			if target_distance < source_distance:
				progress_edges.append((source, target, label))
		return progress_edges

	def _extract_accepting_loop_edges(
		self,
		edges: List[Tuple[str, str, str]],
		accepting: set[str],
	) -> List[Tuple[str, str, str]]:
		if not edges or not accepting:
			return []

		distances = self._distance_to_accepting(edges, accepting)
		accepting_loop_edges: List[Tuple[str, str, str]] = []
		for source, target, label in edges:
			if distances.get(source) == 0 and distances.get(target) == 0:
				accepting_loop_edges.append((source, target, label))
		return accepting_loop_edges

	def _distance_to_accepting(
		self,
		edges: List[Tuple[str, str, str]],
		accepting: set[str],
	) -> Dict[str, int]:
		reverse_graph: Dict[str, List[str]] = {}
		for source, target, _ in edges:
			reverse_graph.setdefault(target, []).append(source)

		distances: Dict[str, int] = {
			state: 0
			for state in accepting
		}
		queue: deque[str] = deque(accepting)

		while queue:
			state = queue.popleft()
			for predecessor in reverse_graph.get(state, []):
				if predecessor in distances:
					continue
				distances[predecessor] = distances[state] + 1
				queue.append(predecessor)

		return distances

	def _build_state_aliases(
		self,
		graph: Dict[str, Any],
		preferred_edges: Optional[List[Tuple[str, str, str]]] = None,
	) -> Dict[str, str]:
		ordered_states: List[str] = []

		def add(state: Optional[str]) -> None:
			if state and state not in ordered_states:
				ordered_states.append(state)

		add(graph.get("init_state"))
		for source, target, _ in preferred_edges or []:
			add(source)
			add(target)
		for source, target, _ in graph.get("edges", []):
			add(source)
			add(target)
		for state in sorted(graph.get("accepting", set())):
			add(state)

		return {
			state: f"q{index}"
			for index, state in enumerate(ordered_states, start=1)
		}

	def _literal_from_label(self, label: str, grounding_map: GroundingMap) -> HTNLiteral:
		is_positive = not label.startswith("!")
		symbol = label[1:] if not is_positive else label

		atom = grounding_map.get_atom(symbol)
		if atom is None:
			return HTNLiteral(
				predicate=symbol,
				args=(),
				is_positive=is_positive,
				source_symbol=symbol,
			)

		return HTNLiteral(
			predicate=atom.predicate,
			args=tuple(atom.args),
			is_positive=is_positive,
			source_symbol=symbol,
		)

	def _transition_suffix_for_literal(self, literal: HTNLiteral) -> str:
		prefix = "" if literal.is_positive else "not_"
		parts = [literal.predicate, *literal.args]
		return prefix + "_".join(self._sanitize_name(part) for part in parts if part)

	def _build_primitive_tasks(self, domain: Any) -> List[HTNTask]:
		actions = [self.parser.parse_action(action) for action in domain.actions]
		return [
			HTNTask(
				name=self._sanitize_name(action.name),
				parameters=tuple(f"X{index + 1}" for index, _ in enumerate(action.parameters)),
				is_primitive=True,
				source_predicates=tuple(
					sorted({literal.predicate for literal in action.positive_effects}),
				),
			)
			for action in actions
		]

	def _request_complete_llm_library(
		self,
		prompt: Dict[str, str],
		domain: Any,
		metadata: Dict[str, Any],
		repair_instruction: Optional[str] = None,
	) -> Tuple[HTNMethodLibrary, str, Optional[str]]:
		last_error: Optional[Exception] = None
		attempt_durations: List[float] = []
		total_start = time.monotonic()

		for attempt in range(1, self.max_attempts + 1):
			metadata["llm_attempts"] = attempt
			retry_instructions: List[str] = []
			if repair_instruction:
				retry_instructions.append(repair_instruction)
			if attempt > 1:
				retry_instructions.append(
					"Your previous response was incomplete. Return the entire JSON object "
					"again from the beginning. Do not summarize, do not omit any fields, "
					"and do not stop until every brace and bracket is closed."
				)
			retry_instruction = "\n\n".join(retry_instructions) if retry_instructions else None
			attempt_start = time.monotonic()
			try:
				response_text, finish_reason = self._call_llm(
					prompt,
					retry_instruction=retry_instruction,
					max_tokens=self.max_tokens,
				)
			except Exception as exc:
				attempt_durations.append(time.monotonic() - attempt_start)
				metadata["llm_attempt_durations_seconds"] = [
					round(duration, 3) for duration in attempt_durations
				]
				metadata["llm_response_time_seconds"] = round(time.monotonic() - total_start, 3)
				last_error = exc
				if attempt == self.max_attempts:
					raise self._build_synthesis_error(
						metadata,
						"llm_call",
						f"LLM request failed after {attempt} attempts: {exc}",
					) from exc
				continue

			attempt_durations.append(time.monotonic() - attempt_start)
			metadata["llm_attempt_durations_seconds"] = [
				round(duration, 3) for duration in attempt_durations
			]
			metadata["llm_response_time_seconds"] = round(time.monotonic() - total_start, 3)
			metadata["llm_response"] = response_text
			metadata["llm_finish_reason"] = finish_reason

			if finish_reason == "length":
				last_error = ValueError(
					"LLM response was truncated before completion (finish_reason=length).",
				)
				if attempt < self.max_attempts:
					continue
				raise self._build_synthesis_error(
					metadata,
					"response_parse",
					(
						"LLM response was truncated before completion "
						f"after {attempt} attempts (finish_reason=length)."
					),
				)

			try:
				parsed_library = self._parse_llm_library(response_text)
			except Exception as exc:
				last_error = exc
				if self._is_retryable_incomplete_response(exc) and attempt < self.max_attempts:
					continue
				raise self._build_synthesis_error(
					metadata,
					"response_parse",
					f"LLM response could not be parsed as a valid HTN library: {exc}",
				) from exc

			return self._normalise_llm_library(parsed_library, domain), response_text, finish_reason

		if last_error is not None:
			raise self._build_synthesis_error(
				metadata,
				"response_parse",
				(
					"LLM response could not be completed after "
					f"{self.max_attempts} attempts: {last_error}"
				),
			)
		raise self._build_synthesis_error(
			metadata,
			"response_parse",
			"LLM response could not be completed after exhausting the retry budget.",
		)

	def _call_llm(
		self,
		prompt: Dict[str, str],
		*,
		retry_instruction: Optional[str] = None,
		max_tokens: Optional[int] = None,
	) -> Tuple[str, Optional[str]]:
		user_content = prompt["user"]
		if retry_instruction:
			user_content = f"{user_content}\n\nRETRY INSTRUCTION:\n{retry_instruction}"
		response = self.client.chat.completions.create(
			model=self.model,
			messages=[
				{"role": "system", "content": prompt["system"]},
				{"role": "user", "content": user_content},
			],
			temperature=0.0,
			max_tokens=max_tokens or self.max_tokens,
			timeout=self.timeout,
		)
		choice = response.choices[0]
		finish_reason = getattr(choice, "finish_reason", None)
		content = (choice.message.content or "").strip()
		return content, finish_reason

	def _parse_llm_library(self, response_text: str) -> HTNMethodLibrary:
		clean_text = self._strip_code_fences(response_text)
		if self._appears_truncated_json(clean_text):
			raise ValueError(
				"LLM response appears truncated before the JSON object closed. "
				"The HTN library was cut off mid-response.",
			)
		payload = json.loads(clean_text)
		if not isinstance(payload, dict):
			raise ValueError("HTN synthesis response must be a JSON object")
		return HTNMethodLibrary.from_dict(payload)

	@staticmethod
	def _appears_truncated_json(text: str) -> bool:
		open_curly = text.count("{")
		close_curly = text.count("}")
		open_square = text.count("[")
		close_square = text.count("]")
		if open_curly > close_curly:
			return True
		if open_square > close_square:
			return True
		return False

	@staticmethod
	def _is_retryable_incomplete_response(exc: Exception) -> bool:
		message = str(exc).lower()
		return (
			"truncated" in message
			or "unterminated" in message
			or "expecting value" in message
			or "expecting ',' delimiter" in message
		)

	@staticmethod
	def _method_step_semantic_signature(step: HTNMethodStep) -> Tuple[Any, ...]:
		return (
			step.step_id,
			step.task_name,
			step.args,
			step.kind,
			step.action_name,
			step.literal.to_signature() if step.literal else None,
			tuple(literal.to_signature() for literal in step.preconditions),
			tuple(literal.to_signature() for literal in step.effects),
		)

	def _method_semantic_signature(self, method: HTNMethod) -> Tuple[Any, ...]:
		return (
			method.task_name,
			method.parameters,
			tuple(literal.to_signature() for literal in method.context),
			tuple(self._method_step_semantic_signature(step) for step in method.subtasks),
			method.ordering,
		)

	def _normalise_llm_library(
		self,
		library: HTNMethodLibrary,
		domain: Any,
	) -> HTNMethodLibrary:
		action_schemas = self._action_schema_map(domain)
		predicate_arity = {
			predicate.name: len(predicate.parameters)
			for predicate in getattr(domain, "predicates", [])
		}
		compound_tasks = [
			HTNTask(
				name=task.name,
				parameters=task.parameters,
				is_primitive=task.is_primitive,
				source_predicates=tuple(
					predicate_name
					for predicate_name in task.source_predicates
					if predicate_name in predicate_arity
					and predicate_arity[predicate_name] == len(task.parameters)
				),
			)
			for task in library.compound_tasks
		]
		compound_task_names = {task.name for task in compound_tasks}
		primitive_task_names = {self._sanitize_name(action.name) for action in domain.actions}

		methods = []
		for method in library.methods:
			normalised_steps = []
			for step in method.subtasks:
				preconditions = step.preconditions
				effects = step.effects
				action_name = step.action_name
				task_name = step.task_name
				kind = self._coerce_step_kind(
					step,
					primitive_task_names=primitive_task_names,
					compound_task_names=compound_task_names,
				)

				if kind == "primitive":
					action_schema = self._resolve_action_schema(step, action_schemas)
					if action_schema is not None:
						action_name = action_schema.name
						task_name = self._sanitize_name(action_schema.name)
						preconditions = self._materialise_action_literals(
							action_schema.preconditions,
							action_schema.parameters,
							step.args,
						)
						effects = self._materialise_action_literals(
							action_schema.effects,
							action_schema.parameters,
							step.args,
						)
					elif step.action_name is not None:
						action_name = step.action_name.replace("_", "-")
						task_name = self._sanitize_name(step.action_name)

				normalised_steps.append(
					HTNMethodStep(
						step_id=step.step_id,
						task_name=task_name,
						args=step.args,
						kind=kind,
						action_name=action_name,
						literal=step.literal,
						preconditions=preconditions,
						effects=effects,
					),
				)

			methods.append(
				HTNMethod(
					method_name=method.method_name,
					task_name=method.task_name,
					parameters=self._normalise_method_parameters(
						method.parameters,
						method.context,
						tuple(normalised_steps),
					),
					context=method.context,
					subtasks=tuple(normalised_steps),
					ordering=method.ordering,
					origin=method.origin,
				),
			)

		return HTNMethodLibrary(
			compound_tasks=compound_tasks,
			primitive_tasks=list(library.primitive_tasks),
			methods=methods,
			target_literals=list(library.target_literals),
			target_task_bindings=list(library.target_task_bindings),
		)

	def _coerce_step_kind(
		self,
		step: HTNMethodStep,
		*,
		primitive_task_names: set[str],
		compound_task_names: set[str],
	) -> str:
		"""Repair obvious `subtask.kind` mismatches from the LLM output."""
		if step.task_name in compound_task_names:
			return "compound"
		if step.task_name in primitive_task_names:
			return "primitive"
		if step.action_name:
			alias = self._sanitize_name(step.action_name)
			if alias in primitive_task_names and alias not in compound_task_names:
				return "primitive"
		return step.kind

	def _apply_negation_resolution_to_library(
		self,
		library: HTNMethodLibrary,
		negation_resolution: NegationResolution,
	) -> HTNMethodLibrary:
		def apply_literal(literal: HTNLiteral) -> HTNLiteral:
			return negation_resolution.apply(literal)

		def apply_step(step: HTNMethodStep) -> HTNMethodStep:
			return HTNMethodStep(
				step_id=step.step_id,
				task_name=step.task_name,
				args=step.args,
				kind=step.kind,
				action_name=step.action_name,
				literal=apply_literal(step.literal) if step.literal else None,
				preconditions=tuple(apply_literal(item) for item in step.preconditions),
				effects=tuple(apply_literal(item) for item in step.effects),
			)

		def apply_method(method: HTNMethod) -> HTNMethod:
			return HTNMethod(
				method_name=method.method_name,
				task_name=method.task_name,
				parameters=method.parameters,
				context=tuple(apply_literal(item) for item in method.context),
				subtasks=tuple(apply_step(step) for step in method.subtasks),
				ordering=method.ordering,
				origin=method.origin,
			)

		return HTNMethodLibrary(
			compound_tasks=list(library.compound_tasks),
			primitive_tasks=list(library.primitive_tasks),
			methods=[apply_method(method) for method in library.methods],
			target_literals=[apply_literal(item) for item in library.target_literals],
			target_task_bindings=list(library.target_task_bindings),
		)

	def _normalise_target_binding_signatures(
		self,
		library: HTNMethodLibrary,
	) -> HTNMethodLibrary:
		canonical_signatures = {
			literal.to_signature()
			for literal in library.target_literals
		}
		alternate_lookup: Dict[str, str] = {}
		for signature in canonical_signatures:
			alternate_lookup[signature] = signature
			if signature.startswith("!"):
				alternate_lookup[f"~{signature[1:]}"] = signature

		bindings = [
			HTNTargetTaskBinding(
				target_literal=alternate_lookup.get(binding.target_literal, binding.target_literal),
				task_name=binding.task_name,
			)
			for binding in library.target_task_bindings
		]
		return HTNMethodLibrary(
			compound_tasks=list(library.compound_tasks),
			primitive_tasks=list(library.primitive_tasks),
			methods=list(library.methods),
			target_literals=list(library.target_literals),
			target_task_bindings=bindings,
		)

	def _normalise_method_parameters(
		self,
		parameters: Tuple[str, ...],
		context: Tuple[HTNLiteral, ...],
		steps: Tuple[HTNMethodStep, ...],
	) -> Tuple[str, ...]:
		ordered_parameters = list(parameters)
		seen = set(ordered_parameters)

		def consider(symbol: str) -> None:
			if not self._looks_like_variable(symbol):
				return
			if symbol in seen:
				return
			seen.add(symbol)
			ordered_parameters.append(symbol)

		for literal in context:
			for arg in literal.args:
				consider(arg)

		for step in steps:
			for arg in step.args:
				consider(arg)
			if step.literal is not None:
				for arg in step.literal.args:
					consider(arg)
			for literal in (*step.preconditions, *step.effects):
				for arg in literal.args:
					consider(arg)

		return tuple(ordered_parameters)

	def _validate_library(self, library: HTNMethodLibrary, domain: Any) -> None:
		compound_task_names = [task.name for task in library.compound_tasks]
		if len(compound_task_names) != len(set(compound_task_names)):
			raise ValueError("HTN library contains duplicate compound task declarations")

		primitive_task_names = [task.name for task in library.primitive_tasks]
		if len(primitive_task_names) != len(set(primitive_task_names)):
			raise ValueError("HTN library contains duplicate primitive task declarations")

		method_names = [method.method_name for method in library.methods]
		if len(method_names) != len(set(method_names)):
			raise ValueError("HTN library contains duplicate method identifiers")

		semantic_signatures: Dict[Tuple[Any, ...], str] = {}
		for method in library.methods:
			signature = self._method_semantic_signature(method)
			existing_method = semantic_signatures.get(signature)
			if existing_method is not None:
				raise ValueError(
					f"Methods '{existing_method}' and '{method.method_name}' are semantically "
					"duplicate. Do not emit repeated method bodies under different names.",
				)
			semantic_signatures[signature] = method.method_name

		primitive_names = {self._sanitize_name(action.name) for action in domain.actions}
		compound_names = {task.name for task in library.compound_tasks}
		all_tasks = compound_names | {task.name for task in library.primitive_tasks}
		task_lookup = {task.name: task for task in library.compound_tasks}
		action_schemas = self._action_schema_map(domain)
		action_types = self._action_type_map(domain)
		predicate_arities = {
			predicate.name: len(predicate.parameters)
			for predicate in getattr(domain, "predicates", [])
		}
		predicate_types = {
			predicate.name: tuple(
				self._parameter_type(parameter)
				for parameter in predicate.parameters
			)
			for predicate in getattr(domain, "predicates", [])
		}

		method_counts: Dict[str, int] = {}
		for method in library.methods:
			method_counts[method.task_name] = method_counts.get(method.task_name, 0) + 1

		if primitive_names - {task.name for task in library.primitive_tasks}:
			raise ValueError("HTN library is missing primitive action tasks")

		for task in library.compound_tasks + library.primitive_tasks:
			if not re.fullmatch(r"[a-z][a-z0-9_]*", task.name):
				raise ValueError(
					f"Invalid task identifier '{task.name}'. "
					"Task names must match [a-z][a-z0-9_]* for AgentSpeak compatibility.",
				)

		for task in library.compound_tasks:
			for predicate_name in task.source_predicates:
				if predicate_name not in predicate_arities:
					raise ValueError(
						f"Task '{task.name}' references unknown source predicate "
						f"'{predicate_name}'. Known predicates: {sorted(predicate_arities)}",
					)
			if len(task.source_predicates) != 1:
				continue
			predicate_name = task.source_predicates[0]
			predicate_arity = predicate_arities[predicate_name]
			if predicate_arity != len(task.parameters):
				raise ValueError(
					f"Task '{task.name}' source predicate '{predicate_name}' arity mismatch: "
					f"task has {len(task.parameters)} args, predicate has {predicate_arity}.",
				)

		for task in library.compound_tasks:
			if self._is_legacy_task_name(task.name):
				raise ValueError(
					f"Legacy task name '{task.name}' is not allowed. "
					"Use semantic task names such as place_on, hold_block, clear_top, or "
					"keep_door_closed.",
				)

		expected_targets = [literal.to_signature() for literal in library.target_literals]
		binding_lookup: Dict[str, str] = {}
		for binding in library.target_task_bindings:
			if binding.target_literal not in expected_targets:
				raise ValueError(
					"target_task_bindings contains an unknown target literal "
					f"'{binding.target_literal}'. Expected one of: {expected_targets}",
				)
			if binding.target_literal in binding_lookup:
				raise ValueError(
					f"Duplicate target_task_binding for '{binding.target_literal}'. "
					"Each target literal must map to exactly one compound task.",
				)
			if binding.task_name not in compound_names:
				raise ValueError(
					f"target_task_binding for '{binding.target_literal}' references unknown "
					f"compound task '{binding.task_name}'. Known compound tasks: "
					f"{sorted(compound_names)}",
				)
			binding_lookup[binding.target_literal] = binding.task_name

		for target_signature in expected_targets:
			if target_signature not in binding_lookup:
				raise ValueError(
					"HTN library is missing a target_task_binding for target literal "
					f"'{target_signature}'.",
				)

			bound_task_name = binding_lookup[target_signature]
			if method_counts.get(bound_task_name, 0) == 0:
				raise ValueError(
					"HTN library is missing a method for the bound top-level task "
					f"'{bound_task_name}' required for target literal '{target_signature}'.",
				)

		self._validate_target_task_bindings(
			library,
			binding_lookup,
			task_lookup,
			action_schemas,
			predicate_arities,
		)

		for literal in library.target_literals:
			self._validate_literal_shape(
				literal,
				predicate_arities,
				"target_literals",
			)
			if literal.is_positive:
				continue
			bound_task_name = binding_lookup[literal.to_signature()]
			bound_methods = [
				method for method in library.methods if method.task_name == bound_task_name
			]
			if not any(method.subtasks for method in bound_methods):
				raise ValueError(
					"Negative target literal "
					f"'{literal.to_signature()}' is bound to task '{bound_task_name}', "
					"but that task has no constructive non-zero-subtask method. "
					"Negative targets must include at least one method that can make the "
					"predicate false, not only a noop/already-satisfied method.",
				)

		for method in library.methods:
			if not re.fullmatch(r"[a-z][a-z0-9_]*", method.method_name):
				raise ValueError(
					f"Invalid method identifier '{method.method_name}'. "
					"Method names must match [a-z][a-z0-9_]* for AgentSpeak compatibility.",
				)
			if method.task_name not in compound_names:
				raise ValueError(
					f"Unknown compound task in method '{method.method_name}': {method.task_name}. "
					f"Known compound tasks: {sorted(compound_names)}",
				)

			required_prefix = f"m_{method.task_name}_"
			if not method.method_name.startswith(required_prefix):
				raise ValueError(
					f"Method '{method.method_name}' must follow the exact naming pattern "
					f"'m_{method.task_name}_<strategy>'.",
				)

			strategy_suffix = method.method_name[len(required_prefix):]
			if not strategy_suffix:
				raise ValueError(
					f"Method '{method.method_name}' must include a strategy suffix after "
					f"'{required_prefix}'.",
				)

			step_ids = {step.step_id for step in method.subtasks}
			if len(step_ids) != len(method.subtasks):
				raise ValueError(
					f"Duplicate step_id values in method '{method.method_name}'. "
					"Each subtask step_id must be unique.",
				)

			if method.subtasks:
				if strategy_suffix == "noop" or strategy_suffix.startswith("already_"):
					raise ValueError(
						f"Non-empty method '{method.method_name}' cannot use the noop/already_* "
						"strategy naming reserved for already-satisfied cases.",
					)
				if len(method.subtasks) > 1 and not method.ordering:
					raise ValueError(
						f"Method '{method.method_name}' must include explicit ordering edges for "
						"multi-step decompositions.",
					)
			else:
				if method.ordering:
					raise ValueError(
						f"Zero-subtask method '{method.method_name}' must have an empty "
						"ordering list.",
					)
				if not method.context:
					raise ValueError(
						f"Zero-subtask method '{method.method_name}' must have a non-empty "
						"context.",
					)

			for before, after in method.ordering:
				if before not in step_ids or after not in step_ids:
					raise ValueError(
						f"Invalid ordering edge ({before}, {after}) in method "
						f"'{method.method_name}'. Known step_ids: {sorted(step_ids)}",
					)

			for step in method.subtasks:
				if step.kind not in {"primitive", "compound"}:
					raise ValueError(
						f"Unsupported subtask kind '{step.kind}' in method "
						f"'{method.method_name}'. Subtasks must use only 'primitive' or "
						"'compound'.",
					)

				if not re.fullmatch(r"[a-z][a-z0-9_]*", step.task_name):
					raise ValueError(
						f"Invalid subtask identifier '{step.task_name}' in method "
						f"'{method.method_name}'. Subtask names must match [a-z][a-z0-9_]*.",
					)

				if step.task_name in primitive_names and step.kind != "primitive":
					raise ValueError(
						f"Subtask '{step.step_id}' in method '{method.method_name}' uses "
						f"primitive task '{step.task_name}' but marks kind='{step.kind}'. "
						"Primitive aliases must use kind='primitive'.",
					)
				if step.task_name in compound_names and step.kind != "compound":
					raise ValueError(
						f"Subtask '{step.step_id}' in method '{method.method_name}' uses "
						f"compound task '{step.task_name}' but marks kind='{step.kind}'. "
						"Compound helpers must use kind='compound'.",
					)

				if step.kind == "primitive":
					if step.task_name not in primitive_names:
						hint = ""
						if step.action_name is not None:
							alias = self._sanitize_name(step.action_name)
							if alias in primitive_names:
								hint = (
									f" Use runtime alias '{alias}' for task_name and keep "
									f"action_name='{step.action_name}'."
								)
						raise ValueError(
							f"Primitive step '{step.step_id}' in method '{method.method_name}' "
							f"references unknown action task '{step.task_name}'. "
							f"Allowed primitive task names: {sorted(primitive_names)}.{hint}",
						)
					continue

				if step.kind == "compound" and step.task_name not in compound_names:
					raise ValueError(
						f"Compound step '{step.step_id}' in method '{method.method_name}' "
						f"references unknown compound task '{step.task_name}'. "
						f"Known compound tasks: {sorted(compound_names)}",
					)

				if step.task_name not in all_tasks:
					raise ValueError(
						f"Unknown subtask reference '{step.task_name}' in method "
						f"'{method.method_name}'. Known tasks: {sorted(all_tasks)}",
					)

				if step.literal is not None:
					self._validate_literal_shape(
						step.literal,
						predicate_arities,
						f"subtask '{step.step_id}' literal in method '{method.method_name}'",
					)
				for index, literal in enumerate(step.preconditions, start=1):
					self._validate_literal_shape(
						literal,
						predicate_arities,
						f"precondition {index} of subtask '{step.step_id}' in method "
						f"'{method.method_name}'",
					)
				for index, literal in enumerate(step.effects, start=1):
					self._validate_literal_shape(
						literal,
						predicate_arities,
						f"effect {index} of subtask '{step.step_id}' in method "
						f"'{method.method_name}'",
					)

			for index, literal in enumerate(method.context, start=1):
				self._validate_literal_shape(
					literal,
					predicate_arities,
					f"context literal {index} of method '{method.method_name}'",
				)

			self._validate_method_variable_binding(method, task_lookup)
			self._validate_method_variable_types(
				method,
				task_lookup,
				action_types,
				predicate_types,
			)

		self._validate_sibling_method_distinguishability(
			library,
			task_lookup,
			action_schemas,
			predicate_arities,
		)

	def _complete_missing_target_guard_methods(
		self,
		library: HTNMethodLibrary,
	) -> Tuple[HTNMethodLibrary, int]:
		literal_lookup = {
			literal.to_signature(): literal
			for literal in library.target_literals
		}
		task_lookup = {
			task.name: task
			for task in library.compound_tasks
		}
		methods = list(library.methods)
		existing_names = {method.method_name for method in methods}
		generated_count = 0

		for binding in library.target_task_bindings:
			literal = literal_lookup.get(binding.target_literal)
			task = task_lookup.get(binding.task_name)
			if literal is None or task is None:
				continue
			if len(task.parameters) != len(literal.args):
				continue

			expected_literal = HTNLiteral(
				predicate=literal.predicate,
				args=tuple(task.parameters),
				is_positive=literal.is_positive,
				source_symbol=None,
			)
			bound_methods = [
				method
				for method in methods
				if method.task_name == task.name
			]
			if any(
				self._method_context_supports_literal(method, expected_literal)
				for method in bound_methods
			):
				continue

			if literal.is_positive:
				base_name = f"m_{task.name}_already_{literal.predicate}"
			else:
				base_name = f"m_{task.name}_already_not_{literal.predicate}"
			method_name = self._unique_method_name(base_name, existing_names)
			existing_names.add(method_name)
			methods.append(
				HTNMethod(
					method_name=method_name,
					task_name=task.name,
					parameters=tuple(task.parameters),
					context=(expected_literal,),
					subtasks=(),
					ordering=(),
					origin="stage3_target_guard_completion",
				),
			)
			generated_count += 1

		if generated_count == 0:
			return library, 0

		return (
			HTNMethodLibrary(
				compound_tasks=list(library.compound_tasks),
				primitive_tasks=list(library.primitive_tasks),
				methods=methods,
				target_literals=list(library.target_literals),
				target_task_bindings=list(library.target_task_bindings),
			),
			generated_count,
		)

	def _prune_redundant_constructive_siblings(
		self,
		library: HTNMethodLibrary,
		domain: Any,
	) -> Tuple[HTNMethodLibrary, int]:
		task_lookup = {task.name: task for task in library.compound_tasks}
		action_schemas = self._action_schema_map(domain)
		predicate_arities = {
			predicate.name: len(predicate.parameters)
			for predicate in getattr(domain, "predicates", [])
		}

		methods_by_task: Dict[str, List[HTNMethod]] = {}
		for method in library.methods:
			methods_by_task.setdefault(method.task_name, []).append(method)

		allowed_method_names: set[str] = set()
		for task_name, methods in methods_by_task.items():
			guard_methods = [method for method in methods if not method.subtasks]
			constructive_methods = [method for method in methods if method.subtasks]
			candidate_constructive_methods = [
				method
				for method in constructive_methods
				if not self._is_direct_self_recursive_method(method)
			]
			if not candidate_constructive_methods:
				candidate_constructive_methods = constructive_methods
			if len(candidate_constructive_methods) <= 1:
				allowed_method_names.update(method.method_name for method in guard_methods)
				allowed_method_names.update(
					method.method_name for method in candidate_constructive_methods
				)
				continue
			constructive_methods = candidate_constructive_methods

			context_signatures: Dict[str, Tuple[str, ...]] = {}
			has_distinguishing_context = False
			for method in constructive_methods:
				promoted_context = self._promoted_method_context(
					method,
					task_lookup,
					action_schemas,
					predicate_arities,
				)
				signature = tuple(sorted(literal.to_signature() for literal in promoted_context))
				context_signatures[method.method_name] = signature
				if signature:
					has_distinguishing_context = True

			allowed_method_names.update(method.method_name for method in guard_methods)
			seen_signatures: set[Tuple[str, ...]] = set()

			if not has_distinguishing_context:
				allowed_method_names.add(constructive_methods[0].method_name)
				continue

			for method in constructive_methods:
				signature = context_signatures[method.method_name]
				if not signature or signature in seen_signatures:
					continue
				seen_signatures.add(signature)
				allowed_method_names.add(method.method_name)

		pruned_methods = [
			method
			for method in library.methods
			if method.method_name in allowed_method_names
		]
		pruned_count = len(library.methods) - len(pruned_methods)
		if pruned_count == 0:
			return library, 0

		return (
			HTNMethodLibrary(
				compound_tasks=list(library.compound_tasks),
				primitive_tasks=list(library.primitive_tasks),
				methods=pruned_methods,
				target_literals=list(library.target_literals),
				target_task_bindings=list(library.target_task_bindings),
			),
			pruned_count,
		)

	@staticmethod
	def _unique_method_name(base_name: str, existing_names: set[str]) -> str:
		if base_name not in existing_names:
			return base_name

		index = 2
		while True:
			candidate = f"{base_name}_{index}"
			if candidate not in existing_names:
				return candidate
			index += 1

	@staticmethod
	def _is_direct_self_recursive_method(method: HTNMethod) -> bool:
		return any(
			step.kind == "compound" and step.task_name == method.task_name
			for step in method.subtasks
		)

	def _validate_target_task_bindings(
		self,
		library: HTNMethodLibrary,
		binding_lookup: Dict[str, str],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
	) -> None:
		literal_lookup = {
			literal.to_signature(): literal
			for literal in library.target_literals
		}
		methods_by_task: Dict[str, List[HTNMethod]] = {}
		for method in library.methods:
			methods_by_task.setdefault(method.task_name, []).append(method)

		for target_signature, task_name in binding_lookup.items():
			literal = literal_lookup[target_signature]
			task = task_lookup[task_name]
			if len(task.parameters) != len(literal.args):
				raise ValueError(
					f"target_task_binding for '{target_signature}' uses task '{task_name}' "
					f"with {len(task.parameters)} parameters, but the target literal has "
					f"{len(literal.args)} arguments. Target-bound tasks must align with the "
					"literal arity so the runtime can bind them correctly.",
				)

			expected_literal = HTNLiteral(
				predicate=literal.predicate,
				args=tuple(task.parameters),
				is_positive=literal.is_positive,
				source_symbol=None,
			)
			bound_methods = methods_by_task.get(task_name, [])

			if not any(
				self._method_context_supports_literal(method, expected_literal)
				for method in bound_methods
			):
				raise ValueError(
					f"target_task_binding for '{target_signature}' uses task '{task_name}', "
					"but none of that task's methods exposes an already-satisfied context "
					f"for '{expected_literal.to_signature()}'. The bound task must clearly "
					"represent the target literal it is meant to satisfy.",
				)

			if literal.is_positive:
				continue

			constructive_methods = [
				method
				for method in bound_methods
				if method.subtasks
			]
			if not constructive_methods:
				continue

			if not any(
				self._method_constructively_supports_literal(
					method,
					expected_literal,
					methods_by_task,
					task_lookup,
					action_schemas,
					predicate_arities,
				)
				for method in constructive_methods
			):
				raise ValueError(
					f"target_task_binding for negative literal '{target_signature}' uses "
					f"task '{task_name}', but none of its constructive methods makes "
					f"'{expected_literal.to_signature()}' true. Negative target bindings "
					"must point to tasks that can actually remove or prevent the relation, "
					"not tasks that only establish the positive form.",
				)

	def _build_synthesis_error(
		self,
		metadata: Dict[str, Any],
		failure_stage: str,
		failure_reason: str,
	) -> HTNSynthesisError:
		error_metadata = dict(metadata)
		error_metadata["failure_stage"] = failure_stage
		error_metadata["failure_reason"] = failure_reason
		return HTNSynthesisError(
			f"HTN synthesis failed during {failure_stage}: {failure_reason}",
			model=error_metadata.get("model"),
			llm_prompt=error_metadata.get("llm_prompt"),
			llm_response=error_metadata.get("llm_response"),
			metadata=error_metadata,
		)

	def _action_schema_map(self, domain: Any) -> Dict[str, Any]:
		action_schemas: Dict[str, Any] = {}
		for action in domain.actions:
			parsed_action = self.parser.parse_action(action)
			action_schemas[action.name] = parsed_action
			action_schemas[self._sanitize_name(action.name)] = parsed_action
		return action_schemas

	def _action_type_map(self, domain: Any) -> Dict[str, Tuple[str, ...]]:
		action_types: Dict[str, Tuple[str, ...]] = {}
		for action in domain.actions:
			type_signature = tuple(
				self._parameter_type(parameter)
				for parameter in action.parameters
			)
			action_types[action.name] = type_signature
			action_types[self._sanitize_name(action.name)] = type_signature
		return action_types

	def _resolve_action_schema(self, step: HTNMethodStep, action_schemas: Dict[str, Any]) -> Any:
		if step.action_name and step.action_name in action_schemas:
			return action_schemas[step.action_name]
		return action_schemas.get(self._sanitize_name(step.task_name))

	def _materialise_action_literals(
		self,
		patterns: Tuple[Any, ...],
		schema_parameters: Tuple[str, ...],
		step_args: Tuple[str, ...],
	) -> Tuple[HTNLiteral, ...]:
		bindings = {
			parameter: arg
			for parameter, arg in zip(schema_parameters, step_args)
		}
		return tuple(
			HTNLiteral(
				predicate=pattern.predicate,
				args=tuple(bindings.get(arg, arg) for arg in pattern.args),
				is_positive=pattern.is_positive,
				source_symbol=None,
			)
			for pattern in patterns
		)

	@staticmethod
	def _sanitize_name(name: str) -> str:
		return name.replace("-", "_")

	@staticmethod
	def _parameter_type(parameter: str) -> str:
		if "-" not in parameter:
			return "OBJECT"
		return parameter.split("-", 1)[1].strip().upper()

	def _validate_method_variable_binding(
		self,
		method: HTNMethod,
		task_lookup: Dict[str, HTNTask],
	) -> None:
		task_schema = task_lookup.get(method.task_name)
		if task_schema is None:
			bound_variables = set(method.parameters)
		else:
			bound_variables = set(method.parameters[: len(task_schema.parameters)])
		ordered_steps = self._ordered_method_steps(method)
		bound_variables = self._extend_bound_variables_from_literals(
			bound_variables,
			method.context,
		)
		self._ensure_literals_are_bound(
			method,
			method.context,
			bound_variables,
			"context",
		)

		for step in ordered_steps:
			bound_variables = self._extend_bound_variables_from_literals(
				bound_variables,
				step.preconditions,
			)
			for arg in step.args:
				if self._is_unbound_variable(arg, bound_variables):
					raise ValueError(
						f"Method '{method.method_name}' uses unbound variable '{arg}' in "
						f"subtask '{step.step_id}'. Every variable must be introduced by "
						"the method parameters or constrained in method.context.",
					)

			for literal in (step.literal, *step.preconditions, *step.effects):
				if literal is None:
					continue
				self._ensure_literals_are_bound(
					method,
					(literal,),
					bound_variables,
					f"subtask '{step.step_id}'",
					)

	def _validate_method_variable_types(
		self,
		method: HTNMethod,
		task_lookup: Dict[str, HTNTask],
		action_types: Dict[str, Tuple[str, ...]],
		predicate_types: Dict[str, Tuple[str, ...]],
	) -> None:
		symbol_types: Dict[str, set[str]] = {}
		task_schema = task_lookup.get(method.task_name)
		if task_schema and len(task_schema.source_predicates) == 1:
			predicate_signature = predicate_types.get(task_schema.source_predicates[0], ())
			for index, parameter in enumerate(task_schema.parameters):
				if index < len(predicate_signature):
					symbol_types.setdefault(parameter, set()).add(predicate_signature[index])

		for literal in method.context:
			self._collect_literal_types(symbol_types, literal, predicate_types)

		for step in method.subtasks:
			if step.kind == "primitive":
				action_signature = action_types.get(step.action_name or "")
				if not action_signature:
					action_signature = action_types.get(step.task_name, ())
				self._collect_argument_types(symbol_types, step.args, action_signature)
			elif step.kind == "compound":
				step_task = task_lookup.get(step.task_name)
				if step_task and len(step_task.source_predicates) == 1:
					predicate_signature = predicate_types.get(step_task.source_predicates[0], ())
					self._collect_argument_types(symbol_types, step.args, predicate_signature)

			for literal in (step.literal, *step.preconditions, *step.effects):
				if literal is None:
					continue
				self._collect_literal_types(symbol_types, literal, predicate_types)

		for parameter in method.parameters:
			if parameter not in symbol_types:
				raise ValueError(
					f"Method '{method.method_name}' variable '{parameter}' has no type evidence.",
				)

		for symbol, candidates in symbol_types.items():
			if not candidates:
				raise ValueError(
					f"Method '{method.method_name}' symbol '{symbol}' has no type evidence.",
				)
			if len(candidates) > 1:
				raise ValueError(
					f"Method '{method.method_name}' uses symbol '{symbol}' with "
					f"conflicting inferred types {sorted(candidates)}.",
				)

	def _collect_literal_types(
		self,
		variable_types: Dict[str, set[str]],
		literal: HTNLiteral,
		predicate_types: Dict[str, Tuple[str, ...]],
	) -> None:
		if literal.is_equality:
			return
		self._collect_argument_types(
			variable_types,
			literal.args,
			predicate_types.get(literal.predicate, ()),
		)

	def _collect_argument_types(
		self,
		variable_types: Dict[str, set[str]],
		args: Tuple[str, ...],
		signature: Tuple[str, ...],
	) -> None:
		for index, arg in enumerate(args):
			if index >= len(signature):
				continue
			variable_types.setdefault(arg, set()).add(signature[index])

	def _ordered_method_steps(self, method: HTNMethod) -> List[HTNMethodStep]:
		if len(method.subtasks) <= 1 or not method.ordering:
			return list(method.subtasks)

		step_lookup = {
			step.step_id: step
			for step in method.subtasks
		}
		dependents: Dict[str, List[str]] = {
			step.step_id: []
			for step in method.subtasks
		}
		in_degree: Dict[str, int] = {
			step.step_id: 0
			for step in method.subtasks
		}

		for before, after in method.ordering:
			if before not in step_lookup or after not in step_lookup:
				return list(method.subtasks)
			dependents[before].append(after)
			in_degree[after] += 1

		ordered_steps: List[HTNMethodStep] = []
		ready = [
			step.step_id
			for step in method.subtasks
			if in_degree[step.step_id] == 0
		]

		while ready:
			current_id = ready.pop(0)
			ordered_steps.append(step_lookup[current_id])
			for next_id in dependents[current_id]:
				in_degree[next_id] -= 1
				if in_degree[next_id] == 0:
					ready.append(next_id)

		if len(ordered_steps) != len(method.subtasks):
			return list(method.subtasks)

		return ordered_steps

	def _extend_bound_variables_from_literals(
		self,
		bound_variables: set[str],
		literals: Tuple[HTNLiteral, ...],
	) -> set[str]:
		known_variables = set(bound_variables)
		changed = True

		while changed:
			changed = False
			for literal in literals:
				if literal.is_equality:
					continue
				literal_variables = [
					arg
					for arg in self._literal_variables(literal)
					if self._looks_like_variable(arg)
				]
				if not literal_variables:
					continue
				has_anchor = any(
					(not self._looks_like_variable(arg)) or arg in known_variables
					for arg in literal.args
				)
				if not has_anchor:
					continue
				new_variables = [
					arg
					for arg in literal_variables
					if arg not in known_variables
				]
				if not new_variables:
					continue
				known_variables.update(new_variables)
				changed = True

		return known_variables

	def _validate_sibling_method_distinguishability(
		self,
		library: HTNMethodLibrary,
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
	) -> None:
		methods_by_task: Dict[str, List[HTNMethod]] = {}
		for method in library.methods:
			methods_by_task.setdefault(method.task_name, []).append(method)

		for task_name, methods in methods_by_task.items():
			constructive_methods = [method for method in methods if method.subtasks]
			if len(constructive_methods) <= 1:
				continue

			seen_signatures: Dict[Tuple[str, ...], str] = {}
			for method in constructive_methods:
				promoted_context = self._promoted_method_context(
					method,
					task_lookup,
					action_schemas,
					predicate_arities,
				)
				if not promoted_context:
					raise ValueError(
						f"Sibling constructive method '{method.method_name}' for task "
						f"'{task_name}' has no distinguishing context. Constructive sibling "
						"methods must be semantically distinguishable.",
					)

				signature = tuple(sorted(literal.to_signature() for literal in promoted_context))
				if signature in seen_signatures:
					raise ValueError(
						f"Sibling constructive methods '{seen_signatures[signature]}' and "
						f"'{method.method_name}' for task '{task_name}' are not "
						"semantically distinguishable: they produce the same promoted "
						f"context {list(signature)}.",
					)
				seen_signatures[signature] = method.method_name

	@staticmethod
	def _method_context_supports_literal(
		method: HTNMethod,
		expected_literal: HTNLiteral,
	) -> bool:
		expected_signature = expected_literal.to_signature()
		return any(
			literal.to_signature() == expected_signature
			for literal in method.context
		)

	def _method_constructively_supports_literal(
		self,
		method: HTNMethod,
		expected_literal: HTNLiteral,
		methods_by_task: Dict[str, List[HTNMethod]],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		parameter_bindings: Optional[Dict[str, str]] = None,
		visiting: Optional[set[Tuple[str, str]]] = None,
	) -> bool:
		expected_signature = expected_literal.to_signature()
		parameter_bindings = dict(parameter_bindings or {})
		visiting = set() if visiting is None else visiting
		visit_key = (method.method_name, expected_signature)
		if visit_key in visiting:
			return False
		visiting.add(visit_key)

		for step in method.subtasks:
			candidate_literals: List[HTNLiteral] = []
			if step.literal is not None:
				candidate_literals.append(step.literal)
			candidate_literals.extend(
				self._step_effect_literals(
					step,
					task_lookup,
					action_schemas,
					predicate_arities,
				),
			)
			if any(
				self._literal_matches_expected_signature(
					literal,
					expected_literal,
					parameter_bindings,
				)
				for literal in candidate_literals
			):
				visiting.remove(visit_key)
				return True

			if step.kind == "compound":
				child_task = task_lookup.get(step.task_name)
				rebound_step_args = tuple(
					parameter_bindings.get(arg, arg)
					for arg in step.args
				)
				child_bindings = {
					parameter: arg
					for parameter, arg in zip(
						child_task.parameters if child_task is not None else (),
						rebound_step_args,
					)
				}
				for child_method in methods_by_task.get(step.task_name, []):
					if self._method_constructively_supports_literal(
						child_method,
						expected_literal,
						methods_by_task,
						task_lookup,
						action_schemas,
						predicate_arities,
						parameter_bindings=child_bindings,
						visiting=visiting,
					):
						visiting.remove(visit_key)
						return True

		visiting.remove(visit_key)
		return False

	def _promoted_method_context(
		self,
		method: HTNMethod,
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
	) -> Tuple[HTNLiteral, ...]:
		context_literals: List[HTNLiteral] = list(method.context)
		seen_signatures = {
			self._literal_signature(literal)
			for literal in context_literals
		}
		bound_variables = set(method.parameters)
		bound_variables.update(self._extend_bound_variables_from_literals(set(), method.context))
		prior_effect_signatures: set[str] = set()

		for step in self._ordered_method_steps(method):
			for literal in self._step_precondition_literals(step, action_schemas):
				if not literal.is_positive:
					continue
				if self._literal_signature(literal) in prior_effect_signatures:
					continue
				if not literal.args:
					continue
				if not any(arg in method.parameters or arg in bound_variables for arg in literal.args):
					continue
				literal_signature = self._literal_signature(literal)
				if literal_signature not in seen_signatures:
					context_literals.append(literal)
					seen_signatures.add(literal_signature)
				bound_variables = self._extend_bound_variables_from_literals(
					set(bound_variables),
					(literal,),
				)

			for effect in self._step_effect_literals(
				step,
				task_lookup,
				action_schemas,
				predicate_arities,
			):
				prior_effect_signatures.add(self._literal_signature(effect))

		return tuple(context_literals)

	def _step_effect_literals(
		self,
		step: HTNMethodStep,
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
	) -> Tuple[HTNLiteral, ...]:
		if step.kind == "primitive":
			effects = list(step.effects)
			seen_signatures = {
				self._literal_signature(literal)
				for literal in effects
			}
			action_schema = self._resolve_action_schema(step, action_schemas)
			if action_schema is None:
				return tuple(effects)
			for literal in self._materialise_action_literals(
				action_schema.effects,
				action_schema.parameters,
				step.args,
			):
				signature = self._literal_signature(literal)
				if signature in seen_signatures:
					continue
				seen_signatures.add(signature)
				effects.append(literal)
			return tuple(effects)
		if step.effects:
			return tuple(step.effects)

		task_schema = task_lookup.get(step.task_name)
		if not task_schema or len(task_schema.source_predicates) != 1:
			return ()
		predicate_name = task_schema.source_predicates[0]
		if predicate_arities.get(predicate_name) != len(step.args):
			return ()
		return (
			HTNLiteral(
				predicate=predicate_name,
				args=tuple(step.args),
				is_positive=True,
				source_symbol=None,
			),
		)

	def _step_precondition_literals(
		self,
		step: HTNMethodStep,
		action_schemas: Dict[str, Any],
	) -> Tuple[HTNLiteral, ...]:
		if step.kind != "primitive":
			return tuple(step.preconditions)
		action_schema = self._resolve_action_schema(step, action_schemas)
		literals = list(step.preconditions)
		seen_signatures = {
			self._literal_signature(literal)
			for literal in literals
		}
		if action_schema is None:
			return tuple(literals)
		for literal in self._materialise_action_literals(
			action_schema.preconditions,
			action_schema.parameters,
			step.args,
		):
			signature = self._literal_signature(literal)
			if signature in seen_signatures:
				continue
			seen_signatures.add(signature)
			literals.append(literal)
		return tuple(literals)

	@staticmethod
	def _literal_signature(literal: HTNLiteral) -> str:
		return literal.to_signature()

	@classmethod
	def _literal_matches_expected_signature(
		cls,
		candidate: HTNLiteral,
		expected: HTNLiteral,
		parameter_bindings: Dict[str, str],
	) -> bool:
		if candidate.predicate != expected.predicate:
			return False
		if candidate.is_positive != expected.is_positive:
			return False
		if len(candidate.args) != len(expected.args):
			return False

		local_bindings: Dict[str, str] = {}
		for raw_candidate_arg, expected_arg in zip(candidate.args, expected.args):
			is_bound_from_parent = raw_candidate_arg in parameter_bindings
			candidate_arg = parameter_bindings.get(raw_candidate_arg, raw_candidate_arg)
			if is_bound_from_parent:
				if candidate_arg != expected_arg:
					return False
				continue
			if cls._looks_like_variable(candidate_arg):
				bound_value = local_bindings.get(candidate_arg)
				if bound_value is None:
					local_bindings[candidate_arg] = expected_arg
					continue
				if bound_value != expected_arg:
					return False
				continue
			if candidate_arg != expected_arg:
				return False

		return True

	@staticmethod
	def _validate_literal_shape(
		literal: HTNLiteral,
		predicate_arities: Dict[str, int],
		location: str,
	) -> None:
		if literal.is_equality:
			if len(literal.args) != 2:
				raise ValueError(
					f"Invalid equality literal '{literal.to_signature()}' in {location}. "
					"Equality and disequality constraints must use exactly two arguments.",
				)
			return

		expected_arity = predicate_arities.get(literal.predicate)
		if expected_arity is None:
			raise ValueError(
				f"Unknown predicate '{literal.predicate}' in {location}. "
				"Only declared domain predicates or '=' constraints are allowed.",
			)
		if len(literal.args) != expected_arity:
			raise ValueError(
				f"Predicate '{literal.predicate}' in {location} uses {len(literal.args)} "
				f"arguments, but the domain arity is {expected_arity}.",
			)

	def _ensure_literals_are_bound(
		self,
		method: HTNMethod,
		literals: Tuple[HTNLiteral, ...],
		bound_variables: set[str],
		location: str,
	) -> None:
		for literal in literals:
			for arg in self._literal_variables(literal):
				if self._is_unbound_variable(arg, bound_variables):
					raise ValueError(
						f"Method '{method.method_name}' uses unbound variable '{arg}' in "
						f"{location}. Every variable must be introduced by the method "
						"parameters or explicitly constrained by context/preconditions.",
					)

	@staticmethod
	def _literal_variables(literal: HTNLiteral) -> Tuple[str, ...]:
		return tuple(literal.args)

	@staticmethod
	def _is_unbound_variable(symbol: str, bound_variables: set[str]) -> bool:
		return (
			bool(symbol)
			and symbol not in bound_variables
			and HTNMethodSynthesizer._looks_like_variable(symbol)
		)

	@staticmethod
	def _looks_like_variable(symbol: str) -> bool:
		return bool(symbol) and symbol[0].isupper()

	@classmethod
	def _is_legacy_task_name(cls, task_name: str) -> bool:
		return task_name.startswith(("achieve_", "maintain_not_", "ensure_", "goal_"))

	@staticmethod
	def _schema_hint() -> str:
		return json.dumps(
			{
				"target_task_bindings": [
					{
						"target_literal": "delivered(X1)",
						"task_name": "deliver_parcel",
					},
				],
				"compound_tasks": [
					{
						"name": "deliver_parcel",
						"parameters": ["X1"],
						"is_primitive": False,
						"source_predicates": ["delivered"],
					},
					{
						"name": "load_parcel",
						"parameters": ["X1"],
						"is_primitive": False,
						"source_predicates": ["loaded"],
					},
				],
				"methods": [
					{
						"method_name": "m_deliver_parcel_drop",
						"task_name": "deliver_parcel",
						"parameters": ["X1"],
						"context": [],
						"subtasks": [
							{
								"step_id": "s1",
								"task_name": "load_parcel",
								"args": ["X1"],
								"kind": "compound",
								"action_name": None,
								"literal": {
									"predicate": "loaded",
									"args": ["X1"],
									"is_positive": True,
									"negation_mode": "naf",
									"source_symbol": None,
								},
								"preconditions": [],
								"effects": [],
							},
							{
								"step_id": "s2",
								"task_name": "drop_parcel",
								"args": ["X1"],
								"kind": "primitive",
								"action_name": "drop-parcel",
								"literal": {
									"predicate": "delivered",
									"args": ["X1"],
									"is_positive": True,
									"negation_mode": "naf",
									"source_symbol": None,
								},
								"preconditions": [],
								"effects": [],
							},
						],
						"ordering": [["s1", "s2"]],
						"origin": "llm",
					},
				],
			},
			indent=2,
		)

	@staticmethod
	def _strip_code_fences(text: str) -> str:
		if not text.startswith("```"):
			return text
		first_newline = text.find("\n")
		closing_fence = text.rfind("```")
		if first_newline == -1 or closing_fence <= first_newline:
			return text
		return text[first_newline + 1:closing_fence].strip()

	@staticmethod
	def _normalise_grounding_map(
		grounding_map: GroundingMap | Dict[str, Any] | None,
	) -> GroundingMap | None:
		if grounding_map is None:
			return None
		if isinstance(grounding_map, GroundingMap):
			return grounding_map
		if isinstance(grounding_map, dict):
			return GroundingMap.from_dict(grounding_map)
		return None
