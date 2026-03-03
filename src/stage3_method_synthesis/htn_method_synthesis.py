"""
HTN method synthesis for Stage 3.

The synthesizer uses LLM output as the only source of compound tasks and methods.
Primitive action tasks are still injected from the domain so Stage 3 can render
and validate executable AgentSpeak wrappers.
"""

from __future__ import annotations

import json
import re
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
)
from utils.hddl_condition_parser import HDDLConditionParser


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
	) -> None:
		self.api_key = api_key
		self.model = model or "deepseek-chat"
		self.base_url = base_url
		self.timeout = timeout
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
	) -> Tuple[HTNMethodLibrary, Dict[str, Any]]:
		"""Create a method library and metadata for logging."""

		normalised_grounding_map = self._normalise_grounding_map(grounding_map)
		target_literals = self.extract_target_literals(normalised_grounding_map, dfa_result)
		primitive_tasks = self._build_primitive_tasks(domain)

		metadata: Dict[str, Any] = {
			"used_llm": False,
			"model": self.model if self.client else None,
			"target_literals": [literal.to_signature() for literal in target_literals],
			"compound_tasks": 0,
			"primitive_tasks": len(primitive_tasks),
			"methods": 0,
			"failure_stage": None,
			"failure_reason": None,
			"llm_prompt": None,
			"llm_response": None,
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

		try:
			response_text = self._call_llm(prompt)
		except Exception as exc:
			raise self._build_synthesis_error(
				metadata,
				"llm_call",
				f"LLM request failed: {exc}",
			) from exc

		metadata["llm_response"] = response_text

		try:
			llm_library = self._normalise_llm_library(
				self._parse_llm_library(response_text),
				domain,
			)
		except Exception as exc:
			raise self._build_synthesis_error(
				metadata,
				"response_parse",
				f"LLM response could not be parsed as a valid HTN library: {exc}",
			) from exc

		llm_only_library = HTNMethodLibrary(
			compound_tasks=list(llm_library.compound_tasks),
			primitive_tasks=primitive_tasks,
			methods=list(llm_library.methods),
			target_literals=list(target_literals),
			target_task_bindings=list(llm_library.target_task_bindings),
		)
		try:
			self._validate_library(llm_only_library, domain)
		except Exception as exc:
			raise self._build_synthesis_error(
				metadata,
				"library_validation",
				f"LLM HTN library failed validation: {exc}",
			) from exc

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

	def _call_llm(self, prompt: Dict[str, str]) -> str:
		response = self.client.chat.completions.create(
			model=self.model,
			messages=[
				{"role": "system", "content": prompt["system"]},
				{"role": "user", "content": prompt["user"]},
			],
			temperature=0.0,
			timeout=self.timeout,
		)
		return response.choices[0].message.content.strip()

	def _parse_llm_library(self, response_text: str) -> HTNMethodLibrary:
		clean_text = self._strip_code_fences(response_text)
		payload = json.loads(clean_text)
		if not isinstance(payload, dict):
			raise ValueError("HTN synthesis response must be a JSON object")
		return HTNMethodLibrary.from_dict(payload)

	def _normalise_llm_library(
		self,
		library: HTNMethodLibrary,
		domain: Any,
	) -> HTNMethodLibrary:
		action_schemas = self._action_schema_map(domain)
		compound_tasks = [
			HTNTask(
				name=task.name,
				parameters=task.parameters,
				is_primitive=task.is_primitive,
				source_predicates=task.source_predicates,
			)
			for task in library.compound_tasks
		]

		methods = []
		for method in library.methods:
			normalised_steps = []
			for step in method.subtasks:
				preconditions = step.preconditions
				effects = step.effects
				action_name = step.action_name

				if step.kind == "primitive":
					action_schema = self._resolve_action_schema(step, action_schemas)
					if action_schema is not None:
						action_name = action_schema.name
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

				normalised_steps.append(
					HTNMethodStep(
						step_id=step.step_id,
						task_name=step.task_name,
						args=step.args,
						kind=step.kind,
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
					parameters=method.parameters,
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

	def _validate_library(self, library: HTNMethodLibrary, domain: Any) -> None:
		primitive_names = {self._sanitize_name(action.name) for action in domain.actions}
		compound_names = {task.name for task in library.compound_tasks}
		all_tasks = compound_names | {task.name for task in library.primitive_tasks}

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

		for literal in library.target_literals:
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
