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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from stage1_interpretation.grounding_map import GroundingMap
from stage3_method_synthesis.htn_prompts import (
	build_prompt_analysis_payload,
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
	) -> None:
		self.api_key = api_key
		self.model = model or "deepseek-chat"
		self.base_url = base_url
		self.timeout = timeout
		self.max_tokens = max_tokens
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
		query_task_anchors: Optional[Sequence[Dict[str, Any]]] = None,
		semantic_objects: Optional[Sequence[str]] = None,
		query_object_inventory: Optional[Sequence[Dict[str, Any]]] = None,
		query_objects: Optional[Sequence[str]] = None,
		derived_analysis: Optional[Dict[str, Any]] = None,
		negation_hints: Optional[Dict[str, Any]] = None,
		ordered_literal_signatures: Optional[Sequence[str]] = None,
		linearise_ordered_literals: bool = True,
	) -> Tuple[HTNMethodLibrary, Dict[str, Any]]:
		"""Create a method library and metadata for logging."""

		normalised_grounding_map = self._normalise_grounding_map(grounding_map)
		target_literals = self.extract_target_literals(
			normalised_grounding_map,
			dfa_result,
			ordered_literal_signatures=ordered_literal_signatures,
			linearise_ordered_literals=linearise_ordered_literals,
		)
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
		action_analysis = self._analyse_domain_actions(domain)
		primitive_tasks = self._build_primitive_tasks(domain)
		normalised_query_task_anchors = self._normalise_query_task_anchors(
			query_task_anchors,
			domain,
		)
		normalised_semantic_objects = tuple(
			str(value).strip()
			for value in (semantic_objects or ())
			if str(value).strip()
		)
		normalised_query_object_inventory = self._normalise_query_object_inventory(
			query_object_inventory,
		)
		normalised_query_objects = tuple(
			str(value).strip()
			for value in (
				query_objects
				or self._query_object_names_from_inventory(normalised_query_object_inventory)
			)
			if str(value).strip()
		)
		prompt_analysis = dict(
			derived_analysis
			or build_prompt_analysis_payload(
				domain,
				target_literals=[literal.to_signature() for literal in target_literals],
				query_task_anchors=normalised_query_task_anchors,
				action_analysis=action_analysis,
			),
		)

		metadata: Dict[str, Any] = {
			"used_llm": False,
			"model": self.model if self.client else None,
			"target_literals": [literal.to_signature() for literal in target_literals],
			"query_task_anchors": list(normalised_query_task_anchors),
			"semantic_objects": list(normalised_semantic_objects),
			"query_object_inventory": list(normalised_query_object_inventory),
			"query_objects": list(normalised_query_objects),
			"negation_resolution": negation_resolution.to_dict(),
			"action_analysis": action_analysis,
			"derived_analysis": prompt_analysis,
			"compound_tasks": 0,
			"primitive_tasks": len(primitive_tasks),
			"methods": 0,
			"failure_stage": None,
			"failure_reason": None,
			"failure_class": None,
			"llm_prompt": None,
			"llm_response": None,
			"llm_finish_reason": None,
			"llm_attempts": 0,
			"llm_generation_attempts": 0,
			"pruned_constructive_siblings": 0,
			"pruned_unreachable_structures": 0,
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
				query_text=query_text or "",
				query_task_anchors=normalised_query_task_anchors,
				semantic_objects=normalised_semantic_objects,
				query_object_inventory=normalised_query_object_inventory,
				query_objects=normalised_query_objects,
				action_analysis=action_analysis,
				derived_analysis=prompt_analysis,
			),
		}
		metadata["llm_prompt"] = prompt

		metadata["llm_generation_attempts"] = 1
		llm_library, response_text, finish_reason = self._request_complete_llm_library(
			prompt,
			domain,
			metadata,
		)

		metadata["llm_response"] = response_text
		metadata["llm_finish_reason"] = finish_reason

		llm_only_library = HTNMethodLibrary(
			compound_tasks=list(llm_library.compound_tasks),
			primitive_tasks=primitive_tasks,
			methods=list(llm_library.methods),
			target_literals=list(target_literals),
			target_task_bindings=list(llm_library.target_task_bindings),
		)
		llm_only_library = self._apply_negation_resolution_to_library(
			llm_only_library,
			negation_resolution,
		)
		llm_only_library = self._normalise_target_binding_signatures(llm_only_library)
		llm_only_library, pruned_count = self._prune_redundant_constructive_siblings(
			llm_only_library,
			domain,
		)
		metadata["pruned_constructive_siblings"] = pruned_count
		llm_only_library, unreachable_pruned_count = self._prune_unreachable_task_structures(
			llm_only_library,
		)
		metadata["pruned_unreachable_structures"] = unreachable_pruned_count
		try:
			self._validate_library(
				llm_only_library,
				domain,
				query_task_anchors=normalised_query_task_anchors,
				query_objects=normalised_query_objects,
				static_predicates=tuple(action_analysis["static_predicates"]),
			)
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
		*,
		ordered_literal_signatures: Optional[Sequence[str]] = None,
		linearise_ordered_literals: bool = True,
	) -> List[HTNLiteral]:
		"""Read atomic transition labels from the simplified DFA."""

		if grounding_map is None:
			return []

		transition_specs = self.extract_progressing_transitions(
			grounding_map,
			dfa_result,
			ordered_literal_signatures=ordered_literal_signatures,
			linearise_ordered_literals=linearise_ordered_literals,
		)
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
		*,
		ordered_literal_signatures: Optional[Sequence[str]] = None,
		linearise_ordered_literals: bool = True,
	) -> List[Dict[str, Any]]:
		"""Preserve DFA state-to-state progress edges for later rendering."""

		if grounding_map is None:
			return []

		if not linearise_ordered_literals:
			unordered_specs = self._build_unordered_transition_specs_for_literal_occurrences(
				ordered_literal_signatures,
			)
			if unordered_specs is not None:
				return unordered_specs

		if linearise_ordered_literals:
			linearised_specs = self._linearise_transition_specs_for_literal_order(
				ordered_literal_signatures,
			)
			if linearised_specs is not None:
				return linearised_specs

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

		return self._reorder_transition_specs_for_literal_order(
			transition_specs,
			ordered_literal_signatures if linearise_ordered_literals else None,
		)

	def _build_unordered_transition_specs_for_literal_occurrences(
		self,
		ordered_literal_signatures: Optional[Sequence[str]],
	) -> Optional[List[Dict[str, Any]]]:
		if not ordered_literal_signatures:
			return None

		ordered_occurrences = [label for label in ordered_literal_signatures if str(label).strip()]
		if not ordered_occurrences:
			return None

		transition_specs: List[Dict[str, Any]] = []
		for index, signature in enumerate(ordered_occurrences, start=1):
			literal = self._literal_from_signature_text(signature)
			if literal is None:
				return None
			suffix = self._transition_suffix_for_literal(literal)
			transition_specs.append(
				{
					"transition_name": f"dfa_step_q1_q1_t{index}_{suffix}",
					"label": literal.to_signature(),
					"raw_label": str(signature).strip(),
					"literal": literal,
					"source_state": "q1",
					"target_state": "q1",
					"raw_source_state": "q1",
					"raw_target_state": "q1",
					"initial_state": "q1",
					"accepting_states": ["q1"],
				},
			)

		return transition_specs

	def _linearise_transition_specs_for_literal_order(
		self,
		ordered_literal_signatures: Optional[Sequence[str]],
	) -> Optional[List[Dict[str, Any]]]:
		if not ordered_literal_signatures:
			return None

		ordered_occurrences = [label for label in ordered_literal_signatures if str(label).strip()]
		if not ordered_occurrences:
			return None

		accepting_state = f"q{len(ordered_occurrences) + 1}"
		transition_specs: List[Dict[str, Any]] = []
		for index, signature in enumerate(ordered_occurrences, start=1):
			literal = self._literal_from_signature_text(signature)
			if literal is None:
				return None
			source_state = f"q{index}"
			target_state = f"q{index + 1}"
			transition_specs.append(
				{
					"transition_name": (
						f"dfa_step_{source_state}_{target_state}_"
						f"{self._transition_suffix_for_literal(literal)}"
					),
					"label": literal.to_signature(),
					"raw_label": str(signature).strip(),
					"literal": literal,
					"source_state": source_state,
					"target_state": target_state,
					"raw_source_state": source_state,
					"raw_target_state": target_state,
					"initial_state": "q1",
					"accepting_states": [accepting_state],
				},
			)

		return transition_specs

	def _reorder_transition_specs_for_literal_order(
		self,
		transition_specs: List[Dict[str, Any]],
		ordered_literal_signatures: Optional[Sequence[str]],
	) -> List[Dict[str, Any]]:
		if not transition_specs or not ordered_literal_signatures:
			return transition_specs

		ordered_occurrences = [label for label in ordered_literal_signatures if str(label).strip()]
		if not ordered_occurrences:
			return transition_specs

		current_labels = [spec["label"] for spec in transition_specs]
		label_to_spec = {}
		for spec in transition_specs:
			label_to_spec.setdefault(spec["label"], spec)
		if not all(label in label_to_spec for label in ordered_occurrences):
			return transition_specs
		if ordered_occurrences == current_labels and len(current_labels) == len(ordered_occurrences):
			return transition_specs

		accepting_state = f"q{len(ordered_occurrences) + 1}"
		reordered_specs: List[Dict[str, Any]] = []
		for index, label in enumerate(ordered_occurrences, start=1):
			spec = dict(label_to_spec[label])
			source_state = f"q{index}"
			target_state = f"q{index + 1}"
			literal = spec["literal"]
			spec.update(
				{
					"transition_name": (
						f"dfa_step_{source_state}_{target_state}_"
						f"{self._transition_suffix_for_literal(literal)}"
					),
					"source_state": source_state,
					"target_state": target_state,
					"raw_source_state": source_state,
					"raw_target_state": target_state,
					"initial_state": "q1",
					"accepting_states": [accepting_state],
				},
			)
			reordered_specs.append(spec)

		return reordered_specs

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

	@staticmethod
	def _literal_from_signature_text(signature: str) -> Optional[HTNLiteral]:
		token = str(signature or "").strip()
		if not token:
			return None
		is_positive = not token.startswith("!")
		if not is_positive:
			token = token[1:].strip()
		predicate, has_args, args_text = token.partition("(")
		predicate = predicate.strip()
		if not predicate:
			return None
		args: Tuple[str, ...] = ()
		if has_args:
			args = tuple(
				str(arg).strip()
				for arg in args_text.rstrip(")").split(",")
				if str(arg).strip()
			)
		return HTNLiteral(
			predicate=predicate,
			args=args,
			is_positive=is_positive,
			source_symbol=None,
		)

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
				source_name=action.name,
			)
			for action in actions
		]

	def _analyse_domain_actions(self, domain: Any) -> Dict[str, Any]:
		def literal_signature(pattern: Any) -> str:
			atom = (
				pattern.predicate
				if not pattern.args
				else f"{pattern.predicate}({', '.join(pattern.args)})"
			)
			return atom if pattern.is_positive else f"not {atom}"

		dynamic_predicates: set[str] = set()
		producer_actions_by_predicate: Dict[str, List[str]] = {}
		producer_patterns_by_predicate: Dict[str, List[Dict[str, Any]]] = {}
		consumer_actions_by_predicate: Dict[str, List[str]] = {}
		consumer_patterns_by_predicate: Dict[str, List[Dict[str, Any]]] = {}
		parsed_actions = []

		for action in domain.actions:
			parsed_action = self.parser.parse_action(action)
			parsed_actions.append(parsed_action)
			for effect in parsed_action.effects:
				if effect.predicate == "=":
					continue
				dynamic_predicates.add(effect.predicate)

		for action, parsed_action in zip(domain.actions, parsed_actions):
			action_name = self._sanitize_name(parsed_action.name)
			action_parameter_types = [
				self._parameter_type(parameter)
				for parameter in action.parameters
			]
			precondition_signatures = [
				literal_signature(pattern)
				for pattern in parsed_action.preconditions
				if pattern.predicate != "="
			]
			positive_effect_signatures = [
				literal_signature(pattern)
				for pattern in parsed_action.effects
				if pattern.predicate != "=" and pattern.is_positive
			]
			negative_effect_signatures = [
				literal_signature(pattern)
				for pattern in parsed_action.effects
				if pattern.predicate != "=" and not pattern.is_positive
			]
			dynamic_precondition_signatures = [
				literal_signature(pattern)
				for pattern in parsed_action.preconditions
				if pattern.predicate != "=" and pattern.predicate in dynamic_predicates
			]
			for effect in parsed_action.effects:
				if effect.predicate == "=" or not effect.is_positive:
					continue
				producer_actions_by_predicate.setdefault(effect.predicate, []).append(action_name)
				producer_patterns_by_predicate.setdefault(effect.predicate, []).append(
					{
						"action_name": action_name,
						"source_action_name": parsed_action.name,
						"action_parameters": list(parsed_action.parameters),
						"action_parameter_types": list(action_parameter_types),
						"effect_args": list(effect.args),
						"effect_signature": literal_signature(effect),
						"precondition_signatures": list(precondition_signatures),
						"dynamic_precondition_signatures": list(dynamic_precondition_signatures),
						"positive_effect_signatures": list(positive_effect_signatures),
						"negative_effect_signatures": list(negative_effect_signatures),
					},
				)
			for precondition in parsed_action.preconditions:
				if precondition.predicate == "=":
					continue
				if precondition.predicate not in dynamic_predicates or not precondition.is_positive:
					continue
				consumer_actions_by_predicate.setdefault(precondition.predicate, []).append(action_name)
				other_dynamic_precondition_signatures = [
					literal_signature(pattern)
					for pattern in parsed_action.preconditions
					if pattern.predicate != "="
					and pattern.predicate in dynamic_predicates
					and pattern.is_positive
					and pattern != precondition
				]
				consumer_patterns_by_predicate.setdefault(precondition.predicate, []).append(
					{
						"action_name": action_name,
						"source_action_name": parsed_action.name,
						"action_parameters": list(parsed_action.parameters),
						"action_parameter_types": list(action_parameter_types),
						"precondition_args": list(precondition.args),
						"precondition_signature": literal_signature(precondition),
						"other_dynamic_precondition_signatures": list(
							other_dynamic_precondition_signatures,
						),
						"positive_effect_signatures": list(positive_effect_signatures),
						"negative_effect_signatures": list(negative_effect_signatures),
					},
				)

		for predicate_name, producer_actions in list(producer_actions_by_predicate.items()):
			producer_actions_by_predicate[predicate_name] = sorted(dict.fromkeys(producer_actions))
		for predicate_name, consumer_actions in list(consumer_actions_by_predicate.items()):
			consumer_actions_by_predicate[predicate_name] = sorted(dict.fromkeys(consumer_actions))
		for predicate_name, patterns in list(producer_patterns_by_predicate.items()):
			producer_patterns_by_predicate[predicate_name] = sorted(
				patterns,
				key=lambda item: (
					item["action_name"],
					item["effect_signature"],
				),
			)
		for predicate_name, patterns in list(consumer_patterns_by_predicate.items()):
			consumer_patterns_by_predicate[predicate_name] = sorted(
				patterns,
				key=lambda item: (
					item["action_name"],
					item["precondition_signature"],
				),
			)

		all_predicates = {
			predicate.name
			for predicate in getattr(domain, "predicates", [])
		}
		return {
			"dynamic_predicates": sorted(dynamic_predicates),
			"static_predicates": sorted(all_predicates - dynamic_predicates),
			"producer_actions_by_predicate": {
				predicate_name: producer_actions_by_predicate.get(predicate_name, [])
				for predicate_name in sorted(dynamic_predicates)
			},
			"producer_patterns_by_predicate": {
				predicate_name: producer_patterns_by_predicate.get(predicate_name, [])
				for predicate_name in sorted(dynamic_predicates)
			},
			"consumer_actions_by_predicate": {
				predicate_name: consumer_actions_by_predicate.get(predicate_name, [])
				for predicate_name in sorted(dynamic_predicates)
			},
			"consumer_patterns_by_predicate": {
				predicate_name: consumer_patterns_by_predicate.get(predicate_name, [])
				for predicate_name in sorted(dynamic_predicates)
			},
		}

	def _normalise_query_task_anchors(
		self,
		query_task_anchors: Optional[Sequence[Dict[str, Any]]],
		domain: Any,
	) -> Tuple[Dict[str, Any], ...]:
		if not query_task_anchors:
			return ()

		raw_to_alias, alias_to_raw = self._declared_task_alias_maps(domain)
		anchors: List[Dict[str, Any]] = []
		for item in query_task_anchors:
			raw_task_name = str(item.get("task_name", "")).strip()
			task_name, source_name = self._normalise_declared_task_identifier(
				raw_task_name,
				raw_to_alias=raw_to_alias,
				alias_to_raw=alias_to_raw,
			)
			if not task_name:
				continue
			args = tuple(
				str(value).strip()
				for value in (item.get("args") or [])
				if str(value).strip()
			)
			anchors.append({
				"task_name": task_name,
				"source_name": source_name or raw_task_name,
				"args": list(args),
			})
		return tuple(anchors)

	@staticmethod
	def _normalise_query_object_inventory(
		query_object_inventory: Optional[Sequence[Dict[str, Any]]],
	) -> Tuple[Dict[str, Any], ...]:
		if not query_object_inventory:
			return ()

		entries: List[Dict[str, Any]] = []
		for entry in query_object_inventory:
			type_name = str(entry.get("type", "")).strip() or "object"
			label = str(entry.get("label", "")).strip() or type_name
			objects = [
				str(item).strip()
				for item in (entry.get("objects") or [])
				if str(item).strip()
			]
			if not objects:
				continue
			entries.append(
				{
					"type": type_name,
					"label": label,
					"objects": objects,
				},
			)
		return tuple(entries)

	@staticmethod
	def _query_object_names_from_inventory(
		query_object_inventory: Sequence[Dict[str, Any]],
	) -> Tuple[str, ...]:
		ordered_names: List[str] = []
		seen: set[str] = set()
		for entry in query_object_inventory:
			for item in entry.get("objects", []):
				object_name = str(item).strip()
				if not object_name or object_name in seen:
					continue
				seen.add(object_name)
				ordered_names.append(object_name)
		return tuple(ordered_names)

	def _request_complete_llm_library(
		self,
		prompt: Dict[str, str],
		domain: Any,
		metadata: Dict[str, Any],
	) -> Tuple[HTNMethodLibrary, str, Optional[str]]:
		total_start = time.monotonic()
		metadata["llm_attempts"] = 1
		attempt_start = time.monotonic()
		try:
			response_text, finish_reason = self._call_llm(
				prompt,
				max_tokens=self.max_tokens,
			)
		except Exception as exc:
			metadata["llm_attempt_durations_seconds"] = [
				round(time.monotonic() - attempt_start, 3),
			]
			metadata["llm_response_time_seconds"] = round(time.monotonic() - total_start, 3)
			raise self._build_synthesis_error(
				metadata,
				"llm_call",
				f"LLM request failed: {exc}",
			) from exc

		metadata["llm_attempt_durations_seconds"] = [
			round(time.monotonic() - attempt_start, 3),
		]
		metadata["llm_response_time_seconds"] = round(time.monotonic() - total_start, 3)
		metadata["llm_response"] = response_text
		metadata["llm_finish_reason"] = finish_reason

		if finish_reason == "length":
			raise self._build_synthesis_error(
				metadata,
				"response_parse",
				"LLM response was truncated before completion (finish_reason=length).",
			)

		try:
			parsed_library = self._parse_llm_library(response_text)
		except Exception as exc:
			raise self._build_synthesis_error(
				metadata,
				"response_parse",
				f"LLM response could not be parsed as a valid HTN library: {exc}",
			) from exc

		return self._normalise_llm_library(parsed_library, domain), response_text, finish_reason

	def _call_llm(
		self,
		prompt: Dict[str, str],
		*,
		max_tokens: Optional[int] = None,
	) -> Tuple[str, Optional[str]]:
		response = self.client.chat.completions.create(
			model=self.model,
			messages=[
				{"role": "system", "content": prompt["system"]},
				{"role": "user", "content": prompt["user"]},
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
		raw_task_to_alias, alias_to_raw_task = self._declared_task_alias_maps(domain)
		predicate_arity = {
			predicate.name: len(predicate.parameters)
			for predicate in getattr(domain, "predicates", [])
		}
		compound_tasks = []
		compound_alias_lookup: Dict[str, str] = {}
		for task in library.compound_tasks:
			normalised_task_name, source_task_name = self._normalise_declared_task_identifier(
				task.name,
				raw_to_alias=raw_task_to_alias,
				alias_to_raw=alias_to_raw_task,
			)
			source_name = task.source_name or source_task_name
			compound_alias_lookup[task.name] = normalised_task_name
			if source_name:
				compound_alias_lookup[source_name] = normalised_task_name
			compound_alias_lookup[normalised_task_name] = normalised_task_name
			compound_tasks.append(
				HTNTask(
					name=normalised_task_name,
					parameters=task.parameters,
					is_primitive=task.is_primitive,
					source_predicates=tuple(
						predicate_name
						for predicate_name in task.source_predicates
						if predicate_name in predicate_arity
						and predicate_arity[predicate_name] == len(task.parameters)
					),
					source_name=source_name,
				)
			)
		compound_task_names = {task.name for task in compound_tasks}
		primitive_task_names = {self._sanitize_name(action.name) for action in domain.actions}

		methods = []
		for method in library.methods:
			normalised_method_task_name = compound_alias_lookup.get(
				method.task_name,
				self._normalise_declared_task_identifier(
					method.task_name,
					raw_to_alias=raw_task_to_alias,
					alias_to_raw=alias_to_raw_task,
				)[0],
			)
			normalised_steps = []
			for step in method.subtasks:
				preconditions = step.preconditions
				effects = step.effects
				action_name = step.action_name
				task_name = compound_alias_lookup.get(
					step.task_name,
					self._normalise_declared_task_identifier(
						step.task_name,
						raw_to_alias=raw_task_to_alias,
						alias_to_raw=alias_to_raw_task,
					)[0],
				)
				kind = self._coerce_step_kind(
					HTNMethodStep(
						step_id=step.step_id,
						task_name=task_name,
						args=step.args,
						kind=step.kind,
						action_name=action_name,
						literal=step.literal,
						preconditions=step.preconditions,
						effects=step.effects,
					),
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
						literal=None if kind == "primitive" else step.literal,
						preconditions=preconditions,
						effects=effects,
					),
				)

			methods.append(
				HTNMethod(
					method_name=self._normalise_method_identifier(
						method.method_name,
						original_task_name=method.task_name,
						normalised_task_name=normalised_method_task_name,
					),
					task_name=normalised_method_task_name,
					parameters=self._normalise_method_parameters(
						method.parameters,
						method.context,
						tuple(normalised_steps),
					),
					task_args=method.task_args,
					context=method.context,
					subtasks=tuple(normalised_steps),
					ordering=method.ordering,
					origin=method.origin,
					source_method_name=(
						method.source_method_name
						or (
							method.method_name
							if self._normalise_method_identifier(
								method.method_name,
								original_task_name=method.task_name,
								normalised_task_name=normalised_method_task_name,
							)
							!= method.method_name
							else None
						)
					),
				),
			)

		methods = self._deduplicate_method_identifiers(methods)

		return HTNMethodLibrary(
			compound_tasks=compound_tasks,
			primitive_tasks=list(library.primitive_tasks),
			methods=methods,
			target_literals=list(library.target_literals),
			target_task_bindings=[
				HTNTargetTaskBinding(
					target_literal=binding.target_literal,
					task_name=compound_alias_lookup.get(
						binding.task_name,
						self._normalise_declared_task_identifier(
							binding.task_name,
							raw_to_alias=raw_task_to_alias,
							alias_to_raw=alias_to_raw_task,
						)[0],
					),
				)
				for binding in library.target_task_bindings
			],
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
				task_args=method.task_args,
				context=tuple(apply_literal(item) for item in method.context),
				subtasks=tuple(apply_step(step) for step in method.subtasks),
				ordering=method.ordering,
				origin=method.origin,
				source_method_name=method.source_method_name,
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
		def normalise_signature_text(text: Any) -> str:
			if isinstance(text, dict):
				predicate = text.get("predicate", "")
				args = tuple(text.get("args", []))
				is_positive = bool(text.get("is_positive", True))
				text = HTNLiteral(
					predicate=predicate,
					args=args,
					is_positive=is_positive,
					negation_mode="naf",
					source_symbol=text.get("source_symbol"),
				).to_signature()
			value = str(text or "").strip()
			value = re.sub(r"\s+", " ", value)
			value = re.sub(r"\(\s*", "(", value)
			value = re.sub(r"\s*\)", ")", value)
			value = re.sub(r"\s*,\s*", ",", value)
			return value

		canonical_signatures = {
			literal.to_signature()
			for literal in library.target_literals
		}
		alternate_lookup: Dict[str, str] = {}
		for signature in canonical_signatures:
			alternate_lookup[signature] = signature
			alternate_lookup[normalise_signature_text(signature)] = signature
			if signature.startswith("!"):
				alternate_lookup[f"~{signature[1:]}"] = signature
				alternate_lookup[normalise_signature_text(f"~{signature[1:]}")] = signature

		bindings = [
			HTNTargetTaskBinding(
				target_literal=alternate_lookup.get(
					normalise_signature_text(binding.target_literal),
					normalise_signature_text(binding.target_literal),
				),
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

	def _validate_library(
		self,
		library: HTNMethodLibrary,
		domain: Any,
		*,
		query_task_anchors: Optional[Sequence[Dict[str, Any]]] = None,
		query_objects: Sequence[str] = (),
		static_predicates: Sequence[str] = (),
	) -> None:
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
		task_types = self._task_type_map(domain)
		predicate_arities = {
			predicate.name: len(predicate.parameters)
			for predicate in getattr(domain, "predicates", [])
		}
		dynamic_predicates = {
			effect.predicate
			for action_schema in dict.fromkeys(action_schemas.values())
			for effect in getattr(action_schema, "effects", ())
			if effect.predicate != "="
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
			if self._is_deprecated_task_name(task.name):
				raise ValueError(
					f"Legacy task name '{task.name}' is not allowed. "
					"Use semantic task names that describe reusable domain operations rather "
					"than legacy achieve_/ensure_/goal_ wrappers.",
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

		self._validate_query_task_alignment(
			library,
			binding_lookup,
			query_task_anchors or (),
			compound_names,
		)
		self._validate_query_object_leakage(
			library,
			query_objects=query_objects,
		)
		self._validate_fresh_static_helper_tasks(
			library,
			declared_task_names={
				task.name
				for task in getattr(domain, "tasks", [])
			},
			static_predicates=set(static_predicates),
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
				if strategy_suffix == "noop":
					raise ValueError(
						f"Non-empty method '{method.method_name}' cannot use the noop "
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
				if step.kind == "primitive":
					self._validate_primitive_step_semantics(
						step,
						method,
						action_schemas,
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
				task_types,
				predicate_types,
			)

		self._validate_sibling_method_distinguishability(
			library,
			task_lookup,
			action_schemas,
			predicate_arities,
		)
		self._validate_dynamic_precondition_support(
			library,
			task_lookup,
			action_schemas,
			predicate_arities,
			dynamic_predicates=dynamic_predicates,
		)
		self._validate_compound_task_semantics(
			library,
			task_lookup,
			action_schemas,
			predicate_arities,
			dynamic_predicates=dynamic_predicates,
			declared_task_names={
				task.name
				for task in getattr(domain, "tasks", [])
			},
			positive_target_task_names={
				binding.task_name
				for binding in library.target_task_bindings
				if any(
					literal.to_signature() == binding.target_literal and literal.is_positive
					for literal in library.target_literals
				)
			},
		)

	def _validate_dynamic_precondition_support(
		self,
		library: HTNMethodLibrary,
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		*,
		dynamic_predicates: set[str],
	) -> None:
		if not dynamic_predicates:
			return

		methods_by_task: Dict[str, List[HTNMethod]] = {}
		for candidate in library.methods:
			methods_by_task.setdefault(candidate.task_name, []).append(candidate)

		for method in library.methods:
			available_positive = {
				literal.to_signature()
				for literal in method.context
				if literal.is_positive and literal.predicate in dynamic_predicates
			}
			for step in self._ordered_method_steps(method):
				if step.kind == "compound":
					missing_child_requirements = [
						signature
						for signature in self._common_child_constructive_requirements(
							step,
							methods_by_task.get(step.task_name, ()),
							task_lookup,
							action_schemas,
							predicate_arities,
							dynamic_predicates=dynamic_predicates,
						)
						if signature not in available_positive
					]
					if missing_child_requirements:
						raise ValueError(
							f"Method '{method.method_name}' reaches compound step "
							f"'{step.step_id}' ({step.task_name}) without first supporting "
							f"its shared dynamic prerequisites {missing_child_requirements} via "
							"method context or earlier subtasks.",
						)

				if step.kind == "primitive":
					missing_dynamic_preconditions = [
						literal.to_signature()
						for literal in self._step_precondition_literals(step, action_schemas)
						if literal.is_positive
						and literal.predicate in dynamic_predicates
						and literal.to_signature() not in available_positive
					]
					if missing_dynamic_preconditions:
						raise ValueError(
							f"Method '{method.method_name}' reaches primitive step "
							f"'{step.step_id}' ({step.task_name}) without first supporting "
							f"its dynamic preconditions {missing_dynamic_preconditions} via "
							"method context or earlier subtasks.",
						)

				for effect in self._step_effect_literals(
					step,
					task_lookup,
					action_schemas,
					predicate_arities,
				):
					if effect.predicate not in dynamic_predicates:
						continue
					positive_signature = HTNLiteral(
						predicate=effect.predicate,
						args=effect.args,
						is_positive=True,
						source_symbol=None,
					).to_signature()
					if effect.is_positive:
						available_positive.add(positive_signature)
					else:
						available_positive.discard(positive_signature)

	def _common_child_constructive_requirements(
		self,
		step: HTNMethodStep,
		child_methods: Sequence[HTNMethod],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		*,
		dynamic_predicates: set[str],
	) -> Tuple[str, ...]:
		if step.kind != "compound":
			return ()

		task_schema = task_lookup.get(step.task_name)
		headline_signature: Optional[str] = None
		if task_schema and len(task_schema.source_predicates) == 1:
			predicate_name = task_schema.source_predicates[0]
			if predicate_arities.get(predicate_name) == len(step.args):
				headline_signature = HTNLiteral(
					predicate=predicate_name,
					args=tuple(step.args),
					is_positive=True,
					source_symbol=None,
				).to_signature()

		requirement_sets: List[set[str]] = []
		for child_method in child_methods:
			if not child_method.subtasks:
				continue
			grounded_literals = self._materialise_method_literals(
				self._promoted_method_context(
					child_method,
					task_lookup,
					action_schemas,
					predicate_arities,
				),
				child_method.parameters,
				step.args,
			)
			required_signatures = {
				literal.to_signature()
				for literal in grounded_literals
				if literal.is_positive and literal.predicate in dynamic_predicates
				and (
					not literal.args
					or set(literal.args).issubset(set(step.args))
				)
			}
			if headline_signature:
				required_signatures.discard(headline_signature)
			requirement_sets.append(required_signatures)

		if not requirement_sets:
			return ()

		return tuple(sorted(set.intersection(*requirement_sets)))

	def _validate_query_task_alignment(
		self,
		library: HTNMethodLibrary,
		binding_lookup: Dict[str, str],
		query_task_anchors: Sequence[Dict[str, Any]],
		compound_names: set[str],
	) -> None:
		if not query_task_anchors:
			return

		anchor_task_names = [
			str(anchor.get("task_name", "")).strip()
			for anchor in query_task_anchors
			if str(anchor.get("task_name", "")).strip()
		]
		missing_tasks = [
			task_name
			for task_name in dict.fromkeys(anchor_task_names)
			if task_name not in compound_names
		]
		if missing_tasks:
			raise ValueError(
				"Stage 3 output must preserve query-mentioned declared tasks. Missing: "
				+ ", ".join(missing_tasks),
			)

		expected_targets = [literal.to_signature() for literal in library.target_literals]
		if len(anchor_task_names) != len(expected_targets):
			return

		for target_signature, anchor_task_name in zip(expected_targets, anchor_task_names):
			bound_task_name = binding_lookup.get(target_signature)
			if bound_task_name != anchor_task_name:
				raise ValueError(
					f"target_task_binding for '{target_signature}' must use the ordered "
					f"query task anchor '{anchor_task_name}', not '{bound_task_name}'.",
				)

	def _validate_query_object_leakage(
		self,
		library: HTNMethodLibrary,
		*,
		query_objects: Sequence[str],
	) -> None:
		grounded_objects = {
			str(value).strip()
			for value in query_objects
			if str(value).strip()
		}
		if not grounded_objects:
			return

		for task in library.compound_tasks:
			for parameter in task.parameters:
				if parameter in grounded_objects:
					raise ValueError(
						f"Compound task '{task.name}' uses grounded query object "
						f"'{parameter}' as a task parameter. Generated tasks must remain "
						"schematic rather than embedding query-specific object names.",
					)

		for method in library.methods:
			for parameter in method.parameters:
				if parameter in grounded_objects:
					raise ValueError(
						f"Method '{method.method_name}' uses grounded query object "
						f"'{parameter}' as a method parameter. Methods must remain "
						"schematic rather than embedding query-specific object names.",
					)
			for symbol in method.task_args:
				if symbol in grounded_objects:
					raise ValueError(
						f"Method '{method.method_name}' uses grounded query object "
						f"'{symbol}' in task_args. Methods must remain schematic.",
					)
			for literal in method.context:
				for symbol in literal.args:
					if symbol in grounded_objects:
						raise ValueError(
							f"Method '{method.method_name}' leaks grounded query object "
							f"'{symbol}' into method context. Use schematic parameters "
							"and bindings instead.",
						)
			for step in method.subtasks:
				for symbol in step.args:
					if symbol in grounded_objects:
						raise ValueError(
							f"Method '{method.method_name}' leaks grounded query object "
							f"'{symbol}' into subtask '{step.step_id}'. Use schematic "
							"parameters and bindings instead.",
						)
				for literal in (step.literal, *step.preconditions, *step.effects):
					if literal is None:
						continue
					for symbol in literal.args:
						if symbol in grounded_objects:
							raise ValueError(
								f"Method '{method.method_name}' leaks grounded query object "
								f"'{symbol}' into subtask '{step.step_id}' metadata. Use "
								"schematic parameters and bindings instead.",
							)

	def _validate_fresh_static_helper_tasks(
		self,
		library: HTNMethodLibrary,
		*,
		declared_task_names: set[str],
		static_predicates: set[str],
	) -> None:
		if not static_predicates:
			return

		for task in library.compound_tasks:
			if task.name in declared_task_names:
				continue
			static_headlines = sorted(
				predicate_name
				for predicate_name in task.source_predicates
				if predicate_name in static_predicates
			)
			if not static_headlines:
				continue
			raise ValueError(
				f"Fresh helper task '{task.name}' cannot headline static predicates "
				f"{static_headlines}. Static predicates must stay in method context.",
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
			task_schema = task_lookup.get(task_name)
			guard_methods = [method for method in methods if not method.subtasks]
			constructive_methods = [method for method in methods if method.subtasks]
			if len(constructive_methods) <= 1:
				allowed_method_names.update(method.method_name for method in guard_methods)
				allowed_method_names.update(
					method.method_name for method in constructive_methods
				)
				continue

			context_signatures: Dict[str, Tuple[str, ...]] = {}
			context_signature_sets: Dict[str, set[str]] = {}
			supportive_constructive_names: set[str] = set()
			expected_literal: Optional[HTNLiteral] = None
			if task_schema is not None and len(task_schema.source_predicates) == 1:
				expected_literal = HTNLiteral(
					predicate=task_schema.source_predicates[0],
					args=tuple(task_schema.parameters),
					is_positive=True,
					source_symbol=None,
				)
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
				context_signature_sets[method.method_name] = set(signature)
				if signature:
					has_distinguishing_context = True
				if (
					expected_literal is not None
					and self._method_constructively_supports_literal(
						method,
						expected_literal,
						methods_by_task,
						task_lookup,
						action_schemas,
						predicate_arities,
					)
				):
					supportive_constructive_names.add(method.method_name)

			allowed_method_names.update(method.method_name for method in guard_methods)
			seen_signatures: set[Tuple[str, ...]] = set()
			kept_empty_signature = False

			if not has_distinguishing_context:
				allowed_method_names.add(constructive_methods[0].method_name)
				continue

			for method in constructive_methods:
				signature = context_signatures[method.method_name]
				if method.method_name in supportive_constructive_names:
					if any(
						other.method_name != method.method_name
						and other.method_name in supportive_constructive_names
						and context_signature_sets[other.method_name]
						< context_signature_sets[method.method_name]
						for other in constructive_methods
					):
						continue
				if not signature:
					if kept_empty_signature:
						continue
					kept_empty_signature = True
					allowed_method_names.add(method.method_name)
					continue
				if signature in seen_signatures:
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
	def _prune_unreachable_task_structures(
		library: HTNMethodLibrary,
	) -> Tuple[HTNMethodLibrary, int]:
		root_task_names = {
			binding.task_name
			for binding in library.target_task_bindings
			if str(binding.task_name).strip()
		}
		if not root_task_names:
			return library, 0

		methods_by_task: Dict[str, List[HTNMethod]] = {}
		for method in library.methods:
			methods_by_task.setdefault(method.task_name, []).append(method)

		reachable_task_names: set[str] = set()
		pending = list(root_task_names)
		while pending:
			task_name = pending.pop()
			if task_name in reachable_task_names:
				continue
			reachable_task_names.add(task_name)
			for method in methods_by_task.get(task_name, ()):
				for step in method.subtasks:
					if step.kind == "compound" and step.task_name not in reachable_task_names:
						pending.append(step.task_name)

		pruned_compound_tasks = [
			task
			for task in library.compound_tasks
			if task.name in reachable_task_names
		]
		pruned_methods = [
			method
			for method in library.methods
			if method.task_name in reachable_task_names
		]
		pruned_count = (
			len(library.compound_tasks) - len(pruned_compound_tasks)
			+ len(library.methods) - len(pruned_methods)
		)
		if pruned_count == 0:
			return library, 0

		return (
			HTNMethodLibrary(
				compound_tasks=pruned_compound_tasks,
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
				self._method_context_supports_literal(
					method,
					expected_literal,
					task,
				)
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

	def _validate_compound_task_semantics(
		self,
		library: HTNMethodLibrary,
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		*,
		dynamic_predicates: set[str],
		declared_task_names: set[str],
		positive_target_task_names: set[str],
	) -> None:
		methods_by_task: Dict[str, List[HTNMethod]] = {}
		for method in library.methods:
			methods_by_task.setdefault(method.task_name, []).append(method)

		for task in library.compound_tasks:
			if (
				task.name not in declared_task_names
				and task.name not in positive_target_task_names
			):
				continue
			if len(task.source_predicates) != 1:
				continue
			expected_literal = HTNLiteral(
				predicate=task.source_predicates[0],
				args=tuple(task.parameters),
				is_positive=True,
				source_symbol=None,
			)
			task_methods = methods_by_task.get(task.name, [])
			if not task_methods:
				continue
			already_supported = any(
				self._method_context_supports_literal(
					method,
					expected_literal,
					task,
				)
				for method in task_methods
			)
			constructive_methods = [
				method
				for method in task_methods
				if method.subtasks
			]
			unsupported_constructive_methods = [
				method
				for method in constructive_methods
				if not self._method_constructively_supports_literal(
					method,
					expected_literal,
					methods_by_task,
					task_lookup,
					action_schemas,
					predicate_arities,
				)
			]
			if unsupported_constructive_methods:
				raise ValueError(
					f"Compound task '{task.name}' has constructive method(s) "
					f"{', '.join(method.method_name for method in unsupported_constructive_methods)} "
					f"that do not make '{expected_literal.to_signature()}' true via real "
					"subtask effects.",
				)
			for method in constructive_methods:
				self._validate_headline_extra_role_support(
					method,
					task,
					expected_literal,
					methods_by_task,
					task_lookup,
					action_schemas,
					predicate_arities,
					dynamic_predicates=dynamic_predicates,
				)
			supportive_constructive_methods = [
				method
				for method in constructive_methods
				if self._method_constructively_supports_literal(
					method,
					expected_literal,
					methods_by_task,
					task_lookup,
					action_schemas,
					predicate_arities,
				)
			]
			if supportive_constructive_methods:
				expected_signature = expected_literal.to_signature()
				if any(
					not self._literals_support_expected_literal(
						self._promoted_method_context(
							method,
							task_lookup,
							action_schemas,
							predicate_arities,
						),
						expected_literal,
						self._method_parameter_bindings(method, task),
					)
					for method in supportive_constructive_methods
				):
					continue
				raise ValueError(
					f"Compound task '{task.name}' can only constructively support "
					f"'{expected_signature}' through methods that still require that same "
					"literal in promoted context. At least one constructive branch must be "
					"usable when the headline literal is currently false.",
				)
			if already_supported and not constructive_methods:
				continue
			raise ValueError(
				f"Compound task '{task.name}' headlines '{expected_literal.to_signature()}', "
				"but none of its methods exposes that literal in context or constructively "
				"supports it via real subtask effects.",
			)

	def _validate_headline_extra_role_support(
		self,
		method: HTNMethod,
		task: HTNTask,
		expected_literal: HTNLiteral,
		methods_by_task: Dict[str, List[HTNMethod]],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		*,
		dynamic_predicates: set[str],
	) -> None:
		if not dynamic_predicates:
			return

		task_parameter_set = set(self._default_method_task_args(method, task))
		method_bindings = self._method_parameter_bindings(method, task)
		supportable_predicates = {
			(candidate_task.source_predicates[0], len(candidate_task.parameters))
			for candidate_task in task_lookup.values()
			if len(candidate_task.source_predicates) == 1
		}
		available_positive = {
			literal.to_signature()
			for literal in method.context
			if literal.is_positive and literal.predicate in dynamic_predicates
		}
		established_by_steps: set[str] = set()
		headline_producer_index: Optional[int] = None
		ordered_steps = self._ordered_method_steps(method)

		for index, step in enumerate(ordered_steps):
			step_effects = self._step_effect_literals(
				step,
				task_lookup,
				action_schemas,
				predicate_arities,
			)
			if any(
				self._literal_matches_expected_signature(
					literal,
					expected_literal,
					method_bindings,
				)
				for literal in step_effects
				if literal.is_positive
			):
				headline_producer_index = index
				break

		if headline_producer_index is not None:
			for index, step in enumerate(ordered_steps[:headline_producer_index]):
				if step.kind != "compound":
					for effect in self._step_effect_literals(
						step,
						task_lookup,
						action_schemas,
						predicate_arities,
					):
						if effect.predicate not in dynamic_predicates:
							continue
						signature = HTNLiteral(
							predicate=effect.predicate,
							args=effect.args,
							is_positive=True,
							source_symbol=None,
						).to_signature()
						if effect.is_positive:
							available_positive.add(signature)
						else:
							available_positive.discard(signature)
					continue

				later_requirements = self._later_dynamic_requirement_signatures(
					ordered_steps[index + 1:headline_producer_index + 1],
					methods_by_task,
					task_lookup,
					action_schemas,
					predicate_arities,
					dynamic_predicates=dynamic_predicates,
				)
				unresolved_later_requirements = later_requirements - available_positive
				if unresolved_later_requirements:
					possible_effects = {
						literal.to_signature()
						for literal in self._possible_positive_step_effect_literals(
							step,
							methods_by_task,
							task_lookup,
							action_schemas,
							predicate_arities,
							dynamic_predicates=dynamic_predicates,
						)
					}
					if not possible_effects & unresolved_later_requirements:
						raise ValueError(
							f"Method '{method.method_name}' inserts compound step "
							f"'{step.step_id}' ({step.task_name}) before the headline "
							f"producer, but that step does not supply any unresolved later "
							f"dynamic requirement in this branch. Later unresolved "
							f"requirements are {sorted(unresolved_later_requirements)}; "
							f"possible effects of '{step.task_name}' here are "
							f"{sorted(possible_effects)}.",
						)

				for literal in self._possible_positive_step_effect_literals(
					step,
					methods_by_task,
					task_lookup,
					action_schemas,
					predicate_arities,
					dynamic_predicates=dynamic_predicates,
				):
					available_positive.add(literal.to_signature())

		for step in ordered_steps:
			step_effects = self._step_effect_literals(
				step,
				task_lookup,
				action_schemas,
				predicate_arities,
			)
			if step.kind == "primitive":
				deleted_signatures = {
					HTNLiteral(
						predicate=literal.predicate,
						args=literal.args,
						is_positive=True,
						source_symbol=None,
					).to_signature()
					for literal in step_effects
					if not literal.is_positive
				}
				produces_headline = any(
					self._literal_matches_expected_signature(
						literal,
						expected_literal,
						method_bindings,
					)
					for literal in step_effects
					if literal.is_positive
				)
				if produces_headline:
					for literal in self._step_precondition_literals(step, action_schemas):
						signature = literal.to_signature()
						if (
							not literal.is_positive
							or literal.predicate not in dynamic_predicates
							or signature not in available_positive
							or signature in established_by_steps
							or (
								signature in deleted_signatures
								and any(arg in task_parameter_set for arg in literal.args)
								and any(arg not in task_parameter_set for arg in literal.args)
							)
							or not any(arg not in task_parameter_set for arg in literal.args)
							or (literal.predicate, len(literal.args)) not in supportable_predicates
						):
							continue
						raise ValueError(
							f"Method '{method.method_name}' uses producer step '{step.step_id}' "
							f"to make '{expected_literal.to_signature()}', but leaves extra-role "
							f"dynamic prerequisite '{signature}' only as context even though a "
							"declared support task can establish it. Support that prerequisite "
							"earlier instead of assuming it.",
						)

			for effect in step_effects:
				if effect.predicate not in dynamic_predicates:
					continue
				signature = HTNLiteral(
					predicate=effect.predicate,
					args=effect.args,
					is_positive=True,
					source_symbol=None,
				).to_signature()
				if effect.is_positive:
					available_positive.add(signature)
					established_by_steps.add(signature)
				else:
					available_positive.discard(signature)

	def _later_dynamic_requirement_signatures(
		self,
		steps: Sequence[HTNMethodStep],
		methods_by_task: Dict[str, List[HTNMethod]],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		*,
		dynamic_predicates: set[str],
	) -> set[str]:
		requirement_signatures: set[str] = set()
		for step in steps:
			if step.kind == "primitive":
				requirement_signatures.update(
					literal.to_signature()
					for literal in self._step_precondition_literals(step, action_schemas)
					if literal.is_positive and literal.predicate in dynamic_predicates
				)
				continue

			requirement_signatures.update(
				self._common_child_constructive_requirements(
					step,
					methods_by_task.get(step.task_name, ()),
					task_lookup,
					action_schemas,
					predicate_arities,
					dynamic_predicates=dynamic_predicates,
				),
			)
		return requirement_signatures

	def _possible_positive_step_effect_literals(
		self,
		step: HTNMethodStep,
		methods_by_task: Dict[str, List[HTNMethod]],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		*,
		dynamic_predicates: set[str],
		visited_tasks: Optional[frozenset[str]] = None,
	) -> Tuple[HTNLiteral, ...]:
		visited = visited_tasks or frozenset()
		if step.kind == "primitive":
			return tuple(
				literal
				for literal in self._step_effect_literals(
					step,
					task_lookup,
					action_schemas,
					predicate_arities,
				)
				if literal.is_positive and literal.predicate in dynamic_predicates
			)

		positive_literals: list[HTNLiteral] = [
			literal
			for literal in self._step_effect_literals(
				step,
				task_lookup,
				action_schemas,
				predicate_arities,
			)
			if literal.is_positive and literal.predicate in dynamic_predicates
		]
		task_schema = task_lookup.get(step.task_name)
		if task_schema is None or step.task_name in visited:
			return tuple(positive_literals)

		for method in methods_by_task.get(step.task_name, ()):
			task_args = self._default_method_task_args(method, task_schema)
			for child_step in self._ordered_method_steps(method):
				materialised_step = HTNMethodStep(
					step_id=child_step.step_id,
					task_name=child_step.task_name,
					args=tuple(
						{
							parameter: arg
							for parameter, arg in zip(task_args, step.args)
						}.get(arg, arg)
						for arg in child_step.args
					),
					kind=child_step.kind,
					action_name=child_step.action_name,
					literal=(
						self._materialise_method_literals(
							(child_step.literal,),
							task_args,
							step.args,
						)[0]
						if child_step.literal is not None
						else None
					),
					preconditions=self._materialise_method_literals(
						child_step.preconditions,
						task_args,
						step.args,
					),
					effects=self._materialise_method_literals(
						child_step.effects,
						task_args,
						step.args,
					),
				)
				for literal in self._possible_positive_step_effect_literals(
					materialised_step,
					methods_by_task,
					task_lookup,
					action_schemas,
					predicate_arities,
					dynamic_predicates=dynamic_predicates,
					visited_tasks=visited | {step.task_name},
				):
					if any(
						existing.to_signature() == literal.to_signature()
						for existing in positive_literals
					):
						continue
					positive_literals.append(literal)
		return tuple(positive_literals)

	def _validate_primitive_step_semantics(
		self,
		step: HTNMethodStep,
		method: HTNMethod,
		action_schemas: Dict[str, Any],
	) -> None:
		action_schema = self._resolve_action_schema(step, action_schemas)
		if action_schema is None:
			return

		materialised_preconditions = {
			literal.to_signature()
			for literal in self._materialise_action_literals(
				action_schema.preconditions,
				action_schema.parameters,
				step.args,
			)
		}
		materialised_effects = {
			literal.to_signature()
			for literal in self._materialise_action_literals(
				action_schema.effects,
				action_schema.parameters,
				step.args,
			)
		}

		if step.literal is not None:
			if not step.literal.is_positive or step.literal.to_signature() not in materialised_effects:
				raise ValueError(
					f"Primitive step '{step.step_id}' in method '{method.method_name}' uses "
					f"literal '{step.literal.to_signature()}', but action '{step.task_name}' "
					"does not make that positive effect true.",
				)

		for literal in step.preconditions:
			if literal.to_signature() not in materialised_preconditions:
				raise ValueError(
					f"Primitive step '{step.step_id}' in method '{method.method_name}' "
					f"declares unsupported precondition '{literal.to_signature()}' for "
					f"action '{step.task_name}'.",
				)

		for literal in step.effects:
			if literal.to_signature() not in materialised_effects:
				raise ValueError(
					f"Primitive step '{step.step_id}' in method '{method.method_name}' "
					f"declares unsupported effect '{literal.to_signature()}' for action "
					f"'{step.task_name}'.",
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
		error_metadata["failure_class"] = self._classify_failure(
			failure_stage,
			failure_reason,
		)
		return HTNSynthesisError(
			f"HTN synthesis failed during {failure_stage}: {failure_reason}",
			model=error_metadata.get("model"),
			llm_prompt=error_metadata.get("llm_prompt"),
			llm_response=error_metadata.get("llm_response"),
			metadata=error_metadata,
		)

	@staticmethod
	def _classify_failure(
		failure_stage: str,
		failure_reason: str,
	) -> str:
		reason = str(failure_reason or "").lower()
		if failure_stage == "preflight":
			return "llm_not_configured"
		if failure_stage == "llm_call":
			return "llm_call_failed"
		if failure_stage == "response_parse":
			return "llm_response_parse_failed"
		if "explicit ordering edges" in reason:
			return "missing_ordering_edges"
		if "uses auxiliary parameter" in reason:
			return "unconstrained_auxiliary_parameter"
		if "shared dynamic prerequisites" in reason:
			return "missing_shared_child_prerequisites"
		if "dynamic preconditions" in reason:
			return "missing_primitive_dynamic_preconditions"
		if "grounded query object" in reason:
			return "grounded_query_object_leakage"
		if failure_stage == "library_validation":
			return "library_validation_failed"
		return f"{failure_stage}_failed"

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

	def _task_type_map(self, domain: Any) -> Dict[str, Tuple[str, ...]]:
		task_types: Dict[str, Tuple[str, ...]] = {}
		for task in getattr(domain, "tasks", []):
			type_signature = tuple(
				self._parameter_type(parameter)
				for parameter in task.parameters
			)
			task_types[task.name] = type_signature
			task_types[self._sanitize_name(task.name)] = type_signature
		return task_types

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

	def _materialise_method_literals(
		self,
		literals: Tuple[HTNLiteral, ...],
		schema_parameters: Tuple[str, ...],
		step_args: Tuple[str, ...],
	) -> Tuple[HTNLiteral, ...]:
		bindings = {
			parameter: arg
			for parameter, arg in zip(schema_parameters, step_args)
		}
		return tuple(
			HTNLiteral(
				predicate=literal.predicate,
				args=tuple(bindings.get(arg, arg) for arg in literal.args),
				is_positive=literal.is_positive,
				source_symbol=None,
			)
			for literal in literals
		)

	@staticmethod
	def _sanitize_name(name: str) -> str:
		return name.replace("-", "_")

	def _declared_task_alias_maps(
		self,
		domain: Any,
	) -> Tuple[Dict[str, str], Dict[str, str]]:
		raw_to_alias: Dict[str, str] = {}
		alias_to_raw: Dict[str, str] = {}
		for task in getattr(domain, "tasks", []):
			raw_name = str(getattr(task, "name", "") or "").strip()
			if not raw_name:
				continue
			alias_name = self._sanitize_name(raw_name)
			raw_to_alias[raw_name] = alias_name
			alias_to_raw[alias_name] = raw_name
		return raw_to_alias, alias_to_raw

	def _normalise_declared_task_identifier(
		self,
		task_name: str,
		*,
		raw_to_alias: Dict[str, str],
		alias_to_raw: Dict[str, str],
	) -> Tuple[str, Optional[str]]:
		candidate = str(task_name or "").strip()
		if not candidate:
			return "", None
		if candidate in raw_to_alias:
			return raw_to_alias[candidate], candidate
		if candidate in alias_to_raw:
			return candidate, alias_to_raw[candidate]
		sanitized_candidate = self._sanitize_name(candidate)
		if sanitized_candidate in alias_to_raw:
			return sanitized_candidate, alias_to_raw[sanitized_candidate]
		return candidate, None

	def _normalise_method_identifier(
		self,
		method_name: str,
		*,
		original_task_name: str,
		normalised_task_name: str,
	) -> str:
		candidate = str(method_name or "").strip()
		if not candidate:
			return candidate

		strategy_suffix = self._extract_method_strategy_suffix(
			candidate,
			original_task_name=original_task_name,
			normalised_task_name=normalised_task_name,
		)
		if strategy_suffix is None:
			return candidate

		normalised_suffix = self._normalise_method_strategy_suffix(strategy_suffix)
		if not normalised_suffix:
			return candidate
		return f"m_{normalised_task_name}_{normalised_suffix}"

	def _extract_method_strategy_suffix(
		self,
		method_name: str,
		*,
		original_task_name: str,
		normalised_task_name: str,
	) -> Optional[str]:
		prefix_candidates = [
			f"m_{normalised_task_name}_",
			f"m_{original_task_name}_",
			f"m_{self._sanitize_name(original_task_name)}_",
		]
		for prefix in prefix_candidates:
			if method_name.startswith(prefix):
				return method_name[len(prefix):]
		return None

	@staticmethod
	def _normalise_method_strategy_suffix(strategy_suffix: str) -> str:
		normalised = re.sub(r"[^A-Za-z0-9]+", "_", str(strategy_suffix).strip())
		normalised = re.sub(r"_+", "_", normalised).strip("_").lower()
		if not normalised:
			return "branch"
		if not normalised[0].isalpha():
			normalised = f"branch_{normalised}"
		return normalised

	@staticmethod
	def _deduplicate_method_identifiers(
		methods: Sequence[HTNMethod],
	) -> List[HTNMethod]:
		seen: Dict[str, int] = {}
		unique_methods: List[HTNMethod] = []
		for method in methods:
			base_name = method.method_name
			count = seen.get(base_name, 0)
			seen[base_name] = count + 1
			if count == 0:
				unique_methods.append(method)
				continue
			unique_methods.append(
				HTNMethod(
					method_name=f"{base_name}_{count + 1}",
					task_name=method.task_name,
					parameters=method.parameters,
					task_args=method.task_args,
					context=method.context,
					subtasks=method.subtasks,
					ordering=method.ordering,
					origin=method.origin,
					source_method_name=method.source_method_name or method.method_name,
				),
			)
		return unique_methods

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
		bound_variables = set(method.parameters)
		ordered_steps = self._ordered_method_steps(method)
		self._validate_auxiliary_parameters_are_constrained_before_use(
			method,
			task_schema,
			ordered_steps,
		)
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

	def _validate_auxiliary_parameters_are_constrained_before_use(
		self,
		method: HTNMethod,
		task_schema: Optional[HTNTask],
		ordered_steps: Sequence[HTNMethodStep],
	) -> None:
		if method.origin == "domain" or task_schema is None:
			return

		declared_arity = len(task_schema.parameters)
		if len(method.parameters) <= declared_arity:
			return

		auxiliary_parameters = tuple(method.parameters[declared_arity:])
		constrained_variables = {
			variable
			for literal in method.context
			for variable in self._literal_variables(literal)
			if variable in auxiliary_parameters
		}

		for step in ordered_steps:
			for literal in step.preconditions:
				for variable in self._literal_variables(literal):
					if variable in auxiliary_parameters:
						constrained_variables.add(variable)

			for variable in auxiliary_parameters:
				if variable in constrained_variables:
					continue
				if variable not in step.args:
					continue
				raise ValueError(
					f"Method '{method.method_name}' uses auxiliary parameter '{variable}' in "
					f"subtask '{step.step_id}' before constraining it in method.context or "
					"earlier step preconditions.",
				)

	def _validate_method_variable_types(
		self,
		method: HTNMethod,
		task_lookup: Dict[str, HTNTask],
		action_types: Dict[str, Tuple[str, ...]],
		task_types: Dict[str, Tuple[str, ...]],
		predicate_types: Dict[str, Tuple[str, ...]],
	) -> None:
		symbol_types: Dict[str, set[str]] = {}
		task_schema = task_lookup.get(method.task_name)
		task_signature = task_types.get(method.task_name, ())
		for index, parameter in enumerate(method.parameters):
			if index >= len(task_signature):
				break
			symbol_types.setdefault(parameter, set()).add(task_signature[index])
		if not task_signature and task_schema and len(task_schema.source_predicates) == 1:
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
				step_task_signature = task_types.get(step.task_name, ())
				if step_task_signature:
					self._collect_argument_types(symbol_types, step.args, step_task_signature)
				step_task = task_lookup.get(step.task_name)
				if not step_task_signature and step_task and len(step_task.source_predicates) == 1:
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
			empty_signature_method: Optional[str] = None
			for method in constructive_methods:
				promoted_context = self._promoted_method_context(
					method,
					task_lookup,
					action_schemas,
					predicate_arities,
				)
				if not promoted_context:
					if empty_signature_method is not None:
						raise ValueError(
							f"Sibling constructive methods '{empty_signature_method}' and "
							f"'{method.method_name}' for task '{task_name}' are not "
							"semantically distinguishable: both act as empty-context "
							"fallback branches.",
						)
					empty_signature_method = method.method_name
					continue

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
	def _default_method_task_args(
		method: HTNMethod,
		task_schema: Optional[HTNTask],
	) -> Tuple[str, ...]:
		if method.task_args:
			return tuple(method.task_args)
		if task_schema is None:
			return tuple(method.parameters)
		declared_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
		if not declared_parameters:
			return tuple(method.parameters)
		leading_parameters = tuple(method.parameters[: len(declared_parameters)])
		if len(leading_parameters) == len(declared_parameters):
			return leading_parameters
		return declared_parameters

	def _method_parameter_bindings(
		self,
		method: HTNMethod,
		task_schema: Optional[HTNTask],
		parameter_bindings: Optional[Dict[str, str]] = None,
	) -> Dict[str, str]:
		bindings = dict(parameter_bindings or {})
		if task_schema is None:
			return bindings
		for method_arg, task_parameter in zip(
			self._default_method_task_args(method, task_schema),
			task_schema.parameters,
		):
			bindings[method_arg] = bindings.get(task_parameter, task_parameter)
		return bindings

	@classmethod
	def _literals_support_expected_literal(
		cls,
		literals: Iterable[HTNLiteral],
		expected_literal: HTNLiteral,
		parameter_bindings: Dict[str, str],
	) -> bool:
		return any(
			cls._literal_matches_expected_signature(
				literal,
				expected_literal,
				parameter_bindings,
			)
			for literal in literals
		)

	def _method_context_supports_literal(
		self,
		method: HTNMethod,
		expected_literal: HTNLiteral,
		task_schema: Optional[HTNTask] = None,
		parameter_bindings: Optional[Dict[str, str]] = None,
	) -> bool:
		return self._literals_support_expected_literal(
			method.context,
			expected_literal,
			self._method_parameter_bindings(
				method,
				task_schema,
				parameter_bindings,
			),
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
		task_schema = task_lookup.get(method.task_name)
		parameter_bindings = self._method_parameter_bindings(
			method,
			task_schema,
			parameter_bindings,
		)
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

		for raw_candidate_arg, expected_arg in zip(candidate.args, expected.args):
			is_bound_from_parent = raw_candidate_arg in parameter_bindings
			candidate_arg = parameter_bindings.get(raw_candidate_arg, raw_candidate_arg)
			if is_bound_from_parent:
				if candidate_arg != expected_arg:
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
	def _is_deprecated_task_name(cls, task_name: str) -> bool:
		return task_name.startswith(("achieve_", "maintain_not_", "ensure_", "goal_"))

	@staticmethod
	def _schema_hint() -> str:
		return (
			'{"target_task_bindings":[{"target_literal":"linked(A, B)","task_name":"do_link"}],'
			'"compound_tasks":[{"name":"do_link","parameters":["A","B"],"is_primitive":false,'
			'"source_predicates":["linked"]}],'
			'"methods":[{"method_name":"m_do_link_noop","task_name":"do_link","parameters":["A","B"],'
			'"context":[{"predicate":"linked","args":["A","B"],"is_positive":true}]},{"method_name":"m_do_link_constructive","task_name":"do_link",'
			'"parameters":["A","B"],"context":[{"predicate":"ready","args":["B"],"is_positive":true}],'
			'"subtasks":[{"step_id":"s1","task_name":"attach","args":["A","B"],"kind":"primitive",'
			'"action_name":"attach"}]}]}'
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
