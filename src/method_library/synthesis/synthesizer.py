"""
Domain-complete HTN method synthesis.

The synthesizer uses language-model output as the source of compound tasks and
methods while keeping primitive action tasks aligned with the input domain.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from method_library.synthesis.schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
	HTNTargetTaskBinding,
)
from method_library.synthesis.prompts import (
	_extend_mapping_with_action_parameters,
	_render_positive_dynamic_requirements,
	_render_positive_static_requirements,
	build_domain_htn_system_prompt,
	build_domain_htn_user_prompt,
	build_domain_prompt_analysis_payload,
	_render_producer_mode_options_for_predicate,
)
from method_library.synthesis.ast_compilation import MethodSynthesisAstCompilationMixin
from method_library.synthesis.errors import HTNSynthesisError
from method_library.synthesis.library_postprocess import MethodSynthesisLibraryPostprocessMixin
from method_library.synthesis.llm_transport import MethodSynthesisLLMTransportMixin
from method_library.validation.minimal_validation import (
	validate_domain_complete_coverage,
	validate_minimal_library,
)
from language_model import create_openai_compatible_client
from utils.config import DEFAULT_METHOD_SYNTHESIS_MODEL
from utils.hddl_condition_parser import HDDLConditionParser

METHOD_SYNTHESIS_MAX_RETRIES = 5


class HTNMethodSynthesizer(
	MethodSynthesisLibraryPostprocessMixin,
	MethodSynthesisAstCompilationMixin,
	MethodSynthesisLLMTransportMixin,
):
	"""Build HTN method libraries for domain-complete synthesis."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		model: Optional[str] = None,
		base_url: Optional[str] = None,
		timeout: float = 60.0,
		max_tokens: int = 8192,
		session_id: Optional[str] = None,
	) -> None:
		self.api_key = api_key
		self.model = model or DEFAULT_METHOD_SYNTHESIS_MODEL
		self.base_url = base_url
		self.timeout = timeout
		self.max_tokens = max_tokens
		self.session_id = session_id
		self.parser = HDDLConditionParser()
		self.client = None

		if api_key:
			self.client = create_openai_compatible_client(
				api_key=api_key,
				base_url=base_url,
				timeout=self.timeout,
				max_retries=0,
			)

	def synthesize_domain_complete(
		self,
		domain: Any,
		*,
		derived_analysis: Optional[Dict[str, Any]] = None,
		query_sequence: Optional[Sequence[Dict[str, Any]]] = None,
		temporal_specifications: Optional[Sequence[Dict[str, Any]]] = None,
	) -> Tuple[HTNMethodLibrary, Dict[str, Any]]:
		"""Create one domain-complete method library for all declared compound tasks."""

		query_sequence = tuple(dict(item) for item in (query_sequence or ()) if isinstance(item, dict))
		temporal_specifications = tuple(
			dict(item) for item in (temporal_specifications or ()) if isinstance(item, dict)
		)
		action_analysis = self._analyse_domain_actions(domain)
		primitive_tasks = self._build_primitive_tasks(domain)
		prompt_analysis = dict(
			derived_analysis
			or build_domain_prompt_analysis_payload(
				domain,
				action_analysis=action_analysis,
			),
		)
		ast_compiler_defaults = self._build_domain_complete_ast_compiler_defaults(domain)
		metadata: Dict[str, Any] = {
			"used_llm": False,
			"model": self.model if self.client else None,
			"prompt_strategy": (
				"query_aligned_domain_contracts"
				if query_sequence or temporal_specifications
				else "compact_domain_contracts"
			),
			"declared_compound_tasks": list(prompt_analysis.get("declared_compound_tasks") or ()),
			"domain_task_contracts": list(prompt_analysis.get("domain_task_contracts") or ()),
			"action_analysis": action_analysis,
			"derived_analysis": prompt_analysis,
			"query_instruction_ids": [
				str(item.get("instruction_id") or item.get("id") or "").strip()
				for item in query_sequence
				if str(item.get("instruction_id") or item.get("id") or "").strip()
			],
			"query_sequence_count": len(query_sequence),
			"temporal_specification_count": len(temporal_specifications),
			"prompt_declared_task_count": len(
				list(prompt_analysis.get("declared_compound_tasks") or ())
			),
			"prompt_domain_task_contract_count": len(
				list(prompt_analysis.get("domain_task_contracts") or ())
			),
			"prompt_reusable_dynamic_resource_count": len(
				list(prompt_analysis.get("reusable_dynamic_resource_predicates") or ())
			),
			"compound_tasks": 0,
			"primitive_tasks": len(primitive_tasks),
			"methods": 0,
			"failure_stage": None,
			"failure_reason": None,
			"failure_class": None,
			"llm_prompt": None,
			"llm_response": None,
			"llm_finish_reason": None,
			"llm_request_id": None,
			"llm_response_mode": None,
			"llm_stream_handshake_seconds": None,
			"llm_first_stream_chunk_seconds": None,
			"llm_first_chunk_seconds": None,
			"llm_first_content_chunk_seconds": None,
			"llm_first_reasoning_chunk_seconds": None,
			"llm_complete_json_seconds": None,
			"llm_reasoning_chunks_ignored": None,
			"llm_reasoning_excluded": None,
			"llm_max_tokens_policy": None,
			"llm_attempts": 0,
			"llm_generation_attempts": 0,
			"pruned_constructive_siblings": 0,
			"pruned_unreachable_structures": 0,
			"timing_profile": {},
		}

		if not self.client:
			raise self._build_synthesis_error(
				metadata,
				"preflight",
				(
					"Method synthesis requires METHOD_SYNTHESIS_API_KEY. "
					"HTN method synthesis only accepts live LLM output."
				),
			)

		prompt_build_start = time.monotonic()
		prompt = {
			"system": build_domain_htn_system_prompt(),
			"user": build_domain_htn_user_prompt(
				domain,
				schema_hint=self._domain_schema_hint(),
				action_analysis=action_analysis,
				derived_analysis=prompt_analysis,
				query_sequence=query_sequence,
				temporal_specifications=temporal_specifications,
			),
		}
		metadata["timing_profile"]["prompt_build_seconds"] = round(
			time.monotonic() - prompt_build_start,
			6,
		)
		metadata["llm_prompt"] = prompt
		metadata["llm_request_count"] = METHOD_SYNTHESIS_MAX_RETRIES + 1
		request_profile = self._method_synthesis_request_profile(prompt=prompt)
		request_max_tokens = int(request_profile.get("completion_max_tokens") or self.max_tokens)
		request_max_tokens = self._apply_method_synthesis_provider_token_ceiling(request_max_tokens)
		metadata["llm_request_max_tokens"] = request_max_tokens
		metadata["llm_generation_attempts"] = 0

		llm_library, response_text, finish_reason = self._request_complete_llm_library(
			prompt,
			domain,
			metadata,
			prompt_analysis=prompt_analysis,
			ast_compiler_defaults=ast_compiler_defaults,
			max_tokens=request_max_tokens,
		)
		metadata["llm_response"] = response_text
		metadata["llm_finish_reason"] = finish_reason

		llm_only_library = HTNMethodLibrary(
			compound_tasks=list(llm_library.compound_tasks),
			primitive_tasks=primitive_tasks,
			methods=list(llm_library.methods),
			target_literals=[],
			target_task_bindings=[],
		)
		llm_only_library = self._attach_source_instruction_ids(
			llm_only_library,
			temporal_specifications=temporal_specifications,
		)
		metadata["used_llm"] = True
		metadata["compound_tasks"] = len(llm_only_library.compound_tasks)
		metadata["primitive_tasks"] = len(llm_only_library.primitive_tasks)
		metadata["methods"] = len(llm_only_library.methods)
		try:
			validate_domain_complete_coverage(domain, llm_only_library)
		except Exception as exc:
			raise self._build_synthesis_error(
				metadata,
				"coverage",
				str(exc),
			) from exc
		try:
			validate_minimal_library(llm_only_library, domain)
		except Exception as exc:
			raise self._build_synthesis_error(
				metadata,
				"semantic_validation",
				str(exc),
			) from exc
		return llm_only_library, metadata

	@staticmethod
	def _attach_source_instruction_ids(
		method_library: HTNMethodLibrary,
		*,
		temporal_specifications: Sequence[Dict[str, Any]],
	) -> HTNMethodLibrary:
		if not temporal_specifications:
			return method_library

		task_lookup = {
			task.name: task
			for task in [*list(method_library.compound_tasks), *list(method_library.primitive_tasks)]
			if str(getattr(task, "name", "")).strip()
		}
		allowed_instruction_ids = {
			str(item.get("instruction_id") or "").strip()
			for item in temporal_specifications
			if str(item.get("instruction_id") or "").strip()
		}
		if not allowed_instruction_ids:
			return method_library

		spec_task_names: List[Tuple[str, set[str]]] = []
		for item in temporal_specifications:
			instruction_id = str(item.get("instruction_id") or "").strip()
			if not instruction_id:
				continue
			referenced_names: set[str] = set()
			for event in item.get("referenced_events") or ():
				event_name = str((event or {}).get("event") or "").strip()
				if not event_name:
					continue
				base_name = event_name.split("(", 1)[0].strip()
				base_name = re.sub(r"__(?:e|event)[1-9][0-9]*$", "", base_name)
				if base_name:
					referenced_names.add(base_name)
			spec_task_names.append((instruction_id, referenced_names))

		updated_methods: List[HTNMethod] = []
		for method in method_library.methods:
			existing_instruction_ids = tuple(
				instruction_id
				for instruction_id in (
					str(value).strip()
					for value in getattr(method, "source_instruction_ids", ())
				)
				if instruction_id and instruction_id in allowed_instruction_ids
			)
			if existing_instruction_ids:
				updated_methods.append(
					replace(method, source_instruction_ids=existing_instruction_ids),
				)
				continue

			relevant_task_names = {str(method.task_name or "").strip()}
			task_schema = task_lookup.get(str(method.task_name or "").strip())
			if task_schema is not None and str(getattr(task_schema, "source_name", "")).strip():
				relevant_task_names.add(str(getattr(task_schema, "source_name", "")).strip())
			for step in tuple(getattr(method, "subtasks", ()) or ()):
				if str(getattr(step, "task_name", "")).strip():
					relevant_task_names.add(str(getattr(step, "task_name", "")).strip())
				if str(getattr(step, "action_name", "")).strip():
					relevant_task_names.add(str(getattr(step, "action_name", "")).strip())

			derived_instruction_ids = tuple(
				instruction_id
				for instruction_id, referenced_names in spec_task_names
				if referenced_names and referenced_names.intersection(relevant_task_names)
			)
			updated_methods.append(
				replace(method, source_instruction_ids=derived_instruction_ids),
			)

		return HTNMethodLibrary(
			compound_tasks=list(method_library.compound_tasks),
			primitive_tasks=list(method_library.primitive_tasks),
			methods=updated_methods,
			target_literals=list(method_library.target_literals),
			target_task_bindings=list(method_library.target_task_bindings),
		)

	def _build_domain_complete_ast_compiler_defaults(
		self,
		domain: Any,
	) -> Dict[str, Any]:
		task_defaults: Dict[str, Dict[str, Any]] = {}
		call_arities: Dict[str, int] = {}
		primitive_aliases: set[str] = set()

		for task in getattr(domain, "tasks", []):
			source_name = str(getattr(task, "name", "")).strip()
			if not source_name:
				continue
			sanitized_name = self._sanitize_name(source_name)
			task_parameters = [
				str(parameter).split("-", 1)[0].strip()
				for parameter in (getattr(task, "parameters", ()) or ())
			]
			default_entry = {
				"name": sanitized_name,
				"parameters": task_parameters,
				"parameter_types": [
					self._parameter_type(parameter)
					for parameter in (getattr(task, "parameters", ()) or ())
				],
				"source_name": source_name,
			}
			task_defaults[source_name] = dict(default_entry)
			task_defaults[sanitized_name] = dict(default_entry)
			call_arities[source_name] = len(task_parameters)
			call_arities[sanitized_name] = len(task_parameters)

		for action in getattr(domain, "actions", []):
			action_name = str(getattr(action, "name", "")).strip()
			if not action_name:
				continue
			sanitized_name = self._sanitize_name(action_name)
			arity = len(getattr(action, "parameters", ()) or ())
			call_arities[action_name] = arity
			call_arities[sanitized_name] = arity
			primitive_aliases.add(action_name)
			primitive_aliases.add(sanitized_name)

		return {
			"task_defaults": task_defaults,
			"primitive_aliases": sorted(primitive_aliases),
			"call_arities": call_arities,
			"strict_hddl_ast": True,
		}

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

		type_parent_map = self._build_domain_type_parent_map(domain)
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
			"type_parent_map": dict(type_parent_map),
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

	def _request_complete_llm_library(
		self,
		prompt: Dict[str, str],
		domain: Any,
		metadata: Dict[str, Any],
		*,
		prompt_analysis: Optional[Dict[str, Any]] = None,
		ast_compiler_defaults: Optional[Dict[str, Any]] = None,
		max_tokens: Optional[int] = None,
	) -> Tuple[HTNMethodLibrary, str, Optional[str]]:
		total_start = time.monotonic()
		metadata.setdefault("timing_profile", {})
		attempt_durations: List[float] = []
		attempt_trace: List[Dict[str, Any]] = []
		response_text: Optional[str] = None
		finish_reason: Optional[str] = None
		transport_metadata: Dict[str, Any] = {}
		attempt_max_tokens = max_tokens
		max_attempts = METHOD_SYNTHESIS_MAX_RETRIES + 1
		last_exception: Optional[Exception] = None
		for attempt_index in range(1, max_attempts + 1):
			attempt_start = time.monotonic()
			self._emit_method_synthesis_progress(
				f"attempt={attempt_index}/{max_attempts} start model={self.model} max_tokens={attempt_max_tokens}",
			)
			try:
				response_text, finish_reason, transport_metadata = self._call_llm(
					prompt,
					max_tokens=attempt_max_tokens,
				)
				llm_roundtrip_seconds = time.monotonic() - attempt_start
				attempt_durations.append(round(llm_roundtrip_seconds, 3))
				attempt_trace.append(
					self._method_synthesis_attempt_trace(
						attempt_index=attempt_index,
						duration_seconds=llm_roundtrip_seconds,
						request_max_tokens=attempt_max_tokens,
						transport_metadata=transport_metadata,
						finish_reason=finish_reason,
						error=None,
					),
				)
				metadata["llm_request_max_tokens"] = attempt_max_tokens
				self._emit_method_synthesis_progress(
					f"attempt={attempt_index} finish_reason={finish_reason!r} duration_seconds={round(llm_roundtrip_seconds, 3)}",
				)
				break
			except Exception as exc:
				last_exception = exc
				transport_metadata = dict(getattr(exc, "transport_metadata", None) or {})
				llm_roundtrip_seconds = time.monotonic() - attempt_start
				attempt_durations.append(round(llm_roundtrip_seconds, 3))
				partial_response = getattr(exc, "partial_text", None)
				partial_finish_reason = getattr(exc, "finish_reason", None)
				attempt_trace.append(
					self._method_synthesis_attempt_trace(
						attempt_index=attempt_index,
						duration_seconds=llm_roundtrip_seconds,
						request_max_tokens=attempt_max_tokens,
						transport_metadata=transport_metadata,
						finish_reason=partial_finish_reason,
						error=str(exc),
					),
				)
				self._emit_method_synthesis_progress(
					f"attempt={attempt_index} error={exc} duration_seconds={round(llm_roundtrip_seconds, 3)}",
				)
				if partial_response:
					metadata["llm_response"] = str(partial_response)
				if partial_finish_reason is not None:
					metadata["llm_finish_reason"] = partial_finish_reason
				if attempt_index >= max_attempts:
					for key in (
						"llm_request_profile",
						"llm_request_id",
						"llm_response_mode",
						"llm_stream_handshake_seconds",
						"llm_first_stream_chunk_seconds",
						"llm_first_chunk_seconds",
						"llm_first_content_chunk_seconds",
						"llm_first_reasoning_chunk_seconds",
						"llm_complete_json_seconds",
						"llm_reasoning_chunks_ignored",
						"llm_completion_max_tokens",
						"llm_max_tokens_policy",
						"llm_thinking_type",
						"llm_reasoning_effort",
					):
						if transport_metadata.get(key) is not None:
							metadata[key] = transport_metadata.get(key)
					metadata["llm_request_max_tokens"] = attempt_max_tokens
					metadata["llm_attempts"] = len(attempt_durations)
					metadata["llm_generation_attempts"] = len(attempt_durations)
					metadata["llm_attempt_durations_seconds"] = list(attempt_durations)
					metadata["llm_attempt_trace"] = list(attempt_trace)
					metadata["llm_response_time_seconds"] = round(time.monotonic() - total_start, 3)
					metadata["timing_profile"]["llm_roundtrip_seconds"] = round(
						sum(attempt_durations),
						6,
					)
					raise self._build_synthesis_error(
						metadata,
						"llm_call",
						f"LLM request failed: {exc}",
					) from exc

		if response_text is None:
			raise self._build_synthesis_error(
				metadata,
				"llm_call",
				f"LLM request failed: {last_exception or 'unknown failure'}",
			)

		metadata["llm_attempts"] = len(attempt_durations)
		metadata["llm_generation_attempts"] = len(attempt_durations)
		metadata["llm_attempt_durations_seconds"] = list(attempt_durations)
		metadata["llm_attempt_trace"] = list(attempt_trace)
		metadata["llm_response_time_seconds"] = round(time.monotonic() - total_start, 3)
		metadata["llm_response"] = response_text
		metadata["llm_finish_reason"] = finish_reason
		for key in (
			"llm_request_profile",
			"llm_request_id",
			"llm_response_mode",
			"llm_stream_handshake_seconds",
			"llm_first_stream_chunk_seconds",
			"llm_first_chunk_seconds",
			"llm_first_content_chunk_seconds",
			"llm_first_reasoning_chunk_seconds",
			"llm_complete_json_seconds",
			"llm_reasoning_chunks_ignored",
			"llm_completion_max_tokens",
			"llm_max_tokens_policy",
			"llm_thinking_type",
			"llm_reasoning_effort",
		):
			if transport_metadata.get(key) is not None:
				metadata[key] = transport_metadata.get(key)
		metadata["timing_profile"]["llm_roundtrip_seconds"] = round(
			sum(attempt_durations),
			6,
		)

		parse_start = time.monotonic()
		try:
			parsed_library = self._parse_llm_library(
				response_text,
				ast_compiler_defaults=ast_compiler_defaults,
			)
		except Exception as exc:
			if finish_reason == "length":
				raise self._build_synthesis_error(
					metadata,
					"response_parse",
					"LLM response was truncated before completion (finish_reason=length).",
				) from exc
			raise self._build_synthesis_error(
				metadata,
				"response_parse",
				f"LLM response could not be parsed as a valid HTN library: {exc}",
			) from exc
		metadata["timing_profile"]["ast_parse_seconds"] = round(
			time.monotonic() - parse_start,
			6,
		)

		return self._normalise_llm_library(
			parsed_library,
			domain,
			prompt_analysis=prompt_analysis,
		), response_text, finish_reason

	@staticmethod
	def _method_synthesis_attempt_trace(
		*,
		attempt_index: int,
		duration_seconds: float,
		request_max_tokens: Optional[int],
		transport_metadata: Dict[str, Any],
		finish_reason: Optional[str],
		error: Optional[str],
	) -> Dict[str, Any]:
		return {
			"attempt": attempt_index,
			"duration_seconds": round(duration_seconds, 6),
			"request_max_tokens": (
				int(request_max_tokens)
				if request_max_tokens is not None
				else None
			),
			"request_profile": transport_metadata.get("llm_request_profile"),
			"reasoning_budget": transport_metadata.get("llm_reasoning_budget"),
			"first_chunk_timeout_seconds": transport_metadata.get(
				"llm_first_chunk_timeout_seconds",
			),
			"request_id": transport_metadata.get("llm_request_id"),
			"response_mode": transport_metadata.get("llm_response_mode"),
			"stream_handshake_seconds": transport_metadata.get(
				"llm_stream_handshake_seconds",
			),
			"first_stream_chunk_seconds": transport_metadata.get(
				"llm_first_stream_chunk_seconds",
			),
			"first_chunk_seconds": transport_metadata.get("llm_first_chunk_seconds"),
			"first_content_chunk_seconds": transport_metadata.get(
				"llm_first_content_chunk_seconds",
			),
			"first_reasoning_chunk_seconds": transport_metadata.get(
				"llm_first_reasoning_chunk_seconds",
			),
			"complete_json_seconds": transport_metadata.get("llm_complete_json_seconds"),
			"finish_reason": finish_reason,
			"error": error,
		}
