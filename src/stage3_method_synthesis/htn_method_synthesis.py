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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
	HTNTargetTaskBinding,
	_parse_invocation_signature,
)
from stage3_method_synthesis.htn_prompts import (
	_render_producer_mode_options_for_predicate,
	build_htn_system_prompt,
	build_htn_user_prompt,
	build_prompt_analysis_payload,
)
from stage3_method_synthesis.task_naming import (
	query_root_alias_task_name,
	sanitize_identifier,
)
from utils.config import DEFAULT_STAGE3_MODEL
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

class LLMStreamingResponseError(RuntimeError):
	"""Raised when a streaming Stage 3 response is unusable but diagnostically valuable."""

	def __init__(
		self,
		message: str,
		*,
		partial_text: Optional[str] = None,
		finish_reason: Optional[str] = None,
	) -> None:
		super().__init__(message)
		self.partial_text = partial_text
		self.finish_reason = finish_reason

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
		self.model = model or DEFAULT_STAGE3_MODEL
		self.base_url = base_url
		self.timeout = timeout
		self.max_tokens = max_tokens
		self.parser = HDDLConditionParser()
		self.client = None

		if api_key:
			from openai import OpenAI

			if base_url:
				self.client = OpenAI(
					api_key=api_key,
					base_url=base_url,
					timeout=self.timeout,
					max_retries=0,
				)
			else:
				self.client = OpenAI(
					api_key=api_key,
					timeout=self.timeout,
					max_retries=0,
				)

	def synthesize(
		self,
		domain: Any,
		*,
		query_text: Optional[str] = None,
		query_task_anchors: Optional[Sequence[Dict[str, Any]]] = None,
		semantic_objects: Optional[Sequence[str]] = None,
		query_object_inventory: Optional[Sequence[Dict[str, Any]]] = None,
		query_objects: Optional[Sequence[str]] = None,
		derived_analysis: Optional[Dict[str, Any]] = None,
		negation_hints: Optional[Dict[str, Any]] = None,
		ordered_literal_signatures: Optional[Sequence[str]] = None,
	) -> Tuple[HTNMethodLibrary, Dict[str, Any]]:
		"""Create a method library and metadata for logging."""

		target_literals = self.extract_target_literals(
			ordered_literal_signatures=ordered_literal_signatures,
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
		ast_compiler_defaults = None

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
			"timing_profile": {},
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

		prompt_build_start = time.monotonic()
		prompt = {
			"system": build_htn_system_prompt(),
			"user": build_htn_user_prompt(
				domain,
				target_literals=[literal.to_signature() for literal in target_literals],
				schema_hint=self._schema_hint(),
				query_text=query_text or "",
				query_task_anchors=normalised_query_task_anchors,
				semantic_objects=normalised_semantic_objects,
				query_object_inventory=normalised_query_object_inventory,
				query_objects=normalised_query_objects,
				action_analysis=action_analysis,
				derived_analysis=prompt_analysis,
			),
		}
		metadata["timing_profile"]["prompt_build_seconds"] = round(
			time.monotonic() - prompt_build_start,
			6,
		)
		metadata["llm_prompt"] = prompt
		request_max_tokens = self._estimate_stage3_response_token_budget(
			prompt_analysis=prompt_analysis,
			ast_compiler_defaults=ast_compiler_defaults,
			default_max_tokens=self.max_tokens,
		)
		request_max_tokens = self._apply_stage3_provider_token_ceiling(request_max_tokens)
		metadata["llm_request_max_tokens"] = request_max_tokens

		metadata["llm_generation_attempts"] = 1
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
			prompt_analysis=prompt_analysis,
		)
		metadata["pruned_constructive_siblings"] = pruned_count
		llm_only_library, unreachable_pruned_count = self._prune_unreachable_task_structures(
			llm_only_library,
			prompt_analysis=prompt_analysis,
		)
		metadata["pruned_unreachable_structures"] = unreachable_pruned_count
		validation_start = time.monotonic()
		try:
			self._validate_library(
				llm_only_library,
				domain,
				query_task_anchors=normalised_query_task_anchors,
				prompt_analysis=prompt_analysis,
				query_objects=normalised_query_objects,
				static_predicates=tuple(action_analysis["static_predicates"]),
				action_analysis=action_analysis,
			)
		except Exception as exc:
			raise self._build_synthesis_error(
				metadata,
				"library_validation",
				f"LLM HTN library failed validation: {exc}",
			) from exc
		metadata["timing_profile"]["library_validation_seconds"] = round(
			time.monotonic() - validation_start,
			6,
		)

		metadata["used_llm"] = True
		metadata["compound_tasks"] = len(llm_only_library.compound_tasks)
		metadata["primitive_tasks"] = len(llm_only_library.primitive_tasks)
		metadata["methods"] = len(llm_only_library.methods)
		return llm_only_library, metadata

	def extract_target_literals(
		self,
		*,
		ordered_literal_signatures: Optional[Sequence[str]] = None,
	) -> List[HTNLiteral]:
		"""Read Stage 1 query literals without depending on Stage 2 DFA artifacts."""

		seen: set[str] = set()
		literals: List[HTNLiteral] = []
		for signature in ordered_literal_signatures or ():
			literal = self._literal_from_signature_text(str(signature).strip())
			if literal is None:
				continue
			label = literal.to_signature()
			if label in seen:
				continue
			seen.add(label)
			literals.append(literal)

		return literals

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
		*,
		prompt_analysis: Optional[Dict[str, Any]] = None,
		ast_compiler_defaults: Optional[Dict[str, Any]] = None,
		max_tokens: Optional[int] = None,
	) -> Tuple[HTNMethodLibrary, str, Optional[str]]:
		total_start = time.monotonic()
		metadata.setdefault("timing_profile", {})
		metadata["llm_attempts"] = 1
		attempt_start = time.monotonic()
		try:
			response_text, finish_reason = self._call_llm(
				prompt,
				max_tokens=max_tokens or self.max_tokens,
			)
		except Exception as exc:
			llm_roundtrip_seconds = time.monotonic() - attempt_start
			metadata["llm_attempt_durations_seconds"] = [
				round(llm_roundtrip_seconds, 3),
			]
			metadata["llm_response_time_seconds"] = round(time.monotonic() - total_start, 3)
			metadata["timing_profile"]["llm_roundtrip_seconds"] = round(
				llm_roundtrip_seconds,
				6,
			)
			partial_response = getattr(exc, "partial_text", None)
			if partial_response:
				metadata["llm_response"] = str(partial_response)
			partial_finish_reason = getattr(exc, "finish_reason", None)
			if partial_finish_reason is not None:
				metadata["llm_finish_reason"] = partial_finish_reason
			raise self._build_synthesis_error(
				metadata,
				"llm_call",
				f"LLM request failed: {exc}",
			) from exc

		llm_roundtrip_seconds = time.monotonic() - attempt_start
		metadata["llm_attempt_durations_seconds"] = [
			round(llm_roundtrip_seconds, 3),
		]
		metadata["llm_response_time_seconds"] = round(time.monotonic() - total_start, 3)
		metadata["llm_response"] = response_text
		metadata["llm_finish_reason"] = finish_reason
		metadata["timing_profile"]["llm_roundtrip_seconds"] = round(
			llm_roundtrip_seconds,
			6,
		)

		if finish_reason == "length":
			raise self._build_synthesis_error(
				metadata,
				"response_parse",
				"LLM response was truncated before completion (finish_reason=length).",
			)

		parse_start = time.monotonic()
		try:
			parsed_library = self._parse_llm_library(
				response_text,
				ast_compiler_defaults=ast_compiler_defaults,
			)
		except Exception as exc:
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
	def _estimate_stage3_response_token_budget(
		*,
		prompt_analysis: Optional[Dict[str, Any]],
		ast_compiler_defaults: Optional[Dict[str, Any]],
		default_max_tokens: int | None = None,
	) -> int:
		max_tokens = int(default_max_tokens or 0)
		if max_tokens <= 0:
			max_tokens = 48000
		defaults = dict(ast_compiler_defaults or {})
		task_count = len(dict(defaults.get("task_defaults") or {}))
		estimated = 6000 + 220 * task_count
		minimum_budget = 12000
		return min(max_tokens, max(minimum_budget, estimated))

	def _apply_stage3_provider_token_ceiling(self, requested_max_tokens: int) -> int:
		"""
		Cap Stage 3 completion budgets for providers that misbehave on oversized
		one-shot JSON requests.

		Minimax returns stable one-shot libraries for the current
		benchmark prompts, but on the OpenRouter-compatible chat path it can
		occasionally respond with an empty assistant envelope when the requested
		completion budget is inflated well beyond the actual JSON need. Keeping a
		compact ceiling preserves one-shot generation while avoiding provider-side
		null-content failures on longer ordered queries.
		"""
		requested = max(int(requested_max_tokens or 0), 1)
		model_name = str(self.model or "").strip().lower()
		if model_name.startswith("minimax/"):
			return min(requested, 8192)
		return requested

	def _call_llm(
		self,
		prompt: Dict[str, str],
		*,
		max_tokens: Optional[int] = None,
	) -> Tuple[str, Optional[str]]:
		response = self._create_chat_completion(prompt, max_tokens=max_tokens)
		return self._consume_llm_response(response)

	def _consume_llm_response(
		self,
		response: object,
	) -> Tuple[str, Optional[str]]:
		if hasattr(response, "choices"):
			choice = response.choices[0]
			finish_reason = getattr(choice, "finish_reason", None)
			content = self._extract_response_text(response)
			return content, finish_reason
		return self._consume_streaming_llm_response(response)

	def _create_chat_completion(
		self,
		prompt: Dict[str, str],
		*,
		max_tokens: Optional[int] = None,
	):
		request_kwargs: Dict[str, Any] = {
			"model": self.model,
			"messages": [
				{"role": "system", "content": prompt["system"]},
				{"role": "user", "content": prompt["user"]},
			],
			"temperature": 0.0,
			"max_tokens": max_tokens or self.max_tokens,
			"timeout": self.timeout,
			"stream": False,
		}
		extra_body = self._openrouter_provider_routing_body()
		if extra_body is not None:
			request_kwargs["extra_body"] = extra_body
		return self.client.chat.completions.create(
			**request_kwargs,
		)

	def _openrouter_provider_routing_body(self) -> Dict[str, Any] | None:
		base_url = str(self.base_url or "").strip().lower()
		if "openrouter.ai" not in base_url:
			return None
		model_name = str(self.model or "").strip()
		if "/" not in model_name:
			return None
		provider_name = model_name.split("/", 1)[0].strip().lower()
		if not provider_name:
			return None
		return {
			"provider": {
				"only": [provider_name],
				"allow_fallbacks": False,
			},
		}

	def _consume_streaming_llm_response(
		self,
		response: object,
	) -> Tuple[str, Optional[str]]:
		parts: list[str] = []
		finish_reason: Optional[str] = None
		close_stream = getattr(response, "close", None)
		for chunk in response:
			choices = getattr(chunk, "choices", None) or ()
			if not choices:
				continue
			choice = choices[0]
			finish_reason = getattr(choice, "finish_reason", None) or finish_reason
			delta = getattr(choice, "delta", None)
			for candidate in (
				getattr(delta, "content", None) if delta is not None else None,
				getattr(delta, "parsed", None) if delta is not None else None,
				getattr(choice, "message", None),
			):
				extracted = self._normalise_response_content(candidate)
				if extracted is not None:
					parts.append(extracted)
			current_text = "".join(parts).strip()
			complete_payload = self._extract_complete_json_payload_text(current_text)
			if complete_payload is not None:
				if callable(close_stream):
					close_stream()
				return complete_payload, finish_reason or "stop"

		text = "".join(parts).strip()
		complete_payload = self._extract_complete_json_payload_text(text)
		if complete_payload is not None:
			if callable(close_stream):
				close_stream()
			return complete_payload, finish_reason or "stop"
		if text:
			raise LLMStreamingResponseError(
				"LLM response did not contain usable textual JSON content. "
				f"finish_reason={finish_reason!r}",
				partial_text=text,
				finish_reason=finish_reason,
			)
		raise LLMStreamingResponseError(
			"LLM response did not contain usable textual JSON content. "
			f"finish_reason={finish_reason!r}",
			finish_reason=finish_reason,
		)

	def _stream_response_has_complete_json(self, text: str) -> bool:
		return self._extract_complete_json_payload_text(text) is not None

	def _extract_response_text(self, response: object) -> str:
		choices = getattr(response, "choices", None) or ()
		if not choices:
			response_dump = response.model_dump() if hasattr(response, "model_dump") else None
			if isinstance(response_dump, dict):
				extracted = self._extract_response_text_from_response_dump(response_dump)
				if extracted is not None:
					return extracted
			raise RuntimeError("LLM response did not include any choices.")

		message = getattr(choices[0], "message", None)
		if message is None:
			raise RuntimeError("LLM response choice did not include a message payload.")

		for candidate in (
			getattr(message, "content", None),
			getattr(message, "parsed", None),
		):
			extracted = self._normalise_response_content(candidate)
			if extracted is not None:
				return extracted

		dumped_message = message.model_dump() if hasattr(message, "model_dump") else None
		if isinstance(dumped_message, dict):
			for key in ("content", "parsed", "output_text", "text"):
				extracted = self._normalise_response_content(dumped_message.get(key))
				if extracted is not None:
					return extracted
			refusal = dumped_message.get("refusal")
			refusal_text = self._normalise_response_content(refusal)
			if refusal_text:
				raise RuntimeError(f"LLM refused Stage 3 response: {refusal_text}")

		response_dump = response.model_dump() if hasattr(response, "model_dump") else None
		if isinstance(response_dump, dict):
			extracted = self._extract_response_text_from_response_dump(response_dump)
			if extracted is not None:
				return extracted

		finish_reason = getattr(choices[0], "finish_reason", None)
		raise RuntimeError(
			"LLM response did not contain usable textual JSON content. "
			f"finish_reason={finish_reason!r}",
		)

	@staticmethod
	def _normalise_response_content(content: object) -> str | None:
		if content is None:
			return None
		if isinstance(content, str):
			text = content.strip()
			return text or None
		if isinstance(content, dict):
			for key in ("text", "value", "content"):
				extracted = HTNMethodSynthesizer._normalise_response_content(content.get(key))
				if extracted is not None:
					return extracted
			try:
				return json.dumps(content, ensure_ascii=False)
			except TypeError:
				return str(content).strip() or None
		if isinstance(content, (list, tuple)):
			parts: list[str] = []
			for item in content:
				extracted = HTNMethodSynthesizer._normalise_response_content(item)
				if extracted is not None:
					parts.append(extracted)
			if not parts:
				return None
			return "\n".join(parts).strip() or None
		text_attr = getattr(content, "text", None)
		extracted = HTNMethodSynthesizer._normalise_response_content(text_attr)
		if extracted is not None:
			return extracted
		value_attr = getattr(content, "value", None)
		extracted = HTNMethodSynthesizer._normalise_response_content(value_attr)
		if extracted is not None:
			return extracted
		stringified = str(content).strip()
		return stringified or None

	@classmethod
	def _extract_response_text_from_response_dump(cls, response_dump: Dict[str, Any]) -> str | None:
		choices = response_dump.get("choices")
		if isinstance(choices, list) and choices:
			first_choice = choices[0]
			if isinstance(first_choice, dict):
				message = first_choice.get("message")
				if isinstance(message, dict):
					for key in ("content", "parsed", "output_text", "text"):
						extracted = cls._normalise_response_content(message.get(key))
						if extracted is not None:
							return extracted
					if any(key in message for key in ("target_task_bindings", "tasks", "compound_tasks", "methods")):
						extracted = cls._normalise_response_content(message)
						if extracted is not None:
							return extracted
				for key in ("content", "parsed", "output_text", "text"):
					extracted = cls._normalise_response_content(first_choice.get(key))
					if extracted is not None:
						return extracted
		for key in ("output_text", "text", "content", "parsed"):
			extracted = cls._normalise_response_content(response_dump.get(key))
			if extracted is not None:
				return extracted
		return None

	@classmethod
	def _extract_complete_json_payload_text(cls, text: str) -> str | None:
		stripped = str(text or "").strip()
		if not stripped:
			return None

		def _decode_from(start_index: int) -> str | None:
			candidate = stripped[start_index:]
			if cls._appears_truncated_json(candidate):
				return None
			try:
				parsed, end_index = json.JSONDecoder().raw_decode(candidate)
			except json.JSONDecodeError:
				return None
			if not isinstance(parsed, (dict, list)):
				return None
			payload = candidate[:end_index].strip()
			return payload or None

		first_nonspace = stripped[0]
		if first_nonspace in "{[":
			payload = _decode_from(0)
			if payload is not None:
				return payload

		object_index = stripped.find("{")
		if object_index != -1:
			payload = _decode_from(object_index)
			return payload

		array_index = stripped.find("[")
		if array_index != -1:
			payload = _decode_from(array_index)
			return payload
		return None

	def _parse_llm_library(
		self,
		response_text: str,
		*,
		ast_compiler_defaults: Optional[Dict[str, Any]] = None,
	) -> HTNMethodLibrary:
		clean_text = self._strip_code_fences(response_text)
		salvaged_tail_payload = self._salvage_missing_object_closer_at_tail(clean_text)
		if self._appears_truncated_json(clean_text):
			if salvaged_tail_payload is not None:
				payload = salvaged_tail_payload
			else:
				raise ValueError(
					"LLM response appears truncated before the JSON object closed. "
					"The HTN library was cut off mid-response.",
				)
		else:
			try:
				payload = json.loads(clean_text)
			except json.JSONDecodeError as original_error:
				salvaged_payload = self._salvage_ast_payload(clean_text)
				if salvaged_payload is not None:
					payload = salvaged_payload
				else:
					missing_object_closer_payload = self._salvage_missing_object_closer(
						clean_text,
						original_error,
					)
					if missing_object_closer_payload is not None:
						payload = missing_object_closer_payload
					else:
						raw_decoded = self._decode_leading_json_object(clean_text)
						if raw_decoded is not None:
							payload = raw_decoded
						else:
							candidate = self._extract_json_object_candidate(clean_text)
							if candidate is None:
								raise ValueError(
									f"HTN synthesis response could not be parsed as JSON: {original_error}"
								) from original_error
							try:
								payload = json.loads(candidate)
							except json.JSONDecodeError as candidate_error:
								raise ValueError(
									"HTN synthesis response could not be parsed as JSON: "
									f"{candidate_error}"
								) from original_error
		if isinstance(payload, list):
			if len(payload) == 1 and isinstance(payload[0], dict):
				payload = payload[0]
			else:
				raise ValueError("HTN synthesis response must be a JSON object")
		if not isinstance(payload, dict):
			raise ValueError("HTN synthesis response must be a JSON object")
		if ast_compiler_defaults:
			payload = self._apply_ast_compiler_defaults(
				payload,
				ast_compiler_defaults=ast_compiler_defaults,
			)
		return HTNMethodLibrary.from_dict(payload)

	@staticmethod
	def _apply_ast_compiler_defaults(
		payload: Dict[str, Any],
		*,
		ast_compiler_defaults: Dict[str, Any],
	) -> Dict[str, Any]:
		if "tasks" not in payload or "compound_tasks" in payload or "methods" in payload:
			return payload
		tasks_payload = payload.get("tasks", [])
		if not isinstance(tasks_payload, list):
			return payload
		task_defaults = dict(ast_compiler_defaults.get("task_defaults") or {})
		default_target_task_bindings = list(
			ast_compiler_defaults.get("target_task_bindings") or (),
		)
		normalised_payload = dict(payload)
		if default_target_task_bindings:
			normalised_payload["target_task_bindings"] = default_target_task_bindings
		elif normalised_payload.get("target_task_bindings") in (None, []):
			normalised_payload["target_task_bindings"] = default_target_task_bindings
		if ast_compiler_defaults.get("primitive_aliases"):
			normalised_payload["primitive_aliases"] = list(
				ast_compiler_defaults.get("primitive_aliases") or (),
			)
		if ast_compiler_defaults.get("call_arities"):
			normalised_payload["call_arities"] = dict(
				ast_compiler_defaults.get("call_arities") or {},
			)
		strict_hddl_ast = bool(ast_compiler_defaults.get("strict_hddl_ast"))
		normalised_tasks: List[Any] = []
		for task_entry in tasks_payload:
			if not isinstance(task_entry, dict):
				normalised_tasks.append(task_entry)
				continue
			task_name = str(task_entry.get("name", "")).strip()
			default_entry = task_defaults.get(task_name)
			if not default_entry:
				normalised_tasks.append(task_entry)
				continue
			raw_task_parameters = [
				str(value).strip()
				for value in (task_entry.get("parameters") or ())
				if str(value).strip()
			]
			merged_entry = dict(default_entry)
			if strict_hddl_ast:
				merged_entry.update(task_entry)
				merged_entry = HTNMethodSynthesizer._normalise_query_root_bridge_layout(
					merged_entry,
					raw_task_parameters=tuple(raw_task_parameters),
					default_task_parameters=tuple(default_entry.get("parameters") or ()),
				)
				merged_entry = HTNMethodSynthesizer._migrate_ast_task_precondition_shorthand(
					merged_entry,
				)
				merged_entry = HTNMethodSynthesizer._migrate_ast_branch_parameter_shorthand(
					merged_entry,
					raw_task_parameters=raw_task_parameters,
					default_task_parameters=tuple(default_entry.get("parameters") or ()),
				)
			else:
				task_entry = HTNMethodSynthesizer._migrate_ast_ordered_subtasks_branch_array_shorthand(
					task_entry,
				)
				merged_entry.update(task_entry)
				merged_entry = HTNMethodSynthesizer._migrate_ast_branch_parameter_shorthand(
					merged_entry,
					raw_task_parameters=raw_task_parameters,
					default_task_parameters=tuple(default_entry.get("parameters") or ()),
				)
				merged_entry = HTNMethodSynthesizer._migrate_ast_branch_field_shorthand(
					merged_entry,
				)
			for fixed_key in ("name", "parameters", "parameter_types", "headline", "source_name"):
				if fixed_key in default_entry:
					merged_entry[fixed_key] = default_entry[fixed_key]
			normalised_tasks.append(merged_entry)
		normalised_payload["tasks"] = normalised_tasks
		return normalised_payload

	@staticmethod

	@staticmethod

	@staticmethod
	def _migrate_ast_ordered_subtasks_branch_array_shorthand(
		task_entry: Dict[str, Any],
	) -> Dict[str, Any]:
		if task_entry.get("constructive") not in (None, [], {}):
			return task_entry
		raw_ordered_subtasks = task_entry.get("ordered_subtasks")
		if not isinstance(raw_ordered_subtasks, (list, tuple)) or not raw_ordered_subtasks:
			return task_entry
		if not all(isinstance(item, dict) for item in raw_ordered_subtasks):
			return task_entry
		branch_keys = {
			"label",
			"parameters",
			"task_args",
			"precondition",
			"context",
			"ordered_subtasks",
			"ordering",
			"orderings",
			"ordering_edges",
			"subtasks",
			"steps",
			"support_before",
			"producer",
			"produce",
			"followup",
			"followups",
		}
		if not all(any(key in item for key in branch_keys) for item in raw_ordered_subtasks):
			return task_entry
		migrated_entry = dict(task_entry)
		migrated_entry["constructive"] = list(raw_ordered_subtasks)
		migrated_entry.pop("ordered_subtasks", None)
		return migrated_entry

	@staticmethod
	def _migrate_ast_branch_parameter_shorthand(
		task_entry: Dict[str, Any],
		*,
		raw_task_parameters: Sequence[str],
		default_task_parameters: Sequence[str],
	) -> Dict[str, Any]:
		"""
		Preserve branch-local AUX witness parameters before fixed task defaults apply.

		Transition-native contracts intentionally keep task headers equal to the
		canonical headline arity. Some providers still place extra branch witness
		parameters on the task object instead of on the constructive branch. When
		that happens, migrate the longer parameter list onto any constructive
		branch that omitted an explicit parameters field.
		"""
		normalised_raw_parameters = tuple(
			str(value).strip()
			for value in raw_task_parameters
			if str(value).strip()
		)
		normalised_default_parameters = tuple(
			str(value).strip()
			for value in default_task_parameters
			if str(value).strip()
		)
		if (
			not normalised_raw_parameters
			or normalised_raw_parameters == normalised_default_parameters
		):
			return task_entry

		migrated_entry = dict(task_entry)
		constructive_payload = task_entry.get("constructive")
		if constructive_payload not in (None, [], {}):
			migrated_entry["constructive"] = (
				HTNMethodSynthesizer._inject_missing_ast_branch_parameters(
					constructive_payload,
					branch_parameters=normalised_raw_parameters,
				)
			)
			return migrated_entry

		branch_payload = HTNMethodSynthesizer._extract_ast_task_level_branch_shorthand(
			task_entry,
		)
		if branch_payload is None:
			return migrated_entry

		migrated_branch = dict(branch_payload)
		migrated_branch.setdefault("parameters", list(normalised_raw_parameters))
		migrated_entry["constructive"] = [migrated_branch]
		for key in (
			"label",
			"task_args",
			"precondition",
			"context",
			"ordered_subtasks",
			"ordering",
			"orderings",
			"ordering_edges",
			"subtasks",
			"steps",
			"support_before",
			"producer",
			"produce",
			"followup",
			"followups",
		):
			migrated_entry.pop(key, None)
		return migrated_entry

	@staticmethod
	def _migrate_ast_task_precondition_shorthand(
		task_entry: Dict[str, Any],
	) -> Dict[str, Any]:
		"""
		Promote shared task-level preconditions onto constructive branches.

		Strict transition-native AST allows a task object to carry a shared
		precondition, but downstream method matching is branch-based. When a
		provider places the contract precondition on the task object instead of
		inside each constructive branch, inherit that context into every branch
		that omitted its own precondition.
		"""
		constructive_payload = task_entry.get("constructive")
		if constructive_payload in (None, [], {}):
			return task_entry

		task_precondition = task_entry.get("precondition")
		if task_precondition in (None, [], {}):
			return task_entry

		migrated_entry = dict(task_entry)
		migrated_entry["constructive"] = (
			HTNMethodSynthesizer._inject_missing_ast_branch_fields(
				constructive_payload,
				inherited_fields={"precondition": task_precondition},
			)
		)
		migrated_entry.pop("precondition", None)
		return migrated_entry

	@staticmethod
	def _migrate_ast_branch_field_shorthand(
		task_entry: Dict[str, Any],
	) -> Dict[str, Any]:
		"""
		Promote task-level branch fields onto constructive branches when present.

		Some providers keep branch-scoped metadata such as precondition or
		support_before on the task object even when they already emit a
		constructive branch array. That shape is semantically unambiguous for a
		single task definition, so normalise it before validation rather than
		rejecting a mechanically recoverable AST variant.
		"""
		constructive_payload = task_entry.get("constructive")
		if constructive_payload in (None, [], {}):
			return task_entry

		inherited_fields = {
			key: task_entry[key]
			for key in (
				"label",
				"task_args",
				"precondition",
				"context",
				"ordered_subtasks",
				"ordering",
				"orderings",
				"ordering_edges",
				"subtasks",
				"steps",
				"support_before",
				"producer",
				"produce",
				"followup",
				"followups",
			)
			if key in task_entry
		}
		if not inherited_fields:
			return task_entry

		migrated_entry = dict(task_entry)
		migrated_entry["constructive"] = (
			HTNMethodSynthesizer._inject_missing_ast_branch_fields(
				constructive_payload,
				inherited_fields=inherited_fields,
			)
		)
		for key in inherited_fields:
			migrated_entry.pop(key, None)
		return migrated_entry

	@staticmethod
	def _inject_missing_ast_branch_parameters(
		constructive_payload: Any,
		*,
		branch_parameters: Sequence[str],
	) -> Any:
		if isinstance(constructive_payload, list):
			return [
				HTNMethodSynthesizer._inject_missing_ast_branch_parameters(
					item,
					branch_parameters=branch_parameters,
				)
				for item in constructive_payload
			]
		if isinstance(constructive_payload, tuple):
			return tuple(
				HTNMethodSynthesizer._inject_missing_ast_branch_parameters(
					item,
					branch_parameters=branch_parameters,
				)
				for item in constructive_payload
			)
		if not isinstance(constructive_payload, dict):
			return constructive_payload

		for wrapper_key in ("branch", "branches"):
			if wrapper_key in constructive_payload:
				migrated_wrapper = dict(constructive_payload)
				migrated_wrapper[wrapper_key] = (
					HTNMethodSynthesizer._inject_missing_ast_branch_parameters(
						constructive_payload.get(wrapper_key),
						branch_parameters=branch_parameters,
					)
				)
				return migrated_wrapper

		if "parameters" in constructive_payload:
			return constructive_payload

		migrated_branch = dict(constructive_payload)
		migrated_branch["parameters"] = list(branch_parameters)
		return migrated_branch

	@staticmethod
	def _inject_missing_ast_branch_fields(
		constructive_payload: Any,
		*,
		inherited_fields: Dict[str, Any],
	) -> Any:
		if isinstance(constructive_payload, list):
			return [
				HTNMethodSynthesizer._inject_missing_ast_branch_fields(
					item,
					inherited_fields=inherited_fields,
				)
				for item in constructive_payload
			]
		if isinstance(constructive_payload, tuple):
			return tuple(
				HTNMethodSynthesizer._inject_missing_ast_branch_fields(
					item,
					inherited_fields=inherited_fields,
				)
				for item in constructive_payload
			)
		if not isinstance(constructive_payload, dict):
			return constructive_payload

		for wrapper_key in ("branch", "branches"):
			if wrapper_key in constructive_payload:
				migrated_wrapper = dict(constructive_payload)
				migrated_wrapper[wrapper_key] = (
					HTNMethodSynthesizer._inject_missing_ast_branch_fields(
						constructive_payload.get(wrapper_key),
						inherited_fields=inherited_fields,
					)
				)
				return migrated_wrapper

		migrated_branch = dict(constructive_payload)
		for key, value in inherited_fields.items():
			migrated_branch.setdefault(key, value)
		return migrated_branch

	@staticmethod
	def _extract_ast_task_level_branch_shorthand(
		task_entry: Dict[str, Any],
	) -> Dict[str, Any] | None:
		branch_keys = (
			"label",
			"task_args",
			"precondition",
			"context",
			"ordered_subtasks",
			"ordering",
			"orderings",
			"ordering_edges",
			"subtasks",
			"steps",
			"support_before",
			"producer",
			"produce",
			"followup",
			"followups",
		)
		if task_entry.get("constructive") not in (None, [], {}):
			return None
		if not any(key in task_entry for key in branch_keys):
			return None
		return {
			key: task_entry[key]
			for key in branch_keys
			if key in task_entry
		}

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
	def _salvage_missing_object_closer_at_tail(
		result_text: str,
	) -> dict | list | None:
		if result_text.count("[") != result_text.count("]"):
			return None
		if result_text.count("{") != result_text.count("}") + 1:
			return None
		candidate_indexes = [
			index
			for index, character in enumerate(result_text)
			if character == "]"
		]
		for insert_index in reversed(candidate_indexes):
			candidate = result_text[:insert_index] + "}" + result_text[insert_index:]
			try:
				return json.loads(candidate)
			except json.JSONDecodeError:
				continue
		return None

	@staticmethod
	def _extract_json_object_candidate(result_text: str) -> str | None:
		start_index = result_text.find("{")
		end_index = result_text.rfind("}")
		if start_index == -1 or end_index == -1 or end_index <= start_index:
			return None
		candidate = result_text[start_index:end_index + 1].strip()
		return candidate or None

	@staticmethod
	def _salvage_missing_object_closer(
		result_text: str,
		error: json.JSONDecodeError,
	) -> dict | list | None:
		position = int(getattr(error, "pos", -1))
		if position < 0 or position >= len(result_text):
			return None
		if result_text[position] != "]":
			return None
		if result_text.count("[") != result_text.count("]"):
			return None
		if result_text.count("{") != result_text.count("}") + 1:
			return None
		candidate = result_text[:position] + "}" + result_text[position:]
		try:
			return json.loads(candidate)
		except json.JSONDecodeError:
			return None

	@staticmethod
	def _decode_leading_json_object(result_text: str) -> dict | None:
		stripped = result_text.lstrip()
		if not stripped.startswith("{"):
			return None
		try:
			decoder = json.JSONDecoder()
			parsed, _ = decoder.raw_decode(stripped)
		except json.JSONDecodeError:
			return None
		return parsed if isinstance(parsed, dict) else None

	@classmethod
	def _salvage_ast_payload(cls, result_text: str) -> dict | None:
		target_task_bindings_array = cls._extract_balanced_array_for_key(
			result_text,
			"target_task_bindings",
		)
		if target_task_bindings_array is None:
			return None
		try:
			target_task_bindings = json.loads(target_task_bindings_array)
		except json.JSONDecodeError:
			return None
		if not isinstance(target_task_bindings, list):
			return None

		task_object_texts = cls._extract_named_task_object_fragments(result_text)
		if not task_object_texts:
			return None

		tasks: List[Dict[str, Any]] = []
		for task_text in task_object_texts:
			try:
				task_payload = json.loads(task_text)
			except json.JSONDecodeError:
				continue
			if not isinstance(task_payload, dict):
				continue
			if not str(task_payload.get("name", "")).strip():
				continue
			tasks.append(task_payload)

		if not tasks:
			return None
		return {
			"target_task_bindings": target_task_bindings,
			"tasks": tasks,
		}

	@classmethod
	def _extract_balanced_array_for_key(
		cls,
		result_text: str,
		key: str,
	) -> str | None:
		match = re.search(rf'"{re.escape(key)}"\s*:', result_text)
		if match is None:
			return None
		index = match.end()
		while index < len(result_text) and result_text[index].isspace():
			index += 1
		if index >= len(result_text) or result_text[index] != "[":
			return None
		end_index = cls._find_matching_delimiter(
			result_text,
			start_index=index,
			open_char="[",
			close_char="]",
		)
		if end_index is None:
			return None
		return result_text[index:end_index + 1]

	@classmethod
	def _extract_named_task_object_fragments(
		cls,
		result_text: str,
	) -> List[str]:
		tasks_match = re.search(r'"tasks"\s*:', result_text)
		if tasks_match is None:
			return []
		search_index = tasks_match.end()
		fragments: List[str] = []
		while True:
			match = re.search(r'\{\s*"name"\s*:', result_text[search_index:])
			if match is None:
				break
			start_index = search_index + match.start()
			end_index = cls._find_matching_delimiter(
				result_text,
				start_index=start_index,
				open_char="{",
				close_char="}",
			)
			if end_index is None:
				break
			fragments.append(result_text[start_index:end_index + 1])
			search_index = end_index + 1
		return fragments

	@staticmethod
	def _find_matching_delimiter(
		result_text: str,
		*,
		start_index: int,
		open_char: str,
		close_char: str,
	) -> int | None:
		depth = 0
		in_string = False
		escape_next = False
		for index in range(start_index, len(result_text)):
			character = result_text[index]
			if in_string:
				if escape_next:
					escape_next = False
				elif character == "\\":
					escape_next = True
				elif character == '"':
					in_string = False
				continue
			if character == '"':
				in_string = True
				continue
			if character == open_char:
				depth += 1
				continue
			if character == close_char:
				depth -= 1
				if depth == 0:
					return index
		return None

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
		*,
		prompt_analysis: Optional[Dict[str, Any]] = None,
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
					headline_literal=task.headline_literal,
					source_name=source_name,
				)
			)
		compound_tasks = self._infer_missing_task_source_predicates(
			compound_tasks,
			library.target_task_bindings,
			library.methods,
			predicate_arity=predicate_arity,
			raw_to_alias=raw_task_to_alias,
			alias_to_raw=alias_to_raw_task,
			prompt_analysis=prompt_analysis,
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

	def _infer_missing_task_source_predicates(
		self,
		compound_tasks: Sequence[HTNTask],
		target_task_bindings: Sequence[HTNTargetTaskBinding],
		methods: Sequence[HTNMethod],
		*,
		predicate_arity: Dict[str, int],
		raw_to_alias: Dict[str, str],
		alias_to_raw: Dict[str, str],
		prompt_analysis: Optional[Dict[str, Any]] = None,
	) -> List[HTNTask]:
		binding_predicates_by_task: Dict[str, List[str]] = {}
		for binding in target_task_bindings:
			literal = self._literal_from_signature_text(binding.target_literal)
			if literal is None:
				continue
			task_name, _ = self._normalise_declared_task_identifier(
				binding.task_name,
				raw_to_alias=raw_to_alias,
				alias_to_raw=alias_to_raw,
			)
			binding_predicates_by_task.setdefault(task_name, [])
			if literal.predicate not in binding_predicates_by_task[task_name]:
				binding_predicates_by_task[task_name].append(literal.predicate)

		task_parameters_by_name = {
			task.name: task.parameters
			for task in compound_tasks
		}
		noop_predicates_by_task: Dict[str, List[str]] = {}
		noop_headlines_by_task: Dict[str, List[HTNLiteral]] = {}
		for method in methods:
			if method.subtasks:
				continue
			positive_context_literals = [
				literal
				for literal in method.context
				if literal.is_positive
			]
			if len(positive_context_literals) != 1:
				continue
			headline_literal = positive_context_literals[0]
			if len(headline_literal.args) != len(method.parameters):
				continue
			task_parameters = task_parameters_by_name.get(method.task_name)
			if task_parameters:
				provisional_task = HTNTask(
					name=method.task_name,
					parameters=task_parameters,
					is_primitive=False,
				)
				method_bindings = self._method_parameter_bindings(
					method,
					provisional_task,
				)
				canonical_headline = HTNLiteral(
					predicate=headline_literal.predicate,
					args=tuple(
						method_bindings.get(argument, argument)
						for argument in headline_literal.args
					),
					is_positive=headline_literal.is_positive,
					source_symbol=headline_literal.source_symbol,
					negation_mode=headline_literal.negation_mode,
				)
				if (
					len(canonical_headline.args) == len(task_parameters)
					and set(canonical_headline.args).issubset(set(task_parameters))
				):
					noop_headlines_by_task.setdefault(method.task_name, [])
					if canonical_headline not in noop_headlines_by_task[method.task_name]:
						noop_headlines_by_task[method.task_name].append(canonical_headline)
			noop_predicates_by_task.setdefault(method.task_name, [])
			if headline_literal.predicate not in noop_predicates_by_task[method.task_name]:
				noop_predicates_by_task[method.task_name].append(headline_literal.predicate)

		headline_candidates_by_task: Dict[str, Tuple[str, ...]] = {}
		for raw_task_name, predicates in (prompt_analysis or {}).get("task_headline_candidates", {}).items():
			task_name, _ = self._normalise_declared_task_identifier(
				raw_task_name,
				raw_to_alias=raw_to_alias,
				alias_to_raw=alias_to_raw,
			)
			headline_candidates_by_task[task_name] = tuple(
				str(predicate_name)
				for predicate_name in predicates
				if str(predicate_name).strip()
			)

		enriched_tasks: List[HTNTask] = []
		for task in compound_tasks:
			headline_literal = task.headline_literal
			if headline_literal is None:
				inferred_headlines = tuple(
					literal
					for literal in noop_headlines_by_task.get(task.name, ())
					if len(literal.args) == len(task.parameters)
				)
				if inferred_headlines:
					headline_literal = inferred_headlines[0]
			if task.source_predicates and headline_literal is not None:
				enriched_tasks.append(
					HTNTask(
						name=task.name,
						parameters=task.parameters,
						is_primitive=task.is_primitive,
						source_predicates=task.source_predicates,
						headline_literal=headline_literal,
						source_name=task.source_name,
					)
				)
				continue
			inferred_predicates = tuple(
				predicate_name
				for predicate_name in (
					(headline_literal.predicate,) if headline_literal is not None else ()
				) or (
					binding_predicates_by_task.get(task.name, ())
					or noop_predicates_by_task.get(task.name, ())
					or headline_candidates_by_task.get(task.name, ())
				)
				if predicate_arity.get(predicate_name) == len(task.parameters)
			)
			enriched_tasks.append(
				HTNTask(
					name=task.name,
					parameters=task.parameters,
					is_primitive=task.is_primitive,
					source_predicates=inferred_predicates[:1],
					headline_literal=headline_literal,
					source_name=task.source_name,
				)
			)
		return enriched_tasks

	@staticmethod
	def _materialise_task_headline_literal(
		task_schema: Optional[HTNTask],
		*,
		bound_args: Optional[Sequence[str]] = None,
		predicate_arities: Optional[Dict[str, int]] = None,
	) -> Optional[HTNLiteral]:
		if task_schema is None:
			return None

		headline_literal = task_schema.headline_literal
		if headline_literal is None:
			if len(task_schema.source_predicates) != 1:
				return None
			predicate_name = task_schema.source_predicates[0]
			if (
				predicate_arities is not None
				and predicate_arities.get(predicate_name) != len(task_schema.parameters)
			):
				return None
			headline_literal = HTNLiteral(
				predicate=predicate_name,
				args=tuple(task_schema.parameters),
				is_positive=True,
				source_symbol=None,
			)

		if bound_args is None:
			return headline_literal
		if len(bound_args) != len(task_schema.parameters):
			return None

		argument_bindings = {
			parameter: argument
			for parameter, argument in zip(task_schema.parameters, bound_args)
		}
		return HTNLiteral(
			predicate=headline_literal.predicate,
			args=tuple(
				argument_bindings.get(argument, argument)
				for argument in headline_literal.args
			),
			is_positive=headline_literal.is_positive,
			source_symbol=headline_literal.source_symbol,
			negation_mode=headline_literal.negation_mode,
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
		prompt_analysis: Optional[Dict[str, Any]] = None,
		query_objects: Sequence[str] = (),
		static_predicates: Sequence[str] = (),
		action_analysis: Optional[Dict[str, Any]] = None,
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
		type_parent_map = self._build_domain_type_parent_map(domain)
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
		normalised_action_analysis = action_analysis or self._analyse_domain_actions(domain)
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
			if task.headline_literal is not None:
				headline_predicate = task.headline_literal.predicate
				if headline_predicate not in predicate_arities:
					raise ValueError(
						f"Task '{task.name}' headline literal references unknown predicate "
						f"'{headline_predicate}'. Known predicates: {sorted(predicate_arities)}",
					)
				if len(task.headline_literal.args) != len(task.parameters):
					raise ValueError(
						f"Task '{task.name}' headline literal '{task.headline_literal.to_signature()}' "
						f"uses {len(task.headline_literal.args)} args, but the task declares "
						f"{len(task.parameters)} parameters.",
					)
				if set(task.headline_literal.args) != set(task.parameters):
					raise ValueError(
						f"Task '{task.name}' headline literal '{task.headline_literal.to_signature()}' "
						"must be a permutation of the task parameters.",
					)
				if task.source_predicates and headline_predicate not in task.source_predicates:
					raise ValueError(
						f"Task '{task.name}' headline literal '{task.headline_literal.to_signature()}' "
						f"conflicts with source_predicates {list(task.source_predicates)}.",
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

		required_contract_task_names = {
			self._sanitize_name(str(payload.get("task_name", "")).strip())
			for section_name in ("query_task_contracts", "support_task_contracts")
			for payload in (prompt_analysis or {}).get(section_name, [])
			if str(payload.get("task_name", "")).strip()
		}
		missing_required_contract_tasks = sorted(
			required_contract_task_names - compound_names
		)
		if missing_required_contract_tasks:
			raise ValueError(
				"LLM HTN library omitted required contract task definitions: "
				f"{missing_required_contract_tasks}. Define every task listed in the prompt "
				"contracts exactly once in tasks.",
			)

		required_helper_specs = [
			{
				"helper_task": self._sanitize_name(str(payload.get("task_name", "")).strip()),
				"parent_task": self._sanitize_name(str(payload.get("required_parent_task", "")).strip()),
				"packaging_task": self._sanitize_name(str(payload.get("required_packaging_task", "")).strip()),
				"helper_target_args": tuple(payload.get("helper_target_args") or ()),
			}
			for payload in (prompt_analysis or {}).get("support_task_contracts", [])
			if str(payload.get("required_parent_task", "")).strip()
			and str(payload.get("required_packaging_task", "")).strip()
		]
		for helper_spec in required_helper_specs:
			for method in library.methods:
				if method.task_name != helper_spec["parent_task"]:
					continue
				parent_task_schema = task_lookup.get(method.task_name)
				parent_task_arity = (
					len(parent_task_schema.parameters)
					if parent_task_schema is not None
					else 0
				)
				parent_task_args = tuple(
					method.task_args
					or method.parameters[:parent_task_arity]
				)
				expected_helper_args: list[str] = []
				for raw_arg in helper_spec["helper_target_args"]:
					text = str(raw_arg).strip()
					if text.startswith("ARG") and text[3:].isdigit():
						position = int(text[3:]) - 1
						if 0 <= position < len(parent_task_args):
							expected_helper_args.append(parent_task_args[position])
							continue
					expected_helper_args.append(text)
				expected_helper_args_tuple = tuple(expected_helper_args)
				packaging_step_indices = [
					index
					for index, step in enumerate(method.subtasks)
					if step.kind == "compound"
					and step.task_name == helper_spec["packaging_task"]
				]
				if not packaging_step_indices:
					continue
				packaging_step_index = min(packaging_step_indices)
				has_required_helper_call = any(
					step.kind == "compound"
					and step.task_name == helper_spec["helper_task"]
					and tuple(step.args) == expected_helper_args_tuple
					for step in method.subtasks[:packaging_step_index]
				)
				if not has_required_helper_call:
					raise ValueError(
						f"Method '{method.method_name}' for task '{method.task_name}' must call "
						f"required helper '{helper_spec['helper_task']}"
						f"{expected_helper_args_tuple}' before packaging child "
						f"'{helper_spec['packaging_task']}'. Do not hard-wire the helper's "
						"primitive producer modes directly into the parent branch.",
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
		declared_task_names = {
			task.name
			for task in getattr(domain, "tasks", [])
		}
		semantic_task_names = set(declared_task_names)
		self._validate_fresh_static_helper_tasks(
			library,
			declared_task_names=semantic_task_names,
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
					if step.task_name == "nop":
						raise ValueError(
							f"Method '{method.method_name}' uses nop in a constructive branch. "
							"already-satisfied cases belong only in noop methods; constructive "
							"branches must use real supporting or producing steps.",
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
				type_parent_map=type_parent_map,
			)

		self._validate_sibling_method_distinguishability(
			library,
			task_lookup,
			action_schemas,
			predicate_arities,
			prompt_analysis=prompt_analysis,
		)
		self._validate_dynamic_precondition_support(
			library,
			task_lookup,
			action_schemas,
			predicate_arities,
			dynamic_predicates=dynamic_predicates,
			prompt_analysis=prompt_analysis,
		)
		self._validate_compound_task_semantics(
			library,
			task_lookup,
			action_schemas,
			predicate_arities,
			dynamic_predicates=dynamic_predicates,
			declared_task_names=semantic_task_names,
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
		prompt_analysis: Optional[Dict[str, Any]] = None,
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
			known_negative = {
				HTNLiteral(
					predicate=literal.predicate,
					args=literal.args,
					is_positive=True,
					source_symbol=None,
				).to_signature()
				for literal in method.context
				if not literal.is_positive and literal.predicate in dynamic_predicates
			}
			for step in self._ordered_method_steps(method):
				if step.kind == "compound":
					child_methods = methods_by_task.get(step.task_name, ())
					if child_methods and not any(
						self._child_method_is_compatible_with_known_dynamic_facts(
							step,
							child_method,
							available_positive=available_positive,
							known_negative=known_negative,
							task_lookup=task_lookup,
							action_schemas=action_schemas,
							predicate_arities=predicate_arities,
							dynamic_predicates=dynamic_predicates,
						)
						for child_method in child_methods
					):
						raise ValueError(
							f"Method '{method.method_name}' reaches compound step "
							f"'{step.step_id}' ({step.task_name}), but no child method for "
							f"'{step.task_name}' is compatible with the dynamic facts already "
							"established or ruled out by this branch.",
						)
					missing_child_requirements = [
						signature
						for signature in self._common_compatible_child_dynamic_requirements(
							step,
							child_methods,
							task_lookup,
							action_schemas,
							predicate_arities,
							dynamic_predicates=dynamic_predicates,
							available_positive=available_positive,
							known_negative=known_negative,
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
						known_negative.discard(positive_signature)
					else:
						available_positive.discard(positive_signature)
						known_negative.add(positive_signature)

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
		headline_literal = self._materialise_task_headline_literal(
			task_schema,
			bound_args=step.args,
			predicate_arities=predicate_arities,
		)
		if headline_literal is not None:
			headline_signature = headline_literal.to_signature()

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

	def _common_compatible_child_dynamic_requirements(
		self,
		step: HTNMethodStep,
		child_methods: Sequence[HTNMethod],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		*,
		dynamic_predicates: set[str],
		available_positive: set[str],
		known_negative: set[str],
	) -> Tuple[str, ...]:
		if step.kind != "compound":
			return ()

		compatible_child_methods = tuple(
			child_method
			for child_method in child_methods
			if self._child_method_is_compatible_with_known_dynamic_facts(
				step,
				child_method,
				available_positive=available_positive,
				known_negative=known_negative,
				task_lookup=task_lookup,
				action_schemas=action_schemas,
				predicate_arities=predicate_arities,
				dynamic_predicates=dynamic_predicates,
			)
		)
		if compatible_child_methods:
			return self._common_child_dynamic_requirements(
				step,
				compatible_child_methods,
				task_lookup,
				action_schemas,
				predicate_arities,
				dynamic_predicates=dynamic_predicates,
			)
		return self._common_child_dynamic_requirements(
			step,
			child_methods,
			task_lookup,
			action_schemas,
			predicate_arities,
			dynamic_predicates=dynamic_predicates,
		)

	def _common_child_dynamic_requirements(
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
		headline_literal = self._materialise_task_headline_literal(
			task_schema,
			bound_args=step.args,
			predicate_arities=predicate_arities,
		)
		if headline_literal is not None:
			headline_signature = headline_literal.to_signature()

		constructive_child_methods = [
			child_method
			for child_method in child_methods
			if child_method.subtasks
		]
		evaluated_child_methods = (
			constructive_child_methods
			if len(constructive_child_methods) > 1
			else list(child_methods)
		)

		requirement_sets: List[set[str]] = []
		for child_method in evaluated_child_methods:
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

	def _possible_child_constructive_requirements(
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
		headline_literal = self._materialise_task_headline_literal(
			task_schema,
			bound_args=step.args,
			predicate_arities=predicate_arities,
		)
		if headline_literal is not None:
			headline_signature = headline_literal.to_signature()

		requirement_signatures: set[str] = set()
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
			for literal in grounded_literals:
				if not literal.is_positive or literal.predicate not in dynamic_predicates:
					continue
				if literal.args and not set(literal.args).issubset(set(step.args)):
					continue
				signature = literal.to_signature()
				if headline_signature and signature == headline_signature:
					continue
				requirement_signatures.add(signature)

		return tuple(sorted(requirement_signatures))

	def _child_method_is_compatible_with_known_dynamic_facts(
		self,
		step: HTNMethodStep,
		child_method: HTNMethod,
		*,
		available_positive: set[str],
		known_negative: set[str],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		dynamic_predicates: set[str],
	) -> bool:
		promoted_context = self._materialise_method_literals(
			self._promoted_method_context(
				child_method,
				task_lookup,
				action_schemas,
				predicate_arities,
			),
			child_method.parameters,
			step.args,
		)
		for literal in promoted_context:
			if literal.predicate not in dynamic_predicates:
				continue
			if self._required_literal_is_contradicted_by_known_dynamic_facts(
				required_literal=literal,
				available_positive=available_positive,
				known_negative=known_negative,
			):
				return False
		return True

	@classmethod
	def _required_literal_is_contradicted_by_known_dynamic_facts(
		cls,
		*,
		required_literal: HTNLiteral,
		available_positive: set[str],
		known_negative: set[str],
	) -> bool:
		relevant_signatures = available_positive if not required_literal.is_positive else known_negative
		for candidate_signature in relevant_signatures:
			candidate_literal = cls._parse_signature_literal(candidate_signature)
			if candidate_literal is None:
				continue
			if candidate_literal.predicate != required_literal.predicate:
				continue
			if len(candidate_literal.args) != len(required_literal.args):
				continue
			local_alias_bindings: Dict[str, str] = {}
			matched = True
			for required_arg, candidate_arg in zip(required_literal.args, candidate_literal.args):
				if str(required_arg).startswith("__child_local_"):
					bound_value = local_alias_bindings.get(required_arg)
					if bound_value is None:
						local_alias_bindings[required_arg] = candidate_arg
					elif bound_value != candidate_arg:
						matched = False
						break
					continue
				if required_arg != candidate_arg:
					matched = False
					break
			if matched:
				return True
		return False

	@staticmethod
	def _parse_signature_literal(signature: str) -> Optional[HTNLiteral]:
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

	@staticmethod

	@staticmethod

	def _is_branch_specific_dynamic_selector(
		self,
		literal: HTNLiteral,
		method: HTNMethod,
		sibling_constructive_methods: Sequence[HTNMethod],
	) -> bool:
		if len(tuple(sibling_constructive_methods)) <= 1:
			return False
		literal_signature = literal.to_signature()
		for sibling in sibling_constructive_methods:
			if sibling.method_name == method.method_name:
				continue
			sibling_context_signatures = {
				context_literal.to_signature()
				for context_literal in sibling.context
				if context_literal.is_positive
			}
			if literal_signature not in sibling_context_signatures:
				return True
		return False

	@staticmethod
	def _literal_from_dict(payload: Any) -> Optional[HTNLiteral]:
		if not isinstance(payload, dict):
			return None
		predicate = str(payload.get("predicate", "")).strip()
		if not predicate:
			return None
		return HTNLiteral(
			predicate=predicate,
			args=tuple(str(arg).strip() for arg in (payload.get("args") or ()) if str(arg).strip()),
			is_positive=bool(payload.get("is_positive", True)),
			source_symbol=payload.get("source_symbol"),
			negation_mode=str(payload.get("negation_mode", "naf") or "naf"),
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
		*,
		prompt_analysis: Optional[Dict[str, Any]] = None,
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
			expected_literal = self._materialise_task_headline_literal(
				task_schema,
				predicate_arities=predicate_arities,
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
		*,
		prompt_analysis: Optional[Dict[str, Any]] = None,
	) -> Tuple[HTNMethodLibrary, int]:
		root_task_names = {
			binding.task_name
			for binding in library.target_task_bindings
			if str(binding.task_name).strip()
		}
		required_contract_task_names = {
			str(payload.get("task_name", "")).strip()
			for section_name in ("query_task_contracts", "support_task_contracts")
			for payload in (prompt_analysis or {}).get(section_name, [])
			if str(payload.get("task_name", "")).strip()
		}
		seed_task_names = root_task_names | required_contract_task_names
		if not seed_task_names:
			return library, 0

		methods_by_task: Dict[str, List[HTNMethod]] = {}
		for method in library.methods:
			methods_by_task.setdefault(method.task_name, []).append(method)

		reachable_task_names: set[str] = set()
		pending = list(seed_task_names)
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
			expected_literal = self._materialise_task_headline_literal(
				task,
				predicate_arities=predicate_arities,
			)
			if expected_literal is None:
				continue
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
		initial_context_positive = {
			literal.to_signature()
			for literal in method.context
			if literal.is_positive and literal.predicate in dynamic_predicates
		}
		available_positive = {
			literal.to_signature()
			for literal in method.context
			if literal.is_positive and literal.predicate in dynamic_predicates
		}
		task_preserved_dynamic_signatures = {
			literal.to_signature()
			for candidate_method in methods_by_task.get(task.name, ())
			if not candidate_method.subtasks
			for literal in candidate_method.context
			if literal.is_positive
			and literal.predicate in dynamic_predicates
			and not self._literal_matches_expected_signature(
				literal,
				expected_literal,
				method_bindings,
			)
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
				later_restorable_signatures = self._later_possible_positive_effect_signatures(
					ordered_steps[index + 1:headline_producer_index],
					methods_by_task,
					task_lookup,
					action_schemas,
					predicate_arities,
					dynamic_predicates=dynamic_predicates,
				)
				if step.kind == "compound":
					possible_negative_literals = tuple(
						self._possible_negative_step_effect_literals(
							step,
							methods_by_task,
							task_lookup,
							action_schemas,
							predicate_arities,
							dynamic_predicates=dynamic_predicates,
						),
					)
				else:
					possible_negative_literals = tuple(
						HTNLiteral(
							predicate=effect.predicate,
							args=effect.args,
							is_positive=True,
							source_symbol=None,
						)
						for effect in self._step_effect_literals(
							step,
							task_lookup,
							action_schemas,
							predicate_arities,
						)
						if not effect.is_positive and effect.predicate in dynamic_predicates
					)
				headline_step = ordered_steps[headline_producer_index]
				headline_precondition_literals = tuple(
					literal
					for literal in self._step_precondition_literals(headline_step, action_schemas)
					if literal.is_positive and literal.predicate in dynamic_predicates
				)
				available_positive_literals = tuple(
					literal
					for signature in available_positive
					for literal in (self._parse_signature_literal(signature),)
					if literal is not None
				)
				later_restorable_literals = tuple(
					literal
					for signature in later_restorable_signatures
					for literal in (self._parse_signature_literal(signature),)
					if literal is not None
				)
				destructive_signatures: set[str] = set()
				for negative_literal in possible_negative_literals:
					if not any(
						self._literal_may_unify_with_expected_signature(
							negative_literal,
							headline_precondition,
							method_bindings,
						)
						for headline_precondition in headline_precondition_literals
					):
						continue
					if not any(
						self._literal_may_unify_with_expected_signature(
							negative_literal,
							available_literal,
							method_bindings,
						)
						for available_literal in available_positive_literals
					):
						continue
					if any(
						self._literal_may_unify_with_expected_signature(
							negative_literal,
							restorable_literal,
							method_bindings,
						)
						for restorable_literal in later_restorable_literals
					):
						continue
					signature = negative_literal.to_signature()
					if self._signature_has_task_and_extra_role(
						signature,
						task_parameter_set,
					):
						destructive_signatures.add(signature)
				if destructive_signatures:
					raise ValueError(
						f"Method '{method.method_name}' inserts step '{step.step_id}' "
						f"({step.task_name}) before headline producer "
						f"'{headline_step.step_id}' ({headline_step.task_name}), but that earlier "
						f"step can invalidate later producer prerequisite(s) "
						f"{sorted(destructive_signatures)} with no restoring step before the "
						"producer runs. Keep such consumed mode-selector or shared literals "
						"available until the producer executes.",
					)
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
				unresolved_preservation_requirements = (
					task_preserved_dynamic_signatures - available_positive
				)
				relevant_unresolved_requirements = (
					unresolved_later_requirements | unresolved_preservation_requirements
				)
				if relevant_unresolved_requirements:
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
					if not possible_effects & relevant_unresolved_requirements:
						raise ValueError(
							f"Method '{method.method_name}' inserts compound step "
							f"'{step.step_id}' ({step.task_name}) before the headline "
							f"producer, but that step does not supply any unresolved later "
							"dynamic requirement or required preserved dynamic literal in this "
							f"branch. Later unresolved requirements are "
							f"{sorted(unresolved_later_requirements)}; preserved dynamic "
							f"obligations are {sorted(unresolved_preservation_requirements)}; "
							f"possible effects of '{step.task_name}' here are "
							f"{sorted(possible_effects)}.",
						)

				for literal in self._compound_step_carryover_positive_literals(
					step,
					methods_by_task,
					task_lookup,
					action_schemas,
					predicate_arities,
					dynamic_predicates=dynamic_predicates,
				):
					available_positive.add(literal.to_signature())

		for step_index, step in enumerate(ordered_steps):
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
						matching_support_steps = [
							earlier_step
							for earlier_step in ordered_steps[: step_index]
							if earlier_step.kind == "compound"
							and (
								child_task := task_lookup.get(earlier_step.task_name)
							) is not None
							and len(child_task.source_predicates) == 1
							and child_task.source_predicates[0] == literal.predicate
							and tuple(earlier_step.args) == tuple(literal.args)
						]
						if (
							literal.is_positive
							and literal.predicate in dynamic_predicates
							and signature in initial_context_positive
							and matching_support_steps
						):
							support_step = matching_support_steps[0]
							raise ValueError(
								f"Method '{method.method_name}' keeps supportable dynamic "
								f"prerequisite '{signature}' in context even though earlier "
								f"support step '{support_step.step_id}' ({support_step.task_name}) "
								"already handles that same literal in this branch. Split the "
								"already-supported and support-then-produce siblings instead of "
								"requiring the supported literal in the same branch context."
							)
						if (
							not literal.is_positive
							or literal.predicate not in dynamic_predicates
							or signature not in available_positive
							or signature in established_by_steps
							or self._signature_has_task_and_extra_role(
								signature,
								task_parameter_set,
							)
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
				self._possible_child_constructive_requirements(
					step,
					methods_by_task.get(step.task_name, ()),
					task_lookup,
					action_schemas,
					predicate_arities,
					dynamic_predicates=dynamic_predicates,
				),
			)
		return requirement_signatures

	def _compound_step_carryover_positive_literals(
		self,
		step: HTNMethodStep,
		methods_by_task: Dict[str, List[HTNMethod]],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		*,
		dynamic_predicates: set[str],
	) -> Tuple[HTNLiteral, ...]:
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
		if step.kind != "compound":
			return tuple(positive_literals)

		task_schema = task_lookup.get(step.task_name)
		if task_schema is None:
			return tuple(positive_literals)

		seen_signatures = {
			literal.to_signature()
			for literal in positive_literals
		}
		for method in methods_by_task.get(step.task_name, ()):
			if method.subtasks:
				continue
			task_args = self._default_method_task_args(method, task_schema)
			for literal in self._materialise_method_literals(
				method.context,
				task_args,
				step.args,
			):
				if not literal.is_positive or literal.predicate not in dynamic_predicates:
					continue
				signature = literal.to_signature()
				if signature in seen_signatures:
					continue
				seen_signatures.add(signature)
				positive_literals.append(literal)
		return tuple(positive_literals)

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

	def _possible_negative_step_effect_literals(
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
				HTNLiteral(
					predicate=literal.predicate,
					args=literal.args,
					is_positive=True,
					source_symbol=None,
				)
				for literal in self._step_effect_literals(
					step,
					task_lookup,
					action_schemas,
					predicate_arities,
				)
				if not literal.is_positive and literal.predicate in dynamic_predicates
			)

		negative_literals: list[HTNLiteral] = [
			HTNLiteral(
				predicate=literal.predicate,
				args=literal.args,
				is_positive=True,
				source_symbol=None,
			)
			for literal in self._step_effect_literals(
				step,
				task_lookup,
				action_schemas,
				predicate_arities,
			)
			if not literal.is_positive and literal.predicate in dynamic_predicates
		]
		task_schema = task_lookup.get(step.task_name)
		if task_schema is None or step.task_name in visited:
			return tuple(negative_literals)

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
				for literal in self._possible_negative_step_effect_literals(
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
						for existing in negative_literals
					):
						continue
					negative_literals.append(literal)
		return tuple(negative_literals)

	def _later_possible_positive_effect_signatures(
		self,
		steps: Sequence[HTNMethodStep],
		methods_by_task: Dict[str, List[HTNMethod]],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		*,
		dynamic_predicates: set[str],
	) -> set[str]:
		signatures: set[str] = set()
		for step in steps:
			signatures.update(
				literal.to_signature()
				for literal in self._possible_positive_step_effect_literals(
					step,
					methods_by_task,
					task_lookup,
					action_schemas,
					predicate_arities,
					dynamic_predicates=dynamic_predicates,
				)
			)
		return signatures

	def _signature_has_task_and_extra_role(
		self,
		signature: str,
		task_parameter_set: set[str],
	) -> bool:
		literal = self._literal_from_signature_text(signature)
		if literal is None or not literal.args:
			return False
		has_task_role = any(arg in task_parameter_set for arg in literal.args)
		has_extra_role = any(arg not in task_parameter_set for arg in literal.args)
		return has_task_role and has_extra_role

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
		local_aliases: Dict[str, str] = {}

		def materialise_symbol(symbol: str) -> str:
			if symbol in bindings:
				return bindings[symbol]
			if symbol in schema_parameters:
				return local_aliases.setdefault(
					symbol,
					f"__child_local_{len(local_aliases) + 1}_{symbol}",
				)
			return symbol

		return tuple(
			HTNLiteral(
				predicate=literal.predicate,
				args=tuple(materialise_symbol(arg) for arg in literal.args),
				is_positive=literal.is_positive,
				source_symbol=None,
			)
			for literal in literals
		)

	@staticmethod
	def _sanitize_name(name: str) -> str:
		return sanitize_identifier(name)

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
			task_lookup,
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
		task_lookup: Dict[str, HTNTask],
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
				if step.kind == "compound" and step.task_name in task_lookup:
					constrained_variables.add(variable)
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
		*,
		type_parent_map: Optional[Dict[str, Optional[str]]] = None,
	) -> None:
		resolved_type_parent_map = dict(type_parent_map or {})
		if "OBJECT" not in resolved_type_parent_map:
			resolved_type_parent_map["OBJECT"] = None

		symbol_types: Dict[str, set[str]] = {}
		task_schema = task_lookup.get(method.task_name)
		task_signature = task_types.get(method.task_name, ())
		for index, parameter in enumerate(method.parameters):
			if index >= len(task_signature):
				break
			symbol_types.setdefault(parameter, set()).add(task_signature[index])
		if not task_signature and task_schema:
			task_headline = self._materialise_task_headline_literal(task_schema)
			if task_headline is not None:
				predicate_signature = predicate_types.get(task_headline.predicate, ())
				for index, parameter in enumerate(task_headline.args):
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
				if not step_task_signature and step_task:
					step_headline = self._materialise_task_headline_literal(
						step_task,
						bound_args=step.args,
					)
					if step_headline is not None:
						predicate_signature = predicate_types.get(step_headline.predicate, ())
						for index, argument in enumerate(step_headline.args):
							if index < len(predicate_signature):
								symbol_types.setdefault(argument, set()).add(predicate_signature[index])

			for literal in (step.literal, *step.preconditions, *step.effects):
				if literal is None:
					continue
				self._collect_literal_types(symbol_types, literal, predicate_types)

		for candidates in symbol_types.values():
			for candidate_type in candidates:
				resolved_type_parent_map.setdefault(candidate_type, "OBJECT")

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
				self._resolve_compatible_method_symbol_type(
					method_name=method.method_name,
					symbol=symbol,
					candidate_types=candidates,
					type_parent_map=resolved_type_parent_map,
				)

	@staticmethod
	def _build_domain_type_parent_map(domain: Any) -> Dict[str, Optional[str]]:
		tokens = [
			str(token).strip().upper()
			for token in (getattr(domain, "types", []) or [])
			if str(token).strip()
		]
		if not tokens:
			return {"OBJECT": None}

		parent_map: Dict[str, Optional[str]] = {}
		pending_children: List[str] = []
		index = 0
		while index < len(tokens):
			token = tokens[index]
			if token == "-":
				if not pending_children or index + 1 >= len(tokens):
					raise ValueError("Malformed HDDL :types declaration (dangling '-').")
				parent_type = tokens[index + 1]
				for child_type in pending_children:
					previous = parent_map.get(child_type)
					if previous is not None and previous != parent_type:
						raise ValueError(
							f"Type '{child_type}' has conflicting parents "
							f"('{previous}' vs '{parent_type}').",
						)
					parent_map[child_type] = parent_type
				pending_children = []
				index += 2
				continue

			pending_children.append(token)
			index += 1

		for child_type in pending_children:
			parent_map.setdefault(child_type, "OBJECT")

		parent_map["OBJECT"] = None
		changed = True
		while changed:
			changed = False
			for parent_type in list(parent_map.values()):
				if parent_type is None or parent_type in parent_map:
					continue
				parent_map[parent_type] = "OBJECT" if parent_type != "OBJECT" else None
				changed = True

		for type_name in list(parent_map.keys()):
			if type_name == "OBJECT":
				parent_map[type_name] = None
				continue
			if parent_map[type_name] == type_name:
				raise ValueError(f"Type '{type_name}' cannot inherit from itself.")

			seen = {type_name}
			cursor = parent_map[type_name]
			while cursor is not None:
				if cursor in seen:
					raise ValueError(f"Cyclic type hierarchy detected at '{type_name}'.")
				seen.add(cursor)
				cursor = parent_map.get(cursor)

		return parent_map

	@staticmethod
	def _is_type_subtype(
		candidate_type: str,
		expected_type: str,
		type_parent_map: Dict[str, Optional[str]],
	) -> bool:
		if candidate_type == expected_type:
			return True
		if (
			candidate_type not in type_parent_map
			or expected_type not in type_parent_map
		):
			return False
		cursor = type_parent_map.get(candidate_type)
		visited = {candidate_type}
		while cursor is not None and cursor not in visited:
			if cursor == expected_type:
				return True
			visited.add(cursor)
			cursor = type_parent_map.get(cursor)
		return False

	@classmethod
	def _resolve_compatible_method_symbol_type(
		cls,
		*,
		method_name: str,
		symbol: str,
		candidate_types: set[str],
		type_parent_map: Dict[str, Optional[str]],
	) -> str:
		unknown_types = sorted(
			type_name
			for type_name in candidate_types
			if type_name not in type_parent_map
		)
		if unknown_types:
			raise ValueError(
				f"Method '{method_name}' symbol '{symbol}' references unknown types "
				f"{unknown_types}.",
			)

		feasible = sorted(
			type_name
			for type_name in type_parent_map
			if all(
				cls._is_type_subtype(type_name, required, type_parent_map)
				for required in candidate_types
			)
		)
		if not feasible:
			raise ValueError(
				f"Method '{method_name}' uses symbol '{symbol}' with conflicting "
				f"inferred types {sorted(candidate_types)}.",
			)

		most_general = sorted(
			type_name
			for type_name in feasible
			if not any(
				other != type_name
				and cls._is_type_subtype(type_name, other, type_parent_map)
				for other in feasible
			)
		)
		if len(most_general) != 1:
			raise ValueError(
				f"Method '{method_name}' symbol '{symbol}' is ambiguous under type "
				f"constraints {sorted(candidate_types)}; candidate schema types="
				f"{most_general}.",
			)
		return most_general[0]

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
		*,
		prompt_analysis: Optional[Dict[str, Any]] = None,
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
			return ()
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
		headline_literal = self._materialise_task_headline_literal(
			task_schema,
			bound_args=step.args,
			predicate_arities=predicate_arities,
		)
		if headline_literal is None:
			return ()
		return (headline_literal,)

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

	@classmethod
	def _literal_may_unify_with_expected_signature(
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

		local_unifier: Dict[str, str] = {}
		for raw_candidate_arg, expected_arg in zip(candidate.args, expected.args):
			if raw_candidate_arg in parameter_bindings:
				candidate_arg = parameter_bindings[raw_candidate_arg]
				if candidate_arg != expected_arg:
					return False
				continue
			candidate_arg = raw_candidate_arg
			bound_expected_arg = local_unifier.get(candidate_arg)
			if bound_expected_arg is None:
				local_unifier[candidate_arg] = expected_arg
				continue
			if bound_expected_arg != expected_arg:
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
			'"tasks":[{"name":"do_link","parameters":["A","B"],"headline":"linked(A, B)",'
			'"noop":{"precondition":["linked(A, B)"]},'
			'"constructive":[{"precondition":["ready(B)"],'
			'"support_before":["prepare(B)"],"producer":"attach(A, B)"}]}]}'
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
