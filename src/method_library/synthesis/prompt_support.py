"""
Compact domain-level prompt support helpers.

These helpers intentionally cover only the current domain-complete stage-owned
generation path. Query-conditioned prompt construction and other retired prompt
contract builders are no longer part of the mainline.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any, Dict, Iterable, Optional, Sequence

from utils.hddl_condition_parser import HDDLConditionParser


def _sanitize_name(name: str) -> str:
	return str(name or "").replace("-", "_")


def _literal_pattern_signature(literal: Any) -> str:
	atom = (
		literal.predicate
		if not literal.args
		else f"{literal.predicate}({', '.join(literal.args)})"
	)
	return atom if literal.is_positive else f"not {atom}"


def _task_invocation_signature(task_name: str, args: Sequence[str]) -> str:
	if not args:
		return f"{task_name}()"
	return f"{task_name}({', '.join(args)})"


def _typed_task_invocation_signature(task_name: str, parameters: Sequence[str]) -> str:
	if not parameters:
		return f"{task_name}()"
	rendered = [
		f"{_parameter_token(parameter)}:{_parameter_type(parameter)}"
		for parameter in parameters
	]
	return f"{task_name}({', '.join(rendered)})"


def _limited_unique(values: Iterable[Any], *, limit: int) -> list[str]:
	ordered: list[str] = []
	seen: set[str] = set()
	for value in values:
		text = str(value).strip()
		if not text or text in seen:
			continue
		seen.add(text)
		ordered.append(text)
		if len(ordered) >= max(int(limit), 1):
			break
	return ordered


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
	ordered: list[str] = []
	seen: set[str] = set()
	for value in values:
		text = str(value).strip()
		if not text or text in seen:
			continue
		seen.add(text)
		ordered.append(text)
	return ordered


def _parameter_token(parameter: str) -> str:
	return str(parameter).split("-", 1)[0].strip()


def _parameter_type(parameter: str) -> str:
	if "-" not in str(parameter):
		return "object"
	type_name = str(parameter).split("-", 1)[1].strip()
	return type_name or "object"


def _normalise_type_name(type_name: str) -> str:
	return _sanitize_name(str(type_name or "object")).lower()


def _normalised_type_parent_map(action_analysis: Dict[str, Any]) -> Dict[str, Optional[str]]:
	type_parent_map = dict(action_analysis.get("type_parent_map") or {})
	return {
		_normalise_type_name(type_name): (
			_normalise_type_name(parent_type) if parent_type is not None else None
		)
		for type_name, parent_type in type_parent_map.items()
	}


def _is_same_or_subtype(
	candidate_type: str,
	required_type: str,
	type_parent_map: Dict[str, Optional[str]],
) -> bool:
	normalized_candidate = _normalise_type_name(candidate_type)
	normalized_required = _normalise_type_name(required_type)
	if normalized_required in {"", "object"}:
		return True
	current_type: Optional[str] = normalized_candidate
	visited: set[str] = set()
	while current_type and current_type not in visited:
		if current_type == normalized_required:
			return True
		visited.add(current_type)
		current_type = type_parent_map.get(current_type)
	return False


def _signature_types_can_biject(
	task_parameter_types: Sequence[str],
	predicate_parameter_types: Sequence[str],
	type_parent_map: Dict[str, Optional[str]],
) -> bool:
	if len(predicate_parameter_types) > len(task_parameter_types):
		return False
	normalized_task_types = tuple(_normalise_type_name(type_name) for type_name in task_parameter_types)
	normalized_predicate_types = tuple(
		_normalise_type_name(type_name) for type_name in predicate_parameter_types
	)
	if not normalized_task_types:
		return True

	def assign(predicate_index: int, used_task_indices: set[int]) -> bool:
		if predicate_index >= len(normalized_predicate_types):
			return True
		required_type = normalized_predicate_types[predicate_index]
		for task_index, task_type in enumerate(normalized_task_types):
			if task_index in used_task_indices:
				continue
			if not _is_same_or_subtype(task_type, required_type, type_parent_map):
				continue
			next_used_task_indices = set(used_task_indices)
			next_used_task_indices.add(task_index)
			if assign(predicate_index + 1, next_used_task_indices):
				return True
		return False

	return assign(0, set())


def _predicate_type_signature_map(domain: Any) -> Dict[str, tuple[str, ...]]:
	return {
		str(predicate.name).strip(): tuple(
			_normalise_type_name(_parameter_type(parameter))
			for parameter in (getattr(predicate, "parameters", ()) or ())
		)
		for predicate in getattr(domain, "predicates", [])
		if str(getattr(predicate, "name", "")).strip()
	}


def _declared_task_schema_map(domain: Any) -> Dict[str, Any]:
	task_schemas: Dict[str, Any] = {}
	for task in getattr(domain, "tasks", []):
		task_name = str(getattr(task, "name", "")).strip()
		if not task_name:
			continue
		task_schemas[task_name] = task
		task_schemas[_sanitize_name(task_name)] = task
	return task_schemas


def _task_schema_can_align_to_predicate(
	task_schema: Any,
	predicate_name: str,
	*,
	predicate_type_signatures: Dict[str, tuple[str, ...]],
	type_parent_map: Dict[str, Optional[str]],
) -> bool:
	if task_schema is None:
		return False
	task_parameter_types = tuple(
		_normalise_type_name(_parameter_type(parameter))
		for parameter in (getattr(task_schema, "parameters", ()) or ())
	)
	predicate_parameter_types = tuple(predicate_type_signatures.get(predicate_name, ()))
	if not predicate_parameter_types:
		return False
	if len(predicate_parameter_types) > len(task_parameter_types):
		return False
	if _signature_types_can_biject(
		task_parameter_types,
		predicate_parameter_types,
		type_parent_map,
	):
		return True
	task_positions = range(len(task_parameter_types))
	for indices in combinations(task_positions, len(predicate_parameter_types)):
		candidate_types = tuple(task_parameter_types[index] for index in indices)
		if _signature_types_can_biject(
			candidate_types,
			predicate_parameter_types,
			type_parent_map,
		):
			return True
	return False


def _aligned_task_parameter_labels_for_predicate(
	predicate_name: str,
	task_parameters: Sequence[str],
	task_parameter_types: Sequence[str],
	action_analysis: Dict[str, Any],
	*,
	producer_pattern: Optional[Dict[str, Any]] = None,
) -> Optional[tuple[str, ...]]:
	labels = tuple(str(label) for label in task_parameters)
	normalized_types = tuple(_sanitize_name(str(type_name or "object")).lower() for type_name in task_parameter_types)
	if len(labels) != len(normalized_types):
		return labels
	patterns = (
		[producer_pattern]
		if producer_pattern is not None
		else list(action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, ()))
	)
	for pattern in patterns:
		if not isinstance(pattern, dict):
			continue
		effect_args = list(pattern.get("effect_args") or ())
		if not effect_args or len(effect_args) > len(labels):
			continue
		action_parameters = list(pattern.get("action_parameters") or ())
		action_parameter_types = list(pattern.get("action_parameter_types") or ())
		token_to_type = {
			token: _sanitize_name(str(parameter_type or "object")).lower()
			for token, parameter_type in zip(action_parameters, action_parameter_types)
		}
		used_indices: set[int] = set()
		aligned_labels: list[str] = []
		for effect_index, effect_token in enumerate(effect_args):
			chosen_index: Optional[int] = None
			effect_type = token_to_type.get(effect_token)
			if effect_type:
				matching_indices = [
					index
					for index, task_type in enumerate(normalized_types)
					if task_type == effect_type and index not in used_indices
				]
				if len(matching_indices) == 1:
					chosen_index = matching_indices[0]
				elif not matching_indices:
					return None
			if chosen_index is None and effect_index < len(labels) and effect_index not in used_indices:
				chosen_index = effect_index
			if chosen_index is None:
				for fallback_index in range(len(labels)):
					if fallback_index not in used_indices:
						chosen_index = fallback_index
						break
			if chosen_index is None:
				return None
			used_indices.add(chosen_index)
			aligned_labels.append(labels[chosen_index])
		return tuple(aligned_labels)
	return None


def _iter_unique_action_patterns(action_analysis: Dict[str, Any]) -> tuple[Dict[str, Any], ...]:
	seen: set[str] = set()
	patterns: list[Dict[str, Any]] = []
	for bucket_name in ("producer_patterns_by_predicate", "consumer_patterns_by_predicate"):
		for bucket in (action_analysis.get(bucket_name, {}) or {}).values():
			for pattern in bucket or ():
				if not isinstance(pattern, dict):
					continue
				action_name = str(pattern.get("action_name") or "").strip()
				if not action_name:
					continue
				serialized = json.dumps(
					{
						"action_name": action_name,
						"action_parameters": list(pattern.get("action_parameters") or ()),
						"action_parameter_types": list(pattern.get("action_parameter_types") or ()),
						"dynamic_preconditions": list(
							pattern.get("dynamic_precondition_signatures") or (),
						),
						"positive_effects": list(pattern.get("positive_effect_signatures") or ()),
					},
					sort_keys=True,
				)
				if serialized in seen:
					continue
				seen.add(serialized)
				patterns.append(dict(pattern))
	return tuple(patterns)


def _fallback_action_template_summaries_for_task(
	task_name: str,
	task_parameters: Sequence[str],
	task_parameter_types: Sequence[str],
	action_analysis: Dict[str, Any],
	*,
	limit: int = 3,
) -> list[str]:
	task_tokens = _name_tokens(task_name)
	type_parent_map = _normalised_type_parent_map(action_analysis)
	best_patterns_by_action_name: Dict[str, tuple[float, str]] = {}
	task_positions = range(len(task_parameters))
	for pattern in _iter_unique_action_patterns(action_analysis):
		action_name = str(pattern.get("action_name") or "").strip()
		action_parameters = list(pattern.get("action_parameters") or ())
		action_parameter_types = [
			_normalise_type_name(parameter_type)
			for parameter_type in (pattern.get("action_parameter_types") or ())
		]
		if not action_name or len(task_parameters) > len(action_parameters):
			continue
		best_rendered_action_args: Optional[tuple[str, ...]] = None
		best_alignment_score = float("-inf")
		for indices in combinations(task_positions, len(task_parameters)):
			candidate_parameter_types = tuple(
				action_parameter_types[index]
				for index in indices
			)
			if not _signature_types_can_biject(
				tuple(_normalise_type_name(task_type) for task_type in task_parameter_types),
				candidate_parameter_types,
				type_parent_map,
			):
				continue
			used_task_indices: set[int] = set()
			token_mapping: Dict[str, str] = {}
			for action_index in indices:
				action_type = _normalise_type_name(action_parameter_types[action_index])
				chosen_task_index: Optional[int] = None
				for task_index, task_type in enumerate(task_parameter_types):
					if task_index in used_task_indices:
						continue
					if _signature_types_can_biject(
						(_normalise_type_name(task_type),),
						(action_type,),
						type_parent_map,
					):
						chosen_task_index = task_index
						break
				if chosen_task_index is None:
					token_mapping = {}
					break
				used_task_indices.add(chosen_task_index)
				token_mapping[str(action_parameters[action_index])] = str(
					task_parameters[chosen_task_index]
				)
			if not token_mapping:
				continue
			rendered_action_args = tuple(
				_extend_mapping_with_action_parameters(
					token_mapping,
					action_parameters,
					action_parameter_types=action_parameter_types,
				)
			)
			extra_parameter_count = len(action_parameters) - len(task_parameters)
			alignment_score = (
				_weighted_token_overlap_score(
					task_tokens,
					_name_tokens(action_name),
					token_frequencies=_domain_name_token_frequencies(action_analysis),
				)
				- 0.25 * extra_parameter_count
			)
			if alignment_score > best_alignment_score:
				best_alignment_score = alignment_score
				best_rendered_action_args = rendered_action_args
		if best_rendered_action_args is None:
			continue
		rendered_requirements = _render_positive_dynamic_requirements(pattern, {
			parameter: value
			for parameter, value in zip(action_parameters, best_rendered_action_args)
		})
		rendered_effects = [
			_render_signature_with_mapping(signature, {
				parameter: value
				for parameter, value in zip(action_parameters, best_rendered_action_args)
			})
			for signature in (pattern.get("positive_effect_signatures") or ())
			if str(signature).strip()
		]
		involved_task_parameters = {
			arg
			for signature in (*rendered_requirements, *rendered_effects)
			for arg in (_parse_literal_signature(signature) or (None, (), False))[1]
			if arg in task_parameters
		}
		suffix = (
			f" [needs {', '.join(rendered_requirements)}]"
			if rendered_requirements
			else ""
		)
		final_score = best_alignment_score + 0.5 * len(involved_task_parameters)
		rendered_pattern = (
			final_score,
			f"{_task_invocation_signature(action_name, best_rendered_action_args)}{suffix}",
		)
		existing_pattern = best_patterns_by_action_name.get(action_name)
		if existing_pattern is None or rendered_pattern[0] > existing_pattern[0]:
			best_patterns_by_action_name[action_name] = rendered_pattern
	rendered_patterns = list(best_patterns_by_action_name.values())
	rendered_patterns.sort(key=lambda item: (-item[0], item[1]))
	return _limited_unique((pattern for _, pattern in rendered_patterns), limit=limit)


def _placeholder_stem(type_name: Optional[str]) -> str:
	text = re.sub(r"[^A-Za-z0-9]+", "_", str(type_name or "AUX").strip()).strip("_")
	if not text:
		return "AUX"
	text = text.upper()
	return text if text[0].isalpha() else f"AUX_{text}"


def _allocate_placeholder_label(
	used_labels: set[str],
	*,
	type_name: Optional[str] = None,
) -> str:
	stem = _placeholder_stem(type_name)
	if not stem.startswith("AUX_"):
		stem = f"AUX_{stem}"
	index = 1
	while True:
		candidate = f"{stem}{index}"
		if candidate not in used_labels:
			used_labels.add(candidate)
			return candidate
		index += 1


def _extend_mapping_with_action_parameters(
	token_mapping: Dict[str, str],
	action_parameters: Sequence[str],
	*,
	action_parameter_types: Sequence[str] = (),
) -> list[str]:
	rendered_action_args: list[str] = []
	used_labels = set(token_mapping.values())
	for index, action_parameter in enumerate(_parameter_token(parameter) for parameter in action_parameters):
		if action_parameter not in token_mapping:
			type_name = action_parameter_types[index] if index < len(action_parameter_types) else None
			token_mapping[action_parameter] = _allocate_placeholder_label(
				used_labels,
				type_name=type_name,
			)
		rendered_action_args.append(token_mapping[action_parameter])
	return rendered_action_args


def _normalise_action_analysis(
	domain: Any,
	action_analysis: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
	if action_analysis is not None:
		return {
			"dynamic_predicates": list(action_analysis.get("dynamic_predicates", ())),
			"static_predicates": list(action_analysis.get("static_predicates", ())),
			"producer_actions_by_predicate": dict(action_analysis.get("producer_actions_by_predicate", {})),
			"producer_patterns_by_predicate": dict(action_analysis.get("producer_patterns_by_predicate", {})),
			"consumer_actions_by_predicate": dict(action_analysis.get("consumer_actions_by_predicate", {})),
			"consumer_patterns_by_predicate": dict(action_analysis.get("consumer_patterns_by_predicate", {})),
			"type_parent_map": dict(action_analysis.get("type_parent_map", {})),
		}
	parser = HDDLConditionParser()

	def literal_signature(pattern: Any) -> str:
		atom = (
			pattern.predicate
			if not pattern.args
			else f"{pattern.predicate}({', '.join(pattern.args)})"
		)
		return atom if pattern.is_positive else f"not {atom}"

	dynamic_predicates: set[str] = set()
	producer_actions_by_predicate: Dict[str, list[str]] = {}
	producer_patterns_by_predicate: Dict[str, list[Dict[str, Any]]] = {}
	consumer_actions_by_predicate: Dict[str, list[str]] = {}
	consumer_patterns_by_predicate: Dict[str, list[Dict[str, Any]]] = {}
	parsed_actions: list[Any] = []
	for action in getattr(domain, "actions", []):
		try:
			parsed_action = parser.parse_action(action)
		except Exception:
			continue
		parsed_actions.append(parsed_action)
		for effect in parsed_action.effects:
			if effect.predicate != "=":
				dynamic_predicates.add(effect.predicate)
	for parsed_action in parsed_actions:
		action_name = _sanitize_name(parsed_action.name)
		action_parameter_types = [
			_parameter_type(parameter)
			for parameter in (parsed_action.parameters or ())
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
			if precondition.predicate == "=" or not precondition.is_positive:
				continue
			if precondition.predicate not in dynamic_predicates:
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
	all_predicates = {
		str(predicate.name).strip()
		for predicate in getattr(domain, "predicates", [])
		if str(getattr(predicate, "name", "")).strip()
	}
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
	return {
		"dynamic_predicates": sorted(dynamic_predicates),
		"static_predicates": sorted(all_predicates - dynamic_predicates),
		"producer_actions_by_predicate": {
			predicate_name: sorted(dict.fromkeys(producer_actions_by_predicate.get(predicate_name, ())))
			for predicate_name in sorted(dynamic_predicates)
		},
		"producer_patterns_by_predicate": {
			predicate_name: producer_patterns_by_predicate.get(predicate_name, [])
			for predicate_name in sorted(dynamic_predicates)
		},
		"consumer_actions_by_predicate": {
			predicate_name: sorted(dict.fromkeys(consumer_actions_by_predicate.get(predicate_name, ())))
			for predicate_name in sorted(dynamic_predicates)
		},
		"consumer_patterns_by_predicate": {
			predicate_name: consumer_patterns_by_predicate.get(predicate_name, [])
			for predicate_name in sorted(dynamic_predicates)
		},
		"type_parent_map": {},
	}


def _reusable_dynamic_resource_payloads(action_analysis: Dict[str, Any]) -> list[Dict[str, Any]]:
	resource_payloads: list[Dict[str, Any]] = []
	for predicate_name in action_analysis.get("dynamic_predicates", ()):
		producer_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, ())
		consumer_patterns = action_analysis.get("consumer_patterns_by_predicate", {}).get(predicate_name, ())
		if not producer_patterns or not consumer_patterns:
			continue
		toggled_by_consumers = any(
			any(
				str(signature).startswith(f"not {predicate_name}")
				for signature in (pattern.get("negative_effect_signatures") or ())
			)
			for pattern in consumer_patterns
		)
		has_zero_ary_mode = any(
			not list(pattern.get("effect_args") or ())
			for pattern in producer_patterns
		) or any(
			not list(pattern.get("precondition_args") or ())
			for pattern in consumer_patterns
		)
		if not toggled_by_consumers and not has_zero_ary_mode:
			continue
		resource_payloads.append(
			{
				"predicate": predicate_name,
				"producer_actions": list(action_analysis.get("producer_actions_by_predicate", {}).get(predicate_name, ())),
				"consumer_actions": list(action_analysis.get("consumer_actions_by_predicate", {}).get(predicate_name, ())),
			}
		)
	return resource_payloads


def _name_tokens(name: str) -> tuple[str, ...]:
	parts = re.split(r"[^a-z0-9]+", _sanitize_name(name).lower())
	return tuple(
		part
		for part in parts
		if part and part not in {"do", "task", "method", "abs", "get", "send", "data", "communicated"}
	)


def _domain_name_token_frequencies(action_analysis: Dict[str, Any]) -> Dict[str, int]:
	counts: Dict[str, int] = {}
	for predicate_name, patterns in (action_analysis.get("producer_patterns_by_predicate", {}) or {}).items():
		for token in set(_name_tokens(predicate_name)):
			counts[token] = counts.get(token, 0) + 1
		for pattern in patterns or ():
			action_name = str(pattern.get("action_name") or "").strip()
			for token in set(_name_tokens(action_name)):
				counts[token] = counts.get(token, 0) + 1
	return counts


def _weighted_token_overlap_score(
	left: Sequence[str],
	right: Sequence[str],
	*,
	token_frequencies: Optional[Dict[str, int]] = None,
) -> float:
	score = 0.0
	frequencies = token_frequencies or {}
	for left_token in left:
		for right_token in right:
			token_weight = 1.0 / max(frequencies.get(left_token, 1), frequencies.get(right_token, 1))
			if left_token == right_token:
				score += 4.0 * token_weight
			elif min(len(left_token), len(right_token)) >= 4:
				if left_token.startswith(right_token) or right_token.startswith(left_token):
					score += 2.0 * token_weight
	return score


def _compact_name_tokens(name: str) -> str:
	return "".join(_name_tokens(name))


def _candidate_support_task_names(
	domain: Any,
	predicate_name: str,
	predicate_args: Sequence[str],
	producer_actions: Sequence[str],
) -> list[str]:
	predicate_tokens = _name_tokens(predicate_name)
	predicate_compact = _compact_name_tokens(predicate_name)
	action_compacts = [
		_compact_name_tokens(action_name)
		for action_name in producer_actions
		if _compact_name_tokens(action_name)
	]
	producer_action_type_signatures = [
		tuple(_normalise_type_name(_parameter_type(parameter)) for parameter in getattr(action, "parameters", ()) or ())
		for action in getattr(domain, "actions", [])
		if str(getattr(action, "name", "")).strip() in set(producer_actions)
	]
	candidates: list[tuple[float, str]] = []
	for task in getattr(domain, "tasks", []):
		task_name = str(getattr(task, "name", "")).strip()
		task_tokens = _name_tokens(task_name)
		if not task_tokens:
			continue
		task_compact = _compact_name_tokens(task_name)
		task_type_signature = tuple(
			_normalise_type_name(_parameter_type(parameter))
			for parameter in getattr(task, "parameters", ()) or ()
		)
		predicate_overlap = _weighted_token_overlap_score(task_tokens, predicate_tokens)
		action_overlap = 0.0
		if task_compact:
			for action_compact in action_compacts:
				if not action_compact:
					continue
				if task_compact == action_compact:
					action_overlap = max(action_overlap, 3.0)
				elif min(len(task_compact), len(action_compact)) >= 4:
					if (
						task_compact.endswith(action_compact)
						or action_compact.endswith(task_compact)
						or task_compact in action_compact
						or action_compact in task_compact
					):
						action_overlap = max(action_overlap, 2.0)
		type_overlap = 0.0
		for producer_type_signature in producer_action_type_signatures:
			if not producer_type_signature or not task_type_signature:
				continue
			if sorted(task_type_signature) == sorted(producer_type_signature):
				type_overlap = max(type_overlap, 2.5)
			elif len(task_type_signature) >= len(producer_type_signature) and all(
				type_name in task_type_signature for type_name in producer_type_signature
			):
				type_overlap = max(type_overlap, 1.5)
		if predicate_overlap <= 0 and action_overlap <= 0 and type_overlap <= 0:
			continue
		if predicate_overlap <= 0 and predicate_compact and task_compact:
			if (
				task_compact != predicate_compact
				and predicate_compact not in task_compact
				and task_compact not in predicate_compact
				and action_overlap < 2.0
				and type_overlap < 2.5
			):
				continue
		score = (2.0 * predicate_overlap) + action_overlap + type_overlap
		if len(getattr(task, "parameters", ()) or ()) == len(predicate_args):
			score += 1.0
		candidates.append((score, task_name))
	candidates.sort(key=lambda item: (-item[0], item[1]))
	return [name for _, name in candidates[:4]]


def _dynamic_support_candidate_map(
	domain: Any,
	action_analysis: Dict[str, Any],
) -> Dict[str, list[str]]:
	parser = HDDLConditionParser()
	producer_actions = action_analysis.get("producer_actions_by_predicate", {})
	candidates_by_predicate: Dict[str, list[str]] = {}
	for action in getattr(domain, "actions", []):
		try:
			parsed_action = parser.parse_action(action)
		except Exception:
			continue
		for precondition in parsed_action.preconditions:
			if precondition.predicate == "=" or precondition.predicate not in action_analysis["dynamic_predicates"]:
				continue
			candidates = _candidate_support_task_names(
				domain,
				precondition.predicate,
				precondition.args,
				producer_actions.get(precondition.predicate, ()),
			)
			if not candidates:
				continue
			bucket = candidates_by_predicate.setdefault(precondition.predicate, [])
			for candidate in candidates:
				if candidate not in bucket:
					bucket.append(candidate)
	return candidates_by_predicate


def _dynamic_support_hint_lines(
	domain: Any,
	action_analysis: Dict[str, Any],
) -> list[str]:
	lines: list[str] = []
	for predicate_name, candidates in _dynamic_support_candidate_map(domain, action_analysis).items():
		for candidate in candidates:
			lines.append(f"- {predicate_name} likely reusable declared tasks: {candidate}")
	return lines


def _same_arity_declared_task_candidates(
	domain: Any,
	task_name: str,
	task_schemas: Dict[str, Any],
) -> list[str]:
	task_schema = task_schemas.get(task_name)
	if task_schema is None:
		return []
	task_parameter_types = [
		_parameter_type(parameter)
		for parameter in getattr(task_schema, "parameters", ()) or ()
	]
	return [
		str(getattr(other_task, "name", "")).strip()
		for other_task in getattr(domain, "tasks", [])
		if str(getattr(other_task, "name", "")).strip() != task_name
		and len(getattr(other_task, "parameters", ()) or ()) == len(getattr(task_schema, "parameters", ()) or ())
		and [
			_parameter_type(parameter)
			for parameter in getattr(other_task, "parameters", ()) or ()
		] == task_parameter_types
	]


def _parse_literal_signature(signature: str) -> Optional[tuple[str, tuple[str, ...], bool]]:
	text = str(signature or "").strip()
	is_positive = True
	if text.startswith("!"):
		is_positive = False
		text = text[1:].strip()
	if "(" not in text or not text.endswith(")"):
		return text, (), is_positive
	predicate, raw_args = text[:-1].split("(", 1)
	args = tuple(part.strip() for part in raw_args.split(",") if part.strip())
	return predicate.strip(), args, is_positive


def _render_signature_with_mapping(signature: str, token_mapping: Dict[str, str]) -> str:
	if not token_mapping:
		return signature
	pattern = re.compile(
		r"(?<![A-Za-z0-9_])("
		+ "|".join(sorted((re.escape(token) for token in token_mapping), key=len, reverse=True))
		+ r")(?![A-Za-z0-9_])",
	)
	return pattern.sub(lambda match: token_mapping[match.group(1)], signature)


def _shared_dynamic_requirements_for_predicate(
	predicate_name: str,
	predicate_args: Sequence[str],
	action_analysis: Dict[str, Any],
	*,
	predicate_arg_types: Sequence[str] = (),
) -> tuple[str, ...]:
	patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, ())
	if not patterns:
		return ()
	requirement_sets: list[set[str]] = []
	for pattern in patterns:
		effect_args = list(pattern.get("effect_args") or ())
		if len(effect_args) != len(predicate_args):
			continue
		aligned_args = (
			_aligned_task_parameter_labels_for_predicate(
				predicate_name,
				predicate_args,
				predicate_arg_types,
				action_analysis,
				producer_pattern=pattern,
			)
			if predicate_arg_types
			else tuple(str(arg) for arg in predicate_args)
		)
		if aligned_args is None:
			continue
		token_mapping = {token: arg for token, arg in zip(effect_args, aligned_args)}
		_extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or (),
			action_parameter_types=pattern.get("action_parameter_types") or (),
		)
		requirement_sets.append(
			{
				_render_signature_with_mapping(signature, token_mapping)
				for signature in (pattern.get("dynamic_precondition_signatures") or ())
				if not str(signature).startswith("not ")
			}
		)
	if not requirement_sets:
		return ()
	return tuple(sorted(set.intersection(*requirement_sets)))


def _render_positive_dynamic_requirements(
	pattern: Dict[str, Any],
	token_mapping: Dict[str, str],
) -> list[str]:
	requirements: list[str] = []
	for signature in pattern.get("dynamic_precondition_signatures") or ():
		rendered_signature = _render_signature_with_mapping(signature, token_mapping)
		parsed_signature = _parse_literal_signature(rendered_signature)
		if parsed_signature is None or not parsed_signature[2]:
			continue
		requirements.append(rendered_signature)
	return requirements


def _render_positive_static_requirements(
	pattern: Dict[str, Any],
	token_mapping: Dict[str, str],
) -> list[str]:
	dynamic_signatures = set(pattern.get("dynamic_precondition_signatures") or ())
	requirements: list[str] = []
	for signature in pattern.get("precondition_signatures") or ():
		if signature in dynamic_signatures:
			continue
		rendered_signature = _render_signature_with_mapping(signature, token_mapping)
		parsed_signature = _parse_literal_signature(rendered_signature)
		if parsed_signature is None or not parsed_signature[2]:
			continue
		requirements.append(rendered_signature)
	return requirements


def _constructive_template_summary_for_task(
	display_name: str,
	task_parameters: Sequence[str],
	predicate_name: str,
	action_analysis: Dict[str, Any],
	*,
	task_parameter_types: Sequence[str] = (),
) -> Optional[str]:
	patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, ())
	if not patterns:
		return None
	rendered_patterns: list[str] = []
	for pattern in patterns:
		effect_args = list(pattern.get("effect_args") or ())
		if not effect_args or len(effect_args) > len(task_parameters):
			continue
		aligned_task_parameters = (
			_aligned_task_parameter_labels_for_predicate(
				predicate_name,
				task_parameters,
				task_parameter_types,
				action_analysis,
				producer_pattern=pattern,
			)
			if task_parameter_types
			else tuple(str(parameter) for parameter in task_parameters)
		)
		if aligned_task_parameters is None:
			continue
		token_mapping = {
			token: task_parameter
			for token, task_parameter in zip(effect_args, aligned_task_parameters)
		}
		rendered_action_args = _extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or (),
			action_parameter_types=pattern.get("action_parameter_types") or (),
		)
		rendered_requirements = _render_positive_dynamic_requirements(pattern, token_mapping)
		extra_role_preconditions: list[str] = []
		shared_role_preconditions: list[str] = []
		for rendered_signature in rendered_requirements:
			parsed_signature = _parse_literal_signature(rendered_signature)
			if parsed_signature is None:
				continue
			_, rendered_args, _ = parsed_signature
			if rendered_args and set(rendered_args).issubset(set(task_parameters)):
				shared_role_preconditions.append(rendered_signature)
			else:
				extra_role_preconditions.append(rendered_signature)
		rendered_call = _task_invocation_signature(pattern["action_name"], rendered_action_args)
		if shared_role_preconditions and extra_role_preconditions:
			suffix = (
				f" [needs {', '.join(shared_role_preconditions)}; "
				f"extra needs {', '.join(extra_role_preconditions)}]"
			)
		elif extra_role_preconditions:
			suffix = f" [extra needs {', '.join(extra_role_preconditions)}]"
		elif shared_role_preconditions:
			suffix = f" [needs {', '.join(shared_role_preconditions)}]"
		else:
			suffix = ""
		rendered_patterns.append(f"{rendered_call}{suffix}")
	if not rendered_patterns:
		return None
	return "; ".join(rendered_patterns)


def _filter_dominated_producer_modes(
	rendered_modes: Sequence[tuple[str, tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...]]]:
	filtered_modes: list[tuple[str, tuple[str, ...]]] = []
	need_sets = [
		{str(signature).strip() for signature in needs if str(signature).strip()}
		for _, needs in rendered_modes
	]
	for index, mode in enumerate(rendered_modes):
		current_need_set = need_sets[index]
		if any(other_index != index and need_sets[other_index] < current_need_set for other_index in range(len(rendered_modes))):
			continue
		filtered_modes.append(mode)
	return filtered_modes


def _render_producer_mode_options_for_predicate(
	predicate_name: str,
	predicate_args: Sequence[str],
	action_analysis: Dict[str, Any],
	*,
	limit: int = 3,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
	rendered_modes: list[tuple[str, tuple[str, ...]]] = []
	seen: set[tuple[str, tuple[str, ...]]] = set()
	target_signature = predicate_name if not predicate_args else f"{predicate_name}({', '.join(predicate_args)})"
	for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, ()):
		effect_args = list(pattern.get("effect_args") or ())
		if len(effect_args) != len(predicate_args):
			continue
		token_mapping = {token: arg for token, arg in zip(effect_args, predicate_args)}
		rendered_action_args = _extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or (),
			action_parameter_types=pattern.get("action_parameter_types") or (),
		)
		rendered_requirements = tuple(_render_positive_dynamic_requirements(pattern, token_mapping))
		if target_signature in rendered_requirements:
			continue
		mode = (
			_task_invocation_signature(pattern["action_name"], rendered_action_args),
			rendered_requirements,
		)
		if mode in seen:
			continue
		seen.add(mode)
		rendered_modes.append(mode)
	return tuple(_filter_dominated_producer_modes(rendered_modes)[:limit])


def _candidate_headline_predicates_for_task(
	task_name: str,
	task_arity: int,
	action_analysis: Dict[str, Any],
) -> list[str]:
	task_tokens = _name_tokens(task_name)
	task_compact = _compact_name_tokens(task_name)
	token_frequencies = _domain_name_token_frequencies(action_analysis)
	exact_arity_predicates: list[tuple[float, str]] = []
	exact_arity_direct_predicate_matches: list[tuple[float, str]] = []
	subset_arity_predicates: list[tuple[float, str]] = []
	for predicate_name, patterns in action_analysis.get("producer_patterns_by_predicate", {}).items():
		compatible_patterns = [
			pattern
			for pattern in patterns
			if 0 < len(pattern.get("effect_args") or ()) <= task_arity
		]
		if not compatible_patterns:
			continue
		predicate_overlap = _weighted_token_overlap_score(
			task_tokens,
			_name_tokens(predicate_name),
			token_frequencies=token_frequencies,
		)
		score = predicate_overlap
		predicate_compact = _compact_name_tokens(predicate_name)
		if task_compact and predicate_compact:
			if task_compact == predicate_compact:
				score += 6
			elif min(len(task_compact), len(predicate_compact)) >= 4:
				if task_compact.endswith(predicate_compact) or predicate_compact.endswith(task_compact):
					score += 4
				elif predicate_compact in task_compact or task_compact in predicate_compact:
					score += 2
		for pattern in compatible_patterns:
			score += _weighted_token_overlap_score(
				task_tokens,
				_name_tokens(str(pattern.get("action_name") or "")),
				token_frequencies=token_frequencies,
			)
		has_exact_arity = any(len(pattern.get("effect_args") or ()) == task_arity for pattern in compatible_patterns)
		score += 2.5 if has_exact_arity else 0.25
		if score <= 0:
			continue
		if has_exact_arity:
			exact_arity_predicates.append((score, predicate_name))
			if predicate_overlap > 0:
				exact_arity_direct_predicate_matches.append((score, predicate_name))
		else:
			subset_arity_predicates.append((score, predicate_name))
	if exact_arity_direct_predicate_matches:
		scored_predicates = exact_arity_direct_predicate_matches
	elif exact_arity_predicates and subset_arity_predicates:
		best_exact_score = max(score for score, _ in exact_arity_predicates)
		best_subset_score = max(score for score, _ in subset_arity_predicates)
		scored_predicates = exact_arity_predicates if best_exact_score >= best_subset_score else subset_arity_predicates
	else:
		scored_predicates = exact_arity_predicates or subset_arity_predicates
	scored_predicates.sort(key=lambda item: (-item[0], item[1]))
	return [predicate_name for _, predicate_name in scored_predicates]


def _task_headline_candidate_map(
	domain: Any,
	action_analysis: Dict[str, Any],
) -> Dict[str, list[str]]:
	task_schemas = _declared_task_schema_map(domain)
	predicate_type_signatures = _predicate_type_signature_map(domain)
	type_parent_map = _normalised_type_parent_map(action_analysis)
	mapping: Dict[str, list[str]] = {}
	for task in getattr(domain, "tasks", []):
		task_name = _sanitize_name(str(getattr(task, "name", "")))
		task_schema = task_schemas.get(str(getattr(task, "name", "")).strip())
		source_predicates = [
			str(predicate_name).strip()
			for predicate_name in (getattr(task, "source_predicates", ()) or ())
			if str(predicate_name).strip()
		]
		raw_candidates = source_predicates or _candidate_headline_predicates_for_task(
			str(getattr(task, "name", "")),
			len(getattr(task, "parameters", ()) or ()),
			action_analysis,
		)
		typed_candidates = [
			predicate_name
			for predicate_name in raw_candidates
			if _task_schema_can_align_to_predicate(
				task_schema,
				predicate_name,
				predicate_type_signatures=predicate_type_signatures,
				type_parent_map=type_parent_map,
			)
		]
		mapping[task_name] = _unique_preserve_order(typed_candidates)[:1]
	return mapping


def _relevant_action_names_for_prompt(
	relevant_predicates: Sequence[str],
	action_analysis: Dict[str, Any],
) -> tuple[str, ...]:
	action_names: set[str] = set()
	for predicate_name in relevant_predicates:
		for patterns_key in ("producer_patterns_by_predicate", "consumer_patterns_by_predicate"):
			for pattern in action_analysis.get(patterns_key, {}).get(predicate_name, ()):
				action_name = _sanitize_name(str(pattern.get("action_name") or "").strip())
				if action_name:
					action_names.add(action_name)
		for actions_key in ("producer_actions_by_predicate", "consumer_actions_by_predicate"):
			for action_name in action_analysis.get(actions_key, {}).get(predicate_name, ()):
				sanitized = _sanitize_name(str(action_name).strip())
				if sanitized:
					action_names.add(sanitized)
	return tuple(sorted(action_names))


def _format_tagged_block(tag_name: str, body: str) -> str:
	content = str(body or "").strip()
	if not content:
		return ""
	return f"<{tag_name}>\\n{content}\\n</{tag_name}>"
