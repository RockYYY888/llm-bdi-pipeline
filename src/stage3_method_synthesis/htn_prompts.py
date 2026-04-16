"""
Prompt builders for Stage 3 HTN method synthesis.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Optional, Sequence

from utils.hddl_condition_parser import HDDLConditionParser


def _sanitize_name(name: str) -> str:
	return name.replace("-", "_")


def _literal_pattern_signature(literal: Any) -> str:
	atom = (
		literal.predicate
		if not literal.args
		else f"{literal.predicate}({', '.join(literal.args)})"
	)
	return atom if literal.is_positive else f"not {atom}"


def _clause_signature(clause: Iterable[Any]) -> str:
	parts = [_literal_pattern_signature(item) for item in clause]
	return " & ".join(parts) if parts else "true"


def _task_invocation_signature(task_name: str, args: Sequence[str]) -> str:
	if not args:
		return f"{task_name}()"
	return f"{task_name}({', '.join(args)})"


def _parameter_token(parameter: str) -> str:
	return str(parameter).split("-", 1)[0].strip()


def _parameter_type(parameter: str) -> str:
	if "-" not in str(parameter):
		return "object"
	type_name = str(parameter).split("-", 1)[1].strip()
	return type_name or "object"


def _generic_parameter_symbols(arity: int) -> tuple[str, ...]:
	base_symbols = ("X", "Y", "Z", "W")
	if arity <= len(base_symbols):
		return base_symbols[:arity]
	return tuple(f"X{index}" for index in range(1, arity + 1))


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
	for index, action_parameter in enumerate(
		_parameter_token(parameter)
		for parameter in action_parameters
	):
		if action_parameter not in token_mapping:
			type_name = (
				action_parameter_types[index]
				if index < len(action_parameter_types)
				else None
			)
			token_mapping[action_parameter] = _allocate_placeholder_label(
				used_labels,
				type_name=type_name,
			)
		rendered_action_args.append(token_mapping[action_parameter])
	return rendered_action_args


def _anchor_display_name(anchor: Dict[str, Any]) -> str:
	return str(anchor.get("source_name") or anchor.get("task_name") or "").strip()


def _declared_task_schema_map(domain: Any) -> Dict[str, Any]:
	task_schemas: Dict[str, Any] = {}
	for task in getattr(domain, "tasks", []):
		task_schemas[str(task.name)] = task
		task_schemas[_sanitize_name(str(task.name))] = task
	return task_schemas


def _normalise_action_analysis(
	domain: Any,
	action_analysis: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
	if action_analysis is not None:
		return {
			"dynamic_predicates": list(action_analysis.get("dynamic_predicates", [])),
			"static_predicates": list(action_analysis.get("static_predicates", [])),
			"producer_actions_by_predicate": dict(
				action_analysis.get("producer_actions_by_predicate", {}),
			),
			"producer_patterns_by_predicate": dict(
				action_analysis.get("producer_patterns_by_predicate", {}),
			),
			"consumer_actions_by_predicate": dict(
				action_analysis.get("consumer_actions_by_predicate", {}),
			),
			"consumer_patterns_by_predicate": dict(
				action_analysis.get("consumer_patterns_by_predicate", {}),
			),
		}

	parser = HDDLConditionParser()
	dynamic_predicates: set[str] = set()
	producer_actions_by_predicate: Dict[str, list[str]] = {}
	consumer_actions_by_predicate: Dict[str, list[str]] = {}
	parsed_actions: list[Any] = []

	for action in getattr(domain, "actions", []):
		try:
			parsed_action = parser.parse_action(action)
		except Exception:
			continue
		parsed_actions.append(parsed_action)
		for effect in parsed_action.effects:
			if effect.predicate == "=":
				continue
			dynamic_predicates.add(effect.predicate)

	for parsed_action in parsed_actions:
		action_name = _sanitize_name(parsed_action.name)
		for effect in parsed_action.effects:
			if effect.predicate == "=" or not effect.is_positive:
				continue
			producer_actions_by_predicate.setdefault(effect.predicate, []).append(
				action_name,
			)
		for precondition in parsed_action.preconditions:
			if precondition.predicate == "=":
				continue
			if precondition.predicate not in dynamic_predicates or not precondition.is_positive:
				continue
			consumer_actions_by_predicate.setdefault(precondition.predicate, []).append(
				action_name,
			)

	for predicate_name, producer_actions in list(producer_actions_by_predicate.items()):
		producer_actions_by_predicate[predicate_name] = sorted(dict.fromkeys(producer_actions))
	for predicate_name, consumer_actions in list(consumer_actions_by_predicate.items()):
		consumer_actions_by_predicate[predicate_name] = sorted(dict.fromkeys(consumer_actions))

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
			predicate_name: []
			for predicate_name in sorted(dynamic_predicates)
		},
		"consumer_actions_by_predicate": {
			predicate_name: consumer_actions_by_predicate.get(predicate_name, [])
			for predicate_name in sorted(dynamic_predicates)
		},
		"consumer_patterns_by_predicate": {
			predicate_name: []
			for predicate_name in sorted(dynamic_predicates)
		},
	}


def _normalise_query_object_inventory(
	query_object_inventory: Optional[Sequence[Dict[str, Any]]],
) -> tuple[Dict[str, Any], ...]:
	if not query_object_inventory:
		return ()

	entries: list[Dict[str, Any]] = []
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


def _query_object_names_from_inventory(
	query_object_inventory: Sequence[Dict[str, Any]],
) -> tuple[str, ...]:
	ordered_names: list[str] = []
	seen: set[str] = set()
	for entry in query_object_inventory:
		for item in entry.get("objects", []):
			object_name = str(item).strip()
			if not object_name or object_name in seen:
				continue
			seen.add(object_name)
			ordered_names.append(object_name)
	return tuple(ordered_names)


def _render_consumer_template_summary_for_task(
	display_name: str,
	task_parameters: Sequence[str],
	predicate_name: str,
	action_analysis: Dict[str, Any],
) -> Optional[str]:
	patterns = action_analysis.get("consumer_patterns_by_predicate", {}).get(predicate_name, [])
	if not patterns:
		return None

	rendered_patterns: list[str] = []
	for pattern in patterns:
		precondition_args = list(pattern.get("precondition_args") or [])
		if len(precondition_args) != len(task_parameters):
			continue
		token_mapping = {
			token: task_parameter
			for token, task_parameter in zip(precondition_args, task_parameters)
		}
		rendered_action_args = _extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or [],
			action_parameter_types=pattern.get("action_parameter_types") or [],
		)
		rendered_signature = _render_signature_with_mapping(
			pattern.get("precondition_signature") or predicate_name,
			token_mapping,
		)
		other_dynamic_preconditions = [
			_render_signature_with_mapping(signature, token_mapping)
			for signature in (pattern.get("other_dynamic_precondition_signatures") or [])
			if not str(signature).startswith("not ")
		]
		rendered_patterns.append(
			f"{_task_invocation_signature(pattern['action_name'], rendered_action_args)} consumes "
			f"{rendered_signature}; sibling dynamic preconditions: "
			f"{', '.join(other_dynamic_preconditions) if other_dynamic_preconditions else 'none'}"
		)

	if not rendered_patterns:
		return None
	return (
		f"{_task_invocation_signature(display_name, task_parameters)} consumer templates: "
		f"{'; '.join(rendered_patterns)}"
	)


def _reusable_dynamic_resource_payloads(action_analysis: Dict[str, Any]) -> list[Dict[str, Any]]:
	resource_payloads: list[Dict[str, Any]] = []
	for predicate_name in action_analysis.get("dynamic_predicates", []):
		producer_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		consumer_patterns = action_analysis.get("consumer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		if not producer_patterns or not consumer_patterns:
			continue
		toggled_by_consumers = any(
			any(
				str(signature).startswith(f"not {predicate_name}")
				or str(signature) == f"not {predicate_name}"
				for signature in (pattern.get("negative_effect_signatures") or [])
			)
			for pattern in consumer_patterns
		)
		has_zero_ary_mode = any(
			not list(pattern.get("effect_args") or [])
			for pattern in producer_patterns
		) or any(
			not list(pattern.get("precondition_args") or [])
			for pattern in consumer_patterns
		)
		if not toggled_by_consumers and not has_zero_ary_mode:
			continue
		resource_payloads.append(
			{
				"predicate": predicate_name,
				"producer_actions": list(
					action_analysis.get("producer_actions_by_predicate", {}).get(
						predicate_name,
						[],
					),
				),
				"consumer_actions": list(
					action_analysis.get("consumer_actions_by_predicate", {}).get(
						predicate_name,
						[],
					),
				),
			},
		)
	return resource_payloads


def build_prompt_analysis_payload(
	domain: Any,
	*,
	target_literals: Sequence[str] = (),
	query_task_anchors: Sequence[Dict[str, Any]] = (),
	action_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	analysis = _normalise_action_analysis(domain, action_analysis)
	shared_dynamic_prerequisites_by_task: Dict[str, list[str]] = {}
	producer_consumer_templates_by_task: Dict[str, list[str]] = {}
	task_headline_candidates = _task_headline_candidate_map(domain, analysis)

	for task in getattr(domain, "tasks", []):
		task_name = _sanitize_name(str(task.name))
		task_parameters = tuple(_parameter_token(parameter) for parameter in task.parameters)
		source_predicates = tuple(
			dict.fromkeys(
				(
					str(predicate_name)
					for predicate_name in (getattr(task, "source_predicates", ()) or ())
					if str(predicate_name).strip()
				)
			)
		) or tuple(task_headline_candidates.get(task_name, ())[:2])
		if not task_parameters or not source_predicates:
			continue

		shared_requirements: list[str] = []
		template_summaries: list[str] = []
		for predicate_name in source_predicates:
			shared_requirements.extend(
				requirement
				for requirement in _shared_dynamic_requirements_for_predicate(
					predicate_name,
					task_parameters,
					analysis,
				)
				if requirement not in shared_requirements
			)
			constructive_template = _constructive_template_summary_for_task(
				task.name,
				task_parameters,
				predicate_name,
				analysis,
			)
			if constructive_template and constructive_template not in template_summaries:
				template_summaries.append(constructive_template)
			consumer_template = _render_consumer_template_summary_for_task(
				task.name,
				task_parameters,
				predicate_name,
				analysis,
			)
			if consumer_template and consumer_template not in template_summaries:
				template_summaries.append(consumer_template)

		if shared_requirements:
			shared_dynamic_prerequisites_by_task[task_name] = shared_requirements
		if template_summaries:
			producer_consumer_templates_by_task[task_name] = template_summaries

	return {
		"ordered_query_task_anchors": [dict(anchor) for anchor in query_task_anchors],
		"shared_dynamic_prerequisites_by_task": shared_dynamic_prerequisites_by_task,
		"producer_consumer_templates_by_task": producer_consumer_templates_by_task,
		"task_headline_candidates": task_headline_candidates,
		"query_task_contracts": _build_query_task_contract_payloads(
			domain,
			target_literals,
			query_task_anchors,
			analysis,
		),
		"support_task_contracts": _build_support_task_contract_payloads(
			domain,
			target_literals,
			query_task_anchors,
			analysis,
		),
		"reusable_dynamic_resource_predicates": _reusable_dynamic_resource_payloads(analysis),
	}


def _name_tokens(name: str) -> tuple[str, ...]:
	parts = re.split(r"[^a-z0-9]+", _sanitize_name(name).lower())
	return tuple(
		part
		for part in parts
		if part and part not in {"do", "task", "method", "abs"}
	)


def _token_overlap_score(left: Sequence[str], right: Sequence[str]) -> int:
	score = 0
	for left_token in left:
		for right_token in right:
			if left_token == right_token:
				score += 4
				continue
			if min(len(left_token), len(right_token)) < 4:
				continue
			if left_token.startswith(right_token) or right_token.startswith(left_token):
				score += 2
	return score


def _domain_name_token_frequencies(action_analysis: Dict[str, Any]) -> Dict[str, int]:
	counts: Dict[str, int] = {}
	for predicate_name, patterns in (
		action_analysis.get("producer_patterns_by_predicate", {}) or {}
	).items():
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
			token_weight = 1.0 / max(
				frequencies.get(left_token, 1),
				frequencies.get(right_token, 1),
			)
			if left_token == right_token:
				score += 4.0 * token_weight
				continue
			if min(len(left_token), len(right_token)) < 4:
				continue
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
	candidates = []
	for task in getattr(domain, "tasks", []):
		task_tokens = _name_tokens(task.name)
		if not task_tokens:
			continue
		task_compact = _compact_name_tokens(str(task.name))
		predicate_overlap = _weighted_token_overlap_score(task_tokens, predicate_tokens)
		action_overlap = 0.0
		if task_compact:
			for action_compact in action_compacts:
				if not action_compact:
					continue
				if task_compact == action_compact:
					action_overlap = max(action_overlap, 3.0)
				elif min(len(task_compact), len(action_compact)) >= 4 and (
					task_compact.endswith(action_compact)
					or action_compact.endswith(task_compact)
					or task_compact in action_compact
					or action_compact in task_compact
				):
					action_overlap = max(action_overlap, 2.0)
		if predicate_overlap <= 0 and action_overlap <= 0:
			continue
		if predicate_overlap <= 0 and predicate_compact and task_compact:
			if (
				task_compact != predicate_compact
				and predicate_compact not in task_compact
				and task_compact not in predicate_compact
			):
				# Without direct predicate evidence, only allow strong action-name alignment.
				if action_overlap < 2.0:
					continue
		score = (2.0 * predicate_overlap) + action_overlap
		if len(task.parameters) == len(predicate_args):
			score += 1
		candidates.append((score, task.name))
	candidates.sort(key=lambda item: (-item[0], item[1]))
	return [name for _, name in candidates[:4]]


def _dynamic_support_hint_lines(
	domain: Any,
	action_analysis: Dict[str, Any],
) -> list[str]:
	parser = HDDLConditionParser()
	producer_actions = action_analysis.get("producer_actions_by_predicate", {})
	lines: list[str] = []
	seen: set[str] = set()
	for action in getattr(domain, "actions", []):
		try:
			parsed_action = parser.parse_action(action)
		except Exception:
			continue
		for precondition in parsed_action.preconditions:
			if precondition.predicate == "=":
				continue
			if precondition.predicate not in action_analysis["dynamic_predicates"]:
				continue
			candidates = _candidate_support_task_names(
				domain,
				precondition.predicate,
				precondition.args,
				producer_actions.get(precondition.predicate, []),
			)
			if not candidates:
				continue
			signature = _literal_pattern_signature(precondition)
			line = (
				f"- {_sanitize_name(parsed_action.name)} needs dynamic {signature}; "
				f"likely reusable declared tasks: {', '.join(candidates)}"
			)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)
	return lines


def _same_arity_task_hint_lines(
	domain: Any,
	query_task_anchors: Sequence[Dict[str, Any]],
) -> list[str]:
	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for anchor in query_task_anchors:
		task_name = anchor.get("task_name")
		display_name = _anchor_display_name(anchor)
		task_schema = task_schemas.get(task_name)
		if task_schema is None:
			continue
		candidates = [
			other_task.name
			for other_task in getattr(domain, "tasks", [])
			if other_task.name != task_name
			and len(other_task.parameters) == len(task_schema.parameters)
		]
		if not candidates:
			continue
		line = (
			f"- {_task_invocation_signature(display_name, task_schema.parameters)} may reuse "
			f"same-arity declared tasks: {', '.join(candidates)}"
		)
		if line in seen:
			continue
		seen.add(line)
		lines.append(line)
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
		other_task.name
		for other_task in getattr(domain, "tasks", [])
		if other_task.name != task_name
		and len(other_task.parameters) == len(task_schema.parameters)
		and [
			_parameter_type(parameter)
			for parameter in getattr(other_task, "parameters", ()) or ()
		] == task_parameter_types
	]


def _same_arity_packaging_candidates_for_query_task(
	domain: Any,
	task_name: str,
	predicate_name: str,
	task_parameters: Sequence[str],
	task_schemas: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> list[Dict[str, Any]]:
	headline_candidates_by_task = _task_headline_candidate_map(domain, action_analysis)
	packaging_candidates: list[Dict[str, Any]] = []
	for candidate in _same_arity_declared_task_candidates(
		domain,
		task_name,
		task_schemas,
	):
		candidate_schema = task_schemas.get(candidate)
		if candidate_schema is None:
			continue
		candidate_parameters = tuple(
			_parameter_token(parameter)
			for parameter in getattr(candidate_schema, "parameters", ()) or ()
		)
		if len(candidate_parameters) != len(task_parameters):
			continue
		constructive_template = _constructive_template_summary_for_task(
			candidate,
			candidate_parameters,
			predicate_name,
			action_analysis,
		)
		explicit_source_predicates = {
			str(value).strip()
			for value in (getattr(candidate_schema, "source_predicates", ()) or ())
			if str(value).strip()
		}
		inferred_headlines = set(
			headline_candidates_by_task.get(_sanitize_name(candidate), []),
		)
		#
		# Same-arity packaging should match HDDL method structure: the child task must
		# either already headline this predicate, or have no independently inferred
		# headline at all. If a task already has a distinct inferred headline, reusing
		# it as packaging for another predicate is too permissive and creates bogus
		# cross-task envelopes such as get_rock_data(?w) packaging
		# communicated_soil_data(?w).
		#
		# Tasks without any inferred headline remain eligible because some domains use
		# support tasks whose target predicate is only recoverable from their aligned
		# constructive template, such as blocksworld do_move(?x, ?y) for on(?x, ?y).
		known_headlines = explicit_source_predicates | inferred_headlines
		if known_headlines and predicate_name not in known_headlines:
			continue
		if constructive_template is None:
			continue
		raw_shared_requirements = _shared_dynamic_requirements_for_predicate(
			predicate_name,
			candidate_parameters,
			action_analysis,
		) or _support_task_precise_shared_requirements(
			domain,
			candidate,
			predicate_name,
			candidate_parameters,
			task_schemas,
			action_analysis,
		)
		shared_requirements = _same_arity_packaging_parent_requirements(
			domain,
			predicate_name,
			candidate_parameters,
			action_analysis,
		) or raw_shared_requirements
		packaging_candidates.append(
			{
				"candidate": candidate,
				"parameters": candidate_parameters,
				"constructive_template": constructive_template,
				"shared_requirements": shared_requirements,
			},
	)
	return packaging_candidates


def _render_same_arity_shared_requirements(
	candidate: Dict[str, Any],
	task_parameters: Sequence[str],
) -> tuple[str, ...]:
	candidate_parameters = tuple(
		_parameter_token(parameter)
		for parameter in candidate.get("parameters", ()) or ()
	)
	render_mapping = {
		raw_parameter: task_parameter
		for raw_parameter, task_parameter in zip(candidate_parameters, task_parameters)
	}
	rendered_requirements: list[str] = []
	for raw_requirement in candidate.get("shared_requirements", ()) or ():
		rendered_requirement = _render_signature_with_mapping(
			str(raw_requirement),
			render_mapping,
		)
		if rendered_requirement and rendered_requirement not in rendered_requirements:
			rendered_requirements.append(rendered_requirement)
	return tuple(rendered_requirements)


def _same_arity_caller_shared_requirements(
	domain: Any,
	predicate_name: str,
	task_parameters: Sequence[str],
	action_analysis: Dict[str, Any],
	candidate: Optional[Dict[str, Any]] = None,
) -> tuple[str, ...]:
	"""Return the caller-shared envelope for a same-arity packaging child.

	The raw producer intersection for the headline predicate can be too narrow for
	a reusable declared task. For example, the final primitive producer may need an
	intermediate dynamic literal that the declared child is expected to establish
	internally. For caller-shared obligations we instead expose the refined parent
	envelope, which lifts transitive dynamic requirements when no declared support
	task headlines the intermediate predicate.
	"""
	refined_requirements = _same_arity_packaging_parent_requirements(
		domain,
		predicate_name,
		task_parameters,
		action_analysis,
	)
	if refined_requirements:
		return refined_requirements
	if candidate is None:
		return ()
	return _render_same_arity_shared_requirements(candidate, task_parameters)


def _query_task_same_arity_packaging_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Optional[Dict[str, Any]] = None,
) -> list[str]:
	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	analysis = action_analysis or {}
	if query_task_anchors and len(query_task_anchors) != len(target_literals):
		return []
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		force_internal_contract = bool(anchor.get("force_internal_contract"))
		task_schema = task_schemas.get(task_name)
		if task_schema is None:
			continue
		if force_internal_contract:
			continue
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue
		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		candidates = _same_arity_packaging_candidates_for_query_task(
			domain,
			task_name,
			predicate_name,
			task_parameters,
			task_schemas,
			analysis,
		)
		if not candidates:
			continue
		for candidate in candidates:
			candidate_name = str(candidate.get("candidate", "")).strip()
			candidate_parameters = tuple(candidate.get("parameters", ()))
			shared_requirements = _same_arity_caller_shared_requirements(
				domain,
				predicate_name,
				task_parameters,
				analysis,
				candidate,
			)
			requirement_text = (
				", ".join(shared_requirements)
				if shared_requirements
				else "none"
			)
			line = (
				f"- {_task_invocation_signature(display_name, task_parameters)}: exact same-arity "
				f"packaging contract for {predicate_name}({', '.join(task_parameters)}) is "
				f"{_task_invocation_signature(candidate_name, candidate_parameters)}. Support "
				f"caller-shared prerequisites {requirement_text} before the child call, then let "
				f"{_task_invocation_signature(candidate_name, candidate_parameters)} own the final "
				f"producer for {predicate_name}({', '.join(task_parameters)}). The listed "
				"caller-shared set is exhaustive at child entry."
			)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)
	return lines


def _same_arity_packaging_parent_requirements(
	domain: Any,
	predicate_name: str,
	task_parameters: Sequence[str],
	action_analysis: Dict[str, Any],
) -> tuple[str, ...]:
	exposed_requirements: set[str] = set()
	for requirement in _shared_dynamic_requirements_for_predicate(
		predicate_name,
		task_parameters,
		action_analysis,
	):
		parsed_requirement = _parse_literal_signature(requirement)
		if parsed_requirement is None:
			continue
		requirement_predicate, requirement_args, requirement_positive = parsed_requirement
		if not requirement_positive:
			continue
		if requirement_args and not all(arg in task_parameters for arg in requirement_args):
			continue

		support_task_candidates = _candidate_support_task_names(
			domain,
			requirement_predicate,
			requirement_args,
			action_analysis.get("producer_actions_by_predicate", {}).get(
				requirement_predicate,
				[],
			),
		)
		if support_task_candidates:
			exposed_requirements.add(requirement)
			continue

		for transitive_requirement in _shared_dynamic_requirements_for_predicate(
			requirement_predicate,
			requirement_args,
			action_analysis,
		):
			parsed_transitive = _parse_literal_signature(transitive_requirement)
			if parsed_transitive is None:
				continue
			_, transitive_args, transitive_positive = parsed_transitive
			if not transitive_positive:
				continue
			if transitive_args and not all(arg in task_parameters for arg in transitive_args):
				continue
			exposed_requirements.add(transitive_requirement)

	return tuple(sorted(exposed_requirements))


def _same_arity_packaging_headline_only_fragment(
	domain: Any,
	packaging_task_name: str,
	predicate_name: str,
	task_parameters: Sequence[str],
	action_analysis: Dict[str, Any],
) -> str:
	exposed_requirements = _same_arity_packaging_parent_requirements(
		domain,
		predicate_name,
		task_parameters,
		action_analysis,
	)
	if len(exposed_requirements) < 2:
		return ""

	task_schemas = _declared_task_schema_map(domain)
	headline_support_pairs: list[tuple[str, str]] = []
	remaining_requirements: list[str] = []

	for requirement in exposed_requirements:
		parsed_requirement = _parse_literal_signature(requirement)
		if parsed_requirement is None:
			continue
		requirement_predicate, requirement_args, requirement_positive = parsed_requirement
		if not requirement_positive:
			continue
		candidate_tasks = [
			candidate
			for candidate in _candidate_support_task_names(
				domain,
				requirement_predicate,
				requirement_args,
				action_analysis.get("producer_actions_by_predicate", {}).get(
					requirement_predicate,
					[],
				),
			)
			if candidate != packaging_task_name
		]
		selected_task: Optional[str] = None
		for candidate_task in candidate_tasks:
			candidate_schema = task_schemas.get(candidate_task)
			if candidate_schema is None:
				continue
			if len(candidate_schema.parameters) != len(requirement_args):
				continue
			selected_task = candidate_task
			break

		if selected_task is None:
			remaining_requirements.append(requirement)
			continue

		headline_support_pairs.append(
			(
				_task_invocation_signature(selected_task, requirement_args),
				requirement,
			)
		)

	if not headline_support_pairs or not remaining_requirements:
		return ""

	support_descriptions = ", ".join(
		f"{task_signature} for {headline_literal}"
		for task_signature, headline_literal in headline_support_pairs
	)
	remaining_text = ", ".join(remaining_requirements)
	return (
		f"earlier support tasks {support_descriptions} only guarantee those headline literals; "
		f"keep {remaining_text} explicit in parent context unless another earlier parent step "
		f"itself headlines {remaining_text}"
	)


def _query_task_same_arity_child_support_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	query_task_names = {
		str(anchor.get("task_name", "")).strip()
		for anchor in query_task_anchors
		if str(anchor.get("task_name", "")).strip()
	}
	lines: list[str] = []
	seen: set[str] = set()
	required_helper_lookup = {
		(
			str(spec.get("query_task_name", "")).strip(),
			str(spec.get("precondition_signature", "")).strip(),
		): spec
		for spec in _required_helper_specs_for_query_targets(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		)
	}
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		force_internal_contract = bool(anchor.get("force_internal_contract"))
		task_schema = task_schemas.get(task_name)
		if task_schema is None:
			continue
		if force_internal_contract:
			continue
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue

		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		candidates = _same_arity_declared_task_candidates(
			domain,
			task_name,
			task_schemas,
		)
		for candidate in candidates:
			exposed_requirements = _same_arity_packaging_parent_requirements(
				domain,
				predicate_name,
				task_parameters,
				action_analysis,
			)
			if not exposed_requirements:
				continue
			requirement_plans = [
				_same_arity_parent_requirement_plan(
					domain,
					requirement,
					action_analysis,
				)
				for requirement in exposed_requirements
			]
			line = (
				f"- {_task_invocation_signature(display_name, task_parameters)}: "
				f"if you invoke {_task_invocation_signature(candidate, task_parameters)} as same-arity "
				f"packaging for {predicate_name}({', '.join(task_parameters)}), first support "
				f"child shared prerequisites {'; '.join(requirement_plans)}."
			)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)
	return lines


def _query_task_zero_ary_parent_context_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	query_task_names = {
		str(anchor.get("task_name", "")).strip()
		for anchor in query_task_anchors
		if str(anchor.get("task_name", "")).strip()
	}
	lines: list[str] = []
	seen: set[str] = set()
	required_helper_lookup = {
		(
			str(spec.get("query_task_name", "")).strip(),
			str(spec.get("precondition_signature", "")).strip(),
		): spec
		for spec in _required_helper_specs_for_query_targets(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		)
	}
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		force_internal_contract = bool(anchor.get("force_internal_contract"))
		task_schema = task_schemas.get(task_name)
		if task_schema is None:
			continue
		if force_internal_contract:
			continue
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue

		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		candidates = _same_arity_declared_task_candidates(
			domain,
			task_name,
			task_schemas,
		)
		for candidate in candidates:
			exposed_requirements = _same_arity_packaging_parent_requirements(
				domain,
				predicate_name,
				task_parameters,
				action_analysis,
			)
			for requirement in exposed_requirements:
				parsed_requirement = _parse_literal_signature(requirement)
				if parsed_requirement is None:
					continue
				_, requirement_args, requirement_positive = parsed_requirement
				if not requirement_positive or requirement_args:
					continue
				line = (
					f"- {_task_invocation_signature(display_name, task_parameters)}: before "
					f"{_task_invocation_signature(candidate, task_parameters)}, place "
					f"{requirement} explicitly in parent context unless an earlier parent "
					"subtask establishes it."
				)
				if line in seen:
					continue
				seen.add(line)
				lines.append(line)
	return lines


def _query_task_same_arity_child_context_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	query_task_names = {
		str(anchor.get("task_name", "")).strip()
		for anchor in query_task_anchors
		if str(anchor.get("task_name", "")).strip()
	}
	lines: list[str] = []
	seen: set[str] = set()
	required_helper_lookup = {
		(
			str(spec.get("query_task_name", "")).strip(),
			str(spec.get("precondition_signature", "")).strip(),
		): spec
		for spec in _required_helper_specs_for_query_targets(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		)
	}
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		task_schema = task_schemas.get(task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if task_schema is None or parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue

		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		candidates = _same_arity_declared_task_candidates(
			domain,
			task_name,
			task_schemas,
		)
		for candidate in candidates[:2]:
			exposed_requirements = _same_arity_packaging_parent_requirements(
				domain,
				predicate_name,
				task_parameters,
				action_analysis,
			)
			if not exposed_requirements:
				continue
			line = (
				f"- {_task_invocation_signature(candidate, task_parameters)}: when used as same-arity "
				f"packaging for {predicate_name}({', '.join(task_parameters)}), every constructive "
				f"sibling should keep {', '.join(exposed_requirements)} explicit in method.context "
				"unless that sibling establishes one of them earlier internally."
			)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)
	return lines


def _query_task_packaging_skeleton_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		force_internal_contract = bool(anchor.get("force_internal_contract"))
		task_schema = task_schemas.get(task_name)
		if task_schema is None:
			continue
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue

		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		candidates = tuple(
			str(item.get("candidate", "")).strip()
			for item in _same_arity_packaging_candidates_for_query_task(
				domain,
				task_name,
				predicate_name,
				task_parameters,
				task_schemas,
				action_analysis,
			)
			if str(item.get("candidate", "")).strip()
		)
		for candidate in candidates[:2]:
			packaging_candidates = {
				str(item.get("candidate", "")).strip(): item
				for item in _same_arity_packaging_candidates_for_query_task(
					domain,
					task_name,
					predicate_name,
					task_parameters,
					task_schemas,
					action_analysis,
				)
			}
			candidate_payload = packaging_candidates.get(candidate)
			exposed_requirements = tuple(
				candidate_payload.get("shared_requirements", ())
				if candidate_payload is not None
				else ()
			)
			context_requirements: list[str] = []
			support_calls: list[str] = []
			for requirement in exposed_requirements:
				parsed_requirement = _parse_literal_signature(requirement)
				if parsed_requirement is None:
					continue
				requirement_predicate, requirement_args, requirement_positive = parsed_requirement
				if not requirement_positive:
					continue
				if not requirement_args:
					if requirement not in context_requirements:
						context_requirements.append(requirement)
					continue
				support_task_candidates = [
					support_task
					for support_task in _candidate_support_task_names(
						domain,
						requirement_predicate,
						requirement_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							requirement_predicate,
							[],
						),
					)
					if support_task != task_name
					and len(getattr(task_schemas.get(support_task), "parameters", ()))
					== len(requirement_args)
				]
				if support_task_candidates:
					rendered_support_call = _task_invocation_signature(
						support_task_candidates[0],
						requirement_args,
					)
					if rendered_support_call not in support_calls:
						support_calls.append(rendered_support_call)
					continue
				if requirement not in context_requirements:
					context_requirements.append(requirement)
			bound_occurrences = [
				tuple(str(arg) for arg in occurrence.get("args", ()))
				for occurrence in query_task_anchors
				if str(occurrence.get("task_name", "")).strip() == task_name
				and len(tuple(occurrence.get("args", ()))) == len(task_parameters)
			]
			for stabilizer_call in _query_task_non_leading_role_stabilizer_plans(
				domain,
				task_schema,
				bound_occurrences=bound_occurrences,
				predicate_name=predicate_name,
				action_analysis=action_analysis,
			):
				if stabilizer_call not in support_calls:
					support_calls.append(stabilizer_call)
			requirement_plans = [
				_same_arity_parent_requirement_plan(
					domain,
					requirement,
					action_analysis,
				)
				for requirement in exposed_requirements
			]
			if requirement_plans:
				line = (
					f"- {_task_invocation_signature(display_name, task_parameters)}: once you choose "
					f"{_task_invocation_signature(candidate, task_parameters)} as same-arity packaging "
					f"for {predicate_name}({', '.join(task_parameters)}), use a parent skeleton that first "
					f"supports {'; '.join(requirement_plans)} and only then calls "
					f"{_task_invocation_signature(candidate, task_parameters)}. Do not keep planning "
					"from the parent task's direct headline producer after selecting the packaging child."
				)
				if context_requirements:
					slot_parts: list[str] = [
						f"precondition/context {', '.join(context_requirements)}",
					]
					if support_calls:
						slot_parts.append(f"support_before {'; '.join(support_calls)}")
					slot_parts.append(
						f"producer {_task_invocation_signature(candidate, task_parameters)}",
					)
					line += (
						f" More concrete AST slot shape: {'; '.join(slot_parts)}."
					)
			else:
				line = (
					f"- {_task_invocation_signature(display_name, task_parameters)}: once you choose "
					f"{_task_invocation_signature(candidate, task_parameters)} as same-arity packaging "
					f"for {predicate_name}({', '.join(task_parameters)}), call that child directly and let "
					f"{_task_invocation_signature(candidate, task_parameters)} own internal support plus "
					"the final producer."
				)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)
	return lines


def _query_task_non_leading_role_stabilizer_plans(
	domain: Any,
	task_schema: Any,
	*,
	bound_occurrences: Sequence[Sequence[str]],
	predicate_name: str,
	action_analysis: Dict[str, Any],
) -> list[str]:
	task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
	if len(task_parameters) < 2 or len(bound_occurrences) < 2:
		return []

	direct_role_requirements: Dict[int, set[str]] = {
		index: set()
		for index in range(1, len(task_parameters))
	}
	for headline_pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
		predicate_name,
		[],
	):
		effect_args = list(headline_pattern.get("effect_args") or [])
		if len(effect_args) != len(task_parameters):
			continue
		token_mapping = {
			token: task_parameter
			for token, task_parameter in zip(effect_args, task_parameters)
		}
		for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
			rendered_signature = _render_signature_with_mapping(
				precondition_signature,
				token_mapping,
			)
			parsed_precondition = _parse_literal_signature(rendered_signature)
			if parsed_precondition is None:
				continue
			_, precondition_args, precondition_positive = parsed_precondition
			if not precondition_positive or len(precondition_args) != 1:
				continue
			for role_index in range(1, len(task_parameters)):
				if precondition_args[0] == task_parameters[role_index]:
					direct_role_requirements[role_index].add(rendered_signature)

	task_schemas = _declared_task_schema_map(domain)
	primary_values = {occurrence[0] for occurrence in bound_occurrences}
	stabilizer_plans: list[str] = []
	seen_plans: set[str] = set()
	for role_index in range(1, len(task_parameters)):
		role_parameter = task_parameters[role_index]
		role_type = _parameter_type(task_schema.parameters[role_index])
		role_values = {
			occurrence[role_index]
			for occurrence in bound_occurrences
			if len(occurrence) == len(task_parameters)
		}
		reused_values = {value for value in role_values if value in primary_values}
		terminal_values = {value for value in role_values if value not in primary_values}
		if not reused_values and not terminal_values:
			continue

		for candidate_task in getattr(domain, "tasks", []):
			candidate_name = str(candidate_task.name)
			if len(candidate_task.parameters) != 1:
				continue
			if _parameter_type(candidate_task.parameters[0]) != role_type:
				continue
			if candidate_name == str(task_schema.name):
				continue
			for candidate_predicate in _candidate_headline_predicates_for_task(
				candidate_name,
				1,
				action_analysis,
			):
				candidate_signature = f"{candidate_predicate}({role_parameter})"
				if candidate_signature in direct_role_requirements[role_index]:
					continue
				if _constructive_template_summary_for_task(
					candidate_name,
					(role_parameter,),
					candidate_predicate,
					action_analysis,
				) is None:
					continue
				plan = _task_invocation_signature(candidate_name, (role_parameter,))
				if plan in seen_plans:
					break
				seen_plans.add(plan)
				stabilizer_plans.append(plan)
				break
			if stabilizer_plans and stabilizer_plans[-1].endswith(f"({role_parameter})"):
				break
	return stabilizer_plans


def _same_arity_parent_requirement_plan(
	domain: Any,
	requirement_signature: str,
	action_analysis: Dict[str, Any],
) -> str:
	parsed_requirement = _parse_literal_signature(requirement_signature)
	if parsed_requirement is None:
		return f"{requirement_signature} via parent context or earlier parent subtasks"

	predicate_name, requirement_args, requirement_positive = parsed_requirement
	if not requirement_positive:
		return f"{requirement_signature} via parent context or earlier parent subtasks"
	if not requirement_args:
		return (
			f"{requirement_signature} explicitly in parent context unless an earlier "
			"parent subtask establishes it"
		)

	task_schemas = _declared_task_schema_map(domain)
	task_candidates = _candidate_support_task_names(
		domain,
		predicate_name,
		requirement_args,
		action_analysis.get("producer_actions_by_predicate", {}).get(
			predicate_name,
			[],
		),
	)
	task_candidates = [
		candidate
		for candidate in task_candidates
		if len(getattr(task_schemas.get(candidate), "parameters", ())) == len(requirement_args)
	]
	if task_candidates:
		return (
			f"{requirement_signature} via "
			f"{' or '.join(_task_invocation_signature(candidate, requirement_args) for candidate in task_candidates[:2])}"
		)

	return f"{requirement_signature} via parent context or earlier parent subtasks"


def _task_shared_and_transitive_requirements(
	predicate_name: str,
	task_parameters: Sequence[str],
	action_analysis: Dict[str, Any],
) -> tuple[str, ...]:
	qualified_requirements: set[str] = set()
	shared_requirements = _shared_dynamic_requirements_for_predicate(
		predicate_name,
		task_parameters,
		action_analysis,
	)
	for requirement in shared_requirements:
		parsed_requirement = _parse_literal_signature(requirement)
		if parsed_requirement is None:
			continue
		_, requirement_args, requirement_positive = parsed_requirement
		if not requirement_positive:
			continue
		if not requirement_args or all(arg in task_parameters for arg in requirement_args):
			qualified_requirements.add(requirement)

	for shared_requirement in shared_requirements:
		parsed_requirement = _parse_literal_signature(shared_requirement)
		if parsed_requirement is None:
			continue
		shared_predicate, shared_args, shared_positive = parsed_requirement
		if not shared_positive:
			continue
		for transitive_requirement in _shared_dynamic_requirements_for_predicate(
			shared_predicate,
			shared_args,
			action_analysis,
		):
			parsed_transitive = _parse_literal_signature(transitive_requirement)
			if parsed_transitive is None:
				continue
			_, transitive_args, transitive_positive = parsed_transitive
			if not transitive_positive:
				continue
			if not transitive_args or all(arg in task_parameters for arg in transitive_args):
				qualified_requirements.add(transitive_requirement)

	return tuple(sorted(qualified_requirements))


def _query_task_same_arity_transitive_requirement_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		task_schema = task_schemas.get(task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if task_schema is None or parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue

		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		packaging_candidates = tuple(
			str(item.get("candidate", "")).strip()
			for item in _same_arity_packaging_candidates_for_query_task(
				domain,
				task_name,
				predicate_name,
				task_parameters,
				task_schemas,
				action_analysis,
			)
			if str(item.get("candidate", "")).strip()
		)
		if not packaging_candidates:
			continue

		headline_summary = _constructive_template_summary_for_task(
			task_name,
			task_parameters,
			predicate_name,
			action_analysis,
		)
		if headline_summary is None:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		mode_fragments: list[str] = []
		mode_seen: set[str] = set()
		specific_mode_fragments: list[str] = []
		specific_mode_seen: set[str] = set()
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(task_parameters):
				continue

			token_mapping = {
				token: task_parameter
				for token, task_parameter in zip(effect_args, task_parameters)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				rendered_signature = _render_signature_with_mapping(
					precondition_signature,
					token_mapping,
				)
				parsed_support = _parse_literal_signature(rendered_signature)
				if parsed_support is None:
					continue
				support_predicate, support_args, support_positive = parsed_support
				if not support_positive:
					continue
				if not set(support_args) & set(task_parameters):
					continue

				support_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
					support_predicate,
					[],
				)
				later_required_signatures = {
					_render_signature_with_mapping(signature, token_mapping)
					for signature in (headline_pattern.get("dynamic_precondition_signatures") or [])
				}
				if not support_patterns:
					continue
				specific_handoff_fragments: list[str] = []
				for support_pattern in support_patterns[:2]:
					support_effect_args = list(support_pattern.get("effect_args") or [])
					if len(support_effect_args) != len(support_args):
						continue
					support_mapping = {
						token: arg
						for token, arg in zip(support_effect_args, support_args)
					}
					rendered_support_action_args = _extend_mapping_with_action_parameters(
						support_mapping,
						support_pattern.get("action_parameters") or [],
						action_parameter_types=support_pattern.get("action_parameter_types") or [],
					)
					extra_role_args = [
						arg
						for arg in rendered_support_action_args
						if arg not in support_args
					]
					if not extra_role_args:
						continue
					specific_handoff_fragments.append(
						f"If you choose mode "
						f"{_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)} "
						f"to make {support_predicate}({', '.join(support_args)}), the next relevant "
						f"step should be the final producer {headline_summary}; do not add support "
						f"or cleanup on extra roles {', '.join(extra_role_args)} unless the chosen "
						"producer or a later step explicitly needs that same headline literal"
					)
					for positive_effect_signature in (
						support_pattern.get("positive_effect_signatures") or []
					):
						rendered_effect_signature = _render_signature_with_mapping(
							positive_effect_signature,
							support_mapping,
						)
						parsed_effect = _parse_literal_signature(rendered_effect_signature)
						if parsed_effect is None:
							continue
						_, effect_args, effect_positive = parsed_effect
						if not effect_positive:
							continue
						if rendered_effect_signature == (
							f"{support_predicate}({', '.join(support_args)})"
							if support_args
							else support_predicate
						):
							continue
						if not any(arg in extra_role_args for arg in effect_args):
							continue
						if rendered_effect_signature in later_required_signatures:
							continue
						specific_handoff_fragments.append(
							f"If you choose mode "
							f"{_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)} "
							f"to make {support_predicate}({', '.join(support_args)}), do not try "
							f"to establish {rendered_effect_signature} before that step unless a "
							f"later step explicitly consumes {rendered_effect_signature}; this "
							f"mode already makes {rendered_effect_signature} as a side effect after it runs"
						)
				support_task_candidates = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						support_predicate,
						support_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							support_predicate,
							[],
						),
					)
					if len(getattr(task_schemas.get(candidate), "parameters", ())) == len(support_args)
				]
				if support_task_candidates:
					fragment = (
						f"Inside this task, support {support_predicate}({', '.join(support_args)}) "
						f"via declared tasks "
						f"{' or '.join(_headline_support_task_invocation_signature(candidate, support_predicate, support_args, task_schemas, action_analysis) for candidate in support_task_candidates[:2])} "
						"before the final producer"
					)
					if fragment not in mode_seen:
						mode_seen.add(fragment)
						mode_fragments.append(fragment)
					continue
				support_summary = _constructive_template_summary_for_task(
					support_predicate,
					support_args,
					support_predicate,
					action_analysis,
				)
				if support_summary is None:
					continue
				fragment = (
					f"Inside this task, support {support_predicate}({', '.join(support_args)}) "
					f"before the final producer using one of the real producer modes {support_summary}"
				)
				if fragment in mode_seen:
					continue
				mode_seen.add(fragment)
				mode_fragments.append(fragment)
				common_mode_requirements: Optional[set[str]] = None
				for support_pattern in support_patterns:
					rendered_requirements = {
						_render_signature_with_mapping(signature, support_mapping)
						for signature in (
							support_pattern.get("dynamic_precondition_signatures") or []
						)
					}
					filtered_requirements = {
						signature
						for signature in rendered_requirements
						if (
							(parsed_signature := _parse_literal_signature(signature)) is not None
							and parsed_signature[2]
							and (
								not parsed_signature[1]
								or set(parsed_signature[1]).issubset(set(task_parameters))
							)
						)
					}
					if common_mode_requirements is None:
						common_mode_requirements = filtered_requirements
					else:
						common_mode_requirements &= filtered_requirements
				if common_mode_requirements:
					common_mode_text = ", ".join(sorted(common_mode_requirements))
					shared_mode_fragment = (
						f"If every real producer mode for {support_predicate}({', '.join(support_args)}) "
						f"still needs {common_mode_text}, keep those literals explicit in each "
						"constructive sibling context of this packaging task instead of "
						"re-establishing them with an extra support subtask inside every sibling"
					)
					if shared_mode_fragment not in mode_seen:
						mode_seen.add(shared_mode_fragment)
						mode_fragments.append(shared_mode_fragment)
				if not support_task_candidates:
					if len(support_patterns) >= 2:
						no_declared_fragment = (
							f"No declared task directly headlines {support_predicate}({', '.join(support_args)}), "
							f"so keep that obligation inside this packaging task and, because its "
							"producer modes differ, prefer separate constructive sibling methods "
							"instead of exposing a mode-specific prerequisite to the parent"
						)
					else:
						no_declared_fragment = (
							f"No declared task directly headlines {support_predicate}({', '.join(support_args)}), "
							"so keep that obligation inside this same-arity packaging task instead of "
							"pushing it back to the parent as an exposed prerequisite"
						)
					if no_declared_fragment not in mode_seen:
						mode_seen.add(no_declared_fragment)
						mode_fragments.append(no_declared_fragment)
					if len(support_patterns) >= 2:
						sibling_fragment = (
							"For generic coverage, implement one constructive sibling per supported "
							"producer mode instead of keeping only a single narrow mode branch"
						)
						if sibling_fragment not in mode_seen:
							mode_seen.add(sibling_fragment)
							mode_fragments.append(sibling_fragment)
				handoff_fragment = (
					f"If one of those support modes produces {support_predicate}({', '.join(support_args)}), "
					f"hand that same literal directly to the final producer {headline_summary}. "
					"Do not insert cleanup on extra-role symbols unless later steps really establish "
					"the cleanup precondition for that same symbol"
				)
				if handoff_fragment not in mode_seen:
					mode_seen.add(handoff_fragment)
					mode_fragments.append(handoff_fragment)
				for handoff_fragment in specific_handoff_fragments:
					if handoff_fragment in mode_seen:
						continue
					mode_seen.add(handoff_fragment)
					mode_fragments.append(handoff_fragment)
					if handoff_fragment not in specific_mode_seen:
						specific_mode_seen.add(handoff_fragment)
						specific_mode_fragments.append(handoff_fragment)

		for candidate in packaging_candidates:
			line = (
				f"- {_task_invocation_signature(candidate, task_parameters)}: if used as same-arity "
				f"packaging for {predicate_name}({', '.join(task_parameters)}), its own constructive "
				f"branch must internally close the headline effect via {headline_summary}."
			)
			if mode_fragments:
				line += f" {' '.join(mode_fragments[:7])}."
			line += (
				" Parent methods should not have to provide unresolved internal support merely "
				"because this packaging task was chosen. Do not call this packaging child and "
				"then repeat the same final producer again in the parent."
			)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)
			for fragment in specific_mode_fragments[:3]:
				specific_line = (
					f"- {_task_invocation_signature(candidate, task_parameters)}: {fragment}."
				)
				if specific_line in seen:
					continue
				seen.add(specific_line)
				lines.append(specific_line)
	return lines


def _aligned_producer_template_lines(action_analysis: Dict[str, Any]) -> list[str]:
	lines: list[str] = []
	for predicate_name, patterns in action_analysis.get("producer_patterns_by_predicate", {}).items():
		for pattern in patterns:
			effect_args = list(pattern.get("effect_args") or [])
			action_parameters = [
				_parameter_token(parameter)
				for parameter in (pattern.get("action_parameters") or [])
			]
			target_labels = (
				["TARGET"]
				if len(effect_args) == 1
				else [f"TARGET{index}" for index in range(1, len(effect_args) + 1)]
			)
			token_mapping = {
				token: label
				for token, label in zip(effect_args, target_labels)
			}
			other_index = 1
			rendered_args = []
			for token in action_parameters:
				if token not in token_mapping:
					token_mapping[token] = f"OTHER{other_index}"
					other_index += 1
				rendered_args.append(token_mapping[token])
			rendered_targets = ", ".join(target_labels)
			rendered_call = (
				f"{pattern['action_name']}({', '.join(rendered_args)})"
				if rendered_args
				else f"{pattern['action_name']}()"
			)
			lines.append(
				f"- {rendered_call} -> {predicate_name}({rendered_targets})"
			)
	return lines


def _shared_producer_requirement_lines(action_analysis: Dict[str, Any]) -> list[str]:
	lines: list[str] = []
	for predicate_name, patterns in action_analysis.get("producer_patterns_by_predicate", {}).items():
		if not patterns:
			continue
		effect_arity = len(patterns[0].get("effect_args") or [])
		if any(len(pattern.get("effect_args") or []) != effect_arity for pattern in patterns):
			continue

		target_labels = (
			["TARGET"]
			if effect_arity == 1
			else [f"TARGET{index}" for index in range(1, effect_arity + 1)]
		)
		requirement_sets: list[set[str]] = []
		for pattern in patterns:
			effect_args = list(pattern.get("effect_args") or [])
			token_mapping = {
				token: label
				for token, label in zip(effect_args, target_labels)
			}
			_extend_mapping_with_action_parameters(
				token_mapping,
				pattern.get("action_parameters") or [],
				action_parameter_types=pattern.get("action_parameter_types") or [],
			)
			requirement_sets.append(
				{
					_render_signature_with_mapping(signature, token_mapping)
					for signature in (pattern.get("dynamic_precondition_signatures") or [])
					if not str(signature).startswith("not ")
				},
			)

		if not requirement_sets:
			continue
		shared_requirements = set.intersection(*requirement_sets)
		if not shared_requirements:
			continue
		lines.append(
			f"- {predicate_name}({', '.join(target_labels)}) shared dynamic prerequisites: "
			f"{', '.join(sorted(shared_requirements))}"
		)
	return lines


def _parse_literal_signature(signature: str) -> Optional[tuple[str, tuple[str, ...], bool]]:
	text = signature.strip()
	is_positive = True
	if text.startswith("!"):
		is_positive = False
		text = text[1:].strip()
	if "(" not in text or not text.endswith(")"):
		return text, (), is_positive
	predicate, raw_args = text[:-1].split("(", 1)
	args = tuple(
		part.strip()
		for part in raw_args.split(",")
		if part.strip()
	)
	return predicate.strip(), args, is_positive


def _signature_mentions_aux_role(signature: str) -> bool:
	parsed_signature = _parse_literal_signature(signature)
	if parsed_signature is None:
		return False
	_, args, _ = parsed_signature
	return any(str(argument).strip().startswith("AUX_") for argument in args)


def _is_aux_binding_requirement(
	requirement_args: Sequence[str],
	*,
	task_parameter_symbols: Collection[str],
	extra_role_symbols: Collection[str],
) -> bool:
	"""Return True when a literal binds task arguments to an auxiliary witness role."""

	return any(arg in task_parameter_symbols for arg in requirement_args) and any(
		arg in extra_role_symbols for arg in requirement_args
	)


def _render_signature_with_mapping(signature: str, token_mapping: Dict[str, str]) -> str:
	if not token_mapping:
		return signature
	pattern = re.compile(
		r"(?<![A-Za-z0-9_])("
		+ "|".join(
			sorted(
				(re.escape(token) for token in token_mapping),
				key=len,
				reverse=True,
			),
		)
		+ r")(?![A-Za-z0-9_])",
	)
	return pattern.sub(lambda match: token_mapping[match.group(1)], signature)


def _shared_dynamic_requirements_for_predicate(
	predicate_name: str,
	predicate_args: Sequence[str],
	action_analysis: Dict[str, Any],
) -> tuple[str, ...]:
	patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, [])
	if not patterns:
		return ()

	requirement_sets: list[set[str]] = []
	for pattern in patterns:
		effect_args = list(pattern.get("effect_args") or [])
		if len(effect_args) != len(predicate_args):
			continue

		token_mapping = {
			token: arg
			for token, arg in zip(effect_args, predicate_args)
		}
		_extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or [],
			action_parameter_types=pattern.get("action_parameter_types") or [],
		)
		requirement_sets.append(
			{
				_render_signature_with_mapping(signature, token_mapping)
				for signature in (pattern.get("dynamic_precondition_signatures") or [])
				if not str(signature).startswith("not ")
			},
		)

	if not requirement_sets:
		return ()
	return tuple(sorted(set.intersection(*requirement_sets)))


def _render_pattern_negative_effect_signatures(
	pattern: Dict[str, Any],
	*,
	effect_args: Sequence[str],
) -> set[str]:
	pattern_effect_args = list(pattern.get("effect_args") or [])
	if len(pattern_effect_args) != len(effect_args):
		return set()

	token_mapping = {
		token: arg
		for token, arg in zip(pattern_effect_args, effect_args)
	}
	_extend_mapping_with_action_parameters(
		token_mapping,
		pattern.get("action_parameters") or [],
		action_parameter_types=pattern.get("action_parameter_types") or [],
	)
	return {
		_render_signature_with_mapping(signature, token_mapping)
		for signature in (pattern.get("negative_effect_signatures") or [])
	}


def _signature_pattern_matches_requirement(
	pattern_signature: str,
	requirement_signature: str,
) -> bool:
	parsed_pattern = _parse_literal_signature(pattern_signature)
	parsed_requirement = _parse_literal_signature(requirement_signature)
	if parsed_pattern is None or parsed_requirement is None:
		return False
	pattern_predicate, pattern_args, pattern_positive = parsed_pattern
	requirement_predicate, requirement_args, requirement_positive = parsed_requirement
	if not pattern_positive or not requirement_positive:
		return False
	if pattern_predicate != requirement_predicate or len(pattern_args) != len(requirement_args):
		return False
	for pattern_arg, requirement_arg in zip(pattern_args, requirement_args):
		if pattern_arg == requirement_arg:
			continue
		if str(pattern_arg).startswith("AUX_"):
			continue
		return False
	return True


def _all_producer_modes_clobber_requirement(
	requirement_signature: str,
	other_requirement_signature: str,
	action_analysis: Dict[str, Any],
) -> bool:
	parsed_requirement = _parse_literal_signature(requirement_signature)
	if parsed_requirement is None:
		return False
	requirement_predicate, requirement_args, requirement_positive = parsed_requirement
	if not requirement_positive:
		return False

	patterns = [
		pattern
		for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
			requirement_predicate,
			[],
		)
		if len(pattern.get("effect_args") or []) == len(requirement_args)
	]
	if not patterns:
		return False

	return all(
		any(
			_signature_pattern_matches_requirement(
				negative_signature,
				other_requirement_signature,
			)
			for negative_signature in _render_pattern_negative_effect_signatures(
				pattern,
				effect_args=requirement_args,
			)
		)
		for pattern in patterns
	)


def _requirements_are_mutually_destructive(
	first_requirement_signature: str,
	second_requirement_signature: str,
	action_analysis: Dict[str, Any],
) -> bool:
	return _all_producer_modes_clobber_requirement(
		first_requirement_signature,
		second_requirement_signature,
		action_analysis,
	) and _all_producer_modes_clobber_requirement(
		second_requirement_signature,
		first_requirement_signature,
		action_analysis,
	)


def _constructive_template_summary_for_task(
	display_name: str,
	task_parameters: Sequence[str],
	predicate_name: str,
	action_analysis: Dict[str, Any],
) -> Optional[str]:
	patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, [])
	if not patterns:
		return None

	rendered_patterns: list[str] = []
	for pattern in patterns:
		effect_args = list(pattern.get("effect_args") or [])
		if len(effect_args) != len(task_parameters):
			continue

		token_mapping = {
			token: task_parameter
			for token, task_parameter in zip(effect_args, task_parameters)
		}
		rendered_action_args = _extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or [],
			action_parameter_types=pattern.get("action_parameter_types") or [],
		)

		rendered_requirements = _render_positive_dynamic_requirements(
			pattern,
			token_mapping,
		)
		extra_role_preconditions = []
		shared_role_preconditions = []
		for rendered_signature in rendered_requirements:
			parsed_signature = _parse_literal_signature(rendered_signature)
			if parsed_signature is None:
				continue
			_, rendered_args, _ = parsed_signature
			if rendered_args and set(rendered_args).issubset(set(task_parameters)):
				shared_role_preconditions.append(rendered_signature)
				continue
			extra_role_preconditions.append(rendered_signature)

		rendered_call = _task_invocation_signature(
			pattern["action_name"],
			rendered_action_args,
		)
		if shared_role_preconditions and extra_role_preconditions:
			precondition_suffix = (
				f" [needs {', '.join(shared_role_preconditions)}; "
				f"extra needs {', '.join(extra_role_preconditions)}]"
			)
		elif extra_role_preconditions:
			precondition_suffix = (
				f" [extra needs {', '.join(extra_role_preconditions)}]"
			)
		elif shared_role_preconditions:
			precondition_suffix = f" [needs {', '.join(shared_role_preconditions)}]"
		else:
			precondition_suffix = ""
		rendered_patterns.append(f"{rendered_call}{precondition_suffix}")

	if not rendered_patterns:
		return None

	return "; ".join(rendered_patterns)


def _render_positive_dynamic_requirements(
	pattern: Dict[str, Any],
	token_mapping: Dict[str, str],
) -> list[str]:
	"""Render the selected producer mode's positive dynamic prerequisites."""

	requirements: list[str] = []
	for signature in pattern.get("dynamic_precondition_signatures") or []:
		rendered_signature = _render_signature_with_mapping(signature, token_mapping)
		parsed_signature = _parse_literal_signature(rendered_signature)
		if parsed_signature is None:
			continue
		_, _, is_positive = parsed_signature
		if not is_positive:
			continue
		requirements.append(rendered_signature)
	return requirements


def _render_positive_static_requirements(
	pattern: Dict[str, Any],
	token_mapping: Dict[str, str],
) -> list[str]:
	"""Render the selected producer mode's positive non-dynamic prerequisites."""

	dynamic_signatures = set(pattern.get("dynamic_precondition_signatures") or [])
	requirements: list[str] = []
	for signature in pattern.get("precondition_signatures") or []:
		if signature in dynamic_signatures:
			continue
		rendered_signature = _render_signature_with_mapping(signature, token_mapping)
		parsed_signature = _parse_literal_signature(rendered_signature)
		if parsed_signature is None:
			continue
		_, _, is_positive = parsed_signature
		if not is_positive:
			continue
		requirements.append(rendered_signature)
	return requirements


def _render_producer_mode_options_for_predicate(
	predicate_name: str,
	predicate_args: Sequence[str],
	action_analysis: Dict[str, Any],
	*,
	limit: int = 3,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
	"""Render aligned producer modes for a predicate with their dynamic needs."""

	rendered_modes: list[tuple[str, tuple[str, ...]]] = []
	seen: set[tuple[str, tuple[str, ...]]] = set()
	target_signature = (
		predicate_name
		if not predicate_args
		else f"{predicate_name}({', '.join(predicate_args)})"
	)
	for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
		predicate_name,
		[],
	):
		effect_args = list(pattern.get("effect_args") or [])
		if len(effect_args) != len(predicate_args):
			continue

		token_mapping = {
			token: arg
			for token, arg in zip(effect_args, predicate_args)
		}
		rendered_action_args = _extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or [],
			action_parameter_types=pattern.get("action_parameter_types") or [],
		)
		rendered_requirements = tuple(
			_render_positive_dynamic_requirements(pattern, token_mapping)
		)
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
	rendered_modes = _filter_dominated_producer_modes(rendered_modes)
	return tuple(rendered_modes[:limit])


def _filter_dominated_producer_modes(
	rendered_modes: Sequence[tuple[str, tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...]]]:
	filtered_modes: list[tuple[str, tuple[str, ...]]] = []
	need_sets = [
		{
			str(signature).strip()
			for signature in needs
			if str(signature).strip()
		}
		for _, needs in rendered_modes
	]
	for index, mode in enumerate(rendered_modes):
		current_need_set = need_sets[index]
		if any(
			other_index != index
			and need_sets[other_index] < current_need_set
			for other_index in range(len(rendered_modes))
		):
			continue
		filtered_modes.append(mode)
	return filtered_modes


def _constructive_template_line_for_task(
	display_name: str,
	task_parameters: Sequence[str],
	predicate_name: str,
	action_analysis: Dict[str, Any],
	*,
	headline_parameters: Optional[Sequence[str]] = None,
	template_parameters: Optional[Sequence[str]] = None,
) -> Optional[str]:
	rendered_patterns = _constructive_template_summary_for_task(
		display_name,
		tuple(template_parameters or task_parameters),
		predicate_name,
		action_analysis,
	)
	if rendered_patterns is None:
		return None

	rendered_headline_parameters = tuple(headline_parameters or task_parameters)
	return (
		f"- {_task_invocation_signature(display_name, task_parameters)} targets "
		f"{predicate_name}({', '.join(rendered_headline_parameters)}); constructive templates: "
		f"{rendered_patterns}. Use one listed template as the final producer."
	)


def _exact_producer_slot_line_for_task(
	display_name: str,
	task_parameters: Sequence[str],
	predicate_name: str,
	action_analysis: Dict[str, Any],
	*,
	template_parameters: Optional[Sequence[str]] = None,
) -> Optional[str]:
	patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
		predicate_name,
		[],
	)
	if not patterns:
		return None

	mapping_parameters = tuple(template_parameters or task_parameters)
	role_labels = tuple(f"ARG{index}" for index in range(1, len(mapping_parameters) + 1))
	producer_slot_objects: list[Dict[str, str]] = []
	for pattern in patterns:
		effect_args = list(pattern.get("effect_args") or [])
		if len(effect_args) != len(mapping_parameters):
			continue
		token_mapping = {
			token: role_label
			for token, role_label in zip(effect_args, role_labels)
		}
		rendered_action_args = _extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or [],
			action_parameter_types=pattern.get("action_parameter_types") or [],
		)
		producer_slot_objects.append(
			{
				"producer": _task_invocation_signature(
					pattern["action_name"],
					rendered_action_args,
				),
			},
		)
	if not producer_slot_objects:
		return None
	if len(producer_slot_objects) == 1 and all(
		"AUX_" not in slot["producer"]
		for slot in producer_slot_objects
	):
		return None

	return (
		f"- {_task_invocation_signature(display_name, task_parameters)}: exact producer slots "
		f"{json.dumps(producer_slot_objects, separators=(',', ':'))}. "
		"Copy producer verbatim; do not reorder arguments or drop listed "
		"support_before/precondition/context obligations."
	)


def _declared_task_producer_template_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = anchor.get("task_name")
		display_name = _anchor_display_name(anchor)
		force_internal_contract = bool(anchor.get("force_internal_contract"))
		task = task_schemas.get(str(task_name))
		if task is None:
			continue
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue
		if predicate_name not in action_analysis.get("dynamic_predicates", ()):
			continue

		task_parameters = tuple(_parameter_token(parameter) for parameter in task.parameters)
		same_arity_candidates = _same_arity_packaging_candidates_for_query_task(
			domain,
			task_name,
			predicate_name,
			task_parameters,
			task_schemas,
			action_analysis,
		)
		if same_arity_candidates and not force_internal_contract:
			continue
		line = _constructive_template_line_for_task(
			display_name,
			task_parameters,
			predicate_name,
			action_analysis,
		)
		if line is None:
			continue
		if line in seen:
			continue
		seen.add(line)
		lines.append(line)
		exact_slot_line = _exact_producer_slot_line_for_task(
			display_name,
			task_parameters,
			predicate_name,
			action_analysis,
		)
		if exact_slot_line is None or exact_slot_line in seen:
			continue
		seen.add(exact_slot_line)
		lines.append(exact_slot_line)
	return lines


def _relevant_support_task_template_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		anchor_task_name = str(anchor.get("task_name", "")).strip()
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue

			headline_mapping = {
				token: f"ARG{index}"
				for index, token in enumerate(effect_args, start=1)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				parsed_precondition = _parse_literal_signature(
					_render_signature_with_mapping(precondition_signature, headline_mapping)
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(headline_mapping.values()):
					continue

				candidate_tasks = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						precondition_predicate,
						precondition_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							precondition_predicate,
							[],
						),
					)
					if candidate != anchor_task_name
					and (
						len(getattr(task_schemas.get(candidate), "parameters", ()))
						== len(precondition_args)
					)
				]
				for candidate_task in candidate_tasks[:2]:
					candidate_schema = task_schemas.get(candidate_task)
					if candidate_schema is None:
						continue
					candidate_parameters = tuple(
						_parameter_token(parameter)
						for parameter in candidate_schema.parameters
					)
					headline_parameters = _aligned_task_parameter_sequence_for_predicate(
						candidate_schema,
						precondition_predicate,
						action_analysis,
						candidate_parameters,
					)
					line = _constructive_template_line_for_task(
						candidate_task,
						candidate_parameters,
						precondition_predicate,
						action_analysis,
						headline_parameters=headline_parameters,
						template_parameters=headline_parameters,
					)
					if line is None or line in seen:
						continue
					seen.add(line)
					lines.append(line)
					exact_slot_line = _exact_producer_slot_line_for_task(
						candidate_task,
						candidate_parameters,
						precondition_predicate,
						action_analysis,
						template_parameters=headline_parameters,
					)
					if exact_slot_line is None or exact_slot_line in seen:
						continue
					seen.add(exact_slot_line)
					lines.append(exact_slot_line)
	return lines


def _alignment_warning_lines_for_task(
	display_name: str,
	task_parameters: Sequence[str],
	predicate_name: str,
	action_analysis: Dict[str, Any],
) -> list[str]:
	patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, [])
	lines: list[str] = []
	seen: set[str] = set()
	for pattern in patterns:
		effect_args = list(pattern.get("effect_args") or [])
		if len(effect_args) != len(task_parameters):
			continue

		token_mapping = {
			token: task_parameter
			for token, task_parameter in zip(effect_args, task_parameters)
		}
		rendered_action_args = _extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or [],
			action_parameter_types=pattern.get("action_parameter_types") or [],
		)
		extra_roles = [
			arg
			for arg in rendered_action_args
			if arg not in task_parameters
		]
		if not extra_roles:
			continue

		rendered_call = _task_invocation_signature(
			pattern["action_name"],
			rendered_action_args,
		)
		if len(task_parameters) == 1 and len(rendered_action_args) == 2:
			target_arg = task_parameters[0]
			extra_role = next(
				(arg for arg in rendered_action_args if arg != target_arg),
				extra_roles[0],
			)
			if rendered_action_args[0] == target_arg:
				swapped_args = (extra_role, target_arg)
			else:
				swapped_args = (target_arg, extra_role)
			swapped_call = _task_invocation_signature(
				pattern["action_name"],
				swapped_args,
			)
			line = (
				f"- {_task_invocation_signature(display_name, task_parameters)}: if you use "
				f"{rendered_call}, keep {target_arg} in the effect-aligned position shown above. "
				f"Do not swap it to {swapped_call}, because that would support "
				f"{predicate_name}({extra_role}) instead of {predicate_name}({target_arg})."
			)
		else:
			line = (
				f"- {_task_invocation_signature(display_name, task_parameters)}: keep task-target "
				f"arguments in the effect-aligned positions shown by {rendered_call}. Do not swap "
				"task targets with extra roles."
			)
		if line in seen:
			continue
		seen.add(line)
		lines.append(line)
	return lines


def _relevant_support_task_alignment_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		anchor_task_name = str(anchor.get("task_name", "")).strip()
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue

			headline_mapping = {
				token: f"ARG{index}"
				for index, token in enumerate(effect_args, start=1)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				parsed_precondition = _parse_literal_signature(
					_render_signature_with_mapping(precondition_signature, headline_mapping)
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(headline_mapping.values()):
					continue

				candidate_tasks = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						precondition_predicate,
						precondition_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							precondition_predicate,
							[],
						),
					)
					if candidate != anchor_task_name
					and (
						len(getattr(task_schemas.get(candidate), "parameters", ()))
						== len(precondition_args)
					)
				]
				for candidate_task in candidate_tasks[:2]:
					candidate_schema = task_schemas.get(candidate_task)
					if candidate_schema is None:
						continue
					candidate_parameters = tuple(
						_parameter_token(parameter)
						for parameter in candidate_schema.parameters
					)
					for line in _alignment_warning_lines_for_task(
						candidate_task,
						candidate_parameters,
						precondition_predicate,
						action_analysis,
					):
						if line in seen:
							continue
						seen.add(line)
						lines.append(line)
	return lines


def _declared_support_task_applicability_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		anchor_task_name = str(anchor.get("task_name", "")).strip()
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue

			headline_mapping = {
				token: f"ARG{index}"
				for index, token in enumerate(effect_args, start=1)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				parsed_precondition = _parse_literal_signature(
					_render_signature_with_mapping(precondition_signature, headline_mapping)
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(headline_mapping.values()):
					continue

				support_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
					precondition_predicate,
					[],
				)
				for support_pattern in support_patterns:
					support_effect_args = list(support_pattern.get("effect_args") or [])
					if len(support_effect_args) != len(precondition_args):
						continue

					support_mapping = {
						token: arg
						for token, arg in zip(support_effect_args, precondition_args)
					}
					_extend_mapping_with_action_parameters(
						support_mapping,
						support_pattern.get("action_parameters") or [],
						action_parameter_types=support_pattern.get("action_parameter_types") or [],
					)
					for support_requirement_signature in (
						support_pattern.get("dynamic_precondition_signatures") or []
					):
						parsed_support_requirement = _parse_literal_signature(
							_render_signature_with_mapping(
								support_requirement_signature,
								support_mapping,
							)
						)
						if parsed_support_requirement is None:
							continue
						(
							support_requirement_predicate,
							support_requirement_args,
							support_requirement_positive,
						) = parsed_support_requirement
						if not support_requirement_positive:
							continue

						candidate_tasks = [
							candidate
							for candidate in _candidate_support_task_names(
								domain,
								support_requirement_predicate,
								support_requirement_args,
								action_analysis.get("producer_actions_by_predicate", {}).get(
									support_requirement_predicate,
									[],
								),
							)
							if candidate != anchor_task_name
							and len(getattr(task_schemas.get(candidate), "parameters", ()))
							== len(support_requirement_args)
						]
						for candidate_task in candidate_tasks[:2]:
							candidate_schema = task_schemas.get(candidate_task)
							if candidate_schema is None:
								continue
							candidate_parameters = tuple(
								_parameter_token(parameter)
								for parameter in candidate_schema.parameters
							)
							headline_parameters = _aligned_task_parameter_sequence_for_predicate(
								candidate_schema,
								support_requirement_predicate,
								action_analysis,
								candidate_parameters,
							)
							template_summary = _constructive_template_summary_for_task(
								candidate_task,
								headline_parameters,
								support_requirement_predicate,
								action_analysis,
							)
							shared_requirements = _shared_dynamic_requirements_for_predicate(
								support_requirement_predicate,
								headline_parameters,
								action_analysis,
							)
							if template_summary is None and not shared_requirements:
								continue

							line = (
								f"- {_task_invocation_signature(candidate_task, candidate_parameters)} "
								f"can serve as a declared support task for "
								f"{support_requirement_predicate}({', '.join(headline_parameters)})"
							)
							if template_summary is not None:
								line += (
									f"; constructive templates: {template_summary}. Use one listed "
									"template as the final producer"
								)
							if shared_requirements:
								line += (
									"; if a parent calls it, first provide shared prerequisites "
									f"{', '.join(shared_requirements)}"
								)
							if line in seen:
								continue
							seen.add(line)
							lines.append(line)
	return lines


def _relevant_support_task_internal_obligation_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	query_task_names = {
		str(anchor.get("task_name", "")).strip()
		for anchor in query_task_anchors
		if str(anchor.get("task_name", "")).strip()
	}
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		anchor_task_name = str(anchor.get("task_name", "")).strip()
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue

			headline_mapping = {
				token: f"ARG{index}"
				for index, token in enumerate(effect_args, start=1)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				parsed_precondition = _parse_literal_signature(
					_render_signature_with_mapping(precondition_signature, headline_mapping)
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(headline_mapping.values()):
					continue

				candidate_tasks = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						precondition_predicate,
						precondition_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							precondition_predicate,
							[],
						),
					)
					if candidate != anchor_task_name
					and (
						len(getattr(task_schemas.get(candidate), "parameters", ()))
						== len(precondition_args)
					)
				]
				for candidate_task in candidate_tasks[:2]:
					candidate_schema = task_schemas.get(candidate_task)
					if candidate_schema is None:
						continue
					candidate_parameters = tuple(
						_parameter_token(parameter)
						for parameter in candidate_schema.parameters
					)
					for support_pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
						precondition_predicate,
						[],
					):
						support_effect_args = list(support_pattern.get("effect_args") or [])
						if len(support_effect_args) != len(candidate_parameters):
							continue

						support_mapping = {
							token: arg
							for token, arg in zip(support_effect_args, candidate_parameters)
						}
						rendered_action_args = _extend_mapping_with_action_parameters(
							support_mapping,
							support_pattern.get("action_parameters") or [],
							action_parameter_types=support_pattern.get("action_parameter_types") or [],
						)
						rendered_consumed_requirements: set[str] = set()
						for signature in (support_pattern.get("negative_effect_signatures") or []):
							rendered_signature = _render_signature_with_mapping(
								signature,
								support_mapping,
							)
							if rendered_signature.startswith("not "):
								rendered_consumed_requirements.add(rendered_signature[4:].strip())
						rendered_requirements: list[str] = []
						for requirement_signature in (
							support_pattern.get("dynamic_precondition_signatures") or []
						):
							rendered_requirement = _render_signature_with_mapping(
								requirement_signature,
								support_mapping,
							)
							parsed_requirement = _parse_literal_signature(rendered_requirement)
							if parsed_requirement is None:
								continue
							_, _, requirement_positive = parsed_requirement
							if not requirement_positive:
								continue
							rendered_requirements.append(rendered_requirement)
						if not rendered_requirements:
							continue

						extra_roles = [
							arg
							for arg in rendered_action_args
							if arg not in candidate_parameters
						]
						requirement_fragments: list[str] = []
						task_parameter_set = set(candidate_parameters)
						extra_role_set = set(extra_roles)
						mode_has_aux_binding_requirement = any(
							(
								(parsed_requirement := _parse_literal_signature(requirement)) is not None
								and parsed_requirement[2]
								and _is_aux_binding_requirement(
									parsed_requirement[1],
									task_parameter_symbols=task_parameter_set,
									extra_role_symbols=extra_role_set,
								)
							)
							for requirement in rendered_requirements
						)
						needs_availability_split_guidance = False
						for rendered_requirement in rendered_requirements:
							parsed_requirement = _parse_literal_signature(rendered_requirement)
							if parsed_requirement is None:
								continue
							req_predicate, req_args, req_positive = parsed_requirement
							if not req_positive:
								continue
							requirement_task_candidates = [
								candidate
								for candidate in _candidate_support_task_names(
									domain,
									req_predicate,
									req_args,
									action_analysis.get("producer_actions_by_predicate", {}).get(
										req_predicate,
										[],
									),
								)
								if candidate not in query_task_names
								if len(getattr(task_schemas.get(candidate), "parameters", ()))
								== len(req_args)
							]
							has_task_parameter = any(arg in task_parameter_set for arg in req_args)
							has_extra_role = any(arg in extra_role_set for arg in req_args)
							is_aux_binding_requirement = _is_aux_binding_requirement(
								req_args,
								task_parameter_symbols=task_parameter_set,
								extra_role_symbols=extra_role_set,
							)
							if is_aux_binding_requirement:
								requirement_fragments.append(
									f"{rendered_requirement} explicit in method.context as the selected producer mode condition"
								)
							elif requirement_task_candidates and has_extra_role:
								if mode_has_aux_binding_requirement:
									needs_availability_split_guidance = True
									requirement_fragments.append(
										f"{rendered_requirement} via "
										f"{' or '.join(_headline_support_task_invocation_signature(candidate, req_predicate, req_args, task_schemas, action_analysis) for candidate in requirement_task_candidates[:2])} "
										f"before {_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}, not in method.context"
									)
								else:
									requirement_fragments.append(
										f"{rendered_requirement} explicit in method.context before "
										f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}; "
										"do not recurse on unconstrained AUX roles for this mode"
									)
							elif requirement_task_candidates and not has_task_parameter:
								requirement_fragments.append(
									f"{rendered_requirement} via "
									f"{' or '.join(_headline_support_task_invocation_signature(candidate, req_predicate, req_args, task_schemas, action_analysis) for candidate in requirement_task_candidates[:2])} "
									f"before {_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}"
								)
							elif requirement_task_candidates:
								requirement_fragments.append(
									f"{rendered_requirement} via "
									f"{' or '.join(_headline_support_task_invocation_signature(candidate, req_predicate, req_args, task_schemas, action_analysis) for candidate in requirement_task_candidates[:2])} "
									f"before {_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}"
								)
							elif req_args:
								requirement_fragments.append(
									f"{rendered_requirement} in method.context or earlier subtasks"
								)
							else:
								requirement_fragments.append(
									f"{rendered_requirement} explicit in method.context or earlier subtasks"
								)
						if not requirement_fragments:
							continue
						line = (
							f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: if a "
							f"constructive sibling uses "
							f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)} "
							f"to make {precondition_predicate}({', '.join(candidate_parameters)}), support "
							f"{'; '.join(requirement_fragments)}"
						)
						if extra_roles:
							line += (
								f"; declare extra roles {', '.join(extra_roles)} in method.parameters"
							)
						if needs_availability_split_guidance:
							line += (
								"; if that extra-role requirement may already hold or require "
								"recursive support, keep those as separate constructive siblings "
								"rather than one precondition-only branch"
							)
						line += ". Prefer simpler modes with fewer extra roles when they remain suitable."
						if line in seen:
							continue
						seen.add(line)
						lines.append(line)
	return lines


def _support_task_precise_shared_requirements(
	domain: Any,
	candidate_task: str,
	support_predicate: str,
	candidate_args: Sequence[str],
	task_schemas: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> tuple[str, ...]:
	candidate_schema = task_schemas.get(candidate_task)
	if candidate_schema is None:
		return ()
	candidate_parameters = tuple(
		_parameter_token(parameter)
		for parameter in getattr(candidate_schema, "parameters", ()) or ()
	)
	if len(candidate_parameters) != len(candidate_args):
		return ()

	headline_predicates = tuple(
		str(predicate_name).strip()
		for predicate_name in (getattr(candidate_schema, "source_predicates", ()) or ())
		if str(predicate_name).strip()
	)
	if support_predicate in headline_predicates:
		selected_headline_predicates = (support_predicate,)
	elif headline_predicates:
		selected_headline_predicates = headline_predicates[:1]
	else:
		selected_headline_predicates = (support_predicate,)

	pattern_shared_sets: list[set[str]] = []
	for headline_predicate in selected_headline_predicates:
		aligned_candidate_args = _aligned_task_parameter_sequence_for_predicate(
			candidate_schema,
			headline_predicate,
			action_analysis,
			candidate_args,
		)
		for support_pattern in action_analysis.get(
			"producer_patterns_by_predicate",
			{},
		).get(headline_predicate, []):
			support_effect_args = list(support_pattern.get("effect_args") or [])
			if len(support_effect_args) != len(candidate_args):
				continue

			support_mapping = {
				token: arg
				for token, arg in zip(support_effect_args, aligned_candidate_args)
			}
			rendered_action_args = _extend_mapping_with_action_parameters(
				support_mapping,
				support_pattern.get("action_parameters") or [],
				action_parameter_types=support_pattern.get("action_parameter_types") or [],
			)
			extra_roles = {
				arg
				for arg in rendered_action_args
				if arg not in candidate_args
			}
			pattern_requirements: set[str] = set()
			for requirement_signature in support_pattern.get("dynamic_precondition_signatures") or []:
				rendered_requirement = _render_signature_with_mapping(
					requirement_signature,
					support_mapping,
				)
				parsed_requirement = _parse_literal_signature(rendered_requirement)
				if parsed_requirement is None:
					continue
				_, req_args, req_positive = parsed_requirement
				if not req_positive:
					continue
				if not req_args or any(arg in extra_roles for arg in req_args):
					continue
				if not all(arg in candidate_args for arg in req_args):
					continue
				pattern_requirements.add(rendered_requirement)
			pattern_shared_sets.append(pattern_requirements)

	if not pattern_shared_sets:
		return ()
	shared_requirements = set.intersection(*pattern_shared_sets)
	return tuple(sorted(shared_requirements))


def _support_task_caller_shared_prerequisite_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	query_task_names = {
		str(anchor.get("task_name", "")).strip()
		for anchor in query_task_anchors
		if str(anchor.get("task_name", "")).strip()
	}
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		anchor_task_name = str(anchor.get("task_name", "")).strip()
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue
			headline_mapping = {
				token: f"ARG{index}"
				for index, token in enumerate(effect_args, start=1)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				parsed_precondition = _parse_literal_signature(
					_render_signature_with_mapping(precondition_signature, headline_mapping)
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(headline_mapping.values()):
					continue

				candidate_tasks = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						precondition_predicate,
						precondition_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							precondition_predicate,
							[],
						),
					)
					if candidate != anchor_task_name
					and len(getattr(task_schemas.get(candidate), "parameters", ())) == len(precondition_args)
				]
				for candidate_task in candidate_tasks[:2]:
					candidate_schema = task_schemas.get(candidate_task)
					if candidate_schema is None:
						continue
					candidate_parameters = tuple(
						_parameter_token(parameter)
						for parameter in candidate_schema.parameters
					)
					shared_requirements = _support_task_precise_shared_requirements(
						domain,
						candidate_task,
						precondition_predicate,
						candidate_parameters,
						task_schemas,
						action_analysis,
					)
					filtered_shared_requirements = []
					for shared_requirement in shared_requirements:
						parsed_shared_requirement = _parse_literal_signature(shared_requirement)
						if parsed_shared_requirement is None:
							filtered_shared_requirements.append(shared_requirement)
							continue
						shared_predicate, shared_args, shared_positive = parsed_shared_requirement
						if not shared_positive or not shared_args:
							filtered_shared_requirements.append(shared_requirement)
							continue
						internal_support_tasks = [
							support_task
							for support_task in _candidate_support_task_names(
								domain,
								shared_predicate,
								shared_args,
								action_analysis.get("producer_actions_by_predicate", {}).get(
									shared_predicate,
									[],
								),
							)
							if support_task != candidate_task
							and support_task not in query_task_names
							and len(getattr(task_schemas.get(support_task), "parameters", ()))
							== len(shared_args)
						]
						if internal_support_tasks:
							continue
						filtered_shared_requirements.append(shared_requirement)
					if not filtered_shared_requirements:
						continue
					line = (
						f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: caller-shared "
						f"dynamic prerequisites {', '.join(filtered_shared_requirements)}. If a parent calls "
						f"{_task_invocation_signature(candidate_task, candidate_parameters)}, support "
						"them before the child call."
					)
					if line in seen:
						continue
					seen.add(line)
					lines.append(line)
	return lines


def _relevant_support_task_recursive_mode_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		anchor_task_name = str(anchor.get("task_name", "")).strip()
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue

			headline_mapping = {
				token: f"ARG{index}"
				for index, token in enumerate(effect_args, start=1)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				parsed_precondition = _parse_literal_signature(
					_render_signature_with_mapping(precondition_signature, headline_mapping)
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(headline_mapping.values()):
					continue

				candidate_tasks = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						precondition_predicate,
						precondition_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							precondition_predicate,
							[],
						),
					)
					if candidate != anchor_task_name
					and (
						len(getattr(task_schemas.get(candidate), "parameters", ()))
						== len(precondition_args)
					)
				]
				for candidate_task in candidate_tasks[:2]:
					candidate_schema = task_schemas.get(candidate_task)
					if candidate_schema is None:
						continue
					candidate_parameters = tuple(
						_parameter_token(parameter)
						for parameter in candidate_schema.parameters
					)
					for support_pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
						precondition_predicate,
						[],
					):
						support_effect_args = list(support_pattern.get("effect_args") or [])
						if len(support_effect_args) != len(candidate_parameters):
							continue

						support_mapping = {
							token: arg
							for token, arg in zip(support_effect_args, candidate_parameters)
						}
						rendered_action_args = _extend_mapping_with_action_parameters(
							support_mapping,
							support_pattern.get("action_parameters") or [],
							action_parameter_types=support_pattern.get("action_parameter_types") or [],
						)
						extra_roles = [
							arg
							for arg in rendered_action_args
							if arg not in candidate_parameters
						]
						if not extra_roles:
							continue
						for requirement_signature in (
							support_pattern.get("dynamic_precondition_signatures") or []
						):
							rendered_requirement = _render_signature_with_mapping(
								requirement_signature,
								support_mapping,
							)
							parsed_requirement = _parse_literal_signature(rendered_requirement)
							if parsed_requirement is None:
								continue
							req_predicate, req_args, req_positive = parsed_requirement
							if not req_positive:
								continue
							requirement_task_candidates = [
								candidate
								for candidate in _candidate_support_task_names(
									domain,
									req_predicate,
									req_args,
									action_analysis.get("producer_actions_by_predicate", {}).get(
										req_predicate,
										[],
									),
								)
								if len(getattr(task_schemas.get(candidate), "parameters", ()))
								== len(req_args)
							]
							if candidate_task not in requirement_task_candidates:
								continue
							if any(arg in candidate_parameters for arg in req_args):
								continue
							line = (
								f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: "
								f"recursive support is valid. If "
								f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)} "
								f"needs {rendered_requirement}, call "
								f"{_headline_support_task_invocation_signature(candidate_task, req_predicate, req_args, task_schemas, action_analysis)} before the primitive step."
							)
							if line in seen:
								continue
							seen.add(line)
							lines.append(line)
							other_requirements = [
								_render_signature_with_mapping(signature, support_mapping)
								for signature in (
									support_pattern.get("dynamic_precondition_signatures") or []
								)
								if _render_signature_with_mapping(signature, support_mapping)
								!= rendered_requirement
							]
							if other_requirements:
								other_needs_line = (
									f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: "
									f"if a constructive sibling adds "
									f"{_headline_support_task_invocation_signature(candidate_task, req_predicate, req_args, task_schemas, action_analysis)} before "
									f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}, "
									f"keep that mode's other unmet needs {', '.join(other_requirements)} "
									"in the same sibling precondition/context unless earlier subtasks in "
									"that sibling already support them. Binding literals among those "
									"needs still constrain the AUX role and cannot be dropped."
								)
								if other_needs_line not in seen:
									seen.add(other_needs_line)
									lines.append(other_needs_line)
								for other_requirement in other_requirements:
									parsed_other_requirement = _parse_literal_signature(
										other_requirement,
									)
									if parsed_other_requirement is None:
										continue
									(
										other_predicate,
										other_args,
										other_positive,
									) = parsed_other_requirement
									if not other_positive or not other_args:
										continue
									restoration_task_candidates = [
										candidate
										for candidate in _candidate_support_task_names(
											domain,
											other_predicate,
											other_args,
											action_analysis.get(
												"producer_actions_by_predicate",
												{},
											).get(other_predicate, []),
										)
										if len(
											getattr(task_schemas.get(candidate), "parameters", ())
										)
										== len(other_args)
									]
									if (
										not restoration_task_candidates
										or not _signature_mentions_aux_role(other_requirement)
									):
										continue
									restoration_calls = " or ".join(
										_headline_support_task_invocation_signature(
											candidate,
											other_predicate,
											other_args,
											task_schemas,
											action_analysis,
										)
										for candidate in restoration_task_candidates[:2]
									)
									restoration_line = (
										f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: "
										f"if a constructive sibling adds "
										f"{_headline_support_task_invocation_signature(candidate_task, req_predicate, req_args, task_schemas, action_analysis)} before "
										f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}, "
										f"do not keep later supportable need {other_requirement} only in "
										f"precondition/context; restore it with {restoration_calls} after "
										f"{_headline_support_task_invocation_signature(candidate_task, req_predicate, req_args, task_schemas, action_analysis)} and before "
										f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}."
									)
									if restoration_line not in seen:
										seen.add(restoration_line)
										lines.append(restoration_line)
								binding_needs = [
									requirement
									for requirement in other_requirements
									if _signature_mentions_aux_role(requirement)
								]
								if binding_needs:
									binding_line = (
										f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: "
										f"if a constructive sibling adds "
										f"{_headline_support_task_invocation_signature(candidate_task, req_predicate, req_args, task_schemas, action_analysis)} before "
										f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}, "
										f"keep binding literal(s) {', '.join(binding_needs)} in that sibling "
										"precondition/context before the first use of the AUX role."
									)
									if binding_line not in seen:
										seen.add(binding_line)
										lines.append(binding_line)
	return lines


def _relevant_support_task_recursive_template_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		anchor_task_name = str(anchor.get("task_name", "")).strip()
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue

			headline_mapping = {
				token: f"ARG{index}"
				for index, token in enumerate(effect_args, start=1)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				parsed_precondition = _parse_literal_signature(
					_render_signature_with_mapping(precondition_signature, headline_mapping)
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(headline_mapping.values()):
					continue

				candidate_tasks = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						precondition_predicate,
						precondition_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							precondition_predicate,
							[],
						),
					)
					if candidate != anchor_task_name
					and (
						len(getattr(task_schemas.get(candidate), "parameters", ()))
						== len(precondition_args)
					)
				]
				for candidate_task in candidate_tasks[:2]:
					candidate_schema = task_schemas.get(candidate_task)
					if candidate_schema is None:
						continue
					candidate_parameters = tuple(
						_parameter_token(parameter)
						for parameter in candidate_schema.parameters
					)
					for support_pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
						precondition_predicate,
						[],
					):
						support_effect_args = list(support_pattern.get("effect_args") or [])
						if len(support_effect_args) != len(candidate_parameters):
							continue

						support_mapping = {
							token: arg
							for token, arg in zip(support_effect_args, candidate_parameters)
						}
						rendered_action_args = _extend_mapping_with_action_parameters(
							support_mapping,
							support_pattern.get("action_parameters") or [],
							action_parameter_types=support_pattern.get("action_parameter_types") or [],
						)
						extra_roles = [
							arg
							for arg in rendered_action_args
							if arg not in candidate_parameters
						]
						if not extra_roles:
							continue

						recursive_requirements: list[str] = []
						mode_context_requirements: list[str] = []
						for requirement_signature in (
							support_pattern.get("dynamic_precondition_signatures") or []
						):
							rendered_requirement = _render_signature_with_mapping(
								requirement_signature,
								support_mapping,
							)
							parsed_requirement = _parse_literal_signature(rendered_requirement)
							if parsed_requirement is None:
								continue
							req_predicate, req_args, req_positive = parsed_requirement
							if not req_positive:
								continue
							requirement_task_candidates = [
								candidate
								for candidate in _candidate_support_task_names(
									domain,
									req_predicate,
									req_args,
									action_analysis.get("producer_actions_by_predicate", {}).get(
										req_predicate,
										[],
									),
								)
								if len(getattr(task_schemas.get(candidate), "parameters", ()))
								== len(req_args)
							]
							if candidate_task in requirement_task_candidates and not any(
								arg in candidate_parameters for arg in req_args
							):
								recursive_requirements.append(rendered_requirement)
							else:
								mode_context_requirements.append(rendered_requirement)
						if not recursive_requirements:
							continue
						if extra_roles:
							filtered_recursive_requirements: list[str] = []
							for requirement in recursive_requirements:
								parsed_recursive_requirement = _parse_literal_signature(requirement)
								if parsed_recursive_requirement is None:
									continue
								_, recursive_args, recursive_positive = parsed_recursive_requirement
								if not recursive_positive:
									continue
								recursive_extra_roles = {
									arg
									for arg in recursive_args
									if arg in extra_roles
								}
								if not recursive_extra_roles:
									filtered_recursive_requirements.append(requirement)
									continue
								has_binding_context = False
								for context_requirement in mode_context_requirements:
									parsed_context_requirement = _parse_literal_signature(
										context_requirement,
									)
									if parsed_context_requirement is None:
										continue
									_, context_args, context_positive = parsed_context_requirement
									if not context_positive:
										continue
									if _is_aux_binding_requirement(
										context_args,
										task_parameter_symbols=set(candidate_parameters),
										extra_role_symbols=recursive_extra_roles,
									):
										has_binding_context = True
										break
								if has_binding_context:
									filtered_recursive_requirements.append(requirement)
							recursive_requirements = filtered_recursive_requirements
						if not recursive_requirements:
							continue
						recursive_support_calls = [
							_task_invocation_signature(
								candidate_task,
								_parse_literal_signature(requirement)[1],
							)
							for requirement in recursive_requirements
						]
						cleanup_followup_call = _cleanup_followup_call_for_support_pattern(
							support_pattern=support_pattern,
							support_mapping=support_mapping,
							extra_roles=extra_roles,
							action_analysis=action_analysis,
						)
						line = (
							f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: "
							f"recursive template for {_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}. "
							f"Keep mode context {'; '.join(mode_context_requirements)} and use subtasks "
							f"{'; '.join(recursive_support_calls)}; "
							f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}."
						)
						if cleanup_followup_call:
							line = line.removesuffix(".") + f"; followup {cleanup_followup_call}."
						if line in seen:
							continue
						seen.add(line)
						lines.append(line)
						slot_line = (
							f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: "
							f"valid recursive slot sibling for "
							f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}: "
							f"precondition/context {'; '.join(mode_context_requirements)}; "
							f"support_before {'; '.join(recursive_support_calls)}; "
							f"producer {_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}"
						)
						if cleanup_followup_call:
							slot_line += f"; followup {cleanup_followup_call}."
						else:
							slot_line += "."
						if slot_line in seen:
							continue
						seen.add(slot_line)
						lines.append(slot_line)
						ast_slot_example = (
							f'- {_task_invocation_signature(candidate_task, candidate_parameters)}: '
							"AST slot shape for that recursive sibling: "
							"{"
							f'"precondition":[{", ".join(json.dumps(item) for item in mode_context_requirements)}],'
							f'"support_before":[{", ".join(json.dumps(item) for item in recursive_support_calls)}],'
							f'"producer":{json.dumps(_task_invocation_signature(support_pattern["action_name"], rendered_action_args))}'
						)
						if cleanup_followup_call:
							ast_slot_example += f',"followup":{json.dumps(cleanup_followup_call)}'
						ast_slot_example += "}."
						if ast_slot_example not in seen:
							seen.add(ast_slot_example)
							lines.append(ast_slot_example)
						already_supported_context_requirements = [
							*mode_context_requirements,
							*recursive_requirements,
						]
						already_supported_ast_slot = (
							f'- {_task_invocation_signature(candidate_task, candidate_parameters)}: '
							"AST slot shape for the already-supported recursive sibling: "
							"{"
							f'"precondition":[{", ".join(json.dumps(item) for item in already_supported_context_requirements)}],'
							f'"producer":{json.dumps(_task_invocation_signature(support_pattern["action_name"], rendered_action_args))}'
						)
						if cleanup_followup_call:
							already_supported_ast_slot += (
								f',"followup":{json.dumps(cleanup_followup_call)}'
							)
						already_supported_ast_slot += "}."
						if already_supported_ast_slot not in seen:
							seen.add(already_supported_ast_slot)
							lines.append(already_supported_ast_slot)
						for requirement, support_call in zip(
							recursive_requirements,
							recursive_support_calls,
						):
							split_line = (
								f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: "
								f"if support_before {support_call} handles false-{requirement} before "
								f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}, "
								f"do not also keep {requirement} in that same sibling "
								"precondition/context; split already-supported and recursive-support "
								"siblings."
							)
							if split_line in seen:
								continue
							seen.add(split_line)
							lines.append(split_line)
	return lines


def _cleanup_followup_call_for_support_pattern(
	*,
	support_pattern: Dict[str, Any],
	support_mapping: Dict[str, str],
	extra_roles: Sequence[str],
	action_analysis: Dict[str, Any],
) -> Optional[str]:
	if len(extra_roles) != 1:
		return None
	extra_role = extra_roles[0]

	rendered_negative_zeroary: list[str] = []
	for signature in (support_pattern.get("negative_effect_signatures") or []):
		rendered_signature = _render_signature_with_mapping(signature, support_mapping)
		if not rendered_signature.startswith("not "):
			continue
		positive_signature = rendered_signature[4:].strip()
		parsed_signature = _parse_literal_signature(positive_signature)
		if parsed_signature is None:
			continue
		_, args, positive = parsed_signature
		if positive and not args:
			rendered_negative_zeroary.append(positive_signature)

	rendered_positive_effects = {
		_render_signature_with_mapping(signature, support_mapping)
		for signature in (support_pattern.get("positive_effect_signatures") or [])
	}
	for lost_zeroary in rendered_negative_zeroary:
		for cleanup_pattern in action_analysis.get(
			"producer_patterns_by_predicate",
			{},
		).get(lost_zeroary, []):
			action_parameters = list(cleanup_pattern.get("action_parameters") or [])
			if len(action_parameters) != 1:
				continue
			cleanup_mapping = {
				_parameter_token(action_parameters[0]): extra_role,
			}
			rendered_cleanup_args = _extend_mapping_with_action_parameters(
				cleanup_mapping,
				action_parameters,
				action_parameter_types=cleanup_pattern.get("action_parameter_types") or [],
			)
			rendered_cleanup_preconditions = {
				_render_signature_with_mapping(signature, cleanup_mapping)
				for signature in (
					cleanup_pattern.get("dynamic_precondition_signatures") or []
				)
			}
			if not rendered_cleanup_preconditions & rendered_positive_effects:
				continue
			return _task_invocation_signature(
				cleanup_pattern["action_name"],
				rendered_cleanup_args,
			)
	return None


def _relevant_support_task_cleanup_template_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		anchor_task_name = str(anchor.get("task_name", "")).strip()
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue

			headline_mapping = {
				token: f"ARG{index}"
				for index, token in enumerate(effect_args, start=1)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				parsed_precondition = _parse_literal_signature(
					_render_signature_with_mapping(precondition_signature, headline_mapping)
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(headline_mapping.values()):
					continue

				candidate_tasks = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						precondition_predicate,
						precondition_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							precondition_predicate,
							[],
						),
					)
					if candidate != anchor_task_name
					and (
						len(getattr(task_schemas.get(candidate), "parameters", ()))
						== len(precondition_args)
					)
				]
				for candidate_task in candidate_tasks[:2]:
					candidate_schema = task_schemas.get(candidate_task)
					if candidate_schema is None:
						continue
					candidate_parameters = tuple(
						_parameter_token(parameter)
						for parameter in candidate_schema.parameters
					)
					for support_pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
						precondition_predicate,
						[],
					):
						support_effect_args = list(support_pattern.get("effect_args") or [])
						if len(support_effect_args) != len(candidate_parameters):
							continue

						support_mapping = {
							token: arg
							for token, arg in zip(support_effect_args, candidate_parameters)
						}
						rendered_action_args = _extend_mapping_with_action_parameters(
							support_mapping,
							support_pattern.get("action_parameters") or [],
							action_parameter_types=support_pattern.get("action_parameter_types") or [],
						)
						extra_roles = [
							arg
							for arg in rendered_action_args
							if arg not in candidate_parameters
						]
						if len(extra_roles) != 1:
							continue
						extra_role = extra_roles[0]

						rendered_negative_zeroary: list[str] = []
						for signature in (support_pattern.get("negative_effect_signatures") or []):
							rendered_signature = _render_signature_with_mapping(
								signature,
								support_mapping,
							)
							if not rendered_signature.startswith("not "):
								continue
							positive_signature = rendered_signature[4:].strip()
							parsed_signature = _parse_literal_signature(positive_signature)
							if parsed_signature is None:
								continue
							_, args, positive = parsed_signature
							if positive and not args:
								rendered_negative_zeroary.append(positive_signature)

						rendered_positive_effects: set[str] = set()
						for signature in (support_pattern.get("positive_effect_signatures") or []):
							rendered_signature = _render_signature_with_mapping(
								signature,
								support_mapping,
							)
							rendered_positive_effects.add(rendered_signature)

						for lost_zeroary in rendered_negative_zeroary:
							for cleanup_pattern in action_analysis.get(
								"producer_patterns_by_predicate",
								{},
							).get(lost_zeroary, []):
								action_parameters = list(cleanup_pattern.get("action_parameters") or [])
								if len(action_parameters) != 1:
									continue
								cleanup_mapping = {
									_parameter_token(action_parameters[0]): extra_role,
								}
								rendered_cleanup_args = _extend_mapping_with_action_parameters(
									cleanup_mapping,
									action_parameters,
									action_parameter_types=cleanup_pattern.get("action_parameter_types") or [],
								)
								rendered_cleanup_preconditions = {
									_render_signature_with_mapping(signature, cleanup_mapping)
									for signature in (
										cleanup_pattern.get("dynamic_precondition_signatures") or []
									)
								}
								if not rendered_cleanup_preconditions & rendered_positive_effects:
									continue
								line = (
									f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: "
									f"cleanup followup after {_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}. "
									f"If this branch should return with {lost_zeroary} restored, use "
									f"followup {_task_invocation_signature(cleanup_pattern['action_name'], rendered_cleanup_args)} "
									"before returning."
								)
								if line in seen:
									continue
								seen.add(line)
								lines.append(line)
	return lines


def _query_task_role_frame_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()

	def append_role_lines(
		*,
		display_name: str,
		task_parameters: Sequence[str],
		predicate_name: str,
		render_prefix: str = "",
	) -> None:
		patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, [])
		for pattern in patterns:
			effect_args = list(pattern.get("effect_args") or [])
			if len(effect_args) != len(task_parameters):
				continue

			token_mapping = {
				token: task_parameter
				for token, task_parameter in zip(effect_args, task_parameters)
			}
			used_labels = set(token_mapping.values())
			rendered_action_args: list[str] = []
			extra_roles: list[str] = []
			action_parameters = list(pattern.get("action_parameters") or [])
			action_parameter_types = list(pattern.get("action_parameter_types") or [])
			for index, action_parameter in enumerate(
				_parameter_token(parameter)
				for parameter in action_parameters
			):
				if action_parameter not in token_mapping:
					type_name = (
						action_parameter_types[index]
						if index < len(action_parameter_types)
						else "object"
					)
					token_mapping[action_parameter] = _allocate_placeholder_label(
						used_labels,
						type_name=type_name,
					)
					extra_roles.append(
						f"{token_mapping[action_parameter]} - {type_name}"
					)
				rendered_action_args.append(token_mapping[action_parameter])

			if not extra_roles:
				continue

			rendered_requirements = _render_positive_dynamic_requirements(
				pattern,
				token_mapping,
			)
			line = (
				f"- {_task_invocation_signature(display_name, task_parameters)}: "
				f"{render_prefix}{_task_invocation_signature(pattern['action_name'], rendered_action_args)} "
				f"introduces extra roles {', '.join(extra_roles)}. Keep them as distinct "
				"method.parameters or earlier schematic child bindings; never substitute "
				"grounded query objects."
			)
			if rendered_requirements:
				line += (
					f" If you choose this mode, support "
					f"{', '.join(rendered_requirements)} before that step."
				)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)

	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		force_internal_contract = bool(anchor.get("force_internal_contract"))
		task_schema = task_schemas.get(task_name)
		if task_schema is None:
			continue

		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue

		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		same_arity_candidates = _same_arity_packaging_candidates_for_query_task(
			domain,
			task_name,
			predicate_name,
			task_parameters,
			task_schemas,
			action_analysis,
		)
		packaging_shared_requirements = {
				str(requirement).strip()
				for candidate in same_arity_candidates
				for requirement in _same_arity_caller_shared_requirements(
					domain,
					predicate_name,
					task_parameters,
					action_analysis,
					candidate,
				)
				if str(requirement).strip()
			}
		candidate_support_tasks = _candidate_support_task_names(
			domain,
			predicate_name,
			task_parameters,
			action_analysis.get("producer_actions_by_predicate", {}).get(predicate_name, []),
		)
		if not any(candidate != task_name for candidate in candidate_support_tasks):
			append_role_lines(
				display_name=display_name,
				task_parameters=task_parameters,
				predicate_name=predicate_name,
			)
		for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, []):
			effect_args = list(pattern.get("effect_args") or [])
			if len(effect_args) != len(task_parameters):
				continue
			token_mapping = {
				token: task_parameter
				for token, task_parameter in zip(effect_args, task_parameters)
			}
			for signature in pattern.get("dynamic_precondition_signatures") or []:
				rendered_support_signature = _render_signature_with_mapping(
					signature,
					token_mapping,
				)
				if (
					not force_internal_contract
					and
					packaging_shared_requirements
					and rendered_support_signature not in packaging_shared_requirements
				):
					continue
				parsed_support = _parse_literal_signature(rendered_support_signature)
				if parsed_support is None:
					continue
				support_predicate, support_args, support_positive = parsed_support
				if not support_positive:
					continue
				if not set(support_args) & set(task_parameters):
					continue
				append_role_lines(
					display_name=display_name,
					task_parameters=support_args,
					predicate_name=support_predicate,
					render_prefix=f"to support {support_predicate}({', '.join(support_args)}), ",
				)
	return lines


def _candidate_headline_predicates_for_task(
	task_name: str,
	task_arity: int,
	action_analysis: Dict[str, Any],
) -> list[str]:
	task_tokens = _name_tokens(task_name)
	task_compact = _compact_name_tokens(task_name)
	token_frequencies = _domain_name_token_frequencies(action_analysis)
	scored_predicates: list[tuple[float, str]] = []
	for predicate_name, patterns in action_analysis.get("producer_patterns_by_predicate", {}).items():
		if not any(len(pattern.get("effect_args") or []) == task_arity for pattern in patterns):
			continue
		score = _weighted_token_overlap_score(
			task_tokens,
			_name_tokens(predicate_name),
			token_frequencies=token_frequencies,
		)
		predicate_compact = _compact_name_tokens(predicate_name)
		if task_compact and predicate_compact:
			if task_compact == predicate_compact:
				score += 6
			elif min(len(task_compact), len(predicate_compact)) >= 4:
				if task_compact.endswith(predicate_compact) or predicate_compact.endswith(task_compact):
					score += 4
				elif predicate_compact in task_compact or task_compact in predicate_compact:
					score += 2
		for pattern in patterns:
			score += _weighted_token_overlap_score(
				task_tokens,
				_name_tokens(str(pattern.get("action_name") or "")),
				token_frequencies=token_frequencies,
			)
		if score <= 0:
			continue
		scored_predicates.append((score, predicate_name))
	scored_predicates.sort(key=lambda item: (-item[0], item[1]))
	return [predicate_name for _, predicate_name in scored_predicates]


def _aligned_task_parameter_sequence_for_predicate(
	task_schema: Any,
	predicate_name: str,
	action_analysis: Dict[str, Any],
	labels: Sequence[str],
) -> tuple[str, ...]:
	task_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
	task_parameter_types = tuple(
		_sanitize_name(_parameter_type(parameter)).lower()
		for parameter in task_parameters
	)
	if len(task_parameter_types) != len(labels):
		return tuple(labels)

	for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
		predicate_name,
		[],
	):
		effect_args = list(pattern.get("effect_args") or [])
		if len(effect_args) != len(labels):
			continue
		action_parameters = list(pattern.get("action_parameters") or [])
		action_parameter_types = list(pattern.get("action_parameter_types") or [])
		token_to_type = {
			token: _sanitize_name(str(parameter_type)).lower()
			for token, parameter_type in zip(action_parameters, action_parameter_types)
		}
		used_indices: set[int] = set()
		aligned_labels: list[str] = []
		for effect_index, effect_token in enumerate(effect_args):
			chosen_index: Optional[int] = None
			effect_type = token_to_type.get(effect_token)
			if effect_type is not None:
				matching_indices = [
					index
					for index, task_type in enumerate(task_parameter_types)
					if task_type == effect_type and index not in used_indices
				]
				if len(matching_indices) == 1:
					chosen_index = matching_indices[0]
			if chosen_index is None and effect_index < len(labels) and effect_index not in used_indices:
				chosen_index = effect_index
			if chosen_index is None:
				for fallback_index in range(len(labels)):
					if fallback_index not in used_indices:
						chosen_index = fallback_index
						break
			if chosen_index is None:
				return tuple(labels)
			used_indices.add(chosen_index)
			aligned_labels.append(str(labels[chosen_index]))
		if len(aligned_labels) == len(labels):
			return tuple(aligned_labels)

	return tuple(labels)


def _task_parameter_sequence_for_headline_predicate(
	task_name: str,
	predicate_name: str,
	predicate_args: Sequence[str],
	task_schemas: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> tuple[str, ...]:
	task_schema = task_schemas.get(task_name)
	if task_schema is None:
		return tuple(str(arg) for arg in predicate_args)

	task_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
	task_parameter_types = tuple(
		_sanitize_name(_parameter_type(parameter)).lower()
		for parameter in task_parameters
	)
	if len(task_parameter_types) != len(predicate_args):
		return tuple(str(arg) for arg in predicate_args)

	for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
		predicate_name,
		[],
	):
		effect_args = list(pattern.get("effect_args") or [])
		if len(effect_args) != len(predicate_args):
			continue
		action_parameters = list(pattern.get("action_parameters") or [])
		action_parameter_types = list(pattern.get("action_parameter_types") or [])
		token_to_type = {
			token: _sanitize_name(str(parameter_type)).lower()
			for token, parameter_type in zip(action_parameters, action_parameter_types)
		}
		used_task_indices: set[int] = set()
		aligned_args: list[Optional[str]] = [None] * len(task_parameters)
		for effect_index, effect_token in enumerate(effect_args):
			chosen_task_index: Optional[int] = None
			effect_type = token_to_type.get(effect_token)
			if effect_type is not None:
				matching_indices = [
					index
					for index, task_type in enumerate(task_parameter_types)
					if task_type == effect_type and index not in used_task_indices
				]
				if len(matching_indices) == 1:
					chosen_task_index = matching_indices[0]
			if (
				chosen_task_index is None
				and effect_index < len(task_parameters)
				and effect_index not in used_task_indices
			):
				chosen_task_index = effect_index
			if chosen_task_index is None:
				for fallback_index in range(len(task_parameters)):
					if fallback_index not in used_task_indices:
						chosen_task_index = fallback_index
						break
			if chosen_task_index is None:
				return tuple(str(arg) for arg in predicate_args)
			used_task_indices.add(chosen_task_index)
			aligned_args[chosen_task_index] = str(predicate_args[effect_index])
		if all(arg is not None for arg in aligned_args):
			return tuple(str(arg) for arg in aligned_args)

	return tuple(str(arg) for arg in predicate_args)


def _headline_support_task_invocation_signature(
	task_name: str,
	predicate_name: str,
	predicate_args: Sequence[str],
	task_schemas: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> str:
	aligned_args = _task_parameter_sequence_for_headline_predicate(
		task_name,
		predicate_name,
		predicate_args,
		task_schemas,
		action_analysis,
	)
	return _task_invocation_signature(task_name, aligned_args)


def _query_task_role_stabilization_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	anchors_by_task_name: Dict[str, list[Dict[str, Any]]] = {}
	for anchor in query_task_anchors:
		task_name = str(anchor.get("task_name", "")).strip()
		if not task_name:
			continue
		anchors_by_task_name.setdefault(task_name, []).append(anchor)

	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		task_schema = task_schemas.get(task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if task_schema is None or parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue
		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		if len(task_parameters) < 2:
			continue

		bound_occurrences = [
			tuple(str(arg) for arg in occurrence.get("args", ()))
			for occurrence in anchors_by_task_name.get(task_name, [])
			if len(tuple(occurrence.get("args", ()))) == len(task_parameters)
		]
		if len(bound_occurrences) < 2:
			continue

		role_labels = tuple(f"ARG{index}" for index in range(1, len(task_parameters) + 1))
		headline_mapping = {
			token: role_label
			for token, role_label in zip(task_parameters, role_labels)
		}
		direct_role_requirements: Dict[int, set[str]] = {
			index: set()
			for index in range(1, len(task_parameters))
		}
		for headline_pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		):
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(task_parameters):
				continue
			token_mapping = {
				token: role_label
				for token, role_label in zip(effect_args, role_labels)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				rendered_signature = _render_signature_with_mapping(
					precondition_signature,
					token_mapping,
				)
				parsed_precondition = _parse_literal_signature(rendered_signature)
				if parsed_precondition is None:
					continue
				_, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive or len(precondition_args) != 1:
					continue
				for role_index in range(1, len(task_parameters)):
					if precondition_args[0] == role_labels[role_index]:
						direct_role_requirements[role_index].add(rendered_signature)

		same_arity_candidates = _same_arity_declared_task_candidates(
			domain,
			task_name,
			task_schemas,
		)
		child_description = (
			f"same-arity packaging child "
			f"{_task_invocation_signature(same_arity_candidates[0], role_labels)}"
			if same_arity_candidates
			else "the final producer"
		)

		primary_values = {occurrence[0] for occurrence in bound_occurrences}
		for role_index in range(1, len(task_parameters)):
			role_label = role_labels[role_index]
			role_type = _parameter_type(task_schema.parameters[role_index])
			role_values = {occurrence[role_index] for occurrence in bound_occurrences}
			reused_values = sorted(value for value in role_values if value in primary_values)
			terminal_values = sorted(value for value in role_values if value not in primary_values)
			if not reused_values and not terminal_values:
				continue

			candidate_stabilizers: list[str] = []
			for candidate_task in getattr(domain, "tasks", []):
				candidate_name = str(candidate_task.name)
				if candidate_name == task_name or len(candidate_task.parameters) != 1:
					continue
				if _parameter_type(candidate_task.parameters[0]) != role_type:
					continue
				for candidate_predicate in _candidate_headline_predicates_for_task(
					candidate_name,
					1,
					action_analysis,
				):
					candidate_signature = f"{candidate_predicate}({role_label})"
					if candidate_signature in direct_role_requirements[role_index]:
						continue
					if _constructive_template_summary_for_task(
						candidate_name,
						(role_label,),
						candidate_predicate,
						action_analysis,
					) is None:
						continue
					candidate_stabilizers.append(
						_task_invocation_signature(candidate_name, (role_label,))
					)
					break
			if not candidate_stabilizers:
				continue

			line = (
				f"- {_task_invocation_signature(display_name, role_labels)}: {role_label} acts as a "
				f"non-leading support/base role in the repeated query skeleton. If a unary "
				f"declared task such as {' or '.join(candidate_stabilizers[:2])} can stabilize "
				f"{role_label} beyond the headline producer's direct prerequisites, keep that "
				f"stabilizer as a real sibling step before {child_description}; do not compress it "
				f"away into only direct support for the final producer. The stabilizer must close "
				"its own headline effect internally."
			)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)
	return lines


def _query_task_child_support_requirement_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	required_helper_lookup = {
		(
			str(spec.get("query_task_name", "")).strip(),
			str(spec.get("precondition_signature", "")).strip(),
		): spec
		for spec in _required_helper_specs_for_query_targets(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		)
	}
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		anchor_task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		task_schema = task_schemas.get(anchor_task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue
		task_args = tuple(f"ARG{index}" for index in range(1, len(target_args) + 1))
		task_signature_parameters = tuple(
			_parameter_token(parameter)
			for parameter in getattr(task_schema, "parameters", ()) or ()
		)
		same_arity_candidates = _same_arity_packaging_candidates_for_query_task(
			domain,
			anchor_task_name,
			predicate_name,
			task_signature_parameters or task_args,
			task_schemas,
			action_analysis,
		)
		packaging_caller_shared_requirements: list[str] = []
		if same_arity_candidates:
			primary_candidate = same_arity_candidates[0]
			candidate_parameters = tuple(primary_candidate.get("parameters", ()))
			candidate_mapping = {
				_parameter_token(raw_parameter): task_arg
				for raw_parameter, task_arg in zip(candidate_parameters, task_args)
			}
			for raw_requirement in primary_candidate.get("shared_requirements", ()):
				rendered_requirement = _render_signature_with_mapping(
					str(raw_requirement),
					candidate_mapping,
				)
				if (
					rendered_requirement
					and rendered_requirement not in packaging_caller_shared_requirements
				):
					packaging_caller_shared_requirements.append(rendered_requirement)

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, [])
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue

			headline_mapping = {
				token: f"ARG{index}"
				for index, token in enumerate(effect_args, start=1)
			}
			headline_action_args = _extend_mapping_with_action_parameters(
				headline_mapping,
				headline_pattern.get("action_parameters") or [],
				action_parameter_types=headline_pattern.get("action_parameter_types") or [],
			)
			headline_call = _task_invocation_signature(
				headline_pattern["action_name"],
				headline_action_args,
			)
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				rendered_precondition_signature = _render_signature_with_mapping(
					precondition_signature,
					headline_mapping,
				)
				parsed_precondition = _parse_literal_signature(
					rendered_precondition_signature
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(headline_mapping.values()):
					continue
				headline_parent_support_pairs: list[tuple[str, str]] = []
				headline_parent_context_requirements: list[str] = []
				if (
					same_arity_candidates
					and packaging_caller_shared_requirements
					and rendered_precondition_signature not in packaging_caller_shared_requirements
				):
					continue
				if same_arity_candidates:
					for sibling_requirement in packaging_caller_shared_requirements:
						if sibling_requirement == rendered_precondition_signature:
							continue
						parsed_sibling_requirement = _parse_literal_signature(
							sibling_requirement,
						)
						if parsed_sibling_requirement is None:
							continue
						sibling_predicate, sibling_args, sibling_positive = (
							parsed_sibling_requirement
						)
						if not sibling_positive:
							continue
						if not sibling_args:
							headline_parent_context_requirements.append(
								sibling_requirement,
							)
							continue
						sibling_support_tasks = [
							support_task
							for support_task in _candidate_support_task_names(
								domain,
								sibling_predicate,
								sibling_args,
								action_analysis.get("producer_actions_by_predicate", {}).get(
									sibling_predicate,
									[],
								),
							)
							if support_task != anchor_task_name
							and (
								len(getattr(task_schemas.get(support_task), "parameters", ()))
								== len(sibling_args)
							)
						]
						if sibling_support_tasks:
							headline_parent_support_pairs.append(
								(
									sibling_requirement,
									_task_invocation_signature(
										sibling_support_tasks[0],
										sibling_args,
									),
								),
							)
						else:
							headline_parent_context_requirements.append(
								sibling_requirement,
							)

				shared_requirements = _shared_dynamic_requirements_for_predicate(
					precondition_predicate,
					precondition_args,
					action_analysis,
				)
				if not shared_requirements:
					continue
				if (
					required_helper_lookup.get(
						(anchor_task_name, rendered_precondition_signature),
					)
					is not None
				):
					continue

				candidate_tasks = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						precondition_predicate,
						precondition_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							precondition_predicate,
							[],
						),
					)
					if candidate != anchor_task_name
					and (
						len(getattr(task_schemas.get(candidate), "parameters", ()))
						== len(precondition_args)
					)
				]
				if not candidate_tasks:
					line = (
						f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
						f"before any helper or child call intended to establish "
						f"{precondition_predicate}({', '.join(precondition_args)}), first support "
						f"its shared prerequisites {', '.join(shared_requirements)} via parent "
						"context or earlier parent subtasks."
					)
					if line not in seen:
						seen.add(line)
						lines.append(line)
					producer_modes = _render_producer_mode_options_for_predicate(
						precondition_predicate,
						precondition_args,
						action_analysis,
						limit=2,
					)
					if producer_modes:
						strict_compact_producer_line = (
							f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
							f"in any compact support_before/producer/followup branch where "
							f"{precondition_predicate}({', '.join(precondition_args)}) is not already "
							f"in branch precondition/context, the producer slot itself must establish it: "
							f"use {' or '.join(mode_call for mode_call, _ in producer_modes)} there. "
							f"Do not put a later child that still requires "
							f"{precondition_predicate}({', '.join(precondition_args)}) into producer."
						)
						if strict_compact_producer_line not in seen:
							seen.add(strict_compact_producer_line)
							lines.append(strict_compact_producer_line)
					for shared_requirement in shared_requirements:
						parsed_requirement = _parse_literal_signature(shared_requirement)
						if parsed_requirement is None:
							continue
						requirement_predicate, requirement_args, requirement_positive = (
							parsed_requirement
						)
						if not requirement_positive:
							continue
						requirement_support_tasks = [
							support_task
							for support_task in _candidate_support_task_names(
								domain,
								requirement_predicate,
								requirement_args,
								action_analysis.get("producer_actions_by_predicate", {}).get(
									requirement_predicate,
									[],
								),
							)
							if len(getattr(task_schemas.get(support_task), "parameters", ()))
							== len(requirement_args)
						]
						if not requirement_support_tasks:
							continue
						preferred_support_task = _task_invocation_signature(
							requirement_support_tasks[0],
							requirement_args,
						)
						if producer_modes:
							mode_text = " or ".join(
								mode_call
								for mode_call, _ in producer_modes
							)
							split_line = (
								f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
								f"if {shared_requirement} may already hold or require support before a "
								f"{precondition_predicate}({', '.join(precondition_args)}) producer, "
								f"use separate constructive siblings: one may require {shared_requirement} "
								f"at entry, another should use {preferred_support_task} before {mode_text}. "
								"Do not collapse into one precondition-only branch."
							)
						else:
							split_line = (
								f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
								f"if {shared_requirement} may already hold or require support "
								f"before establishing {precondition_predicate}({', '.join(precondition_args)}), "
								f"use separate constructive siblings: one may require {shared_requirement} "
								f"at entry, another should use {preferred_support_task} before the consuming step. "
								"Do not collapse into one precondition-only branch."
							)
						if split_line not in seen:
							seen.add(split_line)
							lines.append(split_line)
						if producer_modes:
							unsupported_case_line = (
								f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
								f"if {shared_requirement} is false at entry, at least one constructive sibling must use "
								f"{preferred_support_task} before {mode_text}; do not make every constructive sibling "
								f"require {shared_requirement} at entry."
							)
						else:
							unsupported_case_line = (
								f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
								f"if {shared_requirement} is false at entry, at least one constructive sibling must use "
								f"{preferred_support_task} before the consuming step; do not make every constructive sibling "
								f"require {shared_requirement} at entry."
							)
						if unsupported_case_line not in seen:
							seen.add(unsupported_case_line)
							lines.append(unsupported_case_line)
						for mode_call, mode_requirements in producer_modes:
							if shared_requirement not in mode_requirements:
								continue
							mode_call_name, _, mode_call_args_text = str(mode_call).partition("(")
							mode_call_args = tuple(
								arg.strip()
								for arg in mode_call_args_text.rstrip(")").split(",")
								if arg.strip()
							) if mode_call_name else ()
							mode_has_extra_role = any(
								arg not in {f"ARG{index}" for index in range(1, len(target_args) + 1)}
								for arg in mode_call_args
							)
							remaining_mode_needs = [
								requirement
								for requirement in mode_requirements
								if requirement != shared_requirement
							]
							if not remaining_mode_needs:
								continue
							remaining_needs_line = (
								f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
								f"if a constructive sibling adds {preferred_support_task} before {mode_call}, "
								f"keep that mode's other unmet needs {', '.join(remaining_mode_needs)} in the "
								f"same sibling precondition/context unless earlier subtasks in that sibling "
								"already support them."
							)
							if remaining_needs_line not in seen:
								seen.add(remaining_needs_line)
								lines.append(remaining_needs_line)
							if mode_has_extra_role:
								conflicting_parent_support_pairs = [
									(requirement, support_call)
									for requirement, support_call in headline_parent_support_pairs
									if any(
										_requirements_are_mutually_destructive(
											requirement,
											remaining_need,
											action_analysis,
										)
										for remaining_need in remaining_mode_needs
									)
								]
								supportable_parent_support_calls = [
									support_call
									for requirement, support_call in headline_parent_support_pairs
									if (requirement, support_call) not in conflicting_parent_support_pairs
								]
							else:
								conflicting_parent_support_pairs = []
								supportable_parent_support_calls = [
									support_call
									for _, support_call in headline_parent_support_pairs
								]
							branch_precondition_requirements = list(
								dict.fromkeys(
									[
										*remaining_mode_needs,
										*headline_parent_context_requirements,
										*[
											requirement
											for requirement, _ in conflicting_parent_support_pairs
										],
									],
								),
							)
							branch_support_before_calls = list(
								dict.fromkeys(
									[
										preferred_support_task,
										*supportable_parent_support_calls,
									],
								),
							)
							for support_call in supportable_parent_support_calls:
								if support_call not in branch_support_before_calls:
									branch_support_before_calls.append(support_call)
							support_then_produce_line = (
								f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
								f"valid support-then-produce sibling for the false-{shared_requirement} "
								f"case with {mode_call}: precondition/context {', '.join(branch_precondition_requirements)}; "
								f"support_before {'; '.join(branch_support_before_calls)}; producer {mode_call}"
							)
							if same_arity_candidates:
								packaging_call = _task_invocation_signature(
									str(same_arity_candidates[0].get("candidate", "")).strip(),
									task_args,
								)
								support_then_produce_line += (
									f"; followup {packaging_call}."
								)
							else:
								support_then_produce_line += (
									f"; followup {headline_call}."
								)
							if support_then_produce_line not in seen:
								seen.add(support_then_produce_line)
								lines.append(support_then_produce_line)
							no_duplicate_supported_need_line = (
								f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
								f"if {preferred_support_task} before {mode_call} handles false-{shared_requirement}, "
								f"do not also keep {shared_requirement} in that sibling precondition/context; split siblings."
							)
							if no_duplicate_supported_need_line not in seen:
								seen.add(no_duplicate_supported_need_line)
								lines.append(no_duplicate_supported_need_line)
							binding_needs = [
								requirement
								for requirement in remaining_mode_needs
								if _signature_mentions_aux_role(requirement)
							]
							if binding_needs:
								binding_line = (
									f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
									f"if a constructive sibling adds {preferred_support_task} before {mode_call}, "
									f"keep binding literal(s) {', '.join(binding_needs)} in that sibling "
									"precondition/context before the first use of the AUX role."
								)
								if binding_line not in seen:
									seen.add(binding_line)
									lines.append(binding_line)
					continue

				for candidate_task in candidate_tasks[:2]:
					candidate_schema = task_schemas.get(candidate_task)
					if candidate_schema is None:
						continue
					candidate_args = tuple(
						_parameter_token(parameter)
						for parameter in candidate_schema.parameters
					)
					candidate_mapping = {
						parameter: argument
						for parameter, argument in zip(
							candidate_args,
							precondition_args,
						)
					}
					precise_shared_requirements = _support_task_precise_shared_requirements(
						domain,
						candidate_task,
						precondition_predicate,
						candidate_args,
						task_schemas,
						action_analysis,
					)
					rendered_shared_requirements = tuple(
						dict.fromkeys(
							_render_signature_with_mapping(
								requirement,
								candidate_mapping,
							)
							for requirement in precise_shared_requirements
							if _render_signature_with_mapping(
								requirement,
								candidate_mapping,
							)
						)
					) or shared_requirements
					candidate_support_call = _task_invocation_signature(
						candidate_task,
						precondition_args,
					)
					line = (
						f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
						f"if you use {candidate_support_call} "
						f"to support {precondition_predicate}({', '.join(precondition_args)}), "
						f"first support its shared prerequisites {', '.join(rendered_shared_requirements)} "
						"via parent context or earlier parent subtasks."
					)
					if line in seen:
						continue
					seen.add(line)
					lines.append(line)
					conflicting_parent_support_pairs = [
						(requirement, support_call)
						for requirement, support_call in headline_parent_support_pairs
						if any(
							_requirements_are_mutually_destructive(
								requirement,
								shared_requirement,
								action_analysis,
							)
							for shared_requirement in rendered_shared_requirements
						)
					]
					supportable_parent_support_pairs = [
						(requirement, support_call)
						for requirement, support_call in headline_parent_support_pairs
						if (requirement, support_call) not in conflicting_parent_support_pairs
						and support_call != candidate_support_call
					]
					sibling_support_envelopes: list[str] = []
					support_before_calls = [candidate_support_call]
					for sibling_requirement, sibling_support_call in supportable_parent_support_pairs:
						if sibling_support_call not in support_before_calls:
							support_before_calls.append(sibling_support_call)
						parsed_sibling_requirement = _parse_literal_signature(
							sibling_requirement,
						)
						if parsed_sibling_requirement is None:
							continue
						sibling_predicate, sibling_args, sibling_positive = parsed_sibling_requirement
						if not sibling_positive:
							continue
						sibling_task_candidates = [
							support_task
							for support_task in _candidate_support_task_names(
								domain,
								sibling_predicate,
								sibling_args,
								action_analysis.get("producer_actions_by_predicate", {}).get(
									sibling_predicate,
									[],
								),
							)
							if len(getattr(task_schemas.get(support_task), "parameters", ()))
							== len(sibling_args)
						]
						if not sibling_task_candidates:
							continue
						sibling_task = sibling_task_candidates[0]
						sibling_schema = task_schemas.get(sibling_task)
						if sibling_schema is None:
							continue
						sibling_parameters = tuple(
							_parameter_token(parameter)
							for parameter in sibling_schema.parameters
						)
						sibling_mapping = {
							parameter: argument
							for parameter, argument in zip(
								sibling_parameters,
								sibling_args,
							)
						}
						sibling_precise_shared_requirements = _support_task_precise_shared_requirements(
							domain,
							sibling_task,
							sibling_predicate,
							sibling_parameters,
							task_schemas,
							action_analysis,
						)
						rendered_sibling_shared_requirements = tuple(
							dict.fromkeys(
								_render_signature_with_mapping(
									requirement,
									sibling_mapping,
								)
								for requirement in sibling_precise_shared_requirements
								if _render_signature_with_mapping(
									requirement,
									sibling_mapping,
								)
							)
						) or _shared_dynamic_requirements_for_predicate(
							sibling_predicate,
							sibling_args,
							action_analysis,
						)
						for rendered_requirement in rendered_sibling_shared_requirements:
							if rendered_requirement not in sibling_support_envelopes:
								sibling_support_envelopes.append(rendered_requirement)
					branch_precondition_requirements = list(
						dict.fromkeys(
							[
								*rendered_shared_requirements,
								*sibling_support_envelopes,
								*headline_parent_context_requirements,
								*[
									requirement
									for requirement, _ in conflicting_parent_support_pairs
								],
							]
						),
					)
					if branch_precondition_requirements:
						branch_line = (
							f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
							f"valid support-then-produce sibling for the false-"
							f"{rendered_precondition_signature} case with {candidate_support_call}: "
							f"precondition/context {', '.join(branch_precondition_requirements)}; "
							f"support_before {'; '.join(support_before_calls)}; "
							f"producer {headline_call}."
						)
						if branch_line not in seen:
							seen.add(branch_line)
							lines.append(branch_line)
						branch_ast_line = (
							f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
							"AST slot shape for that support-then-produce sibling: "
							"{"
							f'"precondition":[{", ".join(json.dumps(item) for item in branch_precondition_requirements)}],'
							f'"support_before":[{", ".join(json.dumps(item) for item in support_before_calls)}],'
							f'"producer":{json.dumps(headline_call)}'
							"}."
						)
						if branch_ast_line not in seen:
							seen.add(branch_ast_line)
							lines.append(branch_ast_line)
						no_duplicate_line = (
							f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
							f"if support_before {candidate_support_call} handles false-"
							f"{rendered_precondition_signature} before {headline_call}, do not also keep "
							f"{rendered_precondition_signature} in that same sibling precondition/context; "
							"split already-supported and support-then-produce siblings."
						)
						if no_duplicate_line not in seen:
							seen.add(no_duplicate_line)
							lines.append(no_duplicate_line)
					for sibling_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
						rendered_sibling_signature = _render_signature_with_mapping(
							sibling_signature,
							headline_mapping,
						)
						if rendered_sibling_signature == rendered_precondition_signature:
							continue
						parsed_sibling = _parse_literal_signature(rendered_sibling_signature)
						if parsed_sibling is None or not parsed_sibling[2]:
							continue
						conflicting_shared_requirements = [
							requirement
							for requirement in rendered_shared_requirements
							if _requirements_are_mutually_destructive(
								requirement,
								rendered_sibling_signature,
								action_analysis,
							)
						]
						if not conflicting_shared_requirements:
							continue
						conflict_line = (
							f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
							f"if {candidate_support_call} still needs "
							f"{', '.join(conflicting_shared_requirements)} but the later child or final producer "
							f"expects caller-shared {rendered_sibling_signature}, support "
							f"{precondition_predicate}({', '.join(precondition_args)}) before "
							f"{rendered_sibling_signature} becomes required at child entry. Do not move "
							f"{candidate_support_call} into a later child "
							f"after {rendered_sibling_signature} is already required."
						)
						if conflict_line in seen:
							continue
						seen.add(conflict_line)
						lines.append(conflict_line)
	return lines


def _query_task_role_stabilizer_support_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	anchors_by_task_name: Dict[str, list[Dict[str, Any]]] = {}
	for anchor in query_task_anchors:
		task_name = str(anchor.get("task_name", "")).strip()
		if not task_name:
			continue
		anchors_by_task_name.setdefault(task_name, []).append(anchor)

	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		task_schema = task_schemas.get(task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if task_schema is None or parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue

		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		if len(task_parameters) < 2:
			continue
		bound_occurrences = [
			tuple(str(arg) for arg in occurrence.get("args", ()))
			for occurrence in anchors_by_task_name.get(task_name, [])
			if len(tuple(occurrence.get("args", ()))) == len(task_parameters)
		]
		if len(bound_occurrences) < 2:
			continue

		role_labels = tuple(f"ARG{index}" for index in range(1, len(task_parameters) + 1))
		direct_role_requirements: Dict[int, set[str]] = {
			index: set()
			for index in range(1, len(task_parameters))
		}
		for headline_pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		):
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(task_parameters):
				continue
			token_mapping = {
				token: role_label
				for token, role_label in zip(effect_args, role_labels)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				rendered_signature = _render_signature_with_mapping(
					precondition_signature,
					token_mapping,
				)
				parsed_precondition = _parse_literal_signature(rendered_signature)
				if parsed_precondition is None:
					continue
				_, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive or len(precondition_args) != 1:
					continue
				for role_index in range(1, len(task_parameters)):
					if precondition_args[0] == role_labels[role_index]:
						direct_role_requirements[role_index].add(rendered_signature)

		primary_values = {occurrence[0] for occurrence in bound_occurrences}
		for role_index in range(1, len(task_parameters)):
			role_label = role_labels[role_index]
			role_type = _parameter_type(task_schema.parameters[role_index])
			role_values = {
				occurrence[role_index]
				for occurrence in bound_occurrences
				if len(occurrence) == len(task_parameters)
			}
			reused_values = {value for value in role_values if value in primary_values}
			terminal_values = {value for value in role_values if value not in primary_values}
			if not reused_values and not terminal_values:
				continue

			for candidate_task in getattr(domain, "tasks", []):
				candidate_name = str(candidate_task.name)
				if candidate_name == task_name or len(candidate_task.parameters) != 1:
					continue
				if _parameter_type(candidate_task.parameters[0]) != role_type:
					continue
				for candidate_predicate in _candidate_headline_predicates_for_task(
					candidate_name,
					1,
					action_analysis,
				):
					candidate_signature = f"{candidate_predicate}({role_label})"
					if candidate_signature in direct_role_requirements[role_index]:
						continue
					template_summary = _constructive_template_summary_for_task(
						candidate_name,
						(role_label,),
						candidate_predicate,
						action_analysis,
					)
					if template_summary is None:
						continue
					shared_requirements = _shared_dynamic_requirements_for_predicate(
						candidate_predicate,
						(role_label,),
						action_analysis,
					)
					support_option_fragments: list[str] = []
					self_requiring_modes: list[str] = []
					compatible_modes: list[str] = []
					compatible_mode_specs: list[tuple[str, list[str]]] = []
					compatible_mode_binding_hints: list[str] = []
					compatible_mode_antidetour_hints: list[str] = []
					if shared_requirements:
						for shared_requirement in shared_requirements:
							parsed_requirement = _parse_literal_signature(shared_requirement)
							if parsed_requirement is None:
								continue
							req_predicate, req_args, req_positive = parsed_requirement
							if not req_positive:
								continue
							rendered_options: list[str] = []
							for support_pattern in action_analysis.get(
								"producer_patterns_by_predicate",
								{},
							).get(req_predicate, [])[:3]:
								support_effect_args = list(support_pattern.get("effect_args") or [])
								if len(support_effect_args) != len(req_args):
									continue
								support_mapping = {
									token: arg
									for token, arg in zip(support_effect_args, req_args)
								}
								rendered_action_args = _extend_mapping_with_action_parameters(
									support_mapping,
									support_pattern.get("action_parameters") or [],
									action_parameter_types=support_pattern.get("action_parameter_types")
									or [],
								)
								rendered_requirements = [
									_render_signature_with_mapping(signature, support_mapping)
									for signature in (
										support_pattern.get("dynamic_precondition_signatures") or []
									)
								]
								suffix = (
									f" [needs {', '.join(rendered_requirements)}]"
									if rendered_requirements
									else ""
								)
								rendered_options.append(
									f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}{suffix}"
								)
								action_parameter_types = list(
									support_pattern.get("action_parameter_types") or []
								)
								headline_signature = f"{candidate_predicate}({role_label})"
								if headline_signature in rendered_requirements:
									self_requiring_modes.append(
										_task_invocation_signature(
											support_pattern["action_name"],
											rendered_action_args,
										)
									)
								else:
									mode_call = _task_invocation_signature(
										support_pattern["action_name"],
										rendered_action_args,
									)
									compatible_modes.append(mode_call)
									extra_roles = [
										arg
										for arg in rendered_action_args
										if arg not in req_args
									]
									mode_selector_requirements = [
										requirement
										for requirement in rendered_requirements
										if requirement != headline_signature
									]
									compatible_mode_specs.append(
										(mode_call, mode_selector_requirements),
									)
									if extra_roles:
										binding_hint = (
											f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}: "
											f"declare extra roles {', '.join(extra_roles)} in method.parameters"
										)
										if mode_selector_requirements:
											binding_hint += (
												f" and keep {'; '.join(mode_selector_requirements)} "
												"in method.context before that step"
											)
										compatible_mode_binding_hints.append(binding_hint)
										for extra_role in extra_roles:
											generic_antidetour_hint = (
												f"before "
												f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}, "
												f"do not insert unary support on extra role {extra_role} "
												f"unless its headline literal is one of that mode's real "
												f"dynamic prerequisites {'; '.join(rendered_requirements)}"
											)
											if generic_antidetour_hint not in compatible_mode_antidetour_hints:
												compatible_mode_antidetour_hints.append(
													generic_antidetour_hint
												)
											extra_role_type = None
											for parameter_arg, parameter_type in zip(
												rendered_action_args,
												action_parameter_types,
											):
												if parameter_arg == extra_role:
													extra_role_type = parameter_type
													break
											unrelated_support_tasks: list[str] = []
											for extra_role_task in getattr(domain, "tasks", []):
												if len(extra_role_task.parameters) != 1:
													continue
												extra_role_task_name = str(extra_role_task.name)
												if extra_role_task_name == candidate_name:
													continue
												if (
													extra_role_type is not None
													and _parameter_type(extra_role_task.parameters[0])
													!= extra_role_type
												):
													continue
												for extra_role_predicate in _candidate_headline_predicates_for_task(
													extra_role_task_name,
													1,
													action_analysis,
												):
													extra_role_signature = (
														f"{extra_role_predicate}({extra_role})"
													)
													if extra_role_signature in rendered_requirements:
														continue
													unrelated_support_tasks.append(
														_task_invocation_signature(
															extra_role_task_name,
															(extra_role,),
														)
													)
													break
											if not unrelated_support_tasks:
												unrelated_support_tasks = [
													_task_invocation_signature(
														str(extra_role_task.name),
														(extra_role,),
													)
													for extra_role_task in getattr(domain, "tasks", [])
													if (
														len(extra_role_task.parameters) == 1
														and str(extra_role_task.name) != candidate_name
													)
												]
											if unrelated_support_tasks:
												specific_antidetour_hint = (
													f"before "
													f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}, "
													f"do not insert {' or '.join(unrelated_support_tasks[:2])} "
													f"unless its headline literal is one of that mode's real "
													f"dynamic prerequisites {'; '.join(rendered_requirements)}"
												)
												if (
													specific_antidetour_hint
													not in compatible_mode_antidetour_hints
												):
													compatible_mode_antidetour_hints.append(
														specific_antidetour_hint
													)
							if rendered_options:
								support_option_fragments.append(
									f"{shared_requirement} via {'; '.join(rendered_options)}"
								)
					line_prefix = (
						f"- {_task_invocation_signature(candidate_name, (role_label,))}: "
					)
					role_stabilizer_lines = [
						line_prefix
						+ (
							f"if used as a role stabilizer, its constructive branch must internally "
							f"close {candidate_predicate}({role_label}); constructive template "
							f"{template_summary}."
						),
					]
					downstream_reusable_conditions = sorted(
						requirement
						for requirement in direct_role_requirements[role_index]
						if requirement != candidate_signature
					)
					if downstream_reusable_conditions:
						joined_conditions = "; ".join(downstream_reusable_conditions)
						role_stabilizer_lines.append(
							line_prefix
							+ (
								f"for already-stable reuse of {joined_conditions} on {role_label}, "
								f"include a stable/noop sibling with {joined_conditions} in branch "
								f"context instead of requiring {candidate_predicate}({role_label})."
							)
						)
					if support_option_fragments:
						internal_support_line = (
							line_prefix
							+ f"support remaining internal prerequisites via {'; '.join(support_option_fragments)}."
						)
						if self_requiring_modes:
							internal_support_line += (
								f" For false-{candidate_predicate}({role_label}), do not keep only "
								f"self-requiring modes {'; '.join(self_requiring_modes)}"
							)
							if compatible_modes:
								internal_support_line += (
									f"; prefer compatible modes {'; '.join(compatible_modes)}"
								)
							internal_support_line += "."
						role_stabilizer_lines.append(internal_support_line)
					if self_requiring_modes:
						if not support_option_fragments:
							false_case_line = (
								line_prefix
								+ (
									f"for false-{candidate_predicate}({role_label}), do not keep only "
									f"self-requiring modes {'; '.join(self_requiring_modes)}"
								)
							)
							if compatible_modes:
								false_case_line += (
									f"; prefer compatible modes {'; '.join(compatible_modes)}"
								)
							false_case_line += "."
							role_stabilizer_lines.append(false_case_line)
					if compatible_mode_binding_hints:
						role_stabilizer_lines.append(
							line_prefix
							+ (
								f"if you use a compatible extra-role mode, "
								f"{'; '.join(compatible_mode_binding_hints)}."
							)
						)
					for mode_call, mode_selector_requirements in compatible_mode_specs:
						if not mode_selector_requirements:
							continue
						role_stabilizer_lines.append(
							line_prefix
							+ (
								f"valid compatible slot sibling for false-{candidate_predicate}({role_label}) "
								f"with {mode_call}: precondition/context {'; '.join(mode_selector_requirements)}; "
								f"producer {mode_call}; followup {template_summary}."
							)
						)
					if compatible_mode_antidetour_hints:
						role_stabilizer_lines.append(
							line_prefix
							+ f"also, {'; '.join(compatible_mode_antidetour_hints)}."
						)
					if shared_requirements:
						role_stabilizer_lines.append(
							line_prefix
							+ "parent tasks should not provide those internal stabilizer prerequisites."
						)
					for line in role_stabilizer_lines:
						if line in seen:
							continue
						seen.add(line)
						lines.append(line)
					break
	return lines


def _query_task_support_obligation_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		force_internal_contract = bool(anchor.get("force_internal_contract"))
		task = task_schemas.get(task_name)
		if task is None:
			continue

		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue

		task_parameters = tuple(_parameter_token(parameter) for parameter in task.parameters)
		same_arity_candidates = _same_arity_packaging_candidates_for_query_task(
			domain,
			task_name,
			predicate_name,
			task_parameters,
			task_schemas,
			action_analysis,
		)
		packaging_shared_requirements = {
				str(requirement).strip()
				for candidate in same_arity_candidates
				for requirement in _same_arity_caller_shared_requirements(
					domain,
					predicate_name,
					task_parameters,
					action_analysis,
					candidate,
				)
				if str(requirement).strip()
			}
		if packaging_shared_requirements and not force_internal_contract:
			continue
		patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, [])
		for pattern in patterns:
			effect_args = list(pattern.get("effect_args") or [])
			if len(effect_args) != len(task_parameters):
				continue

			token_mapping = {
				token: task_parameter
				for token, task_parameter in zip(effect_args, task_parameters)
			}
			rendered_action_args = _extend_mapping_with_action_parameters(
				token_mapping,
				pattern.get("action_parameters") or [],
				action_parameter_types=pattern.get("action_parameter_types") or [],
			)

			rendered_preconditions = []
			for signature in pattern.get("dynamic_precondition_signatures") or []:
				rendered_signature = _render_signature_with_mapping(signature, token_mapping)
				if (
					packaging_shared_requirements
					and rendered_signature not in packaging_shared_requirements
				):
					continue
				parsed_signature = _parse_literal_signature(rendered_signature)
				if parsed_signature is None:
					continue
				_, rendered_args, rendered_positive = parsed_signature
				if not rendered_positive:
					continue
				if not set(rendered_args) & set(task_parameters):
					continue
				rendered_preconditions.append(rendered_signature)

			if not rendered_preconditions:
				continue

			line = (
				f"- {_task_invocation_signature(display_name, task_parameters)}: before "
				f"{_task_invocation_signature(pattern['action_name'], rendered_action_args)}, "
				f"support {', '.join(rendered_preconditions)} in earlier subtasks or context."
			)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)
	return lines


def _query_priority_obligation_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	anchors_by_task_name: Dict[str, list[Dict[str, Any]]] = {}
	for anchor in query_task_anchors:
		task_name = str(anchor.get("task_name", "")).strip()
		if task_name:
			anchors_by_task_name.setdefault(task_name, []).append(anchor)

	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		task_schema = task_schemas.get(task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if task_schema is None or parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue
		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		if len(task_parameters) < 2:
			continue
		bound_occurrences = [
			tuple(str(arg) for arg in occurrence.get("args", ()))
			for occurrence in anchors_by_task_name.get(task_name, [])
			if len(tuple(occurrence.get("args", ()))) == len(task_parameters)
		]
		if len(bound_occurrences) < 2:
			continue

		primary_values = {occurrence[0] for occurrence in bound_occurrences}
		same_arity_candidates = _same_arity_declared_task_candidates(
			domain,
			task_name,
			task_schemas,
		)
		if not same_arity_candidates:
			continue
		packaging_child = _task_invocation_signature(same_arity_candidates[0], task_parameters)

		direct_role_requirements: Dict[int, list[str]] = {
			index: []
			for index in range(1, len(task_parameters))
		}
		for headline_pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		):
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(task_parameters):
				continue
			token_mapping = {
				token: role_label
				for token, role_label in zip(effect_args, task_parameters)
			}
			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				rendered_signature = _render_signature_with_mapping(
					precondition_signature,
					token_mapping,
				)
				parsed_precondition = _parse_literal_signature(rendered_signature)
				if parsed_precondition is None:
					continue
				_, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive or len(precondition_args) != 1:
					continue
				for role_index in range(1, len(task_parameters)):
					if precondition_args[0] != task_parameters[role_index]:
						continue
					if rendered_signature in direct_role_requirements[role_index]:
						continue
					direct_role_requirements[role_index].append(rendered_signature)

		for role_index in range(1, len(task_parameters)):
			role_parameter = task_parameters[role_index]
			role_type = _parameter_type(task_schema.parameters[role_index])
			role_values = {
				occurrence[role_index]
				for occurrence in bound_occurrences
				if len(occurrence) == len(task_parameters)
			}
			if not role_values:
				continue
			if not any(value in primary_values for value in role_values) and not any(
				value not in primary_values for value in role_values
			):
				continue

			for candidate_task in getattr(domain, "tasks", []):
				candidate_name = str(candidate_task.name)
				if candidate_name == task_name or len(candidate_task.parameters) != 1:
					continue
				if _parameter_type(candidate_task.parameters[0]) != role_type:
					continue
				for candidate_predicate in _candidate_headline_predicates_for_task(
					candidate_name,
					1,
					action_analysis,
				):
					candidate_signature = f"{candidate_predicate}({role_parameter})"
					role_stable_requirements = direct_role_requirements.get(role_index, [])
					if candidate_signature not in role_stable_requirements:
						continue
					template_summary = _constructive_template_summary_for_task(
						candidate_name,
						(role_parameter,),
						candidate_predicate,
						action_analysis,
					)
					if template_summary is None:
						continue
					stabilizer_call = _task_invocation_signature(candidate_name, (role_parameter,))
					line = (
						f"- {_task_invocation_signature(display_name, task_parameters)}: high-priority "
						f"query skeleton is support {role_parameter}, then {stabilizer_call}, then "
						f"{packaging_child}. Do not omit {stabilizer_call}."
					)
					if line not in seen:
						seen.add(line)
						lines.append(line)
					if role_stable_requirements:
						stable_line = (
							f"- {stabilizer_call}: if {role_parameter} already satisfies downstream "
							f"support-role conditions {', '.join(role_stable_requirements)}, prefer "
							"a stable/noop sibling instead of a destructive constructive branch, "
							f"even when {candidate_predicate}({role_parameter}) is still false."
						)
						if stable_line not in seen:
							seen.add(stable_line)
							lines.append(stable_line)

					shared_requirements = _shared_dynamic_requirements_for_predicate(
						candidate_predicate,
						(role_parameter,),
						action_analysis,
					)
					if not shared_requirements:
						break
					for shared_requirement in shared_requirements:
						parsed_requirement = _parse_literal_signature(shared_requirement)
						if parsed_requirement is None:
							continue
						req_predicate, req_args, req_positive = parsed_requirement
						if not req_positive:
							continue
						self_requiring_modes: list[str] = []
						compatible_modes: list[str] = []
						for support_pattern in action_analysis.get(
							"producer_patterns_by_predicate",
							{},
						).get(req_predicate, [])[:3]:
							support_effect_args = list(support_pattern.get("effect_args") or [])
							if len(support_effect_args) != len(req_args):
								continue
							support_mapping = {
								token: arg
								for token, arg in zip(support_effect_args, req_args)
							}
							rendered_action_args = _extend_mapping_with_action_parameters(
								support_mapping,
								support_pattern.get("action_parameters") or [],
								action_parameter_types=support_pattern.get("action_parameter_types")
								or [],
							)
							rendered_requirements = {
								_render_signature_with_mapping(signature, support_mapping)
								for signature in (
									support_pattern.get("dynamic_precondition_signatures") or []
								)
							}
							rendered_call = _task_invocation_signature(
								support_pattern["action_name"],
								rendered_action_args,
							)
							headline_signature = f"{candidate_predicate}({role_parameter})"
							if headline_signature in rendered_requirements:
								self_requiring_modes.append(rendered_call)
							else:
								compatible_modes.append(rendered_call)
						if self_requiring_modes:
							mode_line = (
								f"- {stabilizer_call}: for the false-{candidate_predicate}({role_parameter}) "
								f"constructive case, do not use self-requiring modes "
								f"{'; '.join(self_requiring_modes)}"
							)
							if compatible_modes:
								mode_line += f"; prefer {'; '.join(compatible_modes)}"
							mode_line += "."
							if mode_line not in seen:
								seen.add(mode_line)
								lines.append(mode_line)
					break
	return lines


def _query_task_support_producer_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	lines: list[str] = []
	seen: set[str] = set()
	task_schemas = _declared_task_schema_map(domain)
	required_helper_lookup = {
		(
			str(spec.get("query_task_name", "")).strip(),
			str(spec.get("precondition_signature", "")).strip(),
		): spec
		for spec in _required_helper_specs_for_query_targets(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		)
	}
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		force_internal_contract = bool(anchor.get("force_internal_contract"))
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue
		task_signature_parameters = tuple(
			_parameter_token(parameter)
			for parameter in getattr(task_schemas.get(task_name), "parameters", ()) or ()
		)
		task_args = tuple(f"ARG{index}" for index in range(1, len(target_args) + 1))
		if anchor.get("args") and len(anchor.get("args", [])) == len(target_args):
			task_args = tuple(
				f"ARG{index}" for index, _ in enumerate(target_args, start=1)
			)
		same_arity_candidates = (
			[]
			if force_internal_contract
			else _same_arity_packaging_candidates_for_query_task(
				domain,
				task_name,
				predicate_name,
				task_signature_parameters,
				task_schemas,
				action_analysis,
			)
		)
		if force_internal_contract and anchor.get("caller_shared_requirements"):
			packaging_shared_requirements = {
				str(requirement).strip()
				for requirement in anchor.get("caller_shared_requirements", ())
				if str(requirement).strip()
			}
		else:
			packaging_shared_requirements = {
				str(requirement).strip()
				for candidate in same_arity_candidates
				for requirement in _same_arity_caller_shared_requirements(
					domain,
					predicate_name,
					task_args,
					action_analysis,
					candidate,
				)
				if str(requirement).strip()
			}
		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, [])
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue

			headline_mapping = {
				token: task_arg
				for token, task_arg in zip(effect_args, task_args)
			}
			rendered_headline_action_args = _extend_mapping_with_action_parameters(
				headline_mapping,
				headline_pattern.get("action_parameters") or [],
				action_parameter_types=headline_pattern.get("action_parameter_types") or [],
			)

			rendered_headline_call = _task_invocation_signature(
				headline_pattern["action_name"],
				rendered_headline_action_args,
			)
			headline_support_calls: list[str] = []
			headline_support_context_requirements: list[str] = []

			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				rendered_precondition_signature = _render_signature_with_mapping(
					precondition_signature,
					headline_mapping,
				)
				if (
					force_internal_contract
					and packaging_shared_requirements
					and rendered_precondition_signature in packaging_shared_requirements
				):
					continue
				if (
					not force_internal_contract
					and packaging_shared_requirements
					and rendered_precondition_signature not in packaging_shared_requirements
				):
					continue
				parsed_precondition = _parse_literal_signature(
					rendered_precondition_signature
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				candidate_declared_tasks = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						precondition_predicate,
						precondition_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							precondition_predicate,
							[],
						),
					)
					if len(getattr(task_schemas.get(candidate), "parameters", ())) == len(precondition_args)
				]
				if not set(precondition_args) & set(task_args) and not candidate_declared_tasks:
					if rendered_precondition_signature not in headline_support_context_requirements:
						headline_support_context_requirements.append(
							rendered_precondition_signature,
						)
					continue
				required_helper_spec = required_helper_lookup.get(
					(task_name, rendered_precondition_signature),
				)
				if required_helper_spec is not None:
					mode_calls = [
						str(pattern_row.get("mode_call", "")).strip()
						for pattern_row in required_helper_spec.get("pattern_rows") or []
						if str(pattern_row.get("mode_call", "")).strip()
					]
					required_helper_call = _task_invocation_signature(
						str(required_helper_spec.get("helper_task_name", "")).strip(),
						tuple(required_helper_spec.get("helper_target_args") or ()),
					)
					packaging_call = str(
						required_helper_spec.get("packaging_call", ""),
					).strip()
					parent_support_calls = list(
						required_helper_spec.get("helper_support_calls") or [],
					)
					helper_entry_requirements = list(
						required_helper_spec.get("common_context_requirements") or [],
					)
					mandatory_helper_line = (
						f"- {_task_invocation_signature(display_name, task_args)}: because no declared task "
						f"headlines {precondition_predicate}({', '.join(precondition_args)}) and "
						"earlier caller-shared support can change which producer mode remains valid, "
						f"use required helper task {required_helper_call} before {packaging_call}. "
					)
					if helper_entry_requirements:
						mandatory_helper_line += (
							f" That helper still needs {', '.join(helper_entry_requirements)} at helper entry, "
							"so keep those in parent precondition/context or establish them before the helper. "
						)
					mandatory_helper_line += "Parent constructive branches that call "
					mandatory_helper_line += f"{packaging_call} must follow "
					if helper_entry_requirements:
						mandatory_helper_line += (
							f"precondition/context {', '.join(helper_entry_requirements)}; "
						)
					mandatory_helper_line += (
						f"support_before {'; '.join(parent_support_calls)}; "
						f"producer {required_helper_call}; followup {packaging_call}."
					)
					if mode_calls:
						mandatory_helper_line += (
							f" Do not place {' or '.join(mode_calls)} directly in the parent branch."
						)
					if mandatory_helper_line not in seen:
						seen.add(mandatory_helper_line)
						lines.append(mandatory_helper_line)
					continue

				if candidate_declared_tasks:
					preferred_support_task = candidate_declared_tasks[0]
					preferred_schema = task_schemas.get(preferred_support_task)
					preferred_parameters = tuple(
						_parameter_token(parameter)
						for parameter in getattr(preferred_schema, "parameters", ()) or ()
					)
					preferred_mapping = {
						parameter: argument
						for parameter, argument in zip(
							preferred_parameters,
							precondition_args,
						)
					}
					preferred_shared_requirements = tuple(
						dict.fromkeys(
							_render_signature_with_mapping(
								requirement,
								preferred_mapping,
							)
							for requirement in _support_task_precise_shared_requirements(
								domain,
								preferred_support_task,
								precondition_predicate,
								preferred_parameters,
								task_schemas,
								action_analysis,
							)
							if _render_signature_with_mapping(
								requirement,
								preferred_mapping,
							)
						)
					)
					preferred_support_call = _task_invocation_signature(
						preferred_support_task,
						precondition_args,
					)
					if preferred_support_call not in headline_support_calls:
						headline_support_calls.append(preferred_support_call)
					for requirement in preferred_shared_requirements:
						if requirement not in headline_support_context_requirements:
							headline_support_context_requirements.append(requirement)
				else:
					if rendered_precondition_signature not in headline_support_context_requirements:
						headline_support_context_requirements.append(
							rendered_precondition_signature,
						)

				support_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
					precondition_predicate,
					[],
				)
				rendered_support_options: list[str] = []
				mode_obligation_lines: list[str] = []
				has_extra_role_support_pattern = False
				for candidate_task in candidate_declared_tasks[:2]:
					candidate_schema = task_schemas.get(candidate_task)
					if candidate_schema is None:
						continue
					candidate_parameters = tuple(
						_parameter_token(parameter)
						for parameter in candidate_schema.parameters
					)
					rendered_support_options.append(
						_task_invocation_signature(candidate_task, precondition_args)
					)
					shared_requirements = _support_task_precise_shared_requirements(
						domain,
						candidate_task,
						precondition_predicate,
						candidate_parameters,
						task_schemas,
						action_analysis,
					)
					rendered_shared_requirements = tuple(
						dict.fromkeys(
							_render_signature_with_mapping(
								requirement,
								{
									parameter: argument
									for parameter, argument in zip(
										candidate_parameters,
										precondition_args,
									)
								},
							)
							for requirement in shared_requirements
							if _render_signature_with_mapping(
								requirement,
								{
									parameter: argument
									for parameter, argument in zip(
										candidate_parameters,
										precondition_args,
									)
								},
							)
						)
					)
					if rendered_shared_requirements:
						mode_obligation_lines.append(
							f"- {_task_invocation_signature(display_name, task_args)}: if the "
							f"constructive branch uses "
							f"{_task_invocation_signature(candidate_task, precondition_args)} "
							f"to establish {precondition_predicate}({', '.join(precondition_args)}), "
							f"support {', '.join(rendered_shared_requirements)} before that step."
						)
				for support_pattern in support_patterns[:3]:
					support_effect_args = list(support_pattern.get("effect_args") or [])
					if len(support_effect_args) != len(precondition_args):
						continue

					support_mapping = {
						token: arg
						for token, arg in zip(support_effect_args, precondition_args)
					}
					rendered_support_action_args = _extend_mapping_with_action_parameters(
						support_mapping,
						support_pattern.get("action_parameters") or [],
						action_parameter_types=support_pattern.get("action_parameter_types")
						or [],
					)

					support_preconditions = [
						signature
						for signature in _render_positive_dynamic_requirements(
							support_pattern,
							support_mapping,
						)
					]
					support_suffix = (
						f" [needs {', '.join(support_preconditions)}]"
						if support_preconditions
						else ""
					)
					rendered_support_options.append(
						f"{_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)}"
						f"{support_suffix}"
					)
					extra_roles = [
						arg
						for arg in rendered_support_action_args
						if arg not in precondition_args
					]
					has_extra_role_support_pattern = (
						has_extra_role_support_pattern or bool(extra_roles)
					)
					if support_preconditions and not extra_roles:
						headline_remaining_requirements = [
							rendered_requirement
							for signature in (
								headline_pattern.get("dynamic_precondition_signatures") or []
							)
							for rendered_requirement in [
								_render_signature_with_mapping(signature, headline_mapping)
							]
							if rendered_requirement
							and rendered_requirement != rendered_precondition_signature
						]
						mode_slot_requirements = list(
							dict.fromkeys(
								[
									*support_preconditions,
									*headline_remaining_requirements,
								]
							)
						)
						mode_line = (
							f"- {_task_invocation_signature(display_name, task_args)}: if the "
							f"constructive branch uses "
							f"{_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)} "
							f"to establish {precondition_predicate}({', '.join(precondition_args)}), "
							f"support {', '.join(support_preconditions)} before that step."
						)
						mode_obligation_lines.append(mode_line)
						mode_slot_line = (
							f"- {_task_invocation_signature(display_name, task_args)}: "
							f"valid constructive sibling for "
							f"{_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)}: "
							f"precondition/context "
							f"{', '.join(mode_slot_requirements) if mode_slot_requirements else 'none'}; "
							f"producer "
							f"{_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)}; "
							f"followup {rendered_headline_call}."
						)
						mode_obligation_lines.append(mode_slot_line)
						mode_ast_slot_line = (
							f"- {_task_invocation_signature(display_name, task_args)}: "
							"AST slot shape for that constructive sibling: "
							"{"
							f'"precondition":[{", ".join(json.dumps(item) for item in mode_slot_requirements)}],'
							f'"producer":{json.dumps(_task_invocation_signature(support_pattern["action_name"], rendered_support_action_args))},'
							f'"followup":{json.dumps(rendered_headline_call)}'
							"}."
						)
						mode_obligation_lines.append(mode_ast_slot_line)
						for support_requirement in support_preconditions:
							parsed_requirement = _parse_literal_signature(support_requirement)
							if parsed_requirement is None:
								continue
							requirement_predicate, requirement_args, requirement_positive = parsed_requirement
							if not requirement_positive:
								continue
							requirement_support_tasks = [
								candidate
								for candidate in _candidate_support_task_names(
									domain,
									requirement_predicate,
									requirement_args,
									action_analysis.get("producer_actions_by_predicate", {}).get(
										requirement_predicate,
										[],
									),
								)
								if len(getattr(task_schemas.get(candidate), "parameters", ())) == len(requirement_args)
							]
							if not requirement_support_tasks:
								continue
							split_line = (
								f"- {_task_invocation_signature(display_name, task_args)}: if "
								f"{support_requirement} may either already hold or require support "
								f"before "
								f"{_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)}, "
								f"use separate constructive siblings: one may require "
								f"{support_requirement} at entry, another should use "
								f"{_task_invocation_signature(requirement_support_tasks[0], requirement_args)} "
								f"before {_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)}. "
								"Do not collapse both cases into one precondition-only branch."
							)
							mode_obligation_lines.append(split_line)
							other_mode_requirements = [
								requirement
								for requirement in support_preconditions
								if requirement != support_requirement
							]
							slot_context_requirements = list(
								dict.fromkeys(
									[
										*other_mode_requirements,
										*headline_remaining_requirements,
									]
								)
							)
							if slot_context_requirements:
								slot_line = (
									f"- {_task_invocation_signature(display_name, task_args)}: "
									f"valid support-then-produce sibling for the false-"
									f"{support_requirement} case with "
									f"{_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)}: "
									f"precondition/context {', '.join(slot_context_requirements)}; "
									f"support_before "
									f"{_task_invocation_signature(requirement_support_tasks[0], requirement_args)}; "
									f"producer "
									f"{_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)}; "
									f"followup {rendered_headline_call}."
								)
								mode_obligation_lines.append(slot_line)
								slot_ast_line = (
									f"- {_task_invocation_signature(display_name, task_args)}: "
									"AST slot shape for that support-then-produce sibling: "
									"{"
									f'"precondition":[{", ".join(json.dumps(item) for item in slot_context_requirements)}],'
									f'"support_before":[{json.dumps(_task_invocation_signature(requirement_support_tasks[0], requirement_args))}],'
									f'"producer":{json.dumps(_task_invocation_signature(support_pattern["action_name"], rendered_support_action_args))},'
									f'"followup":{json.dumps(rendered_headline_call)}'
									"}."
								)
								mode_obligation_lines.append(slot_ast_line)
							no_duplicate_line = (
								f"- {_task_invocation_signature(display_name, task_args)}: if "
								f"support_before {_task_invocation_signature(requirement_support_tasks[0], requirement_args)} "
								f"handles false-{support_requirement} before "
								f"{_task_invocation_signature(support_pattern['action_name'], rendered_support_action_args)}, "
								f"do not also keep {support_requirement} in that same sibling "
								"precondition/context; split already-supported and support-then-produce siblings."
							)
							mode_obligation_lines.append(no_duplicate_line)

				if not rendered_support_options:
					continue

				line = (
					f"- {_task_invocation_signature(display_name, task_args)}: "
					f"{rendered_headline_call} requires "
					f"{precondition_predicate}({', '.join(precondition_args)}). "
					f"Support options: {'; '.join(rendered_support_options)}"
				)
				if line in seen:
					continue
				seen.add(line)
				lines.append(line)
				if (
					force_internal_contract
					and rendered_precondition_signature not in packaging_shared_requirements
				):
					if candidate_declared_tasks:
						preferred_support_task = candidate_declared_tasks[0]
						explicit_support_line = (
							f"- {_task_invocation_signature(display_name, task_args)}: because "
							f"{precondition_predicate}({', '.join(precondition_args)}) is not "
							f"caller-shared here, constructive branches must establish "
							f"{precondition_predicate}({', '.join(precondition_args)}) via "
							f"{_task_invocation_signature(preferred_support_task, precondition_args)} "
							f"before {rendered_headline_call} instead of leaving it in branch "
							"context."
						)
						if explicit_support_line not in seen:
							seen.add(explicit_support_line)
							lines.append(explicit_support_line)
						availability_split_line = (
							f"- {_task_invocation_signature(display_name, task_args)}: if "
							f"{precondition_predicate}({', '.join(precondition_args)}) may either "
							f"already hold or require internal support, use separate constructive "
							f"siblings: one may require {precondition_predicate}({', '.join(precondition_args)}) "
							f"at entry, another should use "
							f"{_task_invocation_signature(preferred_support_task, precondition_args)} "
							f"before {rendered_headline_call}. Do not collapse both cases into one "
							"precondition-only branch."
						)
						if availability_split_line not in seen:
							seen.add(availability_split_line)
							lines.append(availability_split_line)
						remaining_mode_needs = [
							rendered_requirement
							for signature in (
								headline_pattern.get("dynamic_precondition_signatures") or []
							)
							for rendered_requirement in [
								_render_signature_with_mapping(signature, headline_mapping)
							]
							if rendered_requirement
							and rendered_requirement != rendered_precondition_signature
							and (
								not packaging_shared_requirements
								or rendered_requirement in packaging_shared_requirements
								or force_internal_contract
							)
						]
						if remaining_mode_needs:
							remaining_needs_line = (
								f"- {_task_invocation_signature(display_name, task_args)}: if a "
								f"constructive sibling adds "
								f"{_task_invocation_signature(preferred_support_task, precondition_args)} "
								f"before {rendered_headline_call}, keep that mode's other unmet "
								f"needs {', '.join(remaining_mode_needs)} in the same sibling "
								"precondition/context unless earlier subtasks in that sibling "
								"already support them."
							)
							if remaining_needs_line not in seen:
								seen.add(remaining_needs_line)
								lines.append(remaining_needs_line)
							internal_support_then_produce_line = (
								f"- {_task_invocation_signature(display_name, task_args)}: "
								f"valid internal-support sibling for the false-"
								f"{precondition_predicate}({', '.join(precondition_args)}) case: "
								f"precondition/context {', '.join(remaining_mode_needs)}; "
								f"support_before "
								f"{_task_invocation_signature(preferred_support_task, precondition_args)}; "
								f"producer {rendered_headline_call}."
							)
							if internal_support_then_produce_line not in seen:
								seen.add(internal_support_then_produce_line)
								lines.append(internal_support_then_produce_line)
							no_duplicate_supported_need_line = (
								f"- {_task_invocation_signature(display_name, task_args)}: if "
								f"{_task_invocation_signature(preferred_support_task, precondition_args)} "
								f"before {rendered_headline_call} handles false-"
								f"{precondition_predicate}({', '.join(precondition_args)}), do not "
								f"also keep {precondition_predicate}({', '.join(precondition_args)}) "
								"in that same sibling precondition/context; split already-supported "
								"and internal-support siblings."
							)
							if no_duplicate_supported_need_line not in seen:
								seen.add(no_duplicate_supported_need_line)
								lines.append(no_duplicate_supported_need_line)
							conflicting_support_requirements = [
								requirement
								for requirement in _shared_dynamic_requirements_for_predicate(
									precondition_predicate,
									precondition_args,
									action_analysis,
								)
								if any(
									_requirements_are_mutually_destructive(
										requirement,
										remaining_need,
										action_analysis,
									)
									for remaining_need in remaining_mode_needs
								)
							]
							if conflicting_support_requirements:
								conflict_line = (
									f"- {_task_invocation_signature(display_name, task_args)}: if an internal-support sibling adds "
									f"{_task_invocation_signature(preferred_support_task, precondition_args)} "
									f"but that support task still needs {', '.join(conflicting_support_requirements)} "
									f"while the same sibling still needs {', '.join(remaining_mode_needs)}, "
									"do not keep both requirements in one constructive sibling. Split the support "
									"earlier in the parent or keep only a caller-compatible sibling that "
									f"receives {precondition_predicate}({', '.join(precondition_args)}) at entry."
								)
								if conflict_line not in seen:
									seen.add(conflict_line)
									lines.append(conflict_line)
							binding_needs = [
								requirement
								for requirement in remaining_mode_needs
								if _signature_mentions_aux_role(requirement)
							]
							if binding_needs:
								binding_line = (
									f"- {_task_invocation_signature(display_name, task_args)}: if a "
									f"constructive sibling adds "
									f"{_task_invocation_signature(preferred_support_task, precondition_args)} "
									f"before {rendered_headline_call}, keep binding literal(s) "
									f"{', '.join(binding_needs)} in that sibling "
									"precondition/context before the first use of the AUX role."
								)
								if binding_line not in seen:
									seen.add(binding_line)
									lines.append(binding_line)
					else:
						internal_line = (
							f"- {_task_invocation_signature(display_name, task_args)}: because "
							f"{precondition_predicate}({', '.join(precondition_args)}) has listed "
							"support options here and is not caller-shared, constructive branches "
							f"must establish {precondition_predicate}({', '.join(precondition_args)}) "
							"with earlier subtasks inside this task instead of leaving it in branch "
							"context."
						)
						if internal_line not in seen:
							seen.add(internal_line)
							lines.append(internal_line)
				for mode_line in mode_obligation_lines:
					if mode_line in seen:
						continue
					seen.add(mode_line)
					lines.append(mode_line)

				if candidate_declared_tasks:
					preferred_support_task = candidate_declared_tasks[0]
					preferred_schema = task_schemas.get(preferred_support_task)
					preferred_support_parameters = tuple(
						_parameter_token(parameter)
						for parameter in getattr(preferred_schema, "parameters", ()) or ()
					)
					preferred_support_mapping = {
						parameter: argument
						for parameter, argument in zip(
							preferred_support_parameters,
							precondition_args,
						)
					}
					preferred_shared_requirements = tuple(
						dict.fromkeys(
							_render_signature_with_mapping(
								requirement,
								preferred_support_mapping,
							)
							for requirement in _support_task_precise_shared_requirements(
								domain,
								preferred_support_task,
								precondition_predicate,
								preferred_support_parameters,
								task_schemas,
								action_analysis,
							)
							if _render_signature_with_mapping(
								requirement,
								preferred_support_mapping,
							)
						)
					)
					headline_remaining_requirements = tuple(
						dict.fromkeys(
							rendered_requirement
							for signature in (
								headline_pattern.get("dynamic_precondition_signatures") or []
							)
							for rendered_requirement in [
								_render_signature_with_mapping(signature, headline_mapping)
							]
							if rendered_requirement
							and rendered_requirement != rendered_precondition_signature
						)
					)
					support_then_produce_context = tuple(
						dict.fromkeys(
							[
								*preferred_shared_requirements,
								*headline_remaining_requirements,
							]
						)
					)
					if support_then_produce_context:
						support_then_produce_line = (
							f"- {_task_invocation_signature(display_name, task_args)}: "
							f"valid support-then-produce sibling for the false-"
							f"{rendered_precondition_signature} case with "
							f"{_task_invocation_signature(preferred_support_task, precondition_args)}: "
							f"precondition/context {', '.join(support_then_produce_context)}; "
							f"support_before "
							f"{_task_invocation_signature(preferred_support_task, precondition_args)}; "
							f"producer {rendered_headline_call}."
						)
						if support_then_produce_line not in seen:
							seen.add(support_then_produce_line)
							lines.append(support_then_produce_line)
						support_then_produce_ast_line = (
							f"- {_task_invocation_signature(display_name, task_args)}: "
							"AST slot shape for that support-then-produce sibling: "
							"{"
							f'"precondition":[{", ".join(json.dumps(item) for item in support_then_produce_context)}],'
							f'"support_before":[{json.dumps(_task_invocation_signature(preferred_support_task, precondition_args))}],'
							f'"producer":{json.dumps(rendered_headline_call)}'
							"}."
						)
						if support_then_produce_ast_line not in seen:
							seen.add(support_then_produce_ast_line)
							lines.append(support_then_produce_ast_line)
						no_duplicate_support_line = (
							f"- {_task_invocation_signature(display_name, task_args)}: if "
							f"support_before {_task_invocation_signature(preferred_support_task, precondition_args)} "
							f"handles false-{rendered_precondition_signature} before "
							f"{rendered_headline_call}, do not also keep "
							f"{rendered_precondition_signature} in that same sibling "
							"precondition/context; split already-supported and support-then-produce siblings."
						)
						if no_duplicate_support_line not in seen:
							seen.add(no_duplicate_support_line)
							lines.append(no_duplicate_support_line)

				if candidate_declared_tasks:
					continue
				if (
					same_arity_candidates
					and len(packaging_shared_requirements) > 1
					and len(support_patterns) > 1
					and has_extra_role_support_pattern
				):
					helper_task_name = _sanitize_name(f"do_{precondition_predicate}")
					packaging_call = _task_invocation_signature(
						str(same_arity_candidates[0].get("candidate", "")).strip(),
						task_args,
					)
					helper_support_calls: list[str] = []
					for sibling_requirement in packaging_shared_requirements:
						if sibling_requirement == rendered_precondition_signature:
							continue
						parsed_sibling_requirement = _parse_literal_signature(
							sibling_requirement,
						)
						if parsed_sibling_requirement is None:
							continue
						sibling_predicate, sibling_args, sibling_positive = parsed_sibling_requirement
						if not sibling_positive or not sibling_args:
							continue
						sibling_support_tasks = [
							support_task
							for support_task in _candidate_support_task_names(
								domain,
								sibling_predicate,
								sibling_args,
								action_analysis.get("producer_actions_by_predicate", {}).get(
									sibling_predicate,
									[],
								),
							)
							if support_task != task_name
							and (
								len(getattr(task_schemas.get(support_task), "parameters", ()))
								== len(sibling_args)
							)
						]
						if sibling_support_tasks:
							helper_support_calls.append(
								_task_invocation_signature(
									sibling_support_tasks[0],
									sibling_args,
								),
							)
					helper_line = (
						f"- {_task_invocation_signature(display_name, task_args)}: no declared task "
						f"headlines {precondition_predicate}({', '.join(precondition_args)}), but a "
						"same-arity packaging child still expects it as caller-shared while other "
						"caller-shared support may run earlier. Prefer one minimal helper task for that "
						f"predicate here; let that helper own the "
						f"producer choice for {precondition_predicate}({', '.join(precondition_args)}) "
						"after earlier parent supports settle instead of hard-wiring one fragile "
						"producer mode into every parent branch. If you introduce that helper, keep "
						"its parameters aligned with the predicate arity and give it a stable "
						"predicate-aligned name."
					)
					if helper_support_calls:
						helper_line += (
							" More concrete shape: when earlier caller-shared support may change which "
							f"{precondition_predicate}({', '.join(precondition_args)}) producer mode "
							"remains valid, prefer one predicate-aligned helper task "
							f"{_task_invocation_signature(helper_task_name, precondition_args)} rather "
							f"than hard-wiring {' or '.join(rendered_support_options)} directly into the "
							"parent branch. Parent sibling shape: "
							f"support_before {'; '.join(helper_support_calls)}; "
							f"producer {_task_invocation_signature(helper_task_name, precondition_args)}; "
							f"followup {packaging_call}. Inside "
							+ f"{_task_invocation_signature(helper_task_name, precondition_args)}, keep "
							+ f"noop {precondition_predicate}({', '.join(precondition_args)}) and "
							+ "constructive branches that mirror the listed support options."
						)
					if helper_line not in seen:
						seen.add(helper_line)
						lines.append(helper_line)
					continue
				if same_arity_candidates:
					continue
				helper_line = (
					f"- {_task_invocation_signature(display_name, task_args)}: no declared task "
					f"clearly matches {precondition_predicate}({', '.join(precondition_args)}). "
					"A minimal helper task for that dynamic predicate is allowed only as a "
					"last resort if earlier context or subtasks do not already provide it."
				)
				if helper_line in seen:
					continue
				seen.add(helper_line)
				lines.append(helper_line)
			if headline_support_calls and headline_support_context_requirements:
				headline_static_binding_requirements = tuple(
					dict.fromkeys(
						requirement
						for requirement in _render_positive_static_requirements(
							headline_pattern,
							headline_mapping,
						)
						if _signature_mentions_aux_role(requirement)
					)
				)
				complete_context_requirements = tuple(
					dict.fromkeys(
						[
							*headline_support_context_requirements,
							*headline_static_binding_requirements,
						]
					)
				)
				complete_branch_line = (
					f"- {_task_invocation_signature(display_name, task_args)}: "
					f"complete constructive sibling for {rendered_headline_call}: "
					f"precondition/context {', '.join(complete_context_requirements)}; "
					f"support_before {'; '.join(headline_support_calls)}; "
					f"producer {rendered_headline_call}."
				)
				if complete_branch_line not in seen:
					seen.add(complete_branch_line)
					lines.append(complete_branch_line)
				complete_branch_ast_line = (
					f"- {_task_invocation_signature(display_name, task_args)}: "
					"AST slot shape for that complete constructive sibling: "
					"{"
					f'"precondition":[{", ".join(json.dumps(item) for item in complete_context_requirements)}],'
					f'"support_before":[{", ".join(json.dumps(item) for item in headline_support_calls)}],'
					f'"producer":{json.dumps(rendered_headline_call)}'
					"}."
				)
				if complete_branch_ast_line not in seen:
					seen.add(complete_branch_ast_line)
					lines.append(complete_branch_ast_line)
	return lines


def build_htn_system_prompt() -> str:
	# Prompt structure follows a deliberately declarative design:
	# 1. tagged structured blocks and fixed output contracts help models recover
	#    latent structure more reliably (Blevins et al., ACL 2023; Beurer-Kellner
	#    et al., PLDI 2023),
	# 2. prompt-based semantic parsers are sensitive to avoidable variation, so we
	#    canonicalise the instruction layout and keep only salient constraints
	#    (Zhuo et al., EACL 2023),
	# 3. declarative specifications stay closer to the task than free-form
	#    procedural explanations (Ye et al., NeurIPS 2023).
	return (
		"You generate a compact, executable query-specific HTN method library for a symbolic planning domain.\n"
		"The domain methods are unavailable. Infer methods only from the natural-language query, "
		"declared domain tasks, predicates, and primitive action schemas.\n"
		"No later repair pass will add missing methods, wrappers, or bridges for you.\n"
		"Return JSON only. No markdown, no comments, no prose.\n"
		"Return exactly one compact JSON object with exactly these top-level keys: "
		"target_task_bindings, compound_tasks, methods.\n"
		"\n"
		"OUTPUT CONTRACT:\n"
		"- Emit the shortest valid JSON you can. Prefer compact formatting over pretty-printing.\n"
		"- task_name and method_name must match [a-z][a-z0-9_]*.\n"
		"- method_name must be exactly m_{task_name}_{strategy}.\n"
		"- task_args is optional; if omitted, the first declared-task-arity entries of method.parameters are treated as the task arguments in order, so keep those leading parameters aligned with the task signature.\n"
		"- Primitive subtasks must use the provided runtime primitive aliases.\n"
		"- Set primitive step literal to null unless that exact positive literal is a real action effect.\n"
		"- Omit optional empty/default fields whenever possible: task_args when implied by leading parameters, origin when it would just be llm/default, source_method_name, literal when null, preconditions/effects when empty, and ordering only for empty or single-step methods.\n"
		"- Never omit required structural fields. compound_tasks still need name, parameters, is_primitive, source_predicates, and the exact task headline literal in headline. Every subtask still needs step_id, task_name, args, and kind. Primitive subtasks also need action_name.\n"
		"- Every compound subtask must reference a declared compound task from the same JSON.\n"
		"- Every referenced compound subtask must also appear in compound_tasks and have at least one method in methods.\n"
		"- Zero-subtask methods must have non-empty context and empty subtasks/orderings.\n"
		"- Reuse parameterized tasks; never encode grounded constants into task names.\n"
		"- If both Stage 1 semantic-object hints and a query object inventory are supplied, the query inventory is authoritative for grounded instances; the semantic hints may be partial.\n"
		"- Do not use grounded object names as method variables; method.parameters must remain schematic.\n"
		"- Auxiliary variables must appear in method.parameters and be constrained by method context before use in subtasks.\n"
		"- Every auxiliary variable needs typed evidence before reuse.\n"
		"- Keep typed roles separate: one variable cannot stand for incompatible declared types or semantic roles.\n"
		"- Never use deprecated task prefixes achieve_, ensure_, goal_, or maintain_not_.\n"
		"- Every literal-bearing field must use JSON object form with predicate/args/is_positive.\n"
		"- headline must be the exact positive literal the task is meant to establish, with arguments written in the true task-role order even when that order permutes ARG positions.\n"
		"\n"
		"SYNTHESIS POLICY:\n"
		"- When the query explicitly names declared domain tasks, those task names are the primary HTN skeleton.\n"
		"- Create a fresh helper task only if no declared task can express the required dynamic state change.\n"
		"- If no declared task directly headlines an intermediate dynamic predicate but a reusable same-arity declared packaging task exists, prefer that declared packaging task before inventing a helper.\n"
		"- Fresh helper tasks may correspond only to dynamic predicates.\n"
		"- Static predicates are context constraints only; never create helper tasks to establish them.\n"
		"- Bind each target literal to one top-level compound task. If an ordered query-task anchor is supplied for that target, prefer that exact task name.\n"
		"- For every target-bound task, include an already-satisfied/noop method whose context contains the target headline literal and whose subtask list is empty. For declared support/stabilizer tasks, an already-stable/noop branch may instead use the downstream support-role condition when that already makes the role reusable.\n"
		"- Keep the library compact after executability is satisfied. Emit sibling methods only for genuine mode differences, producer-action alternatives, or already-satisfied cases.\n"
		"- Do not generate transitive support-closure libraries or exhaustive missing-support powersets.\n"
		"- Keep static resources, capabilities, topology, and immutable relations in method context when possible.\n"
		"- Every primitive step's dynamic preconditions must be guaranteed by earlier subtasks or method context.\n"
		"- Apply the same rule to compound subtasks when their constructive branches share a dynamic prerequisite.\n"
		"- When sequencing compound subtasks, only rely on a prior compound child's own headline effect and explicitly shared envelope as guaranteed for later siblings; do not rely on incidental internal side effects or cleanup from that child.\n"
		"- If a same-arity declared child is chosen as packaging for the headline effect, let that child own the full constructive path; do not immediately repeat the same final producer in the parent.\n"
		"- If that packaging child still needs an internal support predicate with multiple real producer modes and no declared support task directly headlines it, keep those modes inside the child as separate constructive sibling methods instead of choosing one narrow mode as the only constructive branch.\n"
		"- If a support task can recursively establish an extra-role prerequisite for one of its own producer modes, recursion is allowed and often necessary; place the recursive support-task call before the primitive producer instead of leaving that blocker as an unsupported context assumption.\n"
		"- If a remaining shared dynamic prerequisite has arity 0 and no earlier parent subtask establishes it, usually state it explicitly in the parent method context before the child call.\n"
		"- No free variables; respect declared types and role distinctions.\n"
	)


def build_htn_user_prompt(
	domain: Any,
	target_literals: Iterable[str],
	schema_hint: str,
	*,
	query_text: str = "",
	query_task_anchors: Sequence[Dict[str, Any]] = (),
	semantic_objects: Sequence[str] = (),
	query_object_inventory: Sequence[Dict[str, Any]] = (),
	query_objects: Sequence[str] = (),
	action_analysis: Optional[Dict[str, Any]] = None,
	derived_analysis: Optional[Dict[str, Any]] = None,
) -> str:
	parser = HDDLConditionParser()
	analysis = _normalise_action_analysis(domain, action_analysis)
	prompt_analysis = dict(
		derived_analysis
		or build_prompt_analysis_payload(
			domain,
			query_task_anchors=query_task_anchors,
			action_analysis=analysis,
		),
	)
	query_task_anchors = tuple(
		prompt_analysis.get("ordered_query_task_anchors") or query_task_anchors,
	)
	query_object_inventory = _normalise_query_object_inventory(query_object_inventory)
	query_objects = tuple(
		query_objects
		or _query_object_names_from_inventory(query_object_inventory)
	)

	action_lines = []
	action_branch_hint_lines = []
	for action in domain.actions:
		params = ", ".join(action.parameters) if action.parameters else "none"
		action_lines.append(
			f"- runtime task: {_sanitize_name(action.name)} | source action: {action.name}"
			f"({params}) | pre: {action.preconditions} | eff: {action.effects}",
		)
		try:
			parsed_action = parser.parse_action(action)
		except Exception:
			continue
		if len(parsed_action.precondition_clauses) <= 1:
			continue
		clauses = " OR ".join(
			f"[{_clause_signature(clause)}]"
			for clause in parsed_action.precondition_clauses
		)
		action_branch_hint_lines.append(
			f"- {_sanitize_name(action.name)} applicability branches: {clauses}",
		)

	predicate_lines = [f"- {predicate.to_signature()}" for predicate in domain.predicates]
	task_lines = [f"- {task.to_signature()}" for task in getattr(domain, "tasks", [])]
	type_lines = [f"- {type_name}" for type_name in domain.types]
	task_schemas = _declared_task_schema_map(domain)
	targets = list(target_literals)
	target_lines = "\n".join(
		f"- #{index}: {item}"
		for index, item in enumerate(targets, start=1)
	)
	branch_hints = (
		"\n".join(action_branch_hint_lines)
		if action_branch_hint_lines
		else "- none"
	)
	query_anchor_lines = "\n".join(
		f"- #{index}: {_task_invocation_signature(_anchor_display_name(anchor), anchor.get('args', []))}"
		for index, anchor in enumerate(query_task_anchors, start=1)
	) or "- none detected"
	query_object_lines = "\n".join(
		f"- {item}"
		for item in dict.fromkeys(
			str(value).strip()
			for value in query_objects
			if str(value).strip()
		)
	) or "- none detected"
	semantic_object_lines = "\n".join(
		f"- {item}"
		for item in dict.fromkeys(
			str(value).strip()
			for value in semantic_objects
			if str(value).strip()
		)
	) or "- none detected"
	query_object_inventory_lines = "\n".join(
		f"- {entry['type']}: {', '.join(entry['objects'])}"
		for entry in query_object_inventory
	) or "- none detected"
	anchor_binding_lines = "- no ordered target/task hint available"
	if query_task_anchors and len(query_task_anchors) == len(targets):
		anchor_binding_lines = "\n".join(
			f"- target #{index} {target} -> prefer declared task "
			f"{_task_invocation_signature(_anchor_display_name(anchor), anchor.get('args', []))}"
			for index, (target, anchor) in enumerate(
				zip(targets, query_task_anchors),
				start=1,
			)
		)
	query_task_parameter_alignment_lines = "\n".join(
		f"- {_task_invocation_signature(_anchor_display_name(anchor), tuple(_parameter_token(parameter) for parameter in task_schemas[anchor['task_name']].parameters))}: "
		"if task_args is omitted, keep these task-argument roles as the first "
		"method.parameters in exactly this order before adding support-role variables."
		for anchor in query_task_anchors
		if anchor.get("task_name") in task_schemas
	) or "- none"
	query_priority_obligation_lines = "\n".join(
		_query_priority_obligation_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"

	dynamic_predicate_lines = "\n".join(
		f"- {predicate_name}"
		for predicate_name in analysis["dynamic_predicates"]
	) or "- none"
	static_predicate_lines = "\n".join(
		f"- {predicate_name}"
		for predicate_name in analysis["static_predicates"]
	) or "- none"
	producer_lines = "\n".join(
		f"- {predicate_name} <- {', '.join(actions) if actions else 'none'}"
		for predicate_name, actions in analysis["producer_actions_by_predicate"].items()
	) or "- none"
	producer_pattern_lines = "\n".join(
		f"- {predicate_name} can be produced by {pattern['action_name']} via "
		f"{pattern['effect_signature']}; dynamic preconditions: "
		f"{', '.join(pattern['dynamic_precondition_signatures']) if pattern['dynamic_precondition_signatures'] else 'none'}"
		for predicate_name, patterns in analysis["producer_patterns_by_predicate"].items()
		for pattern in patterns
	) or "- none"
	dynamic_support_lines = "\n".join(
		_dynamic_support_hint_lines(domain, analysis)
	) or "- none"
	aligned_producer_lines = "\n".join(
		_aligned_producer_template_lines(analysis)
	) or "- none"
	shared_producer_requirement_lines = "\n".join(
		_shared_producer_requirement_lines(analysis)
	) or "- none"
	same_arity_task_lines = "\n".join(
		_same_arity_task_hint_lines(domain, query_task_anchors)
	) or "- none"
	query_task_same_arity_packaging_lines = "\n".join(
		_query_task_same_arity_packaging_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_same_arity_transitive_lines = "\n".join(
		_query_task_same_arity_transitive_requirement_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	declared_task_template_lines = "\n".join(
		_declared_task_producer_template_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	relevant_support_task_template_lines = "\n".join(
		_relevant_support_task_template_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	relevant_support_task_alignment_lines = "\n".join(
		_relevant_support_task_alignment_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	declared_support_task_applicability_lines = "\n".join(
		_declared_support_task_applicability_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	relevant_support_task_internal_obligation_lines = "\n".join(
		_relevant_support_task_internal_obligation_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	relevant_support_task_recursive_mode_lines = "\n".join(
		_relevant_support_task_recursive_mode_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	relevant_support_task_recursive_template_lines = "\n".join(
		_relevant_support_task_recursive_template_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	relevant_support_task_cleanup_template_lines = "\n".join(
		_relevant_support_task_cleanup_template_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_support_lines = "\n".join(
		_query_task_support_obligation_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_support_producer_lines = "\n".join(
		_query_task_support_producer_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_role_frame_lines = "\n".join(
		_query_task_role_frame_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_role_stabilization_lines = "\n".join(
		_query_task_role_stabilization_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_role_stabilizer_support_lines = "\n".join(
		_query_task_role_stabilizer_support_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_child_support_lines = "\n".join(
		_query_task_child_support_requirement_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_same_arity_child_support_lines = "\n".join(
		_query_task_same_arity_child_support_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_zero_ary_parent_context_lines = "\n".join(
		_query_task_zero_ary_parent_context_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_same_arity_child_context_lines = "\n".join(
		_query_task_same_arity_child_context_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	query_task_packaging_skeleton_lines = "\n".join(
		_query_task_packaging_skeleton_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	declared_task_shared_prerequisite_lines = "\n".join(
		f"- {_task_invocation_signature(task_name, tuple(_parameter_token(parameter) for parameter in _declared_task_schema_map(domain)[task_name].parameters))}: "
		f"{', '.join(requirements)}"
		for task_name, requirements in (
			prompt_analysis.get("shared_dynamic_prerequisites_by_task", {}) or {}
		).items()
		if task_name in _declared_task_schema_map(domain)
	) or "- none"
	producer_consumer_template_lines = "\n".join(
		f"- {summary}"
		for summaries in (
			prompt_analysis.get("producer_consumer_templates_by_task", {}) or {}
		).values()
		for summary in summaries
	) or "- none"
	reusable_dynamic_resource_lines = "\n".join(
		f"- {item['predicate']} <- produced by "
		f"{', '.join(item.get('producer_actions', [])) or 'none'}; consumed by "
		f"{', '.join(item.get('consumer_actions', [])) or 'none'}"
		for item in (
			prompt_analysis.get("reusable_dynamic_resource_predicates", ()) or ()
		)
	) or "- none"
	binding_hints = "\n".join(
		f'- {{"target_literal": "{literal}", "task_name": "<top_level_declared_or_minimal_dynamic_task>"}}'
		for literal in targets
	)
	semantic_object_section = ""
	if semantic_object_lines != "- none detected":
		semantic_object_section = (
			"STAGE 1 SEMANTIC OBJECT HINTS (goal-semantics only; may be partial and are not authoritative for grounding):\n"
			f"{semantic_object_lines}\n\n"
		)
	query_object_inventory_section = ""
	if query_object_inventory_lines != "- none detected":
		query_object_inventory_section = (
			"QUERY OBJECT INVENTORY BY TYPE (authoritative grounded inventory from the query text):\n"
			f"{query_object_inventory_lines}\n\n"
		)
	query_object_section = ""
	if query_object_lines != "- none detected":
		query_object_section = (
			"QUERY OBJECT NAMES (grounded instances from the current query; never place them inside methods):\n"
			f"{query_object_lines}\n\n"
		)
	declared_task_shared_prerequisite_section = ""
	if len(targets) <= 4 or declared_task_shared_prerequisite_lines != "- none":
		declared_task_shared_prerequisite_section = (
			"DECLARED TASK SHARED DYNAMIC PREREQUISITES:\n"
			f"{declared_task_shared_prerequisite_lines}\n\n"
		)
	query_task_parameter_alignment_section = ""
	if len(targets) <= 4:
		query_task_parameter_alignment_section = (
			"QUERY-TASK PARAMETER ALIGNMENT:\n"
			f"{query_task_parameter_alignment_lines}\n\n"
		)
	query_task_alignment_checklist_line = ""
	if len(targets) <= 4:
		query_task_alignment_checklist_line = (
			"- If task_args is omitted, the leading method.parameters for every query-task method stay aligned with the declared task signature in order; do not start a query-task method with support-role variables.\n"
		)
	producer_consumer_template_section = ""
	reusable_dynamic_resource_section = ""
	if len(targets) <= 4:
		producer_consumer_template_section = (
			"DECLARED TASK PRODUCER/CONSUMER TEMPLATES:\n"
			f"{producer_consumer_template_lines}\n\n"
		)
		reusable_dynamic_resource_section = (
			"REUSABLE DYNAMIC RESOURCE PREDICATES:\n"
			f"{reusable_dynamic_resource_lines}\n\n"
		)
	query_priority_section = ""
	if query_priority_obligation_lines != "- none":
		query_priority_section = (
			"QUERY-SPECIFIC PRIORITY OBLIGATIONS:\n"
			f"{query_priority_obligation_lines}\n\n"
		)
	role_frame_section = ""
	if len(targets) <= 4 and query_task_role_frame_lines != "- none":
		role_frame_section = (
			f"QUERY-TASK EXTRA ROLE FRAMES:\n{query_task_role_frame_lines}\n\n"
		)
	role_stabilization_section = ""
	if query_task_role_stabilization_lines != "- none":
		role_stabilization_section = (
			"QUERY-TASK ROLE STABILIZATION:\n"
			f"{query_task_role_stabilization_lines}\n\n"
		)
	role_stabilizer_support_section = ""
	if query_task_role_stabilizer_support_lines != "- none":
		role_stabilizer_support_section = (
			"QUERY-TASK ROLE STABILIZER INTERNAL SUPPORT:\n"
			f"{query_task_role_stabilizer_support_lines}\n\n"
		)
	relevant_support_task_section = ""
	if relevant_support_task_template_lines != "- none":
		relevant_support_task_section = (
			"RELEVANT SUPPORT TASK CONSTRUCTIVE TEMPLATES:\n"
			f"{relevant_support_task_template_lines}\n\n"
		)
	relevant_support_task_alignment_section = ""
	if relevant_support_task_alignment_lines != "- none":
		relevant_support_task_alignment_section = (
			"RELEVANT SUPPORT TASK ROLE-ALIGNMENT WARNINGS:\n"
			f"{relevant_support_task_alignment_lines}\n\n"
		)
	declared_support_task_applicability_section = ""
	if declared_support_task_applicability_lines != "- none":
		declared_support_task_applicability_section = (
			"DECLARED SUPPORT TASK APPLICABILITY ENVELOPES:\n"
			f"{declared_support_task_applicability_lines}\n\n"
		)
	relevant_support_task_internal_obligation_section = ""
	if relevant_support_task_internal_obligation_lines != "- none":
		relevant_support_task_internal_obligation_section = (
			"RELEVANT SUPPORT TASK INTERNAL OBLIGATIONS:\n"
			f"{relevant_support_task_internal_obligation_lines}\n\n"
		)
	relevant_support_task_recursive_mode_section = ""
	if relevant_support_task_recursive_mode_lines != "- none":
		relevant_support_task_recursive_mode_section = (
			"RELEVANT SUPPORT TASK RECURSIVE MODES:\n"
			f"{relevant_support_task_recursive_mode_lines}\n\n"
		)
	relevant_support_task_recursive_template_section = ""
	if relevant_support_task_recursive_template_lines != "- none":
		relevant_support_task_recursive_template_section = (
			"RELEVANT SUPPORT TASK RECURSIVE TEMPLATES:\n"
			f"{relevant_support_task_recursive_template_lines}\n\n"
		)
	relevant_support_task_cleanup_template_section = ""
	if relevant_support_task_cleanup_template_lines != "- none":
		relevant_support_task_cleanup_template_section = (
			"RELEVANT SUPPORT TASK CLEANUP TEMPLATES:\n"
			f"{relevant_support_task_cleanup_template_lines}\n\n"
		)
	same_arity_transitive_section = ""
	if query_task_same_arity_transitive_lines != "- none":
		same_arity_transitive_section = (
			"QUERY-TASK SAME-ARITY PACKAGING MODES:\n"
			f"{query_task_same_arity_transitive_lines}\n\n"
		)
	child_support_section = ""
	if query_task_child_support_lines != "- none":
		child_support_section = (
			f"QUERY-TASK CHILD SUPPORT PREREQUISITES:\n{query_task_child_support_lines}\n\n"
		)
	same_arity_child_support_section = ""
	if query_task_same_arity_child_support_lines != "- none":
		same_arity_child_support_section = (
			"QUERY-TASK SAME-ARITY CHILD PREREQUISITES:\n"
			f"{query_task_same_arity_child_support_lines}\n\n"
		)
	same_arity_child_context_section = ""
	if query_task_same_arity_child_context_lines != "- none":
		same_arity_child_context_section = (
			"QUERY-TASK SAME-ARITY CHILD CONTEXT OBLIGATIONS:\n"
			f"{query_task_same_arity_child_context_lines}\n\n"
		)
	zero_ary_parent_context_section = ""
	if query_task_zero_ary_parent_context_lines != "- none":
		zero_ary_parent_context_section = (
			"QUERY-TASK ZERO-ARY PARENT CONTEXT:\n"
			f"{query_task_zero_ary_parent_context_lines}\n\n"
		)
	packaging_skeleton_section = ""
	if query_task_packaging_skeleton_lines != "- none":
		packaging_skeleton_section = (
			"QUERY-TASK PACKAGING SKELETONS:\n"
			f"{query_task_packaging_skeleton_lines}\n\n"
		)
	branch_hint_section = ""
	if branch_hints != "- none":
		branch_hint_section = (
			f"ACTION PRECONDITION BRANCH HINTS (DNF):\n{branch_hints}\n\n"
		)
	if len(targets) > 4:
		step_instructions_section = (
			"STEP-BY-STEP INSTRUCTIONS:\n"
			"1. Preserve query-mentioned declared tasks as the top-level skeleton, prefer declared support tasks, and only create a fresh helper when no declared task or same-arity packaging task can express the needed dynamic change; never create helpers for static predicates.\n"
			"2. For each constructive method, choose an aligned final producer or same-arity packaging child for the headline effect and recursively support its dynamic prerequisites instead of leaving them implicit.\n"
			"3. For every target-bound task, include one already-satisfied/noop branch with the headline literal in context. For declared support/stabilizer tasks, a stable/noop branch may instead use the downstream support-role condition when that already makes the role reusable. At least one constructive branch must then make progress when the target/stabilizer headline remains unsupported.\n"
			"4. Respect producer-effect argument alignment exactly; do not use a chain for !P(args) if unresolved support still requires P(args) to already hold.\n"
			"5. If a same-arity packaging child is chosen, let that child own the final producer. Keep unsupported internal predicates inside that child, and if such a predicate has multiple real producer modes with different dynamic prerequisites, encode those modes as separate constructive sibling methods inside the child.\n"
			"6. Auxiliary variables must appear in method.parameters, be constrained before use, keep typed roles separate, and never reuse grounded query object names inside methods. If task_args is omitted, keep the first task-arity method.parameters aligned with the task signature in order.\n"
			"7. Every primitive dynamic precondition and every shared compound-child prerequisite must be supported by method context or earlier subtasks. If the parent is expected to provide a child's shared envelope, restate those assumed literals in each constructive child context unless that child establishes them internally.\n"
			"8. If an extra-role producer mode needs a blocker role that the same declared support task can clear recursively, recurse on that blocker before the primitive producer, and append a real cleanup step before returning only when that cleanup's own preconditions are truly established by earlier steps on the same symbol.\n"
			"8b. If every real producer mode for an internal support literal still needs the same dynamic prerequisite on task arguments or a shared 0-ary resource, keep that requirement explicit in each constructive sibling context of the reusable task instead of re-establishing it with an extra support subtask inside every sibling.\n"
			"9. Every ordered compound support step must justify itself by contributing a headline literal that a later step or the parent headline actually consumes. Do not insert orphan support subtasks whose headline literal is never used downstream.\n"
			"10. Keep static capabilities, topology, visibility, equipment, and immutable relations in context; do not use nop as constructive filler.\n"
			"11. If a method has two or more subtasks, include explicit ordering edges; never rely on subtask array order alone.\n"
			"12. Use sibling methods only for real mode differences or already-satisfied cases, keep the library compact, and do not bypass the query skeleton with a fresh helper-only library.\n"
			"13. Keep the JSON structurally simple: omit empty/default fields and unnecessary literal metadata so the response stays short and well-formed.\n\n"
		)
	else:
		step_instructions_section = (
			"STEP-BY-STEP INSTRUCTIONS:\n"
			"1. If the query explicitly names declared tasks, preserve those task names in the generated library, use query-mentioned declared tasks as top-level bindings whenever they semantically match the ordered target literals, prefer declared supporting tasks over fresh helper tasks, only create a fresh helper task when no declared task can express the required dynamic state change, and never create helper tasks for static predicates.\n"
			"2. For each constructive method, inspect the final producer action or child task that achieves the intended dynamic effect. If that constructive step requires dynamic preconditions, normally establish them via supporting declared tasks or sibling mode branches instead of assuming them in context.\n"
			"3. For a target-bound task that headlines a positive dynamic predicate, include one already-satisfied/noop branch with the headline literal in context. For declared support/stabilizer tasks, a stable/noop branch may instead use the downstream support-role condition when that already makes the role reusable. Keep at least one constructive branch applicable when the target/stabilizer headline is still unsupported; non-empty branches should make progress from states where that literal is false.\n"
			"4. Respect producer-effect argument alignment exactly: only treat an action or subtask as supporting P(args) when its positive effect can instantiate to that same P(args). If a constructive branch is intended for !P(args), do not choose a producer chain whose unresolved preconditions still require that same P(args) to already hold.\n"
			"5. If a query task still has unresolved dynamic preconditions after obvious support tasks, prefer same-arity declared tasks as reusable intermediate abstractions before inventing a fresh helper. Only fall back to a fresh helper when no declared support task or packaging task can responsibly absorb that obligation. If you choose a same-arity declared packaging task, that child should own the final producer for the headline effect instead of being followed by the same final producer again in the parent.\n"
			"5b. If a same-arity packaging task's final producer still needs a dynamic prerequisite that no declared task directly headlines, keep that support obligation inside the packaging task via one of the real producer modes instead of pushing it back to the parent.\n"
			"5c. If that internal support predicate has multiple real producer modes with different dynamic prerequisites, encode those supported modes as separate constructive sibling methods inside the task that owns the final producer. Do not keep only one narrow mode as the sole constructive branch of a reusable generic task.\n"
			"6. If you introduce an auxiliary blocker/intermediate variable, add it to method.parameters, constrain it in method.context before using it in subtasks, and give it explicit typed evidence.\n"
			"7. If a declared task has a constructive producer template, build its constructive branch around one of those aligned templates instead of operating on a different object role. Preserve typed role separation across every method: do not reuse one symbol for incompatible declared types or semantic roles. Never place grounded query object names inside methods; use schematic parameters instead. When a producer needs extra roles beyond the headline arguments, add fresh schematic parameters for those roles.\n"
			"7b. When several real producer modes can achieve the same headline effect, prefer the mode with fewer extra roles and less extra support unless a more complex mode is genuinely needed. If you use an extra-role mode, declare those roles and support their dynamic prerequisites explicitly.\n"
			"8. For every primitive step, each dynamic precondition must already be guaranteed by method context or by earlier subtasks in the ordering. Do not stop after choosing a final producer step if that producer still has unresolved dynamic preconditions; recursively decompose those obligations first. If you intentionally leave a primitive step's dynamic precondition to method applicability instead of earlier subtasks, state it explicitly in method.context. If such a context literal introduces an auxiliary variable, declare that variable in method.parameters and constrain it in method.context. If a compound child task's constructive branches share a dynamic prerequisite, provide that prerequisite in the parent method context or earlier parent subtasks before invoking the child.\n"
			"8b. Apply the same rule to same-arity packaging children: if the packaging child still has a shared applicability envelope, support that envelope in the parent before the child call. For any remaining 0-ary shared prerequisite, prefer explicit parent context unless an earlier parent subtask establishes it.\n"
			"8c. If the parent is expected to supply a same-arity packaging child's shared envelope, every constructive sibling of that child must restate those assumed shared literals in method.context unless that sibling establishes them earlier internally.\n"
			"8c2. If a non-leading task argument is a reused or terminal support/base role and a unary declared task can stabilize it beyond the final producer's direct prerequisites, prefer that unary task before the packaging child or final producer and keep it as a real declared compound task instead of compressing it away.\n"
			"8c2b. For such a unary stabilizer, if the downstream child already only needs a reusable role condition that currently holds, prefer a stable/noop sibling over forcing the stabilizer headline literally true when that constructive branch would destructively rewrite the support role.\n"
			"8c3. When such a unary stabilizer is chosen, its own constructive methods must internally close the stabilizer headline and absorb remaining shared prerequisites; parent methods should not provide those internal stabilizer prerequisites merely because the stabilizer was chosen.\n"
			"8c4. If every real producer mode for an internal support literal still needs the same dynamic prerequisite on task arguments or a shared 0-ary resource, keep that requirement explicit in each constructive sibling context of the reusable task instead of re-establishing it with an extra support subtask inside every sibling.\n"
			"8d. If an extra-role producer mode needs a dynamic prerequisite that itself has a declared support task, prefer supporting that prerequisite via the declared task before the primitive step instead of leaving it only as an assumed context literal.\n"
			"8e. If that declared support task is the same task you are currently defining, recursive support is allowed: call the task on the blocker role first, then execute the primitive producer once the blocker is ready.\n"
			"8f. If a recursive or extra-role producer achieves the headline effect but leaves a transient extra-role predicate or shared resource unusable for the caller, append a real cleanup step before returning so the method finishes in a reusable stable state. Only add cleanup when its own preconditions are genuinely established by earlier steps on that same symbol.\n"
			"8g. Do not insert a support subtask that can invalidate a consumed mode-selector literal before the producer that needs it executes. Keep mode-selector relations explicit in context until the consuming producer runs.\n"
			"8g2. Before an extra-role producer mode ACTION(ARG1, AUX1, ...), only add earlier support for the dynamic prerequisites that ACTION itself actually needs. Do not insert a preparatory task on AUX1 just because AUX1 appears in the mode if that task's headline literal is not one of ACTION's unresolved dynamic prerequisites.\n"
			"8h. When a parent orders compound support tasks before another compound child, only count the earlier support tasks' own headline literals as guaranteed afterwards. If a later child still shares resource_free, handempty, or another prerequisite outside those headline literals, keep it explicit in parent context unless an earlier parent step itself headlines that same prerequisite.\n"
			"8i. Every ordered compound support step must justify itself either by contributing a later-consumed literal or by stabilizing a reused/non-leading role before a later packaging child or producer. Do not insert detours that neither support later requirements nor stabilize such a reused role.\n"
			"9. Do not use nop as filler inside constructive methods. Keep static capabilities, topology, visibility, equipment, and immutable relations in method context unless a declared task genuinely changes them.\n"
			"10. If a method has two or more subtasks, include explicit ordering edges; never rely on subtask array order alone.\n"
			"11. Use action producer alternatives or genuine already-satisfied cases to justify sibling methods; do not enumerate every support powerset. Keep the library compact: only include tasks and methods needed for the target bindings and executable support. Prefer semantic task names and reusable parameterization; do not clone grounded tasks per target literal. Do not bypass the query skeleton with a fresh helper-only library.\n"
			"12. Keep the JSON structurally simple: omit empty/default fields and unnecessary literal metadata so the response stays short and well-formed.\n\n"
		)
	if len(targets) > 4:
		examples_section = (
			"INPUT/OUTPUT EXAMPLES:\n"
			"- Input: parent(ARG1, ARG2) chooses same-arity child(ARG1, ARG2). Output pattern: parent supports the child's shared envelope, then calls child(ARG1, ARG2), and child owns the final producer.\n"
			"- Input: a method has subtasks s1 then s2. Output pattern: include ordering [[\"s1\", \"s2\"]]; do not rely on list position alone.\n"
			"- Input: child(ARG1, ARG2) reaches attach(ARG1, ARG2), mid(ARG1) has producer modes grab(ARG1) [needs free_hand, base(ARG1)] and lift(ARG1, AUX1) [needs attached(ARG1, AUX1), free_hand], and no declared task headlines mid(ARG1). Output pattern: child uses separate constructive sibling methods for the grab and lift modes instead of one narrow branch.\n"
			"- Input: move(ARG1, ARG2) uses a support mode lift(ARG1, AUX1) to make holding(ARG1), and the final producer place(ARG1, ARG2) immediately consumes holding(ARG1). Output pattern: after lift(ARG1, AUX1), continue directly to place(ARG1, ARG2); do not add cleanup on AUX1 unless earlier steps really establish carrying(AUX1) or holding(AUX1).\n"
			"- Input: move(ARG1, ARG2) will later use a producer consume(ARG1, AUX1), and consume(ARG1, AUX1) needs linked(ARG1, AUX1) as the mode selector. Output pattern: do not call a prior support task on AUX1 if that task can remove linked(ARG1, AUX1) before consume(ARG1, AUX1) executes.\n"
			"- Input: stabilize(ARG1) uses producer detach(ARG1, AUX1), and detach(ARG1, AUX1) needs linked(ARG1, AUX1) plus ready(ARG1) but does not need clear(AUX1). Output pattern: keep linked(ARG1, AUX1) explicit, support ready(ARG1) if needed, and do not insert clear_task(AUX1) before detach(ARG1, AUX1).\n"
			"- Input: parent(ARG1, ARG2) calls support(AUX1) before producer(ARG1, AUX1), but no later step needs support(AUX1)'s headline literal and AUX1 is not a reused role that must be stabilized. Output pattern: omit that detour.\n"
			"- Input: attach(ARG1, ARG2) is repeated across the query skeleton, ARG2 is a reused or terminal support/base role, and settle(ARG2) can stabilize ARG2 beyond the final producer's direct prerequisites. Output pattern: perform settle(ARG2) before the same-arity packaging child or final producer.\n"
			"- Input: settle(ARG2) headlines base(ARG2), but the later packaging child only needs clear(ARG2) and clear(ARG2) already holds. Output pattern: use a stable/noop settle(ARG2) sibling; do not destructively force base(ARG2) if that would undo earlier structure built under ARG2.\n"
			"- Input: holding(ARG1) has real producer modes grab(ARG1) [needs clear(ARG1), free_hand] and lift(ARG1, AUX1) [needs clear(ARG1), linked(ARG1, AUX1), free_hand]. Output pattern: keep clear(ARG1) and free_hand explicit in each constructive sibling context of the reusable packaging task instead of calling clear_task(ARG1) inside every sibling.\n"
			"- Input: clear_item(TARGET) is implemented with detach(BLOCKER, TARGET), and clear_item(BLOCKER) is the same declared support task. Output pattern: declare BLOCKER in method.parameters, call clear_item(BLOCKER), then execute detach(BLOCKER, TARGET).\n"
			"- Input: detach(BLOCKER, TARGET) achieves clear_item(TARGET) but leaves carrying(BLOCKER) and not free_hand. Output pattern: clear_item(BLOCKER); detach(BLOCKER, TARGET); store(BLOCKER).\n\n"
		)
		edge_cases_section = (
			"COMMON EDGE CASES:\n"
			"- Multi-step method without ordering: invalid. If there is more than one subtask, add explicit ordering edges.\n"
			"- Shared 0-ary dynamic prerequisites: keep them explicit in parent context when no earlier parent subtask establishes them.\n"
			"- Producer modes differ: keep mode-specific prerequisites inside sibling methods of the packaging child instead of exposing one mode-specific literal to the parent.\n"
			"- Support-mode handoff interrupted: if a support mode already produced the literal the final producer consumes, continue to that final producer instead of inserting unrelated cleanup or detours.\n"
			"- Consumed mode selector destroyed too early: if a later producer needs linked(ARG1, AUX1) or another mode-selector relation, do not run a prior support task on AUX1 when that task can remove the relation before the producer executes.\n"
			"- Extra-role precondition drift: if ACTION(ARG1, AUX1) only needs linked(ARG1, AUX1) and ready(ARG1), do not insert support(AUX1) whose headline is clear(AUX1) or another unrelated literal before ACTION.\n"
			"- Unjustified detour: if support(AUX1) neither feeds a later requirement nor stabilizes a reused/non-leading role, remove it.\n"
			"- Support/base role left unstable: if ARG2 is a reused or terminal support role and a unary declared task can stabilize ARG2 beyond the final producer's direct prerequisites, prefer that unary task before building on ARG2.\n"
			"- Over-eager stabilizer constructivization: if a later child already only needs clear(ARG2), ready(ARG2), or another reusable role condition that currently holds, do not force a unary stabilizer to make base(ARG2) or another headline literal true first when that would dismantle earlier progress.\n"
			"- Shared envelope rebuilt inside every sibling: if all real producer modes for an internal support literal still need clear(ARG1), free_hand, or another same requirement, keep it explicit in constructive sibling context instead of inserting the same support task inside every sibling.\n"
			"- Recursive blocker support skipped: if the same declared support task can clear the blocker role of an extra-role producer, recurse on that blocker before the primitive step.\n"
			"- Recursive branch ends without cleanup: if the extra-role producer leaves a shared resource unavailable, append a real cleanup step before returning.\n"
			"- Impossible cleanup: do not add store(AUX) or put_down(AUX) unless earlier steps really leave carrying(AUX) or holding(AUX) true on that same symbol.\n"
			"- Omitted task_args: keep the first task-arity method.parameters in the same order as the task signature.\n"
			"- Auxiliary variables: do not introduce an extra variable unless it appears in method.parameters and is constrained before use.\n"
			"- Typed role separation: operate(ACTOR, LOCATION, TARGET, TOOL, MODE) must keep ACTOR and TOOL as different variables.\n\n"
		)
	else:
		examples_section = (
			"INPUT/OUTPUT EXAMPLES:\n"
			"- Input: parent(ARG1, ARG2) still needs linked(ARG1, ARG2) and a same-arity child(ARG1, ARG2) is available. Output pattern: parent first supports the child's shared envelope, then calls child(ARG1, ARG2), and the child owns the final producer.\n"
			"- Input: a method has subtasks s1 then s2. Output pattern: include ordering [[\"s1\", \"s2\"]]; do not rely on list position alone.\n"
			"- Input: child(ARG1, ARG2) reaches attach(ARG1, ARG2), and attach needs holding(ARG1) plus ready(ARG2). Output pattern: child supports holding(ARG1) and ready(ARG2) before attach(ARG1, ARG2) instead of leaving them implicit.\n"
			"- Input: no declared task clearly covers holding(ARG1). Output pattern: either keep that support inside a same-arity packaging task via real producer modes, or add one minimal dynamic helper as a last resort.\n"
			"- Input: child(ARG1, ARG2) reaches attach(ARG1, ARG2), holding(ARG1) has producer modes grab(ARG1) [needs free_hand, base(ARG1)] and lift(ARG1, AUX1) [needs attached(ARG1, AUX1), free_hand], and no declared task directly headlines holding(ARG1). Output pattern: child has separate constructive sibling methods for the grab and lift modes instead of one generic branch that assumes only base(ARG1).\n"
			"- Input: a same-arity child still needs a shared 0-ary literal like resource_free. Output pattern: place resource_free explicitly in the parent context unless an earlier parent subtask establishes it.\n"
			"- Input: support(ARG1) and support(ARG2) run before same-arity child(ARG1, ARG2), and the child still shares resource_free. Output pattern: keep resource_free explicit in the parent context unless an earlier parent step itself headlines resource_free; do not rely on incidental side effects of support(ARG1) or support(ARG2).\n"
			"- Input: parent supplies ready(ARG2) before child(ARG1, ARG2). Output pattern: every constructive child sibling either keeps ready(ARG2) in method.context or establishes it internally before the final producer.\n"
			"- Input: free(ARG2) can be produced by release(ARG2) [needs holding(ARG2)] or detach(AUX1, ARG2) [needs attached(AUX1, ARG2), ready(AUX1), free_hand]. Output pattern: prefer release(ARG2) when suitable; if detach(AUX1, ARG2) is used, declare AUX1 and support ready(AUX1) explicitly.\n"
			"- Input: free(ARG2) is produced by detach(AUX1, ARG2), and ready(AUX1) itself has a declared support task prepare(AUX1). Output pattern: support ready(AUX1) via prepare(AUX1) before detach(AUX1, ARG2) instead of only assuming ready(AUX1) in context.\n"
			"- Input: attach(ARG1, ARG2) is repeated across the query skeleton, ARG2 is a reused or terminal support/base role, and settle(ARG2) can stabilize ARG2 beyond the final producer's direct prerequisites. Output pattern: perform settle(ARG2) before the same-arity packaging child or final producer.\n"
			"- Input: settle(ARG2) headlines base(ARG2), but the later packaging child only needs clear(ARG2) and clear(ARG2) already holds. Output pattern: use a stable/noop settle(ARG2) sibling; do not destructively force base(ARG2) if that would undo earlier structure built under ARG2.\n"
			"- Input: holding(ARG1) has real producer modes grab(ARG1) [needs clear(ARG1), free_hand] and lift(ARG1, AUX1) [needs clear(ARG1), linked(ARG1, AUX1), free_hand]. Output pattern: keep clear(ARG1) and free_hand explicit in each constructive sibling context of the reusable packaging task instead of calling clear_task(ARG1) inside every sibling.\n"
			"- Input: clear_item(ARG1) is implemented with detach(AUX1, ARG1), detach(AUX1, ARG1) needs clear_item(AUX1), and clear_item is the same declared support task. Output pattern: a recursive constructive method first calls clear_item(AUX1), then executes detach(AUX1, ARG1).\n"
			"- Input: same recursive-support case as above. Output snippet: {\"method_name\":\"m_clear_item_recursive\",\"task_name\":\"clear_item\",\"parameters\":[\"ARG1\",\"AUX1\"],\"context\":[{\"predicate\":\"attached\",\"args\":[\"AUX1\",\"ARG1\"],\"is_positive\":true},{\"predicate\":\"free_hand\",\"args\":[],\"is_positive\":true}],\"subtasks\":[{\"step_id\":\"s1\",\"task_name\":\"clear_item\",\"args\":[\"AUX1\"],\"kind\":\"compound\"},{\"step_id\":\"s2\",\"task_name\":\"detach\",\"args\":[\"AUX1\",\"ARG1\"],\"kind\":\"primitive\"}],\"ordering\":[[\"s1\",\"s2\"]]}.\n"
			"- Input: detach(AUX1, ARG1) makes clear_item(ARG1) but leaves carrying(AUX1) and not free_hand. Output pattern: clear_item(AUX1); detach(AUX1, ARG1); store(AUX1), so the method returns with free_hand restored.\n"
			"- Input: detach(ARG1, AUX1) needs linked(ARG1, AUX1) as the mode selector. Output pattern: keep linked(ARG1, AUX1) in context until detach(ARG1, AUX1); do not call a support task on AUX1 first if it could remove linked(ARG1, AUX1).\n"
			"- Input: stabilize(ARG1) uses producer detach(ARG1, AUX1), and detach(ARG1, AUX1) needs linked(ARG1, AUX1) plus ready(ARG1) but does not need clear(AUX1). Output pattern: keep linked(ARG1, AUX1) explicit, support ready(ARG1) if needed, and do not insert clear_task(AUX1) before detach(ARG1, AUX1).\n"
			"- Input: parent(ARG1, ARG2) calls support(AUX1) before producer(ARG1, AUX1), but no later step needs support(AUX1)'s headline literal and AUX1 is not a reused role that must be stabilized. Output pattern: omit that detour.\n"
			"- Input: detach(AUX1, ARG2) is used to achieve clear(ARG2). Output pattern: AUX1 appears in method.parameters and is constrained in method.context before the detach step.\n\n"
		)
		edge_cases_section = (
			"COMMON EDGE CASES:\n"
			"- Same-arity packaging chosen: stop planning from the parent task's direct headline producer and switch to supporting the packaging child's shared envelope plus the child call.\n"
			"- Multi-step method without ordering: invalid. If there is more than one subtask, add explicit ordering edges.\n"
			"- Shared 0-ary dynamic prerequisites: keep them explicit in parent context when no earlier parent subtask establishes them.\n"
			"- Earlier compound support subtasks only guarantee their own headline literals: if support(ARG1) and support(ARG2) run before child(ARG1, ARG2), do not assume they also preserve resource_free unless one of those earlier parent steps itself headlines resource_free.\n"
			"- Producer modes differ: keep mode-specific prerequisites inside sibling methods of the packaging child instead of exposing one mode-specific literal to the parent.\n"
			"- Parent-supported child envelope silently omitted: if the caller is expected to provide clear(ARG1), ready(ARG2), or another shared prerequisite, keep that literal explicit in each constructive child context unless the child establishes it internally.\n"
			"- Single narrow mode mistaken for a generic task: if one internal producer mode needs base(ARG1) and another needs attached(ARG1, AUX1), do not keep only the base(ARG1) branch unless the task is intentionally partial.\n"
			"- Complex extra-role mode chosen when a simpler mode exists: if release(ARG2) and detach(AUX1, ARG2) both achieve free(ARG2), do not choose detach(AUX1, ARG2) unless the method really needs that extra-role mode and its extra support.\n"
			"- Support/base role left unstable: if ARG2 is a reused or terminal support role and a unary declared task can stabilize ARG2 beyond the final producer's direct prerequisites, prefer that unary task before building on ARG2.\n"
			"- Over-eager stabilizer constructivization: if a later child already only needs clear(ARG2), ready(ARG2), or another reusable role condition that currently holds, do not force a unary stabilizer to make base(ARG2) or another headline literal true first when that would dismantle earlier progress.\n"
			"- Shared envelope rebuilt inside every sibling: if all real producer modes for an internal support literal still need clear(ARG1), free_hand, or another same requirement, keep it explicit in constructive sibling context instead of inserting the same support task inside every sibling.\n"
			"- Extra-role prerequisite with a declared support task left implicit: if ready(AUX1) has a task prepare(AUX1), prefer prepare(AUX1) before detach(AUX1, ARG2) rather than assuming ready(AUX1) without support.\n"
			"- Recursive blocker support skipped: if clear_item(AUX1) is available for the blocker of detach(AUX1, ARG1), do not leave clear_item(AUX1) as an unsupported assumption when the task is meant to be generic.\n"
			"- Recursive branch ends without cleanup: if detach(AUX1, ARG1) leaves carrying(AUX1) or consumes free_hand, do not return immediately when a real cleanup step can restore the shared resource.\n"
			"- Impossible cleanup: do not add store(AUX1) or put_down(AUX1) unless earlier steps really leave carrying(AUX1) or holding(AUX1) true on that same symbol.\n"
			"- Consumed mode selector destroyed too early: if detach(ARG1, AUX1) needs linked(ARG1, AUX1), do not run a support task on AUX1 first when that task can remove linked(ARG1, AUX1) before detach executes.\n"
			"- Extra-role precondition drift: if ACTION(ARG1, AUX1) only needs linked(ARG1, AUX1) and ready(ARG1), do not insert support(AUX1) whose headline is clear(AUX1) or another unrelated literal before ACTION.\n"
			"- Unjustified detour: if support(AUX1) neither feeds a later requirement nor stabilizes a reused/non-leading role, remove it.\n"
			"- Auxiliary variables: do not introduce an extra variable unless it appears in method.parameters and is constrained before use.\n"
			"- Typed role separation: operate(ACTOR, LOCATION, TARGET, TOOL, MODE) must keep ACTOR and TOOL as different variables.\n"
			"- Declared support task exists: prefer it over raw primitive producer modes for the same dynamic predicate.\n"
			"- Negative or already-satisfied cases: keep them as separate noop/already-satisfied branches instead of mixing them into constructive fallbacks.\n\n"
		)

	return (
		"TASK:\n"
		"Generate one compact but executable JSON HTN library that compiles into valid AgentSpeak.\n\n"
		f"QUERY:\n{query_text or '- none provided'}\n\n"
		f"{semantic_object_section}"
		f"{query_object_inventory_section}"
		f"{query_object_section}"
		f"ORDERED QUERY TASK ANCHORS:\n{query_anchor_lines}\n\n"
		f"ORDERED TARGET/TASK SKELETON HINTS:\n{anchor_binding_lines}\n\n"
		f"{query_priority_section}"
		f"{query_task_parameter_alignment_section}"
		f"QUERY-TASK SUPPORT OBLIGATIONS:\n{query_task_support_lines}\n\n"
		f"QUERY-TASK SUPPORT PRODUCERS:\n{query_task_support_producer_lines}\n\n"
		f"{child_support_section}"
		f"{same_arity_child_support_section}"
		f"{same_arity_child_context_section}"
		f"{zero_ary_parent_context_section}"
		f"{packaging_skeleton_section}"
		f"{role_frame_section}"
		f"DOMAIN:\n{domain.name}\n\n"
		f"DOMAIN TYPES:\n{chr(10).join(type_lines) if type_lines else '- object'}\n\n"
		f"DECLARED DOMAIN TASKS:\n{chr(10).join(task_lines) if task_lines else '- none declared'}\n\n"
		f"PREDICATES:\n{chr(10).join(predicate_lines)}\n\n"
		f"DYNAMIC PREDICATES (changed by action effects):\n{dynamic_predicate_lines}\n\n"
		f"STATIC PREDICATES (never changed by action effects; use as context only):\n{static_predicate_lines}\n\n"
		f"PRODUCER ACTIONS BY DYNAMIC PREDICATE:\n{producer_lines}\n\n"
		f"PRODUCER EFFECT PATTERNS (use these to design constructive methods):\n{producer_pattern_lines}\n\n"
		f"ARGUMENT-ALIGNED PRODUCER TEMPLATES:\n{aligned_producer_lines}\n\n"
		f"SHARED PRODUCER DYNAMIC PREREQUISITES:\n{shared_producer_requirement_lines}\n\n"
		f"{declared_task_shared_prerequisite_section}"
		f"{producer_consumer_template_section}"
		f"{reusable_dynamic_resource_section}"
		f"DYNAMIC PRECONDITION SUPPORT HINTS:\n{dynamic_support_lines}\n\n"
		f"QUERY-TASK SAME-ARITY SUPPORT CANDIDATES:\n{same_arity_task_lines}\n\n"
		f"QUERY-TASK SAME-ARITY PACKAGING HINTS:\n{query_task_same_arity_packaging_lines}\n\n"
		f"{same_arity_transitive_section}"
		f"DECLARED TASK CONSTRUCTIVE TEMPLATES:\n{declared_task_template_lines}\n\n"
		f"{relevant_support_task_section}"
		f"{relevant_support_task_alignment_section}"
		f"{declared_support_task_applicability_section}"
		f"{relevant_support_task_recursive_mode_section}"
		f"{relevant_support_task_recursive_template_section}"
		f"{relevant_support_task_cleanup_template_section}"
		f"{relevant_support_task_internal_obligation_section}"
		f"{role_stabilization_section}"
		f"{role_stabilizer_support_section}"
		f"RUNTIME PRIMITIVE ACTION ALIASES:\n{chr(10).join(action_lines)}\n\n"
		f"{branch_hint_section}"
		f"ORDERED TARGET LITERALS:\n{target_lines}\n\n"
		f"REQUIRED target_task_bindings ENTRIES:\n{binding_hints}\n\n"
		f"{step_instructions_section}"
		f"{examples_section}"
		f"{edge_cases_section}"
		"TOP-LEVEL JSON SHAPE:\n"
		"Only define target_task_bindings, compound_tasks, and methods; primitive tasks are injected automatically.\n"
		f"{schema_hint}\n\n"
		"FINAL CHECKLIST:\n"
		"- Each target literal has exactly one binding, and query-mentioned declared tasks appear in the library when provided.\n"
		"- Every target-bound task includes an already-satisfied/noop method with the headline literal in context; support/stabilizer tasks may use a downstream role-stable condition for their noop branch when appropriate.\n"
		"- Compactness never removes required structure: keep source_predicates plus exact headline on compound tasks, kind on every subtask, action_name on primitive subtasks, and explicit ordering edges on every multi-step method.\n"
		"- If a query-task packaging skeleton or role-stabilization hint names a declared unary stabilizer, keep that stabilizer in compound_tasks with methods instead of omitting it for compactness.\n"
		"- Primitive step literal metadata is usually omitted; if present, it must name an exact positive real action effect. Every referenced compound subtask appears in compound_tasks and has methods.\n"
		"- Fresh helper tasks, if any, correspond only to dynamic predicates. Static predicates appear only as context/preconditions, not helper-task headlines.\n"
		f"{query_task_alignment_checklist_line}"
		"- Every primitive step's dynamic preconditions are supported by method context or earlier subtasks. Every compound step's shared dynamic prerequisites are supported by method context or earlier subtasks. No primitive step relies on an unstated dynamic precondition.\n"
		"- No free variables. No undefined compound subtasks. Every auxiliary variable has an explicit typing witness.\n"
		"- Omit empty/default fields when possible so the JSON stays compact and less error-prone.\n"
		"- Return one complete JSON object and nothing else.\n"
	)


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
	seen: set[str] = set()
	ordered: list[str] = []
	for value in values:
		text = str(value).strip()
		if not text or text in seen:
			continue
		seen.add(text)
		ordered.append(text)
	return ordered


def _task_headline_candidate_map(
	domain: Any,
	action_analysis: Dict[str, Any],
) -> Dict[str, list[str]]:
	mapping: Dict[str, list[str]] = {}
	for task in getattr(domain, "tasks", []):
		task_name = _sanitize_name(str(task.name))
		source_predicates = [
			str(predicate_name).strip()
			for predicate_name in (getattr(task, "source_predicates", ()) or ())
			if str(predicate_name).strip()
		]
		candidates = source_predicates or _candidate_headline_predicates_for_task(
			str(task.name),
			len(getattr(task, "parameters", ()) or ()),
			action_analysis,
		)
		mapping[task_name] = _unique_preserve_order(candidates)[:3]
	return mapping


def _group_contract_lines_by_task(
	lines: Sequence[str],
	task_names: Sequence[str],
) -> Dict[str, list[str]]:
	grouped = {
		task_name: []
		for task_name in _unique_preserve_order(task_names)
	}
	for line in lines:
		text = str(line).strip()
		if not text:
			continue
		for task_name in grouped:
			if text.startswith(f"- {task_name}("):
				payload = text[2:].strip()
				if payload not in grouped[task_name]:
					grouped[task_name].append(payload)
				break
	return {
		task_name: entries
		for task_name, entries in grouped.items()
		if entries
	}


def _build_query_task_contract_payloads(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[Dict[str, Any]]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	query_task_names = [
		_anchor_display_name(anchor)
		for anchor in query_task_anchors
		if _anchor_display_name(anchor)
	]
	if not query_task_names:
		return []

	line_sources = [
		_query_task_noop_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_role_stabilization_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_support_obligation_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_support_producer_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_child_support_requirement_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_same_arity_packaging_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_same_arity_child_support_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_packaging_shared_requirement_support_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_role_frame_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_packaging_skeleton_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_declared_task_producer_template_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
	]
	grouped: Dict[str, list[str]] = {}
	for lines in line_sources:
		for task_name, entries in _group_contract_lines_by_task(lines, query_task_names).items():
			grouped.setdefault(task_name, [])
			for entry in entries:
				if entry not in grouped[task_name]:
					grouped[task_name].append(entry)

	task_schemas = _declared_task_schema_map(domain)
	payloads: list[Dict[str, Any]] = []
	seen_query_tasks: set[str] = set()
	for index, (target_literal, anchor) in enumerate(zip(target_literals, query_task_anchors), start=1):
		display_name = _anchor_display_name(anchor)
		task_name = str(anchor.get("task_name", "")).strip()
		if not display_name or display_name in seen_query_tasks:
			continue
		seen_query_tasks.add(display_name)
		task_schema = task_schemas.get(task_name)
		task_signature = (
			task_schema.to_signature()
			if task_schema is not None and hasattr(task_schema, "to_signature")
			else _task_invocation_signature(
				display_name,
				tuple(str(arg) for arg in anchor.get("args", ())),
			)
		)
		payloads.append(
			{
				"task_name": task_name,
				"display_name": display_name,
				"ordered_binding": {
					"index": index,
					"target_literal": str(target_literal),
					"task_signature": _task_invocation_signature(
						display_name,
						tuple(str(arg) for arg in anchor.get("args", ())),
					),
				},
				"task_signature": task_signature,
				"contract_lines": grouped.get(display_name, []),
			},
		)
	return payloads


def _query_task_noop_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		display_name = _anchor_display_name(anchor)
		task_name = str(anchor.get("task_name", "")).strip()
		task_schema = task_schemas.get(task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if not display_name or task_schema is None or parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue
		task_parameters = tuple(
			_parameter_token(parameter)
			for parameter in getattr(task_schema, "parameters", ()) or ()
		)
		if len(task_parameters) != len(target_args):
			continue
		target_mapping = {
			task_parameter: f"ARG{index}"
			for index, task_parameter in enumerate(task_parameters, start=1)
		}
		rendered_headline = _render_signature_with_mapping(
			f"{predicate_name}({', '.join(task_parameters)})",
			target_mapping,
		)
		line = (
			f"- {_task_invocation_signature(task_name, task_parameters)}: required noop branch "
			f"precondition/context {rendered_headline}. "
			"Include noop explicitly in the task entry; tasks missing noop are invalid."
		)
		if line in seen:
			continue
		seen.add(line)
		lines.append(line)
	return lines


def _query_task_packaging_shared_requirement_support_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	required_helper_lookup = {
		(
			str(spec.get("query_task_name", "")).strip(),
			str(spec.get("precondition_signature", "")).strip(),
		): spec
		for spec in _required_helper_specs_for_query_targets(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		)
	}
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		force_internal_contract = bool(anchor.get("force_internal_contract"))
		task_schema = task_schemas.get(task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if task_schema is None or parsed_target is None:
			continue
		if force_internal_contract:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		task_signature_parameters = tuple(
			_parameter_token(parameter)
			for parameter in getattr(task_schema, "parameters", ()) or ()
		)
		task_args = tuple(f"ARG{index}" for index, _ in enumerate(target_args, start=1))
		candidates = _same_arity_packaging_candidates_for_query_task(
			domain,
			task_name,
			predicate_name,
			task_signature_parameters,
			task_schemas,
			action_analysis,
		)
		for candidate in candidates:
			candidate_name = str(candidate.get("candidate", "")).strip()
			candidate_parameters = tuple(candidate.get("parameters", ()))
			candidate_mapping = {
				_parameter_token(raw_parameter): task_arg
				for raw_parameter, task_arg in zip(candidate_parameters, task_args)
			}
			for raw_requirement in candidate.get("shared_requirements", ()):
				rendered_requirement = _render_signature_with_mapping(
					str(raw_requirement),
					candidate_mapping,
				)
				parsed_requirement = _parse_literal_signature(rendered_requirement)
				if parsed_requirement is None:
					continue
				requirement_predicate, requirement_args, requirement_positive = parsed_requirement
				if not requirement_positive:
					continue
				if (
					required_helper_lookup.get((task_name, rendered_requirement))
					is not None
				):
					continue
				support_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
					requirement_predicate,
					[],
				)
				rendered_support_options: list[str] = []
				mode_obligation_lines: list[str] = []
				for support_pattern in support_patterns[:3]:
					support_effect_args = list(support_pattern.get("effect_args") or [])
					if len(support_effect_args) != len(requirement_args):
						continue
					support_mapping = {
						token: arg
						for token, arg in zip(support_effect_args, requirement_args)
					}
					rendered_action_args = _extend_mapping_with_action_parameters(
						support_mapping,
						support_pattern.get("action_parameters") or [],
						action_parameter_types=support_pattern.get("action_parameter_types")
						or [],
					)
					support_preconditions = list(
						_render_positive_dynamic_requirements(
							support_pattern,
							support_mapping,
						),
					)
					support_suffix = (
						f" [needs {', '.join(support_preconditions)}]"
						if support_preconditions
						else ""
					)
					rendered_support_options.append(
						f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}"
						f"{support_suffix}"
					)
					extra_roles = [
						arg
						for arg in rendered_action_args
						if arg not in requirement_args
					]
					if support_preconditions and not extra_roles:
						mode_obligation_lines.append(
							f"- {_task_invocation_signature(display_name, task_args)}: if the parent uses "
							f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)} "
							f"before {_task_invocation_signature(candidate_name, task_args)} to establish "
							f"{requirement_predicate}({', '.join(requirement_args)}), support "
							f"{', '.join(support_preconditions)} before that step."
						)
				if not rendered_support_options:
					continue
				line = (
					f"- {_task_invocation_signature(display_name, task_args)}: "
					f"{_task_invocation_signature(candidate_name, task_args)} expects caller-shared "
					f"{requirement_predicate}({', '.join(requirement_args)}). Before the child call, "
					f"establish {requirement_predicate}({', '.join(requirement_args)}) via "
					f"{' or '.join(rendered_support_options)}."
				)
				if line in seen:
					continue
				seen.add(line)
				lines.append(line)
				for mode_line in mode_obligation_lines:
					if mode_line in seen:
						continue
					seen.add(mode_line)
					lines.append(mode_line)
	return lines


def _support_task_summary_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	if len(query_task_anchors) != len(target_literals):
		return lines

	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		task_schema = task_schemas.get(task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if task_schema is None or parsed_target is None:
			continue
		predicate_name, _, is_positive = parsed_target
		if not is_positive:
			continue
		parent_parameters = tuple(
			_parameter_token(parameter)
			for parameter in task_schema.parameters
		)
		candidates = _same_arity_packaging_candidates_for_query_task(
			domain,
			task_name,
			predicate_name,
			parent_parameters,
			task_schemas,
			action_analysis,
		)
		if not candidates:
			continue
		for candidate in candidates:
			candidate_name = str(candidate.get("candidate", "")).strip()
			child_parameters = tuple(candidate.get("parameters", ()))
			shared_requirements = _same_arity_caller_shared_requirements(
				domain,
				predicate_name,
				parent_parameters,
				action_analysis,
				candidate,
			)
			requirement_text = ", ".join(shared_requirements) if shared_requirements else "none"
			line = (
				f"- {_task_invocation_signature(candidate_name, child_parameters)}: exact same-arity "
				f"packaging child for {predicate_name}({', '.join(parent_parameters)}) when called by "
				f"{_task_invocation_signature(display_name, parent_parameters)}. Parent-side "
				f"caller-shared prerequisites: {requirement_text}. This set is exhaustive at "
				f"child entry. This child must own the final "
				f"producer for {predicate_name}({', '.join(parent_parameters)})."
			)
			if line not in seen:
				seen.add(line)
				lines.append(line)
	return lines


def _support_task_noop_line(
	domain: Any,
	task_name: str,
	task_schema: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> Optional[str]:
	task_parameters = tuple(
		_parameter_token(parameter)
		for parameter in getattr(task_schema, "parameters", ()) or ()
	)
	if not task_parameters:
		return None
	role_labels = tuple(
		f"ARG{index}"
		for index, _ in enumerate(task_parameters, start=1)
	)
	headline_candidates = _candidate_headline_predicates_for_task(
		task_name,
		len(task_parameters),
		action_analysis,
	)
	if not headline_candidates:
		return None
	rendered_headline = (
		f"{headline_candidates[0]}("
		f"{', '.join(_aligned_task_parameter_sequence_for_predicate(task_schema, headline_candidates[0], action_analysis, role_labels))}"
		f")"
	)
	preferred_stable_condition_sets = _support_task_preferred_stable_noop_conditions(
		domain,
		task_name,
		task_schema,
		target_literals,
		query_task_anchors,
		action_analysis,
	)
	if preferred_stable_condition_sets:
		rendered_condition_sets = [
			"; ".join(condition_set)
			for condition_set in preferred_stable_condition_sets
			if condition_set
		]
		if rendered_condition_sets:
			stable_condition_text = " or ".join(rendered_condition_sets)
			return (
				f"{_task_invocation_signature(task_name, task_parameters)}: when used as a "
				f"role stabilizer, required stable/noop branch precondition/context "
				f"{stable_condition_text} instead of requiring {rendered_headline}. "
				f"If that already makes the downstream role reusable, that branch must "
				f"have no subtasks. If {rendered_headline} itself already holds, a noop "
				"branch is also valid."
			)
	return (
		f"{_task_invocation_signature(task_name, task_parameters)}: required "
		f"stable/noop branch precondition/context {rendered_headline}. "
		"If this already holds, that branch must have no subtasks; do not force "
		"a constructive branch."
	)


def _support_task_preferred_stable_noop_conditions(
	domain: Any,
	task_name: str,
	task_schema: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> tuple[tuple[str, ...], ...]:
	task_parameters = tuple(
		_parameter_token(parameter)
		for parameter in getattr(task_schema, "parameters", ()) or ()
	)
	if len(task_parameters) != 1:
		return ()

	task_headline_candidates = _candidate_headline_predicates_for_task(
		task_name,
		len(task_parameters),
		action_analysis,
	)
	if not task_headline_candidates:
		return ()

	task_schemas = _declared_task_schema_map(domain)
	role_type = _parameter_type(task_schema.parameters[0])
	anchors_by_task_name: dict[str, list[Dict[str, Any]]] = {}
	for anchor in query_task_anchors:
		anchor_name = str(anchor.get("task_name", "")).strip()
		if anchor_name:
			anchors_by_task_name.setdefault(anchor_name, []).append(anchor)

	stable_condition_sets: list[tuple[str, ...]] = []
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		query_task_name = str(anchor.get("task_name", "")).strip()
		query_task_schema = task_schemas.get(query_task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if query_task_schema is None or parsed_target is None:
			continue
		_, _, is_positive = parsed_target
		if not is_positive:
			continue

		query_task_parameters = tuple(
			_parameter_token(parameter)
			for parameter in getattr(query_task_schema, "parameters", ()) or ()
		)
		if len(query_task_parameters) < 2:
			continue

		bound_occurrences = [
			tuple(str(arg) for arg in occurrence.get("args", ()))
			for occurrence in anchors_by_task_name.get(query_task_name, [])
			if len(tuple(occurrence.get("args", ()))) == len(query_task_parameters)
		]
		if len(bound_occurrences) < 2:
			continue

		role_labels = tuple(
			f"ARG{index}"
			for index, _ in enumerate(query_task_parameters, start=1)
		)
		direct_role_requirements: Dict[int, set[str]] = {
			index: set()
			for index in range(1, len(query_task_parameters))
		}
		predicate_name = parsed_target[0]
		for headline_pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		):
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(query_task_parameters):
				continue
			token_mapping = {
				token: role_label
				for token, role_label in zip(effect_args, role_labels)
			}
			for precondition_signature in (
				headline_pattern.get("dynamic_precondition_signatures") or []
			):
				rendered_signature = _render_signature_with_mapping(
					precondition_signature,
					token_mapping,
				)
				parsed_precondition = _parse_literal_signature(rendered_signature)
				if parsed_precondition is None:
					continue
				_, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive or len(precondition_args) != 1:
					continue
				for role_index in range(1, len(query_task_parameters)):
					if precondition_args[0] == role_labels[role_index]:
						direct_role_requirements[role_index].add(rendered_signature)

		for role_index in range(1, len(query_task_parameters)):
			if _parameter_type(query_task_schema.parameters[role_index]) != role_type:
				continue
			role_label = role_labels[role_index]
			for candidate_predicate in task_headline_candidates:
				candidate_signature = f"{candidate_predicate}({role_label})"
				if candidate_signature in direct_role_requirements[role_index]:
					continue
				if (
					_constructive_template_summary_for_task(
						task_name,
						(role_label,),
						candidate_predicate,
						action_analysis,
					)
					is None
				):
					continue
				downstream_reusable_conditions = []
				for requirement in sorted(direct_role_requirements[role_index]):
					if requirement == candidate_signature:
						continue
					parsed_requirement = _parse_literal_signature(requirement)
					if parsed_requirement is None:
						continue
					req_predicate, _, req_positive = parsed_requirement
					if not req_positive:
						continue
					downstream_reusable_conditions.append(f"{req_predicate}(ARG1)")
				if not downstream_reusable_conditions:
					continue
				condition_set = tuple(dict.fromkeys(downstream_reusable_conditions))
				if condition_set not in stable_condition_sets:
					stable_condition_sets.append(condition_set)

	return tuple(stable_condition_sets)


def _required_helper_specs_for_query_targets(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[Dict[str, Any]]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	specs: list[Dict[str, Any]] = []
	seen_keys: set[tuple[str, str]] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		query_task_name = str(anchor.get("task_name", "")).strip()
		task_schema = task_schemas.get(query_task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if task_schema is None or parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		task_signature_parameters = tuple(
			_parameter_token(parameter)
			for parameter in getattr(task_schema, "parameters", ()) or ()
		)
		task_args = tuple(f"ARG{index}" for index in range(1, len(target_args) + 1))
		same_arity_candidates = _same_arity_packaging_candidates_for_query_task(
			domain,
			query_task_name,
			predicate_name,
			task_signature_parameters or task_args,
			task_schemas,
			action_analysis,
		)
		if not same_arity_candidates:
			continue

		packaging_shared_requirements = {
			str(requirement).strip()
			for candidate in same_arity_candidates
			for requirement in _same_arity_caller_shared_requirements(
				domain,
				predicate_name,
				task_args,
				action_analysis,
				candidate,
			)
			if str(requirement).strip()
		}
		if len(packaging_shared_requirements) <= 1:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		)
		for headline_pattern in headline_patterns:
			effect_args = list(headline_pattern.get("effect_args") or [])
			if len(effect_args) != len(target_args):
				continue
			headline_mapping = {
				token: task_arg
				for token, task_arg in zip(effect_args, task_args)
			}
			for precondition_signature in headline_pattern.get(
				"dynamic_precondition_signatures",
			) or []:
				rendered_precondition_signature = _render_signature_with_mapping(
					precondition_signature,
					headline_mapping,
				)
				if rendered_precondition_signature not in packaging_shared_requirements:
					continue
				parsed_precondition = _parse_literal_signature(
					rendered_precondition_signature,
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = (
					parsed_precondition
				)
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(task_args):
					continue

				candidate_declared_tasks = [
					candidate
					for candidate in _candidate_support_task_names(
						domain,
						precondition_predicate,
						precondition_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							precondition_predicate,
							[],
						),
					)
					if len(getattr(task_schemas.get(candidate), "parameters", ()))
					== len(precondition_args)
				]
				if candidate_declared_tasks:
					continue

				support_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
					precondition_predicate,
					[],
				)
				if len(support_patterns) <= 1:
					continue

				helper_support_calls: list[str] = []
				for sibling_requirement in packaging_shared_requirements:
					if sibling_requirement == rendered_precondition_signature:
						continue
					parsed_sibling_requirement = _parse_literal_signature(
						sibling_requirement,
					)
					if parsed_sibling_requirement is None:
						continue
					sibling_predicate, sibling_args, sibling_positive = (
						parsed_sibling_requirement
					)
					if not sibling_positive or not sibling_args:
						continue
					sibling_support_tasks = [
						support_task
						for support_task in _candidate_support_task_names(
							domain,
							sibling_predicate,
							sibling_args,
							action_analysis.get("producer_actions_by_predicate", {}).get(
								sibling_predicate,
								[],
							),
						)
						if support_task != query_task_name
						and (
							len(getattr(task_schemas.get(support_task), "parameters", ()))
							== len(sibling_args)
						)
					]
					if sibling_support_tasks:
						helper_support_calls.append(
							_task_invocation_signature(
								sibling_support_tasks[0],
								sibling_args,
							),
						)
				if not helper_support_calls:
					continue

				pattern_rows: list[Dict[str, Any]] = []
				common_requirements: Optional[set[str]] = None
				has_extra_role_support_pattern = False
				for support_pattern in support_patterns[:3]:
					support_effect_args = list(support_pattern.get("effect_args") or [])
					if len(support_effect_args) != len(precondition_args):
						continue
					support_mapping = {
						token: arg
						for token, arg in zip(support_effect_args, precondition_args)
					}
					rendered_support_action_args = _extend_mapping_with_action_parameters(
						support_mapping,
						support_pattern.get("action_parameters") or [],
						action_parameter_types=support_pattern.get("action_parameter_types")
						or [],
					)
					support_preconditions = list(
						_render_positive_dynamic_requirements(
							support_pattern,
							support_mapping,
						),
					)
					extra_roles = [
						arg
						for arg in rendered_support_action_args
						if arg not in precondition_args
					]
					has_extra_role_support_pattern = (
						has_extra_role_support_pattern or bool(extra_roles)
					)
					pattern_rows.append(
						{
							"mode_call": _task_invocation_signature(
								support_pattern["action_name"],
								rendered_support_action_args,
							),
							"mode_requirements": support_preconditions,
							"has_extra_role": bool(extra_roles),
						},
					)
					if common_requirements is None:
						common_requirements = set(support_preconditions)
					else:
						common_requirements &= set(support_preconditions)

				if not has_extra_role_support_pattern or len(pattern_rows) <= 1:
					continue

				helper_task_name = _sanitize_name(f"do_{precondition_predicate}")
				helper_parameters = _generic_parameter_symbols(len(precondition_args))
				helper_signature = _task_invocation_signature(
					helper_task_name,
					helper_parameters,
				)
				common_requirements = common_requirements or set()
				common_support_calls: list[str] = []
				common_context_requirements: list[str] = []
				for requirement in common_requirements:
					parsed_requirement = _parse_literal_signature(requirement)
					if parsed_requirement is None:
						continue
					requirement_predicate, requirement_args, requirement_positive = (
						parsed_requirement
					)
					if not requirement_positive:
						continue
					support_tasks = [
						support_task
						for support_task in _candidate_support_task_names(
							domain,
							requirement_predicate,
							requirement_args,
							action_analysis.get("producer_actions_by_predicate", {}).get(
								requirement_predicate,
								[],
							),
						)
						if len(getattr(task_schemas.get(support_task), "parameters", ()))
						== len(requirement_args)
					]
					if support_tasks:
						common_support_calls.append(
							_task_invocation_signature(support_tasks[0], requirement_args),
						)
					else:
						common_context_requirements.append(requirement)

				spec_key = (query_task_name, rendered_precondition_signature)
				if spec_key in seen_keys:
					continue
				seen_keys.add(spec_key)
				specs.append(
					{
						"query_task_name": query_task_name,
						"precondition_signature": rendered_precondition_signature,
						"precondition_predicate": precondition_predicate,
						"precondition_args": list(precondition_args),
						"helper_task_name": helper_task_name,
						"helper_parameters": list(helper_parameters),
						"helper_signature": helper_signature,
						"helper_target_args": list(precondition_args),
						"packaging_task_name": str(
							same_arity_candidates[0].get("candidate", "")
						).strip(),
						"packaging_call": _task_invocation_signature(
							str(same_arity_candidates[0].get("candidate", "")).strip(),
							task_args,
						),
						"helper_support_calls": helper_support_calls,
						"common_requirements": sorted(common_requirements),
						"common_support_calls": common_support_calls,
						"common_context_requirements": common_context_requirements,
						"pattern_rows": pattern_rows,
					},
				)

	return specs


def _build_required_helper_task_contract_payloads(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[Dict[str, Any]]:
	payloads_by_name: dict[str, dict[str, Any]] = {}
	for helper_spec in _required_helper_specs_for_query_targets(
		domain,
		target_literals,
		query_task_anchors,
		action_analysis,
	):
		helper_task_name = str(helper_spec.get("helper_task_name", "")).strip()
		if not helper_task_name:
			continue
		precondition_predicate = str(helper_spec.get("precondition_predicate", "")).strip()
		helper_parameters = tuple(helper_spec.get("helper_parameters") or ())
		helper_arg_tokens = tuple(
			f"ARG{index}" for index in range(1, len(helper_parameters) + 1)
		)
		helper_signature = str(helper_spec.get("helper_signature", "")).strip()
		common_requirements = list(helper_spec.get("common_requirements") or [])
		common_support_calls = list(helper_spec.get("common_support_calls") or [])
		common_context_requirements = list(
			helper_spec.get("common_context_requirements") or []
		)

		contract_lines: list[str] = [
			f"{helper_signature}: required stable/noop branch precondition/context "
			f"{precondition_predicate}({', '.join(helper_arg_tokens)}). If this already "
			"holds, that branch must have no subtasks; do not force a constructive branch.",
		]
		template_line = _constructive_template_line_for_task(
			helper_task_name,
			helper_parameters,
			precondition_predicate,
			action_analysis,
		)
		if template_line is not None:
			contract_lines.append(template_line.removeprefix("- "))
		if common_support_calls:
			contract_lines.append(
				f"{helper_signature}: because all listed producer templates still need "
				f"{', '.join(common_requirements)}, support those reusable needs via "
				f"{'; '.join(common_support_calls)} inside this helper before the final "
				"producer instead of pushing them back to the parent task.",
			)
		for pattern_row in helper_spec.get("pattern_rows") or []:
			mode_call = str(pattern_row.get("mode_call", "")).strip()
			mode_requirements = list(pattern_row.get("mode_requirements") or [])
			mode_has_extra_role = bool(pattern_row.get("has_extra_role"))
			mode_specific_requirements = [
				requirement
				for requirement in mode_requirements
				if requirement not in common_requirements
			]
			context_requirements = [
				*common_context_requirements,
				*mode_specific_requirements,
			]
			line = (
				f"{helper_signature}: valid constructive sibling for {mode_call}: "
				f"precondition/context {', '.join(context_requirements) if context_requirements else 'none'}; "
			)
			if common_support_calls:
				line += f"support_before {'; '.join(common_support_calls)}; "
			line += f"producer {mode_call}."
			contract_lines.append(line)
			if mode_has_extra_role:
				binding_requirements = [
					requirement
					for requirement in mode_specific_requirements
					if _signature_mentions_aux_role(requirement)
				]
				if binding_requirements:
					contract_lines.append(
						f"{helper_signature}: if a constructive sibling uses {mode_call}, keep "
						f"binding literal(s) {', '.join(binding_requirements)} explicit in "
						"branch precondition/context before the producer step.",
					)
		for line in _alignment_warning_lines_for_task(
			helper_task_name,
			helper_parameters,
			precondition_predicate,
			action_analysis,
		):
			contract_lines.append(line.removeprefix("- "))

		payloads_by_name.setdefault(
			helper_task_name,
			{
				"task_name": helper_task_name,
				"display_name": helper_task_name,
				"task_signature": helper_signature,
				"required_parent_task": str(helper_spec.get("query_task_name", "")).strip(),
				"required_packaging_task": str(
					helper_spec.get("packaging_task_name", "")
				).strip(),
				"helper_target_args": list(helper_spec.get("helper_target_args") or []),
				"contract_lines": [],
			},
		)
		for line in contract_lines:
			if line not in payloads_by_name[helper_task_name]["contract_lines"]:
				payloads_by_name[helper_task_name]["contract_lines"].append(line)

	return list(payloads_by_name.values())


def _build_support_task_contract_payloads(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[Dict[str, Any]]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	query_task_names = {
		_anchor_display_name(anchor)
		for anchor in query_task_anchors
		if _anchor_display_name(anchor)
	}
	support_task_names = [
		str(task.name)
		for task in getattr(domain, "tasks", [])
		if str(task.name) not in query_task_names
	]
	helper_payloads = _build_required_helper_task_contract_payloads(
		domain,
		target_literals,
		query_task_anchors,
		action_analysis,
	)
	if not support_task_names and not helper_payloads:
		return []

	line_sources = [
		_support_task_summary_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_same_arity_transitive_requirement_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_same_arity_child_context_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_query_task_role_stabilizer_support_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_packaging_child_internal_contract_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_support_task_caller_shared_prerequisite_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_relevant_support_task_template_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_relevant_support_task_alignment_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_declared_support_task_applicability_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_relevant_support_task_internal_obligation_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_relevant_support_task_recursive_mode_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_relevant_support_task_recursive_template_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
		_relevant_support_task_cleanup_template_lines(
			domain,
			target_literals,
			query_task_anchors,
			action_analysis,
		),
	]
	grouped: Dict[str, list[str]] = {}
	for lines in line_sources:
		for task_name, entries in _group_contract_lines_by_task(lines, support_task_names).items():
			grouped.setdefault(task_name, [])
			for entry in entries:
				if entry not in grouped[task_name]:
					grouped[task_name].append(entry)

	payloads: list[Dict[str, Any]] = []
	for task in getattr(domain, "tasks", []):
		task_name = str(task.name)
		entries = grouped.get(task_name)
		if not entries:
			continue
		for mode_line in _support_task_headline_mode_slot_lines_for_task(
			task_name,
			task,
			domain,
			action_analysis,
			tuple(query_task_names),
		):
			if mode_line not in entries:
				entries.append(mode_line)
		noop_line = _support_task_noop_line(
			domain,
			task_name,
			task,
			target_literals,
			query_task_anchors,
			action_analysis,
		)
		if noop_line and noop_line not in entries:
			entries = [noop_line, *entries]
		payloads.append(
			{
				"task_name": _sanitize_name(task_name),
				"display_name": task_name,
				"task_signature": task.to_signature() if hasattr(task, "to_signature") else task_name,
				"contract_lines": entries,
			},
		)
	for helper_payload in helper_payloads:
		payloads.append(helper_payload)
	return payloads


def _support_task_headline_mode_slot_lines_for_task(
	task_name: str,
	task: Any,
	domain: Any,
	action_analysis: Dict[str, Any],
	query_task_names: Sequence[str],
) -> list[str]:
	lines: list[str] = []
	seen: set[str] = set()
	task_schemas = _declared_task_schema_map(domain)
	task_parameters = tuple(
		_parameter_token(parameter)
		for parameter in getattr(task, "parameters", ()) or ()
	)
	if not task_parameters:
		return []
	role_labels = tuple(
		f"ARG{index}"
		for index, _ in enumerate(task_parameters, start=1)
	)
	task_signature = _task_invocation_signature(task_name, task_parameters)
	headline_predicates = tuple(
		str(predicate_name).strip()
		for predicate_name in (getattr(task, "source_predicates", ()) or ())
		if str(predicate_name).strip()
	) or tuple(
		_candidate_headline_predicates_for_task(
			task_name,
			len(task_parameters),
			action_analysis,
		)[:1]
	)
	for headline_predicate in headline_predicates:
		aligned_role_labels = _aligned_task_parameter_sequence_for_predicate(
			task,
			headline_predicate,
			action_analysis,
			role_labels,
		)
		for producer_pattern in action_analysis.get(
			"producer_patterns_by_predicate",
			{},
		).get(headline_predicate, []):
			effect_args = list(producer_pattern.get("effect_args") or [])
			if len(effect_args) != len(task_parameters):
				continue
			token_mapping = {
				token: role_label
				for token, role_label in zip(effect_args, aligned_role_labels)
			}
			rendered_action_args = _extend_mapping_with_action_parameters(
				token_mapping,
				producer_pattern.get("action_parameters") or [],
				action_parameter_types=producer_pattern.get("action_parameter_types") or [],
			)
			rendered_requirements = [
				_render_signature_with_mapping(signature, token_mapping)
				for signature in (
					producer_pattern.get("dynamic_precondition_signatures") or []
				)
			]
			static_requirements = list(
				dict.fromkeys(
					requirement
					for requirement in _render_positive_static_requirements(
						producer_pattern,
						token_mapping,
					)
					if requirement
				)
			)
			mode_context_requirements = list(
				dict.fromkeys(
					[
						*rendered_requirements,
						*static_requirements,
					]
				)
			)
			mode_call = _task_invocation_signature(
				producer_pattern["action_name"],
				rendered_action_args,
			)
			supportable_requirements: list[tuple[str, str, tuple[str, ...], str]] = []
			for rendered_requirement in rendered_requirements:
				parsed_requirement = _parse_literal_signature(rendered_requirement)
				if parsed_requirement is None:
					continue
				requirement_predicate, requirement_args, requirement_positive = parsed_requirement
				if not requirement_positive or not requirement_args:
					continue
				requirement_support_tasks = [
					candidate_task
					for candidate_task in _candidate_support_task_names(
						domain,
						requirement_predicate,
						requirement_args,
						action_analysis.get("producer_actions_by_predicate", {}).get(
							requirement_predicate,
							[],
						),
					)
					if candidate_task != task_name
					and candidate_task not in query_task_names
					and len(getattr(task_schemas.get(candidate_task), "parameters", ()))
					== len(requirement_args)
				]
				if requirement_support_tasks:
					supportable_requirements.append(
						(
							rendered_requirement,
							requirement_predicate,
							tuple(requirement_args),
							requirement_support_tasks[0],
						)
					)
			line = (
				f"- {task_signature}: valid constructive sibling for {mode_call}: "
				f"precondition/context "
				f"{'; '.join(mode_context_requirements) if mode_context_requirements else 'none'}; "
				f"producer {mode_call}."
			)
			if line not in seen:
				seen.add(line)
				lines.append(line)
			for (
				support_requirement,
				support_predicate,
				support_args,
				support_task_name,
			) in supportable_requirements:
				primary_support_call = _headline_support_task_invocation_signature(
					support_task_name,
					support_predicate,
					support_args,
					task_schemas,
					action_analysis,
				)
				split_line = (
					f"- {task_signature}: if {support_requirement} may either already hold or "
					f"require internal support, use separate constructive siblings: one may "
					f"require {support_requirement} at entry, another should use "
					f"{primary_support_call} before "
					f"{mode_call}. Do not collapse both cases into one precondition-only branch."
				)
				if split_line not in seen:
					seen.add(split_line)
					lines.append(split_line)
				remaining_requirements = [
					requirement
					for requirement in mode_context_requirements
					if requirement != support_requirement
				]
				support_task_shared_requirements = list(
					dict.fromkeys(
						requirement
						for requirement in _support_task_precise_shared_requirements(
							domain,
							support_task_name,
							support_predicate,
							support_args,
							task_schemas,
							action_analysis,
						)
						if requirement and requirement != support_requirement
					)
				)
				internal_support_requirements = list(
					dict.fromkeys(
						[
							*remaining_requirements,
							*support_task_shared_requirements,
						]
					)
				)
				support_before_calls = [primary_support_call]
				current_requirement_index = next(
					(
						index
						for index, requirement in enumerate(rendered_requirements)
						if requirement == support_requirement
					),
					-1,
				)
				for requirement in tuple(internal_support_requirements):
					parsed_requirement = _parse_literal_signature(requirement)
					if parsed_requirement is None:
						continue
					req_predicate, req_args, req_positive = parsed_requirement
					if (
						not req_positive
						or not req_args
						or not _signature_mentions_aux_role(requirement)
					):
						continue
					other_requirement_index = next(
						(
							index
							for index, candidate_requirement in enumerate(rendered_requirements)
							if candidate_requirement == requirement
						),
						-1,
					)
					if (
						current_requirement_index >= 0
						and other_requirement_index >= 0
						and other_requirement_index <= current_requirement_index
					):
						continue
					restoration_task_candidates = [
						candidate_task
						for candidate_task in _candidate_support_task_names(
							domain,
							req_predicate,
							req_args,
							action_analysis.get("producer_actions_by_predicate", {}).get(
								req_predicate,
								[],
							),
						)
						if candidate_task not in {task_name, support_task_name}
						and candidate_task not in query_task_names
						and len(getattr(task_schemas.get(candidate_task), "parameters", ()))
						== len(req_args)
					]
					if not restoration_task_candidates:
						continue
					restoration_task_name = restoration_task_candidates[0]
					restoration_call = _headline_support_task_invocation_signature(
						restoration_task_name,
						req_predicate,
						req_args,
						task_schemas,
						action_analysis,
					)
					if restoration_call not in support_before_calls:
						support_before_calls.append(restoration_call)
					internal_support_requirements = [
						item
						for item in internal_support_requirements
						if item != requirement
					]
					restoration_shared_requirements = list(
						dict.fromkeys(
							shared_requirement
							for shared_requirement in _support_task_precise_shared_requirements(
								domain,
								restoration_task_name,
								req_predicate,
								req_args,
								task_schemas,
								action_analysis,
							)
							if shared_requirement and shared_requirement != requirement
						)
					)
					internal_support_requirements = list(
						dict.fromkeys(
							[
								*internal_support_requirements,
								*restoration_shared_requirements,
							]
						)
					)
				internal_support_line = (
					f"- {task_signature}: valid internal-support sibling for the false-"
					f"{support_requirement} case: precondition/context "
					f"{'; '.join(internal_support_requirements) if internal_support_requirements else 'none'}; "
					f"support_before {'; '.join(support_before_calls)}; "
					f"producer {mode_call}."
				)
				if internal_support_line not in seen:
					seen.add(internal_support_line)
					lines.append(internal_support_line)
				internal_support_ast_line = (
					f"- {task_signature}: AST slot shape for that internal-support sibling: "
					"{"
					f'"precondition":[{", ".join(json.dumps(item) for item in internal_support_requirements)}],'
					f'"support_before":[{", ".join(json.dumps(item) for item in support_before_calls)}],'
					f'"producer":{json.dumps(mode_call)}'
					"}."
				)
				if internal_support_ast_line not in seen:
					seen.add(internal_support_ast_line)
					lines.append(internal_support_ast_line)
				no_duplicate_line = (
					f"- {task_signature}: if support_before "
					f"{primary_support_call} handles "
					f"false-{support_requirement} before {mode_call}, do not also keep "
					f"{support_requirement} in that same sibling precondition/context; split "
					"already-supported and internal-support siblings."
				)
				if no_duplicate_line not in seen:
					seen.add(no_duplicate_line)
					lines.append(no_duplicate_line)
	return lines


def _packaging_child_internal_contract_lines(
	domain: Any,
	target_literals: Sequence[str],
	query_task_anchors: Sequence[Dict[str, Any]],
	action_analysis: Dict[str, Any],
) -> list[str]:
	if not query_task_anchors or len(query_task_anchors) != len(target_literals):
		return []

	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()

	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		task_schema = task_schemas.get(task_name)
		parsed_target = _parse_literal_signature(target_signature)
		if task_schema is None or parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		task_parameters = tuple(
			_parameter_token(parameter)
			for parameter in getattr(task_schema, "parameters", ()) or ()
		)
		candidates = _same_arity_packaging_candidates_for_query_task(
			domain,
			task_name,
			predicate_name,
			task_parameters,
			task_schemas,
			action_analysis,
		)
		if not candidates:
			continue

		child_target_signature = f"{predicate_name}({', '.join(f'ARG{index}' for index, _ in enumerate(target_args, start=1))})"
		for candidate in candidates:
			candidate_name = str(candidate.get("candidate", "")).strip()
			candidate_parameters = tuple(candidate.get("parameters", ()))
			rendered_shared_requirements = list(
				_same_arity_caller_shared_requirements(
					domain,
					predicate_name,
					tuple(
						f"ARG{index}"
						for index, _ in enumerate(candidate_parameters, start=1)
					),
					action_analysis,
					candidate,
				),
			)
			caller_shared_mapping = {
				_parameter_token(raw_parameter): f"ARG{index}"
				for index, raw_parameter in enumerate(candidate_parameters, start=1)
			}
			pseudo_anchor = {
				"task_name": candidate_name,
				"source_name": candidate_name,
				"args": list(candidate_parameters),
				"force_internal_contract": True,
				"caller_shared_requirements": rendered_shared_requirements,
			}
			sources = (
				_declared_task_producer_template_lines,
				_query_task_support_producer_lines,
			)
			for source in sources:
				for line in source(
					domain,
					(child_target_signature,),
					(pseudo_anchor,),
					action_analysis,
				):
					if line in seen:
						continue
					seen.add(line)
					lines.append(line)
			inverse_child_mapping = {
				rendered_label: raw_parameter
				for raw_parameter, rendered_label in caller_shared_mapping.items()
			}
			rendered_child_shared_requirements = [
				_render_signature_with_mapping(
					requirement,
					inverse_child_mapping,
				)
				for requirement in rendered_shared_requirements
			]
			child_shared_requirements = _shared_dynamic_requirements_for_predicate(
				predicate_name,
				tuple(
					f"ARG{index}"
					for index, _ in enumerate(candidate_parameters, start=1)
				),
				action_analysis,
			)
			remaining_entry_requirements = [
				_render_signature_with_mapping(
					requirement,
					inverse_child_mapping,
				)
				for requirement in child_shared_requirements
				if requirement not in rendered_shared_requirements
			]
			if remaining_entry_requirements:
				entry_line = (
					f"- {_task_invocation_signature(candidate_name, candidate_parameters)}: with "
					f"parent-side caller-shared prerequisites "
					f"{', '.join(rendered_child_shared_requirements) if rendered_child_shared_requirements else 'none'}, "
					f"keep the remaining shared final-producer prerequisites "
					f"{', '.join(remaining_entry_requirements)} explicit in the constructive "
					"branch precondition/context at child entry unless earlier child subtasks "
					"establish them."
				)
				if entry_line not in seen:
					seen.add(entry_line)
					lines.append(entry_line)

	return lines


def _format_tagged_block(tag_name: str, body: str) -> str:
	content = body.strip()
	if not content:
		return ""
	return f"<{tag_name}>\n{content}\n</{tag_name}>"


def _render_contract_blocks(
	tag_name: str,
	payloads: Sequence[Dict[str, Any]],
) -> str:
	blocks: list[str] = []
	for payload in payloads:
		lines = [
			_compact_contract_line(str(line).strip())
			for line in payload.get("contract_lines", ())
			if str(line).strip()
		]
		lines = _select_salient_contract_lines(tag_name, lines)
		if not lines and tag_name != "query_task_contract":
			continue
		header_lines = []
		ordered_binding = payload.get("ordered_binding") or {}
		if ordered_binding:
			header_lines.append(
				f"ordered_binding #{ordered_binding.get('index')}: "
				f"{ordered_binding.get('target_literal')} -> {ordered_binding.get('task_signature')}"
			)
		rendered_contract_lines = [
			line if line.startswith("- ") else f"- {line}"
			for line in lines
		]
		body = "\n".join(
			[f"- {line}" for line in header_lines]
			+ rendered_contract_lines
		)
		blocks.append(
			f"<{tag_name} name=\"{payload.get('display_name')}\">\n{body}\n</{tag_name}>"
		)
	return "\n".join(blocks) if blocks else f"<{tag_name}s>\n- none\n</{tag_name}s>"


def _select_salient_contract_lines(
	tag_name: str,
	lines: Sequence[str],
) -> list[str]:
	"""
	Keep the contract block compact while preserving the highest-value synthesis cues.

	The live prompt follows a schema-driven, structure-guided layout. To keep the
	one-shot synthesis step stable, retain the minimal set of contract lines that
	define headline producers, caller-shared prerequisites, recursive support, and
	exact packaging obligations, while trimming repetitive explanatory variants.
	"""
	normalised_lines = [str(line).strip() for line in lines if str(line).strip()]
	if not normalised_lines:
		return []
	if tag_name == "support_task_contract":
		has_exact_compatible_slot = any(
			"valid compatible slot sibling" in line
			for line in normalised_lines
		)
		has_explicit_constructive_slot = any(
			"valid constructive sibling for" in line
			for line in normalised_lines
		)
		if has_exact_compatible_slot:
			normalised_lines = [
				line
				for line in normalised_lines
				if "support remaining internal prerequisites via" not in line
				and "if you use a compatible extra-role mode" not in line
			]
		if has_explicit_constructive_slot:
			normalised_lines = [
				line
				for line in normalised_lines
				if "can serve as a declared support task" not in line
			]
		recursive_slot_action_names = _recursive_slot_action_names(normalised_lines)
		if recursive_slot_action_names:
			normalised_lines = [
				line
				for line in normalised_lines
				if not _is_redundant_recursive_constructive_line(
					line,
					recursive_slot_action_names,
				)
			]
	if tag_name == "query_task_contract":
		has_complete_constructive_slot = any(
			"complete constructive sibling for" in line
			for line in normalised_lines
		)
		if has_complete_constructive_slot:
			normalised_lines = [
				line
				for line in normalised_lines
				if "valid support-then-produce sibling" not in line
				and "AST slot shape for that support-then-produce sibling" not in line
			]
	limit_by_tag = {
		"query_task_contract": 9,
		"support_task_contract": 7,
	}
	limit = limit_by_tag.get(tag_name)
	if limit is None or len(normalised_lines) <= limit:
		return normalised_lines
	has_same_arity_packaging_contract = any(
		"exact same-arity packaging contract" in line
		for line in normalised_lines
	)

	def priority(line: str) -> tuple[int, int]:
		if "required noop branch precondition/context" in line:
			return (0, 1)
		if "required stable/noop branch precondition/context" in line:
			return (0, 2)
		if "use required helper task" in line:
			return (0, 3)
		if "exact same-arity packaging contract" in line or "exact same-arity packaging child" in line:
			return (0, 0)
		if "acts as a non-leading support/base role" in line:
			return (1, 1)
		if "minimal helper task for that predicate" in line:
			return (1, 2)
		if "predicate-aligned helper task" in line or "Parent sibling shape:" in line:
			return (1, 3)
		if "expects caller-shared" in line and "Before the child call, establish" in line:
			return (1, 4)
		if "same-arity packaging for" in line and "parent skeleton" in line:
			return (1, 5)
		if "must internally close the headline effect via" in line:
			return (1, 6)
		if "valid constructive sibling for" in line:
			return (2, 0)
		if "AST slot shape for that constructive sibling" in line:
			return (2, 1)
		if "valid internal-support sibling" in line:
			return (2, 2)
		if "AST slot shape for that internal-support sibling" in line:
			return (2, 3)
		if "split already-supported and internal-support siblings" in line:
			return (2, 4)
		if "complete constructive sibling for" in line:
			return (2, 5)
		if "AST slot shape for that complete constructive sibling" in line:
			return (2, 6)
		if "valid support-then-produce sibling" in line:
			return (2, 7)
		if "AST slot shape for that support-then-produce sibling" in line:
			return (2, 8)
		if "split already-supported and support-then-produce siblings" in line:
			return (2, 9)
		if "Do not swap it to" in line:
			return (2, 10)
		if "exact producer slots" in line:
			return (2, 11)
		if "targets " in line and "templates:" in line:
			return (2, 12)
		if (
			has_same_arity_packaging_contract
			and "Support options:" in line
			and "expects caller-shared" not in line
		):
			return (13, 0)
		if "Support options:" in line or " requires " in line:
			return (3, 0)
		if (
			"do not also keep" in line
			and (
				"split siblings" in line
				or "recursive-support siblings" in line
			)
		):
			return (4, 1)
		if "AST slot shape for the already-supported recursive sibling" in line:
			return (4, 2)
		if "AST slot shape for that recursive sibling" in line:
			return (4, 3)
		if "cleanup followup after" in line:
			return (2, 2)
		if "if the constructive branch uses" in line or "if the parent uses" in line:
			return (5, 0)
		if "must establish" in line and "instead of leaving it in branch context" in line:
			return (6, 0)
		if "support remaining internal prerequisites via" in line:
			return (6, 1)
		if "do not keep only self-requiring modes" in line:
			return (6, 2)
		if "parent tasks should not provide those internal stabilizer prerequisites" in line:
			return (6, 3)
		if "Do not collapse both cases into one precondition-only branch." in line:
			return (7, 0)
		if "before any helper or child call" in line:
			return (8, 0)
		if "caller-shared dynamic prerequisites" in line:
			return (9, 0)
		if "if a constructive sibling uses stack(" in line or "if a constructive sibling uses unstack(" in line:
			return (10, 0)
		if "keep binding literal(s)" in line:
			return (11, 0)
		if "other unmet needs" in line:
			return (12, 0)
		if "if a constructive sibling uses" in line or "if the constructive branch uses" in line:
			return (12, 1)
		if "recursive support is valid" in line:
			return (13, 0)
		if "recursive template" in line:
			return (15, 0)
		if "introduces extra roles" in line:
			return (16, 0)
		if "can serve as a declared support task" in line:
			return (18, 0)
		return (19, 0)

	selected_indices = {
		index
		for _, index in sorted(
			((priority(line), index) for index, line in enumerate(normalised_lines)),
			key=lambda item: (item[0], item[1]),
		)[:limit]
	}
	if tag_name == "support_task_contract":
		required_markers = (
			"must internally close the headline effect via",
			"AST slot shape for that recursive sibling",
			"AST slot shape for the already-supported recursive sibling",
			"split already-supported and recursive-support siblings",
			"cleanup followup after",
		)
		required_indices = {
			index
			for index, line in enumerate(normalised_lines)
			if any(marker in line for marker in required_markers)
		}
		for required_index in sorted(required_indices):
			if required_index in selected_indices:
				continue
			removable_candidates = [
				index
				for index in selected_indices
				if index not in required_indices
			]
			if not removable_candidates:
				continue
			worst_index = max(
				removable_candidates,
				key=lambda index: (priority(normalised_lines[index]), index),
			)
			selected_indices.remove(worst_index)
			selected_indices.add(required_index)
	selected_indices = sorted(selected_indices)
	return [normalised_lines[index] for index in selected_indices]


def _recursive_slot_action_names(lines: Sequence[str]) -> set[str]:
	action_names: set[str] = set()
	for line in lines:
		if "AST slot shape for" not in line or "recursive sibling" not in line:
			continue
		match = re.search(r'"producer":"([A-Za-z0-9_-]+)\(', line)
		if match is not None:
			action_names.add(match.group(1))
	return action_names


def _is_redundant_recursive_constructive_line(
	line: str,
	recursive_slot_action_names: set[str],
) -> bool:
	if "valid constructive sibling for" not in line or "AUX_" not in line:
		return False
	return any(
		f"valid constructive sibling for {action_name}(" in line
		for action_name in recursive_slot_action_names
	)


def _compact_contract_line(line: str) -> str:
	text = line.strip()
	if not text:
		return text

	replacements = (
		(
			" in earlier subtasks or context.",
			" before the step.",
		),
		(
			" via parent context or earlier parent subtasks.",
			" before the child call.",
		),
		(
			" Keep them as distinct method.parameters or earlier schematic child bindings; "
			"never substitute grounded query objects.",
			" Keep those extra roles schematic.",
		),
		(
			" explicit in method.context as the selected producer mode condition",
			" explicit in branch context",
		),
		(
			" explicit in method.context",
			" explicit in branch context",
		),
		(
			" in method.context",
			" in branch context",
		),
		(
			"method.context",
			"branch context",
		),
		(
			"method.parameters",
			"parameters",
		),
		(
			"inside methods",
			"inside task definitions",
		),
		(
			" via parent context or earlier parent subtasks.",
			" before the child call.",
		),
		(
			" Prefer simpler modes with fewer extra roles when they remain suitable.",
			" Prefer the simpler valid mode.",
		),
	)
	for source, target in replacements:
		text = text.replace(source, target)

	pattern_rewrites = (
		(
			r"^(?P<head>.+): before (?P<step>.+), support (?P<support>.+) before the step\.$",
			r"\g<head>: support \g<support> before \g<step>.",
		),
		(
			r"^(?P<head>.+): if you use (?P<child>.+) to support (?P<goal>.+), first support "
			r"its shared prerequisites (?P<requirements>.+) before the child call\.$",
			r"\g<head>: via \g<child>, parent must provide shared \g<requirements> before the child call.",
		),
		(
			r"^(?P<head>.+) targets (?P<goal>.+); constructive templates: (?P<templates>.+)$",
			r"\g<head> targets \g<goal>; templates: \g<templates>",
		),
		(
			r"^(?P<head>.+) can serve as a declared support task for (?P<goal>.+); "
			r"constructive templates: (?P<templates>.+); if a parent calls it, first provide "
			r"shared prerequisites (?P<requirements>.+)$",
			r"\g<head> supports \g<goal>; templates: \g<templates>; parent shared \g<requirements>.",
		),
	)
	for pattern, replacement in pattern_rewrites:
		updated = re.sub(pattern, replacement, text)
		if updated != text:
			text = updated
			break

	return re.sub(r"\s+", " ", text).strip()


def _target_predicate_names(target_literals: Sequence[str]) -> tuple[str, ...]:
	predicate_names: list[str] = []
	seen: set[str] = set()
	for signature in target_literals:
		parsed_literal = _parse_literal_signature(signature)
		if parsed_literal is None:
			continue
		predicate_name, _, is_positive = parsed_literal
		if not is_positive or predicate_name in seen:
			continue
		seen.add(predicate_name)
		predicate_names.append(predicate_name)
	return tuple(predicate_names)


def _relevant_dynamic_predicates_for_prompt(
	target_literals: Sequence[str],
	action_analysis: Dict[str, Any],
) -> tuple[str, ...]:
	relevant: set[str] = set()
	pending = list(_target_predicate_names(target_literals))
	while pending:
		predicate_name = pending.pop()
		if predicate_name in relevant:
			continue
		relevant.add(predicate_name)
		for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		):
			for requirement_signature in pattern.get("dynamic_precondition_signatures") or []:
				parsed_requirement = _parse_literal_signature(requirement_signature)
				if parsed_requirement is None:
					continue
				requirement_predicate, _, is_positive = parsed_requirement
				if not is_positive or requirement_predicate in relevant:
					continue
				pending.append(requirement_predicate)
	return tuple(sorted(relevant))


def _relevant_action_names_for_prompt(
	relevant_predicates: Sequence[str],
	action_analysis: Dict[str, Any],
) -> tuple[str, ...]:
	action_names: set[str] = set()
	for predicate_name in relevant_predicates:
		for patterns_key in (
			"producer_patterns_by_predicate",
			"consumer_patterns_by_predicate",
		):
			for pattern in action_analysis.get(patterns_key, {}).get(predicate_name, []):
				action_name = _sanitize_name(str(pattern.get("action_name") or "").strip())
				if action_name:
					action_names.add(action_name)
		for actions_key in (
			"producer_actions_by_predicate",
			"consumer_actions_by_predicate",
		):
			for action_name in action_analysis.get(actions_key, {}).get(predicate_name, []):
				sanitized = _sanitize_name(str(action_name).strip())
				if sanitized:
					action_names.add(sanitized)
	return tuple(sorted(action_names))


def _render_relevant_action_lines(
	domain: Any,
	relevant_action_names: Sequence[str],
) -> list[str]:
	selected_names = set(relevant_action_names)
	parser = HDDLConditionParser()
	lines: list[str] = []
	for action in getattr(domain, "actions", []):
		action_name = _sanitize_name(str(getattr(action, "name", "")).strip())
		if selected_names and action_name not in selected_names:
			continue
		try:
			parsed_action = parser.parse_action(action)
		except Exception:
			parsed_action = None
		if parsed_action is None:
			if hasattr(action, "to_description"):
				lines.append(f"- {action.to_description()}")
				continue
			lines.append(
				"- "
				f"{getattr(action, 'name', 'unknown')}({', '.join(getattr(action, 'parameters', []) or []) or 'none'})\n"
				f"    Pre: {getattr(action, 'preconditions', '()')}\n"
				f"    Eff: {getattr(action, 'effects', '()')}"
			)
			continue

		preconditions = [
			_clause_signature(clause)
			for clause in (parsed_action.precondition_clauses or [])
		] or ["true"]
		effects = [
			_literal_pattern_signature(effect)
			for effect in parsed_action.effects
		] or ["none"]
		lines.append(
			"- "
			f"{parsed_action.name}({', '.join(parsed_action.parameters) or 'none'}): "
			f"needs {' | '.join(preconditions)}; effects {', '.join(effects)}"
		)
	if lines:
		return lines
	for action in getattr(domain, "actions", []):
		try:
			parsed_action = parser.parse_action(action)
		except Exception:
			parsed_action = None
		if parsed_action is None:
			if hasattr(action, "to_description"):
				lines.append(f"- {action.to_description()}")
				continue
			lines.append(
				"- "
				f"{getattr(action, 'name', 'unknown')}({', '.join(getattr(action, 'parameters', []) or []) or 'none'})\n"
				f"    Pre: {getattr(action, 'preconditions', '()')}\n"
				f"    Eff: {getattr(action, 'effects', '()')}"
			)
			continue
		preconditions = [
			_clause_signature(clause)
			for clause in (parsed_action.precondition_clauses or [])
		] or ["true"]
		effects = [
			_literal_pattern_signature(effect)
			for effect in parsed_action.effects
		] or ["none"]
		lines.append(
			"- "
			f"{parsed_action.name}({', '.join(parsed_action.parameters) or 'none'}): "
			f"needs {' | '.join(preconditions)}; effects {', '.join(effects)}"
		)
	return lines or ["- none"]

# Active prompt builder. The layout is intentionally declarative and tagged:
# structured prompt blocks and fixed output contracts improve structured
# prediction reliability (Blevins et al., ACL 2023; Beurer-Kellner et al.,
# PLDI 2023), and prompt robustness for semantic parsing improves when
# avoidable prompt variation is reduced (Zhuo et al., EACL 2023).
def build_htn_system_prompt() -> str:
	return (
			"Synthesize one executable HTN method library from query contracts, declared "
			"tasks, predicates, and primitive action schemas only.\n"
			"Emit JSON only. No markdown or reasoning trace.\n"
			"Return minified JSON only: no breaks or extra fields.\n"
			"Single shot only: no second pass, repair, or hidden methods.\n"
			"Return one JSON object with top-level keys target_task_bindings and tasks.\n"
		"\n"
		"GLOBAL RULES:\n"
			"- Preserve ordered query skeleton; define each required task exactly once.\n"
			"- Every task named under required_tasks or support_task_contracts is mandatory; define it exactly once in tasks.\n"
			"- Do not duplicate tasks or branches per grounded target binding.\n"
			"- Grounded query objects may appear only in target_task_bindings and ordered bindings.\n"
		"- query inventory is authoritative for top-level grounding only.\n"
		"- Never invent type predicates unless the domain declares them.\n"
		"- never invent type predicates such as block(X) or rover(R).\n"
		"- ordering must be explicit pairwise edges. Never emit a chain edge like [[\"s1\",\"s2\",\"s3\"]].\n"
		"- Do not infer new packaging candidates or caller-shared envelopes.\n"
			"- Prefer declared support tasks; fresh helpers only if no listed task can discharge the dynamic predicate.\n"
			"- Each target-bound task needs a noop branch whose precondition/context already contains its headline literal; noop is mandatory for every target-bound task entry.\n"
			"- Support every dynamic prerequisite before its consuming step; do not hide unmet dynamic prerequisites in branch context; use those listed options or declared support tasks instead of inventing a fresh helper.\n"
			"- Treat each constructive branch like one HDDL :method: :precondition plus subtasks.\n"
			"- Before a compound child call, satisfy only caller-shared prerequisites; the child owns internal support and final producer.\n"
		"- If a same-arity packaging child is listed, the constructive branch must call that packaging child.\n"
			"- If a same-arity packaging child is listed, end parent ordered_subtasks with that child call, not the child's final primitive producer.\n"
			"- For same-arity packaging, the parent boundary is exactly the listed caller-shared set; keep child-only prerequisites inside the child.\n"
			"- Use ARG* for task-signature roles and AUX_* for extra witness roles; constrain AUX_* before use.\n"
		"- If a branch chooses ACTION(...), support every listed need for that mode before ACTION.\n"
			"- For a chosen producer mode, keep every other listed unmet need explicit until supported; never keep only a subset.\n"
		"- If support_before TASK handles false-P, do not also require P in that sibling precondition/context; split already-P and false-P siblings.\n"
			"- If a listed need is supportable, add support earlier in ordered_subtasks. If that need binds AUX_* to ARG*, keep it explicit as the selected producer-mode condition until use.\n"
		"- For AUX_* modes, any dynamic prerequisite linking AUX1 to a task argument or shared resource is mandatory.\n"
			"- Support tasks should return in a reusable stable state. Add named cleanup followups when a recursive/extra-role producer consumes a reusable resource.\n"
			"- Template argument positions are exact. If a listed template is ACTION(AUX1, ARG1), do not swap it to ACTION(ARG1, AUX1); effect positions must match headline ARG roles.\n"
		"- If a contract lists exact producer slots, copy one producer invocation verbatim; other branch obligations still apply.\n"
		"- Think in HDDL :method grammar: each branch is one method with :task, :precondition, and subtasks.\n"
		"- Pattern: if final producer PLACE(ARG1, ARG2) needs READY(ARG2), split already-ready and support-then-produce siblings.\n"
		"\n"
			"OUTPUT SHAPE:\n"
			"- target_task_bindings items use only target_literal and task_name.\n"
			"- Each tasks entry defines one compound task once, with fields name, parameters, noop, constructive. source_predicates is optional compiler metadata and may be omitted.\n"
		"- For every target-bound task, noop must be present as an object, even if the task has only one constructive branch.\n"
		"- Every compound child you call must also appear as another task entry in tasks.\n"
		"- materialize it as its own task entry with branches in the same JSON.\n"
		"- Never invent aggregate/root wrapper tasks that merely sequence the ordered query tasks.\n"
			"- constructive is non-empty. Split sibling branches when producer-mode or support-availability changes the method skeleton.\n"
		"- Branch precondition may be written as precondition or context.\n"
			"- For total orders, prefer compact slots: precondition/context, support_before, producer, followup(s); compiler expands them to ordered_subtasks.\n"
			"- Use explicit ordered_subtasks/ordering only when compact slots are insufficient.\n"
			"- Literals may be objects or compact strings such as on(ARG1, ARG2) or !clear(ARG1).\n"
			"- ordered_subtasks items may be compact invocation strings; ids/kinds are inferred.\n"
	)


def build_htn_user_prompt(
	domain: Any,
	target_literals: Iterable[str],
	schema_hint: str,
	*,
	query_text: str = "",
	query_task_anchors: Sequence[Dict[str, Any]] = (),
	semantic_objects: Sequence[str] = (),
	query_object_inventory: Sequence[Dict[str, Any]] = (),
	query_objects: Sequence[str] = (),
	action_analysis: Optional[Dict[str, Any]] = None,
	derived_analysis: Optional[Dict[str, Any]] = None,
) -> str:
	targets = [str(target).strip() for target in target_literals if str(target).strip()]
	analysis = _normalise_action_analysis(domain, action_analysis)
	prompt_analysis = dict(
		derived_analysis
		or build_prompt_analysis_payload(
			domain,
			target_literals=targets,
			query_task_anchors=query_task_anchors,
			action_analysis=analysis,
		)
	)
	query_task_anchors = tuple(
		prompt_analysis.get("ordered_query_task_anchors") or query_task_anchors,
	)
	query_task_contracts = list(
		prompt_analysis.get("query_task_contracts")
		or _build_query_task_contract_payloads(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	)
	support_task_contracts = list(
		prompt_analysis.get("support_task_contracts")
		or _build_support_task_contract_payloads(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	)
	semantic_objects = tuple(
		str(value).strip()
		for value in semantic_objects
		if str(value).strip()
	)
	query_object_inventory = _normalise_query_object_inventory(query_object_inventory)
	query_objects = tuple(
		str(value).strip()
		for value in (
			query_objects
			or _query_object_names_from_inventory(query_object_inventory)
		)
		if str(value).strip()
	)

	query_binding_lines = _unique_preserve_order(
		[
			f"target #{index}: {target} -> {_task_invocation_signature(_anchor_display_name(anchor), tuple(str(arg) for arg in anchor.get('args', ()) ))}"
			for index, (target, anchor) in enumerate(zip(targets, query_task_anchors), start=1)
			if _anchor_display_name(anchor)
		]
	)
	semantic_object_lines = [f"- {value}" for value in semantic_objects] or ["- none"]
	query_inventory_lines = [
		f"- {entry['type']}: {', '.join(entry['objects'])}"
		for entry in query_object_inventory
	] or ["- none"]
	query_inventory_summary_lines = [
		f"- {entry['type']}: {len(entry['objects'])} object(s)"
		for entry in query_object_inventory
	] or ["- none"]
	task_scope_signatures = _unique_preserve_order(
		[
			str(payload.get("task_signature", "")).strip()
			for payload in query_task_contracts + support_task_contracts
			if str(payload.get("task_signature", "")).strip()
		]
	)
	required_task_lines = [f"- {line}" for line in task_scope_signatures] or ["- none"]
	relevant_dynamic_predicates = _relevant_dynamic_predicates_for_prompt(targets, analysis)
	dynamic_predicate_lines = [
		f"- {predicate_name}"
		for predicate_name in relevant_dynamic_predicates
	] or ["- none"]
	relevant_action_names = _relevant_action_names_for_prompt(
		relevant_dynamic_predicates,
		analysis,
	)
	action_lines = [f"- {action_name}" for action_name in relevant_action_names] or ["- none"]

	grounding_block = "\n".join(
		[
			"query_type_inventory:",
			*query_inventory_summary_lines,
			"",
			"grounding_rules:",
			"- Ordered target bindings below are the authoritative grounded binding source for Stage 3.",
			"- This inventory is summary-only; do not reuse its grounded constants in methods.",
			"- Type names are not predicates.",
			"- Do not copy grounded object names into methods; methods must stay schematic.",
		]
	)
	ordered_target_binding_entries = [f"- {line}" for line in query_binding_lines] or ["- none"]
	task_scope_entries = [f"- {line}" for line in task_scope_signatures] or ["- none"]
	ordered_bindings_block = "\n".join(
		[
			"ordered_target_bindings:",
			*ordered_target_binding_entries,
		]
	)
	required_tasks_block = "\n".join(
		[
			"required_tasks:",
			*required_task_lines,
			"",
			"task_rules:",
			"- Define each listed task exactly once in tasks.",
			"- Add a fresh helper only if no listed task can responsibly discharge a required dynamic predicate.",
		]
	)
	query_task_contract_block = _render_contract_blocks(
		"query_task_contract",
		query_task_contracts,
	)
	support_task_contract_block = _render_contract_blocks(
		"support_task_contract",
		support_task_contracts,
	)
	domain_summary_block = "\n".join(
		[
			f"domain: {domain.name}",
			"relevant_dynamic_predicates:",
			*dynamic_predicate_lines,
			"",
			"relevant_primitive_actions:",
			*action_lines,
		]
	)
	instructions_block = "\n".join(
		[
			"1. Read query_task_contracts first; they define the top-level synthesis skeleton.",
			"2. Read support_task_contracts second; every named support task is mandatory, and child-internal support stays inside the child.",
			"3. If an exact same-arity packaging contract is listed, parent supports only the listed caller-shared prerequisites, then calls that child.",
			"3b. For same-arity packaging, treat the listed caller-shared set as the complete parent boundary. Keep later child-only prerequisites inside the child.",
			"4. If a line lists ACTION [needs ...] or [extra needs ...], satisfy those needs before ACTION. For supportable AUX_* needs, use earlier subtasks instead of leaving unmet dynamic prerequisites in constructive context.",
			"4b. If support_before TASK(...) handles false-P, do not also keep P in that sibling precondition/context; split already-P and false-P siblings.",
			"5. Use ARG1..ARGn for task-signature roles and AUX_* for extra roles. Declaring AUX_* is not enough; constrain it before first use.",
			"6. Keep template argument positions exact. If a listed template says ACTION(AUX1, ARG1), do not swap it to ACTION(ARG1, AUX1).",
			"7. Every called compound child must be materialized in tasks. Do not invent aggregate/root wrappers such as do_world, do_all, goal_root, or __top.",
			"8. Use one constructive branch by default; add siblings when producer-mode or support-availability differences change the valid entry precondition or ordered-subtask skeleton.",
			"9. Prefer ordered_subtasks for total orders; otherwise ordering uses pairwise edges only: [[\"s1\",\"s2\"],[\"s2\",\"s3\"]].",
			"11. If a listed need binds ARG* to AUX_* for the chosen mode, keep that binding literal in constructive precondition/context until the consuming producer step instead of trying to synthesize it earlier.",
			"12. When a contract line already names precondition/context, support_before, producer, and followup, copy that slot shape directly into the constructive branch instead of freehanding a different ordered_subtasks skeleton.",
			"12a. If a contract line lists exact producer slots, copy one producer invocation verbatim into producer and still keep the listed branch obligations.",
			"12b. If a contract line already gives precondition/context, producer, and followup but no support_before, do not invent extra support_before steps; add one only when it establishes a listed branch precondition or followup need.",
			"13. If a query-task contract names a unary role stabilizer before the packaging child or final producer, keep that stabilizer as a real sibling step; do not omit it for compactness.",
			"14. If a support-task contract names a cleanup followup after the producer, keep that followup in the branch so the task returns in a reusable stable state.",
		]
	)

	sections = [
		_format_tagged_block(
			"task",
			"Generate one executable JSON HTN library that compiles into valid AgentSpeak. Keep it compact only after branch obligations are explicit.",
		),
		_format_tagged_block(
			"query_summary",
			"Use the ordered query bindings below as the canonical decomposition. "
			"Do not copy grounded constants from the original sentence into task definitions.",
		),
		_format_tagged_block("grounding", grounding_block),
		_format_tagged_block("ordered_query_bindings", ordered_bindings_block),
		_format_tagged_block("required_tasks", required_tasks_block),
		_format_tagged_block("query_task_contracts", query_task_contract_block),
		_format_tagged_block("support_task_contracts", support_task_contract_block),
		_format_tagged_block("domain_summary", domain_summary_block),
		_format_tagged_block("instructions", instructions_block),
			_format_tagged_block(
				"output_schema",
				"Only define target_task_bindings and tasks; primitive tasks are injected automatically.\n"
				"Each noop or constructive entry compiles to one HDDL method. Prefer precondition/subtasks/ordering field names because they mirror HDDL grammar. Omit source_predicates when unsure; the compiler can recover task headlines from bindings/contracts.\n"
				f"{schema_hint}",
			),
	]
	return "\n\n".join(section for section in sections if section).strip() + "\n"


def build_domain_prompt_analysis_payload(
	domain: Any,
	*,
	action_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Build compact domain-complete contracts for all declared compound tasks."""

	analysis = _normalise_action_analysis(domain, action_analysis)
	task_headline_candidates = _task_headline_candidate_map(domain, analysis)
	shared_dynamic_prerequisites_by_task: Dict[str, list[str]] = {}
	producer_consumer_templates_by_task: Dict[str, list[str]] = {}
	domain_task_contracts: list[Dict[str, Any]] = []

	for task in getattr(domain, "tasks", []):
		task_name = str(getattr(task, "name", "")).strip()
		if not task_name:
			continue
		sanitized_task_name = _sanitize_name(task_name)
		task_parameters = tuple(_parameter_token(parameter) for parameter in task.parameters)
		headline_candidates = tuple(
			dict.fromkeys(
				str(predicate_name).strip()
				for predicate_name in (
					getattr(task, "source_predicates", ())
					or task_headline_candidates.get(sanitized_task_name, ())
				)
				if str(predicate_name).strip()
			)
		)
		shared_requirements: list[str] = []
		template_summaries: list[str] = []
		for predicate_name in headline_candidates:
			for requirement in _shared_dynamic_requirements_for_predicate(
				predicate_name,
				task_parameters,
				analysis,
			):
				if requirement not in shared_requirements:
					shared_requirements.append(requirement)
			constructive_template = _constructive_template_summary_for_task(
				task_name,
				task_parameters,
				predicate_name,
				analysis,
			)
			if constructive_template and constructive_template not in template_summaries:
				template_summaries.append(constructive_template)
			consumer_template = _render_consumer_template_summary_for_task(
				task_name,
				task_parameters,
				predicate_name,
				analysis,
			)
			if consumer_template and consumer_template not in template_summaries:
				template_summaries.append(consumer_template)
		shared_dynamic_prerequisites_by_task[sanitized_task_name] = shared_requirements
		producer_consumer_templates_by_task[sanitized_task_name] = template_summaries
		domain_task_contracts.append(
			{
				"task_name": task_name,
				"task_signature": _task_invocation_signature(task_name, task_parameters),
				"parameters": list(task_parameters),
				"headline_candidates": list(headline_candidates),
				"shared_dynamic_prerequisites": list(shared_requirements),
				"producer_consumer_templates": list(template_summaries),
			}
		)

	return {
		"declared_compound_tasks": [
			_task_invocation_signature(
				str(getattr(task, "name", "")).strip(),
				tuple(_parameter_token(parameter) for parameter in task.parameters),
			)
			for task in getattr(domain, "tasks", [])
			if str(getattr(task, "name", "")).strip()
		],
		"task_headline_candidates": task_headline_candidates,
		"shared_dynamic_prerequisites_by_task": shared_dynamic_prerequisites_by_task,
		"producer_consumer_templates_by_task": producer_consumer_templates_by_task,
		"reusable_dynamic_resource_predicates": _reusable_dynamic_resource_payloads(analysis),
		"domain_task_contracts": domain_task_contracts,
	}


def build_domain_htn_system_prompt() -> str:
	"""System prompt for one-shot domain-complete Stage 3 synthesis."""

	return (
		"Synthesize one executable domain-complete HTN method library from declared "
		"compound tasks, predicates, and primitive action schemas only.\n"
		"Emit JSON only. No markdown or reasoning trace.\n"
		"Return minified JSON only: no breaks or extra fields.\n"
		"Single shot only: no second pass, repair, or hidden methods.\n"
		"Return one JSON object with top-level key tasks.\n"
		"\n"
		"GLOBAL RULES:\n"
		"- Define every declared compound task exactly once in tasks.\n"
		"- Do not use grounded query constants, problem facts, or benchmark-specific bindings.\n"
		"- Keep methods schematic and domain-complete.\n"
		"- ordering must be explicit pairwise edges. Never emit a chain edge like [[\"s1\",\"s2\",\"s3\"]].\n"
		"- Every compound child you call must also appear as another task entry in tasks.\n"
		"- Prefer declared support structure implied by the contracts; do not invent aggregate wrappers.\n"
		"- Support dynamic prerequisites before the consuming producer step.\n"
		"- Use ARG* for task-signature roles and AUX_* for extra witness roles.\n"
		"- Think in HDDL :method grammar: each constructive branch is one reusable method.\n"
		"\n"
		"OUTPUT SHAPE:\n"
		"- Only emit tasks; primitive tasks are injected automatically.\n"
		"- Each tasks entry defines one compound task once, with fields name, parameters, noop, constructive.\n"
		"- constructive is non-empty.\n"
		"- Branch precondition may be written as precondition or context.\n"
		"- For total orders, prefer compact slots: precondition/context, support_before, producer, followup(s).\n"
		"- Use explicit ordered_subtasks/ordering only when compact slots are insufficient.\n"
	)


def _render_domain_task_contract_blocks(domain_task_contracts: Sequence[Dict[str, Any]]) -> str:
	blocks: list[str] = []
	for payload in domain_task_contracts:
		task_signature = str(payload.get("task_signature", "")).strip()
		if not task_signature:
			continue
		headline_candidates = payload.get("headline_candidates") or ["none"]
		shared_dynamic_prerequisites = payload.get("shared_dynamic_prerequisites") or ["none"]
		producer_consumer_templates = payload.get("producer_consumer_templates") or ["none"]
		blocks.append(
			"\n".join(
				[
					f"task_signature: {task_signature}",
					f"headline_candidates: {', '.join(str(item) for item in headline_candidates)}",
					"shared_dynamic_prerequisites:",
					*[
						f"- {item}"
						for item in shared_dynamic_prerequisites
					],
					"producer_consumer_templates:",
					*[
						f"- {item}"
						for item in producer_consumer_templates
					],
				]
			)
		)
	return "\n\n".join(blocks).strip()


def build_domain_htn_user_prompt(
	domain: Any,
	schema_hint: str,
	*,
	action_analysis: Optional[Dict[str, Any]] = None,
	derived_analysis: Optional[Dict[str, Any]] = None,
) -> str:
	"""User prompt for one-shot domain-complete Stage 3 synthesis."""

	analysis = _normalise_action_analysis(domain, action_analysis)
	prompt_analysis = dict(
		derived_analysis
		or build_domain_prompt_analysis_payload(
			domain,
			action_analysis=analysis,
		)
	)
	declared_compound_tasks = list(prompt_analysis.get("declared_compound_tasks") or ())
	domain_task_contracts = list(prompt_analysis.get("domain_task_contracts") or ())
	relevant_dynamic_predicates = _unique_preserve_order(
		predicate_name
		for payload in domain_task_contracts
		for predicate_name in (payload.get("headline_candidates") or ())
		if str(predicate_name).strip()
	)
	action_names = _relevant_action_names_for_prompt(relevant_dynamic_predicates, analysis)
	if not action_names:
		action_names = [
			str(getattr(action, "name", "")).strip()
			for action in getattr(domain, "actions", [])
			if str(getattr(action, "name", "")).strip()
		]
	resource_lines = [
		f"- {payload['predicate']}"
		for payload in (prompt_analysis.get("reusable_dynamic_resource_predicates") or ())
		if str(payload.get("predicate") or "").strip()
	] or ["- none"]
	task_lines = [f"- {task_signature}" for task_signature in declared_compound_tasks] or ["- none"]
	action_lines = [f"- {action_name}" for action_name in action_names] or ["- none"]
	domain_summary_block = "\n".join(
		[
			f"domain: {domain.name}",
			"declared_compound_tasks:",
			*task_lines,
			"",
			"relevant_primitive_actions:",
			*action_lines,
			"",
			"reusable_dynamic_resources:",
			*resource_lines,
		]
	)
	instructions_block = "\n".join(
		[
			"1. Read domain_task_contracts first; they define the full declared task coverage you must materialize.",
			"2. Define every declared compound task exactly once in tasks.",
			"3. Keep methods schematic. Do not ground with benchmark constants or problem objects.",
			"4. If a task contract lists shared dynamic prerequisites, support them before the consuming producer step.",
			"5. Preserve argument positions exactly across task headlines, producers, and support tasks.",
			"6. Keep child-internal support inside the child unless the contract marks it as caller-shared.",
			"7. Prefer reusable support structure over duplicating primitive producer logic in multiple parents.",
		]
	)
	sections = [
		_format_tagged_block(
			"task",
			"Generate one executable JSON HTN library for the whole domain. "
			"Do not condition the library on any single benchmark query.",
		),
		_format_tagged_block("domain_summary", domain_summary_block),
		_format_tagged_block(
			"domain_task_contracts",
			_render_domain_task_contract_blocks(domain_task_contracts),
		),
		_format_tagged_block("instructions", instructions_block),
		_format_tagged_block(
			"output_schema",
			"Only define tasks; primitive tasks are injected automatically.\n"
			"Each noop or constructive entry compiles to one HDDL method. Prefer precondition/subtasks/ordering field names because they mirror HDDL grammar. Omit source_predicates when unsure; the compiler can recover task headlines from contracts.\n"
			f"{schema_hint}",
		),
	]
	return "\n\n".join(section for section in sections if section).strip() + "\n"
