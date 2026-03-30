"""
Prompt builders for Stage 3 HTN method synthesis.
"""

from __future__ import annotations

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


def _compact_name_tokens(name: str) -> str:
	return "".join(_name_tokens(name))


def _candidate_support_task_names(
	domain: Any,
	predicate_name: str,
	predicate_args: Sequence[str],
	producer_actions: Sequence[str],
) -> list[str]:
	reference_tokens = _name_tokens(predicate_name)
	for action_name in producer_actions:
		reference_tokens += _name_tokens(action_name)
	candidates = []
	for task in getattr(domain, "tasks", []):
		task_tokens = _name_tokens(task.name)
		if not task_tokens:
			continue
		score = _token_overlap_score(task_tokens, reference_tokens)
		if score <= 0:
			continue
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
		if (
			predicate_name not in explicit_source_predicates
			and predicate_name not in inferred_headlines
			and constructive_template is None
		):
			continue
		if constructive_template is None:
			continue
		shared_requirements = _support_task_precise_shared_requirements(
			domain,
			candidate,
			predicate_name,
			candidate_parameters,
			task_schemas,
			action_analysis,
		)
		packaging_candidates.append(
			{
				"candidate": candidate,
				"parameters": candidate_parameters,
				"constructive_template": constructive_template,
				"shared_requirements": shared_requirements,
			},
		)
	return packaging_candidates


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
			shared_requirements = tuple(candidate.get("shared_requirements", ()))
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
				f"producer for {predicate_name}({', '.join(task_parameters)})."
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
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
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
	lines: list[str] = []
	seen: set[str] = set()
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
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
			requirement_plans = [
				_same_arity_parent_requirement_plan(
					domain,
					requirement,
					action_analysis,
				)
				for requirement in exposed_requirements
			]
			requirement_plans.extend(
				_query_task_non_leading_role_stabilizer_plans(
					domain,
					task_schema,
					bound_occurrences=[
						tuple(str(arg) for arg in occurrence.get("args", ()))
						for occurrence in query_task_anchors
						if str(occurrence.get("task_name", "")).strip() == task_name
						and len(tuple(occurrence.get("args", ()))) == len(task_parameters)
					],
					predicate_name=predicate_name,
					action_analysis=action_analysis,
				)
			)
			line = (
				f"- {_task_invocation_signature(display_name, task_parameters)}: once you choose "
				f"{_task_invocation_signature(candidate, task_parameters)} as same-arity packaging "
				f"for {predicate_name}({', '.join(task_parameters)}), use a parent skeleton that first "
				f"supports {'; '.join(requirement_plans)} and only then calls "
				f"{_task_invocation_signature(candidate, task_parameters)}. Do not compress away any "
				"declared unary stabilizer in that skeleton; keep it as a real compound task with "
				"methods. Any unary stabilizer in that skeleton must internally close its own "
				"headline effect and absorb its remaining shared prerequisites inside its own "
				"methods; the parent should not provide those internal stabilizer prerequisites. "
				"Do not keep planning from the parent task's direct headline producer after selecting "
				"the packaging child."
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
		packaging_candidates = _same_arity_declared_task_candidates(
			domain,
			task_name,
			task_schemas,
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
						f"{' or '.join(_task_invocation_signature(candidate, support_args) for candidate in support_task_candidates[:2])} "
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

		rendered_preconditions = []
		for signature in pattern.get("dynamic_precondition_signatures") or []:
			rendered_signature = _render_signature_with_mapping(signature, token_mapping)
			parsed_signature = _parse_literal_signature(rendered_signature)
			if parsed_signature is None:
				continue
			_, rendered_args, rendered_positive = parsed_signature
			if not rendered_positive:
				continue
			if rendered_args and not set(rendered_args) & set(task_parameters):
				continue
			rendered_preconditions.append(rendered_signature)

		rendered_call = _task_invocation_signature(
			pattern["action_name"],
			rendered_action_args,
		)
		precondition_suffix = (
			f" [needs {', '.join(rendered_preconditions)}]"
			if rendered_preconditions
			else ""
		)
		rendered_patterns.append(f"{rendered_call}{precondition_suffix}")

	if not rendered_patterns:
		return None

	return "; ".join(rendered_patterns)


def _constructive_template_line_for_task(
	display_name: str,
	task_parameters: Sequence[str],
	predicate_name: str,
	action_analysis: Dict[str, Any],
) -> Optional[str]:
	rendered_patterns = _constructive_template_summary_for_task(
		display_name,
		task_parameters,
		predicate_name,
		action_analysis,
	)
	if rendered_patterns is None:
		return None

	return (
		f"- {_task_invocation_signature(display_name, task_parameters)} targets "
		f"{predicate_name}({', '.join(task_parameters)}); constructive templates: "
		f"{rendered_patterns}"
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
					line = _constructive_template_line_for_task(
						candidate_task,
						candidate_parameters,
						precondition_predicate,
						action_analysis,
					)
					if line is None or line in seen:
						continue
					seen.add(line)
					lines.append(line)
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
			extra_role = next((arg for arg in rendered_action_args if arg != target_arg), extra_roles[0])
			swapped_call = _task_invocation_signature(
				pattern["action_name"],
				(target_arg, extra_role),
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
							template_summary = _constructive_template_summary_for_task(
								candidate_task,
								candidate_parameters,
								support_requirement_predicate,
								action_analysis,
							)
							shared_requirements = _shared_dynamic_requirements_for_predicate(
								support_requirement_predicate,
								candidate_parameters,
								action_analysis,
							)
							if template_summary is None and not shared_requirements:
								continue

							line = (
								f"- {_task_invocation_signature(candidate_task, candidate_parameters)} "
								f"can serve as a declared support task for "
								f"{support_requirement_predicate}({', '.join(candidate_parameters)})"
							)
							if template_summary is not None:
								line += f"; constructive templates: {template_summary}"
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
								if len(getattr(task_schemas.get(candidate), "parameters", ()))
								== len(req_args)
							]
							has_task_parameter = any(arg in task_parameter_set for arg in req_args)
							has_extra_role = any(arg in extra_role_set for arg in req_args)
							if requirement_task_candidates and not has_task_parameter:
								requirement_fragments.append(
									f"{rendered_requirement} via "
									f"{' or '.join(_task_invocation_signature(candidate, req_args) for candidate in requirement_task_candidates[:2])} "
									f"before {_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}"
								)
							elif rendered_requirement in rendered_consumed_requirements:
								requirement_fragments.append(
									f"{rendered_requirement} explicit in method.context as the selected producer mode condition"
								)
							elif requirement_task_candidates:
								requirement_fragments.append(
									f"{rendered_requirement} via "
									f"{' or '.join(_task_invocation_signature(candidate, req_args) for candidate in requirement_task_candidates[:2])} "
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
	headline_candidates_by_task = _task_headline_candidate_map(domain, action_analysis)
	candidate_parameters = tuple(
		_parameter_token(parameter)
		for parameter in getattr(candidate_schema, "parameters", ()) or ()
	)
	if len(candidate_parameters) != len(candidate_args):
		return ()

	pattern_shared_sets: list[set[str]] = []
	for support_pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
		support_predicate,
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
		extra_roles = {
			arg
			for arg in rendered_action_args
			if arg not in candidate_parameters
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
			req_predicate, req_args, req_positive = parsed_requirement
			if not req_positive:
				continue
			if not req_args or any(arg in extra_roles for arg in req_args):
				continue
			if not all(arg in candidate_parameters for arg in req_args):
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
				if candidate != candidate_task
				if len(getattr(task_schemas.get(candidate), "parameters", ())) == len(req_args)
				and req_predicate in headline_candidates_by_task.get(_sanitize_name(candidate), [])
			]
			if requirement_task_candidates:
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
					if not shared_requirements:
						continue
					line = (
						f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: caller-shared "
						f"dynamic prerequisites {', '.join(shared_requirements)}. If a parent calls "
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
								f"{_task_invocation_signature(candidate_task, req_args)} before the primitive step."
							)
							if line in seen:
								continue
							seen.add(line)
							lines.append(line)
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
						line = (
							f"- {_task_invocation_signature(candidate_task, candidate_parameters)}: "
							f"recursive template for {_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}. "
							f"Keep mode context {'; '.join(mode_context_requirements)} and use subtasks "
							f"{'; '.join(_task_invocation_signature(candidate_task, _parse_literal_signature(requirement)[1]) for requirement in recursive_requirements)}; "
							f"{_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}."
						)
						if line in seen:
							continue
						seen.add(line)
						lines.append(line)
	return lines


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
									f"cleanup template after {_task_invocation_signature(support_pattern['action_name'], rendered_action_args)}. "
									f"If this branch should return with {lost_zeroary} restored, append "
									f"{_task_invocation_signature(cleanup_pattern['action_name'], rendered_cleanup_args)} "
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

			line = (
				f"- {_task_invocation_signature(display_name, task_parameters)}: "
				f"{render_prefix}{_task_invocation_signature(pattern['action_name'], rendered_action_args)} "
				f"introduces extra roles {', '.join(extra_roles)}. Keep them as distinct "
				"method.parameters or earlier schematic child bindings; never substitute "
				"grounded query objects."
			)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)

	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
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
				parsed_support = _parse_literal_signature(
					_render_signature_with_mapping(signature, token_mapping)
				)
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
	scored_predicates: list[tuple[int, str]] = []
	for predicate_name, patterns in action_analysis.get("producer_patterns_by_predicate", {}).items():
		if not any(len(pattern.get("effect_args") or []) == task_arity for pattern in patterns):
			continue
		score = _token_overlap_score(task_tokens, _name_tokens(predicate_name))
		predicate_compact = _compact_name_tokens(predicate_name)
		if task_compact and predicate_compact:
			if task_compact == predicate_compact:
				score += 6
			elif task_compact.endswith(predicate_compact) or predicate_compact.endswith(task_compact):
				score += 4
			elif predicate_compact in task_compact or task_compact in predicate_compact:
				score += 2
		for pattern in patterns:
			score += _token_overlap_score(
				task_tokens,
				_name_tokens(str(pattern.get("action_name") or "")),
			)
		if score <= 0:
			continue
		scored_predicates.append((score, predicate_name))
	scored_predicates.sort(key=lambda item: (-item[0], item[1]))
	return [predicate_name for _, predicate_name in scored_predicates]


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
				f"{role_label} beyond the headline producer's direct prerequisites, prefer it "
				f"before {child_description}. The stabilizer task itself must internally close its "
				"headline effect and absorb any remaining shared prerequisites inside its own "
				"methods; parent methods should not provide those internal stabilizer "
				"prerequisites merely because the stabilizer was chosen."
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
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		anchor_task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

		headline_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, [])
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

				shared_requirements = _shared_dynamic_requirements_for_predicate(
					precondition_predicate,
					precondition_args,
					action_analysis,
				)
				if not shared_requirements:
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
					continue

				for candidate_task in candidate_tasks[:2]:
					candidate_schema = task_schemas.get(candidate_task)
					if candidate_schema is None:
						continue
					candidate_args = tuple(
						_parameter_token(parameter)
						for parameter in candidate_schema.parameters
					)
					precise_shared_requirements = _support_task_precise_shared_requirements(
						domain,
						candidate_task,
						precondition_predicate,
						candidate_args,
						task_schemas,
						action_analysis,
					)
					rendered_shared_requirements = (
						precise_shared_requirements
						or shared_requirements
					)
					line = (
						f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
						f"if you use {_task_invocation_signature(candidate_task, candidate_args)} "
						f"to support {precondition_predicate}({', '.join(precondition_args)}), "
						f"first support its shared prerequisites {', '.join(rendered_shared_requirements)} "
						"via parent context or earlier parent subtasks."
					)
					if line in seen:
						continue
					seen.add(line)
					lines.append(line)
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
									compatible_modes.append(
										_task_invocation_signature(
											support_pattern["action_name"],
											rendered_action_args,
										)
									)
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
					line = (
						f"- {_task_invocation_signature(candidate_name, (role_label,))}: if used as a "
						f"role stabilizer, its constructive branch must internally close "
						f"{candidate_predicate}({role_label}); constructive template {template_summary}"
					)
					downstream_reusable_conditions = sorted(
						requirement
						for requirement in direct_role_requirements[role_index]
						if requirement != candidate_signature
					)
					if downstream_reusable_conditions:
						joined_conditions = "; ".join(downstream_reusable_conditions)
						line += (
							f". For already-stable cases where a later child or producer only "
							f"reuses {joined_conditions} on {role_label}, include a stable/noop "
							f"sibling with {joined_conditions} in method.context instead of "
							f"requiring {candidate_predicate}({role_label})"
						)
					if support_option_fragments:
						line += f". Support remaining prerequisites via {'; '.join(support_option_fragments)}"
					if self_requiring_modes:
						line += (
							f". For the false-{candidate_predicate}({role_label}) constructive case, "
							f"do not rely on self-requiring modes {'; '.join(self_requiring_modes)}"
						)
						if compatible_modes:
							line += f"; prefer compatible modes {'; '.join(compatible_modes)}"
					if compatible_mode_binding_hints:
						line += (
							f". If you use a compatible extra-role mode, "
							f"{'; '.join(compatible_mode_binding_hints)}"
						)
					if compatible_mode_antidetour_hints:
						line += f". Also, {'; '.join(compatible_mode_antidetour_hints)}"
					if shared_requirements:
						line += (
							". Parent tasks should not provide those internal stabilizer "
							"prerequisites"
						)
					line += "."
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
					role_stable_requirements = direct_role_requirements.get(role_index, [])
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
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
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
		same_arity_candidates = _same_arity_packaging_candidates_for_query_task(
			domain,
			task_name,
			predicate_name,
			task_signature_parameters,
			task_schemas,
			action_analysis,
		)

		task_args = tuple(f"ARG{index}" for index in range(1, len(target_args) + 1))
		if anchor.get("args") and len(anchor.get("args", [])) == len(target_args):
			task_args = tuple(
				f"ARG{index}" for index, _ in enumerate(target_args, start=1)
			)
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

			for precondition_signature in headline_pattern.get("dynamic_precondition_signatures") or []:
				parsed_precondition = _parse_literal_signature(
					_render_signature_with_mapping(precondition_signature, headline_mapping)
				)
				if parsed_precondition is None:
					continue
				precondition_predicate, precondition_args, precondition_positive = parsed_precondition
				if not precondition_positive:
					continue
				if not set(precondition_args) & set(task_args):
					continue

				support_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
					precondition_predicate,
					[],
				)
				rendered_support_options: list[str] = []
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
						_render_signature_with_mapping(signature, support_mapping)
						for signature in (support_pattern.get("dynamic_precondition_signatures") or [])
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

				candidate_declared_tasks = _candidate_support_task_names(
					domain,
					precondition_predicate,
					precondition_args,
					action_analysis.get("producer_actions_by_predicate", {}).get(
						precondition_predicate,
						[],
					),
				)
				if candidate_declared_tasks:
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
	return lines


def build_htn_system_prompt() -> str:
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
		"- Never omit required structural fields. compound_tasks still need name, parameters, is_primitive, and source_predicates. Every subtask still needs step_id, task_name, args, and kind. Primitive subtasks also need action_name.\n"
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
		"\n"
		"SYNTHESIS POLICY:\n"
		"- When the query explicitly names declared domain tasks, those task names are the primary HTN skeleton.\n"
		"- Prefer declared domain task names over fresh helper tasks.\n"
		"- Create a fresh helper task only if no declared task can express the required dynamic state change.\n"
		"- If no declared task clearly covers a required dynamic precondition, add one minimal helper instead of leaving that support unsatisfied.\n"
		"- If no declared task directly headlines an intermediate dynamic predicate but a reusable same-arity declared packaging task exists, prefer that declared packaging task before inventing a helper.\n"
		"- Fresh helper tasks may correspond only to dynamic predicates.\n"
		"- Static predicates are context constraints only; never create helper tasks to establish them.\n"
		"- Bind each target literal to one top-level compound task. If an ordered query-task anchor is supplied for that target, prefer that exact task name.\n"
		"- For every target-bound task, include an already-satisfied/noop method whose context contains the target headline literal and whose subtask list is empty. For declared support/stabilizer tasks, an already-stable/noop branch may instead use the downstream support-role condition when that already makes the role reusable.\n"
		"- Keep the library compact after executability is satisfied. Emit sibling methods only for genuine mode differences, producer-action alternatives, or already-satisfied cases.\n"
		"- Do not generate transitive support-closure libraries or exhaustive missing-support powersets.\n"
		"- Keep static resources, capabilities, topology, and immutable relations in method context when possible.\n"
		"- Every primitive step's dynamic preconditions must be guaranteed by earlier subtasks or method context.\n"
		"- If a primitive dynamic precondition is not achieved earlier, include it positively in method context.\n"
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
		"- Compactness never removes required structure: keep source_predicates on compound tasks, kind on every subtask, action_name on primitive subtasks, and explicit ordering edges on every multi-step method.\n"
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
		_query_task_role_frame_lines(
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
		_query_task_role_stabilizer_support_lines(
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
			shared_requirements = tuple(candidate.get("shared_requirements", ()))
			requirement_text = ", ".join(shared_requirements) if shared_requirements else "none"
			line = (
				f"- {_task_invocation_signature(candidate_name, child_parameters)}: exact same-arity "
				f"packaging child for {predicate_name}({', '.join(parent_parameters)}) when called by "
				f"{_task_invocation_signature(display_name, parent_parameters)}. Parent-side "
				f"caller-shared prerequisites: {requirement_text}. This child must own the final "
				f"producer for {predicate_name}({', '.join(parent_parameters)})."
			)
			if line not in seen:
				seen.add(line)
				lines.append(line)
	return lines


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
	if not support_task_names:
		return []

	line_sources = [
		_support_task_summary_lines(
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
		payloads.append(
			{
				"task_name": _sanitize_name(task_name),
				"display_name": task_name,
				"task_signature": task.to_signature() if hasattr(task, "to_signature") else task_name,
				"contract_lines": entries,
			},
		)
	return payloads


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
		if not lines and tag_name != "query_task_contract":
			continue
		header_lines = []
		ordered_binding = payload.get("ordered_binding") or {}
		if ordered_binding:
			header_lines.append(
				f"ordered_binding #{ordered_binding.get('index')}: "
				f"{ordered_binding.get('target_literal')} -> {ordered_binding.get('task_signature')}"
			)
		body = "\n".join(
			[f"- {line}" for line in header_lines]
			+ [f"- {line}" for line in lines]
		)
		blocks.append(
			f"<{tag_name} name=\"{payload.get('display_name')}\">\n{body}\n</{tag_name}>"
		)
	return "\n".join(blocks) if blocks else f"<{tag_name}s>\n- none\n</{tag_name}s>"


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
			" explicit in method.context",
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


def build_htn_system_prompt() -> str:
	return (
		"You synthesize one compact, executable HTN method library from the query, declared tasks, "
		"predicates, and primitive action schemas only.\n"
		"No second pass exists. No repair pass exists. No hidden methods exist.\n"
		"Return JSON only. No markdown, no prose, no comments.\n"
		"Return exactly one object with top-level keys target_task_bindings, compound_tasks, methods.\n"
		"\n"
		"GLOBAL RULES:\n"
		"- Preserve query-mentioned declared tasks as the top-level skeleton whenever they match the ordered targets.\n"
		"- Prefer declared support tasks over fresh helpers; create a fresh helper only for a dynamic predicate that no declared task can responsibly cover.\n"
		"- Static predicates are context constraints only; never create helpers for them.\n"
		"- Never invent aggregate/root wrapper tasks that merely sequence the ordered query tasks; target_task_bindings already define the top-level roots.\n"
		"- Each target-bound task needs an already-satisfied/noop method with the headline literal in context.\n"
		"- Every primitive dynamic precondition must be supported by method context or earlier subtasks.\n"
		"- Every compound child call must satisfy the child's shared dynamic prerequisites before the child is called.\n"
		"- Only rely on a previous compound child's own headline effect and explicitly shared envelope, never incidental internal side effects.\n"
		"- Use same-arity packaging only when the user prompt provides an exact packaging contract. If selected, support only the listed caller-shared prerequisites before the child call and let that child own the final producer. Do not infer new packaging candidates or new caller-shared envelopes on your own.\n"
		"- Auxiliary variables must remain schematic, appear in method.parameters, and be constrained before use by declared predicates, equality constraints, or earlier subtask bindings; never invent type predicates such as block(X) or rover(R) unless the domain explicitly declares them.\n"
		"- Grounded query objects may appear in target literals and ordered top-level bindings only. If query inventory and Stage 1 semantic hints conflict, the query inventory is authoritative for top-level grounding only.\n"
		"- When a method chooses one producer mode or primitive branch, it must support the full listed dynamic preconditions of that selected mode before the step.\n"
		"\n"
		"OUTPUT CONTRACT:\n"
		"- task_name and method_name must match [a-z][a-z0-9_]*.\n"
		"- method_name must be exactly m_{task_name}_{strategy}.\n"
		"- task_args is optional; if omitted, the leading declared-task-arity method.parameters are the task arguments in order.\n"
		"- Primitive subtasks must use the provided runtime primitive aliases.\n"
		"- Zero-subtask methods must have non-empty context and empty subtasks/orderings.\n"
		"- If a method has two or more subtasks, ordering must be explicit pairwise edges such as [[\"s1\",\"s2\"],[\"s2\",\"s3\"]]. Never emit a chain edge like [[\"s1\",\"s2\",\"s3\"]].\n"
		"- Compactness is part of correctness: default to one noop branch and one constructive branch per task, and add siblings only for real producer-mode differences.\n"
		"- Omit optional empty/default fields when possible, but never omit required structural fields. Every literal-bearing field must use JSON object form with predicate/args/is_positive.\n"
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
	query_anchor_lines = _unique_preserve_order(
		[
			f"#{index}: {_task_invocation_signature(_anchor_display_name(anchor), tuple(str(arg) for arg in anchor.get('args', ()) ))}"
			for index, anchor in enumerate(query_task_anchors, start=1)
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
	relevant_dynamic_predicates = _relevant_dynamic_predicates_for_prompt(targets, analysis)
	dynamic_predicate_lines = [
		f"- {predicate_name}"
		for predicate_name in relevant_dynamic_predicates
	] or ["- none"]
	action_lines = _render_relevant_action_lines(
		domain,
		_relevant_action_names_for_prompt(relevant_dynamic_predicates, analysis),
	)

	grounding_block = "\n".join(
		[
			"query_type_inventory:",
			*query_inventory_summary_lines,
			"",
			"grounding_rules:",
			"- Ordered target bindings below are the authoritative grounded binding source for Stage 3.",
			"- This typed inventory is summary-only. It tells you which types exist and how many objects were named in the query, not which grounded constants to reuse in methods.",
			"- Do not copy grounded object names into methods; methods must stay schematic.",
		]
	)
	ordered_query_anchor_entries = [f"- {line}" for line in query_anchor_lines] or ["- none"]
	ordered_target_binding_entries = [f"- {line}" for line in query_binding_lines] or ["- none"]
	task_scope_entries = [f"- {line}" for line in task_scope_signatures] or ["- none"]
	ordered_bindings_block = "\n".join(
		[
			"ordered_query_task_anchors:",
			*ordered_query_anchor_entries,
			"",
			"ordered_target_bindings:",
			*ordered_target_binding_entries,
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
			"primitive_actions:",
			*action_lines,
		]
	)
	instructions_block = "\n".join(
		[
			"1. Read query_task_contracts first. They are the canonical synthesis skeleton.",
			"2. Read support_task_contracts second. If a parent calls a support child, satisfy every listed caller-shared prerequisite before the child call, and keep the child's internal support inside that child.",
			"3. Preserve ordered query task names in target_task_bindings and as top-level compound tasks when they are supplied. Do not invent aggregate/root wrappers such as do_world, do_all, goal_root, or __top to sequence those query tasks.",
			"4. Every task that appears in target_task_bindings must include an already-satisfied/noop method whose context contains that task's headline literal and whose subtasks are empty.",
			"5. Prefer declared support tasks over fresh helpers. Fresh helpers are allowed only when no declared task can responsibly own the dynamic predicate.",
			"6. Same-arity packaging is allowed only when an exact packaging contract is listed. If selected, support its listed caller-shared prerequisites first and then let that child own the final producer.",
			"7. Use ARG1..ARGn for task-signature roles and AUX_* for extra roles. Grounded query object names may appear only in target_task_bindings and ordered top-level bindings, never inside methods. Type names are not predicates.",
			"8. If a contract line lists ACTION [needs p, q, r], a method that chooses ACTION must support all of p, q, and r before ACTION. Multi-step methods require explicit pairwise ordering edges. Every AUX_* variable must be constrained before use; declaring AUX_* in method.parameters alone is insufficient.",
			"9. ordering: for subtasks s1 then s2 then s3, emit [[\"s1\",\"s2\"],[\"s2\",\"s3\"]]. Never emit [[\"s1\",\"s2\",\"s3\"]].",
			"10. packaging_envelope: if child(ARG1, ARG2) has constructive siblings with contexts {ready(ARG1), clear(ARG2)} and {ready(ARG1), linked(ARG2)}, then the caller-shared envelope is ready(ARG1) only.",
			"11. Never invent type predicates.",
		]
	)

	sections = [
		_format_tagged_block(
			"task",
			"Generate one compact but executable JSON HTN library that compiles into valid AgentSpeak.",
		),
		_format_tagged_block(
			"query_summary",
			"Use the ordered query bindings below as the canonical query decomposition. "
			"Do not copy grounded constants from the original sentence into methods.",
		),
		_format_tagged_block("grounding", grounding_block),
		_format_tagged_block("ordered_query_bindings", ordered_bindings_block),
		_format_tagged_block("query_task_contracts", query_task_contract_block),
		_format_tagged_block("support_task_contracts", support_task_contract_block),
		_format_tagged_block("domain_summary", domain_summary_block),
		_format_tagged_block("instructions", instructions_block),
		_format_tagged_block(
			"output_schema",
			"Only define target_task_bindings, compound_tasks, and methods; primitive tasks are injected automatically.\n"
			f"{schema_hint}",
		),
	]
	return "\n\n".join(section for section in sections if section).strip() + "\n"
