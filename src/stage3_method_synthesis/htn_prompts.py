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
		}

	parser = HDDLConditionParser()
	dynamic_predicates: set[str] = set()
	producer_actions_by_predicate: Dict[str, list[str]] = {}

	for action in getattr(domain, "actions", []):
		try:
			parsed_action = parser.parse_action(action)
		except Exception:
			continue
		for effect in parsed_action.effects:
			if effect.predicate == "=":
				continue
			dynamic_predicates.add(effect.predicate)
			if not effect.is_positive:
				continue
			producer_actions_by_predicate.setdefault(effect.predicate, []).append(
				_sanitize_name(parsed_action.name),
			)

	for predicate_name, producer_actions in list(producer_actions_by_predicate.items()):
		producer_actions_by_predicate[predicate_name] = sorted(dict.fromkeys(producer_actions))

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
	return [
		other_task.name
		for other_task in getattr(domain, "tasks", [])
		if other_task.name != task_name
		and len(other_task.parameters) == len(task_schema.parameters)
	]


def _query_task_same_arity_packaging_lines(
	domain: Any,
	query_task_anchors: Sequence[Dict[str, Any]],
) -> list[str]:
	task_schemas = _declared_task_schema_map(domain)
	lines: list[str] = []
	seen: set[str] = set()
	for anchor in query_task_anchors:
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
		task_schema = task_schemas.get(task_name)
		if task_schema is None:
			continue
		task_parameters = tuple(_parameter_token(parameter) for parameter in task_schema.parameters)
		candidates = _same_arity_declared_task_candidates(
			domain,
			task_name,
			task_schemas,
		)
		if not candidates:
			continue
		line = (
			f"- {_task_invocation_signature(display_name, task_parameters)}: if the final producer "
			f"still leaves unresolved dynamic support, prefer same-arity declared tasks "
			f"{', '.join(_task_invocation_signature(candidate, task_parameters) for candidate in candidates)} "
			"before inventing a fresh helper."
		)
		if line in seen:
			continue
		seen.add(line)
		lines.append(line)
	return lines


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
				if len(support_patterns) < 2:
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
					f"To supply {support_predicate}({', '.join(support_args)}), cover real producer modes "
					f"{support_summary}"
				)
				if fragment in mode_seen:
					continue
				mode_seen.add(fragment)
				mode_fragments.append(fragment)

		for candidate in packaging_candidates:
			line = (
				f"- {_task_invocation_signature(candidate, task_parameters)}: if used as same-arity "
				f"packaging for {predicate_name}({', '.join(task_parameters)}), it should itself reach "
				f"the headline effect via {headline_summary}."
			)
			if mode_fragments:
				line += f" {' '.join(mode_fragments[:2])}."
			line += (
				" Do not call this packaging child and then repeat the same final producer "
				"again in the parent."
			)
			if line in seen:
				continue
			seen.add(line)
			lines.append(line)
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
					line = (
						f"- {_task_invocation_signature(display_name, tuple(f'ARG{index}' for index in range(1, len(target_args) + 1)))}: "
						f"if you use {_task_invocation_signature(candidate_task, candidate_args)} "
						f"to support {precondition_predicate}({', '.join(precondition_args)}), "
						f"first support its shared prerequisites {', '.join(shared_requirements)} "
						"via parent context or earlier parent subtasks."
					)
					if line in seen:
						continue
					seen.add(line)
					lines.append(line)
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
		same_arity_candidates = _same_arity_declared_task_candidates(
			domain,
			task_name,
			task_schemas,
		)
		parsed_target = _parse_literal_signature(target_signature)
		if parsed_target is None:
			continue
		predicate_name, target_args, is_positive = parsed_target
		if not is_positive:
			continue

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
				remaining_requirements = _task_shared_and_transitive_requirements(
					predicate_name,
					task_args,
					action_analysis,
				)
				if same_arity_candidates:
					packaging_line = (
						f"- {_task_invocation_signature(display_name, task_args)}: no declared task "
						f"directly headlines {precondition_predicate}({', '.join(precondition_args)}). "
						f"Before inventing a helper, prefer same-arity declared packaging tasks "
						f"{', '.join(_task_invocation_signature(candidate, task_args) for candidate in same_arity_candidates)}"
					)
					if remaining_requirements:
						packaging_line += (
							f" and let those declared tasks absorb the remaining dynamic support "
							f"needed for {predicate_name}({', '.join(task_args)})"
						)
					packaging_line += "."
					if packaging_line in seen:
						continue
					seen.add(packaging_line)
					lines.append(packaging_line)
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
		"- task_name and method_name must match [a-z][a-z0-9_]*.\n"
		"- method_name must be exactly m_{task_name}_{strategy}.\n"
		"- Primitive subtasks must use the provided runtime primitive aliases.\n"
		"- Set primitive step literal to null unless that exact positive literal is a real action effect of that primitive step.\n"
		"- Primitive step literal is optional metadata; most primitive steps should set literal to null.\n"
		"- Every compound subtask must reference a declared compound task from the same JSON.\n"
		"- Every referenced compound subtask must also appear in compound_tasks and have at least one method in methods.\n"
		"- Zero-subtask methods must have non-empty context and empty subtasks/orderings.\n"
		"- Reuse parameterized tasks; never encode grounded constants into task names.\n"
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
		"- Keep the library compact after executability is satisfied. Emit sibling methods only for genuine mode differences, producer-action alternatives, or already-satisfied cases.\n"
		"- Do not generate transitive support-closure libraries or exhaustive missing-support powersets.\n"
		"- Keep static resources, capabilities, topology, and immutable relations in method context when possible.\n"
		"- Every primitive step's dynamic preconditions must be guaranteed by earlier subtasks or method context.\n"
		"- If a primitive dynamic precondition is not achieved earlier, include it positively in method context.\n"
		"- Apply the same rule to compound subtasks when their constructive branches share a dynamic prerequisite.\n"
		"- If a same-arity declared child is chosen as packaging for the headline effect, let that child own the full constructive path; do not immediately repeat the same final producer in the parent.\n"
		"- No free variables; respect declared types and role distinctions.\n"
	)


def build_htn_user_prompt(
	domain: Any,
	target_literals: Iterable[str],
	schema_hint: str,
	*,
	query_text: str = "",
	query_task_anchors: Sequence[Dict[str, Any]] = (),
	query_objects: Sequence[str] = (),
	action_analysis: Optional[Dict[str, Any]] = None,
) -> str:
	parser = HDDLConditionParser()
	analysis = _normalise_action_analysis(domain, action_analysis)

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
		_query_task_same_arity_packaging_lines(domain, query_task_anchors)
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
	declared_support_task_applicability_lines = "\n".join(
		_declared_support_task_applicability_lines(
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
	query_task_child_support_lines = "\n".join(
		_query_task_child_support_requirement_lines(
			domain,
			targets,
			query_task_anchors,
			analysis,
		)
	) or "- none"
	binding_hints = "\n".join(
		f'- {{"target_literal": "{literal}", "task_name": "<top_level_declared_or_minimal_dynamic_task>"}}'
		for literal in targets
	)
	query_object_section = ""
	if query_object_lines != "- none detected":
		query_object_section = (
			"QUERY OBJECT NAMES (grounded instances from the current query; never place them inside methods):\n"
			f"{query_object_lines}\n\n"
		)
	role_frame_section = ""
	if len(targets) <= 4 and query_task_role_frame_lines != "- none":
		role_frame_section = (
			f"QUERY-TASK EXTRA ROLE FRAMES:\n{query_task_role_frame_lines}\n\n"
		)
	relevant_support_task_section = ""
	if len(targets) <= 4 and relevant_support_task_template_lines != "- none":
		relevant_support_task_section = (
			"RELEVANT SUPPORT TASK CONSTRUCTIVE TEMPLATES:\n"
			f"{relevant_support_task_template_lines}\n\n"
		)
	declared_support_task_applicability_section = ""
	if len(targets) <= 4 and declared_support_task_applicability_lines != "- none":
		declared_support_task_applicability_section = (
			"DECLARED SUPPORT TASK APPLICABILITY ENVELOPES:\n"
			f"{declared_support_task_applicability_lines}\n\n"
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
	branch_hint_section = ""
	if branch_hints != "- none":
		branch_hint_section = (
			f"ACTION PRECONDITION BRANCH HINTS (DNF):\n{branch_hints}\n\n"
		)

	return (
		"TASK:\n"
		"Generate one compact but executable JSON HTN library that compiles into valid AgentSpeak.\n\n"
		f"QUERY:\n{query_text or '- none provided'}\n\n"
		f"{query_object_section}"
		f"ORDERED QUERY TASK ANCHORS:\n{query_anchor_lines}\n\n"
		f"ORDERED TARGET/TASK SKELETON HINTS:\n{anchor_binding_lines}\n\n"
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
		f"DYNAMIC PRECONDITION SUPPORT HINTS:\n{dynamic_support_lines}\n\n"
		f"QUERY-TASK SAME-ARITY SUPPORT CANDIDATES:\n{same_arity_task_lines}\n\n"
		f"QUERY-TASK SAME-ARITY PACKAGING HINTS:\n{query_task_same_arity_packaging_lines}\n\n"
		f"{same_arity_transitive_section}"
		f"DECLARED TASK CONSTRUCTIVE TEMPLATES:\n{declared_task_template_lines}\n\n"
		f"{relevant_support_task_section}"
		f"{declared_support_task_applicability_section}"
		f"{role_frame_section}"
		f"QUERY-TASK SUPPORT OBLIGATIONS:\n{query_task_support_lines}\n\n"
		f"QUERY-TASK SUPPORT PRODUCERS:\n{query_task_support_producer_lines}\n\n"
		f"{child_support_section}"
		f"RUNTIME PRIMITIVE ACTION ALIASES:\n{chr(10).join(action_lines)}\n\n"
		f"{branch_hint_section}"
		f"ORDERED TARGET LITERALS:\n{target_lines}\n\n"
		f"REQUIRED target_task_bindings ENTRIES:\n{binding_hints}\n\n"
		"CONSTRUCTION RULES:\n"
		"1. If the query explicitly names declared tasks, preserve those task names in the generated library, use query-mentioned declared tasks as top-level bindings whenever they semantically match the ordered target literals, prefer declared supporting tasks over fresh helper tasks, only create a fresh helper task when no declared task can express the required dynamic state change, and never create helper tasks for static predicates.\n"
		"2. For each constructive method, inspect the final producer action or child task that achieves the intended dynamic effect. If that constructive step requires dynamic preconditions, normally establish them via supporting declared tasks or sibling mode branches instead of assuming them in context.\n"
		"3. For a task that headlines a positive dynamic predicate, at least one constructive branch must stay applicable when that headline literal is currently false. The already-satisfied branch should be the only branch that assumes the headline literal itself; non-empty branches should make progress from states where that literal is false.\n"
		"4. Respect producer-effect argument alignment exactly: only treat an action or subtask as supporting P(args) when its positive effect can instantiate to that same P(args). If a constructive branch is intended for !P(args), do not choose a producer chain whose unresolved preconditions still require that same P(args) to already hold.\n"
		"5. If a query task still has unresolved dynamic preconditions after obvious support tasks, prefer same-arity declared tasks as reusable intermediate abstractions before inventing a fresh helper. Only fall back to a fresh helper when no declared support task or packaging task can responsibly absorb that obligation. If you choose a same-arity declared packaging task, that child should own the final producer for the headline effect instead of being followed by the same final producer again in the parent.\n"
		"6. If you introduce an auxiliary blocker/intermediate variable, add it to method.parameters, constrain it in method.context before using it in subtasks, and give it explicit typed evidence.\n"
		"7. If a declared task has a constructive producer template, build its constructive branch around one of those aligned templates instead of operating on a different object role. Preserve typed role separation across every method: do not reuse one symbol for incompatible declared types or semantic roles. Never place grounded query object names inside methods; use schematic parameters instead. When a producer needs extra roles beyond the headline arguments, add fresh schematic parameters for those roles.\n"
		"8. For every primitive step, each dynamic precondition must already be guaranteed by method context or by earlier subtasks in the ordering. Do not stop after choosing a final producer step if that producer still has unresolved dynamic preconditions; recursively decompose those obligations first. If you intentionally leave a primitive step's dynamic precondition to method applicability instead of earlier subtasks, state it explicitly in method.context. If such a context literal introduces an auxiliary variable, declare that variable in method.parameters and constrain it in method.context. If a compound child task's constructive branches share a dynamic prerequisite, provide that prerequisite in the parent method context or earlier parent subtasks before invoking the child.\n"
		"9. Do not use nop as filler inside constructive methods. Keep static capabilities, topology, visibility, equipment, and immutable relations in method context unless a declared task genuinely changes them.\n"
		"10. Use action producer alternatives or genuine already-satisfied cases to justify sibling methods; do not enumerate every support powerset. Keep the library compact: only include tasks and methods needed for the target bindings and executable support. Prefer semantic task names and reusable parameterization; do not clone grounded tasks per target literal. Do not bypass the query skeleton with a fresh helper-only library.\n\n"
		"MICRO-EXAMPLES:\n"
		"- Valid: if attach(ARG1, ARG2) needs holding(ARG1) and ready(ARG2), support both before attach(ARG1, ARG2).\n"
		"- Invalid recursion: if clear_top(AUX1) is followed by detach(AUX1, ARG1), do not assume detach's resource_free precondition was satisfied unless context or earlier subtasks make it true.\n"
		"- Valid fallback: if no declared task or reusable same-arity packaging task clearly covers holding(ARG1), add one minimal helper for that dynamic predicate instead of omitting it.\n"
		"- Valid same-arity packaging: if parent(ARG1, ARG2) delegates to child(ARG1, ARG2) as a same-arity packaging task for linked(ARG1, ARG2), let child own the final attach(ARG1, ARG2) rather than calling child(ARG1, ARG2) and then attach(ARG1, ARG2) again in the parent.\n"
		"- Parent-child obligation: if parent(ARG1, ARG2) calls child(ARG1) and every constructive branch of child(ARG1) needs ready(ARG1) and resource_free, parent must provide both before the child call.\n"
		"- Valid auxiliary binding: detach(AUX1, ARG2) for clear(ARG2) requires AUX1 in method.parameters and method.context.\n"
		"- Valid typed roles: operate(ACTOR, LOCATION, TARGET, TOOL, MODE) must keep ACTOR and TOOL as different variables.\n\n"
		"TOP-LEVEL JSON SHAPE:\n"
		"Only define target_task_bindings, compound_tasks, and methods; primitive tasks are injected automatically.\n"
		f"{schema_hint}\n\n"
		"FINAL CHECKLIST:\n"
		"- Each target literal has exactly one binding, and query-mentioned declared tasks appear in the library when provided.\n"
		"- Primitive step literal is null unless that exact positive literal is a real action effect, and every referenced compound subtask appears in compound_tasks and has methods.\n"
		"- Fresh helper tasks, if any, correspond only to dynamic predicates. Static predicates appear only as context/preconditions, not helper-task headlines.\n"
		"- Every primitive step's dynamic preconditions are supported by method context or earlier subtasks. Every compound step's shared dynamic prerequisites are supported by method context or earlier subtasks. No primitive step relies on an unstated dynamic precondition.\n"
		"- No free variables. No undefined compound subtasks. Every auxiliary variable has an explicit typing witness.\n"
		"- Return one complete JSON object and nothing else.\n"
	)
