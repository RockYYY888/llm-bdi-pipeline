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
		candidates = [
			other_task.name
			for other_task in getattr(domain, "tasks", [])
			if other_task.name != task_name
			and len(other_task.parameters) == len(task_schema.parameters)
		]
		if not candidates:
			continue
		line = (
			f"- {_task_invocation_signature(display_name, task_parameters)}: if the final producer "
			f"still leaves unresolved dynamic support, consider same-arity declared tasks "
			f"{', '.join(_task_invocation_signature(candidate, task_parameters) for candidate in candidates)} "
			"before inventing a fresh helper."
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
			aux_index = 1
			for action_parameter in (
				_parameter_token(parameter)
				for parameter in (pattern.get("action_parameters") or [])
			):
				if action_parameter not in token_mapping:
					token_mapping[action_parameter] = f"AUX{aux_index}"
					aux_index += 1
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
	rendered_signature = signature
	for original, replacement in token_mapping.items():
		rendered_signature = re.sub(
			rf"(?<![A-Za-z0-9_]){re.escape(original)}(?![A-Za-z0-9_])",
			replacement,
			rendered_signature,
		)
	return rendered_signature


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
		patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, [])
		if not patterns:
			continue

		rendered_patterns: list[str] = []
		for pattern in patterns:
			effect_args = list(pattern.get("effect_args") or [])
			if len(effect_args) != len(task_parameters):
				continue

			token_mapping = {
				token: task_parameter
				for token, task_parameter in zip(effect_args, task_parameters)
			}
			aux_index = 1
			action_parameters = [
				_parameter_token(parameter)
				for parameter in (pattern.get("action_parameters") or [])
			]
			rendered_action_args: list[str] = []
			for action_parameter in action_parameters:
				if action_parameter not in token_mapping:
					token_mapping[action_parameter] = f"AUX{aux_index}"
					aux_index += 1
				rendered_action_args.append(token_mapping[action_parameter])

			rendered_preconditions = []
			for signature in pattern.get("dynamic_precondition_signatures") or []:
				rendered_preconditions.append(
					_render_signature_with_mapping(signature, token_mapping)
				)

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
			continue

		line = (
			f"- {_task_invocation_signature(display_name, task_parameters)} targets "
			f"{predicate_name}({', '.join(task_parameters)}); constructive templates: "
			f"{'; '.join(rendered_patterns)}"
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
			aux_index = 1
			rendered_action_args: list[str] = []
			for action_parameter in (
				_parameter_token(parameter)
				for parameter in (pattern.get("action_parameters") or [])
			):
				if action_parameter not in token_mapping:
					token_mapping[action_parameter] = f"AUX{aux_index}"
					aux_index += 1
				rendered_action_args.append(token_mapping[action_parameter])

			rendered_preconditions = []
			for signature in pattern.get("dynamic_precondition_signatures") or []:
				rendered_preconditions.append(
					_render_signature_with_mapping(signature, token_mapping)
				)

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
	for target_signature, anchor in zip(target_literals, query_task_anchors):
		task_name = str(anchor.get("task_name", "")).strip()
		display_name = _anchor_display_name(anchor)
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
			headline_aux_index = 1
			rendered_headline_action_args: list[str] = []
			for action_parameter in (
				_parameter_token(parameter)
				for parameter in (headline_pattern.get("action_parameters") or [])
			):
				if action_parameter not in headline_mapping:
					headline_mapping[action_parameter] = f"AUX{headline_aux_index}"
					headline_aux_index += 1
				rendered_headline_action_args.append(headline_mapping[action_parameter])

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

				support_patterns = action_analysis.get("producer_patterns_by_predicate", {}).get(
					precondition_predicate,
					[],
				)
				rendered_support_options: list[str] = []
				for support_pattern in support_patterns[:2]:
					support_effect_args = list(support_pattern.get("effect_args") or [])
					if len(support_effect_args) != len(precondition_args):
						continue

					support_mapping = {
						token: arg
						for token, arg in zip(support_effect_args, precondition_args)
					}
					support_aux_index = 1
					rendered_support_action_args: list[str] = []
					for action_parameter in (
						_parameter_token(parameter)
						for parameter in (support_pattern.get("action_parameters") or [])
					):
						if action_parameter not in support_mapping:
							support_mapping[action_parameter] = f"AUX{support_aux_index}"
							support_aux_index += 1
						rendered_support_action_args.append(support_mapping[action_parameter])

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
				helper_line = (
					f"- {_task_invocation_signature(display_name, task_args)}: no declared task "
					f"clearly matches {precondition_predicate}({', '.join(precondition_args)}). "
					"A minimal helper task for that dynamic predicate is allowed if earlier "
					"context or subtasks do not already provide it."
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
		"- Keep typed roles separate: one variable cannot stand for incompatible declared types or semantic roles.\n"
		"- Never use deprecated task prefixes achieve_, ensure_, goal_, or maintain_not_.\n"
		"- Every literal-bearing field must use JSON object form with predicate/args/is_positive.\n"
		"\n"
		"SYNTHESIS POLICY:\n"
		"- When the query explicitly names declared domain tasks, those task names are the primary HTN skeleton.\n"
		"- Prefer declared domain task names over fresh helper tasks.\n"
		"- Create a fresh helper task only if no declared task can express the required dynamic state change.\n"
		"- If no declared task clearly covers a required dynamic precondition, add one minimal helper instead of leaving that support unsatisfied.\n"
		"- Fresh helper tasks may correspond only to dynamic predicates.\n"
		"- Static predicates are context constraints only; never create helper tasks to establish them.\n"
		"- Bind each target literal to one top-level compound task. If an ordered query-task anchor is supplied for that target, prefer that exact task name.\n"
		"- Keep the library compact after executability is satisfied. Emit sibling methods only for genuine mode differences, producer-action alternatives, or already-satisfied cases.\n"
		"- Do not generate transitive support-closure libraries or exhaustive missing-support powersets.\n"
		"- Keep static resources, capabilities, topology, and immutable relations in method context when possible.\n"
		"- Every primitive step's dynamic preconditions must be guaranteed by earlier subtasks or method context.\n"
		"- If a primitive dynamic precondition is not achieved earlier, include it positively in method context.\n"
		"- Apply the same rule to compound subtasks when their constructive branches share a dynamic prerequisite.\n"
		"- No free variables; respect declared types and role distinctions.\n"
	)


def build_htn_user_prompt(
	domain: Any,
	target_literals: Iterable[str],
	schema_hint: str,
	*,
	query_text: str = "",
	query_task_anchors: Sequence[Dict[str, Any]] = (),
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
	declared_task_template_lines = "\n".join(
		_declared_task_producer_template_lines(
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
	binding_hints = "\n".join(
		f'- {{"target_literal": "{literal}", "task_name": "<top_level_declared_or_minimal_dynamic_task>"}}'
		for literal in targets
	)

	return (
		"TASK:\n"
		"Generate one compact but executable JSON HTN library that compiles into valid AgentSpeak.\n\n"
		"There is no hidden repair or supplementation step after this synthesis.\n\n"
		f"QUERY:\n{query_text or '- none provided'}\n\n"
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
		f"DECLARED TASK CONSTRUCTIVE TEMPLATES:\n{declared_task_template_lines}\n\n"
		f"QUERY-TASK SUPPORT OBLIGATIONS:\n{query_task_support_lines}\n\n"
		f"QUERY-TASK SUPPORT PRODUCERS:\n{query_task_support_producer_lines}\n\n"
		f"RUNTIME PRIMITIVE ACTION ALIASES:\n{chr(10).join(action_lines)}\n\n"
		f"ACTION PRECONDITION BRANCH HINTS (DNF):\n{branch_hints}\n\n"
		f"ORDERED TARGET LITERALS:\n{target_lines}\n\n"
		f"REQUIRED target_task_bindings ENTRIES:\n{binding_hints}\n\n"
		"CONSTRUCTION RULES:\n"
		"1. If the query explicitly names declared tasks, preserve those task names in the generated library.\n"
		"2. Use query-mentioned declared tasks as top-level bindings whenever they semantically match the ordered target literals.\n"
		"3. Prefer declared supporting tasks over fresh helper tasks.\n"
		"4. Only create a fresh helper task when no declared task can express the required dynamic state change.\n"
		"5. Never create helper tasks for static predicates.\n"
		"6. For each constructive method, inspect the final producer action or child task that achieves the intended dynamic effect.\n"
		"7. If that constructive step requires dynamic preconditions, normally establish them via supporting declared tasks or sibling mode branches instead of assuming them in context.\n"
		"8. For a task that headlines a positive dynamic predicate, at least one constructive branch must stay applicable when that headline literal is currently false.\n"
		"9. The already-satisfied branch should be the only branch that assumes the headline literal itself; non-empty branches should make progress from states where that literal is false.\n"
		"10. Respect producer-effect argument alignment exactly: only treat an action or subtask as supporting P(args) when its positive effect can instantiate to that same P(args).\n"
		"11. If a constructive branch is intended for !P(args), do not choose a producer chain whose unresolved preconditions still require that same P(args) to already hold.\n"
		"12. If a query task still has unresolved dynamic preconditions after obvious support tasks, consider same-arity declared tasks as reusable intermediate abstractions before inventing a fresh helper.\n"
		"13. If you introduce an auxiliary blocker/intermediate variable, add it to method.parameters and constrain it in method.context before referencing it in subtasks.\n"
		"14. If a declared task has a constructive producer template, build its constructive branch around one of those aligned templates instead of operating on a different object role.\n"
		"14a. Preserve typed role separation across every method: do not reuse one symbol for incompatible declared types or semantic roles.\n"
		"15. For every primitive step, each dynamic precondition must already be guaranteed by method context or by earlier subtasks in the ordering.\n"
		"15a. Do not stop after choosing a final producer step if that producer still has unresolved dynamic preconditions; recursively decompose those obligations first.\n"
		"15b. If you intentionally leave a primitive step's dynamic precondition to method applicability instead of earlier subtasks, state it explicitly in method.context.\n"
		"15c. If such a context literal introduces an auxiliary variable, declare that variable in method.parameters and constrain it in method.context.\n"
		"15d. If a compound child task's constructive branches share a dynamic prerequisite, provide that prerequisite in the parent method context or earlier parent subtasks before invoking the child.\n"
		"16. Do not use nop as filler inside constructive methods.\n"
		"17. Keep static capabilities, topology, visibility, equipment, and immutable relations in method context unless a declared task genuinely changes them.\n"
		"18. Use action producer alternatives or genuine already-satisfied cases to justify sibling methods; do not enumerate every support powerset.\n"
		"19. Keep the library compact: only include tasks and methods needed for the target bindings and executable support.\n"
		"20. Prefer semantic task names and reusable parameterization; do not clone grounded tasks per target literal.\n"
		"21. Do not bypass the query skeleton with a fresh helper-only library.\n\n"
		"DECISION PRIORITY:\n"
		"1. Valid JSON, bound variables, executable methods.\n"
		"2. Query-task alignment, then minimal reusable size.\n\n"
		"MICRO-EXAMPLES:\n"
		"- Valid: if attach(ARG1, ARG2) needs holding(ARG1) and ready(ARG2), support both before attach(ARG1, ARG2).\n"
		"- Invalid: ready(ARG1), ready(ARG2), attach(ARG1, ARG2) when holding(ARG1) was never established or stated in context.\n"
		"- Valid fallback: if no declared task clearly covers holding(ARG1), add one minimal helper for that dynamic predicate instead of omitting it.\n"
		"- Valid context use: if grab(ARG1) supports holding(ARG1) but still needs on_surface(ARG1) and resource_free, then state those in method.context or support them earlier.\n"
		"- Valid auxiliary binding: detach(AUX1, ARG2) for clear(ARG2) requires AUX1 in method.parameters and method.context.\n"
		"- Valid typed roles: operate(ACTOR, LOCATION, TARGET, TOOL, MODE) must keep ACTOR and TOOL as different variables.\n\n"
		"TOP-LEVEL JSON SHAPE:\n"
		"Only define target_task_bindings, compound_tasks, and methods; primitive tasks are injected automatically.\n"
		f"{schema_hint}\n\n"
		"FINAL CHECKLIST:\n"
		"- Each target literal has exactly one binding.\n"
		"- Query-mentioned declared tasks appear in the library when provided.\n"
		"- Primitive step literal is null unless that exact positive literal is a real action effect.\n"
		"- Every referenced compound subtask appears in compound_tasks and has methods.\n"
		"- Fresh helper tasks, if any, correspond only to dynamic predicates.\n"
		"- Static predicates appear only as context/preconditions, not helper-task headlines.\n"
		"- Every primitive step's dynamic preconditions are supported by method context or earlier subtasks.\n"
		"- Every compound step's shared dynamic prerequisites are supported by method context or earlier subtasks.\n"
		"- No primitive step relies on an unstated dynamic precondition.\n"
		"- No free variables.\n"
		"- No undefined compound subtasks.\n"
		"- Return one complete JSON object and nothing else.\n"
	)
