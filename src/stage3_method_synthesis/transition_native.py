"""
Transition-native Stage 3 contracts and prompt builders.

This module defines the deterministic task skeleton used by the redesigned
Stage 3 pipeline:
- query-root alias tasks preserve the official root-task interface only for
  Stage 6 and Stage 7 compatibility,
- internal synthesis tasks are the raw DFA progress transitions themselves.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from stage3_method_synthesis.htn_prompts import _render_producer_mode_options_for_predicate
from stage3_method_synthesis.htn_schema import HTNLiteral, _parse_signature_literal


def sanitize_identifier(value: str) -> str:
	text = str(value or "").strip().replace("-", "_")
	buffer: List[str] = []
	for character in text:
		if character.isalnum() or character == "_":
			buffer.append(character.lower())
		else:
			buffer.append("_")
	sanitized = "".join(buffer).strip("_")
	while "__" in sanitized:
		sanitized = sanitized.replace("__", "_")
	if not sanitized:
		sanitized = "item"
	if not sanitized[0].isalpha():
		sanitized = f"t_{sanitized}"
	return sanitized


def query_root_alias_task_name(index: int, source_task_name: str) -> str:
	return f"query_root_{int(index)}_{sanitize_identifier(source_task_name)}"


def parameter_symbols(arity: int) -> Tuple[str, ...]:
	return tuple(f"ARG{index}" for index in range(1, int(arity) + 1))


def parameterise_literal_from_grounded_args(
	literal: HTNLiteral,
	grounded_args: Sequence[str],
	parameters: Sequence[str],
) -> HTNLiteral:
	if not grounded_args:
		return HTNLiteral(
			predicate=literal.predicate,
			args=tuple(parameters[: len(literal.args)]),
			is_positive=literal.is_positive,
			source_symbol=None,
			negation_mode=literal.negation_mode,
		)

	position_queues: Dict[str, deque[str]] = defaultdict(deque)
	for grounded_arg, parameter in zip(grounded_args, parameters):
		position_queues[str(grounded_arg)].append(str(parameter))

	mapped_args: List[str] = []
	for index, literal_arg in enumerate(literal.args):
		candidates = position_queues.get(str(literal_arg))
		if candidates:
			mapped_args.append(candidates[0])
			continue
		if index < len(parameters):
			mapped_args.append(str(parameters[index]))
			continue
		mapped_args.append(str(literal_arg))

	return HTNLiteral(
		predicate=literal.predicate,
		args=tuple(mapped_args),
		is_positive=literal.is_positive,
		source_symbol=None,
		negation_mode=literal.negation_mode,
	)


def build_transition_native_prompt_analysis(
	*,
	target_literals: Sequence[HTNLiteral],
	query_task_anchors: Sequence[Dict[str, Any]],
	transition_specs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
	query_root_alias_tasks: List[Dict[str, Any]] = []
	target_task_bindings: List[Dict[str, Any]] = []

	for index, (target_literal, anchor) in enumerate(
		zip(target_literals, query_task_anchors),
		start=1,
	):
		source_task_name = str(anchor.get("task_name", "")).strip()
		grounded_args = tuple(str(arg).strip() for arg in (anchor.get("args") or ()))
		parameters = parameter_symbols(len(grounded_args))
		internal_name = query_root_alias_task_name(index, source_task_name)
		headline_literal = parameterise_literal_from_grounded_args(
			target_literal,
			grounded_args,
			parameters,
		)
		query_root_alias_tasks.append(
			{
				"index": index,
				"name": internal_name,
				"source_name": source_task_name,
				"parameters": list(parameters),
				"headline_literal": headline_literal.to_dict(),
				"target_literal": target_literal.to_signature(),
				"grounded_args": list(grounded_args),
			},
		)
		target_task_bindings.append(
			{
				"target_literal": target_literal.to_signature(),
				"task_name": internal_name,
			},
		)

	transition_tasks: List[Dict[str, Any]] = []
	for spec in transition_specs:
		literal = spec["literal"]
		parameters = parameter_symbols(len(literal.args))
		headline_literal = HTNLiteral(
			predicate=literal.predicate,
			args=parameters,
			is_positive=literal.is_positive,
			source_symbol=None,
			negation_mode=literal.negation_mode,
		)
		transition_tasks.append(
			{
				"name": str(spec["transition_name"]),
				"parameters": list(parameters),
				"headline_literal": headline_literal.to_dict(),
				"target_literal": literal.to_signature(),
				"raw_label": str(spec.get("raw_label", "")).strip(),
				"source_state": str(spec.get("source_state", "")).strip(),
				"target_state": str(spec.get("target_state", "")).strip(),
				"accepting_states": list(spec.get("accepting_states", ()) or ()),
			},
		)

	return {
		"transition_native": True,
		"query_root_alias_tasks": query_root_alias_tasks,
		"transition_tasks": transition_tasks,
		"target_task_bindings": target_task_bindings,
	}


def build_transition_native_system_prompt() -> str:
	return (
		"You generate a compact, executable HTN method library in one shot.\n"
		"The internal compound-task skeleton is fixed in advance and is listed in the user prompt.\n"
		"Do not invent extra top-level query-root aliases or omit any required transition task.\n"
		"Return JSON only. No markdown, no comments, no prose.\n"
		"Return exactly one JSON object with exactly these top-level keys: "
		"target_task_bindings, tasks.\n"
		"Each tasks[] entry must be one HDDL-shaped compound task object with: "
		"name, parameters, headline, optional source_name, optional noop, optional constructive.\n"
		"The noop branch stands for an already-satisfied method with precondition/context only and no subtasks.\n"
		"Every query-root alias task must include a noop branch whose precondition/context is exactly its "
		"headline literal, so the bound task itself still denotes the already-satisfied target.\n"
		"Each constructive branch stands for one method with: optional label, optional parameters, "
		"optional precondition, and ordered_subtasks or compact support_before/producer/followup fields.\n"
		"Canonical subtask form inside ordered_subtasks is either an invocation string like "
		"\"stack(ARG1, ARG2)\" or an object {\"call\":\"stack\",\"args\":[\"ARG1\",\"ARG2\"]}.\n"
		"Use compact support_before/producer/followup only at the branch level, not nested inside "
		"ordered_subtasks.\n"
		"If a branch-level producer is a bare task name with no explicit arguments, it means a single "
		"subtask call on the task's own ARG order.\n"
		"If you choose a listed producer/support mode with extra roles, copy every listed dynamic need "
		"verbatim; never transpose ARG and AUX positions inside a required relation.\n"
		"Support obligations recurse: if a chosen support mode still has listed dynamic needs, keep "
		"supporting those needs until the first primitive step is applicable.\n"
		"Every referenced compound subtask must also appear in tasks[].\n"
		"Primitive steps must use the provided runtime primitive aliases.\n"
		"Only query-root alias tasks may carry source_name, and source_name must equal the official "
		"query-mentioned root task name exactly.\n"
		"Transition tasks and helper tasks must not carry source_name.\n"
		"Do not use grounded object names as method parameters. Use only schematic ARG/AUX variables.\n"
		"Fresh helper tasks are allowed only for dynamic predicates that can be established by primitive actions.\n"
		"Preserve every required target_task_binding exactly as listed.\n"
	)


def build_transition_native_user_prompt(
	domain: Any,
	*,
	query_text: str,
	target_literals: Sequence[HTNLiteral],
	query_task_anchors: Sequence[Dict[str, Any]],
	transition_specs: Sequence[Dict[str, Any]],
	prompt_analysis: Dict[str, Any],
	action_analysis: Optional[Dict[str, Any]] = None,
	query_object_inventory: Sequence[Dict[str, Any]] = (),
) -> str:
	predicate_lines = [
		f"- {predicate.to_signature()}"
		for predicate in getattr(domain, "predicates", [])
	]
	action_lines = [
		(
			f"- runtime task: {sanitize_identifier(action.name)} | source action: {action.name}"
			f"({', '.join(action.parameters) if action.parameters else 'none'}) | "
			f"pre: {action.preconditions} | eff: {action.effects}"
		)
		for action in getattr(domain, "actions", [])
	]
	target_lines = [
		f"- #{index}: {literal.to_signature()}"
		for index, literal in enumerate(target_literals, start=1)
	]
	query_anchor_lines = [
		f"- #{index}: {anchor.get('task_name')}({', '.join(anchor.get('args') or [])})"
		for index, anchor in enumerate(query_task_anchors, start=1)
	]
	query_root_lines = [
		(
			f"- {item['name']} | source_name={item['source_name']} | "
			f"parameters=({', '.join(item['parameters'])}) | "
			f"headline={_dict_literal_signature(item['headline_literal'])}"
		)
		for item in prompt_analysis.get("query_root_alias_tasks", [])
	]
	transition_lines = [
		(
			f"- {item['name']} | states={item['source_state']}->{item['target_state']} | "
			f"parameters=({', '.join(item['parameters'])}) | "
			f"headline={_dict_literal_signature(item['headline_literal'])}"
		)
		for item in prompt_analysis.get("transition_tasks", [])
	]
	binding_lines = [
		f'- {{"target_literal":"{item["target_literal"]}","task_name":"{item["task_name"]}"}}'
		for item in prompt_analysis.get("target_task_bindings", [])
	]
	object_inventory_lines = [
		f"- {entry['type']}: {', '.join(entry['objects'])}"
		for entry in query_object_inventory
	]
	transition_contract_lines = _transition_support_contract_lines(
		prompt_analysis=prompt_analysis,
		action_analysis=action_analysis or {},
	)
	helper_candidate_lines = _helper_task_candidate_lines(
		prompt_analysis=prompt_analysis,
		action_analysis=action_analysis or {},
	)

	return (
		f"NATURAL LANGUAGE QUERY:\n{query_text.strip() or '- none'}\n\n"
		"QUERY ROOT INVOCATIONS:\n"
		+ ("\n".join(query_anchor_lines) if query_anchor_lines else "- none")
		+ "\n\n"
		"TARGET LITERALS:\n"
		+ ("\n".join(target_lines) if target_lines else "- none")
		+ "\n\n"
		"REQUIRED TOP-LEVEL QUERY-ROOT ALIAS TASKS:\n"
		+ ("\n".join(query_root_lines) if query_root_lines else "- none")
		+ "\n\n"
		"REQUIRED TRANSITION TASKS:\n"
		+ ("\n".join(transition_lines) if transition_lines else "- none")
		+ "\n\n"
		"REQUIRED target_task_bindings:\n"
		+ ("\n".join(binding_lines) if binding_lines else "- none")
		+ "\n\n"
		"PREDICATES:\n"
		+ ("\n".join(predicate_lines) if predicate_lines else "- none")
		+ "\n\n"
		"PRIMITIVE ACTION SCHEMAS:\n"
		+ ("\n".join(action_lines) if action_lines else "- none")
		+ "\n\n"
		"QUERY OBJECT INVENTORY:\n"
		+ ("\n".join(object_inventory_lines) if object_inventory_lines else "- none")
		+ "\n\n"
		"TRANSITION SUPPORT CONTRACTS:\n"
		+ ("\n".join(transition_contract_lines) if transition_contract_lines else "- none")
		+ "\n\n"
		"ALLOWED HELPER TASK CANDIDATES:\n"
		+ ("\n".join(helper_candidate_lines) if helper_candidate_lines else "- none")
		+ "\n\n"
		"CANONICAL JSON SHAPES:\n"
		'- query-root alias task: {"noop":{"precondition":["HEADLINE(ARG1, ARG2)"]},"constructive":[{"ordered_subtasks":[{"call":"REQUIRED_TRANSITION_TASK","args":["ARG1","ARG2"]}]}]}\n'
		'- transition constructive branch: {"precondition":["NEED1","NEED2"],"ordered_subtasks":[{"call":"RUNTIME_OR_HELPER_TASK","args":["ARG1"]},{"call":"RUNTIME_OR_HELPER_TASK","args":["ARG1","ARG2"]}]}\n'
		'- compact branch-level shortcut is allowed only as: {"support_before":["helper(ARG1)"],"producer":"runtime_or_transition(ARG1, ARG2)","followup":"cleanup(ARG1)"}\n'
		'- inside ordered_subtasks, do not use nested support_before/producer/followup; use invocation strings or call/args objects only.\n'
		'- if a query-root alias directly wraps one transition task, preserve the same ARG order in that child call.\n\n'
		'- if a listed support mode is unstack(AUX1, ARG2) [needs on(AUX1, ARG2), clear(AUX1), handempty], keep on(AUX1, ARG2) exactly; do not transpose it to on(ARG2, AUX1).\n\n'
		"CONSTRAINTS:\n"
		"- Emit every required query-root alias task exactly once in tasks[].\n"
		"- Emit every required transition task exactly once in tasks[].\n"
		"- Every query-root alias task must have source_name and must decompose only into required transition tasks or fresh dynamic helpers.\n"
		"- Every query-root alias task must include a noop branch whose precondition/context is exactly its listed headline literal.\n"
		"- Every transition task must have the exact listed headline and no source_name.\n"
		"- If a transition or query-root alias is already satisfied, use a noop branch with that headline in precondition/context.\n"
		"- Constructive branches must support every primitive dynamic precondition before the producer action.\n"
		"- For every transition task, choose a listed direct producer mode and satisfy every listed dynamic need before that producer executes.\n"
		"- When you choose a listed producer/support mode, copy its listed dynamic needs with the same argument order; do not flip ARG* and AUX* inside a required relation.\n"
		"- Apply the same rule recursively to support modes: if your chosen support mode still needs another dynamic predicate, create or call a helper task for that need before the support mode runs.\n"
		"- If a dynamic need is not already true and no existing query-root or transition task headlines it, prefer one listed helper-task candidate with the exact same headline literal instead of assuming the need.\n"
		"- If a helper is needed, its headline must be a dynamic predicate and it must also appear in tasks[].\n"
		"- Use the shortest valid JSON you can.\n"
	)


def _dict_literal_signature(payload: Dict[str, Any]) -> str:
	predicate = str(payload.get("predicate", "")).strip()
	args = [str(arg).strip() for arg in payload.get("args", []) if str(arg).strip()]
	if args:
		return f"{predicate}({', '.join(args)})"
	return predicate


def _transition_support_contract_lines(
	*,
	prompt_analysis: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> List[str]:
	lines: List[str] = []
	for transition_task in prompt_analysis.get("transition_tasks", []):
		headline_payload = transition_task.get("headline_literal") or {}
		headline_literal = HTNLiteral(
			predicate=str(headline_payload.get("predicate", "")).strip(),
			args=tuple(str(arg).strip() for arg in headline_payload.get("args", []) if str(arg).strip()),
			is_positive=bool(headline_payload.get("is_positive", True)),
			source_symbol=None,
			negation_mode=str(headline_payload.get("negation_mode", "naf") or "naf"),
		)
		if not headline_literal.predicate or not headline_literal.is_positive:
			continue
		mode_options = _render_producer_mode_options_for_predicate(
			headline_literal.predicate,
			headline_literal.args,
			action_analysis,
			limit=3,
		)
		if not mode_options:
			continue
		rendered_modes: List[str] = []
		requirement_queue: deque[Tuple[str, int]] = deque()
		seen_requirement_signatures: set[str] = set()
		for mode_call, needs in mode_options:
			needs = tuple(str(value).strip() for value in (needs or ()) if str(value).strip())
			if needs:
				rendered_modes.append(f"{mode_call} [needs {', '.join(needs)}]")
			else:
				rendered_modes.append(f"{mode_call} [needs none]")
			for signature in needs:
				if signature not in seen_requirement_signatures:
					seen_requirement_signatures.add(signature)
					requirement_queue.append((signature, 1))
		lines.append(
			f"- {transition_task['name']}: headline {_dict_literal_signature(headline_payload)}. "
			f"Direct producer modes: {'; '.join(rendered_modes)}."
		)
		support_line_budget = 0
		while requirement_queue and support_line_budget < 8:
			requirement_signature, depth = requirement_queue.popleft()
			parsed_requirement = _parse_signature_literal(requirement_signature)
			if parsed_requirement is None or not parsed_requirement.is_positive:
				continue
			support_modes = _render_producer_mode_options_for_predicate(
				parsed_requirement.predicate,
				parsed_requirement.args,
				action_analysis,
				limit=3,
			)
			if support_modes:
				rendered_support_modes = []
				for support_call, support_needs in support_modes:
					support_needs = tuple(
						str(value).strip()
						for value in (support_needs or ())
						if str(value).strip()
					)
					if support_needs:
						rendered_support_modes.append(
							f"{support_call} [needs {', '.join(support_needs)}]"
						)
					else:
						rendered_support_modes.append(f"{support_call} [needs none]")
					if depth < 2:
						for nested_signature in support_needs:
							if nested_signature not in seen_requirement_signatures:
								seen_requirement_signatures.add(nested_signature)
								requirement_queue.append((nested_signature, depth + 1))
				lines.append(
					f"  Support {requirement_signature} with one of: "
					f"{'; '.join(rendered_support_modes)}."
				)
				support_line_budget += 1
			else:
				lines.append(
					f"  If {requirement_signature} is not already true, add one helper task whose "
					f"headline is exactly {requirement_signature} and support it before the final producer."
				)
				support_line_budget += 1
	return lines


def _helper_task_candidate_lines(
	*,
	prompt_analysis: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> List[str]:
	seen_signatures: set[str] = set()
	lines: List[str] = []
	for helper_literal in _recommended_helper_literals(
		prompt_analysis=prompt_analysis,
		action_analysis=action_analysis,
	):
		signature = helper_literal.to_signature()
		if signature in seen_signatures:
			continue
		seen_signatures.add(signature)
		parameters = [str(arg).strip() for arg in helper_literal.args if str(arg).strip()]
		name = f"helper_{sanitize_identifier('_'.join((helper_literal.predicate, *parameters)))}"
		lines.append(
			f"- {name} | headline={signature} | parameters=({', '.join(parameters) if parameters else 'none'})"
		)
	return lines


def _recommended_helper_literals(
	*,
	prompt_analysis: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> Tuple[HTNLiteral, ...]:
	queue: deque[Tuple[HTNLiteral, int]] = deque()
	seen_signatures: set[str] = set()
	helper_literals: List[HTNLiteral] = []

	for transition_task in prompt_analysis.get("transition_tasks", []):
		headline_payload = transition_task.get("headline_literal") or {}
		headline_literal = HTNLiteral(
			predicate=str(headline_payload.get("predicate", "")).strip(),
			args=tuple(str(arg).strip() for arg in headline_payload.get("args", []) if str(arg).strip()),
			is_positive=bool(headline_payload.get("is_positive", True)),
			source_symbol=None,
			negation_mode=str(headline_payload.get("negation_mode", "naf") or "naf"),
		)
		if not headline_literal.predicate or not headline_literal.is_positive:
			continue
		for _, needs in _render_producer_mode_options_for_predicate(
			headline_literal.predicate,
			headline_literal.args,
			action_analysis,
			limit=3,
		):
			for signature in needs:
				parsed = _parse_signature_literal(signature)
				if parsed is None or not parsed.is_positive:
					continue
				queue.append((parsed, 1))

	while queue:
		literal, depth = queue.popleft()
		signature = literal.to_signature()
		if signature in seen_signatures:
			continue
		seen_signatures.add(signature)
		helper_literals.append(
			HTNLiteral(
				predicate=literal.predicate,
				args=tuple(str(arg).strip() for arg in literal.args if str(arg).strip()),
				is_positive=True,
				source_symbol=None,
				negation_mode=literal.negation_mode,
			),
		)
		if depth >= 2:
			continue
		for _, nested_needs in _render_producer_mode_options_for_predicate(
			literal.predicate,
			literal.args,
			action_analysis,
			limit=3,
		):
			for nested_signature in nested_needs:
				nested_literal = _parse_signature_literal(nested_signature)
				if nested_literal is None or not nested_literal.is_positive:
					continue
				queue.append((nested_literal, depth + 1))

	return tuple(helper_literals)
