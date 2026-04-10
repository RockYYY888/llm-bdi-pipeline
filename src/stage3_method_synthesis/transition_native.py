"""
Transition-native Stage 3 contracts and prompt builders.

This module defines the deterministic task skeleton used by the redesigned
Stage 3 pipeline:
- query-root alias tasks preserve the official root-task interface only for
  Stage 6 and Stage 7 compatibility,
- internal synthesis tasks are the raw DFA progress transitions themselves.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Tuple

from stage3_method_synthesis.htn_prompts import (
	_is_aux_binding_requirement,
	_extend_mapping_with_action_parameters,
	_filter_dominated_producer_modes,
	_render_positive_dynamic_requirements,
	_render_positive_static_requirements,
	_render_producer_mode_options_for_predicate,
	_render_signature_with_mapping,
	_same_arity_caller_shared_requirements,
	_task_invocation_signature,
)
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


def context_parameter_symbols(arity: int) -> Tuple[str, ...]:
	return tuple(f"AUX_CTX{index}" for index in range(1, int(arity) + 1))


def helper_task_name_for_literal(literal: HTNLiteral) -> str:
	return f"helper_{sanitize_identifier(literal.predicate)}"


def canonicalise_helper_literal(literal: HTNLiteral) -> HTNLiteral:
	formal_parameters = parameter_symbols(len(tuple(literal.args or ())))
	return HTNLiteral(
		predicate=literal.predicate,
		args=formal_parameters,
		is_positive=literal.is_positive,
		source_symbol=None,
		negation_mode=literal.negation_mode,
	)


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
	ordered_target_payloads: List[Dict[str, Any]] = []

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
		ordered_target_payloads.append(
			{
				"index": index,
				"literal": target_literal,
				"grounded_args": tuple(grounded_args),
				"target_literal": target_literal.to_signature(),
			},
		)
		target_task_bindings.append(
			{
				"target_literal": target_literal.to_signature(),
				"task_name": internal_name,
			},
		)

	ordered_linear_chain = (
		len(transition_specs) == len(ordered_target_payloads)
		and all(
			spec["literal"].to_signature() == ordered_target_payloads[index]["target_literal"]
			for index, spec in enumerate(transition_specs)
		)
	)

	transition_tasks: List[Dict[str, Any]] = []
	for spec_index, spec in enumerate(transition_specs, start=1):
		literal = spec["literal"]
		literal_signature = literal.to_signature()
		occurrence_index = spec_index if ordered_linear_chain else None
		prior_literals = tuple(
			payload["literal"]
			for payload in ordered_target_payloads
			if occurrence_index is not None and int(payload["index"]) < occurrence_index
		)
		parameters, symbol_map = _transition_parameter_context(
			current_literal=literal,
			prior_literals=prior_literals,
		)
		headline_literal = HTNLiteral(
			predicate=literal.predicate,
			args=tuple(
				str(symbol_map.get(str(argument).strip(), str(argument).strip()))
				for argument in literal.args
			),
			is_positive=literal.is_positive,
			source_symbol=None,
			negation_mode=literal.negation_mode,
		)
		retained_prefix_literals = [
			_parameterise_literal_with_symbol_map(previous_literal, symbol_map).to_signature()
			for previous_literal in prior_literals
		]
		retained_prefix_grounded_literals = [
			previous_literal.to_signature()
			for previous_literal in prior_literals
		]
		transition_tasks.append(
			{
				"name": str(spec["transition_name"]),
				"parameters": list(parameters),
				"headline_literal": headline_literal.to_dict(),
				"target_literal": literal.to_signature(),
				"grounded_args": list(literal.args),
				"retained_prefix_literals": retained_prefix_literals,
				"retained_prefix_grounded_literals": retained_prefix_grounded_literals,
				"query_root_index": occurrence_index,
				"raw_label": str(spec.get("raw_label", "")).strip(),
				"source_state": str(spec.get("source_state", "")).strip(),
				"target_state": str(spec.get("target_state", "")).strip(),
				"accepting_states": list(spec.get("accepting_states", ()) or ()),
			},
		)

	transition_by_index = {
		int(payload["query_root_index"]): payload
		for payload in transition_tasks
		if payload.get("query_root_index") is not None
	}
	for payload in query_root_alias_tasks:
		transition_payload = transition_by_index.get(int(payload["index"]))
		if transition_payload is None:
			continue
		transition_parameters = tuple(
			str(value).strip()
			for value in (transition_payload.get("parameters") or ())
			if str(value).strip()
		)
		payload["bridge_parameters"] = list(transition_parameters)
		payload["bridge_precondition"] = list(
			transition_payload.get("retained_prefix_literals") or (),
		)

	return {
		"transition_native": True,
		"query_root_alias_tasks": query_root_alias_tasks,
		"transition_tasks": transition_tasks,
		"target_task_bindings": target_task_bindings,
	}


def build_transition_native_system_prompt() -> str:
	return (
		"You generate one executable HTN method library in JSON.\n"
		"Return minified JSON only. No markdown. No comments. No prose.\n"
		"Return exactly one object with top-level keys: target_task_bindings, tasks.\n"
		"Treat every constructive branch as one HDDL-style :method body.\n"
		"Use the provided contracts as authoritative compiled obligations. "
		"Do not infer new helper choices, new ordering logic, or new task structure from "
		"the natural-language query.\n"
		"Emit every required query_root_* and dfa_step_* task exactly once. "
		"Use [] for target_task_bindings unless you intentionally need non-default bindings.\n"
		"Use compiler-owned task defaults unless you must override them. "
		"Do not restate default task headers, headline literals, source_name, or noop branches. "
		"Only query_root_* may set source_name.\n"
		"Each task object may contain only: name, optional parameters, optional source_name, "
		"optional precondition, optional noop, required constructive.\n"
		"Each constructive branch may contain only: optional parameters, optional precondition, "
		"required ordered_subtasks.\n"
		"Do not use task-level ordered_subtasks as a branch shorthand. "
		"Do not use support_before, producer, produce, followup, followups, steps, subtasks, "
		"ordering, orderings, or ordering_edges anywhere in the output.\n"
		"Do not merge required branches. If a contract lists N constructive branches for a task, "
		"emit exactly N constructive branch objects for that task.\n"
		"Each query_root_* task is a pure administrative alias bridge to required dfa_step_* tasks. "
		"Never give query_root_* any helper step or primitive action. "
		"Each query_root_* constructive branch must contain exactly one dfa_step_* call. "
		"Never inline helper_* calls, primitive actions, or the body of a dfa_step_* branch inside "
		"query_root_*.\n"
		"If a transition task must preserve earlier ordered goals, use the branch ordered_subtasks "
		"exactly as listed in REQUIRED BRANCH CONTRACTS JSON. That ordered list already encodes the "
		"staged support order: stable preparation, retained-prefix restoration, volatile final support, "
		"then final producer.\n"
		"If a task has compiler-owned default precondition literals, treat them as fixed context. "
		"You may omit restating them, but you must not delete, weaken, or replace them.\n"
		"Only if REQUIRED BRANCH CONTRACTS JSON explicitly lists query_root_* precondition literals "
		"may you restate them. Never add the query_root headline literal or any other extra dynamic predicate "
		"to a query_root constructive precondition/context. If the listed query_root precondition is empty, "
		"omit precondition entirely.\n"
		"If a branch introduces AUX* symbols, every AUX* must be justified by a genuine AUX witness-binding "
		"literal or by the exact ordered_subtasks contract that establishes it.\n"
		"When TASK INVENTORY JSON lists parameter_types for a task, those types are authoritative exact slot types. "
		"Do not reuse an ARG slot as a different type, even if a predicate supertype would allow it.\n"
		"Call every compound child with exactly the task-header parameters listed in TASK INVENTORY JSON. "
		"Branch-local parameters are local existential witnesses for that task's own method branch only; "
		"never pass a branch-local AUX symbol from a parent call unless it also appears in the callee's "
		"task-header parameter list.\n"
		"Any referenced helper_* or other compound child must also appear in tasks[]. "
		"Primitive steps must use the listed runtime primitive aliases.\n"
		"Use only schematic ARG* / AUX* symbols, preserve ARG/AUX argument order, and do not use "
		"grounded objects as task parameters.\n"
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
	action_payload = [
		sanitize_identifier(action.name)
		for action in getattr(domain, "actions", [])
	]
	required_branch_contracts = dict(prompt_analysis.get("required_branch_contracts", {}) or {})
	if not required_branch_contracts:
		required_branch_contracts = build_transition_native_required_branch_contracts(
			prompt_analysis=prompt_analysis,
			action_analysis=action_analysis or {},
		)
	compiler_defaults = build_transition_native_ast_compiler_defaults(
		prompt_analysis=prompt_analysis,
		action_analysis=action_analysis or {},
	)
	branch_contracts_payload = _prompt_required_branch_contracts_payload(
		required_branch_contracts=required_branch_contracts,
		task_defaults=compiler_defaults.get("task_defaults", {}) or {},
	)
	task_inventory_payload = _prompt_task_defaults_payload(
		prompt_analysis=prompt_analysis,
		required_branch_contracts=required_branch_contracts,
		task_defaults=compiler_defaults.get("task_defaults", {}) or {},
	)
	query_root_bridge_lines = _query_root_bridge_contract_lines(
		prompt_analysis=prompt_analysis,
	)

	return (
		"TASK INVENTORY JSON:\n"
		+ json.dumps(task_inventory_payload, ensure_ascii=False, separators=(",", ":"))
		+ "\n\n"
		+ "REQUIRED QUERY_ROOT BRIDGE RULES:\n"
		+ ("\n".join(query_root_bridge_lines) if query_root_bridge_lines else "- none")
		+ "\n\n"
		+ "REQUIRED BRANCH CONTRACTS JSON:\n"
		+ json.dumps(branch_contracts_payload, ensure_ascii=False, separators=(",", ":"))
		+ "\n\n"
		+ "ALLOWED PRIMITIVE ACTIONS JSON:\n"
		+ json.dumps(action_payload, ensure_ascii=False, separators=(",", ":"))
		+ "\n\n"
		+ f"QUERY REFERENCE ONLY:\n{query_text.strip() or '- none'}\n\n"
		+ "OUTPUT SHAPE EXAMPLES:\n"
		+ '- {"target_task_bindings":[],"tasks":[...]}\n'
		+ '- {"name":"query_root_*","constructive":[{"ordered_subtasks":["dfa_step_q1_q2_goal(ARG1)"]}]}\n'
		+ '- {"name":"dfa_step_*","constructive":[{"ordered_subtasks":["helper_x(ARG1)","primitive_or_helper(ARG1, ARG2)"]}]}\n'
		+ '- {"name":"helper_clear","constructive":[{"parameters":["ARG1","AUX1"],"precondition":["on(AUX1, ARG1)"],"ordered_subtasks":["helper_clear(AUX1)","helper_handempty","unstack(AUX1, ARG1)"]}]}\n'
		+ "HARD PROHIBITIONS:\n"
		+ "- query_root_* must never inline helper_* or primitive actions; only one dfa_step_* call per constructive branch\n"
		+ "- do not copy a dfa_step_* ordered_subtasks body into query_root_*\n"
	)


def _prompt_task_defaults_payload(
	*,
	prompt_analysis: Dict[str, Any],
	required_branch_contracts: Dict[str, List[Dict[str, Any]]],
	task_defaults: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
	ordered_task_names: List[str] = []
	seen_names: set[str] = set()
	for section_name in ("query_root_alias_tasks", "transition_tasks"):
		for payload in prompt_analysis.get(section_name, []) or ():
			task_name = str(payload.get("name", "")).strip()
			if task_name and task_name not in seen_names:
				seen_names.add(task_name)
				ordered_task_names.append(task_name)
	for task_name in sorted(required_branch_contracts):
		if task_name not in seen_names:
			seen_names.add(task_name)
			ordered_task_names.append(task_name)

	task_entries: List[Dict[str, Any]] = []
	for task_name in ordered_task_names:
		defaults = dict(task_defaults.get(task_name) or {})
		if not defaults:
			continue
		task_entry: Dict[str, Any] = {
			"name": task_name,
			"parameters": list(defaults.get("parameters", ()) or ()),
		}
		parameter_types = [
			str(value).strip()
			for value in (defaults.get("parameter_types") or ())
			if str(value).strip()
		]
		if parameter_types:
			task_entry["parameter_types"] = parameter_types
		source_name = str(defaults.get("source_name", "")).strip()
		if source_name:
			task_entry["source_name"] = source_name
		task_entries.append(task_entry)
	return task_entries


def _prompt_required_branch_contracts_payload(
	*,
	required_branch_contracts: Dict[str, List[Dict[str, Any]]],
	task_defaults: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
	ordered_payload: Dict[str, List[Dict[str, Any]]] = {}
	for task_name in sorted(required_branch_contracts):
		task_default_parameters = [
			str(value).strip()
			for value in (task_defaults.get(task_name, {}).get("parameters") or ())
			if str(value).strip()
		]
		branch_payloads: List[Dict[str, Any]] = []
		for branch in required_branch_contracts.get(task_name, []) or ():
			payload: Dict[str, Any] = {}
			parameters = [str(value).strip() for value in (branch.get("parameters") or ()) if str(value).strip()]
			precondition = [str(value).strip() for value in (branch.get("precondition") or ()) if str(value).strip()]
			ordered_subtasks = [str(value).strip() for value in (branch.get("ordered_subtasks") or ()) if str(value).strip()]
			if parameters and parameters != task_default_parameters:
				payload["parameters"] = parameters
			if precondition:
				payload["precondition"] = precondition
			payload["ordered_subtasks"] = ordered_subtasks
			branch_payloads.append(payload)
		ordered_payload[task_name] = branch_payloads
	return ordered_payload


def build_transition_native_ast_compiler_defaults(
	*,
	prompt_analysis: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> Dict[str, Any]:
	task_defaults: Dict[str, Dict[str, Any]] = {}
	primitive_aliases: list[str] = []
	seen_primitive_aliases: set[str] = set()
	call_arities: Dict[str, int] = {}
	for item in prompt_analysis.get("query_root_alias_tasks", []):
		headline = _dict_literal_signature(item.get("headline_literal") or {})
		task_name = str(item["name"]).strip()
		parameters = list(item.get("parameters") or [])
		task_defaults[task_name] = {
			"name": task_name,
			"parameters": parameters,
			"headline": headline,
			"source_name": str(item.get("source_name", "")).strip(),
			"noop": headline,
			"grounded_args": list(item.get("grounded_args") or []),
		}
		if item.get("parameter_types"):
			task_defaults[task_name]["parameter_types"] = list(item.get("parameter_types") or [])
		bridge_precondition = [
			str(value).strip()
			for value in (item.get("bridge_precondition") or ())
			if str(value).strip()
		]
		if bridge_precondition:
			task_defaults[task_name]["precondition"] = bridge_precondition
		call_arities[task_name] = len(parameters)
	for item in prompt_analysis.get("transition_tasks", []):
		headline = _dict_literal_signature(item.get("headline_literal") or {})
		task_name = str(item["name"]).strip()
		parameters = list(item.get("parameters") or [])
		retained_prefix_literals = [
			str(value).strip()
			for value in (item.get("retained_prefix_literals") or ())
			if str(value).strip()
		]
		noop_contract: str | List[str]
		if retained_prefix_literals:
			noop_contract = [*retained_prefix_literals, headline]
		else:
			noop_contract = headline
		task_defaults[task_name] = {
			"name": task_name,
			"parameters": parameters,
			"headline": headline,
			"noop": noop_contract,
			"grounded_args": list(item.get("grounded_args") or []),
		}
		if item.get("parameter_types"):
			task_defaults[task_name]["parameter_types"] = list(item.get("parameter_types") or [])
		call_arities[task_name] = len(parameters)
	for helper_literal in _recommended_helper_literals(
		prompt_analysis=prompt_analysis,
		action_analysis=action_analysis,
	):
		canonical_literal = canonicalise_helper_literal(helper_literal)
		headline = canonical_literal.to_signature()
		task_name = helper_task_name_for_literal(canonical_literal)
		task_defaults.setdefault(
			task_name,
			{
				"name": task_name,
				"parameters": list(canonical_literal.args),
				"headline": headline,
				"noop": headline,
			},
		)
		call_arities.setdefault(task_name, len(canonical_literal.args))
	for pattern_mapping in (
		action_analysis.get("producer_patterns_by_predicate", {}),
		action_analysis.get("consumer_patterns_by_predicate", {}),
	):
		for patterns in pattern_mapping.values():
			for pattern in patterns or ():
				action_name = str(pattern.get("action_name", "")).strip()
				if action_name and action_name not in seen_primitive_aliases:
					seen_primitive_aliases.add(action_name)
					primitive_aliases.append(action_name)
				if action_name:
					call_arities.setdefault(
						action_name,
						len(tuple(pattern.get("action_parameters") or ())),
					)
	return {
		"task_defaults": task_defaults,
		"target_task_bindings": list(prompt_analysis.get("target_task_bindings", [])),
		"primitive_aliases": primitive_aliases,
		"call_arities": call_arities,
		"strict_hddl_ast": True,
	}


def build_transition_native_required_branch_contracts(
	*,
	prompt_analysis: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
	required_contracts: Dict[str, List[Dict[str, Any]]] = {}
	transition_tasks_by_target_literal: Dict[str, List[Dict[str, Any]]] = {}
	transition_payloads_by_index: Dict[int, Dict[str, Any]] = {}
	for payload in prompt_analysis.get("transition_tasks", []):
		target_literal = str(payload.get("target_literal", "")).strip()
		if not target_literal:
			continue
		transition_tasks_by_target_literal.setdefault(target_literal, []).append(payload)
		query_root_index = payload.get("query_root_index")
		if query_root_index is None:
			continue
		try:
			transition_payloads_by_index[int(query_root_index)] = payload
		except (TypeError, ValueError):
			continue

	for payload in prompt_analysis.get("query_root_alias_tasks", []):
		task_name = str(payload.get("name", "")).strip()
		target_literal = str(payload.get("target_literal", "")).strip()
		parameters = tuple(
			str(value).strip()
			for value in (payload.get("bridge_parameters") or payload.get("parameters") or ())
			if str(value).strip()
		)
		bridge_precondition = tuple(
			str(value).strip()
			for value in (payload.get("bridge_precondition") or ())
			if str(value).strip()
		)
		if not task_name or not target_literal:
			continue
		branches: List[Dict[str, Any]] = []
		for transition_payload in transition_tasks_by_target_literal.get(target_literal, ()):
			transition_name = str(transition_payload.get("name", "")).strip()
			if not transition_name:
				continue
			invocation = (
				f"{transition_name}({', '.join(parameters)})"
				if parameters
				else transition_name
			)
			branches.append(
				_branch_contract_dict(
					parameters=parameters,
					precondition=bridge_precondition,
					ordered_subtasks=(invocation,),
				)
			)
		if branches:
			required_contracts[task_name] = branches

	for payload in prompt_analysis.get("transition_tasks", []):
		task_name = str(payload.get("name", "")).strip()
		headline_payload = payload.get("headline_literal") or {}
		headline_literal = HTNLiteral(
			predicate=str(headline_payload.get("predicate", "")).strip(),
			args=tuple(str(arg).strip() for arg in headline_payload.get("args", []) if str(arg).strip()),
			is_positive=bool(headline_payload.get("is_positive", True)),
			source_symbol=None,
			negation_mode=str(headline_payload.get("negation_mode", "naf") or "naf"),
		)
		if not task_name or not headline_literal.predicate or not headline_literal.is_positive:
			continue
		base_task_parameters = tuple(
			str(value).strip()
			for value in (payload.get("parameters") or ())
			if str(value).strip()
		)
		if not base_task_parameters:
			base_task_parameters = tuple(headline_literal.args)
		retained_prefix_literals = tuple(
			str(value).strip()
			for value in (payload.get("retained_prefix_literals") or ())
			if str(value).strip()
		)
		previous_transition_support, retained_prefix_available_literals = (
			_ordered_prefix_transition_support_contract(
			current_payload=payload,
			transition_payloads_by_index=transition_payloads_by_index,
		)
		)
		retained_prefix_support_calls = (
			(previous_transition_support,)
			if previous_transition_support
			else tuple(
				_helper_call_for_literal(parsed_literal)
				for parsed_literal in (
					_parse_signature_literal(signature)
					for signature in retained_prefix_literals
				)
				if parsed_literal is not None and parsed_literal.is_positive
			)
		)
		branches: List[Dict[str, Any]] = []
		for mode_call, needs, static_requirements in _render_producer_mode_contract_options_for_predicate(
				headline_literal.predicate,
				headline_literal.args,
				action_analysis,
				limit=3,
			):
			raw_needs = tuple(
				str(value).strip()
				for value in (needs or ())
				if str(value).strip()
			)
			needs = _filter_requirements_already_available(
				raw_needs,
				available_literals=retained_prefix_available_literals,
			)
			extra_role_symbols = _branch_extra_role_symbols(
				mode_call=mode_call,
				needs=needs,
				task_parameter_symbols=base_task_parameters,
			)
			branch_parameters = tuple(base_task_parameters) + tuple(
				symbol
				for symbol in extra_role_symbols
				if symbol not in base_task_parameters
			)
			preserved_prefix_literals = tuple(
				parsed_literal
				for parsed_literal in (
					_parse_signature_literal(signature)
					for signature in retained_prefix_literals
				)
				if parsed_literal is not None and parsed_literal.is_positive
			)
			preparatory_support_calls = _dedupe_invocation_signatures(
				_prepare_support_calls_for_preserved_prefix(
					requirement_signatures=needs,
					task_parameter_symbols=base_task_parameters,
					preserved_literals=preserved_prefix_literals,
					action_analysis=action_analysis,
				),
			)
			support_calls, residual_context, restabilise_calls = _support_plan_contract_for_requirements(
				needs,
				task_parameter_symbols=base_task_parameters,
				extra_role_symbols=extra_role_symbols,
				preserved_literals=preserved_prefix_literals,
				action_analysis=action_analysis,
				current_headline_literal=headline_literal,
				preserve_aux_binding_context=False,
			)
			realised_helper_calls = set(preparatory_support_calls) | set(support_calls)
			same_arity_support_preconditions = _dedupe_invocation_signatures(
				tuple(
					signature
					for requirement_signature in needs
					for signature in _same_arity_positive_precondition_envelope(
						requirement_signature,
						action_analysis=action_analysis,
					)
					if (
						(parsed_requirement := _parse_signature_literal(requirement_signature)) is not None
						and _helper_call_for_literal(parsed_requirement) in realised_helper_calls
					)
				),
			)
			retained_prefix_support = _dedupe_invocation_signatures(
				retained_prefix_support_calls,
			)
			branch_precondition = _dedupe_invocation_signatures(
				(*same_arity_support_preconditions, *static_requirements, *residual_context)
				if retained_prefix_support
				else (
					*same_arity_support_preconditions,
					*static_requirements,
					*retained_prefix_literals,
					*residual_context,
				),
			)
			ordered_subtasks = (
				*preparatory_support_calls,
				*support_calls,
				*retained_prefix_support,
				*restabilise_calls,
				mode_call,
			)
			branches.append(
				_branch_contract_dict(
					parameters=branch_parameters,
					precondition=branch_precondition,
					ordered_subtasks=ordered_subtasks,
				)
			)
		if branches:
			required_contracts[task_name] = branches

	for helper_literal in _recommended_helper_literals(
		prompt_analysis=prompt_analysis,
		action_analysis=action_analysis,
	):
		canonical_literal = canonicalise_helper_literal(helper_literal)
		task_name = helper_task_name_for_literal(canonical_literal)
		branches: List[Dict[str, Any]] = []
		for mode_call, needs, static_requirements in _render_producer_mode_contract_options_for_predicate(
				canonical_literal.predicate,
				canonical_literal.args,
				action_analysis,
				limit=3,
			):
			needs = tuple(str(value).strip() for value in (needs or ()) if str(value).strip())
			extra_role_symbols = _branch_extra_role_symbols(
				mode_call=mode_call,
				needs=needs,
				task_parameter_symbols=canonical_literal.args,
			)
			support_calls, residual_context, restabilise_calls = _support_plan_contract_for_requirements(
				needs,
				task_parameter_symbols=canonical_literal.args,
				extra_role_symbols=extra_role_symbols,
				action_analysis=action_analysis,
				current_headline_literal=canonical_literal,
			)
			branches.append(
				_branch_contract_dict(
					parameters=(*canonical_literal.args, *extra_role_symbols),
					precondition=(*static_requirements, *residual_context),
					ordered_subtasks=(*support_calls, *restabilise_calls, mode_call),
				)
			)
		if branches:
			required_contracts[task_name] = branches

	return required_contracts


def _dict_literal_signature(payload: Dict[str, Any]) -> str:
	predicate = str(payload.get("predicate", "")).strip()
	args = [str(arg).strip() for arg in payload.get("args", []) if str(arg).strip()]
	if args:
		return f"{predicate}({', '.join(args)})"
	return predicate


def _helper_call_for_literal(literal: HTNLiteral) -> str:
	args = ", ".join(str(arg).strip() for arg in literal.args if str(arg).strip())
	helper_name = helper_task_name_for_literal(literal)
	return f"{helper_name}({args})" if args else helper_name


def _render_producer_mode_contract_options_for_predicate(
	predicate_name: str,
	predicate_args: Sequence[str],
	action_analysis: Dict[str, Any],
	*,
	limit: int = 3,
) -> Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...]:
	rendered_modes: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
	seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
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
		rendered_dynamic_requirements = tuple(
			_render_positive_dynamic_requirements(pattern, token_mapping)
		)
		if target_signature in rendered_dynamic_requirements:
			continue
		rendered_static_requirements = tuple(
			_render_positive_static_requirements(pattern, token_mapping)
		)
		mode = (
			_task_invocation_signature(pattern["action_name"], rendered_action_args),
			rendered_dynamic_requirements,
			rendered_static_requirements,
		)
		if mode in seen:
			continue
		seen.add(mode)
		rendered_modes.append(mode)
	filtered_dynamic_modes = {
		(mode_call, requirements)
		for mode_call, requirements in _filter_dominated_producer_modes(
			[(mode_call, requirements) for mode_call, requirements, _ in rendered_modes]
		)
	}
	filtered_modes = [
		mode
		for mode in rendered_modes
		if (mode[0], mode[1]) in filtered_dynamic_modes
	]
	if limit > 0:
		filtered_modes = filtered_modes[:limit]
	return tuple(filtered_modes)


def _same_arity_positive_precondition_envelope(
	requirement_signature: str,
	*,
	action_analysis: Dict[str, Any],
) -> Tuple[str, ...]:
	parsed_requirement = _parse_signature_literal(requirement_signature)
	if parsed_requirement is None or not parsed_requirement.is_positive:
		return ()
	task_parameters = tuple(str(arg).strip() for arg in parsed_requirement.args if str(arg).strip())
	if not task_parameters:
		return ()
	headline_signature = parsed_requirement.to_signature()
	requirement_sets: List[set[str]] = []
	for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
		parsed_requirement.predicate,
		[],
	):
		effect_args = [
			str(value).strip()
			for value in (pattern.get("effect_args") or ())
			if str(value).strip()
		]
		if len(effect_args) != len(task_parameters):
			continue
		token_mapping = {
			token: arg
			for token, arg in zip(effect_args, task_parameters)
		}
		requirements: set[str] = set()
		for rendered_signature in _render_positive_static_requirements(pattern, token_mapping):
			rendered_literal = _parse_signature_literal(rendered_signature)
			if rendered_literal is None or not rendered_literal.is_positive:
				continue
			if rendered_literal.to_signature() == headline_signature:
				continue
			if rendered_literal.args and not set(rendered_literal.args).issubset(set(task_parameters)):
				continue
			requirements.add(rendered_literal.to_signature())
		requirement_sets.append(requirements)
	if not requirement_sets:
		return ()
	return tuple(sorted(set.intersection(*requirement_sets)))


def _transition_payload_symbolic_grounded_pairs(
	payload: Dict[str, Any],
) -> Tuple[Tuple[str, HTNLiteral], ...]:
	pairs: List[Tuple[str, HTNLiteral]] = []
	retained_symbolic = [
		str(value).strip()
		for value in (payload.get("retained_prefix_literals") or ())
		if str(value).strip()
	]
	retained_grounded = [
		str(value).strip()
		for value in (payload.get("retained_prefix_grounded_literals") or ())
		if str(value).strip()
	]
	for grounded_signature, symbolic_signature in zip(retained_grounded, retained_symbolic):
		parsed_literal = _parse_signature_literal(symbolic_signature)
		if grounded_signature and parsed_literal is not None:
			pairs.append((grounded_signature, parsed_literal))
	headline_payload = payload.get("headline_literal") or {}
	headline_literal = HTNLiteral(
		predicate=str(headline_payload.get("predicate", "")).strip(),
		args=tuple(
			str(arg).strip()
			for arg in (headline_payload.get("args") or ())
			if str(arg).strip()
		),
		is_positive=bool(headline_payload.get("is_positive", True)),
		source_symbol=None,
		negation_mode=str(headline_payload.get("negation_mode", "naf") or "naf"),
	)
	target_literal = str(payload.get("target_literal", "")).strip()
	if target_literal and headline_literal.predicate and headline_literal.is_positive:
		pairs.append((target_literal, headline_literal))
	return tuple(pairs)


def _ordered_prefix_transition_support_contract(
	*,
	current_payload: Dict[str, Any],
	transition_payloads_by_index: Dict[int, Dict[str, Any]],
) -> tuple[str, tuple[HTNLiteral, ...]]:
	current_index = current_payload.get("query_root_index")
	if current_index is None:
		return "", ()
	try:
		previous_payload = transition_payloads_by_index.get(int(current_index) - 1)
	except (TypeError, ValueError):
		return "", ()
	if not previous_payload:
		return "", ()

	previous_pairs = _transition_payload_symbolic_grounded_pairs(previous_payload)
	current_pairs = {
		grounded_signature: symbolic_literal
		for grounded_signature, symbolic_literal in _transition_payload_symbolic_grounded_pairs(
			current_payload,
		)
	}
	if not previous_pairs or not current_pairs:
		return ""

	symbol_mapping: Dict[str, str] = {}
	for grounded_signature, previous_literal in previous_pairs:
		current_literal = current_pairs.get(grounded_signature)
		if current_literal is None:
			return "", ()
		if (
			previous_literal.predicate != current_literal.predicate
			or previous_literal.is_positive != current_literal.is_positive
			or len(previous_literal.args) != len(current_literal.args)
		):
			return "", ()
		for previous_arg, current_arg in zip(previous_literal.args, current_literal.args):
			previous_symbol = str(previous_arg).strip()
			current_symbol = str(current_arg).strip()
			if not previous_symbol or not current_symbol:
				return "", ()
			bound_symbol = symbol_mapping.get(previous_symbol)
			if bound_symbol is None:
				symbol_mapping[previous_symbol] = current_symbol
			elif bound_symbol != current_symbol:
				return "", ()

	previous_parameters = [
		str(value).strip()
		for value in (previous_payload.get("parameters") or ())
		if str(value).strip()
	]
	if not previous_parameters:
		return "", ()
	try:
		invocation_args = tuple(symbol_mapping[parameter] for parameter in previous_parameters)
	except KeyError:
		return "", ()
	transition_name = str(previous_payload.get("name", "")).strip()
	if not transition_name or any(not arg for arg in invocation_args):
		return "", ()

	guaranteed_literals: list[HTNLiteral] = []
	seen_signatures: set[str] = set()
	for grounded_signature, _ in previous_pairs:
		current_literal = current_pairs.get(grounded_signature)
		if current_literal is None or not current_literal.is_positive:
			continue
		signature = current_literal.to_signature()
		if not signature or signature in seen_signatures:
			continue
		seen_signatures.add(signature)
		guaranteed_literals.append(current_literal)
	return (
		f"{transition_name}({', '.join(invocation_args)})",
		tuple(guaranteed_literals),
	)


def _filter_requirements_already_available(
	requirement_signatures: Sequence[str],
	*,
	available_literals: Sequence[HTNLiteral],
) -> tuple[str, ...]:
	filtered_signatures: list[str] = []
	for requirement_signature in requirement_signatures:
		parsed_requirement = _parse_signature_literal(str(requirement_signature).strip())
		if parsed_requirement is None or not parsed_requirement.is_positive:
			filtered_signatures.append(str(requirement_signature).strip())
			continue
		if any(
			_literal_may_match_for_disturbance(
				parsed_requirement,
				available_literal,
			)
			for available_literal in available_literals
		):
			continue
		filtered_signatures.append(parsed_requirement.to_signature())
	return tuple(signature for signature in filtered_signatures if signature)


def _prepare_support_calls_for_preserved_prefix(
	*,
	requirement_signatures: Sequence[str],
	task_parameter_symbols: Collection[str],
	preserved_literals: Sequence[HTNLiteral],
	action_analysis: Optional[Dict[str, Any]],
) -> tuple[str, ...]:
	if action_analysis is None or not preserved_literals:
		return ()

	preparatory_calls: list[str] = []
	seen_calls: set[str] = set()
	for requirement_signature in requirement_signatures:
		requirement_literal = _parse_signature_literal(str(requirement_signature).strip())
		if requirement_literal is None or not requirement_literal.is_positive:
			continue
		if not any(
			_support_requirement_disturbance_score(
				support_literal=requirement_literal,
				preserved_literal=preserved_literal,
				action_analysis=action_analysis,
			) > 0
			or _support_requirement_disturbance_score(
				support_literal=preserved_literal,
				preserved_literal=requirement_literal,
				action_analysis=action_analysis,
			) > 0
			for preserved_literal in preserved_literals
		):
			continue
		for helper_call in _preparatory_mode_support_calls_for_requirement(
			requirement_literal=requirement_literal,
			task_parameter_symbols=task_parameter_symbols,
			preserved_literals=preserved_literals,
			action_analysis=action_analysis,
		):
			if helper_call in seen_calls:
				continue
			seen_calls.add(helper_call)
			preparatory_calls.append(helper_call)
	return tuple(preparatory_calls)


def _preparatory_mode_support_calls_for_requirement(
	*,
	requirement_literal: HTNLiteral,
	task_parameter_symbols: Collection[str],
	preserved_literals: Sequence[HTNLiteral],
	action_analysis: Dict[str, Any],
) -> tuple[str, ...]:
	best_calls: tuple[str, ...] = ()
	preserved_symbols = {
		str(arg).strip()
		for literal in preserved_literals
		for arg in literal.args
		if str(arg).strip()
	}
	for mode_call, mode_needs in _render_producer_mode_options_for_predicate(
		requirement_literal.predicate,
		requirement_literal.args,
		action_analysis,
		limit=3,
	):
		parsed_mode_call = _parse_signature_literal(str(mode_call).strip())
		if parsed_mode_call is None or not parsed_mode_call.is_positive:
			continue
		mode_symbols = tuple(str(arg).strip() for arg in parsed_mode_call.args if str(arg).strip())
		if any(
			symbol.startswith("AUX_") and symbol not in task_parameter_symbols
			for symbol in mode_symbols
		):
			continue
		candidate_calls: list[str] = []
		seen_calls: set[str] = set()
		for need_signature in mode_needs:
			need_literal = _parse_signature_literal(str(need_signature).strip())
			if need_literal is None or not need_literal.is_positive:
				continue
			if need_literal.to_signature() == requirement_literal.to_signature():
				continue
			need_symbols = tuple(str(arg).strip() for arg in need_literal.args if str(arg).strip())
			if any(
				symbol.startswith("AUX_") and symbol not in task_parameter_symbols
				for symbol in need_symbols
			):
				continue
			if _requirement_must_remain_context(
				need_literal,
				task_parameter_symbols=task_parameter_symbols,
				extra_role_symbols=(),
			):
				continue
			if not need_symbols:
				continue
			if preserved_symbols & set(need_symbols):
				continue
			if not _render_producer_mode_options_for_predicate(
				need_literal.predicate,
				need_literal.args,
				action_analysis,
				limit=3,
			):
				continue
			helper_call = _helper_call_for_literal(need_literal)
			if helper_call in seen_calls:
				continue
			seen_calls.add(helper_call)
			candidate_calls.append(helper_call)
		if candidate_calls and (
			not best_calls or len(candidate_calls) < len(best_calls)
		):
			best_calls = tuple(candidate_calls)
	return best_calls


def _render_ast_branch_object(
	*,
	parameters: Sequence[str] = (),
	precondition: Sequence[str] = (),
	ordered_subtasks: Sequence[str] = (),
) -> str:
	payload: Dict[str, Any] = {}
	if parameters:
		payload["parameters"] = [str(value).strip() for value in parameters if str(value).strip()]
	if precondition:
		payload["precondition"] = [str(value).strip() for value in precondition if str(value).strip()]
	payload["ordered_subtasks"] = [
		str(value).strip()
		for value in ordered_subtasks
		if str(value).strip()
	]
	return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _branch_contract_dict(
	*,
	parameters: Sequence[str],
	precondition: Sequence[str],
	ordered_subtasks: Sequence[str] = (),
) -> Dict[str, Any]:
	payload = {
		"parameters": [str(value).strip() for value in parameters if str(value).strip()],
		"precondition": [str(value).strip() for value in precondition if str(value).strip()],
		"ordered_subtasks": [
			str(value).strip()
			for value in ordered_subtasks
			if str(value).strip()
		],
	}
	return payload


def _branch_extra_role_symbols(
	*,
	mode_call: str,
	needs: Sequence[str],
	task_parameter_symbols: Collection[str],
) -> tuple[str, ...]:
	branch_symbols: list[str] = []
	seen_symbols: set[str] = set()
	candidate_signatures = [str(mode_call).strip(), *[str(item).strip() for item in needs]]
	for signature in candidate_signatures:
		parsed_signature = _parse_signature_literal(signature)
		if parsed_signature is None:
			continue
		for argument in parsed_signature.args:
			symbol = str(argument).strip()
			if (
				not symbol
				or not symbol.startswith("AUX_")
				or symbol in task_parameter_symbols
				or symbol in seen_symbols
			):
				continue
			seen_symbols.add(symbol)
			branch_symbols.append(symbol)
	return tuple(branch_symbols)


def _requirement_must_remain_context(
	requirement: HTNLiteral,
	*,
	task_parameter_symbols: Collection[str],
	extra_role_symbols: Collection[str],
	current_headline_literal: Optional[HTNLiteral] = None,
) -> bool:
	requirement_args = tuple(str(arg).strip() for arg in requirement.args if str(arg).strip())
	if not requirement_args:
		return False
	if not any(arg in extra_role_symbols for arg in requirement_args):
		return False
	if (
		current_headline_literal is not None
		and requirement.is_positive
		and requirement.predicate == current_headline_literal.predicate
	):
		return False
	if _is_aux_binding_requirement(
		requirement_args,
		task_parameter_symbols=task_parameter_symbols,
		extra_role_symbols=extra_role_symbols,
	):
		return True
	return False


def _support_plan_for_requirements(
	needs: Sequence[str],
	*,
	task_parameter_symbols: Collection[str],
	extra_role_symbols: Collection[str] = (),
	action_analysis: Optional[Dict[str, Any]] = None,
	include_recursive_caller_shared: bool = True,
	preserve_aux_binding_context: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
	support_calls, residual_context_literals, _ = _support_plan_contract_for_requirements(
		needs,
		task_parameter_symbols=task_parameter_symbols,
		extra_role_symbols=extra_role_symbols,
		action_analysis=action_analysis,
		current_headline_literal=None,
		include_recursive_caller_shared=include_recursive_caller_shared,
		preserve_aux_binding_context=preserve_aux_binding_context,
	)
	return support_calls, residual_context_literals


def _support_plan_contract_for_requirements(
	needs: Sequence[str],
	*,
	task_parameter_symbols: Collection[str],
	extra_role_symbols: Collection[str] = (),
	preserved_literals: Sequence[HTNLiteral] = (),
	action_analysis: Optional[Dict[str, Any]] = None,
	current_headline_literal: Optional[HTNLiteral] = None,
	include_recursive_caller_shared: bool = True,
	preserve_aux_binding_context: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
	support_calls: list[str] = []
	residual_context_literals: list[str] = []
	seen_support_calls: set[str] = set()
	seen_context_literals: set[str] = set()
	support_entries: list[tuple[HTNLiteral, str]] = []

	for requirement_signature in needs:
		parsed_requirement = _parse_signature_literal(str(requirement_signature).strip())
		if parsed_requirement is None or not parsed_requirement.is_positive:
			continue
		producer_options = None
		if action_analysis is not None:
			producer_options = _render_producer_mode_options_for_predicate(
				parsed_requirement.predicate,
				parsed_requirement.args,
				action_analysis,
				limit=3,
			)
		if preserve_aux_binding_context and _requirement_must_remain_context(
			parsed_requirement,
			task_parameter_symbols=task_parameter_symbols,
			extra_role_symbols=extra_role_symbols,
			current_headline_literal=current_headline_literal,
		):
			if not producer_options:
				rendered_signature = parsed_requirement.to_signature()
				if rendered_signature not in seen_context_literals:
					seen_context_literals.add(rendered_signature)
					residual_context_literals.append(rendered_signature)
				continue
		if action_analysis is not None and not producer_options:
			rendered_signature = parsed_requirement.to_signature()
			if rendered_signature not in seen_context_literals:
				seen_context_literals.add(rendered_signature)
				residual_context_literals.append(rendered_signature)
			continue
		if (
			action_analysis is not None
			and current_headline_literal is not None
			and _support_requirement_is_positive_cycle(
				requirement_literal=parsed_requirement,
				current_headline_literal=current_headline_literal,
				action_analysis=action_analysis,
			)
		):
			rendered_signature = parsed_requirement.to_signature()
			if rendered_signature not in seen_context_literals:
				seen_context_literals.add(rendered_signature)
				residual_context_literals.append(rendered_signature)
			continue
		helper_call = _helper_call_for_literal(parsed_requirement)
		if helper_call not in seen_support_calls:
			seen_support_calls.add(helper_call)
			support_calls.append(helper_call)
		support_entries.append((parsed_requirement, helper_call))
		if action_analysis is not None and include_recursive_caller_shared:
			for shared_signature in _helper_caller_shared_requirements(
				parsed_requirement.predicate,
				parsed_requirement.args,
				action_analysis,
			):
				if shared_signature not in seen_context_literals:
					seen_context_literals.add(shared_signature)
					residual_context_literals.append(shared_signature)

	volatile_support_calls: set[str] = set()
	ordered_support_calls = tuple(support_calls)
	if action_analysis is not None and len(support_entries) > 1:
		preservation_order = _ordered_support_literals_for_preservation(
			tuple(literal for literal, _ in support_entries),
			action_analysis=action_analysis,
		)
		if preservation_order is not None and len(preservation_order) > 1:
			ordered_support_calls = tuple(
				_helper_call_for_literal(literal)
				for literal in preservation_order
			)
			support_entries = [
				(literal, _helper_call_for_literal(literal))
				for literal in preservation_order
			]
		disturbance_scores: Dict[tuple[str, str], int] = {}
		for support_literal, _ in support_entries:
			support_signature = support_literal.to_signature()
			for preserved_literal, _ in support_entries:
				preserved_signature = preserved_literal.to_signature()
				if support_signature == preserved_signature:
					continue
				disturbance_scores[(support_signature, preserved_signature)] = (
					_support_requirement_disturbance_score(
						support_literal=support_literal,
						preserved_literal=preserved_literal,
						action_analysis=action_analysis,
					)
				)
		for supported_literal, helper_call in support_entries:
			supported_signature = supported_literal.to_signature()
			for other_literal, _ in support_entries:
				other_signature = other_literal.to_signature()
				if other_signature == supported_signature:
					continue
				if disturbance_scores.get(
					(other_signature, supported_signature),
					0,
				) > disturbance_scores.get(
					(supported_signature, other_signature),
					0,
				):
					volatile_support_calls.add(helper_call)
					break

	if action_analysis is not None and preserved_literals:
		for supported_literal, helper_call in support_entries:
			if helper_call in volatile_support_calls:
				continue
			for preserved_literal in preserved_literals:
				if (
					_support_requirement_disturbance_score(
						support_literal=supported_literal,
						preserved_literal=preserved_literal,
						action_analysis=action_analysis,
					) > 0
					or _support_requirement_disturbance_score(
						support_literal=preserved_literal,
						preserved_literal=supported_literal,
						action_analysis=action_analysis,
					) > 0
				):
					volatile_support_calls.add(helper_call)
					break

	support_before = tuple(
		call
		for call in ordered_support_calls
		if call not in volatile_support_calls
	)
	followup = tuple(
		call
		for call in ordered_support_calls
		if call in volatile_support_calls
	)
	return support_before, tuple(residual_context_literals), followup


def _support_requirement_is_positive_cycle(
	*,
	requirement_literal: HTNLiteral,
	current_headline_literal: HTNLiteral,
	action_analysis: Dict[str, Any],
) -> bool:
	current_signature = current_headline_literal.to_signature()
	if not current_signature:
		return False

	mode_options = _render_producer_mode_options_for_predicate(
		requirement_literal.predicate,
		requirement_literal.args,
		action_analysis,
		limit=3,
	)
	if not mode_options:
		return False

	for _, nested_needs in mode_options:
		nested_positive_signatures = {
			parsed_need.to_signature()
			for parsed_need in (
				_parse_signature_literal(str(signature).strip())
				for signature in (nested_needs or ())
			)
			if parsed_need is not None and parsed_need.is_positive
		}
		if current_signature not in nested_positive_signatures:
			return False
	return True


def _support_requirement_disturbance_score(
	*,
	support_literal: HTNLiteral,
	preserved_literal: HTNLiteral,
	action_analysis: Dict[str, Any],
	_seen_pairs: Optional[set[tuple[str, str]]] = None,
) -> int:
	direct_score = _direct_support_requirement_disturbance_score(
		support_literal=support_literal,
		preserved_literal=preserved_literal,
		action_analysis=action_analysis,
		_seen_pairs=_seen_pairs,
	)
	if direct_score > 0:
		return direct_score

	for shared_literal in _shared_support_requirements_for_literal(
		preserved_literal=preserved_literal,
		action_analysis=action_analysis,
	):
		if _direct_support_requirement_disturbance_score(
			support_literal=support_literal,
			preserved_literal=shared_literal,
			action_analysis=action_analysis,
			_seen_pairs=_seen_pairs,
		) > 0 and _direct_support_requirement_disturbance_score(
			support_literal=shared_literal,
			preserved_literal=preserved_literal,
			action_analysis=action_analysis,
			_seen_pairs=_seen_pairs,
		) > 0:
			return 1
	return 0


def _direct_support_requirement_disturbance_score(
	*,
	support_literal: HTNLiteral,
	preserved_literal: HTNLiteral,
	action_analysis: Dict[str, Any],
	_seen_pairs: Optional[set[tuple[str, str]]] = None,
) -> int:
	pair_key = (
		support_literal.to_signature(),
		preserved_literal.to_signature(),
	)
	if not pair_key[0] or not pair_key[1]:
		return 0
	if _seen_pairs is None:
		_seen_pairs = set()
	if pair_key in _seen_pairs:
		return 0
	_seen_pairs.add(pair_key)

	max_score = 0
	for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
		support_literal.predicate,
		[],
	):
		effect_args = [
			str(value).strip()
			for value in (pattern.get("effect_args") or [])
			if str(value).strip()
		]
		if len(effect_args) != len(support_literal.args):
			continue
		token_mapping = {
			token: argument
			for token, argument in zip(effect_args, support_literal.args)
		}
		_extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or [],
			action_parameter_types=pattern.get("action_parameter_types") or [],
		)
		for negative_signature in pattern.get("negative_effect_signatures") or []:
			rendered_signature = _render_signature_with_mapping(
				str(negative_signature),
				token_mapping,
			)
			rendered_literal = _parse_signature_literal(rendered_signature)
			if rendered_literal is None:
				continue
			if _literal_may_match_for_disturbance(
				rendered_literal,
				preserved_literal,
			):
				return 2
		for nested_signature in pattern.get("dynamic_precondition_signatures") or []:
			rendered_signature = _render_signature_with_mapping(
				str(nested_signature),
				token_mapping,
			)
			nested_literal = _parse_signature_literal(rendered_signature)
			if nested_literal is None or not nested_literal.is_positive:
				continue
			if _literal_may_match_for_disturbance(
				nested_literal,
				preserved_literal,
			):
				max_score = max(max_score, 1)
				continue
			if _direct_support_requirement_disturbance_score(
				support_literal=nested_literal,
				preserved_literal=preserved_literal,
				action_analysis=action_analysis,
				_seen_pairs=_seen_pairs,
			) > 0:
				max_score = max(max_score, 1)
	return max_score


def _shared_support_requirements_for_literal(
	*,
	preserved_literal: HTNLiteral,
	action_analysis: Dict[str, Any],
) -> tuple[HTNLiteral, ...]:
	shared_literals: list[HTNLiteral] = []
	seen_signatures: set[str] = set()
	shared_requirement_sets: list[set[str]] = []
	for _, needs in _render_producer_mode_options_for_predicate(
		preserved_literal.predicate,
		preserved_literal.args,
		action_analysis,
		limit=3,
	):
		shared_requirement_sets.append({
			parsed_need.to_signature()
			for parsed_need in (
				_parse_signature_literal(str(signature).strip())
				for signature in (needs or ())
			)
			if parsed_need is not None and parsed_need.is_positive
		})
	if shared_requirement_sets:
		for shared_signature in sorted(set.intersection(*shared_requirement_sets)):
			parsed_shared = _parse_signature_literal(shared_signature)
			if parsed_shared is None or not parsed_shared.is_positive:
				continue
			if parsed_shared.args:
				continue
			canonical_signature = parsed_shared.to_signature()
			if canonical_signature in seen_signatures:
				continue
			seen_signatures.add(canonical_signature)
			shared_literals.append(parsed_shared)
	return tuple(shared_literals)


def _ordered_support_literals_for_preservation(
	support_literals: Sequence[HTNLiteral],
	*,
	action_analysis: Dict[str, Any],
) -> Optional[tuple[HTNLiteral, ...]]:
	literals = tuple(support_literals)
	if len(literals) <= 1:
		return literals
	best_order: Optional[tuple[HTNLiteral, ...]] = None
	best_rank: Optional[tuple[int, ...]] = None
	for candidate_order in _permutations_of_literals(literals):
		if _support_order_is_preservation_feasible(
			candidate_order,
			action_analysis=action_analysis,
		):
			candidate_rank = _support_order_disruption_rank(
				candidate_order,
				action_analysis=action_analysis,
			)
			if best_rank is None or candidate_rank > best_rank:
				best_order = candidate_order
				best_rank = candidate_rank
	return best_order


def _support_order_is_preservation_feasible(
	ordered_literals: Sequence[HTNLiteral],
	*,
	action_analysis: Dict[str, Any],
) -> bool:
	preserved_literals: tuple[HTNLiteral, ...] = ()
	for literal in ordered_literals:
		if not _can_support_literal_preserving(
			target_literal=literal,
			preserved_literals=preserved_literals,
			action_analysis=action_analysis,
		):
			return False
		preserved_literals = (*preserved_literals, literal)
	return True


def _can_support_literal_preserving(
	*,
	target_literal: HTNLiteral,
	preserved_literals: Sequence[HTNLiteral],
	action_analysis: Dict[str, Any],
	_seen_states: Optional[set[tuple[str, tuple[str, ...]]]] = None,
) -> bool:
	state_key = (
		target_literal.to_signature(),
		tuple(sorted(literal.to_signature() for literal in preserved_literals)),
	)
	if not state_key[0]:
		return False
	if _seen_states is None:
		_seen_states = set()
	if state_key in _seen_states:
		return True
	_seen_states.add(state_key)

	for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
		target_literal.predicate,
		[],
	):
		effect_args = [
			str(value).strip()
			for value in (pattern.get("effect_args") or [])
			if str(value).strip()
		]
		if len(effect_args) != len(target_literal.args):
			continue
		token_mapping = {
			token: argument
			for token, argument in zip(effect_args, target_literal.args)
		}
		_extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or [],
			action_parameter_types=pattern.get("action_parameter_types") or [],
		)
		extra_role_symbols = tuple(
			symbol
			for symbol in token_mapping.values()
			if symbol not in target_literal.args
		)
		if any(
			_literal_may_match_for_disturbance(
				rendered_literal,
				preserved_literal,
			)
			for negative_signature in (pattern.get("negative_effect_signatures") or [])
			for rendered_literal in (
				_parse_signature_literal(
					_render_signature_with_mapping(
						str(negative_signature),
						token_mapping,
					),
				),
			)
			if rendered_literal is not None
			for preserved_literal in preserved_literals
		):
			continue
		support_requirements = _supportable_positive_requirements_for_pattern(
			pattern=pattern,
			token_mapping=token_mapping,
			task_parameter_symbols=target_literal.args,
			extra_role_symbols=extra_role_symbols,
			action_analysis=action_analysis,
		)
		if _support_requirements_are_preservation_feasible(
			support_requirements,
			preserved_literals=preserved_literals,
			action_analysis=action_analysis,
			_seen_states=_seen_states,
		):
			return True
	return False


def _support_requirements_are_preservation_feasible(
	requirements: Sequence[HTNLiteral],
	*,
	preserved_literals: Sequence[HTNLiteral],
	action_analysis: Dict[str, Any],
	_seen_states: set[tuple[str, tuple[str, ...]]],
) -> bool:
	if not requirements:
		return True
	for candidate_order in _permutations_of_literals(tuple(requirements)):
		achieved_literals = tuple(preserved_literals)
		order_feasible = True
		for literal in candidate_order:
			achieved_signatures = {
				item.to_signature()
				for item in achieved_literals
			}
			if literal.to_signature() in achieved_signatures:
				continue
			if not any(
				_support_requirement_disturbance_score(
					support_literal=literal,
					preserved_literal=preserved_literal,
					action_analysis=action_analysis,
				) > 0
				for preserved_literal in achieved_literals
			):
				continue
			if not _can_support_literal_preserving(
				target_literal=literal,
				preserved_literals=achieved_literals,
				action_analysis=action_analysis,
				_seen_states=_seen_states,
			):
				order_feasible = False
				break
			achieved_literals = (*achieved_literals, literal)
		if order_feasible:
			return True
	return False


def _supportable_positive_requirements_for_pattern(
	*,
	pattern: Dict[str, Any],
	token_mapping: Dict[str, str],
	task_parameter_symbols: Sequence[str],
	extra_role_symbols: Sequence[str],
	action_analysis: Dict[str, Any],
) -> tuple[HTNLiteral, ...]:
	requirements: list[HTNLiteral] = []
	for signature in pattern.get("dynamic_precondition_signatures") or []:
		rendered_signature = _render_signature_with_mapping(
			str(signature),
			token_mapping,
		)
		parsed_literal = _parse_signature_literal(rendered_signature)
		if parsed_literal is None or not parsed_literal.is_positive:
			continue
		if _requirement_must_remain_context(
			parsed_literal,
			task_parameter_symbols=task_parameter_symbols,
			extra_role_symbols=extra_role_symbols,
		):
			continue
		if not _render_producer_mode_options_for_predicate(
			parsed_literal.predicate,
			parsed_literal.args,
			action_analysis,
			limit=3,
		):
			continue
		requirements.append(parsed_literal)
	return tuple(requirements)


def _permutations_of_literals(
	literals: Sequence[HTNLiteral],
) -> tuple[tuple[HTNLiteral, ...], ...]:
	if len(literals) <= 1:
		return (tuple(literals),)
	orders: list[tuple[HTNLiteral, ...]] = []
	for index, literal in enumerate(literals):
		remaining = tuple(literals[:index]) + tuple(literals[index + 1 :])
		for suffix in _permutations_of_literals(remaining):
			orders.append((literal, *suffix))
	return tuple(orders)


def _support_order_disruption_rank(
	ordered_literals: Sequence[HTNLiteral],
	*,
	action_analysis: Dict[str, Any],
) -> tuple[int, ...]:
	per_position_scores: list[int] = []
	for index, literal in enumerate(ordered_literals):
		prior_literals = ordered_literals[:index]
		per_position_scores.append(
			sum(
				_support_precondition_disruption_count(
					support_literal=literal,
					preserved_literal=prior_literal,
					action_analysis=action_analysis,
				)
				for prior_literal in prior_literals
			),
		)
	return tuple(per_position_scores)


def _support_precondition_disruption_count(
	*,
	support_literal: HTNLiteral,
	preserved_literal: HTNLiteral,
	action_analysis: Dict[str, Any],
) -> int:
	preserved_requirements = _union_positive_requirements_for_literal(
		preserved_literal=preserved_literal,
		action_analysis=action_analysis,
	)
	if not preserved_requirements:
		return 0

	disrupting_mode_count = 0
	for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
		support_literal.predicate,
		[],
	):
		effect_args = [
			str(value).strip()
			for value in (pattern.get("effect_args") or [])
			if str(value).strip()
		]
		if len(effect_args) != len(support_literal.args):
			continue
		token_mapping = {
			token: argument
			for token, argument in zip(effect_args, support_literal.args)
		}
		rendered_action_args = _extend_mapping_with_action_parameters(
			token_mapping,
			pattern.get("action_parameters") or [],
			action_parameter_types=pattern.get("action_parameter_types") or [],
		)
		rendered_needs = tuple(
			parsed_need.to_signature()
			for parsed_need in (
				_parse_signature_literal(
					_render_signature_with_mapping(
						str(signature),
						token_mapping,
					),
				)
				for signature in (pattern.get("dynamic_precondition_signatures") or ())
			)
			if parsed_need is not None and parsed_need.is_positive
		)
		if (
			_task_invocation_signature(pattern.get("action_name", ""), rendered_action_args),
			rendered_needs,
		) not in set(
			_render_producer_mode_options_for_predicate(
				support_literal.predicate,
				support_literal.args,
				action_analysis,
				limit=3,
			),
		):
			continue
		for negative_signature in pattern.get("negative_effect_signatures") or []:
			rendered_literal = _parse_signature_literal(
				_render_signature_with_mapping(
					str(negative_signature),
					token_mapping,
				),
			)
			if rendered_literal is None:
				continue
			if any(
				_literal_may_match_for_disturbance(
					rendered_literal,
					requirement_literal,
				)
				for requirement_literal in preserved_requirements
			):
				disrupting_mode_count += 1
				break
	return disrupting_mode_count


def _union_positive_requirements_for_literal(
	*,
	preserved_literal: HTNLiteral,
	action_analysis: Dict[str, Any],
) -> tuple[HTNLiteral, ...]:
	seen_signatures: set[str] = set()
	requirements: list[HTNLiteral] = []
	for _, needs in _render_producer_mode_options_for_predicate(
		preserved_literal.predicate,
		preserved_literal.args,
		action_analysis,
		limit=3,
	):
		for signature in needs:
			parsed_need = _parse_signature_literal(str(signature).strip())
			if parsed_need is None or not parsed_need.is_positive:
				continue
			canonical_signature = parsed_need.to_signature()
			if canonical_signature in seen_signatures:
				continue
			seen_signatures.add(canonical_signature)
			requirements.append(parsed_need)
	return tuple(requirements)


def _literal_may_match_for_disturbance(
	candidate_literal: HTNLiteral,
	preserved_literal: HTNLiteral,
) -> bool:
	if candidate_literal.predicate != preserved_literal.predicate:
		return False
	if len(candidate_literal.args) != len(preserved_literal.args):
		return False
	return all(
		_symbols_may_unify_for_disturbance(
			str(candidate).strip(),
			str(preserved).strip(),
		)
		for candidate, preserved in zip(
			candidate_literal.args,
			preserved_literal.args,
		)
	)


def _symbols_may_unify_for_disturbance(
	left_symbol: str,
	right_symbol: str,
) -> bool:
	if not left_symbol or not right_symbol:
		return False
	if left_symbol == right_symbol:
		return True
	return (
		_symbol_is_existential_for_disturbance(left_symbol)
		or _symbol_is_existential_for_disturbance(right_symbol)
	)


def _symbol_is_existential_for_disturbance(symbol: str) -> bool:
	text = str(symbol).strip()
	return text.startswith("AUX") or text.startswith("?")


def _helper_caller_shared_requirements(
	predicate_name: str,
	predicate_args: Sequence[str],
	action_analysis: Dict[str, Any],
) -> tuple[str, ...]:
	mode_options = _render_producer_mode_options_for_predicate(
		predicate_name,
		predicate_args,
		action_analysis,
		limit=3,
	)
	if not mode_options:
		return ()

	residual_sets: list[set[str]] = []
	task_parameter_symbols = {
		str(symbol).strip()
		for symbol in predicate_args
		if str(symbol).strip()
	}
	for mode_call, needs in mode_options:
		extra_role_symbols = _branch_extra_role_symbols(
			mode_call=mode_call,
			needs=needs,
			task_parameter_symbols=predicate_args,
		)
		_, residual_context = _support_plan_for_requirements(
			needs,
			task_parameter_symbols=predicate_args,
			extra_role_symbols=extra_role_symbols,
			action_analysis=action_analysis,
			include_recursive_caller_shared=False,
		)
		filtered_residual_context: set[str] = set()
		for residual_signature in residual_context:
			parsed_residual = _parse_signature_literal(residual_signature)
			if parsed_residual is None:
				continue
			residual_symbols = {
				str(argument).strip()
				for argument in parsed_residual.args
				if str(argument).strip()
			}
			if residual_symbols and not residual_symbols <= task_parameter_symbols:
				continue
			filtered_residual_context.add(parsed_residual.to_signature())
		residual_sets.append(filtered_residual_context)

	if not residual_sets:
		return ()
	return tuple(sorted(set.intersection(*residual_sets)))


def _transition_support_contract_lines(
	*,
	domain: Any,
	prompt_analysis: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> List[str]:
	lines: List[str] = []
	grouped_tasks: Dict[str, List[Dict[str, Any]]] = {}
	for transition_task in prompt_analysis.get("transition_tasks", []):
		headline_signature = _dict_literal_signature(transition_task.get("headline_literal") or {})
		grouped_tasks.setdefault(headline_signature, []).append(transition_task)

	for headline_signature, transition_group in grouped_tasks.items():
		transition_task = transition_group[0]
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
		lines.append(
			f"- {headline_signature} | tasks={', '.join(str(item['name']) for item in transition_group)}"
		)
		lines.append(
			f"  required constructive branches per listed task: {len(mode_options)}"
		)
		for mode_call, needs in mode_options:
			needs = tuple(str(value).strip() for value in (needs or ()) if str(value).strip())
			required_block = (
				"{"
				+ ", ".join(needs)
				+ "}"
				if needs
				else "{none}"
			)
			lines.append(
				f"  branch: producer={mode_call} | require all of {required_block}"
			)
			support_calls, residual_context, restabilise_calls = _support_plan_contract_for_requirements(
				needs,
				task_parameter_symbols=headline_literal.args,
				action_analysis=action_analysis,
				current_headline_literal=headline_literal,
				preserve_aux_binding_context=False,
			)
			context_block = (
				"{"
				+ ", ".join(residual_context)
				+ "}"
				if residual_context
				else "{none}"
			)
			ordered_block = "{" + ", ".join((*support_calls, *restabilise_calls, mode_call)) + "}"
			lines.append(
				f"    exact branch skeleton: ordered_subtasks={ordered_block}"
				f" | keep context only {context_block} before {mode_call}"
			)
			extra_role_symbols = _branch_extra_role_symbols(
				mode_call=mode_call,
				needs=needs,
				task_parameter_symbols=headline_literal.args,
			)
			ordered_subtasks = (*support_calls, *restabilise_calls, mode_call)
			lines.append(
				"    exact AST branch object: "
				+ _render_ast_branch_object(
					parameters=(*headline_literal.args, *extra_role_symbols),
					precondition=residual_context,
					ordered_subtasks=ordered_subtasks,
				)
			)
			for requirement_signature in needs:
				parsed_requirement = _parse_signature_literal(requirement_signature)
				if parsed_requirement is None or not parsed_requirement.is_positive:
					continue
				helper_call = _helper_call_for_literal(parsed_requirement)
				if helper_call not in support_calls:
					continue
				shared_requirements = _same_arity_caller_shared_requirements(
					domain,
					parsed_requirement.predicate,
					parsed_requirement.args,
					action_analysis,
				)
				if shared_requirements:
					lines.append(
						f"  if you satisfy {requirement_signature} via {helper_call}, first satisfy "
						f"caller-shared {{{', '.join(shared_requirements)}}} before that helper call"
					)
				elif not restabilise_calls:
					lines.append(
						f"  if {requirement_signature} is not already true at entry, satisfy it via "
						f"{helper_call} before {mode_call}"
					)
	return lines


def _query_root_bridge_contract_lines(
	*,
	prompt_analysis: Dict[str, Any],
) -> List[str]:
	transition_names_by_target_literal: Dict[str, List[str]] = {}
	for payload in prompt_analysis.get("transition_tasks", []):
		target_literal = str(payload.get("target_literal", "")).strip()
		transition_name = str(payload.get("name", "")).strip()
		if not target_literal or not transition_name:
			continue
		transition_names_by_target_literal.setdefault(target_literal, []).append(
			transition_name,
		)

	lines: List[str] = []
	for payload in prompt_analysis.get("query_root_alias_tasks", []):
		query_root_name = str(payload.get("name", "")).strip()
		target_literal = str(payload.get("target_literal", "")).strip()
		parameters = tuple(
			str(value).strip()
			for value in (payload.get("parameters") or ())
			if str(value).strip()
		)
		bridge_parameters = tuple(
			str(value).strip()
			for value in (payload.get("bridge_parameters") or payload.get("parameters") or ())
			if str(value).strip()
		)
		bridge_precondition = tuple(
			str(value).strip()
			for value in (payload.get("bridge_precondition") or ())
			if str(value).strip()
		)
		matching_transitions = transition_names_by_target_literal.get(target_literal, [])
		if not query_root_name:
			continue
		if matching_transitions:
			lines.append(
				f"- {query_root_name} -> required bridge branches: {len(matching_transitions)}"
			)
			bridge_invocations = [
				(
					f"{transition_name}({', '.join(bridge_parameters)})"
					if bridge_parameters
					else transition_name
				)
				for transition_name in matching_transitions
			]
			if len(bridge_invocations) == 1:
				if bridge_parameters == parameters and not bridge_precondition:
					lines.append(
						f'  exact compact task payload suffix: {{"ordered_subtasks":["{bridge_invocations[0]}"]}}'
						"; no precondition; no helpers; no primitives"
					)
				else:
					branch_payload: Dict[str, Any] = {
						"ordered_subtasks": [bridge_invocations[0]],
					}
					if bridge_parameters != parameters:
						branch_payload["parameters"] = list(bridge_parameters)
					if bridge_precondition:
						branch_payload["precondition"] = list(bridge_precondition)
					lines.append(
						"  exact AST branch object: "
						+ json.dumps(branch_payload, ensure_ascii=False, separators=(",", ":"))
						+ "; no helpers; no primitives"
					)
			else:
				lines.append(
					"  multi-bridge case: use REQUIRED BRANCH CONTRACTS JSON exactly; one "
					"dfa_step_* call per branch; no helpers; no primitives; no extra "
					"precondition beyond the listed branch precondition"
				)
		else:
			lines.append(
				f"- {query_root_name} has no matching transition task for target literal "
				f"{target_literal}; treat this as unsupported and do not invent a replacement."
			)
	return lines


def _helper_task_candidate_lines(
	*,
	domain: Any,
	prompt_analysis: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> List[str]:
	seen_signatures: set[str] = set()
	lines: List[str] = []
	for helper_literal in _recommended_helper_literals(
		prompt_analysis=prompt_analysis,
		action_analysis=action_analysis,
	):
		canonical_literal = canonicalise_helper_literal(helper_literal)
		signature = canonical_literal.to_signature()
		if signature in seen_signatures:
			continue
		seen_signatures.add(signature)
		parameters = [str(arg).strip() for arg in canonical_literal.args if str(arg).strip()]
		name = helper_task_name_for_literal(canonical_literal)
		shared_requirements = _same_arity_caller_shared_requirements(
			domain,
			canonical_literal.predicate,
			canonical_literal.args,
			action_analysis,
		)
		line = (
			f"- {name} | headline={signature} | parameters="
			f"({', '.join(parameters) if parameters else 'none'})"
		)
		if shared_requirements:
			line += f" | caller_shared={{{', '.join(shared_requirements)}}}"
		lines.append(line)
	return lines


def _helper_contract_template_lines(
	*,
	prompt_analysis: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> List[str]:
	lines: List[str] = []
	seen_signatures: set[str] = set()
	for helper_literal in _recommended_helper_literals(
		prompt_analysis=prompt_analysis,
		action_analysis=action_analysis,
	):
		canonical_literal = canonicalise_helper_literal(helper_literal)
		signature = canonical_literal.to_signature()
		if signature in seen_signatures:
			continue
		seen_signatures.add(signature)
		mode_options = _render_producer_mode_options_for_predicate(
			canonical_literal.predicate,
			canonical_literal.args,
			action_analysis,
			limit=3,
		)
		if not mode_options:
			continue
		mentions_auxiliary_role = False
		name = helper_task_name_for_literal(canonical_literal)
		lines.append(f"- {name} | headline={signature}")
		lines.append(f"  required constructive branches: {len(mode_options)}")
		for mode_call, needs in mode_options:
				needs = tuple(str(value).strip() for value in (needs or ()) if str(value).strip())
				mentions_auxiliary_role = mentions_auxiliary_role or ("AUX_" in mode_call)
				mentions_auxiliary_role = mentions_auxiliary_role or any("AUX_" in value for value in needs)
				required_block = (
					"{"
					+ ", ".join(needs)
					+ "}"
					if needs
					else "{none}"
				)
				lines.append(
					f"  branch: producer={mode_call} | require all of {required_block}"
				)
				extra_role_symbols = _branch_extra_role_symbols(
					mode_call=mode_call,
					needs=needs,
					task_parameter_symbols=canonical_literal.args,
				)
				recursive_support_calls, residual_context_literals, restabilise_calls = _support_plan_contract_for_requirements(
					needs,
					task_parameter_symbols=canonical_literal.args,
					extra_role_symbols=extra_role_symbols,
					action_analysis=action_analysis,
					current_headline_literal=canonical_literal,
				)
				if recursive_support_calls or residual_context_literals or restabilise_calls:
					context_block = (
						"{"
						+ ", ".join(residual_context_literals)
						+ "}"
						if residual_context_literals
						else "{none}"
					)
					ordered_block = (
						"{"
						+ ", ".join((*recursive_support_calls, *restabilise_calls, mode_call))
						+ "}"
					)
					lines.append(
						f"    recursive ordered_subtasks={ordered_block}"
						f" | keep context only {context_block}"
					)
				lines.append(
					"    exact AST branch object: "
					+ _render_ast_branch_object(
						parameters=(*canonical_literal.args, *extra_role_symbols),
						precondition=residual_context_literals,
						ordered_subtasks=(
							*recursive_support_calls,
							*restabilise_calls,
							mode_call,
						),
					)
				)
		if mentions_auxiliary_role:
			lines.append(
				f"  AUX rule for {name}: if a branch uses AUX, declare AUX in branch parameters "
				"and keep only genuine AUX witness-binding literals in precondition/context."
			)
	return lines


def _recommended_helper_literals(
	*,
	prompt_analysis: Dict[str, Any],
	action_analysis: Dict[str, Any],
) -> Tuple[HTNLiteral, ...]:
	queue: deque[HTNLiteral] = deque()
	seen_signatures: set[str] = set()
	helper_literals: List[HTNLiteral] = []

	for transition_task in prompt_analysis.get("transition_tasks", []):
		for signature in (transition_task.get("retained_prefix_literals") or ()):
			parsed = _parse_signature_literal(str(signature).strip())
			if parsed is not None and parsed.is_positive:
				queue.append(parsed)
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
				queue.append(parsed)

	while queue:
		literal = queue.popleft()
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
				queue.append(nested_literal)

	return tuple(helper_literals)


def _transition_parameter_context(
	*,
	current_literal: HTNLiteral,
	prior_literals: Sequence[HTNLiteral],
) -> Tuple[Tuple[str, ...], Dict[str, str]]:
	current_args = tuple(str(argument).strip() for argument in current_literal.args if str(argument).strip())
	current_symbols = parameter_symbols(len(current_args))
	symbol_map: Dict[str, str] = {
		argument: symbol
		for argument, symbol in zip(current_args, current_symbols)
	}
	extra_objects: List[str] = []
	seen_extra_objects = set(current_args)
	for literal in prior_literals:
		for argument in literal.args:
			candidate = str(argument).strip()
			if not candidate or candidate in seen_extra_objects:
				continue
			seen_extra_objects.add(candidate)
			extra_objects.append(candidate)
	extra_symbols = context_parameter_symbols(len(extra_objects))
	for argument, symbol in zip(extra_objects, extra_symbols):
		symbol_map[argument] = symbol
	return (*current_symbols, *extra_symbols), symbol_map


def _parameterise_literal_with_symbol_map(
	literal: HTNLiteral,
	symbol_map: Dict[str, str],
) -> HTNLiteral:
	return HTNLiteral(
		predicate=literal.predicate,
		args=tuple(
			str(symbol_map.get(str(argument).strip(), str(argument).strip()))
			for argument in literal.args
		),
		is_positive=literal.is_positive,
		source_symbol=None,
		negation_mode=literal.negation_mode,
	)


def _dedupe_invocation_signatures(signatures: Sequence[str]) -> Tuple[str, ...]:
	ordered: List[str] = []
	seen: set[str] = set()
	for signature in signatures:
		text = str(signature).strip()
		if not text or text in seen:
			continue
		seen.add(text)
		ordered.append(text)
	return tuple(ordered)
