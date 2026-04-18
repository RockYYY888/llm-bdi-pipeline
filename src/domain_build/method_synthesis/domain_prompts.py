"""
Domain-complete HTN prompt builders.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from utils.hddl_condition_parser import HDDLConditionParser

from .semantic_rules import (
	_constructive_template_summary_for_task,
	_declared_task_schema_map,
	_dynamic_support_candidate_map,
	_dynamic_support_hint_lines,
	_format_tagged_block,
	_limited_unique,
	_literal_pattern_signature,
	_name_tokens,
	_normalise_action_analysis,
	_parameter_token,
	_parameter_type,
	_relevant_action_names_for_prompt,
	_reusable_dynamic_resource_payloads,
	_same_arity_declared_task_candidates,
	_sanitize_name,
	_shared_dynamic_requirements_for_predicate,
	_task_headline_candidate_map,
	_task_invocation_signature,
	_typed_task_invocation_signature,
	_unique_preserve_order,
)

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
		task_parameter_types = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(task, "parameters", ()) or ())
		)
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
				predicate_arg_types=task_parameter_types,
			):
				if requirement not in shared_requirements:
					shared_requirements.append(requirement)
			constructive_template = _constructive_template_summary_for_task(
				task_name,
				task_parameters,
				predicate_name,
				analysis,
				task_parameter_types=task_parameter_types,
			)
			if constructive_template and constructive_template not in template_summaries:
				template_summaries.append(constructive_template)
		shared_requirements = _limited_unique(shared_requirements, limit=4)
		template_summaries = _limited_unique(template_summaries, limit=2)
		shared_dynamic_prerequisites_by_task[sanitized_task_name] = shared_requirements
		producer_consumer_templates_by_task[sanitized_task_name] = template_summaries
		domain_task_contracts.append(
			{
				"task_name": task_name,
				"task_signature": _task_invocation_signature(task_name, task_parameters),
				"typed_task_signature": _typed_task_invocation_signature(
					task_name,
					getattr(task, "parameters", ()) or (),
				),
				"parameters": list(task_parameters),
				"parameter_types": list(task_parameter_types),
				"headline_candidates": list(headline_candidates),
				"shared_dynamic_prerequisites": list(shared_requirements),
				"producer_consumer_templates": list(template_summaries),
				"composition_support_tasks": [],
				"recursive_support_predicates": [],
			}
		)

	support_task_palette = [
		(
			str(contract.get("task_name") or "").strip(),
			str(contract.get("task_signature") or "").strip(),
			list(contract.get("headline_candidates") or ()),
		)
		for contract in domain_task_contracts
		if list(contract.get("headline_candidates") or ())
	]
	task_schemas = _declared_task_schema_map(domain)
	generic_resource_predicates = {
		str(payload.get("predicate") or "").strip()
		for payload in _reusable_dynamic_resource_payloads(analysis)
		if str(payload.get("predicate") or "").strip()
	}
	palette_eligible_generic_predicates = {
		predicate_name
		for predicate_name in generic_resource_predicates
		if any(
			predicate_name in candidate_headlines
			for _, _, candidate_headlines in support_task_palette
		)
	}
	task_signature_by_name = {
		str(contract.get("task_name") or "").strip(): str(contract.get("task_signature") or "").strip()
		for contract in domain_task_contracts
		if str(contract.get("task_name") or "").strip()
	}
	task_headlines_by_name = {
		str(contract.get("task_name") or "").strip(): list(contract.get("headline_candidates") or ())
		for contract in domain_task_contracts
		if str(contract.get("task_name") or "").strip()
	}
	support_candidates_by_predicate = _dynamic_support_candidate_map(domain, analysis)
	for contract in domain_task_contracts:
		task_name = str(contract.get("task_name") or "").strip()
		task_signature = str(contract.get("task_signature") or "").strip()
		relevant_predicates = {
			str(predicate_name).strip()
			for predicate_name in list(contract.get("headline_candidates") or ())
			if str(predicate_name).strip()
		}
		relevant_predicates.update(
			str(requirement).split("(", 1)[0].strip()
			for requirement in list(contract.get("shared_dynamic_prerequisites") or ())
			if str(requirement).strip()
			and "(" in str(requirement)
			and (
				str(requirement).split("(", 1)[0].strip() not in generic_resource_predicates
				or str(requirement).split("(", 1)[0].strip() in palette_eligible_generic_predicates
			)
		)
		composition_support_tasks: list[str] = []
		recursive_support_predicates: list[str] = []
		for candidate_name, candidate_signature, candidate_headlines in support_task_palette:
			if not candidate_signature or not candidate_headlines:
				continue
			headline_overlap = [
				headline
				for headline in candidate_headlines
				if headline in relevant_predicates
			]
			if not headline_overlap:
				continue
			if candidate_signature == task_signature:
				entry = (
					f"{candidate_signature} stabilizes {', '.join(headline_overlap)} "
					"(recursive reuse allowed)"
				)
			else:
				entry = f"{candidate_signature} stabilizes {', '.join(headline_overlap)}"
			if entry not in composition_support_tasks:
				composition_support_tasks.append(entry)
			if candidate_signature == task_signature:
				for headline in headline_overlap:
					if headline not in recursive_support_predicates:
						recursive_support_predicates.append(headline)
		for predicate_name in sorted(relevant_predicates):
			for candidate_name in support_candidates_by_predicate.get(predicate_name, []):
				candidate_signature = task_signature_by_name.get(candidate_name)
				candidate_headlines = task_headlines_by_name.get(candidate_name) or []
				if not candidate_signature:
					continue
				predicate_tokens = set(_name_tokens(predicate_name))
				task_name_tokens = set(_name_tokens(task_name))
				candidate_name_tokens = set(_name_tokens(candidate_name))
				if (
					predicate_name not in candidate_headlines
					and not (predicate_tokens & candidate_name_tokens)
					and not (task_name_tokens & candidate_name_tokens)
				):
					continue
				if candidate_signature == task_signature:
					entry = (
						f"{candidate_signature} stabilizes {predicate_name} "
						"(recursive reuse allowed)"
					)
				else:
					entry = f"{candidate_signature} stabilizes {predicate_name}"
				if entry not in composition_support_tasks:
					composition_support_tasks.append(entry)
				if candidate_signature == task_signature and predicate_name not in recursive_support_predicates:
					recursive_support_predicates.append(predicate_name)
		if not composition_support_tasks and task_name:
			task_schema = task_schemas.get(task_name)
			task_parameter_types = {
				_parameter_type(parameter)
				for parameter in (getattr(task_schema, "parameters", ()) or ())
			}
			for candidate_name in _same_arity_declared_task_candidates(
				domain,
				task_name,
				task_schemas,
			):
				candidate_signature = task_signature_by_name.get(candidate_name)
				candidate_headlines = task_headlines_by_name.get(candidate_name) or []
				if not candidate_signature or not candidate_headlines:
					continue
				entry = f"{candidate_signature} stabilizes {', '.join(candidate_headlines)}"
				if entry not in composition_support_tasks:
					composition_support_tasks.append(entry)
			for candidate_name, candidate_signature, candidate_headlines in support_task_palette:
				if candidate_name == task_name or not candidate_headlines:
					continue
				candidate_schema = task_schemas.get(candidate_name)
				candidate_parameter_types = {
					_parameter_type(parameter)
					for parameter in (getattr(candidate_schema, "parameters", ()) or ())
				}
				if not candidate_parameter_types or not candidate_parameter_types.issubset(task_parameter_types):
					continue
				entry = f"{candidate_signature} stabilizes {', '.join(candidate_headlines)}"
				if entry not in composition_support_tasks:
					composition_support_tasks.append(entry)
		contract["composition_support_tasks"] = (
			_limited_unique(composition_support_tasks, limit=6) or ["none"]
		)
		contract["recursive_support_predicates"] = recursive_support_predicates or ["none"]

	return {
		"declared_compound_tasks": [
			_task_invocation_signature(
				str(getattr(task, "name", "")).strip(),
				tuple(_parameter_token(parameter) for parameter in task.parameters),
			)
			for task in getattr(domain, "tasks", [])
			if str(getattr(task, "name", "")).strip()
		],
		"typed_declared_compound_tasks": [
			_typed_task_invocation_signature(
				str(getattr(task, "name", "")).strip(),
				tuple(getattr(task, "parameters", ()) or ()),
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

def _render_domain_task_contract_blocks(domain_task_contracts: Sequence[Dict[str, Any]]) -> str:
	blocks: list[str] = []
	for payload in domain_task_contracts:
		task_signature = str(payload.get("task_signature", "")).strip()
		if not task_signature:
			continue
		headline_candidates = payload.get("headline_candidates") or ["none"]
		typed_task_signature = str(payload.get("typed_task_signature") or task_signature).strip()
		parameter_types = payload.get("parameter_types") or ["object"] * len(
			payload.get("parameters") or ()
		)
		shared_dynamic_prerequisites = payload.get("shared_dynamic_prerequisites") or ["none"]
		producer_consumer_templates = payload.get("producer_consumer_templates") or ["none"]
		composition_support_tasks = payload.get("composition_support_tasks") or ["none"]
		recursive_support_predicates = payload.get("recursive_support_predicates") or ["none"]
		blocks.append(
			"\n".join(
				[
					f"task_signature: {task_signature}",
					f"typed_task_signature: {typed_task_signature}",
					f"parameter_types: {', '.join(str(item) for item in parameter_types)}",
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
					"composition_support_tasks:",
					*[
						f"- {item}"
						for item in composition_support_tasks
					],
					"recursive_support_predicates:",
					*[
						f"- {item}"
						for item in recursive_support_predicates
					],
				]
			)
		)
	return "\n\n".join(blocks).strip()

def _render_domain_action_schema_blocks(domain: Any) -> str:
	parser = HDDLConditionParser()
	blocks: list[str] = []
	for action in getattr(domain, "actions", []):
		parsed_action = parser.parse_action(action)
		parameter_signature = ", ".join(
			f"{_parameter_token(parameter)}:{_parameter_type(parameter)}"
			for parameter in getattr(action, "parameters", ())
		)
		preconditions = [
			_literal_pattern_signature(pattern)
			for pattern in parsed_action.preconditions
			if pattern.predicate != "="
		]
		add_effects = [
			_literal_pattern_signature(pattern)
			for pattern in parsed_action.effects
			if pattern.predicate != "=" and pattern.is_positive
		]
		delete_effects = [
			_literal_pattern_signature(pattern)
			for pattern in parsed_action.effects
			if pattern.predicate != "=" and not pattern.is_positive
		]
		blocks.append(
			"\n".join(
				[
					f"{parsed_action.name}({parameter_signature})",
					f"  preconditions: {', '.join(preconditions) if preconditions else 'true'}",
					f"  add_effects: {', '.join(add_effects) if add_effects else 'none'}",
					f"  delete_effects: {', '.join(delete_effects) if delete_effects else 'none'}",
				]
			)
		)
	return "\n\n".join(blocks).strip()

def _render_domain_dynamic_predicate_contract_blocks(action_analysis: Dict[str, Any]) -> str:
	blocks: list[str] = []
	for predicate_name in action_analysis.get("dynamic_predicates", []):
		producer_lines: list[str] = []
		for pattern in action_analysis.get("producer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		):
			parameter_signature = ", ".join(
				f"{_parameter_token(parameter)}:{_parameter_type(parameter)}"
				for parameter in (pattern.get("action_parameters") or [])
			)
			dynamic_needs = [
				str(signature)
				for signature in (pattern.get("dynamic_precondition_signatures") or [])
				if str(signature).strip() and not str(signature).startswith("not ")
			]
			producer_lines.append(
				f"- {pattern['source_action_name']}({parameter_signature}) "
				f"adds {pattern['effect_signature']} "
				f"[needs {', '.join(dynamic_needs) if dynamic_needs else 'no dynamic prerequisites'}]"
			)
		consumer_lines: list[str] = []
		for pattern in action_analysis.get("consumer_patterns_by_predicate", {}).get(
			predicate_name,
			[],
		):
			parameter_signature = ", ".join(
				f"{_parameter_token(parameter)}:{_parameter_type(parameter)}"
				for parameter in (pattern.get("action_parameters") or [])
			)
			other_dynamic_needs = [
				str(signature)
				for signature in (pattern.get("other_dynamic_precondition_signatures") or [])
				if str(signature).strip() and not str(signature).startswith("not ")
			]
			consumer_lines.append(
				f"- {pattern['source_action_name']}({parameter_signature}) "
				f"consumes {pattern['precondition_signature']} "
				f"[other dynamic prerequisites: {', '.join(other_dynamic_needs) if other_dynamic_needs else 'none'}]"
			)
		blocks.append(
			"\n".join(
				[
					f"predicate: {predicate_name}",
					"producer_modes:",
					*(producer_lines or ["- none"]),
					"consumer_modes:",
					*(consumer_lines or ["- none"]),
				]
			)
		)
	return "\n\n".join(blocks).strip()

def build_domain_htn_system_prompt() -> str:
	"""System prompt for one-shot domain-complete method synthesis synthesis."""

	return (
		"ROLE:\n"
		"You are writing a typed symbolic HTN method library, not prose. Treat the task as "
		"constrained program synthesis over the provided domain signature.\n"
		"\n"
		"OBJECTIVE:\n"
		"Synthesize one executable domain-complete HTN library from the typed domain "
		"signature only.\n"
		"\n"
		"ALLOWED MATERIAL:\n"
		"- Use only task names, action names, predicates, and symbolic variables justified by the prompt.\n"
		"- Use ARG* for task-signature roles and AUX_* for extra witness roles.\n"
		"- Keep methods schematic, typed, and query-independent.\n"
		"\n"
		"FORBIDDEN MATERIAL:\n"
		"- Never invent benchmark-specific constants, hidden helper predicates, domain-specific shortcuts, or grounded bindings.\n"
		"- Never use problem facts, benchmark instance names, or hidden planning knowledge outside the prompt.\n"
		"- Never emit markdown, reasoning trace, explanatory text, or extra schema fields.\n"
		"\n"
		"SYNTHESIS POLICY:\n"
		"- Define every declared compound task exactly once in tasks.\n"
		"- The tasks list is closed-world: define exactly the declared compound tasks and no extra compound helpers.\n"
		"- Each constructive branch must compile to one reusable HDDL-style method.\n"
		"- Every declared task must have at least one executable constructive branch; noop-only coverage is invalid.\n"
		"- Every compound child you call must also appear as another task entry in tasks.\n"
		"- Support dynamic prerequisites before the consuming producer step.\n"
		"- When the compact slots are insufficient, fall back to explicit ordered_subtasks and ordering.\n"
		"\n"
		"LOGIC AND SCHEMA RULES:\n"
		"- Preconditions and context are conjunctions only. Never emit or(...), exists(...), forall(...), or synthetic boolean wrapper predicates.\n"
		"- Use equality only as ARG1 == ARG2 or ARG1 != ARG2, never equal(ARG1, ARG2).\n"
		"- ordering must be explicit pairwise edges. Never emit a chain edge like [[\"s1\",\"s2\",\"s3\"]].\n"
		"- Branch precondition may be written as precondition or context.\n"
		"- For total orders, prefer compact slots: precondition/context, support_before, producer, followup(s).\n"
		"\n"
		"OUTPUT CONTRACT:\n"
		"- Emit JSON only.\n"
		"- Return minified JSON only: no commentary, no markdown fences, no blank output.\n"
		"- Return one JSON object with top-level key tasks.\n"
		"- Only emit tasks; primitive tasks are injected automatically.\n"
		"- Each task entry uses fields name, parameters, noop, constructive.\n"
		"\n"
		"SILENT QUALITY GATE BEFORE OUTPUT:\n"
		"- Check that all declared tasks are present exactly once.\n"
		"- Check that all constructive branches are executable and typed.\n"
		"- Check that recursive branches change at least one witness argument.\n"
		"- Check that all called compound children are defined.\n"
		"- Then output final JSON only.\n"
	)

def build_domain_htn_user_prompt(
	domain: Any,
	schema_hint: str,
	*,
	action_analysis: Optional[Dict[str, Any]] = None,
	derived_analysis: Optional[Dict[str, Any]] = None,
) -> str:
	"""User prompt for one-shot domain-complete method synthesis synthesis."""

	analysis = _normalise_action_analysis(domain, action_analysis)
	prompt_analysis = dict(
		derived_analysis
		or build_domain_prompt_analysis_payload(
			domain,
			action_analysis=analysis,
		)
	)
	declared_compound_tasks = list(prompt_analysis.get("declared_compound_tasks") or ())
	typed_declared_compound_tasks = list(
		prompt_analysis.get("typed_declared_compound_tasks") or ()
	)
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
	typed_task_lines = [
		f"- {task_signature}"
		for task_signature in typed_declared_compound_tasks
	] or ["- none"]
	action_lines = [f"- {action_name}" for action_name in action_names] or ["- none"]
	type_lines = [
		f"- {type_name}"
		for type_name in sorted(
			str(getattr(type_obj, "name", "")).strip()
			for type_obj in getattr(domain, "types", [])
			if str(getattr(type_obj, "name", "")).strip()
		)
	] or ["- object"]
	predicate_lines = [
		f"- {_task_invocation_signature(str(getattr(predicate, 'name', '')).strip(), tuple(_parameter_token(parameter) for parameter in getattr(predicate, 'parameters', ()) or ()))}"
		for predicate in getattr(domain, "predicates", [])
		if str(getattr(predicate, "name", "")).strip()
	] or ["- none"]
	dynamic_predicate_lines = [
		f"- {predicate_name}"
		for predicate_name in (analysis.get("dynamic_predicates") or [])
	] or ["- none"]
	action_schema_block = _render_domain_action_schema_blocks(domain)
	dynamic_predicate_contract_block = _render_domain_dynamic_predicate_contract_blocks(analysis)
	support_task_hint_lines = "\n".join(
		_dynamic_support_hint_lines(domain, analysis)
	) or "none"
	domain_summary_block = "\n".join(
		[
			f"domain: {domain.name}",
			"domain_types:",
			*type_lines,
			"",
			"declared_compound_tasks:",
			*task_lines,
			"",
			"typed_declared_compound_tasks:",
			*typed_task_lines,
			"",
			"predicates:",
			*predicate_lines,
			"",
			"dynamic_predicates:",
			*dynamic_predicate_lines,
			"",
			"relevant_primitive_actions:",
			*action_lines,
			"",
			"primitive_action_schemas:",
			action_schema_block or "none",
			"",
			"dynamic_predicate_contracts:",
			dynamic_predicate_contract_block or "none",
			"",
			"support_task_hints:",
			support_task_hint_lines,
			"",
			"reusable_dynamic_resources:",
			*resource_lines,
		]
	)
	instructions_block = "\n".join(
		[
			"1. Read domain_task_contracts first; they are the compact domain-wide synthesis contract.",
			"2. Define every declared compound task exactly once in tasks.",
			"2b. The tasks list is closed-world: do not invent extra compound helpers. Encode helper behaviour as another branch of a declared task or via primitive actions.",
			"3. Keep methods schematic and query-independent. Do not ground with benchmark constants or problem objects.",
			"4. Use only symbols licensed by the provided domain summary and task contracts.",
			"5. Preserve argument positions exactly across task headlines, producers, support tasks, and recursive siblings.",
			"6. If a task contract lists shared dynamic prerequisites, support them before the consuming producer step.",
			"7. If producer_consumer_templates exposes a direct primitive producer, include at least one non-recursive branch that uses it.",
			"8. If a task does not align to one obvious primitive producer, decompose it through declared support tasks and primitive action schemas instead of omitting it.",
			"8b. If a declared task lacks headline_candidates or producer_consumer_templates, treat it as a helper task and define it with relevant primitive action schemas.",
			"9. Prefer reusable support structure over duplicating primitive producer logic in multiple parents.",
			"10. Treat composition_support_tasks as the preferred reusable palette.",
			"11. Keep child-internal support inside the child unless the contract marks it as caller-shared.",
			"12. If the task headline already holds, use noop instead of a destructive constructive branch.",
			"13. Every declared task needs at least one executable constructive branch with real subtasks; noop-only coverage is invalid.",
			"14. If no subtasks are needed because the headline already holds, that branch belongs under noop, not constructive.",
			"15. Earlier support must preserve or establish the later producer's required dynamic literals.",
			"15a. In every constructive branch, each required dynamic literal must be explicit before the consuming producer step.",
			"16. If the final producer needs holding(ARG1), loaded(ARG1), or another control/resource literal, add a sibling that acquires it first.",
			"17. If support_task_hints names a declared task for a producer precondition, prefer that declared support task.",
			"18. If recursive_support_predicates is non-empty, any recursive child must change at least one AUX_* witness argument.",
			"19. Never recurse with an identical argument tuple or to the same immediate target before the final producer.",
			"20. Recursive reuse should be an extra sibling around a direct producer branch, not the only constructive route.",
			"21. For externally completed headlines, include the final outward producer in at least one branch.",
			"22. Treat typed_task_signature and parameter_types as hard constraints.",
			"23. Respect primitive action arity exactly. A zero-parameter action must be emitted with zero arguments.",
			"24. Preconditions and context are conjunctions only. Do not emit or(...), quantified formulas, or shorthand predicates.",
			"25. Express equality only with == or != between symbols, never equal(...).",
			"26. Never emit dummy padding steps or nop fillers inside constructive branches.",
			"27. Never use noop(...) or nop(...) as a constructive producer or placeholder step. already-satisfied cases belong only in noop.",
			"28. When the compact slots are insufficient, switch to explicit ordered_subtasks and ordering instead of inventing new schema fields.",
		]
	)
	construction_protocol_block = "\n".join(
		[
			"1. For each task, identify its headline or helper purpose from typed_task_signature, headline_candidates, and support_task_hints.",
			"2. Add one minimal typed noop case for the already-satisfied condition.",
			"3. Add one direct constructive branch when a primitive producer template exists.",
			"4. Use support_before only to establish missing direct prerequisites or unblock the producer.",
			"5. Add followup only when the producer reaches the headline but leaves a shared resource unstable.",
			"6. Add recursive or transitive siblings only after a direct branch exists, and only through changed AUX_* witnesses.",
		]
	)
	forbidden_patterns_block = "\n".join(
		[
			"- Invalid: omitting a declared task because it has no obvious headline predicate.",
			"- Invalid: inventing a new compound helper outside the declared task set.",
			"- Invalid: using a support child unrelated to the current headline or its direct prerequisites.",
			"- Invalid: recursive self-calls with an identical argument tuple.",
			"- Invalid: a constructive branch that stops at an intermediate artifact instead of the final headline.",
			"- Invalid: binding a typed task parameter into an incompatible predicate or action slot.",
			"- Invalid: using noop(...) or nop(...) as a producer.",
		]
	)
	examples_block = "\n".join(
		[
			"1. Transitive topology recursion:",
			"- If relocate(ARG1, ARG2) lacks a direct producer, stabilize an intermediate witness or blocker, then call the producer that reaches ARG2.",
			"2. Recursive blocker-clearing pattern:",
			"- If clear_target(ARG1) requires removing AUX1 from ARG1, clear AUX1 first, then remove AUX1 from ARG1, then clean up only if a shared resource was destabilized.",
			"3. Resource-acquisition pattern:",
			"- If the final producer needs holding(ARG1), calibrated(ARG1), empty(ARG_STORE), or at(ARG1, ARG2), add a sibling that acquires it first.",
			"4. Compact support-before pattern:",
			"- support_before [do_clear(ARG1), do_clear(ARG2)] then producer stack(ARG1, ARG2).",
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
		_format_tagged_block("construction_protocol", construction_protocol_block),
		_format_tagged_block("forbidden_patterns", forbidden_patterns_block),
		_format_tagged_block("abstract_composition_examples", examples_block),
		_format_tagged_block(
			"silent_self_check",
			"Silently verify: every declared task appears once; every constructive branch has a real subtask; every referenced compound child is defined; recursive siblings change an AUX_* witness; typed_task_signature is respected; final answer is minified JSON only.",
		),
		_format_tagged_block(
			"output_schema",
			"Only define tasks; primitive tasks are injected automatically.\n"
			"Define every declared compound task once, including helper tasks.\n"
			"Each noop or constructive entry compiles to one HDDL method. Prefer precondition/subtasks/ordering. Omit source_predicates when unsure.\n"
			"Constructive branch shape: precondition/context, optional support_before, one producer, optional followup or ordered_subtasks/ordering.\n"
			f"{schema_hint}",
		),
	]
	return "\n\n".join(section for section in sections if section).strip() + "\n"
