"""
Query-conditioned HTN prompt builders.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from .semantic_rules import (
	_anchor_display_name,
	_build_query_task_contract_payloads,
	_build_support_task_contract_payloads,
	_constructive_template_summary_for_task,
	_format_tagged_block,
	_normalise_action_analysis,
	_normalise_query_object_inventory,
	_parameter_token,
	_query_object_names_from_inventory,
	_relevant_action_names_for_prompt,
	_relevant_dynamic_predicates_for_prompt,
	_render_consumer_template_summary_for_task,
	_render_contract_blocks,
	_reusable_dynamic_resource_payloads,
	_sanitize_name,
	_shared_dynamic_requirements_for_predicate,
	_task_headline_candidate_map,
	_task_invocation_signature,
	_unique_preserve_order,
)

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
			"- Ordered target bindings below are the authoritative grounded binding source for method synthesis.",
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
			"Generate one executable JSON HTN library with stable planner-safe identifiers. Keep it compact only after branch obligations are explicit.",
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
