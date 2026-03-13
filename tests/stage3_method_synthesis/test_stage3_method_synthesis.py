"""
Focused tests for Stage 3 HTN method synthesis.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import (
	LTLFormula,
	LTLSpecification,
	TemporalOperator,
)
from stage3_method_synthesis.htn_method_synthesis import HTNMethodSynthesizer, HTNSynthesisError
from stage3_method_synthesis.htn_prompts import (
	build_htn_system_prompt,
	build_htn_user_prompt,
)
from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
	HTNTargetTaskBinding,
)
from utils.config import get_config
from utils.hddl_parser import HDDLParser


def _domain():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "tests"
		/ "fixtures"
		/ "domains"
		/ "legacy_blocksworld"
		/ "domain.hddl"
	)
	return HDDLParser.parse_domain(str(domain_path))


def _marsrover_domain():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "src"
		/ "domains"
		/ "marsrover"
		/ "domain.hddl"
	)
	return HDDLParser.parse_domain(str(domain_path))


def _live_stage3_kwargs() -> dict:
	config = get_config()
	if not config.validate():
		pytest.skip("Stage 3 live tests require a valid OPENAI_API_KEY")

	return {
		"api_key": config.openai_api_key,
		"model": config.openai_model,
		"base_url": config.openai_base_url,
		"timeout": float(config.openai_timeout),
	}


def _atomic_formula(predicate: str, args: list[str]) -> LTLFormula:
	return LTLFormula(
		operator=None,
		predicate={predicate: args},
		sub_formulas=[],
		logical_op=None,
	)


def _eventually_on_spec() -> LTLSpecification:
	spec = LTLSpecification()
	spec.objects = ["a", "b"]
	spec.grounding_map = GroundingMap()
	spec.grounding_map.add_atom("on_a_b", "on", ["a", "b"])
	spec.formulas = [
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[_atomic_formula("on", ["a", "b"])],
			logical_op=None,
		),
	]
	return spec


def _dfa_result_for_labels(*labels: str) -> dict:
	edges = "\n".join(f'  0 -> 1 [label="{label}"];' for label in labels)
	return {
		"dfa_dot": (
			"digraph {\n"
			"  0 [shape=circle];\n"
			"  1 [shape=doublecircle];\n"
			f"{edges}\n"
			"}\n"
		),
	}


def test_extract_target_literals_discards_non_progressing_transitions():
	synthesizer = HTNMethodSynthesizer()
	grounding_map = GroundingMap()
	grounding_map.add_atom("clear_a", "clear", ["a"])
	grounding_map.add_atom("on_a_b", "on", ["a", "b"])
	grounding_map.add_atom("on_b_c", "on", ["b", "c"])
	dfa_result = {
		"dfa_dot": (
			"digraph MONA_DFA {\n"
			"  node [shape = doublecircle]; 3;\n"
			"  node [shape = circle]; 1 2 s1 s2;\n"
			"  init [shape = plaintext, label = \"\"];\n"
			"  init -> 1;\n"
			"  1 -> 2 [label=\"!clear_a\"];\n"
			"  1 -> s1 [label=\"clear_a\"];\n"
			"  2 -> 2 [label=\"!clear_a\"];\n"
			"  2 -> 2 [label=\"clear_a\"];\n"
			"  3 -> 3 [label=\"!clear_a\"];\n"
			"  3 -> 3 [label=\"clear_a\"];\n"
			"  s1 -> 2 [label=\"!on_a_b\"];\n"
			"  s1 -> s2 [label=\"on_a_b\"];\n"
			"  s2 -> 2 [label=\"!on_b_c\"];\n"
			"  s2 -> 3 [label=\"on_b_c\"];\n"
			"}\n"
		),
	}

	literals = synthesizer.extract_target_literals(grounding_map, dfa_result)

	assert [literal.to_signature() for literal in literals] == [
		"clear(a)",
		"on(a, b)",
		"on(b, c)",
	]

	transition_specs = synthesizer.extract_progressing_transitions(grounding_map, dfa_result)

	assert [
		(
			spec["transition_name"],
			spec["source_state"],
			spec["target_state"],
			spec["label"],
		)
		for spec in transition_specs
	] == [
		("dfa_step_q1_q2_clear_a", "q1", "q2", "clear(a)"),
		("dfa_step_q2_q3_on_a_b", "q2", "q3", "on(a, b)"),
		("dfa_step_q3_q4_on_b_c", "q3", "q4", "on(b, c)"),
	]


def test_extract_progressing_transitions_can_follow_explicit_literal_order():
	synthesizer = HTNMethodSynthesizer()
	grounding_map = GroundingMap()
	grounding_map.add_atom("on_b4_b2", "on", ["b4", "b2"])
	grounding_map.add_atom("on_b1_b4", "on", ["b1", "b4"])
	grounding_map.add_atom("on_b3_b1", "on", ["b3", "b1"])
	dfa_result = {
		"dfa_dot": (
			"digraph MONA_DFA {\n"
			"  node [shape = doublecircle]; 3;\n"
			"  node [shape = circle]; 1 s1 s2;\n"
			"  init [shape = plaintext, label = \"\"];\n"
			"  init -> 1;\n"
			"  1 -> s1 [label=\"on_b1_b4\"];\n"
			"  s1 -> s2 [label=\"on_b3_b1\"];\n"
			"  s2 -> 3 [label=\"on_b4_b2\"];\n"
			"}\n"
		),
	}

	transition_specs = synthesizer.extract_progressing_transitions(
		grounding_map,
		dfa_result,
		ordered_literal_signatures=[
			"on(b4, b2)",
			"on(b1, b4)",
			"on(b3, b1)",
		],
	)

	assert [
		(
			spec["transition_name"],
			spec["source_state"],
			spec["target_state"],
			spec["label"],
		)
		for spec in transition_specs
	] == [
		("dfa_step_q1_q2_on_b4_b2", "q1", "q2", "on(b4, b2)"),
		("dfa_step_q2_q3_on_b1_b4", "q2", "q3", "on(b1, b4)"),
		("dfa_step_q3_q4_on_b3_b1", "q3", "q4", "on(b3, b1)"),
	]


def test_extract_progressing_transitions_can_drop_auxiliary_dfa_labels_when_query_order_is_known():
	synthesizer = HTNMethodSynthesizer()
	grounding_map = GroundingMap()
	grounding_map.add_atom("on_b4_b2", "on", ["b4", "b2"])
	grounding_map.add_atom("on_b1_b4", "on", ["b1", "b4"])
	grounding_map.add_atom("on_b3_b1", "on", ["b3", "b1"])
	dfa_result = {
		"dfa_dot": (
			"digraph MONA_DFA {\n"
			"  node [shape = doublecircle]; 4;\n"
			"  node [shape = circle]; 1 s1 s2 s3;\n"
			"  init [shape = plaintext, label = \"\"];\n"
			"  init -> 1;\n"
			"  1 -> s1 [label=\"on_b1_b4\"];\n"
			"  s1 -> s2 [label=\"!on_b1_b4\"];\n"
			"  s2 -> s3 [label=\"on_b3_b1\"];\n"
			"  s3 -> 4 [label=\"on_b4_b2\"];\n"
			"}\n"
		),
	}

	transition_specs = synthesizer.extract_progressing_transitions(
		grounding_map,
		dfa_result,
		ordered_literal_signatures=[
			"on(b4, b2)",
			"on(b1, b4)",
			"on(b3, b1)",
		],
	)

	assert [spec["label"] for spec in transition_specs] == [
		"on(b4, b2)",
		"on(b1, b4)",
		"on(b3, b1)",
	]


def test_synthesize_requires_live_llm():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "src"
		/ "domains"
		/ "blocksworld"
		/ "domain.hddl"
	)
	domain = HDDLParser.parse_domain(str(domain_path))
	grounding_map = GroundingMap()
	grounding_map.add_atom("on_b4_b2", "on", ["b4", "b2"])
	grounding_map.add_atom("on_b1_b4", "on", ["b1", "b4"])
	grounding_map.add_atom("on_b3_b1", "on", ["b3", "b1"])
	dfa_result = {
		"dfa_dot": (
			"digraph MONA_DFA {\n"
			"  node [shape = doublecircle]; 3;\n"
			"  node [shape = circle]; 1 s1 s2;\n"
			"  init [shape = plaintext, label = \"\"];\n"
			"  init -> 1;\n"
			"  1 -> s1 [label=\"on_b1_b4\"];\n"
			"  s1 -> s2 [label=\"on_b3_b1\"];\n"
			"  s2 -> 3 [label=\"on_b4_b2\"];\n"
			"}\n"
		),
	}

	with pytest.raises(HTNSynthesisError) as exc_info:
		HTNMethodSynthesizer().synthesize(
			domain=domain,
			grounding_map=grounding_map,
			dfa_result=dfa_result,
			ordered_literal_signatures=["on(b4, b2)", "on(b1, b4)", "on(b3, b1)"],
		)

	assert "requires a configured OPENAI_API_KEY" in str(exc_info.value)


def test_extract_target_literals_keeps_accepting_loops_when_no_progress_edge_exists():
	synthesizer = HTNMethodSynthesizer()
	grounding_map = GroundingMap()
	grounding_map.add_atom("clear_a", "clear", ["a"])
	dfa_result = {
		"dfa_dot": (
			"digraph MONA_DFA {\n"
			"  0 [shape = doublecircle];\n"
			"  1 [shape = circle];\n"
			"  0 -> 0 [label=\"clear_a\"];\n"
			"  0 -> 1 [label=\"!clear_a\"];\n"
			"  1 -> 1 [label=\"clear_a\"];\n"
			"  1 -> 1 [label=\"!clear_a\"];\n"
			"}\n"
		),
	}

	literals = synthesizer.extract_target_literals(grounding_map, dfa_result)

	assert [literal.to_signature() for literal in literals] == ["clear(a)"]


def test_method_synthesizer_uses_live_llm_output():
	domain = _domain()
	spec = _eventually_on_spec()
	dfa_result = _dfa_result_for_labels("on_a_b")

	library, metadata = HTNMethodSynthesizer(
		**_live_stage3_kwargs(),
	).synthesize(
		domain=domain,
		grounding_map=spec.grounding_map,
		dfa_result=dfa_result,
	)

	assert metadata["used_llm"] is True
	assert metadata["llm_prompt"] is not None
	assert metadata["llm_response"]
	assert metadata["target_literals"] == ["on(a, b)"]
	assert metadata["compound_tasks"] >= 1
	assert metadata["methods"] >= 1
	assert library.compound_tasks
	assert library.methods
	assert library.target_task_bindings

	primitive_task_names = {task.name for task in library.primitive_tasks}
	assert primitive_task_names == {
		"pick_up",
		"pick_up_from_table",
		"put_on_block",
		"put_down",
	}

	compound_task_names = {task.name for task in library.compound_tasks}
	assert all(method.task_name in compound_task_names for method in library.methods)
	assert all(
		not task_name.startswith(("achieve_", "maintain_not_"))
		for task_name in compound_task_names
	)

	bound_task = library.task_name_for_literal(library.target_literals[0])
	assert bound_task in compound_task_names


def test_method_synthesizer_requires_target_task_binding_for_each_target_literal():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
	)

	with pytest.raises(ValueError, match="missing a target_task_binding"):
		synthesizer._validate_library(library, domain)


def test_normalise_library_repairs_subtask_kind_by_declared_task_sets():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("communicate_rock_data", ("P",), False, ("communicated_rock_data",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_communicate_rock_data_direct",
				task_name="communicate_rock_data",
				parameters=("P",),
				context=(HTNLiteral("have_rock_analysis", ("ROVER", "P"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="communicate_rock_data",
						args=("ROVER", "L", "P", "X", "Y"),
						kind="primitive",
						action_name="communicate_rock_data",
						literal=HTNLiteral("communicated_rock_data", ("P",), True, None),
						preconditions=(),
						effects=(),
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("communicated_rock_data", ("waypoint2",), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding("communicated_rock_data(waypoint2)", "communicate_rock_data"),
		],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	[method] = normalised.methods
	[step] = method.subtasks
	assert step.kind == "compound"
	assert step.task_name == "communicate_rock_data"


def test_validate_library_rejects_constant_symbol_type_conflicts():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("communicate_image", ("OBJECTIVE", "MODE"), False, ("communicated_image_data",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_communicate_image_noop",
				task_name="communicate_image",
				parameters=("OBJECTIVE", "MODE"),
				context=(
					HTNLiteral("communicated_image_data", ("objective0", "low_res"), True, None),
					HTNLiteral("at_lander", ("objective0", "waypoint0"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("communicated_image_data", ("objective0", "low_res"), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding(
				"communicated_image_data(objective0, low_res)",
				"communicate_image",
			),
		],
	)

	with pytest.raises(ValueError, match="objective0"):
		synthesizer._validate_library(library, domain)


def test_validate_library_rejects_task_source_predicate_arity_mismatch():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("navigate_rover", ("ROVER", "FROM_WP", "TO_WP"), False, ("at",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_navigate_rover_direct",
				task_name="navigate_rover",
				parameters=("ROVER", "FROM_WP", "TO_WP"),
				context=(
					HTNLiteral("at", ("ROVER", "FROM_WP"), True, None),
					HTNLiteral("can_traverse", ("ROVER", "FROM_WP", "TO_WP"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("at", ("rover0", "waypoint5"), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("at(rover0, waypoint5)", "navigate_rover")],
	)

	with pytest.raises(ValueError, match="arity mismatch"):
		synthesizer._validate_library(library, domain)


def test_target_guard_completion_adds_missing_already_satisfied_method():
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
		],
		methods=[
			HTNMethod(
				method_name="m_place_on_direct",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("holding", ("BLOCK1",), True, None),
					HTNLiteral("clear", ("BLOCK2",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="put-on-block",
					),
				),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("on(a, b)", "place_on"),
		],
	)

	completed_library, generated_count = synthesizer._complete_missing_target_guard_methods(library)

	assert generated_count == 1
	generated_methods = [
		method
		for method in completed_library.methods
		if method.origin == "stage3_target_guard_completion"
	]
	assert len(generated_methods) == 1
	assert generated_methods[0].task_name == "place_on"
	assert generated_methods[0].context == (
		HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
	)
	assert generated_methods[0].subtasks == ()


def test_stage3_prompts_make_binding_and_naming_rules_explicit():
	system_prompt = build_htn_system_prompt()
	user_prompt = build_htn_user_prompt(
		_domain(),
		["on(a, b)", "!clear(a)"],
		'{"target_task_bindings": [], "compound_tasks": [], "methods": []}',
	)

	assert "OUTPUT CONTRACT:" in system_prompt
	assert "HIDDEN REASONING RECIPE (DO NOT OUTPUT IT):" in system_prompt
	assert "Prefer reusable parameterized compound tasks" in system_prompt
	assert "Do not encode grounded object constants into compound task names" in system_prompt
	assert "Never use legacy task prefixes achieve_, ensure_, goal_, or maintain_not_" in system_prompt
	assert "must be JSON literal objects, never strings" in system_prompt
	assert "PRIORITY ORDER: validity of JSON and bindings > executability of methods > semantic alignment > branch coverage." in system_prompt
	assert "not handempty alone does not identify what is being held" in system_prompt
	assert "identify every non-equivalent final primitive action or final helper" in system_prompt
	assert "If several different final actions can establish the same literal" in system_prompt
	assert "Helper coverage is not a substitute for target-task coverage" in system_prompt
	assert "missing-first, missing-second, and missing-both" in system_prompt
	assert "Keep a missing-one-support branch only when the already-satisfied support can remain true" in system_prompt
	assert "direct-blocker and recursively-blocked-blocker cases" in system_prompt
	assert "recursive blocked-blocker sibling is mandatory" in system_prompt
	assert "positive support literal q(BLOCKER, ...)" in system_prompt
	assert "check helper after-effects" in system_prompt
	assert "A missing-support branch is invalid if its current context already blocks the first helper" in system_prompt
	assert "Prefer reusable helper end states over transient ones" in system_prompt
	assert "A clear-like helper is incomplete if it only has already-satisfied and direct-blocker methods" in user_prompt
	assert "Reject any helper cycle" in system_prompt
	assert "Disjunctive action applicability from or / imply must become distinguishable sibling methods." in system_prompt
	assert "Never reveal chain-of-thought or hidden reasoning." in system_prompt

	assert "DOMAIN TYPES:" in user_prompt
	assert "DECLARED DOMAIN TASKS:" in user_prompt
	assert "ACTION PRECONDITION BRANCH HINTS (DNF):" in user_prompt
	assert "REQUIRED target_task_bindings ENTRIES:" in user_prompt
	assert "MANDATORY TARGET-TASK CONSTRUCTION PROTOCOL:" in user_prompt
	assert "DECISION PRIORITY:" in user_prompt
	assert "If a candidate sibling is not executable, omit it" in user_prompt
	assert "Reuse one parameterized task" in user_prompt
	assert "missing-both-support" in user_prompt
	assert "Keep a missing-one-support branch only when the already-satisfied support can stay true" in user_prompt
	assert "Few-shot guidance (illustrative only):" in user_prompt
	assert "Example A: branch families for a positive target" in user_prompt
	assert "Example B: blocked-clear resource conflict" in user_prompt
	assert "Example C: OR / IMPLY must become sibling methods" in user_prompt
	assert "Example D: recursive blocker removal" in user_prompt
	assert "Example E: post-helper resource conflict" in user_prompt
	assert "Example F: invalid missing-second-support branch" in user_prompt
	assert "Example G: do not preserve a conflicting support" in user_prompt
	assert "Example G2: explicit repaired missing_clear branch" in user_prompt
	assert "Example H: underspecified recovery branch" in user_prompt
	assert "Example I: reusable helper end state" in user_prompt
	assert "Example J: omit underspecified make_holding branch" in user_prompt
	assert "Example J2: same literal, different final actions" in user_prompt
	assert "Example J3: helper coverage does not replace target coverage" in user_prompt
	assert "Example K: exact clear-helper recursion pattern" in user_prompt
	assert "missing-both coverage is mandatory unless provably impossible" in user_prompt
	assert "do not clone a new grounded task for each target literal instance" in user_prompt
	assert "Never use legacy task prefixes achieve_, ensure_, goal_, or maintain_not_" in user_prompt
	assert "Do not invent grounded task names like achieve_p_a_b" in user_prompt
	assert "recursive-blocker: on(BLOCKER, TARGET) & not clear(BLOCKER)" in user_prompt
	assert "complementary-support rule" in user_prompt
	assert "Invalid pattern: only the already-satisfied branch and the direct-blocker branch." in user_prompt
	assert "positive support literal about a blocker/support object" in user_prompt
	assert "This is mandatory." in user_prompt
	assert "direct blocker-removal branch requires clear(BLOCKER)" in user_prompt
	assert "must use object form with predicate/args/is_positive" in user_prompt
	assert "Do not use unbound disposer variables" in user_prompt
	assert "inventing put_down(Z)" in user_prompt
	assert "carried object is not named in context or parameters" in user_prompt
	assert "clear_slot(SLOT); pick_up(ITEM)" in user_prompt
	assert "helper likely leaves holding(BLOCKER)" in user_prompt
	assert "already blocks its first helper" in user_prompt
	assert "omit that missing-second-support branch" in user_prompt
	assert "holding(X) & not clear(Y)" in user_prompt
	assert "put_down(X); make_clear(Y); make_holding(X); stack(X, Y)" in user_prompt
	assert "put_down(Z); pick_up(X)" in user_prompt
	assert "treat recovery as underspecified and omit that branch" in user_prompt
	assert "omit the missing_handempty sibling unless the carried object is already bound" in user_prompt
	assert "table-mode acquisition and stack-mode acquisition" in user_prompt
	assert "multiple primitive actions can establish the same headline literal" in user_prompt
	assert "Helper coverage does not replace target-task coverage" in user_prompt
	assert "do not emit `holding(X) & not clear(Y) -> clear_helper(Y); final_step`" in user_prompt
	assert "unstack(BLOCKER, TARGET); put_down(BLOCKER)" in user_prompt
	assert "context on(B, X) & not clear(B)" in user_prompt
	assert "FINAL SILENT CHECKLIST:" in user_prompt
	assert "Return one complete JSON object and nothing else." in user_prompt


def test_stage3_prompt_stays_compact_for_multi_goal_blocksworld_case():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	targets = [
		"on(b3, b5)",
		"on(b6, b3)",
		"on(b1, b6)",
		"on(b2, b1)",
		"on(b4, b2)",
		"on(b7, b4)",
	]
	system_prompt = build_htn_system_prompt()
	user_prompt = build_htn_user_prompt(
		domain,
		targets,
		HTNMethodSynthesizer._schema_hint(),
	)

	assert len(system_prompt) + len(user_prompt) < 21500


def test_stage3_user_prompt_includes_disjunctive_action_branch_hints():
	domain = type(
		"DomainStub",
		(),
		{
			"name": "branch_domain",
			"types": ["object"],
			"predicates": [],
			"actions": [
				type(
					"ActionStub",
					(),
					{
						"name": "probe",
						"parameters": ["?x - object"],
						"preconditions": "(or (clear ?x) (holding ?x))",
						"effects": "(and (checked ?x))",
					},
				)(),
				type(
					"ActionStub",
					(),
					{
						"name": "seal_if_clear",
						"parameters": ["?x - object"],
						"preconditions": "(imply (clear ?x) (holding ?x))",
						"effects": "(and (sealed ?x))",
					},
				)(),
			],
		},
	)()

	user_prompt = build_htn_user_prompt(
		domain,
		["checked(a)"],
		'{"target_task_bindings": [], "compound_tasks": [], "methods": []}',
	)

	assert "probe applicability branches: [clear(?x)] OR [holding(?x)]" in user_prompt
	assert "seal_if_clear applicability branches: [not clear(?x)] OR [holding(?x)]" in user_prompt


def test_method_synthesizer_rejects_llm_identifiers_that_need_silent_sanitising():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place-on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place-on_noop",
				task_name="place-on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)
	assert normalised.compound_tasks[0].name == "place-on"

	with pytest.raises(ValueError, match="Invalid task identifier 'place-on'"):
		synthesizer._validate_library(normalised, domain)


def test_normalise_llm_library_rewrites_primitive_action_name_to_source_hddl_name():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B",), False, ("holding",)),
			HTNTask("clear_top", ("B",), False, ("clear",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("B",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("B",),
						kind="compound",
						action_name=None,
						literal=None,
						preconditions=(),
						effects=(),
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up_from_table",
						args=("B",),
						kind="primitive",
						action_name="pick_up_from_table",
						literal=None,
						preconditions=(),
						effects=(),
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)
	primitive_step = normalised.methods[0].subtasks[1]

	assert primitive_step.task_name == "pick_up_from_table"
	assert primitive_step.action_name == "pick-up-from-table"


def test_normalise_llm_library_promotes_used_local_variables_into_method_parameters():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("make_not_clear", ("BLOCK",), False, ("clear",)),
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_make_not_clear_acquire",
				task_name="make_not_clear",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK"),
						kind="primitive",
						action_name="put_on_block",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	assert normalised.methods[0].parameters == ("BLOCK", "BLOCK1")


def test_method_validation_rejects_legacy_task_prefixes():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("achieve_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_achieve_on_stack",
				task_name="achieve_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="Legacy task name 'achieve_on' is not allowed"):
		synthesizer._validate_library(library, domain)


def test_method_validation_enforces_method_name_contract():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="place_on_stack",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("on(a, b)", "place_on"),
		],
	)

	with pytest.raises(ValueError, match="must follow the exact naming pattern"):
		synthesizer._validate_library(library, domain)


def test_negative_target_requires_constructive_method():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("keep_not_clear", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_keep_not_clear_noop",
				task_name="keep_not_clear",
				parameters=("B",),
				context=(
					HTNLiteral("clear", ("B",), False, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("clear", ("a",), False, "clear_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("!clear(a)", "keep_not_clear"),
		],
	)

	with pytest.raises(ValueError, match="has no constructive non-zero-subtask method"):
		synthesizer._validate_library(library, domain)


def test_negative_target_binding_must_match_negative_semantics():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="put-on-block",
						literal=HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), False, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("!on(a, b)", "place_on"),
		],
	)

	with pytest.raises(ValueError, match="none of that task's methods exposes an already-satisfied context"):
		synthesizer._validate_library(library, domain)


def test_synthesize_forces_negative_literals_to_naf_signatures(monkeypatch):
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	synthesizer.client = object()

	grounding_map = GroundingMap()
	grounding_map.add_atom("on_a_b", "on", ["a", "b"])
	dfa_result = {
		"dfa_dot": (
			"digraph MONA_DFA {\n"
			"  node [shape = doublecircle]; 1;\n"
			"  node [shape = circle]; 0;\n"
			"  init [shape = plaintext, label = \"\"];\n"
			"  init -> 0;\n"
			"  0 -> 1 [label=\"!on_a_b\"];\n"
			"}\n"
		),
	}

	llm_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("remove_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_remove_on_noop",
				task_name="remove_on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), False, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_remove_on_pickup",
				task_name="remove_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("B1", "B2"),
						kind="primitive",
						action_name="pick-up",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[
			HTNTargetTaskBinding("!on(a, b)", "remove_on"),
		],
	)

	def _fake_request_complete_llm_library(
		prompt,
		domain_obj,
		metadata,
	):
		return llm_library, '{"ok": true}', "stop"

	monkeypatch.setattr(
		synthesizer,
		"_request_complete_llm_library",
		_fake_request_complete_llm_library,
	)

	library, metadata = synthesizer.synthesize(
		domain=domain,
		grounding_map=grounding_map,
		dfa_result=dfa_result,
		query_text="Keep on(a,b) explicitly false.",
	)

	assert library.target_literals[0].negation_mode == "naf"
	assert library.target_literals[0].to_signature() == "!on(a, b)"
	assert metadata["target_literals"] == ["!on(a, b)"]
	assert metadata["negation_resolution"]["mode_by_predicate"] == {"on/2": "naf"}
	assert metadata["negation_resolution"]["policy"] == "all_naf"
	assert library.target_task_bindings[0].target_literal == "!on(a, b)"


def test_method_validation_rejects_unbound_free_variables_in_subtasks():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("clear_top", ("B",), False, ("clear",)),
			HTNTask("hold_block", ("B",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_clear_top_recursive",
				task_name="clear_top",
				parameters=("B",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("TOP",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_down",
						args=("TOP",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="uses unbound variable 'TOP'"):
		synthesizer._validate_library(library, domain)


def test_method_validation_allows_local_variables_when_bound_in_context():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("clear_top", ("B",), False, ("clear",)),
			HTNTask("hold_block", ("B",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_clear_top_recursive",
				task_name="clear_top",
				parameters=("B",),
				context=(
					HTNLiteral("on", ("TOP", "B"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("TOP",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_down",
						args=("TOP",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)


def test_method_validation_allows_local_variables_when_bound_in_preconditions():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B",), False, ("holding",)),
			HTNTask("clear_top", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("B1",),
				context=(
					HTNLiteral("holding", ("B1",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("B",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("B",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up",
						args=("B", "SUPPORT"),
						kind="primitive",
						action_name="pick-up",
						preconditions=(
							HTNLiteral("on", ("B", "SUPPORT"), True, None),
						),
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)


def test_primitive_alias_cannot_use_non_primitive_subtask_kind():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B1",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("B1",),
				context=(
					HTNLiteral("holding", ("B1",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("B1", "B2"),
						kind="compound",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("holding", ("a",), True, "holding_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("holding(a)", "hold_block"),
		],
	)

	with pytest.raises(ValueError, match="Primitive aliases must use kind='primitive'"):
		synthesizer._validate_library(library, domain)


def test_sibling_constructive_methods_must_have_distinguishable_contexts():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
			HTNTask("clear_top", ("BLOCK",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("holding", ("BLOCK",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("BLOCK",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up_from_table",
						args=("BLOCK",),
						kind="primitive",
						action_name="pick-up-from-table",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_table_again",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("BLOCK",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up_from_table",
						args=("BLOCK",),
						kind="primitive",
						action_name="pick-up-from-table",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("holding", ("a",), True, "holding_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("holding(a)", "hold_block"),
		],
	)

	with pytest.raises(ValueError, match="semantically duplicate|not semantically distinguishable"):
		synthesizer._validate_library(library, domain)


def test_redundant_constructive_siblings_are_pruned_before_validation():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
			HTNTask("hold_block", ("BLOCK1",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_acquire",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="put-on-block",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("BLOCK1",),
				context=(
					HTNLiteral("holding", ("BLOCK1",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("on(a, b)", "place_on"),
		],
	)

	pruned_library, pruned_count = synthesizer._prune_redundant_constructive_siblings(
		library,
		domain,
	)

	assert pruned_count == 1
	assert {method.method_name for method in pruned_library.methods} == {
		"m_place_on_noop",
		"m_place_on_stack",
		"m_hold_block_noop",
	}
	synthesizer._validate_library(pruned_library, domain)


def test_direct_self_recursive_siblings_are_preserved_when_contexts_are_distinct():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
			HTNTask("clear_top", ("BLOCK",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("ontable", ("BLOCK",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up_from_table",
						args=("BLOCK",),
						kind="primitive",
						action_name="pick-up-from-table",
					),
				),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_clear_first",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("clear", ("BLOCK",), False, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("BLOCK",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="hold_block",
						args=("BLOCK",),
						kind="compound",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("clear", ("BLOCK",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
	)

	pruned_library, pruned_count = synthesizer._prune_redundant_constructive_siblings(
		library,
		domain,
	)

	assert pruned_count == 0
	assert {method.method_name for method in pruned_library.methods} == {
		"m_hold_block_from_table",
		"m_hold_block_clear_first",
		"m_clear_top_noop",
	}


def test_method_validation_rejects_conflicting_variable_types():
	domain = SimpleNamespace(
		name="typed_domain",
		actions=[
			SimpleNamespace(
				name="place",
				parameters=["?item - block", "?slot - location"],
				preconditions="(and)",
				effects="(and (stored ?item))",
			),
		],
		predicates=[
			SimpleNamespace(
				name="stored",
				parameters=["?item - block"],
				to_signature=lambda: "stored(?item - block)",
			),
		],
		requirements=[],
		types=["block", "location"],
	)
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("misbind", ("BLOCK",), False, ("stored",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_misbind_noop",
				task_name="misbind",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("stored", ("BLOCK",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_misbind_conflict",
				task_name="misbind",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="place",
						args=("BLOCK", "BLOCK"),
						kind="primitive",
						action_name="place",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("stored", ("a",), True, "stored_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("stored(a)", "misbind"),
		],
	)

	with pytest.raises(ValueError, match="conflicting inferred types"):
		synthesizer._validate_library(library, domain)


def test_method_validation_accepts_supported_equality_constraints():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("keep_apart", ("BLOCK1", "BLOCK2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_keep_apart_distinct",
				task_name="keep_apart",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("=", ("BLOCK1", "BLOCK2"), False, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)


def test_parse_llm_library_rejects_truncated_json_with_clear_error():
	synthesizer = HTNMethodSynthesizer()

	with pytest.raises(ValueError, match="appears truncated"):
		synthesizer._parse_llm_library(
			'{"target_task_bindings": [], "compound_tasks": [',
		)


def test_validate_library_rejects_semantically_duplicate_methods():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("holding", ("BLOCK1",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_duplicate_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("holding", ("BLOCK1",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="semantically duplicate"):
		synthesizer._validate_library(library, domain)


def test_request_complete_llm_library_fails_on_truncated_json():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer(max_attempts=2)
	prompt = {"system": "system", "user": "user"}
	metadata = {"llm_attempts": 0}

	def fake_call_llm(
		prompt_payload: dict,
		*,
		retry_instruction: str | None = None,
		max_tokens: int | None = None,
	):
		return ('{"compound_tasks": [', "length")

	synthesizer._call_llm = fake_call_llm  # type: ignore[method-assign]

	with pytest.raises(HTNSynthesisError, match="truncated before completion"):
		synthesizer._request_complete_llm_library(
			prompt,
			domain,
			metadata,
		)

	assert metadata["llm_attempts"] == 1
	assert len(metadata["llm_attempt_durations_seconds"]) == 1
	assert metadata["llm_response_time_seconds"] >= 0


def test_negative_target_binding_accepts_helper_mediated_removal():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("remove_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_remove_on_noop",
				task_name="remove_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("on", ("BLOCK1", "BLOCK2"), False, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_remove_on_from_block",
				task_name="remove_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("BLOCK", "SUPPORT"),
				context=(
					HTNLiteral("on", ("BLOCK", "SUPPORT"), True, None),
					HTNLiteral("clear", ("BLOCK",), True, None),
					HTNLiteral("handempty", (), True, None),
					HTNLiteral("=", ("BLOCK", "SUPPORT"), False, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("BLOCK", "SUPPORT"),
						kind="primitive",
						action_name="pick-up",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), False, "on_a_b")],
		target_task_bindings=[HTNTargetTaskBinding("!on(a, b)", "remove_on")],
	)

	synthesizer._validate_library(library, domain)
