"""
Focused tests for PANDA HDDL export and plan parsing.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
)
from stage4_panda_planning.panda_planner import PANDAPlanner
from stage4_panda_planning.problem_builder import (
	PANDAProblemBuilder,
	PANDAProblemBuilderConfig,
)
from utils.hddl_parser import HDDLParser


def _domain():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "src"
		/ "domains"
		/ "blocksworld"
		/ "domain.hddl"
	)
	return HDDLParser.parse_domain(str(domain_path))


def _library() -> HTNMethodLibrary:
	return HTNMethodLibrary(
		compound_tasks=[
			HTNTask("achieve_on", ("B1", "B2"), False, ("on",)),
			HTNTask("achieve_holding", ("B",), False, ("holding",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="achieve_on__via_put_on_block",
				task_name="achieve_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="achieve_holding",
						args=("B1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_on_block",
						args=("B1", "B2"),
						kind="primitive",
						action_name="put-on-block",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
	)


def test_panda_problem_export_builds_canonical_blocksworld_init():
	builder = PANDAProblemBuilder()
	problem_hddl = builder.build_problem_hddl(
		domain=_domain(),
		domain_name="blocksworld_transition_1",
		objects=("a", "b"),
		task_name="achieve_on",
		task_args=("a", "b"),
	)

	assert "(:domain blocksworld_transition_1)" in problem_hddl
	assert "(:objects a b - block)" in problem_hddl
	assert "(handempty)" in problem_hddl
	assert "(ontable a)" in problem_hddl
	assert "(clear b)" in problem_hddl
	assert "(t1 (achieve_on a b))" in problem_hddl


def test_panda_problem_builder_accepts_explicit_initial_fact_configuration():
	builder = PANDAProblemBuilder(
		config=PANDAProblemBuilderConfig(
			initial_facts=("(handempty)", "(ontable a)"),
		),
	)
	problem_hddl = builder.build_problem_hddl(
		domain=_domain(),
		domain_name="blocksworld_transition_1",
		objects=("a", "b"),
		task_name="achieve_on",
		task_args=("a", "b"),
	)

	assert "(handempty)" in problem_hddl
	assert "(ontable a)" in problem_hddl
	assert "(ontable b)" not in problem_hddl
	assert "(clear a)" not in problem_hddl


def test_panda_domain_export_uses_llm_method_library():
	planner = PANDAPlanner()
	domain_hddl = planner._build_domain_hddl(
		domain=_domain(),
		method_library=_library(),
		domain_name="blocksworld_transition_1",
	)

	assert "(define (domain blocksworld_transition_1)" in domain_hddl
	assert "(:task achieve_on" in domain_hddl
	assert "(:method achieve_on__via_put_on_block" in domain_hddl
	assert ":subtasks (and" in domain_hddl
	assert "(s2 (put-on-block ?b1 ?b2))" in domain_hddl
	assert "(< s1 s2)" in domain_hddl
	assert "(:action put-on-block" in domain_hddl
	assert "(:method achieve_on__guard" in domain_hddl
	assert ":precondition (and (on ?b1 ?b2))" in domain_hddl


def test_panda_domain_export_adds_positive_guard_for_non_target_negative_helper():
	planner = PANDAPlanner()
	domain_hddl = planner._build_domain_hddl(
		domain=_domain(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("maintain_not_handempty", (), False, ("handempty",)),
			],
			primitive_tasks=[],
			methods=[],
			target_literals=[HTNLiteral("on", ("a", "b"), True, "on_a_b")],
		),
		domain_name="blocksworld_transition_1",
	)

	assert "(:method maintain_not_handempty__guard" in domain_hddl
	assert ":precondition (and (handempty))" in domain_hddl


def test_panda_plan_parser_extracts_primitive_steps():
	planner = PANDAPlanner()
	plan_text = "0: (pick-up-from-table a)\n1: (put-on-block a b)\n"
	steps = planner._parse_plan_steps(plan_text, _domain())

	assert [step.task_name for step in steps] == ["pick_up_from_table", "put_on_block"]
	assert steps[0].args == ("a",)
	assert steps[1].args == ("a", "b")


def test_panda_plan_parser_extracts_converted_plan_steps():
	planner = PANDAPlanner()
	plan_text = (
		"==>\n"
		"15 pick-up-from-table a\n"
		"16 put-on-block a b\n"
		"root 2\n"
		"2 achieve_on a b -> achieve_on__via_put_on_block 8 3 16\n"
	)
	steps = planner._parse_plan_steps(plan_text, _domain())

	assert [step.task_name for step in steps] == ["pick_up_from_table", "put_on_block"]
	assert steps[0].args == ("a",)
	assert steps[1].args == ("a", "b")
