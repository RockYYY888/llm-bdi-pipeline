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
	planner = PANDAPlanner()
	problem_hddl = planner._build_problem_hddl(
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
	assert "(s2 (put_on_block ?b1 ?b2))" in domain_hddl
	assert "(:action put-on-block" in domain_hddl


def test_panda_plan_parser_extracts_primitive_steps():
	planner = PANDAPlanner()
	plan_text = "0: (pick-up-from-table a)\n1: (put-on-block a b)\n"
	steps = planner._parse_plan_steps(plan_text, _domain())

	assert [step.task_name for step in steps] == ["pick_up_from_table", "put_on_block"]
	assert steps[0].args == ("a",)
	assert steps[1].args == ("a", "b")
