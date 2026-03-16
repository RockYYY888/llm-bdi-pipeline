"""
Focused integration tests for Stage 4 PANDA planning.
"""

import sys
from pathlib import Path

_tests_dir = str(Path(__file__).parent.parent)
if _tests_dir not in sys.path:
	sys.path.insert(0, _tests_dir)
_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTargetTaskBinding,
	HTNTask,
)
from stage4_panda_planning.panda_planner import PANDAPlanner
from utils.hddl_parser import HDDLParser

OFFICIAL_BLOCKSWORLD_DOMAIN_FILE = (
	Path(__file__).parent.parent.parent / "src" / "domains" / "blocksworld" / "domain.hddl"
)


def _domain():
	return HDDLParser.parse_domain(str(OFFICIAL_BLOCKSWORLD_DOMAIN_FILE))


def test_stage4_planner_generates_plan_for_valid_blocksworld_library(tmp_path):
	domain = _domain()
	planner = PANDAPlanner(workspace=str(tmp_path))
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("X", "Y"), False, ("on",)),
			HTNTask("hold_block", ("X",), False, ("holding",)),
		],
		primitive_tasks=[
			HTNTask("pick_up", ("X",), True),
			HTNTask("stack", ("X", "Y"), True),
		],
		methods=[
			HTNMethod(
				method_name="m_place_on_already",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("on", ("X", "Y"), True, None),),
			),
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("clear", ("Y",), True, None),),
				subtasks=(
					HTNMethodStep("s1", "hold_block", ("X",), "compound"),
					HTNMethodStep("s2", "stack", ("X", "Y"), "primitive", action_name="stack"),
				),
				ordering=(("s1", "s2"),),
			),
			HTNMethod(
				method_name="m_hold_block_already",
				task_name="hold_block",
				parameters=("X",),
				context=(HTNLiteral("holding", ("X",), True, None),),
			),
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("X",),
				context=(
					HTNLiteral("ontable", ("X",), True, None),
					HTNLiteral("clear", ("X",), True, None),
				),
				subtasks=(
					HTNMethodStep("s1", "pick_up", ("X",), "primitive", action_name="pick-up"),
				),
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), True, "on_a_b")],
		target_task_bindings=[HTNTargetTaskBinding("on(a, b)", "place_on")],
	)
	target_literal = method_library.target_literals[0]
	plan = planner.plan(
		domain=domain,
		method_library=method_library,
		objects=("a", "b"),
		target_literal=target_literal,
		task_name="place_on",
		transition_name="transition_1",
		initial_facts=("(ontable a)", "(clear a)", "(clear b)", "(handempty)"),
	)

	assert plan.task_name == "place_on"
	assert plan.task_args == ("a", "b")
	assert plan.steps
	assert [step.task_name for step in plan.steps] == ["pick_up", "stack"]
	assert "(t1 (place_on a b))" in plan.problem_hddl


def test_stage4_domain_export_preserves_declared_source_names_for_hyphenated_tasks(tmp_path):
	domain = HDDLParser.parse_domain("src/domains/marsrover/domain.hddl")
	planner = PANDAPlanner(workspace=str(tmp_path))
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				"empty_store",
				("STORE", "ROVER"),
				False,
				("empty",),
				source_name="empty-store",
			),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_empty_store_ready",
				task_name="empty_store",
				parameters=("STORE", "ROVER"),
				context=(HTNLiteral("empty", ("STORE",), True, None),),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	domain_hddl = planner._build_domain_hddl(domain, method_library, "marsrover_alias_test")

	assert "(:task empty-store" in domain_hddl
	assert ":task (empty-store ?store ?rover)" in domain_hddl
	assert "(:task empty_store" not in domain_hddl
