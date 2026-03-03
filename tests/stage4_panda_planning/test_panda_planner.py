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
	HTNTargetTaskBinding,
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
			HTNTask("place_on", ("B1", "B2"), False, ("on",)),
			HTNTask("hold_block", ("B",), False, ("holding",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
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
		target_task_bindings=[
			HTNTargetTaskBinding("on(a, b)", "place_on"),
		],
	)


def test_panda_planner_resolves_relative_workspace_to_absolute_path():
	planner = PANDAPlanner(workspace="tests/logs")

	assert planner.workspace is not None
	assert planner.workspace.is_absolute()


def test_panda_problem_export_builds_canonical_blocksworld_init():
	builder = PANDAProblemBuilder()
	problem_hddl = builder.build_problem_hddl(
		domain=_domain(),
		domain_name="blocksworld_transition_1",
		objects=("a", "b"),
		task_name="place_on",
		task_args=("a", "b"),
	)

	assert "(:domain blocksworld_transition_1)" in problem_hddl
	assert "(:objects a b - block)" in problem_hddl
	assert "(handempty)" in problem_hddl
	assert "(ontable a)" in problem_hddl
	assert "(clear b)" in problem_hddl
	assert "(t1 (place_on a b))" in problem_hddl


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
		task_name="place_on",
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
	assert "(:task place_on" in domain_hddl
	assert "(:method m_place_on_stack" in domain_hddl
	assert ":subtasks (and" in domain_hddl
	assert "(s2 (put-on-block ?b1 ?b2))" in domain_hddl
	assert "(< s1 s2)" in domain_hddl
	assert "(:action put-on-block" in domain_hddl
	assert "(:method m_place_on_noop" in domain_hddl
	assert ":precondition (and (on ?b1 ?b2))" in domain_hddl


def test_panda_domain_export_adds_positive_guard_for_non_target_helper():
	planner = PANDAPlanner()
	domain_hddl = planner._build_domain_hddl(
		domain=_domain(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("free_hand", (), False, ("handempty",)),
			],
			primitive_tasks=[],
			methods=[],
			target_literals=[],
			target_task_bindings=[],
		),
		domain_name="blocksworld_transition_1",
	)

	assert "(:method m_free_hand_noop" in domain_hddl
	assert ":precondition (and (handempty))" in domain_hddl


def test_panda_domain_export_adds_negative_guard_for_negative_target_task():
	planner = PANDAPlanner()
	domain_hddl = planner._build_domain_hddl(
		domain=_domain(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("keep_clear", ("B",), False, ("clear",)),
			],
			primitive_tasks=[],
			methods=[],
			target_literals=[HTNLiteral("clear", ("a",), False, "clear_a")],
			target_task_bindings=[HTNTargetTaskBinding("!clear(a)", "keep_clear")],
		),
		domain_name="blocksworld_transition_1",
	)

	assert "(:method m_keep_clear_noop" in domain_hddl
	assert ":precondition (and (not (clear ?b)))" in domain_hddl


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
		"2 place_on a b -> m_place_on_stack 8 3 16\n"
	)
	steps = planner._parse_plan_steps(plan_text, _domain())

	assert [step.task_name for step in steps] == ["pick_up_from_table", "put_on_block"]
	assert steps[0].args == ("a",)
	assert steps[1].args == ("a", "b")


def test_root_method_wrapper_forces_selected_root_method_without_removing_helpers():
	planner = PANDAPlanner()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
			HTNTask("clear_top", ("BLOCK",), False, ("clear",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(HTNLiteral("holding", ("BLOCK",), True, None),),
			),
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("BLOCK",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("BLOCK",),
						kind="compound",
					),
				),
			),
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("BLOCK",),
				context=(HTNLiteral("clear", ("BLOCK",), True, None),),
			),
		],
	)

	wrapped, wrapper_task_name = planner._wrap_library_for_root_method(library, library.methods[1])

	assert [method.method_name for method in wrapped.methods if method.task_name == "hold_block"] == [
		"m_hold_block_noop",
		"m_hold_block_from_table",
	]
	assert [method.method_name for method in wrapped.methods if method.task_name == "clear_top"] == [
		"m_clear_top_noop",
	]
	assert wrapper_task_name.startswith("validate_m_hold_block_from_table")
	wrapper_task = next(task for task in wrapped.compound_tasks if task.name == wrapper_task_name)
	assert wrapper_task.parameters == ("BLOCK",)
	assert [method.method_name for method in wrapped.methods if method.task_name == wrapper_task_name] == [
		f"m_{wrapper_task_name}_entry",
	]


def test_panda_domain_export_renders_equality_constraints():
	planner = PANDAPlanner()
	domain_hddl = planner._build_domain_hddl(
		domain=_domain(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("keep_apart", ("LEFT", "RIGHT"), False, ()),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_keep_apart_distinct",
					task_name="keep_apart",
					parameters=("LEFT", "RIGHT"),
					context=(
						HTNLiteral("=", ("LEFT", "RIGHT"), False, None),
					),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		domain_name="blocksworld_transition_1",
	)

	assert ":precondition (and (not (= ?left ?right)))" in domain_hddl


def test_domain_export_can_suppress_auto_generated_guard_for_root_task():
	planner = PANDAPlanner()
	domain_hddl = planner._build_domain_hddl(
		domain=_domain(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_place_on_stack",
					task_name="place_on",
					parameters=("BLOCK1", "BLOCK2"),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="put_on_block",
							args=("BLOCK1", "BLOCK2"),
							kind="primitive",
							action_name="put-on-block",
						),
					),
				),
			],
			target_literals=[HTNLiteral("on", ("a", "b"), True, "on_a_b")],
			target_task_bindings=[HTNTargetTaskBinding("on(a, b)", "place_on")],
		),
		domain_name="blocksworld_transition_1",
		suppress_guard_tasks={"place_on"},
	)

	assert "(:method m_place_on_noop" not in domain_hddl
