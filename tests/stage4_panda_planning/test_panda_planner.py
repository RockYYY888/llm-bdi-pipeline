"""
Focused tests for PANDA HDDL export and plan parsing.
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
	HTNTask,
	HTNTargetTaskBinding,
)
from stage4_panda_planning.panda_planner import PANDAPlanner
from stage4_panda_planning.problem_builder import (
	PANDAProblemBuilder,
	PANDAProblemBuilderConfig,
)
from utils.hddl_parser import HDDLParser

OFFICIAL_BLOCKSWORLD_DOMAIN_FILE = (
	Path(__file__).parent.parent.parent / "src" / "domains" / "blocksworld" / "domain.hddl"
)


def _domain():
	return HDDLParser.parse_domain(str(OFFICIAL_BLOCKSWORLD_DOMAIN_FILE))


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
						action_name="stack",
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


def test_panda_problem_export_uses_explicit_blocksworld_style_defaults_when_requested():
	builder = PANDAProblemBuilder(
		config=PANDAProblemBuilderConfig(
			global_initial_predicates=("handempty",),
			per_object_initial_predicates=("ontable", "clear"),
		),
	)
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


def test_panda_problem_export_is_generic_by_default_without_domain_specific_facts():
	builder = PANDAProblemBuilder()
	problem_hddl = builder.build_problem_hddl(
		domain=_domain(),
		domain_name="blocksworld_transition_1",
		objects=("a", "b"),
		task_name="place_on",
		task_args=("a", "b"),
	)

	assert "(handempty)" not in problem_hddl
	assert "(ontable a)" not in problem_hddl
	assert "(clear b)" not in problem_hddl


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
	assert "(s2 (stack ?b1 ?b2))" in domain_hddl
	assert "(< s1 s2)" in domain_hddl
	assert "(:action stack" in domain_hddl


def test_panda_domain_export_defaults_task_parameter_type_to_object():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "src"
		/ "domains"
		/ "marsrover"
		/ "domain.hddl"
	)
	rover_domain = HDDLParser.parse_domain(str(domain_path))
	planner = PANDAPlanner()
	domain_hddl = planner._build_domain_hddl(
		domain=rover_domain,
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("navigate_to", ("ROVER", "FROM", "TO"), False, ()),
			],
			primitive_tasks=[],
			methods=[],
			target_literals=[],
			target_task_bindings=[],
		),
		domain_name="rover_transition_1",
	)

	assert "(:task navigate_to" in domain_hddl
	assert ":parameters (?rover - object ?from - object ?to - object)" in domain_hddl


def test_panda_domain_export_uses_leading_method_parameters_when_task_args_omitted():
	planner = PANDAPlanner()
	domain_hddl = planner._build_domain_hddl(
		domain=_domain(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("clear_top", ("B",), False, ("clear",)),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_clear_top_put_down",
					task_name="clear_top",
					parameters=("TARGET",),
					context=(HTNLiteral("holding", ("TARGET",), True, None),),
					subtasks=(
						HTNMethodStep(
							"s1",
							"put_down",
							("TARGET",),
							"primitive",
							action_name="put-down",
						),
					),
					ordering=(),
					origin="llm",
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		domain_name="blocksworld_transition_rename",
	)

	assert ":task (clear_top ?target)" in domain_hddl

def test_panda_plan_parser_extracts_primitive_steps():
	planner = PANDAPlanner()
	plan_text = "0: (pick-up a)\n1: (stack a b)\n"
	steps = planner._parse_plan_steps(plan_text, _domain())

	assert [step.task_name for step in steps] == ["pick_up", "stack"]
	assert steps[0].args == ("a",)
	assert steps[1].args == ("a", "b")


def test_panda_plan_parser_extracts_converted_plan_steps():
	planner = PANDAPlanner()
	plan_text = (
		"==>\n"
		"15 pick-up a\n"
		"16 stack a b\n"
		"root 2\n"
		"2 place_on a b -> m_place_on_stack 8 3 16\n"
	)
	steps = planner._parse_plan_steps(plan_text, _domain())

	assert [step.task_name for step in steps] == ["pick_up", "stack"]
	assert steps[0].args == ("a",)
	assert steps[1].args == ("a", "b")



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


def test_panda_domain_export_preserves_lowercase_method_schema_variables():
	planner = PANDAPlanner()
	domain_hddl = planner._build_domain_hddl(
		domain=_domain(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("do_move", ("x", "y"), False, ()),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_do_move_constructive_unstack",
					task_name="do_move",
					parameters=("x", "y", "aux"),
					task_args=("x", "y"),
					context=(
						HTNLiteral("on", ("x", "aux"), True, None),
						HTNLiteral("clear", ("x",), True, None),
						HTNLiteral("handempty", (), True, None),
						HTNLiteral("clear", ("y",), True, None),
					),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="unstack",
							args=("x", "aux"),
							kind="primitive",
							action_name="unstack",
						),
						HTNMethodStep(
							step_id="s2",
							task_name="stack",
							args=("x", "y"),
							kind="primitive",
							action_name="stack",
						),
					),
					ordering=(("s1", "s2"),),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		domain_name="blocksworld_transition_1",
	)

	assert ":parameters (?x - block ?y - block ?aux - block)" in domain_hddl
	assert ":task (do_move ?x ?y)" in domain_hddl
	assert ":precondition (and (on ?x ?aux) (clear ?x) (handempty) (clear ?y))" in domain_hddl
	assert "(s1 (unstack ?x ?aux))" in domain_hddl
