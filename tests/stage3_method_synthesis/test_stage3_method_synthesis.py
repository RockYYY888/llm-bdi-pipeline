"""
Focused tests for Stage 3 HTN method synthesis.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_tests_dir = str(Path(__file__).parent.parent)
if _tests_dir not in sys.path:
	sys.path.insert(0, _tests_dir)
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
	_render_signature_with_mapping,
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

OFFICIAL_BLOCKSWORLD_DOMAIN_FILE = (
	Path(__file__).parent.parent.parent / "src" / "domains" / "blocksworld" / "domain.hddl"
)


def _domain():
	return HDDLParser.parse_domain(str(OFFICIAL_BLOCKSWORLD_DOMAIN_FILE))


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
	synthesizer = HTNMethodSynthesizer(**_live_stage3_kwargs())

	try:
		library, metadata = synthesizer.synthesize(
			domain=domain,
			grounding_map=spec.grounding_map,
			dfa_result=dfa_result,
		)
	except HTNSynthesisError as exc:
		metadata = exc.metadata
		assert metadata["llm_prompt"] is not None
		assert metadata["llm_response"]
		assert metadata["target_literals"] == ["on(a, b)"]
		assert metadata["failure_stage"] == "library_validation"
		return

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
		"nop",
		"pick_up",
		"put_down",
		"stack",
		"unstack",
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

def test_stage3_prompts_make_binding_and_naming_rules_explicit():
	synthesizer = HTNMethodSynthesizer()
	domain = _marsrover_domain()
	system_prompt = build_htn_system_prompt()
	user_prompt = build_htn_user_prompt(
		domain,
		["communicated_soil_data(waypoint2)", "communicated_image_data(objective1, high_res)"],
		'{"target_task_bindings": [], "compound_tasks": [], "methods": []}',
		query_text=(
			"Using rover rover0, waypoint waypoint2, mode high_res, and objective objective1, "
			"complete the tasks get_soil_data(waypoint2) and get_image_data(objective1, high_res)."
		),
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
			{"task_name": "get_image_data", "args": ["objective1", "high_res"]},
		),
		query_objects=("rover0", "waypoint2", "high_res", "objective1", "camera0"),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)
	lower_user_prompt = user_prompt.lower()

	assert "OUTPUT CONTRACT:" in system_prompt
	assert "When the query explicitly names declared domain tasks" in system_prompt
	assert "Fresh helper tasks may correspond only to dynamic predicates." in system_prompt
	assert "Static predicates are context constraints only" in system_prompt
	assert "Set primitive step literal to null unless that exact positive literal is a real action effect" in system_prompt
	assert "Every referenced compound subtask must also appear in compound_tasks and have at least one method in methods." in system_prompt
	assert "Auxiliary variables must appear in method.parameters and be constrained by method context before use in subtasks." in system_prompt
	assert "Do not use grounded object names as method variables" in system_prompt
	assert "Keep typed roles separate" in system_prompt
	assert "Do not generate transitive support-closure libraries" in system_prompt
	assert "Never use deprecated task prefixes achieve_, ensure_, goal_, or maintain_not_" in system_prompt
	assert "Apply the same rule to compound subtasks when their constructive branches share a dynamic prerequisite." in system_prompt
	assert "If a same-arity declared child is chosen as packaging for the headline effect" in system_prompt

	assert "QUERY:" in user_prompt
	assert "QUERY OBJECT NAMES (grounded instances from the current query; never place them inside methods):" in user_prompt
	assert "- rover0" in user_prompt
	assert "- camera0" in user_prompt
	assert "ORDERED QUERY TASK ANCHORS:" in user_prompt
	assert "#1: get_soil_data(waypoint2)" in user_prompt
	assert "#2: get_image_data(objective1, high_res)" in user_prompt
	assert "ORDERED TARGET/TASK SKELETON HINTS:" in user_prompt
	assert "target #1 communicated_soil_data(waypoint2) -> prefer declared task get_soil_data(waypoint2)" in user_prompt
	assert "STATIC PREDICATES (never changed by action effects; use as context only):" in user_prompt
	assert "- equipped_for_soil_analysis" in user_prompt
	assert "- visible_from" in user_prompt
	assert "DYNAMIC PREDICATES (changed by action effects):" in user_prompt
	assert "- communicated_soil_data" in user_prompt
	assert "- have_image" in user_prompt
	assert "PRODUCER ACTIONS BY DYNAMIC PREDICATE:" in user_prompt
	assert "- communicated_soil_data <- communicate_soil_data" in user_prompt
	assert "- have_image <- take_image" in user_prompt
	assert "PRODUCER EFFECT PATTERNS (use these to design constructive methods):" in user_prompt
	assert "- calibrated can be produced by calibrate via calibrated(?i, ?r);" in user_prompt
	assert "ARGUMENT-ALIGNED PRODUCER TEMPLATES:" in user_prompt
	assert "- navigate(TARGET1, OTHER1, TARGET2) -> at(TARGET1, TARGET2)" in user_prompt
	assert "SHARED PRODUCER DYNAMIC PREREQUISITES:" in user_prompt
	assert "DYNAMIC PRECONDITION SUPPORT HINTS:" in user_prompt
	assert "sample_soil needs dynamic empty(?s); likely reusable declared tasks: empty-store" in user_prompt
	assert "communicate_soil_data needs dynamic at(?r, ?x); likely reusable declared tasks: navigate_abs" in user_prompt
	assert "QUERY-TASK SAME-ARITY SUPPORT CANDIDATES:" in user_prompt
	assert "DECLARED TASK CONSTRUCTIVE TEMPLATES:" in user_prompt
	assert "get_soil_data(?waypoint) targets communicated_soil_data(?waypoint);" in user_prompt
	assert "DECLARED SUPPORT TASK APPLICABILITY ENVELOPES:" in user_prompt
	assert "navigate_abs(" in user_prompt
	assert "can serve as a declared support task for at(" in user_prompt
	assert "if a parent calls it, first provide shared prerequisites" in user_prompt
	assert "QUERY-TASK EXTRA ROLE FRAMES:" in user_prompt
	assert "to support have_image(?r, ?objective, ?mode), take_image(?r, WAYPOINT1, ?objective, CAMERA1, ?mode)" in user_prompt
	assert "introduces extra roles WAYPOINT1 - WAYPOINT, CAMERA1 - CAMERA" in user_prompt
	assert "sample_soil(ROVER1, STORE1, ARG1)" in user_prompt
	assert "take_image(ROVER1, WAYPOINT1, ARG1, CAMERA1, ARG2)" in user_prompt
	assert "get_image_data(?r, ARG1, ARG2)" not in user_prompt
	assert "only create a fresh helper task when no declared task can express the required dynamic state change" in lower_user_prompt
	assert "never create helper tasks for static predicates" in lower_user_prompt
	assert "if that constructive step requires dynamic preconditions" in lower_user_prompt
	assert "at least one constructive branch must stay applicable when that headline literal is currently false" in user_prompt
	assert "prefer same-arity declared tasks as reusable intermediate abstractions" in user_prompt
	assert "that child should own the final producer for the headline effect" in user_prompt
	assert "do not choose a producer chain whose unresolved preconditions still require that same P(args) to already hold" in user_prompt
	assert "add it to method.parameters, constrain it in method.context before using it in subtasks" in user_prompt
	assert "build its constructive branch around one of those aligned templates" in user_prompt
	assert "Preserve typed role separation across every method" in user_prompt
	assert "each dynamic precondition must already be guaranteed by method context or by earlier subtasks" in user_prompt
	assert "state it explicitly in method.context" in user_prompt
	assert "If a compound child task's constructive branches share a dynamic prerequisite" in user_prompt
	assert "never place grounded query object names inside methods; use schematic parameters instead" in lower_user_prompt
	assert "when a producer needs extra roles beyond the headline arguments" in lower_user_prompt
	assert "Do not use nop as filler inside constructive methods." in user_prompt
	assert "do not enumerate every support powerset" in user_prompt
	assert "do not bypass the query skeleton with a fresh helper-only library" in lower_user_prompt
	assert "primitive step literal is null unless that exact positive literal is a real action effect" in lower_user_prompt
	assert "every referenced compound subtask appears in compound_tasks and has methods" in lower_user_prompt
	assert "every primitive step's dynamic preconditions are supported by method context or earlier subtasks" in lower_user_prompt
	assert "every compound step's shared dynamic prerequisites are supported by method context or earlier subtasks" in lower_user_prompt
	assert "no primitive step relies on an unstated dynamic precondition" in lower_user_prompt
	assert "every auxiliary variable has an explicit typing witness" in lower_user_prompt
	assert "operate(ACTOR, LOCATION, TARGET, TOOL, MODE) must keep ACTOR and TOOL as different variables." in user_prompt


def test_stage3_prompt_makes_child_shared_support_requirements_explicit_for_query_tasks():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["on(a, b)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text="Using blocks a and b, complete the tasks do_put_on(a, b).",
		query_task_anchors=(
			{"task_name": "do_put_on", "args": ["a", "b"]},
		),
		query_objects=("a", "b"),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert "QUERY-TASK CHILD SUPPORT PREREQUISITES:" in user_prompt
	assert "QUERY-TASK SAME-ARITY PACKAGING MODES:" in user_prompt
	assert "before any helper or child call intended to establish holding(ARG1), first support its shared prerequisites clear(ARG1), handempty" in user_prompt
	assert "unstack(ARG1, BLOCK1)" in user_prompt
	assert "do_clear(?x) targets clear(?x); constructive templates: put_down(?x) [needs holding(?x)]; stack(?x, BLOCK1) [needs holding(?x)]; unstack(BLOCK1, ?x) [needs on(BLOCK1, ?x), handempty]" in user_prompt
	assert "no declared task directly headlines holding(ARG1). Before inventing a helper, prefer same-arity declared packaging tasks do_move(ARG1, ARG2) and let those declared tasks absorb the remaining dynamic support needed for on(ARG1, ARG2)." in user_prompt
	assert "if used as same-arity packaging for on(?x, ?y), it should itself reach the headline effect via stack(?x, ?y) [needs holding(?x), clear(?y)]" in user_prompt
	assert "Do not call this packaging child and then repeat the same final producer again in the parent." in user_prompt


def test_render_signature_with_mapping_does_not_cascade_replacements():
	assert _render_signature_with_mapping(
		"on(?x, ?y)",
		{"?y": "?x", "?x": "BLOCK1"},
	) == "on(BLOCK1, ?x)"


def test_stage3_prompt_uses_declared_source_names_for_hyphenated_task_anchors():
	domain = _marsrover_domain()
	user_prompt = build_htn_user_prompt(
		domain,
		["empty(s0)"],
		'{"target_task_bindings": [], "compound_tasks": [], "methods": []}',
		query_text="Using store s0 and rover rover0, complete the tasks empty-store(s0, rover0).",
		query_task_anchors=(
			{
				"task_name": "empty_store",
				"source_name": "empty-store",
				"args": ["s0", "rover0"],
			},
		),
		query_objects=("s0", "rover0"),
		action_analysis=HTNMethodSynthesizer()._analyse_domain_actions(domain),
	)

	assert "#1: empty-store(s0, rover0)" in user_prompt
	assert "prefer declared task empty-store(s0, rover0)" in user_prompt
	assert "empty_store(s0, rover0)" not in user_prompt


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
		query_text=(
			"Using blocks b1, b2, b3, b4, b5, b6, and b7, complete the tasks "
			"do_put_on(b3, b5), do_put_on(b6, b3), do_put_on(b1, b6), "
			"do_put_on(b2, b1), do_put_on(b4, b2), and do_put_on(b7, b4)."
		),
		query_task_anchors=tuple(
			{"task_name": "do_put_on", "args": list(args)}
			for args in (
				("b3", "b5"),
				("b6", "b3"),
				("b1", "b6"),
				("b2", "b1"),
				("b4", "b2"),
				("b7", "b4"),
			)
		),
		action_analysis=HTNMethodSynthesizer()._analyse_domain_actions(domain),
	)

	assert len(system_prompt) + len(user_prompt) < 18600


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


def test_action_analysis_includes_producer_effect_patterns():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(domain)

	assert "producer_patterns_by_predicate" in analysis
	clear_patterns = analysis["producer_patterns_by_predicate"]["clear"]
	assert any(
		pattern["effect_signature"] == "clear(?y)"
		and pattern["action_name"] == "unstack"
		and "on(?x, ?y)" in pattern["dynamic_precondition_signatures"]
		for pattern in clear_patterns
	)


def test_validate_library_requires_query_anchor_tasks_and_bindings():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	analysis = synthesizer._analyse_domain_actions(domain)
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("get_soil_data", ("WAYPOINT",), False, ("communicated_soil_data",)),
			HTNTask("soil_report_ready", ("WAYPOINT",), False, ("communicated_soil_data",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_get_soil_data_ready",
				task_name="get_soil_data",
				parameters=("WAYPOINT",),
				context=(HTNLiteral("communicated_soil_data", ("WAYPOINT",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_soil_report_ready_ready",
				task_name="soil_report_ready",
				parameters=("WAYPOINT",),
				context=(HTNLiteral("communicated_soil_data", ("WAYPOINT",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("communicated_soil_data", ("waypoint2",), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding("communicated_soil_data(waypoint2)", "soil_report_ready"),
		],
	)

	with pytest.raises(ValueError, match="must use the ordered query task anchor 'get_soil_data'"):
		synthesizer._validate_library(
			library,
			domain,
			query_task_anchors=(
				{"task_name": "get_soil_data", "args": ["waypoint2"]},
			),
			static_predicates=tuple(analysis["static_predicates"]),
		)


def test_validate_library_rejects_fresh_helper_for_static_predicate():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	analysis = synthesizer._analyse_domain_actions(domain)
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("remember_equipment", ("ROVER",), False, ("equipped_for_soil_analysis",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_remember_equipment_ready",
				task_name="remember_equipment",
				parameters=("ROVER",),
				context=(HTNLiteral("equipped_for_soil_analysis", ("ROVER",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="cannot headline static predicates"):
		synthesizer._validate_library(
			library,
			domain,
			static_predicates=tuple(analysis["static_predicates"]),
		)


def test_validate_library_rejects_primitive_step_literal_not_in_action_effects():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_clear", ("X",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_clear_bad_pickup",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("ontable", ("X",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("X",),
						kind="primitive",
						action_name="pick-up",
						literal=HTNLiteral("clear", ("X",), True, None),
					),
				),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="does not make that positive effect true"):
		synthesizer._validate_library(library, domain)


def test_validate_library_requires_compound_task_to_support_its_headline_predicate():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_on_table", ("X",), False, ("ontable",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_on_table_bad_noop",
				task_name="do_on_table",
				parameters=("X",),
				context=(HTNLiteral("clear", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="headlines 'ontable\\(X\\)'"):
		synthesizer._validate_library(library, domain)


def test_validate_library_rejects_constructive_branch_that_still_requires_headline_literal():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_on_table", ("X",), False, ("ontable",)),
			HTNTask("do_clear", ("X",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_clear_already",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("clear", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_on_table_bad_pickup",
				task_name="do_on_table",
				parameters=("X",),
				context=(
					HTNLiteral("ontable", ("X",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_clear",
						args=("X",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up",
						args=("X",),
						kind="primitive",
						action_name="pick-up",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="put_down",
						args=("X",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="headline literal is currently false"):
		synthesizer._validate_library(library, domain)


def test_validate_library_requires_dynamic_preconditions_to_be_supported_before_primitive_step():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_clear", ("X",), False, ("clear",)),
			HTNTask("do_on_table", ("X",), False, ("ontable",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_on_table_already",
				task_name="do_on_table",
				parameters=("X",),
				context=(HTNLiteral("ontable", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_bad_preconditions",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("clear", ("X",), False, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_on_table",
						args=("X",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up",
						args=("X",),
						kind="primitive",
						action_name="pick-up",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="without first supporting its dynamic preconditions"):
		synthesizer._validate_library(library, domain)


def test_validate_library_rejects_compound_steps_that_skip_shared_child_dynamic_support():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_put_on", ("X", "Y"), False, ("on",)),
			HTNTask("do_clear", ("X",), False, ("clear",)),
			HTNTask("hold_block", ("X",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_put_on_constructive",
				task_name="do_put_on",
				parameters=("X", "Y"),
				context=(),
				subtasks=(
					HTNMethodStep("s1", "hold_block", ("X",), "compound"),
					HTNMethodStep("s2", "do_clear", ("Y",), "compound"),
					HTNMethodStep(
						"s3",
						"stack",
						("X", "Y"),
						"primitive",
						action_name="stack",
					),
				),
				ordering=(("s1", "s3"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_already",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("clear", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_already",
				task_name="hold_block",
				parameters=("X",),
				context=(HTNLiteral("holding", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("X",),
				context=(
					HTNLiteral("clear", ("X",), True, None),
					HTNLiteral("ontable", ("X",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						"s1",
						"pick_up",
						("X",),
						"primitive",
						action_name="pick-up",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("X", "AUX"),
				context=(
					HTNLiteral("on", ("X", "AUX"), True, None),
					HTNLiteral("clear", ("X",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						"s1",
						"unstack",
						("X", "AUX"),
						"primitive",
						action_name="unstack",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="shared dynamic prerequisites"):
		synthesizer._validate_library(library, domain)


def test_validate_library_rejects_grounded_query_object_leakage():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("get_image_data", ("OBJECTIVE", "MODE"), False, ("communicated_image_data",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_get_image_data_constructive",
				task_name="get_image_data",
				parameters=("OBJECTIVE", "MODE"),
				context=(
					HTNLiteral("on_board", ("camera0", "ROVER"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="take_image",
						args=("ROVER", "WAYPOINT", "OBJECTIVE", "camera0", "MODE"),
						kind="primitive",
						action_name="take_image",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("communicated_image_data", ("objective1", "high_res"), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding("communicated_image_data(objective1, high_res)", "get_image_data"),
		],
	)

	with pytest.raises(ValueError, match="grounded query object 'camera0'"):
		synthesizer._validate_library(
			library,
			domain,
			query_objects=("camera0", "objective1", "high_res"),
		)


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
			HTNTask("cover_top", ("BLOCK",), False, ("clear",)),
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_cover_top_acquire",
				task_name="cover_top",
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


def test_method_validation_rejects_deprecated_task_prefixes():
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
				context=(
					HTNLiteral("on", ("B1", "B2"), True, None),
					HTNLiteral("clear", ("B1",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="unstack",
						args=("B1", "B2"),
						kind="primitive",
						action_name="unstack",
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


def test_target_binding_normalisation_accepts_object_form_literal():
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[],
		primitive_tasks=[],
		methods=[],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, None),
		],
		target_task_bindings=[
			HTNTargetTaskBinding(
				{
					"predicate": "on",
					"args": ["a", "b"],
					"is_positive": True,
				},
				"place_on",
			),
		],
	)

	normalised = synthesizer._normalise_target_binding_signatures(library)

	assert normalised.target_task_bindings[0].target_literal == "on(a, b)"


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
				context=(
					HTNLiteral("on", ("B", "SUPPORT"), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("B",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="unstack",
						args=("B", "SUPPORT"),
						kind="primitive",
						action_name="unstack",
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
			HTNTask("clear_top", ("BLOCK2",), False, ("clear",)),
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
				method_name="m_place_on_stack_primary",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("handempty", (), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="clear_top",
						args=("BLOCK2",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="stack",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="stack",
					),
				),
				ordering=(("s1", "s3"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_stack_backup",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("handempty", (), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="clear_top",
						args=("BLOCK2",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="stack",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="stack",
					),
				),
				ordering=(("s1", "s3"), ("s2", "s3")),
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
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("BLOCK2",),
				context=(
					HTNLiteral("clear", ("BLOCK2",), True, None),
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
		"m_place_on_stack_primary",
		"m_hold_block_noop",
		"m_clear_top_noop",
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


def test_single_empty_context_fallback_constructive_sibling_is_preserved():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("X", "Y"), False, ("on",)),
			HTNTask("hold_block", ("X",), False, ("holding",)),
			HTNTask("clear_top", ("Y",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_already",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("on", ("X", "Y"), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_missing_holding",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("clear", ("Y",), True, None),),
				subtasks=(
					HTNMethodStep("s1", "hold_block", ("X",), "compound"),
					HTNMethodStep("s2", "stack", ("X", "Y"), "primitive", "stack"),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_missing_clear",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("holding", ("X",), True, None),),
				subtasks=(
					HTNMethodStep("s1", "clear_top", ("Y",), "compound"),
					HTNMethodStep("s2", "stack", ("X", "Y"), "primitive", "stack"),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_missing_both",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(),
				subtasks=(
					HTNMethodStep("s1", "clear_top", ("Y",), "compound"),
					HTNMethodStep("s2", "hold_block", ("X",), "compound"),
					HTNMethodStep("s3", "stack", ("X", "Y"), "primitive", "stack"),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("X",),
				context=(HTNLiteral("holding", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("Y",),
				context=(HTNLiteral("clear", ("Y",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), True, "on_a_b")],
		target_task_bindings=[HTNTargetTaskBinding("on(a, b)", "place_on")],
	)

	pruned_library, pruned_count = synthesizer._prune_redundant_constructive_siblings(
		library,
		domain,
	)

	assert pruned_count == 0
	assert {method.method_name for method in pruned_library.methods} >= {
		"m_place_on_missing_holding",
		"m_place_on_missing_clear",
		"m_place_on_missing_both",
	}
	synthesizer._validate_library(pruned_library, domain)


def test_multiple_empty_context_fallback_constructive_siblings_are_rejected():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("X", "Y"), False, ("on",)),
			HTNTask("clear_top", ("Y",), False, ("clear",)),
			HTNTask("hold_block", ("X",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_missing_both",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(),
				subtasks=(
					HTNMethodStep("s1", "clear_top", ("Y",), "compound"),
					HTNMethodStep("s2", "hold_block", ("X",), "compound"),
					HTNMethodStep("s3", "stack", ("X", "Y"), "primitive", "stack"),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_fallback_2",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(),
				subtasks=(
					HTNMethodStep("s1", "hold_block", ("X",), "compound"),
					HTNMethodStep("s2", "clear_top", ("Y",), "compound"),
					HTNMethodStep("s3", "stack", ("X", "Y"), "primitive", "stack"),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("Y",),
				context=(HTNLiteral("clear", ("Y",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("X",),
				context=(HTNLiteral("holding", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="empty-context fallback branches"):
		synthesizer._validate_library(library, domain)


def test_declared_hyphenated_task_names_are_normalised_internally_but_keep_source_names():
	synthesizer = HTNMethodSynthesizer()
	domain = _marsrover_domain()

	anchors = synthesizer._normalise_query_task_anchors(
		(
			{"task_name": "empty-store", "args": ["s0", "rover0"]},
			{"task_name": "navigate_abs", "args": ["rover0", "waypoint1"]},
		),
		domain,
	)
	assert anchors == (
		{
			"task_name": "empty_store",
			"source_name": "empty-store",
			"args": ["s0", "rover0"],
		},
		{
			"task_name": "navigate_abs",
			"source_name": "navigate_abs",
			"args": ["rover0", "waypoint1"],
		},
	)

	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("empty-store", ("STORE", "ROVER"), False, ("empty",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_empty-store_ready",
				task_name="empty-store",
				parameters=("STORE", "ROVER"),
				context=(HTNLiteral("empty", ("STORE",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("empty", ("s0",), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("empty(s0)", "empty-store")],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	assert [task.name for task in normalised.compound_tasks] == ["empty_store"]
	assert [task.source_name for task in normalised.compound_tasks] == ["empty-store"]
	assert [method.task_name for method in normalised.methods] == ["empty_store"]
	assert [method.method_name for method in normalised.methods] == ["m_empty_store_ready"]
	assert [binding.task_name for binding in normalised.target_task_bindings] == ["empty_store"]


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


def test_method_type_validation_prefers_declared_task_signature_over_source_predicate_order():
	synthesizer = HTNMethodSynthesizer()
	task_lookup = {
		"calibrate_abs": HTNTask(
			"calibrate_abs",
			("ROVER", "CAMERA"),
			False,
			("calibrated",),
		),
	}
	method = HTNMethod(
		method_name="m_calibrate_abs_constructive",
		task_name="calibrate_abs",
		parameters=("ROVER", "CAMERA"),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="calibrate",
				args=("ROVER", "CAMERA"),
				kind="primitive",
				action_name="calibrate",
			),
		),
		origin="llm",
	)

	synthesizer._validate_method_variable_types(
		method,
		task_lookup,
		action_types={"calibrate": ("ROVER", "CAMERA")},
		task_types={"calibrate_abs": ("ROVER", "CAMERA")},
		predicate_types={"calibrated": ("CAMERA", "ROVER")},
	)


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
	synthesizer = HTNMethodSynthesizer()
	prompt = {"system": "system", "user": "user"}
	metadata = {"llm_attempts": 0}

	def fake_call_llm(
		prompt_payload: dict,
		*,
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


def test_negative_target_binding_rejects_helper_call_without_shared_dynamic_support():
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
						task_name="unstack",
						args=("BLOCK", "SUPPORT"),
						kind="primitive",
						action_name="unstack",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), False, "on_a_b")],
		target_task_bindings=[HTNTargetTaskBinding("!on(a, b)", "remove_on")],
	)

	with pytest.raises(ValueError, match="shared dynamic prerequisites"):
		synthesizer._validate_library(library, domain)
