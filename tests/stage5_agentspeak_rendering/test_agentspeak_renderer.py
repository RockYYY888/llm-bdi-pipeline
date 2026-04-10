"""
Focused tests for Stage 5 AgentSpeak rendering.
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
from stage4_panda_planning.panda_schema import PANDAPlanResult
from stage5_agentspeak_rendering.agentspeak_renderer import AgentSpeakRenderer
from utils.hddl_parser import HDDLParser

OFFICIAL_BLOCKSWORLD_DOMAIN_FILE = (
	Path(__file__).parent.parent.parent / "src" / "domains" / "blocksworld" / "domain.hddl"
)
OFFICIAL_MARSROVER_DOMAIN_FILE = (
	Path(__file__).parent.parent.parent / "src" / "domains" / "marsrover" / "domain.hddl"
)
OFFICIAL_SATELLITE_DOMAIN_FILE = (
	Path(__file__).parent.parent.parent / "src" / "domains" / "satellite" / "domain.hddl"
)
OFFICIAL_TRANSPORT_DOMAIN_FILE = (
	Path(__file__).parent.parent.parent / "src" / "domains" / "transport" / "domain.hddl"
)


def _domain():
	return HDDLParser.parse_domain(str(OFFICIAL_BLOCKSWORLD_DOMAIN_FILE))


def _marsrover_domain():
	return HDDLParser.parse_domain(str(OFFICIAL_MARSROVER_DOMAIN_FILE))


def _satellite_domain():
	return HDDLParser.parse_domain(str(OFFICIAL_SATELLITE_DOMAIN_FILE))


def _transport_domain():
	return HDDLParser.parse_domain(str(OFFICIAL_TRANSPORT_DOMAIN_FILE))


def _parse_domain_text(tmp_path: Path, filename: str, content: str):
	domain_file = tmp_path / filename
	domain_file.write_text(content.strip())
	return HDDLParser.parse_domain(str(domain_file))


def _method_library() -> HTNMethodLibrary:
	return HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="place_on",
				parameters=("B1", "B2"),
				is_primitive=False,
				source_predicates=("on",),
			),
			HTNTask(
				name="hold_block",
				parameters=("B1",),
				is_primitive=False,
				source_predicates=("holding",),
			),
		],
		primitive_tasks=[
			HTNTask(
				name="pick_up",
				parameters=("B1",),
				is_primitive=True,
			),
			HTNTask(
				name="stack",
				parameters=("B1", "B2"),
				is_primitive=True,
			),
		],
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(HTNLiteral("on", ("B1", "B2")),),
			),
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("B1", "B2"),
				subtasks=(
					HTNMethodStep(
						step_id="s2",
						task_name="stack",
						args=("B1", "B2"),
						kind="primitive",
						action_name="stack",
					),
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("B1",),
						kind="compound",
					),
				),
				ordering=(("s1", "s2"),),
			),
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("B1",),
				context=(
					HTNLiteral("ontable", ("B1",)),
					HTNLiteral("clear", ("B1",)),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("B1",),
						kind="primitive",
						action_name="pick-up",
					),
				),
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), True, "on_a_b")],
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="on(a, b)", task_name="place_on"),
		],
	)


def test_renderer_emits_method_library_and_state_aware_wrappers():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=_method_library(),
		plan_records=[
			{
				"transition_name": "dfa_step_q1_q2_on_a_b",
				"source_state": "q1",
				"target_state": "q2",
				"initial_state": "q1",
				"accepting_states": ["q2"],
				"label": "on(a, b)",
				"plan": PANDAPlanResult(
					task_name="place_on",
					task_args=("a", "b"),
					target_literal=HTNLiteral("on", ("a", "b"), True, "on_a_b"),
				),
			},
		],
	)

	assert "/* HTN Method Plans */" in code
	assert "dfa_state(q1)." in code
	assert "accepting_state(q2)." in code
	assert 'dfa_edge_label(dfa_step_q1_q2_on_a_b, "on(a, b)").' in code
	assert "+!place_on(BLOCK1, BLOCK2) : on(BLOCK1, BLOCK2) <-" in code
	assert any(
		line.startswith("+!place_on(BLOCK1, BLOCK2) : ")
		and line != "+!place_on(BLOCK1, BLOCK2) : on(BLOCK1, BLOCK2) <-"
		for line in code.splitlines()
	)
	assert "\t!hold_block(BLOCK1);" in code
	assert "\t!stack(BLOCK1, BLOCK2)." in code
	assert "+!hold_block(BLOCK) : ontable(BLOCK) & clear(BLOCK) <-" in code
	assert "\t!pick_up(BLOCK)." in code
	assert "+!dfa_step_q1_q2_on_a_b : dfa_state(q1) <-" in code
	assert "\t!place_on(a, b);" in code
	assert "\t-dfa_state(q1);" in code
	assert "\t+dfa_state(q2)." in code
	assert "+!run_dfa : dfa_state(q1) <-" in code
	assert "\t!dfa_step_q1_q2_on_a_b;" in code
	assert "\t!run_dfa." in code
	assert "+!run_dfa : dfa_state(q2) & accepting_state(q2) <-" in code
	assert "+!stack(BLOCK1, BLOCK2) :" in code


def test_renderer_accepts_zero_step_wrappers_when_stage4_returns_no_witness_steps():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=_method_library(),
		plan_records=[
			{
				"transition_name": "dfa_step_q2_q2_not_on_a_b",
				"source_state": "q2",
				"target_state": "q2",
				"initial_state": "q2",
				"accepting_states": ["q2"],
				"label": "!on(a, b)",
				"plan": PANDAPlanResult(
					task_name="keep_apart",
					task_args=("a", "b"),
					target_literal=HTNLiteral("on", ("a", "b"), False, "on_a_b"),
					steps=[],
				),
			},
		],
	)

	assert "/* HTN Method Plans */" in code
	assert "+!dfa_step_q2_q2_not_on_a_b : dfa_state(q2) <-" in code
	assert "\t!keep_apart(a, b)." in code
	assert "\t-dfa_state(q2);" not in code
	assert "\t+dfa_state(q2)." not in code
	assert "+!run_dfa : dfa_state(q2) & accepting_state(q2) <-" in code
	assert "+!run_dfa : dfa_state(q2) <-" not in code


def test_renderer_hoists_binding_preconditions_into_method_context():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B",), False, ("holding",)),
			HTNTask("clear_top", ("B",), False, ("clear",)),
		],
		primitive_tasks=[],
		methods=[
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
						action_name="unstack",
						preconditions=(
							HTNLiteral("on", ("B", "SUPPORT"), True, None),
						),
					),
				),
				ordering=(("s1", "s2"),),
			),
		],
	)

	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=method_library,
		plan_records=[],
	)

	assert "+!hold_block(BLOCK1) : on(BLOCK1, BLOCK2) <-" in code
	assert "\t!pick_up(BLOCK1, BLOCK2)." in code


def test_renderer_sanitizes_hyphenated_predicate_functors():
	renderer = AgentSpeakRenderer()

	rendered = renderer._render_literal(
		HTNLiteral("capacity-predecessor", ("CAPACITY1", "CAPACITY2"), True, None),
		{},
	)

	assert rendered == "capacity_predecessor(CAPACITY1, CAPACITY2)"


def test_renderer_sanitizes_hyphenated_type_names_when_emitting_variables():
	renderer = AgentSpeakRenderer()

	names = renderer._canonical_param_names(
		("CAPACITY-NUMBER", "CAPACITY-NUMBER"),
		{"CAPACITY-NUMBER": 2},
	)

	assert names == ("CAPACITY_NUMBER1", "CAPACITY_NUMBER2")


def test_renderer_emits_object_type_guards_for_typed_compound_bindings():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_transport_domain(),
		objects=("truck-0", "package-0", "city-loc-0", "city-loc-1"),
		typed_objects=(
			("truck-0", "vehicle"),
			("package-0", "package"),
			("city-loc-0", "location"),
			("city-loc-1", "location"),
		),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("deliver", ("PACKAGE", "LOCATION"), False, source_name="deliver"),
				HTNTask("get_to", ("VEHICLE", "LOCATION"), False, source_name="get-to"),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m-deliver",
					task_name="deliver",
					parameters=("PACKAGE", "DESTINATION", "VEHICLE", "SOURCE"),
					task_args=("PACKAGE", "DESTINATION"),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="get_to",
							args=("VEHICLE", "SOURCE"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s2",
							task_name="get_to",
							args=("VEHICLE", "DESTINATION"),
							kind="compound",
						),
					),
					ordering=(("s1", "s2"),),
				),
				HTNMethod(
					method_name="m-drive-to",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="drive",
							args=("VEHICLE", "SOURCE", "DESTINATION"),
							kind="primitive",
							action_name="drive",
						),
					),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	assert 'object_type("truck-0", vehicle).' in code
	assert 'object_type("package-0", package).' in code
	assert "object_type(VEHICLE, vehicle)" in code
	assert "object_type(LOCATION2, location)" in code
	assert "\t!get_to(VEHICLE, LOCATION2);" in code
	assert "\t!get_to(VEHICLE, LOCATION1)." in code


def test_renderer_emits_parent_type_beliefs_for_subtyped_objects():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_satellite_domain(),
		objects=("GroundStation2", "Phenomenon4"),
		typed_objects=(
			("GroundStation2", "calib_direction"),
			("Phenomenon4", "image_direction"),
		),
		method_library=HTNMethodLibrary(
			compound_tasks=[],
			primitive_tasks=[],
			methods=[],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	assert 'object_type("GroundStation2", calib_direction).' in code
	assert 'object_type("GroundStation2", direction).' in code
	assert 'object_type("Phenomenon4", image_direction).' in code
	assert 'object_type("Phenomenon4", direction).' in code


def test_renderer_hoists_safe_later_compound_binding_literals():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_transport_domain(),
		objects=("truck-0", "package-0", "city-loc-0", "city-loc-1", "capacity-0", "capacity-1"),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("deliver", ("PACKAGE", "DESTINATION"), False, source_name="deliver"),
				HTNTask("get_to", ("VEHICLE", "LOCATION"), False, source_name="get-to"),
				HTNTask("load", ("VEHICLE", "LOCATION", "PACKAGE"), False, source_name="load"),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m-deliver",
					task_name="deliver",
					parameters=("PACKAGE", "DESTINATION", "SOURCE", "VEHICLE"),
					task_args=("PACKAGE", "DESTINATION"),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="get_to",
							args=("VEHICLE", "SOURCE"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s2",
							task_name="load",
							args=("VEHICLE", "SOURCE", "PACKAGE"),
							kind="compound",
						),
					),
					ordering=(("s1", "s2"),),
				),
				HTNMethod(
					method_name="m-drive-to",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="drive",
							args=("VEHICLE", "SOURCE", "DESTINATION"),
							kind="primitive",
							action_name="drive",
						),
					),
				),
				HTNMethod(
					method_name="m-load",
					task_name="load",
					parameters=("VEHICLE", "LOCATION", "PACKAGE", "CAPACITY1", "CAPACITY2"),
					task_args=("VEHICLE", "LOCATION", "PACKAGE"),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="pick_up",
							args=("VEHICLE", "LOCATION", "PACKAGE", "CAPACITY1", "CAPACITY2"),
							kind="primitive",
							action_name="pick-up",
						),
					),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	deliver_lines = [
		line for line in code.splitlines() if line.startswith("+!deliver(")
	]
	assert any(
		"capacity(VEHICLE, CAPACITY_NUMBER2)" in line
		or "at(VEHICLE, LOCATION)" in line
		for line in deliver_lines
	)


def test_renderer_keeps_recursive_first_child_specialisation_bindings():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_transport_domain(),
		objects=("truck-0", "city-loc-0", "city-loc-1", "city-loc-2"),
		typed_objects=(
			("truck-0", "vehicle"),
			("city-loc-0", "location"),
			("city-loc-1", "location"),
			("city-loc-2", "location"),
		),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("get_to", ("VEHICLE", "DESTINATION"), False, source_name="get-to"),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_drive_to",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					context=(
						HTNLiteral("at", ("VEHICLE", "SOURCE"), True, None),
						HTNLiteral("road", ("SOURCE", "DESTINATION"), True, None),
					),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="drive",
							args=("VEHICLE", "SOURCE", "DESTINATION"),
							kind="primitive",
							action_name="drive",
						),
					),
				),
				HTNMethod(
					method_name="m_drive_to_via",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "MID", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					context=(
						HTNLiteral("road", ("MID", "DESTINATION"), True, None),
					),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="get_to",
							args=("VEHICLE", "MID"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s2",
							task_name="drive",
							args=("VEHICLE", "MID", "DESTINATION"),
							kind="primitive",
							action_name="drive",
						),
					),
					ordering=(("s1", "s2"),),
				),
				HTNMethod(
					method_name="m_i_am_there",
					task_name="get_to",
					parameters=("VEHICLE", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					context=(HTNLiteral("at", ("VEHICLE", "DESTINATION"), True, None),),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="noop",
							args=("VEHICLE", "DESTINATION"),
							kind="primitive",
							action_name="noop",
						),
					),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	get_to_lines = [
		line for line in code.splitlines() if line.startswith("+!get_to(")
	]
	assert any(
		"road(" in line
		and "at(" in line
		for line in get_to_lines
	)
	lines = code.splitlines()
	recursive_header = next(
		lines[index - 1]
		for index, line in enumerate(lines)
		if 'trace_method(m_drive_to_via' in line
	)
	assert "road(" in recursive_header
	assert "at(" not in recursive_header
	assert "\\==" in recursive_header


def test_renderer_keeps_recursive_first_child_bindings_when_child_methods_infer_context():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_transport_domain(),
		objects=("truck-0", "city-loc-0", "city-loc-1", "city-loc-2"),
		typed_objects=(
			("truck-0", "vehicle"),
			("city-loc-0", "location"),
			("city-loc-1", "location"),
			("city-loc-2", "location"),
		),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("get_to", ("VEHICLE", "DESTINATION"), False, source_name="get-to"),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_drive_to",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="drive",
							args=("VEHICLE", "SOURCE", "DESTINATION"),
							kind="primitive",
							action_name="drive",
						),
					),
				),
				HTNMethod(
					method_name="m_drive_to_via",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "MID", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="get_to",
							args=("VEHICLE", "MID"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s2",
							task_name="drive",
							args=("VEHICLE", "MID", "DESTINATION"),
							kind="primitive",
							action_name="drive",
						),
					),
					ordering=(("s1", "s2"),),
				),
				HTNMethod(
					method_name="m_i_am_there",
					task_name="get_to",
					parameters=("VEHICLE", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="noop",
							args=("VEHICLE", "DESTINATION"),
							kind="primitive",
							action_name="noop",
						),
					),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	get_to_lines = [
		line for line in code.splitlines() if line.startswith("+!get_to(")
	]
	assert any(
		"road(" in line and "at(" in line
		for line in get_to_lines
	)
	lines = code.splitlines()
	recursive_header = next(
		lines[index - 1]
		for index, line in enumerate(lines)
		if 'trace_method(m_drive_to_via' in line
	)
	assert "road(" in recursive_header
	assert "at(" not in recursive_header
	assert "\\==" in recursive_header


def test_renderer_orders_effect_entailing_and_direct_methods_before_recursive_methods():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_transport_domain(),
		objects=("truck-0", "city-loc-0", "city-loc-1", "city-loc-2"),
		typed_objects=(
			("truck-0", "vehicle"),
			("city-loc-0", "location"),
			("city-loc-1", "location"),
			("city-loc-2", "location"),
		),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("get_to", ("VEHICLE", "DESTINATION"), False, source_name="get-to"),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_drive_to",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="drive",
							args=("VEHICLE", "SOURCE", "DESTINATION"),
							kind="primitive",
							action_name="drive",
						),
					),
				),
				HTNMethod(
					method_name="m_drive_to_via",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "MID", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="get_to",
							args=("VEHICLE", "MID"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s2",
							task_name="drive",
							args=("VEHICLE", "MID", "DESTINATION"),
							kind="primitive",
							action_name="drive",
						),
					),
					ordering=(("s1", "s2"),),
				),
				HTNMethod(
					method_name="m_i_am_there",
					task_name="get_to",
					parameters=("VEHICLE", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="noop",
							args=("VEHICLE", "DESTINATION"),
							kind="primitive",
							action_name="noop",
						),
					),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	i_am_there_index = code.index("trace_method(m_i_am_there")
	drive_to_index = code.index("trace_method(m_drive_to")
	drive_to_via_index = code.index("trace_method(m_drive_to_via")

	assert i_am_there_index < drive_to_index < drive_to_via_index


def test_renderer_does_not_specialise_parent_plans_through_recursive_child_tasks():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_transport_domain(),
		objects=("truck-0", "city-loc-0", "city-loc-1", "city-loc-2", "package-0"),
		typed_objects=(
			("truck-0", "vehicle"),
			("city-loc-0", "location"),
			("city-loc-1", "location"),
			("city-loc-2", "location"),
			("package-0", "package"),
		),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("deliver", ("PACKAGE", "DESTINATION"), False, source_name="deliver"),
				HTNTask("get_to", ("VEHICLE", "LOCATION"), False, source_name="get-to"),
				HTNTask("load", ("VEHICLE", "LOCATION", "PACKAGE"), False, source_name="load"),
				HTNTask("unload", ("VEHICLE", "LOCATION", "PACKAGE"), False, source_name="unload"),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_deliver",
					task_name="deliver",
					parameters=("PACKAGE", "PICKUP", "DESTINATION", "VEHICLE"),
					task_args=("PACKAGE", "DESTINATION"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="get_to",
							args=("VEHICLE", "PICKUP"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s2",
							task_name="load",
							args=("VEHICLE", "PICKUP", "PACKAGE"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s3",
							task_name="get_to",
							args=("VEHICLE", "DESTINATION"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s4",
							task_name="unload",
							args=("VEHICLE", "DESTINATION", "PACKAGE"),
							kind="compound",
						),
					),
					ordering=(("s1", "s2"), ("s2", "s3"), ("s3", "s4")),
				),
				HTNMethod(
					method_name="m_load",
					task_name="load",
					parameters=("VEHICLE", "LOCATION", "PACKAGE", "CAPACITY1", "CAPACITY2"),
					task_args=("VEHICLE", "LOCATION", "PACKAGE"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="pick_up",
							args=("VEHICLE", "LOCATION", "PACKAGE", "CAPACITY1", "CAPACITY2"),
							kind="primitive",
							action_name="pick_up",
						),
					),
				),
				HTNMethod(
					method_name="m_unload",
					task_name="unload",
					parameters=("VEHICLE", "LOCATION", "PACKAGE", "CAPACITY1", "CAPACITY2"),
					task_args=("VEHICLE", "LOCATION", "PACKAGE"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="drop",
							args=("VEHICLE", "LOCATION", "PACKAGE", "CAPACITY1", "CAPACITY2"),
							kind="primitive",
							action_name="drop",
						),
					),
				),
				HTNMethod(
					method_name="m_drive_to",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "LOCATION"),
					task_args=("VEHICLE", "LOCATION"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="drive",
							args=("VEHICLE", "SOURCE", "LOCATION"),
							kind="primitive",
							action_name="drive",
						),
					),
				),
				HTNMethod(
					method_name="m_drive_to_via",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "MID", "LOCATION"),
					task_args=("VEHICLE", "LOCATION"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="get_to",
							args=("VEHICLE", "MID"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s2",
							task_name="drive",
							args=("VEHICLE", "MID", "LOCATION"),
							kind="primitive",
							action_name="drive",
						),
					),
					ordering=(("s1", "s2"),),
				),
				HTNMethod(
					method_name="m_i_am_there",
					task_name="get_to",
					parameters=("VEHICLE", "LOCATION"),
					task_args=("VEHICLE", "LOCATION"),
					context=(),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="noop",
							args=("VEHICLE", "LOCATION"),
							kind="primitive",
							action_name="noop",
						),
					),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	deliver_lines = [
		line for line in code.splitlines() if line.startswith("+!deliver(")
	]
	assert len(deliver_lines) == 1
	assert "road(" not in deliver_lines[0]


def test_renderer_prefers_precise_concrete_compound_effects_over_abstract_witnesses():
	renderer = AgentSpeakRenderer()
	from test_pipeline import _official_domain_method_library

	domain = _transport_domain()
	method_library = _official_domain_method_library(
		"transport",
		target_literal_signatures=[],
		query_task_anchors=[],
	)
	task_lookup = {
		task.name: task
		for task in method_library.compound_tasks + method_library.primitive_tasks
	}
	methods_by_task = {}
	for method in method_library.methods:
		methods_by_task.setdefault(method.task_name, []).append(method)
	render_specs = renderer._build_task_render_specs(domain, method_library)
	deliver_method = next(
		method
		for method in method_library.methods
		if method.method_name == "m-deliver"
	)
	ordered_steps = renderer._ordered_method_steps(deliver_method)
	render_spec = render_specs["deliver"]
	variable_map = renderer._method_variable_map(
		deliver_method,
		ordered_steps,
		render_spec,
	)
	context_literals = renderer._method_context_literals(
		deliver_method,
		ordered_steps,
		task_lookup,
		methods_by_task,
		render_spec,
		variable_map,
		{},
		{},
	)

	assert any(
		literal.predicate == "at" and literal.args == ("?p", "?l1")
		for literal in context_literals
	)
	assert not any(
		literal.predicate == "at" and literal.args == ("LOCATABLE", "?l1")
		for literal in context_literals
	)


def test_renderer_prefers_explicit_method_task_args_for_task_parameter_typing(tmp_path):
	renderer = AgentSpeakRenderer()
	domain = _parse_domain_text(
		tmp_path,
		"delivery_domain.hddl",
		"""
(define (domain delivery_world)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types package vehicle location)
  (:predicates
    (pkg_at ?p - package ?l - location)
    (truck_at ?v - vehicle ?l - location)
  )
  (:task deliver
    :parameters (?p - package ?l - location)
  )
  (:task load
    :parameters (?v - vehicle ?p - package)
  )
  (:action drive
    :parameters (?v - vehicle ?from - location ?to - location)
    :precondition (and (truck_at ?v ?from))
    :effect (and (truck_at ?v ?to) (not (truck_at ?v ?from)))
  )
  (:action pick_up
    :parameters (?v - vehicle ?l - location ?p - package)
    :precondition (and (truck_at ?v ?l) (pkg_at ?p ?l))
    :effect (and)
  )
)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("deliver", ("PKG", "DEST"), False, ()),
			HTNTask("load", ("TRUCK", "PKG"), False, ()),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_deliver",
				task_name="deliver",
				parameters=("DEPOT", "TRUCK", "PKG", "DEST"),
				task_args=("PKG", "DEST"),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="load",
						args=("TRUCK", "PKG"),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="drive",
						args=("TRUCK", "DEPOT", "DEST"),
						kind="primitive",
						action_name="drive",
					),
				),
				ordering=(("s1", "s2"),),
			),
			HTNMethod(
				method_name="m_load",
				task_name="load",
				parameters=("DEPOT", "TRUCK", "PKG"),
				task_args=("TRUCK", "PKG"),
				context=(
					HTNLiteral("truck_at", ("TRUCK", "DEPOT"), True, None),
					HTNLiteral("pkg_at", ("PKG", "DEPOT"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("TRUCK", "DEPOT", "PKG"),
						kind="primitive",
						action_name="pick_up",
					),
				),
				ordering=(),
			),
		],
	)

	render_specs = renderer._build_task_render_specs(domain, method_library)
	deliver_method = method_library.methods[0]
	variable_map = renderer._method_variable_map(
		deliver_method,
		renderer._ordered_method_steps(deliver_method),
		render_specs["deliver"],
	)

	assert render_specs["deliver"]["task_param_types"] == ("PACKAGE", "LOCATION")
	assert variable_map["PKG"] == "PACKAGE"
	assert variable_map["DEST"].startswith("LOCATION")
	assert variable_map["DEPOT"].startswith("LOCATION")
	assert variable_map["DEST"] != variable_map["DEPOT"]


def test_renderer_sanitises_hyphenated_task_functors_for_jason():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("empty-store", ("STORE", "ROVER"), False, ("empty",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_empty_store_noop",
				task_name="empty-store",
				parameters=("STORE", "ROVER"),
				context=(HTNLiteral("empty", ("STORE",), True, None),),
			),
		],
		target_literals=[HTNLiteral("empty", ("rover0store",), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("empty(rover0store)", "empty-store")],
	)

	code = renderer.generate(
		domain=_marsrover_domain(),
		objects=("rover0store", "rover0"),
		method_library=method_library,
		plan_records=[
			{
				"transition_name": "dfa_step_q0_q1_empty_rover0store",
				"source_state": "q0",
				"target_state": "q1",
				"initial_state": "q0",
				"accepting_states": ["q1"],
				"label": "empty(rover0store)",
				"plan": PANDAPlanResult(
					task_name="empty-store",
					task_args=("rover0store", "rover0"),
					target_literal=HTNLiteral("empty", ("rover0store",), True, None),
				),
			},
		],
	)

	assert "+!empty_store(STORE, ROVER) : empty(STORE) <-" in code
	assert "\t!empty_store(rover0store, rover0);" in code
	assert "empty-store" not in code


def test_renderer_uses_task_args_not_leading_method_parameters_for_trigger_bindings():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("navigate_abs", ("ROVER", "TO"), False, ("at",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_navigate_abs_direct",
				task_name="navigate_abs",
				parameters=("ROVER", "FROM", "TO"),
				task_args=("ROVER", "TO"),
				context=(HTNLiteral("at", ("ROVER", "FROM"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="navigate",
						args=("ROVER", "FROM", "TO"),
						kind="primitive",
						action_name="navigate",
					),
				),
			),
		],
	)

	code = renderer.generate(
		domain=_marsrover_domain(),
		objects=("rover0", "waypoint0", "waypoint3"),
		method_library=method_library,
		plan_records=[],
	)

	assert "+!navigate_abs(ROVER, WAYPOINT1) : at(ROVER, WAYPOINT2)" in code
	assert "\t!navigate(ROVER, WAYPOINT2, WAYPOINT1)." in code


def test_renderer_hoists_action_preconditions_for_hddl_style_question_mark_variables():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("navigate_abs", ("ROVER", "TO"), False, ("at",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_navigate_abs_direct_hddl",
				task_name="navigate_abs",
				parameters=("?rover", "?from", "?to"),
				task_args=("?rover", "?to"),
				context=(HTNLiteral("at", ("?rover", "?from"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="navigate",
						args=("?rover", "?from", "?to"),
						kind="primitive",
						action_name="navigate",
					),
				),
			),
		],
	)

	code = renderer.generate(
		domain=_marsrover_domain(),
		objects=("rover0", "waypoint0", "waypoint3"),
		method_library=method_library,
		plan_records=[],
	)

	assert "can_traverse(ROVER, WAYPOINT2, WAYPOINT1)" in code
	assert "available(ROVER)" in code
	assert "visible(WAYPOINT2, WAYPOINT1)" in code


def test_renderer_orders_positive_context_binders_before_negative_guards():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("navigate_abs", ("ROVER", "TO"), False, ("at",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_navigate_abs_recursive",
				task_name="navigate_abs",
				parameters=("?rover", "?from", "?to", "?mid"),
				task_args=("?rover", "?to"),
				context=(
					HTNLiteral("visited", ("?mid",), False, None),
					HTNLiteral("can_traverse", ("?rover", "?from", "?mid"), True, None),
					HTNLiteral("at", ("?rover", "?from"), True, None),
				),
				subtasks=(),
			),
		],
	)

	code = renderer.generate(
		domain=_marsrover_domain(),
		objects=("rover0", "waypoint0", "waypoint1"),
		method_library=method_library,
		plan_records=[],
	)

	context_line = next(
		line for line in code.splitlines() if line.startswith("+!navigate_abs(")
	)
	assert context_line.index("can_traverse(ROVER, WAYPOINT2, WAYPOINT3)") < context_line.index(
		"not visited(WAYPOINT3)",
	)


def test_renderer_does_not_hoist_precondition_already_supported_by_packaging_child():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("empty-store", ("STORE", "ROVER"), False, ("empty",)),
			HTNTask("get_rock_data", ("WAYPOINT",), False, ("communicated_rock_data",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_empty_store_noop",
				task_name="empty-store",
				parameters=("STORE", "ROVER"),
				context=(HTNLiteral("empty", ("STORE",), True, None),),
			),
			HTNMethod(
				method_name="m_get_rock_data",
				task_name="get_rock_data",
				parameters=("WAYPOINT", "ROVER", "STORE"),
				context=(
					HTNLiteral("at_rock_sample", ("WAYPOINT",), True, None),
					HTNLiteral("equipped_for_rock_analysis", ("ROVER",), True, None),
					HTNLiteral("store_of", ("STORE", "ROVER"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="empty-store",
						args=("STORE", "ROVER"),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="sample_rock",
						args=("ROVER", "STORE", "WAYPOINT"),
						kind="primitive",
						action_name="sample_rock",
					),
				),
				ordering=(("s1", "s2"),),
			),
		],
	)

	code = renderer.generate(
		domain=_marsrover_domain(),
		objects=("rover0", "rover0store", "waypoint3"),
		method_library=method_library,
		plan_records=[],
	)

	context_line = next(
		line for line in code.splitlines() if line.startswith("+!get_rock_data(")
	)
	assert "empty(" not in context_line


def test_renderer_prefers_declared_task_parameter_order_over_source_predicate_order():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("calibrate_abs", ("ROVER", "CAMERA"), False, ("calibrated",)),
		],
		primitive_tasks=[],
		methods=[],
	)

	render_specs = renderer._build_task_render_specs(_marsrover_domain(), method_library)

	assert render_specs["calibrate_abs"]["task_param_types"] == ("ROVER", "CAMERA")


def test_renderer_projects_compound_effects_by_declared_task_types():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("calibrate_abs", ("ROVER", "CAMERA"), False, ("calibrated",)),
			HTNTask("get_image_data", ("OBJECTIVE", "MODE"), False, ("communicated_image_data",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_get_image_data",
				task_name="get_image_data",
				parameters=("OBJECTIVE", "MODE", "ROVER", "CAMERA", "WAYPOINT"),
				task_args=("OBJECTIVE", "MODE"),
				context=(
					HTNLiteral("equipped_for_imaging", ("ROVER",), True, None),
					HTNLiteral("on_board", ("CAMERA", "ROVER"), True, None),
					HTNLiteral("supports", ("CAMERA", "MODE"), True, None),
					HTNLiteral("visible_from", ("OBJECTIVE", "WAYPOINT"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="calibrate_abs",
						args=("ROVER", "CAMERA"),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="take_image",
						args=("ROVER", "WAYPOINT", "OBJECTIVE", "CAMERA", "MODE"),
						kind="primitive",
						action_name="take_image",
					),
				),
				ordering=(("s1", "s2"),),
			),
		],
	)

	code = renderer.generate(
		domain=_marsrover_domain(),
		objects=("rover0", "camera0", "objective0", "colour", "waypoint0"),
		method_library=method_library,
		plan_records=[],
	)

	context_line = next(
		line for line in code.splitlines() if line.startswith("+!get_image_data(")
	)
	assert "calibrated(" not in context_line


def test_renderer_specialises_parent_plan_with_first_compound_child_applicability():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("navigate_abs", ("ROVER", "TO"), False, ("at",)),
			HTNTask("get_soil_data", ("WAYPOINT",), False, ("communicated_soil_data",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_navigate_abs_direct",
				task_name="navigate_abs",
				parameters=("ROVER", "FROM", "TO"),
				task_args=("ROVER", "TO"),
				context=(HTNLiteral("at", ("ROVER", "FROM"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="navigate",
						args=("ROVER", "FROM", "TO"),
						kind="primitive",
						action_name="navigate",
					),
				),
			),
			HTNMethod(
				method_name="m_get_soil_data",
				task_name="get_soil_data",
				parameters=("WAYPOINT", "ROVER", "STORE"),
				task_args=("WAYPOINT",),
				context=(
					HTNLiteral("store_of", ("STORE", "ROVER"), True, None),
					HTNLiteral("equipped_for_soil_analysis", ("ROVER",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="navigate_abs",
						args=("ROVER", "WAYPOINT"),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="sample_soil",
						args=("ROVER", "STORE", "WAYPOINT"),
						kind="primitive",
						action_name="sample_soil",
					),
				),
				ordering=(("s1", "s2"),),
			),
		],
	)

	code = renderer.generate(
		domain=_marsrover_domain(),
		objects=("rover0", "rover0store", "waypoint0", "waypoint2"),
		method_library=method_library,
		plan_records=[],
	)

	get_soil_lines = [
		line for line in code.splitlines() if line.startswith("+!get_soil_data(")
	]
	assert any("at(ROVER, WAYPOINT2)" in line for line in get_soil_lines)
	assert any("can_traverse(ROVER, WAYPOINT2, WAYPOINT)" in line for line in get_soil_lines)


def test_renderer_specialises_parent_only_with_child_entry_requirements(tmp_path):
	renderer = AgentSpeakRenderer()
	domain = _parse_domain_text(
		tmp_path,
		"routing_world.hddl",
		"""
(define (domain routing_world)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types vehicle location)
  (:predicates
    (at ?v - vehicle ?l - location)
    (road ?from - location ?to - location)
  )
  (:task go_to
    :parameters (?v - vehicle ?to - location)
  )
  (:task execute
    :parameters (?to - location)
  )
  (:action drive
    :parameters (?v - vehicle ?from - location ?to - location)
    :precondition (and (at ?v ?from) (road ?from ?to))
    :effect (and (at ?v ?to) (not (at ?v ?from)))
  )
  (:action use_destination
    :parameters (?v - vehicle ?to - location)
    :precondition (and (at ?v ?to))
    :effect (and)
  )
)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("go_to", ("VEHICLE", "DESTINATION"), False, ()),
			HTNTask("execute", ("DESTINATION",), False, ()),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_go_to_via",
				task_name="go_to",
				parameters=("VEHICLE", "SOURCE", "MID", "DESTINATION"),
				task_args=("VEHICLE", "DESTINATION"),
				context=(HTNLiteral("road", ("MID", "DESTINATION"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="drive",
						args=("VEHICLE", "SOURCE", "MID"),
						kind="primitive",
						action_name="drive",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="drive",
						args=("VEHICLE", "MID", "DESTINATION"),
						kind="primitive",
						action_name="drive",
					),
				),
				ordering=(("s1", "s2"),),
			),
			HTNMethod(
				method_name="m_execute",
				task_name="execute",
				parameters=("DESTINATION", "VEHICLE"),
				task_args=("DESTINATION",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="go_to",
						args=("VEHICLE", "DESTINATION"),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="use_destination",
						args=("VEHICLE", "DESTINATION"),
						kind="primitive",
						action_name="use_destination",
					),
				),
				ordering=(("s1", "s2"),),
			),
		],
	)

	code = renderer.generate(
		domain=domain,
		objects=("vehicle0", "location0", "location1", "location2"),
		method_library=method_library,
		plan_records=[],
	)

	execute_line = next(line for line in code.splitlines() if line.startswith("+!execute("))
	assert "road(" in execute_line
	assert "at(VEHICLE, LOCATION" in execute_line
	assert execute_line.count("at(VEHICLE, LOCATION") == 1


def test_renderer_does_not_hoist_child_internal_progress_preconditions_for_later_compound_steps(
	tmp_path,
):
	renderer = AgentSpeakRenderer()
	domain = _parse_domain_text(
		tmp_path,
		"prepare_world.hddl",
		"""
(define (domain prepare_world)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types vehicle location)
  (:predicates
    (armed ?v - vehicle)
    (ready ?v - vehicle)
    (at ?v - vehicle ?l - location)
  )
  (:task finish
    :parameters (?v - vehicle ?to - location)
  )
  (:task execute
    :parameters (?to - location)
  )
  (:action arm
    :parameters (?v - vehicle)
    :precondition (and)
    :effect (and (armed ?v))
  )
  (:action ready_up
    :parameters (?v - vehicle)
    :precondition (and (armed ?v))
    :effect (and (ready ?v))
  )
  (:action complete
    :parameters (?v - vehicle ?to - location)
    :precondition (and (ready ?v) (at ?v ?to))
    :effect (and)
  )
)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("finish", ("VEHICLE", "DESTINATION"), False, ()),
			HTNTask("execute", ("DESTINATION",), False, ()),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_finish",
				task_name="finish",
				parameters=("VEHICLE", "DESTINATION"),
				task_args=("VEHICLE", "DESTINATION"),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="ready_up",
						args=("VEHICLE",),
						kind="primitive",
						action_name="ready_up",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="complete",
						args=("VEHICLE", "DESTINATION"),
						kind="primitive",
						action_name="complete",
					),
				),
				ordering=(("s1", "s2"),),
			),
			HTNMethod(
				method_name="m_execute",
				task_name="execute",
				parameters=("DESTINATION", "VEHICLE"),
				task_args=("DESTINATION",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="arm",
						args=("VEHICLE",),
						kind="primitive",
						action_name="arm",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="finish",
						args=("VEHICLE", "DESTINATION"),
						kind="compound",
					),
				),
				ordering=(("s1", "s2"),),
			),
		],
	)

	code = renderer.generate(
		domain=domain,
		objects=("vehicle0", "location0"),
		method_library=method_library,
		plan_records=[],
	)

	execute_line = next(line for line in code.splitlines() if line.startswith("+!execute("))
	assert "at(VEHICLE, LOCATION)" in execute_line
	assert "ready(VEHICLE)" not in execute_line


def test_renderer_does_not_hoist_child_preconditions_already_satisfied_by_prior_effects(tmp_path):
	renderer = AgentSpeakRenderer()
	domain = _parse_domain_text(
		tmp_path,
		"power_domain.hddl",
		"""
(define (domain power_world)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types device mode)
  (:predicates
    (installed ?d - device)
    (powered ?d - device)
    (configured ?d - device ?m - mode)
  )
  (:task enable
    :parameters (?d - device)
  )
  (:task finish
    :parameters (?d - device)
  )
  (:action turn_on
    :parameters (?d - device)
    :precondition (and (installed ?d))
    :effect (and (powered ?d))
  )
)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("enable", ("DEVICE",), False, ()),
			HTNTask("finish", ("DEVICE",), False, ()),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_enable",
				task_name="enable",
				parameters=("DEVICE",),
				task_args=("DEVICE",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="turn_on",
						args=("DEVICE",),
						kind="primitive",
						action_name="turn_on",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="finish",
						args=("DEVICE",),
						kind="compound",
					),
				),
				ordering=(("s1", "s2"),),
			),
			HTNMethod(
				method_name="m_finish",
				task_name="finish",
				parameters=("DEVICE", "MODE"),
				task_args=("DEVICE",),
				context=(
					HTNLiteral("powered", ("DEVICE",), True, None),
					HTNLiteral("configured", ("DEVICE", "MODE"), True, None),
				),
				subtasks=(),
			),
		],
	)

	code = renderer.generate(
		domain=domain,
		objects=("device0", "mode0"),
		method_library=method_library,
		plan_records=[],
	)

	enable_line = next(line for line in code.splitlines() if line.startswith("+!enable("))
	assert "installed(DEVICE)" in enable_line
	assert "configured(DEVICE, MODE)" in enable_line
	assert "powered(DEVICE)" not in enable_line


def test_renderer_does_not_hoist_later_preconditions_for_predicates_modified_earlier(tmp_path):
	renderer = AgentSpeakRenderer()
	domain = _parse_domain_text(
		tmp_path,
		"mode_domain.hddl",
		"""
(define (domain mode_world)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types mode)
  (:predicates
    (available)
    (active ?m - mode)
  )
  (:task execute
    :parameters (?target - mode)
  )
  (:action switch_mode
    :parameters (?from - mode ?to - mode)
    :precondition (and (available))
    :effect (and (active ?to) (not (active ?from)))
  )
  (:action use_mode
    :parameters (?m - mode)
    :precondition (and (active ?m))
    :effect (and)
  )
)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("execute", ("TARGET",), False, ())],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_execute",
				task_name="execute",
				parameters=("CURRENT", "TARGET"),
				task_args=("TARGET",),
				context=(HTNLiteral("available", (), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="switch_mode",
						args=("CURRENT", "TARGET"),
						kind="primitive",
						action_name="switch_mode",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="use_mode",
						args=("TARGET",),
						kind="primitive",
						action_name="use_mode",
					),
				),
				ordering=(("s1", "s2"),),
			),
		],
	)

	code = renderer.generate(
		domain=domain,
		objects=("mode0", "mode1"),
		method_library=method_library,
		plan_records=[],
	)

	execute_line = next(line for line in code.splitlines() if line.startswith("+!execute("))
	assert "available" in execute_line
	assert "active(" not in execute_line


def test_renderer_quotes_problem_objects_that_are_not_safe_agentspeak_atoms(tmp_path):
	renderer = AgentSpeakRenderer()
	domain = _parse_domain_text(
		tmp_path,
		"observe_domain.hddl",
		"""
(define (domain observe_world)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types target)
  (:predicates
    (seen ?t - target)
  )
  (:task observe
    :parameters (?t - target)
  )
  (:action snap
    :parameters (?t - target)
    :precondition (and)
    :effect (and (seen ?t))
  )
)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("observe", ("TARGET",), False, ("seen",))],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_observe",
				task_name="observe",
				parameters=("TARGET",),
				task_args=("TARGET",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="snap",
						args=("TARGET",),
						kind="primitive",
						action_name="snap",
					),
				),
				ordering=(),
			),
		],
		target_literals=[HTNLiteral("seen", ("Phenomenon4",), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("seen(Phenomenon4)", "observe")],
	)

	code = renderer.generate(
		domain=domain,
		objects=("Phenomenon4",),
		method_library=method_library,
		plan_records=[
			{
				"transition_name": "dfa_step_q1_q2_seen_Phenomenon4",
				"source_state": "q1",
				"target_state": "q2",
				"initial_state": "q1",
				"accepting_states": ["q2"],
				"label": "seen(Phenomenon4)",
				"plan": PANDAPlanResult(
					task_name="observe",
					task_args=("Phenomenon4",),
					target_literal=HTNLiteral("seen", ("Phenomenon4",), True, None),
				),
			},
		],
	)

	assert 'object("Phenomenon4").' in code
	assert '!observe("Phenomenon4");' in code


def test_renderer_embeds_runtime_method_trace_logging():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("empty-store", ("STORE", "ROVER"), False, ("empty",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m-empty-store-1",
				task_name="empty-store",
				parameters=("STORE", "ROVER"),
				context=(HTNLiteral("empty", ("STORE",), True, None),),
			),
		],
	)

	code = renderer.generate(
		domain=_marsrover_domain(),
		objects=("rover0store", "rover0"),
		method_library=method_library,
		plan_records=[],
	)

	assert 'runtime trace method ' in code
	assert 'trace_method("m-empty-store-1", STORE, ROVER)' in code


def test_renderer_stops_recursive_compound_effect_summary_cycles():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("loop_task", ("ITEM",), False, ("done",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_loop_task",
				task_name="loop_task",
				parameters=("ITEM",),
				task_args=("ITEM",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="loop_task",
						args=("ITEM",),
						kind="compound",
					),
				),
			),
		],
	)

	code = renderer.generate(
		domain=_domain(),
		objects=("a",),
		method_library=method_library,
		plan_records=[],
	)

	assert "+!loop_task(" in code


def test_renderer_stops_mutual_recursive_compound_effect_summary_cycles():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("helper_holding", ("ITEM",), False, ("holding",)),
			HTNTask("helper_clear", ("ITEM",), False, ("clear",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_helper_holding",
				task_name="helper_holding",
				parameters=("ITEM",),
				task_args=("ITEM",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="helper_clear",
						args=("ITEM",),
						kind="compound",
					),
				),
			),
			HTNMethod(
				method_name="m_helper_clear",
				task_name="helper_clear",
				parameters=("ITEM",),
				task_args=("ITEM",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="helper_holding",
						args=("ITEM",),
						kind="compound",
					),
				),
			),
		],
	)

	code = renderer.generate(
		domain=_domain(),
		objects=("a",),
		method_library=method_library,
		plan_records=[],
	)

	assert "+!helper_holding(" in code
	assert "+!helper_clear(" in code


def test_renderer_hoists_branch_discriminators_from_action_schema():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B",), False, ("holding",)),
			HTNTask("clear_top", ("B",), False, ("clear",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("B",),
				context=(
					HTNLiteral("holding", ("B",), True, None),
				),
			),
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
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up_from_table",
						args=("B",),
						kind="primitive",
						action_name="pick-up",
					),
				),
				ordering=(("s1", "s2"),),
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
						action_name="unstack",
						preconditions=(
							HTNLiteral("on", ("B", "SUPPORT"), True, None),
						),
					),
				),
				ordering=(("s1", "s2"),),
			),
		],
	)

	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=method_library,
		plan_records=[],
	)

	assert "+!hold_block(BLOCK1) : holding(BLOCK1) <-" in code
	assert "+!hold_block(BLOCK1) : ontable(BLOCK1) <-" in code
	assert "\t!clear_top(BLOCK1);" in code
	assert "\t!pick_up_from_table(BLOCK1)." in code
	assert "+!hold_block(BLOCK1) : on(BLOCK1, BLOCK2) <-" in code
	assert "\t!pick_up(BLOCK1, BLOCK2)." in code


def test_renderer_emits_agentspeak_equality_and_disequality_constraints():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
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
			HTNMethod(
				method_name="m_keep_apart_same",
				task_name="keep_apart",
				parameters=("LEFT", "RIGHT"),
				context=(
					HTNLiteral("=", ("LEFT", "RIGHT"), True, None),
				),
			),
		],
	)

	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=method_library,
		plan_records=[],
	)

	assert "+!keep_apart(BLOCK1, BLOCK2) : BLOCK1 \\== BLOCK2 <-" in code
	assert "+!keep_apart(BLOCK1, BLOCK2) : BLOCK1 == BLOCK2 <-" in code


def test_renderer_renders_all_negative_contexts_as_naf():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("check_clear", ("B",), False, ("clear",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_check_clear_naf",
				task_name="check_clear",
				parameters=("B",),
				context=(
					HTNLiteral("clear", ("B",), False, None, "naf"),
				),
			),
				HTNMethod(
					method_name="m_check_clear_strong",
					task_name="check_clear",
					parameters=("B",),
					context=(
						HTNLiteral("clear", ("B",), False, None, "strong"),
					),
				),
			],
			target_literals=[
				HTNLiteral("clear", ("a",), False, None, "strong"),
			],
			target_task_bindings=[
				HTNTargetTaskBinding("!clear(a)", "check_clear"),
			],
		)

	code = renderer.generate(
		domain=_domain(),
		objects=("a",),
		method_library=method_library,
		plan_records=[],
	)

	assert "+!check_clear(BLOCK) : not clear(BLOCK) <-" in code
	assert "~clear(BLOCK)" not in code


def test_renderer_does_not_emit_tilde_effect_updates():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		target_literals=[
			HTNLiteral("clear", ("a",), False, None, "strong"),
		],
	)

	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=method_library,
		plan_records=[],
	)

	assert "~clear(" not in code


def test_renderer_supports_or_in_primitive_precondition_context():
	renderer = AgentSpeakRenderer()
	action = type(
		"ActionStub",
		(),
		{
			"name": "probe",
			"parameters": ["?x - object"],
			"preconditions": "(or (clear ?x) (holding ?x))",
			"effects": "(and (checked ?x))",
		},
	)()
	domain = type(
		"DomainStub",
		(),
		{
			"name": "demo",
			"actions": [action],
			"predicates": [],
			"types": ["object"],
		},
	)()

	code = renderer.generate(
		domain=domain,
		objects=("a",),
		method_library=HTNMethodLibrary(),
		plan_records=[],
	)

	assert "+!probe(OBJECT) : clear(OBJECT) | holding(OBJECT) <-" in code


def test_renderer_identifies_pure_noop_methods_generically():
	renderer = AgentSpeakRenderer()
	noop_method = HTNMethod(
		method_name="m_clear_top_noop",
		task_name="clear_top",
		parameters=("BLOCK1",),
		context=(HTNLiteral("clear", ("BLOCK1",), True, None),),
		subtasks=(),
	)
	decomposing_method = HTNMethod(
		method_name="m_clear_top_recursive",
		task_name="clear_top",
		parameters=("BLOCK1",),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="pick_up",
				args=("BLOCK1", "BLOCK2"),
				kind="primitive",
				action_name="unstack",
			),
		),
	)

	assert renderer._method_is_pure_noop(noop_method) is True
	assert renderer._method_is_pure_noop(decomposing_method) is False


def test_renderer_generate_accepts_ordered_query_sequence_flag():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=HTNMethodLibrary(),
		plan_records=[],
		ordered_query_sequence=False,
	)

	assert "domain(" in code


def test_renderer_supports_imply_in_primitive_precondition_context():
	renderer = AgentSpeakRenderer()
	action = type(
		"ActionStub",
		(),
		{
			"name": "seal_if_clear",
			"parameters": ["?x - object"],
			"preconditions": "(imply (clear ?x) (holding ?x))",
			"effects": "(and (sealed ?x))",
		},
	)()
	domain = type(
		"DomainStub",
		(),
		{
			"name": "demo",
			"actions": [action],
			"predicates": [],
			"types": ["object"],
		},
	)()

	code = renderer.generate(
		domain=domain,
		objects=("a",),
		method_library=HTNMethodLibrary(),
		plan_records=[],
	)

	assert "+!seal_if_clear(OBJECT) : not clear(OBJECT) | holding(OBJECT) <-" in code


def test_renderer_uses_zero_arity_action_functor_without_empty_parentheses(tmp_path):
	domain_file = tmp_path / "zero_arity_domain.hddl"
	domain_file.write_text(
		"""
(define (domain zero_arity)
  (:requirements :typing :hierarchy)
  (:types block)
  (:predicates (ready))
  (:action nop
    :parameters ()
    :precondition (and)
    :effect (and)
  )
)
		""".strip(),
	)
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=HDDLParser.parse_domain(str(domain_file)),
		objects=(),
		method_library=HTNMethodLibrary(),
		plan_records=[],
	)

	assert "\tnop." in code
	assert "nop()." not in code
