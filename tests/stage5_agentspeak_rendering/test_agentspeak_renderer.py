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
from stage5_agentspeak_rendering.dfa_runtime import build_agentspeak_transition_specs
from stage1_interpretation.grounding_map import GroundingMap
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
	assert "\t-dfa_state(q1);" in code
	assert "\t!dfa_step_q1_q2_on_a_b(a, b);" in code
	assert "\t+dfa_state(q2)." in code
	assert "+!run_dfa : dfa_state(q1) <-" in code
	assert "\t!dfa_step_q1_q2_on_a_b;" in code
	assert "\t!run_dfa." in code
	assert "+!run_dfa : dfa_state(q2) & accepting_state(q2) <-" in code
	assert "+!stack(BLOCK1, BLOCK2) :" in code

def test_renderer_transition_wrappers_prefer_transition_task_for_non_transition_plan_record():
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
				"query_root_task_name": "query_root_1_do_put_on",
				"query_root_task_args": ["a", "b"],
				"plan": PANDAPlanResult(
					task_name="place_on",
					task_args=("a", "b"),
					target_literal=HTNLiteral("on", ("a", "b"), True, "on_a_b"),
				),
			},
		],
	)

	assert "+!dfa_step_q1_q2_on_a_b : dfa_state(q1) <-" in code
	assert "\t-dfa_state(q1);" in code
	assert "\t!dfa_step_q1_q2_on_a_b(a, b);" in code
	assert "\t!query_root_1_do_put_on(a, b);" not in code
	assert "\t!place_on(a, b);" not in code

def test_renderer_raw_dfa_control_uses_numeric_query_steps_and_domain_tasks():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=_method_library(),
		plan_records=[],
		transition_specs=[
			{
				"transition_name": "dfa_t1",
				"source_state": "q1",
				"target_state": "q2",
				"initial_state": "q1",
				"accepting_states": ["q2"],
				"raw_label": "on_a_b",
				"guard_context": "on(a, b)",
			},
		],
		query_task_anchors=[
			{"task_name": "place_on", "args": ["a", "b"]},
		],
	)

	assert "query_step(1)." in code
	assert "+!dfa_t1 : dfa_state(q1) & on(a, b) <-" in code
	assert "+!advance_dfa : dfa_state(q1) & on(a, b) <-" in code
	assert "+!run_dfa : query_step(1) <-" in code
	assert "\t!place_on(a, b);" in code
	assert "\t-query_step(1);" in code
	assert "\t+query_step(2);" in code
	assert 'query_step("1")' not in code
	assert 'query_step("2")' not in code

def test_build_agentspeak_transition_specs_uses_unit_progress_edges_for_query_steps():
	transition_specs = build_agentspeak_transition_specs(
		dfa_result={
			"dfa_dot": """
digraph MONA_DFA {
 rankdir = LR;
 node [shape = doublecircle]; 4;
 node [shape = circle]; 1;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 1 [label="~query_step_1 & ~query_step_2"];
 1 -> 2 [label="query_step_1 & ~query_step_2"];
 1 -> 3 [label="query_step_2 & ~query_step_1"];
 1 -> 4 [label="query_step_1 & query_step_2"];
 2 -> 2 [label="~query_step_2"];
 2 -> 4 [label="query_step_2"];
 3 -> 3 [label="true"];
 4 -> 4 [label="true"];
}
""",
		},
		grounding_map=GroundingMap(),
	)

	assert [spec["query_step_index"] for spec in transition_specs] == [1, 2]
	assert [spec["raw_label"] for spec in transition_specs] == [
		"query_step_1",
		"query_step_2",
	]
	assert all(spec["stateless_query_step"] is True for spec in transition_specs)

def test_build_agentspeak_transition_specs_keeps_stateful_query_step_edges_for_ordered_queries():
	transition_specs = build_agentspeak_transition_specs(
		dfa_result={
			"dfa_dot": """
digraph MONA_DFA {
 rankdir = LR;
 node [shape = doublecircle]; 4;
 node [shape = circle]; 1;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 1 [label="~query_step_1 & ~query_step_2"];
 1 -> 2 [label="query_step_1 & ~query_step_2"];
 1 -> 3 [label="query_step_2 & ~query_step_1"];
 2 -> 4 [label="query_step_2"];
 3 -> 4 [label="query_step_1"];
 4 -> 4 [label="true"];
}
""",
		},
		grounding_map=GroundingMap(),
		ordered_query_sequence=True,
	)

	assert [spec["query_step_index"] for spec in transition_specs] == [1, 2, 2, 1]
	assert all("stateless_query_step" not in spec for spec in transition_specs)

def test_build_agentspeak_transition_specs_prefers_query_order_within_same_source_state():
	transition_specs = build_agentspeak_transition_specs(
		dfa_result={
			"dfa_dot": """
digraph MONA_DFA {
 rankdir = LR;
 node [shape = doublecircle]; 4;
 node [shape = circle]; 1;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 2 [label="query_step_3"];
 1 -> 3 [label="query_step_2"];
 1 -> 4 [label="query_step_1"];
 2 -> 4 [label="true"];
 3 -> 4 [label="true"];
 4 -> 4 [label="true"];
}
""",
		},
		grounding_map=GroundingMap(),
	)

	assert [spec["query_step_index"] for spec in transition_specs] == [1, 2, 3]


def test_build_agentspeak_transition_specs_supports_symbolic_unordered_query_step_monitor():
	transition_specs = build_agentspeak_transition_specs(
		dfa_result={
			"symbolic_query_step_monitor": {
				"mode": "unordered_eventually_conjunction",
				"query_step_indices": [1, 2, 3],
				"initial_state": "q0",
				"accepting_states": [],
				"num_states": 8,
				"num_transitions": 12,
			},
		},
		grounding_map=GroundingMap(),
	)

	assert [spec["query_step_index"] for spec in transition_specs] == [1, 2, 3]
	assert [spec["raw_label"] for spec in transition_specs] == [
		"query_step_1",
		"query_step_2",
		"query_step_3",
	]
	assert [spec["source_state"] for spec in transition_specs] == ["q0", "q0", "q0"]
	assert all(spec["stateless_query_step"] is True for spec in transition_specs)
	assert [spec["transition_name"] for spec in transition_specs] == [
		"dfa_t1",
		"dfa_t2",
		"dfa_t3",
	]


def test_build_agentspeak_transition_specs_preserves_generic_progress_edges():
	grounding_map = GroundingMap()
	grounding_map.add_atom("clear_a", "clear", ["a"])
	grounding_map.add_atom("on_a_b", "on", ["a", "b"])
	grounding_map.add_atom("on_b_c", "on", ["b", "c"])

	transition_specs = build_agentspeak_transition_specs(
		dfa_result={
			"dfa_dot": """
digraph MONA_DFA {
 node [shape = doublecircle]; 3;
 node [shape = circle]; 1 2 s1 s2;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 2 [label="!clear_a"];
 1 -> s1 [label="clear_a"];
 2 -> 2 [label="!clear_a"];
 2 -> 2 [label="clear_a"];
 3 -> 3 [label="!clear_a"];
 3 -> 3 [label="clear_a"];
 s1 -> 2 [label="!on_a_b"];
 s1 -> s2 [label="on_a_b"];
 s2 -> 2 [label="!on_b_c"];
 s2 -> 3 [label="on_b_c"];
}
""",
		},
		grounding_map=grounding_map,
	)

	assert sorted(
		(
			spec["transition_name"],
			spec["source_state"],
			spec["target_state"],
			spec["raw_label"],
		)
		for spec in transition_specs
	) == [
		("dfa_t1", "q1", "q2", "clear_a"),
		("dfa_t2", "q3", "q4", "on_b_c"),
		("dfa_t3", "q2", "q3", "on_a_b"),
	]
	assert sorted(spec["guard_context"] for spec in transition_specs) == [
		"clear(a)",
		"on(a, b)",
		"on(b, c)",
	]


def test_build_agentspeak_transition_specs_preserves_generic_unordered_progress_edges():
	grounding_map = GroundingMap()
	grounding_map.add_atom("goal_a", "goal", ["a"])
	grounding_map.add_atom("goal_b", "goal", ["b"])

	transition_specs = build_agentspeak_transition_specs(
		dfa_result={
			"dfa_dot": """
digraph MONA_DFA {
 node [shape = doublecircle]; 2;
 node [shape = circle]; 1;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 2 [label="goal_a"];
 1 -> 2 [label="goal_b"];
}
""",
		},
		grounding_map=grounding_map,
	)

	assert [
		(spec["transition_name"], spec["source_state"], spec["target_state"], spec["raw_label"])
		for spec in transition_specs
	] == [
		("dfa_t1", "q1", "q2", "goal_a"),
		("dfa_t2", "q1", "q2", "goal_b"),
	]
	assert [spec["guard_context"] for spec in transition_specs] == [
		"goal(a)",
		"goal(b)",
	]

def test_renderer_raw_dfa_query_step_transition_executes_anchor_task():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=_method_library(),
		plan_records=[],
		transition_specs=[
			{
				"transition_name": "dfa_t1",
				"source_state": "q1",
				"target_state": "q2",
				"initial_state": "q1",
				"accepting_states": ["q2"],
				"raw_label": "query_step_1",
				"guard_context": "true",
				"query_step_index": 1,
			},
		],
		query_task_anchors=[
			{"task_name": "place_on", "args": ["a", "b"]},
		],
	)

	assert "+!dfa_t1 : dfa_state(q1) <-" in code
	assert "\t!place_on(a, b);" in code
	assert "\t-dfa_state(q1);" in code
	assert "\t+dfa_state(q2)." in code
	assert "+!run_dfa : dfa_state(q1) <-" in code
	assert "\t!dfa_t1;" in code
	assert "\t!run_dfa." in code
	assert "+!advance_dfa" not in code

def test_renderer_raw_dfa_stateless_query_step_transition_executes_anchor_task_without_dfa_guard():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=_method_library(),
		plan_records=[],
		transition_specs=[
			{
				"transition_name": "dfa_t1",
				"source_state": "q1",
				"target_state": "q1",
				"initial_state": "q1",
				"accepting_states": ["q2"],
				"raw_label": "query_step_1",
				"guard_context": "true",
				"query_step_index": 1,
				"stateless_query_step": True,
			},
		],
		query_task_anchors=[
			{"task_name": "place_on", "args": ["a", "b"]},
		],
	)

	assert "+!dfa_t1 : true <-" in code
	assert "\t!place_on(a, b)." in code
	assert "\t-dfa_state(q1);" not in code
	assert "\t+dfa_state(q1)." not in code

def test_renderer_labels_query_step_edges_with_target_literal_signatures():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=_method_library(),
		plan_records=[],
		transition_specs=[
			{
				"transition_name": "dfa_t1",
				"source_state": "q1",
				"target_state": "q2",
				"initial_state": "q1",
				"accepting_states": ["q2"],
				"raw_label": "query_step_1",
				"guard_context": "true",
				"query_step_index": 1,
			},
		],
		query_task_anchors=[
			{
				"task_name": "place_on",
				"args": ["a", "b"],
				"literal_signature": "on(a, b)",
			},
		],
	)

	assert 'dfa_edge_label(dfa_t1, "on(a, b)").' in code
	assert 'dfa_edge_label(dfa_t1, "query_step_1").' not in code

def test_renderer_raw_dfa_mode_does_not_fall_back_to_plan_records_when_no_edges():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=HTNMethodLibrary(),
		plan_records=[
			{
				"source_state": "q1",
				"target_state": "q2",
				"label": "on(a, b)",
			},
		],
		transition_specs=[],
		query_task_anchors=[
			{"task_name": "place_on", "args": ["a", "b"]},
		],
	)

	assert "dfa_edge_label(" not in code

def test_renderer_keeps_stage4_actual_plan_out_of_rendered_method_library(tmp_path):
	domain = _parse_domain_text(
		tmp_path,
		"deliver-domain.hddl",
		"""
		(define (domain deliver_demo)
			(:requirements :typing :hierarchy)
			(:types object package truck location)
			(:predicates
				(in ?p - package ?t - truck)
				(at ?x - object ?l - location)
			)
			(:task deliver
				:parameters (?p - package ?l - location)
			)
			(:task helper_in
				:parameters (?p - package ?t - truck)
			)
			(:task helper_at
				:parameters (?x - object ?l - location)
			)
		)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("deliver", ("P", "L"), False, ("at",)),
			HTNTask("helper_in", ("P", "T"), False, ("in",)),
			HTNTask("helper_at", ("X", "L"), False, ("at",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_deliver",
				task_name="deliver",
				parameters=("P", "L", "T"),
				task_args=("P", "L"),
				subtasks=(
					HTNMethodStep("s1", "helper_in", ("P", "T"), "compound"),
					HTNMethodStep("s2", "helper_at", ("T", "L"), "compound"),
				),
				ordering=(("s1", "s2"),),
			),
			HTNMethod(
				method_name="m_helper_in_noop",
				task_name="helper_in",
				parameters=("P", "T"),
				task_args=("P", "T"),
				context=(HTNLiteral("in", ("P", "T"), True, None),),
			),
			HTNMethod(
				method_name="m_helper_at_noop",
				task_name="helper_at",
				parameters=("X", "L"),
				task_args=("X", "L"),
				context=(HTNLiteral("at", ("X", "L"), True, None),),
			),
		],
	)
	renderer = AgentSpeakRenderer()

	code = renderer.generate(
		domain=domain,
		objects=("package0", "truck0", "city1"),
		method_library=method_library,
		plan_records=[
			{
				"transition_name": "dfa_step_q1_q2_done_package0_city1",
				"source_state": "q1",
				"target_state": "q2",
				"initial_state": "q1",
				"accepting_states": ["q2"],
				"label": "done(package0, city1)",
				"plan": PANDAPlanResult(
					task_name="dfa_step_q1_q2_done_package0_city1",
					task_args=("package0", "city1"),
					target_literal=HTNLiteral("done", ("package0", "city1"), True, None),
					actual_plan="\n".join(
						[
							"==>",
							"1 helper_in package0 truck0 -> m_helper_in_noop",
							"2 helper_at truck0 city1 -> m_helper_at_noop",
							"root 3",
							"3 deliver package0 city1 -> m_deliver 1 2",
						],
					)
					+ "\n",
				),
			},
		],
	)

	assert "+!deliver(package0, city1) : in(package0, truck0) & at(truck0, city1) <-" not in code
	assert "+!helper_in(package0, truck0) : in(package0, truck0) <-" not in code
	assert "+!helper_at(truck0, city1) : at(truck0, city1) <-" not in code
	assert "+!deliver(PACKAGE, LOCATION) : at(TRUCK, LOCATION) <-" in code
	assert "\t!helper_in(PACKAGE, TRUCK);" in code
	assert "\t!helper_at(TRUCK, LOCATION)." in code

def test_renderer_skips_exact_grounded_chunks_with_synthetic_witness_placeholders(tmp_path):
	domain = _parse_domain_text(
		tmp_path,
		"deliver-domain.hddl",
		"""
		(define (domain deliver_demo)
			(:requirements :typing :hierarchy)
			(:types object package truck location)
			(:predicates
				(in ?p - package ?t - truck)
				(at ?x - object ?l - location)
			)
			(:task deliver
				:parameters (?p - package ?l - location)
			)
			(:task helper_in
				:parameters (?p - package ?t - truck)
			)
		)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("deliver", ("P", "L"), False, ("at",)),
			HTNTask("helper_in", ("P", "T"), False, ("in",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_deliver",
				task_name="deliver",
				parameters=("P", "L", "T"),
				task_args=("P", "L"),
				subtasks=(HTNMethodStep("s1", "helper_in", ("P", "T"), "compound"),),
				ordering=(),
			),
			HTNMethod(
				method_name="m_helper_in_noop",
				task_name="helper_in",
				parameters=("P", "T"),
				task_args=("P", "T"),
				context=(HTNLiteral("in", ("P", "T"), True, None),),
			),
		],
	)
	renderer = AgentSpeakRenderer()

	code = renderer.generate(
		domain=domain,
		objects=("package0", "city1"),
		method_library=method_library,
		plan_records=[
			{
				"transition_name": "dfa_step_q1_q2_done_package0_city1",
				"source_state": "q1",
				"target_state": "q2",
				"initial_state": "q1",
				"accepting_states": ["q2"],
				"label": "done(package0, city1)",
				"plan": PANDAPlanResult(
					task_name="dfa_step_q1_q2_done_package0_city1",
					task_args=("package0", "city1"),
					target_literal=HTNLiteral("done", ("package0", "city1"), True, None),
					actual_plan="\n".join(
						[
							"==>",
							"1 helper_in package0 witness_vehicle_1 -> m_helper_in_noop",
							"root 2",
							"2 deliver package0 city1 -> m_deliver 1",
						],
					)
					+ "\n",
				),
			},
		],
	)

	assert "+!deliver(package0, city1)" not in code
	assert "witness_vehicle_1" not in code

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
	assert "\t-dfa_state(q2);" in code
	assert "\t!dfa_step_q2_q2_not_on_a_b(a, b);" in code
	assert "\t+dfa_state(q2)." in code
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
	recursive_headers = [
		lines[index - 1]
		for index, line in enumerate(lines)
		if '"m_drive_to_via"' in line and "runtime trace method flat" in line
	]
	assert any("road(" in header and "at(" in header for header in recursive_headers)
	assert any("road(" in header and "at(" not in header for header in recursive_headers)
	assert all("\\==" in header for header in recursive_headers)

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
	recursive_headers = [
		lines[index - 1]
		for index, line in enumerate(lines)
		if '"m_drive_to_via"' in line and "runtime trace method flat" in line
	]
	assert any("road(" in header and "at(" in header for header in recursive_headers)
	assert any("road(" in header and "at(" not in header for header in recursive_headers)
	assert all("\\==" in header for header in recursive_headers)

def test_renderer_adds_recursive_ancestor_guard_before_self_recursive_call():
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
					method_name="m_drive_to_via",
					task_name="get_to",
					parameters=("VEHICLE", "SOURCE", "MID", "DESTINATION"),
					task_args=("VEHICLE", "DESTINATION"),
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
			],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	assert "pipeline.no_ancestor_goal(get_to, VEHICLE, LOCATION3);" in code
	assert code.index("pipeline.no_ancestor_goal(get_to, VEHICLE, LOCATION3);") < code.index(
		"!get_to(VEHICLE, LOCATION3);"
	)

def test_renderer_omits_ancestor_guard_before_non_recursive_compound_child_call():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("achieve", ("X",), False, ("done",)),
				HTNTask("support", ("X",), False, ("ready",)),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_achieve",
					task_name="achieve",
					parameters=("X",),
					task_args=("X",),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="support",
							args=("X",),
							kind="compound",
						),
					),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	assert "pipeline.no_ancestor_goal(support, BLOCK);" not in code
	assert "!support(BLOCK)." in code

def test_renderer_adds_ancestor_guard_before_indirect_recursive_child_call():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("achieve", ("X",), False, ("done",)),
				HTNTask("support", ("X",), False, ("ready",)),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_achieve",
					task_name="achieve",
					parameters=("X",),
					task_args=("X",),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="support",
							args=("X",),
							kind="compound",
						),
					),
				),
				HTNMethod(
					method_name="m_support",
					task_name="support",
					parameters=("X",),
					task_args=("X",),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="achieve",
							args=("X",),
							kind="compound",
						),
					),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	assert "pipeline.no_ancestor_goal(achieve, BLOCK);" in code
	assert code.index("pipeline.no_ancestor_goal(achieve, BLOCK);") < code.index(
		"!achieve(BLOCK).",
	)

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

	i_am_there_index = code.index('"m_i_am_there"')
	drive_to_index = code.index('"m_drive_to"')
	drive_to_via_index = code.index('"m_drive_to_via"')

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
	assert "\t!dfa_step_q0_q1_empty_rover0store(rover0store);" in code
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

def test_renderer_propagates_first_child_specialisation_through_first_child_chain(tmp_path):
	renderer = AgentSpeakRenderer()
	domain = _parse_domain_text(
		tmp_path,
		"first_child_chain.hddl",
		"""
(define (domain first_child_chain)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types vehicle location)
  (:predicates
    (at ?v - vehicle ?l - location)
    (road ?from - location ?to - location)
  )
  (:task navigate_abs
    :parameters (?v - vehicle ?to - location)
  )
  (:task collect_data
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
  (:action collect
    :parameters (?v - vehicle ?to - location)
    :precondition (and (at ?v ?to))
    :effect (and)
  )
  (:action report
    :parameters (?v - vehicle ?to - location)
    :precondition (and (at ?v ?to))
    :effect (and)
  )
)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("navigate_abs", ("VEHICLE", "DESTINATION"), False, ()),
			HTNTask("collect_data", ("VEHICLE", "DESTINATION"), False, ()),
			HTNTask("execute", ("DESTINATION",), False, ()),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_navigate_abs_noop",
				task_name="navigate_abs",
				parameters=("VEHICLE", "DESTINATION"),
				task_args=("VEHICLE", "DESTINATION"),
				context=(HTNLiteral("at", ("VEHICLE", "DESTINATION"), True, None),),
				subtasks=(),
			),
			HTNMethod(
				method_name="m_navigate_abs_direct",
				task_name="navigate_abs",
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
				method_name="m_collect_data",
				task_name="collect_data",
				parameters=("VEHICLE", "DESTINATION"),
				task_args=("VEHICLE", "DESTINATION"),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="navigate_abs",
						args=("VEHICLE", "DESTINATION"),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="collect",
						args=("VEHICLE", "DESTINATION"),
						kind="primitive",
						action_name="collect",
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
						task_name="collect_data",
						args=("VEHICLE", "DESTINATION"),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="report",
						args=("VEHICLE", "DESTINATION"),
						kind="primitive",
						action_name="report",
					),
				),
				ordering=(("s1", "s2"),),
			),
		],
	)

	code = renderer.generate(
		domain=domain,
		objects=("vehicle0", "location0", "location1"),
		method_library=method_library,
		plan_records=[],
	)

	execute_lines = [line for line in code.splitlines() if line.startswith("+!execute(")]
	assert any("at(VEHICLE, LOCATION)" in line for line in execute_lines)

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

def test_renderer_only_hoists_first_compound_child_entry_requirements(tmp_path):
	renderer = AgentSpeakRenderer()
	domain = _parse_domain_text(
		tmp_path,
		"ordered_compound_chain.hddl",
		"""
(define (domain ordered_compound_chain)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types vehicle location)
  (:predicates
    (armed ?v - vehicle)
    (ready ?v - vehicle)
    (at ?v - vehicle ?l - location)
    (road ?from - location ?to - location)
  )
  (:task prepare
    :parameters (?v - vehicle)
  )
  (:task navigate_abs
    :parameters (?v - vehicle ?to - location)
  )
  (:task go_to
    :parameters (?v - vehicle ?to - location)
  )
  (:task execute
    :parameters (?v - vehicle ?to - location)
  )
  (:action ready_up
    :parameters (?v - vehicle)
    :precondition (and (armed ?v))
    :effect (and (ready ?v))
  )
  (:action drive
    :parameters (?v - vehicle ?from - location ?to - location)
    :precondition (and (ready ?v) (at ?v ?from) (road ?from ?to))
    :effect (and (at ?v ?to) (not (at ?v ?from)))
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
			HTNTask("prepare", ("VEHICLE",), False, ()),
			HTNTask("navigate_abs", ("VEHICLE", "DESTINATION"), False, ()),
			HTNTask("go_to", ("VEHICLE", "DESTINATION"), False, ()),
			HTNTask("execute", ("VEHICLE", "DESTINATION"), False, ()),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_prepare",
				task_name="prepare",
				parameters=("VEHICLE",),
				task_args=("VEHICLE",),
				context=(HTNLiteral("armed", ("VEHICLE",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="ready_up",
						args=("VEHICLE",),
						kind="primitive",
						action_name="ready_up",
					),
				),
			),
			HTNMethod(
				method_name="m_navigate_abs",
				task_name="navigate_abs",
				parameters=("VEHICLE", "SOURCE", "DESTINATION"),
				task_args=("VEHICLE", "DESTINATION"),
				context=(
					HTNLiteral("at", ("VEHICLE", "SOURCE"), True, None),
					HTNLiteral("road", ("SOURCE", "DESTINATION"), True, None),
					HTNLiteral("ready", ("VEHICLE",), True, None),
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
				method_name="m_go_to",
				task_name="go_to",
				parameters=("VEHICLE", "DESTINATION"),
				task_args=("VEHICLE", "DESTINATION"),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="navigate_abs",
						args=("VEHICLE", "DESTINATION"),
						kind="compound",
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
				parameters=("VEHICLE", "DESTINATION"),
				task_args=("VEHICLE", "DESTINATION"),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="prepare",
						args=("VEHICLE",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="go_to",
						args=("VEHICLE", "DESTINATION"),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="complete",
						args=("VEHICLE", "DESTINATION"),
						kind="primitive",
						action_name="complete",
					),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
			),
		],
	)

	code = renderer.generate(
		domain=domain,
		objects=("vehicle0", "location0", "location1"),
		method_library=method_library,
		plan_records=[],
	)

	execute_line = next(line for line in code.splitlines() if line.startswith("+!execute("))
	assert "armed(VEHICLE)" in execute_line
	assert "at(VEHICLE, LOCATION)" not in execute_line
	assert "road(" not in execute_line
	assert "ready(VEHICLE)" not in execute_line

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
	assert '!dfa_step_q1_q2_seen_Phenomenon4("Phenomenon4");' in code

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
	assert '"runtime trace method flat ", "m-empty-store-1", "|", STORE, "|", ROVER' in code

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

def test_renderer_priority_uses_explicit_context_only():
	class GuardedRenderer(AgentSpeakRenderer):
		def _method_context_literals(self, *args, **kwargs):
			raise AssertionError("render-priority sorting must not expand context")

	renderer = GuardedRenderer()
	method = HTNMethod(
		method_name="m_achieve_decompose",
		task_name="achieve",
		parameters=("X",),
		task_args=("X",),
		subtasks=tuple(
			HTNMethodStep(
				step_id=f"s{index}",
				task_name="helper",
				args=("X",),
				kind="compound",
			)
			for index in range(2)
		),
	)
	task_lookup = {
		"achieve": HTNTask(
			name="achieve",
			parameters=("X",),
			is_primitive=False,
			source_predicates=("done",),
		),
	}

	priority = renderer._method_render_priority(
		method,
		0,
		task_lookup,
		{"achieve": [method]},
		{"achieve": {}},
		{},
		{},
		{"done(X)"},
	)

	assert priority[0] == 1

def test_renderer_task_ordering_uses_headline_without_effect_template_expansion():
	class GuardedRenderer(AgentSpeakRenderer):
		def _compound_task_effect_templates(self, *args, **kwargs):
			raise AssertionError("render-priority sorting must not expand task effects")

	renderer = GuardedRenderer()
	noop_method = HTNMethod(
		method_name="m_achieve_noop",
		task_name="achieve",
		parameters=("X",),
		task_args=("X",),
		context=(HTNLiteral("done", ("X",), True, None),),
		subtasks=(),
	)
	constructive_method = HTNMethod(
		method_name="m_achieve_constructive",
		task_name="achieve",
		parameters=("X",),
		task_args=("X",),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="helper",
				args=("X",),
				kind="compound",
			),
		),
	)
	task_lookup = {
		"achieve": HTNTask(
			name="achieve",
			parameters=("X",),
			is_primitive=False,
			source_predicates=("done",),
			headline_literal=HTNLiteral("done", ("X",), True, None),
		),
	}

	ordered = renderer._ordered_task_methods_for_rendering(
		"achieve",
		(constructive_method, noop_method),
		task_lookup,
		{"achieve": [constructive_method, noop_method]},
		{"achieve": {}},
		{},
		{},
	)

	assert ordered[0] is noop_method

def test_renderer_task_ordering_preserves_noop_for_runtime_progress_guards():
	renderer = AgentSpeakRenderer()
	destructive_method = HTNMethod(
		method_name="m_do_on_table_destructive",
		task_name="do_on_table",
		parameters=("BLOCK1", "BLOCK2"),
		task_args=("BLOCK1",),
		context=(
			HTNLiteral("clear", ("BLOCK1",), True, None),
			HTNLiteral("handempty", (), True, None),
			HTNLiteral("ontable", ("BLOCK1",), False, None),
			HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
		),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="unstack",
				args=("BLOCK1", "BLOCK2"),
				kind="primitive",
			),
			HTNMethodStep(
				step_id="s2",
				task_name="put_down",
				args=("BLOCK1",),
				kind="primitive",
			),
		),
	)
	noop_method = HTNMethod(
		method_name="m_do_on_table_noop",
		task_name="do_on_table",
		parameters=("BLOCK1",),
		task_args=("BLOCK1",),
		context=(HTNLiteral("clear", ("BLOCK1",), True, None),),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="nop",
				args=(),
				kind="primitive",
			),
		),
	)
	task_lookup = {
		"do_on_table": HTNTask(
			name="do_on_table",
			parameters=("BLOCK1",),
			is_primitive=False,
			source_predicates=("ontable",),
			headline_literal=HTNLiteral("ontable", ("BLOCK1",), True, None),
		),
	}

	ordered = renderer._ordered_task_methods_for_rendering(
		"do_on_table",
		(destructive_method, noop_method),
		task_lookup,
		{"do_on_table": [destructive_method, noop_method]},
		{"do_on_table": {}},
		{},
		{},
	)

	assert ordered[0] is noop_method

def test_renderer_does_not_lift_later_step_preconditions_past_prior_compound_step(tmp_path):
	domain = _parse_domain_text(
		tmp_path,
		"satellite-mini.hddl",
		"""
		(define (domain satellite_mini)
			(:requirements :typing :hierarchy)
			(:types satellite instrument direction image_direction mode)
			(:predicates
				(on_board ?i - instrument ?s - satellite)
				(pointing ?s - satellite ?d - direction)
				(power_avail ?s - satellite)
				(power_on ?i - instrument)
				(calibrated ?i - instrument)
				(supports ?i - instrument ?m - mode)
			)
			(:task do_observation
				:parameters (?d - image_direction ?m - mode)
			)
			(:task activate_instrument
				:parameters (?s - satellite ?i - instrument)
			)
			(:action turn_to
				:parameters (?s - satellite ?d_new - direction ?d_prev - direction)
				:precondition (and (pointing ?s ?d_prev))
				:effect (and (pointing ?s ?d_new))
			)
			(:action take_image
				:parameters (?s - satellite ?d - image_direction ?i - instrument ?m - mode)
				:precondition (and
					(calibrated ?i)
					(pointing ?s ?d)
					(on_board ?i ?s)
					(power_on ?i)
					(supports ?i ?m)
				)
				:effect (and)
			)
		)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_observation", ("IMAGE_DIRECTION", "MODE"), False, ("have_image",)),
			HTNTask("activate_instrument", ("SATELLITE", "INSTRUMENT"), False, ("calibrated",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="method0",
				task_name="do_observation",
				parameters=("IMAGE_DIRECTION", "MODE", "SATELLITE", "INSTRUMENT", "DIRECTION"),
				task_args=("IMAGE_DIRECTION", "MODE"),
				subtasks=(
					HTNMethodStep("task0", "activate_instrument", ("SATELLITE", "INSTRUMENT"), "compound"),
					HTNMethodStep("task1", "turn_to", ("SATELLITE", "IMAGE_DIRECTION", "DIRECTION"), "primitive"),
					HTNMethodStep("task2", "take_image", ("SATELLITE", "IMAGE_DIRECTION", "INSTRUMENT", "MODE"), "primitive"),
				),
				ordering=(("task0", "task1"), ("task1", "task2")),
			),
			HTNMethod(
				method_name="method5",
				task_name="activate_instrument",
				parameters=("SATELLITE", "INSTRUMENT"),
				task_args=("SATELLITE", "INSTRUMENT"),
				context=(
					HTNLiteral("on_board", ("INSTRUMENT", "SATELLITE"), True, None),
					HTNLiteral("power_avail", ("SATELLITE",), True, None),
				),
				subtasks=(
					HTNMethodStep("task0", "switch_on", ("INSTRUMENT", "SATELLITE"), "primitive"),
				),
			),
		],
	)
	renderer = AgentSpeakRenderer()

	code = renderer.generate(
		domain=domain,
		objects=("satellite0", "instrument0", "star0", "star5", "image1"),
		typed_objects=(
			("satellite0", "satellite"),
			("instrument0", "instrument"),
			("star0", "direction"),
			("star5", "image_direction"),
			("image1", "mode"),
		),
		method_library=method_library,
		plan_records=[],
	)

	assert "+!do_observation(IMAGE_DIRECTION, MODE) : " in code
	assert "pointing(SATELLITE, DIRECTION)" not in code

def test_renderer_stops_recursive_first_child_specialisation_after_child_primitive_prefix(tmp_path):
	domain = _parse_domain_text(
		tmp_path,
		"satellite-recursive-boundary.hddl",
		"""
		(define (domain satellite_recursive_boundary)
			(:requirements :typing :hierarchy)
			(:types satellite instrument direction image_direction mode)
			(:predicates
				(on_board ?i - instrument ?s - satellite)
				(pointing ?s - satellite ?d - direction)
				(power_avail ?s - satellite)
				(power_on ?i - instrument)
				(calibrated ?i - instrument)
				(calibration_target ?i - instrument ?d - direction)
				(supports ?i - instrument ?m - mode)
			)
			(:task do_observation
				:parameters (?d - image_direction ?m - mode)
			)
			(:task activate_instrument
				:parameters (?s - satellite ?i - instrument)
			)
			(:task auto_calibrate
				:parameters (?s - satellite ?i - instrument)
			)
			(:action switch_on
				:parameters (?i - instrument ?s - satellite)
				:precondition (and (on_board ?i ?s) (power_avail ?s))
				:effect (and (power_on ?i))
			)
			(:action turn_to
				:parameters (?s - satellite ?d_new - direction ?d_prev - direction)
				:precondition (and (pointing ?s ?d_prev))
				:effect (and (pointing ?s ?d_new))
			)
			(:action calibrate
				:parameters (?s - satellite ?i - instrument ?d - direction)
				:precondition (and
					(on_board ?i ?s)
					(calibration_target ?i ?d)
					(pointing ?s ?d)
					(power_on ?i)
				)
				:effect (and (calibrated ?i))
			)
			(:action take_image
				:parameters (?s - satellite ?d - image_direction ?i - instrument ?m - mode)
				:precondition (and
					(on_board ?i ?s)
					(power_on ?i)
					(calibrated ?i)
					(pointing ?s ?d)
					(supports ?i ?m)
				)
				:effect (and)
			)
		)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_observation", ("IMAGE_DIRECTION", "MODE"), False, ()),
			HTNTask("activate_instrument", ("SATELLITE", "INSTRUMENT"), False, ()),
			HTNTask("auto_calibrate", ("SATELLITE", "INSTRUMENT"), False, ()),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_do_observation",
				task_name="do_observation",
				parameters=("SATELLITE", "INSTRUMENT", "IMAGE_DIRECTION", "MODE", "DIRECTION"),
				task_args=("IMAGE_DIRECTION", "MODE"),
				subtasks=(
					HTNMethodStep("task0", "activate_instrument", ("SATELLITE", "INSTRUMENT"), "compound"),
					HTNMethodStep(
						"task1",
						"turn_to",
						("SATELLITE", "IMAGE_DIRECTION", "DIRECTION"),
						"primitive",
						"turn_to",
					),
					HTNMethodStep(
						"task2",
						"take_image",
						("SATELLITE", "IMAGE_DIRECTION", "INSTRUMENT", "MODE"),
						"primitive",
						"take_image",
					),
				),
				ordering=(("task0", "task1"), ("task1", "task2")),
			),
			HTNMethod(
				method_name="m_activate_instrument",
				task_name="activate_instrument",
				parameters=("SATELLITE", "INSTRUMENT"),
				task_args=("SATELLITE", "INSTRUMENT"),
				subtasks=(
					HTNMethodStep(
						"task0",
						"switch_on",
						("INSTRUMENT", "SATELLITE"),
						"primitive",
						"switch_on",
					),
					HTNMethodStep(
						"task1",
						"auto_calibrate",
						("SATELLITE", "INSTRUMENT"),
						"compound",
					),
				),
				ordering=(("task0", "task1"),),
			),
			HTNMethod(
				method_name="m_auto_calibrate_direct",
				task_name="auto_calibrate",
				parameters=("SATELLITE", "INSTRUMENT", "CALIB_DIRECTION"),
				task_args=("SATELLITE", "INSTRUMENT"),
				subtasks=(
					HTNMethodStep(
						"task0",
						"calibrate",
						("SATELLITE", "INSTRUMENT", "CALIB_DIRECTION"),
						"primitive",
						"calibrate",
					),
				),
			),
			HTNMethod(
				method_name="m_auto_calibrate_turn_then_calibrate",
				task_name="auto_calibrate",
				parameters=("SATELLITE", "INSTRUMENT", "CALIB_DIRECTION", "DIRECTION"),
				task_args=("SATELLITE", "INSTRUMENT"),
				subtasks=(
					HTNMethodStep(
						"task0",
						"turn_to",
						("SATELLITE", "CALIB_DIRECTION", "DIRECTION"),
						"primitive",
						"turn_to",
					),
					HTNMethodStep(
						"task1",
						"calibrate",
						("SATELLITE", "INSTRUMENT", "CALIB_DIRECTION"),
						"primitive",
						"calibrate",
					),
				),
				ordering=(("task0", "task1"),),
			),
		],
	)
	renderer = AgentSpeakRenderer()

	code = renderer.generate(
		domain=domain,
		objects=("satellite0", "instrument0", "star0", "star1", "image0", "mode0"),
		typed_objects=(
			("satellite0", "satellite"),
			("instrument0", "instrument"),
			("star0", "direction"),
			("star1", "image_direction"),
			("image0", "image_direction"),
			("mode0", "mode"),
		),
		method_library=method_library,
		plan_records=[],
	)

	do_observation_lines = [
		line for line in code.splitlines() if line.startswith("+!do_observation(")
	]
	assert all("calibration_target(" not in line for line in do_observation_lines)

	activate_instrument_lines = [
		line for line in code.splitlines() if line.startswith("+!activate_instrument(")
	]
	assert any("calibration_target(" in line for line in activate_instrument_lines)

def test_renderer_keeps_static_later_step_preconditions_after_prior_compound_step(tmp_path):
	domain = _parse_domain_text(
		tmp_path,
		"transport-mini.hddl",
		"""
		(define (domain transport_mini)
			(:requirements :typing :hierarchy)
			(:types vehicle location)
			(:predicates
				(at ?v - vehicle ?l - location)
				(road ?l1 - location ?l2 - location)
			)
			(:task get_to
				:parameters (?v - vehicle ?target - location)
			)
			(:task helper_at
				:parameters (?v - vehicle ?mid - location)
			)
			(:action drive
				:parameters (?v - vehicle ?mid - location ?target - location)
				:precondition (and (at ?v ?mid) (road ?mid ?target))
				:effect (and (at ?v ?target) (not (at ?v ?mid)))
			)
			(:action noop
				:parameters (?v - vehicle ?mid - location)
				:precondition (and (at ?v ?mid))
				:effect (and)
			)
		)
		""",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("get_to", ("VEHICLE", "TARGET"), False),
			HTNTask("helper_at", ("VEHICLE", "MID"), False),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="helper-at-noop",
				task_name="helper_at",
				parameters=("VEHICLE", "MID"),
				task_args=("VEHICLE", "MID"),
				context=(
					HTNLiteral("at", ("VEHICLE", "MID"), True, None),
				),
				subtasks=(
					HTNMethodStep("s1", "noop", ("VEHICLE", "MID"), "primitive"),
				),
			),
			HTNMethod(
				method_name="get-to-via",
				task_name="get_to",
				parameters=("VEHICLE", "TARGET", "MID"),
				task_args=("VEHICLE", "TARGET"),
				subtasks=(
					HTNMethodStep("s1", "helper_at", ("VEHICLE", "MID"), "compound"),
					HTNMethodStep("s2", "drive", ("VEHICLE", "MID", "TARGET"), "primitive"),
				),
				ordering=(("s1", "s2"),),
			),
		],
	)
	renderer = AgentSpeakRenderer()

	code = renderer.generate(
		domain=domain,
		objects=("truck0", "loc0", "loc1"),
		typed_objects=(
			("truck0", "vehicle"),
			("loc0", "location"),
			("loc1", "location"),
		),
		method_library=method_library,
		plan_records=[],
	)

	rendered = next(
		line
		for line in code.splitlines()
		if line.startswith("+!get_to(")
	)

	assert "road(LOCATION2, LOCATION1)" in rendered

def test_renderer_effect_predicate_closure_handles_long_linear_task_chain():
	renderer = AgentSpeakRenderer()
	chain_length = 1100
	methods_by_task = {}
	render_specs = {}
	for index in range(chain_length + 1):
		task_name = f"task_{index}"
		render_specs[task_name] = {
			"task_param_types": ("BLOCK",),
			"predicate_types": {"done": ("BLOCK",)},
		}
		if index == chain_length:
			methods_by_task[task_name] = [
				HTNMethod(
					method_name=f"m_{task_name}_finish",
					task_name=task_name,
					parameters=("X",),
					task_args=("X",),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="finish",
							args=("X",),
							kind="primitive",
							effects=(HTNLiteral("done", ("X",), True, None),),
						),
					),
				),
			]
			continue
		child_task_name = f"task_{index + 1}"
		methods_by_task[task_name] = [
			HTNMethod(
				method_name=f"m_{task_name}_chain",
				task_name=task_name,
				parameters=("X",),
				task_args=("X",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name=child_task_name,
						args=("X",),
						kind="compound",
					),
				),
			),
		]
	for spec in render_specs.values():
		spec["task_render_specs"] = render_specs

	root_step = HTNMethodStep(
		step_id="root",
		task_name="task_0",
		args=("X",),
		kind="compound",
	)
	effect_predicate_cache = {}

	predicates = renderer._step_effect_predicates(
		root_step,
		methods_by_task,
		render_specs["task_0"],
		effect_predicate_cache,
	)

	assert predicates == ("done",)
	assert effect_predicate_cache["task_0"] == ("done",)

def test_renderer_decomposing_method_plan_uses_explicit_context_only():
	class GuardedRenderer(AgentSpeakRenderer):
		def _method_context_literals(self, *args, **kwargs):
			raise AssertionError("decomposing method rendering must not expand context")

	renderer = GuardedRenderer()
	method = HTNMethod(
		method_name="m_decomposition",
		task_name="achieve",
		parameters=("X",),
		task_args=("X",),
		context=(HTNLiteral("ready", ("X",), True, None),),
		subtasks=tuple(
			HTNMethodStep(
				step_id=f"s{index}",
				task_name="helper",
				args=("X",),
				kind="compound",
			)
			for index in range(2)
		),
	)
	task_lookup = {
		"achieve": HTNTask(
			name="achieve",
			parameters=("X",),
			is_primitive=False,
			source_predicates=("done",),
		),
		"helper": HTNTask(
			name="helper",
			parameters=("X",),
			is_primitive=False,
			source_predicates=("ready",),
		),
	}

	lines = renderer._render_method_plan(
		method,
		task_lookup,
		{},
		{"achieve": {"trigger_args": ("X",), "task_param_types": ("OBJECT",)}},
		{},
		{},
	)

	assert any("+!achieve(X) : ready(X) <-" in line for line in lines)

def test_renderer_large_method_plan_uses_simple_variable_map():
	class GuardedRenderer(AgentSpeakRenderer):
		def _infer_method_variable_types(self, *args, **kwargs):
			raise AssertionError("large method rendering must not infer variable types")

	renderer = GuardedRenderer()
	step_count = renderer._VARIABLE_MAP_INFERENCE_STEP_LIMIT + 1
	method = HTNMethod(
		method_name="m_large_decomposition",
		task_name="achieve",
		parameters=("X",),
		task_args=("X",),
		subtasks=tuple(
			HTNMethodStep(
				step_id=f"s{index}",
				task_name="helper",
				args=("X", "AUX"),
				kind="compound",
			)
			for index in range(step_count)
		),
	)
	task_lookup = {
		"achieve": HTNTask(
			name="achieve",
			parameters=("X",),
			is_primitive=False,
			source_predicates=("done",),
		),
		"helper": HTNTask(
			name="helper",
			parameters=("X", "AUX"),
			is_primitive=False,
			source_predicates=("ready",),
		),
	}

	lines = renderer._render_method_plan(
		method,
		task_lookup,
		{},
		{"achieve": {"trigger_args": ("X",), "task_param_types": ("OBJECT",)}},
		{},
		{},
	)

	assert any("!helper(X, AUX)" in line for line in lines)

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

def test_renderer_ignores_synthetic_witness_prompt_groundings():
	renderer = AgentSpeakRenderer()
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("deliver", ("PACKAGE", "LOCATION"), False, ("at",))],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_deliver",
				task_name="deliver",
				parameters=("PACKAGE", "LOCATION", "VEHICLE"),
				task_args=("PACKAGE", "LOCATION"),
				subtasks=(HTNMethodStep("s1", "helper_in", ("PACKAGE", "VEHICLE"), "compound"),),
				ordering=(),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)
	code = renderer.generate(
		domain=_transport_domain(),
		objects=("package-0", "city-loc-1"),
		typed_objects=(
			("package-0", "package"),
			("city-loc-1", "location"),
		),
		method_library=method_library,
		plan_records=[],
		prompt_analysis={
			"transition_tasks": [
				{
					"name": "deliver",
					"parameters": ["PACKAGE", "LOCATION"],
					"context_parameters": ["VEHICLE"],
					"grounded_symbol_map": {
						"package-0": "PACKAGE",
						"city-loc-1": "LOCATION",
						"witness_vehicle_1": "VEHICLE",
					},
				},
			],
		},
	)

	assert "witness_vehicle_1" not in code

def test_renderer_simple_variable_map_lifts_lowercase_witness_parameters_into_asl_variables():
	renderer = AgentSpeakRenderer()
	renderer._VARIABLE_MAP_INFERENCE_STEP_LIMIT = 0
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("unload", ("VEHICLE", "LOCATION", "PACKAGE"), False, ("at",))],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m-unload",
				task_name="unload",
				parameters=(
					"VEHICLE",
					"LOCATION",
					"PACKAGE",
					"witness_vehicle_1",
					"witness_capacity-number_1",
				),
				task_args=("VEHICLE", "LOCATION", "PACKAGE"),
				context=(
					HTNLiteral("at", ("witness_vehicle_1", "LOCATION"), True, None),
					HTNLiteral("in", ("PACKAGE", "witness_vehicle_1"), True, None),
					HTNLiteral("capacity", ("witness_vehicle_1", "witness_capacity-number_1"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						"s1",
						"drop",
						("witness_vehicle_1", "LOCATION", "PACKAGE", "witness_capacity-number_1"),
						"primitive",
						action_name="drop",
					),
				),
				ordering=(),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)
	code = renderer.generate(
		domain=_transport_domain(),
		objects=("truck-0", "city-loc-0", "package-0"),
		typed_objects=(
			("truck-0", "vehicle"),
			("city-loc-0", "location"),
			("package-0", "package"),
		),
		method_library=method_library,
		plan_records=[],
	)

	assert "witness_vehicle_1" not in code
	assert "witness_capacity-number_1" not in code
	assert "WITNESS_VEHICLE_1" in code
	assert "WITNESS_CAPACITY_NUMBER_1" in code

def test_renderer_narrows_task_bound_variable_types_from_branch_action_usage():
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
				HTNTask("helper_at", ("LOCATABLE", "LOCATION"), False, ("at",)),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_helper_at_noop",
					task_name="helper_at",
					parameters=("LOCATABLE", "LOCATION"),
					task_args=("LOCATABLE", "LOCATION"),
					context=(HTNLiteral("at", ("LOCATABLE", "LOCATION"), True, None),),
				),
				HTNMethod(
					method_name="m_helper_at_constructive_1",
					task_name="helper_at",
					parameters=("LOCATABLE", "SOURCE", "LOCATION"),
					task_args=("LOCATABLE", "LOCATION"),
					context=(HTNLiteral("road", ("SOURCE", "LOCATION"), True, None),),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="drive",
							args=("LOCATABLE", "SOURCE", "LOCATION"),
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

	constructive_header = next(
		line
		for line in code.splitlines()
		if line.startswith("+!helper_at(") and "road(" in line
	)
	assert "object_type(LOCATABLE, vehicle)" in constructive_header
	assert "object_type(LOCATABLE, locatable)" not in constructive_header

def test_renderer_emits_type_guards_for_body_only_action_witnesses():
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
				HTNTask("helper_in", ("PACKAGE", "VEHICLE"), False, ("in",)),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m_helper_in_constructive_1",
					task_name="helper_in",
					parameters=("PACKAGE", "VEHICLE", "LOCATION", "CAPACITY1", "CAPACITY2"),
					task_args=("PACKAGE", "VEHICLE"),
					context=(HTNLiteral("capacity_predecessor", ("CAPACITY1", "CAPACITY2"), True, None),),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="helper_at",
							args=("VEHICLE", "LOCATION"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s2",
							task_name="helper_at",
							args=("PACKAGE", "LOCATION"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s3",
							task_name="pick_up",
							args=("VEHICLE", "LOCATION", "PACKAGE", "CAPACITY1", "CAPACITY2"),
							kind="primitive",
							action_name="pick_up",
						),
					),
				),
			],
			target_literals=[],
			target_task_bindings=[],
		),
		plan_records=[],
	)

	constructive_header = next(
		line
		for line in code.splitlines()
		if line.startswith("+!helper_in(") and "capacity_predecessor(" in line
	)
	assert "object_type(LOCATION, location)" in constructive_header

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
