"""
Focused tests for Stage 5 AgentSpeak rendering.
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
	HTNTargetTaskBinding,
	HTNTask,
)
from stage4_panda_planning.panda_schema import PANDAPlanResult
from stage5_agentspeak_rendering.agentspeak_renderer import AgentSpeakRenderer
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
				name="pick_up_from_table",
				parameters=("B1",),
				is_primitive=True,
			),
			HTNTask(
				name="put_on_block",
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
						task_name="put_on_block",
						args=("B1", "B2"),
						kind="primitive",
						action_name="put-on-block",
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
						task_name="pick_up_from_table",
						args=("B1",),
						kind="primitive",
						action_name="pick-up-from-table",
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
	assert "+!place_on(BLOCK1, BLOCK2) : clear(BLOCK2) <-" in code
	assert "\t!hold_block(BLOCK1);" in code
	assert "\t!put_on_block(BLOCK1, BLOCK2)." in code
	assert "+!hold_block(BLOCK) : ontable(BLOCK) & clear(BLOCK) <-" in code
	assert "\t!pick_up_from_table(BLOCK)." in code
	assert "+!dfa_step_q1_q2_on_a_b : dfa_state(q1) <-" in code
	assert "\t!place_on(a, b);" in code
	assert "\t-dfa_state(q1);" in code
	assert "\t+dfa_state(q2)." in code
	assert "+!run_dfa : dfa_state(q1) <-" in code
	assert "\t!dfa_step_q1_q2_on_a_b;" in code
	assert "\t!run_dfa." in code
	assert "+!run_dfa : dfa_state(q2) & accepting_state(q2) <-" in code
	assert "+!put_on_block(BLOCK1, BLOCK2) :" in code


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
						action_name="pick-up",
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
						action_name="pick-up-from-table",
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
						action_name="pick-up",
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
