from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from stage4_panda_planning import panda_planner as panda_planner_module
from stage6_jason_validation.jason_runner import JasonRunner, JasonValidationError

OFFICIAL_BLOCKSWORLD_P01 = (
	Path(__file__).parent.parent.parent / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"
)

def _sample_action_schemas():
	return [
		{
			"functor": "pick_up",
			"source_name": "pick-up",
			"parameters": ["?x", "?y"],
			"preconditions": [
				{"predicate": "clear", "args": ["?x"], "is_positive": True},
				{"predicate": "on", "args": ["?x", "?y"], "is_positive": True},
			],
			"precondition_clauses": [
				[
					{"predicate": "clear", "args": ["?x"], "is_positive": True},
					{"predicate": "on", "args": ["?x", "?y"], "is_positive": True},
				],
			],
			"effects": [
				{"predicate": "holding", "args": ["?x"], "is_positive": True},
				{"predicate": "on", "args": ["?x", "?y"], "is_positive": False},
			],
		},
	]

def test_hddl_fact_to_atom_ignores_negative_and_equality():
	runner = JasonRunner()
	assert runner._hddl_fact_to_atom("(on a b)") == "on(a,b)"
	assert runner._hddl_fact_to_atom("(handempty)") == "handempty"
	assert runner._hddl_fact_to_atom("(not (on a b))") is None
	assert runner._hddl_fact_to_atom("(= a b)") is None

def test_hddl_fact_to_atom_quotes_unsafe_constants():
	runner = JasonRunner()
	assert runner._hddl_fact_to_atom("(pointing satellite0 Phenomenon6)") == (
		'pointing(satellite0,"Phenomenon6")'
	)

def test_hddl_fact_to_atom_sanitizes_hyphenated_predicates():
	runner = JasonRunner()

	assert runner._hddl_fact_to_atom("(capacity-predecessor capacity-0 capacity-1)") == (
		'capacity_predecessor("capacity-0","capacity-1")'
	)

def test_runtime_world_to_hddl_facts_restores_source_predicate_names_and_tokens():
	runner = JasonRunner()

	facts = runner._runtime_world_to_hddl_facts(
		[
			'capacity_predecessor("capacity-0","capacity-1")',
			'at("package-0","city-loc-4")',
		],
		predicate_name_map={
			"capacity_predecessor": "capacity-predecessor",
			"at": "at",
		},
	)

	assert facts == (
		"(at package-0 city-loc-4)",
		"(capacity-predecessor capacity-0 capacity-1)",
	)

def test_strip_seed_fact_beliefs_removes_runtime_world_atoms_but_keeps_metadata():
	runner = JasonRunner()
	agentspeak_code = """
/* Initial Beliefs */
domain(transport).
at("truck-0", "city-loc-2").
capacity_predecessor("capacity-0", "capacity-1").
object("truck-0").

/* Primitive Action Plans */
""".strip() + "\n"

	rewritten = runner._strip_seed_fact_beliefs(
		agentspeak_code,
		seed_facts=[
			"(at truck-0 city-loc-2)",
			"(capacity-predecessor capacity-0 capacity-1)",
		],
	)

	assert 'at("truck-0", "city-loc-2").' not in rewritten
	assert 'capacity_predecessor("capacity-0", "capacity-1").' not in rewritten
	assert "domain(transport)." in rewritten
	assert 'object("truck-0").' in rewritten

def test_runner_asl_includes_accepting_and_target_validation_without_manual_seeding():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q2).",
			],
		),
		[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="clear", args=("b",), is_positive=False, source_symbol=None),
		],
	)

	assert "+!verify_targets : on(a, b) & not clear(b) <-" in asl
	assert "?dfa_state(FINAL_STATE)" in asl
	assert "?accepting_state(FINAL_STATE)" in asl
	assert "!execute." in asl
	assert "+!reset_execution_state : dfa_state(CURRENT_STATE) <-" not in asl
	assert "+!execute_round_1 :" not in asl
	assert "\t!run_dfa;" in asl
	assert "+on(a" not in asl

def test_runner_asl_compiles_ordered_progress_chain_without_dfa_round_search():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q4).",
				'dfa_edge_label(dfa_t1, "goal(a)").',
				'dfa_edge_label(dfa_t2, "goal(b)").',
				'dfa_edge_label(dfa_t3, "goal(c)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_t1 : dfa_state(q1) <-",
				"\t!task_a;",
				"\t-dfa_state(q1);",
				"\t+dfa_state(q2).",
				"",
				"+!dfa_t2 : dfa_state(q2) <-",
				"\t!task_b;",
				"\t-dfa_state(q2);",
				"\t+dfa_state(q3).",
				"",
				"+!dfa_t3 : dfa_state(q3) <-",
				"\t!task_c;",
				"\t-dfa_state(q3);",
				"\t+dfa_state(q4).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : dfa_state(q4) & accepting_state(q4) <-",
				"\ttrue.",
				"",
				"/* Primitive Action Plans */",
				"+!task_a : true <-",
				"\ttrue.",
				"+!task_b : true <-",
				"\ttrue.",
				"+!task_c : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="goal", args=("c",), is_positive=True, source_symbol=None),
		],
	)

	assert "+!verify_targets : target_seen(t1) & target_seen(t2) & target_seen(t3) <-" in asl
	assert "+!execute_round_1 :" not in asl
	assert "+!reset_execution_state :" not in asl
	assert "+!execute_progress_chain_1 : true <-" in asl
	execution_section = asl.split("/* Execution Entry */", 1)[1]
	assert "\t!execute_progress_chain_1." in execution_section
	assert "\t!dfa_t1;" in execution_section
	assert "\t!dfa_t2;" in execution_section
	assert "\t!dfa_t3;" in execution_section
	assert "\t!run_dfa;" not in execution_section
	assert "?accepting_state(FINAL_STATE)" in execution_section

def test_runner_asl_guards_noop_only_for_unmet_progress_transition():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="do_on_table",
				parameters=("BLOCK1",),
				is_primitive=False,
				source_predicates=("ontable",),
			),
		],
		methods=[
			HTNMethod(
				method_name="m3_do_on_table",
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
			),
			HTNMethod(
				method_name="m2_do_on_table",
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
			),
		],
	)
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q2).",
				'dfa_edge_label(dfa_t1, "ontable(b26)").',
				"",
				"/* HTN Method Plans */",
				"+!do_on_table(BLOCK1) : clear(BLOCK1) <-",
				'\t.print("runtime trace method flat ", "m3_do_on_table", "|", BLOCK1);',
				"\t!nop.",
				"",
				"+!do_on_table(BLOCK1) : clear(BLOCK1) & handempty & not ontable(BLOCK1) & on(BLOCK1, BLOCK2) <-",
				'\t.print("runtime trace method flat ", "m2_do_on_table", "|", BLOCK1);',
				"\t!unstack(BLOCK1, BLOCK2);",
				"\t!put_down(BLOCK1).",
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_t1 : dfa_state(q1) <-",
				"\t!do_on_table(b26);",
				"\t-dfa_state(q1);",
				"\t+dfa_state(q2).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : dfa_state(q2) & accepting_state(q2) <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="ontable", args=("b26",), is_positive=True, source_symbol=None),
		],
		method_library=method_library,
	)

	assert "+!do_on_table(BLOCK1) : clear(BLOCK1) & ontable(BLOCK1) <-" in asl
	assert "+!dfa_t1 : dfa_state(q1) & ontable(b26) <-" in asl
	assert "+!dfa_t1 : dfa_state(q1) & not ontable(b26) <-" in asl
	assert "\t+progress_target_1(do_on_table, ontable, b26);" in asl
	assert "\t-do_on_table" not in asl
	assert "\t-progress_target_1(do_on_table, ontable, b26);" in asl

def test_runner_asl_restricts_unsafe_noop_to_prefix_protected_targets():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="do_on_table",
				parameters=("BLOCK1",),
				is_primitive=False,
				source_predicates=("ontable",),
			),
		],
		methods=[
			HTNMethod(
				method_name="m3_do_on_table",
				task_name="do_on_table",
				parameters=("BLOCK1",),
				task_args=("BLOCK1",),
				context=(
					HTNLiteral(
						predicate="clear",
						args=("BLOCK1",),
						is_positive=True,
						source_symbol=None,
					),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="nop",
						args=(),
						kind="primitive",
						action_name="nop",
					),
				),
			),
		],
	)

	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				'dfa_edge_label(dfa_t1, "on(b6, b3)").',
				"",
				"/* HTN Method Plans */",
				"+!do_on_table(BLOCK1) : clear(BLOCK1) <-",
				"\t!nop.",
				"",
				"/* DFA Transition Wrappers */",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="on", args=("b6", "b3"), is_positive=True, source_symbol=None),
		],
		protected_target_literals=[
			HTNLiteral(predicate="on", args=("b6", "b3"), is_positive=True, source_symbol=None),
		],
		method_library=method_library,
	)

	assert "+!do_on_table(BLOCK1) : clear(BLOCK1) & ontable(BLOCK1) <-" in asl
	assert "protected_target_on(BLOCK1, b3)" in asl
	assert "protected_target_on(b6, BLOCK1)" not in asl
	assert "+!do_on_table(BLOCK1) : clear(BLOCK1) <-" not in asl

def test_runner_asl_does_not_allow_clear_target_to_short_circuit_do_on_table():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="do_on_table",
				parameters=("BLOCK1",),
				is_primitive=False,
				source_predicates=("ontable",),
			),
		],
		methods=[
			HTNMethod(
				method_name="m3_do_on_table",
				task_name="do_on_table",
				parameters=("BLOCK1",),
				task_args=("BLOCK1",),
				context=(
					HTNLiteral(
						predicate="clear",
						args=("BLOCK1",),
						is_positive=True,
						source_symbol=None,
					),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="nop",
						args=(),
						kind="primitive",
						action_name="nop",
					),
				),
			),
		],
	)

	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"",
				"/* HTN Method Plans */",
				"+!do_on_table(BLOCK1) : clear(BLOCK1) <-",
				"\t!nop.",
				"",
				"/* DFA Transition Wrappers */",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="clear", args=("b2",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="ontable", args=("b9",), is_positive=True, source_symbol=None),
		],
		method_library=method_library,
	)

	assert "+!do_on_table(BLOCK1) : clear(BLOCK1) & ontable(BLOCK1) <-" in asl
	assert "protected_target_clear(" not in asl
	assert "+!do_on_table(BLOCK1) : clear(BLOCK1) <-" not in asl

def test_runner_asl_uses_projected_goal_protection_for_grounded_runtime_noop():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="do_on_table",
				parameters=("BLOCK1",),
				is_primitive=False,
				source_predicates=("ontable",),
			),
		],
		methods=[
			HTNMethod(
				method_name="m3_do_on_table",
				task_name="do_on_table",
				parameters=("BLOCK1",),
				task_args=("BLOCK1",),
				context=(
					HTNLiteral(
						predicate="clear",
						args=("BLOCK1",),
						is_positive=True,
						source_symbol=None,
					),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="nop",
						args=(),
						kind="primitive",
						action_name="nop",
					),
				),
			),
		],
	)

	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				'dfa_edge_label(dfa_t1, "clear(b2)").',
				'dfa_edge_label(dfa_t2, "on(b15, b9)").',
				"",
				"/* HTN Method Plans */",
				"+!do_on_table(b15) : clear(b15) <-",
				"\t!nop.",
				"",
				"/* DFA Transition Wrappers */",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="clear", args=("b2",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="on", args=("b15", "b9"), is_positive=True, source_symbol=None),
		],
		protected_target_literals=[
			HTNLiteral(predicate="on", args=("b15", "b9"), is_positive=True, source_symbol=None),
		],
		method_library=method_library,
	)

	assert "protected_target_clear(" not in asl
	assert "+!mark_target_t1 : clear(b2) <-" in asl
	assert "+!mark_target_t2 : on(b15, b9) & protected_target_on(b15, b9) <-" in asl
	assert "+!do_on_table(b15) : clear(b15) & protected_target_on(b15, b9) <-" in asl

def test_runner_asl_preserves_head_variables_in_projected_goal_protection_contexts():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="do_on_table",
				parameters=("BLOCK1",),
				is_primitive=False,
				source_predicates=("ontable",),
			),
		],
		methods=[
			HTNMethod(
				method_name="m3_do_on_table",
				task_name="do_on_table",
				parameters=("BLOCK1",),
				task_args=("BLOCK1",),
				context=(
					HTNLiteral(
						predicate="clear",
						args=("BLOCK1",),
						is_positive=True,
						source_symbol=None,
					),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="nop",
						args=(),
						kind="primitive",
						action_name="nop",
					),
				),
			),
		],
	)

	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				'dfa_edge_label(dfa_t1, "on(b15, b9)").',
				"",
				"/* HTN Method Plans */",
				"+!do_on_table(BLOCK1) : clear(BLOCK1) <-",
				"\t!nop.",
				"",
				"/* DFA Transition Wrappers */",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="on", args=("b15", "b9"), is_positive=True, source_symbol=None),
		],
		protected_target_literals=[
			HTNLiteral(predicate="on", args=("b15", "b9"), is_positive=True, source_symbol=None),
		],
		method_library=method_library,
	)

	assert 'protected_target_on("BLOCK1", b9)' not in asl
	assert "protected_target_on(BLOCK1, b9)" in asl
	assert "protected_target_on(b15, BLOCK1)" not in asl

def test_runner_asl_keeps_rendered_noop_context_without_protected_target_expansion():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="get_to",
				parameters=("VEHICLE", "LOCATION1"),
				is_primitive=False,
				source_predicates=("at",),
			),
		],
		methods=[
			HTNMethod(
				method_name="m-i-am-there",
				task_name="get_to",
				parameters=("VEHICLE", "LOCATION1"),
				task_args=("VEHICLE", "LOCATION1"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="nop",
						args=(),
						kind="primitive",
						action_name="nop",
					),
				),
			),
		],
	)

	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"",
				"/* HTN Method Plans */",
				"+!get_to(VEHICLE, LOCATION1) : at(VEHICLE, LOCATION1) <-",
				'\t.print("runtime trace method flat ", "m-i-am-there", "|", VEHICLE, "|", LOCATION1);',
				"\t!nop.",
				"",
				"/* DFA Transition Wrappers */",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(
				predicate="at",
				args=("package-0", "city-loc-0"),
				is_positive=True,
				source_symbol=None,
			),
		],
		method_library=method_library,
	)

	assert "+!get_to(VEHICLE, LOCATION1) : at(VEHICLE, LOCATION1) <-" in asl
	assert "protected_target_at(" not in asl
	assert asl.count("+!get_to(") == 1

def test_runner_asl_keeps_noop_safe_when_source_predicate_uses_subset_of_task_args():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="empty_store",
				parameters=("STORE", "ROVER"),
				is_primitive=False,
				source_predicates=("empty",),
			),
		],
		methods=[
			HTNMethod(
				method_name="m-empty-store-1",
				task_name="empty_store",
				parameters=("STORE", "ROVER"),
				task_args=("STORE", "ROVER"),
				context=(
					HTNLiteral(
						predicate="empty",
						args=("STORE",),
						is_positive=True,
						source_symbol=None,
					),
				),
				subtasks=(),
			),
		],
	)

	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"",
				"/* HTN Method Plans */",
				"+!empty_store(STORE, ROVER) : empty(STORE) <-",
				"\ttrue.",
				"",
				"/* DFA Transition Wrappers */",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(
				predicate="communicated_soil_data",
				args=("waypoint2",),
				is_positive=True,
				source_symbol=None,
			),
		],
		method_library=method_library,
	)

	assert "+!empty_store(STORE, ROVER) : empty(STORE) <-" in asl
	assert "empty(STORE, ROVER)" not in asl
	assert "protected_target_communicated_soil_data(STORE)" not in asl

def test_progress_target_guard_atom_quotes_runtime_terms_with_hyphens():
	assert JasonRunner._progress_target_guard_atom(
		"deliver",
		"at",
		("package-0", "city-loc-0"),
	) == 'progress_target_2(deliver, at, "package-0", "city-loc-0")'

def test_runner_asl_accepting_state_completion_mode_skips_target_rounds():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q2).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : dfa_state(q2) & accepting_state(q2) <-",
				"\ttrue.",
			],
		),
		[],
		completion_mode="accepting_state",
	)

	assert "+!verify_targets :" not in asl
	assert "+!reset_execution_state :" not in asl
	assert "+!execute_round_1 :" not in asl
	assert "+!execute : true <-" in asl
	assert "!run_dfa" in asl
	assert "?dfa_state(FINAL_STATE)" in asl
	assert "?accepting_state(FINAL_STATE)" in asl
	assert '.print("execute success")' in asl
	assert "target_seen(" not in asl
	assert "blocked_target(" not in asl

def test_runner_asl_guided_execution_replays_method_trace_and_actions():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"",
				"/* Primitive Action Plans */",
				"+!pick_up(X) : true <-",
				"\tpick_up(X).",
				"",
				"/* HTN Method Plans */",
				"+!place_on(A, B) : true <-",
				"\ttrue.",
			],
		),
		[],
		guided_action_path=('pick_up("GroundStation2")', 'stack("GroundStation2", b)'),
		guided_method_trace=(
			{"method_name": "m_place_on_stack", "task_args": ["GroundStation2", "b"]},
			{"method_name": "m_hold_block_pickup", "task_args": ["GroundStation2"]},
		),
	)

	assert "runtime trace method" not in asl
	assert '!pick_up("GroundStation2");' in asl
	assert '!stack("GroundStation2", b);' in asl
	assert "+!guided_replay_1 : true <-" in asl
	assert "+!run_dfa" not in asl
	assert "+!verify_targets" not in asl

def test_runner_asl_guided_prefix_continues_with_runtime_goal():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q2).",
				'dfa_edge_label(dfa_step_q1_q2_on_a_b, "on(a, b)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_step_q1_q2_on_a_b : not target_seen(t1) & not blocked_target(t1) <-",
				"\t!place_on(a, b).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : target_seen(t1) <-",
				"\ttrue.",
				"",
				"+!run_dfa : not target_seen(t1) & not blocked_target(t1) <-",
				"\t!dfa_step_q1_q2_on_a_b;",
				"\t!clear_blocked_targets;",
				"\t!run_dfa.",
				"",
				"-!dfa_step_q1_q2_on_a_b : not target_seen(t1) <-",
				"\t+blocked_target(t1);",
				"\t!run_dfa.",
				"",
				"/* Primitive Action Plans */",
				"+!pick_up(X) : true <-",
				"\tpick_up(X).",
				"",
				"/* HTN Method Plans */",
				"+!place_on(A, B) : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
		],
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
			],
		),
		ordered_query_sequence=False,
		guided_action_path=('pick_up("GroundStation2")',),
		guided_method_trace=(
			{"method_name": "m_place_on_stack", "task_args": ["GroundStation2", "b"]},
		),
		guided_continue_with_runtime_goal=True,
		guided_completed_target_ids=("t1",),
	)

	assert '.print("runtime trace method flat ", "m_place_on_stack", "|", "GroundStation2", "|", b)' not in asl
	assert "+!guided_prefix_replay_1 : true <-" in asl
	assert '!pick_up("GroundStation2");' in asl
	assert "!mark_target_t1;" in asl
	assert "!run_dfa;" in asl
	assert "!verify_targets;" in asl
	assert "+!verify_targets" in asl

def test_runner_asl_guided_prefix_filters_completed_targets_from_unordered_control_plan():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q4).",
				'dfa_edge_label(dfa_step_q1_q2_goal_a, "goal(a)").',
				'dfa_edge_label(dfa_step_q2_q3_goal_b, "goal(b)").',
				'dfa_edge_label(dfa_step_q3_q4_goal_c, "goal(c)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_step_q1_q2_goal_a : not target_seen(t1) & not blocked_target(t1) <-",
				"\t!task_a.",
				"",
				"+!dfa_step_q2_q3_goal_b : not target_seen(t2) & not blocked_target(t2) <-",
				"\t!task_b.",
				"",
				"+!dfa_step_q3_q4_goal_c : not target_seen(t3) & not blocked_target(t3) <-",
				"\t!task_c.",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : target_seen(t1) & target_seen(t2) & target_seen(t3) <-",
				"\ttrue.",
				"",
				"/* Primitive Action Plans */",
				"+!task_a : true <-",
				"\ttrue.",
				"+!task_b : true <-",
				"\ttrue.",
				"+!task_c : true <-",
				"\ttrue.",
				"",
				"/* HTN Method Plans */",
			],
		),
		[
			HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="goal", args=("c",), is_positive=True, source_symbol=None),
		],
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("dfa_step_q1_q2_goal_a", (), False, ("goal",)),
				HTNTask("dfa_step_q2_q3_goal_b", (), False, ("goal",)),
				HTNTask("dfa_step_q3_q4_goal_c", (), False, ("goal",)),
			],
		),
		ordered_query_sequence=False,
		guided_action_path=("task_a",),
		guided_method_trace=(
			{"method_name": "m_guided_trace", "task_args": []},
		),
		guided_continue_with_runtime_goal=True,
		guided_completed_target_ids=("t1", "t2"),
	)

	start = asl.find("/* DFA Control Plans */")
	assert start != -1
	control_section = asl[start:]
	assert "+!run_dfa : target_seen(t1)" not in control_section
	assert "+!run_dfa : target_seen(t2)" not in control_section
	assert "not target_seen(t1) & not blocked_target(t1)" not in control_section
	assert "not target_seen(t2) & not blocked_target(t2)" not in control_section
	assert "not target_seen(t3)" in control_section
	assert "dfa_step_q3_q4_goal_c" in control_section

def test_runner_asl_guided_prefix_filters_completed_literal_variants():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q4).",
				'dfa_edge_label(dfa_step_q1_q2_goal_a, "goal(a)").',
				'dfa_edge_label(dfa_step_q2_q3_goal_a, "goal(a)").',
				'dfa_edge_label(dfa_step_q3_q4_goal_b, "goal(b)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_step_q1_q2_goal_a : not target_seen(t1) & not blocked_target(t1) <-",
				"\t!task_a.",
				"",
				"+!dfa_step_q2_q3_goal_a : not target_seen(t2) & not blocked_target(t2) <-",
				"\t!task_a.",
				"",
				"+!dfa_step_q3_q4_goal_b : not target_seen(t3) & not blocked_target(t3) <-",
				"\t!task_b.",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : target_seen(t1) & target_seen(t2) & target_seen(t3) <-",
				"\ttrue.",
				"",
				"/* Primitive Action Plans */",
				"+!task_a : true <-",
				"\ttrue.",
				"+!task_b : true <-",
				"\ttrue.",
				"",
				"/* HTN Method Plans */",
			],
		),
		[
			HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
		],
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("dfa_step_q1_q2_goal_a", (), False, ("goal",)),
				HTNTask("dfa_step_q2_q3_goal_a", (), False, ("goal",)),
				HTNTask("dfa_step_q3_q4_goal_b", (), False, ("goal",)),
			],
		),
		ordered_query_sequence=False,
		guided_action_path=("task_a",),
		guided_method_trace=(
			{"method_name": "m_guided_trace", "task_args": []},
		),
		guided_continue_with_runtime_goal=True,
		guided_completed_target_ids=("t1",),
	)

	start = asl.find("/* DFA Control Plans */")
	assert start != -1
	control_section = asl[start:]
	assert "+!run_dfa : not target_seen(t1)" not in control_section
	assert "+!run_dfa : not target_seen(t2)" in control_section
	assert "dfa_step_q3_q4_goal_b" in control_section

def test_runner_asl_guided_prefix_syncs_dfa_state_for_current_transition_style():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q3).",
				'dfa_edge_label(dfa_step_q1_q2_goal_a, "goal(a)").',
				'dfa_edge_label(dfa_step_q2_q3_goal_b, "goal(b)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_step_q1_q2_goal_a : dfa_state(q1) <-",
				"\t!task_a;",
				"\t-dfa_state(q1);",
				"\t+dfa_state(q2).",
				"",
				"+!dfa_step_q2_q3_goal_b : dfa_state(q2) <-",
				"\t!task_b;",
				"\t-dfa_state(q2);",
				"\t+dfa_state(q3).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : dfa_state(q3) & accepting_state(q3) <-",
				"\ttrue.",
				"",
				"+!run_dfa : dfa_state(q1) <-",
				"\t!dfa_step_q1_q2_goal_a;",
				"\t!run_dfa.",
				"",
				"+!run_dfa : dfa_state(q2) <-",
				"\t!dfa_step_q2_q3_goal_b;",
				"\t!run_dfa.",
				"",
				"/* Primitive Action Plans */",
				"+!task_a : true <-",
				"\ttask_a.",
				"",
				"+!task_b : true <-",
				"\ttask_b.",
				"",
				"/* HTN Method Plans */",
				"+!do_goal(A) : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
		],
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("do_goal", ("X",), False, ("goal",)),
			],
		),
		ordered_query_sequence=False,
		guided_action_path=("task_a",),
		guided_method_trace=(
			{"method_name": "m_do_goal", "task_args": ["a"]},
		),
		guided_continue_with_runtime_goal=True,
		guided_completed_target_ids=("t1",),
	)

	assert "!mark_target_t1;" in asl
	assert "!advance_dfa_for_t1;" in asl
	assert "+!advance_dfa_for_t1 : target_seen(t1) & dfa_state(q1) <-" in asl
	assert "\t-dfa_state(q1);" in asl
	assert "\t+dfa_state(q2)." in asl
	assert "+!advance_dfa_for_t1 : true <-" in asl

def test_runner_asl_unordered_target_completion_skips_dfa_round_reset_logic():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q2).",
				'dfa_edge_label(dfa_step_q1_q2_on_a_b, "on(a, b)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_step_q1_q2_on_a_b : not target_seen(t1) & not blocked_target(t1) <-",
				"\t!place_on(a, b).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : target_seen(t1) <-",
				"\ttrue.",
				"",
				"+!run_dfa : not target_seen(t1) & not blocked_target(t1) <-",
				"\t!dfa_step_q1_q2_on_a_b;",
				"\t!clear_blocked_targets;",
				"\t!run_dfa.",
				"",
				"-!dfa_step_q1_q2_on_a_b : not target_seen(t1) <-",
				"\t+blocked_target(t1);",
				"\t!run_dfa.",
			],
		),
		[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
		],
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
			],
		),
		ordered_query_sequence=False,
	)

	assert "+!verify_targets : target_seen(t1) <-" in asl
	assert "+!execute : true <-" in asl
	assert "!run_dfa" in asl
	assert "!verify_targets" in asl
	assert "+!reset_execution_state :" not in asl
	assert "+!execute_round_1 :" not in asl
	assert "?dfa_state(FINAL_STATE)" not in asl
	assert "?accepting_state(FINAL_STATE)" not in asl
	assert "blocked_runtime_goal(run_dfa)" not in asl

def test_runner_asl_rewrites_unordered_current_style_control_to_target_centric_runtime():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q1).",
				'dfa_edge_label(dfa_step_q1_q1_goal_a, "goal(a)").',
				'dfa_edge_label(dfa_step_q1_q1_goal_b, "goal(b)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_step_q1_q1_goal_a : dfa_state(q1) <-",
				"\t!task_a;",
				"\t+dfa_state(q1).",
				"",
				"+!dfa_step_q1_q1_goal_b : dfa_state(q1) <-",
				"\t!task_b;",
				"\t+dfa_state(q1).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : dfa_state(q1) & accepting_state(q1) <-",
				"\ttrue.",
				"",
				"/* Primitive Action Plans */",
				"+!task_a : true <-",
				"\ttask_a.",
				"",
				"+!task_b : true <-",
				"\ttask_b.",
				"",
				"/* HTN Method Plans */",
				"+!do_goal(X) : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
		],
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("do_goal", ("X",), False, ("goal",)),
			],
		),
		ordered_query_sequence=False,
	)

	assert "+!run_dfa : target_seen(t1) & target_seen(t2) <-" in asl
	assert "+!clear_blocked_targets : blocked_target(TARGET_ID) <-" in asl
	assert "+!run_dfa : not target_seen(t1) & not blocked_target(t1) <-" in asl
	assert "!advance_target_t1;" in asl
	assert "+!advance_target_t1 : dfa_state(q1) <-" in asl
	assert "\t!dfa_step_q1_q1_goal_a." in asl
	assert "-!advance_target_t1 : not target_seen(t1) <-" in asl
	assert "+!run_dfa : dfa_state(q1) & accepting_state(q1) <-" not in asl

def test_runner_asl_unordered_control_blocks_failed_transition_not_whole_target():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q3).",
				'dfa_edge_label(dfa_t1, "goal(a)").',
				'dfa_edge_label(dfa_t2, "goal(a)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_t1 : dfa_state(q1) <-",
				"\t!do_goal(a);",
				"\t-dfa_state(q1);",
				"\t+dfa_state(q2).",
				"",
				"+!dfa_t2 : dfa_state(q2) <-",
				"\t!do_goal(a);",
				"\t-dfa_state(q2);",
				"\t+dfa_state(q3).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : dfa_state(q3) & accepting_state(q3) <-",
				"\ttrue.",
				"",
				"/* Primitive Action Plans */",
				"+!do_goal(X) : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
		],
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("do_goal", ("X",), False, ("goal",)),
			],
		),
		ordered_query_sequence=False,
	)

	control_section = asl.split("/* DFA Control Plans */", 1)[1].split(
		"/* Target Observation Plans */",
		1,
	)[0]
	failure_section = asl.split("/* Failure Handlers */", 1)[1].split(
		"/* Execution Entry */",
		1,
	)[0]
	assert "not blocked_target(t1)" in control_section
	assert "+!advance_target_t1 : dfa_state(q1) <-" in control_section
	assert "+!advance_target_t1 : dfa_state(q2) <-" in control_section
	assert "\t!dfa_t1." in control_section
	assert "\t!dfa_t2." in control_section
	assert "blocked_transition(" not in control_section
	assert "target_retry(t1, r1)" in control_section
	assert "\t!advance_target_t1." in control_section
	assert "\t.stopMAS." in failure_section


def test_runner_asl_unordered_control_clears_runtime_blockers_before_retry():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q2).",
				'dfa_edge_label(dfa_t1, "goal(a)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_t1 : dfa_state(q1) <-",
				"\t!do_goal(a);",
				"\t-dfa_state(q1);",
				"\t+dfa_state(q2).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : dfa_state(q2) & accepting_state(q2) <-",
				"\ttrue.",
				"",
				"/* Primitive Action Plans */",
				"+!do_goal(X) : true <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
		],
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask("do_goal", ("X",), False, ("goal",)),
			],
		),
		ordered_query_sequence=False,
	)

	assert "+!clear_runtime_blockers : blocked_runtime_goal(do_goal, X) <-" in asl
	assert (
		"-!advance_target_t1 : not target_seen(t1) & not target_retry(t1, r1) <-\n"
		"\t+target_retry(t1, r1);\n"
		"\t!clear_runtime_blockers;\n"
		"\t!advance_target_t1."
	) in asl


def test_reorder_unordered_control_plan_blocks_uses_preferred_target_order():
	runner = JasonRunner()
	code = "\n".join(
		[
			"domain(test).",
			"",
			"/* DFA Control Plans */",
			"+!run_dfa : target_seen(t1) & target_seen(t2) <-",
			"\ttrue.",
			"",
			"+!clear_blocked_targets : blocked_target(TARGET_ID) <-",
			"\t-blocked_target(TARGET_ID);",
			"\t!clear_blocked_targets.",
			"",
			"+!clear_blocked_targets : true <-",
			"\ttrue.",
			"",
			"+!run_dfa : not target_seen(t1) & not blocked_target(t1) <-",
			"\t!dfa_step_q1_q2_goal_a;",
			"\t!clear_blocked_targets;",
			"\t!run_dfa.",
			"",
			"-!dfa_step_q1_q2_goal_a : not target_seen(t1) <-",
			"\t+blocked_target(t1);",
			"\t!run_dfa.",
			"",
			"+!run_dfa : not target_seen(t2) & not blocked_target(t2) <-",
			"\t!dfa_step_q2_q3_goal_b;",
			"\t!clear_blocked_targets;",
			"\t!run_dfa.",
			"",
			"-!dfa_step_q2_q3_goal_b : not target_seen(t2) <-",
			"\t+blocked_target(t2);",
			"\t!run_dfa.",
			"",
			"+!run_dfa : true <-",
			"\t.fail.",
		],
	)

	reordered = runner._reorder_unordered_control_plan_blocks(code, ["t2", "t1"])

	assert reordered.startswith("domain(test).\n")
	assert reordered.index("+!run_dfa : not target_seen(t2) & not blocked_target(t2) <-") < reordered.index(
		"+!run_dfa : not target_seen(t1) & not blocked_target(t1) <-",
	)
	assert reordered.index("-!dfa_step_q2_q3_goal_b : not target_seen(t2) <-") < reordered.index(
		"-!dfa_step_q1_q2_goal_a : not target_seen(t1) <-",
	)
	assert reordered.index("+!run_dfa : true <-") > reordered.index(
		"-!dfa_step_q1_q2_goal_a : not target_seen(t1) <-",
	)

def test_reorder_unordered_control_plan_blocks_preserves_current_dfa_state_style():
	runner = JasonRunner()
	code = "\n".join(
		[
			"domain(test).",
			"",
			"/* DFA Transition Wrappers */",
			"+!dfa_step_q1_q2_goal_a : dfa_state(q1) <-",
			"\t!task_a;",
			"\t!mark_target_t1;",
			"\t-dfa_state(q1);",
			"\t+dfa_state(q2).",
			"",
			"+!dfa_step_q1_q3_goal_b : dfa_state(q1) <-",
			"\t!task_b;",
			"\t!mark_target_t2;",
			"\t-dfa_state(q1);",
			"\t+dfa_state(q3).",
			"",
			"/* DFA Control Plans */",
			"+!run_dfa : dfa_state(q4) & accepting_state(q4) <-",
			"\ttrue.",
			"",
			"+!run_dfa : dfa_state(q1) <-",
			"\t!dfa_step_q1_q2_goal_a;",
			"\t!run_dfa.",
			"",
			"+!run_dfa : dfa_state(q1) <-",
			"\t!dfa_step_q1_q3_goal_b;",
			"\t!run_dfa.",
			"",
			"+!run_dfa : true <-",
			"\t.fail.",
		],
	)

	reordered = runner._reorder_unordered_control_plan_blocks(code, ["t2", "t1"])

	assert reordered.startswith("domain(test).\n")
	assert reordered.index("+!run_dfa : dfa_state(q4) & accepting_state(q4) <-") < reordered.index(
		"+!run_dfa : dfa_state(q1) <-",
	)
	assert reordered.index("!dfa_step_q1_q3_goal_b;") < reordered.index(
		"!dfa_step_q1_q2_goal_a;",
	)
	assert reordered.count("+!run_dfa : dfa_state(q1) <-") == 2
	assert reordered.index("+!run_dfa : true <-") > reordered.index("!dfa_step_q1_q2_goal_a;")

def test_reorder_unordered_control_plan_blocks_keeps_following_sections_outside_control_region():
	runner = JasonRunner()
	code = "\n".join(
		[
			"domain(test).",
			"",
			"/* DFA Transition Wrappers */",
			"+!dfa_step_q1_q1_goal_a : dfa_state(q1) <-",
			"\t!task_a.",
			"",
			"+!dfa_step_q1_q1_goal_b : dfa_state(q1) <-",
			"\t!task_b.",
			"",
			"/* DFA Control Plans */",
			"+!run_dfa : target_seen(t1) & target_seen(t2) <-",
			"\ttrue.",
			"",
			"+!run_dfa : not target_seen(t1) & not blocked_target(t1) <-",
			"\t!dfa_step_q1_q1_goal_a;",
			"\t!clear_blocked_targets;",
			"\t!run_dfa.",
			"",
			"-!dfa_step_q1_q1_goal_a : not target_seen(t1) <-",
			"\t+blocked_target(t1);",
			"\t!run_dfa.",
			"",
			"+!run_dfa : not target_seen(t2) & not blocked_target(t2) <-",
			"\t!dfa_step_q1_q1_goal_b;",
			"\t!clear_blocked_targets;",
			"\t!run_dfa.",
			"",
			"-!dfa_step_q1_q1_goal_b : not target_seen(t2) <-",
			"\t+blocked_target(t2);",
			"\t!run_dfa.",
			"",
			"+!run_dfa : true <-",
			"\t.fail.",
			"",
			"/* Target Observation Plans */",
			"+!mark_target_t1 : goal(a) <-",
			"\t+target_seen(t1).",
			"",
			"/* Failure Handlers */",
			"-!run_dfa : true <-",
			"\t.fail.",
			"",
			"/* Execution Entry */",
			"!execute.",
		],
	)

	reordered = runner._reorder_unordered_control_plan_blocks(code, ["t2", "t1"])

	assert reordered.index("/* Target Observation Plans */") > reordered.index("+!run_dfa : true <-")
	assert reordered.index("/* Failure Handlers */") > reordered.index("/* Target Observation Plans */")
	assert reordered.index("/* Execution Entry */") > reordered.index("/* Failure Handlers */")
	assert reordered.count("/* Target Observation Plans */") == 1
	assert reordered.count("+!run_dfa : true <-") == 1
	assert reordered.index("+!run_dfa : not target_seen(t2) & not blocked_target(t2) <-") < reordered.index(
		"+!run_dfa : not target_seen(t1) & not blocked_target(t1) <-",
	)

def test_infer_unordered_target_execution_order_replays_world_between_targets(monkeypatch, tmp_path):
	class FakePlanner:
		def __init__(self, *args, **kwargs):
			pass

		def toolchain_available(self) -> bool:
			return True

		def plan(
			self,
			*,
			target_literal,
			initial_facts,
			**kwargs,
		):
			signature = target_literal.to_signature()
			fact_set = set(initial_facts)
			if signature == "goal(a)":
				return SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="unlock", task_name="unlock", args=()),
					],
				)
			if signature == "goal(b)":
				if "(ready)" not in fact_set:
					raise ValueError("not yet reachable")
				return SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="finish", task_name="finish", args=()),
					],
				)
			raise AssertionError(f"unexpected target {signature}")

	monkeypatch.setattr(panda_planner_module, "PANDAPlanner", FakePlanner)

	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(a)", task_name="task_a"),
			HTNTargetTaskBinding(target_literal="goal(b)", task_name="task_b"),
		],
	)
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "unlock",
				"source_name": "unlock",
				"parameters": [],
				"effects": [
					{"predicate": "ready", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "finish",
				"source_name": "finish",
				"parameters": [],
				"effects": [
					{"predicate": "done", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
	)

	assert ordering["target_ids"] == ["t1", "t2"]
	assert ordering["target_signatures"] == ["goal(a)", "goal(b)"]

def test_infer_unordered_target_execution_order_reuses_replayable_stage4_plan_records(
	monkeypatch,
	tmp_path,
):
	class FakePlanner:
		def __init__(self, *args, **kwargs):
			pass

		def toolchain_available(self) -> bool:
			return True

		def plan(self, **kwargs):
			raise AssertionError("planner should not be called when replayable plan records exist")

	monkeypatch.setattr(panda_planner_module, "PANDAPlanner", FakePlanner)

	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(a)", task_name="task_a"),
			HTNTargetTaskBinding(target_literal="goal(b)", task_name="task_b"),
		],
	)
	plan_records = [
		{
			"label": "goal(a)",
			"plan": SimpleNamespace(
				steps=[SimpleNamespace(action_name="do_a", task_name="do_a", args=())],
			),
		},
		{
			"label": "goal(b)",
			"plan": SimpleNamespace(
				steps=[SimpleNamespace(action_name="do_b", task_name="do_b", args=())],
			),
		},
	]
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "do_a",
				"source_name": "do_a",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "after_a", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_b",
				"source_name": "do_b",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "after_b", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=plan_records,
	)

	assert ordering["target_ids"] == ["t1", "t2"]

def test_infer_unordered_target_execution_order_ignores_witness_fallback_and_uses_runtime_probe(
	monkeypatch,
	tmp_path,
):
	class FakePlanner:
		def __init__(self, *args, **kwargs):
			pass

		def toolchain_available(self) -> bool:
			return True

		def plan(
			self,
			*,
			target_literal,
			initial_facts,
			**kwargs,
		):
			signature = target_literal.to_signature()
			fact_set = set(initial_facts)
			if signature == "goal(a)":
				return SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_a", task_name="do_a", args=())],
				)
			if signature == "goal(b)":
				if "(at_x)" not in fact_set:
					raise ValueError("goal(b) is not reachable yet")
				return SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_b_exact", task_name="do_b_exact", args=())],
					actual_plan="\n".join(
						[
							"root 1",
							"1 task_b b -> m-task_b 2",
							"2 do_b_exact",
						],
					),
				)
			if signature == "goal(c)":
				if "(at_x)" not in fact_set or "(c_available)" not in fact_set:
					raise ValueError("goal(c) is not reachable yet")
				return SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_c", task_name="do_c", args=())],
				)
			raise AssertionError(f"unexpected target {signature}")

	monkeypatch.setattr(panda_planner_module, "PANDAPlanner", FakePlanner)

	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("c",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(a)", task_name="task_a"),
			HTNTargetTaskBinding(target_literal="goal(b)", task_name="task_b"),
			HTNTargetTaskBinding(target_literal="goal(c)", task_name="task_c"),
		],
	)
	plan_records = [
		{
			"target_id": "t1",
			"target_literal": target_literals[0],
			"task_name": "task_a",
			"task_args": ("a",),
			"runtime_seed_kind": "problem_exact",
			"plan": SimpleNamespace(
				steps=[SimpleNamespace(action_name="do_a", task_name="do_a", args=())],
			),
		},
		{
			"target_id": "t2",
			"target_literal": target_literals[1],
			"task_name": "task_b",
			"task_args": ("b",),
			"runtime_seed_kind": "witness_fallback",
			"plan": SimpleNamespace(
				steps=[
					SimpleNamespace(
						action_name="do_b_witness",
						task_name="do_b_witness",
						args=(),
					),
				],
			),
		},
	]
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "do_a",
				"source_name": "do_a",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "at_x", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_b_witness",
				"source_name": "do_b_witness",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "done_b", "args": [], "is_positive": True},
					{"predicate": "c_available", "args": [], "is_positive": False},
				],
			},
			{
				"functor": "do_b_exact",
				"source_name": "do_b_exact",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "at_x", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "done_b", "args": [], "is_positive": True},
					{"predicate": "c_available", "args": [], "is_positive": False},
				],
			},
			{
				"functor": "do_c",
				"source_name": "do_c",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "at_x", "args": [], "is_positive": True},
						{"predicate": "c_available", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "done_c", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(c_available)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=plan_records,
		prefer_query_order=True,
	)

	assert ordering["target_ids"] == ["t1", "t3", "t2"]
	assert ordering["target_signatures"] == ["goal(a)", "goal(c)", "goal(b)"]
	assert ordering["guided_method_trace"] == [
		{"method_name": "m-task_b", "task_args": ["b"]},
	]
	assert ordering["guided_action_path"] == [
		"do_a",
		"do_c",
		"do_b_exact",
	]

def test_infer_unordered_target_execution_order_switches_to_exact_suffix_after_greedy_prefix(
	monkeypatch,
	tmp_path,
):
	class FakePlanner:
		def __init__(self, *args, **kwargs):
			pass

		def toolchain_available(self) -> bool:
			return True

		def plan(
			self,
			*,
			target_literal,
			initial_facts,
			**kwargs,
		):
			signature = target_literal.to_signature()
			fact_set = set(initial_facts)
			if signature == "goal(prep)":
				return SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_prep", task_name="do_prep", args=())],
				)
			if signature == "goal(side1)":
				return SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_side1", task_name="do_side1", args=())],
				)
			if signature == "goal(side2)":
				return SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_side2", task_name="do_side2", args=())],
				)
			if signature == "goal(bridge)":
				if "(prep)" not in fact_set:
					raise ValueError("bridge not reachable yet")
				return SimpleNamespace(
					steps=[
						SimpleNamespace(
							action_name="do_bridge_exact",
							task_name="do_bridge_exact",
							args=(),
						),
					],
				)
			if signature == "goal(consume)":
				return SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_consume", task_name="do_consume", args=())],
				)
			if signature == "goal(final)":
				if "(prep)" not in fact_set or "(resource)" not in fact_set:
					raise ValueError("final not reachable yet")
				return SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_final", task_name="do_final", args=())],
				)
			raise AssertionError(f"unexpected target {signature}")

	monkeypatch.setattr(panda_planner_module, "PANDAPlanner", FakePlanner)

	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("prep",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("side1",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("side2",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("bridge",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("consume",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("final",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(prep)", task_name="task_prep"),
			HTNTargetTaskBinding(target_literal="goal(side1)", task_name="task_side1"),
			HTNTargetTaskBinding(target_literal="goal(side2)", task_name="task_side2"),
			HTNTargetTaskBinding(target_literal="goal(bridge)", task_name="task_bridge"),
			HTNTargetTaskBinding(target_literal="goal(consume)", task_name="task_consume"),
			HTNTargetTaskBinding(target_literal="goal(final)", task_name="task_final"),
		],
	)
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "do_prep",
				"source_name": "do_prep",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "prep", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_side1",
				"source_name": "do_side1",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "side1_done", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_side2",
				"source_name": "do_side2",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "side2_done", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_bridge_exact",
				"source_name": "do_bridge_exact",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "prep", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "bridge_done", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_consume",
				"source_name": "do_consume",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "resource", "args": [], "is_positive": False},
					{"predicate": "consume_done", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_final",
				"source_name": "do_final",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "prep", "args": [], "is_positive": True},
						{"predicate": "resource", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "final_done", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(resource)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=[
			{
				"target_id": "t1",
				"target_literal": target_literals[0],
				"task_name": "task_prep",
				"task_args": ("prep",),
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_prep", task_name="do_prep", args=())],
				),
			},
			{
				"target_id": "t2",
				"target_literal": target_literals[1],
				"task_name": "task_side1",
				"task_args": ("side1",),
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_side1", task_name="do_side1", args=())],
				),
			},
			{
				"target_id": "t3",
				"target_literal": target_literals[2],
				"task_name": "task_side2",
				"task_args": ("side2",),
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_side2", task_name="do_side2", args=())],
				),
			},
			{
				"target_id": "t4",
				"target_literal": target_literals[3],
				"task_name": "task_bridge",
				"task_args": ("bridge",),
				"runtime_seed_kind": "witness_fallback",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_bridge_witness", task_name="do_bridge_witness", args=())],
				),
			},
			{
				"target_id": "t5",
				"target_literal": target_literals[4],
				"task_name": "task_consume",
				"task_args": ("consume",),
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_consume", task_name="do_consume", args=())],
				),
			},
			{
				"target_id": "t6",
				"target_literal": target_literals[5],
				"task_name": "task_final",
				"task_args": ("final",),
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_final", task_name="do_final", args=())],
				),
			},
		],
		prefer_query_order=True,
	)

	assert ordering["target_ids"][0] == "t1"
	assert ordering["target_ids"].index("t4") < ordering["target_ids"].index("t5")
	assert ordering["target_ids"].index("t6") < ordering["target_ids"].index("t5")

def test_infer_unordered_target_execution_order_exact_suffix_only_probes_earliest_rescue_target(
	monkeypatch,
	tmp_path,
):
	class FakePlanner:
		def __init__(self, *args, **kwargs):
			pass

		def toolchain_available(self) -> bool:
			return True

		def plan(
			self,
			*,
			target_literal,
			initial_facts,
			**kwargs,
		):
			signature = target_literal.to_signature()
			fact_set = set(initial_facts)
			if signature == "goal(bridge_a)":
				if "(prep)" not in fact_set:
					raise ValueError("bridge_a not reachable yet")
				return SimpleNamespace(
					steps=[
						SimpleNamespace(
							action_name="do_bridge_a_exact",
							task_name="do_bridge_a_exact",
							args=(),
						),
					],
				)
			if signature == "goal(bridge_b)":
				if "(bridge_a_done)" not in fact_set:
					raise AssertionError(
						"exact suffix should not probe a later unreplayable target "
						"before the earliest rescue target succeeds"
					)
				return SimpleNamespace(
					steps=[
						SimpleNamespace(
							action_name="do_bridge_b_exact",
							task_name="do_bridge_b_exact",
							args=(),
						),
					],
				)
			raise AssertionError(f"unexpected target {signature}")

	monkeypatch.setattr(panda_planner_module, "PANDAPlanner", FakePlanner)

	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("prep",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("bridge_a",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("bridge_b",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("consume",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("final_a",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("final_b",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(prep)", task_name="task_prep"),
			HTNTargetTaskBinding(target_literal="goal(bridge_a)", task_name="task_bridge_a"),
			HTNTargetTaskBinding(target_literal="goal(bridge_b)", task_name="task_bridge_b"),
			HTNTargetTaskBinding(target_literal="goal(consume)", task_name="task_consume"),
			HTNTargetTaskBinding(target_literal="goal(final_a)", task_name="task_final_a"),
			HTNTargetTaskBinding(target_literal="goal(final_b)", task_name="task_final_b"),
		],
	)

	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "do_prep",
				"source_name": "do_prep",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "prep", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_bridge_a_exact",
				"source_name": "do_bridge_a_exact",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "prep", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "bridge_a_done", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_bridge_b_exact",
				"source_name": "do_bridge_b_exact",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "bridge_a_done", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "bridge_b_done", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_consume",
				"source_name": "do_consume",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "resource", "args": [], "is_positive": False},
					{"predicate": "consume_done", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_final_a",
				"source_name": "do_final_a",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "prep", "args": [], "is_positive": True},
						{"predicate": "resource", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "final_a_done", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_final_b",
				"source_name": "do_final_b",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "bridge_a_done", "args": [], "is_positive": True},
						{"predicate": "resource", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "final_b_done", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(resource)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=[
			{
				"target_id": "t1",
				"target_literal": target_literals[0],
				"task_name": "task_prep",
				"task_args": ("prep",),
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_prep", task_name="do_prep", args=())],
				),
			},
			{
				"target_id": "t2",
				"target_literal": target_literals[1],
				"task_name": "task_bridge_a",
				"task_args": ("bridge_a",),
				"runtime_seed_kind": "witness_fallback",
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(
							action_name="do_bridge_a_witness",
							task_name="do_bridge_a_witness",
							args=(),
						),
					],
				),
			},
			{
				"target_id": "t3",
				"target_literal": target_literals[2],
				"task_name": "task_bridge_b",
				"task_args": ("bridge_b",),
				"runtime_seed_kind": "witness_fallback",
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(
							action_name="do_bridge_b_witness",
							task_name="do_bridge_b_witness",
							args=(),
						),
					],
				),
			},
			{
				"target_id": "t4",
				"target_literal": target_literals[3],
				"task_name": "task_consume",
				"task_args": ("consume",),
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_consume", task_name="do_consume", args=())],
				),
			},
			{
				"target_id": "t5",
				"target_literal": target_literals[4],
				"task_name": "task_final_a",
				"task_args": ("final_a",),
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_final_a", task_name="do_final_a", args=())],
				),
			},
			{
				"target_id": "t6",
				"target_literal": target_literals[5],
				"task_name": "task_final_b",
				"task_args": ("final_b",),
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_final_b", task_name="do_final_b", args=())],
				),
			},
		],
		prefer_query_order=True,
	)

	assert ordering["target_ids"][0] == "t1"
	assert ordering["target_ids"].index("t2") < ordering["target_ids"].index("t4")
	assert ordering["target_ids"].index("t3") < ordering["target_ids"].index("t4")

def test_infer_unordered_target_execution_order_prefers_shorter_exact_record_plan(tmp_path):
	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("first",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("second",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("third",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(first)", task_name="task_first"),
			HTNTargetTaskBinding(target_literal="goal(second)", task_name="task_second"),
			HTNTargetTaskBinding(target_literal="goal(third)", task_name="task_third"),
		],
	)
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "unlock_first",
				"source_name": "unlock_first",
				"parameters": [],
				"effects": [
					{"predicate": "after_first", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "unlock_third",
				"source_name": "unlock_third",
				"parameters": [],
				"effects": [
					{"predicate": "after_third", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "finish_second",
				"source_name": "finish_second",
				"parameters": [],
				"effects": [
					{"predicate": "done", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=[
			{
				"target_id": "t1",
				"target_literal": target_literals[0],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="unlock_first", task_name="unlock_first", args=()),
						SimpleNamespace(action_name="unlock_first", task_name="unlock_first", args=()),
						SimpleNamespace(action_name="unlock_first", task_name="unlock_first", args=()),
					],
				),
			},
			{
				"target_id": "t2",
				"target_literal": target_literals[1],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="finish_second", task_name="finish_second", args=()),
					],
				),
			},
			{
				"target_id": "t3",
				"target_literal": target_literals[2],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="unlock_third", task_name="unlock_third", args=()),
						SimpleNamespace(action_name="unlock_third", task_name="unlock_third", args=()),
					],
				),
			},
		],
	)

	assert ordering["target_ids"] == ["t2", "t3", "t1"]
	assert ordering["target_signatures"] == ["goal(second)", "goal(third)", "goal(first)"]

def test_infer_unordered_target_execution_order_prefers_more_reachable_follow_ons(
	tmp_path,
):
	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("first",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("second",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("third",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(first)", task_name="task_first"),
			HTNTargetTaskBinding(target_literal="goal(second)", task_name="task_second"),
			HTNTargetTaskBinding(target_literal="goal(third)", task_name="task_third"),
		],
	)
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "do_first",
				"source_name": "do_first",
				"parameters": [],
				"effects": [
					{"predicate": "after_first", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_second",
				"source_name": "do_second",
				"parameters": [],
				"effects": [
					{"predicate": "after_second", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_third",
				"source_name": "do_third",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_second", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "after_third", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=[
			{
				"target_id": "t1",
				"target_literal": target_literals[0],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_first", task_name="do_first", args=()),
					],
				),
			},
			{
				"target_id": "t2",
				"target_literal": target_literals[1],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_second", task_name="do_second", args=()),
						SimpleNamespace(action_name="do_second", task_name="do_second", args=()),
					],
				),
			},
			{
				"target_id": "t3",
				"target_literal": target_literals[2],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_third", task_name="do_third", args=()),
					],
				),
			},
		],
	)

	assert ordering["target_ids"][0] == "t2"
	assert ordering["target_signatures"][0] == "goal(second)"

def test_infer_unordered_target_execution_order_promotes_earliest_blocked_future_target(
	tmp_path,
):
	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("first",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("second",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("third",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(first)", task_name="task_first"),
			HTNTargetTaskBinding(target_literal="goal(second)", task_name="task_second"),
			HTNTargetTaskBinding(target_literal="goal(third)", task_name="task_third"),
		],
	)
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "do_first",
				"source_name": "do_first",
				"parameters": [],
				"effects": [
					{"predicate": "after_first", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_second",
				"source_name": "do_second",
				"parameters": [],
				"effects": [
					{"predicate": "after_second", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_third",
				"source_name": "do_third",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "start", "args": [], "is_positive": True},
						{"predicate": "after_first", "args": [], "is_positive": False},
					],
				],
				"effects": [
					{"predicate": "after_third", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=[
			{
				"target_id": "t1",
				"target_literal": target_literals[0],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_first", task_name="do_first", args=()),
					],
				),
			},
			{
				"target_id": "t2",
				"target_literal": target_literals[1],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_second", task_name="do_second", args=()),
					],
				),
			},
			{
				"target_id": "t3",
				"target_literal": target_literals[2],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_third", task_name="do_third", args=()),
					],
				),
			},
		],
		prefer_query_order=True,
	)

	assert ordering["target_ids"][0] == "t3"

def test_infer_unordered_target_execution_order_probes_query_prefix_before_large_set_drift(
	monkeypatch,
	tmp_path,
):
	class FakePlanner:
		def __init__(self, *args, **kwargs):
			pass

		def toolchain_available(self) -> bool:
			return True

		def plan(
			self,
			*,
			target_literal,
			initial_facts,
			**kwargs,
		):
			signature = target_literal.to_signature()
			fact_set = set(initial_facts)
			if signature != "goal(second)":
				raise AssertionError(f"unexpected probe for {signature}")
			if "(after_first)" not in fact_set:
				raise ValueError("second is not reachable before first")
			return SimpleNamespace(
				steps=[
					SimpleNamespace(action_name="do_second_exact", task_name="do_second_exact", args=()),
				],
			)

	monkeypatch.setattr(panda_planner_module, "PANDAPlanner", FakePlanner)

	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("first",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("second",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("third",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("fourth",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("fifth",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("sixth",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("seventh",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("eighth",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(first)", task_name="task_first"),
			HTNTargetTaskBinding(target_literal="goal(second)", task_name="task_second"),
			HTNTargetTaskBinding(target_literal="goal(third)", task_name="task_third"),
			HTNTargetTaskBinding(target_literal="goal(fourth)", task_name="task_fourth"),
			HTNTargetTaskBinding(target_literal="goal(fifth)", task_name="task_fifth"),
			HTNTargetTaskBinding(target_literal="goal(sixth)", task_name="task_sixth"),
			HTNTargetTaskBinding(target_literal="goal(seventh)", task_name="task_seventh"),
			HTNTargetTaskBinding(target_literal="goal(eighth)", task_name="task_eighth"),
		],
	)
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "do_first",
				"source_name": "do_first",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "after_first", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_second_exact",
				"source_name": "do_second_exact",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_first", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "after_second", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_third",
				"source_name": "do_third",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "after_third", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_fourth",
				"source_name": "do_fourth",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "after_fourth", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_fifth",
				"source_name": "do_fifth",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "after_fifth", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=[
			{
				"target_id": "t1",
				"target_literal": target_literals[0],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_first", task_name="do_first", args=())],
				),
			},
			{
				"target_id": "t2",
				"target_literal": target_literals[1],
				"runtime_seed_kind": "witness_fallback",
				"task_name": "task_second",
				"task_args": ("second",),
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_second_witness", task_name="do_second_witness", args=()),
					],
				),
			},
			{
				"target_id": "t3",
				"target_literal": target_literals[2],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_third", task_name="do_third", args=())],
				),
			},
			{
				"target_id": "t4",
				"target_literal": target_literals[3],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_fourth", task_name="do_fourth", args=())],
				),
			},
			{
				"target_id": "t5",
				"target_literal": target_literals[4],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_fifth", task_name="do_fifth", args=())],
				),
			},
		],
		prefer_query_order=True,
	)

	assert ordering["target_ids"][:2] == ["t1", "t2"]

def test_infer_unordered_target_execution_order_keeps_runtime_probeable_query_prefix_ahead_of_more_reachable_later_targets(
	monkeypatch,
	tmp_path,
):
	class FakePlanner:
		def __init__(self, *args, **kwargs):
			pass

		def toolchain_available(self) -> bool:
			return True

		def plan(
			self,
			*,
			target_literal,
			initial_facts,
			**kwargs,
		):
			signature = target_literal.to_signature()
			fact_set = set(initial_facts)
			if signature != "goal(second)":
				raise AssertionError(f"unexpected probe for {signature}")
			if "(after_first)" not in fact_set:
				raise ValueError("second is not reachable before first")
			return SimpleNamespace(
				steps=[
					SimpleNamespace(
						action_name="do_second_exact",
						task_name="do_second_exact",
						args=(),
					),
				],
			)

	monkeypatch.setattr(panda_planner_module, "PANDAPlanner", FakePlanner)

	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("first",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("second",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("third",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("fourth",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("fifth",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("sixth",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("seventh",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("eighth",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(first)", task_name="task_first"),
			HTNTargetTaskBinding(target_literal="goal(second)", task_name="task_second"),
			HTNTargetTaskBinding(target_literal="goal(third)", task_name="task_third"),
			HTNTargetTaskBinding(target_literal="goal(fourth)", task_name="task_fourth"),
			HTNTargetTaskBinding(target_literal="goal(fifth)", task_name="task_fifth"),
			HTNTargetTaskBinding(target_literal="goal(sixth)", task_name="task_sixth"),
			HTNTargetTaskBinding(target_literal="goal(seventh)", task_name="task_seventh"),
			HTNTargetTaskBinding(target_literal="goal(eighth)", task_name="task_eighth"),
		],
	)
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "do_first",
				"source_name": "do_first",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": "after_first", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_second_exact",
				"source_name": "do_second_exact",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_first", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "after_second", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_third",
				"source_name": "do_third",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_first", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "after_third", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_fourth",
				"source_name": "do_fourth",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_third", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "after_fourth", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_fifth",
				"source_name": "do_fifth",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_third", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "after_fifth", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_sixth",
				"source_name": "do_sixth",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_third", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "after_sixth", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_seventh",
				"source_name": "do_seventh",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_third", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "after_seventh", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_eighth",
				"source_name": "do_eighth",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_third", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "after_eighth", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=[
			{
				"target_id": "t1",
				"target_literal": target_literals[0],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_first", task_name="do_first", args=())],
				),
			},
			{
				"target_id": "t2",
				"target_literal": target_literals[1],
				"runtime_seed_kind": "witness_fallback",
				"task_name": "task_second",
				"task_args": ("second",),
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(
							action_name="do_second_witness",
							task_name="do_second_witness",
							args=(),
						),
					],
				),
			},
			{
				"target_id": "t3",
				"target_literal": target_literals[2],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_third", task_name="do_third", args=())],
				),
			},
			{
				"target_id": "t4",
				"target_literal": target_literals[3],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_fourth", task_name="do_fourth", args=())],
				),
			},
			{
				"target_id": "t5",
				"target_literal": target_literals[4],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_fifth", task_name="do_fifth", args=())],
				),
			},
			{
				"target_id": "t6",
				"target_literal": target_literals[5],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_sixth", task_name="do_sixth", args=())],
				),
			},
			{
				"target_id": "t7",
				"target_literal": target_literals[6],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(
							action_name="do_seventh",
							task_name="do_seventh",
							args=(),
						),
					],
				),
			},
			{
				"target_id": "t8",
				"target_literal": target_literals[7],
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[SimpleNamespace(action_name="do_eighth", task_name="do_eighth", args=())],
				),
			},
		],
		prefer_query_order=True,
	)

	assert ordering["target_ids"][:2] == ["t1", "t2"]

def test_infer_unordered_target_execution_order_uses_exact_replay_suffix_for_small_set(
	tmp_path,
):
	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("first",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("second",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("third",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("fourth",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(first)", task_name="task_first"),
			HTNTargetTaskBinding(target_literal="goal(second)", task_name="task_second"),
			HTNTargetTaskBinding(target_literal="goal(third)", task_name="task_third"),
			HTNTargetTaskBinding(target_literal="goal(fourth)", task_name="task_fourth"),
		],
	)
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "do_first",
				"source_name": "do_first",
				"parameters": [],
				"effects": [
					{"predicate": "after_first", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_second",
				"source_name": "do_second",
				"parameters": [],
				"effects": [
					{"predicate": "after_second", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_third",
				"source_name": "do_third",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_first", "args": [], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "after_third", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "do_fourth",
				"source_name": "do_fourth",
				"parameters": [],
				"precondition_clauses": [
					[
						{"predicate": "after_second", "args": [], "is_positive": True},
						{"predicate": "after_first", "args": [], "is_positive": False},
					],
				],
				"effects": [
					{"predicate": "after_fourth", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=[
			{
				"target_id": "t1",
				"target_literal": target_literals[0],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_first", task_name="do_first", args=()),
					],
				),
			},
			{
				"target_id": "t2",
				"target_literal": target_literals[1],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_second", task_name="do_second", args=()),
					],
				),
			},
			{
				"target_id": "t3",
				"target_literal": target_literals[2],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_third", task_name="do_third", args=()),
					],
				),
			},
			{
				"target_id": "t4",
				"target_literal": target_literals[3],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="do_fourth", task_name="do_fourth", args=()),
					],
				),
			},
		],
	)

	assert ordering["target_ids"] == ["t2", "t4", "t1", "t3"]
	assert ordering["target_signatures"] == [
		"goal(second)",
		"goal(fourth)",
		"goal(first)",
		"goal(third)",
	]
	assert ordering["guided_action_path"] == [
		"do_second",
		"do_fourth",
		"do_first",
		"do_third",
	]

def test_infer_unordered_target_execution_order_skips_exact_suffix_when_small_set_is_fully_replayable(
	monkeypatch,
	tmp_path,
):
	class FakePlanner:
		def __init__(self, *args, **kwargs):
			raise AssertionError("fully replayable small suffix should not construct planner")

	monkeypatch.setattr(panda_planner_module, "PANDAPlanner", FakePlanner)

	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=(str(index),), is_positive=True, source_symbol=None)
		for index in range(1, 7)
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(
				target_literal=literal.to_signature(),
				task_name=f"task_{index}",
			)
			for index, literal in enumerate(target_literals, start=1)
		],
	)

	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": f"do_{index}",
				"source_name": f"do_{index}",
				"parameters": [],
				"precondition_clauses": [[]],
				"effects": [
					{"predicate": f"done_{index}", "args": [], "is_positive": True},
				],
			}
			for index in range(1, 7)
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=[
			{
				"target_id": f"t{index}",
				"target_literal": literal,
				"runtime_seed_kind": "problem_exact",
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(
							action_name=f"do_{index}",
							task_name=f"do_{index}",
							args=(),
						),
					],
				),
			}
			for index, literal in enumerate(target_literals, start=1)
		],
		prefer_query_order=True,
	)

	assert ordering["target_ids"] == [f"t{index}" for index in range(1, 7)]

def test_infer_unordered_target_execution_order_defaults_to_query_order_without_runtime_records(
	tmp_path,
):
	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(a)", task_name="task_a"),
			HTNTargetTaskBinding(target_literal="goal(b)", task_name="task_b"),
		],
	)
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "unlock",
				"source_name": "unlock",
				"parameters": [],
				"effects": [
					{"predicate": "ready", "args": [], "is_positive": True},
				],
			},
			{
				"functor": "finish",
				"source_name": "finish",
				"parameters": [],
				"effects": [
					{"predicate": "done", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
	)

	assert ordering["target_ids"] == ["t1", "t2"]

def test_infer_unordered_target_execution_order_skips_global_ranking_for_large_target_sets(
	monkeypatch,
	tmp_path,
):
	class FakePlanner:
		def __init__(self, *args, **kwargs):
			pass

		def toolchain_available(self) -> bool:
			return True

		def plan(self, **kwargs):
			raise AssertionError("large unordered target sets should not call planner ranking")

	monkeypatch.setattr(panda_planner_module, "PANDAPlanner", FakePlanner)

	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=(str(index),), is_positive=True, source_symbol=None)
		for index in range(25)
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(
				target_literal=literal.to_signature(),
				task_name=f"task_{index}",
			)
			for index, literal in enumerate(target_literals, start=1)
		],
	)

	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
	)

	assert ordering["target_ids"] == [f"t{index}" for index in range(1, 26)]
	assert ordering["target_signatures"] == [
		f"goal({index})"
		for index in range(25)
	]

def test_infer_unordered_target_execution_order_keeps_query_suffix_for_unreplayable_targets(
	tmp_path,
):
	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
	]
	method_library = HTNMethodLibrary(
		target_literals=list(target_literals),
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="goal(a)", task_name="task_a"),
			HTNTargetTaskBinding(target_literal="goal(b)", task_name="task_b"),
		],
	)
	ordering = runner._infer_unordered_target_execution_order(
		target_literals=target_literals,
		method_library=method_library,
		action_schemas=[
			{
				"functor": "unlock",
				"source_name": "unlock",
				"parameters": [],
				"effects": [
					{"predicate": "ready", "args": [], "is_positive": True},
				],
			},
		],
		seed_facts=["(start)"],
		runtime_objects=(),
		object_types={},
		planning_domain=SimpleNamespace(name="test"),
		output_path=tmp_path,
		plan_records=[
			{
				"target_id": "t1",
				"target_literal": target_literals[0],
				"plan": SimpleNamespace(
					steps=[
						SimpleNamespace(action_name="unlock", task_name="unlock", args=()),
					],
				),
			},
		],
	)

	assert ordering["target_ids"] == ["t1", "t2"]
	assert ordering["target_signatures"] == ["goal(a)", "goal(b)"]

def test_runner_asl_injects_runtime_objects_and_parent_types():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"/* Initial Beliefs */",
				"domain(test).",
				"",
				"/* Primitive Action Plans */",
				"+!noop : true <-",
				"\ttrue.",
			],
		),
		[],
		runtime_objects=("GroundStation2", "truck-0"),
		object_types={
			"GroundStation2": "calib_direction",
			"truck-0": "vehicle",
		},
		type_parent_map={
			"calib_direction": "direction",
			"direction": "object",
			"vehicle": "object",
			"object": None,
		},
	)

	assert 'object("GroundStation2").' in asl
	assert 'object_type("GroundStation2", calib_direction).' in asl
	assert 'object_type("GroundStation2", direction).' in asl
	assert 'object("truck-0").' in asl
	assert 'object_type("truck-0", vehicle).' in asl

def test_runner_asl_forces_negative_targets_to_naf_notation():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"domain(test).\ndfa_state(q1).",
		[
			HTNLiteral(
				predicate="clear",
				args=("b",),
				is_positive=False,
				source_symbol=None,
				negation_mode="strong",
			),
		],
	)

	assert "+!verify_targets : not clear(b) <-" in asl
	assert "~clear(b)" not in asl

def test_runner_asl_uses_seen_target_observations_when_dfa_edge_labels_match_targets():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q2).",
				'dfa_edge_label(dfa_step_q1_q2_on_a_b, "on(a, b)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_step_q1_q2_on_a_b : dfa_state(q1) <-",
				"\t!place_on(a, b);",
				"\t-dfa_state(q1);",
				"\t+dfa_state(q2).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : dfa_state(q2) & accepting_state(q2) <-",
				"\ttrue.",
				"",
				"+!run_dfa : dfa_state(q1) <-",
				"\t!dfa_step_q1_q2_on_a_b;",
				"\t!run_dfa.",
			],
		),
		[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
		],
	)

	assert "+!verify_targets : target_seen(t1) <-" in asl
	assert "+!mark_target_t1 : on(a, b) & protected_target_on(a, b) <-" in asl
	assert "+!mark_target_t1 : on(a, b) & not protected_target_on(a, b) <-" in asl
	assert "\t!place_on(a, b);" in asl
	assert "\t!mark_target_t1;" in asl
	assert "\t+target_seen(t1);" in asl
	assert "\t+protected_target_on(a, b)." in asl

def test_runner_asl_marks_targets_after_the_last_ordered_task_call():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q2).",
				"accepting_state(q3).",
				'dfa_edge_label(dfa_step_q2_q3_on_b1_b4, "on(b1, b4)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_step_q2_q3_on_b1_b4 : dfa_state(q2) <-",
				"\t!do_put_on(b4, b2);",
				"\t!do_put_on(b1, b4);",
				"\t-dfa_state(q2);",
				"\t+dfa_state(q3).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : dfa_state(q3) & accepting_state(q3) <-",
				"\ttrue.",
				"",
				"+!run_dfa : dfa_state(q2) <-",
				"\t!dfa_step_q2_q3_on_b1_b4;",
				"\t!run_dfa.",
			],
		),
		[
			HTNLiteral(predicate="on", args=("b1", "b4"), is_positive=True, source_symbol=None),
		],
	)

	wrapper_block = asl.split(
		"+!dfa_step_q2_q3_on_b1_b4 : dfa_state(q2) & on(b1, b4) <-",
		1,
	)[1].split(
		"/* DFA Control Plans */",
		1,
	)[0]
	assert wrapper_block.index("\t!do_put_on(b1, b4);") < wrapper_block.index("\t!mark_target_t1;")
	assert wrapper_block.index("\t!mark_target_t1;") < wrapper_block.index("\t-dfa_state(q2);")

def test_runner_asl_marks_negative_target_absence_as_protected():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"\n".join(
			[
				"domain(test).",
				"dfa_state(q1).",
				"accepting_state(q2).",
				'dfa_edge_label(dfa_step_q1_q2_not_clear_b, "!clear(b)").',
				"",
				"/* DFA Transition Wrappers */",
				"+!dfa_step_q1_q2_not_clear_b : dfa_state(q1) <-",
				"\t!stabilise(b);",
				"\t-dfa_state(q1);",
				"\t+dfa_state(q2).",
				"",
				"/* DFA Control Plans */",
				"+!run_dfa : dfa_state(q2) & accepting_state(q2) <-",
				"\ttrue.",
			],
		),
		[
			HTNLiteral(predicate="clear", args=("b",), is_positive=False, source_symbol=None),
		],
	)

	assert "+!mark_target_t1 : not clear(b) & protected_absence_clear(b) <-" in asl
	assert "+!mark_target_t1 : not clear(b) & not protected_absence_clear(b) <-" in asl
	assert "\t+protected_absence_clear(b)." in asl

def test_extract_initial_dfa_state_reads_header_belief():
	runner = JasonRunner()

	assert runner._extract_initial_dfa_state("domain(test).\ndfa_state(q7).\n") == "q7"
	assert runner._extract_initial_dfa_state("domain(test).\n") is None

def test_extract_method_trace_strips_quoted_task_args():
	runner = JasonRunner()
	trace = runner._extract_method_trace(
		'[agent] runtime trace method trace_method("method0", "Phenomenon4", thermograph0)\n',
	)

	assert trace == [
		{
			"method_name": "method0",
			"task_args": ["Phenomenon4", "thermograph0"],
		},
	]

def test_literal_to_context_expression_quotes_unsafe_constants():
	context = JasonRunner._literal_to_context_expression(
		HTNLiteral("have_image", ("Phenomenon4", "thermograph0"), True, None),
	)

	assert context == 'have_image("Phenomenon4", thermograph0)'

def test_literal_to_context_expression_sanitizes_hyphenated_predicates():
	context = JasonRunner._literal_to_context_expression(
		HTNLiteral("capacity-predecessor", ("capacity-0", "capacity-1"), True, None),
	)

	assert context == 'capacity_predecessor("capacity-0", "capacity-1")'

def test_rewrite_primitive_wrappers_runs_external_action_then_mirrors_effects():
	runner = JasonRunner()
	code = """
/* Primitive Action Plans */
+!pick_up(BLOCK1, BLOCK2) : handempty <-
\tpick_up(BLOCK1, BLOCK2);
\t-on(BLOCK1, BLOCK2);
\t+holding(BLOCK1).

/* HTN Method Plans */
+!demo : true <-
\ttrue.
""".strip()
	rewritten = runner._rewrite_primitive_wrappers_for_environment(code)
	assert "\tpick_up(BLOCK1, BLOCK2);" in rewritten
	assert "\t-on(BLOCK1, BLOCK2);" in rewritten
	assert "\t+holding(BLOCK1);" in rewritten
	assert "\t.perceive." in rewritten
	assert rewritten.index("\tpick_up(BLOCK1, BLOCK2);") < rewritten.index("\t-on(BLOCK1, BLOCK2);")
	assert rewritten.index("\t-on(BLOCK1, BLOCK2);") < rewritten.index("\t+holding(BLOCK1);")
	assert rewritten.index("\t+holding(BLOCK1);") < rewritten.index("\t.perceive.")

def test_select_java_prefers_highest_supported_version(monkeypatch):
	runner = JasonRunner()
	candidates = ["/java24", "/java23", "/java17"]
	versions = {
		"/java24": 24,
		"/java23": 23,
		"/java17": 17,
	}
	monkeypatch.setattr(runner, "_discover_java_candidates", lambda: candidates)
	monkeypatch.setattr(runner, "_probe_java_binary", lambda path: versions[path])

	assert runner._select_java_binary() == ("/java23", 23)

def test_select_java_raises_when_no_supported_version(monkeypatch):
	runner = JasonRunner()
	candidates = ["/java16", "/java24"]
	versions = {
		"/java16": 16,
		"/java24": 24,
	}
	monkeypatch.setattr(runner, "_discover_java_candidates", lambda: candidates)
	monkeypatch.setattr(runner, "_probe_java_binary", lambda path: versions[path])

	with pytest.raises(JasonValidationError) as exc_info:
		runner._select_java_binary()

	assert "requires Java 17-23" in str(exc_info.value)
	assert exc_info.value.metadata["candidates"] == versions

def test_ensure_jason_jar_triggers_build_when_missing(monkeypatch, tmp_path):
	stage6_dir = tmp_path / "stage6"
	stage6_dir.mkdir()
	runner = JasonRunner(stage6_dir=stage6_dir)
	jar_file = stage6_dir / "jason_src" / "jason-cli" / "build" / "bin" / "jason-cli-all-3.3.1.jar"
	jar_file.parent.mkdir(parents=True)
	jar_file.write_text("jar")

	find_calls = {"count": 0}
	build_calls: list[str] = []

	def fake_find():
		find_calls["count"] += 1
		return None if find_calls["count"] == 1 else jar_file

	monkeypatch.setattr(runner, "_find_jason_jar", fake_find)
	monkeypatch.setattr(runner, "_build_jason_cli", lambda java_bin: build_calls.append(java_bin))

	resolved = runner._ensure_jason_jar("/java23")
	assert resolved == jar_file
	assert build_calls == ["/java23"]

def test_validate_success_writes_stage6_artifacts(monkeypatch, tmp_path):
	stage6_dir = tmp_path / "stage6"
	stage6_dir.mkdir()
	runner = JasonRunner(stage6_dir=stage6_dir, timeout_seconds=1)
	jar_file = tmp_path / "jason-cli-all.jar"
	log_conf = tmp_path / "console-info.properties"
	jar_file.write_text("jar")
	log_conf.write_text("log")

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("/java23", 23))
	monkeypatch.setattr(runner, "_select_javac_binary", lambda java_bin: "/javac23")
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jar_file)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(
		runner,
		"_compile_environment_java",
		lambda **kwargs: (tmp_path / "Stage6PipelineEnvironment.class").write_text("class"),
	)
	monkeypatch.setattr(
		subprocess,
		"run",
		lambda *args, **kwargs: subprocess.CompletedProcess(
			args=args[0],
			returncode=0,
			stdout=(
				"runtime env ready\n"
				"runtime trace method trace_method(m_do_put_on_domain_2,a,b)\n"
				"runtime env action success pick-up(a,b)\n"
				"execute start\n"
				"execute success\n"
			),
			stderr="",
		),
	)

	result = runner.validate(
		agentspeak_code="domain(test).\n/* Primitive Action Plans */\n+!pick_up(B1,B2):true<-\n\tpick_up(B1,B2);\n\t+holding(B1).\n\n/* HTN Method Plans */",
		target_literals=[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
		],
		action_schemas=_sample_action_schemas(),
		seed_facts=("(clear a)", "(on a b)"),
		domain_name="blocksworld",
		output_dir=tmp_path,
	)

	assert result.status == "success"
	assert result.environment_adapter["success"] is True
	assert (tmp_path / "agentspeak_generated.asl").exists()
	assert (tmp_path / "jason_runner.mas2j").exists()
	assert (tmp_path / "Stage6PipelineEnvironment.java").exists()
	assert (tmp_path / "Stage6PipelineEnvironment.class").exists()
	assert (tmp_path / "jason_stdout.txt").exists()
	assert (tmp_path / "jason_stderr.txt").exists()
	assert (tmp_path / "jason_validation.json").exists()
	assert (tmp_path / "action_path.txt").exists()
	assert (tmp_path / "method_trace.json").exists()
	assert (tmp_path / "action_path.txt").read_text() == "pick-up(a,b)\n"

	validation_payload = json.loads((tmp_path / "jason_validation.json").read_text())
	assert validation_payload["status"] == "success"
	assert validation_payload["environment_adapter"]["success"] is True
	assert validation_payload["failure_class"] is None
	assert validation_payload["action_path"] == ["pick-up(a,b)"]
	assert validation_payload["method_trace"] == [
		{"method_name": "m_do_put_on_domain_2", "task_args": ["a", "b"]},
	]
	assert validation_payload["failed_goals"] == []
	assert validation_payload["consistency_checks"]["action_path_schema_replay"]["passed"] is True
	assert validation_payload["consistency_checks"]["method_trace_reconstruction"]["passed"] is None
	assert validation_payload["artifacts"]["action_path"] == str(tmp_path / "action_path.txt")
	assert validation_payload["artifacts"]["method_trace"] == str(tmp_path / "method_trace.json")

def test_validate_does_not_runtime_specialise_method_plans(monkeypatch, tmp_path):
	stage6_dir = tmp_path / "stage6"
	stage6_dir.mkdir()
	runner = JasonRunner(stage6_dir=stage6_dir, timeout_seconds=1)
	jar_file = tmp_path / "jason-cli-all.jar"
	log_conf = tmp_path / "console-info.properties"
	jar_file.write_text("jar")
	log_conf.write_text("log")

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("/java23", 23))
	monkeypatch.setattr(runner, "_select_javac_binary", lambda java_bin: "/javac23")
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jar_file)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(
		runner,
		"_compile_environment_java",
		lambda **kwargs: (tmp_path / "Stage6PipelineEnvironment.class").write_text("class"),
	)
	monkeypatch.setattr(
		runner,
		"_ground_local_witness_method_plans",
		lambda *args, **kwargs: (_ for _ in ()).throw(
			AssertionError("Stage 6 should execute Stage 5 ASL directly"),
		),
	)
	monkeypatch.setattr(
		runner,
		"_run_consistency_checks",
		lambda **kwargs: {
			"success": True,
			"action_path_schema_replay": {"passed": True},
			"method_trace_reconstruction": {"passed": True},
		},
	)
	monkeypatch.setattr(
		subprocess,
		"run",
		lambda *args, **kwargs: subprocess.CompletedProcess(
			args=args[0],
			returncode=0,
			stdout="runtime env ready\nexecute start\nexecute success\n",
			stderr="",
		),
	)

	result = runner.validate(
		agentspeak_code="domain(test).\n/* HTN Method Plans */\n+!deliver(package0,city1):true<-\n\ttrue.\n",
		target_literals=[],
		action_schemas=_sample_action_schemas(),
		seed_facts=("(start)",),
		domain_name="transport",
		output_dir=tmp_path,
	)

	assert result.status == "success"

def test_extract_action_path_preserves_runtime_order():
	runner = JasonRunner()
	stdout = "\n".join(
		[
			"runtime env ready",
			"runtime env action success pick-up(a,b)",
			"runtime env action success put-on-block(a,c)",
			"execute success",
		],
	)

	assert runner._extract_action_path(stdout) == [
		"pick-up(a,b)",
		"put-on-block(a,c)",
	]

def test_extract_method_trace_preserves_runtime_order():
	runner = JasonRunner()
	stdout = "\n".join(
		[
			"runtime trace method trace_method(m_drive_to_domain_1,rover0,waypoint1)",
			"runtime trace method trace_method(m_send_data_domain_2,rover0,waypoint1,channel0)",
		],
	)

	assert runner._extract_method_trace(stdout) == [
		{"method_name": "m_drive_to_domain_1", "task_args": ["rover0", "waypoint1"]},
		{
			"method_name": "m_send_data_domain_2",
			"task_args": ["rover0", "waypoint1", "channel0"],
		},
	]

def test_extract_panda_method_trace_preserves_hierarchical_order():
	runner = JasonRunner()
	plan_text = "\n".join(
		[
			"root 1",
			"1 get_image_data objective1 high_res -> m-get_image_data 2 3",
			"2 calibrate_abs rover3 camera2 -> m-calibrate_abs 4",
			"3 send_image_data rover3 objective1 high_res -> m-send_image_data 5",
			"4 take_image rover3 waypoint2 objective1 camera2 high_res",
			"5 communicate_image_data rover3 lander1 objective1 high_res waypoint2 waypoint0",
		],
	)

	assert runner._extract_panda_method_trace(plan_text) == [
		{"method_name": "m-get_image_data", "task_args": ["objective1", "high_res"]},
		{"method_name": "m-calibrate_abs", "task_args": ["rover3", "camera2"]},
		{
			"method_name": "m-send_image_data",
			"task_args": ["rover3", "objective1", "high_res"],
		},
	]

def test_prioritise_guided_method_chunks_reorders_matching_grounded_method_first():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!send_image_data(rover1, OBJECTIVE, MODE) : true <-
	.print("runtime trace method flat ", "m-send_image_data", "|", rover1, "|", OBJECTIVE, "|", MODE);
	!communicate_image_data(rover1, lander1, OBJECTIVE, MODE, waypoint0, waypoint1).

+!send_image_data(rover3, OBJECTIVE, MODE) : true <-
	.print("runtime trace method flat ", "m-send_image_data", "|", rover3, "|", OBJECTIVE, "|", MODE);
	!communicate_image_data(rover3, lander1, OBJECTIVE, MODE, waypoint2, waypoint0).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-send_image_data",
				"task_args": ["rover3", "objective1", "high_res"],
			},
		],
	)

	assert rewritten.index("+!send_image_data(rover3, OBJECTIVE, MODE) : true <-") < rewritten.index(
		"+!send_image_data(rover1, OBJECTIVE, MODE) : true <-",
	)

def test_prioritise_guided_method_chunks_uses_action_path_to_break_internal_witness_ties():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!send_rock_data(rover1, WAYPOINT1) : true <-
	.print("runtime trace method flat ", "m-send_rock_data", "|", rover1, "|", WAYPOINT1);
	!communicate_rock_data(rover1, general, WAYPOINT1, waypoint5, waypoint1).

+!send_rock_data(rover1, WAYPOINT1) : true <-
	.print("runtime trace method flat ", "m-send_rock_data", "|", rover1, "|", WAYPOINT1);
	!communicate_rock_data(rover1, general, WAYPOINT1, waypoint0, waypoint1).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-send_rock_data",
				"task_args": ["rover1", "waypoint6"],
			},
		],
		[
			"communicate_rock_data(rover1,general,waypoint6,waypoint0,waypoint1)",
		],
	)

	assert "waypoint0, waypoint1" in rewritten
	assert "waypoint5, waypoint1" in rewritten
	assert rewritten.index("waypoint0, waypoint1") < rewritten.index("waypoint5, waypoint1")

def test_prioritise_guided_method_chunks_prefers_more_grounded_head_when_matches_tie():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!send_rock_data(rover1, WAYPOINT1) : true <-
	.print("runtime trace method flat ", "m-send_rock_data", "|", rover1, "|", WAYPOINT1);
	!communicate_rock_data(rover1, general, WAYPOINT1, waypoint0, waypoint1).

+!send_rock_data(rover1, waypoint6) : true <-
	.print("runtime trace method flat ", "m-send_rock_data", "|", rover1, "|", waypoint6);
	!communicate_rock_data(rover1, general, waypoint6, waypoint0, waypoint1).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-send_rock_data",
				"task_args": ["rover1", "waypoint6"],
			},
		],
		[
			"communicate_rock_data(rover1,general,waypoint6,waypoint0,waypoint1)",
		],
	)

	assert "+!send_rock_data(rover1, waypoint6) : true <-" in rewritten
	assert "+!send_rock_data(rover1, WAYPOINT1) : true <-" in rewritten
	assert rewritten.index("+!send_rock_data(rover1, waypoint6) : true <-") < rewritten.index(
		"+!send_rock_data(rover1, WAYPOINT1) : true <-",
	)

def test_prioritise_guided_method_chunks_prefers_grounded_current_invocation_over_earlier_generic_trace():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!send_rock_data(rover1, WAYPOINT1) : true <-
	.print("runtime trace method flat ", "m-send_rock_data", "|", rover1, "|", WAYPOINT1);
	!communicate_rock_data(rover1, general, WAYPOINT1, waypoint5, waypoint1).

+!send_rock_data(rover1, waypoint6) : true <-
	.print("runtime trace method flat ", "m-send_rock_data", "|", rover1, "|", waypoint6);
	!communicate_rock_data(rover1, general, waypoint6, waypoint0, waypoint1).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-send_rock_data",
				"task_args": ["rover1", "waypoint0"],
			},
			{
				"method_name": "m-send_rock_data",
				"task_args": ["rover1", "waypoint6"],
			},
		],
		[
			"communicate_rock_data(rover1,general,waypoint6,waypoint0,waypoint1)",
		],
	)

	assert rewritten.index("+!send_rock_data(rover1, waypoint6) : true <-") < rewritten.index(
		"+!send_rock_data(rover1, WAYPOINT1) : true <-",
	)

def test_prioritise_guided_method_chunks_prefers_simpler_base_case_over_recursive_via():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!get_to(VEHICLE, LOCATION1) : road(LOCATION2, LOCATION1) <-
	.print("runtime trace method flat ", "m-drive-to-via", "|", VEHICLE, "|", LOCATION1);
	!get_to(VEHICLE, LOCATION2);
	!drive(VEHICLE, LOCATION2, LOCATION1).

+!get_to(VEHICLE, LOCATION1) : at(VEHICLE, LOCATION1) <-
	.print("runtime trace method flat ", "m-i-am-there", "|", VEHICLE, "|", LOCATION1);
	!noop(VEHICLE, LOCATION1).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-drive-to-via",
				"task_args": ["truck-0", "city-loc-1"],
			},
		],
	)

	assert rewritten.index('+!get_to(VEHICLE, LOCATION1) : at(VEHICLE, LOCATION1) <-') < rewritten.index(
		'+!get_to(VEHICLE, LOCATION1) : road(LOCATION2, LOCATION1) <-',
	)

def test_prioritise_guided_method_chunks_prefers_direct_action_body_over_recursive_via():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!get_to("truck-0", "city-loc-2") : at("truck-0", "city-loc-1") & road("city-loc-1", "city-loc-2") <-
	.print("runtime trace method flat ", "m-drive-to-via", "|", "truck-0", "|", "city-loc-2");
	!get_to("truck-0", "city-loc-1");
	!drive("truck-0", "city-loc-1", "city-loc-2").

+!get_to("truck-0", "city-loc-2") : at("truck-0", "city-loc-1") & road("city-loc-1", "city-loc-2") <-
	.print("runtime trace method flat ", "m-drive-to", "|", "truck-0", "|", "city-loc-2");
	!drive("truck-0", "city-loc-1", "city-loc-2").

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-drive-to-via",
				"task_args": ["truck-0", "city-loc-2"],
			},
		],
		[
			"drive(truck-0,city-loc-1,city-loc-2)",
		],
	)

	assert rewritten.index('"m-drive-to", "|", "truck-0", "|", "city-loc-2"') < rewritten.index(
		'"m-drive-to-via", "|", "truck-0", "|", "city-loc-2"',
	)

def test_prioritise_guided_method_chunks_reorders_generic_chunk_when_body_is_only_partially_guided():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!calibrate_abs(rover0, camera4) : true <-
	.print("runtime trace method flat ", "m-calibrate_abs", "|", rover0, "|", camera4);
	!navigate_abs(rover0, WAYPOINT);
	!calibrate(rover0, camera4, objective3, WAYPOINT).

+!calibrate_abs(rover0, camera4) : true <-
	.print("runtime trace method flat ", "m-calibrate_abs", "|", rover0, "|", camera4);
	!navigate_abs(rover0, waypoint3);
	!calibrate(rover0, camera4, objective3, waypoint3).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-calibrate_abs",
				"task_args": ["rover0", "camera4"],
			},
		],
		[
			"calibrate(rover0,camera4,objective3,waypoint3)",
		],
	)

	assert "!calibrate(rover0, camera4, objective3, waypoint3)." in rewritten
	assert "!calibrate(rover0, camera4, objective3, WAYPOINT)." in rewritten
	assert rewritten.index("!calibrate(rover0, camera4, objective3, waypoint3).") < rewritten.index(
		"!calibrate(rover0, camera4, objective3, WAYPOINT).",
	)

def test_prioritise_guided_method_chunks_keeps_distinct_exact_action_signatures():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!navigate_abs(rover1, waypoint8) : true <-
	.print("runtime trace method flat ", "m-navigate_abs-4", "|", rover1, "|", waypoint8);
	!navigate(rover1, WAYPOINT2, waypoint6);
	!visit(waypoint6);
	!navigate(rover1, waypoint6, waypoint8);
	!unvisit(waypoint6).

+!navigate_abs(rover1, waypoint8) : true <-
	.print("runtime trace method flat ", "m-navigate_abs-4", "|", rover1, "|", waypoint8);
	!navigate(rover1, waypoint4, waypoint6);
	!visit(waypoint6);
	!navigate(rover1, waypoint6, waypoint8);
	!unvisit(waypoint6).

+!navigate_abs(rover1, waypoint8) : true <-
	.print("runtime trace method flat ", "m-navigate_abs-4", "|", rover1, "|", waypoint8);
	!navigate(rover1, waypoint5, waypoint6);
	!visit(waypoint6);
	!navigate(rover1, waypoint6, waypoint8);
	!unvisit(waypoint6).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-navigate_abs-4",
				"task_args": ["rover1", "waypoint8"],
			},
		],
		[
			"navigate(rover1,waypoint4,waypoint6)",
			"visit(waypoint6)",
			"navigate(rover1,waypoint6,waypoint8)",
			"unvisit(waypoint6)",
			"navigate(rover1,waypoint5,waypoint6)",
			"visit(waypoint6)",
			"navigate(rover1,waypoint6,waypoint8)",
			"unvisit(waypoint6)",
		],
	)

	assert rewritten.count("+!navigate_abs(rover1, waypoint8) : true <-") == 2
	assert "!navigate(rover1, waypoint4, waypoint6);" in rewritten
	assert "!navigate(rover1, waypoint5, waypoint6);" in rewritten
	assert "!navigate(rover1, WAYPOINT2, waypoint6);" not in rewritten

def test_prioritise_guided_method_chunks_preserves_root_task_fallbacks():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!get_soil_data(WAYPOINT) : true <-
	.print("runtime trace method flat ", "m-get_soil_data", "|", WAYPOINT);
	!sample_soil(rover0, rover0store, WAYPOINT).

+!get_soil_data(waypoint3) : true <-
	.print("runtime trace method flat ", "m-get_soil_data", "|", waypoint3);
	!sample_soil(rover0, rover0store, waypoint3).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-get_soil_data",
				"task_args": ["waypoint3"],
			},
		],
		[
			"sample_soil(rover0,rover0store,waypoint3)",
		],
		preserve_task_names={"get_soil_data"},
	)

	assert "+!get_soil_data(WAYPOINT) : true <-" in rewritten
	assert "+!get_soil_data(waypoint3) : true <-" in rewritten

def test_prioritise_guided_method_chunks_rejects_type_inconsistent_specialisation():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!get_soil_data(waypoint3) : object_type(waypoint3, waypoint) & object_type(rover0, rover) & object_type(rover0store, store) & can_traverse(rover0, WAYPOINT2, waypoint3) & at(rover0, WAYPOINT2) & visible(WAYPOINT2, waypoint3) <-
	.print("runtime trace method flat ", "m-get_soil_data", "|", waypoint3);
	!sample_soil(rover0, rover0store, waypoint3).

+!get_soil_data(waypoint3) : object_type(waypoint3, waypoint) & object_type(rover0, rover) & object_type(rover0store, store) & can_traverse(rover0, rover0store, waypoint3) & at(rover0, rover0store) & visible(rover0store, waypoint3) <-
	.print("runtime trace method flat ", "m-get_soil_data", "|", waypoint3);
	!sample_soil(rover0, rover0store, waypoint3).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-get_soil_data",
				"task_args": ["waypoint3"],
			},
		],
		[
			"sample_soil(rover0,rover0store,waypoint3)",
		],
		predicate_argument_types={
			("can_traverse", 0): {"rover"},
			("can_traverse", 1): {"waypoint"},
			("can_traverse", 2): {"waypoint"},
			("at", 0): {"rover"},
			("at", 1): {"waypoint"},
			("visible", 0): {"waypoint"},
			("visible", 1): {"waypoint"},
		},
		object_types={
			"rover0": "rover",
			"rover0store": "store",
			"waypoint3": "waypoint",
		},
	)

	assert "at(rover0, WAYPOINT2)" in rewritten
	assert "at(rover0, rover0store)" not in rewritten

def test_prioritise_guided_method_chunks_skips_body_only_witness_specialisation():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */

+!support_task(TARGET) : true <-
	.print("runtime trace method flat ", "m-support_task", "|", TARGET);
	!probe(WITNESS).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._prioritise_guided_method_chunks(
		agentspeak_code,
		[
			{
				"method_name": "m-support_task",
				"task_args": ["goal0"],
			},
		],
		[
			"probe(witness1)",
		],
	)

	assert "!probe(WITNESS)." in rewritten
	assert "!probe(witness1)." not in rewritten

def test_validate_prefers_guided_method_trace_when_provided(tmp_path, monkeypatch):
	stage6_dir = Path("src/stage6_jason_validation").resolve()
	runner = JasonRunner(stage6_dir=stage6_dir, timeout_seconds=1)
	jar_file = tmp_path / "jason-cli-all.jar"
	log_conf = tmp_path / "console-info.properties"
	jar_file.write_text("jar")
	log_conf.write_text("log")

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("/java23", 23))
	monkeypatch.setattr(runner, "_select_javac_binary", lambda java_bin: "/javac23")
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jar_file)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(
		runner,
		"_compile_environment_java",
		lambda **kwargs: (tmp_path / "Stage6PipelineEnvironment.class").write_text("class"),
	)
	monkeypatch.setattr(
		subprocess,
		"run",
		lambda *args, **kwargs: subprocess.CompletedProcess(
			args=args[0],
			returncode=0,
			stdout=(
				"runtime env ready\n"
				"runtime trace method trace_method(m_runtime_trace,wrong,arg)\n"
				"runtime env action success pick-up(a,b)\n"
				"execute start\n"
				"execute success\n"
			),
			stderr="",
		),
	)

	result = runner.validate(
		agentspeak_code="domain(test).\n/* Primitive Action Plans */\n+!pick_up(B1,B2):true<-\n\tpick_up(B1,B2).\n\n/* HTN Method Plans */",
		target_literals=[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
		],
		action_schemas=_sample_action_schemas(),
		seed_facts=("(clear a)", "(on a b)"),
		domain_name="blocksworld",
		output_dir=tmp_path,
		guided_method_trace=(
			{"method_name": "m_guided_trace", "task_args": ["a", "b"]},
		),
		skip_method_trace_reconstruction=True,
	)

	assert result.status == "success"
	assert result.method_trace == [{"method_name": "m_guided_trace", "task_args": ["a", "b"]}]
	validation_payload = json.loads((tmp_path / "jason_validation.json").read_text())
	assert validation_payload["method_trace"] == [
		{"method_name": "m_guided_trace", "task_args": ["a", "b"]},
	]
	assert validation_payload["consistency_checks"]["method_trace_reconstruction"]["message"] == (
		"skipped: authoritative guided hierarchical plan available"
	)

def test_validate_uses_runtime_method_trace_when_guided_prefix_continues(tmp_path, monkeypatch):
	stage6_dir = Path("src/stage6_jason_validation").resolve()
	runner = JasonRunner(stage6_dir=stage6_dir, timeout_seconds=1)
	jar_file = tmp_path / "jason-cli-all.jar"
	log_conf = tmp_path / "console-info.properties"
	jar_file.write_text("jar")
	log_conf.write_text("log")

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("/java23", 23))
	monkeypatch.setattr(runner, "_select_javac_binary", lambda java_bin: "/javac23")
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jar_file)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(
		runner,
		"_compile_environment_java",
		lambda **kwargs: (tmp_path / "Stage6PipelineEnvironment.class").write_text("class"),
	)
	monkeypatch.setattr(
		runner,
		"_run_consistency_checks",
		lambda **kwargs: {
			"success": True,
			"action_path_schema_replay": {"passed": True},
			"method_trace_reconstruction": {"passed": True},
		},
	)
	monkeypatch.setattr(
		subprocess,
		"run",
		lambda *args, **kwargs: subprocess.CompletedProcess(
			args=args[0],
			returncode=0,
			stdout=(
				"runtime env ready\n"
				"runtime env action success pick_up(a,b)\n"
				"runtime env action success pick_up(c,d)\n"
				"execute start\n"
				"execute success\n"
			),
			stderr=(
				'[agentspeak_generated] runtime trace method trace_method(m_guided_trace,"a","b")\n'
				"[agentspeak_generated] runtime trace method trace_method(m_runtime_trace,c,d)\n"
			),
		),
	)

	result = runner.validate(
		agentspeak_code="domain(test).\n/* Primitive Action Plans */\n+!pick_up(B1,B2):true<-\n\tpick_up(B1,B2).\n\n/* HTN Method Plans */",
		target_literals=[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
		],
		action_schemas=_sample_action_schemas(),
		seed_facts=("(clear a)", "(on a b)"),
		domain_name="blocksworld",
		output_dir=tmp_path,
		guided_action_path=("pick_up(a,b)",),
		guided_method_trace=(
			{"method_name": "m_guided_trace", "task_args": ["a", "b"]},
		),
		guided_continue_with_runtime_goal=True,
	)

	assert result.status == "success"
	assert result.method_trace == [
		{"method_name": "m_guided_trace", "task_args": ["a", "b"]},
		{"method_name": "m_runtime_trace", "task_args": ["c", "d"]},
	]

def test_validate_reorders_remaining_unordered_targets_after_guided_prefix(tmp_path, monkeypatch):
	stage6_dir = Path("src/stage6_jason_validation").resolve()
	runner = JasonRunner(stage6_dir=stage6_dir, timeout_seconds=1)
	jar_file = tmp_path / "jason-cli-all.jar"
	log_conf = tmp_path / "console-info.properties"
	jar_file.write_text("jar")
	log_conf.write_text("log")
	captured = {}

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("/java23", 23))
	monkeypatch.setattr(runner, "_select_javac_binary", lambda java_bin: "/javac23")
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jar_file)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(
		runner,
		"_build_runner_asl",
		lambda *args, **kwargs: "domain(test).",
	)
	monkeypatch.setattr(
		runner,
		"_compile_environment_java",
		lambda **kwargs: (tmp_path / "Stage6PipelineEnvironment.class").write_text("class"),
	)
	monkeypatch.setattr(
		runner,
		"_run_consistency_checks",
		lambda **kwargs: {
			"success": True,
			"action_path_schema_replay": {"passed": True},
			"method_trace_reconstruction": {"passed": True},
		},
	)

	def capture_ordering(**kwargs):
		captured["ordering"] = {
			"seed_facts": tuple(kwargs["seed_facts"]),
			"signatures": tuple(literal.to_signature() for literal in kwargs["target_literals"]),
			"target_id_offset": kwargs["target_id_offset"],
		}
		return {
			"target_ids": ["t3", "t2"],
			"target_signatures": ["goal(c)", "goal(b)"],
		}

	def capture_reordered(code, preferred_target_ids):
		captured["reordered_ids"] = list(preferred_target_ids)
		return code

	monkeypatch.setattr(
		runner,
		"_infer_unordered_target_execution_order",
		capture_ordering,
	)
	monkeypatch.setattr(
		runner,
		"_reorder_unordered_control_plan_blocks",
		capture_reordered,
	)
	monkeypatch.setattr(
		subprocess,
		"run",
		lambda *args, **kwargs: subprocess.CompletedProcess(
			args=args[0],
			returncode=0,
			stdout="runtime env ready\nexecute success\n",
			stderr="",
		),
	)

	runner.validate(
		agentspeak_code="domain(test).",
		target_literals=[
			HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="goal", args=("c",), is_positive=True, source_symbol=None),
		],
		method_library=HTNMethodLibrary(
			target_literals=[
				HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
				HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
				HTNLiteral(predicate="goal", args=("c",), is_positive=True, source_symbol=None),
			],
			target_task_bindings=[
				HTNTargetTaskBinding(target_literal="goal(a)", task_name="task_a"),
				HTNTargetTaskBinding(target_literal="goal(b)", task_name="task_b"),
				HTNTargetTaskBinding(target_literal="goal(c)", task_name="task_c"),
			],
		),
		action_schemas=_sample_action_schemas(),
		seed_facts=("(start)",),
		domain_name="blocksworld",
		output_dir=tmp_path,
		ordered_query_sequence=False,
		planning_domain=SimpleNamespace(name="test", predicates=()),
		guided_action_path=("pick_up(a,b)",),
		guided_continue_with_runtime_goal=True,
		guided_post_seed_facts=("(after_prefix)",),
		guided_completed_target_count=1,
	)

	assert captured["ordering"]["seed_facts"] == ("(after_prefix)",)
	assert captured["ordering"]["signatures"] == ("goal(b)", "goal(c)")
	assert captured["ordering"]["target_id_offset"] == 1
	assert captured["reordered_ids"] == ["t3", "t2"]

def test_validate_uses_conservative_reordering_for_direct_unordered_runtime(tmp_path, monkeypatch):
	stage6_dir = Path("src/stage6_jason_validation").resolve()
	runner = JasonRunner(stage6_dir=stage6_dir, timeout_seconds=1)
	jar_file = tmp_path / "jason-cli-all.jar"
	log_conf = tmp_path / "console-info.properties"
	jar_file.write_text("jar")
	log_conf.write_text("log")
	captured: dict[str, object] = {}

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("/java23", 23))
	monkeypatch.setattr(runner, "_select_javac_binary", lambda java_bin: "/javac23")
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jar_file)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(runner, "_build_runner_asl", lambda *args, **kwargs: "domain(test).")
	monkeypatch.setattr(
		runner,
		"_compile_environment_java",
		lambda **kwargs: (tmp_path / "Stage6PipelineEnvironment.class").write_text("class"),
	)
	monkeypatch.setattr(
		runner,
		"_run_consistency_checks",
		lambda **kwargs: {
			"success": True,
			"action_path_schema_replay": {"passed": True},
			"method_trace_reconstruction": {"passed": True},
		},
	)
	def capture_ordering(**kwargs):
		captured["prefer_query_order"] = kwargs["prefer_query_order"]
		captured["ordering"] = tuple(
			literal.to_signature()
			for literal in kwargs["target_literals"]
		)
		return {
			"target_ids": ["t2", "t1"],
			"target_signatures": ["goal(b)", "goal(a)"],
		}

	def capture_reordered(code, preferred_target_ids):
		captured["reordered_ids"] = list(preferred_target_ids)
		return code

	monkeypatch.setattr(runner, "_infer_unordered_target_execution_order", capture_ordering)
	monkeypatch.setattr(runner, "_reorder_unordered_control_plan_blocks", capture_reordered)
	monkeypatch.setattr(
		subprocess,
		"run",
		lambda *args, **kwargs: subprocess.CompletedProcess(
			args=args[0],
			returncode=0,
			stdout="runtime env ready\nexecute success\n",
			stderr="",
		),
	)

	result = runner.validate(
		agentspeak_code="domain(test).",
		target_literals=[
			HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
		],
		method_library=HTNMethodLibrary(
			target_literals=[
				HTNLiteral(predicate="goal", args=("a",), is_positive=True, source_symbol=None),
				HTNLiteral(predicate="goal", args=("b",), is_positive=True, source_symbol=None),
			],
		),
		action_schemas=_sample_action_schemas(),
		seed_facts=("(start)",),
		domain_name="blocksworld",
		output_dir=tmp_path,
		ordered_query_sequence=False,
		planning_domain=SimpleNamespace(name="test", predicates=()),
	)

	assert result.status == "success"
	assert captured["prefer_query_order"] is True
	assert captured["ordering"] == ("goal(a)", "goal(b)")
	assert captured["reordered_ids"] == ["t2", "t1"]

def test_action_path_schema_replay_detects_shared_resource_precondition_violation():
	runner = JasonRunner()
	check = runner._replay_action_path_against_schemas(
		action_path=["navigate(rover0, waypoint0, waypoint1)"],
		action_schemas=[
			{
				"functor": "navigate",
				"source_name": "navigate",
				"parameters": ["?r", "?from", "?to"],
				"precondition_clauses": [
					[
						{"predicate": "available", "args": ["?r"], "is_positive": True},
						{"predicate": "at", "args": ["?r", "?from"], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "at", "args": ["?r", "?from"], "is_positive": False},
					{"predicate": "at", "args": ["?r", "?to"], "is_positive": True},
				],
			},
		],
		seed_facts=["(at rover0 waypoint0)"],
	)

	assert check["passed"] is False
	assert check["failure_class"] == "action_path_precondition_violation"

def test_action_path_schema_replay_detects_store_lifecycle_violation():
	runner = JasonRunner()
	check = runner._replay_action_path_against_schemas(
		action_path=[
			"sample_soil(rover0, store0, waypoint1)",
			"sample_soil(rover0, store0, waypoint1)",
		],
		action_schemas=[
			{
				"functor": "sample_soil",
				"source_name": "sample_soil",
				"parameters": ["?r", "?s", "?wp"],
				"precondition_clauses": [
					[
						{"predicate": "available", "args": ["?r"], "is_positive": True},
						{"predicate": "at", "args": ["?r", "?wp"], "is_positive": True},
						{"predicate": "at_soil_sample", "args": ["?wp"], "is_positive": True},
						{"predicate": "empty", "args": ["?s"], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "full", "args": ["?s"], "is_positive": True},
					{"predicate": "empty", "args": ["?s"], "is_positive": False},
				],
			},
		],
		seed_facts=[
			"(available rover0)",
			"(at rover0 waypoint1)",
			"(at_soil_sample waypoint1)",
			"(empty store0)",
		],
	)

	assert check["passed"] is False
	assert check["failure_class"] == "action_path_precondition_violation"
	assert check["checked_steps"] == 1

def test_action_path_schema_replay_detects_calibration_lifecycle_violation():
	runner = JasonRunner()
	check = runner._replay_action_path_against_schemas(
		action_path=[
			"take_image(rover0, waypoint0, objective0, camera0, colour)",
			"take_image(rover0, waypoint0, objective0, camera0, colour)",
		],
		action_schemas=[
			{
				"functor": "take_image",
				"source_name": "take_image",
				"parameters": ["?r", "?wp", "?objective", "?camera", "?mode"],
				"precondition_clauses": [
					[
						{"predicate": "calibrated", "args": ["?camera", "?r"], "is_positive": True},
						{"predicate": "available", "args": ["?r"], "is_positive": True},
						{"predicate": "at", "args": ["?r", "?wp"], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "have_image", "args": ["?r", "?objective", "?mode"], "is_positive": True},
					{"predicate": "calibrated", "args": ["?camera", "?r"], "is_positive": False},
				],
			},
		],
		seed_facts=[
			"(calibrated camera0 rover0)",
			"(available rover0)",
			"(at rover0 waypoint0)",
		],
	)

	assert check["passed"] is False
	assert check["failure_class"] == "action_path_precondition_violation"
	assert check["checked_steps"] == 1

def test_method_trace_reconstruction_detects_missing_root_entries():
	if not OFFICIAL_BLOCKSWORLD_P01.exists():
		pytest.skip(f"Missing blocksworld problem file: {OFFICIAL_BLOCKSWORLD_P01}")

	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_put_on", ("BLOCK1", "BLOCK2"), False, ("on",), source_name="do_put_on"),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_do_put_on_noop",
				task_name="do_put_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	check = runner._check_method_trace_reconstruction(
		action_path=[],
		method_trace=[
			{"method_name": "m_do_put_on_noop", "task_args": ["b4", "b2"]},
			{"method_name": "m_do_put_on_noop", "task_args": ["b1", "b4"]},
		],
		method_library=method_library,
		problem_file=OFFICIAL_BLOCKSWORLD_P01,
	)

	assert check["passed"] is False
	assert check["failure_class"] == "method_trace_reconstruction_failed"

def test_extract_failed_goals_preserves_runtime_order():
	runner = JasonRunner()
	stdout = "\n".join(
		[
			"runtime goal failed fail_goal(hold_block,b4)",
			"runtime goal failed fail_goal(clear_top,b2)",
		],
	)

	assert runner._extract_failed_goals(stdout) == [
		"hold_block,b4",
		"clear_top,b2",
	]

def test_augment_method_trace_with_query_root_bridges_skips_when_trace_already_contains_bridge_tasks():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="query_root_1_do_on_table",
				parameters=("ARG1",),
				is_primitive=False,
				source_predicates=("ontable",),
				source_name="do_on_table",
			),
			HTNTask(
				name="dfa_step_q1_q2_ontable_b275",
				parameters=("ARG1",),
				is_primitive=False,
				source_predicates=("ontable",),
			),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_query_root_1_do_on_table_noop",
				task_name="query_root_1_do_on_table",
				parameters=("ARG1",),
				task_args=("ARG1",),
				context=(
					HTNLiteral("ontable", ("ARG1",), True, None),
				),
				subtasks=(),
				ordering=(),
			),
			HTNMethod(
				method_name="m_query_root_1_do_on_table_constructive_1",
				task_name="query_root_1_do_on_table",
				parameters=("ARG1",),
				task_args=("ARG1",),
				subtasks=(
					HTNMethodStep(
						"s1",
						"dfa_step_q1_q2_ontable_b275",
						("ARG1",),
						"compound",
					),
				),
				ordering=(),
			),
			HTNMethod(
				method_name="m_dfa_step_q1_q2_ontable_b275_constructive_1",
				task_name="dfa_step_q1_q2_ontable_b275",
				parameters=("ARG1",),
				task_args=("ARG1",),
				subtasks=(),
				ordering=(),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	trace = runner._augment_method_trace_with_query_root_bridges(
		method_trace=[
			{"method_name": "m_query_root_1_do_on_table_noop", "task_args": ["b275"]},
			{"method_name": "m_dfa_step_q1_q2_ontable_b275_constructive_1", "task_args": ["b275"]},
		],
		method_library=method_library,
		problem_file=OFFICIAL_BLOCKSWORLD_P01,
	)

	assert trace == [
		{"method_name": "m_query_root_1_do_on_table_noop", "task_args": ["b275"]},
		{"method_name": "m_dfa_step_q1_q2_ontable_b275_constructive_1", "task_args": ["b275"]},
	]

def test_render_failure_handlers_canonicalise_hddl_style_task_parameters():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_put_on", ("?x", "?y"), False, ("on",)),
		],
		primitive_tasks=[],
		methods=[],
		target_literals=[],
		target_task_bindings=[],
	)

	lines = runner._render_failure_handlers(method_library)
	rendered = "\n".join(lines)

	assert "-!do_put_on(X, Y) : true <-" in rendered
	assert 'fail_goal(do_put_on, X, Y)' in rendered
	assert "+blocked_runtime_goal(do_put_on, X, Y)" in rendered

def test_render_failure_handlers_sanitise_hyphenated_task_functors():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("empty-store", ("?store", "?rover"), False, ("empty",)),
		],
		primitive_tasks=[],
		methods=[],
		target_literals=[],
		target_task_bindings=[],
	)

	lines = runner._render_failure_handlers(method_library)
	rendered = "\n".join(lines)

	assert '-!empty_store(STORE, ROVER) : true <-' in rendered
	assert 'fail_goal("empty-store", STORE, ROVER)' in rendered
	assert '+blocked_runtime_goal(empty_store, STORE, ROVER)' in rendered

def test_render_failure_handlers_skip_target_bound_compound_tasks():
	runner = JasonRunner()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("get_rock_data", ("?waypoint",), False, ("communicated_rock_data",)),
			HTNTask("navigate_abs", ("?rover", "?to"), False, ("at",)),
		],
		primitive_tasks=[],
		methods=[],
		target_literals=[],
		target_task_bindings=[
			HTNTargetTaskBinding(
				target_literal="communicated_rock_data(waypoint7)",
				task_name="get_rock_data",
			),
		],
	)

	lines = runner._render_failure_handlers(method_library)
	rendered = "\n".join(lines)

	assert "-!get_rock_data(WAYPOINT) : true <-" not in rendered
	assert "+blocked_runtime_goal(get_rock_data, WAYPOINT)" not in rendered
	assert "-!navigate_abs(ROVER, TO) : true <-" in rendered
	assert "+blocked_runtime_goal(navigate_abs, ROVER, TO)" in rendered

def test_render_method_trace_statement_quotes_hyphenated_method_names():
	runner = JasonRunner()
	method = HTNMethod(
		method_name="m-empty-store-1",
		task_name="empty-store",
		parameters=("STORE", "ROVER"),
		context=(),
		subtasks=(),
	)

	line = runner._render_method_trace_statement(
		method,
		"+!empty_store(STORE, ROVER) : empty(STORE) <-",
	)

	assert '"runtime trace method flat ", "m-empty-store-1", "|", STORE, "|", ROVER' in line

def test_extract_method_trace_strips_quotes_from_method_names():
	runner = JasonRunner()
	stdout = 'runtime trace method trace_method("m-empty-store-1", rover0store, rover0)\n'

	trace = runner._extract_method_trace(stdout)

	assert trace == [
		{
			"method_name": "m-empty-store-1",
			"task_args": ["rover0store", "rover0"],
		},
	]

def test_extract_method_trace_accepts_flat_runtime_trace_payloads():
	runner = JasonRunner()
	stdout = (
		"[agent] runtime trace method flat m-empty-store-1|rover0store|rover0\n"
	)

	trace = runner._extract_method_trace(stdout)

	assert trace == [
		{
			"method_name": "m-empty-store-1",
			"task_args": ["rover0store", "rover0"],
		},
	]

def test_instrument_method_plans_skips_preinstrumented_code():
	runner = JasonRunner()
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
	agentspeak_code = (
		"/* HTN Method Plans */\n"
		"+!empty_store(STORE, ROVER) : empty(STORE) <-\n"
		'\t.print("runtime trace method ", trace_method("m-empty-store-1", STORE, ROVER));\n'
		"\ttrue.\n\n"
		"/* DFA Transition Wrappers */\n"
	)

	instrumented = runner._instrument_method_plans(agentspeak_code, method_library)

	assert instrumented == agentspeak_code

def test_combine_process_output_accepts_byte_streams():
	runner = JasonRunner()

	stdout = runner._normalise_process_output(b"execute start\n")
	stderr = runner._normalise_process_output(b"runtime env ready\n")

	assert runner._combine_process_output(stdout, stderr) == (
		"execute start\nruntime env ready\n"
	)

def test_validate_fails_when_environment_ready_marker_missing(monkeypatch, tmp_path):
	stage6_dir = tmp_path / "stage6"
	stage6_dir.mkdir()
	runner = JasonRunner(stage6_dir=stage6_dir, timeout_seconds=1)
	jar_file = tmp_path / "jason-cli-all.jar"
	log_conf = tmp_path / "console-info.properties"
	jar_file.write_text("jar")
	log_conf.write_text("log")

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("/java23", 23))
	monkeypatch.setattr(runner, "_select_javac_binary", lambda java_bin: "/javac23")
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jar_file)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(
		runner,
		"_compile_environment_java",
		lambda **kwargs: (tmp_path / "Stage6PipelineEnvironment.class").write_text("class"),
	)
	monkeypatch.setattr(
		subprocess,
		"run",
		lambda *args, **kwargs: subprocess.CompletedProcess(
			args=args[0],
			returncode=0,
			stdout="execute start\nexecute success\n",
			stderr="",
		),
	)

	with pytest.raises(JasonValidationError) as exc_info:
		runner.validate(
			agentspeak_code="domain(test).",
			target_literals=[],
			action_schemas=_sample_action_schemas(),
			domain_name="blocksworld",
			output_dir=tmp_path,
		)

	assert "environment adapter validation failed" in str(exc_info.value)
	assert exc_info.value.metadata["environment_adapter"]["success"] is False

def test_validate_timeout_is_reported(monkeypatch, tmp_path):
	stage6_dir = tmp_path / "stage6"
	stage6_dir.mkdir()
	runner = JasonRunner(stage6_dir=stage6_dir, timeout_seconds=1)
	jar_file = tmp_path / "jason-cli-all.jar"
	log_conf = tmp_path / "console-info.properties"
	jar_file.write_text("jar")
	log_conf.write_text("log")

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("/java23", 23))
	monkeypatch.setattr(runner, "_select_javac_binary", lambda java_bin: "/javac23")
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jar_file)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(
		runner,
		"_compile_environment_java",
		lambda **kwargs: (tmp_path / "Stage6PipelineEnvironment.class").write_text("class"),
	)

	def raise_timeout(*args, **kwargs):
		raise subprocess.TimeoutExpired(
			cmd=args[0],
			timeout=1,
			output="runtime env ready\nexecute start\n",
			stderr="timeout",
		)

	monkeypatch.setattr(subprocess, "run", raise_timeout)

	with pytest.raises(JasonValidationError) as exc_info:
		runner.validate(
			agentspeak_code="domain(test).",
			target_literals=[],
			action_schemas=_sample_action_schemas(),
			domain_name="blocksworld",
			output_dir=tmp_path,
		)

	assert "timeout" in str(exc_info.value)
	assert exc_info.value.metadata["timed_out"] is True
	assert (tmp_path / "jason_validation.json").exists()

def test_environment_java_source_uses_single_world_set_for_negative_semantics():
	runner = JasonRunner()
	java_source = runner._build_environment_java_source(
		action_schemas=[
			{
				"functor": "seal",
				"parameters": ["?x"],
				"preconditions": [
					{
						"predicate": "clear",
						"args": ["?x"],
						"is_positive": False,
					},
				],
				"precondition_clauses": [
					[
						{
							"predicate": "clear",
							"args": ["?x"],
							"is_positive": False,
						},
					],
				],
				"effects": [
					{
						"predicate": "clear",
						"args": ["?x"],
						"is_positive": False,
					},
				],
			},
		],
		seed_facts=["(not (clear a))"],
		target_literals=[
			HTNLiteral(
					predicate="clear",
					args=("a",),
					is_positive=False,
					source_symbol=None,
					negation_mode="strong",
				),
			],
	)

	assert 'new Pattern("clear", false, new String[]{"?x"})' in java_source
	assert "Pattern[][] preconditionClauses" in java_source
	assert "for (Pattern[] clause : preconditionClauses)" in java_source
	assert "private final Set<String> strongNegatives" not in java_source
	assert "holds = !world.contains(grounded);" in java_source
	assert "strongNegatives" not in java_source

def test_environment_java_source_supports_disjunctive_precondition_clauses():
	runner = JasonRunner()
	java_source = runner._build_environment_java_source(
		action_schemas=[
			{
				"functor": "probe",
				"parameters": ["?x"],
				"precondition_clauses": [
					[
						{
							"predicate": "clear",
							"args": ["?x"],
							"is_positive": True,
						},
					],
					[
						{
							"predicate": "holding",
							"args": ["?x"],
							"is_positive": True,
						},
					],
				],
				"effects": [],
			},
		],
		seed_facts=[],
		target_literals=[],
	)

	assert 'new Pattern[][]{new Pattern[]{new Pattern("clear", true, new String[]{"?x"})}, ' in java_source
	assert 'new Pattern[]{new Pattern("holding", true, new String[]{"?x"})}}' in java_source

def test_environment_java_source_treats_true_as_builtin_noop():
	runner = JasonRunner()
	java_source = runner._build_environment_java_source(
		action_schemas=_sample_action_schemas(),
		seed_facts=[],
		target_literals=[],
	)

	assert 'if ("true".equals(action.getFunctor()) && action.getArity() == 0)' in java_source

def test_ground_local_witness_method_plans_specialises_transport_like_helper_clause():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_in(PACKAGE, VEHICLE) : object_type(PACKAGE, package) & object_type(VEHICLE, vehicle) & object_type(LOCATION, location) & object_type(CAPACITY_NUMBER1, capacity_number) & object_type(CAPACITY_NUMBER2, capacity_number) & at(VEHICLE, LOCATION) & capacity_predecessor(CAPACITY_NUMBER1, CAPACITY_NUMBER2) <-
	.print("runtime trace method flat ", "m_helper_in_constructive_1", "|", PACKAGE, "|", VEHICLE);
	!helper_at(VEHICLE, LOCATION);
	!helper_capacity(VEHICLE, CAPACITY_NUMBER2);
	!pick_up(VEHICLE, LOCATION, PACKAGE, CAPACITY_NUMBER1, CAPACITY_NUMBER2).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._ground_local_witness_method_plans(
		agentspeak_code,
		seed_facts=[
			"(at truck-0 city-loc-2)",
			"(capacity-predecessor capacity-0 capacity-1)",
		],
		runtime_objects=(
			"package-0",
			"truck-0",
			"city-loc-2",
			"capacity-0",
			"capacity-1",
		),
		object_types={
			"package-0": "package",
			"truck-0": "truck",
			"city-loc-2": "location",
			"capacity-0": "capacity_number",
			"capacity-1": "capacity_number",
		},
		type_parent_map={
			"package": "locatable",
			"truck": "vehicle",
			"vehicle": "locatable",
			"locatable": "object",
			"location": "object",
			"capacity_number": "object",
		},
	)

	assert '+!helper_in(PACKAGE, "truck-0") :' in rewritten
	assert '!helper_at("truck-0", "city-loc-2");' in rewritten
	assert '!helper_capacity("truck-0", "capacity-1");' in rewritten
	assert '!pick_up("truck-0", "city-loc-2", PACKAGE, "capacity-0", "capacity-1").' in rewritten
	assert rewritten.count("+!helper_in(") >= 2

def test_ground_local_witness_method_plans_preserves_clause_without_groundable_local_types():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_in(PACKAGE, VEHICLE) : object_type(PACKAGE, package) & object_type(VEHICLE, vehicle) <-
	.print("runtime trace method flat ", "m_helper_in_constructive_1", "|", PACKAGE, "|", VEHICLE);
	!helper_at(VEHICLE, LOCATION).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._ground_local_witness_method_plans(
		agentspeak_code,
		seed_facts=["(at truck-0 city-loc-2)"],
		runtime_objects=("package-0", "truck-0", "city-loc-2"),
		object_types={
			"package-0": "package",
			"truck-0": "truck",
			"city-loc-2": "location",
		},
		type_parent_map={
			"package": "locatable",
			"truck": "vehicle",
			"vehicle": "locatable",
			"locatable": "object",
			"location": "object",
		},
	)

	assert rewritten == agentspeak_code

def test_ground_local_witness_method_plans_keeps_noop_body_specialisation_when_first_pass_unchanged():
	runner = JasonRunner()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(LOCATABLE, LOCATION1) : at(LOCATABLE, LOCATION1) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", LOCATABLE, "|", LOCATION1);
	true.

+!helper_in(PACKAGE, VEHICLE) : object_type(PACKAGE, package) & object_type(VEHICLE, vehicle) <-
	.print("runtime trace method flat ", "m_helper_in_constructive_1", "|", PACKAGE, "|", VEHICLE);
	!helper_at(PACKAGE, LOCATION).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = runner._ground_local_witness_method_plans(
		agentspeak_code,
		seed_facts=["(at package-0 city-loc-1)"],
		runtime_objects=("package-0", "truck-0", "city-loc-1"),
		object_types={
			"package-0": "package",
			"truck-0": "truck",
			"city-loc-1": "location",
		},
		type_parent_map={
			"package": "locatable",
			"truck": "vehicle",
			"vehicle": "locatable",
			"locatable": "object",
			"location": "object",
		},
	)

	assert rewritten != agentspeak_code
	assert '+!helper_in("package-0", VEHICLE) :' in rewritten
	assert '!helper_at("package-0", "city-loc-1").' in rewritten

def test_order_runtime_method_plan_chunks_prefers_later_body_noop_support():
	runner = JasonRunner()
	fact_index, type_domains = runner._runtime_fact_index_for_local_witness_grounding(
		seed_facts=[
			"(at package-0 city-loc-1)",
			"(at truck-0 city-loc-2)",
			"(capacity truck-0 capacity-1)",
			"(capacity-predecessor capacity-0 capacity-1)",
		],
		runtime_objects=("package-0", "truck-0", "city-loc-1", "city-loc-2", "capacity-1"),
		object_types={
			"package-0": "package",
			"truck-0": "truck",
			"city-loc-1": "location",
			"city-loc-2": "location",
			"capacity-1": "capacity_number",
		},
		type_parent_map={
			"package": "locatable",
			"truck": "vehicle",
			"vehicle": "locatable",
			"locatable": "object",
			"location": "object",
			"capacity_number": "object",
		},
	)
	chunks = [
		"\n".join(
			[
				'+!helper_at(LOCATABLE, LOCATION1) : at(LOCATABLE, LOCATION1) <-',
				'\t.print("runtime trace method flat ", "m_helper_at_noop", "|", LOCATABLE, "|", LOCATION1);',
				"\ttrue.",
			],
		),
		"\n".join(
			[
				'+!helper_capacity(VEHICLE, CAPACITY_NUMBER1) : capacity(VEHICLE, CAPACITY_NUMBER1) <-',
				'\t.print("runtime trace method flat ", "m_helper_capacity_noop", "|", VEHICLE, "|", CAPACITY_NUMBER1);',
				"\ttrue.",
			],
		),
		"\n".join(
			[
				'+!helper_in(PACKAGE, VEHICLE) : object_type(PACKAGE, package) & object_type(VEHICLE, vehicle) & object_type("city-loc-2", location) & object_type("capacity-1", capacity_number) & capacity_predecessor("capacity-0", "capacity-1") <-',
				'\t.print("runtime trace method flat ", "m_helper_in_constructive_1", "|", PACKAGE, "|", VEHICLE);',
				'\t!helper_at(VEHICLE, "city-loc-2");',
				'\t!helper_at(PACKAGE, "city-loc-2");',
				'\t!helper_capacity(VEHICLE, "capacity-1").',
			],
		),
		"\n".join(
			[
				'+!helper_in(PACKAGE, VEHICLE) : object_type(PACKAGE, package) & object_type(VEHICLE, vehicle) & object_type("city-loc-1", location) & object_type("capacity-1", capacity_number) & capacity_predecessor("capacity-0", "capacity-1") <-',
				'\t.print("runtime trace method flat ", "m_helper_in_constructive_1", "|", PACKAGE, "|", VEHICLE);',
				'\t!helper_at(VEHICLE, "city-loc-1");',
				'\t!helper_at(PACKAGE, "city-loc-1");',
				'\t!helper_capacity(VEHICLE, "capacity-1").',
			],
		),
	]

	ordered = runner._order_runtime_method_plan_chunks(
		chunks,
		fact_index=fact_index,
		type_domains=type_domains,
	)
	helper_in_chunks = [
		chunk
		for chunk in ordered
		if chunk.startswith("+!helper_in(")
	]

	assert '"city-loc-2"' in helper_in_chunks[0]
	assert '"city-loc-1"' in helper_in_chunks[1]

def test_order_runtime_method_plan_chunks_prefers_specific_constructive_contexts():
	runner = JasonRunner()
	fact_index, type_domains = runner._runtime_fact_index_for_local_witness_grounding(
		seed_facts=["(at truck-0 city-loc-0)", "(road city-loc-0 city-loc-1)"],
		runtime_objects=("truck-0", "city-loc-0", "city-loc-1", "city-loc-2"),
		object_types={
			"truck-0": "truck",
			"city-loc-0": "location",
			"city-loc-1": "location",
			"city-loc-2": "location",
		},
		type_parent_map={
			"truck": "vehicle",
			"vehicle": "locatable",
			"locatable": "object",
			"location": "object",
		},
	)
	chunks = [
		"\n".join(
			[
				'+!helper_at(LOCATABLE, LOCATION1) : road(LOCATION2, LOCATION1) & object_type(LOCATABLE, vehicle) & object_type(LOCATION1, location) & object_type(LOCATION2, location) & LOCATION2 \\== LOCATION1 <-',
				'\t.print("runtime trace method flat ", "m_helper_at_constructive_1", "|", LOCATABLE, "|", LOCATION1);',
				"\t!helper_at(LOCATABLE, LOCATION2);",
				"\t!drive(LOCATABLE, LOCATION2, LOCATION1).",
			],
		),
		"\n".join(
			[
				'+!helper_at(LOCATABLE, LOCATION1) : road(LOCATION2, LOCATION1) & object_type(LOCATABLE, vehicle) & object_type(LOCATION1, location) & object_type(LOCATION2, location) & LOCATION2 \\== LOCATION1 & at(LOCATABLE, LOCATION2) <-',
				'\t.print("runtime trace method flat ", "m_helper_at_constructive_1", "|", LOCATABLE, "|", LOCATION1);',
				"\t!helper_at(LOCATABLE, LOCATION2);",
				"\t!drive(LOCATABLE, LOCATION2, LOCATION1).",
			],
		),
		"\n".join(
			[
				'+!helper_at("truck-0", "city-loc-1") : road("city-loc-0", "city-loc-1") & object_type("truck-0", vehicle) & object_type("city-loc-1", location) & object_type("city-loc-0", location) & "city-loc-0" \\== "city-loc-1" <-',
				'\t.print("runtime trace method flat ", "m_helper_at_constructive_1", "|", "truck-0", "|", "city-loc-1");',
				'\t!helper_at("truck-0", "city-loc-0");',
				'\t!drive("truck-0", "city-loc-0", "city-loc-1").',
			],
		),
	]

	ordered = runner._order_runtime_method_plan_chunks(
		chunks,
		fact_index=fact_index,
		type_domains=type_domains,
	)

	assert ordered[2].startswith("+!helper_at(LOCATABLE, LOCATION1) :")
	assert "& at(LOCATABLE, LOCATION2)" in ordered[2]
	assert ordered[1].startswith('+!helper_at("truck-0", "city-loc-1") :')

def test_order_runtime_method_plan_chunks_prefers_grounded_context_witnesses():
	runner = JasonRunner()
	fact_index, type_domains = runner._runtime_fact_index_for_local_witness_grounding(
		seed_facts=[
			"(at package-0 city-loc-3)",
			"(at truck-0 city-loc-3)",
			"(capacity truck-0 capacity-2)",
			"(capacity-predecessor capacity-0 capacity-1)",
			"(capacity-predecessor capacity-1 capacity-2)",
		],
		runtime_objects=(
			"package-0",
			"truck-0",
			"city-loc-3",
			"capacity-0",
			"capacity-1",
			"capacity-2",
		),
		object_types={
			"package-0": "package",
			"truck-0": "truck",
			"city-loc-3": "location",
			"capacity-0": "capacity_number",
			"capacity-1": "capacity_number",
			"capacity-2": "capacity_number",
		},
		type_parent_map={
			"package": "locatable",
			"truck": "vehicle",
			"vehicle": "locatable",
			"locatable": "object",
			"location": "object",
			"capacity_number": "object",
		},
	)
	chunks = [
		"\n".join(
			[
				'+!helper_in("package-0", "truck-0") : object_type("package-0", package) & object_type("truck-0", vehicle) & object_type("city-loc-3", location) & object_type(CAPACITY_NUMBER1, capacity_number) & object_type(CAPACITY_NUMBER2, capacity_number) & at("truck-0", "city-loc-3") & capacity_predecessor(CAPACITY_NUMBER1, CAPACITY_NUMBER2) <-',
				'\t.print("runtime trace method flat ", "m_helper_in_constructive_1", "|", "package-0", "|", "truck-0");',
				'\t!helper_at("truck-0", "city-loc-3");',
				'\t!helper_at("package-0", "city-loc-3");',
				'\t!helper_capacity("truck-0", CAPACITY_NUMBER2).',
			],
		),
		"\n".join(
			[
				'+!helper_in("package-0", "truck-0") : object_type("package-0", package) & object_type("truck-0", vehicle) & object_type("city-loc-3", location) & object_type(CAPACITY_NUMBER1, capacity_number) & object_type("capacity-2", capacity_number) & at("truck-0", "city-loc-3") & capacity_predecessor(CAPACITY_NUMBER1, "capacity-2") <-',
				'\t.print("runtime trace method flat ", "m_helper_in_constructive_1", "|", "package-0", "|", "truck-0");',
				'\t!helper_at("truck-0", "city-loc-3");',
				'\t!helper_at("package-0", "city-loc-3");',
				'\t!helper_capacity("truck-0", "capacity-2").',
			],
		),
	]

	ordered = runner._order_runtime_method_plan_chunks(
		chunks,
		fact_index=fact_index,
		type_domains=type_domains,
	)

	assert 'object_type("capacity-2", capacity_number)' in ordered[1]
	assert '!helper_capacity("truck-0", "capacity-2").' in ordered[1]

def test_order_runtime_method_plan_chunks_prefers_body_pairs_supported_by_current_facts():
	runner = JasonRunner()
	fact_index, type_domains = runner._runtime_fact_index_for_local_witness_grounding(
		seed_facts=[
			"(at carrier-0 place-0)",
			"(state carrier-0 level-2)",
			"(link level-0 level-1)",
			"(link level-1 level-2)",
		],
		runtime_objects=(
			"item-0",
			"carrier-0",
			"place-0",
			"target-0",
			"level-0",
			"level-1",
			"level-2",
		),
		object_types={
			"item-0": "item",
			"carrier-0": "carrier",
			"place-0": "place",
			"target-0": "place",
			"level-0": "level",
			"level-1": "level",
			"level-2": "level",
		},
		type_parent_map={
			"item": "object",
			"carrier": "object",
			"place": "object",
			"level": "object",
		},
	)
	chunks = [
		"\n".join(
			[
				'+!advance(ITEM, "target-0") : object_type(ITEM, item) & object_type("carrier-0", carrier) & object_type("target-0", place) & object_type("level-0", level) & object_type("level-1", level) & link("level-0", "level-1") & at("carrier-0", "place-0") <-',
				'\t.print("runtime trace method flat ", "m_advance_constructive", "|", ITEM, "|", "target-0");',
				'\t!support(ITEM, "carrier-0");',
				'\t!restore("carrier-0", "target-0", ITEM, "level-0", "level-1").',
			],
		),
		"\n".join(
			[
				'+!advance(ITEM, "target-0") : object_type(ITEM, item) & object_type("carrier-0", carrier) & object_type("target-0", place) & object_type("level-1", level) & object_type("level-2", level) & link("level-1", "level-2") & at("carrier-0", "place-0") <-',
				'\t.print("runtime trace method flat ", "m_advance_constructive", "|", ITEM, "|", "target-0");',
				'\t!support(ITEM, "carrier-0");',
				'\t!restore("carrier-0", "target-0", ITEM, "level-1", "level-2").',
			],
		),
	]

	ordered = runner._order_runtime_method_plan_chunks(
		chunks,
		fact_index=fact_index,
		type_domains=type_domains,
	)

	assert '!restore("carrier-0", "target-0", ITEM, "level-1", "level-2").' in ordered[1]

def test_promote_body_no_ancestor_guards_to_context_preserves_body_guard():
	runner = JasonRunner()
	chunk = "\n".join(
		[
			"+!helper_at(LOCATABLE, LOCATION1) : road(LOCATION2, LOCATION1) <-",
			"\tpipeline.no_ancestor_goal(helper_at, LOCATABLE, LOCATION2);",
			"\t!helper_at(LOCATABLE, LOCATION2);",
			"\t!drive(LOCATABLE, LOCATION2, LOCATION1).",
		],
	)

	rewritten = runner._promote_body_no_ancestor_guards_to_context([chunk])

	assert len(rewritten) == 1
	assert (
		"+!helper_at(LOCATABLE, LOCATION1) : "
		"road(LOCATION2, LOCATION1) & "
		"pipeline.no_ancestor_goal(helper_at, LOCATABLE, LOCATION2) <-"
	) in rewritten[0]
	assert "\tpipeline.no_ancestor_goal(helper_at, LOCATABLE, LOCATION2);" in rewritten[0]

def test_specialise_chunks_from_noop_prefix_contexts_adds_guarded_variants():
	runner = JasonRunner()
	chunks = [
		"\n".join(
			[
				"+!helper_in(ITEM, CARRIER) : in(ITEM, CARRIER) <-",
				'\t.print("runtime trace method flat ", "m_helper_in_noop", "|", ITEM, "|", CARRIER);',
				"\ttrue.",
			],
		),
		"\n".join(
			[
				"+!helper_capacity(CARRIER, LEVEL) : capacity(CARRIER, LEVEL) <-",
				'\t.print("runtime trace method flat ", "m_helper_capacity_noop", "|", CARRIER, "|", LEVEL);',
				"\ttrue.",
			],
		),
		"\n".join(
			[
				"+!deliver(ITEM, TARGET) : capacity_predecessor(LOW, HIGH) <-",
				'\t.print("runtime trace method flat ", "m_deliver", "|", ITEM, "|", TARGET);',
				"\t!helper_in(ITEM, CARRIER);",
				"\t!helper_capacity(CARRIER, LOW);",
				"\t!helper_at(CARRIER, TARGET);",
				"\t!drop(CARRIER, TARGET, ITEM, LOW, HIGH).",
			],
		),
	]

	specialised = runner._specialise_chunks_from_noop_prefix_contexts(
		chunks,
		max_candidates_per_chunk=16,
	)

	assert any(
		chunk.startswith(
			"+!deliver(ITEM, TARGET) : "
			"capacity_predecessor(LOW, HIGH) & "
			"in(ITEM, CARRIER) & "
			"capacity(CARRIER, LOW) <-",
		)
		for chunk in specialised
	)
	assert not any(
		chunk.startswith(
			"+!deliver(ITEM, TARGET) : "
			"capacity_predecessor(LOW, HIGH) & "
			"in(ITEM, CARRIER) <-",
		)
		for chunk in specialised
	)
	assert specialised[-1] == chunks[-1]

def test_specialise_chunks_from_noop_body_support_binds_head_variables_from_support_world():
	runner = JasonRunner()
	fact_index, type_domains = runner._runtime_fact_index_for_local_witness_grounding(
		seed_facts=["(at truck-0 city-loc-0)"],
		runtime_objects=("truck-0", "city-loc-0", "city-loc-1"),
		object_types={
			"truck-0": "truck",
			"city-loc-0": "location",
			"city-loc-1": "location",
		},
		type_parent_map={
			"truck": "vehicle",
			"vehicle": "locatable",
			"locatable": "object",
			"location": "object",
		},
	)
	chunks = [
		"\n".join(
			[
				'+!helper_at(LOCATABLE, LOCATION1) : at(LOCATABLE, LOCATION1) <-',
				'\t.print("runtime trace method flat ", "m_helper_at_noop", "|", LOCATABLE, "|", LOCATION1);',
				"\ttrue.",
			],
		),
		"\n".join(
			[
				'+!helper_at(LOCATABLE, "city-loc-1") : road("city-loc-0", "city-loc-1") & object_type(LOCATABLE, vehicle) & object_type("city-loc-1", location) & object_type("city-loc-0", location) <-',
				'\t.print("runtime trace method flat ", "m_helper_at_constructive_1", "|", LOCATABLE, "|", "city-loc-1");',
				'\t!helper_at(LOCATABLE, "city-loc-0");',
				'\t!drive(LOCATABLE, "city-loc-0", "city-loc-1").',
			],
		),
	]

	specialised = runner._specialise_chunks_from_noop_body_support(
		chunks,
		fact_index=fact_index,
		type_domains=type_domains,
		max_candidates_per_chunk=16,
	)

	assert any(
		chunk.startswith('+!helper_at("truck-0", "city-loc-1") :')
		for chunk in specialised
	)

def test_specialise_chunks_from_noop_body_support_combines_compatible_body_bindings():
	runner = JasonRunner()
	fact_index, type_domains = runner._runtime_fact_index_for_local_witness_grounding(
		seed_facts=[
			"(at package-0 city-loc-1)",
			"(at package-1 city-loc-2)",
			"(at truck-0 city-loc-2)",
			"(capacity truck-0 capacity-1)",
			"(capacity-predecessor capacity-0 capacity-1)",
		],
		runtime_objects=(
			"package-0",
			"package-1",
			"truck-0",
			"city-loc-1",
			"city-loc-2",
			"capacity-1",
		),
		object_types={
			"package-0": "package",
			"package-1": "package",
			"truck-0": "truck",
			"city-loc-1": "location",
			"city-loc-2": "location",
			"capacity-1": "capacity_number",
		},
		type_parent_map={
			"package": "locatable",
			"truck": "vehicle",
			"vehicle": "locatable",
			"locatable": "object",
			"location": "object",
			"capacity_number": "object",
		},
	)
	chunks = [
		"\n".join(
			[
				"+!helper_at(LOCATABLE, LOCATION1) : at(LOCATABLE, LOCATION1) <-",
				'\t.print("runtime trace method flat ", "m_helper_at_noop", "|", LOCATABLE, "|", LOCATION1);',
				"\ttrue.",
			],
		),
		"\n".join(
			[
				"+!helper_capacity(VEHICLE, CAPACITY_NUMBER1) : capacity(VEHICLE, CAPACITY_NUMBER1) <-",
				'\t.print("runtime trace method flat ", "m_helper_capacity_noop", "|", VEHICLE, "|", CAPACITY_NUMBER1);',
				"\ttrue.",
			],
		),
		"\n".join(
			[
				"+!helper_in(PACKAGE, VEHICLE) : object_type(PACKAGE, package) & object_type(VEHICLE, vehicle) & object_type(LOCATION, location) & object_type(CAPACITY_NUMBER1, capacity_number) & object_type(CAPACITY_NUMBER2, capacity_number) & capacity_predecessor(CAPACITY_NUMBER1, CAPACITY_NUMBER2) <-",
				'\t.print("runtime trace method flat ", "m_helper_in_constructive_1", "|", PACKAGE, "|", VEHICLE);',
				"\t!helper_at(VEHICLE, LOCATION);",
				"\t!helper_at(PACKAGE, LOCATION);",
				"\t!helper_capacity(VEHICLE, CAPACITY_NUMBER2);",
				"\t!pick_up(VEHICLE, LOCATION, PACKAGE, CAPACITY_NUMBER1, CAPACITY_NUMBER2).",
			],
		),
	]

	specialised = runner._specialise_chunks_from_noop_body_support(
		chunks,
		fact_index=fact_index,
		type_domains=type_domains,
		max_candidates_per_chunk=64,
	)

	assert any(
		chunk.startswith('+!helper_in("package-0", "truck-0") :')
		and '!pick_up("truck-0", "city-loc-1", "package-0", CAPACITY_NUMBER1, "capacity-1").' in chunk
		for chunk in specialised
	)
def test_runner_builds_recursive_ancestor_guard_internal_action_source():
	runner = JasonRunner()
	java_source = runner._build_no_ancestor_goal_internal_action_source()

	assert "package pipeline;" in java_source
	assert "public class no_ancestor_goal extends DefaultInternalAction" in java_source
	assert "for (IntendedMeans intendedMeans : currentIntention)" in java_source
	assert "requestedFunctor.equals(canonicalText(literal.getFunctor()))" in java_source
