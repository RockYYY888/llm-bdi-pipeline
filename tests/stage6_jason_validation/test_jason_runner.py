from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNTask,
)
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
	assert "+!reset_execution_state : dfa_state(CURRENT_STATE) <-" in asl
	assert "+dfa_state(q1)" in asl
	assert "+!execute_round_1 : on(a, b) & not clear(b) <-" in asl
	assert "+!execute_round_4 : true <-" in asl
	assert "+on(a" not in asl


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
	assert "+!mark_target_t1 : on(a, b) <-" in asl
	assert "\t+target_seen(t1)." in asl
	assert "\t!mark_target_t1;" in asl


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


def test_rewrite_primitive_wrappers_keeps_only_external_action_call():
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
	assert "\tpick_up(BLOCK1, BLOCK2)." in rewritten
	assert "\t-on(BLOCK1, BLOCK2);" not in rewritten
	assert "\t+holding(BLOCK1)." not in rewritten


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

	assert 'trace_method("m-empty-store-1", STORE, ROVER)' in line


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
