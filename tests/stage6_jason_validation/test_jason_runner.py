from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage3_method_synthesis.htn_schema import HTNLiteral
from stage6_jason_validation.jason_runner import JasonRunner, JasonValidationError


def _sample_action_schemas():
	return [
		{
			"functor": "pick_up",
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


def test_runner_asl_includes_accepting_and_target_validation_without_manual_seeding():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"domain(test).",
		[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="clear", args=("b",), is_positive=False, source_symbol=None),
		],
	)

	assert "+!verify_targets : on(a, b) & not clear(b) <-" in asl
	assert "?dfa_state(FINAL_STATE)" in asl
	assert "?accepting_state(FINAL_STATE)" in asl
	assert "!verify_targets" in asl
	assert "!execute." in asl
	assert "+on(a" not in asl


def test_runner_asl_forces_negative_targets_to_naf_notation():
	runner = JasonRunner()
	asl = runner._build_runner_asl(
		"domain(test).",
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
				"runtime env action success pick_up(a,b)\n"
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
	assert (tmp_path / "action_path.txt").read_text() == "pick_up(a,b)\n"

	validation_payload = json.loads((tmp_path / "jason_validation.json").read_text())
	assert validation_payload["status"] == "success"
	assert validation_payload["environment_adapter"]["success"] is True
	assert validation_payload["action_path"] == ["pick_up(a,b)"]
	assert validation_payload["artifacts"]["action_path"] == str(tmp_path / "action_path.txt")


def test_extract_action_path_preserves_runtime_order():
	runner = JasonRunner()
	stdout = "\n".join(
		[
			"runtime env ready",
			"runtime env action success pick_up(a,b)",
			"runtime env action success put_on_block(a,c)",
			"execute success",
		],
	)

	assert runner._extract_action_path(stdout) == [
		"pick_up(a,b)",
		"put_on_block(a,c)",
	]


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
