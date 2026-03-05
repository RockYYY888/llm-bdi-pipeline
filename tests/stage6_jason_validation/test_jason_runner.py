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


def test_seed_beliefs_only_keeps_positive_non_equality_literals():
	runner = JasonRunner()
	target_literals = [
		HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="clear", args=("a",), is_positive=False, source_symbol=None),
		HTNLiteral(predicate="=", args=("a", "b"), is_positive=True, source_symbol=None),
	]

	assert runner._seed_beliefs_from_literals(target_literals) == ["on(a, b)"]


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
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jar_file)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(
		subprocess,
		"run",
		lambda *args, **kwargs: subprocess.CompletedProcess(
			args=args[0],
			returncode=0,
			stdout="stage6 exec start\nstage6 exec success\n",
			stderr="",
		),
	)

	result = runner.validate(
		agentspeak_code="domain(test).",
		target_literals=[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=True, source_symbol=None),
		],
		domain_name="blocksworld",
		output_dir=tmp_path,
	)

	assert result.status == "success"
	assert (tmp_path / "jason_runner_agent.asl").exists()
	assert (tmp_path / "jason_runner.mas2j").exists()
	assert (tmp_path / "jason_stdout.txt").exists()
	assert (tmp_path / "jason_stderr.txt").exists()
	assert (tmp_path / "jason_validation.json").exists()

	validation_payload = json.loads((tmp_path / "jason_validation.json").read_text())
	assert validation_payload["status"] == "success"


def test_validate_timeout_is_reported(monkeypatch, tmp_path):
	stage6_dir = tmp_path / "stage6"
	stage6_dir.mkdir()
	runner = JasonRunner(stage6_dir=stage6_dir, timeout_seconds=1)
	jar_file = tmp_path / "jason-cli-all.jar"
	log_conf = tmp_path / "console-info.properties"
	jar_file.write_text("jar")
	log_conf.write_text("log")

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("/java23", 23))
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jar_file)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)

	def raise_timeout(*args, **kwargs):
		raise subprocess.TimeoutExpired(
			cmd=args[0],
			timeout=1,
			output="stage6 exec start\n",
			stderr="timeout",
		)

	monkeypatch.setattr(subprocess, "run", raise_timeout)

	with pytest.raises(JasonValidationError) as exc_info:
		runner.validate(
			agentspeak_code="domain(test).",
			target_literals=[],
			domain_name="blocksworld",
			output_dir=tmp_path,
		)

	assert "timeout" in str(exc_info.value)
	assert exc_info.value.metadata["timed_out"] is True
	assert (tmp_path / "jason_validation.json").exists()
