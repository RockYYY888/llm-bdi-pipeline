from __future__ import annotations

import sys
from pathlib import Path

import pytest

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage3_method_synthesis.htn_schema import HTNLiteral
from stage6_jason_validation.jason_runner import JasonRunner


def _stage6_ready() -> bool:
	return JasonRunner().toolchain_available()


def test_stage6_runs_query2_sample_agentspeak(tmp_path):
	if not _stage6_ready():
		pytest.skip("Stage 6 integration requires Java 17-23 and Jason runtime toolchain")

	sample_asl = Path("tests/logs/20260304_123250_dfa_agentspeak/agentspeak_generated.asl")
	if not sample_asl.exists():
		pytest.skip(f"Sample ASL missing: {sample_asl}")

	runner = JasonRunner(timeout_seconds=60)
	result = runner.validate(
		agentspeak_code=sample_asl.read_text(),
		target_literals=[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=False, source_symbol=None),
		],
		domain_name="blocksworld",
		output_dir=tmp_path,
	)

	assert result.status == "success"
	assert "stage6 exec success" in result.stdout
	assert "stage6 exec failed" not in result.stdout


def test_stage6_runs_mixed_goal_sample_agentspeak(tmp_path):
	if not _stage6_ready():
		pytest.skip("Stage 6 integration requires Java 17-23 and Jason runtime toolchain")

	sample_asl = Path("tests/logs/20260304_123745_dfa_agentspeak/agentspeak_generated.asl")
	if not sample_asl.exists():
		pytest.skip(f"Sample ASL missing: {sample_asl}")

	runner = JasonRunner(timeout_seconds=60)
	result = runner.validate(
		agentspeak_code=sample_asl.read_text(),
		target_literals=[
			HTNLiteral(predicate="clear", args=("c",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="clear", args=("d",), is_positive=False, source_symbol=None),
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=False, source_symbol=None),
			HTNLiteral(predicate="on", args=("c", "d"), is_positive=True, source_symbol=None),
		],
		domain_name="blocksworld",
		output_dir=tmp_path,
	)

	assert result.status == "success"
	assert "stage6 exec success" in result.stdout
	assert "stage6 exec failed" not in result.stdout
