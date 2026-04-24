from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import tests.method_library.run_generated_problem_root_baseline as generated_baseline_runner
import tests.method_library.run_generated_domain_build_sweep as generated_domain_sweep_runner
from tests.support.plan_library_generation_support import (
	DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS,
	DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS,
	apply_generated_runtime_defaults,
	DOMAIN_FILES,
	load_domain_query_cases,
	query_id_sort_key,
	run_generated_domain_build,
)
from tests.support.htn_evaluation_support import run_generated_problem_root_case
import tests.support.htn_evaluation_support as generated_support


LIVE_GENERATED_SMOKE_ENABLED = bool(os.getenv("RUN_LIVE_GENERATED_BASELINE_SMOKE")) and bool(
	os.getenv("METHOD_SYNTHESIS_API_KEY"),
)


@pytest.fixture(scope="module", params=sorted(DOMAIN_FILES))
def generated_domain_build_report(request):
	if not LIVE_GENERATED_SMOKE_ENABLED:
		pytest.skip("Live generated baseline smoke is disabled.")
	domain_key = str(request.param)
	report = run_generated_domain_build(domain_key)
	return domain_key, report


def test_generated_domain_build_smoke(generated_domain_build_report) -> None:
	domain_key, report = generated_domain_build_report
	assert report["success"], domain_key
	assert report["source_domain_kind"] == "masked_official"
	assert report["method_synthesis_model"] == "deepseek-v4-pro"
	assert report["generated_method_count"] > 0
	execution = report["execution"]
	assert execution["mode"] == "plan_library_generation"
	assert execution["method_synthesis"]["status"] == "success"
	assert execution["domain_gate"]["status"] == "success"


def test_generated_problem_root_smoke(generated_domain_build_report) -> None:
	domain_key, report = generated_domain_build_report
	if not report["success"]:
		pytest.skip(f"Generated domain build failed for {domain_key}; skipping problem-root smoke.")
	query_cases = load_domain_query_cases(domain_key, limit=1)
	query_id = sorted(query_cases, key=query_id_sort_key)[0]
	run_report = run_generated_problem_root_case(
		domain_key,
		query_id,
		generated_domain_file=str((report.get("artifact_paths") or {}).get("generated_domain") or ""),
	)
	assert run_report["success"], run_report
	execution = run_report["execution"]
	assert execution["mode"] == "generated_problem_root_execution"
	assert "goal_grounding" not in execution
	assert "method_synthesis" not in execution
	assert "domain_gate" not in execution
	assert execution["plan_solve"]["status"] == "success"
	assert execution["plan_verification"]["status"] == "success"


def test_generated_full_sweep_defaults_to_four_domain_parallelism() -> None:
	with patch.object(generated_baseline_runner, "_run_full_baseline", return_value=0) as run_full:
		with patch.object(
			sys,
			"argv",
			["run_generated_problem_root_baseline.py"],
		):
			assert generated_baseline_runner.main() == 0

	run_full.assert_called_once_with(
		max_concurrent_domains=len(generated_baseline_runner.DOMAIN_KEYS),
	)


def test_generated_domain_build_sweep_defaults_to_four_domain_parallelism() -> None:
	with patch.object(generated_domain_sweep_runner, "_run_full_sweep", return_value=0) as run_full:
		with patch.object(
			sys,
			"argv",
			["run_generated_domain_build_sweep.py"],
		):
			assert generated_domain_sweep_runner.main() == 0

	run_full.assert_called_once_with(
		max_concurrent_domains=len(generated_domain_sweep_runner.DOMAIN_KEYS),
	)


def test_generated_single_domain_can_reuse_existing_generated_domain_file() -> None:
	with patch.object(
		generated_support,
		"load_domain_query_cases",
		return_value={"query_1": {"problem_file": "p.hddl"}},
	), patch.object(
		generated_support,
		"run_generated_problem_root_case",
		return_value={
			"query_id": "query_1",
			"case": {"problem_file": "p.hddl"},
			"success": True,
			"outcome_bucket": "hierarchical_plan_verified",
			"log_dir": Path("/tmp/generated-problem-root"),
			"plan_solve": {"summary": {"status": "success"}},
			"plan_verification": {
				"summary": {"status": "success"},
				"artifacts": {"selected_solver_id": "lifted_panda_sat"},
			},
		},
	) as run_case:
		run_dir = PROJECT_ROOT / "tests" / "generated" / "tmp_generated_problem_root_reuse"
		run_dir.mkdir(parents=True, exist_ok=True)
		generated_baseline_runner._RUN_QUERY_IDS = ["query_1"]
		generated_baseline_runner._RUN_GENERATED_DOMAIN_FILE = "/tmp/generated_domain.hddl"

		try:
			assert generated_baseline_runner._run_single_domain("blocksworld", run_dir) == 0
		finally:
			generated_baseline_runner._RUN_QUERY_IDS = []
			generated_baseline_runner._RUN_GENERATED_DOMAIN_FILE = ""

	run_case.assert_called_once_with(
		"blocksworld",
		"query_1",
		generated_domain_file="/tmp/generated_domain.hddl",
	)
	summary = json.loads((run_dir / "blocksworld.summary.json").read_text())
	assert summary["domain_build"]["domain_build_invocations"] == 0
	assert summary["domain_build"]["reused_generated_domain"] is True
	assert summary["verified_successes"] == 1


def test_generated_runtime_defaults_raise_short_timeouts() -> None:
	env = {
		"METHOD_SYNTHESIS_TIMEOUT": "45",
		"PLANNING_TIMEOUT": "10",
	}

	normalised = apply_generated_runtime_defaults(env)

	assert normalised["METHOD_SYNTHESIS_TIMEOUT"] == str(
		DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS,
	)
	assert normalised["PLANNING_TIMEOUT"] == str(
		DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS,
	)
	assert normalised["METHOD_SYNTHESIS_PROGRESS"] == "1"
	assert normalised["DOMAIN_GATE_PROGRESS"] == "1"


def test_generated_full_sweep_env_enforces_stable_timeouts() -> None:
	with patch.dict("os.environ", {"METHOD_SYNTHESIS_TIMEOUT": "45", "PLANNING_TIMEOUT": "10"}, clear=False):
		env = generated_baseline_runner._build_env()

	assert env["METHOD_SYNTHESIS_TIMEOUT"] == str(
		DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS,
	)
	assert env["PLANNING_TIMEOUT"] == str(
		DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS,
	)
	assert env["METHOD_SYNTHESIS_PROGRESS"] == "1"
	assert env["DOMAIN_GATE_PROGRESS"] == "1"
