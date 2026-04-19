from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

import tests.offline_method_generation.run_generated_problem_root_baseline as generated_baseline_runner
from tests.support.ground_truth_baseline_support import (
	DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS,
	DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS,
	apply_generated_runtime_defaults,
	DOMAIN_FILES,
	load_domain_query_cases,
	query_id_sort_key,
	run_generated_domain_build,
	run_generated_problem_root_case,
)


LIVE_GENERATED_SMOKE_ENABLED = bool(os.getenv("RUN_LIVE_GENERATED_BASELINE_SMOKE")) and bool(
	os.getenv("OPENAI_API_KEY"),
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
	assert report["method_synthesis_model"] == "minimax/minimax-m2.7"
	assert report["generated_method_count"] > 0
	execution = report["execution"]
	assert execution["mode"] == "offline_method_generation"
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


def test_generated_full_sweep_env_enforces_stable_timeouts() -> None:
	with patch.dict("os.environ", {"METHOD_SYNTHESIS_TIMEOUT": "45", "PLANNING_TIMEOUT": "10"}, clear=False):
		env = generated_baseline_runner._build_env()

	assert env["METHOD_SYNTHESIS_TIMEOUT"] == str(
		DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS,
	)
	assert env["PLANNING_TIMEOUT"] == str(
		DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS,
	)
