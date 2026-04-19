from __future__ import annotations

import pytest

from tests.support.ground_truth_baseline_support import (
	DOMAIN_FILES,
	load_domain_query_cases,
	query_id_sort_key,
	run_domain_problem_root_case,
	run_official_domain_gate_preflight,
)


@pytest.mark.parametrize("domain_key", sorted(DOMAIN_FILES))
def test_official_domain_gate_preflight_smoke(domain_key: str) -> None:
	report = run_official_domain_gate_preflight(domain_key)
	assert report["success"], report
	execution = report["execution"]
	assert execution["mode"] == "official_domain_preflight"
	assert "method_synthesis" not in execution
	assert execution["domain_gate"]["status"] == "success"


@pytest.mark.parametrize("domain_key", sorted(DOMAIN_FILES))
def test_official_problem_root_baseline_smoke(domain_key: str) -> None:
	query_cases = load_domain_query_cases(domain_key, limit=1)
	query_id = sorted(query_cases, key=query_id_sort_key)[0]
	report = run_domain_problem_root_case(domain_key, query_id)
	assert report["success"], report
	execution = report["execution"]
	assert execution["mode"] == "official_problem_root_execution"
	assert "goal_grounding" not in execution
	assert "method_synthesis" not in execution
	assert "domain_gate" not in execution
	assert execution["plan_solve"]["status"] == "success"
	assert execution["plan_verification"]["status"] == "success"
