from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from planning.backends import LiftedPandaBackend, PandaDealerBackend
from planning.official_benchmark import OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS
from planning.plan_models import PANDAPlanResult
from planning.representations import PlanningRepresentation
from pipeline.domain_complete_pipeline import DomainCompletePipeline

import tests.support.ground_truth_baseline_support as baseline_support
from tests.support.ground_truth_baseline_support import DOMAIN_FILES, build_official_method_library


def test_official_method_library_clears_query_specific_targets() -> None:
	method_library = build_official_method_library(DOMAIN_FILES["blocksworld"])
	assert method_library.target_literals == []
	assert method_library.target_task_bindings == []


def test_problem_structure_analysis_detects_total_order_blocksworld() -> None:
	pipeline = DomainCompletePipeline(
		domain_file=DOMAIN_FILES["blocksworld"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl").resolve()
		),
	)
	structure = pipeline._official_problem_root_structure_analysis()
	assert structure.is_total_order is True
	assert structure.requires_linearization is False


def test_problem_structure_analysis_detects_partial_order_transport() -> None:
	pipeline = DomainCompletePipeline(
		domain_file=DOMAIN_FILES["transport"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "transport" / "problems" / "pfile01.hddl").resolve()
		),
	)
	structure = pipeline._official_problem_root_structure_analysis()
	assert structure.is_total_order is False
	assert structure.requires_linearization is True


def test_official_problem_root_timeout_is_benchmark_pinned() -> None:
	pipeline = DomainCompletePipeline(
		domain_file=DOMAIN_FILES["transport"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "transport" / "problems" / "pfile39.hddl").resolve()
		),
	)
	pipeline.config = SimpleNamespace(planning_timeout=5)
	assert pipeline._official_problem_root_planning_timeout_seconds() == (
		OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS
	)


def test_pandadealer_backend_uses_full_backend_timeout_budget() -> None:
	backend = PandaDealerBackend()
	backend.planner.plan_hddl_files = Mock(  # type: ignore[method-assign]
		return_value=Mock(spec=PANDAPlanResult),
	)
	representation = PlanningRepresentation(
		representation_id="linearized_total_order",
		representation_source="linearized",
		ordering_kind="total_order",
		domain_file="/tmp/domain.hddl",
		problem_file="/tmp/problem.hddl",
		compilation_profile="semantics_preserving_linearization",
	)
	backend.solve(
		domain=object(),
		representation=representation,
		task_name="deliver",
		task_args=("package-0", "city-loc-0"),
		timeout_seconds=1800.0,
	)
	kwargs = backend.planner.plan_hddl_files.call_args.kwargs  # type: ignore[union-attr]
	solver_configs = kwargs["solver_configs"]
	assert len(solver_configs) == 1
	assert "timeout_seconds" not in solver_configs[0]
	assert kwargs["timeout_seconds"] == 1800.0


def test_lifted_panda_backend_uses_full_backend_timeout_budget() -> None:
	backend = LiftedPandaBackend()
	backend.planner.plan_linearized_hddl_files = Mock(  # type: ignore[method-assign]
		return_value=Mock(spec=PANDAPlanResult),
	)
	representation = PlanningRepresentation(
		representation_id="linearized_total_order",
		representation_source="linearized",
		ordering_kind="total_order",
		domain_file="/tmp/domain.hddl",
		problem_file="/tmp/problem.hddl",
		compilation_profile="semantics_preserving_linearization",
	)
	backend.solve(
		domain=object(),
		representation=representation,
		task_name="deliver",
		task_args=("package-0", "city-loc-0"),
		timeout_seconds=1800.0,
	)
	kwargs = backend.planner.plan_linearized_hddl_files.call_args.kwargs  # type: ignore[union-attr]
	solver_configs = kwargs["solver_configs"]
	assert len(solver_configs) == 1
	assert "timeout_seconds" not in solver_configs[0]
	assert kwargs["timeout_seconds"] == 1800.0


def test_run_official_problem_root_baseline_for_domain_filters_query_ids() -> None:
	load_cases = Mock(
		return_value={
			"query_01": {"problem_file": "pfile01.hddl", "instruction": "q1"},
			"query_02": {"problem_file": "pfile02.hddl", "instruction": "q2"},
			"query_03": {"problem_file": "pfile03.hddl", "instruction": "q3"},
		},
	)
	run_case = Mock(
		side_effect=[
			{
				"query_id": "query_01",
				"case": {"problem_file": "pfile01.hddl"},
				"log_dir": Path("/tmp/query-01"),
				"success": False,
				"outcome_bucket": "no_plan_from_solver",
				"plan_solve": {"summary": {"status": "failed"}},
				"plan_verification": {"summary": {"status": "failed"}, "artifacts": {}},
			},
			{
				"query_id": "query_03",
				"case": {"problem_file": "pfile03.hddl"},
				"log_dir": Path("/tmp/query-03"),
				"success": True,
				"outcome_bucket": "hierarchical_plan_verified",
				"plan_solve": {"summary": {"status": "success"}},
				"plan_verification": {
					"summary": {"status": "success"},
					"artifacts": {"selected_solver_id": "sat"},
				},
			},
		],
	)
	run_gate = Mock(
		return_value={
			"success": True,
			"log_dir": Path("/tmp/domain-gate"),
			"artifact_root": Path("/tmp/domain-gate-artifacts"),
			"domain_gate": {"validated_task_count": 3},
		},
	)

	original_load = baseline_support.load_domain_query_cases
	original_run_case = baseline_support.run_domain_problem_root_case
	original_run_gate = baseline_support.run_official_domain_gate_preflight
	try:
		baseline_support.load_domain_query_cases = load_cases
		baseline_support.run_domain_problem_root_case = run_case
		baseline_support.run_official_domain_gate_preflight = run_gate
		summary = baseline_support.run_official_problem_root_baseline_for_domain(
			"transport",
			query_ids=("query_03", "query_01"),
		)
	finally:
		baseline_support.load_domain_query_cases = original_load
		baseline_support.run_domain_problem_root_case = original_run_case
		baseline_support.run_official_domain_gate_preflight = original_run_gate

	assert summary["selected_query_ids"] == ["query_01", "query_03"]
	assert summary["total_queries"] == 2
	assert summary["verified_successes"] == 1
	assert summary["solver_no_plan_failures"] == 1
	assert run_case.call_args_list[0].args == ("transport", "query_01")
	assert run_case.call_args_list[1].args == ("transport", "query_03")
