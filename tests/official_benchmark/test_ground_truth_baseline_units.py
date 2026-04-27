from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from planning.primary_planner import LiftedPandaSatPlanner
from planning.official_benchmark import OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS
from planning.panda_sat import PANDAPlanner, PANDAPlanningError
from planning.plan_models import PANDAPlanResult
from planning.process_capture import (
	PROCESS_OUTPUT_PREVIEW_BYTE_LIMIT,
	read_full_process_output,
	run_subprocess_to_files,
)
from planning.representations import PlanningRepresentation, RepresentationBuildResult
from htn_evaluation.pipeline import HTNEvaluationPipeline
from htn_evaluation.problem_root_evaluator import HTNProblemRootEvaluator
import htn_evaluation.problem_root_evaluator as problem_root_evaluator
import htn_evaluation.problem_root_runtime as problem_root_runtime
from htn_evaluation.result_tables import (
	HTN_PLANNER_IDS,
	PRIMARY_HTN_PLANNER_ID,
	SINGLE_PLANNER_MODE,
	build_planner_capability_rows,
	build_problem_capability_rows,
	build_problem_result_row,
	build_track_summary,
	validate_planner_id,
	write_planner_capability_matrix,
	write_problem_capability_matrix,
)

import tests.support.htn_evaluation_support as baseline_support
import tests.run_official_problem_root_baseline as baseline_runner
from tests.support.plan_library_generation_support import DOMAIN_FILES, build_official_method_library


def test_official_method_library_clears_query_specific_targets() -> None:
	method_library = build_official_method_library(DOMAIN_FILES["blocksworld"])
	assert method_library.target_literals == []
	assert method_library.target_task_bindings == []


def test_problem_structure_analysis_detects_total_order_blocksworld() -> None:
	pipeline = HTNEvaluationPipeline(
		domain_file=DOMAIN_FILES["blocksworld"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl").resolve()
		),
	)
	structure = pipeline._official_problem_root_structure_analysis()
	assert structure.is_total_order is True
	assert structure.requires_linearization is False


def test_problem_structure_analysis_detects_partial_order_transport() -> None:
	pipeline = HTNEvaluationPipeline(
		domain_file=DOMAIN_FILES["transport"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "transport" / "problems" / "pfile01.hddl").resolve()
		),
	)
	structure = pipeline._official_problem_root_structure_analysis()
	assert structure.is_total_order is False
	assert structure.requires_linearization is True


def test_official_problem_root_timeout_is_benchmark_pinned() -> None:
	pipeline = HTNEvaluationPipeline(
		domain_file=DOMAIN_FILES["transport"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "transport" / "problems" / "pfile39.hddl").resolve()
		),
	)
	pipeline.config = SimpleNamespace(planning_timeout=5)
	assert pipeline._official_problem_root_planning_timeout_seconds() == (
		OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS
	)


def test_official_problem_root_resource_profile_matches_ipc_limits() -> None:
	pipeline = HTNEvaluationPipeline(
		domain_file=DOMAIN_FILES["transport"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "transport" / "problems" / "pfile39.hddl").resolve()
		),
	)
	profile = pipeline.context._official_problem_root_resource_profile()
	assert profile == {
		"planning_timeout_seconds": 1800.0,
		"memory_limit_mib": 8192,
		"cpu_count": 1,
	}


def test_only_lifted_panda_sat_is_supported_as_primary_baseline() -> None:
	assert HTN_PLANNER_IDS == (PRIMARY_HTN_PLANNER_ID,)
	assert validate_planner_id(None, evaluation_mode=SINGLE_PLANNER_MODE) == (
		PRIMARY_HTN_PLANNER_ID
	)


def test_panda_plan_parser_decodes_linearized_identifier_tokens() -> None:
	planner = PANDAPlanner()
	domain = SimpleNamespace(
		actions=[
			SimpleNamespace(name="wait"),
			SimpleNamespace(name="pick-up"),
			SimpleNamespace(name="drop"),
		],
	)
	plan_text = planner._decode_linearized_plan_tokens(
		"""
==>
0 wait truckMINUS_0 cityMINUS_locMINUS_0
1 pickMINUS_up truckMINUS_0 cityMINUS_locMINUS_0 packageMINUS_0 capacityMINUS_2 capacityMINUS_3
2 drop truckMINUS_0 cityMINUS_locMINUS_3 packageMINUS_0 capacityMINUS_2 capacityMINUS_3
root
""",
	)

	steps = planner._parse_plan_steps(plan_text, domain)

	assert [step.action_name for step in steps] == ["wait", "pick-up", "drop"]
	assert steps[1].args == (
		"truck-0",
		"city-loc-0",
		"package-0",
		"capacity-2",
		"capacity-3",
	)


def test_lifted_panda_primary_planner_uses_full_planning_timeout_budget() -> None:
	planner = LiftedPandaSatPlanner()
	planner.planner.plan_linearized_hddl_files = Mock(  # type: ignore[method-assign]
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
	planner.solve(
		domain=object(),
		representation=representation,
		task_name="deliver",
		task_args=("package-0", "city-loc-0"),
		timeout_seconds=1800.0,
	)
	kwargs = planner.planner.plan_linearized_hddl_files.call_args.kwargs  # type: ignore[union-attr]
	solver_configs = kwargs["solver_configs"]
	assert len(solver_configs) == 1
	assert "timeout_seconds" not in solver_configs[0]
	assert kwargs["timeout_seconds"] == 1800.0


def test_run_official_problem_root_baseline_for_domain_filters_query_ids(
	tmp_path: Path,
) -> None:
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
			output_root=tmp_path / PRIMARY_HTN_PLANNER_ID / "transport",
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
	assert run_case.call_args_list[0].kwargs["logs_dir"] == (
		tmp_path / PRIMARY_HTN_PLANNER_ID / "transport" / "query_logs"
	)


def test_merge_primary_planner_output_dir_skips_unreadable_planner_root(
	tmp_path: Path,
) -> None:
	pipeline = HTNEvaluationPipeline(
		domain_file=DOMAIN_FILES["blocksworld"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl").resolve()
		),
	)
	pipeline.output_dir = str(tmp_path / "selected-root")
	unreadable_root = tmp_path / "planner-root"
	unreadable_root.mkdir(parents=True, exist_ok=True)
	original_iterdir = Path.iterdir

	def fake_iterdir(path: Path):
		if path == unreadable_root:
			raise PermissionError("operation not permitted")
		return original_iterdir(path)

	with patch.object(Path, "iterdir", fake_iterdir):
		pipeline._merge_primary_planner_output_dir(unreadable_root)

	assert (tmp_path / "selected-root").exists()


def test_close_planner_queue_closes_and_joins_thread() -> None:
	close = Mock()
	join_thread = Mock()
	queue_stub = SimpleNamespace(close=close, join_thread=join_thread)

	HTNEvaluationPipeline._close_planner_queue(queue_stub)

	close.assert_called_once_with()
	join_thread.assert_called_once_with()


def test_primary_htn_planner_delegates_to_hierarchical_task_network_evaluator() -> None:
	pipeline = HTNEvaluationPipeline(
		domain_file=DOMAIN_FILES["blocksworld"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl").resolve()
		),
	)

	class FakeEvaluator:
		def __init__(self) -> None:
			self.calls: list[dict[str, object]] = []

		def execute_problem_root_evaluation(
			self,
			*,
			method_library=None,
			evaluation_mode: str,
			planner_id: str | None,
		):
			self.calls.append(
				{
					"method_library": method_library,
					"evaluation_mode": evaluation_mode,
					"planner_id": planner_id,
				},
			)
			return {"plan_solve": {"summary": {"status": "success"}}, "plan_verification": {"summary": {"status": "success"}}}

	fake_evaluator = FakeEvaluator()
	pipeline._htn_problem_root_evaluator_instance = fake_evaluator  # type: ignore[assignment]

	result = pipeline._execute_primary_htn_planner(method_library="sentinel")

	assert result["plan_solve"]["summary"]["status"] == "success"
	assert fake_evaluator.calls == [
		{
			"method_library": "sentinel",
			"evaluation_mode": SINGLE_PLANNER_MODE,
			"planner_id": PRIMARY_HTN_PLANNER_ID,
		},
	]


def test_problem_root_evaluation_delegates_mode_and_planner_id() -> None:
	pipeline = HTNEvaluationPipeline(
		domain_file=DOMAIN_FILES["blocksworld"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl").resolve()
		),
	)

	class FakeEvaluator:
		def __init__(self) -> None:
			self.calls: list[dict[str, object]] = []

		def execute_problem_root_evaluation(
			self,
			*,
			method_library=None,
			evaluation_mode: str,
			planner_id: str | None,
		):
			self.calls.append(
				{
					"method_library": method_library,
					"evaluation_mode": evaluation_mode,
					"planner_id": planner_id,
				},
			)
			return {"plan_solve": {"summary": {"status": "success"}}}

	fake_evaluator = FakeEvaluator()
	pipeline._htn_problem_root_evaluator_instance = fake_evaluator  # type: ignore[assignment]

	pipeline.execute_problem_root_evaluation(
		method_library="sentinel",
		evaluation_mode=SINGLE_PLANNER_MODE,
		planner_id="lifted_panda_sat",
	)

	assert fake_evaluator.calls == [
		{
			"method_library": "sentinel",
			"evaluation_mode": SINGLE_PLANNER_MODE,
			"planner_id": "lifted_panda_sat",
		},
	]


def test_run_official_problem_root_baseline_for_domain_writes_mode_specific_result_tables(
	tmp_path: Path,
) -> None:
	load_cases = Mock(
		return_value={
			"query_01": {"problem_file": "pfile01.hddl", "instruction": "q1"},
			"query_02": {"problem_file": "pfile02.hddl", "instruction": "q2"},
		},
	)
	run_case = Mock(
		side_effect=[
			{
				"query_id": "query_01",
				"case": {"problem_file": "pfile01.hddl", "instruction": "q1"},
				"log_dir": Path("/tmp/query-01"),
				"success": True,
				"outcome_bucket": "hierarchical_plan_verified",
				"execution": {
					"execution_time_seconds": 12.5,
					"timings": {
						"plan_solve": {
							"total_seconds": 10.0,
							"metadata": {
								"representation_build_seconds": 1.25,
								"planner_wallclock_seconds": 11.0,
							},
						},
						"plan_verification": {
							"total_seconds": 2.0,
							"metadata": {
								"planner_wallclock_seconds": 11.0,
							},
						},
					},
				},
				"plan_solve": {"summary": {"status": "success"}},
				"plan_verification": {
					"summary": {"status": "success"},
					"artifacts": {
						"selected_solver_id": "sat",
						"selected_planner_id": "lifted_panda_sat",
						"selected_representation_id": "linearized_total_order",
					},
				},
			},
			{
				"query_id": "query_02",
				"case": {"problem_file": "pfile02.hddl", "instruction": "q2"},
				"log_dir": Path("/tmp/query-02"),
				"success": False,
				"outcome_bucket": "no_plan_from_solver",
				"execution": {
					"execution_time_seconds": 20.0,
					"timings": {
						"plan_solve": {
							"total_seconds": 19.0,
							"metadata": {
								"representation_build_seconds": 0.5,
								"planner_wallclock_seconds": 19.0,
							},
						},
						"plan_verification": {
							"total_seconds": 0.5,
							"metadata": {
								"planner_wallclock_seconds": 19.0,
							},
						},
					},
				},
				"plan_solve": {"summary": {"status": "failed"}},
				"plan_verification": {"summary": {"status": "failed"}, "artifacts": {}},
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
	original_generated_baseline_dir = baseline_support.GENERATED_BASELINE_DIR
	try:
		baseline_support.load_domain_query_cases = load_cases
		baseline_support.run_domain_problem_root_case = run_case
		baseline_support.run_official_domain_gate_preflight = run_gate
		baseline_support.GENERATED_BASELINE_DIR = tmp_path
		summary = baseline_support.run_official_problem_root_baseline_for_domain(
			"transport",
			query_ids=("query_01", "query_02"),
			evaluation_mode=SINGLE_PLANNER_MODE,
			planner_id="lifted_panda_sat",
		)
	finally:
		baseline_support.load_domain_query_cases = original_load
		baseline_support.run_domain_problem_root_case = original_run_case
		baseline_support.run_official_domain_gate_preflight = original_run_gate
		baseline_support.GENERATED_BASELINE_DIR = original_generated_baseline_dir

	assert summary["evaluation_mode"] == SINGLE_PLANNER_MODE
	assert summary["requested_planner_id"] == "lifted_panda_sat"
	assert summary["verified_success_count"] == 1
	assert summary["bucket_counts"]["no_plan_from_solver"] == 1
	assert summary["execution_time_seconds_total"] == 32.5
	assert summary["plan_solve_time_seconds_total"] == 29.0
	assert summary["plan_verification_time_seconds_total"] == 2.5
	problem_results_path = Path(summary["output_paths"]["problem_results"])
	domain_summary_path = Path(summary["output_paths"]["domain_summary"])
	assert problem_results_path.exists()
	assert domain_summary_path.exists()
	assert problem_results_path.parent == tmp_path / "lifted_panda_sat" / "transport"
	problem_rows = json.loads(problem_results_path.read_text())
	assert problem_rows[0]["execution_time_seconds"] == 12.5
	assert problem_rows[0]["plan_solve_time_seconds"] == 10.0
	assert problem_rows[0]["plan_verification_time_seconds"] == 2.0
	assert problem_rows[0]["representation_build_seconds"] == 1.25
	assert problem_rows[0]["planner_wallclock_seconds"] == 11.0


def test_run_official_problem_root_baseline_for_domain_resumes_completed_queries(
	tmp_path: Path,
) -> None:
	output_root = tmp_path / PRIMARY_HTN_PLANNER_ID / "transport"
	output_root.mkdir(parents=True, exist_ok=True)
	existing_problem_row = {
		"domain_key": "transport",
		"query_id": "query_01",
		"problem_file": "pfile01.hddl",
		"instruction": "q1",
		"evaluation_mode": SINGLE_PLANNER_MODE,
		"requested_planner_id": PRIMARY_HTN_PLANNER_ID,
		"track_id": PRIMARY_HTN_PLANNER_ID,
		"ipc_verified_success": True,
		"outcome_bucket": "hierarchical_plan_verified",
		"log_dir": "/tmp/query-01",
		"execution_time_seconds": 10.0,
		"plan_solve_time_seconds": 8.0,
		"plan_verification_time_seconds": 2.0,
		"representation_build_seconds": 0.5,
		"planner_wallclock_seconds": 9.0,
		"plan_solve_status": "success",
		"plan_verification_status": "success",
		"selected_solver_id": "sat",
		"selected_planner_id": "lifted_panda_sat",
		"selected_representation_id": "linearized_total_order",
	}
	(output_root / "problem_results.json").write_text(
		json.dumps([existing_problem_row], indent=2),
	)
	(output_root / "domain_summary.json").write_text(
		json.dumps(
			{
				"domain_gate_preflight": {
					"success": True,
					"log_dir": "/tmp/domain-gate",
					"artifact_root": "/tmp/domain-gate-artifacts",
					"validated_task_count": 3,
				},
			},
			indent=2,
		),
	)

	load_cases = Mock(
		return_value={
			"query_01": {"problem_file": "pfile01.hddl", "instruction": "q1"},
			"query_02": {"problem_file": "pfile02.hddl", "instruction": "q2"},
		},
	)
	run_case = Mock(
		return_value={
			"query_id": "query_02",
			"case": {"problem_file": "pfile02.hddl", "instruction": "q2"},
			"log_dir": Path("/tmp/query-02"),
			"success": False,
			"outcome_bucket": "no_plan_from_solver",
			"execution": {
				"execution_time_seconds": 20.0,
				"timings": {
					"plan_solve": {
						"total_seconds": 19.0,
						"metadata": {
							"representation_build_seconds": 0.5,
							"planner_wallclock_seconds": 19.0,
						},
					},
					"plan_verification": {
						"total_seconds": 0.5,
						"metadata": {"planner_wallclock_seconds": 19.0},
					},
				},
			},
			"plan_solve": {"summary": {"status": "failed"}},
			"plan_verification": {"summary": {"status": "failed"}, "artifacts": {}},
		},
	)

	original_load = baseline_support.load_domain_query_cases
	original_run_case = baseline_support.run_domain_problem_root_case
	try:
		baseline_support.load_domain_query_cases = load_cases
		baseline_support.run_domain_problem_root_case = run_case
		summary = baseline_support.run_official_problem_root_baseline_for_domain(
			"transport",
			query_ids=("query_01", "query_02"),
			output_root=output_root,
		)
	finally:
		baseline_support.load_domain_query_cases = original_load
		baseline_support.run_domain_problem_root_case = original_run_case

	run_case.assert_called_once()
	assert run_case.call_args.args[:2] == ("transport", "query_02")
	assert run_case.call_args.kwargs["logs_dir"] == output_root / "query_logs"
	assert summary["attempted_problem_count"] == 2
	assert summary["verified_success_count"] == 1
	problem_rows = json.loads((output_root / "problem_results.json").read_text())
	assert [row["query_id"] for row in problem_rows] == ["query_01", "query_02"]


def test_htn_query_log_timestamp_includes_query_id_to_avoid_collisions() -> None:
	query_1_timestamp = baseline_support._query_log_timestamp("query_1")
	query_2_timestamp = baseline_support._query_log_timestamp("query_2")

	assert query_1_timestamp.endswith("_query_1")
	assert query_2_timestamp.endswith("_query_2")
	assert query_1_timestamp != query_2_timestamp


def test_run_official_problem_root_baseline_for_domain_preserves_unselected_existing_query_rows(
	tmp_path: Path,
) -> None:
	output_root = tmp_path / PRIMARY_HTN_PLANNER_ID / "transport"
	output_root.mkdir(parents=True, exist_ok=True)
	(output_root / "problem_results.json").write_text(
		json.dumps(
			[
				{
					"domain_key": "transport",
					"query_id": "query_01",
					"problem_file": "pfile01.hddl",
					"instruction": "q1",
					"evaluation_mode": SINGLE_PLANNER_MODE,
					"requested_planner_id": PRIMARY_HTN_PLANNER_ID,
					"track_id": PRIMARY_HTN_PLANNER_ID,
					"ipc_verified_success": True,
					"outcome_bucket": "hierarchical_plan_verified",
					"log_dir": "/tmp/query-01",
					"execution_time_seconds": 10.0,
					"plan_solve_time_seconds": 8.0,
					"plan_verification_time_seconds": 2.0,
					"representation_build_seconds": 0.5,
					"planner_wallclock_seconds": 9.0,
					"plan_solve_status": "success",
					"plan_verification_status": "success",
					"selected_solver_id": "sat",
					"selected_planner_id": "lifted_panda_sat",
					"selected_representation_id": "linearized_total_order",
				},
			],
			indent=2,
		),
	)
	(output_root / "domain_summary.json").write_text(
		json.dumps({"domain_gate_preflight": {"success": True}}, indent=2),
	)

	load_cases = Mock(
		return_value={
			"query_01": {"problem_file": "pfile01.hddl", "instruction": "q1"},
			"query_10": {"problem_file": "pfile10.hddl", "instruction": "q10"},
		},
	)
	run_case = Mock(
		return_value={
			"query_id": "query_10",
			"case": {"problem_file": "pfile10.hddl", "instruction": "q10"},
			"log_dir": Path("/tmp/query-10"),
			"success": True,
			"outcome_bucket": "hierarchical_plan_verified",
			"execution": {
				"execution_time_seconds": 12.5,
				"timings": {
					"plan_solve": {
						"total_seconds": 10.0,
						"metadata": {
							"representation_build_seconds": 1.25,
							"planner_wallclock_seconds": 11.0,
						},
					},
					"plan_verification": {
						"total_seconds": 2.0,
						"metadata": {"planner_wallclock_seconds": 11.0},
					},
				},
			},
			"plan_solve": {"summary": {"status": "success"}},
			"plan_verification": {
				"summary": {"status": "success"},
				"artifacts": {
					"selected_solver_id": "sat",
					"selected_planner_id": "lifted_panda_sat",
					"selected_representation_id": "linearized_total_order",
				},
			},
		},
	)

	original_load = baseline_support.load_domain_query_cases
	original_run_case = baseline_support.run_domain_problem_root_case
	try:
		baseline_support.load_domain_query_cases = load_cases
		baseline_support.run_domain_problem_root_case = run_case
		summary = baseline_support.run_official_problem_root_baseline_for_domain(
			"transport",
			query_ids=("query_10",),
			output_root=output_root,
		)
	finally:
		baseline_support.load_domain_query_cases = original_load
		baseline_support.run_domain_problem_root_case = original_run_case

	assert summary["attempted_problem_count"] == 2
	problem_rows = json.loads((output_root / "problem_results.json").read_text())
	assert [row["query_id"] for row in problem_rows] == ["query_01", "query_10"]


def test_run_official_problem_root_baseline_for_domain_persists_partial_query_checkpoint_before_failure(
	tmp_path: Path,
) -> None:
	output_root = tmp_path / PRIMARY_HTN_PLANNER_ID / "transport"
	load_cases = Mock(
		return_value={
			"query_01": {"problem_file": "pfile01.hddl", "instruction": "q1"},
			"query_02": {"problem_file": "pfile02.hddl", "instruction": "q2"},
		},
	)
	run_case = Mock(
		side_effect=[
			{
				"query_id": "query_01",
				"case": {"problem_file": "pfile01.hddl", "instruction": "q1"},
				"log_dir": Path("/tmp/query-01"),
				"success": True,
				"outcome_bucket": "hierarchical_plan_verified",
				"execution": {
					"execution_time_seconds": 12.5,
					"timings": {
						"plan_solve": {
							"total_seconds": 10.0,
							"metadata": {
								"representation_build_seconds": 1.25,
								"planner_wallclock_seconds": 11.0,
							},
						},
						"plan_verification": {
							"total_seconds": 2.0,
							"metadata": {"planner_wallclock_seconds": 11.0},
						},
					},
				},
				"plan_solve": {"summary": {"status": "success"}},
				"plan_verification": {
					"summary": {"status": "success"},
					"artifacts": {
						"selected_solver_id": "sat",
						"selected_planner_id": "lifted_panda_sat",
						"selected_representation_id": "linearized_total_order",
					},
				},
			},
			RuntimeError("boom"),
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
		try:
			baseline_support.run_official_problem_root_baseline_for_domain(
				"transport",
				query_ids=("query_01", "query_02"),
				output_root=output_root,
			)
			raise AssertionError("Expected the second query to fail.")
		except RuntimeError as exc:
			assert str(exc) == "boom"
	finally:
		baseline_support.load_domain_query_cases = original_load
		baseline_support.run_domain_problem_root_case = original_run_case
		baseline_support.run_official_domain_gate_preflight = original_run_gate

	problem_rows = json.loads((output_root / "problem_results.json").read_text())
	assert [row["query_id"] for row in problem_rows] == ["query_01"]
	domain_summary = json.loads((output_root / "domain_summary.json").read_text())
	assert domain_summary["attempted_problem_count"] == 1
	assert domain_summary["verified_success_count"] == 1


def test_result_tables_build_planner_capability_matrix_rows_and_csv(
	tmp_path: Path,
) -> None:
	problem_row = build_problem_result_row(
		domain_key="blocksworld",
		query_id="query_01",
		case={"problem_file": "p01.hddl", "instruction": "move"},
		report={
			"success": True,
			"outcome_bucket": "hierarchical_plan_verified",
			"log_dir": Path("/tmp/log"),
			"execution": {
				"execution_time_seconds": 17.0,
				"timings": {
					"plan_solve": {
						"total_seconds": 12.0,
						"metadata": {
							"representation_build_seconds": 0.7,
							"planner_wallclock_seconds": 13.5,
						},
					},
					"plan_verification": {
						"total_seconds": 5.0,
						"metadata": {"planner_wallclock_seconds": 13.5},
					},
				},
			},
			"plan_solve": {"summary": {"status": "success"}},
			"plan_verification": {
				"summary": {"status": "success"},
				"artifacts": {
					"selected_solver_id": "sat",
					"selected_planner_id": "lifted_panda_sat",
					"selected_representation_id": "linearized_total_order",
				},
			},
		},
		evaluation_mode=SINGLE_PLANNER_MODE,
		planner_id=None,
	)
	track_summary = build_track_summary(
		run_dir=tmp_path,
		domain_summaries={
			"blocksworld": {
				"attempted_problem_count": 1,
				"execution_time_seconds_total": 17.0,
				"execution_time_seconds_average": 17.0,
				"plan_solve_time_seconds_total": 12.0,
				"plan_solve_time_seconds_average": 12.0,
				"plan_verification_time_seconds_total": 5.0,
				"plan_verification_time_seconds_average": 5.0,
				"verified_success_count": 1,
				"bucket_counts": {
					"hierarchical_plan_verified": 1,
					"primitive_plan_valid_but_hierarchical_rejected": 0,
					"primitive_plan_invalid": 0,
					"no_plan_from_solver": 0,
					"unknown_failure": 0,
				},
				"query_results": [
					{
						"query_id": "query_01",
						"problem_file": "p01.hddl",
						"log_dir": "/tmp/log",
						"success": True,
						"outcome_bucket": "hierarchical_plan_verified",
						"execution_time_seconds": 17.0,
						"plan_solve_time_seconds": 12.0,
						"plan_verification_time_seconds": 5.0,
						"representation_build_seconds": 0.7,
						"planner_wallclock_seconds": 13.5,
						"plan_solve_status": "success",
						"plan_verification_status": "success",
						"selected_solver_id": "sat",
						"selected_planner_id": "lifted_panda_sat",
						"selected_representation_id": "linearized_total_order",
					},
				],
			},
		},
		evaluation_mode=SINGLE_PLANNER_MODE,
		planner_id=None,
	)
	rows = build_planner_capability_rows((track_summary,))
	problem_rows = build_problem_capability_rows((track_summary,))
	paths = write_planner_capability_matrix(tmp_path, rows=rows)
	problem_paths = write_problem_capability_matrix(tmp_path, rows=problem_rows)

	assert problem_row["selected_planner_id"] == "lifted_panda_sat"
	assert problem_row["execution_time_seconds"] == 17.0
	assert rows[0]["track_id"] == PRIMARY_HTN_PLANNER_ID
	assert rows[0]["execution_time_seconds_total"] == 17.0
	assert rows[0]["plan_solve_time_seconds_total"] == 12.0
	assert rows[0]["plan_verification_time_seconds_total"] == 5.0
	assert rows[0]["verified_success_count"] == 1
	assert Path(paths["planner_capability_matrix_json"]).exists()
	assert Path(paths["planner_capability_matrix_csv"]).exists()
	assert "domain_key" in Path(paths["planner_capability_matrix_csv"]).read_text()
	assert problem_rows[0]["query_id"] == "query_01"
	assert problem_rows[0]["execution_time_seconds"] == 17.0
	assert problem_rows[0]["planner_wallclock_seconds"] == 13.5
	assert Path(problem_paths["problem_capability_matrix_json"]).exists()
	assert Path(problem_paths["problem_capability_matrix_csv"]).exists()
	assert "query_id" in Path(problem_paths["problem_capability_matrix_csv"]).read_text()


def test_sequential_full_baseline_writes_incremental_track_outputs(
	tmp_path: Path,
) -> None:
	original_domain_keys = baseline_runner.DOMAIN_KEYS
	baseline_runner.DOMAIN_KEYS = ("blocksworld", "transport")
	try:
		def fake_domain_runner(
			domain_key: str,
			evaluation_mode: str,
			planner_id: str | None,
		) -> dict[str, object]:
			return {
				"domain_key": domain_key,
				"evaluation_mode": evaluation_mode,
				"requested_planner_id": planner_id,
				"attempted_problem_count": 1,
				"execution_time_seconds_total": 12.0,
				"execution_time_seconds_average": 12.0,
				"plan_solve_time_seconds_total": 9.0,
				"plan_solve_time_seconds_average": 9.0,
				"plan_verification_time_seconds_total": 3.0,
				"plan_verification_time_seconds_average": 3.0,
				"verified_success_count": 1,
				"verified_successes": 1,
				"hierarchical_rejection_failures": 0,
				"primitive_invalid_failures": 0,
				"solver_no_plan_failures": 0,
				"unknown_failures": 0,
				"bucket_counts": {
					"hierarchical_plan_verified": 1,
					"primitive_plan_valid_but_hierarchical_rejected": 0,
					"primitive_plan_invalid": 0,
					"no_plan_from_solver": 0,
					"unknown_failure": 0,
				},
				"query_results": [
					{
						"query_id": f"{domain_key}_query_01",
						"problem_file": f"{domain_key}.hddl",
						"log_dir": f"/tmp/{domain_key}",
						"success": True,
						"outcome_bucket": "hierarchical_plan_verified",
						"execution_time_seconds": 12.0,
						"plan_solve_time_seconds": 9.0,
						"plan_verification_time_seconds": 3.0,
						"representation_build_seconds": 0.5,
						"planner_wallclock_seconds": 10.5,
						"plan_solve_status": "success",
						"plan_verification_status": "success",
						"selected_solver_id": "sat",
						"selected_planner_id": "lifted_panda_sat",
						"selected_representation_id": "linearized_total_order",
					},
				],
			}

		summary = baseline_runner._run_sequential_full_baseline(
			run_dir=tmp_path,
			evaluation_mode=SINGLE_PLANNER_MODE,
			planner_id=PRIMARY_HTN_PLANNER_ID,
			track_id=PRIMARY_HTN_PLANNER_ID,
			domain_runner=fake_domain_runner,
		)
	finally:
		baseline_runner.DOMAIN_KEYS = original_domain_keys

	assert summary["complete"] is True
	assert Path(tmp_path / "blocksworld.summary.json").exists()
	assert Path(tmp_path / "transport.summary.json").exists()
	assert Path(tmp_path / "track_summary.json").exists()
	assert Path(tmp_path / "summary.json").exists()
	assert Path(tmp_path / "planner_capability_matrix.json").exists()
	assert Path(tmp_path / "planner_capability_matrix.csv").exists()
	assert Path(tmp_path / "problem_capability_matrix.json").exists()
	assert Path(tmp_path / "problem_capability_matrix.csv").exists()
	problem_rows = json.loads((tmp_path / "problem_capability_matrix.json").read_text())
	assert problem_rows[0]["execution_time_seconds"] == 12.0
	assert problem_rows[0]["plan_solve_time_seconds"] == 9.0
	assert problem_rows[0]["plan_verification_time_seconds"] == 3.0


def test_sequential_full_baseline_resumes_from_existing_domain_summaries(
	tmp_path: Path,
) -> None:
	original_domain_keys = baseline_runner.DOMAIN_KEYS
	baseline_runner.DOMAIN_KEYS = ("blocksworld", "transport")
	(tmp_path / "blocksworld.summary.json").write_text(
		json.dumps(
			{
				"domain_key": "blocksworld",
				"attempted_problem_count": 1,
				"execution_time_seconds_total": 5.0,
				"execution_time_seconds_average": 5.0,
				"plan_solve_time_seconds_total": 4.0,
				"plan_solve_time_seconds_average": 4.0,
				"plan_verification_time_seconds_total": 1.0,
				"plan_verification_time_seconds_average": 1.0,
				"verified_success_count": 1,
				"verified_successes": 1,
				"hierarchical_rejection_failures": 0,
				"primitive_invalid_failures": 0,
				"solver_no_plan_failures": 0,
				"unknown_failures": 0,
				"bucket_counts": {
					"hierarchical_plan_verified": 1,
					"primitive_plan_valid_but_hierarchical_rejected": 0,
					"primitive_plan_invalid": 0,
					"no_plan_from_solver": 0,
					"unknown_failure": 0,
				},
				"query_results": [
					{
						"query_id": "blocksworld_query_01",
						"problem_file": "blocksworld.hddl",
						"log_dir": "/tmp/blocksworld",
						"success": True,
						"outcome_bucket": "hierarchical_plan_verified",
						"execution_time_seconds": 5.0,
						"plan_solve_time_seconds": 4.0,
						"plan_verification_time_seconds": 1.0,
						"representation_build_seconds": 0.2,
						"planner_wallclock_seconds": 4.5,
						"plan_solve_status": "success",
						"plan_verification_status": "success",
						"selected_solver_id": "sat",
						"selected_planner_id": "lifted_panda_sat",
						"selected_representation_id": "linearized_total_order",
					},
				],
			},
			indent=2,
		),
	)
	calls: list[str] = []
	try:
		def fake_domain_runner(
			domain_key: str,
			evaluation_mode: str,
			planner_id: str | None,
		) -> dict[str, object]:
			calls.append(domain_key)
			return {
				"domain_key": domain_key,
				"evaluation_mode": evaluation_mode,
				"requested_planner_id": planner_id,
				"attempted_problem_count": 1,
				"execution_time_seconds_total": 12.0,
				"execution_time_seconds_average": 12.0,
				"plan_solve_time_seconds_total": 9.0,
				"plan_solve_time_seconds_average": 9.0,
				"plan_verification_time_seconds_total": 3.0,
				"plan_verification_time_seconds_average": 3.0,
				"verified_success_count": 1,
				"verified_successes": 1,
				"hierarchical_rejection_failures": 0,
				"primitive_invalid_failures": 0,
				"solver_no_plan_failures": 0,
				"unknown_failures": 0,
				"bucket_counts": {
					"hierarchical_plan_verified": 1,
					"primitive_plan_valid_but_hierarchical_rejected": 0,
					"primitive_plan_invalid": 0,
					"no_plan_from_solver": 0,
					"unknown_failure": 0,
				},
				"query_results": [
					{
						"query_id": f"{domain_key}_query_01",
						"problem_file": f"{domain_key}.hddl",
						"log_dir": f"/tmp/{domain_key}",
						"success": True,
						"outcome_bucket": "hierarchical_plan_verified",
						"execution_time_seconds": 12.0,
						"plan_solve_time_seconds": 9.0,
						"plan_verification_time_seconds": 3.0,
						"representation_build_seconds": 0.5,
						"planner_wallclock_seconds": 10.5,
						"plan_solve_status": "success",
						"plan_verification_status": "success",
						"selected_solver_id": "sat",
						"selected_planner_id": "lifted_panda_sat",
						"selected_representation_id": "linearized_total_order",
					},
				],
			}

		summary = baseline_runner._run_sequential_full_baseline(
			run_dir=tmp_path,
			evaluation_mode=SINGLE_PLANNER_MODE,
			planner_id=PRIMARY_HTN_PLANNER_ID,
			track_id=PRIMARY_HTN_PLANNER_ID,
			domain_runner=fake_domain_runner,
		)
	finally:
		baseline_runner.DOMAIN_KEYS = original_domain_keys

	assert calls == ["transport"]
	assert summary["complete"] is True
	assert "blocksworld" in summary["completed_domains"]
	assert "transport" in summary["completed_domains"]


def test_track_pass_matrix_writes_compact_pass_status(
	tmp_path: Path,
) -> None:
	paths = baseline_runner._write_track_pass_matrix(
		tmp_path,
		{
			PRIMARY_HTN_PLANNER_ID: {
				"evaluation_mode": SINGLE_PLANNER_MODE,
				"requested_planner_id": PRIMARY_HTN_PLANNER_ID,
				"complete": True,
				"completed_domains": ["blocksworld", "marsrover", "satellite", "transport"],
				"track_summary": {
					"total_queries": 115,
					"verified_success_count": 110,
				},
			},
		},
	)
	rows = json.loads(Path(paths["track_pass_matrix_json"]).read_text())
	assert rows[0]["track_id"] == PRIMARY_HTN_PLANNER_ID
	assert rows[0]["pass"] is False
	assert rows[0]["all_queries_verified"] is False
	assert rows[0]["sweep_complete"] is True
	assert rows[0]["total_query_count"] == 115
	assert "verified_success_count" in Path(paths["track_pass_matrix_csv"]).read_text()
	assert "all_queries_verified" in Path(paths["track_pass_matrix_csv"]).read_text()


def test_cleanup_reclaims_only_live_project_planning_processes() -> None:
	mock_result = SimpleNamespace(
		stdout="\n".join(
			[
				f"101 1 4096 {PROJECT_ROOT}/.local/pandaPI-full/bin/pandaPIengine {PROJECT_ROOT}/tests/generated/logs/x",
				f"102 1 0 {PROJECT_ROOT}/.local/pandaPI-full/bin/pandaPIengine {PROJECT_ROOT}/tests/generated/logs/y",
				f"103 1 2048 /usr/bin/python unrelated.py",
				f"104 1 1024 {PROJECT_ROOT}/tests/run_official_problem_root_baseline.py --all-tracks",
				f"105 1 1024 {PROJECT_ROOT}/.venv/bin/python -c from multiprocessing.spawn import spawn_main",
			],
		),
	)
	killed: list[int] = []
	with patch.object(baseline_runner.subprocess, "run", return_value=mock_result), patch.object(
		baseline_runner.os,
		"getpid",
		return_value=104,
	), patch.object(baseline_runner.os, "kill", side_effect=lambda pid, _sig: killed.append(pid)):
		baseline_runner._cleanup_htn_evaluation_resources()

	assert killed == [101, 105]


def test_run_single_domain_uses_domain_output_root_without_marking_partial_domain_complete(
	tmp_path: Path,
) -> None:
	original_query_ids = list(baseline_runner._RUN_QUERY_IDS)
	original_evaluation_mode = baseline_runner._RUN_EVALUATION_MODE
	original_planner_id = baseline_runner._RUN_PLANNER_ID
	mock_summary = {
		"attempted_problem_count": 1,
		"verified_success_count": 1,
	}
	with patch(
		"tests.support.htn_evaluation_support.load_domain_query_cases",
		return_value={"query_10": {}, "query_11": {}},
	), patch(
		"tests.support.htn_evaluation_support.run_official_problem_root_baseline_for_domain",
		return_value=mock_summary,
	) as run_domain:
		try:
			baseline_runner._RUN_QUERY_IDS = ["query_10"]
			baseline_runner._RUN_EVALUATION_MODE = SINGLE_PLANNER_MODE
			baseline_runner._RUN_PLANNER_ID = PRIMARY_HTN_PLANNER_ID
			result = baseline_runner._run_single_domain("transport", tmp_path)
		finally:
			baseline_runner._RUN_QUERY_IDS = original_query_ids
			baseline_runner._RUN_EVALUATION_MODE = original_evaluation_mode
			baseline_runner._RUN_PLANNER_ID = original_planner_id

	assert result == 0
	assert run_domain.call_args.kwargs["output_root"] == tmp_path / "transport"
	assert not (tmp_path / "transport.summary.json").exists()


def test_run_single_domain_writes_top_level_summary_when_domain_is_complete(
	tmp_path: Path,
) -> None:
	original_query_ids = list(baseline_runner._RUN_QUERY_IDS)
	original_evaluation_mode = baseline_runner._RUN_EVALUATION_MODE
	original_planner_id = baseline_runner._RUN_PLANNER_ID
	mock_summary = {
		"attempted_problem_count": 2,
		"verified_success_count": 2,
	}
	with patch(
		"tests.support.htn_evaluation_support.load_domain_query_cases",
		return_value={"query_10": {}, "query_11": {}},
	), patch(
		"tests.support.htn_evaluation_support.run_official_problem_root_baseline_for_domain",
		return_value=mock_summary,
	):
		try:
			baseline_runner._RUN_QUERY_IDS = []
			baseline_runner._RUN_EVALUATION_MODE = SINGLE_PLANNER_MODE
			baseline_runner._RUN_PLANNER_ID = PRIMARY_HTN_PLANNER_ID
			result = baseline_runner._run_single_domain("transport", tmp_path)
		finally:
			baseline_runner._RUN_QUERY_IDS = original_query_ids
			baseline_runner._RUN_EVALUATION_MODE = original_evaluation_mode
			baseline_runner._RUN_PLANNER_ID = original_planner_id

	assert result == 0
	assert (tmp_path / "transport.summary.json").exists()


def test_launch_detached_controller_spawns_new_session_and_writes_state(
	tmp_path: Path,
) -> None:
	process = SimpleNamespace(pid=43210)
	with patch.object(
		baseline_runner.subprocess,
		"Popen",
		return_value=process,
	) as popen, patch.object(
		baseline_runner,
		"_timestamp",
		return_value="20260420_200000",
	):
		state = baseline_runner._launch_detached_controller(
			run_dir=tmp_path,
			all_tracks=True,
			domain=None,
			query_ids=("query_1", "query_2"),
			evaluation_mode=SINGLE_PLANNER_MODE,
			planner_id=PRIMARY_HTN_PLANNER_ID,
		)

	assert state["status"] == "launched"
	assert state["pid"] == 43210
	assert Path(state["log_file"]).name == "controller_20260420_200000.out"
	assert (tmp_path / "controller.pid").read_text().strip() == "43210"
	written_state = json.loads((tmp_path / "controller_state.json").read_text())
	assert written_state["pid"] == 43210
	assert written_state["all_tracks"] is True
	assert written_state["query_ids"] == ["query_1", "query_2"]
	popen.assert_called_once()
	assert popen.call_args.kwargs["start_new_session"] is True
	assert popen.call_args.kwargs["stdin"] == subprocess.DEVNULL
	assert popen.call_args.kwargs["env"]["HTN_EVAL_IGNORE_SIGTERM"] == "1"
	assert popen.call_args.kwargs["env"]["HTN_EVAL_CONTROLLER_LOG_FILE"].endswith(
		"controller_20260420_200000.out",
	)
	command = popen.call_args.args[0]
	assert command[:3] == [sys.executable, "-u", str((PROJECT_ROOT / "tests" / "run_official_problem_root_baseline.py").resolve())]
	assert "--all-tracks" in command


def test_launch_detached_controller_reuses_existing_live_controller(
	tmp_path: Path,
) -> None:
	(tmp_path / "controller_state.json").write_text(
		json.dumps(
			{
				"pid": 55555,
				"log_file": str(tmp_path / "existing.out"),
				"run_dir": str(tmp_path),
			},
			indent=2,
		),
	)
	with patch.object(
		baseline_runner,
		"_pid_is_alive",
		return_value=True,
	), patch.object(
		baseline_runner.subprocess,
		"Popen",
	) as popen:
		state = baseline_runner._launch_detached_controller(
			run_dir=tmp_path,
			all_tracks=True,
			domain=None,
			query_ids=(),
			evaluation_mode=SINGLE_PLANNER_MODE,
			planner_id=PRIMARY_HTN_PLANNER_ID,
		)

	assert state["status"] == "already_running"
	assert state["pid"] == 55555
	popen.assert_not_called()


def test_sequential_full_baseline_cleans_resources_between_domains(
	tmp_path: Path,
) -> None:
	original_domain_keys = baseline_runner.DOMAIN_KEYS
	baseline_runner.DOMAIN_KEYS = ("blocksworld", "transport")
	cleanup_calls: list[str] = []
	try:
		def fake_domain_runner(
			domain_key: str,
			evaluation_mode: str,
			planner_id: str | None,
		) -> dict[str, object]:
			return {
				"domain_key": domain_key,
				"evaluation_mode": evaluation_mode,
				"requested_planner_id": planner_id,
				"attempted_problem_count": 1,
				"execution_time_seconds_total": 12.0,
				"execution_time_seconds_average": 12.0,
				"plan_solve_time_seconds_total": 9.0,
				"plan_solve_time_seconds_average": 9.0,
				"plan_verification_time_seconds_total": 3.0,
				"plan_verification_time_seconds_average": 3.0,
				"verified_success_count": 1,
				"verified_successes": 1,
				"hierarchical_rejection_failures": 0,
				"primitive_invalid_failures": 0,
				"solver_no_plan_failures": 0,
				"unknown_failures": 0,
				"bucket_counts": {
					"hierarchical_plan_verified": 1,
					"primitive_plan_valid_but_hierarchical_rejected": 0,
					"primitive_plan_invalid": 0,
					"no_plan_from_solver": 0,
					"unknown_failure": 0,
				},
				"query_results": [
					{
						"query_id": f"{domain_key}_query_01",
						"problem_file": f"{domain_key}.hddl",
						"log_dir": f"/tmp/{domain_key}",
						"success": True,
						"outcome_bucket": "hierarchical_plan_verified",
						"execution_time_seconds": 12.0,
						"plan_solve_time_seconds": 9.0,
						"plan_verification_time_seconds": 3.0,
						"representation_build_seconds": 0.5,
						"planner_wallclock_seconds": 10.5,
						"plan_solve_status": "success",
						"plan_verification_status": "success",
						"selected_solver_id": "sat",
						"selected_planner_id": "lifted_panda_sat",
						"selected_representation_id": "linearized_total_order",
					},
				],
			}

		with patch.object(
			baseline_runner,
			"_cleanup_htn_evaluation_resources",
			side_effect=lambda: cleanup_calls.append("cleanup"),
		):
			summary = baseline_runner._run_sequential_full_baseline(
				run_dir=tmp_path,
				evaluation_mode=SINGLE_PLANNER_MODE,
				planner_id=PRIMARY_HTN_PLANNER_ID,
				track_id=PRIMARY_HTN_PLANNER_ID,
				domain_runner=fake_domain_runner,
			)
	finally:
		baseline_runner.DOMAIN_KEYS = original_domain_keys

	assert summary["complete"] is True
	assert len(cleanup_calls) == 6


def test_register_controller_runtime_creates_missing_run_dir(tmp_path: Path) -> None:
	run_dir = tmp_path / "missing" / "track"

	baseline_runner._register_controller_runtime(run_dir)

	assert run_dir.exists()
	assert (run_dir / "controller.pid").exists()


def test_supported_planner_ids_are_stable() -> None:
	assert HTN_PLANNER_IDS == (PRIMARY_HTN_PLANNER_ID,)


def test_apply_official_resource_profile_records_memory_and_cpu_enforcement() -> None:
	with patch.object(
		problem_root_runtime.sys,
		"platform",
		"linux",
	), patch.object(
		problem_root_runtime.resource,
		"setrlimit",
	) as setrlimit, patch.object(
		problem_root_runtime.os,
		"sched_getaffinity",
		return_value={3, 7},
		create=True,
	), patch.object(
		problem_root_runtime.os,
		"sched_setaffinity",
		create=True,
	) as setaffinity:
		profile = problem_root_runtime._apply_official_resource_profile(
			memory_limit_mib=8192,
			cpu_count=1,
		)

	assert setrlimit.called
	setaffinity.assert_called_once_with(0, {3})
	assert profile["memory_limit_enforced"] is True
	assert profile["cpu_affinity_enforced"] is True
	assert profile["requested_memory_limit_mib"] == 8192
	assert profile["requested_cpu_count"] == 1


def test_htn_verifier_evidence_is_compacted_without_raw_stdout(
	tmp_path: Path,
) -> None:
	output_file = tmp_path / "verifier.txt"
	output_file.write_text("existing oversized verifier output")
	result = SimpleNamespace(
		to_dict=lambda: {
			"tool_available": True,
			"command": ["pandaPIparser", "-V"],
			"plan_file": str(tmp_path / "plan.txt"),
			"output_file": str(output_file),
			"stdout": "a" * 20_000,
			"stderr": "b" * 12_000,
			"primitive_plan_only": False,
			"primitive_plan_executable": True,
			"verification_result": True,
			"reached_goal_state": True,
			"plan_kind": "hierarchical",
			"build_warning": None,
			"error": None,
		},
	)

	payload = problem_root_runtime._compact_verification_result(
		result,
		json_filename="verification.json",
	)

	assert "stdout" not in payload
	assert "stderr" not in payload
	assert payload["stdout_chars"] == 20_000
	assert payload["stderr_chars"] == 12_000
	assert "truncated" in payload["stdout_preview"]
	assert output_file.stat().st_size < 10_000
	written_payload = json.loads((tmp_path / "verification.json").read_text())
	assert "stdout" not in written_payload
	assert "stderr" not in written_payload


def test_primitive_executable_goal_reached_plan_is_hierarchical_rejection(
	tmp_path: Path,
) -> None:
	domain_file = tmp_path / "domain.hddl"
	problem_file = tmp_path / "problem.hddl"
	actual_plan = tmp_path / "actual.plan"
	output_dir = tmp_path / "out"
	domain_file.write_text("(domain)")
	problem_file.write_text("(problem)")
	actual_plan.write_text("==>\nroot\n")
	output_dir.mkdir()

	class FakeVerifier:
		@staticmethod
		def tool_available() -> bool:
			return True

		@staticmethod
		def verify_primitive_plan(**kwargs: object) -> object:
			output_path = Path(str(kwargs["output_dir"]))
			plan_file = output_path / str(kwargs["plan_filename"])
			output_file = output_path / str(kwargs["output_filename"])
			plan_file.write_text("primitive")
			output_file.write_text("Primitive plan alone executable: true")
			return SimpleNamespace(
				tool_available=True,
				primitive_plan_executable=True,
				verification_result=False,
				reached_goal_state=True,
				to_dict=lambda: {
					"tool_available": True,
					"command": ["pandaPIparser", "-V"],
					"plan_file": str(plan_file),
					"output_file": str(output_file),
					"stdout": "",
					"stderr": "",
					"primitive_plan_only": True,
					"primitive_plan_executable": True,
					"verification_result": False,
					"reached_goal_state": True,
					"plan_kind": "primitive_only",
					"build_warning": None,
					"error": "verifier exited with code 1",
				},
			)

		@staticmethod
		def verify_plan_text(**kwargs: object) -> object:
			output_path = Path(str(kwargs["output_dir"]))
			plan_file = output_path / str(kwargs["plan_filename"])
			output_file = output_path / str(kwargs["output_filename"])
			plan_file.write_text("hierarchical")
			output_file.write_text("Plan verification result: false")
			return SimpleNamespace(
				tool_available=True,
				plan_kind="hierarchical",
				verification_result=False,
				to_dict=lambda: {
					"tool_available": True,
					"command": ["pandaPIparser", "-V"],
					"plan_file": str(plan_file),
					"output_file": str(output_file),
					"stdout": "",
					"stderr": "",
					"primitive_plan_only": False,
					"primitive_plan_executable": None,
					"verification_result": False,
					"reached_goal_state": None,
					"plan_kind": "hierarchical",
					"build_warning": None,
					"error": "verifier exited with code 1",
				},
			)

	fake_context = SimpleNamespace(
		domain_file=str(domain_file),
		problem_file=str(problem_file),
		output_dir=output_dir,
		logger=problem_root_runtime._NullExecutionLogger(),
		_record_step_timing=lambda *_args, **_kwargs: None,
	)

	result = problem_root_runtime.verify_primary_planner_solution(
		fake_context,
		verifier=FakeVerifier(),
		plan_solve_data={"summary": {"status": "success"}},
		plan_solve_artifacts={
			"solver_candidates": [
				{
					"solver_id": PRIMARY_HTN_PLANNER_ID,
					"mode": "sat",
					"status": "success",
					"action_path": ["drive(truck-0, a, b)"],
					"actual_plan_path": str(actual_plan),
				},
			],
		},
		stage_start=0.0,
	)

	assert result["summary"]["status"] == "failed"
	assert result["summary"]["failure_bucket"] == "primitive_plan_valid_but_hierarchical_rejected"
	assert result["artifacts"]["selected_bucket"] == "primitive_plan_valid_but_hierarchical_rejected"


def test_htn_plan_solve_evidence_drops_large_inline_runtime_payloads(
	tmp_path: Path,
) -> None:
	actual_plan = tmp_path / "plan.actual"
	actual_plan.write_text("do-a\n")
	payload = {
		"summary": {"status": "success", "step_count": 2},
		"artifacts": {
			"planner_id": PRIMARY_HTN_PLANNER_ID,
			"status": "success",
			"planning_mode": "official_problem_root",
			"engine_mode": "sat",
			"solver_id": "sat",
			"planning_representation": {
				"representation_id": "linearized_total_order",
				"representation_source": "linearized",
				"ordering_kind": "total_order",
				"domain_file": str(tmp_path / "domain.hddl"),
				"problem_file": str(tmp_path / "problem.hddl"),
				"metadata": {"large": "x" * 10_000},
			},
			"task_network": [{"task_name": "root", "args": []}],
			"ordering_edges": [{"before": "s1", "after": "s2"}],
			"action_path": ["a", "b"],
			"method_trace": [{"method": "m"}],
			"guided_hierarchical_plan_text": "plan text" * 1000,
			"solver_candidates": [
				{
					"solver_id": "sat",
					"status": "success",
					"action_path": ["a", "b"],
					"actual_plan_path": str(actual_plan),
					"engine_stdout": "raw stdout" * 1000,
				},
			],
			"artifacts": {
				"actual_plan": str(actual_plan),
			},
		},
	}

	compacted = problem_root_runtime._compact_plan_solve_data_for_parent(payload)
	artifacts = compacted["artifacts"]

	assert "action_path" not in artifacts
	assert "method_trace" not in artifacts
	assert "guided_hierarchical_plan_text" not in artifacts
	assert artifacts["task_network_count"] == 1
	assert artifacts["ordering_edge_count"] == 1
	assert artifacts["solver_candidates"][0]["action_path_count"] == 2
	assert "engine_stdout" not in artifacts["solver_candidates"][0]
	assert artifacts["artifact_files"]["actual_plan"] == str(actual_plan)


def test_planning_tasks_can_filter_to_one_requested_planner() -> None:
	if not baseline_support.lifted_linear_toolchain_available():
		import pytest

		pytest.skip("Lifted Linear/PANDA toolchain is not available.")
	pipeline = HTNEvaluationPipeline(
		domain_file=DOMAIN_FILES["transport"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "transport" / "problems" / "pfile01.hddl").resolve()
		),
	)
	pipeline.output_dir = str(PROJECT_ROOT / "tests" / "generated" / "tmp-planning-tasks")
	evaluator = HTNProblemRootEvaluator(pipeline.context)

	tasks = evaluator.planning_tasks(planner_id="lifted_panda_sat")

	assert tasks
	assert all(task.planner_id == "lifted_panda_sat" for task in tasks)


def test_lifted_planner_forces_linearized_representation_for_total_order_problem(
	tmp_path: Path,
) -> None:
	original_representation = PlanningRepresentation(
		representation_id="original_total_order",
		representation_source="original",
		ordering_kind="total_order",
		domain_file=str(tmp_path / "domain.hddl"),
		problem_file=str(tmp_path / "problem.hddl"),
		compilation_profile="identity",
		metadata={"requires_linearization": False},
	)
	context = SimpleNamespace(
		output_dir=tmp_path,
		domain_file=str(tmp_path / "domain.hddl"),
		problem_file=str(tmp_path / "problem.hddl"),
		_build_problem_representations=Mock(
			return_value=RepresentationBuildResult(
				structure=SimpleNamespace(requires_linearization=False),
				representations=(original_representation,),
			),
		),
	)
	linearized_domain = tmp_path / "domain.linearized.hddl"
	linearized_problem = tmp_path / "problem.linearized.hddl"
	with patch.object(problem_root_evaluator, "LiftedLinearPlanner") as planner_cls:
		planner_cls.return_value.linearize_hddl_files.return_value = {
			"linearized_domain_file": str(linearized_domain),
			"linearized_problem_file": str(linearized_problem),
			"linearizer_seconds": 0.1,
		}

		tasks = HTNProblemRootEvaluator(context).planning_tasks(
			planner_id="lifted_panda_sat",
		)

	assert len(tasks) == 1
	assert tasks[0].planner_id == "lifted_panda_sat"
	assert tasks[0].representation.representation_id == "linearized_total_order"
	assert tasks[0].representation.metadata["forced_for_primary_planner_capability"] is True


def test_primary_planner_evaluation_records_representation_build_failure_as_query_failure(
	tmp_path: Path,
) -> None:
	original_representation = PlanningRepresentation(
		representation_id="original_total_order",
		representation_source="original",
		ordering_kind="total_order",
		domain_file=str(tmp_path / "domain.hddl"),
		problem_file=str(tmp_path / "problem.hddl"),
		compilation_profile="identity",
		metadata={"requires_linearization": False},
	)
	context = SimpleNamespace(
		output_dir=tmp_path,
		domain_file=str(tmp_path / "domain.hddl"),
		problem_file=str(tmp_path / "problem.hddl"),
		_build_problem_representations=Mock(
			return_value=RepresentationBuildResult(
				structure=SimpleNamespace(requires_linearization=False),
				representations=(original_representation,),
			),
		),
		_official_problem_root_planning_timeout_seconds=Mock(return_value=1800.0),
	)
	with patch.object(problem_root_evaluator, "LiftedLinearPlanner") as planner_cls:
		planner_cls.return_value.linearize_hddl_files.side_effect = PANDAPlanningError(
			"Lifted Linear linearization failed.",
			metadata={
				"linearizer_stdout": "linearizer out",
				"linearizer_stderr": "linearizer err",
				"engine_attempts": [{"solver_id": "lifted_linear_config_2"}],
			},
		)

		result = HTNProblemRootEvaluator(context).run_primary_planner_evaluation(
			evaluation_mode=SINGLE_PLANNER_MODE,
			planner_id="lifted_panda_sat",
		)

	assert result["planning_tasks"] == []
	assert len(result["attempts"]) == 1
	selected_attempt = result["selected_attempt"]
	assert selected_attempt["planner_id"] == "lifted_panda_sat"
	assert selected_attempt["representation_id"] == "linearized_total_order"
	assert selected_attempt["selected_bucket"] == "no_plan_from_solver"
	assert selected_attempt["plan_solve_data"]["summary"]["status"] == "failed"
	assert selected_attempt["plan_solve_data"]["summary"]["failure_stage"] == "representation_build"
	assert "Lifted Linear linearization failed" in selected_attempt["plan_solve_data"]["summary"]["failure_reason"]
	assert selected_attempt["stdout"] == "linearizer out"
	assert selected_attempt["stderr"] == "linearizer err"


def test_primary_planner_evaluation_runs_tasks_sequentially_to_cap_memory_pressure(
	tmp_path: Path,
) -> None:
	pipeline = HTNEvaluationPipeline(
		domain_file=DOMAIN_FILES["transport"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "transport" / "problems" / "pfile01.hddl").resolve()
		),
	)
	pipeline.output_dir = str(tmp_path / "official-eval")
	evaluator = HTNProblemRootEvaluator(pipeline.context)

	def make_task(task_id: str, representation_id: str) -> SimpleNamespace:
		representation = SimpleNamespace(
			representation_id=representation_id,
			to_dict=lambda: {"representation_id": representation_id},
		)
		return SimpleNamespace(
			task_id=task_id,
			planner_id="lifted_panda_sat",
			representation=representation,
			to_dict=lambda: {
				"task_id": task_id,
				"planner_id": "lifted_panda_sat",
				"representation": {"representation_id": representation_id},
			},
		)

	tasks = (
		make_task("task_a", "rep_a"),
		make_task("task_b", "rep_b"),
		make_task("task_c", "rep_c"),
	)
	evaluator.planning_tasks = Mock(return_value=tasks)  # type: ignore[method-assign]
	pipeline.context._official_problem_root_planning_timeout_seconds = Mock(return_value=60.0)  # type: ignore[method-assign]

	class FakeQueue:
		def __init__(self) -> None:
			self.items: list[dict[str, object]] = []

		def put(self, item: dict[str, object]) -> None:
			self.items.append(item)

		def get(self, timeout: float | None = None) -> dict[str, object]:
			if not self.items:
				raise queue.Empty
			return self.items.pop(0)

		def close(self) -> None:
			return None

		def join_thread(self) -> None:
			return None

	class FakeProcess:
		active_count = 0
		max_active_count = 0

		def __init__(self, *, kwargs: dict[str, object]) -> None:
			self.kwargs = kwargs
			self.pid = id(self)
			self._alive = False

		def start(self) -> None:
			FakeProcess.active_count += 1
			FakeProcess.max_active_count = max(
				FakeProcess.max_active_count,
				FakeProcess.active_count,
			)
			self._alive = True
			task_payload = dict(self.kwargs["task_payload"])  # type: ignore[index]
			output_dir = str(self.kwargs["output_dir"])  # type: ignore[index]
			self.kwargs["result_queue"].put(  # type: ignore[index]
				{
					"message_type": "primary_planner_attempt",
					"planner_id": task_payload["planner_id"],
					"task_id": task_payload["task_id"],
					"representation_id": task_payload["representation"]["representation_id"],
					"output_dir": output_dir,
					"plan_solve_data": {
						"summary": {"status": "failed"},
						"artifacts": {},
					},
					"plan_verification_data": {
						"summary": {"status": "failed"},
						"artifacts": {},
					},
					"plan_solve_seconds": 1.0,
					"plan_verification_seconds": 0.0,
					"total_seconds": 1.0,
					"success": False,
					"selected_bucket": "no_plan_from_solver",
					"stdout": "",
					"stderr": "",
				},
			)

		def join(self, timeout: float | None = None) -> None:
			if self._alive:
				self._alive = False
				FakeProcess.active_count -= 1

		def is_alive(self) -> bool:
			return self._alive

		def terminate(self) -> None:
			self.join()

		def kill(self) -> None:
			self.join()

	class FakeContext:
		def Queue(self) -> FakeQueue:
			return FakeQueue()

		def Process(self, target=None, kwargs=None):  # type: ignore[no-untyped-def]
			return FakeProcess(kwargs=kwargs or {})

	with patch.object(
		problem_root_evaluator.multiprocessing,
		"get_context",
		return_value=FakeContext(),
	):
		result = evaluator.run_primary_planner_evaluation(
			evaluation_mode=SINGLE_PLANNER_MODE,
			planner_id="lifted_panda_sat",
		)

	assert len(result["attempts"]) == 3
	assert FakeProcess.max_active_count == 1


def test_run_subprocess_to_files_spools_large_outputs_without_returning_full_payload(
	tmp_path: Path,
) -> None:
	large_stdout = "S" * (PROCESS_OUTPUT_PREVIEW_BYTE_LIMIT + 4096)
	large_stderr = "E" * (PROCESS_OUTPUT_PREVIEW_BYTE_LIMIT + 2048)
	result = run_subprocess_to_files(
		[
			sys.executable,
			"-c",
			(
				"import sys; "
				f"sys.stdout.write({large_stdout!r}); "
				f"sys.stderr.write({large_stderr!r})"
			),
		],
		work_dir=tmp_path,
		output_label="oversized_solver_output",
		timeout_seconds=10.0,
	)

	assert result["returncode"] == 0
	assert result["stdout_truncated"] is True
	assert result["stderr_truncated"] is True
	assert "...[truncated " in result["stdout"]
	assert "...[truncated " in result["stderr"]
	assert len(result["stdout"]) < len(large_stdout)
	assert len(result["stderr"]) < len(large_stderr)
	assert read_full_process_output(result["stdout_path"]) == large_stdout
	assert read_full_process_output(result["stderr_path"]) == large_stderr


def test_spawn_worker_import_path_stays_valid_for_htn_runtime() -> None:
	env = dict(os.environ)
	pythonpath_entries = [str(SRC_ROOT), str(PROJECT_ROOT)]
	existing_pythonpath = env.get("PYTHONPATH", "")
	if existing_pythonpath:
		pythonpath_entries.append(existing_pythonpath)
	env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
	result = subprocess.run(
		[
			sys.executable,
			"-c",
			"import htn_evaluation.problem_root_runtime",
		],
		cwd=PROJECT_ROOT,
		env=env,
		capture_output=True,
		text=True,
		check=False,
	)
	assert result.returncode == 0, result.stderr
