from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from planning.backends import LiftedPandaBackend, PandaDealerBackend
from planning.official_benchmark import OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS
from planning.plan_models import PANDAPlanResult
from planning.process_capture import (
	PROCESS_OUTPUT_PREVIEW_BYTE_LIMIT,
	read_full_process_output,
	run_subprocess_to_files,
)
from planning.representations import PlanningRepresentation
from htn_evaluation.pipeline import HTNEvaluationPipeline
from htn_evaluation.problem_root_evaluator import HTNProblemRootEvaluator
from htn_evaluation.result_tables import (
	HTN_PLANNER_IDS,
	PLANNER_OR_RACE_MODE,
	SINGLE_PLANNER_MODE,
	build_planner_capability_rows,
	build_problem_result_row,
	build_track_summary,
	write_planner_capability_matrix,
)

import tests.support.htn_evaluation_support as baseline_support
from tests.support.offline_generation_support import DOMAIN_FILES, build_official_method_library


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


def test_merge_official_backend_output_dir_skips_unreadable_backend_root(
	tmp_path: Path,
) -> None:
	pipeline = HTNEvaluationPipeline(
		domain_file=DOMAIN_FILES["blocksworld"],
		problem_file=str(
			(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl").resolve()
		),
	)
	pipeline.output_dir = str(tmp_path / "selected-root")
	unreadable_root = tmp_path / "backend-root"
	unreadable_root.mkdir(parents=True, exist_ok=True)
	original_iterdir = Path.iterdir

	def fake_iterdir(path: Path):
		if path == unreadable_root:
			raise PermissionError("operation not permitted")
		return original_iterdir(path)

	with patch.object(Path, "iterdir", fake_iterdir):
		pipeline._merge_official_backend_output_dir(unreadable_root)

	assert (tmp_path / "selected-root").exists()


def test_close_backend_race_queue_closes_and_joins_thread() -> None:
	close = Mock()
	join_thread = Mock()
	queue_stub = SimpleNamespace(close=close, join_thread=join_thread)

	HTNEvaluationPipeline._close_backend_race_queue(queue_stub)

	close.assert_called_once_with()
	join_thread.assert_called_once_with()


def test_parallel_solver_race_delegates_to_hierarchical_task_network_evaluator() -> None:
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

	result = pipeline._execute_official_problem_root_parallel_solver_race(method_library="sentinel")

	assert result["plan_solve"]["summary"]["status"] == "success"
	assert fake_evaluator.calls == [
		{
			"method_library": "sentinel",
			"evaluation_mode": PLANNER_OR_RACE_MODE,
			"planner_id": None,
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
				"plan_solve": {"summary": {"status": "success"}},
				"plan_verification": {
					"summary": {"status": "success"},
					"artifacts": {
						"selected_solver_id": "sat",
						"selected_backend_name": "lifted_panda_sat",
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
	problem_results_path = Path(summary["output_paths"]["problem_results"])
	domain_summary_path = Path(summary["output_paths"]["domain_summary"])
	assert problem_results_path.exists()
	assert domain_summary_path.exists()
	assert problem_results_path.parent == tmp_path / "lifted_panda_sat" / "transport"


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
			"plan_solve": {"summary": {"status": "success"}},
			"plan_verification": {
				"summary": {"status": "success"},
				"artifacts": {
					"selected_solver_id": "sat",
					"selected_backend_name": "lifted_panda_sat",
					"selected_representation_id": "linearized_total_order",
				},
			},
		},
		evaluation_mode=PLANNER_OR_RACE_MODE,
		planner_id=None,
	)
	track_summary = build_track_summary(
		run_dir=tmp_path,
		domain_summaries={
			"blocksworld": {
				"attempted_problem_count": 1,
				"verified_success_count": 1,
				"bucket_counts": {
					"hierarchical_plan_verified": 1,
					"primitive_plan_valid_but_hierarchical_rejected": 0,
					"primitive_plan_invalid": 0,
					"no_plan_from_solver": 0,
					"unknown_failure": 0,
				},
			},
		},
		evaluation_mode=PLANNER_OR_RACE_MODE,
		planner_id=None,
	)
	rows = build_planner_capability_rows((track_summary,))
	paths = write_planner_capability_matrix(tmp_path, rows=rows)

	assert problem_row["selected_backend_name"] == "lifted_panda_sat"
	assert rows[0]["track_id"] == PLANNER_OR_RACE_MODE
	assert rows[0]["verified_success_count"] == 1
	assert Path(paths["planner_capability_matrix_json"]).exists()
	assert Path(paths["planner_capability_matrix_csv"]).exists()
	assert "domain_key" in Path(paths["planner_capability_matrix_csv"]).read_text()


def test_supported_planner_ids_are_stable() -> None:
	assert HTN_PLANNER_IDS == (
		"panda_pi_portfolio",
		"pandadealer_agile_lama",
		"lifted_panda_sat",
	)


def test_planning_tasks_can_filter_to_one_requested_planner() -> None:
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
	assert all(task.backend_name == "lifted_panda_sat" for task in tasks)


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
