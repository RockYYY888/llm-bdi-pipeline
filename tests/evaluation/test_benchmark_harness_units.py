from __future__ import annotations

import sys
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from tests.support.plan_library_evaluation_support import (
	GOAL_GROUNDING_PROVIDER_UNAVAILABLE_BUCKET,
	BENCHMARK_EVALUATION_DOMAIN_SOURCE,
	BENCHMARK_EVALUATION_LIBRARY_SOURCE,
	GENERATED_LIBRARY_ARTIFACT_REQUIRED_FILES,
	_classify_evaluation_failure,
	_compact_failure_signature,
	_extract_failure_signature,
	_extract_reported_evaluation_domain_source,
	apply_evaluation_runtime_defaults,
	run_plan_library_evaluation_case,
	run_plan_library_evaluation_benchmark_for_domain,
)
import tests.support.plan_library_evaluation_support as benchmark_support
import tests.run_plan_library_evaluation_benchmark as benchmark_runner
from tests.run_plan_library_evaluation_benchmark import (
	DomainRun,
	_collect_domain_run_result,
	_load_existing_domain_summaries,
	_load_failed_query_ids_for_run,
	_write_latest_run_manifest,
	_write_run_summary_snapshot,
)
from evaluation.artifacts import TemporalGroundingResult
from temporal_specification import TemporalSpecificationRecord


def _write_generated_library_artifact_files(artifact_root: Path) -> None:
	artifact_root.mkdir(parents=True, exist_ok=True)
	for file_name in GENERATED_LIBRARY_ARTIFACT_REQUIRED_FILES:
		(artifact_root / file_name).write_text("{}", encoding="utf-8")


def test_evaluation_benchmark_runtime_defaults_pin_benchmark_domain_source() -> None:
	env = apply_evaluation_runtime_defaults({})

	assert env["EVALUATION_DOMAIN_SOURCE"] == BENCHMARK_EVALUATION_DOMAIN_SOURCE
	assert env["EVALUATION_DOMAIN_SOURCE"] == "benchmark"


def test_generated_library_artifact_reuses_only_complete_cached_bundle(
	tmp_path: Path,
	monkeypatch,
) -> None:
	artifact_root = tmp_path / "generated_masked" / "blocksworld"
	_write_generated_library_artifact_files(artifact_root)

	def fail_if_called(domain_key: str) -> dict[str, object]:
		raise AssertionError(f"unexpected generated rebuild for {domain_key}")

	monkeypatch.setattr(
		benchmark_support,
		"GENERATED_MASKED_DOMAIN_BUILDS_DIR",
		tmp_path / "generated_masked",
	)
	monkeypatch.setattr(benchmark_support, "run_generated_domain_build", fail_if_called)

	assert (
		benchmark_support.ensure_generated_library_artifact("blocksworld")
		== artifact_root.resolve()
	)


def test_generated_library_artifact_rebuilds_incomplete_cached_bundle(
	tmp_path: Path,
	monkeypatch,
) -> None:
	artifact_root = tmp_path / "generated_masked" / "blocksworld"
	artifact_root.mkdir(parents=True)
	(artifact_root / "method_library.json").write_text("{}", encoding="utf-8")
	calls: list[str] = []

	def fake_run_generated_domain_build(domain_key: str) -> dict[str, object]:
		calls.append(domain_key)
		_write_generated_library_artifact_files(artifact_root)
		return {"success": True, "artifact_root": artifact_root}

	monkeypatch.setattr(
		benchmark_support,
		"GENERATED_MASKED_DOMAIN_BUILDS_DIR",
		tmp_path / "generated_masked",
	)
	monkeypatch.setattr(
		benchmark_support,
		"run_generated_domain_build",
		fake_run_generated_domain_build,
	)

	assert (
		benchmark_support.ensure_generated_library_artifact("blocksworld")
		== artifact_root.resolve()
	)
	assert calls == ["blocksworld"]


def test_full_benchmark_domain_launcher_forwards_selected_query_ids(
	tmp_path: Path,
	monkeypatch,
) -> None:
	captured: dict[str, object] = {}

	class FakeProcess:
		pid = 12345

		def wait(self) -> int:
			return 0

	def fake_popen(command, **kwargs):
		captured["command"] = list(command)
		captured["kwargs"] = dict(kwargs)
		return FakeProcess()

	monkeypatch.setattr(benchmark_runner.subprocess, "Popen", fake_popen)
	monkeypatch.setattr(benchmark_runner, "_RUN_QUERY_IDS", ["query_1"])
	monkeypatch.setattr(benchmark_runner, "_RUN_FAILED_ONLY_QUERY_IDS", {})
	monkeypatch.setattr(benchmark_runner, "_RUN_LIBRARY_SOURCE", "official")
	monkeypatch.setattr(benchmark_runner, "_RUN_RUNTIME_BACKEND", "jason")
	monkeypatch.setattr(benchmark_runner, "_RUN_RESUME", False)

	domain_run = benchmark_runner._start_domain_run(tmp_path, "blocksworld", {})
	domain_run.output_handle.close()

	assert captured["command"].count("--query-id") == 1
	query_flag_index = captured["command"].index("--query-id")
	assert captured["command"][query_flag_index + 1] == "query_1"
	assert "--runtime-backend" in captured["command"]
	assert captured["command"][captured["command"].index("--runtime-backend") + 1] == "jason"


def test_evaluation_benchmark_execution_source_defaults_to_benchmark_when_metadata_missing() -> None:
	assert _extract_reported_evaluation_domain_source({}) == "benchmark"


def test_goal_grounding_provider_unavailable_has_separate_bucket() -> None:
	bucket = _classify_evaluation_failure(
		{
			"success": False,
			"step": "goal_grounding",
			"failure_class": "goal_grounding_provider_unavailable",
		},
		{
			"goal_grounding": {
				"metadata": {"failure_class": "goal_grounding_provider_unavailable"},
				"error": "Goal-grounding provider did not return usable completion text.",
			},
		},
	)

	assert bucket == GOAL_GROUNDING_PROVIDER_UNAVAILABLE_BUCKET


def test_goal_grounding_validation_error_remains_grounding_failed() -> None:
	bucket = _classify_evaluation_failure(
		{
			"success": False,
			"step": "goal_grounding",
			"error": 'LTLf formula references unknown grounded task "do_fake".',
		},
		{
			"goal_grounding": {
				"metadata": {"failure_class": "goal_grounding_failed"},
				"error": 'LTLf formula references unknown grounded task "do_fake".',
			},
		},
	)

	assert bucket == "goal_grounding_failed"


def test_runtime_repair_success_uses_separate_goal_verified_bucket() -> None:
	bucket = _classify_evaluation_failure(
		{
			"success": True,
			"plan_verification_summary": {
				"plan_kind": "primitive_only",
				"primitive_plan_executable": True,
				"runtime_goal_reached": True,
			},
		},
		{},
	)

	assert bucket == "runtime_goal_verified"

	assert _classify_evaluation_failure(
		{
			"success": True,
			"plan_verification": {
				"summary": {
					"plan_kind": "primitive_only",
					"primitive_plan_executable": True,
					"runtime_goal_reached": True,
				},
			},
		},
		{},
	) == "runtime_goal_verified"


def test_evaluation_benchmark_standard_library_source_is_benchmark() -> None:
	assert BENCHMARK_EVALUATION_LIBRARY_SOURCE == "benchmark"


def test_full_sweep_writes_latest_run_manifest_files(tmp_path: Path) -> None:
	run_dir = tmp_path / "20260420_220000"
	run_dir.mkdir(parents=True)
	(run_dir / "summary.txt").write_text("run_id: 20260420_220000\nverified_successes: 3\n")
	summary = {
		"run_id": "20260420_220000",
		"run_dir": str(run_dir),
		"evaluation_domain_source": "benchmark",
		"library_source": "benchmark",
		"total_queries": 4,
		"verified_successes": 3,
		"completed_domains": ["blocksworld"],
		"internal_failures": [],
		"complete": False,
	}

	original_runs_root = _write_latest_run_manifest.__globals__["RUNS_ROOT"]
	_write_latest_run_manifest.__globals__["RUNS_ROOT"] = tmp_path
	try:
		_write_latest_run_manifest(run_dir, summary)
	finally:
		_write_latest_run_manifest.__globals__["RUNS_ROOT"] = original_runs_root

	latest = json.loads((tmp_path / "latest.json").read_text())
	assert latest["run_id"] == "20260420_220000"
	assert latest["run_dir"] == str(run_dir)
	assert latest["summary_json"] == str(run_dir / "summary.json")
	assert latest["summary_txt"] == str(run_dir / "summary.txt")
	assert (tmp_path / "latest_summary.txt").read_text().startswith("run_id: 20260420_220000\n")
	assert json.loads((tmp_path / "latest_summary.json").read_text())["verified_successes"] == 3
	assert "verified_successes: 3\n" in (tmp_path / "latest_summary.txt").read_text()


def test_domain_benchmark_resume_reuses_query_checkpoints(
	tmp_path: Path,
	monkeypatch,
) -> None:
	query_cases = {
		"q1": {"problem_file": tmp_path / "p1.hddl", "instruction": "q1 instruction"},
		"q2": {"problem_file": tmp_path / "p2.hddl", "instruction": "q2 instruction"},
	}
	monkeypatch.setattr(
		benchmark_support,
		"load_domain_query_cases",
		lambda domain_key: query_cases,
	)
	calls: list[tuple[str, str]] = []

	def fake_run_plan_library_evaluation_case(
		domain_key: str,
		query_id: str,
		*,
		library_source: str,
		runtime_backend: str = "jason",
		logs_root: str | Path | None = None,
	):
		calls.append((query_id, str(Path(logs_root or "").resolve())))
		return {
			"query_id": query_id,
			"instruction": str(query_cases[query_id]["instruction"]),
			"problem_file": str(query_cases[query_id]["problem_file"]),
			"library_source": library_source,
			"runtime_backend": runtime_backend,
			"success": True,
			"result": {"success": True, "step": ""},
			"outcome_bucket": "hierarchical_plan_verified",
			"log_dir": str((Path(logs_root or tmp_path) / query_id).resolve()),
			"execution": {
				"runtime_execution": {
					"metadata": {"verification_mode": "original_problem"},
				},
			},
			"failure_signature": {
				"ltlf_formula": None,
				"ltlf_atom_count": 0,
				"ltlf_operator_counts": {},
				"jason_failure_class": None,
				"failed_goals": [],
				"verifier_missing_goal_facts": [],
			},
			"evaluation_domain_source": "benchmark",
		}

	monkeypatch.setattr(
		benchmark_support,
		"run_plan_library_evaluation_case",
		fake_run_plan_library_evaluation_case,
	)
	run_root = tmp_path / "run-001"
	domain_root = run_root / "blocksworld"
	query_results_root = domain_root / "query_results"
	query_results_root.mkdir(parents=True)
	(query_results_root / "q1.json").write_text(
		json.dumps(
			{
				"run_id": "run-001",
				"domain_key": "blocksworld",
				"query_id": "q1",
				"instruction": "q1 instruction",
				"problem_file": str(query_cases["q1"]["problem_file"]),
				"library_source": "benchmark",
				"runtime_backend": "jason",
				"success": True,
				"result": {"success": True, "step": ""},
				"outcome_bucket": "hierarchical_plan_verified",
				"log_dir": str((domain_root / "logs" / "q1").resolve()),
				"execution": {
					"runtime_execution": {
						"metadata": {"verification_mode": "original_problem"},
					},
				},
				"failure_signature": {
					"ltlf_formula": None,
					"ltlf_atom_count": 0,
					"ltlf_operator_counts": {},
					"jason_failure_class": None,
					"failed_goals": [],
					"verifier_missing_goal_facts": [],
				},
				"evaluation_domain_source": "benchmark",
			},
			indent=2,
		),
	)

	summary = run_plan_library_evaluation_benchmark_for_domain(
		"blocksworld",
		query_ids=("q1", "q2"),
		output_root=run_root,
		run_id="run-001",
		resume=True,
	)

	assert calls == [("q2", str((domain_root / "logs").resolve()))]
	assert summary["complete"] is True
	assert summary["run_id"] == "run-001"
	assert summary["completed_query_ids"] == ["q1", "q2"]
	assert summary["resumed_query_ids"] == ["q1"]
	assert summary["query_results"][0]["run_id"] == "run-001"
	assert summary["query_results"][0]["query_result_path"].endswith("q1.json")
	assert summary["query_results"][0]["ltlf_formula"] is None
	assert (run_root / "blocksworld.summary.json").exists()
	assert (domain_root / "summary.json").exists()
	assert (domain_root / "state.json").exists()
	assert json.loads((query_results_root / "q2.json").read_text()) == {
		"run_id": "run-001",
		"domain_key": "blocksworld",
		"query_id": "q2",
		"instruction": "q2 instruction",
		"problem_file": str(query_cases["q2"]["problem_file"]),
		"library_source": "benchmark",
		"runtime_backend": "jason",
		"success": True,
		"result": {"success": True, "step": ""},
		"outcome_bucket": "hierarchical_plan_verified",
		"goal_grounding_failure_class": "",
		"log_dir": str((domain_root / "logs" / "q2").resolve()),
		"execution_path": str((domain_root / "logs" / "q2" / "execution.json").resolve()),
		"execution_summary": {
			"runtime_execution": {
				"metadata": {"verification_mode": "original_problem"},
			},
		},
		"verification_mode": "original_problem",
		"failure_signature": {
			"ltlf_formula": None,
			"ltlf_atom_count": 0,
			"ltlf_operator_counts": {},
			"jason_failure_class": None,
			"failed_goals": [],
			"verifier_missing_goal_facts": [],
		},
		"ltlf_formula": None,
		"ltlf_atom_count": 0,
		"ltlf_operator_counts": {},
		"jason_failure_class": None,
		"failed_goals": [],
		"verifier_missing_goal_facts": [],
		"evaluation_domain_source": "benchmark",
	}


def test_domain_benchmark_supports_single_query_runs(
	tmp_path: Path,
	monkeypatch,
) -> None:
	query_cases = {
		"q1": {"problem_file": tmp_path / "p1.hddl", "instruction": "q1 instruction"},
		"q2": {"problem_file": tmp_path / "p2.hddl", "instruction": "q2 instruction"},
	}
	monkeypatch.setattr(
		benchmark_support,
		"load_domain_query_cases",
		lambda domain_key: query_cases,
	)

	def fake_run_plan_library_evaluation_case(
		domain_key: str,
		query_id: str,
		*,
		library_source: str,
		runtime_backend: str = "jason",
		logs_root: str | Path | None = None,
	):
		return {
			"query_id": query_id,
			"instruction": str(query_cases[query_id]["instruction"]),
			"problem_file": str(query_cases[query_id]["problem_file"]),
			"library_source": library_source,
			"runtime_backend": runtime_backend,
			"success": False,
			"result": {"success": False, "step": "runtime_execution"},
			"outcome_bucket": "runtime_execution_failed",
			"log_dir": str((Path(logs_root or tmp_path) / query_id).resolve()),
			"execution": {"jason_failure_class": "runtime_failure_marker"},
			"failure_signature": {
				"ltlf_formula": None,
				"ltlf_atom_count": 0,
				"ltlf_operator_counts": {},
				"jason_failure_class": "runtime_failure_marker",
				"failed_goals": [],
				"verifier_missing_goal_facts": [],
			},
			"evaluation_domain_source": "benchmark",
		}

	monkeypatch.setattr(
		benchmark_support,
		"run_plan_library_evaluation_case",
		fake_run_plan_library_evaluation_case,
	)
	summary = run_plan_library_evaluation_benchmark_for_domain(
		"blocksworld",
		query_ids=("q2",),
		output_root=tmp_path / "single-query-run",
		run_id="single-query-run",
	)

	assert summary["total_queries"] == 1
	assert summary["selected_query_ids"] == ["q2"]
	assert summary["runtime_execution_failures"] == 1
	assert summary["complete"] is True
	assert summary["query_results"][0]["run_id"] == "single-query-run"
	assert summary["query_results"][0]["jason_failure_class"] == "runtime_failure_marker"
	assert summary["query_results"][0]["query_result_path"].endswith("q2.json")
	assert (
		tmp_path
		/ "single-query-run"
		/ "blocksworld"
		/ "query_results"
		/ "q2.json"
	).exists()


def test_query_report_checkpoint_compacts_long_query_and_formula_text() -> None:
	long_instruction = " ".join("move_block" for _ in range(600))
	long_formula = " & ".join(f"do_move(b{i},b{i + 1})" for i in range(500))

	report = benchmark_support._serialize_query_report(
		{
			"run_id": "compact-run",
			"domain_key": "blocksworld",
			"query_id": "query_30",
			"case": {
				"instruction": long_instruction,
				"problem_file": "/tmp/problem.hddl",
			},
			"library_source": "official",
			"success": True,
			"result": {"success": True},
			"outcome_bucket": "runtime_goal_verified",
			"execution": {
				"ltlf_formula": long_formula,
				"runtime_execution": {
					"metadata": {"verification_mode": "original_problem"},
				},
			},
			"failure_signature": {
				"ltlf_formula": long_formula,
				"ltlf_atom_count": 500,
				"ltlf_operator_counts": {"&": 499},
				"jason_failure_class": None,
				"failed_goals": [],
				"verifier_missing_goal_facts": [],
			},
			"evaluation_domain_source": "benchmark",
		},
	)

	assert report["instruction_truncated"] is True
	assert report["instruction_chars"] == len(long_instruction)
	assert len(report["instruction_sha256"]) == 64
	assert len(report["instruction"]) < len(long_instruction)
	assert report["ltlf_formula_truncated"] is True
	assert report["ltlf_formula_chars"] == len(long_formula)
	assert len(report["ltlf_formula_sha256"]) == 64
	assert len(report["ltlf_formula"]) < len(long_formula)
	assert report["failure_signature"]["ltlf_formula_truncated"] is True


def test_failure_signature_preserves_counts_when_formula_is_truncated() -> None:
	signature = _extract_failure_signature(
		{
			"ltlf_formula": "do_move(b1,b2) & X(do_move(b2,b3))... [truncated]",
			"ltlf_atom_count": 500,
			"ltlf_operator_counts": {"X": 499, "&": 499},
			"failure_signature": {
				"ltlf_formula_truncated": True,
				"ltlf_formula_chars": 12000,
				"ltlf_formula_sha256": "a" * 64,
				"ltlf_atom_count": 500,
				"ltlf_operator_counts": {"X": 499, "&": 499},
			},
		},
		{"success": True},
	)

	assert signature["ltlf_atom_count"] == 500
	assert signature["ltlf_operator_counts"] == {"X": 499, "&": 499}
	assert signature["ltlf_formula_truncated"] is True
	assert signature["ltlf_formula_chars"] == 12000
	assert signature["ltlf_formula_sha256"] == "a" * 64


def test_compact_failure_signature_truncates_failed_goal_lists() -> None:
	signature = _compact_failure_signature(
		{
			"failed_goals": [f"goal_{index}" for index in range(25)],
			"verifier_missing_goal_facts": [f"fact_{index}" for index in range(22)],
		},
	)

	assert signature["failed_goals"] == [f"goal_{index}" for index in range(20)]
	assert signature["failed_goals_truncated"] is True
	assert signature["failed_goals_count"] == 25
	assert signature["verifier_missing_goal_facts"] == [
		f"fact_{index}"
		for index in range(20)
	]
	assert signature["verifier_missing_goal_facts_truncated"] is True
	assert signature["verifier_missing_goal_facts_count"] == 22


def test_evaluation_case_uses_stored_temporal_specification(
	tmp_path: Path,
	monkeypatch,
) -> None:
	problem_file = tmp_path / "p1.hddl"
	problem_file.write_text("(define (problem p1))")
	temporal_specification = TemporalSpecificationRecord(
		instruction_id="query_1",
		source_text="stored instruction",
		ltlf_formula="F(do_move(b1,b2))",
		referenced_events=(),
		problem_file="p1.hddl",
	)
	captured: dict[str, object] = {}

	monkeypatch.setattr(
		benchmark_support,
		"load_domain_query_cases",
		lambda domain_key: {
			"query_1": {
				"instruction": "nl instruction should not be grounded",
				"problem_file": str(problem_file),
			},
		},
	)
	monkeypatch.setattr(
		benchmark_support,
		"load_domain_temporal_specifications",
		lambda domain_key, query_ids=None: {"query_1": temporal_specification},
	)
	monkeypatch.setattr(
		benchmark_support,
		"resolve_plan_library_input",
		lambda domain_key, *, library_source: "library-root",
	)
	monkeypatch.setattr(
		benchmark_support,
		"load_plan_library_artifact_bundle",
		lambda library_input: type("FakeBundle", (), {"method_library": object()})(),
	)

	def fake_temporal_specification_to_grounding_result(
		*,
		temporal_specification,
		method_library,
		problem,
		task_type_map,
	):
		captured["temporal_specification"] = temporal_specification
		return TemporalGroundingResult(
			query_text=temporal_specification.source_text,
			ltlf_formula=temporal_specification.ltlf_formula,
			subgoals=(object(),),
			typed_objects={},
			query_object_inventory=(),
			diagnostics=(),
		)

	monkeypatch.setattr(
		benchmark_support,
		"_temporal_specification_to_grounding_result",
		fake_temporal_specification_to_grounding_result,
	)

	class FakeOrchestrator:
		def __init__(
			self,
			*,
			domain_file,
			problem_file,
			evaluation_domain_source,
			runtime_backend="jason",
		):
			self.problem = object()
			self.task_type_map = {}
			self.logger = None

		def execute_query_with_library(self, *args, **kwargs):
			raise AssertionError("Benchmark evaluation must not call live NL grounding.")

		def execute_grounded_query_with_library(
			self,
			nl_query,
			*,
			library_artifact,
			grounding_result,
		):
			captured["nl_query"] = nl_query
			captured["grounding_result"] = grounding_result
			log_dir = Path(self.logger.logs_dir) / "query_1"
			log_dir.mkdir(parents=True)
			(log_dir / "execution.json").write_text(
				json.dumps(
					{
						"goal_grounding": {
							"metadata": {
								"evaluation_domain_source": "benchmark",
								"grounding_mode": "precomputed_temporal_specification",
							},
							"artifacts": {
								"ltlf_formula": grounding_result.ltlf_formula,
							},
						},
						"runtime_execution": {
							"metadata": {"verification_mode": "original_problem"},
						},
						"ltlf_formula": grounding_result.ltlf_formula,
					},
				),
			)
			return {
				"success": True,
				"step": "",
				"log_path": str(log_dir / "execution.txt"),
			}

	monkeypatch.setattr(
		benchmark_support,
		"PlanLibraryEvaluationOrchestrator",
		FakeOrchestrator,
	)

	report = run_plan_library_evaluation_case(
		"blocksworld",
		"query_1",
		library_source="official",
		logs_root=tmp_path / "logs",
	)

	assert captured["temporal_specification"] is temporal_specification
	assert captured["nl_query"] == "stored instruction"
	assert captured["grounding_result"].ltlf_formula == "F(do_move(b1,b2))"
	assert report["case"]["ltlf_formula"] == "F(do_move(b1,b2))"
	assert report["failure_signature"]["ltlf_formula"] == "F(do_move(b1,b2))"
	assert report["evaluation_domain_source"] == "benchmark"


def test_full_sweep_summary_snapshot_writes_incremental_run_state(tmp_path: Path) -> None:
	summary = _write_run_summary_snapshot(
		run_dir=tmp_path / "20260420_220000",
		run_id="20260420_220000",
		domain_summaries={
			"blocksworld": {
				"total_queries": 30,
				"verified_successes": 2,
				"goal_grounding_failures": 1,
				"agentspeak_rendering_failures": 0,
				"runtime_execution_failures": 4,
				"plan_verification_failures": 5,
				"hierarchical_rejection_failures": 0,
				"unknown_failures": 0,
				"complete": True,
				"completed_query_ids": ["query_1", "query_2"],
				"remaining_query_ids": [],
			},
		},
		internal_failures=["transport"],
		max_concurrent_domains=1,
	)

	assert summary["run_id"] == "20260420_220000"
	assert summary["completed_domains"] == ["blocksworld"]
	assert summary["partial_domains"] == []
	assert summary["completed_query_count"] == 2
	assert summary["bdi_runtime_successes"] == 2
	assert summary["hierarchical_compatibility_successes"] == 0
	assert summary["runtime_goal_verified_successes"] == 0
	assert "transport" in summary["internal_failures"]
	assert (tmp_path / "20260420_220000" / "summary.json").exists()
	assert (tmp_path / "20260420_220000" / "summary.txt").read_text().startswith(
		"run_id: 20260420_220000\n",
	)


def test_full_sweep_summary_snapshot_tracks_partial_domains(tmp_path: Path) -> None:
	summary = _write_run_summary_snapshot(
		run_dir=tmp_path / "20260420_230000",
		run_id="20260420_230000",
		domain_summaries={
			"transport": {
				"total_queries": 40,
				"verified_successes": 16,
				"goal_grounding_failures": 0,
				"agentspeak_rendering_failures": 0,
				"runtime_execution_failures": 8,
				"plan_verification_failures": 0,
				"hierarchical_rejection_failures": 0,
				"unknown_failures": 0,
				"complete": False,
				"completed_query_ids": [f"query_{index}" for index in range(1, 25)],
				"remaining_query_ids": [f"query_{index}" for index in range(25, 41)],
			},
		},
		internal_failures=["transport"],
		max_concurrent_domains=4,
	)

	assert summary["completed_domains"] == []
	assert summary["partial_domains"] == ["transport"]
	assert "transport" in summary["pending_domains"]
	assert summary["total_queries"] == 40
	assert summary["completed_query_count"] == 24
	assert summary["remaining_query_count"] == 16
	assert summary["complete"] is False


def test_full_sweep_resume_loads_partial_domain_summaries(tmp_path: Path) -> None:
	run_dir = tmp_path / "20260420_230000"
	run_dir.mkdir(parents=True)
	(run_dir / "transport.summary.json").write_text(
		json.dumps(
			{
				"total_queries": 40,
				"complete": False,
				"completed_query_ids": ["query_1"],
				"remaining_query_ids": ["query_2"],
			},
			indent=2,
		),
	)

	summaries = _load_existing_domain_summaries(run_dir)

	assert "transport" in summaries
	assert summaries["transport"]["complete"] is False


def test_collect_domain_run_result_keeps_partial_summary_on_child_failure(
	tmp_path: Path,
) -> None:
	class FakeProcess:
		def wait(self) -> int:
			return 9

	class FakeOutputHandle:
		def __init__(self) -> None:
			self.closed = False

		def close(self) -> None:
			self.closed = True

	output_handle = FakeOutputHandle()
	summary_path = tmp_path / "transport.summary.json"
	output_path = tmp_path / "transport.out"
	summary_path.write_text(
		json.dumps(
			{
				"domain_key": "transport",
				"total_queries": 40,
				"complete": False,
				"completed_query_ids": ["query_1"],
				"remaining_query_ids": ["query_2"],
				"verified_successes": 1,
				"goal_grounding_failures": 0,
				"agentspeak_rendering_failures": 0,
				"runtime_execution_failures": 0,
				"plan_verification_failures": 0,
				"hierarchical_rejection_failures": 0,
				"unknown_failures": 0,
			},
			indent=2,
		),
	)
	run = DomainRun(
		name="transport",
		command=["python", "runner.py"],
		output_path=output_path,
		summary_path=summary_path,
		output_handle=output_handle,  # type: ignore[arg-type]
		process=FakeProcess(),  # type: ignore[arg-type]
	)
	domain_summaries: dict[str, dict[str, object]] = {}
	internal_failures: list[str] = []

	_collect_domain_run_result(run, domain_summaries, internal_failures)

	assert output_handle.closed is True
	assert internal_failures == ["transport"]
	assert domain_summaries["transport"]["process_return_code"] == 9
	assert domain_summaries["transport"]["process_failed"] is True
	assert domain_summaries["transport"]["process_output_path"] == str(output_path)
	assert domain_summaries["transport"]["completed_query_ids"] == ["query_1"]


def test_load_failed_query_ids_for_run_collects_failed_queries(tmp_path: Path) -> None:
	run_root = tmp_path / "run-001"
	domain_root = run_root / "blocksworld"
	domain_root.mkdir(parents=True)
	(domain_root / "summary.json").write_text(
		json.dumps(
			{
				"query_results": [
					{"query_id": "q1", "success": False},
					{"query_id": "q2", "success": True},
					{"query_id": "q3", "success": False},
				],
			},
			indent=2,
		),
	)

	failed_query_ids = _load_failed_query_ids_for_run("run-001", runs_root=tmp_path)

	assert failed_query_ids == {"blocksworld": ["q1", "q3"]}
