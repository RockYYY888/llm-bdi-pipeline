from __future__ import annotations

import sys
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from tests.support.online_query_solution_benchmark_support import (
	ONLINE_BENCHMARK_DOMAIN_SOURCE,
	ONLINE_BENCHMARK_LIBRARY_SOURCE,
	_extract_reported_online_domain_source,
	apply_online_query_runtime_defaults,
	run_online_query_solution_benchmark_for_domain,
)
import tests.support.online_query_solution_benchmark_support as benchmark_support
from tests.run_online_query_solution_benchmark import (
	_load_failed_query_ids_for_run,
	_write_latest_run_manifest,
	_write_run_summary_snapshot,
)


def test_online_benchmark_runtime_defaults_pin_benchmark_domain_source() -> None:
	env = apply_online_query_runtime_defaults({})

	assert env["ONLINE_DOMAIN_SOURCE"] == ONLINE_BENCHMARK_DOMAIN_SOURCE
	assert env["ONLINE_DOMAIN_SOURCE"] == "benchmark"


def test_online_benchmark_execution_source_defaults_to_benchmark_when_metadata_missing() -> None:
	assert _extract_reported_online_domain_source({}) == "benchmark"


def test_online_benchmark_standard_library_source_is_benchmark() -> None:
	assert ONLINE_BENCHMARK_LIBRARY_SOURCE == "benchmark"


def test_full_sweep_writes_latest_run_manifest_files(tmp_path: Path) -> None:
	run_dir = tmp_path / "20260420_220000"
	run_dir.mkdir(parents=True)
	(run_dir / "summary.txt").write_text("run_id: 20260420_220000\nverified_successes: 3\n")
	summary = {
		"run_id": "20260420_220000",
		"run_dir": str(run_dir),
		"online_domain_source": "benchmark",
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

	def fake_run_online_query_case(
		domain_key: str,
		query_id: str,
		*,
		library_source: str,
		logs_root: str | Path | None = None,
	):
		calls.append((query_id, str(Path(logs_root or "").resolve())))
		return {
			"query_id": query_id,
			"instruction": str(query_cases[query_id]["instruction"]),
			"problem_file": str(query_cases[query_id]["problem_file"]),
			"library_source": library_source,
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
				"mona_failure_signature": None,
				"jason_failure_class": None,
				"failed_goals": [],
				"verifier_missing_goal_facts": [],
			},
			"online_domain_source": "benchmark",
		}

	monkeypatch.setattr(
		benchmark_support,
		"run_online_query_case",
		fake_run_online_query_case,
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
					"mona_failure_signature": None,
					"jason_failure_class": None,
					"failed_goals": [],
					"verifier_missing_goal_facts": [],
				},
				"online_domain_source": "benchmark",
			},
			indent=2,
		),
	)

	summary = run_online_query_solution_benchmark_for_domain(
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
		"success": True,
		"result": {"success": True, "step": ""},
		"outcome_bucket": "hierarchical_plan_verified",
		"log_dir": str((domain_root / "logs" / "q2").resolve()),
		"execution": {
			"runtime_execution": {
				"metadata": {"verification_mode": "original_problem"},
			},
		},
		"failure_signature": {
			"ltlf_formula": None,
			"ltlf_atom_count": 0,
			"ltlf_operator_counts": {},
			"mona_failure_signature": None,
			"jason_failure_class": None,
			"failed_goals": [],
			"verifier_missing_goal_facts": [],
		},
		"ltlf_formula": None,
		"ltlf_atom_count": 0,
		"ltlf_operator_counts": {},
		"mona_failure_signature": None,
		"jason_failure_class": None,
		"failed_goals": [],
		"verifier_missing_goal_facts": [],
		"online_domain_source": "benchmark",
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

	def fake_run_online_query_case(
		domain_key: str,
		query_id: str,
		*,
		library_source: str,
		logs_root: str | Path | None = None,
	):
		return {
			"query_id": query_id,
			"instruction": str(query_cases[query_id]["instruction"]),
			"problem_file": str(query_cases[query_id]["problem_file"]),
			"library_source": library_source,
			"success": False,
			"result": {"success": False, "step": "runtime_execution"},
			"outcome_bucket": "runtime_execution_failed",
			"log_dir": str((Path(logs_root or tmp_path) / query_id).resolve()),
			"execution": {"jason_failure_class": "runtime_failure_marker"},
			"failure_signature": {
				"ltlf_formula": None,
				"ltlf_atom_count": 0,
				"ltlf_operator_counts": {},
				"mona_failure_signature": None,
				"jason_failure_class": "runtime_failure_marker",
				"failed_goals": [],
				"verifier_missing_goal_facts": [],
			},
			"online_domain_source": "benchmark",
		}

	monkeypatch.setattr(
		benchmark_support,
		"run_online_query_case",
		fake_run_online_query_case,
	)
	summary = run_online_query_solution_benchmark_for_domain(
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


def test_full_sweep_summary_snapshot_writes_incremental_run_state(tmp_path: Path) -> None:
	summary = _write_run_summary_snapshot(
		run_dir=tmp_path / "20260420_220000",
		run_id="20260420_220000",
		domain_summaries={
			"blocksworld": {
				"total_queries": 30,
				"verified_successes": 2,
				"goal_grounding_failures": 1,
				"temporal_compilation_failures": 3,
				"agentspeak_rendering_failures": 0,
				"runtime_execution_failures": 4,
				"plan_verification_failures": 5,
				"hierarchical_rejection_failures": 0,
				"unknown_failures": 0,
			},
		},
		internal_failures=["transport"],
		max_concurrent_domains=1,
	)

	assert summary["run_id"] == "20260420_220000"
	assert summary["completed_domains"] == ["blocksworld"]
	assert "transport" in summary["internal_failures"]
	assert (tmp_path / "20260420_220000" / "summary.json").exists()
	assert (tmp_path / "20260420_220000" / "summary.txt").read_text().startswith(
		"run_id: 20260420_220000\n",
	)


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
