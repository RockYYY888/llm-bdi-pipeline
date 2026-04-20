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
)
from tests.run_online_query_solution_benchmark import _write_latest_run_manifest


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
	(run_dir / "summary.txt").write_text("verified_successes: 3\n")
	summary = {
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
	assert latest["run_dir"] == str(run_dir)
	assert latest["summary_json"] == str(run_dir / "summary.json")
	assert latest["summary_txt"] == str(run_dir / "summary.txt")
	assert json.loads((tmp_path / "latest_summary.json").read_text())["verified_successes"] == 3
	assert (tmp_path / "latest_summary.txt").read_text() == "verified_successes: 3\n"
