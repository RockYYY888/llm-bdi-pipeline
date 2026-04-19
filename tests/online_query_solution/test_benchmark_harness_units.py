from __future__ import annotations

import sys
from pathlib import Path


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


def test_online_benchmark_runtime_defaults_pin_benchmark_domain_source() -> None:
	env = apply_online_query_runtime_defaults({})

	assert env["ONLINE_DOMAIN_SOURCE"] == ONLINE_BENCHMARK_DOMAIN_SOURCE
	assert env["ONLINE_DOMAIN_SOURCE"] == "benchmark"


def test_online_benchmark_execution_source_defaults_to_benchmark_when_metadata_missing() -> None:
	assert _extract_reported_online_domain_source({}) == "benchmark"


def test_online_benchmark_standard_library_source_is_benchmark() -> None:
	assert ONLINE_BENCHMARK_LIBRARY_SOURCE == "benchmark"
