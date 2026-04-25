"""
Official benchmark profile for the ground-truth planning baseline.
"""

from __future__ import annotations

OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS = 1800.0
OFFICIAL_BENCHMARK_MEMORY_LIMIT_MIB = 8192
OFFICIAL_BENCHMARK_CPU_COUNT = 1

OFFICIAL_LIFTED_LINEAR_SOLVER_ID = "lifted_linear_config_2"
OFFICIAL_LIFTED_LINEAR_INNER_SOLVER_ID = "sat"
