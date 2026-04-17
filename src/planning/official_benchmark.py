"""
Official benchmark profile for the ground-truth planning baseline.
"""

from __future__ import annotations

from typing import Tuple


OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS = 1800.0
OFFICIAL_BENCHMARK_MEMORY_LIMIT_MIB = 8192
OFFICIAL_BENCHMARK_CPU_COUNT = 1

OFFICIAL_PANDA_PI_SOLVER_IDS: Tuple[str, ...] = (
	"progression_rc2_ff",
	"progression_rc2_add",
	"progression_rc2_lmc",
	"progression_suboptimal",
	"sat",
	"bdd",
	"translation_fd",
)

OFFICIAL_PANDADEALER_SOLVER_ID = "pandadealer_agile_lama"
OFFICIAL_LIFTED_LINEAR_SOLVER_ID = "lifted_linear_config_2"
OFFICIAL_LIFTED_LINEAR_INNER_SOLVER_ID = "sat"

OFFICIAL_PANDADEALER_ENGINE_ARGS: Tuple[str, ...] = (
	"--heuristic=lama(lazy=false;ha=false;lm=lmc;useLMOrd=false;h=add;search=gbfs)",
)

OFFICIAL_TRANSLATION_FAST_DOWNWARD_CONFIGURATION = "lazy-cea()"

OFFICIAL_BACKEND_SELECTION_RULE = "parallel_backend_any_success"
