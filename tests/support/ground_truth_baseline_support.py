from __future__ import annotations

"""
Compatibility re-export for legacy test imports.

Method-library generation helpers live in tests.support.plan_library_generation_support.
Hierarchical Task Network evaluation helpers live in tests.support.htn_evaluation_support.
"""

from tests.support.htn_evaluation_support import (
	load_domain_query_cases,
	query_id_sort_key,
	run_domain_problem_root_case,
	run_generated_problem_root_baseline_for_domain,
	run_generated_problem_root_case,
	run_official_problem_root_baseline_for_domain,
)
from tests.support.method_library_validation_support import run_official_domain_gate_preflight
from tests.support.plan_library_generation_support import (
	DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS,
	DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS,
	DOMAIN_FILES,
	GENERATED_BASELINE_DIR,
	GENERATED_DOMAIN_BUILDS_DIR,
	GENERATED_LOGS_DIR,
	GENERATED_MASKED_BASELINE_DIR,
	GENERATED_MASKED_DOMAIN_BUILDS_DIR,
	apply_generated_runtime_defaults,
	build_method_library_from_domain_file,
	build_official_method_library,
	run_generated_domain_build,
)

__all__ = [
	"DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS",
	"DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS",
	"DOMAIN_FILES",
	"GENERATED_BASELINE_DIR",
	"GENERATED_DOMAIN_BUILDS_DIR",
	"GENERATED_LOGS_DIR",
	"GENERATED_MASKED_BASELINE_DIR",
	"GENERATED_MASKED_DOMAIN_BUILDS_DIR",
	"apply_generated_runtime_defaults",
	"build_method_library_from_domain_file",
	"build_official_method_library",
	"load_domain_query_cases",
	"query_id_sort_key",
	"run_domain_problem_root_case",
	"run_generated_domain_build",
	"run_generated_problem_root_baseline_for_domain",
	"run_generated_problem_root_case",
	"run_official_domain_gate_preflight",
	"run_official_problem_root_baseline_for_domain",
]
