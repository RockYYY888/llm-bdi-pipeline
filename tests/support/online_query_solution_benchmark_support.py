from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
	sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from pipeline.execution_logger import ExecutionLogger
from online_query_solution.pipeline import OnlineQuerySolutionPipeline
from utils.benchmark_query_dataset import load_problem_query_cases

from tests.support.offline_generation_support import (
	DOMAIN_FILES,
	GENERATED_LOGS_DIR,
	GENERATED_MASKED_DOMAIN_BUILDS_DIR,
	apply_generated_runtime_defaults,
	build_official_method_library,
	query_id_sort_key,
	run_generated_domain_build,
)


ONLINE_BENCHMARK_RESULTS_DIR = PROJECT_ROOT / "tests" / "generated" / "online_query_solution"
ONLINE_BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ONLINE_BENCHMARK_DOMAIN_SOURCE = "benchmark"
ONLINE_BENCHMARK_LIBRARY_SOURCE = "benchmark"


def apply_online_query_runtime_defaults(
	env: Optional[MutableMapping[str, str]] = None,
) -> MutableMapping[str, str]:
	target_env = apply_generated_runtime_defaults(env)
	target_env["ONLINE_DOMAIN_SOURCE"] = ONLINE_BENCHMARK_DOMAIN_SOURCE
	return target_env


def _extract_reported_online_domain_source(execution: Dict[str, Any]) -> str:
	candidate_paths = (
		("goal_grounding", "metadata", "online_domain_source"),
		("temporal_compilation", "metadata", "online_domain_source"),
		("agentspeak_rendering", "metadata", "online_domain_source"),
		("runtime_execution", "metadata", "online_domain_source"),
		("plan_verification", "metadata", "online_domain_source"),
	)
	for step_name, metadata_key, source_key in candidate_paths:
		step_payload = execution.get(step_name)
		if not isinstance(step_payload, dict):
			continue
		metadata = step_payload.get(metadata_key)
		if not isinstance(metadata, dict):
			continue
		source = str(metadata.get(source_key) or "").strip().lower()
		if source:
			return source
	return ONLINE_BENCHMARK_DOMAIN_SOURCE


def _classify_online_query_failure(result: Dict[str, Any], execution: Dict[str, Any]) -> str:
	if bool(result.get("success")):
		return "hierarchical_plan_verified"

	step = str(result.get("step") or "").strip()
	if step == "goal_grounding":
		return "goal_grounding_failed"
	if step == "temporal_compilation":
		return "temporal_compilation_failed"
	if step == "agentspeak_rendering":
		return "agentspeak_rendering_failed"
	if step == "runtime_execution":
		return "runtime_execution_failed"
	if step == "plan_verification":
		verification_status = str(
			((result.get("plan_verification") or {}).get("summary") or {}).get("status") or "",
		).strip()
		if verification_status == "failed":
			return "hierarchical_rejection_failed"
		return "plan_verification_failed"

	verification_payload = dict(execution.get("plan_verification") or {})
	if verification_payload.get("status") == "failed":
		return "hierarchical_rejection_failed"
	return "unknown_failure"


def ensure_online_domain_library_artifact(domain_key: str) -> Path:
	artifact_root = GENERATED_MASKED_DOMAIN_BUILDS_DIR / domain_key
	method_library_path = artifact_root / "method_library.json"
	if method_library_path.exists():
		return artifact_root
	build_report = run_generated_domain_build(domain_key)
	if not build_report.get("success"):
		raise RuntimeError(f"Offline domain build failed for {domain_key}: {build_report}")
	return Path(build_report["artifact_root"]).resolve()


def resolve_online_method_library_input(
	domain_key: str,
	*,
	library_source: str,
) -> str | Any:
	source = str(library_source or ONLINE_BENCHMARK_LIBRARY_SOURCE).strip().lower()
	if source in {"benchmark", "official"}:
		return build_official_method_library(DOMAIN_FILES[domain_key])
	if source == "generated":
		return str(ensure_online_domain_library_artifact(domain_key))
	raise ValueError(f"Unsupported online benchmark library source '{library_source}'.")


def load_domain_query_cases(domain_key: str) -> Dict[str, Dict[str, Any]]:
	return load_problem_query_cases(
		PROJECT_ROOT / "src" / "domains" / domain_key / "problems",
		limit=10**9,
	)


def run_online_query_case(
	domain_key: str,
	query_id: str,
	*,
	library_source: str = ONLINE_BENCHMARK_LIBRARY_SOURCE,
) -> Dict[str, Any]:
	apply_online_query_runtime_defaults()
	query_cases = load_domain_query_cases(domain_key)
	case = query_cases[query_id]
	domain_file = DOMAIN_FILES[domain_key]
	method_library_input = resolve_online_method_library_input(
		domain_key,
		library_source=library_source,
	)

	pipeline = OnlineQuerySolutionPipeline(
		domain_file=domain_file,
		problem_file=str(case["problem_file"]),
		online_domain_source=ONLINE_BENCHMARK_DOMAIN_SOURCE,
	)
	pipeline.pipeline.logger = ExecutionLogger(logs_dir=str(GENERATED_LOGS_DIR), run_origin="tests")
	result = pipeline.execute_query_with_library(
		case["instruction"],
		library_artifact=method_library_input,
	)
	log_path = result.get("log_path")
	log_dir = Path(str(log_path)).resolve().parent if log_path else None
	execution = (
		json.loads((log_dir / "execution.json").read_text())
		if log_dir is not None and (log_dir / "execution.json").exists()
		else {}
	)
	reported_online_domain_source = _extract_reported_online_domain_source(execution)
	if reported_online_domain_source != ONLINE_BENCHMARK_DOMAIN_SOURCE:
		raise AssertionError(
			"Online benchmark sweep must run against the benchmark domain source. "
			f"Observed: {reported_online_domain_source}",
		)
	outcome_bucket = _classify_online_query_failure(result, execution)

	return {
		"query_id": query_id,
		"case": case,
		"library_source": library_source,
		"success": bool(result.get("success")),
		"result": result,
		"outcome_bucket": outcome_bucket,
		"log_dir": log_dir,
		"execution": execution,
		"online_domain_source": reported_online_domain_source,
	}


def run_online_query_solution_benchmark_for_domain(
	domain_key: str,
	*,
	query_ids: Optional[Sequence[str]] = None,
	library_source: str = ONLINE_BENCHMARK_LIBRARY_SOURCE,
) -> Dict[str, Any]:
	normalized_library_source = str(library_source or ONLINE_BENCHMARK_LIBRARY_SOURCE).strip().lower()
	if normalized_library_source == "generated":
		library_artifact_ref: str | None = str(ensure_online_domain_library_artifact(domain_key))
	else:
		library_artifact_ref = None
	query_cases = load_domain_query_cases(domain_key)
	selected_query_ids = (
		tuple(sorted(query_ids, key=query_id_sort_key))
		if query_ids
		else tuple(sorted(query_cases, key=query_id_sort_key))
	)
	missing_query_ids = [query_id for query_id in selected_query_ids if query_id not in query_cases]
	if missing_query_ids:
		raise KeyError(
			f"Unknown query ids for domain '{domain_key}': {', '.join(missing_query_ids)}",
		)

	query_reports = [
		run_online_query_case(
			domain_key,
			query_id,
			library_source=normalized_library_source,
		)
		for query_id in selected_query_ids
	]
	counts = {
		"hierarchical_plan_verified": 0,
		"goal_grounding_failed": 0,
		"temporal_compilation_failed": 0,
		"agentspeak_rendering_failed": 0,
		"runtime_execution_failed": 0,
		"plan_verification_failed": 0,
		"hierarchical_rejection_failed": 0,
		"unknown_failure": 0,
	}
	for report in query_reports:
		bucket = str(report.get("outcome_bucket") or "unknown_failure")
		counts[bucket] = counts.get(bucket, 0) + 1

	summary = {
		"domain_key": domain_key,
		"online_domain_source": ONLINE_BENCHMARK_DOMAIN_SOURCE,
		"library_source": normalized_library_source,
		"domain_file": DOMAIN_FILES[domain_key],
		"library_artifact_root": library_artifact_ref,
		"total_queries": len(query_reports),
		"selected_query_ids": list(selected_query_ids),
		"verified_successes": counts.get("hierarchical_plan_verified", 0),
		"goal_grounding_failures": counts.get("goal_grounding_failed", 0),
		"temporal_compilation_failures": counts.get("temporal_compilation_failed", 0),
		"agentspeak_rendering_failures": counts.get("agentspeak_rendering_failed", 0),
		"runtime_execution_failures": counts.get("runtime_execution_failed", 0),
		"plan_verification_failures": counts.get("plan_verification_failed", 0),
		"hierarchical_rejection_failures": counts.get("hierarchical_rejection_failed", 0),
		"unknown_failures": counts.get("unknown_failure", 0),
		"query_results": [
			{
				"query_id": report["query_id"],
				"problem_file": str(report["case"]["problem_file"]),
				"log_dir": str(report["log_dir"]),
				"success": bool(report["success"]),
				"outcome_bucket": report["outcome_bucket"],
				"step": str((report.get("result") or {}).get("step") or ""),
				"verification_mode": str(
					(((report.get("execution") or {}).get("runtime_execution") or {}).get("metadata") or {}).get(
						"verification_mode",
					)
					or "",
				),
				"online_domain_source": str(report.get("online_domain_source") or ""),
				"library_source": str(report.get("library_source") or ""),
			}
			for report in query_reports
		],
	}
	output_root = ONLINE_BENCHMARK_RESULTS_DIR / domain_key
	output_root.mkdir(parents=True, exist_ok=True)
	(output_root / "summary.json").write_text(json.dumps(summary, indent=2))
	return summary


__all__ = [
	"ONLINE_BENCHMARK_RESULTS_DIR",
	"ONLINE_BENCHMARK_DOMAIN_SOURCE",
	"ONLINE_BENCHMARK_LIBRARY_SOURCE",
	"apply_online_query_runtime_defaults",
	"ensure_online_domain_library_artifact",
	"load_domain_query_cases",
	"run_online_query_case",
	"run_online_query_solution_benchmark_for_domain",
]
