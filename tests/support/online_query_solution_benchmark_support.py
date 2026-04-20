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
	logs_root: str | Path | None = None,
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
	resolved_logs_root = (
		Path(logs_root).resolve()
		if logs_root is not None
		else GENERATED_LOGS_DIR
	)
	pipeline.pipeline.logger = ExecutionLogger(logs_dir=str(resolved_logs_root), run_origin="tests")
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
		"problem_file": str(case["problem_file"]),
		"library_source": library_source,
		"success": bool(result.get("success")),
		"result": result,
		"outcome_bucket": outcome_bucket,
		"log_dir": log_dir,
		"execution": execution,
		"online_domain_source": reported_online_domain_source,
	}


def _domain_output_root(
	output_root: str | Path | None,
	domain_key: str,
) -> Path:
	if output_root is None:
		return (ONLINE_BENCHMARK_RESULTS_DIR / domain_key).resolve()
	return (Path(output_root).resolve() / domain_key).resolve()


def _query_results_root(domain_output_root: Path) -> Path:
	return domain_output_root / "query_results"


def _query_result_path(domain_output_root: Path, query_id: str) -> Path:
	return _query_results_root(domain_output_root) / f"{query_id}.json"


def _domain_state_path(domain_output_root: Path) -> Path:
	return domain_output_root / "state.json"


def _domain_summary_path(domain_output_root: Path) -> Path:
	return domain_output_root / "summary.json"


def _top_level_domain_summary_path(
	output_root: str | Path | None,
	domain_key: str,
) -> Path:
	if output_root is None:
		return _domain_summary_path(_domain_output_root(output_root, domain_key))
	return Path(output_root).resolve() / f"{domain_key}.summary.json"


def _serialize_query_report(report: Dict[str, Any]) -> Dict[str, Any]:
	problem_file = str(
		report.get("problem_file")
		or ((report.get("case") or {}).get("problem_file") or ""),
	)
	instruction = str(
		report.get("instruction")
		or ((report.get("case") or {}).get("instruction") or ""),
	).strip()
	log_dir = report.get("log_dir")
	return {
		"run_id": report.get("run_id"),
		"domain_key": str(report.get("domain_key") or ""),
		"query_id": str(report.get("query_id") or ""),
		"instruction": instruction,
		"problem_file": problem_file,
		"library_source": str(report.get("library_source") or ""),
		"success": bool(report.get("success")),
		"result": dict(report.get("result") or {}),
		"outcome_bucket": str(report.get("outcome_bucket") or "unknown_failure"),
		"log_dir": str(log_dir) if log_dir else None,
		"execution": dict(report.get("execution") or {}),
		"online_domain_source": str(report.get("online_domain_source") or ""),
	}


def _load_query_report_checkpoint(
	query_result_path: Path,
	*,
	domain_key: str,
	query_id: str,
	library_source: str,
) -> Optional[Dict[str, Any]]:
	if not query_result_path.exists():
		return None
	try:
		payload = json.loads(query_result_path.read_text())
	except json.JSONDecodeError:
		return None
	if str(payload.get("domain_key") or "") != domain_key:
		return None
	if str(payload.get("query_id") or "") != query_id:
		return None
	if str(payload.get("library_source") or "") != library_source:
		return None
	if (
		str(payload.get("online_domain_source") or "").strip().lower()
		!= ONLINE_BENCHMARK_DOMAIN_SOURCE
	):
		return None
	return _serialize_query_report(payload)


def _count_query_outcomes(query_reports: Sequence[Dict[str, Any]]) -> Dict[str, int]:
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
	return counts


def _build_domain_summary(
	*,
	domain_key: str,
	run_id: str | None,
	output_root: str | Path | None,
	domain_output_root: Path,
	selected_query_ids: Sequence[str],
	query_reports: Sequence[Dict[str, Any]],
	library_source: str,
	library_artifact_ref: str | None,
	resume: bool,
	resumed_query_ids: Sequence[str],
) -> Dict[str, Any]:
	counts = _count_query_outcomes(query_reports)
	completed_query_ids = [str(report.get("query_id") or "") for report in query_reports]
	completed_query_id_set = set(completed_query_ids)
	remaining_query_ids = [
		query_id
		for query_id in selected_query_ids
		if query_id not in completed_query_id_set
	]
	summary = {
		"run_id": run_id,
		"domain_key": domain_key,
		"online_domain_source": ONLINE_BENCHMARK_DOMAIN_SOURCE,
		"library_source": library_source,
		"domain_file": DOMAIN_FILES[domain_key],
		"library_artifact_root": library_artifact_ref,
		"output_root": str(Path(output_root).resolve()) if output_root is not None else None,
		"domain_output_root": str(domain_output_root),
		"query_results_root": str(_query_results_root(domain_output_root)),
		"logs_root": str((domain_output_root / "logs").resolve()),
		"total_queries": len(selected_query_ids),
		"selected_query_ids": list(selected_query_ids),
		"completed_query_ids": completed_query_ids,
		"remaining_query_ids": remaining_query_ids,
		"resumed_query_ids": list(resumed_query_ids),
		"resume_enabled": bool(resume),
		"complete": not remaining_query_ids,
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
				"query_id": str(report.get("query_id") or ""),
				"run_id": report.get("run_id"),
				"problem_file": str(report.get("problem_file") or ""),
				"log_dir": str(report.get("log_dir") or ""),
				"query_result_path": str(
					_query_result_path(
						domain_output_root,
						str(report.get("query_id") or ""),
					).resolve(),
				),
				"success": bool(report.get("success")),
				"outcome_bucket": str(report.get("outcome_bucket") or ""),
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
	return summary


def _write_domain_summary_artifacts(
	*,
	output_root: str | Path | None,
	domain_key: str,
	domain_output_root: Path,
	summary: Dict[str, Any],
) -> None:
	domain_output_root.mkdir(parents=True, exist_ok=True)
	_query_results_root(domain_output_root).mkdir(parents=True, exist_ok=True)
	(domain_output_root / "logs").mkdir(parents=True, exist_ok=True)
	_domain_state_path(domain_output_root).write_text(json.dumps(summary, indent=2))
	_domain_summary_path(domain_output_root).write_text(json.dumps(summary, indent=2))
	top_level_summary_path = _top_level_domain_summary_path(output_root, domain_key)
	if top_level_summary_path != _domain_summary_path(domain_output_root):
		top_level_summary_path.write_text(json.dumps(summary, indent=2))


def run_online_query_solution_benchmark_for_domain(
	domain_key: str,
	*,
	query_ids: Optional[Sequence[str]] = None,
	library_source: str = ONLINE_BENCHMARK_LIBRARY_SOURCE,
	output_root: str | Path | None = None,
	run_id: str | None = None,
	resume: bool = False,
) -> Dict[str, Any]:
	normalized_library_source = str(library_source or ONLINE_BENCHMARK_LIBRARY_SOURCE).strip().lower()
	if normalized_library_source == "generated":
		library_artifact_ref: str | None = str(ensure_online_domain_library_artifact(domain_key))
	else:
		library_artifact_ref = None
	domain_output_root = _domain_output_root(output_root, domain_key)
	domain_output_root.mkdir(parents=True, exist_ok=True)
	_query_results_root(domain_output_root).mkdir(parents=True, exist_ok=True)
	(domain_output_root / "logs").mkdir(parents=True, exist_ok=True)
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
	query_reports_by_id: Dict[str, Dict[str, Any]] = {}
	resumed_query_ids: list[str] = []
	if resume:
		for query_id in selected_query_ids:
			cached_report = _load_query_report_checkpoint(
				_query_result_path(domain_output_root, query_id),
				domain_key=domain_key,
				query_id=query_id,
				library_source=normalized_library_source,
			)
			if cached_report is None:
				continue
			query_reports_by_id[query_id] = cached_report
			resumed_query_ids.append(query_id)

	for query_id in selected_query_ids:
		if query_id in query_reports_by_id:
			continue
		query_report = _serialize_query_report(
			{
				**run_online_query_case(
					domain_key,
					query_id,
					library_source=normalized_library_source,
					logs_root=domain_output_root / "logs",
				),
				"run_id": run_id,
				"domain_key": domain_key,
			},
		)
		query_reports_by_id[query_id] = query_report
		_query_result_path(domain_output_root, query_id).write_text(
			json.dumps(query_report, indent=2),
		)
		partial_reports = [
			query_reports_by_id[current_query_id]
			for current_query_id in selected_query_ids
			if current_query_id in query_reports_by_id
		]
		partial_summary = _build_domain_summary(
			domain_key=domain_key,
			run_id=run_id,
			output_root=output_root,
			domain_output_root=domain_output_root,
			selected_query_ids=selected_query_ids,
			query_reports=partial_reports,
			library_source=normalized_library_source,
			library_artifact_ref=library_artifact_ref,
			resume=resume,
			resumed_query_ids=resumed_query_ids,
		)
		_write_domain_summary_artifacts(
			output_root=output_root,
			domain_key=domain_key,
			domain_output_root=domain_output_root,
			summary=partial_summary,
		)

	query_reports = [
		query_reports_by_id[query_id]
		for query_id in selected_query_ids
		if query_id in query_reports_by_id
	]
	summary = _build_domain_summary(
		domain_key=domain_key,
		run_id=run_id,
		output_root=output_root,
		domain_output_root=domain_output_root,
		selected_query_ids=selected_query_ids,
		query_reports=query_reports,
		library_source=normalized_library_source,
		library_artifact_ref=library_artifact_ref,
		resume=resume,
		resumed_query_ids=resumed_query_ids,
	)
	_write_domain_summary_artifacts(
		output_root=output_root,
		domain_key=domain_key,
		domain_output_root=domain_output_root,
		summary=summary,
	)
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
