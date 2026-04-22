"""
Stable result schemas for Hierarchical Task Network evaluation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from planning.backends import default_official_backends


PLANNER_OR_RACE_MODE = "planner_or_race"
SINGLE_PLANNER_MODE = "single_planner"
HTN_EVALUATION_MODES: Tuple[str, ...] = (
	PLANNER_OR_RACE_MODE,
	SINGLE_PLANNER_MODE,
)
HTN_OUTCOME_BUCKETS: Tuple[str, ...] = (
	"hierarchical_plan_verified",
	"primitive_plan_valid_but_hierarchical_rejected",
	"primitive_plan_invalid",
	"no_plan_from_solver",
	"unknown_failure",
)
HTN_PLANNER_IDS: Tuple[str, ...] = tuple(
	backend.backend_name
	for backend in default_official_backends()
)
OR_AGGREGATION_MODE = "single_planner_success_union"
OR_FAILURE_BUCKET_PRIORITY: Mapping[str, int] = {
	"primitive_plan_valid_but_hierarchical_rejected": 0,
	"primitive_plan_invalid": 1,
	"no_plan_from_solver": 2,
	"unknown_failure": 3,
	"hierarchical_plan_verified": 4,
}


def validate_evaluation_mode(evaluation_mode: str) -> str:
	mode = str(evaluation_mode or "").strip()
	if mode not in HTN_EVALUATION_MODES:
		raise ValueError(
			f"Unsupported HTN evaluation mode '{mode or 'unknown'}'. "
			f"Expected one of: {', '.join(HTN_EVALUATION_MODES)}.",
		)
	return mode


def validate_planner_id(
	planner_id: Optional[str],
	*,
	evaluation_mode: str,
) -> Optional[str]:
	mode = validate_evaluation_mode(evaluation_mode)
	normalized = str(planner_id or "").strip() or None
	if mode == PLANNER_OR_RACE_MODE:
		if normalized is not None:
			raise ValueError(
				"planner_or_race does not accept a planner_id; it runs all planner paths.",
			)
		return None
	if normalized is None:
		raise ValueError("single_planner mode requires a planner_id.")
	if normalized not in HTN_PLANNER_IDS:
		raise ValueError(
			f"Unsupported HTN planner id '{normalized}'. "
			f"Expected one of: {', '.join(HTN_PLANNER_IDS)}.",
		)
	return normalized


def planner_track_id(
	*,
	evaluation_mode: str,
	planner_id: Optional[str],
) -> str:
	mode = validate_evaluation_mode(evaluation_mode)
	normalized_planner_id = validate_planner_id(
		planner_id,
		evaluation_mode=mode,
	)
	if mode == PLANNER_OR_RACE_MODE:
		return PLANNER_OR_RACE_MODE
	return str(normalized_planner_id)


def empty_bucket_counts() -> Dict[str, int]:
	return {bucket: 0 for bucket in HTN_OUTCOME_BUCKETS}


def _coerce_float(value: Any) -> Optional[float]:
	try:
		if value is None:
			return None
		return float(value)
	except (TypeError, ValueError):
		return None


def _coerce_bool(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "y"}
	return bool(value)


def _inner_track_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
	return dict(summary.get("track_summary") or summary)


def _index_single_planner_track_summaries(
	track_summaries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
	indexed: Dict[str, Mapping[str, Any]] = {}
	for fallback_track_id, summary in track_summaries.items():
		inner = _inner_track_summary(summary)
		if str(inner.get("evaluation_mode") or "") != SINGLE_PLANNER_MODE:
			continue
		planner_id = str(
			inner.get("requested_planner_id")
			or inner.get("track_id")
			or fallback_track_id,
		)
		if planner_id in HTN_PLANNER_IDS:
			indexed[planner_id] = inner
	return indexed


def _query_order_key(query_id: str) -> Tuple[str, int, str]:
	prefix, sep, suffix = query_id.rpartition("_")
	if sep and suffix.isdigit():
		return (prefix, int(suffix), query_id)
	return (query_id, 0, query_id)


def _ordered_query_ids(
	domain_summaries: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
	ordered: List[str] = []
	seen: set[str] = set()
	for domain_summary in domain_summaries:
		for query_result in list(domain_summary.get("query_results") or ()):
			query_id = str(query_result.get("query_id") or "")
			if not query_id or query_id in seen:
				continue
			seen.add(query_id)
			ordered.append(query_id)
	return tuple(sorted(ordered, key=_query_order_key))


def _query_results_by_id(
	domain_summary: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
	results: Dict[str, Mapping[str, Any]] = {}
	for query_result in list(domain_summary.get("query_results") or ()):
		query_id = str(query_result.get("query_id") or "")
		if query_id:
			results[query_id] = dict(query_result)
	return results


def _select_or_query_result(
	query_id: str,
	candidates: Sequence[Tuple[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
	successful = [
		(planner_id, query_result)
		for planner_id, query_result in candidates
		if _coerce_bool(query_result.get("success"))
	]
	attempted_planner_ids = [planner_id for planner_id, _query_result in candidates]
	successful_planner_ids = [planner_id for planner_id, _query_result in successful]
	if successful:
		selected_planner_id, selected_query_result = successful[0]
		selected = dict(selected_query_result)
		selected["success"] = True
		selected["ipc_verified_success"] = True
		selected["outcome_bucket"] = "hierarchical_plan_verified"
		selected["failure_reason"] = ""
	else:
		def failure_sort_key(candidate: Tuple[str, Mapping[str, Any]]) -> Tuple[int, int]:
			planner_id, query_result = candidate
			bucket = str(query_result.get("outcome_bucket") or "unknown_failure")
			planner_order = HTN_PLANNER_IDS.index(planner_id) if planner_id in HTN_PLANNER_IDS else 999
			return (int(OR_FAILURE_BUCKET_PRIORITY.get(bucket, 999)), planner_order)

		selected_planner_id, selected_query_result = sorted(candidates, key=failure_sort_key)[0]
		selected = dict(selected_query_result)
		selected["success"] = False
		selected["ipc_verified_success"] = False
		selected["outcome_bucket"] = str(selected.get("outcome_bucket") or "unknown_failure")
		if not str(selected.get("failure_reason") or ""):
			buckets = ",".join(
				f"{planner_id}:{query_result.get('outcome_bucket') or 'unknown_failure'}"
				for planner_id, query_result in candidates
			)
			selected["failure_reason"] = f"planner_or_race_all_planners_failed: {buckets}"
	selected["query_id"] = query_id
	selected["selected_planner_track_id"] = selected_planner_id
	selected["successful_planner_ids"] = tuple(successful_planner_ids)
	selected["attempted_planner_ids"] = tuple(attempted_planner_ids)
	selected["aggregation_mode"] = OR_AGGREGATION_MODE
	return selected


def build_planner_or_race_track_summary_from_single_planner_tracks(
	*,
	run_dir: Path,
	track_summaries: Mapping[str, Mapping[str, Any]],
	domain_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
	"""
	Build planner_or_race as a capability union over completed single-planner tracks.

	The aggregate is intentionally not a runtime solver invocation. It answers the
	evaluation question "does any official planner path produce an IPC-verified
	plan for this problem?" without imposing an artificial sequential shared
	timeout that can starve later planner paths.
	"""
	single_planner_summaries = _index_single_planner_track_summaries(track_summaries)
	missing_planner_ids = [
		planner_id
		for planner_id in HTN_PLANNER_IDS
		if planner_id not in single_planner_summaries
	]
	if missing_planner_ids:
		raise ValueError(
			"Cannot build planner_or_race aggregate without completed single-planner "
			f"tracks: {', '.join(missing_planner_ids)}.",
		)
	for planner_id, track_summary in single_planner_summaries.items():
		if not bool(track_summary.get("complete", True)):
			raise ValueError(
				f"Cannot build planner_or_race aggregate from incomplete track '{planner_id}'.",
			)

	if domain_keys is None:
		discovered_domain_keys: List[str] = []
		seen_domains: set[str] = set()
		for planner_id in HTN_PLANNER_IDS:
			for domain_key in dict(single_planner_summaries[planner_id].get("domains") or {}):
				if domain_key in seen_domains:
					continue
				seen_domains.add(str(domain_key))
				discovered_domain_keys.append(str(domain_key))
		domain_keys = tuple(discovered_domain_keys)

	aggregate_domain_summaries: Dict[str, Mapping[str, Any]] = {}
	for domain_key in domain_keys:
		planner_domain_summaries = [
			dict(single_planner_summaries[planner_id].get("domains") or {}).get(domain_key)
			for planner_id in HTN_PLANNER_IDS
		]
		if any(domain_summary is None for domain_summary in planner_domain_summaries):
			raise ValueError(
				f"Cannot build planner_or_race aggregate; domain '{domain_key}' is missing "
				"from at least one single-planner track.",
			)
		non_null_domain_summaries = [
			dict(domain_summary)
			for domain_summary in planner_domain_summaries
			if domain_summary is not None
		]
		query_maps = {
			planner_id: _query_results_by_id(domain_summary)
			for planner_id, domain_summary in zip(HTN_PLANNER_IDS, non_null_domain_summaries)
		}
		problem_rows: List[Dict[str, Any]] = []
		for query_id in _ordered_query_ids(non_null_domain_summaries):
			candidates = [
				(planner_id, query_maps[planner_id][query_id])
				for planner_id in HTN_PLANNER_IDS
				if query_id in query_maps[planner_id]
			]
			if len(candidates) != len(HTN_PLANNER_IDS):
				raise ValueError(
					f"Cannot build planner_or_race aggregate; query '{query_id}' in "
					f"domain '{domain_key}' is missing from at least one single-planner track.",
				)
			problem_rows.append(_select_or_query_result(query_id, candidates))
		aggregate_domain_summaries[str(domain_key)] = build_domain_summary(
			domain_key=str(domain_key),
			problem_rows=problem_rows,
			evaluation_mode=PLANNER_OR_RACE_MODE,
			planner_id=None,
		)

	track_summary = build_track_summary(
		run_dir=run_dir,
		domain_summaries=aggregate_domain_summaries,
		evaluation_mode=PLANNER_OR_RACE_MODE,
		planner_id=None,
	)
	track_summary["aggregation_mode"] = OR_AGGREGATION_MODE
	track_summary["aggregation_source_tracks"] = list(HTN_PLANNER_IDS)
	track_summary["complete"] = True
	track_summary["completed_domains"] = sorted(aggregate_domain_summaries)
	return track_summary


def build_problem_result_row(
	*,
	domain_key: str,
	query_id: str,
	case: Mapping[str, Any],
	report: Mapping[str, Any],
	evaluation_mode: str,
	planner_id: Optional[str],
) -> Dict[str, Any]:
	mode = validate_evaluation_mode(evaluation_mode)
	normalized_planner_id = validate_planner_id(planner_id, evaluation_mode=mode)
	track_id = planner_track_id(
		evaluation_mode=mode,
		planner_id=normalized_planner_id,
	)
	plan_solve = dict(report.get("plan_solve") or {})
	plan_verification = dict(report.get("plan_verification") or {})
	plan_solve_summary = dict(plan_solve.get("summary") or {})
	plan_solve_artifacts = dict(plan_solve.get("artifacts") or {})
	plan_verification_summary = dict(plan_verification.get("summary") or {})
	plan_verification_artifacts = dict(plan_verification.get("artifacts") or {})
	execution = dict(report.get("execution") or {})
	timings = dict(execution.get("timings") or {})
	plan_solve_timing = dict(timings.get("plan_solve") or {})
	plan_verification_timing = dict(timings.get("plan_verification") or {})
	plan_solve_timing_metadata = dict(plan_solve_timing.get("metadata") or {})
	plan_verification_timing_metadata = dict(
		plan_verification_timing.get("metadata") or {},
	)
	outcome_bucket = str(report.get("outcome_bucket") or "unknown_failure")
	if outcome_bucket not in HTN_OUTCOME_BUCKETS:
		outcome_bucket = "unknown_failure"
	failure_reason = (
		plan_verification_artifacts.get("failure_reason")
		or plan_verification_summary.get("failure_reason")
		or plan_solve_artifacts.get("failure_reason")
		or plan_solve_summary.get("failure_reason")
		or ""
	)

	return {
		"domain_key": domain_key,
		"query_id": query_id,
		"problem_file": str(case.get("problem_file") or ""),
		"instruction": str(case.get("instruction") or ""),
		"evaluation_mode": mode,
		"requested_planner_id": normalized_planner_id,
		"track_id": track_id,
		"ipc_verified_success": bool(report.get("success")),
		"outcome_bucket": outcome_bucket,
		"log_dir": str(report.get("log_dir") or ""),
		"execution_time_seconds": _coerce_float(execution.get("execution_time_seconds")),
		"plan_solve_time_seconds": _coerce_float(plan_solve_timing.get("total_seconds")),
		"plan_verification_time_seconds": _coerce_float(
			plan_verification_timing.get("total_seconds"),
		),
		"representation_build_seconds": _coerce_float(
			plan_solve_timing_metadata.get("representation_build_seconds"),
		),
		"solver_race_wallclock_seconds": _coerce_float(
			plan_solve_timing_metadata.get("race_wallclock_seconds")
			or plan_verification_timing_metadata.get("race_wallclock_seconds"),
		),
		"plan_solve_status": plan_solve_summary.get("status"),
		"plan_verification_status": plan_verification_summary.get("status"),
		"selected_solver_id": plan_verification_artifacts.get("selected_solver_id"),
		"selected_backend_name": plan_verification_artifacts.get("selected_backend_name"),
		"selected_representation_id": plan_verification_artifacts.get(
			"selected_representation_id",
		),
		"failure_reason": str(failure_reason),
	}


def build_domain_summary(
	*,
	domain_key: str,
	problem_rows: Sequence[Mapping[str, Any]],
	evaluation_mode: str,
	planner_id: Optional[str],
	domain_gate_preflight: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
	mode = validate_evaluation_mode(evaluation_mode)
	normalized_planner_id = validate_planner_id(planner_id, evaluation_mode=mode)
	track_id = planner_track_id(
		evaluation_mode=mode,
		planner_id=normalized_planner_id,
	)
	counts = empty_bucket_counts()
	execution_time_seconds_total = 0.0
	plan_solve_time_seconds_total = 0.0
	plan_verification_time_seconds_total = 0.0
	execution_time_case_count = 0
	plan_solve_time_case_count = 0
	plan_verification_time_case_count = 0
	for row in problem_rows:
		bucket = str(row.get("outcome_bucket") or "unknown_failure")
		if bucket not in counts:
			bucket = "unknown_failure"
		counts[bucket] += 1
		execution_time_seconds = _coerce_float(row.get("execution_time_seconds"))
		if execution_time_seconds is not None:
			execution_time_seconds_total += execution_time_seconds
			execution_time_case_count += 1
		plan_solve_time_seconds = _coerce_float(row.get("plan_solve_time_seconds"))
		if plan_solve_time_seconds is not None:
			plan_solve_time_seconds_total += plan_solve_time_seconds
			plan_solve_time_case_count += 1
		plan_verification_time_seconds = _coerce_float(
			row.get("plan_verification_time_seconds"),
		)
		if plan_verification_time_seconds is not None:
			plan_verification_time_seconds_total += plan_verification_time_seconds
			plan_verification_time_case_count += 1

	summary = {
		"domain_key": domain_key,
		"evaluation_mode": mode,
		"requested_planner_id": normalized_planner_id,
		"track_id": track_id,
		"attempted_problem_count": len(problem_rows),
		"execution_time_seconds_total": execution_time_seconds_total,
		"execution_time_seconds_average": (
			execution_time_seconds_total / execution_time_case_count
			if execution_time_case_count > 0
			else None
		),
		"plan_solve_time_seconds_total": plan_solve_time_seconds_total,
		"plan_solve_time_seconds_average": (
			plan_solve_time_seconds_total / plan_solve_time_case_count
			if plan_solve_time_case_count > 0
			else None
		),
		"plan_verification_time_seconds_total": plan_verification_time_seconds_total,
		"plan_verification_time_seconds_average": (
			plan_verification_time_seconds_total / plan_verification_time_case_count
			if plan_verification_time_case_count > 0
			else None
		),
		"selected_query_ids": [
			str(row.get("query_id") or "")
			for row in problem_rows
		],
		"bucket_counts": counts,
		"verified_success_count": counts["hierarchical_plan_verified"],
		"domain_gate_preflight": dict(domain_gate_preflight or {}),
		"query_results": [
			{
				"query_id": str(row.get("query_id") or ""),
				"problem_file": str(row.get("problem_file") or ""),
				"log_dir": str(row.get("log_dir") or ""),
				"success": bool(row.get("ipc_verified_success")),
				"outcome_bucket": str(row.get("outcome_bucket") or "unknown_failure"),
				"execution_time_seconds": row.get("execution_time_seconds"),
				"plan_solve_time_seconds": row.get("plan_solve_time_seconds"),
				"plan_verification_time_seconds": row.get("plan_verification_time_seconds"),
				"representation_build_seconds": row.get("representation_build_seconds"),
				"solver_race_wallclock_seconds": row.get("solver_race_wallclock_seconds"),
				"plan_solve_status": row.get("plan_solve_status"),
				"plan_verification_status": row.get("plan_verification_status"),
					"selected_solver_id": row.get("selected_solver_id"),
					"selected_backend_name": row.get("selected_backend_name"),
					"selected_representation_id": row.get("selected_representation_id"),
					"failure_reason": row.get("failure_reason"),
					"selected_planner_track_id": row.get("selected_planner_track_id"),
					"successful_planner_ids": list(row.get("successful_planner_ids") or ()),
					"attempted_planner_ids": list(row.get("attempted_planner_ids") or ()),
					"aggregation_mode": row.get("aggregation_mode"),
				}
				for row in problem_rows
			],
		"total_queries": len(problem_rows),
		"verified_successes": counts["hierarchical_plan_verified"],
		"hierarchical_rejection_failures": counts[
			"primitive_plan_valid_but_hierarchical_rejected"
		],
		"primitive_invalid_failures": counts["primitive_plan_invalid"],
		"solver_no_plan_failures": counts["no_plan_from_solver"],
		"unknown_failures": counts["unknown_failure"],
	}
	return summary


def write_domain_results(
	output_root: Path,
	*,
	problem_rows: Sequence[Mapping[str, Any]],
	domain_summary: Mapping[str, Any],
) -> Dict[str, str]:
	output_root.mkdir(parents=True, exist_ok=True)
	problem_results_path = output_root / "problem_results.json"
	domain_summary_path = output_root / "domain_summary.json"
	legacy_summary_path = output_root / "summary.json"
	problem_results_path.write_text(json.dumps(list(problem_rows), indent=2))
	domain_summary_path.write_text(json.dumps(dict(domain_summary), indent=2))
	legacy_summary_path.write_text(json.dumps(dict(domain_summary), indent=2))
	return {
		"problem_results": str(problem_results_path),
		"domain_summary": str(domain_summary_path),
		"legacy_summary": str(legacy_summary_path),
	}


def build_track_summary(
	*,
	run_dir: Path,
	domain_summaries: Mapping[str, Mapping[str, Any]],
	evaluation_mode: str,
	planner_id: Optional[str],
) -> Dict[str, Any]:
	mode = validate_evaluation_mode(evaluation_mode)
	normalized_planner_id = validate_planner_id(planner_id, evaluation_mode=mode)
	track_id = planner_track_id(
		evaluation_mode=mode,
		planner_id=normalized_planner_id,
	)
	counts = empty_bucket_counts()
	total_queries = 0
	execution_time_seconds_total = 0.0
	plan_solve_time_seconds_total = 0.0
	plan_verification_time_seconds_total = 0.0
	for domain_summary in domain_summaries.values():
		total_queries += int(domain_summary.get("attempted_problem_count") or 0)
		execution_time_seconds_total += float(
			domain_summary.get("execution_time_seconds_total") or 0.0,
		)
		plan_solve_time_seconds_total += float(
			domain_summary.get("plan_solve_time_seconds_total") or 0.0,
		)
		plan_verification_time_seconds_total += float(
			domain_summary.get("plan_verification_time_seconds_total") or 0.0,
		)
		bucket_counts = dict(domain_summary.get("bucket_counts") or {})
		for bucket in HTN_OUTCOME_BUCKETS:
			counts[bucket] += int(bucket_counts.get(bucket, 0))

	return {
		"run_dir": str(run_dir),
		"track_id": track_id,
		"evaluation_mode": mode,
		"requested_planner_id": normalized_planner_id,
		"attempted_problem_count": total_queries,
		"execution_time_seconds_total": execution_time_seconds_total,
		"plan_solve_time_seconds_total": plan_solve_time_seconds_total,
		"plan_verification_time_seconds_total": plan_verification_time_seconds_total,
		"bucket_counts": counts,
		"verified_success_count": counts["hierarchical_plan_verified"],
		"domains": dict(domain_summaries),
		"total_queries": total_queries,
		"verified_successes": counts["hierarchical_plan_verified"],
		"hierarchical_rejection_failures": counts[
			"primitive_plan_valid_but_hierarchical_rejected"
		],
		"primitive_invalid_failures": counts["primitive_plan_invalid"],
		"solver_no_plan_failures": counts["no_plan_from_solver"],
		"unknown_failures": counts["unknown_failure"],
	}


def build_planner_capability_rows(
	track_summaries: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
	rows = []
	for track_summary in track_summaries:
		for domain_key, domain_summary in dict(track_summary.get("domains") or {}).items():
			bucket_counts = dict(domain_summary.get("bucket_counts") or {})
			row = {
				"track_id": str(track_summary.get("track_id") or ""),
				"evaluation_mode": str(track_summary.get("evaluation_mode") or ""),
				"requested_planner_id": track_summary.get("requested_planner_id"),
				"domain_key": str(domain_key),
				"attempted_problem_count": int(
					domain_summary.get("attempted_problem_count") or 0,
				),
				"execution_time_seconds_total": float(
					domain_summary.get("execution_time_seconds_total") or 0.0,
				),
				"execution_time_seconds_average": _coerce_float(
					domain_summary.get("execution_time_seconds_average"),
				),
				"plan_solve_time_seconds_total": float(
					domain_summary.get("plan_solve_time_seconds_total") or 0.0,
				),
				"plan_solve_time_seconds_average": _coerce_float(
					domain_summary.get("plan_solve_time_seconds_average"),
				),
				"plan_verification_time_seconds_total": float(
					domain_summary.get("plan_verification_time_seconds_total") or 0.0,
				),
				"plan_verification_time_seconds_average": _coerce_float(
					domain_summary.get("plan_verification_time_seconds_average"),
				),
				"verified_success_count": int(
					domain_summary.get("verified_success_count")
					or domain_summary.get("verified_successes")
					or 0,
				),
			}
			for bucket in HTN_OUTCOME_BUCKETS:
				row[bucket] = int(bucket_counts.get(bucket, 0))
			rows.append(row)
	return tuple(rows)


def build_problem_capability_rows(
	track_summaries: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
	rows = []
	for track_summary in track_summaries:
		track_id = str(track_summary.get("track_id") or "")
		evaluation_mode = str(track_summary.get("evaluation_mode") or "")
		requested_planner_id = track_summary.get("requested_planner_id")
		for domain_key, domain_summary in dict(track_summary.get("domains") or {}).items():
			for query_result in list(domain_summary.get("query_results") or ()):
				row = {
					"track_id": track_id,
					"evaluation_mode": evaluation_mode,
					"requested_planner_id": requested_planner_id,
					"domain_key": str(domain_key),
					"query_id": str(query_result.get("query_id") or ""),
					"problem_file": str(query_result.get("problem_file") or ""),
					"log_dir": str(query_result.get("log_dir") or ""),
					"ipc_verified_success": bool(query_result.get("success")),
					"outcome_bucket": str(
						query_result.get("outcome_bucket") or "unknown_failure"
					),
					"execution_time_seconds": _coerce_float(
						query_result.get("execution_time_seconds"),
					),
					"plan_solve_time_seconds": _coerce_float(
						query_result.get("plan_solve_time_seconds"),
					),
					"plan_verification_time_seconds": _coerce_float(
						query_result.get("plan_verification_time_seconds"),
					),
					"representation_build_seconds": _coerce_float(
						query_result.get("representation_build_seconds"),
					),
					"solver_race_wallclock_seconds": _coerce_float(
						query_result.get("solver_race_wallclock_seconds"),
					),
					"plan_solve_status": query_result.get("plan_solve_status"),
					"plan_verification_status": query_result.get("plan_verification_status"),
					"selected_solver_id": query_result.get("selected_solver_id"),
					"selected_backend_name": query_result.get("selected_backend_name"),
					"selected_representation_id": query_result.get("selected_representation_id"),
					"failure_reason": str(query_result.get("failure_reason") or ""),
					"selected_planner_track_id": query_result.get("selected_planner_track_id"),
					"successful_planner_ids": ",".join(
						str(planner_id)
						for planner_id in list(query_result.get("successful_planner_ids") or ())
					),
					"attempted_planner_ids": ",".join(
						str(planner_id)
						for planner_id in list(query_result.get("attempted_planner_ids") or ())
					),
					"aggregation_mode": query_result.get("aggregation_mode"),
				}
				rows.append(row)
	return tuple(rows)


def write_planner_capability_matrix(
	output_root: Path,
	*,
	rows: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
	output_root.mkdir(parents=True, exist_ok=True)
	json_path = output_root / "planner_capability_matrix.json"
	csv_path = output_root / "planner_capability_matrix.csv"
	row_payload = [dict(row) for row in rows]
	json_path.write_text(json.dumps(row_payload, indent=2))
	fieldnames = [
		"track_id",
		"evaluation_mode",
		"requested_planner_id",
		"domain_key",
		"attempted_problem_count",
		"execution_time_seconds_total",
		"execution_time_seconds_average",
		"plan_solve_time_seconds_total",
		"plan_solve_time_seconds_average",
		"plan_verification_time_seconds_total",
		"plan_verification_time_seconds_average",
		"verified_success_count",
		*HTN_OUTCOME_BUCKETS,
	]
	with csv_path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in row_payload:
			writer.writerow({field: row.get(field) for field in fieldnames})
	return {
		"planner_capability_matrix_json": str(json_path),
		"planner_capability_matrix_csv": str(csv_path),
	}


def write_problem_capability_matrix(
	output_root: Path,
	*,
	rows: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
	output_root.mkdir(parents=True, exist_ok=True)
	json_path = output_root / "problem_capability_matrix.json"
	csv_path = output_root / "problem_capability_matrix.csv"
	row_payload = [dict(row) for row in rows]
	json_path.write_text(json.dumps(row_payload, indent=2))
	fieldnames = [
		"track_id",
		"evaluation_mode",
		"requested_planner_id",
		"domain_key",
		"query_id",
		"problem_file",
		"log_dir",
		"ipc_verified_success",
		"outcome_bucket",
		"execution_time_seconds",
		"plan_solve_time_seconds",
		"plan_verification_time_seconds",
		"representation_build_seconds",
		"solver_race_wallclock_seconds",
		"plan_solve_status",
		"plan_verification_status",
		"selected_solver_id",
		"selected_backend_name",
		"selected_representation_id",
			"failure_reason",
			"selected_planner_track_id",
			"successful_planner_ids",
			"attempted_planner_ids",
			"aggregation_mode",
		]
	with csv_path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in row_payload:
			writer.writerow({field: row.get(field) for field in fieldnames})
	return {
		"problem_capability_matrix_json": str(json_path),
		"problem_capability_matrix_csv": str(csv_path),
	}
