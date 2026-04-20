"""
Stable result schemas for Hierarchical Task Network evaluation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

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
	plan_verification_summary = dict(plan_verification.get("summary") or {})
	plan_verification_artifacts = dict(plan_verification.get("artifacts") or {})
	outcome_bucket = str(report.get("outcome_bucket") or "unknown_failure")
	if outcome_bucket not in HTN_OUTCOME_BUCKETS:
		outcome_bucket = "unknown_failure"

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
		"plan_solve_status": plan_solve_summary.get("status"),
		"plan_verification_status": plan_verification_summary.get("status"),
		"selected_solver_id": plan_verification_artifacts.get("selected_solver_id"),
		"selected_backend_name": plan_verification_artifacts.get("selected_backend_name"),
		"selected_representation_id": plan_verification_artifacts.get(
			"selected_representation_id",
		),
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
	for row in problem_rows:
		bucket = str(row.get("outcome_bucket") or "unknown_failure")
		if bucket not in counts:
			bucket = "unknown_failure"
		counts[bucket] += 1

	summary = {
		"domain_key": domain_key,
		"evaluation_mode": mode,
		"requested_planner_id": normalized_planner_id,
		"track_id": track_id,
		"attempted_problem_count": len(problem_rows),
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
				"plan_solve_status": row.get("plan_solve_status"),
				"plan_verification_status": row.get("plan_verification_status"),
				"selected_solver_id": row.get("selected_solver_id"),
				"selected_backend_name": row.get("selected_backend_name"),
				"selected_representation_id": row.get("selected_representation_id"),
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
	for domain_summary in domain_summaries.values():
		total_queries += int(domain_summary.get("attempted_problem_count") or 0)
		bucket_counts = dict(domain_summary.get("bucket_counts") or {})
		for bucket in HTN_OUTCOME_BUCKETS:
			counts[bucket] += int(bucket_counts.get(bucket, 0))

	return {
		"run_dir": str(run_dir),
		"track_id": track_id,
		"evaluation_mode": mode,
		"requested_planner_id": normalized_planner_id,
		"attempted_problem_count": total_queries,
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

