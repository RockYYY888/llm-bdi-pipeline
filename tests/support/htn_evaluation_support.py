from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
	sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from pipeline.execution_logger import ExecutionLogger
from htn_evaluation.pipeline import HTNEvaluationPipeline

from tests.support.offline_generation_support import (
	DOMAIN_FILES,
	GENERATED_BASELINE_DIR,
	GENERATED_LOGS_DIR,
	GENERATED_MASKED_BASELINE_DIR,
	apply_generated_runtime_defaults,
	build_official_method_library,
	load_domain_query_cases,
	query_id_sort_key,
	run_generated_domain_build,
)
from tests.support.offline_domain_gate_support import run_official_domain_gate_preflight


def run_domain_problem_root_case(domain_key: str, query_id: str) -> Dict[str, Any]:
	query_cases = load_domain_query_cases(domain_key)
	case = query_cases[query_id]
	domain_file = DOMAIN_FILES[domain_key]
	pipeline = HTNEvaluationPipeline(
		domain_file=domain_file,
		problem_file=str(case["problem_file"]),
	)
	pipeline.logger = ExecutionLogger(logs_dir=str(GENERATED_LOGS_DIR), run_origin="tests")
	pipeline.logger.start_pipeline(
		case["instruction"],
		mode="official_problem_root_execution",
		domain_file=domain_file,
		problem_file=str(case["problem_file"]),
		domain_name=pipeline.domain.name,
		problem_name=pipeline.problem.name if pipeline.problem else None,
		output_dir=str(GENERATED_LOGS_DIR),
	)
	pipeline.output_dir = pipeline.logger.current_log_dir
	if pipeline.logger.current_record is not None and pipeline.output_dir is not None:
		pipeline.logger.current_record.output_dir = str(pipeline.output_dir)
		pipeline.logger._save_current_state()

	method_library = build_official_method_library(domain_file)
	race_result = pipeline._execute_official_problem_root_parallel_solver_race(method_library)
	plan_solve = dict(race_result.get("plan_solve") or {})
	plan_verification = dict(race_result.get("plan_verification") or {})
	success = (
		(plan_solve.get("summary") or {}).get("status") == "success"
		and (plan_verification.get("summary") or {}).get("status") == "success"
	)
	log_path = pipeline.logger.end_pipeline(success=success)
	log_dir = Path(log_path).parent
	execution = json.loads((log_dir / "execution.json").read_text())

	verifier_artifacts = dict(plan_verification.get("artifacts") or {})
	outcome_bucket = (
		verifier_artifacts.get("selected_bucket")
		or (plan_verification.get("summary") or {}).get("failure_bucket")
		or (plan_solve.get("summary") or {}).get("failure_bucket")
		or ("hierarchical_plan_verified" if success else "unknown_failure")
	)
	return {
		"query_id": query_id,
		"case": case,
		"success": success,
		"outcome_bucket": outcome_bucket,
		"log_dir": log_dir,
		"execution": execution,
		"plan_solve": plan_solve,
		"plan_verification": plan_verification,
	}


def run_official_problem_root_baseline_for_domain(
	domain_key: str,
	*,
	query_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
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
		run_domain_problem_root_case(domain_key, query_id)
		for query_id in selected_query_ids
	]
	counts = {
		"hierarchical_plan_verified": 0,
		"primitive_plan_valid_but_hierarchical_rejected": 0,
		"primitive_plan_invalid": 0,
		"no_plan_from_solver": 0,
		"unknown_failure": 0,
	}
	for report in query_reports:
		bucket = str(report.get("outcome_bucket") or "unknown_failure")
		counts[bucket] = counts.get(bucket, 0) + 1

	domain_gate_report = run_official_domain_gate_preflight(domain_key)
	summary = {
		"domain_key": domain_key,
		"domain_gate_preflight": {
			"success": bool(domain_gate_report.get("success")),
			"log_dir": str(domain_gate_report.get("log_dir")),
			"artifact_root": str(domain_gate_report.get("artifact_root")),
			"validated_task_count": (
				(domain_gate_report.get("domain_gate") or {}).get("validated_task_count")
			),
		},
		"total_queries": len(query_reports),
		"selected_query_ids": list(selected_query_ids),
		"verified_successes": counts.get("hierarchical_plan_verified", 0),
		"hierarchical_rejection_failures": counts.get(
			"primitive_plan_valid_but_hierarchical_rejected",
			0,
		),
		"primitive_invalid_failures": counts.get("primitive_plan_invalid", 0),
		"solver_no_plan_failures": counts.get("no_plan_from_solver", 0),
		"unknown_failures": counts.get("unknown_failure", 0),
		"query_results": [
			{
				"query_id": report["query_id"],
				"problem_file": str(report["case"]["problem_file"]),
				"log_dir": str(report["log_dir"]),
				"success": bool(report["success"]),
				"outcome_bucket": report["outcome_bucket"],
				"plan_solve_status": (
					(report.get("plan_solve", {}).get("summary", {}) or {}).get("status")
				),
				"plan_verification_status": (
					(report.get("plan_verification", {}).get("summary", {}) or {}).get("status")
				),
				"selected_solver_id": (
					(report.get("plan_verification", {}).get("artifacts", {}) or {}).get(
						"selected_solver_id"
					)
				),
			}
			for report in query_reports
		],
	}
	output_root = GENERATED_BASELINE_DIR / domain_key
	output_root.mkdir(parents=True, exist_ok=True)
	(output_root / "summary.json").write_text(json.dumps(summary, indent=2))
	return summary


def run_generated_problem_root_case(
	domain_key: str,
	query_id: str,
	*,
	generated_domain_file: str,
) -> Dict[str, Any]:
	apply_generated_runtime_defaults()
	query_cases = load_domain_query_cases(domain_key)
	case = query_cases[query_id]
	pipeline = HTNEvaluationPipeline(
		domain_file=str(Path(generated_domain_file).resolve()),
		problem_file=str(case["problem_file"]),
	)
	pipeline.logger = ExecutionLogger(logs_dir=str(GENERATED_LOGS_DIR), run_origin="tests")
	pipeline.logger.start_pipeline(
		case["instruction"],
		mode="generated_problem_root_execution",
		domain_file=str(Path(generated_domain_file).resolve()),
		problem_file=str(case["problem_file"]),
		domain_name=pipeline.domain.name,
		problem_name=pipeline.problem.name if pipeline.problem else None,
		output_dir=str(GENERATED_LOGS_DIR),
	)
	pipeline.output_dir = pipeline.logger.current_log_dir
	if pipeline.logger.current_record is not None and pipeline.output_dir is not None:
		pipeline.logger.current_record.output_dir = str(pipeline.output_dir)
		pipeline.logger._save_current_state()

	race_result = pipeline._execute_official_problem_root_parallel_solver_race()
	plan_solve = dict(race_result.get("plan_solve") or {})
	plan_verification = dict(race_result.get("plan_verification") or {})
	success = (
		(plan_solve.get("summary") or {}).get("status") == "success"
		and (plan_verification.get("summary") or {}).get("status") == "success"
	)
	log_path = pipeline.logger.end_pipeline(success=success)
	log_dir = Path(log_path).parent
	execution = json.loads((log_dir / "execution.json").read_text())
	verifier_artifacts = dict(plan_verification.get("artifacts") or {})
	outcome_bucket = (
		verifier_artifacts.get("selected_bucket")
		or (plan_verification.get("summary") or {}).get("failure_bucket")
		or (plan_solve.get("summary") or {}).get("failure_bucket")
		or ("hierarchical_plan_verified" if success else "unknown_failure")
	)
	return {
		"query_id": query_id,
		"case": case,
		"success": success,
		"outcome_bucket": outcome_bucket,
		"log_dir": log_dir,
		"execution": execution,
		"plan_solve": plan_solve,
		"plan_verification": plan_verification,
	}


def run_generated_problem_root_baseline_for_domain(
	domain_key: str,
	*,
	query_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
	build_report = run_generated_domain_build(domain_key)
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

	query_reports = []
	if build_report.get("success"):
		generated_domain_file = str(
			(build_report.get("artifact_paths") or {}).get("generated_domain") or ""
		)
		query_reports = [
			run_generated_problem_root_case(
				domain_key,
				query_id,
				generated_domain_file=generated_domain_file,
			)
			for query_id in selected_query_ids
		]

	counts = {
		"hierarchical_plan_verified": 0,
		"primitive_plan_valid_but_hierarchical_rejected": 0,
		"primitive_plan_invalid": 0,
		"no_plan_from_solver": 0,
		"unknown_failure": 0,
	}
	for report in query_reports:
		bucket = str(report.get("outcome_bucket") or "unknown_failure")
		counts[bucket] = counts.get(bucket, 0) + 1

	summary = {
		"domain_key": domain_key,
		"domain_build": {
			"success": bool(build_report.get("success")),
			"log_dir": str(build_report.get("log_dir")),
			"artifact_root": str(build_report.get("artifact_root")),
			"source_domain_kind": str(build_report.get("source_domain_kind") or ""),
			"masked_domain_file": str(
				(build_report.get("artifact_paths") or {}).get("masked_domain") or ""
			),
			"generated_domain_file": str(
				(build_report.get("artifact_paths") or {}).get("generated_domain") or ""
			),
			"domain_build_invocations": 1,
			"llm_attempted": bool(build_report.get("llm_attempted")),
			"llm_generation_attempts": int(build_report.get("llm_generation_attempts") or 0),
			"llm_attempts": int(build_report.get("llm_attempts") or 0),
			"llm_request_id": str(build_report.get("llm_request_id") or ""),
			"llm_response_mode": str(build_report.get("llm_response_mode") or ""),
			"llm_first_chunk_seconds": build_report.get("llm_first_chunk_seconds"),
			"llm_complete_json_seconds": build_report.get("llm_complete_json_seconds"),
			"method_synthesis_model": str(build_report.get("method_synthesis_model") or ""),
			"generated_method_count": int(build_report.get("generated_method_count") or 0),
		},
		"domain_gate_preflight": {
			"success": bool((build_report.get("domain_gate") or {}).get("validated_task_count") is not None),
			"log_dir": str(build_report.get("log_dir")),
			"artifact_root": str(build_report.get("artifact_root")),
			"validated_task_count": (build_report.get("domain_gate") or {}).get(
				"validated_task_count"
			),
		},
		"total_queries": len(query_reports),
		"selected_query_ids": list(selected_query_ids),
		"llm_generation_attempts_total": int(build_report.get("llm_generation_attempts") or 0),
		"verified_successes": counts.get("hierarchical_plan_verified", 0),
		"hierarchical_rejection_failures": counts.get(
			"primitive_plan_valid_but_hierarchical_rejected",
			0,
		),
		"primitive_invalid_failures": counts.get("primitive_plan_invalid", 0),
		"solver_no_plan_failures": counts.get("no_plan_from_solver", 0),
		"unknown_failures": counts.get("unknown_failure", 0),
		"query_results": [
			{
				"query_id": report["query_id"],
				"problem_file": str(report["case"]["problem_file"]),
				"log_dir": str(report["log_dir"]),
				"success": bool(report["success"]),
				"outcome_bucket": report["outcome_bucket"],
				"plan_solve_status": (
					(report.get("plan_solve", {}).get("summary", {}) or {}).get("status")
				),
				"plan_verification_status": (
					(report.get("plan_verification", {}).get("summary", {}) or {}).get("status")
				),
				"selected_solver_id": (
					(report.get("plan_verification", {}).get("artifacts", {}) or {}).get(
						"selected_solver_id"
					)
				),
			}
			for report in query_reports
		],
	}
	output_root = GENERATED_MASKED_BASELINE_DIR / domain_key
	output_root.mkdir(parents=True, exist_ok=True)
	(output_root / "summary.json").write_text(json.dumps(summary, indent=2))
	return summary


__all__ = [
	"DOMAIN_FILES",
	"load_domain_query_cases",
	"query_id_sort_key",
	"run_domain_problem_root_case",
	"run_generated_domain_build",
	"run_generated_problem_root_baseline_for_domain",
	"run_generated_problem_root_case",
	"run_official_domain_gate_preflight",
	"run_official_problem_root_baseline_for_domain",
]
