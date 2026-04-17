from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
	sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from domain_build.method_synthesis.schema import HTNLiteral, HTNMethod, HTNMethodLibrary, HTNMethodStep, HTNTask
from domain_build.method_synthesis.synthesizer import HTNMethodSynthesizer
from pipeline.domain_complete_pipeline import DomainCompletePipeline
from pipeline.execution_logger import ExecutionLogger
from utils.benchmark_query_dataset import (
	DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS,
	DEFAULT_BENCHMARK_QUERY_DOMAIN_PROBLEM_DIRS,
	load_problem_query_cases,
)
from utils.hddl_condition_parser import HDDLConditionParser
from utils.hddl_parser import HDDLParser


TESTS_ROOT = PROJECT_ROOT / "tests"
GENERATED_ROOT = TESTS_ROOT / "generated"
GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
GENERATED_LOGS_DIR = GENERATED_ROOT / "logs"
GENERATED_LOGS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DOMAIN_BUILDS_DIR = GENERATED_ROOT / "domain_builds"
GENERATED_DOMAIN_BUILDS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_BASELINE_DIR = GENERATED_ROOT / "official_ground_truth"
GENERATED_BASELINE_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_FILES = {
	domain_key: str((PROJECT_ROOT / "src" / "domains" / domain_key / "domain.hddl").resolve())
	for domain_key in ("blocksworld", "marsrover", "satellite", "transport")
}


def query_id_sort_key(query_id: str) -> tuple[int, str]:
	match = re.fullmatch(r"query_(\d+)", str(query_id).strip())
	if match is None:
		return (10**9, str(query_id))
	return (int(match.group(1)), str(query_id))


def load_domain_query_cases(domain_key: str, *, limit: int = 0) -> Dict[str, Dict[str, Any]]:
	pattern = DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS[domain_key]
	query_limit = limit if limit > 0 else 10**9
	return load_problem_query_cases(
		DEFAULT_BENCHMARK_QUERY_DOMAIN_PROBLEM_DIRS[domain_key],
		limit=query_limit,
		pattern=pattern,
	)


def build_official_method_library(domain_file: str) -> HTNMethodLibrary:
	domain = HDDLParser.parse_domain(domain_file)
	condition_parser = HDDLConditionParser()
	synthesizer = HTNMethodSynthesizer()
	primitive_action_names = {action.name for action in domain.actions}
	compound_tasks = [
		HTNTask(
			name=task.name,
			parameters=tuple(condition_parser._extract_parameter_names(task.parameters)),
			is_primitive=False,
			source_predicates=(),
			source_name=task.name,
		)
		for task in domain.tasks
	]
	primitive_tasks = synthesizer._build_primitive_tasks(domain)
	methods: List[HTNMethod] = []
	for method in domain.methods:
		context = tuple(
			HTNLiteral(
				predicate=literal.predicate,
				args=tuple(literal.args),
				is_positive=literal.is_positive,
				source_symbol=None,
			)
			for literal in condition_parser.parse_literals(
				method.precondition,
				action_name=method.name,
				scope="official_method_precondition",
			)
		)
		subtasks = tuple(
			HTNMethodStep(
				step_id=step.label,
				task_name=(
					synthesizer._sanitize_name(step.task_name)
					if step.task_name in primitive_action_names
					else step.task_name
				),
				args=tuple(step.args),
				kind="primitive" if step.task_name in primitive_action_names else "compound",
				action_name=step.task_name if step.task_name in primitive_action_names else None,
			)
			for step in method.subtasks
		)
		methods.append(
			HTNMethod(
				method_name=method.name,
				task_name=method.task_name,
				parameters=tuple(condition_parser._extract_parameter_names(method.parameters)),
				task_args=tuple(method.task_args),
				context=context,
				subtasks=subtasks,
				ordering=tuple(tuple(edge) for edge in method.ordering),
				origin="official_hddl",
				source_method_name=method.name,
			)
		)
	return HTNMethodLibrary(
		compound_tasks=compound_tasks,
		primitive_tasks=primitive_tasks,
		methods=methods,
		target_literals=[],
		target_task_bindings=[],
	)


def run_official_domain_gate_preflight(domain_key: str) -> Dict[str, Any]:
	domain_file = DOMAIN_FILES[domain_key]
	pipeline = DomainCompletePipeline(domain_file=domain_file)
	pipeline.logger = ExecutionLogger(logs_dir=str(GENERATED_LOGS_DIR), run_origin="tests")
	pipeline.logger.start_pipeline(
		f"Official domain gate preflight for {domain_key}",
		mode="official_domain_preflight",
		domain_file=domain_file,
		domain_name=pipeline.domain.name,
		output_dir=str(GENERATED_LOGS_DIR),
	)
	pipeline.output_dir = pipeline.logger.current_log_dir
	if pipeline.logger.current_record is not None and pipeline.output_dir is not None:
		pipeline.logger.current_record.output_dir = str(pipeline.output_dir)
		pipeline.logger._save_current_state()

	method_library = build_official_method_library(domain_file)
	domain_gate_summary = pipeline._validate_domain_library(method_library)
	success = domain_gate_summary is not None
	log_path = pipeline.logger.end_pipeline(success=success)
	log_dir = Path(log_path).parent
	execution = json.loads((log_dir / "execution.json").read_text())

	artifact_root = GENERATED_DOMAIN_BUILDS_DIR / "official_ground_truth" / domain_key
	artifact_root.mkdir(parents=True, exist_ok=True)
	artifact_path = artifact_root / "domain_gate.json"
	artifact_path.write_text(
		json.dumps(
			{
				"success": success,
				"domain_gate": domain_gate_summary,
				"log_dir": str(log_dir),
			},
			indent=2,
		)
	)
	return {
		"success": success,
		"domain_gate": domain_gate_summary,
		"log_dir": log_dir,
		"artifact_root": artifact_root,
		"execution": execution,
	}


def run_domain_problem_root_case(domain_key: str, query_id: str) -> Dict[str, Any]:
	query_cases = load_domain_query_cases(domain_key)
	case = query_cases[query_id]
	domain_file = DOMAIN_FILES[domain_key]
	pipeline = DomainCompletePipeline(
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


def run_official_problem_root_baseline_for_domain(domain_key: str) -> Dict[str, Any]:
	query_cases = load_domain_query_cases(domain_key)
	query_reports = [
		run_domain_problem_root_case(domain_key, query_id)
		for query_id in sorted(query_cases, key=query_id_sort_key)
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
					(report.get("plan_verification", {}).get("artifacts", {}) or {}).get("selected_solver_id")
				),
			}
			for report in query_reports
		],
	}
	output_root = GENERATED_BASELINE_DIR / domain_key
	output_root.mkdir(parents=True, exist_ok=True)
	(output_root / "summary.json").write_text(json.dumps(summary, indent=2))
	return summary


__all__ = [
	"DOMAIN_FILES",
	"build_official_method_library",
	"load_domain_query_cases",
	"query_id_sort_key",
	"run_domain_problem_root_case",
	"run_official_domain_gate_preflight",
	"run_official_problem_root_baseline_for_domain",
]
