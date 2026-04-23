from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TextIO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
RUNS_ROOT = PROJECT_ROOT / "tests" / "method_library" / "generated" / "generated_masked_full"
DOMAIN_KEYS = ("blocksworld", "marsrover", "satellite", "transport")


@dataclass
class DomainRun:
	name: str
	command: List[str]
	output_path: Path
	summary_path: Path
	output_handle: TextIO
	process: subprocess.Popen[str]


def _timestamp() -> str:
	return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _build_env() -> Dict[str, str]:
	from tests.support.plan_library_generation_support import apply_generated_runtime_defaults

	env = os.environ.copy()
	env["PYTHONPATH"] = os.pathsep.join(
		[
			str(PROJECT_ROOT / "src"),
			str(PROJECT_ROOT),
		],
	)
	return dict(apply_generated_runtime_defaults(env))


def _start_domain_run(run_dir: Path, domain_key: str, env: Dict[str, str]) -> DomainRun:
	output_path = run_dir / f"{domain_key}.out"
	summary_path = run_dir / f"{domain_key}.summary.json"
	command = [
		sys.executable,
		str(Path(__file__).resolve()),
		"--domain",
		domain_key,
		"--run-dir",
		str(run_dir),
	]
	output_handle = output_path.open("w")
	process = subprocess.Popen(
		command,
		cwd=PROJECT_ROOT,
		env=env,
		stdout=output_handle,
		stderr=subprocess.STDOUT,
		text=True,
	)
	return DomainRun(
		name=domain_key,
		command=command,
		output_path=output_path,
		summary_path=summary_path,
		output_handle=output_handle,
		process=process,
	)


def _record_launch_metadata(launch_path: Path, runs: List[DomainRun]) -> None:
	launch_metadata = {
		run.name: {
			"pid": run.process.pid,
			"command": run.command,
			"output_path": str(run.output_path),
			"summary_path": str(run.summary_path),
		}
		for run in runs
	}
	launch_path.write_text(json.dumps(launch_metadata, indent=2))


def _collect_domain_run_result(
	run: DomainRun,
	domain_summaries: Dict[str, Dict[str, object]],
	internal_failures: List[str],
) -> None:
	try:
		return_code = run.process.wait()
	finally:
		run.output_handle.close()
	if return_code != 0:
		internal_failures.append(run.name)
		return
	if not run.summary_path.exists():
		internal_failures.append(run.name)
		return
	domain_summaries[run.name] = json.loads(run.summary_path.read_text())


def _aggregate_domain_summaries(
	run_dir: Path,
	domain_summaries: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
	total_queries = sum(int(summary.get("total_queries", 0)) for summary in domain_summaries.values())
	domain_build_pass = sum(
		1
		for summary in domain_summaries.values()
		if bool((summary.get("domain_build") or {}).get("success"))
	)
	domain_gate_pass = sum(
		1
		for summary in domain_summaries.values()
		if bool((summary.get("domain_gate_preflight") or {}).get("success"))
	)
	return {
		"run_dir": str(run_dir),
		"domain_build_pass": domain_build_pass,
		"domain_build_fail": len(domain_summaries) - domain_build_pass,
		"domain_gate_pass": domain_gate_pass,
		"domain_gate_fail": len(domain_summaries) - domain_gate_pass,
		"llm_generation_attempts_total": sum(
			int(summary.get("llm_generation_attempts_total", 0))
			for summary in domain_summaries.values()
		),
		"total_queries": total_queries,
		"solver_no_plan_failures": sum(
			int(summary.get("solver_no_plan_failures", 0))
			for summary in domain_summaries.values()
		),
		"primitive_invalid_failures": sum(
			int(summary.get("primitive_invalid_failures", 0))
			for summary in domain_summaries.values()
		),
		"hierarchical_rejection_failures": sum(
			int(summary.get("hierarchical_rejection_failures", 0))
			for summary in domain_summaries.values()
		),
		"verified_successes": sum(
			int(summary.get("verified_successes", 0))
			for summary in domain_summaries.values()
		),
		"unknown_failures": sum(
			int(summary.get("unknown_failures", 0))
			for summary in domain_summaries.values()
		),
		"domains": domain_summaries,
	}


def _write_human_summary(run_dir: Path, summary: Dict[str, object]) -> None:
	lines = [
		f"run_dir: {summary['run_dir']}",
		f"domain_build_pass: {summary['domain_build_pass']}",
		f"domain_build_fail: {summary['domain_build_fail']}",
		f"domain_gate_pass: {summary['domain_gate_pass']}",
		f"domain_gate_fail: {summary['domain_gate_fail']}",
		f"llm_generation_attempts_total: {summary['llm_generation_attempts_total']}",
		f"total_queries: {summary['total_queries']}",
		f"solver_no_plan_failures: {summary['solver_no_plan_failures']}",
		f"primitive_invalid_failures: {summary['primitive_invalid_failures']}",
		f"hierarchical_rejection_failures: {summary['hierarchical_rejection_failures']}",
		f"verified_successes: {summary['verified_successes']}",
		f"unknown_failures: {summary['unknown_failures']}",
	]
	for domain_key, domain_summary in dict(summary.get("domains") or {}).items():
		lines.append(
			f"{domain_key}: build_success={bool((domain_summary.get('domain_build') or {}).get('success'))}, "
			f"gate_success={bool((domain_summary.get('domain_gate_preflight') or {}).get('success'))}, "
			f"llm_generation_attempts={int(domain_summary.get('llm_generation_attempts_total', 0))}, "
			f"queries={domain_summary.get('total_queries')}, "
			f"verified={domain_summary.get('verified_successes')}, "
			f"hierarchical_rejected={domain_summary.get('hierarchical_rejection_failures')}, "
			f"primitive_invalid={domain_summary.get('primitive_invalid_failures')}, "
			f"no_plan={domain_summary.get('solver_no_plan_failures')}",
		)
	(run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
	print("\n".join(lines))


def _run_single_domain(domain_key: str, run_dir: Path) -> int:
	from tests.support.htn_evaluation_support import (
		run_generated_problem_root_baseline_for_domain,
		run_generated_problem_root_case,
	)
	from tests.support.plan_library_generation_support import (
		load_domain_query_cases,
		query_id_sort_key,
	)

	if _RUN_GENERATED_DOMAIN_FILE:
		query_cases = load_domain_query_cases(domain_key)
		selected_query_ids = (
			tuple(sorted(_RUN_QUERY_IDS, key=query_id_sort_key))
			if _RUN_QUERY_IDS
			else tuple(sorted(query_cases, key=query_id_sort_key))
		)
		query_reports = [
			run_generated_problem_root_case(
				domain_key,
				query_id,
				generated_domain_file=_RUN_GENERATED_DOMAIN_FILE,
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
				"success": True,
				"log_dir": None,
				"artifact_root": None,
				"source_domain_kind": "generated",
				"masked_domain_file": None,
				"generated_domain_file": _RUN_GENERATED_DOMAIN_FILE,
				"domain_build_invocations": 0,
				"reused_generated_domain": True,
				"llm_attempted": False,
				"llm_generation_attempts": 0,
				"llm_attempts": 0,
				"llm_request_id": "",
				"llm_response_mode": "",
				"llm_first_chunk_seconds": None,
				"llm_complete_json_seconds": None,
				"method_synthesis_model": "",
				"generated_method_count": 0,
			},
			"domain_gate_preflight": {
				"success": True,
				"log_dir": None,
				"artifact_root": None,
				"validated_task_count": None,
				"reused_generated_domain": True,
			},
			"total_queries": len(query_reports),
			"selected_query_ids": list(selected_query_ids),
			"llm_generation_attempts_total": 0,
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
	else:
		summary = run_generated_problem_root_baseline_for_domain(
			domain_key,
			query_ids=tuple(_RUN_QUERY_IDS) if _RUN_QUERY_IDS else None,
		)
	summary_path = run_dir / f"{domain_key}.summary.json"
	summary_path.write_text(json.dumps(summary, indent=2))
	print(json.dumps(summary, indent=2))
	return 0


def _run_full_baseline(*, max_concurrent_domains: int = len(DOMAIN_KEYS)) -> int:
	if max_concurrent_domains < 1:
		raise ValueError("max_concurrent_domains must be at least 1")
	run_dir = RUNS_ROOT / _timestamp()
	run_dir.mkdir(parents=True, exist_ok=True)
	env = _build_env()
	launch_path = run_dir / "launch.json"
	pending_domains = list(DOMAIN_KEYS)
	active_runs: List[DomainRun] = []

	domain_summaries: Dict[str, Dict[str, object]] = {}
	internal_failures: List[str] = []
	while pending_domains or active_runs:
		while pending_domains and len(active_runs) < max_concurrent_domains:
			domain_key = pending_domains.pop(0)
			active_runs.append(_start_domain_run(run_dir, domain_key, env))
			_record_launch_metadata(launch_path, active_runs)
		if not active_runs:
			continue
		run = active_runs.pop(0)
		_collect_domain_run_result(run, domain_summaries, internal_failures)

	summary = _aggregate_domain_summaries(run_dir, domain_summaries)
	summary["max_concurrent_domains"] = max_concurrent_domains
	summary["internal_failures"] = internal_failures
	summary["completed_domains"] = sorted(domain_summaries)
	summary["complete"] = len(domain_summaries) == len(DOMAIN_KEYS) and not internal_failures
	(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
	_write_human_summary(run_dir, summary)
	return 0 if summary["complete"] else 1


_RUN_QUERY_IDS: List[str] = []
_RUN_GENERATED_DOMAIN_FILE: str = ""


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--domain", choices=DOMAIN_KEYS)
	parser.add_argument("--run-dir")
	parser.add_argument("--query-id", action="append", default=[])
	parser.add_argument("--generated-domain-file")
	parser.add_argument("--max-concurrent-domains", type=int, default=len(DOMAIN_KEYS))
	args = parser.parse_args()
	global _RUN_QUERY_IDS, _RUN_GENERATED_DOMAIN_FILE
	_RUN_QUERY_IDS = list(args.query_id or [])
	_RUN_GENERATED_DOMAIN_FILE = str(args.generated_domain_file or "").strip()

	if args.domain:
		if not args.run_dir:
			raise SystemExit("--run-dir is required when --domain is set")
		run_dir = Path(args.run_dir).resolve()
		run_dir.mkdir(parents=True, exist_ok=True)
		return _run_single_domain(args.domain, run_dir)

	return _run_full_baseline(max_concurrent_domains=args.max_concurrent_domains)


if __name__ == "__main__":
	raise SystemExit(main())
