from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
RUNS_ROOT = PROJECT_ROOT / "tests" / "generated" / "official_ground_truth_full"
DOMAIN_KEYS = ("blocksworld", "marsrover", "satellite", "transport")


@dataclass
class DomainRun:
	name: str
	command: List[str]
	output_path: Path
	summary_path: Path
	process: subprocess.Popen[str]


def _timestamp() -> str:
	return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _build_env() -> Dict[str, str]:
	env = os.environ.copy()
	env["PYTHONPATH"] = os.pathsep.join(
		[
			str(PROJECT_ROOT / "src"),
			str(PROJECT_ROOT),
		],
	)
	return env


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
		process=process,
	)


def _aggregate_domain_summaries(run_dir: Path, domain_summaries: Dict[str, Dict[str, object]]) -> Dict[str, object]:
	total_queries = sum(int(summary.get("total_queries", 0)) for summary in domain_summaries.values())
	domain_gate_pass = sum(
		1
		for summary in domain_summaries.values()
		if bool((summary.get("domain_gate_preflight") or {}).get("success"))
	)
	return {
		"run_dir": str(run_dir),
		"domain_gate_pass": domain_gate_pass,
		"domain_gate_fail": len(domain_summaries) - domain_gate_pass,
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
		f"domain_gate_pass: {summary['domain_gate_pass']}",
		f"domain_gate_fail: {summary['domain_gate_fail']}",
		f"total_queries: {summary['total_queries']}",
		f"solver_no_plan_failures: {summary['solver_no_plan_failures']}",
		f"primitive_invalid_failures: {summary['primitive_invalid_failures']}",
		f"hierarchical_rejection_failures: {summary['hierarchical_rejection_failures']}",
		f"verified_successes: {summary['verified_successes']}",
		f"unknown_failures: {summary['unknown_failures']}",
	]
	for domain_key, domain_summary in dict(summary.get("domains") or {}).items():
		lines.append(
			f"{domain_key}: gate_success={bool((domain_summary.get('domain_gate_preflight') or {}).get('success'))}, "
			f"queries={domain_summary.get('total_queries')}, "
			f"verified={domain_summary.get('verified_successes')}, "
			f"hierarchical_rejected={domain_summary.get('hierarchical_rejection_failures')}, "
			f"primitive_invalid={domain_summary.get('primitive_invalid_failures')}, "
			f"no_plan={domain_summary.get('solver_no_plan_failures')}",
		)
	(run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
	print("\n".join(lines))


def _run_single_domain(domain_key: str, run_dir: Path) -> int:
	from tests.support.ground_truth_baseline_support import run_official_problem_root_baseline_for_domain

	summary = run_official_problem_root_baseline_for_domain(
		domain_key,
		query_ids=tuple(_RUN_QUERY_IDS) if _RUN_QUERY_IDS else None,
	)
	summary_path = run_dir / f"{domain_key}.summary.json"
	summary_path.write_text(json.dumps(summary, indent=2))
	print(json.dumps(summary, indent=2))
	return 0


def _run_parallel_full_baseline() -> int:
	run_dir = RUNS_ROOT / _timestamp()
	run_dir.mkdir(parents=True, exist_ok=True)
	env = _build_env()
	runs = [_start_domain_run(run_dir, domain_key, env) for domain_key in DOMAIN_KEYS]

	launch_metadata = {
		run.name: {
			"pid": run.process.pid,
			"command": run.command,
			"output_path": str(run.output_path),
			"summary_path": str(run.summary_path),
		}
		for run in runs
	}
	(run_dir / "launch.json").write_text(json.dumps(launch_metadata, indent=2))

	domain_summaries: Dict[str, Dict[str, object]] = {}
	internal_failures: List[str] = []
	for run in runs:
		return_code = run.process.wait()
		if return_code != 0:
			internal_failures.append(run.name)
			continue
		if not run.summary_path.exists():
			internal_failures.append(run.name)
			continue
		domain_summaries[run.name] = json.loads(run.summary_path.read_text())

	summary = _aggregate_domain_summaries(run_dir, domain_summaries)
	summary["internal_failures"] = internal_failures
	summary["completed_domains"] = sorted(domain_summaries)
	summary["complete"] = len(domain_summaries) == len(DOMAIN_KEYS) and not internal_failures
	(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
	_write_human_summary(run_dir, summary)
	return 0 if summary["complete"] else 1


_RUN_QUERY_IDS: List[str] = []


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--domain", choices=DOMAIN_KEYS)
	parser.add_argument("--run-dir")
	parser.add_argument("--query-id", action="append", default=[])
	args = parser.parse_args()
	global _RUN_QUERY_IDS
	_RUN_QUERY_IDS = list(args.query_id or [])

	if args.domain:
		if not args.run_dir:
			raise SystemExit("--run-dir is required when --domain is set")
		run_dir = Path(args.run_dir).resolve()
		run_dir.mkdir(parents=True, exist_ok=True)
		return _run_single_domain(args.domain, run_dir)

	return _run_parallel_full_baseline()


if __name__ == "__main__":
	raise SystemExit(main())
