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
RUNS_ROOT = PROJECT_ROOT / "tests" / "generated" / "online_query_solution_full"
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
	from tests.support.online_query_solution_benchmark_support import (
		apply_online_query_runtime_defaults,
	)

	env = os.environ.copy()
	env["PYTHONPATH"] = os.pathsep.join(
		[
			str(PROJECT_ROOT / "src"),
			str(PROJECT_ROOT),
		],
	)
	return dict(apply_online_query_runtime_defaults(env))


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
	return {
		"run_dir": str(run_dir),
		"online_domain_source": "benchmark",
		"library_source": _RUN_LIBRARY_SOURCE,
		"total_queries": total_queries,
		"verified_successes": sum(
			int(summary.get("verified_successes", 0))
			for summary in domain_summaries.values()
		),
		"goal_grounding_failures": sum(
			int(summary.get("goal_grounding_failures", 0))
			for summary in domain_summaries.values()
		),
		"temporal_compilation_failures": sum(
			int(summary.get("temporal_compilation_failures", 0))
			for summary in domain_summaries.values()
		),
		"agentspeak_rendering_failures": sum(
			int(summary.get("agentspeak_rendering_failures", 0))
			for summary in domain_summaries.values()
		),
		"runtime_execution_failures": sum(
			int(summary.get("runtime_execution_failures", 0))
			for summary in domain_summaries.values()
		),
		"plan_verification_failures": sum(
			int(summary.get("plan_verification_failures", 0))
			for summary in domain_summaries.values()
		),
		"hierarchical_rejection_failures": sum(
			int(summary.get("hierarchical_rejection_failures", 0))
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
		f"online_domain_source: {summary['online_domain_source']}",
		f"library_source: {summary['library_source']}",
		f"total_queries: {summary['total_queries']}",
		f"verified_successes: {summary['verified_successes']}",
		f"goal_grounding_failures: {summary['goal_grounding_failures']}",
		f"temporal_compilation_failures: {summary['temporal_compilation_failures']}",
		f"agentspeak_rendering_failures: {summary['agentspeak_rendering_failures']}",
		f"runtime_execution_failures: {summary['runtime_execution_failures']}",
		f"plan_verification_failures: {summary['plan_verification_failures']}",
		f"hierarchical_rejection_failures: {summary['hierarchical_rejection_failures']}",
		f"unknown_failures: {summary['unknown_failures']}",
	]
	for domain_key, domain_summary in dict(summary.get("domains") or {}).items():
		lines.append(
			f"{domain_key}: queries={domain_summary.get('total_queries')}, "
			f"verified={domain_summary.get('verified_successes')}, "
			f"grounding_failed={domain_summary.get('goal_grounding_failures')}, "
			f"runtime_failed={domain_summary.get('runtime_execution_failures')}, "
			f"verification_failed={domain_summary.get('plan_verification_failures')}, "
			f"hierarchical_rejected={domain_summary.get('hierarchical_rejection_failures')}",
		)
	(run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
	print("\n".join(lines))


def _run_single_domain(domain_key: str, run_dir: Path) -> int:
	from tests.support.online_query_solution_benchmark_support import (
		run_online_query_solution_benchmark_for_domain,
	)

	summary = run_online_query_solution_benchmark_for_domain(
		domain_key,
		query_ids=tuple(_RUN_QUERY_IDS) if _RUN_QUERY_IDS else None,
		library_source=_RUN_LIBRARY_SOURCE,
	)
	summary_path = run_dir / f"{domain_key}.summary.json"
	summary_path.write_text(json.dumps(summary, indent=2))
	print(json.dumps(summary, indent=2))
	return 0


def _run_full_benchmark(*, max_concurrent_domains: int = 1) -> int:
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
_RUN_LIBRARY_SOURCE = "official"


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--domain", choices=DOMAIN_KEYS)
	parser.add_argument("--run-dir")
	parser.add_argument("--query-id", action="append", default=[])
	parser.add_argument("--max-concurrent-domains", type=int, default=1)
	parser.add_argument("--library-source", choices=("official", "generated"), default="official")
	args = parser.parse_args()
	global _RUN_QUERY_IDS, _RUN_LIBRARY_SOURCE
	_RUN_QUERY_IDS = list(args.query_id or [])
	_RUN_LIBRARY_SOURCE = str(args.library_source)

	if args.domain:
		if not args.run_dir:
			raise SystemExit("--run-dir is required when --domain is set")
		run_dir = Path(args.run_dir).resolve()
		run_dir.mkdir(parents=True, exist_ok=True)
		return _run_single_domain(args.domain, run_dir)

	return _run_full_benchmark(max_concurrent_domains=args.max_concurrent_domains)


if __name__ == "__main__":
	raise SystemExit(main())
