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
RUNS_ROOT = PROJECT_ROOT / "tests" / "generated" / "plan_library_evaluation_full"
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
	from tests.support.plan_library_evaluation_support import (
		apply_evaluation_runtime_defaults,
	)

	env = os.environ.copy()
	env["PYTHONPATH"] = os.pathsep.join(
		[
			str(PROJECT_ROOT / "src"),
			str(PROJECT_ROOT),
		],
	)
	return dict(apply_evaluation_runtime_defaults(env))


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
		"--library-source",
		_RUN_LIBRARY_SOURCE,
		"--runtime-backend",
		_RUN_RUNTIME_BACKEND,
	]
	for query_id in _RUN_QUERY_IDS:
		command.extend(["--query-id", str(query_id)])
	for query_id in _RUN_FAILED_ONLY_QUERY_IDS.get(domain_key, ()):
		command.extend(["--query-id", str(query_id)])
	if _RUN_RESUME:
		command.append("--resume")
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
	if not run.summary_path.exists():
		if run.name not in internal_failures:
			internal_failures.append(run.name)
		return
	summary = json.loads(run.summary_path.read_text())
	summary["process_return_code"] = return_code
	summary["process_failed"] = return_code != 0
	summary["process_output_path"] = str(run.output_path)
	domain_summaries[run.name] = summary
	if return_code != 0 and run.name not in internal_failures:
		internal_failures.append(run.name)


def _load_existing_domain_summaries(run_dir: Path) -> Dict[str, Dict[str, object]]:
	domain_summaries: Dict[str, Dict[str, object]] = {}
	for domain_key in DOMAIN_KEYS:
		summary_path = run_dir / f"{domain_key}.summary.json"
		if not summary_path.exists():
			continue
		try:
			summary = json.loads(summary_path.read_text())
		except json.JSONDecodeError:
			continue
		domain_summaries[domain_key] = summary
	return domain_summaries


def _load_failed_query_ids_for_run(
	run_id: str,
	*,
	runs_root: Path = RUNS_ROOT,
) -> Dict[str, List[str]]:
	run_dir = runs_root / str(run_id).strip()
	failed_query_ids_by_domain: Dict[str, List[str]] = {}
	for domain_key in DOMAIN_KEYS:
		summary_path = run_dir / domain_key / "summary.json"
		if not summary_path.exists():
			summary_path = run_dir / f"{domain_key}.summary.json"
		if not summary_path.exists():
			continue
		try:
			summary = json.loads(summary_path.read_text())
		except json.JSONDecodeError:
			continue
		failed_query_ids = [
			str(result.get("query_id") or "").strip()
			for result in (summary.get("query_results") or ())
			if str(result.get("query_id") or "").strip()
			and not bool(result.get("success"))
		]
		if failed_query_ids:
			failed_query_ids_by_domain[domain_key] = failed_query_ids
	return failed_query_ids_by_domain


def _aggregate_domain_summaries(
	run_dir: Path,
	domain_summaries: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
	total_queries = sum(int(summary.get("total_queries", 0)) for summary in domain_summaries.values())
	completed_query_count = sum(
		len(summary.get("completed_query_ids") or summary.get("query_results") or [])
		for summary in domain_summaries.values()
	)
	remaining_query_count = sum(
		len(summary.get("remaining_query_ids") or [])
		for summary in domain_summaries.values()
	)
	return {
		"run_dir": str(run_dir),
		"evaluation_domain_source": "benchmark",
		"library_source": _RUN_LIBRARY_SOURCE,
		"runtime_backend": _RUN_RUNTIME_BACKEND,
		"total_queries": total_queries,
		"completed_query_count": completed_query_count,
		"remaining_query_count": remaining_query_count,
		"verified_successes": sum(
			int(summary.get("verified_successes", 0))
			for summary in domain_summaries.values()
		),
		"bdi_runtime_successes": sum(
			int(
				summary.get(
					"bdi_runtime_successes",
					summary.get("verified_successes", 0),
				),
			)
			for summary in domain_summaries.values()
		),
		"hierarchical_compatibility_successes": sum(
			int(
				summary.get(
					"hierarchical_compatibility_successes",
					summary.get("hierarchical_verified_successes", 0),
				),
			)
			for summary in domain_summaries.values()
		),
		"runtime_goal_verified_successes": sum(
			int(summary.get("runtime_goal_verified_successes", 0))
			for summary in domain_summaries.values()
		),
		"goal_grounding_failures": sum(
			int(summary.get("goal_grounding_failures", 0))
			for summary in domain_summaries.values()
		),
		"goal_grounding_provider_failures": sum(
			int(summary.get("goal_grounding_provider_failures", 0))
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
		f"run_id: {summary.get('run_id') or run_dir.name}",
		f"run_dir: {summary['run_dir']}",
		f"evaluation_domain_source: {summary['evaluation_domain_source']}",
		f"library_source: {summary['library_source']}",
		f"runtime_backend: {summary.get('runtime_backend', 'jason')}",
		f"total_queries: {summary['total_queries']}",
		f"completed_query_count: {summary.get('completed_query_count', 0)}",
		f"remaining_query_count: {summary.get('remaining_query_count', 0)}",
		f"verified_successes: {summary['verified_successes']}",
		(
			"bdi_runtime_successes: "
			f"{summary.get('bdi_runtime_successes', summary['verified_successes'])}"
		),
		(
			"hierarchical_compatibility_successes: "
			f"{summary.get('hierarchical_compatibility_successes', 0)}"
		),
		f"runtime_goal_verified_successes: {summary.get('runtime_goal_verified_successes', 0)}",
		f"goal_grounding_failures: {summary['goal_grounding_failures']}",
		f"goal_grounding_provider_failures: {summary.get('goal_grounding_provider_failures', 0)}",
		f"agentspeak_rendering_failures: {summary['agentspeak_rendering_failures']}",
		f"runtime_execution_failures: {summary['runtime_execution_failures']}",
		f"plan_verification_failures: {summary['plan_verification_failures']}",
		f"hierarchical_rejection_failures: {summary['hierarchical_rejection_failures']}",
		f"unknown_failures: {summary['unknown_failures']}",
	]
	for domain_key, domain_summary in dict(summary.get("domains") or {}).items():
		bdi_runtime_count = domain_summary.get(
			"bdi_runtime_successes",
			domain_summary.get("verified_successes"),
		)
		hierarchical_compatibility_count = domain_summary.get(
			"hierarchical_compatibility_successes",
			domain_summary.get("hierarchical_verified_successes", 0),
		)
		lines.append(
			f"{domain_key}: queries={domain_summary.get('total_queries')}, "
			f"completed={len(domain_summary.get('completed_query_ids') or domain_summary.get('query_results') or [])}, "
			f"remaining={len(domain_summary.get('remaining_query_ids') or [])}, "
			f"bdi_runtime={bdi_runtime_count}, "
			f"hierarchical_compatibility={hierarchical_compatibility_count}, "
			f"runtime_goal={domain_summary.get('runtime_goal_verified_successes', 0)}, "
			f"grounding_failed={domain_summary.get('goal_grounding_failures')}, "
			f"grounding_provider_failed={domain_summary.get('goal_grounding_provider_failures', 0)}, "
			f"runtime_failed={domain_summary.get('runtime_execution_failures')}, "
			f"verification_failed={domain_summary.get('plan_verification_failures')}, "
			f"hierarchical_rejected={domain_summary.get('hierarchical_rejection_failures')}",
		)
	(run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
	print("\n".join(lines))


def _compact_summary_for_console(summary: Dict[str, object]) -> Dict[str, object]:
	compact = {
		key: value
		for key, value in summary.items()
		if key != "query_results"
	}
	query_results = []
	for result in summary.get("query_results") or []:
		if not isinstance(result, dict):
			continue
		query_results.append(
			{
				"query_id": result.get("query_id"),
				"problem_file": result.get("problem_file"),
				"success": result.get("success"),
				"outcome_bucket": result.get("outcome_bucket"),
				"step": result.get("step"),
				"verification_mode": result.get("verification_mode"),
				"runtime_backend": result.get("runtime_backend"),
				"execution_path": result.get("execution_path"),
				"ltlf_atom_count": result.get("ltlf_atom_count"),
				"jason_failure_class": result.get("jason_failure_class"),
				"failed_goals": result.get("failed_goals"),
				"verifier_missing_goal_facts": result.get("verifier_missing_goal_facts"),
			},
		)
	compact["query_results"] = query_results
	return compact


def _write_latest_run_manifest(run_dir: Path, summary: Dict[str, object]) -> None:
	latest_manifest = {
		"run_id": summary.get("run_id"),
		"run_dir": str(run_dir),
		"summary_json": str(run_dir / "summary.json"),
		"summary_txt": str(run_dir / "summary.txt"),
		"evaluation_domain_source": summary.get("evaluation_domain_source"),
		"library_source": summary.get("library_source"),
		"runtime_backend": summary.get("runtime_backend"),
		"total_queries": summary.get("total_queries"),
		"completed_query_count": summary.get("completed_query_count"),
		"remaining_query_count": summary.get("remaining_query_count"),
		"verified_successes": summary.get("verified_successes"),
		"bdi_runtime_successes": summary.get("bdi_runtime_successes"),
		"hierarchical_compatibility_successes": summary.get(
			"hierarchical_compatibility_successes",
		),
		"runtime_goal_verified_successes": summary.get("runtime_goal_verified_successes"),
		"goal_grounding_provider_failures": summary.get("goal_grounding_provider_failures"),
		"completed_domains": list(summary.get("completed_domains") or []),
		"internal_failures": list(summary.get("internal_failures") or []),
		"complete": bool(summary.get("complete")),
	}
	RUNS_ROOT.mkdir(parents=True, exist_ok=True)
	(RUNS_ROOT / "latest.json").write_text(json.dumps(latest_manifest, indent=2))
	(RUNS_ROOT / "latest_summary.json").write_text(json.dumps(summary, indent=2))
	summary_text_path = run_dir / "summary.txt"
	if summary_text_path.exists():
		(RUNS_ROOT / "latest_summary.txt").write_text(summary_text_path.read_text())


def _write_run_summary_snapshot(
	*,
	run_dir: Path,
	run_id: str,
	domain_summaries: Dict[str, Dict[str, object]],
	internal_failures: List[str],
	max_concurrent_domains: int,
) -> Dict[str, object]:
	run_dir.mkdir(parents=True, exist_ok=True)
	summary = _aggregate_domain_summaries(run_dir, domain_summaries)
	summary["run_id"] = run_id
	summary["max_concurrent_domains"] = max_concurrent_domains
	summary["internal_failures"] = list(internal_failures)
	summary["completed_domains"] = sorted(
		domain_key
		for domain_key, domain_summary in domain_summaries.items()
		if bool(domain_summary.get("complete"))
	)
	summary["partial_domains"] = sorted(
		domain_key
		for domain_key, domain_summary in domain_summaries.items()
		if not bool(domain_summary.get("complete"))
	)
	summary["pending_domains"] = [
		domain_key
		for domain_key in DOMAIN_KEYS
		if domain_key not in domain_summaries
		or not bool(domain_summaries[domain_key].get("complete"))
	]
	summary["complete"] = (
		len(summary["completed_domains"]) == len(DOMAIN_KEYS)
		and not internal_failures
	)
	(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
	_write_human_summary(run_dir, summary)
	_write_latest_run_manifest(run_dir, summary)
	return summary


def _run_single_domain(domain_key: str, run_dir: Path, *, resume: bool = False) -> int:
	from tests.support.plan_library_evaluation_support import (
		run_plan_library_evaluation_benchmark_for_domain,
	)

	summary = run_plan_library_evaluation_benchmark_for_domain(
		domain_key,
		query_ids=(
			tuple(_RUN_QUERY_IDS)
			if _RUN_QUERY_IDS
			else tuple(_RUN_FAILED_ONLY_QUERY_IDS.get(domain_key, ())) or None
		),
		library_source=_RUN_LIBRARY_SOURCE,
		runtime_backend=_RUN_RUNTIME_BACKEND,
		output_root=run_dir,
		run_id=run_dir.name,
		resume=resume,
	)
	summary_path = run_dir / f"{domain_key}.summary.json"
	summary_path.write_text(json.dumps(summary, indent=2))
	print(json.dumps(_compact_summary_for_console(summary), indent=2))
	return 0


def _run_full_benchmark(*, max_concurrent_domains: int = 1, run_id: str | None = None) -> int:
	if max_concurrent_domains < 1:
		raise ValueError("max_concurrent_domains must be at least 1")
	resolved_run_id = run_id or _timestamp()
	run_dir = RUNS_ROOT / resolved_run_id
	if not _RUN_RESUME and run_dir.exists() and any(run_dir.iterdir()):
		raise ValueError(
			f"Run directory already exists for run id '{resolved_run_id}'. "
			"Use --resume-run-id to continue it.",
		)
	run_dir.mkdir(parents=True, exist_ok=True)
	env = _build_env()
	launch_path = run_dir / "launch.json"
	domain_summaries = _load_existing_domain_summaries(run_dir) if _RUN_RESUME else {}
	pending_domains = [
		domain_key
		for domain_key in DOMAIN_KEYS
		if (
			domain_key not in domain_summaries
			or not bool(domain_summaries[domain_key].get("complete"))
		)
		and (
			not _RUN_FAILED_ONLY_QUERY_IDS
			or _RUN_FAILED_ONLY_QUERY_IDS.get(domain_key)
		)
	]
	active_runs: List[DomainRun] = []
	internal_failures: List[str] = []
	_write_run_summary_snapshot(
		run_dir=run_dir,
		run_id=resolved_run_id,
		domain_summaries=domain_summaries,
		internal_failures=internal_failures,
		max_concurrent_domains=max_concurrent_domains,
	)

	while pending_domains or active_runs:
		while pending_domains and len(active_runs) < max_concurrent_domains:
			domain_key = pending_domains.pop(0)
			active_runs.append(_start_domain_run(run_dir, domain_key, env))
			_record_launch_metadata(launch_path, active_runs)
		if not active_runs:
			continue
		completed_index = next(
			(
				index
				for index, candidate in enumerate(active_runs)
				if candidate.process.poll() is not None
			),
			None,
		)
		if completed_index is None:
			time.sleep(5)
			continue
		run = active_runs.pop(completed_index)
		_collect_domain_run_result(run, domain_summaries, internal_failures)
		_write_run_summary_snapshot(
			run_dir=run_dir,
			run_id=resolved_run_id,
			domain_summaries=domain_summaries,
			internal_failures=internal_failures,
			max_concurrent_domains=max_concurrent_domains,
		)
	summary = _write_run_summary_snapshot(
		run_dir=run_dir,
		run_id=resolved_run_id,
		domain_summaries=domain_summaries,
		internal_failures=internal_failures,
		max_concurrent_domains=max_concurrent_domains,
	)
	return 0 if summary["complete"] else 1


_RUN_QUERY_IDS: List[str] = []
_RUN_LIBRARY_SOURCE = "benchmark"
_RUN_RUNTIME_BACKEND = "jason"
_RUN_RESUME = False
_RUN_FAILED_ONLY_QUERY_IDS: Dict[str, List[str]] = {}


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--domain", choices=DOMAIN_KEYS)
	parser.add_argument("--run-dir")
	parser.add_argument("--run-id")
	parser.add_argument("--resume-run-id")
	parser.add_argument("--replay-failed-run-id")
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--query-id", action="append", default=[])
	parser.add_argument("--max-concurrent-domains", type=int, default=1)
	parser.add_argument(
		"--library-source",
		choices=("benchmark", "official", "generated"),
		default="benchmark",
	)
	parser.add_argument(
		"--runtime-backend",
		choices=("jason",),
		default="jason",
	)
	args = parser.parse_args()
	global _RUN_QUERY_IDS, _RUN_LIBRARY_SOURCE, _RUN_RUNTIME_BACKEND, _RUN_RESUME, _RUN_FAILED_ONLY_QUERY_IDS
	_RUN_QUERY_IDS = list(args.query_id or [])
	_RUN_LIBRARY_SOURCE = str(args.library_source)
	_RUN_RUNTIME_BACKEND = str(args.runtime_backend)
	_RUN_RESUME = bool(args.resume or args.resume_run_id)
	_RUN_FAILED_ONLY_QUERY_IDS = (
		_load_failed_query_ids_for_run(str(args.replay_failed_run_id).strip())
		if args.replay_failed_run_id
		else {}
	)
	resolved_run_id = str(args.resume_run_id or args.run_id or "").strip() or None

	if args.domain:
		if args.run_dir:
			run_dir = Path(args.run_dir).resolve()
		elif resolved_run_id:
			run_dir = (RUNS_ROOT / resolved_run_id).resolve()
		else:
			raise SystemExit("--run-dir or --run-id is required when --domain is set")
		run_dir.mkdir(parents=True, exist_ok=True)
		return _run_single_domain(args.domain, run_dir, resume=_RUN_RESUME)

	return _run_full_benchmark(
		max_concurrent_domains=args.max_concurrent_domains,
		run_id=resolved_run_id,
	)


if __name__ == "__main__":
	raise SystemExit(main())
