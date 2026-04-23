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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
RUNS_ROOT = PROJECT_ROOT / "tests" / "method_library" / "generated" / "domain_build_sweeps"
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
	return {
		"run_dir": str(run_dir),
		"domain_build_pass": sum(
			1
			for summary in domain_summaries.values()
			if bool((summary.get("domain_build") or {}).get("success"))
		),
		"domain_build_fail": sum(
			1
			for summary in domain_summaries.values()
			if not bool((summary.get("domain_build") or {}).get("success"))
		),
		"domain_gate_pass": sum(
			1
			for summary in domain_summaries.values()
			if bool((summary.get("domain_gate_preflight") or {}).get("success"))
		),
		"domain_gate_fail": sum(
			1
			for summary in domain_summaries.values()
			if not bool((summary.get("domain_gate_preflight") or {}).get("success"))
		),
		"llm_generation_attempts_total": sum(
			int(summary.get("llm_generation_attempts_total", 0))
			for summary in domain_summaries.values()
		),
		"generated_method_count_total": sum(
			int((summary.get("domain_build") or {}).get("generated_method_count", 0))
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
		f"generated_method_count_total: {summary['generated_method_count_total']}",
	]
	for domain_key, domain_summary in dict(summary.get("domains") or {}).items():
		domain_build = dict(domain_summary.get("domain_build") or {})
		domain_gate = dict(domain_summary.get("domain_gate_preflight") or {})
		lines.append(
			f"{domain_key}: build_success={bool(domain_build.get('success'))}, "
			f"gate_success={bool(domain_gate.get('success'))}, "
			f"llm_generation_attempts={int(domain_summary.get('llm_generation_attempts_total', 0))}, "
			f"stream_handshake={domain_build.get('llm_stream_handshake_seconds')}, "
			f"first_stream_chunk={domain_build.get('llm_first_stream_chunk_seconds')}, "
			f"first_content_chunk={domain_build.get('llm_first_content_chunk_seconds')}, "
			f"complete_json={domain_build.get('llm_complete_json_seconds')}, "
			f"methods={domain_build.get('generated_method_count')}",
		)
	(run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
	print("\n".join(lines))


def _run_single_domain(domain_key: str, run_dir: Path) -> int:
	from tests.support.plan_library_generation_support import run_generated_domain_build

	report = run_generated_domain_build(domain_key)
	execution = dict(report.get("execution") or {})
	method_synthesis = dict(execution.get("method_synthesis") or {})
	domain_gate = dict(execution.get("domain_gate") or {})
	summary = {
		"domain_key": domain_key,
		"domain_build": {
			"success": bool(report.get("success")),
			"log_dir": str(report.get("log_dir") or ""),
			"artifact_root": str(report.get("artifact_root") or ""),
			"source_domain_kind": str(report.get("source_domain_kind") or ""),
			"generated_domain_file": str(((report.get("artifact_paths") or {}).get("generated_domain")) or ""),
			"llm_attempted": bool(report.get("llm_attempted")),
			"llm_generation_attempts": int(report.get("llm_generation_attempts") or 0),
			"llm_attempts": int(report.get("llm_attempts") or 0),
			"llm_request_id": str(report.get("llm_request_id") or ""),
			"llm_response_mode": str(report.get("llm_response_mode") or ""),
			"llm_stream_handshake_seconds": report.get("llm_stream_handshake_seconds"),
			"llm_first_stream_chunk_seconds": report.get("llm_first_stream_chunk_seconds"),
			"llm_first_chunk_seconds": report.get("llm_first_chunk_seconds"),
			"llm_first_content_chunk_seconds": report.get("llm_first_content_chunk_seconds"),
			"llm_complete_json_seconds": report.get("llm_complete_json_seconds"),
			"method_synthesis_model": str(report.get("method_synthesis_model") or ""),
			"generated_method_count": int(report.get("generated_method_count") or 0),
			"method_status": method_synthesis.get("status"),
			"method_error": method_synthesis.get("error"),
		},
		"domain_gate_preflight": {
			"success": str(domain_gate.get("status") or "").lower() == "success",
			"status": domain_gate.get("status"),
			"error": domain_gate.get("error"),
			"validated_task_count": int((domain_gate.get("artifacts") or {}).get("validated_task_count") or 0)
			if isinstance(domain_gate.get("artifacts"), dict)
			else 0,
		},
		"llm_generation_attempts_total": int(report.get("llm_generation_attempts") or 0),
	}
	(run_dir / f"{domain_key}.summary.json").write_text(json.dumps(summary, indent=2))
	print(json.dumps(summary, indent=2))
	return 0 if summary["domain_build"]["success"] else 1


def _run_full_sweep(max_concurrent_domains: int) -> int:
	run_dir = RUNS_ROOT / _timestamp()
	run_dir.mkdir(parents=True, exist_ok=True)
	env = _build_env()
	pending = list(DOMAIN_KEYS)
	active_runs: List[DomainRun] = []
	domain_summaries: Dict[str, Dict[str, object]] = {}
	internal_failures: List[str] = []

	while pending or active_runs:
		while pending and len(active_runs) < max(1, max_concurrent_domains):
			domain_key = pending.pop(0)
			active_runs.append(_start_domain_run(run_dir, domain_key, env))
		_record_launch_metadata(run_dir / "launches.json", active_runs)

		still_active: List[DomainRun] = []
		for run in active_runs:
			return_code = run.process.poll()
			if return_code is None:
				still_active.append(run)
				continue
			_collect_domain_run_result(run, domain_summaries, internal_failures)
		active_runs = still_active
		if active_runs:
			time.sleep(1.0)

	for failed_domain in internal_failures:
		domain_summaries.setdefault(
			failed_domain,
			{
				"domain_key": failed_domain,
				"domain_build": {"success": False},
				"domain_gate_preflight": {"success": False},
				"llm_generation_attempts_total": 0,
			},
		)

	summary = _aggregate_domain_summaries(run_dir, domain_summaries)
	(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
	_write_human_summary(run_dir, summary)
	return 0 if not internal_failures else 1


def main(argv: List[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--domain", choices=DOMAIN_KEYS)
	parser.add_argument("--run-dir", default="")
	parser.add_argument(
		"--max-concurrent-domains",
		type=int,
		default=len(DOMAIN_KEYS),
	)
	args = parser.parse_args(argv)

	if args.domain:
		run_dir = Path(args.run_dir) if args.run_dir else RUNS_ROOT / _timestamp()
		run_dir.mkdir(parents=True, exist_ok=True)
		return _run_single_domain(args.domain, run_dir)

	return _run_full_sweep(max_concurrent_domains=args.max_concurrent_domains)


if __name__ == "__main__":
	raise SystemExit(main())
