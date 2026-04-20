from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
RUNS_ROOT = PROJECT_ROOT / "tests" / "generated" / "official_ground_truth_full"
DOMAIN_KEYS = ("blocksworld", "marsrover", "satellite", "transport")


def _timestamp() -> str:
	return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _run_single_domain(domain_key: str, run_dir: Path) -> int:
	from tests.support.htn_evaluation_support import (
		run_official_problem_root_baseline_for_domain,
	)

	summary = run_official_problem_root_baseline_for_domain(
		domain_key,
		query_ids=tuple(_RUN_QUERY_IDS) if _RUN_QUERY_IDS else None,
		evaluation_mode=_RUN_EVALUATION_MODE,
		planner_id=_RUN_PLANNER_ID,
	)
	summary_path = run_dir / f"{domain_key}.summary.json"
	summary_path.write_text(json.dumps(summary, indent=2))
	print(json.dumps(summary, indent=2))
	return 0


def _write_text_summary(run_dir: Path, summary: Mapping[str, Any]) -> None:
	track_summary = dict(summary.get("track_summary") or {})
	lines = [
		f"run_dir: {summary['run_dir']}",
		f"track_id: {summary['track_id']}",
		f"evaluation_mode: {summary['evaluation_mode']}",
		f"requested_planner_id: {summary['requested_planner_id']}",
		f"complete: {summary['complete']}",
		f"completed_domains: {', '.join(summary.get('completed_domains') or [])}",
		f"attempted_problem_count: {track_summary.get('attempted_problem_count', 0)}",
		f"verified_success_count: {track_summary.get('verified_success_count', 0)}",
	]
	for domain_key, domain_summary in dict(track_summary.get("domains") or {}).items():
		lines.append(
			f"{domain_key}: attempted={domain_summary.get('attempted_problem_count', 0)}, "
			f"verified={domain_summary.get('verified_success_count', 0)}, "
			f"no_plan={domain_summary.get('solver_no_plan_failures', 0)}, "
			f"primitive_invalid={domain_summary.get('primitive_invalid_failures', 0)}, "
			f"hierarchical_rejected={domain_summary.get('hierarchical_rejection_failures', 0)}",
		)
	(run_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def _load_existing_domain_summaries(run_dir: Path) -> Dict[str, Mapping[str, Any]]:
	domain_summaries: Dict[str, Mapping[str, Any]] = {}
	for domain_key in DOMAIN_KEYS:
		summary_path = run_dir / f"{domain_key}.summary.json"
		if summary_path.exists():
			domain_summaries[domain_key] = json.loads(summary_path.read_text())
	return domain_summaries


def _write_track_outputs(
	*,
	run_dir: Path,
	domain_summaries: Mapping[str, Mapping[str, Any]],
	evaluation_mode: str,
	planner_id: Optional[str],
	track_id: str,
	internal_failures: Optional[List[Mapping[str, str]]] = None,
) -> Dict[str, Any]:
	from htn_evaluation.result_tables import (
		build_planner_capability_rows,
		build_problem_capability_rows,
		build_track_summary,
		write_planner_capability_matrix,
		write_problem_capability_matrix,
	)

	track_summary = build_track_summary(
		run_dir=run_dir,
		domain_summaries=dict(domain_summaries),
		evaluation_mode=evaluation_mode,
		planner_id=planner_id,
	)
	track_summary["track_id"] = track_id
	track_summary["completed_domains"] = sorted(domain_summaries)
	track_summary["complete"] = len(domain_summaries) == len(DOMAIN_KEYS)
	(run_dir / "track_summary.json").write_text(json.dumps(track_summary, indent=2))

	planner_paths = write_planner_capability_matrix(
		run_dir,
		rows=build_planner_capability_rows((track_summary,)),
	)
	problem_paths = write_problem_capability_matrix(
		run_dir,
		rows=build_problem_capability_rows((track_summary,)),
	)
	summary = {
		"run_dir": str(run_dir),
		"track_id": track_id,
		"evaluation_mode": evaluation_mode,
		"requested_planner_id": planner_id,
		"completed_domains": sorted(domain_summaries),
		"complete": (
			len(domain_summaries) == len(DOMAIN_KEYS)
			and not list(internal_failures or [])
		),
		"internal_failures": list(internal_failures or []),
		"track_summary": track_summary,
		"output_paths": {
			**planner_paths,
			**problem_paths,
		},
	}
	(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
	_write_text_summary(run_dir, summary)
	return summary


def _cleanup_htn_evaluation_resources() -> None:
	result = subprocess.run(
		[
			"ps",
			"-Ao",
			"pid=,ppid=,command=",
		],
		capture_output=True,
		text=True,
		check=False,
	)
	for line in result.stdout.splitlines():
		parts = line.strip().split(None, 2)
		if len(parts) < 3:
			continue
		pid_text, ppid_text, command = parts
		if ppid_text != "1":
			continue
		if str(PROJECT_ROOT) not in command:
			continue
		if "run_official_problem_root_baseline.py" in command:
			continue
		if not any(
			marker in command
			for marker in (
				"pandaPIengine",
				"pandadealer",
				"spawn_main",
			)
		):
			continue
		try:
			os.kill(int(pid_text), signal.SIGKILL)
		except (OSError, ValueError):
			continue


def _build_track_pass_rows(
	track_summaries: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	for track_id, summary in track_summaries.items():
		track_summary = dict(summary.get("track_summary") or {})
		rows.append(
			{
				"track_id": track_id,
				"evaluation_mode": summary.get("evaluation_mode"),
				"requested_planner_id": summary.get("requested_planner_id"),
				"complete": bool(summary.get("complete")),
				"pass": bool(summary.get("complete")),
				"completed_domain_count": len(summary.get("completed_domains") or []),
				"verified_success_count": track_summary.get("verified_success_count", 0),
			},
		)
	return rows


def _write_track_pass_matrix(
	run_dir: Path,
	track_summaries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
	run_dir.mkdir(parents=True, exist_ok=True)
	rows = _build_track_pass_rows(track_summaries)
	json_path = run_dir / "track_pass_matrix.json"
	csv_path = run_dir / "track_pass_matrix.csv"
	json_path.write_text(json.dumps(rows, indent=2))
	fieldnames = [
		"track_id",
		"evaluation_mode",
		"requested_planner_id",
		"complete",
		"pass",
		"completed_domain_count",
		"verified_success_count",
	]
	with csv_path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow({field: row.get(field) for field in fieldnames})
	return {
		"track_pass_matrix_json": str(json_path),
		"track_pass_matrix_csv": str(csv_path),
	}


def _write_all_tracks_state(
	run_dir: Path,
	track_summaries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
	state = {
		"run_dir": str(run_dir),
		"completed_tracks": [
			track_id
			for track_id, summary in track_summaries.items()
			if bool(summary.get("complete"))
		],
		"tracks": dict(track_summaries),
	}
	(run_dir / "all_tracks_state.json").write_text(json.dumps(state, indent=2))
	return state


def _run_sequential_full_baseline(
	*,
	run_dir: Path,
	evaluation_mode: str,
	planner_id: Optional[str],
	track_id: str,
	domain_runner: Optional[
		Callable[[str, str, Optional[str]], Mapping[str, Any]]
	] = None,
) -> Dict[str, Any]:
	from tests.support.htn_evaluation_support import (
		run_official_problem_root_baseline_for_domain,
	)

	run_dir.mkdir(parents=True, exist_ok=True)
	runner = domain_runner or (
		lambda domain_key, mode, normalized_planner_id: (
			run_official_problem_root_baseline_for_domain(
				domain_key,
				evaluation_mode=mode,
				planner_id=normalized_planner_id,
			)
		)
	)
	domain_summaries = _load_existing_domain_summaries(run_dir)
	internal_failures: List[Mapping[str, str]] = []
	if domain_summaries:
		_write_track_outputs(
			run_dir=run_dir,
			domain_summaries=domain_summaries,
			evaluation_mode=evaluation_mode,
			planner_id=planner_id,
			track_id=track_id,
			internal_failures=internal_failures,
		)
	for domain_key in DOMAIN_KEYS:
		if domain_key in domain_summaries:
			print(f"[TRACK] resume skip domain={domain_key}", flush=True)
			continue
		print(f"[TRACK] starting domain={domain_key}", flush=True)
		try:
			domain_summary = runner(domain_key, evaluation_mode, planner_id)
		except Exception:
			internal_failures.append(
				{
					"domain_key": domain_key,
					"traceback": traceback.format_exc(),
				},
			)
			break
		domain_summaries[domain_key] = domain_summary
		(run_dir / f"{domain_key}.summary.json").write_text(
			json.dumps(dict(domain_summary), indent=2),
		)
		_write_track_outputs(
			run_dir=run_dir,
			domain_summaries=domain_summaries,
			evaluation_mode=evaluation_mode,
			planner_id=planner_id,
			track_id=track_id,
			internal_failures=internal_failures,
		)
		print(f"[TRACK] finished domain={domain_key}", flush=True)
	return _write_track_outputs(
		run_dir=run_dir,
		domain_summaries=domain_summaries,
		evaluation_mode=evaluation_mode,
		planner_id=planner_id,
		track_id=track_id,
		internal_failures=internal_failures,
	)


_RUN_QUERY_IDS: List[str] = []
_RUN_EVALUATION_MODE = "planner_or_race"
_RUN_PLANNER_ID: str | None = None


def _track_specs() -> List[Dict[str, str | None]]:
	from htn_evaluation.result_tables import HTN_PLANNER_IDS, PLANNER_OR_RACE_MODE

	specs: List[Dict[str, str | None]] = [
		{
			"track_id": PLANNER_OR_RACE_MODE,
			"evaluation_mode": PLANNER_OR_RACE_MODE,
			"planner_id": None,
		},
	]
	for planner_id in HTN_PLANNER_IDS:
		specs.append(
			{
				"track_id": planner_id,
				"evaluation_mode": "single_planner",
				"planner_id": planner_id,
			},
		)
	return specs


def _run_all_tracks(run_dir: Optional[Path] = None) -> int:
	root_run_dir = run_dir or (RUNS_ROOT / _timestamp())
	root_run_dir.mkdir(parents=True, exist_ok=True)
	track_summaries: Dict[str, Dict[str, Any]] = {}
	for spec in _track_specs():
		track_id = str(spec["track_id"])
		track_dir = root_run_dir / track_id
		existing_summary_path = track_dir / "summary.json"
		if existing_summary_path.exists():
			existing_summary = json.loads(existing_summary_path.read_text())
			track_summaries[track_id] = existing_summary
			if bool(existing_summary.get("complete")):
				_write_track_pass_matrix(root_run_dir, track_summaries)
				_write_all_tracks_state(root_run_dir, track_summaries)
				print(f"[ALL_TRACKS] resume skip track={track_id}", flush=True)
				continue

		thread_result: Dict[str, Dict[str, Any]] = {}

		def _thread_target() -> None:
			thread_result["summary"] = _run_sequential_full_baseline(
				run_dir=track_dir,
				evaluation_mode=str(spec["evaluation_mode"]),
				planner_id=spec["planner_id"],
				track_id=track_id,
			)

		track_thread = threading.Thread(
			target=_thread_target,
			name=f"track-{track_id}",
			daemon=False,
		)
		track_thread.start()
		track_thread.join()
		track_summary = thread_result["summary"]
		track_summaries[track_id] = track_summary
		_cleanup_htn_evaluation_resources()
		_write_track_pass_matrix(root_run_dir, track_summaries)
		_write_all_tracks_state(root_run_dir, track_summaries)

	combined_summary = {
		"run_dir": str(root_run_dir),
		"tracks": track_summaries,
		"complete": all(
			bool(track_summary.get("complete"))
			for track_summary in track_summaries.values()
		),
	}
	(root_run_dir / "summary.json").write_text(json.dumps(combined_summary, indent=2))
	print(json.dumps(combined_summary, indent=2))
	return 0 if combined_summary["complete"] else 1


def main() -> int:
	from htn_evaluation.result_tables import (
		PLANNER_OR_RACE_MODE,
		planner_track_id,
		validate_evaluation_mode,
		validate_planner_id,
	)

	parser = argparse.ArgumentParser()
	parser.add_argument("--domain", choices=DOMAIN_KEYS)
	parser.add_argument("--run-dir")
	parser.add_argument("--query-id", action="append", default=[])
	parser.add_argument(
		"--evaluation-mode",
		default=PLANNER_OR_RACE_MODE,
		choices=("planner_or_race", "single_planner"),
	)
	parser.add_argument("--planner-id")
	parser.add_argument("--all-tracks", action="store_true")
	args = parser.parse_args()
	global _RUN_EVALUATION_MODE, _RUN_PLANNER_ID, _RUN_QUERY_IDS
	_RUN_QUERY_IDS = list(args.query_id or [])
	_RUN_EVALUATION_MODE = validate_evaluation_mode(args.evaluation_mode)
	_RUN_PLANNER_ID = validate_planner_id(
		args.planner_id,
		evaluation_mode=_RUN_EVALUATION_MODE,
	)

	if args.all_tracks:
		root_run_dir = Path(args.run_dir).resolve() if args.run_dir else None
		return _run_all_tracks(run_dir=root_run_dir)

	if args.domain:
		if not args.run_dir:
			raise SystemExit("--run-dir is required when --domain is set")
		run_dir = Path(args.run_dir).resolve()
		run_dir.mkdir(parents=True, exist_ok=True)
		return _run_single_domain(args.domain, run_dir)

	run_dir = Path(args.run_dir).resolve() if args.run_dir else (RUNS_ROOT / _timestamp())
	track_id = planner_track_id(
		evaluation_mode=_RUN_EVALUATION_MODE,
		planner_id=_RUN_PLANNER_ID,
	)
	summary = _run_sequential_full_baseline(
		run_dir=run_dir,
		evaluation_mode=_RUN_EVALUATION_MODE,
		planner_id=_RUN_PLANNER_ID,
		track_id=track_id,
	)
	return 0 if summary["complete"] else 1


if __name__ == "__main__":
	raise SystemExit(main())
