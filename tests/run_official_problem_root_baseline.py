from __future__ import annotations

import argparse
import atexit
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


def _pid_is_alive(pid: int) -> bool:
	try:
		os.kill(pid, 0)
	except OSError:
		return False
	return True


def _controller_pid_path(run_dir: Path) -> Path:
	return run_dir / "controller.pid"


def _controller_state_path(run_dir: Path) -> Path:
	return run_dir / "controller_state.json"


def _controller_log_path(run_dir: Path) -> Path:
	return run_dir / f"controller_{_timestamp()}.out"


def _build_controller_command(
	*,
	run_dir: Path,
	all_tracks: bool,
	domain: Optional[str],
	query_ids: Sequence[str],
	evaluation_mode: str,
	planner_id: Optional[str],
) -> List[str]:
	command = [
		sys.executable,
		"-u",
		str(Path(__file__).resolve()),
		"--run-dir",
		str(run_dir),
		"--evaluation-mode",
		evaluation_mode,
	]
	if all_tracks:
		command.append("--all-tracks")
	if domain:
		command.extend(["--domain", domain])
	if planner_id:
		command.extend(["--planner-id", planner_id])
	for query_id in query_ids:
		command.extend(["--query-id", query_id])
	return command


def _write_controller_state(
	run_dir: Path,
	*,
	pid: int,
	log_file: Path,
	command: Sequence[str],
	all_tracks: bool,
	domain: Optional[str],
	evaluation_mode: str,
	planner_id: Optional[str],
	query_ids: Sequence[str],
) -> Dict[str, Any]:
	state = {
		"pid": pid,
		"log_file": str(log_file),
		"command": list(command),
		"all_tracks": all_tracks,
		"domain": domain,
		"evaluation_mode": evaluation_mode,
		"planner_id": planner_id,
		"query_ids": list(query_ids),
		"run_dir": str(run_dir),
	}
	_controller_pid_path(run_dir).write_text(f"{pid}\n")
	_controller_state_path(run_dir).write_text(json.dumps(state, indent=2))
	return state


def _read_controller_state(run_dir: Path) -> Optional[Dict[str, Any]]:
	state_path = _controller_state_path(run_dir)
	if not state_path.exists():
		return None
	try:
		return dict(json.loads(state_path.read_text()))
	except Exception:
		return None


def _register_controller_runtime(run_dir: Path) -> None:
	run_dir.mkdir(parents=True, exist_ok=True)
	pid_path = _controller_pid_path(run_dir)
	state_path = _controller_state_path(run_dir)
	pid_path.write_text(f"{os.getpid()}\n")
	log_file_env = os.environ.get("HTN_EVAL_CONTROLLER_LOG_FILE")

	def _cleanup_controller_files() -> None:
		try:
			if pid_path.exists():
				pid_path.unlink()
		except OSError:
			pass
		try:
			state = _read_controller_state(run_dir)
			if state is not None:
				state["last_exit_pid"] = os.getpid()
				state["active"] = False
				state_path.write_text(json.dumps(state, indent=2))
		except Exception:
			pass

	def _handle_sigterm(_signum: int, _frame: Any) -> None:
		try:
			state = _read_controller_state(run_dir) or {}
			state["last_signal"] = "SIGTERM"
			state["sigterm_ignored_count"] = int(state.get("sigterm_ignored_count") or 0) + 1
			state["active"] = True
			state_path.write_text(json.dumps(state, indent=2))
		except Exception:
			pass
		message = (
			"[CONTROLLER] Ignored SIGTERM to keep detached HTN sweep alive."
			f" count={json.loads(state_path.read_text()).get('sigterm_ignored_count', 1) if state_path.exists() else 1}\n"
		)
		try:
			if log_file_env:
				with Path(log_file_env).open("a") as handle:
					handle.write(message)
			else:
				sys.stderr.write(message)
				sys.stderr.flush()
		except Exception:
			pass

	atexit.register(_cleanup_controller_files)
	if os.environ.get("HTN_EVAL_IGNORE_SIGTERM") == "1":
		signal.signal(signal.SIGTERM, _handle_sigterm)


def _launch_detached_controller(
	*,
	run_dir: Path,
	all_tracks: bool,
	domain: Optional[str],
	query_ids: Sequence[str],
	evaluation_mode: str,
	planner_id: Optional[str],
) -> Dict[str, Any]:
	run_dir.mkdir(parents=True, exist_ok=True)
	existing_state = _read_controller_state(run_dir)
	if existing_state is not None:
		try:
			existing_pid = int(existing_state.get("pid") or 0)
		except (TypeError, ValueError):
			existing_pid = 0
		if existing_pid > 0 and _pid_is_alive(existing_pid):
			return {
				"status": "already_running",
				**existing_state,
			}

	command = _build_controller_command(
		run_dir=run_dir,
		all_tracks=all_tracks,
		domain=domain,
		query_ids=query_ids,
		evaluation_mode=evaluation_mode,
		planner_id=planner_id,
	)
	log_file = _controller_log_path(run_dir)
	env = dict(os.environ)
	required_paths = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
	existing_pythonpath = env.get("PYTHONPATH", "")
	pythonpath_parts = [part for part in existing_pythonpath.split(os.pathsep) if part]
	for required_path in reversed(required_paths):
		if required_path not in pythonpath_parts:
			pythonpath_parts.insert(0, required_path)
	env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
	env["HTN_EVAL_IGNORE_SIGTERM"] = "1"
	env["HTN_EVAL_CONTROLLER_LOG_FILE"] = str(log_file)
	with log_file.open("a") as log_handle:
		process = subprocess.Popen(
			command,
			cwd=PROJECT_ROOT,
			env=env,
			stdin=subprocess.DEVNULL,
			stdout=log_handle,
			stderr=log_handle,
			start_new_session=True,
			close_fds=True,
		)
	state = _write_controller_state(
		run_dir,
		pid=process.pid,
		log_file=log_file,
		command=command,
		all_tracks=all_tracks,
		domain=domain,
		evaluation_mode=evaluation_mode,
		planner_id=planner_id,
		query_ids=query_ids,
	)
	state["status"] = "launched"
	return state


def _run_single_domain(domain_key: str, run_dir: Path) -> int:
	from tests.support.htn_evaluation_support import (
		load_domain_query_cases,
		run_official_problem_root_baseline_for_domain,
	)

	output_root = run_dir / domain_key
	summary = run_official_problem_root_baseline_for_domain(
		domain_key,
		query_ids=tuple(_RUN_QUERY_IDS) if _RUN_QUERY_IDS else None,
		evaluation_mode=_RUN_EVALUATION_MODE,
		planner_id=_RUN_PLANNER_ID,
		output_root=output_root,
	)
	total_query_count = len(load_domain_query_cases(domain_key))
	if int(summary.get("attempted_problem_count") or 0) >= total_query_count:
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
			"pid=,ppid=,rss=,command=",
		],
		capture_output=True,
		text=True,
		check=False,
	)
	current_pid = os.getpid()
	for line in result.stdout.splitlines():
		parts = line.strip().split(None, 3)
		if len(parts) < 4:
			continue
		pid_text, _ppid_text, rss_text, command = parts
		try:
			pid = int(pid_text)
		except ValueError:
			continue
		if pid == current_pid:
			continue
		try:
			rss = int(rss_text)
		except ValueError:
			rss = 0
		if rss <= 0:
			continue
		if str(PROJECT_ROOT) not in command:
			continue
		if not any(
			marker in command
			for marker in (
				"pandaPIengine",
				"spawn_main",
				"run_official_problem_root_baseline.py",
			)
		):
			continue
		try:
			os.kill(pid, signal.SIGKILL)
		except OSError:
			continue


def _build_track_pass_rows(
	track_summaries: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	for track_id, summary in track_summaries.items():
		track_summary = dict(summary.get("track_summary") or {})
		total_query_count = int(track_summary.get("total_queries") or 0)
		verified_success_count = int(track_summary.get("verified_success_count") or 0)
		sweep_complete = bool(summary.get("complete"))
		all_queries_verified = sweep_complete and total_query_count > 0 and (
			verified_success_count == total_query_count
		)
		rows.append(
			{
				"track_id": track_id,
				"evaluation_mode": summary.get("evaluation_mode"),
				"requested_planner_id": summary.get("requested_planner_id"),
				"complete": sweep_complete,
				"sweep_complete": sweep_complete,
				"all_queries_verified": all_queries_verified,
				"pass": all_queries_verified,
				"completed_domain_count": len(summary.get("completed_domains") or []),
				"total_query_count": total_query_count,
				"verified_success_count": verified_success_count,
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
		"sweep_complete",
		"all_queries_verified",
		"pass",
		"completed_domain_count",
		"total_query_count",
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
	_cleanup_htn_evaluation_resources()
	runner = domain_runner or (
		lambda domain_key, mode, normalized_planner_id: (
			run_official_problem_root_baseline_for_domain(
				domain_key,
				evaluation_mode=mode,
				planner_id=normalized_planner_id,
				output_root=run_dir / domain_key,
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
		_cleanup_htn_evaluation_resources()
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
			_cleanup_htn_evaluation_resources()
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
		_cleanup_htn_evaluation_resources()
	_cleanup_htn_evaluation_resources()
	return _write_track_outputs(
		run_dir=run_dir,
		domain_summaries=domain_summaries,
		evaluation_mode=evaluation_mode,
		planner_id=planner_id,
		track_id=track_id,
		internal_failures=internal_failures,
	)


_RUN_QUERY_IDS: List[str] = []
_RUN_EVALUATION_MODE = "single_planner"
_RUN_PLANNER_ID: str | None = "lifted_panda_sat"


def _track_specs() -> List[Dict[str, str | None]]:
	from htn_evaluation.result_tables import HTN_PLANNER_IDS, SINGLE_PLANNER_MODE

	specs: List[Dict[str, str | None]] = []
	for planner_id in HTN_PLANNER_IDS:
		specs.append(
			{
				"track_id": planner_id,
				"evaluation_mode": SINGLE_PLANNER_MODE,
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
		PRIMARY_HTN_PLANNER_ID,
		SINGLE_PLANNER_MODE,
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
		default=SINGLE_PLANNER_MODE,
		choices=(SINGLE_PLANNER_MODE,),
	)
	parser.add_argument("--planner-id", default=PRIMARY_HTN_PLANNER_ID)
	parser.add_argument("--all-tracks", action="store_true")
	parser.add_argument("--launch-detached", action="store_true")
	args = parser.parse_args()
	global _RUN_EVALUATION_MODE, _RUN_PLANNER_ID, _RUN_QUERY_IDS
	_RUN_QUERY_IDS = list(args.query_id or [])
	_RUN_EVALUATION_MODE = validate_evaluation_mode(args.evaluation_mode)
	_RUN_PLANNER_ID = validate_planner_id(
		args.planner_id,
		evaluation_mode=_RUN_EVALUATION_MODE,
	)

	if args.launch_detached:
		if args.domain:
			if not args.run_dir:
				raise SystemExit("--run-dir is required when --domain is set")
			detached_run_dir = Path(args.run_dir).resolve()
		elif args.all_tracks:
			detached_run_dir = Path(args.run_dir).resolve() if args.run_dir else (RUNS_ROOT / _timestamp())
		else:
			detached_run_dir = Path(args.run_dir).resolve() if args.run_dir else (RUNS_ROOT / _timestamp())
		state = _launch_detached_controller(
			run_dir=detached_run_dir,
			all_tracks=bool(args.all_tracks),
			domain=args.domain,
			query_ids=tuple(_RUN_QUERY_IDS),
			evaluation_mode=_RUN_EVALUATION_MODE,
			planner_id=_RUN_PLANNER_ID,
		)
		print(json.dumps(state, indent=2))
		return 0

	if args.all_tracks:
		root_run_dir = Path(args.run_dir).resolve() if args.run_dir else None
		if root_run_dir is not None:
			_register_controller_runtime(root_run_dir)
		return _run_all_tracks(run_dir=root_run_dir)

	if args.domain:
		if not args.run_dir:
			raise SystemExit("--run-dir is required when --domain is set")
		run_dir = Path(args.run_dir).resolve()
		run_dir.mkdir(parents=True, exist_ok=True)
		_register_controller_runtime(run_dir)
		return _run_single_domain(args.domain, run_dir)

	run_dir = Path(args.run_dir).resolve() if args.run_dir else (RUNS_ROOT / _timestamp())
	_register_controller_runtime(run_dir)
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
