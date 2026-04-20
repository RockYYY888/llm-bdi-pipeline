from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional


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
	domain_summaries: Dict[str, Mapping[str, Any]] = {}
	internal_failures: List[Mapping[str, str]] = []
	for domain_key in DOMAIN_KEYS:
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


def _run_all_tracks() -> int:
	run_dir = RUNS_ROOT / _timestamp()
	run_dir.mkdir(parents=True, exist_ok=True)
	track_summaries: Dict[str, Dict[str, Any]] = {}
	for spec in _track_specs():
		track_id = str(spec["track_id"])
		track_dir = run_dir / track_id
		track_summary = _run_sequential_full_baseline(
			run_dir=track_dir,
			evaluation_mode=str(spec["evaluation_mode"]),
			planner_id=spec["planner_id"],
			track_id=track_id,
		)
		track_summaries[track_id] = track_summary
	combined_summary = {
		"run_dir": str(run_dir),
		"tracks": track_summaries,
		"complete": all(
			bool(track_summary.get("complete"))
			for track_summary in track_summaries.values()
		),
	}
	(run_dir / "summary.json").write_text(json.dumps(combined_summary, indent=2))
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
		return _run_all_tracks()

	if args.domain:
		if not args.run_dir:
			raise SystemExit("--run-dir is required when --domain is set")
		run_dir = Path(args.run_dir).resolve()
		run_dir.mkdir(parents=True, exist_ok=True)
		return _run_single_domain(args.domain, run_dir)

	run_dir = RUNS_ROOT / _timestamp()
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
