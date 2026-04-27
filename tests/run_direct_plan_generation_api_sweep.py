from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) in sys.path:
	sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from domain_model import load_query_sequence_records
from evaluation.direct_plan_baseline import (
	DirectPlanGenerator,
	build_direct_plan_system_prompt,
	build_direct_plan_user_prompt,
	run_direct_plan_baseline_case,
)
from tests.support.plan_library_evaluation_support import (
	DOMAIN_FILES,
	load_domain_query_cases,
	query_id_sort_key,
)
from utils.hddl_parser import HDDLParser


RUNS_ROOT = PROJECT_ROOT / "tests" / "generated" / "direct_plan_generation_api_sweep"
DOMAIN_KEYS = ("blocksworld", "marsrover", "satellite", "transport")


def _timestamp() -> str:
	return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _load_temporal_specifications(domain_key: str, query_ids: Sequence[str]) -> Dict[str, Any]:
	_, records = load_query_sequence_records(
		domain_file=DOMAIN_FILES[domain_key],
		query_domain=domain_key,
		query_ids=query_ids,
	)
	return {record.instruction_id: record for record in records}


def _build_prompt_payload(domain_key: str, query_id: str) -> Dict[str, Any]:
	query_cases = load_domain_query_cases(domain_key)
	case = query_cases[query_id]
	temporal_specification = _load_temporal_specifications(domain_key, (query_id,))[query_id]
	domain = HDDLParser.parse_domain(DOMAIN_FILES[domain_key])
	problem = HDDLParser.parse_problem(str(case["problem_file"]))
	system_prompt = build_direct_plan_system_prompt()
	user_prompt = build_direct_plan_user_prompt(
		domain=domain,
		problem=problem,
		temporal_specification=temporal_specification,
		instruction=str(case["instruction"]),
	)
	return {
		"case": case,
		"system_prompt": system_prompt,
		"user_prompt": user_prompt,
		"temporal_specification": temporal_specification,
	}


def _run_case(
	*,
	run_root: Path,
	domain_key: str,
	query_id: str,
	generator: DirectPlanGenerator,
	verify: bool,
) -> Dict[str, Any]:
	prompt_payload = _build_prompt_payload(domain_key, query_id)
	case = prompt_payload["case"]
	try:
		result = run_direct_plan_baseline_case(
			domain_key=domain_key,
			query_id=query_id,
			domain_file=DOMAIN_FILES[domain_key],
			problem_file=case["problem_file"],
			instruction=str(case["instruction"]),
			temporal_specification=prompt_payload["temporal_specification"],
			output_dir=run_root / domain_key / "query_results" / query_id,
			generator=generator,
			verify=verify,
			system_prompt_override=str(prompt_payload["system_prompt"]),
			user_prompt_override=str(prompt_payload["user_prompt"]),
		)
	except Exception as exc:
		return _record_api_failure(
			run_root=run_root,
			domain_key=domain_key,
			query_id=query_id,
			prompt_payload=prompt_payload,
			error=exc,
			verify=verify,
		)
	return {
		**result.to_dict(),
		"api_failed": False,
	}


def _record_api_failure(
	*,
	run_root: Path,
	domain_key: str,
	query_id: str,
	prompt_payload: Dict[str, Any],
	error: Exception,
	verify: bool,
) -> Dict[str, Any]:
	case = prompt_payload["case"]
	output_dir = run_root / domain_key / "query_results" / query_id
	output_dir.mkdir(parents=True, exist_ok=True)
	prompt_file = output_dir / "prompt.json"
	raw_response_file = output_dir / "response.json"
	plan_file = output_dir / "plan.txt"
	validation_file = output_dir / "direct_plan_validation.json"
	prompt_file.write_text(
		json.dumps(
			{
				"domain_key": domain_key,
				"query_id": query_id,
				"domain_file": str(Path(DOMAIN_FILES[domain_key]).resolve()),
				"problem_file": str(Path(str(case["problem_file"])).resolve()),
				"system": str(prompt_payload["system_prompt"]),
				"user": str(prompt_payload["user_prompt"]),
			},
			indent=2,
		),
	)
	raw_response_file.write_text(
		json.dumps(
			{
				"response_text": "",
				"language_model": {
					"source": "api",
					"status": "failed",
					"error": str(error),
				},
			},
			indent=2,
		),
	)
	plan_file.write_text("")
	payload = {
		"domain_key": domain_key,
		"query_id": query_id,
		"problem_file": str(Path(str(case["problem_file"])).resolve()),
		"output_dir": str(output_dir.resolve()),
		"prompt_file": str(prompt_file.resolve()),
		"raw_response_file": str(raw_response_file.resolve()),
		"plan_file": str(plan_file.resolve()),
		"validation_file": str(validation_file.resolve()),
		"parseable": False,
		"executable": False,
		"goal_reached": False,
		"success": False,
		"diagnostics": [],
		"verification_skipped": not verify,
		"error": str(error),
		"api_failed": True,
	}
	validation_file.write_text(json.dumps(payload, indent=2))
	return payload


def _validation_path(run_root: Path, domain_key: str, query_id: str) -> Path:
	return run_root / domain_key / "query_results" / query_id / "direct_plan_validation.json"


def _write_summary(
	*,
	run_root: Path,
	results_by_domain: Dict[str, list[Dict[str, Any]]],
	selected_by_domain: Dict[str, Sequence[str]],
	resumed: Dict[str, list[str]],
) -> None:
	domain_summaries = {}
	for domain_key, selected_query_ids in selected_by_domain.items():
		results = results_by_domain.get(domain_key, [])
		completed = [str(result.get("query_id") or "") for result in results]
		summary = {
			"run_root": str(run_root.resolve()),
			"domain_key": domain_key,
			"baseline": "direct_plan_generation_api",
			"total_queries": len(selected_query_ids),
			"selected_query_ids": list(selected_query_ids),
			"completed_query_ids": completed,
			"remaining_query_ids": [
				query_id for query_id in selected_query_ids if query_id not in set(completed)
			],
			"resumed_query_ids": list(resumed.get(domain_key, [])),
			"parseable": sum(1 for result in results if result.get("parseable") is True),
			"parse_failures": sum(1 for result in results if result.get("parseable") is not True),
			"api_failures": sum(1 for result in results if result.get("api_failed") is True),
			"executable": sum(1 for result in results if result.get("executable") is True),
			"goal_reached": sum(1 for result in results if result.get("goal_reached") is True),
			"successes": sum(1 for result in results if result.get("success") is True),
			"query_results": results,
		}
		domain_root = run_root / domain_key
		domain_root.mkdir(parents=True, exist_ok=True)
		(domain_root / "summary.json").write_text(json.dumps(summary, indent=2))
		(run_root / f"{domain_key}.summary.json").write_text(json.dumps(summary, indent=2))
		domain_summaries[domain_key] = summary
	total_summary = {
		"run_root": str(run_root.resolve()),
		"baseline": "direct_plan_generation_api",
		"total_queries": sum(len(query_ids) for query_ids in selected_by_domain.values()),
		"completed_query_count": sum(
			len(summary["completed_query_ids"]) for summary in domain_summaries.values()
		),
		"remaining_query_count": sum(
			len(summary["remaining_query_ids"]) for summary in domain_summaries.values()
		),
		"parseable": sum(int(summary["parseable"]) for summary in domain_summaries.values()),
		"parse_failures": sum(int(summary["parse_failures"]) for summary in domain_summaries.values()),
		"api_failures": sum(int(summary["api_failures"]) for summary in domain_summaries.values()),
		"executable": sum(int(summary["executable"]) for summary in domain_summaries.values()),
		"goal_reached": sum(int(summary["goal_reached"]) for summary in domain_summaries.values()),
		"successes": sum(int(summary["successes"]) for summary in domain_summaries.values()),
		"domains": domain_summaries,
	}
	run_root.mkdir(parents=True, exist_ok=True)
	(run_root / "summary.json").write_text(json.dumps(total_summary, indent=2))


def _selected_queries(domain_key: str, explicit_query_ids: Sequence[str]) -> tuple[str, ...]:
	query_cases = load_domain_query_cases(domain_key)
	if explicit_query_ids:
		return tuple(sorted(explicit_query_ids, key=query_id_sort_key))
	return tuple(sorted(query_cases, key=query_id_sort_key))


def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
	run_root = Path(args.run_dir).resolve() if args.run_dir else RUNS_ROOT / _timestamp()
	domain_keys = tuple(args.domain or DOMAIN_KEYS)
	selected_by_domain = {
		domain_key: _selected_queries(domain_key, tuple(args.query_id or ()))
		for domain_key in domain_keys
	}
	results_by_domain: Dict[str, list[Dict[str, Any]]] = {domain_key: [] for domain_key in domain_keys}
	resumed: Dict[str, list[str]] = {domain_key: [] for domain_key in domain_keys}
	generator = DirectPlanGenerator(
		timeout=float(args.timeout_seconds),
		max_tokens=int(args.max_tokens),
	)
	for domain_key in domain_keys:
		for query_id in selected_by_domain[domain_key]:
			validation_path = _validation_path(run_root, domain_key, query_id)
			if args.resume and validation_path.exists():
				cached_result = json.loads(validation_path.read_text())
				cached_skip_state = bool(cached_result.get("verification_skipped"))
				if cached_skip_state == bool(args.skip_verifier):
					results_by_domain[domain_key].append(cached_result)
					resumed[domain_key].append(query_id)
					_write_summary(
						run_root=run_root,
						results_by_domain=results_by_domain,
						selected_by_domain=selected_by_domain,
						resumed=resumed,
					)
					continue
			result = _run_case(
				run_root=run_root,
				domain_key=domain_key,
				query_id=query_id,
				generator=generator,
				verify=not bool(args.skip_verifier),
			)
			results_by_domain[domain_key].append(result)
			_write_summary(
				run_root=run_root,
				results_by_domain=results_by_domain,
				selected_by_domain=selected_by_domain,
				resumed=resumed,
			)
	return json.loads((run_root / "summary.json").read_text())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run the direct plan-generation baseline through the API one query at a time.",
	)
	parser.add_argument("--domain", choices=DOMAIN_KEYS, action="append")
	parser.add_argument("--query-id", action="append", default=[])
	parser.add_argument("--run-dir")
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--timeout-seconds", type=float, default=1800.0)
	parser.add_argument("--max-tokens", type=int, default=24000)
	parser.add_argument("--skip-verifier", action="store_true")
	return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	os.environ.setdefault("PYTHONPATH", f"{SRC_ROOT}{os.pathsep}{PROJECT_ROOT}")
	args = parse_args(argv)
	summary = run_sweep(args)
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
