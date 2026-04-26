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
	run_direct_plan_baseline_case,
)
from tests.support.plan_library_evaluation_support import (
	DOMAIN_FILES,
	load_domain_query_cases,
	query_id_sort_key,
)


RUNS_ROOT = PROJECT_ROOT / "tests" / "generated" / "direct_plan_generation_baseline"
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


def _response_text_from_file(path: str | None) -> str | None:
	if not path:
		return None
	response_path = Path(path).resolve()
	payload = json.loads(response_path.read_text())
	if isinstance(payload, str):
		return payload
	if isinstance(payload, dict):
		for key in ("response_text", "text", "content", "message"):
			value = payload.get(key)
			if isinstance(value, str) and value.strip():
				return value
		if "plan_lines" in payload:
			return json.dumps(payload)
	raise ValueError(f"Could not extract response text from {response_path}.")


def run_domain(args: argparse.Namespace) -> Dict[str, Any]:
	domain_key = str(args.domain)
	query_cases = load_domain_query_cases(domain_key)
	query_ids = tuple(args.query_id or sorted(query_cases, key=query_id_sort_key))
	temporal_specifications = _load_temporal_specifications(domain_key, query_ids)
	run_root = Path(args.run_dir).resolve() if args.run_dir else RUNS_ROOT / _timestamp()
	domain_root = run_root / domain_key
	query_results_root = domain_root / "query_results"
	query_results_root.mkdir(parents=True, exist_ok=True)
	generator = None if args.response_file else DirectPlanGenerator()
	results = []
	resumed_query_ids = []
	for query_id in query_ids:
		result_path = query_results_root / query_id / "direct_plan_validation.json"
		if args.resume and result_path.exists():
			results.append(json.loads(result_path.read_text()))
			resumed_query_ids.append(query_id)
			continue
		case = query_cases[query_id]
		temporal_specification = temporal_specifications[query_id]
		result = run_direct_plan_baseline_case(
			domain_key=domain_key,
			query_id=query_id,
			domain_file=DOMAIN_FILES[domain_key],
			problem_file=case["problem_file"],
			instruction=str(case["instruction"]),
			temporal_specification=temporal_specification,
			output_dir=query_results_root / query_id,
			response_text=_response_text_from_file(args.response_file),
			generator=generator,
		)
		results.append(result.to_dict())
	summary = _build_summary(
		run_root=run_root,
		domain_key=domain_key,
		query_ids=query_ids,
		results=results,
		resume=bool(args.resume),
		resumed_query_ids=resumed_query_ids,
	)
	domain_root.mkdir(parents=True, exist_ok=True)
	(domain_root / "summary.json").write_text(json.dumps(summary, indent=2))
	(run_root / f"{domain_key}.summary.json").write_text(json.dumps(summary, indent=2))
	return summary


def _build_summary(
	*,
	run_root: Path,
	domain_key: str,
	query_ids: Sequence[str],
	results: Sequence[Dict[str, Any]],
	resume: bool,
	resumed_query_ids: Sequence[str],
) -> Dict[str, Any]:
	completed_query_ids = [str(result.get("query_id") or "") for result in results]
	return {
		"run_root": str(run_root),
		"domain_key": domain_key,
		"baseline": "direct_plan_generation",
		"total_queries": len(query_ids),
		"selected_query_ids": list(query_ids),
		"completed_query_ids": completed_query_ids,
		"remaining_query_ids": [
			query_id for query_id in query_ids if query_id not in set(completed_query_ids)
		],
		"resume_enabled": resume,
		"resumed_query_ids": list(resumed_query_ids),
		"parseable": sum(1 for result in results if result.get("parseable") is True),
		"executable": sum(1 for result in results if result.get("executable") is True),
		"goal_reached": sum(1 for result in results if result.get("goal_reached") is True),
		"successes": sum(1 for result in results if result.get("success") is True),
		"query_results": list(results),
	}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run the direct LLM plan-generation baseline.",
	)
	parser.add_argument("--domain", choices=DOMAIN_KEYS, required=True)
	parser.add_argument("--query-id", action="append", default=[])
	parser.add_argument("--run-dir")
	parser.add_argument("--response-file")
	parser.add_argument("--resume", action="store_true")
	return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	os.environ.setdefault("PYTHONPATH", f"{SRC_ROOT}{os.pathsep}{PROJECT_ROOT}")
	args = parse_args(argv)
	summary = run_domain(args)
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
