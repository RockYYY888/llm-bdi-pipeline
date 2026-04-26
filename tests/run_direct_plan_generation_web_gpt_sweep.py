from __future__ import annotations

import argparse
import json
import os
import subprocess
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


RUNS_ROOT = PROJECT_ROOT / "tests" / "generated" / "direct_plan_generation_baseline"
WEB_GPT_RUNNER = (
	Path.home()
	/ ".codex"
	/ "skills"
	/ "web-gpt"
	/ "scripts"
	/ "run_chatgpt_project_prompt_batch_cdp.py"
)
DOMAIN_KEYS = ("blocksworld", "marsrover", "satellite", "transport")


class WebGptCaseError(RuntimeError):
	def __init__(self, message: str, *, log_path: Path | None = None) -> None:
		super().__init__(message)
		self.log_path = log_path


def _timestamp() -> str:
	return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _case_id(domain_key: str, query_id: str) -> str:
	return f"{domain_key}__{query_id}"


def _load_temporal_specifications(domain_key: str, query_ids: Sequence[str]) -> Dict[str, Any]:
	_, records = load_query_sequence_records(
		domain_file=DOMAIN_FILES[domain_key],
		query_domain=domain_key,
		query_ids=query_ids,
	)
	return {record.instruction_id: record for record in records}


def _build_prompt(domain_key: str, query_id: str) -> Dict[str, Any]:
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
		"id": _case_id(domain_key, query_id),
		"prompt": "\n\n".join(
			[
				"SYSTEM PROMPT:",
				system_prompt,
				"USER PROMPT:",
				user_prompt,
			],
		),
		"system_prompt": system_prompt,
		"user_prompt": user_prompt,
		"case": case,
		"temporal_specification": temporal_specification,
	}


def _write_single_prompt_file(
	*,
	run_root: Path,
	domain_key: str,
	query_id: str,
	prompt: str,
) -> Path:
	prompt_path = run_root / "web_gpt_prompts" / domain_key / f"{query_id}.json"
	prompt_path.parent.mkdir(parents=True, exist_ok=True)
	prompt_path.write_text(
		json.dumps(
			[
				{
					"id": _case_id(domain_key, query_id),
					"prompt": prompt,
				},
			],
			indent=2,
		),
	)
	return prompt_path


def _extract_last_assistant_text(item_path: Path) -> str:
	payload = json.loads(item_path.read_text())
	target = payload.get("target")
	if not isinstance(target, dict):
		raise ValueError(f"Missing target payload in {item_path}.")
	text = str(target.get("last_assistant_text") or "").strip()
	if text:
		return text
	messages = target.get("messages") or []
	for message in reversed(messages):
		if isinstance(message, dict) and str(message.get("role") or "") == "assistant":
			text = str(message.get("text") or "").strip()
			if text:
				return text
	raise ValueError(f"Could not extract assistant response text from {item_path}.")


def _run_web_gpt_single(
	*,
	prompt_file: Path,
	run_root: Path,
	domain_key: str,
	query_id: str,
	timeout_seconds: float,
	settle_seconds: float,
) -> Path:
	run_id = f"web_gpt_{_case_id(domain_key, query_id)}"
	output_dir = run_root / "web_gpt_runs"
	command = [
		os.getenv("WEB_GPT_PYTHON", "python"),
		str(WEB_GPT_RUNNER),
		"--prompt-file",
		str(prompt_file),
		"--output-dir",
		str(output_dir),
		"--run-id",
		run_id,
		"--timeout-seconds",
		str(timeout_seconds),
		"--settle-seconds",
		str(settle_seconds),
	]
	completed = subprocess.run(
		command,
		cwd=PROJECT_ROOT,
		text=True,
		capture_output=True,
		check=False,
	)
	run_log_path = run_root / "web_gpt_logs" / domain_key / f"{query_id}.log"
	run_log_path.parent.mkdir(parents=True, exist_ok=True)
	run_log_path.write_text(
		"\n".join(
			[
				"$ " + " ".join(command),
				"",
				"STDOUT:",
				completed.stdout,
				"",
				"STDERR:",
				completed.stderr,
			],
		),
	)
	item_path = output_dir / run_id / "items" / f"{_case_id(domain_key, query_id)}.json"
	if completed.returncode != 0:
		raise WebGptCaseError(
			f"web-gpt failed for {domain_key}/{query_id}; "
			f"log={run_log_path}; item={item_path if item_path.exists() else 'missing'}",
			log_path=run_log_path,
		)
	if not item_path.exists():
		raise WebGptCaseError(
			f"web-gpt did not write expected item file: {item_path}",
			log_path=run_log_path,
		)
	return item_path


def _run_case(
	*,
	run_root: Path,
	domain_key: str,
	query_id: str,
	timeout_seconds: float,
	settle_seconds: float,
	verify: bool,
) -> Dict[str, Any]:
	prompt_payload = _build_prompt(domain_key, query_id)
	prompt_file = _write_single_prompt_file(
		run_root=run_root,
		domain_key=domain_key,
		query_id=query_id,
		prompt=str(prompt_payload["prompt"]),
	)
	try:
		item_path = _run_web_gpt_single(
			prompt_file=prompt_file,
			run_root=run_root,
			domain_key=domain_key,
			query_id=query_id,
			timeout_seconds=timeout_seconds,
			settle_seconds=settle_seconds,
		)
	except Exception as exc:
		return _record_web_gpt_failure(
			run_root=run_root,
			domain_key=domain_key,
			query_id=query_id,
			prompt_payload=prompt_payload,
			error=exc,
			log_path=getattr(exc, "log_path", None),
			verify=verify,
		)
	response_text = _extract_last_assistant_text(item_path)
	response_text_path = run_root / "web_gpt_responses" / domain_key / f"{query_id}.txt"
	response_text_path.parent.mkdir(parents=True, exist_ok=True)
	response_text_path.write_text(response_text)
	case = prompt_payload["case"]
	result = run_direct_plan_baseline_case(
		domain_key=domain_key,
		query_id=query_id,
		domain_file=DOMAIN_FILES[domain_key],
		problem_file=case["problem_file"],
		instruction=str(case["instruction"]),
		temporal_specification=prompt_payload["temporal_specification"],
		output_dir=run_root / domain_key / "query_results" / query_id,
		response_text=response_text,
		verify=verify,
		system_prompt_override=str(prompt_payload["system_prompt"]),
		user_prompt_override=str(prompt_payload["user_prompt"]),
	)
	return {
		**result.to_dict(),
		"web_gpt_item_path": str(item_path.resolve()),
		"web_gpt_response_text_path": str(response_text_path.resolve()),
		"web_gpt_failed": False,
	}


def _record_web_gpt_failure(
	*,
	run_root: Path,
	domain_key: str,
	query_id: str,
	prompt_payload: Dict[str, Any],
	error: Exception,
	log_path: Path | None,
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
				"llm": {
					"source": "web_gpt",
					"status": "failed",
					"error": str(error),
					"log_path": str(log_path.resolve()) if log_path else None,
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
		"web_gpt_failed": True,
		"web_gpt_log_path": str(log_path.resolve()) if log_path else None,
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
			"baseline": "direct_plan_generation_web_gpt",
			"total_queries": len(selected_query_ids),
			"selected_query_ids": list(selected_query_ids),
			"completed_query_ids": completed,
			"remaining_query_ids": [
				query_id for query_id in selected_query_ids if query_id not in set(completed)
			],
			"resumed_query_ids": list(resumed.get(domain_key, [])),
			"parseable": sum(1 for result in results if result.get("parseable") is True),
			"parse_failures": sum(1 for result in results if result.get("parseable") is not True),
			"web_gpt_failures": sum(1 for result in results if result.get("web_gpt_failed") is True),
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
		"baseline": "direct_plan_generation_web_gpt",
		"total_queries": sum(len(query_ids) for query_ids in selected_by_domain.values()),
		"completed_query_count": sum(
			len(summary["completed_query_ids"]) for summary in domain_summaries.values()
		),
		"remaining_query_count": sum(
			len(summary["remaining_query_ids"]) for summary in domain_summaries.values()
		),
		"parseable": sum(int(summary["parseable"]) for summary in domain_summaries.values()),
		"parse_failures": sum(int(summary["parse_failures"]) for summary in domain_summaries.values()),
		"web_gpt_failures": sum(int(summary["web_gpt_failures"]) for summary in domain_summaries.values()),
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
				timeout_seconds=float(args.timeout_seconds),
				settle_seconds=float(args.settle_seconds),
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
		description="Run direct plan-generation baseline through web-gpt one query at a time.",
	)
	parser.add_argument("--domain", choices=DOMAIN_KEYS, action="append")
	parser.add_argument("--query-id", action="append", default=[])
	parser.add_argument("--run-dir")
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--timeout-seconds", type=float, default=1800.0)
	parser.add_argument("--settle-seconds", type=float, default=20.0)
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
