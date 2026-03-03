"""
Live integration harness for the full blocksworld pipeline.

This file is the canonical acceptance entry point:
- pytest uses it for live end-to-end verification
- CLI can run a named query case: `python tests/test_pipeline.py query_2`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from ltl_bdi_pipeline import LTL_BDI_Pipeline
from stage4_panda_planning.panda_planner import PANDAPlanner
from utils.config import get_config
from utils.pipeline_logger import PipelineLogger


BANNED_TASK_PREFIXES = ("achieve_", "maintain_not_", "ensure_", "goal_")

QUERY_CASES: Dict[str, Dict[str, Any]] = {
	"query_1": {
		"instruction": "Using blocks a and b, arrange them so that a is on b.",
		"required_target_literals": ["on(a, b)"],
		"description": "Positive reachability goal",
	},
	"query_2": {
		"instruction": "Using blocks a and b, make sure a is not on b.",
		"required_target_literals": ["!on(a, b)"],
		"description": "Negative safety goal",
	},
	"query_3": {
		"instruction": "Using blocks a and b, make sure a is not clear.",
		"required_target_literals": ["!clear(a)"],
		"description": "Negative reachability goal",
	},
	"query_4": {
		"instruction": (
			"Using blocks a, b, and c, arrange them so that a is on b, "
			"b is on c, and a is clear."
		),
		"required_target_literals": ["on(a, b)", "on(b, c)", "clear(a)"],
		"description": "Three-goal stacking case",
	},
	"query_5": {
		"instruction": (
			"Using blocks a, b, c, d, and e, arrange them so that a is on b, "
			"b is on c, c is on d, d is on e, and a is clear."
		),
		"required_target_literals": [
			"on(a, b)",
			"on(b, c)",
			"on(c, d)",
			"on(d, e)",
			"clear(a)",
		],
		"description": "Five-goal tower case",
	},
	"query_6": {
		"instruction": (
			"Using blocks a, b, c, and d, make sure a is not on b, "
			"c is on d, c is clear, and d is not clear."
		),
		"required_target_literals": ["!on(a, b)", "on(c, d)", "clear(c)", "!clear(d)"],
		"description": "Mixed positive-negative combination case",
	},
}


def _ensure_live_dependencies() -> None:
	config = get_config()
	if not config.validate():
		pytest.skip("Live pipeline tests require a valid OPENAI_API_KEY")
	if not PANDAPlanner().toolchain_available():
		pytest.skip("Live pipeline tests require pandaPIparser, pandaPIgrounder, and pandaPIengine")


def _required_artifact_paths(log_dir: Path) -> List[Path]:
	return [
		log_dir / "execution.json",
		log_dir / "execution.txt",
		log_dir / "grounding_map.json",
		log_dir / "dfa_original.dot",
		log_dir / "dfa_simplified.dot",
		log_dir / "agentspeak_generated.asl",
		log_dir / "htn_method_library.json",
		log_dir / "panda_transitions.json",
		log_dir / "dfa.json",
	]


def _run_query_case(query_id: str) -> Dict[str, Any]:
	if query_id not in QUERY_CASES:
		raise KeyError(
			f"Unknown query id '{query_id}'. Available query ids: {sorted(QUERY_CASES)}",
		)

	case = QUERY_CASES[query_id]
	pipeline = LTL_BDI_Pipeline()
	test_logs_dir = Path(__file__).parent / "logs"
	pipeline.logger = PipelineLogger(logs_dir=str(test_logs_dir))

	result = pipeline.execute(case["instruction"], mode="dfa_agentspeak")
	log_dir = pipeline.logger.current_log_dir
	if log_dir is None:
		raise RuntimeError(f"{query_id} did not produce a log directory")

	execution_json_path = log_dir / "execution.json"
	execution_txt_path = log_dir / "execution.txt"
	execution = json.loads(execution_json_path.read_text())
	execution_txt = execution_txt_path.read_text()

	bug_messages: List[str] = []

	if not result["success"]:
		bug_messages.append("pipeline returned success=False")

	for path in _required_artifact_paths(log_dir):
		if not path.exists():
			bug_messages.append(f"missing log artifact: {path.name}")

	if (log_dir / "generated_code.asl").exists():
		bug_messages.append("unexpected legacy generated_code.asl artifact exists")

	if execution["natural_language"] != case["instruction"]:
		bug_messages.append("execution.json natural_language does not match selected query")

	for stage_key in ("stage1_status", "stage2_status", "stage3_status", "stage4_status", "stage5_status"):
		if execution.get(stage_key) != "success":
			bug_messages.append(f"{stage_key} is not success")

	if execution.get("stage3_used_llm") is not True:
		bug_messages.append("Stage 3 did not record live LLM usage")

	if execution.get("stage4_backend") != "pandaPI":
		bug_messages.append("Stage 4 backend is not pandaPI")

	stage3_library = execution.get("stage3_method_library") or {}
	target_bindings = stage3_library.get("target_task_bindings") or []
	if not target_bindings:
		bug_messages.append("Stage 3 produced no target_task_bindings")

	target_literals = execution.get("stage3_metadata", {}).get("target_literals", [])
	for required_target_literal in case["required_target_literals"]:
		if required_target_literal not in target_literals:
			bug_messages.append(
				f"required target literal '{required_target_literal}' not present in Stage 3 metadata",
			)

	stage5_code = execution.get("stage5_agentspeak") or ""
	for binding in target_bindings:
		task_name = binding["task_name"]
		if task_name.startswith(BANNED_TASK_PREFIXES):
			bug_messages.append(f"legacy task prefix still present: {task_name}")
		if (
			f"+!{task_name}(" not in stage5_code
			and f"+!{task_name} :" not in stage5_code
		):
			bug_messages.append(f"bound task '{task_name}' is missing from Stage 5 code")

	if "STAGE 3: DFA → HTN Method Synthesis" not in execution_txt:
		bug_messages.append("execution.txt is missing Stage 3 section")
	if "STAGE 4: HTN Method Library → PANDA Planning" not in execution_txt:
		bug_messages.append("execution.txt is missing Stage 4 section")
	if "STAGE 5: PANDA Plans → AgentSpeak Rendering" not in execution_txt:
		bug_messages.append("execution.txt is missing Stage 5 section")

	return {
		"query_id": query_id,
		"case": case,
		"result": result,
		"log_dir": log_dir,
		"execution": execution,
		"bug_messages": bug_messages,
		"has_bug": bool(bug_messages),
	}


@pytest.mark.parametrize("query_id", sorted(QUERY_CASES))
def test_blocksworld_pipeline_query_case(query_id: str):
	_ensure_live_dependencies()
	report = _run_query_case(query_id)
	assert report["has_bug"] is False, "\n".join(report["bug_messages"])


def _print_cli_report(report: Dict[str, Any]) -> None:
	query_id = report["query_id"]
	case = report["case"]
	execution = report["execution"]

	print(f"{query_id}: {case['description']}")
	print(f"Instruction: {case['instruction']}")
	print(f"Log Dir: {report['log_dir']}")
	print(f"Success: {report['result']['success']}")
	print(f"Stage 3 Target Literals: {execution.get('stage3_metadata', {}).get('target_literals', [])}")
	print(
		f"Stage 3 Target Task Bindings: "
		f"{(execution.get('stage3_method_library') or {}).get('target_task_bindings', [])}",
	)
	print(f"Has Bug: {report['has_bug']}")
	if report["bug_messages"]:
		for message in report["bug_messages"]:
			print(f"  - {message}")
	print("")


def main(argv: List[str]) -> int:
	config = get_config()
	if not config.validate():
		print("Live pipeline CLI requires a valid OPENAI_API_KEY.")
		return 2
	if not PANDAPlanner().toolchain_available():
		print("Live pipeline CLI requires pandaPIparser, pandaPIgrounder, and pandaPIengine.")
		return 2

	if len(argv) > 2:
		print("Usage: python tests/test_pipeline.py [query_i|all]")
		return 2

	selector = argv[1] if len(argv) == 2 else "query_1"
	if selector == "all":
		query_ids = sorted(QUERY_CASES)
	else:
		if selector not in QUERY_CASES:
			print(f"Unknown query id '{selector}'. Available: {sorted(QUERY_CASES)} or 'all'")
			return 2
		query_ids = [selector]

	reports = [_run_query_case(query_id) for query_id in query_ids]
	for report in reports:
		_print_cli_report(report)

	return 1 if any(report["has_bug"] for report in reports) else 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv))
