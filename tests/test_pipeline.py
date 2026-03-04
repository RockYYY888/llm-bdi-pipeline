"""
Live integration harness for the full blocksworld pipeline.

This file is the canonical acceptance entry point:
- pytest uses it for live end-to-end verification
- CLI can run a named query case: `python tests/test_pipeline.py query_2`
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

import ltl_bdi_pipeline as pipeline_module
from ltl_bdi_pipeline import LTL_BDI_Pipeline
from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTargetTaskBinding,
	HTNTask,
)
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


def _agent_vars(text: str) -> set[str]:
	return set(re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", text))


def _method_free_variable_messages(stage5_code: str) -> List[str]:
	start_marker = "/* HTN Method Plans */"
	end_marker = "/* DFA Transition Wrappers */"
	start_index = stage5_code.find(start_marker)
	end_index = stage5_code.find(end_marker)
	if start_index == -1 or end_index == -1 or end_index <= start_index:
		return []

	method_section = stage5_code[start_index + len(start_marker):end_index].strip()
	if not method_section:
		return []

	chunks = [
		chunk.strip()
		for chunk in re.split(r"\n\s*\n", method_section)
		if chunk.strip()
	]
	free_variable_messages: List[str] = []

	for chunk in chunks:
		lines = [line.rstrip() for line in chunk.splitlines() if line.strip()]
		if not lines:
			continue

		header = lines[0].strip()
		header_match = re.match(r"^\+!([a-z][a-z0-9_]*)\(([^)]*)\)\s*:\s*(.*?)\s*<-$", header)
		if not header_match:
			continue

		task_name, trigger_args_text, context_text = header_match.groups()
		trigger_vars = _agent_vars(trigger_args_text)
		context_vars = set() if context_text.strip() == "true" else _agent_vars(context_text)
		allowed_vars = trigger_vars | context_vars

		for body_line in lines[1:]:
			body_text = body_line.strip().rstrip(";.")
			if body_text == "true":
				continue
			body_vars = _agent_vars(body_text)
			free_vars = sorted(body_vars - allowed_vars)
			if free_vars:
				free_variable_messages.append(
					f"Stage 5 method '{task_name}' uses free body variables {free_vars} "
					f"not present in trigger/context: {body_text}",
				)

	return free_variable_messages


def _reachable_compound_task_names(stage3_library: Dict[str, Any]) -> List[str]:
	task_names = {
		binding["task_name"]
		for binding in (stage3_library.get("target_task_bindings") or [])
	}
	methods = stage3_library.get("methods") or []
	queue = list(task_names)

	while queue:
		task_name = queue.pop(0)
		for method in methods:
			if method.get("task_name") != task_name:
				continue
			for step in method.get("subtasks", []):
				if step.get("kind") != "compound":
					continue
				helper_name = step.get("task_name")
				if not helper_name or helper_name in task_names:
					continue
				task_names.add(helper_name)
				queue.append(helper_name)

	return sorted(task_names)


def _binding_semantic_messages(stage3_library: Dict[str, Any]) -> List[str]:
	compound_tasks = {
		task["name"]: task
		for task in (stage3_library.get("compound_tasks") or [])
	}
	methods_by_task: Dict[str, List[Dict[str, Any]]] = {}
	for method in (stage3_library.get("methods") or []):
		methods_by_task.setdefault(method["task_name"], []).append(method)

	messages: List[str] = []
	for binding in (stage3_library.get("target_task_bindings") or []):
		target_literal = binding["target_literal"]
		if not target_literal.startswith("!"):
			continue
		task_name = binding["task_name"]
		task = compound_tasks.get(task_name)
		if task is None:
			continue

		predicate_part = target_literal[1:]
		predicate_name, _, args_part = predicate_part.partition("(")
		task_parameters = task.get("parameters") or []
		expected_guard_signature = (
			f"!{predicate_name}({', '.join(task_parameters)})"
			if args_part
			else f"!{predicate_name}"
		)
		methods = methods_by_task.get(task_name, [])
		has_negative_guard = any(
			any(
				(
					("!" if not literal.get("is_positive", True) else "")
					+ literal["predicate"]
					+ (
						f"({', '.join(literal.get('args', []))})"
						if literal.get("args")
						else ""
					)
				) == expected_guard_signature
				for literal in (method.get("context") or [])
			)
			for method in methods
		)
		if not has_negative_guard:
			messages.append(
				f"Negative target binding '{target_literal}' -> '{task_name}' has no matching "
				f"negative guard context '{expected_guard_signature}'.",
			)

	return messages


def test_query_relevant_task_names_starts_from_target_bindings_not_only_witnesses():
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("top_goal", ("BLOCK",), False, ("clear",)),
			HTNTask("hidden_helper", ("BLOCK",), False, ("holding",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_top_goal_use_helper",
				task_name="top_goal",
				parameters=("BLOCK",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hidden_helper",
						args=("BLOCK",),
						kind="compound",
					),
				),
			),
			HTNMethod(
				method_name="m_hidden_helper_noop",
				task_name="hidden_helper",
				parameters=("BLOCK",),
				context=(HTNLiteral("holding", ("BLOCK",), True, None),),
			),
		],
		target_literals=[HTNLiteral("clear", ("a",), True, "clear_a")],
		target_task_bindings=[
			HTNTargetTaskBinding("clear(a)", "top_goal"),
		],
	)
	plan_records = []

	relevant = LTL_BDI_Pipeline._query_relevant_task_names(method_library, plan_records)

	assert relevant == ["hidden_helper", "top_goal"]


def test_stage3_summary_preserves_llm_timing_metadata(tmp_path, monkeypatch):
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",))],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),),
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), True, "on_a_b")],
		target_task_bindings=[HTNTargetTaskBinding("on(a, b)", "place_on")],
	)

	class FakeSynthesizer:
		def __init__(self, *args, **kwargs):
			pass

		def synthesize(self, domain, grounding_map, dfa_result):
			return method_library, {
				"used_llm": True,
				"model": "deepseek-chat",
				"target_literals": ["on(a, b)"],
				"compound_tasks": 1,
				"primitive_tasks": 0,
				"methods": 1,
				"llm_prompt": {"system": "SYSTEM", "user": "USER"},
				"llm_response": '{"ok": true}',
				"llm_finish_reason": "stop",
				"llm_attempts": 2,
				"llm_response_time_seconds": 3.21,
				"llm_attempt_durations_seconds": [1.0, 2.21],
			}

		def extract_progressing_transitions(self, grounding_map, dfa_result):
			return []

	monkeypatch.setattr(pipeline_module, "HTNMethodSynthesizer", FakeSynthesizer)

	pipeline = LTL_BDI_Pipeline()
	pipeline.logger = PipelineLogger(logs_dir=str(tmp_path))
	pipeline.logger.start_pipeline(
		"demo instruction",
		mode="dfa_agentspeak",
		domain_file=pipeline.domain_file,
		output_dir=str(tmp_path),
	)
	pipeline.output_dir = pipeline.logger.current_log_dir

	_, stage3_data = pipeline._stage3_method_synthesis(
		SimpleNamespace(grounding_map=None),
		{"dfa_dot": 'digraph { 0 -> 1 [label="on_a_b"]; }'},
	)

	assert stage3_data is not None
	assert stage3_data["summary"]["llm_attempts"] == 2
	assert stage3_data["summary"]["llm_response_time_seconds"] == 3.21
	assert stage3_data["summary"]["llm_attempt_durations_seconds"] == [1.0, 2.21]


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
	bug_messages.extend(_binding_semantic_messages(stage3_library))

	target_literals = execution.get("stage3_metadata", {}).get("target_literals", [])
	for required_target_literal in case["required_target_literals"]:
		if required_target_literal not in target_literals:
			bug_messages.append(
				f"required target literal '{required_target_literal}' not present in Stage 3 metadata",
			)

	stage4_artifacts = execution.get("stage4_artifacts") or {}
	method_validations = stage4_artifacts.get("method_validations") or []
	if not method_validations:
		bug_messages.append("Stage 4 recorded no per-method validation artifacts")
	if any(item.get("status") != "success" for item in method_validations):
		failed_methods = sorted(
			item.get("method_name")
			for item in method_validations
			if item.get("status") != "success"
		)
		bug_messages.append(
			f"Stage 4 has failed per-method validations: {failed_methods}",
		)

	expected_method_names = {
		method["method_name"]
		for method in (stage3_library.get("methods") or [])
		if method.get("task_name") in _reachable_compound_task_names(stage3_library)
	}
	validated_method_names = {
		item.get("method_name")
		for item in method_validations
		if item.get("method_name")
	}
	missing_method_validations = sorted(expected_method_names - validated_method_names)
	if missing_method_validations:
		bug_messages.append(
			"Stage 4 did not validate all reachable sibling methods: "
			f"{missing_method_validations}",
		)

	stage5_code = execution.get("stage5_agentspeak") or ""
	if "/* HTN Method Plans */" not in stage5_code:
		bug_messages.append("Stage 5 code is missing rendered HTN method plans")
	if "dfa_edge_label(" not in stage5_code:
		bug_messages.append("Stage 5 code is missing dfa_edge_label metadata")
	if "dfa_state(" not in stage5_code:
		bug_messages.append("Stage 5 code is missing dfa_state state-tracking facts or guards")
	if "+!dfa_step_" not in stage5_code:
		bug_messages.append("Stage 5 code is missing state-aware dfa_step wrappers")
	if "/* PANDA Goal Plans */" in stage5_code:
		bug_messages.append("legacy PANDA-only task plan section still present in Stage 5 code")
	if "target_label(" in stage5_code:
		bug_messages.append("legacy target_label facts still present in Stage 5 code")
	if "+!transition_" in stage5_code:
		bug_messages.append("legacy transition_i wrappers still present in Stage 5 code")
	bug_messages.extend(_method_free_variable_messages(stage5_code))

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
	if "STAGE 5: HTN Methods + DFA Wrappers → AgentSpeak Rendering" not in execution_txt:
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


def test_method_validation_initial_facts_are_branch_specific():
	pipeline = LTL_BDI_Pipeline()
	planner = PANDAPlanner()
	method = HTNMethod(
		method_name="m_hold_block_from_block",
		task_name="hold_block",
		parameters=("BLOCK1",),
		context=(),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="clear_top",
				args=("BLOCK1",),
				kind="compound",
			),
			HTNMethodStep(
				step_id="s2",
				task_name="pick_up",
				args=("BLOCK1", "BLOCK2"),
				kind="primitive",
				action_name="pick-up",
				preconditions=(
					HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
				),
			),
		),
		ordering=(("s1", "s2"),),
	)
	method_library = HTNMethodLibrary(
		methods=[
			method,
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("BLOCK1",),
				context=(HTNLiteral("clear", ("BLOCK1",), True, None),),
			),
		],
	)

	facts = pipeline._method_validation_initial_facts(
		planner,
		method,
		method_library,
		("a",),
		("a", "b", "c"),
	)

	assert "(handempty)" in facts
	assert "(clear a)" in facts
	assert "(on a b)" in facts


def test_method_validation_initial_facts_avoid_conflicting_global_defaults():
	pipeline = LTL_BDI_Pipeline()
	planner = PANDAPlanner()
	method = HTNMethod(
		method_name="m_place_on_direct",
		task_name="place_on",
		parameters=("BLOCK1", "BLOCK2"),
		context=(),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="hold_block",
				args=("BLOCK1",),
				kind="compound",
			),
			HTNMethodStep(
				step_id="s2",
				task_name="clear_top",
				args=("BLOCK2",),
				kind="compound",
			),
			HTNMethodStep(
				step_id="s3",
				task_name="put_on_block",
				args=("BLOCK1", "BLOCK2"),
				kind="primitive",
				action_name="put-on-block",
			),
		),
		ordering=(("s1", "s2"), ("s2", "s3")),
	)
	method_library = HTNMethodLibrary(
		methods=[
			method,
			HTNMethod(
				method_name="m_hold_block_already",
				task_name="hold_block",
				parameters=("BLOCK1",),
				context=(HTNLiteral("holding", ("BLOCK1",), True, None),),
			),
			HTNMethod(
				method_name="m_clear_top_already",
				task_name="clear_top",
				parameters=("BLOCK1",),
				context=(HTNLiteral("clear", ("BLOCK1",), True, None),),
			),
		],
	)

	facts = pipeline._method_validation_initial_facts(
		planner,
		method,
		method_library,
		("a", "b"),
		("a", "b", "c"),
	)

	assert "(holding a)" in facts
	assert "(clear b)" in facts
	assert "(handempty)" not in facts


def test_task_witness_initial_facts_merge_sibling_branches():
	pipeline = LTL_BDI_Pipeline()
	planner = PANDAPlanner()
	method_library = HTNMethodLibrary(
		methods=[
			HTNMethod(
				method_name="m_place_on_direct",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("holding", ("BLOCK1",), True, None),
					HTNLiteral("clear", ("BLOCK2",), True, None),
				),
			),
			HTNMethod(
				method_name="m_place_on_acquire",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up_from_table",
						args=("BLOCK1",),
						kind="primitive",
						action_name="pick-up-from-table",
					),
				),
			),
		],
	)

	facts = pipeline._task_witness_initial_facts(
		planner,
		"place_on",
		method_library,
		("a", "b"),
		("a", "b", "c"),
	)

	assert "(holding a)" in facts
	assert "(clear b)" in facts
	assert "(ontable a)" in facts


def test_method_validation_initial_facts_allocate_typed_witness_objects():
	pipeline = LTL_BDI_Pipeline()
	planner = PANDAPlanner()
	method = HTNMethod(
		method_name="m_remove_on_clear_first",
		task_name="remove_on",
		parameters=("BLOCK1", "BLOCK2"),
		context=(
			HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
			HTNLiteral("clear", ("BLOCK1",), False, None),
		),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="clear_top",
				args=("BLOCK1",),
				kind="compound",
			),
			HTNMethodStep(
				step_id="s2",
				task_name="remove_on",
				args=("BLOCK1", "BLOCK2"),
				kind="compound",
			),
		),
		ordering=(("s1", "s2"),),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("remove_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
			HTNTask("clear_top", ("BLOCK1",), False, ("clear",)),
		],
		methods=[
			method,
			HTNMethod(
				method_name="m_clear_top_remove",
				task_name="clear_top",
				parameters=("BLOCK1", "BLOCK3"),
				context=(HTNLiteral("holding", ("BLOCK3",), True, None),),
				subtasks=(),
			),
		],
	)
	object_pool = ["a"]
	object_types = {"a": "block"}

	facts = pipeline._method_validation_initial_facts(
		planner,
		method,
		method_library,
		("a", "b"),
		("a", "b"),
		object_pool=object_pool,
		object_types=object_types,
	)

	assert "(holding witness_block_1)" in facts
	assert "witness_block_1" in object_pool
	assert object_types["witness_block_1"] == "block"


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
