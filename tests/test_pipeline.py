"""
Live integration harness for benchmark-backed pipeline acceptance.

This file is the canonical acceptance entry point:
- pytest uses it for live end-to-end verification
- CLI can run a named query case: `python tests/test_pipeline.py query_2`
"""

from __future__ import annotations

import json
import os
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
from ltl_bdi_pipeline import LTL_BDI_Pipeline, TypeResolutionError
from stage1_interpretation.ltlf_formula import LTLFormula, LogicalOperator, TemporalOperator
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTargetTaskBinding,
	HTNTask,
)
from stage4_panda_planning.panda_planner import PANDAPlanner
from stage6_jason_validation.jason_runner import JasonRunner
from utils.config import get_config
from utils.hddl_parser import HDDLParser
from utils.ipc_plan_verifier import IPCPlanVerifier
from utils.pipeline_logger import PipelineLogger

OFFICIAL_BLOCKSWORLD_DOMAIN_FILE = str(
	(Path(__file__).parent.parent / "src" / "domains" / "blocksworld" / "domain.hddl").resolve(),
)
BLOCKSWORLD_PROBLEM_DIR = (
	Path(__file__).parent.parent / "src" / "domains" / "blocksworld" / "problems"
)
MARSROVER_DOMAIN_FILE = str(
	(Path(__file__).parent.parent / "src" / "domains" / "marsrover" / "domain.hddl").resolve(),
)
MARSROVER_PROBLEM_DIR = (
	Path(__file__).parent.parent / "src" / "domains" / "marsrover" / "problems"
)


BANNED_TASK_PREFIXES = ("achieve_", "maintain_not_", "ensure_", "goal_")
IPC_PLAN_VERIFIER = IPCPlanVerifier()

MARSROVER_BASE_QUERY_CASES: Dict[str, Dict[str, Any]] = {
	"rover_query_1": {
		"instruction": (
			"Using rover rover0 and waypoints waypoint1 and waypoint5, "
			"make sure rover0 is at waypoint5."
		),
		"required_target_literals": ["at(rover0, waypoint5)"],
		"description": "Rover navigation reachability goal",
	},
	"rover_query_2": {
		"instruction": (
			"Using waypoint waypoint2, make sure rock data from waypoint2 is communicated."
		),
		"required_target_literals": ["communicated_rock_data(waypoint2)"],
		"description": "Rover communicated rock data goal",
	},
	"rover_query_3": {
		"instruction": (
			"Using objective objective0 and mode low_res, make sure image data for objective0 "
			"in low_res mode is communicated."
		),
		"required_target_literals": ["communicated_image_data(objective0, low_res)"],
		"description": "Rover communicated image data goal",
	},
}


def _literal_signature(predicate: str, args: List[str], is_positive: bool = True) -> str:
	atom = predicate if not args else f"{predicate}({', '.join(args)})"
	return atom if is_positive else f"!{atom}"


def _serialise_nl_list(items: List[str]) -> str:
	if not items:
		return ""
	if len(items) == 1:
		return items[0]
	if len(items) == 2:
		return f"{items[0]} and {items[1]}"
	return f"{', '.join(items[:-1])}, and {items[-1]}"


def _blocksworld_task_invocation_to_target_literal(
	task_name: str,
	args: List[str],
) -> Dict[str, Any] | None:
	if task_name in {"do_put_on", "do_move"} and len(args) >= 2:
		return {
			"predicate": "on",
			"args": [args[0], args[1]],
		}
	if task_name == "do_on_table" and args:
		return {
			"predicate": "ontable",
			"args": [args[0]],
		}
	if task_name == "do_clear" and args:
		return {
			"predicate": "clear",
			"args": [args[0]],
		}
	return None


def _blocksworld_literal_to_nl_goal(predicate: str, args: List[str]) -> str:
	if predicate == "on" and len(args) == 2:
		return f"{args[0]} is on {args[1]}"
	if predicate == "ontable" and len(args) == 1:
		return f"{args[0]} is on the table"
	if predicate == "clear" and len(args) == 1:
		return f"{args[0]} is clear"
	if not args:
		return f"{predicate} holds"
	return f"{predicate}({', '.join(args)}) holds"


def _order_blocksworld_goal_literals(
	literals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	on_literals = [
		item for item in literals
		if item.get("predicate") == "on" and len(item.get("args", [])) == 2
	]
	other_literals = [
		item for item in literals
		if item not in on_literals
	]
	if len(on_literals) <= 1:
		return literals

	signature_to_literal = {
		_literal_signature(item["predicate"], item["args"]): item
		for item in on_literals
	}
	supporter_by_top = {
		item["args"][0]: _literal_signature(item["predicate"], item["args"])
		for item in on_literals
	}
	dependents: Dict[str, List[str]] = {
		signature: []
		for signature in signature_to_literal
	}
	in_degree: Dict[str, int] = {
		signature: 0
		for signature in signature_to_literal
	}
	for item in on_literals:
		signature = _literal_signature(item["predicate"], item["args"])
		support_block = item["args"][1]
		predecessor = supporter_by_top.get(support_block)
		if predecessor is None:
			continue
		dependents[predecessor].append(signature)
		in_degree[signature] += 1

	ordered_signatures: List[str] = []
	queue = [
		_literal_signature(item["predicate"], item["args"])
		for item in on_literals
		if in_degree[_literal_signature(item["predicate"], item["args"])] == 0
	]
	seen = set(queue)
	while queue:
		signature = queue.pop(0)
		ordered_signatures.append(signature)
		for dependent in dependents[signature]:
			in_degree[dependent] -= 1
			if in_degree[dependent] == 0 and dependent not in seen:
				queue.append(dependent)
				seen.add(dependent)

	if len(ordered_signatures) != len(on_literals):
		return literals

	ordered_literals = [signature_to_literal[signature] for signature in ordered_signatures]
	return ordered_literals + other_literals


def _build_case_from_blocksworld_problem(problem_path: Path) -> Dict[str, Any] | None:
	problem = HDDLParser.parse_problem(str(problem_path))
	derived_literals: List[Dict[str, Any]] = []
	seen = set()

	for fact in problem.goal_facts:
		if not fact.is_positive:
			continue
		signature = _literal_signature(fact.predicate, fact.args)
		if signature in seen:
			continue
		seen.add(signature)
		derived_literals.append(
			{
				"predicate": fact.predicate,
				"args": list(fact.args),
			},
		)

	if not derived_literals:
		for invocation in problem.htn_tasks:
			literal = _blocksworld_task_invocation_to_target_literal(
				invocation.task_name,
				invocation.args,
			)
			if literal is None:
				continue
			signature = _literal_signature(literal["predicate"], literal["args"])
			if signature in seen:
				continue
			seen.add(signature)
			derived_literals.append(literal)

	if not derived_literals:
		return None
	derived_literals = _order_blocksworld_goal_literals(derived_literals)

	goal_text = ", and ".join(
		_blocksworld_literal_to_nl_goal(item["predicate"], item["args"])
		for item in derived_literals
	)

	return {
		"instruction": (
			f"Using blocks {_serialise_nl_list(problem.objects)}, arrange them so that {goal_text}."
		),
		"required_target_literals": [
			_literal_signature(item["predicate"], item["args"])
			for item in derived_literals
		],
		"problem_file": str(problem_path.resolve()),
		"minimum_action_count": 1,
		"description": f"Auto-generated from {problem_path.name} ({problem.name})",
	}


def _load_blocksworld_problem_query_cases(limit: int = 3) -> Dict[str, Dict[str, Any]]:
	if not BLOCKSWORLD_PROBLEM_DIR.exists():
		return {}
	cases: Dict[str, Dict[str, Any]] = {}
	problem_paths = sorted(BLOCKSWORLD_PROBLEM_DIR.glob("p*.hddl"))[:limit]
	for index, problem_path in enumerate(problem_paths, start=1):
		case = _build_case_from_blocksworld_problem(problem_path)
		if case is None:
			continue
		cases[f"query_{index}"] = case
	return cases


QUERY_CASES: Dict[str, Dict[str, Any]] = _load_blocksworld_problem_query_cases()


def _task_invocation_to_target_literal(
	task_name: str,
	args: List[str],
) -> Dict[str, Any] | None:
	"""
	Convert rover HTN top-level tasks to canonical predicate targets.

	This is derived from the rover domain method/action semantics and lets us
	reverse official `pfile*.hddl` missions into NL query assertions.
	"""
	if task_name in {"get_soil_data", "send_soil_data"} and args:
		return {
			"predicate": "communicated_soil_data",
			"args": [args[-1]],
		}
	if task_name in {"get_rock_data", "send_rock_data"} and args:
		return {
			"predicate": "communicated_rock_data",
			"args": [args[-1]],
		}
	if task_name in {"get_image_data", "send_image_data"} and len(args) >= 2:
		return {
			"predicate": "communicated_image_data",
			"args": args[-2:],
		}
	if task_name == "navigate_abs" and len(args) >= 2:
		return {
			"predicate": "at",
			"args": [args[0], args[1]],
		}
	if task_name == "calibrate_abs" and len(args) >= 2:
		return {
			"predicate": "calibrated",
			"args": [args[1], args[0]],
		}
	if task_name == "empty-store" and args:
		return {
			"predicate": "empty",
			"args": [args[0]],
		}
	return None


def _literal_to_nl_goal(predicate: str, args: List[str]) -> str:
	if predicate == "communicated_soil_data" and len(args) == 1:
		return f"soil data from {args[0]} is communicated"
	if predicate == "communicated_rock_data" and len(args) == 1:
		return f"rock data from {args[0]} is communicated"
	if predicate == "communicated_image_data" and len(args) == 2:
		return f"image data for {args[0]} in {args[1]} mode is communicated"
	if predicate == "at" and len(args) == 2:
		return f"{args[0]} is at {args[1]}"
	if predicate == "calibrated" and len(args) == 2:
		return f"{args[0]} is calibrated for {args[1]}"
	if predicate == "empty" and len(args) == 1:
		return f"{args[0]} is empty"
	if not args:
		return f"{predicate} holds"
	return f"{predicate}({', '.join(args)}) holds"


def _build_case_from_rover_problem(problem_path: Path) -> Dict[str, Any] | None:
	problem = HDDLParser.parse_problem(str(problem_path))
	derived_literals: List[Dict[str, Any]] = []
	seen = set()
	for invocation in problem.htn_tasks:
		literal = _task_invocation_to_target_literal(invocation.task_name, invocation.args)
		if literal is None:
			continue
		signature = _literal_signature(literal["predicate"], literal["args"])
		if signature in seen:
			continue
		seen.add(signature)
		derived_literals.append(literal)
		if len(derived_literals) >= 3:
			break

	if not derived_literals:
		return None

	relevant_objects = []
	for literal in derived_literals:
		for arg in literal["args"]:
			if arg not in relevant_objects:
				relevant_objects.append(arg)
	if not relevant_objects:
		relevant_objects = problem.objects[:4]

	initial_anchor = next(
		(
			fact
			for fact in problem.init_facts
			if fact.predicate == "at"
			and len(fact.args) == 2
			and fact.args[0].startswith("rover")
		),
		None,
	)
	goal_text = ", and ".join(
		_literal_to_nl_goal(item["predicate"], item["args"])
		for item in derived_literals
	)
	instruction_parts = [
		f"Using objects {', '.join(relevant_objects)}, make sure {goal_text}.",
	]
	if initial_anchor is not None:
		instruction_parts.insert(
			0,
			f"Initially {initial_anchor.args[0]} is at {initial_anchor.args[1]}.",
		)

	return {
		"instruction": " ".join(instruction_parts),
		"required_target_literals": [
			_literal_signature(item["predicate"], item["args"])
			for item in derived_literals
		],
		"description": f"Auto-generated from {problem_path.name} ({problem.name})",
	}


def _load_rover_problem_query_cases(limit: int = 5) -> Dict[str, Dict[str, Any]]:
	if not MARSROVER_PROBLEM_DIR.exists():
		return {}
	cases: Dict[str, Dict[str, Any]] = {}
	problem_paths = sorted(MARSROVER_PROBLEM_DIR.glob("pfile*.hddl"))[:limit]
	for index, problem_path in enumerate(problem_paths, start=1):
		case = _build_case_from_rover_problem(problem_path)
		if case is None:
			continue
		case_id = f"rover_problem_{index:02d}"
		cases[case_id] = case
	return cases


MARSROVER_QUERY_CASES: Dict[str, Dict[str, Any]] = {
	**MARSROVER_BASE_QUERY_CASES,
	**_load_rover_problem_query_cases(),
}


def _pytest_selected_case_ids(
	case_map: Dict[str, Dict[str, Any]],
	*,
	query_env: str,
	all_env: str,
	default_query: str,
) -> List[str]:
	"""Default to a single live query; opt in to full sweep explicitly."""
	run_all = os.getenv(all_env, "").lower() in {"1", "true", "yes"}
	if run_all:
		return sorted(case_map)

	query_id = os.getenv(query_env, default_query)
	if query_id not in case_map:
		query_id = default_query
	return [query_id]


def _pytest_selected_query_ids() -> List[str]:
	return _pytest_selected_case_ids(
		QUERY_CASES,
		query_env="PIPELINE_TEST_QUERY",
		all_env="PIPELINE_TEST_ALL",
		default_query="query_1",
	)


def _pytest_selected_rover_query_ids() -> List[str]:
	return _pytest_selected_case_ids(
		MARSROVER_QUERY_CASES,
		query_env="PIPELINE_TEST_ROVER_QUERY",
		all_env="PIPELINE_TEST_ROVER_ALL",
		default_query="rover_query_1",
	)


def _is_rover_live_enabled() -> bool:
	return os.getenv("PIPELINE_TEST_ROVER", "").lower() in {"1", "true", "yes"}


def _ensure_live_dependencies() -> None:
	config = get_config()
	if not config.validate():
		pytest.skip("Live pipeline tests require a valid OPENAI_API_KEY")
	if not PANDAPlanner().toolchain_available():
		pytest.skip("Live pipeline tests require pandaPIparser, pandaPIgrounder, and pandaPIengine")
	if not JasonRunner().toolchain_available():
		pytest.skip("Live pipeline tests require Stage 6 Java 17-23 and Jason runtime toolchain")
	if not IPC_PLAN_VERIFIER.tool_available():
		pytest.skip("Live pipeline tests require the official pandaPIparser verifier on PATH")


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
		log_dir / "jason_runner.mas2j",
		log_dir / "jason_stdout.txt",
		log_dir / "jason_stderr.txt",
		log_dir / "jason_validation.json",
		log_dir / "action_path.txt",
		log_dir / "ipc_official_plan.txt",
		log_dir / "ipc_official_verifier.txt",
		log_dir / "ipc_official_verification.json",
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


def test_representative_task_args_uses_typed_witness_placeholders(tmp_path):
	domain_file = tmp_path / "domain_move.hddl"
	domain_file.write_text(
		"""
(define (domain move_domain)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types rover waypoint)
  (:predicates
    (at ?r - rover ?w - waypoint)
  )
  (:task move_to
    :parameters (?r - rover ?w - waypoint)
  )
  (:action navigate
    :parameters (?r - rover ?from - waypoint ?to - waypoint)
    :precondition (and (at ?r ?from))
    :effect (and (not (at ?r ?from)) (at ?r ?to))
  )
)
		""".strip(),
	)
	pipeline = LTL_BDI_Pipeline(domain_file=str(domain_file))
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("move_to", ("ROVER", "WAYPOINT"), False, ("at",)),
		],
		primitive_tasks=[],
		methods=[],
		target_literals=[],
		target_task_bindings=[],
	)

	args = pipeline._representative_task_args(
		"move_to",
		method_library,
		("rover0", "waypoint1"),
		[],
	)

	assert args == ("witness_rover_1", "witness_waypoint_2")


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

		def synthesize(
			self,
			domain,
			grounding_map,
			dfa_result,
			*,
			query_text=None,
			negation_hints=None,
		):
			return method_library, {
				"used_llm": True,
				"model": "deepseek-chat",
				"target_literals": ["on(a, b)"],
				"negation_resolution": {"predicates": [], "mode_by_predicate": {}},
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

	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
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


def test_pipeline_requires_explicit_domain_file():
	with pytest.raises(ValueError, match="domain_file is required"):
		LTL_BDI_Pipeline(domain_file=None)  # type: ignore[arg-type]


def test_seed_validation_scope_preserves_multi_type_object_assignments(tmp_path):
	domain_file = tmp_path / "domain_transport.hddl"
	domain_file.write_text(
		"""
(define (domain transport)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types robot package location)
  (:predicates
    (at_robot ?r - robot ?l - location)
    (at_package ?p - package ?l - location)
  )
  (:task deliver
    :parameters (?r - robot ?p - package ?to - location)
  )
  (:action drive
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and (at_robot ?r ?from))
    :effect (and (at_robot ?r ?to) (not (at_robot ?r ?from)))
  )
)
		""".strip(),
	)
	pipeline = LTL_BDI_Pipeline(domain_file=str(domain_file))
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("deliver", ("ROBOT", "PACKAGE", "LOCATION"), False, ("at_package",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_deliver_drive",
				task_name="deliver",
				parameters=("ROBOT", "PACKAGE", "LOCATION", "FROM"),
				context=(
					HTNLiteral("at_robot", ("ROBOT", "FROM"), True, None),
					HTNLiteral("at_package", ("PACKAGE", "FROM"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="drive",
						args=("ROBOT", "FROM", "LOCATION"),
						kind="primitive",
						action_name="drive",
					),
				),
				ordering=(),
			),
		],
		target_literals=[
			HTNLiteral("at_package", ("pkg1", "loc2"), True, None),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("at_package(pkg1, loc2)", "deliver"),
		],
	)
	object_pool, object_types = pipeline._seed_validation_scope(
		"deliver",
		method_library,
		("r1", "pkg1", "loc2"),
		("r1", "pkg1", "loc1", "loc2"),
	)

	assert set(object_pool) == {"r1", "pkg1", "loc2"}
	assert object_types["r1"] == "robot"
	assert object_types["pkg1"] == "package"
	assert object_types["loc2"] == "location"


def test_seed_validation_scope_fails_for_ambiguous_parent_type(tmp_path):
	domain_file = tmp_path / "domain_vehicle.hddl"
	domain_file.write_text(
		"""
(define (domain vehicle_world)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types car truck - vehicle vehicle location)
  (:predicates
    (at ?v - vehicle ?l - location)
  )
  (:task reach
    :parameters (?v - vehicle ?l - location)
  )
  (:action noop
    :parameters (?v - vehicle ?l - location)
    :precondition (and (at ?v ?l))
    :effect (and (at ?v ?l))
  )
)
		""".strip(),
	)
	pipeline = LTL_BDI_Pipeline(domain_file=str(domain_file))
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("reach", ("VEHICLE", "LOCATION"), False, ("at",))],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_reach_noop",
				task_name="reach",
				parameters=("VEHICLE", "LOCATION"),
				context=(HTNLiteral("at", ("VEHICLE", "LOCATION"), True, None),),
			),
		],
		target_literals=[HTNLiteral("at", ("v1", "l1"), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("at(v1, l1)", "reach")],
	)

	with pytest.raises(TypeResolutionError, match="ambiguous"):
		pipeline._seed_validation_scope(
			"reach",
			method_library,
			("v1", "l1"),
			("v1", "l1"),
		)


def test_stage3_type_validation_fails_for_untyped_method_variable(tmp_path):
	domain_file = tmp_path / "domain_typed.hddl"
	domain_file.write_text(
		"""
(define (domain typed_world)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types block location)
  (:predicates
    (at ?b - block ?l - location)
  )
  (:task move
    :parameters (?b - block ?l - location)
  )
  (:action noop
    :parameters (?b - block ?l - location)
    :precondition (and (at ?b ?l))
    :effect (and (at ?b ?l))
  )
)
		""".strip(),
	)
	pipeline = LTL_BDI_Pipeline(domain_file=str(domain_file))
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("move", ("BLOCK", "LOCATION"), False, ("at",))],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_move_bad",
				task_name="move",
				parameters=("BLOCK", "LOCATION", "UNBOUND"),
				context=(HTNLiteral("=", ("UNBOUND", "BLOCK"), True, None),),
				subtasks=(),
				ordering=(),
			),
		],
		target_literals=[HTNLiteral("at", ("b1", "l1"), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("at(b1, l1)", "move")],
	)

	with pytest.raises(TypeResolutionError, match="UNBOUND"):
		pipeline._validate_method_library_typing(method_library)


def test_stage1_object_universe_merges_constants_from_atoms_and_formulas():
	generator = NLToLTLfGenerator()
	formula = LTLFormula(
		operator=None,
		predicate={"communicated_image_data": ["objective0", "low_res"]},
		sub_formulas=[],
		logical_op=None,
	)
	objects = generator._augment_objects_from_formulas_and_atoms(
		["objective0"],
		[formula],
		[
			{
				"symbol": "communicated_image_data_objective0_low_res",
				"predicate": "communicated_image_data",
				"args": ["objective0", "low_res"],
			},
		],
	)
	assert objects == ["objective0", "low_res"]


def test_ordered_literal_signatures_extracts_eventually_wrapped_atoms():
	spec = SimpleNamespace(
		formulas=[
			LTLFormula(
				operator=None,
				predicate=None,
				sub_formulas=[
					LTLFormula(
						operator=TemporalOperator.FINALLY,
						predicate=None,
						sub_formulas=[
							LTLFormula(
								operator=None,
								predicate={"on": ["b1", "b4"]},
								sub_formulas=[],
								logical_op=None,
							),
						],
						logical_op=None,
					),
					LTLFormula(
						operator=TemporalOperator.FINALLY,
						predicate=None,
						sub_formulas=[
							LTLFormula(
								operator=None,
								predicate={"on": ["b3", "b1"]},
								sub_formulas=[],
								logical_op=None,
							),
						],
						logical_op=None,
					),
				],
				logical_op=LogicalOperator.AND,
			),
		],
	)

	assert LTL_BDI_Pipeline._ordered_literal_signatures_from_spec(spec) == (
		"on(b1, b4)",
		"on(b3, b1)",
	)


def test_blocksworld_problem_query_case_generation_from_htn_tasks():
	problem_path = (
		Path(__file__).parent.parent
		/ "src"
		/ "domains"
		/ "blocksworld"
		/ "problems"
		/ "p01.hddl"
	)
	if not problem_path.exists():
		pytest.skip(f"Missing blocksworld problem file: {problem_path}")

	case = _build_case_from_blocksworld_problem(problem_path)
	assert case is not None
	assert case["instruction"]
	assert case["required_target_literals"]
	assert case["problem_file"] == str(problem_path.resolve())
	assert case["required_target_literals"] == ["on(b1, b4)", "on(b3, b1)"]


def test_rover_problem_query_case_generation_from_htn_tasks():
	problem_path = (
		Path(__file__).parent.parent
		/ "src"
		/ "domains"
		/ "marsrover"
		/ "problems"
		/ "pfile01.hddl"
	)
	if not problem_path.exists():
		pytest.skip(f"Missing rover problem file: {problem_path}")

	case = _build_case_from_rover_problem(problem_path)
	assert case is not None
	assert case["instruction"]
	assert case["required_target_literals"]
	assert any(
		item.startswith("communicated_") or item.startswith("at(") or item.startswith("calibrated(")
		for item in case["required_target_literals"]
	)


def test_stage6_object_type_resolution_ignores_unused_query_objects():
	pipeline = LTL_BDI_Pipeline(domain_file=MARSROVER_DOMAIN_FILE)
	method_library = HTNMethodLibrary(
		compound_tasks=[],
		primitive_tasks=[],
		methods=[],
		target_literals=[HTNLiteral("at", ("rover0", "waypoint5"), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("at(rover0, waypoint5)", "move_rover")],
	)

	resolved = pipeline._stage6_object_types(
		("rover0", "waypoint1", "waypoint5"),
		method_library,
		("(at rover0 waypoint5)",),
	)

	assert resolved["rover0"] == "rover"
	assert resolved["waypoint5"] == "waypoint"
	assert "waypoint1" not in resolved


def _run_query_case(
	query_id: str,
	*,
	query_cases: Dict[str, Dict[str, Any]],
	domain_file: str,
) -> Dict[str, Any]:
	if query_id not in query_cases:
		raise KeyError(
			f"Unknown query id '{query_id}'. Available query ids: {sorted(query_cases)}",
		)

	case = query_cases[query_id]
	domain_action_names = set(HDDLParser.parse_domain(domain_file).get_action_names())
	pipeline = LTL_BDI_Pipeline(
		domain_file=domain_file,
		problem_file=case.get("problem_file"),
	)
	test_logs_dir = Path(__file__).parent / "logs"
	pipeline.logger = PipelineLogger(logs_dir=str(test_logs_dir), run_origin="tests")

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

	if (log_dir / "generated_code.asl").exists():
		bug_messages.append("unexpected legacy generated_code.asl artifact exists")

	if execution["natural_language"] != case["instruction"]:
		bug_messages.append("execution.json natural_language does not match selected query")
	if execution.get("run_origin") != "tests":
		bug_messages.append("execution.json run_origin is not tests")

	for stage_key in (
		"stage1_status",
		"stage2_status",
		"stage3_status",
		"stage4_status",
		"stage5_status",
		"stage6_status",
	):
		if execution.get(stage_key) != "success":
			bug_messages.append(f"{stage_key} is not success")

	stage3_metadata = execution.get("stage3_metadata", {}) or {}
	if not (
		execution.get("stage3_used_llm") is True
		or stage3_metadata.get("domain_projection_used") is True
	):
		bug_messages.append("Stage 3 recorded neither live LLM usage nor domain projection")

	if execution.get("stage4_backend") != "pandaPI":
		bug_messages.append("Stage 4 backend is not pandaPI")

	stage3_library = execution.get("stage3_method_library") or {}
	target_bindings = stage3_library.get("target_task_bindings") or []
	if not target_bindings:
		bug_messages.append("Stage 3 produced no target_task_bindings")
	bug_messages.extend(_binding_semantic_messages(stage3_library))

	target_literals = stage3_metadata.get("target_literals", [])
	for required_target_literal in case.get("required_target_literals", []):
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
	if "STAGE 6: AgentSpeak → Jason Runtime Validation" not in execution_txt:
		bug_messages.append("execution.txt is missing Stage 6 section")

	stage6_artifacts = execution.get("stage6_artifacts") or {}
	if stage6_artifacts.get("status") != "success":
		bug_messages.append("Stage 6 status payload is not success")
	if stage6_artifacts.get("backend") != "RunLocalMAS":
		bug_messages.append("Stage 6 backend is not RunLocalMAS")
	if stage6_artifacts.get("timed_out"):
		bug_messages.append("Stage 6 run timed out")
	stage6_stdout = stage6_artifacts.get("stdout") or ""
	if "execute success" not in stage6_stdout:
		bug_messages.append("Stage 6 stdout missing success marker")
	if "execute failed" in stage6_stdout:
		bug_messages.append("Stage 6 stdout contains failure marker")
	action_path = stage6_artifacts.get("action_path") or []
	if not isinstance(action_path, list):
		bug_messages.append("Stage 6 action_path is not a list")
	minimum_action_count = case.get("minimum_action_count")
	if isinstance(minimum_action_count, int) and len(action_path) < minimum_action_count:
		bug_messages.append(
			f"Stage 6 action_path shorter than expected minimum {minimum_action_count}: {action_path}",
		)
	expected_action_path = case.get("expected_action_path")
	if expected_action_path is not None and action_path != expected_action_path:
		bug_messages.append(
			f"Stage 6 action_path mismatch: expected {expected_action_path}, got {action_path}",
		)
	for action_step in action_path:
		match = re.match(r"^([^\s(]+)\(", action_step)
		if match is None:
			bug_messages.append(f"Stage 6 action_path step has invalid format: {action_step}")
			continue
		if match.group(1) not in domain_action_names:
			bug_messages.append(
				f"Stage 6 action_path step is not a domain action: {action_step}",
			)
	action_path_file = log_dir / "action_path.txt"
	if action_path_file.exists():
		file_actions = [line.strip() for line in action_path_file.read_text().splitlines() if line.strip()]
		if file_actions != action_path:
			bug_messages.append("Stage 6 action_path.txt does not match execution.json action_path")
	if (log_dir / "jason_runner_agent.asl").exists():
		bug_messages.append("legacy runtime-only jason_runner_agent.asl artifact still present")

	official_verifier_report = None
	problem_file = case.get("problem_file")
	if problem_file:
		official_verifier_report = IPC_PLAN_VERIFIER.verify_primitive_plan(
			domain_file=domain_file,
			problem_file=problem_file,
			action_path=action_path,
			output_dir=log_dir,
		)
		if not official_verifier_report.tool_available:
			bug_messages.append("official IPC verifier is not available on PATH")
		elif official_verifier_report.primitive_plan_executable is not True:
			bug_messages.append(
				"official IPC primitive-plan verifier did not accept the runtime action_path",
			)

	for path in _required_artifact_paths(log_dir):
		if not path.exists():
			bug_messages.append(f"missing log artifact: {path.name}")

	return {
		"query_id": query_id,
		"case": case,
		"result": result,
		"log_dir": log_dir,
		"execution": execution,
		"official_verifier": (
			official_verifier_report.to_dict()
			if official_verifier_report is not None
			else None
		),
		"bug_messages": bug_messages,
		"has_bug": bool(bug_messages),
	}


def _write_legacy_style_blocksworld_domain(tmp_path: Path) -> str:
	domain_file = tmp_path / "legacy_style_blocksworld.hddl"
	domain_file.write_text(
		"""
(define (domain blocksworld_legacy_style)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types block)
  (:predicates
    (holding ?x - block)
    (handempty)
    (ontable ?x - block)
    (on ?x - block ?y - block)
    (clear ?x - block)
  )
  (:action pick-up
    :parameters (?x - block ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (clear ?x)) (not (handempty)))
  )
  (:action pick-up-from-table
    :parameters (?x - block)
    :precondition (and (ontable ?x) (clear ?x) (handempty))
    :effect (and (holding ?x) (not (ontable ?x)) (not (clear ?x)) (not (handempty)))
  )
  (:action put-on-block
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y) (clear ?x) (not (holding ?x)) (not (clear ?y)) (handempty))
  )
  (:action put-down
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (ontable ?x) (clear ?x) (not (holding ?x)) (handempty))
  )
)
		""".strip(),
	)
	return str(domain_file)


def test_method_validation_initial_facts_are_branch_specific(tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=_write_legacy_style_blocksworld_domain(tmp_path))
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
	assert any(fact.startswith("(on a ") for fact in facts)


def test_method_validation_initial_facts_avoid_conflicting_global_defaults(tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=_write_legacy_style_blocksworld_domain(tmp_path))
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


def test_task_witness_initial_facts_merge_sibling_branches(tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=_write_legacy_style_blocksworld_domain(tmp_path))
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
				context=(HTNLiteral("clear", ("BLOCK2",), True, None),),
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


def test_method_validation_initial_facts_allocate_typed_witness_objects(tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=_write_legacy_style_blocksworld_domain(tmp_path))
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

	holding_facts = [fact for fact in facts if fact.startswith("(holding ")]
	assert holding_facts
	holding_obj = holding_facts[0].split()[1].rstrip(")")
	assert holding_obj in object_pool


@pytest.mark.parametrize("query_id", _pytest_selected_query_ids())
def test_blocksworld_pipeline_query_case(query_id: str):
	_ensure_live_dependencies()
	report = _run_query_case(
		query_id,
		query_cases=QUERY_CASES,
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
	)
	assert report["has_bug"] is False, "\n".join(report["bug_messages"])


@pytest.mark.parametrize("query_id", _pytest_selected_rover_query_ids())
def test_marsrover_pipeline_query_case(query_id: str):
	if not _is_rover_live_enabled():
		pytest.skip("Set PIPELINE_TEST_ROVER=1 to enable MarsRover live query tests")
	if not Path(MARSROVER_DOMAIN_FILE).exists():
		pytest.skip(f"MarsRover domain file missing: {MARSROVER_DOMAIN_FILE}")
	_ensure_live_dependencies()
	report = _run_query_case(
		query_id,
		query_cases=MARSROVER_QUERY_CASES,
		domain_file=MARSROVER_DOMAIN_FILE,
	)
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
	official_verifier = report.get("official_verifier") or {}
	if official_verifier:
		print(
			"Official IPC Primitive Verification: "
			f"primitive_plan_executable={official_verifier.get('primitive_plan_executable')}, "
			f"verification_result={official_verifier.get('verification_result')}",
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
	if not IPC_PLAN_VERIFIER.tool_available():
		print("Live pipeline CLI requires the official pandaPIparser verifier on PATH.")
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

	reports = [
		_run_query_case(
			query_id,
			query_cases=QUERY_CASES,
			domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		)
		for query_id in query_ids
	]
	for report in reports:
		_print_cli_report(report)

	return 1 if any(report["has_bug"] for report in reports) else 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv))
