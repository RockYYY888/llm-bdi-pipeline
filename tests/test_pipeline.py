"""
Live end-to-end acceptance harness for reverse-generated official benchmark queries.

This file is the canonical acceptance entry point:
- pytest uses it only for end-to-end verification
- CLI can run `python tests/test_pipeline.py query_2`, `all`, or `list`
- current live query cases are reverse-generated from official IPC benchmark
  problems for Blocksworld `p01`-`p03` and Marsrover `pfile01`-`pfile03`
  into single-sentence natural-language instructions
- non-E2E pipeline unit tests live in `tests/test_pipeline_units.py`
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
from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import (
	LTLFormula,
	LTLSpecification,
	LogicalOperator,
	TemporalOperator,
)
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


def _task_invocation_to_query_clause(task_name: str, args: List[str]) -> str:
	if not args:
		return f"{task_name}()"
	return f"{task_name}({', '.join(args)})"


def _typed_object_phrase(problem: Any) -> str:
	return _typed_object_phrase_for_objects(problem, problem.objects)


def _typed_object_phrase_for_objects(problem: Any, objects: List[str]) -> str:
	if not objects:
		return "Using the task arguments"

	grouped_objects: Dict[str, List[str]] = {}
	type_order: List[str] = []
	for obj in objects:
		object_type = problem.object_types.get(obj) or "object"
		if object_type not in grouped_objects:
			grouped_objects[object_type] = []
			type_order.append(object_type)
		grouped_objects[object_type].append(obj)

	if len(type_order) == 1 and type_order[0] != "object":
		object_type = type_order[0]
		type_phrase = object_type if object_type.endswith("s") else f"{object_type}s"
		return f"Using {type_phrase} {_serialise_nl_list(grouped_objects[object_type])}"

	group_phrases = []
	for object_type in type_order:
		members = grouped_objects[object_type]
		if object_type == "object":
			group_phrases.append(_serialise_nl_list(members))
			continue
		type_phrase = object_type if len(members) == 1 else (
			object_type if object_type.endswith("s") else f"{object_type}s"
		)
		group_phrases.append(f"{type_phrase} {_serialise_nl_list(members)}")
	return f"Using {_serialise_nl_list(group_phrases)}"


def _build_case_from_problem(problem_path: Path) -> Dict[str, Any] | None:
	problem = HDDLParser.parse_problem(str(problem_path))
	task_clauses = [
		_task_invocation_to_query_clause(invocation.task_name, invocation.args)
		for invocation in problem.htn_tasks
	]
	if not task_clauses:
		return None

	return {
		"instruction": (
			f"{_typed_object_phrase(problem)}, complete the tasks "
			f"{_serialise_nl_list(task_clauses)}."
		),
		"required_task_clauses": task_clauses,
		"problem_file": str(problem_path.resolve()),
		"minimum_action_count": 1,
		"description": f"Auto-generated from {problem_path.name} ({problem.name})",
	}


def _load_problem_query_cases(
	problem_dir: Path,
	*,
	limit: int = 3,
) -> Dict[str, Dict[str, Any]]:
	if not problem_dir.exists():
		return {}
	cases: Dict[str, Dict[str, Any]] = {}
	problem_paths = sorted(problem_dir.glob("p*.hddl"))[:limit]
	for index, problem_path in enumerate(problem_paths, start=1):
		case = _build_case_from_problem(problem_path)
		if case is None:
			continue
		cases[f"query_{index}"] = case
	return cases


BLOCKSWORLD_QUERY_CASES: Dict[str, Dict[str, Any]] = _load_problem_query_cases(
	BLOCKSWORLD_PROBLEM_DIR,
	limit=3,
)
MARSROVER_QUERY_CASES: Dict[str, Dict[str, Any]] = _load_problem_query_cases(
	MARSROVER_PROBLEM_DIR,
	limit=3,
)
QUERY_CASES: Dict[str, Dict[str, Any]] = BLOCKSWORLD_QUERY_CASES


def _expected_execution_identity(
	domain_file: str,
	problem_file: str | None,
) -> Dict[str, str]:
	expected = {
		"domain_name": HDDLParser.parse_domain(domain_file).name,
	}
	if problem_file:
		expected["problem_name"] = HDDLParser.parse_problem(problem_file).name
	return expected


def _pytest_selected_case_ids(
	case_map: Dict[str, Dict[str, Any]],
	*,
	query_env: str,
	all_env: str,
	default_query: str | None,
) -> List[str]:
	"""Default to a single live query; opt in to full sweep explicitly."""
	run_all = os.getenv(all_env, "").lower() in {"1", "true", "yes"}
	if run_all:
		return sorted(case_map)

	if default_query is None:
		query_id = os.getenv(query_env, "")
		if not query_id:
			return sorted(case_map)
	else:
		query_id = os.getenv(query_env, default_query)
	if query_id not in case_map:
		if default_query is None:
			return sorted(case_map)
		query_id = default_query
	return [query_id]


def _pytest_selected_query_ids() -> List[str]:
	return _pytest_selected_case_ids(
		BLOCKSWORLD_QUERY_CASES,
		query_env="PIPELINE_TEST_QUERY",
		all_env="PIPELINE_TEST_ALL",
		default_query=None,
	)


def _pytest_selected_marsrover_query_ids() -> List[str]:
	return _pytest_selected_case_ids(
		MARSROVER_QUERY_CASES,
		query_env="PIPELINE_TEST_MARSROVER_QUERY",
		all_env="PIPELINE_TEST_MARSROVER_ALL",
		default_query=None,
	)


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
		log_dir / "method_trace.json",
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

def assert_stage3_summary_preserves_llm_timing_metadata(tmp_path, monkeypatch):
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
			query_task_anchors=None,
			negation_hints=None,
			ordered_literal_signatures=None,
			query_objects=None,
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

		def extract_progressing_transitions(
			self,
			grounding_map,
			dfa_result,
			*,
			ordered_literal_signatures=None,
		):
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


def assert_pipeline_requires_explicit_domain_file():
	with pytest.raises(ValueError, match="domain_file is required"):
		LTL_BDI_Pipeline(domain_file=None)  # type: ignore[arg-type]


def assert_seed_validation_scope_preserves_multi_type_object_assignments(tmp_path):
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


def assert_seed_validation_scope_fails_for_ambiguous_parent_type(tmp_path):
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


def assert_stage3_type_validation_fails_for_untyped_method_variable(tmp_path):
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


def assert_stage1_object_universe_merges_constants_from_atoms_and_formulas():
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


def assert_ordered_literal_signatures_extracts_eventually_wrapped_atoms():
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


def assert_official_blocksworld_problem_query_case_generation_from_problem_tasks():
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

	case = _build_case_from_problem(problem_path)
	assert case is not None
	assert case["instruction"] == (
		"Using blocks b1, b2, b3, b4, and b5, "
		"complete the tasks do_put_on(b4, b2), do_put_on(b1, b4), and do_put_on(b3, b1)."
	)
	assert case["required_task_clauses"] == [
		"do_put_on(b4, b2)",
		"do_put_on(b1, b4)",
		"do_put_on(b3, b1)",
	]
	assert case["problem_file"] == str(problem_path.resolve())


def assert_official_marsrover_problem_query_case_generation_from_problem_tasks():
	problem_path = (
		Path(__file__).parent.parent
		/ "src"
		/ "domains"
		/ "marsrover"
		/ "problems"
		/ "pfile01.hddl"
	)
	if not problem_path.exists():
		pytest.skip(f"Missing marsrover problem file: {problem_path}")

	case = _build_case_from_problem(problem_path)
	assert case is not None
	assert case["instruction"] == (
		"Using lander general, modes colour, high_res, and low_res, rover rover0, "
		"store rover0store, waypoints waypoint0, waypoint1, waypoint2, and waypoint3, "
		"camera camera0, and objectives objective0 and objective1, complete the tasks "
		"get_soil_data(waypoint2), get_rock_data(waypoint3), and "
		"get_image_data(objective1, high_res)."
	)
	assert case["required_task_clauses"] == [
		"get_soil_data(waypoint2)",
		"get_rock_data(waypoint3)",
		"get_image_data(objective1, high_res)",
	]
	assert case["problem_file"] == str(problem_path.resolve())


def assert_query_task_anchor_extraction_uses_declared_tasks_only():
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	anchors = pipeline._extract_query_task_anchors(
		"Using blocks b1 and b2, complete the tasks do_put_on(b1, b2), "
		"invented_task(b2), and do_clear(b1).",
	)

	assert anchors == (
		{"task_name": "do_put_on", "args": ["b1", "b2"]},
		{"task_name": "do_clear", "args": ["b1"]},
	)


def assert_expected_execution_identity_is_derived_from_selected_domain_and_problem():
	blocksworld_identity = _expected_execution_identity(
		OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		BLOCKSWORLD_QUERY_CASES["query_1"]["problem_file"],
	)
	marsrover_identity = _expected_execution_identity(
		MARSROVER_DOMAIN_FILE,
		MARSROVER_QUERY_CASES["query_1"]["problem_file"],
	)

	assert blocksworld_identity == {
		"domain_name": "BLOCKS",
		"problem_name": "BW-rand-5",
	}
	assert marsrover_identity == {
		"domain_name": "rover",
		"problem_name": "roverprob1234",
	}


def assert_stage1_generation_uses_only_instruction_even_with_problem_file(monkeypatch):
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

	captured: Dict[str, Any] = {}

	def fake_generate(self, nl_instruction):
		captured["instruction"] = nl_instruction
		spec = LTLSpecification()
		spec.add_formula(LTLFormula(None, "true", [], None))
		spec.objects = ["b1", "b2", "b3", "b4", "b5"]
		spec.grounding_map = GroundingMap()
		spec.source_instruction = nl_instruction
		return spec, {"system": "stub", "user": "stub"}, "{\"ok\": true}"

	monkeypatch.setattr(NLToLTLfGenerator, "generate", fake_generate)

	pipeline = LTL_BDI_Pipeline(
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
	)
	spec = pipeline._stage1_parse_nl(
		"Using blocks b1, b2, b3, b4, and b5, arrange them so that b4 is on b2, "
		"and b1 is on b4, and b3 is on b1.",
	)

	assert captured["instruction"] == spec.source_instruction
	assert "p01.hddl" not in captured["instruction"]
	assert "(on b4 b2)" not in captured["instruction"]


def assert_stage6_object_type_resolution_ignores_unused_query_objects():
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
	expected_identity = _expected_execution_identity(
		domain_file,
		case.get("problem_file"),
	)

	bug_messages: List[str] = []

	if not result["success"]:
		bug_messages.append("pipeline returned success=False")

	if (log_dir / "generated_code.asl").exists():
		bug_messages.append("unexpected deprecated generated_code.asl artifact exists")

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
		"stage7_status",
	):
		if execution.get(stage_key) != "success":
			bug_messages.append(f"{stage_key} is not success")

	for key, expected_value in expected_identity.items():
		if execution.get(key) != expected_value:
			bug_messages.append(
				f"execution.json {key} mismatch: expected {expected_value}, "
				f"got {execution.get(key)}",
			)
	log_dir_name = log_dir.name
	if expected_identity.get("domain_name") and expected_identity.get("problem_name"):
		expected_suffix = (
			f"_{expected_identity['domain_name']}_{expected_identity['problem_name']}"
		)
		if not log_dir_name.endswith(expected_suffix):
			bug_messages.append(
				f"log directory name does not end with {expected_suffix}: {log_dir_name}",
			)

	stage3_metadata = execution.get("stage3_metadata", {}) or {}
	if execution.get("stage3_used_llm") is not True:
		bug_messages.append("Stage 3 did not record live LLM synthesis")

	if execution.get("stage4_backend") != "pandaPI":
		bug_messages.append("Stage 4 backend is not pandaPI")

	stage3_library = execution.get("stage3_method_library") or {}
	target_bindings = stage3_library.get("target_task_bindings") or []
	if not target_bindings:
		bug_messages.append("Stage 3 produced no target_task_bindings")
	bug_messages.extend(_binding_semantic_messages(stage3_library))

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
		bug_messages.append("deprecated PANDA-only task plan section still present in Stage 5 code")
	if "target_label(" in stage5_code:
		bug_messages.append("deprecated target_label facts still present in Stage 5 code")
	if "+!transition_" in stage5_code:
		bug_messages.append("deprecated transition_i wrappers still present in Stage 5 code")
	bug_messages.extend(_method_free_variable_messages(stage5_code))

	for binding in target_bindings:
		task_name = binding["task_name"]
		if task_name.startswith(BANNED_TASK_PREFIXES):
			bug_messages.append(f"deprecated task prefix still present: {task_name}")
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
	if "STAGE 7: Official IPC HTN Plan Verification" not in execution_txt:
		bug_messages.append("execution.txt is missing Stage 7 section")

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
	method_trace = stage6_artifacts.get("method_trace") or []
	if not isinstance(method_trace, list):
		bug_messages.append("Stage 6 method_trace is not a list")
	method_trace_file = log_dir / "method_trace.json"
	if method_trace_file.exists():
		file_trace = json.loads(method_trace_file.read_text())
		if file_trace != method_trace:
			bug_messages.append("Stage 6 method_trace.json does not match execution.json method_trace")
	if (log_dir / "jason_runner_agent.asl").exists():
		bug_messages.append("deprecated runtime-only jason_runner_agent.asl artifact still present")

	stage7_artifacts = execution.get("stage7_artifacts") or {}
	if stage7_artifacts.get("tool_available") is not True:
		bug_messages.append("official IPC verifier is not available on PATH")
	if stage7_artifacts.get("plan_kind") != "hierarchical":
		bug_messages.append("official IPC verifier did not validate a hierarchical plan")
	if stage7_artifacts.get("verification_result") is not True:
		bug_messages.append("official IPC HTN verifier did not accept the generated plan")
	if stage7_artifacts.get("primitive_plan_executable") is not True:
		bug_messages.append("official IPC verifier did not mark the primitive plan as executable")
	if stage7_artifacts.get("reached_goal_state") is not True:
		bug_messages.append("official IPC verifier did not report goal-state achievement")

	for path in _required_artifact_paths(log_dir):
		if not path.exists():
			bug_messages.append(f"missing log artifact: {path.name}")

	return {
		"query_id": query_id,
		"case": case,
		"result": result,
		"log_dir": log_dir,
		"execution": execution,
		"official_verifier": stage7_artifacts or None,
		"bug_messages": bug_messages,
		"has_bug": bool(bug_messages),
	}


def assert_method_validation_initial_facts_are_branch_specific(tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
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
				action_name="unstack",
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


def assert_method_validation_initial_facts_avoid_conflicting_global_defaults(tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
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
				action_name="stack",
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


def assert_task_witness_initial_facts_merge_sibling_branches(tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
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
						action_name="pick-up",
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


def assert_transition_witness_initial_facts_exclude_positive_target_literal(tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	initial_facts = (
		"(clear b)",
		"(ontable a)",
		"(on a b)",
	)
	filtered = pipeline._transition_witness_initial_facts(
		initial_facts,
		HTNLiteral("on", ("a", "b"), True, None),
	)

	assert filtered == (
		"(clear b)",
		"(ontable a)",
	)


def assert_method_validation_initial_facts_allocate_typed_witness_objects(tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
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


def assert_stage7_prefers_planned_hierarchical_verification(tmp_path, monkeypatch):
	calls: list[str] = []

	class FakeResult:
		def __init__(self):
			self.tool_available = True
			self.command = ["planned"]
			self.plan_file = str(tmp_path / "ipc_official_plan.txt")
			self.output_file = str(tmp_path / "ipc_official_verifier.txt")
			self.stdout = "Plan is executable: true\nPlan verification result: true\n"
			self.stderr = ""
			self.primitive_plan_only = False
			self.primitive_plan_executable = True
			self.verification_result = True
			self.reached_goal_state = True
			self.plan_kind = "hierarchical"
			self.build_warning = None
			self.error = None

		def to_dict(self):
			return {
				"tool_available": self.tool_available,
				"command": list(self.command),
				"plan_file": self.plan_file,
				"output_file": self.output_file,
				"stdout": self.stdout,
				"stderr": self.stderr,
				"primitive_plan_only": self.primitive_plan_only,
				"primitive_plan_executable": self.primitive_plan_executable,
				"verification_result": self.verification_result,
				"reached_goal_state": self.reached_goal_state,
				"plan_kind": self.plan_kind,
				"build_warning": self.build_warning,
				"error": self.error,
			}

	class FakeVerifier:
		def tool_available(self):
			return True

		def planning_toolchain_available(self):
			return True

		def verify_planned_hierarchical_plan(self, **kwargs):
			calls.append("planned")
			return FakeResult()

		def verify_plan(self, **kwargs):
			raise AssertionError(
				"trace-based verification should not run when the planner toolchain is available"
			)

	monkeypatch.setattr(pipeline_module, "IPCPlanVerifier", FakeVerifier)

	pipeline = LTL_BDI_Pipeline(
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		problem_file=str((BLOCKSWORLD_PROBLEM_DIR / "p01.hddl").resolve()),
	)
	pipeline.output_dir = Path(tmp_path)
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("do_put_on", ("X", "Y"), False, ("on",))],
		primitive_tasks=[],
		methods=[],
		target_literals=[],
		target_task_bindings=[],
	)

	result = pipeline._stage7_official_verification(
		ltl_spec=None,
		method_library=method_library,
		stage6_data={"artifacts": {"action_path": [], "method_trace": []}},
	)

	assert calls == ["planned"]
	assert result is not None
	assert result["summary"]["status"] == "success"
	assert result["artifacts"]["plan_kind"] == "hierarchical"


def assert_method_validation_initial_facts_ground_lowercase_schema_variables(tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	planner = PANDAPlanner()
	method = HTNMethod(
		method_name="m_do_move_unstack",
		task_name="do_move",
		parameters=("x", "y", "aux"),
		task_args=("x", "y"),
		context=(
			HTNLiteral("on", ("x", "aux"), True, None),
			HTNLiteral("clear", ("x",), True, None),
			HTNLiteral("handempty", (), True, None),
			HTNLiteral("clear", ("y",), True, None),
		),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="unstack",
				args=("x", "aux"),
				kind="primitive",
				action_name="unstack",
			),
			HTNMethodStep(
				step_id="s2",
				task_name="stack",
				args=("x", "y"),
				kind="primitive",
				action_name="stack",
			),
		),
		ordering=(("s1", "s2"),),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_move", ("x", "y"), False, ("on",)),
		],
		methods=[method],
	)

	facts = pipeline._method_validation_initial_facts(
		planner,
		method,
		method_library,
		("a", "b"),
		("a", "b", "c"),
		object_pool=["a", "b", "c"],
		object_types={"a": "block", "b": "block", "c": "block"},
	)

	assert "(on a c)" in facts
	assert "(clear a)" in facts
	assert "(clear b)" in facts
	assert "(handempty)" in facts
	assert all(" x" not in fact and " y" not in fact and " aux" not in fact for fact in facts)


@pytest.mark.parametrize("query_id", _pytest_selected_query_ids())
def test_blocksworld_pipeline_query_case(query_id: str):
	_ensure_live_dependencies()
	report = _run_query_case(
		query_id,
		query_cases=BLOCKSWORLD_QUERY_CASES,
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
	)
	assert report["has_bug"] is False, "\n".join(report["bug_messages"])


@pytest.mark.parametrize("query_id", _pytest_selected_marsrover_query_ids())
def test_marsrover_pipeline_query_case(query_id: str):
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
			"Official IPC Verification: "
			f"plan_kind={official_verifier.get('plan_kind')}, "
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
		print("Usage: python tests/test_pipeline.py [query_i|all|list]")
		return 2

	selector = argv[1] if len(argv) == 2 else "query_1"
	if selector == "list":
		for query_id in sorted(QUERY_CASES):
			case = QUERY_CASES[query_id]
			print(f"{query_id}: {case['description']}")
			print(f"Instruction: {case['instruction']}")
			print("")
		return 0
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
