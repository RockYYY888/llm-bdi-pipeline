"""
Live end-to-end acceptance harness for the stored benchmark query dataset.

This file is the canonical acceptance entry point:
- pytest uses it only for end-to-end verification
- CLI can run `python tests/test_pipeline.py query_2`, `all`, or `list`
- current live query cases are loaded from a versioned stored query dataset whose
  entries are canonically reverse-generated from official IPC benchmark root HTN
  tasks and typed object inventories
- non-E2E pipeline unit tests live in `tests/test_pipeline_units.py`
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

import ltl_bdi_pipeline as pipeline_module
from ltl_bdi_pipeline import LTL_BDI_Pipeline, TypeResolutionError
from pipeline_artifacts import DomainBuildArtifact, persist_domain_build_artifact
from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import (
	LTLFormula,
	LTLSpecification,
	LogicalOperator,
	TemporalOperator,
)
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
from stage3_method_synthesis.htn_method_synthesis import (
	HTNMethodSynthesizer as RealHTNMethodSynthesizer,
)
from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTargetTaskBinding,
	HTNTask,
	_parse_invocation_signature,
	_parse_signature_literal,
)
from stage3_method_synthesis.htn_prompts import (
	build_domain_prompt_analysis_payload,
	build_prompt_analysis_payload,
)
from stage4_panda_planning.panda_planner import PANDAPlanner
from stage4_panda_planning.panda_schema import PANDAPlanResult, PANDAPlanStep
from stage5_agentspeak_rendering.agentspeak_renderer import AgentSpeakRenderer
from stage6_jason_validation.jason_runner import JasonRunner
from utils.config import get_config
from utils.benchmark_query_dataset import (
	DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS,
	build_case_from_problem as _build_case_from_problem,
	load_benchmark_query_dataset as _load_benchmark_query_dataset,
	load_problem_query_cases as _load_problem_query_cases,
	query_referenced_problem_objects as _query_referenced_problem_objects,
	serialise_nl_list as _serialise_nl_list,
	serialise_task_clause_sequence as _serialise_task_clause_sequence,
	task_invocation_to_query_clause as _task_invocation_to_query_clause,
	typed_object_phrase as _typed_object_phrase,
	typed_object_phrase_for_objects as _typed_object_phrase_for_objects,
)
from utils.hddl_condition_parser import HDDLConditionParser
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
SATELLITE_DOMAIN_FILE = str(
	(Path(__file__).parent.parent / "src" / "domains" / "satellite" / "domain.hddl").resolve(),
)
SATELLITE_PROBLEM_DIR = (
	Path(__file__).parent.parent / "src" / "domains" / "satellite" / "problems"
)
TRANSPORT_DOMAIN_FILE = str(
	(Path(__file__).parent.parent / "src" / "domains" / "transport" / "domain.hddl").resolve(),
)
TRANSPORT_PROBLEM_DIR = (
	Path(__file__).parent.parent / "src" / "domains" / "transport" / "problems"
)

BANNED_TASK_PREFIXES = ("achieve_", "maintain_not_", "ensure_", "goal_")
IPC_PLAN_VERIFIER = IPCPlanVerifier()

BLOCKSWORLD_OFFICIAL_TASK_SOURCE_PREDICATES = {
	"do_put_on": ("on",),
	"do_on_table": ("clear",),
	"do_move": ("on",),
	"do_clear": ("clear",),
}

MARSROVER_OFFICIAL_TASK_SOURCE_PREDICATES = {
	"calibrate_abs": ("calibrated",),
	"empty-store": ("empty",),
	"navigate_abs": ("at",),
	"get_image_data": ("communicated_image_data",),
	"get_rock_data": ("communicated_rock_data",),
	"get_soil_data": ("communicated_soil_data",),
	"send_image_data": ("communicated_image_data",),
	"send_rock_data": ("communicated_rock_data",),
	"send_soil_data": ("communicated_soil_data",),
}

OFFICIAL_STAGE3_MASK_TASK_SOURCE_PREDICATES: Dict[str, Dict[str, tuple[str, ...]]] = {
	"blocksworld": BLOCKSWORLD_OFFICIAL_TASK_SOURCE_PREDICATES,
	"marsrover": MARSROVER_OFFICIAL_TASK_SOURCE_PREDICATES,
	"satellite": {},
	"transport": {},
}

def _literal_signature(predicate: str, args: List[str], is_positive: bool = True) -> str:
	atom = predicate if not args else f"{predicate}({', '.join(args)})"
	return atom if is_positive else f"!{atom}"

FULL_BENCHMARK_QUERY_LIMIT = 10_000
TESTS_GENERATED_DIR = Path(__file__).parent / "generated"
TESTS_GENERATED_LOGS_DIR = TESTS_GENERATED_DIR / "logs"
TESTS_GENERATED_DOMAIN_BUILDS_DIR = TESTS_GENERATED_DIR / "domain_builds"
_OFFICIAL_STAGE3_MASK_DOMAIN_ARTIFACT_CACHE: Dict[str, Path] = {}


def _benchmark_query_id_sort_key(query_id: str) -> tuple[int, str]:
	match = re.fullmatch(r"query_(\d+)", str(query_id).strip())
	if match is None:
		return (10**9, str(query_id))
	return (int(match.group(1)), str(query_id))

BLOCKSWORLD_QUERY_CASES: Dict[str, Dict[str, Any]] = _load_problem_query_cases(
	BLOCKSWORLD_PROBLEM_DIR,
	limit=FULL_BENCHMARK_QUERY_LIMIT,
)
MARSROVER_QUERY_CASES: Dict[str, Dict[str, Any]] = _load_problem_query_cases(
	MARSROVER_PROBLEM_DIR,
	limit=FULL_BENCHMARK_QUERY_LIMIT,
)
SATELLITE_QUERY_CASES: Dict[str, Dict[str, Any]] = _load_problem_query_cases(
	SATELLITE_PROBLEM_DIR,
	limit=FULL_BENCHMARK_QUERY_LIMIT,
	pattern="*.hddl",
)
TRANSPORT_QUERY_CASES: Dict[str, Dict[str, Any]] = _load_problem_query_cases(
	TRANSPORT_PROBLEM_DIR,
	limit=FULL_BENCHMARK_QUERY_LIMIT,
)
QUERY_CASES: Dict[str, Dict[str, Any]] = BLOCKSWORLD_QUERY_CASES
CLI_QUERY_CASE_GROUPS: Dict[str, Dict[str, Dict[str, Any]]] = {
	"blocksworld": BLOCKSWORLD_QUERY_CASES,
	"marsrover": MARSROVER_QUERY_CASES,
	"satellite": SATELLITE_QUERY_CASES,
	"transport": TRANSPORT_QUERY_CASES,
}
CLI_DOMAIN_FILES: Dict[str, str] = {
	"blocksworld": OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
	"marsrover": MARSROVER_DOMAIN_FILE,
	"satellite": SATELLITE_DOMAIN_FILE,
	"transport": TRANSPORT_DOMAIN_FILE,
}

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
		return sorted(case_map, key=_benchmark_query_id_sort_key)

	if default_query is None:
		query_id = os.getenv(query_env, "")
		if not query_id:
			return sorted(case_map, key=_benchmark_query_id_sort_key)
	else:
		query_id = os.getenv(query_env, default_query)
	if query_id not in case_map:
		if default_query is None:
			return sorted(case_map, key=_benchmark_query_id_sort_key)
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

def _resolve_cli_selection(argv: List[str]) -> tuple[str, Dict[str, Dict[str, Any]], str]:
	if len(argv) > 3:
		raise ValueError(
			"Usage: python tests/test_pipeline.py "
			"[blocksworld|marsrover|satellite|transport] [query_i|all|list] or "
			"[blocksworld:query_i|marsrover:all|satellite:all|transport:list]",
		)

	if len(argv) == 1:
		return "blocksworld", BLOCKSWORLD_QUERY_CASES, "query_1"

	first_arg = argv[1]
	if first_arg == "list":
		return "blocksworld", BLOCKSWORLD_QUERY_CASES, "list"
	if ":" in first_arg:
		domain_key, selector = first_arg.split(":", 1)
		if domain_key not in CLI_QUERY_CASE_GROUPS:
			raise ValueError(f"Unknown domain '{domain_key}'. Available: {sorted(CLI_QUERY_CASE_GROUPS)}")
		return domain_key, CLI_QUERY_CASE_GROUPS[domain_key], selector or "query_1"
	if first_arg in CLI_QUERY_CASE_GROUPS:
		selector = argv[2] if len(argv) == 3 else "all"
		return first_arg, CLI_QUERY_CASE_GROUPS[first_arg], selector
	return "blocksworld", BLOCKSWORLD_QUERY_CASES, first_arg

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
		log_dir / "dfa.dot",
		log_dir / "stage5_agentspeak.asl",
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

def _resolve_log_artifact_path(log_dir: Path, relative_path: Any) -> Path | None:
	if not relative_path:
		return None
	candidate = Path(str(relative_path))
	if candidate.is_absolute():
		return candidate
	return log_dir / candidate

def _read_json_artifact(log_dir: Path, relative_path: Any) -> Any:
	artifact_path = _resolve_log_artifact_path(log_dir, relative_path)
	if artifact_path is None or not artifact_path.exists():
		return None
	return json.loads(artifact_path.read_text())

def _read_text_artifact(log_dir: Path, relative_path: Any) -> str:
	artifact_path = _resolve_log_artifact_path(log_dir, relative_path)
	if artifact_path is None or not artifact_path.exists():
		return ""
	return artifact_path.read_text()

def _load_stage3_method_library(execution: Dict[str, Any], log_dir: Path) -> Dict[str, Any]:
	stage3_payload = execution.get("stage3_method_library")
	if not isinstance(stage3_payload, dict):
		return {}
	artifact_path = stage3_payload.get("artifact_path")
	if artifact_path:
		loaded = _read_json_artifact(log_dir, artifact_path)
		if isinstance(loaded, dict):
			return loaded
	return stage3_payload

def _load_stage5_agentspeak_code(execution: Dict[str, Any], log_dir: Path) -> str:
	stage5_payload = execution.get("stage5_agentspeak")
	if not isinstance(stage5_payload, str) or not stage5_payload:
		return ""
	if "\n" in stage5_payload or "/* " in stage5_payload:
		return stage5_payload
	return _read_text_artifact(log_dir, stage5_payload)

def _load_stage6_artifacts(execution: Dict[str, Any], log_dir: Path) -> Dict[str, Any]:
	stage6_payload = execution.get("stage6_artifacts")
	if not isinstance(stage6_payload, dict):
		return {}
	loaded = dict(stage6_payload)
	if "stdout" not in loaded and loaded.get("stdout_path"):
		loaded["stdout"] = _read_text_artifact(log_dir, loaded["stdout_path"])
	if "stderr" not in loaded and loaded.get("stderr_path"):
		loaded["stderr"] = _read_text_artifact(log_dir, loaded["stderr_path"])
	if "action_path" not in loaded and loaded.get("action_path_path"):
		action_path_text = _read_text_artifact(log_dir, loaded["action_path_path"])
		loaded["action_path"] = [
			line.strip()
			for line in action_path_text.splitlines()
			if line.strip()
		]
	if "method_trace" not in loaded and loaded.get("method_trace_path"):
		method_trace = _read_json_artifact(log_dir, loaded["method_trace_path"])
		if isinstance(method_trace, list):
			loaded["method_trace"] = method_trace
	return loaded

def _load_stage7_artifacts(execution: Dict[str, Any], log_dir: Path) -> Dict[str, Any]:
	stage7_payload = execution.get("stage7_artifacts")
	if not isinstance(stage7_payload, dict):
		return {}
	loaded = _read_json_artifact(log_dir, "ipc_official_verification.json")
	if isinstance(loaded, dict):
		merged = dict(loaded)
		merged.update(stage7_payload)
		return merged
	return dict(stage7_payload)

def _literal_from_signature_text(signature: str) -> HTNLiteral:
	text = str(signature).strip()
	is_positive = not text.startswith("!")
	if not is_positive:
		text = text[1:].strip()
	if "(" not in text:
		return HTNLiteral(text, (), is_positive, None)
	predicate, _, remainder = text.partition("(")
	args_text = remainder[:-1] if remainder.endswith(")") else remainder
	args = tuple(arg.strip() for arg in args_text.split(",") if arg.strip())
	return HTNLiteral(predicate.strip(), args, is_positive, None)

def _official_hddl_method_library(
	*,
	domain_file: str,
	task_source_predicates: Dict[str, tuple[str, ...]],
	target_literal_signatures: List[str],
	query_task_anchors: List[Dict[str, Any]],
) -> HTNMethodLibrary:
	domain = HDDLParser.parse_domain(domain_file)
	condition_parser = HDDLConditionParser()
	synthesizer = RealHTNMethodSynthesizer()
	primitive_action_names = {action.name for action in domain.actions}
	compound_tasks = [
			HTNTask(
				name=task.name,
				parameters=tuple(condition_parser._extract_parameter_names(task.parameters)),
				is_primitive=False,
				source_predicates=(),
				source_name=task.name,
			)
			for task in domain.tasks
		]
	primitive_tasks = synthesizer._build_primitive_tasks(domain)
	methods: List[HTNMethod] = []
	for method in domain.methods:
		context = tuple(
			HTNLiteral(
				predicate=pattern.predicate,
				args=tuple(pattern.args),
				is_positive=pattern.is_positive,
				source_symbol=None,
			)
			for pattern in condition_parser.parse_literals(
				method.precondition,
				action_name=method.name,
				scope="method_precondition",
			)
		)
		subtasks = tuple(
			HTNMethodStep(
				step_id=step.label,
				task_name=(
					synthesizer._sanitize_name(step.task_name)
					if step.task_name in primitive_action_names
					else step.task_name
				),
				args=tuple(step.args),
				kind="primitive" if step.task_name in primitive_action_names else "compound",
				action_name=step.task_name if step.task_name in primitive_action_names else None,
			)
			for step in method.subtasks
		)
		methods.append(
			HTNMethod(
				method_name=method.name,
				task_name=method.task_name,
				parameters=tuple(condition_parser._extract_parameter_names(method.parameters)),
				task_args=tuple(method.task_args),
				context=context,
				subtasks=subtasks,
				ordering=tuple(tuple(edge) for edge in method.ordering),
				origin="official_hddl",
			),
		)

	method_library = HTNMethodLibrary(
		compound_tasks=compound_tasks,
		primitive_tasks=primitive_tasks,
		methods=methods,
	)
	inferred_source_predicates = _infer_official_task_source_predicates(
		domain=domain,
		method_library=method_library,
		target_literal_signatures=target_literal_signatures,
		query_task_anchors=query_task_anchors,
		fallback_source_predicates=task_source_predicates,
	)
	method_library.compound_tasks = [
		HTNTask(
			name=task.name,
			parameters=task.parameters,
			is_primitive=task.is_primitive,
			source_predicates=inferred_source_predicates.get(task.name, task.source_predicates),
			source_name=task.source_name,
		)
		for task in method_library.compound_tasks
	]
	target_literals, target_task_bindings = _official_task_headline_targets(
		domain=domain,
		method_library=method_library,
		target_literal_signatures=target_literal_signatures,
		query_task_anchors=query_task_anchors,
	)
	method_library.target_literals = target_literals
	method_library.target_task_bindings = target_task_bindings
	return method_library

def _infer_official_task_source_predicates(
	*,
	domain: Any,
	method_library: HTNMethodLibrary,
	target_literal_signatures: List[str],
	query_task_anchors: List[Dict[str, Any]],
	fallback_source_predicates: Dict[str, tuple[str, ...]],
) -> Dict[str, tuple[str, ...]]:
	renderer = AgentSpeakRenderer()
	task_lookup = {
		task.name: task
		for task in method_library.compound_tasks + method_library.primitive_tasks
	}
	methods_by_task: Dict[str, List[HTNMethod]] = {}
	for method in method_library.methods:
		methods_by_task.setdefault(method.task_name, []).append(method)
	task_render_specs = renderer._build_task_render_specs(domain, method_library)
	effect_cache: Dict[str, tuple[Any, ...]] = {}
	root_target_signatures_by_task: Dict[str, set[str]] = {}
	for signature, anchor in zip(target_literal_signatures, query_task_anchors):
		task_name = str(anchor.get("task_name") or "").strip()
		if not task_name:
			continue
		root_target_signatures_by_task.setdefault(task_name, set()).add(signature)

	inferred: Dict[str, tuple[str, ...]] = {}
	for task in method_library.compound_tasks:
		candidate_literals = [
			literal
			for literal in renderer._compound_task_effect_templates(
				task.name,
				task_lookup,
				methods_by_task,
				task_render_specs,
				effect_cache,
			)
			if getattr(literal, "is_positive", True)
		]
		if not candidate_literals:
			if task.name in fallback_source_predicates:
				inferred[task.name] = fallback_source_predicates[task.name]
			continue

		task_parameters = tuple(task.parameters or ())
		task_parameter_set = set(task_parameters)
		root_targets = root_target_signatures_by_task.get(task.name, set())
		candidate_scores: Dict[str, Tuple[int, int, int, int, int]] = {}
		candidate_predicates: Dict[str, str] = {}

		for literal in candidate_literals:
			signature = renderer._literal_signature(literal)
			shape_signature = _literal_shape_signature(literal)
			negated_context_match = 0
			pure_noop_positive_match = 0
			for method in methods_by_task.get(task.name, ()):
				task_binding_parameters = renderer._task_binding_parameters(
					method,
					len(task_parameters),
				)
				task_bindings = {
					parameter: task_parameter
					for parameter, task_parameter in zip(task_binding_parameters, task_parameters)
				}
				for context_literal in getattr(method, "context", ()) or ():
					lifted = renderer._lift_literal_to_task_scope(context_literal, task_bindings)
					if lifted is None:
						continue
					if _literal_shape_signature(lifted) != shape_signature:
						continue
					if not getattr(context_literal, "is_positive", True):
						negated_context_match = 1
					elif renderer._method_is_pure_noop(method):
						pure_noop_positive_match = 1

			non_task_variables = sum(
				1
				for arg in getattr(literal, "args", ()) or ()
				if renderer._looks_like_variable(arg) and arg not in task_parameter_set
			)
			task_parameter_coverage = len(
				{
					arg
					for arg in getattr(literal, "args", ()) or ()
					if arg in task_parameter_set
				},
			)
			score = (
				1 if signature in root_targets else 0,
				pure_noop_positive_match,
				negated_context_match,
				task_parameter_coverage,
				-non_task_variables,
			)
			existing = candidate_scores.get(signature)
			if existing is None or score > existing:
				candidate_scores[signature] = score
				candidate_predicates[signature] = str(getattr(literal, "predicate", "")).strip()

		if candidate_scores:
			best_score = max(candidate_scores.values())
			best_predicates = tuple(
				dict.fromkeys(
					candidate_predicates[signature]
					for signature, score in candidate_scores.items()
					if score == best_score and candidate_predicates.get(signature)
				),
			)
			fallback_predicates = tuple(
				predicate
				for predicate in fallback_source_predicates.get(task.name, ())
				if predicate in best_predicates
			)
			predicate_names = fallback_predicates or best_predicates
			if predicate_names:
				inferred[task.name] = predicate_names
				continue

		if task.name in fallback_source_predicates:
			inferred[task.name] = fallback_source_predicates[task.name]

	return inferred

def _literal_shape_signature(literal: Any) -> tuple[str, tuple[str, ...]]:
	return (
		str(getattr(literal, "predicate", "")).strip(),
		tuple(str(arg) for arg in (getattr(literal, "args", ()) or ())),
	)

def _official_task_headline_targets(
	*,
	domain: Any,
	method_library: HTNMethodLibrary,
	target_literal_signatures: List[str],
	query_task_anchors: List[Dict[str, Any]],
) -> tuple[List[HTNLiteral], List[HTNTargetTaskBinding]]:
	if not query_task_anchors:
		return (
			[_literal_from_signature_text(signature) for signature in target_literal_signatures],
			[],
		)

	renderer = AgentSpeakRenderer()
	task_lookup = {
		task.name: task
		for task in method_library.compound_tasks + method_library.primitive_tasks
	}
	methods_by_task: Dict[str, List[HTNMethod]] = {}
	for method in method_library.methods:
		methods_by_task.setdefault(method.task_name, []).append(method)
	task_render_specs = renderer._build_task_render_specs(domain, method_library)
	effect_cache: Dict[str, tuple[Any, ...]] = {}
	stage1_literal_signatures = set(target_literal_signatures)
	target_literals: List[HTNLiteral] = []
	target_task_bindings: List[HTNTargetTaskBinding] = []
	seen_signatures: set[str] = set()

	for anchor in query_task_anchors:
		task_name = anchor.get("task_name")
		if not task_name:
			continue
		task_schema = task_lookup.get(task_name)
		if task_schema is None:
			continue
		task_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
		anchor_args = tuple(anchor.get("args", ()) or ())
		bindings = {
			parameter: arg
			for parameter, arg in zip(task_parameters, anchor_args)
		}
		candidate_literals = [
			HTNLiteral(
				predicate=template.predicate,
				args=tuple(bindings.get(arg, arg) for arg in template.args),
				is_positive=template.is_positive,
				source_symbol=None,
			)
			for template in renderer._compound_task_effect_templates(
				task_name,
				task_lookup,
				methods_by_task,
				task_render_specs,
				effect_cache,
			)
		]
		if not stage1_literal_signatures:
			task_spec = task_render_specs.get(task_name, {})
			predicate_types = task_spec.get("predicate_types", {})
			task_types = tuple(task_spec.get("task_param_types", ()))
			primary_literal_signatures: set[str] = set()
			for predicate_name in (getattr(task_schema, "source_predicates", ()) or ()):
				projected_args = renderer._project_compound_effect_args(
					task_parameters,
					task_types,
					predicate_types.get(predicate_name, ()),
				)
				if projected_args is None:
					continue
				primary_literal_signatures.add(
					HTNLiteral(
						predicate=str(predicate_name).strip(),
						args=tuple(bindings.get(arg, arg) for arg in projected_args),
						is_positive=True,
						source_symbol=None,
					).to_signature(),
				)
			if primary_literal_signatures:
				candidate_literals = [
					literal
					for literal in candidate_literals
					if literal.to_signature() in primary_literal_signatures
				]
		if stage1_literal_signatures:
			candidate_literals = [
				literal
				for literal in candidate_literals
				if literal.to_signature() in stage1_literal_signatures
			]
		for literal in candidate_literals:
			signature = literal.to_signature()
			if signature in seen_signatures:
				continue
			seen_signatures.add(signature)
			target_literals.append(literal)
			target_task_bindings.append(HTNTargetTaskBinding(signature, task_name))

	if target_literals:
		return target_literals, target_task_bindings

	return (
		[_literal_from_signature_text(signature) for signature in target_literal_signatures],
		[
			HTNTargetTaskBinding(signature, anchor["task_name"])
			for signature, anchor in zip(target_literal_signatures, query_task_anchors)
			if anchor.get("task_name")
		],
	)

def assert_official_stage3_mask_infers_blocksworld_internal_task_headlines_generically():
	method_library = _official_hddl_method_library(
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		task_source_predicates=BLOCKSWORLD_OFFICIAL_TASK_SOURCE_PREDICATES,
		target_literal_signatures=["on(b3, b5)"],
		query_task_anchors=[{"task_name": "do_put_on", "args": ["b3", "b5"]}],
	)
	task_lookup = {
		task.name: task
		for task in method_library.compound_tasks
	}

	assert task_lookup["do_on_table"].source_predicates in {("ontable",), ("clear",)}
	assert task_lookup["do_clear"].source_predicates == ("clear",)
	assert task_lookup["do_move"].source_predicates == ("on",)

OFFICIAL_STAGE3_MASK_DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
	"blocksworld": {
		"domain_file": OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		"query_cases": BLOCKSWORLD_QUERY_CASES,
	},
	"marsrover": {
		"domain_file": MARSROVER_DOMAIN_FILE,
		"query_cases": MARSROVER_QUERY_CASES,
	},
	"satellite": {
		"domain_file": SATELLITE_DOMAIN_FILE,
		"query_cases": SATELLITE_QUERY_CASES,
	},
	"transport": {
		"domain_file": TRANSPORT_DOMAIN_FILE,
		"query_cases": TRANSPORT_QUERY_CASES,
	},
}

def _official_domain_method_library(
	domain_key: str,
	*,
	target_literal_signatures: List[str] | None = None,
	query_task_anchors: List[Dict[str, Any]] | None = None,
) -> HTNMethodLibrary:
	domain_config = OFFICIAL_STAGE3_MASK_DOMAIN_CONFIGS[domain_key]
	return _official_hddl_method_library(
		domain_file=domain_config["domain_file"],
		task_source_predicates=OFFICIAL_STAGE3_MASK_TASK_SOURCE_PREDICATES[domain_key],
		target_literal_signatures=target_literal_signatures or [],
		query_task_anchors=query_task_anchors or [],
	)

def _build_oracle_task_grounded_stage1_spec(
	pipeline: LTL_BDI_Pipeline,
	nl_instruction: str,
) -> LTLSpecification:
	query_object_inventory = tuple(pipeline._extract_query_object_inventory(nl_instruction))
	query_task_anchors = tuple(pipeline._extract_query_task_anchors(nl_instruction))
	if not query_task_anchors:
		raise ValueError(
			"Oracle Stage 1 mask requires at least one declared query task invocation.",
		)

	ordered_sequence = pipeline._query_requests_ordered_task_sequence(nl_instruction)
	literal_signatures, formulas = pipeline._canonical_task_grounded_formulas(
		query_task_anchors,
		query_object_inventory=query_object_inventory,
		ordered_sequence=ordered_sequence,
	)
	if not literal_signatures or not formulas:
		raise ValueError(
			"Oracle Stage 1 mask could not derive canonical task-grounded formulas from query.",
		)

	semantic_objects: List[str] = []
	seen_objects = set()
	for entry in query_object_inventory:
		for object_name in entry.get("objects") or ():
			token = str(object_name).strip()
			if not token or token in seen_objects:
				continue
			seen_objects.add(token)
			semantic_objects.append(token)
	for signature in literal_signatures:
		_, has_args, args_text = str(signature).partition("(")
		if not has_args:
			continue
		for raw_arg in args_text.rstrip(")").split(","):
			token = raw_arg.strip()
			if not token or token in seen_objects or token.startswith("?"):
				continue
			seen_objects.add(token)
			semantic_objects.append(token)

	ltl_spec = LTLSpecification()
	ltl_spec.formulas = list(formulas)
	ltl_spec.objects = semantic_objects
	ltl_spec.source_instruction = nl_instruction
	ltl_spec.negation_hints = {}
	ltl_spec.query_object_inventory = list(query_object_inventory)
	ltl_spec.query_task_literal_signatures = list(literal_signatures)
	ltl_spec.query_task_sequence_is_ordered = ordered_sequence
	ltl_spec.grounding_map = NLToLTLfGenerator()._create_grounding_map(ltl_spec)
	return ltl_spec


def assert_query_execution_context_derives_targets_outside_stage3():
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	method_library = _official_domain_method_library("blocksworld")
	method_library.target_literals = []
	method_library.target_task_bindings = []
	ltl_spec = _build_oracle_task_grounded_stage1_spec(
		pipeline,
		BLOCKSWORLD_QUERY_CASES["query_1"]["instruction"],
	)

	query_context = pipeline._build_query_execution_context(ltl_spec, method_library)

	assert list(method_library.target_literals) == []
	assert list(method_library.target_task_bindings) == []
	assert [literal.to_signature() for literal in query_context.target_literals] == list(
		ltl_spec.query_task_literal_signatures,
	)
	assert len(query_context.target_task_bindings) == len(query_context.target_literals)
	assert query_context.query_task_anchors
	assert query_context.query_task_network


def assert_stage4_domain_gate_omits_query_runtime_records(monkeypatch):
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	pipeline.output_dir = Path(tempfile.mkdtemp(prefix="stage4_domain_gate_"))
	method_library = _official_domain_method_library("blocksworld")
	method_library.target_literals = []
	method_library.target_task_bindings = []

	class FakePlanner:
		def __init__(self, workspace=None):
			self.workspace = workspace

		def plan(
			self,
			domain,
			method_library,
			objects,
			target_literal,
			task_name,
			transition_name,
			**kwargs,
		):
			work_dir = Path(tempfile.mkdtemp(prefix=f"{transition_name}_"))
			return PANDAPlanResult(
				task_name=task_name,
				task_args=tuple(kwargs.get("task_args") or ()),
				target_literal=target_literal,
				steps=[],
				raw_plan="noop",
				actual_plan="noop",
				work_dir=str(work_dir),
				timing_profile={},
			)

	monkeypatch.setattr(pipeline_module, "PANDAPlanner", FakePlanner)
	monkeypatch.setattr(
		pipeline,
		"_task_witness_initial_facts",
		lambda *args, **kwargs: (),
	)

	stage4_data = pipeline._stage4_domain_gate(method_library)

	assert stage4_data is not None
	assert stage4_data["gate_type"] == "domain_complete"
	assert stage4_data["query_specific_runtime_records"] == 0
	assert "unordered_runtime_plan_records" not in stage4_data
	assert "query_validations" not in stage4_data
	assert stage4_data["validated_task_count"] == len(method_library.compound_tasks)


def assert_build_domain_library_persists_stable_artifacts(monkeypatch, tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	pipeline.logger = PipelineLogger(logs_dir=str(TESTS_GENERATED_LOGS_DIR), run_origin="tests")
	method_library = _official_domain_method_library("blocksworld")
	method_library.target_literals = []
	method_library.target_task_bindings = []

	stage3_summary = {
		"used_llm": False,
		"declared_compound_tasks": [task.name for task in method_library.compound_tasks],
		"compound_tasks": len(method_library.compound_tasks),
		"primitive_tasks": len(method_library.primitive_tasks),
		"methods": len(method_library.methods),
	}
	stage4_summary = {
		"gate_type": "domain_complete",
		"validated_task_count": len(method_library.compound_tasks),
		"query_specific_runtime_records": 0,
	}

	monkeypatch.setattr(
		pipeline,
		"_stage3_domain_method_synthesis",
		lambda: (method_library, stage3_summary),
	)
	monkeypatch.setattr(
		pipeline,
		"_stage4_domain_gate",
		lambda library: stage4_summary,
	)

	result = pipeline.build_domain_library(output_root=str(tmp_path))

	assert result["success"] is True
	method_library_path = Path(result["artifact_paths"]["method_library"])
	stage3_metadata_path = Path(result["artifact_paths"]["stage3_metadata"])
	stage4_domain_gate_path = Path(result["artifact_paths"]["stage4_domain_gate"])
	assert method_library_path.exists()
	assert stage3_metadata_path.exists()
	assert stage4_domain_gate_path.exists()
	loaded_method_library = HTNMethodLibrary.from_dict(json.loads(method_library_path.read_text()))
	assert loaded_method_library.target_literals == []
	assert loaded_method_library.target_task_bindings == []


def assert_execute_query_with_library_uses_cached_domain_artifact(monkeypatch, tmp_path):
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	pipeline.logger = PipelineLogger(logs_dir=str(TESTS_GENERATED_LOGS_DIR), run_origin="tests")
	method_library = _official_domain_method_library("blocksworld")
	method_library.target_literals = []
	method_library.target_task_bindings = []
	artifact_root = tmp_path / "blocksworld"
	artifact = DomainBuildArtifact(
		domain_name="blocksworld",
		method_library=method_library,
		stage3_metadata={"used_llm": False},
		stage4_domain_gate={"gate_type": "domain_complete"},
	)
	persist_domain_build_artifact(
		artifact_root=artifact_root,
		artifact=artifact,
	)

	def oracle_stage1_parse_nl(nl_instruction: str):
		return _build_oracle_task_grounded_stage1_spec(pipeline, nl_instruction)

	monkeypatch.setattr(pipeline, "_stage1_parse_nl", oracle_stage1_parse_nl)
	monkeypatch.setattr(
		pipeline,
		"_stage2_dfa_generation",
		lambda ltl_spec: {"states": ["q0"], "transitions": [], "accepting_states": ["q0"]},
	)

	def fake_stage5(ltl_spec, dfa_result, runtime_method_library, plan_records, *, query_context=None):
		assert query_context is not None
		assert runtime_method_library.target_literals
		assert runtime_method_library.target_task_bindings
		assert method_library.target_literals == []
		assert method_library.target_task_bindings == []
		return "+!run_dfa <- true.", {"transition_count": 0}

	monkeypatch.setattr(pipeline, "_stage5_agentspeak_rendering", fake_stage5)
	monkeypatch.setattr(
		pipeline,
		"_stage6_jason_validation",
		lambda *args, **kwargs: {"summary": {"status": "success"}, "artifacts": {}},
	)
	monkeypatch.setattr(
		pipeline,
		"_stage7_official_verification",
		lambda *args, **kwargs: {"summary": {"status": "success"}},
	)

	result = pipeline.execute_query_with_library(
		BLOCKSWORLD_QUERY_CASES["query_1"]["instruction"],
		library_artifact=artifact_root,
	)

	assert result["success"] is True
	assert result["query_execution_context"]["target_literals"]


def _official_stage3_mask_synthesizer_class(domain_key: str):
	class OfficialStage3MaskSynthesizer:
		def __init__(self, *args, **kwargs):
			self._delegate = RealHTNMethodSynthesizer()

		def synthesize_domain_complete(
			self,
			domain,
			*,
			derived_analysis=None,
		):
			method_library = _official_domain_method_library(domain_key)
			method_library.target_literals = []
			method_library.target_task_bindings = []
			prompt_analysis = build_domain_prompt_analysis_payload(
				domain,
				action_analysis={},
			)
			return method_library, {
				"used_llm": False,
				"model": None,
				"declared_compound_tasks": list(prompt_analysis.get("declared_compound_tasks") or ()),
				"domain_task_contracts": list(prompt_analysis.get("domain_task_contracts") or ()),
				"action_analysis": {},
				"derived_analysis": prompt_analysis,
				"compound_tasks": len(method_library.compound_tasks),
				"primitive_tasks": len(method_library.primitive_tasks),
				"methods": len(method_library.methods),
				"failure_stage": None,
				"failure_reason": None,
				"failure_class": None,
				"llm_prompt": None,
				"llm_response": None,
				"llm_finish_reason": None,
				"llm_attempts": 0,
				"llm_generation_attempts": 0,
			}

		def synthesize(
			self,
			domain,
			*,
			query_text=None,
			query_task_anchors=None,
			semantic_objects=None,
			query_object_inventory=None,
			query_objects=None,
			derived_analysis=None,
			negation_hints=None,
			ordered_literal_signatures=None,
		):
			target_literal_signatures = list(ordered_literal_signatures or ())
			anchors = list(query_task_anchors or ())
			method_library = _official_domain_method_library(
				domain_key,
				target_literal_signatures=target_literal_signatures,
				query_task_anchors=anchors,
			)
			if not target_literal_signatures:
				target_literal_signatures = [
					literal.to_signature()
					for literal in method_library.target_literals
				]
			prompt_analysis = build_prompt_analysis_payload(
				domain,
				target_literals=target_literal_signatures,
				query_task_anchors=anchors,
			)
			return method_library, {
				"used_llm": False,
				"model": None,
				"target_literals": target_literal_signatures,
				"query_task_anchors": anchors,
				"semantic_objects": list(semantic_objects or ()),
				"query_object_inventory": list(query_object_inventory or ()),
				"query_objects": list(query_objects or ()),
				"negation_resolution": {"predicates": [], "mode_by_predicate": {}},
				"action_analysis": {},
				"derived_analysis": prompt_analysis,
				"compound_tasks": len(method_library.compound_tasks),
				"primitive_tasks": len(method_library.primitive_tasks),
				"methods": len(method_library.methods),
				"failure_stage": None,
				"failure_reason": None,
				"failure_class": None,
				"llm_prompt": None,
				"llm_response": None,
				"llm_finish_reason": None,
				"llm_attempts": 0,
				"llm_generation_attempts": 0,
			}

	return OfficialStage3MaskSynthesizer


def _build_official_stage3_mask_domain_artifact(domain_key: str, monkeypatch) -> Path:
	cached = _OFFICIAL_STAGE3_MASK_DOMAIN_ARTIFACT_CACHE.get(domain_key)
	if cached is not None:
		return cached

	domain_config = OFFICIAL_STAGE3_MASK_DOMAIN_CONFIGS[domain_key]
	monkeypatch.setattr(
		pipeline_module,
		"HTNMethodSynthesizer",
		_official_stage3_mask_synthesizer_class(domain_key),
	)
	pipeline = LTL_BDI_Pipeline(domain_file=domain_config["domain_file"])
	pipeline.logger = PipelineLogger(logs_dir=str(TESTS_GENERATED_LOGS_DIR), run_origin="tests")
	result = pipeline.build_domain_library(
		output_root=str(TESTS_GENERATED_DOMAIN_BUILDS_DIR / "official_stage3_mask" / domain_key),
	)
	assert result["success"] is True
	artifact_root = Path(result["artifact_paths"]["method_library"]).parent
	_OFFICIAL_STAGE3_MASK_DOMAIN_ARTIFACT_CACHE[domain_key] = artifact_root
	return artifact_root

def _run_domain_query_case_with_official_stage3_mask(
	domain_key: str,
	query_id: str,
	monkeypatch,
	*,
	query_cases: Dict[str, Dict[str, Any]] | None = None,
	mask_stage1: bool = False,
) -> Dict[str, Any]:
	domain_config = OFFICIAL_STAGE3_MASK_DOMAIN_CONFIGS[domain_key]
	case_map = query_cases or domain_config["query_cases"]
	case = case_map[query_id]
	artifact_root = _build_official_stage3_mask_domain_artifact(domain_key, monkeypatch)

	pipeline = LTL_BDI_Pipeline(
		domain_file=domain_config["domain_file"],
		problem_file=case.get("problem_file"),
	)
	pipeline.logger = PipelineLogger(logs_dir=str(TESTS_GENERATED_LOGS_DIR), run_origin="tests")
	if mask_stage1:
		def oracle_stage1_parse_nl(nl_instruction: str):
			print("\n[STAGE 1] Natural Language -> LTLf Specification")
			print("-"*80)
			try:
				ltl_spec = _build_oracle_task_grounded_stage1_spec(pipeline, nl_instruction)
				pipeline.logger.log_stage1_success(
					ltl_spec.to_dict(),
					used_llm=False,
				)
				print(
					f"✓ Oracle task-grounded Stage 1 mask: "
					f"{[formula.to_string() for formula in ltl_spec.formulas]}",
				)
				print(f"  Objects: {ltl_spec.objects}")
				return ltl_spec
			except Exception as exc:
				pipeline.logger.log_stage1_error(str(exc))
				print(f"✗ Stage 1 Failed: {exc}")
				return None

		monkeypatch.setattr(pipeline, "_stage1_parse_nl", oracle_stage1_parse_nl)
	result = pipeline.execute_query_with_library(
		case["instruction"],
		library_artifact=artifact_root,
		mode="dfa_agentspeak",
	)
	log_dir = pipeline.logger.current_log_dir
	if log_dir is None:
		raise RuntimeError(f"{domain_key}:{query_id} did not produce a log directory")
	execution = json.loads((log_dir / "execution.json").read_text())
	stage6_artifacts = _load_stage6_artifacts(execution, log_dir)
	stage7_artifacts = _load_stage7_artifacts(execution, log_dir)
	stage3_summary = json.loads((artifact_root / "stage3_metadata.json").read_text())
	stage4_summary = json.loads((artifact_root / "stage4_domain_gate.json").read_text())

	bug_messages: List[str] = []
	if not result["success"]:
		bug_messages.append("pipeline returned success=False")
	failed_stages: List[str] = []
	for stage_key in (
		"stage1_status",
		"stage2_status",
		"stage5_status",
		"stage6_status",
		"stage7_status",
	):
		if execution.get(stage_key) != "success":
			failed_stages.append(stage_key)
	if failed_stages:
		bug_messages.extend(f"{stage_key} is not success" for stage_key in failed_stages)
	if execution.get("stage3_status") not in (None, "", "pending"):
		bug_messages.append("query execution log should not record Stage 3 status")
	if execution.get("stage4_status") not in (None, "", "pending"):
		bug_messages.append("query execution log should not record Stage 4 status")
	if stage3_summary.get("used_llm") is not False:
		bug_messages.append("official Stage 3 mask should not mark used_llm=True")
	if not stage3_summary.get("domain_task_contracts"):
		bug_messages.append("cached domain Stage 3 metadata is missing domain task contracts")
	if stage4_summary.get("gate_type") != "domain_complete":
		bug_messages.append("cached Stage 4 gate is not domain_complete")
	if not stage4_summary.get("validated_task_count"):
		bug_messages.append("cached Stage 4 gate validated zero tasks")
	if execution.get("stage6_status") == "success" and stage6_artifacts.get("status") != "success":
		bug_messages.append("Stage 6 status payload is not success")
	if execution.get("stage7_status") == "success" and stage7_artifacts.get("plan_kind") != "hierarchical":
		bug_messages.append("official IPC verifier did not validate a hierarchical plan")
	if (
		execution.get("stage7_status") == "success"
		and stage7_artifacts.get("verification_result") is not True
	):
		bug_messages.append("official IPC HTN verifier did not accept the generated plan")

	return {
		"query_id": query_id,
		"case": case,
		"result": result,
		"log_dir": log_dir,
		"execution": execution,
		"domain_build_stage3": stage3_summary,
		"domain_build_stage4": stage4_summary,
		"bug_messages": bug_messages,
		"has_bug": bool(bug_messages),
	}

def _run_query_case_with_official_stage3_mask(query_id: str, monkeypatch) -> Dict[str, Any]:
	return _run_domain_query_case_with_official_stage3_mask("blocksworld", query_id, monkeypatch)

def _run_marsrover_query_case_with_official_stage3_mask(query_id: str, monkeypatch) -> Dict[str, Any]:
	return _run_domain_query_case_with_official_stage3_mask("marsrover", query_id, monkeypatch)

def _run_satellite_query_case_with_official_stage3_mask(query_id: str, monkeypatch) -> Dict[str, Any]:
	return _run_domain_query_case_with_official_stage3_mask("satellite", query_id, monkeypatch)

def _run_transport_query_case_with_official_stage3_mask(query_id: str, monkeypatch) -> Dict[str, Any]:
	return _run_domain_query_case_with_official_stage3_mask("transport", query_id, monkeypatch)

def _run_domain_query_case_with_official_stage1_stage3_mask(
	domain_key: str,
	query_id: str,
	monkeypatch,
	*,
	query_cases: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
	return _run_domain_query_case_with_official_stage3_mask(
		domain_key,
		query_id,
		monkeypatch,
		query_cases=query_cases,
		mask_stage1=True,
	)

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
			if body_text.startswith("!") or body_text.startswith("?"):
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
			*,
			query_text=None,
			query_task_anchors=None,
			semantic_objects=None,
			query_object_inventory=None,
			negation_hints=None,
			ordered_literal_signatures=None,
			query_objects=None,
			derived_analysis=None,
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

	_, stage3_data = pipeline._stage3_method_synthesis(SimpleNamespace(grounding_map=None))

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

def assert_seed_validation_scope_accepts_parent_type_when_leaf_type_is_unconstrained(tmp_path):
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

	object_pool, object_types = pipeline._seed_validation_scope(
		"reach",
		method_library,
		("v1", "l1"),
		("v1", "l1"),
	)

	assert object_pool == ["v1", "l1"]
	assert object_types == {
		"v1": "vehicle",
		"l1": "location",
	}

def assert_seed_validation_scope_uses_query_type_for_disjunctive_method_branches(tmp_path):
	domain_file = tmp_path / "domain_disjunctive_transport.hddl"
	domain_file.write_text(
		"""
(define (domain disjunctive_transport)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types vehicle package - locatable locatable location)
  (:predicates
    (at ?x - locatable ?l - location)
    (loaded ?p - package ?v - vehicle)
  )
  (:task reach
    :parameters (?x - locatable ?l - location)
  )
  (:action move
    :parameters (?v - vehicle ?from - location ?to - location)
    :precondition (and (at ?v ?from))
    :effect (and (not (at ?v ?from)) (at ?v ?to))
  )
  (:action drop
    :parameters (?v - vehicle ?l - location ?p - package)
    :precondition (and (at ?v ?l) (loaded ?p ?v))
    :effect (and (not (loaded ?p ?v)) (at ?p ?l))
  )
)
		""".strip(),
	)
	pipeline = LTL_BDI_Pipeline(domain_file=str(domain_file))
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				"reach",
				("ARG1", "ARG2"),
				False,
				("at",),
				headline_literal=HTNLiteral("at", ("ARG1", "ARG2"), True, None),
			),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_reach_vehicle",
				task_name="reach",
				parameters=("ARG1", "ARG2", "FROM"),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="move",
						args=("ARG1", "FROM", "ARG2"),
						kind="primitive",
						action_name="move",
					),
				),
				ordering=(),
			),
			HTNMethod(
				method_name="m_reach_package",
				task_name="reach",
				parameters=("ARG1", "ARG2", "VEHICLE"),
				context=(HTNLiteral("loaded", ("ARG1", "VEHICLE"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="drop",
						args=("VEHICLE", "ARG2", "ARG1"),
						kind="primitive",
						action_name="drop",
					),
				),
				ordering=(),
			),
		],
		target_literals=[HTNLiteral("at", ("pkg1", "loc1"), True, None)],
		target_task_bindings=[],
	)

	object_pool, object_types = pipeline._seed_validation_scope(
		"reach",
		method_library,
		("pkg1", "loc1"),
		("pkg1", "loc1"),
		explicit_object_types={"pkg1": "package", "loc1": "location"},
	)

	assert object_types["pkg1"] == "package"
	assert object_types["loc1"] == "location"
	facts = pipeline._task_witness_initial_facts(
		None,
		"reach",
		method_library,
		("pkg1", "loc1"),
		("pkg1", "loc1"),
		object_pool=object_pool,
		object_types=object_types,
	)
	assert "(loaded pkg1 witness_vehicle_1)" in facts

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

def assert_stage3_type_validation_indexes_large_target_bindings(tmp_path, monkeypatch):
	domain_file = tmp_path / "domain_blocks.hddl"
	domain_file.write_text(
		"""
(define (domain block_world)
  (:requirements :typing :hierarchy)
  (:types block)
  (:predicates
    (clear ?b - block)
  )
  (:task do_clear
    :parameters (?b - block)
  )
  (:action noop
    :parameters (?b - block)
    :precondition (and (clear ?b))
    :effect (and (clear ?b))
  )
)
		""".strip(),
	)
	pipeline = LTL_BDI_Pipeline(domain_file=str(domain_file))
	target_count = 256
	target_literals = [
		HTNLiteral("clear", (f"b{index}",), True, None)
		for index in range(target_count)
	]
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("do_clear", ("BLOCK",), False, ("clear",))],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_do_clear_noop",
				task_name="do_clear",
				parameters=("BLOCK",),
				task_args=("BLOCK",),
				context=(),
				subtasks=(),
				ordering=(),
			),
		],
		target_literals=target_literals,
		target_task_bindings=[
			HTNTargetTaskBinding(literal.to_signature(), "do_clear")
			for literal in target_literals
		],
	)

	literal_type_candidate_calls = 0
	original_literal_type_candidates = pipeline._literal_type_candidates

	def counted_literal_type_candidates(literal):
		nonlocal literal_type_candidate_calls
		literal_type_candidate_calls += 1
		return original_literal_type_candidates(literal)

	monkeypatch.setattr(
		pipeline,
		"_literal_type_candidates",
		counted_literal_type_candidates,
	)

	pipeline._validate_method_library_typing(method_library)

	assert literal_type_candidate_calls <= target_count + len(method_library.methods) + 1

def assert_problem_domain_name_mismatch_allows_compatible_problem(tmp_path):
	domain_file = tmp_path / "domain_transport.hddl"
	domain_file.write_text(
		"""
(define (domain transport)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types location vehicle package)
  (:predicates
    (at ?x - package ?l - location)
    (truck_at ?v - vehicle ?l - location)
  )
  (:task deliver
    :parameters (?p - package ?l - location)
  )
  (:action noop
    :parameters (?v - vehicle ?l - location)
    :precondition (and (truck_at ?v ?l))
    :effect ()
  )
)
		""".strip(),
	)
	problem_file = tmp_path / "problem_transport_alias.hddl"
	problem_file.write_text(
		"""
(define (problem transport_alias)
  (:domain domain_htn)
  (:objects
    truck0 - vehicle
    package0 - package
    loc0 - location
  )
  (:htn
    :tasks (and
      (deliver package0 loc0)
    )
    :ordering ()
    :constraints ()
  )
  (:init
    (at package0 loc0)
    (truck_at truck0 loc0)
  )
)
		""".strip(),
	)

	pipeline = LTL_BDI_Pipeline(
		domain_file=str(domain_file),
		problem_file=str(problem_file),
	)

	assert pipeline.problem is not None
	assert pipeline.problem.domain_name == "domain_htn"
	assert pipeline.domain.name == "transport"

def assert_problem_domain_name_mismatch_rejects_incompatible_problem(tmp_path):
	domain_file = tmp_path / "domain_transport.hddl"
	domain_file.write_text(
		"""
(define (domain transport)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types location vehicle package)
  (:predicates
    (at ?x - package ?l - location)
  )
  (:task deliver
    :parameters (?p - package ?l - location)
  )
  (:action noop
    :parameters (?p - package ?l - location)
    :precondition (and (at ?p ?l))
    :effect ()
  )
)
		""".strip(),
	)
	problem_file = tmp_path / "problem_transport_bad.hddl"
	problem_file.write_text(
		"""
(define (problem transport_bad)
  (:domain domain_htn)
  (:objects
    package0 - package
    loc0 - location
  )
  (:htn
    :tasks (and
      (pickup package0 loc0)
    )
    :ordering ()
    :constraints ()
  )
  (:init
    (at package0 loc0)
  )
)
		""".strip(),
	)

	with pytest.raises(ValueError, match="HTN task is not declared"):
		LTL_BDI_Pipeline(
			domain_file=str(domain_file),
			problem_file=str(problem_file),
		)

def assert_method_task_argument_type_hints_prefer_explicit_task_args(tmp_path):
	domain_file = tmp_path / "domain_delivery.hddl"
	domain_file.write_text(
		"""
(define (domain delivery_world)
  (:requirements :typing :hierarchy :negative-preconditions)
  (:types package vehicle location)
  (:predicates
    (pkg_at ?p - package ?l - location)
    (truck_at ?v - vehicle ?l - location)
  )
  (:task deliver
    :parameters (?p - package ?l - location)
  )
  (:action drive
    :parameters (?v - vehicle ?from - location ?to - location)
    :precondition (and (truck_at ?v ?from))
    :effect (and (truck_at ?v ?to) (not (truck_at ?v ?from)))
  )
)
		""".strip(),
	)
	pipeline = LTL_BDI_Pipeline(domain_file=str(domain_file))
	method = HTNMethod(
		method_name="m_deliver_drive",
		task_name="deliver",
		parameters=("FROM", "TRUCK", "PKG", "DEST"),
		task_args=("PKG", "DEST"),
		context=(
			HTNLiteral("truck_at", ("TRUCK", "FROM"), True, None),
			HTNLiteral("pkg_at", ("PKG", "FROM"), True, None),
		),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="drive",
				args=("TRUCK", "FROM", "DEST"),
				kind="primitive",
				action_name="drive",
			),
		),
		ordering=(),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("deliver", ("PKG", "DEST"), False, ())],
		primitive_tasks=[],
		methods=[method],
		target_literals=[HTNLiteral("pkg_at", ("pkg0", "loc1"), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("pkg_at(pkg0, loc1)", "deliver")],
	)

	variable_types = pipeline._method_variable_type_hints(method, method_library)

	assert variable_types == {
		"DEST": "location",
		"FROM": "location",
		"PKG": "package",
		"TRUCK": "vehicle",
	}
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

def assert_ordered_literal_signatures_preserve_identical_occurrences():
	spec = SimpleNamespace(
		formulas=[
			LTLFormula(
				operator=TemporalOperator.FINALLY,
				predicate=None,
				sub_formulas=[
					LTLFormula(
						operator=None,
						predicate={"on": ["b10", "b6"]},
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
						predicate={"on": ["b5", "b10"]},
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
						predicate={"on": ["b10", "b6"]},
						sub_formulas=[],
						logical_op=None,
					),
				],
				logical_op=None,
			),
		],
	)

	assert LTL_BDI_Pipeline._ordered_literal_signatures_from_spec(spec) == (
		"on(b10, b6)",
		"on(b5, b10)",
		"on(b10, b6)",
	)

def assert_ordered_literal_signatures_prefer_query_task_literal_contract():
	spec = SimpleNamespace(
		query_task_literal_signatures=["on(b10, b6)", "on(b5, b10)", "on(b10, b6)"],
		formulas=[
			LTLFormula(
				operator=TemporalOperator.FINALLY,
				predicate=None,
				sub_formulas=[
					LTLFormula(
						operator=None,
						predicate=None,
						sub_formulas=[
							LTLFormula(
								operator=None,
								predicate={"on": ["b10", "b6"]},
								sub_formulas=[],
								logical_op=None,
							),
							LTLFormula(
								operator=TemporalOperator.FINALLY,
								predicate=None,
								sub_formulas=[
									LTLFormula(
										operator=None,
										predicate=None,
										sub_formulas=[
											LTLFormula(
												operator=None,
												predicate={"on": ["b10", "b6"]},
												sub_formulas=[],
												logical_op=None,
											),
											LTLFormula(
												operator=None,
												predicate={"on": ["b5", "b10"]},
												sub_formulas=[],
												logical_op=None,
											),
										],
										logical_op=LogicalOperator.AND,
									),
								],
								logical_op=None,
							),
						],
						logical_op=LogicalOperator.AND,
					),
				],
				logical_op=None,
			),
		],
	)

	assert LTL_BDI_Pipeline._ordered_literal_signatures_from_spec(spec) == (
		"on(b10, b6)",
		"on(b5, b10)",
		"on(b10, b6)",
	)

def assert_stage1_task_grounded_eventual_targets_drop_support_only_literals():
	pipeline = LTL_BDI_Pipeline(domain_file=str(MARSROVER_DOMAIN_FILE))
	spec = LTLSpecification()
	spec.add_formula(
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[
				LTLFormula(
					operator=None,
					predicate=None,
					sub_formulas=[
						LTLFormula(
							operator=None,
							predicate={"have_soil_analysis": ["rover0", "waypoint2"]},
							sub_formulas=[],
							logical_op=None,
						),
						LTLFormula(
							operator=None,
							predicate={"communicated_soil_data": ["waypoint2"]},
							sub_formulas=[],
							logical_op=None,
						),
					],
					logical_op=LogicalOperator.AND,
				),
			],
			logical_op=None,
		),
	)

	pipeline._normalise_task_grounded_stage1_spec(
		spec,
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
		),
	)

	assert LTL_BDI_Pipeline._ordered_literal_signatures_from_spec(spec) == (
		"communicated_soil_data(waypoint2)",
	)

def assert_stage1_task_grounded_singleton_support_targets_canonicalize_to_headline():
	pipeline = LTL_BDI_Pipeline(domain_file=str(MARSROVER_DOMAIN_FILE))
	spec = LTLSpecification()
	spec.add_formula(
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[
				LTLFormula(
					operator=None,
					predicate={"have_soil_analysis": ["rover0", "waypoint2"]},
					sub_formulas=[],
					logical_op=None,
				),
			],
			logical_op=None,
		),
	)

	pipeline._normalise_task_grounded_stage1_spec(
		spec,
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
		),
	)

	assert LTL_BDI_Pipeline._ordered_literal_signatures_from_spec(spec) == (
		"communicated_soil_data(waypoint2)",
	)

def assert_stage1_task_grounded_joint_eventual_bundle_canonicalizes_per_anchor():
	pipeline = LTL_BDI_Pipeline(domain_file=str(MARSROVER_DOMAIN_FILE))
	spec = LTLSpecification()
	spec.add_formula(
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[
				LTLFormula(
					operator=None,
					predicate=None,
					sub_formulas=[
						LTLFormula(
							operator=None,
							predicate={"have_soil_analysis": ["rover0", "waypoint2"]},
							sub_formulas=[],
							logical_op=None,
						),
						LTLFormula(
							operator=None,
							predicate={"communicated_soil_data": ["waypoint2"]},
							sub_formulas=[],
							logical_op=None,
						),
						LTLFormula(
							operator=None,
							predicate={"have_rock_analysis": ["rover0", "waypoint3"]},
							sub_formulas=[],
							logical_op=None,
						),
						LTLFormula(
							operator=None,
							predicate={"communicated_rock_data": ["waypoint3"]},
							sub_formulas=[],
							logical_op=None,
						),
						LTLFormula(
							operator=None,
							predicate={"have_image": ["rover0", "objective1", "high_res"]},
							sub_formulas=[],
							logical_op=None,
						),
						LTLFormula(
							operator=None,
							predicate={"communicated_image_data": ["objective1", "high_res"]},
							sub_formulas=[],
							logical_op=None,
						),
					],
					logical_op=LogicalOperator.AND,
				),
			],
			logical_op=None,
		),
	)

	pipeline._normalise_task_grounded_stage1_spec(
		spec,
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
			{"task_name": "get_rock_data", "args": ["waypoint3"]},
			{"task_name": "get_image_data", "args": ["objective1", "high_res"]},
		),
	)

	assert LTL_BDI_Pipeline._ordered_literal_signatures_from_spec(spec) == (
		"communicated_soil_data(waypoint2)",
		"communicated_rock_data(waypoint3)",
		"communicated_image_data(objective1, high_res)",
	)

def assert_stage1_task_grounded_satellite_headline_prefers_task_signature_match():
	pipeline = LTL_BDI_Pipeline(domain_file=str(SATELLITE_DOMAIN_FILE))
	spec = LTLSpecification()
	spec.add_formula(
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[
				LTLFormula(
					operator=None,
					predicate=None,
					sub_formulas=[
						LTLFormula(
							operator=None,
							predicate={"calibrated": ["instrument0"]},
							sub_formulas=[],
							logical_op=None,
						),
						LTLFormula(
							operator=None,
							predicate={"have_image": ["Phenomenon4", "thermograph0"]},
							sub_formulas=[],
							logical_op=None,
						),
					],
					logical_op=LogicalOperator.AND,
				),
			],
			logical_op=None,
		),
	)

	pipeline._normalise_task_grounded_stage1_spec(
		spec,
		query_task_anchors=(
			{"task_name": "do_observation", "args": ["Phenomenon4", "thermograph0"]},
		),
	)

	assert LTL_BDI_Pipeline._ordered_literal_signatures_from_spec(spec) == (
		"have_image(Phenomenon4, thermograph0)",
	)

def assert_stage1_task_grounded_satellite_variable_arguments_use_typed_witness_grounding():
	pipeline = LTL_BDI_Pipeline(domain_file=str(SATELLITE_DOMAIN_FILE))
	spec = LTLSpecification()
	spec.query_object_inventory = [
		{"type": "instrument", "label": "instruments", "objects": ["instrument0", "instrument1"]},
		{"type": "satellite", "label": "satellites", "objects": ["satellite0", "satellite1"]},
		{"type": "mode", "label": "mode", "objects": ["image1"]},
		{"type": "calib_direction", "label": "calib_direction", "objects": ["star0"]},
		{
			"type": "image_direction",
			"label": "image_directions",
			"objects": ["star5", "phenomenon1", "phenomenon2"],
		},
	]
	spec.add_formula(
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[
				LTLFormula(
					operator=None,
					predicate={"have_image": ["?direction1", "?mode1"]},
					sub_formulas=[],
					logical_op=None,
				),
			],
			logical_op=None,
		),
	)

	pipeline._normalise_task_grounded_stage1_spec(
		spec,
		query_task_anchors=(
			{"task_name": "do_observation", "args": ["?direction1", "?mode1"]},
		),
	)

	assert LTL_BDI_Pipeline._ordered_literal_signatures_from_spec(spec) == (
		"have_image(star5, image1)",
	)

def assert_stage1_task_grounded_objects_prefer_query_inventory_over_placeholder_tokens():
	pipeline = LTL_BDI_Pipeline(domain_file=str(MARSROVER_DOMAIN_FILE))
	spec = LTLSpecification()
	spec.objects = ["ROVER", "waypoint1"]
	spec.query_object_inventory = [
		{
			"type": "waypoint",
			"label": "waypoints",
			"objects": ["waypoint1", "waypoint3", "waypoint4", "waypoint5"],
		},
		{
			"type": "objective",
			"label": "objectives",
			"objects": ["objective0", "objective2"],
		},
		{
			"type": "mode",
			"label": "modes",
			"objects": ["low_res", "high_res"],
		},
	]
	spec.add_formula(
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[
				LTLFormula(
					operator=None,
					predicate={"communicated_soil_data": ["waypoint1"]},
					sub_formulas=[],
					logical_op=None,
				),
			],
			logical_op=None,
		),
	)

	pipeline._normalise_task_grounded_stage1_spec(
		spec,
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint1"]},
		),
		query_text=(
			"Using waypoints waypoint1, waypoint3, waypoint4, and waypoint5, "
			"objectives objective0 and objective2, and modes low_res and "
			"high_res, complete the tasks get_soil_data(waypoint1)."
		),
	)

	assert spec.objects == [
		"waypoint1",
		"waypoint3",
		"waypoint4",
		"waypoint5",
		"objective0",
		"objective2",
		"low_res",
		"high_res",
	]

def assert_ipc_plan_verifier_grounds_parameterised_root_tasks_from_trace():
	verifier = IPCPlanVerifier()
	grounded = verifier._resolve_task_args_from_trace(
		("?direction1", "?mode1"),
		("star5", "image1"),
	)

	assert grounded == ("star5", "image1")

def assert_ipc_plan_verifier_reconstructs_multi_root_depth_first_method_traces():
	problem = HDDLParser.parse_problem(
		str(SATELLITE_PROBLEM_DIR / "2obs-1sat-1mod.hddl"),
	)
	method_library = _official_domain_method_library(
		"satellite",
		target_literal_signatures=[],
		query_task_anchors=[],
	)
	verifier = IPCPlanVerifier()
	trace_entries = verifier._normalise_method_trace(
		[
			{"method_name": "method0", "task_args": ["Phenomenon4", "thermograph0"]},
			{"method_name": "method5", "task_args": ["satellite0", "instrument0"]},
			{"method_name": "method6", "task_args": ["satellite0", "instrument0"]},
			{"method_name": "method0", "task_args": ["Star5", "thermograph0"]},
			{"method_name": "method4", "task_args": ["satellite0", "instrument0"]},
			{"method_name": "method6", "task_args": ["satellite0", "instrument0"]},
		],
	)
	actions = [
		("switch_on", ("instrument0", "satellite0")),
		("turn_to", ("satellite0", "GroundStation2", "Phenomenon6")),
		("calibrate", ("satellite0", "instrument0", "GroundStation2")),
		("turn_to", ("satellite0", "Phenomenon4", "GroundStation2")),
		("take_image", ("satellite0", "Phenomenon4", "instrument0", "thermograph0")),
		("switch_off", ("instrument0", "satellite0")),
		("switch_on", ("instrument0", "satellite0")),
		("turn_to", ("satellite0", "GroundStation2", "Phenomenon4")),
		("calibrate", ("satellite0", "instrument0", "GroundStation2")),
		("turn_to", ("satellite0", "Star5", "GroundStation2")),
		("take_image", ("satellite0", "Star5", "instrument0", "thermograph0")),
	]

	root_nodes, action_index, trace_index = verifier._reconstruct_hierarchy(
		method_library=method_library,
		root_tasks=problem.htn_tasks,
		actions=actions,
		trace_entries=trace_entries,
		root_tasks_ordered=bool(problem.htn_ordered),
	)

	assert action_index == len(actions)
	assert trace_index == len(trace_entries)
	assert [tuple(node.args) for node in root_nodes] == [
		("Phenomenon4", "thermograph0"),
		("Star5", "thermograph0"),
	]

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
		"Using blocks b4, b2, b1, and b3, "
		"complete the tasks do_put_on(b4, b2), then do_put_on(b1, b4), "
		"then do_put_on(b3, b1)."
	)
	assert case["required_task_clauses"] == [
		"do_put_on(b4, b2)",
		"do_put_on(b1, b4)",
		"do_put_on(b3, b1)",
	]
	assert case["problem_file"] == str(problem_path.resolve())

def assert_official_blocksworld_problem_query_case_generation_preserves_identical_root_task_occurrences():
	problem_path = (
		Path(__file__).parent.parent
		/ "src"
		/ "domains"
		/ "blocksworld"
		/ "problems"
		/ "p04.hddl"
	)
	if not problem_path.exists():
		pytest.skip(f"Missing blocksworld problem file: {problem_path}")

	case = _build_case_from_problem(problem_path)
	assert case is not None
	assert case["required_task_clauses"] == [
		"do_put_on(b10, b6)",
		"do_put_on(b5, b10)",
		"do_put_on(b2, b5)",
		"do_put_on(b9, b2)",
		"do_put_on(b1, b9)",
		"do_put_on(b11, b1)",
		"do_put_on(b10, b6)",
		"do_put_on(b5, b10)",
		"do_put_on(b2, b5)",
		"do_put_on(b9, b2)",
		"do_put_on(b1, b9)",
		"do_put_on(b11, b1)",
		"do_put_on(b4, b11)",
		"do_put_on(b7, b4)",
	]
	assert case["instruction"] == (
		"Using blocks b10, b6, b5, b2, b9, b1, b11, b4, and b7, "
		"complete the tasks do_put_on(b10, b6), then do_put_on(b5, b10), "
		"then do_put_on(b2, b5), then do_put_on(b9, b2), then do_put_on(b1, b9), "
		"then do_put_on(b11, b1), then do_put_on(b10, b6), then do_put_on(b5, b10), "
		"then do_put_on(b2, b5), then do_put_on(b9, b2), then do_put_on(b1, b9), "
		"then do_put_on(b11, b1), then do_put_on(b4, b11), then do_put_on(b7, b4)."
	)

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
		"Using waypoints waypoint2 and waypoint3, objective objective1, and mode high_res, "
		"complete the tasks "
		"get_soil_data(waypoint2), get_rock_data(waypoint3), and "
		"get_image_data(objective1, high_res)."
	)
	assert case["required_task_clauses"] == [
		"get_soil_data(waypoint2)",
		"get_rock_data(waypoint3)",
		"get_image_data(objective1, high_res)",
	]
	assert case["problem_file"] == str(problem_path.resolve())

def assert_official_satellite_problem_query_case_generation_from_problem_tasks():
	problem_path = (
		Path(__file__).parent.parent
		/ "src"
		/ "domains"
		/ "satellite"
		/ "problems"
		/ "1obs-1sat-1mod.hddl"
	)
	if not problem_path.exists():
		pytest.skip(f"Missing satellite problem file: {problem_path}")

	case = _build_case_from_problem(problem_path)
	assert case is not None
	assert case["instruction"] == (
		"Using image_direction Phenomenon4 and mode thermograph0, "
		"complete the tasks do_observation(Phenomenon4, thermograph0)."
	)
	assert case["required_task_clauses"] == [
		"do_observation(Phenomenon4, thermograph0)",
	]
	assert case["problem_file"] == str(problem_path.resolve())

def assert_benchmark_query_generation_uses_only_query_referenced_objects_when_grounded():
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

	problem = HDDLParser.parse_problem(str(problem_path))
	assert _query_referenced_problem_objects(problem) == [
		"waypoint2",
		"waypoint3",
		"objective1",
		"high_res",
	]

def assert_benchmark_query_dataset_matches_canonical_problem_generation():
	for domain_key, problem_dir in {
		"blocksworld": BLOCKSWORLD_PROBLEM_DIR,
		"marsrover": MARSROVER_PROBLEM_DIR,
		"satellite": SATELLITE_PROBLEM_DIR,
		"transport": TRANSPORT_PROBLEM_DIR,
	}.items():
		query_cases = _load_problem_query_cases(
			problem_dir,
			limit=10_000,
			pattern=DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS[domain_key],
		)
		expected_cases: Dict[str, Dict[str, Any]] = {}
		problem_paths = sorted(problem_dir.glob(DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS[domain_key]))
		for index, problem_path in enumerate(problem_paths, start=1):
			case = _build_case_from_problem(problem_path)
			if case is None:
				continue
			expected_cases[f"query_{index}"] = case
		assert query_cases == expected_cases

def assert_benchmark_query_dataset_metadata_points_to_query_protocol():
	dataset = _load_benchmark_query_dataset()
	assert dataset["dataset_kind"] == "stored_benchmark_queries"
	assert dataset["query_protocol_document"] == "docs/query_protocol.md"
	assert dataset["generator"] == "canonical_root_task_query_v2"

def assert_benchmark_query_dataset_cases_store_only_natural_language_inputs():
	dataset = _load_benchmark_query_dataset()
	for domain_record in dataset["domains"].values():
		for stored_case in (domain_record.get("cases") or {}).values():
			assert isinstance(stored_case, str)
			assert stored_case.strip()

def assert_benchmark_query_dataset_instructions_respect_protocol():
	for query_cases in (
		_load_problem_query_cases(BLOCKSWORLD_PROBLEM_DIR, limit=10_000),
		_load_problem_query_cases(MARSROVER_PROBLEM_DIR, limit=10_000),
		_load_problem_query_cases(SATELLITE_PROBLEM_DIR, limit=10_000, pattern="*.hddl"),
		_load_problem_query_cases(TRANSPORT_PROBLEM_DIR, limit=10_000),
	):
		for query_id, case in query_cases.items():
			instruction = case["instruction"]
			lowered = instruction.lower()
			assert "\n" not in instruction, query_id
			assert instruction.endswith("."), query_id
			assert "complete the tasks " in instruction, query_id
			assert "problem.hddl" not in lowered, query_id
			assert ":init" not in lowered, query_id
			assert ":goal" not in lowered, query_id
			assert "official initial state" not in lowered, query_id
			assert "decomposition strategy" not in lowered, query_id
			assert "repair" not in lowered, query_id
			for clause in case["required_task_clauses"]:
				assert clause in instruction, (query_id, clause)


def test_benchmark_query_id_sort_key_orders_numeric_suffixes():
	query_ids = ["query_10", "query_2", "query_1", "query_30", "query_9"]

	assert sorted(query_ids, key=_benchmark_query_id_sort_key) == [
		"query_1",
		"query_2",
		"query_9",
		"query_10",
		"query_30",
	]

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

def assert_task_grounded_canonical_formulas_preserve_identical_obligation_occurrences():
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	signatures, formulas = pipeline._canonical_task_grounded_formulas(
		(
			{"task_name": "do_put_on", "args": ["b1", "b2"]},
			{"task_name": "do_put_on", "args": ["b1", "b2"]},
			{"task_name": "do_put_on", "args": ["b3", "b1"]},
		),
		query_object_inventory=(),
	)

	assert signatures == ("on(b1, b2)", "on(b1, b2)", "on(b3, b1)")
	assert [formula.to_string() for formula in formulas] == [
		"F(query_step_1)",
		"F(query_step_2)",
		"F(query_step_3)",
	]

def assert_task_grounded_stage1_normalisation_marks_default_queries_as_unordered():
	pipeline = LTL_BDI_Pipeline(domain_file=MARSROVER_DOMAIN_FILE)
	ltl_spec = LTLSpecification()
	ltl_spec.formulas = [
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[
				LTLFormula(
					operator=None,
					predicate={"communicated_soil_data": ["waypoint5"]},
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
					predicate={"communicated_soil_data": ["waypoint1"]},
					sub_formulas=[],
					logical_op=None,
				),
			],
			logical_op=None,
		),
	]
	ltl_spec.objects = ["waypoint1", "waypoint5", "rover1", "rover1store"]
	ltl_spec.grounding_map = GroundingMap()
	ltl_spec.source_instruction = (
		"Using rover rover1, store rover1store, and waypoints waypoint1 and waypoint5, "
		"complete the tasks get_soil_data(waypoint5) and get_soil_data(waypoint1)."
	)
	ltl_spec.negation_hints = {}
	ltl_spec.query_object_inventory = []

	pipeline._normalise_task_grounded_stage1_spec(
		ltl_spec,
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint5"]},
			{"task_name": "get_soil_data", "args": ["waypoint1"]},
		),
		query_text=ltl_spec.source_instruction,
	)

	assert ltl_spec.query_task_sequence_is_ordered is False
	assert [formula.to_string() for formula in ltl_spec.formulas] == [
		"F(query_step_1)",
		"F(query_step_2)",
	]

def assert_task_grounded_canonical_formulas_preserve_order_when_query_requests_sequence():
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	signatures, formulas = pipeline._canonical_task_grounded_formulas(
		(
			{"task_name": "do_put_on", "args": ["b4", "b2"]},
			{"task_name": "do_put_on", "args": ["b1", "b4"]},
			{"task_name": "do_put_on", "args": ["b3", "b1"]},
		),
		query_object_inventory=(),
		ordered_sequence=True,
	)

	assert signatures == ("on(b4, b2)", "on(b1, b4)", "on(b3, b1)")
	assert [formula.to_string() for formula in formulas] == [
		"(!(query_step_2) U query_step_1)",
		"(!(query_step_3) U query_step_2)",
		"F(query_step_3)",
	]

def assert_task_grounded_stage1_normalisation_overrides_independent_eventuals_for_ordered_queries():
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	ltl_spec = LTLSpecification()
	ltl_spec.formulas = [
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[
				LTLFormula(
					operator=None,
					predicate={"on": ["b4", "b2"]},
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
	]
	ltl_spec.objects = ["b1", "b2", "b3", "b4"]
	ltl_spec.grounding_map = GroundingMap()
	ltl_spec.source_instruction = (
		"Using blocks b1, b2, b3, and b4, complete the tasks "
		"do_put_on(b4, b2), then do_put_on(b1, b4), then do_put_on(b3, b1)."
	)
	ltl_spec.negation_hints = {}
	ltl_spec.query_object_inventory = []

	pipeline._normalise_task_grounded_stage1_spec(
		ltl_spec,
		query_task_anchors=(
			{"task_name": "do_put_on", "args": ["b4", "b2"]},
			{"task_name": "do_put_on", "args": ["b1", "b4"]},
			{"task_name": "do_put_on", "args": ["b3", "b1"]},
		),
		query_text=ltl_spec.source_instruction,
	)

	assert [formula.to_string() for formula in ltl_spec.formulas] == [
		"(!(query_step_2) U query_step_1)",
		"(!(query_step_3) U query_step_2)",
		"F(query_step_3)",
	]
	assert ltl_spec.query_task_literal_signatures == [
		"on(b4, b2)",
		"on(b1, b4)",
		"on(b3, b1)",
	]

def assert_task_grounded_ordered_formulas_encode_sequential_task_events():
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	signatures, formulas = pipeline._canonical_task_grounded_formulas(
		(
			{"task_name": "do_put_on", "args": ["b1", "b2"]},
			{"task_name": "do_put_on", "args": ["b1", "b2"]},
			{"task_name": "do_put_on", "args": ["b3", "b1"]},
		),
		query_object_inventory=(),
		ordered_sequence=True,
	)

	assert signatures == ("on(b1, b2)", "on(b1, b2)", "on(b3, b1)")
	assert [formula.to_string() for formula in formulas] == [
		"(!(query_step_2) U query_step_1)",
		"(!(query_step_3) U query_step_2)",
		"F(query_step_3)",
	]

def test_task_grounded_canonical_formulas_encode_ordered_queries_as_query_step_events():
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	signatures, formulas = pipeline._canonical_task_grounded_formulas(
		(
			{"task_name": "do_put_on", "args": ["b1", "b2"]},
			{"task_name": "do_put_on", "args": ["b1", "b2"]},
			{"task_name": "do_put_on", "args": ["b3", "b1"]},
		),
		query_object_inventory=(),
		ordered_sequence=True,
	)

	assert signatures == ("on(b1, b2)", "on(b1, b2)", "on(b3, b1)")
	assert [formula.to_string() for formula in formulas] == [
		"(!(query_step_2) U query_step_1)",
		"(!(query_step_3) U query_step_2)",
		"F(query_step_3)",
	]

def assert_query_object_inventory_extraction_uses_minimal_query_referenced_inventory():
	pipeline = LTL_BDI_Pipeline(domain_file=MARSROVER_DOMAIN_FILE)
	inventory = pipeline._extract_query_object_inventory(
		MARSROVER_QUERY_CASES["query_3"]["instruction"],
	)

	assert inventory == (
		{
			"type": "waypoint",
			"label": "waypoints",
			"objects": ["waypoint2", "waypoint0"],
		},
		{"type": "objective", "label": "objective", "objects": ["objective0"]},
		{"type": "mode", "label": "mode", "objects": ["colour"]},
	)

def assert_stage3_uses_query_inventory_for_grounding_even_when_semantic_objects_are_partial(
	tmp_path,
	monkeypatch,
):
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("get_soil_data", ("WAYPOINT",), False, ("communicated_soil_data",))],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_get_soil_data_noop",
				task_name="get_soil_data",
				parameters=("WAYPOINT",),
				context=(
					HTNLiteral("communicated_soil_data", ("WAYPOINT",), True, None),
				),
			),
		],
		target_literals=[HTNLiteral("communicated_soil_data", ("waypoint2",), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding("communicated_soil_data(waypoint2)", "get_soil_data"),
		],
	)
	captured: Dict[str, Any] = {}

	class FakeSynthesizer:
		def __init__(self, *args, **kwargs):
			pass

		def synthesize(
			self,
			domain,
			*,
			query_text=None,
			query_task_anchors=None,
			semantic_objects=None,
			query_object_inventory=None,
			query_objects=None,
			derived_analysis=None,
			negation_hints=None,
			ordered_literal_signatures=None,
		):
			captured["semantic_objects"] = semantic_objects
			captured["query_object_inventory"] = query_object_inventory
			captured["query_objects"] = query_objects
			captured["derived_analysis"] = derived_analysis
			return method_library, {
				"used_llm": True,
				"model": "deepseek-chat",
				"target_literals": ["communicated_soil_data(waypoint2)"],
				"query_task_anchors": list(query_task_anchors or ()),
				"semantic_objects": list(semantic_objects or ()),
				"query_object_inventory": list(query_object_inventory or ()),
				"query_objects": list(query_objects or ()),
				"derived_analysis": dict(derived_analysis or {}),
				"negation_resolution": {"predicates": [], "mode_by_predicate": {}},
				"action_analysis": {},
				"compound_tasks": 1,
				"primitive_tasks": 0,
				"methods": 1,
				"llm_prompt": {"system": "SYSTEM", "user": "USER"},
				"llm_response": '{"ok": true}',
				"llm_finish_reason": "stop",
				"llm_attempts": 1,
				"llm_response_time_seconds": 1.0,
				"llm_attempt_durations_seconds": [1.0],
				"failure_class": None,
			}

	monkeypatch.setattr(pipeline_module, "HTNMethodSynthesizer", FakeSynthesizer)

	pipeline = LTL_BDI_Pipeline(domain_file=MARSROVER_DOMAIN_FILE)
	pipeline.logger = PipelineLogger(logs_dir=str(tmp_path))
	pipeline.logger.start_pipeline(
		"demo instruction",
		mode="dfa_agentspeak",
		domain_file=pipeline.domain_file,
		output_dir=str(tmp_path),
	)
	pipeline.output_dir = pipeline.logger.current_log_dir

	ltl_spec = SimpleNamespace(
		objects=["waypoint2", "waypoint0", "objective0", "colour"],
		query_object_inventory=[],
		source_instruction=MARSROVER_QUERY_CASES["query_3"]["instruction"],
		negation_hints={},
		formulas=[],
		query_task_literal_signatures=["communicated_soil_data(waypoint2)"],
	)
	_, stage3_data = pipeline._stage3_method_synthesis(ltl_spec)

	assert stage3_data is not None
	assert captured["semantic_objects"] == ("waypoint2", "waypoint0", "objective0", "colour")
	assert captured["query_objects"] == (
		"waypoint2",
		"waypoint0",
		"objective0",
		"colour",
	)
	assert captured["query_object_inventory"][0] == {
		"type": "waypoint",
		"label": "waypoints",
		"objects": ["waypoint2", "waypoint0"],
	}

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

def assert_stage3_transition_extraction_respects_unordered_query_semantics(
	tmp_path,
	monkeypatch,
):
	method_library = HTNMethodLibrary(
		compound_tasks=[HTNTask("get_soil_data", ("WAYPOINT",), False, ("communicated_soil_data",))],
		primitive_tasks=[],
		methods=[],
		target_literals=[HTNLiteral("communicated_soil_data", ("waypoint2",), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding("communicated_soil_data(waypoint2)", "get_soil_data"),
		],
	)
	captured: Dict[str, Any] = {}

	class FakeSynthesizer:
		def __init__(self, *args, **kwargs):
			pass

		def synthesize(
			self,
			domain,
			*,
			query_text=None,
			query_task_anchors=None,
			semantic_objects=None,
			query_object_inventory=None,
			query_objects=None,
			derived_analysis=None,
			negation_hints=None,
			ordered_literal_signatures=None,
		):
			captured["ordered_literal_signatures"] = tuple(ordered_literal_signatures or ())
			return method_library, {
				"used_llm": True,
				"model": "deepseek-chat",
				"target_literals": ["communicated_soil_data(waypoint2)"],
				"query_task_anchors": list(query_task_anchors or ()),
				"semantic_objects": list(semantic_objects or ()),
				"query_object_inventory": list(query_object_inventory or ()),
				"query_objects": list(query_objects or ()),
				"derived_analysis": dict(derived_analysis or {}),
				"negation_resolution": {"predicates": [], "mode_by_predicate": {}},
				"action_analysis": {},
				"compound_tasks": 1,
				"primitive_tasks": 0,
				"methods": 0,
				"llm_prompt": {"system": "SYSTEM", "user": "USER"},
				"llm_response": '{"ok": true}',
				"llm_finish_reason": "stop",
				"llm_attempts": 1,
				"llm_response_time_seconds": 1.0,
				"llm_attempt_durations_seconds": [1.0],
				"failure_class": None,
			}

	monkeypatch.setattr(pipeline_module, "HTNMethodSynthesizer", FakeSynthesizer)

	pipeline = LTL_BDI_Pipeline(domain_file=MARSROVER_DOMAIN_FILE)
	pipeline.logger = PipelineLogger(logs_dir=str(tmp_path))
	pipeline.logger.start_pipeline(
		"demo instruction",
		mode="dfa_agentspeak",
		domain_file=pipeline.domain_file,
		output_dir=str(tmp_path),
	)
	pipeline.output_dir = pipeline.logger.current_log_dir

	ltl_spec = _build_oracle_task_grounded_stage1_spec(
		pipeline,
		MARSROVER_QUERY_CASES["query_3"]["instruction"],
	)
	assert pipeline._query_task_sequence_is_ordered(ltl_spec) is False

	_, stage3_data = pipeline._stage3_method_synthesis(ltl_spec)

	assert stage3_data is not None
	assert captured["ordered_literal_signatures"] == tuple(ltl_spec.query_task_literal_signatures)

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

def assert_oracle_task_grounded_stage1_mask_builds_canonical_blocksworld_spec():
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	instruction = BLOCKSWORLD_QUERY_CASES["query_1"]["instruction"]

	spec = _build_oracle_task_grounded_stage1_spec(pipeline, instruction)

	assert spec.source_instruction == instruction
	assert spec.query_task_literal_signatures == [
		"on(b4, b2)",
		"on(b1, b4)",
		"on(b3, b1)",
	]
	assert [formula.to_string() for formula in spec.formulas] == [
		"(!(query_step_2) U query_step_1)",
		"(!(query_step_3) U query_step_2)",
		"F(query_step_3)",
	]
	assert "b4" in spec.objects
	assert spec.query_task_sequence_is_ordered is True
	assert spec.grounding_map is not None

def assert_oracle_task_grounded_stage1_mask_grounds_parameterised_satellite_queries():
	problem_path = SATELLITE_PROBLEM_DIR / "1obs-2sat-1mod.hddl"
	if not problem_path.exists():
		pytest.skip(f"Missing satellite problem file: {problem_path}")

	case = _build_case_from_problem(problem_path)
	assert case is not None

	pipeline = LTL_BDI_Pipeline(domain_file=SATELLITE_DOMAIN_FILE)
	spec = _build_oracle_task_grounded_stage1_spec(pipeline, case["instruction"])

	assert spec.formulas
	assert spec.query_task_literal_signatures
	assert all("?" not in signature for signature in spec.query_task_literal_signatures)
	assert all("?" not in formula.to_string() for formula in spec.formulas)
	assert spec.query_task_sequence_is_ordered is False
	assert spec.grounding_map is not None

def assert_pipeline_logger_stage1_payload_omits_recursive_formula_tree(tmp_path):
	logger = PipelineLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"Using blocks b1, b2, b3, and b4, complete the tasks do_put_on(b4, b2), then "
		"do_put_on(b1, b4), then do_put_on(b3, b1).",
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		domain_name="BLOCKS",
		problem_name="demo",
		output_dir=str(tmp_path),
	)
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	spec = _build_oracle_task_grounded_stage1_spec(
		pipeline,
		"Using blocks b1, b2, b3, and b4, complete the tasks do_put_on(b4, b2), then "
		"do_put_on(b1, b4), then do_put_on(b3, b1).",
	)

	logger.log_stage1_success(spec.to_dict(), used_llm=False)

	assert logger.current_log_dir is not None
	execution = json.loads((logger.current_log_dir / "execution.json").read_text())
	stage1_spec = execution["stage1_ltlf_spec"]
	assert "formulas" not in stage1_spec
	assert stage1_spec["formulas_string"] == [
		"(!(query_step_2) U query_step_1)",
		"(!(query_step_3) U query_step_2)",
		"F(query_step_3)",
	]

def assert_pipeline_logger_stage2_payload_omits_embedded_dfa_bodies(tmp_path):
	logger = PipelineLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"Using package-0, complete the tasks deliver(package-0, city-loc-0).",
		domain_file=TRANSPORT_DOMAIN_FILE,
		domain_name="transport",
		problem_name="demo",
		output_dir=str(tmp_path),
	)

	dfa_result = {
		"formula": "F(at(package-0, city-loc-0))",
		"dfa_dot": 'digraph MONA_DFA { init -> 1; 1 -> 2 [label="at_package_0_city_loc_0"]; }',
		"num_states": 2,
		"num_transitions": 1,
		"construction": "stub",
	}

	logger.log_stage2_dfas(None, dfa_result, "Success")

	assert logger.current_log_dir is not None
	execution_json = json.loads((logger.current_log_dir / "execution.json").read_text())
	stage2_result = execution_json["stage2_dfa_result"]
	assert "dfa_dot" not in stage2_result
	assert stage2_result["dfa_path"] == "dfa.dot"

	execution_txt = (logger.current_log_dir / "execution.txt").read_text()
	assert "Full DFA bodies are stored in dfa.dot" in execution_txt
	assert 'label="at_package_0_city_loc_0"' not in execution_txt

def assert_pipeline_logger_stage4_payload_omits_embedded_transition_dump(tmp_path):
	logger = PipelineLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"Using package-0, complete the tasks deliver(package-0, city-loc-0).",
		domain_file=TRANSPORT_DOMAIN_FILE,
		domain_name="transport",
		problem_name="demo",
		output_dir=str(tmp_path),
	)

	stage4_artifacts = {
		"transitions": [
			{
				"source_state": "q1",
				"target_state": "q2",
				"transition_name": "dfa_step_q1_q2_demo",
				"initial_facts": ["(road city-loc-0 city-loc-1)"],
				"plan": {
					"task_name": "deliver",
					"task_args": ["package-0", "city-loc-0"],
					"steps": [],
				},
			},
		],
	}

	logger.log_stage4_panda_planning(stage4_artifacts, "Success", metadata={"backend": "pandaPI"})

	assert logger.current_log_dir is not None
	execution_json = json.loads((logger.current_log_dir / "execution.json").read_text())
	stage4_result = execution_json["stage4_artifacts"]
	assert "transitions" not in stage4_result
	assert stage4_result["transition_count"] == 1
	assert stage4_result["transitions_path"] == "panda_transitions.json"

	execution_txt = (logger.current_log_dir / "execution.txt").read_text()
	assert "Artifact path: panda_transitions.json" in execution_txt
	assert '"source_state": "q1"' not in execution_txt

def assert_stage4_plan_artifact_summary_records_relative_paths(tmp_path):
	pipeline = LTL_BDI_Pipeline.__new__(LTL_BDI_Pipeline)
	pipeline.output_dir = tmp_path

	work_dir = tmp_path / "panda" / "dfa_step_q1_q2_demo"
	work_dir.mkdir(parents=True)
	for filename in (
		"domain.hddl",
		"problem.hddl",
		"problem.psas",
		"problem.psas.grounded",
		"plan.original",
		"plan.actual",
	):
		(work_dir / filename).write_text(f"{filename}\n")

	plan = PANDAPlanResult(
		task_name="dfa_step_q1_q2_demo",
		task_args=("package-0", "city-loc-0"),
		target_literal=HTNLiteral(
			predicate="delivered",
			args=("package-0",),
			is_positive=True,
		),
		raw_plan="root\nstep\n",
		actual_plan="root\nstep\n",
		work_dir=str(work_dir),
		timing_profile={"total_seconds": 0.5},
	)

	summary = pipeline._stage4_plan_artifact_summary(plan)
	assert summary["work_dir"] == "panda/dfa_step_q1_q2_demo"
	assert summary["artifacts"]["domain_hddl"] == "panda/dfa_step_q1_q2_demo/domain.hddl"
	assert summary["artifacts"]["actual_plan"] == "panda/dfa_step_q1_q2_demo/plan.actual"
	assert summary["raw_plan_line_count"] == 2
	assert summary["actual_plan_line_count"] == 2
	assert "raw_plan" not in summary
	assert "actual_plan" not in summary

	initial_facts_path = pipeline._stage4_write_sequence_artifact(
		plan,
		"validation_initial_facts.txt",
		["(at package-0 city-loc-0)"],
	)
	assert initial_facts_path == "panda/dfa_step_q1_q2_demo/validation_initial_facts.txt"
	assert (work_dir / "validation_initial_facts.txt").read_text() == (
		"(at package-0 city-loc-0)\n"
	)

def assert_pipeline_logger_stage3_payload_records_only_artifact_path(tmp_path):
	logger = PipelineLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"Using package-0, complete the tasks deliver(package-0, city-loc-0).",
		domain_file=TRANSPORT_DOMAIN_FILE,
		domain_name="transport",
		problem_name="demo",
		output_dir=str(tmp_path),
	)

	method_library = {
		"compound_tasks": [{"name": "deliver", "parameters": ["package-0", "city-loc-0"]}],
		"primitive_tasks": [{"name": "drive-truck", "parameters": ["truck-0", "city-loc-0"]}],
		"methods": [{"method_name": "m_deliver_demo", "task_name": "deliver"}],
		"target_literals": [{"predicate": "at", "args": ["package-0", "city-loc-0"], "is_positive": True}],
		"target_task_bindings": [{"task_name": "deliver", "arguments": ["package-0", "city-loc-0"]}],
	}

	logger.log_stage3_method_synthesis(method_library, "Success", metadata={"backend": "oracle"})

	assert logger.current_log_dir is not None
	execution_json = json.loads((logger.current_log_dir / "execution.json").read_text())
	assert execution_json["stage3_method_library"] == {"artifact_path": "htn_method_library.json"}
	assert execution_json["stage3_metadata"]["target_task_bindings"] == [
		{"task_name": "deliver", "arguments": ["package-0", "city-loc-0"]},
	]
	saved_library = json.loads((logger.current_log_dir / "htn_method_library.json").read_text())
	assert saved_library["methods"][0]["method_name"] == "m_deliver_demo"

	execution_txt = (logger.current_log_dir / "execution.txt").read_text()
	assert "Artifact path: htn_method_library.json" in execution_txt
	assert '"method_name": "m_deliver_demo"' not in execution_txt

def assert_pipeline_logger_stage5_payload_records_only_artifact_path(tmp_path):
	logger = PipelineLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"Using package-0, complete the tasks deliver(package-0, city-loc-0).",
		domain_file=TRANSPORT_DOMAIN_FILE,
		domain_name="transport",
		problem_name="demo",
		output_dir=str(tmp_path),
	)

	logger.log_stage5_agentspeak_rendering(
		"/* HTN Method Plans */\n+!deliver(P, L) <- true.\n",
		"Success",
		metadata={"code_size_chars": 49},
	)

	assert logger.current_log_dir is not None
	execution_json = json.loads((logger.current_log_dir / "execution.json").read_text())
	assert execution_json["stage5_agentspeak"] == "stage5_agentspeak.asl"

	execution_txt = (logger.current_log_dir / "execution.txt").read_text()
	assert "Artifact path: stage5_agentspeak.asl" in execution_txt
	assert "+!deliver(P, L)" not in execution_txt

def assert_pipeline_logger_stage6_payload_records_relative_artifact_paths(tmp_path):
	logger = PipelineLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"Using package-0, complete the tasks deliver(package-0, city-loc-0).",
		domain_file=TRANSPORT_DOMAIN_FILE,
		domain_name="transport",
		problem_name="demo",
		output_dir=str(tmp_path),
	)
	assert logger.current_log_dir is not None

	action_path_file = logger.current_log_dir / "action_path.txt"
	action_path_file.write_text("drive-truck(truck-0, city-loc-0, city-loc-1)\n")
	method_trace_file = logger.current_log_dir / "method_trace.json"
	method_trace_file.write_text(json.dumps([{"task": "deliver"}]))
	stdout_file = logger.current_log_dir / "jason_stdout.txt"
	stdout_file.write_text("execute success\n")
	stderr_file = logger.current_log_dir / "jason_stderr.txt"
	stderr_file.write_text("")

	logger.log_stage6_jason_validation(
		{
			"status": "success",
			"backend": "RunLocalMAS",
			"timed_out": False,
			"failed_goals": [],
			"artifacts": {
				"action_path": str(action_path_file),
				"method_trace": str(method_trace_file),
				"jason_stdout": str(stdout_file),
				"jason_stderr": str(stderr_file),
			},
		},
		"Success",
	)

	execution_json = json.loads((logger.current_log_dir / "execution.json").read_text())
	stage6_artifacts = execution_json["stage6_artifacts"]
	assert "action_path" not in stage6_artifacts
	assert "method_trace" not in stage6_artifacts
	assert "stdout" not in stage6_artifacts
	assert stage6_artifacts["action_path_path"] == "action_path.txt"
	assert stage6_artifacts["method_trace_path"] == "method_trace.json"
	assert stage6_artifacts["stdout_path"] == "jason_stdout.txt"
	assert stage6_artifacts["stderr_path"] == "jason_stderr.txt"

	execution_txt = (logger.current_log_dir / "execution.txt").read_text()
	assert '"action_path_path": "action_path.txt"' in execution_txt
	assert "execute success" not in execution_txt

def assert_pipeline_logger_stage7_payload_records_relative_artifact_paths(tmp_path):
	logger = PipelineLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"Using package-0, complete the tasks deliver(package-0, city-loc-0).",
		domain_file=TRANSPORT_DOMAIN_FILE,
		domain_name="transport",
		problem_name="demo",
		output_dir=str(tmp_path),
	)
	assert logger.current_log_dir is not None

	plan_file = logger.current_log_dir / "ipc_official_plan.txt"
	plan_file.write_text("0 drive-truck truck-0 city-loc-0 city-loc-1\n")
	output_file = logger.current_log_dir / "ipc_official_verifier.txt"
	output_file.write_text("Plan verification result: true\n")

	logger.log_stage7_official_verification(
		{
			"tool_available": True,
			"plan_kind": "hierarchical",
			"verification_result": True,
			"primitive_plan_executable": True,
			"reached_goal_state": True,
			"plan_file": str(plan_file),
			"output_file": str(output_file),
			"stdout": "huge verifier stdout",
			"stderr": "",
		},
		"Success",
	)

	execution_json = json.loads((logger.current_log_dir / "execution.json").read_text())
	stage7_artifacts = execution_json["stage7_artifacts"]
	assert stage7_artifacts["plan_file"] == "ipc_official_plan.txt"
	assert stage7_artifacts["output_file"] == "ipc_official_verifier.txt"
	assert "stdout" not in stage7_artifacts
	assert "stderr" not in stage7_artifacts

	execution_txt = (logger.current_log_dir / "execution.txt").read_text()
	assert '"plan_file": "ipc_official_plan.txt"' in execution_txt
	assert "huge verifier stdout" not in execution_txt

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

def assert_stage6_object_type_resolution_preserves_problem_typed_runtime_symbols():
	problem_path = MARSROVER_PROBLEM_DIR / "pfile01.hddl"
	if not problem_path.exists():
		pytest.skip(f"Missing marsrover problem file: {problem_path}")

	pipeline = LTL_BDI_Pipeline(
		domain_file=MARSROVER_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[],
		primitive_tasks=[],
		methods=[],
		target_literals=[HTNLiteral("at", ("rover0", "waypoint0"), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("at(rover0, waypoint0)", "navigate_abs")],
	)

	resolved = pipeline._stage6_object_types(
		tuple(pipeline.problem.objects),
		method_library,
		("(at rover0 waypoint0)",),
		problem_object_types=pipeline.problem.object_types,
	)

	assert resolved["low_res"] == "mode"
	assert resolved["high_res"] == "mode"
	assert resolved["colour"] == "mode"

def assert_stage6_problem_seed_facts_ignore_stage4_witness_facts():
	problem_path = BLOCKSWORLD_PROBLEM_DIR / "p01.hddl"
	if not problem_path.exists():
		pytest.skip(f"Missing blocksworld problem file: {problem_path}")

	pipeline = LTL_BDI_Pipeline(
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
	)
	seed_facts, seed_source = pipeline._stage6_runtime_seed_facts(
		plan_records=[
			{
				"transition_name": "dfa_step_q1_q2",
				"initial_facts": ["(witness_fact bogus)"],
			},
		],
		target_literals=(),
	)

	assert seed_source == "problem_init:p01.hddl"
	assert "(witness_fact bogus)" not in seed_facts
	assert tuple(pipeline._render_problem_fact(fact) for fact in pipeline.problem.init_facts) == seed_facts

def assert_stage6_query_task_network_grounds_parameterised_query_anchors_from_literal_contract():
	problem_path = SATELLITE_PROBLEM_DIR / "1obs-2sat-1mod.hddl"
	if not problem_path.exists():
		pytest.skip(f"Missing satellite problem file: {problem_path}")

	pipeline = LTL_BDI_Pipeline(
		domain_file=SATELLITE_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
	)
	nl_instruction = (
		"Using instruments instrument0 and instrument1, satellites satellite0 and satellite1, "
		"mode image1, calib_direction star0, and image_directions star5, phenomenon1, and "
		"phenomenon2, complete the tasks do_observation(?direction1, ?mode1)."
	)
	ltl_spec = _build_oracle_task_grounded_stage1_spec(
		pipeline,
		nl_instruction,
	)
	method_library = _official_domain_method_library(
		"satellite",
		target_literal_signatures=list(ltl_spec.query_task_literal_signatures),
		query_task_anchors=list(pipeline._extract_query_task_anchors(nl_instruction)),
	)

	task_network = pipeline._stage6_query_task_network(ltl_spec, method_library)

	assert task_network == (("query_root_1_do_observation", ("star5", "image1")),)

def assert_stage4_query_task_network_grounds_parameterised_query_anchors_from_literal_contract():
	problem_path = SATELLITE_PROBLEM_DIR / "1obs-2sat-1mod.hddl"
	if not problem_path.exists():
		pytest.skip(f"Missing satellite problem file: {problem_path}")

	pipeline = LTL_BDI_Pipeline(
		domain_file=SATELLITE_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
	)
	nl_instruction = (
		"Using instruments instrument0 and instrument1, satellites satellite0 and satellite1, "
		"mode image1, calib_direction star0, and image_directions star5, phenomenon1, and "
		"phenomenon2, complete the tasks do_observation(?direction1, ?mode1)."
	)
	ltl_spec = _build_oracle_task_grounded_stage1_spec(
		pipeline,
		nl_instruction,
	)
	method_library = _official_domain_method_library(
		"satellite",
		target_literal_signatures=list(ltl_spec.query_task_literal_signatures),
		query_task_anchors=list(pipeline._extract_query_task_anchors(nl_instruction)),
	)

	task_network = pipeline._stage4_query_task_network(ltl_spec, method_library)

	assert task_network == (("do_observation", ("star5", "image1")),)

def assert_stage5_agentspeak_rendering_grounds_parameterised_query_anchors_from_literal_contract(
	tmp_path,
	monkeypatch,
):
	problem_path = SATELLITE_PROBLEM_DIR / "1obs-2sat-1mod.hddl"
	if not problem_path.exists():
		pytest.skip(f"Missing satellite problem file: {problem_path}")

	pipeline = LTL_BDI_Pipeline(
		domain_file=SATELLITE_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
	)
	pipeline.output_dir = Path(tmp_path)
	pipeline.logger = PipelineLogger(logs_dir=str(tmp_path / "logs"), run_origin="tests")
	nl_instruction = (
		"Using instruments instrument0 and instrument1, satellites satellite0 and satellite1, "
		"mode image1, calib_direction star0, and image_directions star5, phenomenon1, and "
		"phenomenon2, complete the tasks do_observation(?direction1, ?mode1)."
	)
	pipeline.logger.start_pipeline(
		natural_language=nl_instruction,
		domain_file=SATELLITE_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
		domain_name="SATELLITE",
		problem_name="stage5_timing_probe",
		output_dir=str(tmp_path),
		timestamp="stage5_timing_probe",
	)
	ltl_spec = _build_oracle_task_grounded_stage1_spec(
		pipeline,
		nl_instruction,
	)
	method_library = _official_domain_method_library(
		"satellite",
		target_literal_signatures=list(ltl_spec.query_task_literal_signatures),
		query_task_anchors=list(pipeline._extract_query_task_anchors(nl_instruction)),
	)
	captured: Dict[str, Any] = {}

	def _fake_build_agentspeak_transition_specs(*args, **kwargs):
		return (
			{
				"transition_name": "dfa_t1",
				"source_state": "q1",
				"target_state": "q2",
				"initial_state": "q1",
				"accepting_states": ("q2",),
				"guard_context": "have_image(star5, image1)",
			},
		)

	def _fake_generate(self, **kwargs):
		captured.update(kwargs)
		return "/* Execution Entry */\n!execute.\n"

	monkeypatch.setattr(
		pipeline_module,
		"build_agentspeak_transition_specs",
		_fake_build_agentspeak_transition_specs,
	)
	monkeypatch.setattr(AgentSpeakRenderer, "generate", _fake_generate)

	asl_code, metadata = pipeline._stage5_agentspeak_rendering(
		ltl_spec,
		dfa_result={"dot_graph": "digraph {}"},
		method_library=method_library,
		plan_records=[],
	)

	assert asl_code is not None
	assert metadata is not None
	assert captured["query_task_anchors"] == [
		{
			"task_name": "do_observation",
			"args": ["star5", "image1"],
			"literal_signature": "have_image(star5, image1)",
		},
	]
	stage5_timing = pipeline.logger.current_record.timing_profile["stage5"]
	breakdown = stage5_timing["breakdown_seconds"]
	assert breakdown["transition_spec_build_seconds"] >= 0.0
	assert breakdown["query_anchor_setup_seconds"] >= 0.0
	assert breakdown["render_seconds"] >= 0.0
	assert breakdown["write_output_seconds"] >= 0.0

def assert_query_task_sequence_is_ordered_requires_explicit_stage1_signal():
	implicit_then_spec = SimpleNamespace(
		query_task_sequence_is_ordered=None,
		source_instruction="Complete task_a(a), then task_b(b).",
	)
	explicit_ordered_spec = SimpleNamespace(
		query_task_sequence_is_ordered=True,
		source_instruction="Complete task_a(a), then task_b(b).",
	)
	explicit_unordered_spec = SimpleNamespace(
		query_task_sequence_is_ordered=False,
		source_instruction="Complete task_a(a), task_b(b).",
	)

	assert LTL_BDI_Pipeline._query_task_sequence_is_ordered(implicit_then_spec) is False
	assert LTL_BDI_Pipeline._query_task_sequence_is_ordered(explicit_ordered_spec) is True
	assert LTL_BDI_Pipeline._query_task_sequence_is_ordered(explicit_unordered_spec) is False

def assert_stage6_mainline_lowers_agentspeak_with_runtime_seed_facts(tmp_path, monkeypatch):
	pipeline = LTL_BDI_Pipeline(
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		problem_file=str((BLOCKSWORLD_PROBLEM_DIR / "p01.hddl").resolve()),
	)
	pipeline.output_dir = Path(tmp_path)
	ltl_spec = SimpleNamespace(
		objects=("b1", "b2"),
		source_instruction="Complete the task do_put_on(b1, b2).",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[],
		primitive_tasks=[],
		methods=[],
		target_literals=[],
		target_task_bindings=[],
	)
	lowering_calls: List[Dict[str, Any]] = []
	captured: Dict[str, Any] = {}

	class FakeValidationResult:
		def __init__(self):
			self.backend = "RunLocalMAS"
			self.status = "success"
			self.java_path = "/usr/bin/java"
			self.java_version = 17
			self.javac_path = "/usr/bin/javac"
			self.jason_jar = "/tmp/jason.jar"
			self.exit_code = 0
			self.timed_out = False
			self.action_path = []
			self.artifacts = {"action_path": "/tmp/action_path.txt"}
			self.environment_adapter = "JasonEnvironmentAdapter"
			self.failure_class = None
			self.consistency_checks = {"passed": True}
			self.timing_profile = {"mas_run_seconds": 0.1}

		def to_dict(self):
			return {"artifacts": dict(self.artifacts), "timing_profile": dict(self.timing_profile)}

	def _fake_compile(self, agentspeak_code, **kwargs):
		lowering_calls.append({"agentspeak_code": agentspeak_code, **kwargs})
		return f"{agentspeak_code}\n% lowered"

	def _fake_validate(self, **kwargs):
		captured.update(kwargs)
		return FakeValidationResult()

	monkeypatch.setattr(pipeline_module.ASLMethodLowering, "compile_method_plans", _fake_compile)
	monkeypatch.setattr(JasonRunner, "validate", _fake_validate)

	result = pipeline._stage6_jason_validation(
		ltl_spec,
		method_library,
		plan_records=[],
		asl_code="/* Execution Entry */\n!execute.\n",
	)

	assert result is not None
	assert len(lowering_calls) == 1
	assert lowering_calls[0]["seed_facts"]
	assert lowering_calls[0]["runtime_objects"]
	assert lowering_calls[0]["object_types"]
	assert lowering_calls[0]["type_parent_map"] == pipeline.type_parent_map
	assert captured["agentspeak_code"].endswith("% lowered")

def assert_stage4_prunes_validation_library_to_reachable_method_subgraph():
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("dfa_step_q1_q2_on_a_b", ("ARG1", "ARG2"), False, ("on",)),
			HTNTask("helper_clear", ("ARG1",), False, ("clear",)),
			HTNTask("unrelated_transition", ("ARG1",), False, ("clear",)),
		],
		primitive_tasks=[
			HTNTask("stack", ("ARG1", "ARG2"), True, ("on",), source_name="stack"),
		],
		methods=[
			HTNMethod(
				method_name="m_dfa_step_q1_q2_on_a_b",
				task_name="dfa_step_q1_q2_on_a_b",
				parameters=("ARG1", "ARG2"),
				subtasks=(
					HTNMethodStep("s1", "helper_clear", ("ARG2",), "compound"),
					HTNMethodStep("s2", "stack", ("ARG1", "ARG2"), "primitive"),
				),
			),
			HTNMethod(
				method_name="m_helper_clear",
				task_name="helper_clear",
				parameters=("ARG1",),
				subtasks=(),
			),
			HTNMethod(
				method_name="m_unrelated_transition",
				task_name="unrelated_transition",
				parameters=("ARG1",),
				subtasks=(),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	pruned = LTL_BDI_Pipeline._stage4_pruned_method_library_for_task(
		method_library,
		"dfa_step_q1_q2_on_a_b",
	)

	assert [task.name for task in pruned.compound_tasks] == [
		"dfa_step_q1_q2_on_a_b",
		"helper_clear",
	]
	assert [method.method_name for method in pruned.methods] == [
		"m_dfa_step_q1_q2_on_a_b",
		"m_helper_clear",
	]
	assert [task.name for task in pruned.primitive_tasks] == ["stack"]

def assert_stage4_transition_pruning_keeps_only_noop_sibling_transitions():
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("dfa_step_q1_q2_on_a_b", ("ARG1", "ARG2"), False, ("on",)),
			HTNTask("dfa_step_q2_q3_on_b_c", ("ARG1", "ARG2"), False, ("on",)),
			HTNTask("helper_clear", ("ARG1",), False, ("clear",)),
		],
		primitive_tasks=[
			HTNTask("stack", ("ARG1", "ARG2"), True, ("on",), source_name="stack"),
		],
		methods=[
			HTNMethod(
				method_name="m_root_transition",
				task_name="dfa_step_q1_q2_on_a_b",
				parameters=("ARG1", "ARG2"),
				subtasks=(
					HTNMethodStep("s1", "dfa_step_q2_q3_on_b_c", ("ARG1", "ARG2"), "compound"),
				),
			),
			HTNMethod(
				method_name="m_sibling_transition_noop",
				task_name="dfa_step_q2_q3_on_b_c",
				parameters=("ARG1", "ARG2"),
				subtasks=(),
			),
			HTNMethod(
				method_name="m_sibling_transition_constructive",
				task_name="dfa_step_q2_q3_on_b_c",
				parameters=("ARG1", "ARG2"),
				subtasks=(
					HTNMethodStep("s1", "helper_clear", ("ARG2",), "compound"),
					HTNMethodStep("s2", "stack", ("ARG1", "ARG2"), "primitive"),
				),
			),
			HTNMethod(
				method_name="m_helper_clear",
				task_name="helper_clear",
				parameters=("ARG1",),
				subtasks=(),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	pruned = LTL_BDI_Pipeline._stage4_pruned_method_library_for_task(
		method_library,
		"dfa_step_q1_q2_on_a_b",
	)

	assert [task.name for task in pruned.compound_tasks] == [
		"dfa_step_q1_q2_on_a_b",
		"dfa_step_q2_q3_on_b_c",
	]
	assert [method.method_name for method in pruned.methods] == [
		"m_root_transition",
		"m_sibling_transition_noop",
	]

def assert_stage4_compacts_large_validation_methods_without_collapsing_objects():
	root_parameters = ("ARG1", "ARG2", *(f"AUX{i}" for i in range(60)))
	task_args = tuple(f"obj{i}" for i in range(len(root_parameters)))
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				"dfa_step_large",
				root_parameters,
				False,
				("target",),
				headline_literal=HTNLiteral("target", ("ARG1", "ARG2")),
			),
			HTNTask("helper_target", ("ARG1",), False, ("target",)),
			HTNTask("helper_context", ("ARG1", "ARG2"), False, ("context",)),
		],
		primitive_tasks=[
			HTNTask("make_target", ("ARG1", "ARG2"), True, ("target",)),
		],
		methods=[
			HTNMethod(
				method_name="m_dfa_step_large_noop",
				task_name="dfa_step_large",
				parameters=root_parameters,
				task_args=root_parameters,
				context=tuple(
					HTNLiteral("context", (parameter, "ARG2"))
					for parameter in root_parameters[2:]
				),
				subtasks=(),
			),
			HTNMethod(
				method_name="m_dfa_step_large_constructive",
				task_name="dfa_step_large",
				parameters=root_parameters,
				task_args=root_parameters,
				subtasks=(
					HTNMethodStep("s1", "helper_target", ("ARG1",), "compound"),
					*(
						HTNMethodStep(
							f"s{i + 2}",
							"helper_context",
							(parameter, "ARG2"),
							"compound",
						)
						for i, parameter in enumerate(root_parameters[2:])
					),
					HTNMethodStep("s99", "make_target", ("ARG1", "ARG2"), "primitive"),
				),
			),
			HTNMethod(
				method_name="m_helper_target",
				task_name="helper_target",
				parameters=("ARG1",),
				subtasks=(),
			),
			HTNMethod(
				method_name="m_helper_context",
				task_name="helper_context",
				parameters=("ARG1", "ARG2"),
				subtasks=(),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	compacted_library, validation_args, projection = (
		LTL_BDI_Pipeline._stage4_compact_validation_library_for_task(
			method_library,
			"dfa_step_large",
			task_args,
			target_literal=HTNLiteral("target", ("obj0", "obj1")),
			compact_arg_threshold=8,
			max_compound_steps=3,
		)
	)

	assert projection is not None
	assert len(validation_args) < len(task_args)
	assert validation_args == task_args[:3]
	compacted_task = compacted_library.task_for_name("dfa_step_large")
	assert compacted_task.parameters == ("ARG1", "ARG2", "AUX0")
	compacted_methods = compacted_library.methods_for_task("dfa_step_large")
	assert [method.method_name for method in compacted_methods] == [
		"m_dfa_step_large_constructive",
	]
	assert [step.task_name for step in compacted_methods[0].subtasks] == [
		"helper_target",
		"helper_context",
		"make_target",
	]
	assert projection["mode"] == "validation_method_body_compaction"
	assert projection["original_task_arg_count"] == len(task_args)
	assert projection["validation_task_arg_count"] == len(validation_args)

def assert_stage5_augmented_seed_facts_include_stage4_replay_worlds(monkeypatch):
	pipeline = LTL_BDI_Pipeline.__new__(LTL_BDI_Pipeline)
	pipeline.predicate_name_map = {"at": "at"}
	monkeypatch.setattr(
		pipeline,
		"_stage6_action_schemas",
		lambda: [
			{
				"functor": "move",
				"source_name": "move",
				"parameters": ["?rover", "?from", "?to"],
				"preconditions": [
					{"predicate": "at", "args": ["?rover", "?from"], "is_positive": True},
				],
				"precondition_clauses": [
					[
						{"predicate": "at", "args": ["?rover", "?from"], "is_positive": True},
					],
				],
				"effects": [
					{"predicate": "at", "args": ["?rover", "?to"], "is_positive": True},
					{"predicate": "at", "args": ["?rover", "?from"], "is_positive": False},
				],
			},
		],
	)

	seed_facts = ("(at rover2 waypoint0)",)
	plan_records = [
		{
			"initial_facts": list(seed_facts),
			"plan": PANDAPlanResult(
				task_name="query_root_1_get_soil_data",
				task_args=("waypoint4",),
				target_literal=None,
				steps=[
					PANDAPlanStep(
						task_name="move",
						action_name="move",
						args=("rover2", "waypoint0", "waypoint4"),
					),
				],
			),
		},
	]

	augmented = pipeline._stage5_augmented_seed_facts_from_plan_records(
		seed_facts,
		plan_records,
	)

	assert "(at rover2 waypoint0)" in augmented
	assert "(at rover2 waypoint4)" in augmented

def assert_stage6_symbolic_literal_binding_resolves_large_ordered_prefix_iteratively():
	pipeline = LTL_BDI_Pipeline.__new__(LTL_BDI_Pipeline)
	symbolic_requirements = [
		HTNLiteral("on", (f"AUX_CTX{index}", f"AUX_CTX{index + 1}"))
		for index in range(1, 1500)
	]
	concrete_history = [
		HTNLiteral("on", (f"b{index}", f"b{index + 1}"))
		for index in range(1, 1500)
	]

	bindings = pipeline._stage6_resolve_symbolic_literal_bindings(
		symbolic_requirements=symbolic_requirements,
		concrete_history=concrete_history,
		seed_bindings={},
	)

	assert bindings is not None
	assert bindings["AUX_CTX1"] == "b1"
	assert bindings["AUX_CTX1499"] == "b1499"
	assert bindings["AUX_CTX1500"] == "b1500"

def assert_stage4_same_library_query_task_gate_plans_unique_query_task_schema_cases():
	problem_path = BLOCKSWORLD_PROBLEM_DIR / "p01.hddl"
	if not problem_path.exists():
		pytest.skip(f"Missing blocksworld problem file: {problem_path}")

	pipeline = LTL_BDI_Pipeline(
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
	)
	pipeline.output_dir = Path(tempfile.mkdtemp(prefix="stage4_query_task_gate_"))
	instruction = BLOCKSWORLD_QUERY_CASES["query_1"]["instruction"]
	ltl_spec = _build_oracle_task_grounded_stage1_spec(pipeline, instruction)
	method_library = _official_domain_method_library(
		"blocksworld",
		target_literal_signatures=list(ltl_spec.query_task_literal_signatures),
		query_task_anchors=list(pipeline._extract_query_task_anchors(ltl_spec.source_instruction)),
	)
	captured: List[Dict[str, Any]] = []

	class DummyPlanner:
		def __init__(self, *args, **kwargs):
			pass

		def plan(self, **kwargs):
			captured.append(
				{
					"task_name": kwargs.get("task_name"),
					"task_args": tuple(kwargs.get("task_args") or ()),
					"task_network": tuple(
						(task_name, tuple(task_args))
						for task_name, task_args in (kwargs.get("task_network") or ())
					),
					"task_network_ordered": kwargs.get("task_network_ordered"),
				},
			)
			task_name = str(kwargs.get("task_name") or "")
			task_args = tuple(str(arg) for arg in (kwargs.get("task_args") or ()))
			return PANDAPlanResult(
				task_name=task_name,
				task_args=task_args,
				target_literal=kwargs.get("target_literal"),
				steps=[],
				raw_plan="\n".join(["==>", "root 1", f"1 {task_name} {' '.join(task_args)} -> noop"]),
				actual_plan="\n".join(["==>", "root 1", f"1 {task_name} {' '.join(task_args)} -> noop"]),
				work_dir=f"dummy-{task_name}",
				timing_profile={},
			)

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(pipeline_module, "PANDAPlanner", DummyPlanner)
	monkeypatch.setattr(
		LTL_BDI_Pipeline,
		"_task_witness_initial_facts",
		lambda self, planner, task_name, method_library, task_args, objects, object_pool, object_types: (),
	)
	try:
		plan_records, _stage4_artifacts = pipeline._stage4_panda_planning(
			ltl_spec,
			method_library,
		)
	finally:
		monkeypatch.undo()

	assert len(plan_records) == 1
	assert plan_records[0]["validation_name"] == "query_task_schema_1_do_put_on"
	assert plan_records[0]["task_network"] == (
		("do_put_on", ("b4", "b2")),
		("do_put_on", ("b1", "b4")),
		("do_put_on", ("b3", "b1")),
	)
	assert captured == [
		{
			"task_name": "do_put_on",
			"task_args": ("b4", "b2"),
			"task_network": (),
			"task_network_ordered": None,
		},
	]

def assert_stage4_query_task_gate_declares_generated_witness_objects():
	problem_path = BLOCKSWORLD_PROBLEM_DIR / "p01.hddl"
	if not problem_path.exists():
		pytest.skip(f"Missing blocksworld problem file: {problem_path}")

	pipeline = LTL_BDI_Pipeline(
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
	)
	pipeline.output_dir = Path(tempfile.mkdtemp(prefix="stage4_witness_objects_"))
	instruction = BLOCKSWORLD_QUERY_CASES["query_1"]["instruction"]
	ltl_spec = _build_oracle_task_grounded_stage1_spec(pipeline, instruction)
	method_library = _official_domain_method_library(
		"blocksworld",
		target_literal_signatures=list(ltl_spec.query_task_literal_signatures),
		query_task_anchors=list(pipeline._extract_query_task_anchors(ltl_spec.source_instruction)),
	)
	captured: List[Dict[str, Any]] = []

	class DummyPlanner:
		def __init__(self, *args, **kwargs):
			pass

		def plan(self, **kwargs):
			captured.append(
				{
					"objects": tuple(kwargs.get("objects") or ()),
					"typed_objects": tuple(kwargs.get("typed_objects") or ()),
					"initial_facts": tuple(kwargs.get("initial_facts") or ()),
				},
			)
			task_name = str(kwargs.get("task_name") or "")
			task_args = tuple(str(arg) for arg in (kwargs.get("task_args") or ()))
			return PANDAPlanResult(
				task_name=task_name,
				task_args=task_args,
				target_literal=kwargs.get("target_literal"),
				steps=[],
				raw_plan="\n".join(["==>", "root 1", f"1 {task_name} {' '.join(task_args)} -> noop"]),
				actual_plan="\n".join(["==>", "root 1", f"1 {task_name} {' '.join(task_args)} -> noop"]),
				work_dir=f"dummy-{task_name}",
				timing_profile={},
			)

	def fake_task_witness_initial_facts(
		self,
		planner,
		task_name,
		method_library,
		task_args,
		objects,
		object_pool,
		object_types,
	):
		if "witness_block_1" not in object_pool:
			object_pool.append("witness_block_1")
		object_types["witness_block_1"] = "block"
		return ("(clear witness_block_1)",)

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(pipeline_module, "PANDAPlanner", DummyPlanner)
	monkeypatch.setattr(
		LTL_BDI_Pipeline,
		"_task_witness_initial_facts",
		fake_task_witness_initial_facts,
	)
	try:
		plan_records, _stage4_artifacts = pipeline._stage4_panda_planning(
			ltl_spec,
			method_library,
		)
	finally:
		monkeypatch.undo()

	assert plan_records
	assert captured
	assert "witness_block_1" in captured[0]["objects"]
	assert ("witness_block_1", "block") in captured[0]["typed_objects"]
	assert "(clear witness_block_1)" in captured[0]["initial_facts"]

def assert_stage6_object_types_use_problem_type_map_without_target_binding_scan():
	problem_path = BLOCKSWORLD_PROBLEM_DIR / "p01.hddl"
	if not problem_path.exists():
		pytest.skip(f"Missing blocksworld problem file: {problem_path}")

	pipeline = LTL_BDI_Pipeline(
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
	)
	method_library = SimpleNamespace(target_literals=(), target_task_bindings=())
	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(
		LTL_BDI_Pipeline,
		"_target_literal_type_candidates",
		lambda *args, **kwargs: pytest.fail("Stage 6 should trust problem object types"),
	)
	monkeypatch.setattr(
		LTL_BDI_Pipeline,
		"_task_argument_type_candidates",
		lambda *args, **kwargs: pytest.fail("Stage 6 should not scan target bindings"),
	)
	try:
		object_types = pipeline._stage6_object_types(
			("b1", "b2"),
			method_library,
			seed_facts=("(clear b1)",),
			problem_object_types={"b1": "block", "b2": "block"},
		)
	finally:
		monkeypatch.undo()

	assert object_types == {"b1": "block", "b2": "block"}

def assert_stage6_query_goal_projection_drops_transient_ordered_support_effects():
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="do_clear",
				parameters=("X",),
				is_primitive=False,
				source_predicates=("clear",),
			),
			HTNTask(
				name="do_put_on",
				parameters=("X", "Y"),
				is_primitive=False,
				source_predicates=("on",),
			),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_do_clear",
				task_name="do_clear",
				parameters=("X",),
				task_args=("X",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="make_clear",
						args=("X",),
						kind="primitive",
						action_name="make_clear",
					),
				),
			),
			HTNMethod(
				method_name="m_do_put_on",
				task_name="do_put_on",
				parameters=("X", "Y"),
				task_args=("X", "Y"),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="stack_on",
						args=("X", "Y"),
						kind="primitive",
						action_name="stack_on",
					),
				),
			),
		],
	)
	action_schemas = [
		{
			"functor": "make_clear",
			"parameters": ["X"],
			"effects": [
				{"predicate": "clear", "args": ["X"], "is_positive": True},
			],
		},
		{
			"functor": "stack_on",
			"parameters": ["X", "Y"],
			"effects": [
				{"predicate": "on", "args": ["X", "Y"], "is_positive": True},
				{"predicate": "clear", "args": ["Y"], "is_positive": False},
			],
		},
	]

	goal_facts = pipeline._stage6_query_goal_facts(
		ltl_spec=SimpleNamespace(query_task_literal_signatures=("clear(b2)", "on(b1, b2)")),
		task_network=(
			("do_clear", ("b2",)),
			("do_put_on", ("b1", "b2")),
		),
		method_library=method_library,
		action_schemas=action_schemas,
	)

	assert goal_facts == ("(on b1 b2)",)

def assert_stage6_protected_target_literals_drop_transient_ordered_support_effects():
	pipeline = LTL_BDI_Pipeline(domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="do_clear",
				parameters=("X",),
				is_primitive=False,
				source_predicates=("clear",),
			),
			HTNTask(
				name="do_on_table",
				parameters=("X",),
				is_primitive=False,
				source_predicates=("ontable",),
			),
			HTNTask(
				name="do_put_on",
				parameters=("X", "Y"),
				is_primitive=False,
				source_predicates=("on",),
			),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_do_clear",
				task_name="do_clear",
				parameters=("X",),
				task_args=("X",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="make_clear",
						args=("X",),
						kind="primitive",
						action_name="make_clear",
					),
				),
			),
			HTNMethod(
				method_name="m_do_on_table",
				task_name="do_on_table",
				parameters=("X",),
				task_args=("X",),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="move_to_table",
						args=("X",),
						kind="primitive",
						action_name="move_to_table",
					),
				),
			),
			HTNMethod(
				method_name="m_do_put_on",
				task_name="do_put_on",
				parameters=("X", "Y"),
				task_args=("X", "Y"),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="stack_on",
						args=("X", "Y"),
						kind="primitive",
						action_name="stack_on",
					),
				),
			),
		],
		target_literals=[
			HTNLiteral(predicate="clear", args=("b2",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="ontable", args=("b2",), is_positive=True, source_symbol=None),
			HTNLiteral(predicate="on", args=("b17", "b2"), is_positive=True, source_symbol=None),
		],
	)
	action_schemas = [
		{
			"functor": "make_clear",
			"parameters": ["X"],
			"effects": [
				{"predicate": "clear", "args": ["X"], "is_positive": True},
			],
		},
		{
			"functor": "move_to_table",
			"parameters": ["X"],
			"effects": [
				{"predicate": "ontable", "args": ["X"], "is_positive": True},
			],
		},
		{
			"functor": "stack_on",
			"parameters": ["X", "Y"],
			"effects": [
				{"predicate": "on", "args": ["X", "Y"], "is_positive": True},
				{"predicate": "clear", "args": ["Y"], "is_positive": False},
			],
		},
	]
	ltl_spec = SimpleNamespace(
		source_instruction="complete the tasks do_clear(b2), then do_on_table(b2), then do_put_on(b17, b2).",
		query_task_literal_signatures=("clear(b2)", "ontable(b2)", "on(b17, b2)"),
		query_object_inventory=("b2", "b17"),
	)

	protected_literals = pipeline._stage6_protected_target_literals(
		ltl_spec=ltl_spec,
		method_library=method_library,
		action_schemas=action_schemas,
	)

	assert [literal.to_signature() for literal in protected_literals] == [
		"ontable(b2)",
		"on(b17, b2)",
	]

def assert_stage7_prefers_guided_hierarchical_plan_text():
	problem_path = BLOCKSWORLD_PROBLEM_DIR / "p01.hddl"
	if not problem_path.exists():
		pytest.skip(f"Missing blocksworld problem file: {problem_path}")

	pipeline = LTL_BDI_Pipeline(
		domain_file=OFFICIAL_BLOCKSWORLD_DOMAIN_FILE,
		problem_file=str(problem_path.resolve()),
	)
	pipeline.output_dir = Path(tempfile.mkdtemp(prefix="stage7_verifier_"))
	method_library = HTNMethodLibrary(
		compound_tasks=(
			HTNTask(
				name="query_root_1_do_put_on",
				parameters=("ARG1", "ARG2"),
				is_primitive=False,
				source_predicates=("on",),
				headline_literal=HTNLiteral("on", ("ARG1", "ARG2"), True, None),
				source_name="do_put_on",
			),
			HTNTask(
				name="dfa_step_q1_q2_on_b1_b2",
				parameters=("ARG1", "ARG2"),
				is_primitive=False,
				source_predicates=("on",),
				headline_literal=HTNLiteral("on", ("ARG1", "ARG2"), True, None),
				source_name=None,
			),
		),
		primitive_tasks=(),
		methods=(),
		target_literals=(HTNLiteral("on", ("b1", "b2"), True, None),),
		target_task_bindings=(HTNTargetTaskBinding("on(b1, b2)", "query_root_1_do_put_on"),),
	)
	captured: Dict[str, Any] = {"verify_plan": 0, "verify_plan_text": 0}

	class DummyResult:
		def __init__(self):
			self.tool_available = True
			self.plan_kind = "hierarchical"
			self.verification_result = True
			self.primitive_plan_executable = True
			self.reached_goal_state = True
			self.build_warning = None

		def to_dict(self):
			return {
				"tool_available": True,
				"plan_kind": "hierarchical",
				"verification_result": True,
				"primitive_plan_executable": True,
				"reached_goal_state": True,
				"build_warning": None,
			}

	class DummyVerifier:
		def tool_available(self):
			return True

		def verify_plan_text(self, **kwargs):
			captured["verify_plan_text"] += 1
			captured["plan_text"] = kwargs.get("plan_text")
			return DummyResult()

		def verify_plan(self, **kwargs):
			captured["verify_plan"] += 1
			return DummyResult()

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(pipeline_module, "IPCPlanVerifier", DummyVerifier)
	try:
		result = pipeline._stage7_official_verification(
			None,
			method_library,
			{
				"artifacts": {
					"guided_hierarchical_plan_text": (
						"==>\n"
						"0 stack b1 b2\n"
						"root 2\n"
						"2 query_root_1_do_put_on b1 b2 -> m_query_root_1_do_put_on_constructive 4\n"
						"4 dfa_step_q1_q2_on_b1_b2 b1 b2 -> m_dfa_step_q1_q2_on_b1_b2_constructive\n"
					),
					"action_path": "/tmp/action_path.txt",
					"method_trace": "/tmp/method_trace.json",
				},
			},
		)
	finally:
		monkeypatch.undo()

	assert result is not None
	assert captured["verify_plan_text"] == 1
	assert captured["verify_plan"] == 0
	assert captured["plan_text"] == (
		"==>\n"
		"0 stack b1 b2\n"
		"root 2\n"
		"2 do_put_on b1 b2 -> m_query_root_1_do_put_on_constructive 4\n"
		"4 dfa_step_q1_q2_on_b1_b2 b1 b2 -> m_dfa_step_q1_q2_on_b1_b2_constructive\n"
	)

def _run_query_case(
	query_id: str,
	*,
	query_cases: Dict[str, Dict[str, Any]],
	domain_file: str,
) -> Dict[str, Any]:
	if query_id not in query_cases:
		raise KeyError(
			f"Unknown query id '{query_id}'. Available query ids: {sorted(query_cases, key=_benchmark_query_id_sort_key)}",
		)

	case = query_cases[query_id]
	domain_action_names = set(HDDLParser.parse_domain(domain_file).get_action_names())
	pipeline = LTL_BDI_Pipeline(
		domain_file=domain_file,
		problem_file=case.get("problem_file"),
	)
	pipeline.logger = PipelineLogger(logs_dir=str(TESTS_GENERATED_LOGS_DIR), run_origin="tests")

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

	stage3_library = _load_stage3_method_library(execution, log_dir)
	target_bindings = stage3_library.get("target_task_bindings") or []
	if not target_bindings:
		bug_messages.append("Stage 3 produced no target_task_bindings")
	bug_messages.extend(_binding_semantic_messages(stage3_library))

	stage5_code = _load_stage5_agentspeak_code(execution, log_dir)
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

	stage6_artifacts = _load_stage6_artifacts(execution, log_dir)
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

	stage7_artifacts = _load_stage7_artifacts(execution, log_dir)
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

def assert_stage7_verifies_exported_hierarchical_plan_even_when_planner_available(
	tmp_path,
	monkeypatch,
):
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
			raise AssertionError(
				"Stage 7 should verify the exported runtime plan rather than replanning it"
			)

		def verify_plan(self, **kwargs):
			calls.append("trace")
			assert kwargs["action_path"] == ["stack(a, b)"]
			assert kwargs["method_trace"] == [{"method_name": "m_put_on", "task_args": ["a", "b"]}]
			return FakeResult()

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
		stage6_data={
			"artifacts": {
				"action_path": ["stack(a, b)"],
				"method_trace": [{"method_name": "m_put_on", "task_args": ["a", "b"]}],
			},
		},
	)

	assert calls == ["trace"]
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

@pytest.mark.parametrize("query_id", sorted(BLOCKSWORLD_QUERY_CASES, key=_benchmark_query_id_sort_key))
def test_blocksworld_pipeline_query_case_with_official_stage3_mask(query_id: str, monkeypatch):
	_ensure_live_dependencies()
	report = _run_query_case_with_official_stage3_mask(query_id, monkeypatch)
	assert report["has_bug"] is False, "\n".join(report["bug_messages"])

@pytest.mark.parametrize("query_id", sorted(MARSROVER_QUERY_CASES, key=_benchmark_query_id_sort_key))
def test_marsrover_pipeline_query_case_with_official_stage3_mask(query_id: str, monkeypatch):
	_ensure_live_dependencies()
	report = _run_marsrover_query_case_with_official_stage3_mask(query_id, monkeypatch)
	assert report["has_bug"] is False, "\n".join(report["bug_messages"])

@pytest.mark.parametrize("query_id", sorted(SATELLITE_QUERY_CASES, key=_benchmark_query_id_sort_key))
def test_satellite_pipeline_query_case_with_official_stage3_mask(query_id: str, monkeypatch):
	_ensure_live_dependencies()
	report = _run_satellite_query_case_with_official_stage3_mask(query_id, monkeypatch)
	assert report["has_bug"] is False, "\n".join(report["bug_messages"])

@pytest.mark.parametrize("query_id", sorted(TRANSPORT_QUERY_CASES, key=_benchmark_query_id_sort_key))
def test_transport_pipeline_query_case_with_official_stage3_mask(query_id: str, monkeypatch):
	_ensure_live_dependencies()
	report = _run_transport_query_case_with_official_stage3_mask(query_id, monkeypatch)
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
	execution = report["execution"] or {}
	log_dir = Path(report["log_dir"]) if report.get("log_dir") else None
	stage3_metadata = execution.get("stage3_metadata") or {}
	stage3_library = (
		_load_stage3_method_library(execution, log_dir)
		if log_dir is not None
		else (execution.get("stage3_method_library") or {})
	)

	print(f"{query_id}: {case['description']}")
	print(f"Instruction: {case['instruction']}")
	print(f"Log Dir: {report['log_dir']}")
	print(f"Success: {report['result']['success']}")
	print(f"Stage 3 Target Literals: {stage3_metadata.get('target_literals', [])}")
	print(
		f"Stage 3 Target Task Bindings: "
		f"{stage3_library.get('target_task_bindings', [])}",
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

def test_print_cli_report_handles_missing_stage3_metadata(capsys):
	_print_cli_report(
		{
			"query_id": "query_3",
			"case": {
				"description": "Auto-generated from p03.hddl",
				"instruction": "Complete the tasks do_put_on(b3, b4).",
			},
			"log_dir": "/tmp/fake-log",
			"result": {"success": False},
			"execution": {
				"stage3_metadata": None,
				"stage3_method_library": None,
			},
			"official_verifier": None,
			"has_bug": True,
			"bug_messages": ["pipeline returned success=False"],
		},
	)

	rendered = capsys.readouterr().out
	assert "Stage 3 Target Literals: []" in rendered
	assert "Stage 3 Target Task Bindings: []" in rendered

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

	try:
		domain_key, query_cases, selector = _resolve_cli_selection(argv)
	except ValueError as exc:
		print(str(exc))
		return 2
	if selector == "list":
		for cli_domain_key, case_map in CLI_QUERY_CASE_GROUPS.items():
			print(f"[{cli_domain_key}]")
			for query_id in sorted(case_map, key=_benchmark_query_id_sort_key):
				case = case_map[query_id]
				print(f"{query_id}: {case['description']}")
				print(f"Instruction: {case['instruction']}")
				print("")
		return 0
	if selector == "all":
		query_ids = sorted(query_cases, key=_benchmark_query_id_sort_key)
	else:
		if selector not in query_cases:
			print(
				f"Unknown query id '{selector}' for domain '{domain_key}'. "
				f"Available: {sorted(query_cases, key=_benchmark_query_id_sort_key)} or 'all'",
			)
			return 2
		query_ids = [selector]

	reports = [
		_run_query_case(
			query_id,
			query_cases=query_cases,
			domain_file=CLI_DOMAIN_FILES[domain_key],
		)
		for query_id in query_ids
	]
	for report in reports:
		_print_cli_report(report)

	return 1 if any(report["has_bug"] for report in reports) else 0

if __name__ == "__main__":
	raise SystemExit(main(sys.argv))
