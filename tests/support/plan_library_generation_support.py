from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
	sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from method_library.synthesis.schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
)
from method_library.synthesis.synthesizer import HTNMethodSynthesizer
from execution_logging.execution_logger import ExecutionLogger
from plan_library import PlanLibraryGenerationPipeline, load_plan_library_artifact_bundle
from utils.config import (
	DEFAULT_METHOD_SYNTHESIS_TIMEOUT_SECONDS,
	DEFAULT_PLANNING_TIMEOUT_SECONDS,
)
from utils.benchmark_query_dataset import (
	DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS,
	DEFAULT_BENCHMARK_QUERY_DOMAIN_PROBLEM_DIRS,
	load_problem_query_cases,
)
from utils.hddl_condition_parser import HDDLConditionParser
from utils.hddl_parser import HDDLParser


TESTS_ROOT = PROJECT_ROOT / "tests"
GENERATED_ROOT = TESTS_ROOT / "method_library" / "generated"
GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
GENERATED_LOGS_DIR = GENERATED_ROOT / "logs"
GENERATED_LOGS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DOMAIN_BUILDS_DIR = GENERATED_ROOT / "domain_builds"
GENERATED_DOMAIN_BUILDS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_BASELINE_DIR = GENERATED_ROOT / "official_ground_truth"
GENERATED_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_MASKED_DOMAIN_BUILDS_DIR = GENERATED_DOMAIN_BUILDS_DIR / "generated_masked"
GENERATED_MASKED_DOMAIN_BUILDS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_MASKED_BASELINE_DIR = GENERATED_ROOT / "generated_masked_problem_root"
GENERATED_MASKED_BASELINE_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_FILES = {
	domain_key: str((PROJECT_ROOT / "src" / "domains" / domain_key / "domain.hddl").resolve())
	for domain_key in ("blocksworld", "marsrover", "satellite", "transport")
}
DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS = DEFAULT_METHOD_SYNTHESIS_TIMEOUT_SECONDS
DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS = DEFAULT_PLANNING_TIMEOUT_SECONDS


def _coerce_timeout_seconds(value: object, fallback: int) -> int:
	try:
		seconds = int(str(value))
	except (TypeError, ValueError):
		return int(fallback)
	return max(seconds, 1)


def apply_generated_runtime_defaults(
	env: Optional[MutableMapping[str, str]] = None,
) -> MutableMapping[str, str]:
	target_env = env if env is not None else os.environ
	method_synthesis_timeout = max(
		_coerce_timeout_seconds(
			target_env.get("METHOD_SYNTHESIS_TIMEOUT"),
			DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS,
		),
		DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS,
	)
	planning_timeout = max(
		_coerce_timeout_seconds(
			target_env.get("PLANNING_TIMEOUT"),
			DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS,
		),
		DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS,
	)
	target_env["METHOD_SYNTHESIS_TIMEOUT"] = str(method_synthesis_timeout)
	target_env["PLANNING_TIMEOUT"] = str(planning_timeout)
	target_env.setdefault("METHOD_SYNTHESIS_PROGRESS", "1")
	target_env.setdefault("DOMAIN_GATE_PROGRESS", "1")
	return target_env


def query_id_sort_key(query_id: str) -> tuple[int, str]:
	match = re.fullmatch(r"query_(\d+)", str(query_id).strip())
	if match is None:
		return (10**9, str(query_id))
	return (int(match.group(1)), str(query_id))


def load_domain_query_cases(domain_key: str, *, limit: int = 0) -> Dict[str, Dict[str, Any]]:
	pattern = DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS[domain_key]
	query_limit = limit if limit > 0 else 10**9
	return load_problem_query_cases(
		DEFAULT_BENCHMARK_QUERY_DOMAIN_PROBLEM_DIRS[domain_key],
		limit=query_limit,
		pattern=pattern,
	)


def build_method_library_from_domain_file(domain_file: str) -> HTNMethodLibrary:
	domain = HDDLParser.parse_domain(domain_file)
	condition_parser = HDDLConditionParser()
	synthesizer = HTNMethodSynthesizer()
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
				predicate=literal.predicate,
				args=tuple(literal.args),
				is_positive=literal.is_positive,
				source_symbol=None,
			)
			for literal in condition_parser.parse_literals(
				method.precondition,
				action_name=method.name,
				scope="official_method_precondition",
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
				source_method_name=method.name,
			)
		)
	return HTNMethodLibrary(
		compound_tasks=compound_tasks,
		primitive_tasks=primitive_tasks,
		methods=methods,
		target_literals=[],
		target_task_bindings=[],
	)


def build_official_method_library(domain_file: str) -> HTNMethodLibrary:
	return build_method_library_from_domain_file(domain_file)


def run_generated_domain_build(domain_key: str) -> Dict[str, Any]:
	apply_generated_runtime_defaults()
	domain_file = DOMAIN_FILES[domain_key]
	artifact_root = GENERATED_MASKED_DOMAIN_BUILDS_DIR / domain_key
	pipeline = PlanLibraryGenerationPipeline(domain_file=domain_file)
	pipeline.logger = ExecutionLogger(logs_dir=str(GENERATED_LOGS_DIR), run_origin="tests")
	result = pipeline.build_library_bundle(output_root=str(artifact_root))
	log_dir = Path(str(result["log_path"])).parent if result.get("log_path") else None
	execution = (
		json.loads((log_dir / "execution.json").read_text())
		if log_dir is not None and (log_dir / "execution.json").exists()
		else {}
	)
	artifact = (
		load_plan_library_artifact_bundle(artifact_root)
		if bool(result.get("success")) and artifact_root.exists()
		else None
	)
	method_synthesis_metadata = dict(
		((artifact.method_synthesis_metadata if artifact is not None else None) or {}),
	)
	artifact_summary = dict(result.get("artifact_summary") or result.get("artifact") or {})
	return {
		"success": bool(result.get("success")),
		"artifact_root": artifact_root,
		"artifact_paths": dict(result.get("artifact_paths") or {}),
		"artifact": artifact,
		"log_dir": log_dir,
		"execution": execution,
		"library_validation": dict(artifact_summary.get("library_validation") or {}),
		"method_synthesis_metadata": method_synthesis_metadata,
		"source_domain_kind": str(method_synthesis_metadata.get("source_domain_kind") or ""),
		"llm_request_id": str(method_synthesis_metadata.get("llm_request_id") or ""),
		"llm_response_mode": str(method_synthesis_metadata.get("llm_response_mode") or ""),
		"llm_stream_handshake_seconds": method_synthesis_metadata.get(
			"llm_stream_handshake_seconds",
		),
		"llm_first_stream_chunk_seconds": method_synthesis_metadata.get(
			"llm_first_stream_chunk_seconds",
		),
		"llm_first_chunk_seconds": method_synthesis_metadata.get("llm_first_chunk_seconds"),
		"llm_first_content_chunk_seconds": method_synthesis_metadata.get(
			"llm_first_content_chunk_seconds",
		),
		"llm_complete_json_seconds": method_synthesis_metadata.get("llm_complete_json_seconds"),
		"llm_attempted": bool(method_synthesis_metadata.get("llm_attempted")),
		"llm_generation_attempts": int(method_synthesis_metadata.get("llm_generation_attempts") or 0),
		"llm_attempts": int(method_synthesis_metadata.get("llm_attempts") or 0),
		"method_synthesis_model": str(method_synthesis_metadata.get("model") or ""),
		"generated_method_count": len(
			list((artifact.method_library.methods if artifact is not None else ()))
		),
	}


__all__ = [
	"DEFAULT_GENERATED_METHOD_SYNTHESIS_TIMEOUT_SECONDS",
	"DEFAULT_GENERATED_PLANNING_TIMEOUT_SECONDS",
	"DOMAIN_FILES",
	"GENERATED_BASELINE_DIR",
	"GENERATED_DOMAIN_BUILDS_DIR",
	"GENERATED_LOGS_DIR",
	"GENERATED_MASKED_BASELINE_DIR",
	"GENERATED_MASKED_DOMAIN_BUILDS_DIR",
	"apply_generated_runtime_defaults",
	"build_method_library_from_domain_file",
	"build_official_method_library",
	"load_domain_query_cases",
	"query_id_sort_key",
	"run_generated_domain_build",
]
