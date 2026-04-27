"""
Jason runtime runner.

Runs generated AgentSpeak code with Jason (RunLocalMAS), boots a real Jason
`Environment` implementation for domain action semantics, and returns structured
runtime metadata for pipeline logging.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from method_library.synthesis.naming import query_root_alias_task_name
from method_library.synthesis.schema import HTNMethodLibrary
from evaluation.jason_runtime.environment_adapter import (
	EnvironmentAdapterResult,
	Stage6EnvironmentAdapter,
	build_environment_adapter,
)
from plan_library.models import PlanLibrary


class JasonValidationError(RuntimeError):
	"""Raised when Jason runtime validation fails."""

	def __init__(self, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
		super().__init__(message)
		self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class JasonValidationResult:
	"""Structured result for Jason runtime validation."""

	status: str
	backend: str
	java_path: Optional[str]
	java_version: Optional[int]
	javac_path: Optional[str]
	jason_jar: Optional[str]
	exit_code: Optional[int]
	timed_out: bool
	stdout: str
	stderr: str
	action_path: List[str]
	method_trace: List[Dict[str, Any]]
	failed_goals: List[str]
	environment_adapter: Dict[str, Any]
	failure_class: Optional[str]
	consistency_checks: Dict[str, Any]
	artifacts: Dict[str, Any]
	timing_profile: Dict[str, Any]

	def to_dict(self) -> Dict[str, Any]:
		return {
			"status": self.status,
			"backend": self.backend,
			"java_path": self.java_path,
			"java_version": self.java_version,
			"javac_path": self.javac_path,
			"jason_jar": self.jason_jar,
			"exit_code": self.exit_code,
			"timed_out": self.timed_out,
			"stdout": self.stdout,
			"stderr": self.stderr,
			"action_path": list(self.action_path),
			"method_trace": list(self.method_trace),
			"failed_goals": list(self.failed_goals),
			"environment_adapter": dict(self.environment_adapter),
			"failure_class": self.failure_class,
			"consistency_checks": dict(self.consistency_checks),
			"artifacts": dict(self.artifacts),
			"timing_profile": dict(self.timing_profile),
		}

	def to_compact_dict(self) -> Dict[str, Any]:
		"""Return a bounded validation record for JSON logs.

		The full stdout/stderr/action path/method trace are written as separate
		artifacts by the runner. Keeping them out of this JSON prevents sweep
		summaries from duplicating large runtime traces.
		"""

		artifacts = dict(self.artifacts)
		return {
			"status": self.status,
			"backend": self.backend,
			"java_path": self.java_path,
			"java_version": self.java_version,
			"javac_path": self.javac_path,
			"jason_jar": self.jason_jar,
			"exit_code": self.exit_code,
			"timed_out": self.timed_out,
			"stdout_path": artifacts.get("jason_stdout"),
			"stderr_path": artifacts.get("jason_stderr"),
			"stdout_bytes": len(self.stdout.encode("utf-8")),
			"stderr_bytes": len(self.stderr.encode("utf-8")),
			"stdout_tail": _tail_text(self.stdout),
			"stderr_tail": _tail_text(self.stderr),
			"action_path_count": len(self.action_path),
			"method_trace_count": len(self.method_trace),
			"failed_goals": list(self.failed_goals),
			"environment_adapter": dict(self.environment_adapter),
			"failure_class": self.failure_class,
			"consistency_checks": dict(self.consistency_checks),
			"artifacts": artifacts,
			"timing_profile": dict(self.timing_profile),
		}


def _tail_text(text: str, *, limit: int = 4000) -> str:
	value = str(text or "")
	if len(value) <= limit:
		return value
	return value[-limit:]


def _extract_runtime_plan_projection(agentspeak_code: str) -> str:
	"""Extract the runtime-grounded method-plan projection from a runnable program."""

	text = str(agentspeak_code or "")
	start_marker = "/* HTN Method Plans */"
	start_index = text.find(start_marker)
	if start_index == -1:
		return ""
	end_candidates = [
		index
		for marker in ("/* Failure Handlers */", "/* Execution Entry */")
		for index in [text.find(marker, start_index + len(start_marker))]
		if index != -1
	]
	end_index = min(end_candidates) if end_candidates else len(text)
	return text[start_index:end_index].strip() + "\n"


class JasonRunner:
	"""Run rendered AgentSpeak code in Jason and validate runtime outcomes."""

	backend_name = "RunLocalMAS"
	success_marker = "execute success"
	failure_marker = "execute failed"
	failed_goal_record_limit = 64
	runtime_output_artifact_limit_chars = 500_000
	method_trace_record_limit = 2_000
	min_java_major = 23
	max_java_major = 23
	environment_class_name = "JasonPipelineEnvironment"

	def __init__(
		self,
		*,
		runtime_dir: str | Path | None = None,
		timeout_seconds: int = 120,
		environment_adapter: Stage6EnvironmentAdapter | None = None,
		environment_adapter_name: str | None = None,
	) -> None:
		base_dir = (
			Path(runtime_dir).resolve()
			if runtime_dir is not None
			else Path(__file__).resolve().parent
		)
		self.runtime_dir = base_dir
		self.jason_src_dir = self.runtime_dir / "jason_src"
		self.timeout_seconds = timeout_seconds
		adapter_name = (
			environment_adapter_name
			or os.getenv("JASON_RUNTIME_ENV_ADAPTER")
			or os.getenv("STAGE6_ENV_ADAPTER")
		)
		self.environment_adapter = environment_adapter or build_environment_adapter(adapter_name)
		self._action_schema_lookup_cache: Dict[int, Dict[str, Dict[str, Any]]] = {}

	def validate(
		self,
		*,
		agentspeak_code: str,
		action_schemas: Sequence[Dict[str, Any]],
		method_library: HTNMethodLibrary | None = None,
		plan_library: PlanLibrary | None = None,
		seed_facts: Sequence[str] = (),
		runtime_objects: Sequence[str] = (),
		object_types: Optional[Dict[str, str]] = None,
		type_parent_map: Optional[Dict[str, Optional[str]]] = None,
		query_goals: Sequence[Any] = (),
		goal_facts: Sequence[str] = (),
		domain_name: str,
		problem_file: str | Path | None = None,
		output_dir: str | Path,
	) -> JasonValidationResult:
		"""Execute Jason validation and return a structured result."""
		total_start = time.perf_counter()
		timing_profile: Dict[str, float] = {}

		if not action_schemas:
			raise JasonValidationError(
				"Jason runtime requires action schemas for real environment execution.",
				metadata={"action_schema_count": 0},
			)

		output_path = Path(output_dir).resolve()
		output_path.mkdir(parents=True, exist_ok=True)

		runtime_resolution_start = time.perf_counter()
		java_bin, java_major = self._select_java_binary()
		javac_bin = self._select_javac_binary(java_bin)
		jason_jar = self._ensure_jason_jar(java_bin)
		log_conf = self._resolve_log_config()
		timing_profile["runtime_resolution_seconds"] = (
			time.perf_counter() - runtime_resolution_start
		)

		runner_asl_path = output_path / "agentspeak_generated.asl"
		runtime_projection_path = output_path / "runtime_grounding_projection.asl"
		runner_mas2j_path = output_path / "jason_runner.mas2j"
		env_java_path = output_path / f"{self.environment_class_name}.java"
		env_class_path = output_path / f"{self.environment_class_name}.class"
		no_ancestor_goal_java_path = output_path / "pipeline" / "no_ancestor_goal.java"
		no_ancestor_goal_class_path = output_path / "pipeline" / "no_ancestor_goal.class"
		choose_runtime_choice_java_path = output_path / "pipeline" / "choose_runtime_choice.java"
		choose_runtime_choice_class_path = output_path / "pipeline" / "choose_runtime_choice.class"
		stdout_path = output_path / "jason_stdout.txt"
		stderr_path = output_path / "jason_stderr.txt"
		action_path_path = output_path / "action_path.txt"
		method_trace_path = output_path / "method_trace.json"
		validation_json_path = output_path / "jason_validation.json"

		source_build_start = time.perf_counter()
		runtime_agentspeak_code = self._strip_seed_fact_beliefs(
			agentspeak_code,
			seed_facts=seed_facts,
		)
		runner_asl = self._build_runner_asl(
			runtime_agentspeak_code,
			method_library=method_library,
			plan_library=plan_library,
			action_schemas=action_schemas,
			seed_facts=seed_facts,
			runtime_objects=runtime_objects,
			object_types=object_types or {},
			type_parent_map=type_parent_map or {},
			query_goals=query_goals,
			goal_facts=goal_facts,
		)
		runner_mas2j = self._build_runner_mas2j(domain_name)
		env_source = self._build_environment_java_source(
			action_schemas=action_schemas,
			seed_facts=seed_facts,
		)
		no_ancestor_goal_source = self._build_no_ancestor_goal_internal_action_source()
		choose_runtime_choice_source = self._build_choose_runtime_choice_internal_action_source()
		timing_profile["source_build_seconds"] = time.perf_counter() - source_build_start
		write_sources_start = time.perf_counter()
		runtime_plan_projection = _extract_runtime_plan_projection(runner_asl)
		runner_asl_path.write_text(runner_asl)
		runtime_projection_path.write_text(runtime_plan_projection)
		runner_mas2j_path.write_text(runner_mas2j)
		env_java_path.write_text(env_source)
		no_ancestor_goal_java_path.parent.mkdir(parents=True, exist_ok=True)
		no_ancestor_goal_java_path.write_text(no_ancestor_goal_source)
		choose_runtime_choice_java_path.write_text(choose_runtime_choice_source)
		timing_profile["write_sources_seconds"] = time.perf_counter() - write_sources_start
		compile_start = time.perf_counter()
		self._compile_environment_java(
			java_bin=java_bin,
			javac_bin=javac_bin,
			jason_jar=jason_jar,
			env_java_path=env_java_path,
			output_path=output_path,
		)
		timing_profile["environment_compile_seconds"] = time.perf_counter() - compile_start
		if not env_class_path.exists():
			raise JasonValidationError(
				"Jason environment class compilation completed but class file is missing.",
				metadata={
					"environment_java": str(env_java_path),
					"environment_class": str(env_class_path),
				},
			)
		needs_recursive_ancestor_guard = "pipeline.no_ancestor_goal(" in runner_asl
		if needs_recursive_ancestor_guard and not no_ancestor_goal_class_path.exists():
			raise JasonValidationError(
				"Jason internal action compilation completed but class file is missing.",
				metadata={
					"internal_action_java": str(no_ancestor_goal_java_path),
					"internal_action_class": str(no_ancestor_goal_class_path),
				},
			)
		needs_choice_stack = "pipeline.choose_runtime_choice(" in runner_asl
		if needs_choice_stack and not choose_runtime_choice_class_path.exists():
			raise JasonValidationError(
				"Jason choice-stack internal action compilation completed but class file is missing.",
				metadata={
					"internal_action_java": str(choose_runtime_choice_java_path),
					"internal_action_class": str(choose_runtime_choice_class_path),
				},
			)

		runtime_classpath = os.pathsep.join([str(jason_jar), str(output_path)])
		command = [
			java_bin,
			"-cp",
			runtime_classpath,
			"jason.infra.local.RunLocalMAS",
			runner_mas2j_path.name,
			"--log-conf",
			str(log_conf),
		]

		timed_out = False
		exit_code: Optional[int] = None
		raw_stdout: str | bytes = ""
		raw_stderr: str | bytes = ""

		try:
			mas_run_start = time.perf_counter()
			result = subprocess.run(
				command,
				cwd=output_path,
				text=True,
				capture_output=True,
				check=False,
				timeout=self.timeout_seconds,
			)
			exit_code = result.returncode
			raw_stdout = result.stdout
			raw_stderr = result.stderr
			timing_profile["mas_run_seconds"] = time.perf_counter() - mas_run_start
		except subprocess.TimeoutExpired as exc:
			timed_out = True
			raw_stdout = exc.stdout or ""
			raw_stderr = exc.stderr or ""
			timing_profile["mas_run_seconds"] = time.perf_counter() - mas_run_start

		output_processing_start = time.perf_counter()
		stdout_text = self._normalise_process_output(raw_stdout)
		stderr_text = self._normalise_process_output(raw_stderr)
		stdout = self._combine_process_output(stdout_text, stderr_text)
		stderr = stderr_text
		action_path = self._extract_action_path(stdout)
		method_trace_output = stderr_text if "runtime trace method" in stderr_text else stdout
		raw_method_trace = self._extract_method_trace(method_trace_output)
		method_trace, method_trace_original_count, method_trace_truncated = (
			self._cap_method_trace_records(raw_method_trace)
		)
		observed_failed_goals = self._extract_failed_goals(stdout)
		goal_repair_pass_count = self._extract_goal_repair_pass_count(stdout)
		timing_profile["output_processing_seconds"] = (
			time.perf_counter() - output_processing_start
		)

		artifact_write_start = time.perf_counter()
		stdout_artifact, stdout_truncated = self._bounded_runtime_output_artifact(stdout)
		stderr_artifact, stderr_truncated = self._bounded_runtime_output_artifact(stderr)
		stdout_path.write_text(stdout_artifact)
		stderr_path.write_text(stderr_artifact)
		action_path_path.write_text(self._render_action_path(action_path))
		method_trace_path.write_text(json.dumps(method_trace, indent=2))
		timing_profile["artifact_write_seconds"] = time.perf_counter() - artifact_write_start

		artifacts = {
			"source_plan_library_kind": "S",
			"runtime_projection_kind": "S_{I,g}",
			"runtime_projection_scope": "evaluation_grounding_projection",
			"agentspeak_generated": str(runner_asl_path),
			"runtime_grounding_projection": str(runtime_projection_path),
			"jason_runner_mas2j": str(runner_mas2j_path),
			"runtime_environment_java": str(env_java_path),
			"runtime_environment_class": str(env_class_path),
			"jason_stdout": str(stdout_path),
			"jason_stderr": str(stderr_path),
			"action_path": str(action_path_path),
			"method_trace": str(method_trace_path),
			"jason_validation": str(validation_json_path),
			"goal_repair_pass_count": goal_repair_pass_count,
			"stdout_artifact_truncated": stdout_truncated,
			"stderr_artifact_truncated": stderr_truncated,
			"stdout_chars": len(stdout),
			"stderr_chars": len(stderr),
			"stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
			"stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
			"method_trace_original_count": method_trace_original_count,
			"method_trace_truncated": method_trace_truncated,
		}
		environment_validation_start = time.perf_counter()
		environment_result = self.environment_adapter.validate(stdout=stdout, stderr=stderr)
		timing_profile["environment_validation_seconds"] = (
			time.perf_counter() - environment_validation_start
		)
		consistency_start = time.perf_counter()
		try:
			consistency_checks = self._run_consistency_checks(
				action_path=action_path,
				method_trace=method_trace,
				method_library=method_library,
				action_schemas=action_schemas,
				seed_facts=seed_facts,
				problem_file=problem_file,
				skip_method_trace_diagnostics=(
					goal_repair_pass_count > 1
					or method_trace_truncated
				),
			)
		except Exception as exc:
			consistency_checks = {
				"diagnostics_only": True,
				"success": None,
				"failure_class": "consistency_diagnostics_exception",
				"message": str(exc),
			}
		timing_profile["consistency_checks_seconds"] = time.perf_counter() - consistency_start
		is_success = self._is_successful_run(
			stdout=stdout,
			exit_code=exit_code,
			timed_out=timed_out,
			environment_result=environment_result,
		)
		status = "success" if is_success else "failed"
		failure_class = None if is_success else self._failure_class(
			stdout,
			exit_code,
			timed_out,
			environment_result,
		)
		failed_goals = [] if is_success else observed_failed_goals
		artifacts["recovered_failed_goals"] = (
			observed_failed_goals if is_success else []
		)
		timing_profile["total_seconds"] = time.perf_counter() - total_start
		result_payload = JasonValidationResult(
			status=status,
			backend=self.backend_name,
			java_path=java_bin,
			java_version=java_major,
			javac_path=javac_bin,
			jason_jar=str(jason_jar),
			exit_code=exit_code,
			timed_out=timed_out,
			stdout=stdout,
			stderr=stderr,
			action_path=action_path,
			method_trace=method_trace,
			failed_goals=failed_goals,
			environment_adapter=environment_result.to_dict(),
			failure_class=failure_class,
			consistency_checks=consistency_checks,
			artifacts=artifacts,
			timing_profile=timing_profile,
		)
		validation_json_path.write_text(json.dumps(result_payload.to_compact_dict(), indent=2))

		if not is_success:
			failure_reason = self._failure_reason(
				stdout,
				stderr,
				exit_code,
				timed_out,
				environment_result,
			)
			raise JasonValidationError(
				f"Jason runtime validation failed: {failure_reason}",
				metadata=result_payload.to_compact_dict(),
			)

		return result_payload

	def _action_schema_lookup(
		self,
		action_schemas: Sequence[Dict[str, Any]],
	) -> Dict[str, Dict[str, Any]]:
		cache_key = id(action_schemas)
		cached = self._action_schema_lookup_cache.get(cache_key)
		if cached is not None:
			return cached
		schema_lookup: Dict[str, Dict[str, Any]] = {}
		for schema in action_schemas:
			functor = str(schema.get("functor", "")).strip()
			source_name = str(schema.get("source_name", "")).strip()
			if functor:
				schema_lookup.setdefault(functor, schema)
			if source_name:
				schema_lookup.setdefault(source_name, schema)
		if len(self._action_schema_lookup_cache) >= 16:
			self._action_schema_lookup_cache.clear()
		self._action_schema_lookup_cache[cache_key] = schema_lookup
		return schema_lookup

	def _replay_plan_steps_into_world(
		self,
		*,
		world: Set[str],
		steps: Sequence[Any],
		schema_lookup: Dict[str, Dict[str, Any]],
	) -> Set[str]:
		next_world = set(world)
		for step in steps:
			step_name = str(getattr(step, "action_name", None) or getattr(step, "task_name", "")).strip()
			schema = schema_lookup.get(step_name)
			if schema is None:
				continue
			parameters = [str(item) for item in (schema.get("parameters") or [])]
			bindings: Dict[str, str] = {}
			for parameter, value in zip(parameters, getattr(step, "args", ()) or ()):
				token = self._canonical_runtime_token(parameter)
				bindings[token] = str(value)
				if token.startswith("?"):
					bindings[token[1:]] = str(value)

			for effect in self._ordered_runtime_effects(schema.get("effects") or []):
				predicate = str(effect.get("predicate", "")).strip()
				if not predicate or predicate == "=":
					continue
				grounded = self._ground_runtime_pattern(
					predicate,
					effect.get("args") or [],
					bindings,
				)
				if effect.get("is_positive", True):
					next_world.add(grounded)
				else:
					next_world.discard(grounded)
		return next_world

	def _runtime_world_to_hddl_facts(
		self,
		world: Sequence[str],
		*,
		predicate_name_map: Optional[Dict[str, str]] = None,
	) -> Tuple[str, ...]:
		return tuple(
			self._runtime_atom_to_hddl_fact(
				atom,
				predicate_name_map=predicate_name_map,
			)
			for atom in sorted(world)
			if atom
		)

	@classmethod
	def _runtime_atom_to_hddl_fact(
		cls,
		atom: str,
		*,
		predicate_name_map: Optional[Dict[str, str]] = None,
	) -> str:
		text = (atom or "").strip()
		if not text:
			return "()"
		if "(" not in text:
			predicate = (
				predicate_name_map.get(text, text)
				if predicate_name_map is not None
				else text
			)
			return f"({predicate})"
		functor, remainder = text.split("(", 1)
		functor = functor.strip()
		predicate = (
			predicate_name_map.get(functor, functor)
			if predicate_name_map is not None
			else functor
		)
		args_text = remainder[:-1].strip()
		if not args_text:
			return f"({predicate})"
		args = [
			cls._canonical_runtime_token(part.strip())
			for part in args_text.split(",")
			if part.strip()
		]
		return f"({predicate} {' '.join(args)})"

	@classmethod
	def _runtime_predicate_name_map(
		cls,
		*,
		action_schemas: Sequence[Dict[str, Any]] = (),
		predicate_names: Sequence[str] = (),
	) -> Dict[str, str]:
		mapping: Dict[str, str] = {}

		def register(name: Any) -> None:
			source_name = str(name or "").strip()
			if not source_name or source_name == "=":
				return
			mapping.setdefault(cls._sanitize_name(source_name), source_name)
			mapping.setdefault(source_name, source_name)

		for predicate_name in predicate_names or ():
			register(predicate_name)
		for schema in action_schemas or ():
			for collection_name in ("preconditions", "effects"):
				for literal in schema.get(collection_name) or ():
					if isinstance(literal, dict):
						register(literal.get("predicate"))
			for clause in schema.get("precondition_clauses") or ():
				for literal in clause or ():
					if isinstance(literal, dict):
						register(literal.get("predicate"))
		return mapping

	def toolchain_available(self) -> bool:
		"""Return whether Java and Jason runtime requirements are available."""

		try:
			java_bin, _ = self._select_java_binary()
			self._select_javac_binary(java_bin)
			self._ensure_jason_jar(java_bin)
			self._resolve_log_config()
			return True
		except Exception:
			return False

	def _build_runner_asl(
		self,
		agentspeak_code: str,
		*,
		method_library: HTNMethodLibrary | None = None,
		plan_library: PlanLibrary | None = None,
		action_schemas: Sequence[Dict[str, Any]] = (),
		seed_facts: Sequence[str] = (),
		runtime_objects: Sequence[str] = (),
		object_types: Optional[Dict[str, str]] = None,
		type_parent_map: Optional[Dict[str, Optional[str]]] = None,
		query_goals: Sequence[Any] = (),
		goal_facts: Sequence[str] = (),
	) -> str:
		runtime_ready_code = self._inject_runtime_object_beliefs(
			agentspeak_code,
			seed_facts=seed_facts,
			runtime_objects=runtime_objects,
			object_types=object_types or {},
			type_parent_map=type_parent_map or {},
		)
		environment_ready_code = self._rewrite_primitive_wrappers_for_environment(runtime_ready_code)
		method_goal_ready_code = self._rewrite_method_primitive_actions_to_goals(
			environment_ready_code,
			action_schemas=action_schemas,
		)
		deferred_type_ready_code = self._defer_type_only_local_context_guards(
			method_goal_ready_code,
		)
		trace_ready_code = self._instrument_method_plans(
			deferred_type_ready_code,
			method_library,
			plan_library=plan_library,
		)
		goal_context = self._render_goal_fact_context(goal_facts)
		failure_repair_enabled = bool(query_goals) and self._failure_repair_enabled()
		goal_retry_enabled = bool(goal_context) or failure_repair_enabled
		lowered_code = self._ground_local_witness_method_plans(
			trace_ready_code,
			seed_facts=seed_facts,
			runtime_objects=runtime_objects,
			object_types=object_types or {},
			type_parent_map=type_parent_map or {},
			enable_blocked_goal_guards=failure_repair_enabled,
		)
		lines = [
			lowered_code.rstrip(),
			"",
			*self._render_failure_handlers(
				method_library,
				plan_library=plan_library,
				action_schemas=action_schemas,
				allow_repair=failure_repair_enabled,
				query_goal_count=len(query_goals),
			),
			"",
			"/* Execution Entry */",
			"!execute.",
			"",
			"+!execute : true <-",
		]
		execute_body = self._render_execute_body(
			query_goals=query_goals,
			goal_context=goal_context,
			repair_enabled=goal_retry_enabled,
		)
		lines.extend(
			self._indent_body(execute_body),
		)
		lines.append("")
		if goal_retry_enabled:
			if query_goals:
				lines.extend(
					self._render_retry_query_goal_plans(
						query_goals,
						goal_facts=goal_facts,
						method_library=method_library,
						action_schemas=action_schemas,
					),
				)
			lines.extend(
				self._render_goal_repair_plans(
					query_goals=query_goals,
					goal_context=goal_context,
					track_runtime_failures=failure_repair_enabled,
				),
			)
		lines.append("-!execute : true <-")
		lines.extend(self._indent_body(['.print("execute failed")', ".stopMAS"]))
		lines.append("")
		return "\n".join(lines)

	def _render_execute_body(
		self,
		*,
		query_goals: Sequence[Any],
		goal_context: str,
		repair_enabled: bool = False,
	) -> Tuple[str, ...]:
		if goal_context:
			return (
				'.print("execute start")',
				".perceive",
				"!finish_or_retry_0",
				".stopMAS",
			)
		if repair_enabled:
			query_execution = (
				("!runtime_execute_from_1",)
				if query_goals
				else self._render_retry_query_goal_calls(query_goals)
			)
			return (
				'.print("execute start")',
				".perceive",
				*query_execution,
				"!finish_or_retry_0",
				".stopMAS",
			)
		return (
			'.print("execute start")',
			".perceive",
			*self._render_query_goal_calls(query_goals),
			'.print("execute success")',
			".stopMAS",
		)

	def _render_goal_repair_plans(
		self,
		*,
		query_goals: Sequence[Any],
		goal_context: str,
		track_runtime_failures: bool = False,
	) -> List[str]:
		if not goal_context:
			lines: List[str] = []
			success_context = "not runtime_pass_failed"
			lines.append(f"+!finish_or_retry_0 : {success_context} <-")
			lines.extend(self._indent_body(['.print("execute success")']))
			lines.append("")
			lines.append("+!finish_or_retry_0 : true <-")
			lines.extend(self._indent_body(['.print("execute failed")', ".stopMAS"]))
			lines.append("")
			return lines

		pass_count = self._goal_repair_pass_count()
		lines: List[str] = []
		query_goal_calls: Tuple[str, ...]
		if track_runtime_failures and query_goals:
			query_goal_calls = ("!runtime_execute_from_1",)
		else:
			query_goal_calls = self._render_retry_query_goal_calls(query_goals)
		for pass_index in range(0, pass_count + 1):
			success_context = (
				f"{goal_context} & not runtime_pass_failed"
				if goal_context and track_runtime_failures
				else goal_context
				if goal_context
				else "not runtime_pass_failed"
			)
			lines.append(f"+!finish_or_retry_{pass_index} : {success_context} <-")
			lines.extend(self._indent_body(['.print("execute success")']))
			lines.append("")
			if pass_index >= pass_count:
				lines.append(f"+!finish_or_retry_{pass_index} : true <-")
				lines.extend(self._indent_body(['.print("execute failed")', ".stopMAS"]))
				lines.append("")
				continue
			next_pass = pass_index + 1
			lines.append(f"+!finish_or_retry_{pass_index} : true <-")
			pass_body: List[str] = [
				f'.print("runtime query pass ", {next_pass})',
			]
			if track_runtime_failures:
				pass_body.extend(
					[
						"if (runtime_pass_failed) { -runtime_pass_failed }",
						"!runtime_clear_local_repair_state",
						".abolish(blocked_runtime_choice(_))",
						".abolish(runtime_backtracked_choice(_, _))",
					],
				)
				if query_goals:
					pass_body.append("!runtime_clear_query_progress_from_1")
			pass_body.extend(query_goal_calls or ("true",))
			pass_body.append(f"!finish_or_retry_{next_pass}")
			lines.extend(self._indent_body(pass_body))
			lines.append("")
		return lines

	@staticmethod
	def _goal_repair_pass_count() -> int:
		raw_value = os.getenv("JASON_RUNTIME_GOAL_REPAIR_PASSES", "").strip()
		if not raw_value:
			return 3
		try:
			return max(1, int(raw_value))
		except ValueError:
			return 3

	def _render_retry_query_goal_plans(
		self,
		query_goals: Sequence[Any],
		*,
		goal_facts: Sequence[str] = (),
		method_library: HTNMethodLibrary | None = None,
		action_schemas: Sequence[Dict[str, Any]] = (),
	) -> List[str]:
		lines: List[str] = ["/* Runtime Query Goal Wrappers */"]
		completion_contexts = self._query_goal_completion_contexts(
			query_goals,
			goal_facts=goal_facts,
			method_library=method_library,
			action_schemas=action_schemas,
		)
		query_goal_calls = self._render_query_goal_calls(query_goals)
		for index, goal_call in enumerate(query_goal_calls, start=1):
			goal_name = f"runtime_query_goal_{index}"
			marker = f"runtime_query_goal_completed({index})"
			mark_goal = f"runtime_mark_query_goal_{index}"
			checkpoint = f"runtime_query_checkpoint({index})"
			context_alternatives = completion_contexts[index - 1] if index - 1 < len(completion_contexts) else ()
			if context_alternatives:
				for context in context_alternatives:
					lines.append(f"+!{goal_name} : {context} <-")
					lines.extend(self._indent_body(("true",)))
					lines.append("")
			else:
				lines.append(f"+!{goal_name} : {marker} <-")
				lines.extend(self._indent_body(("true",)))
				lines.append("")
			lines.append(f"+!{goal_name} : runtime_pass_failed <-")
			lines.extend(self._indent_body(("true",)))
			lines.append("")
			lines.append(f"+!{goal_name} : true <-")
			lines.extend(
				self._indent_body(
					(
						f"runtime_snapshot({checkpoint})",
						f"runtime_set_active_query_goal({index})",
						goal_call,
						f"!{mark_goal}",
					),
				),
			)
			lines.append("")
			lines.append(f"+!{mark_goal} : not runtime_pass_failed <-")
			lines.extend(
				self._indent_body(
					(
						f"runtime_clear_active_query_goal({index})",
						f"+{marker}",
					),
				),
			)
			lines.append("")
			lines.append(f"+!{mark_goal} : true <-")
			lines.extend(
				self._indent_body(
					(
						f"runtime_restore({checkpoint})",
						".perceive",
						f"runtime_clear_active_query_goal({index})",
						".fail",
					),
				),
			)
			lines.append("")
			lines.append(f"-!{goal_name} : true <-")
			lines.extend(
				self._indent_body(
					(
						f"runtime_restore({checkpoint})",
						".perceive",
						f"runtime_clear_active_query_goal({index})",
						".fail",
					),
				),
			)
			lines.append("")
		if query_goal_calls:
			lines.extend(self._render_runtime_query_driver_plans(len(query_goal_calls)))
			lines.extend(self._render_runtime_local_repair_cleanup_plans())
			lines.extend(self._render_runtime_query_cleanup_plans(len(query_goal_calls)))
		return lines

	def _render_runtime_query_driver_plans(self, query_goal_count: int) -> List[str]:
		if query_goal_count <= 0:
			return []
		lines: List[str] = ["/* Runtime Query Driver */"]
		lines.append(f"+!runtime_execute_from_{query_goal_count + 1} : true <-")
		lines.extend(self._indent_body(("true",)))
		lines.append("")
		for index in range(1, query_goal_count + 1):
			lines.append(f"+!runtime_execute_from_{index} : true <-")
			lines.extend(
				self._indent_body(
					(
						f"!runtime_query_goal_{index}",
						f"!runtime_execute_from_{index + 1}",
					),
				),
			)
			lines.append("")
			lines.extend(
				self._render_runtime_query_backtrack_plans(
					goal_name=f"runtime_backtrack_from_{index}",
					source_index=index,
				),
			)
			for target_index in range(index, 0, -1):
				lines.append(
					f"-!runtime_execute_from_{index} : "
					f"pipeline.choose_runtime_choice({index}, {target_index}, CHOICE) <-",
				)
				lines.extend(
					self._indent_body(
						self._runtime_query_backtrack_body(
							source_index=index,
							target_index=target_index,
						),
					),
				)
				lines.append("")
				for choice_predicate in self._runtime_query_choice_predicates():
					lines.append(
						f"-!runtime_execute_from_{index} : "
						f"{choice_predicate}({target_index}, CHOICE) "
						f"& not runtime_backtracked_choice({target_index}, CHOICE) <-",
					)
					lines.extend(
						self._indent_body(
							self._runtime_query_backtrack_body(
								source_index=index,
								target_index=target_index,
							),
						),
					)
					lines.append("")
			lines.append(f"-!runtime_execute_from_{index} : true <-")
			lines.extend(
				self._indent_body(
					(
						f'.print("runtime query branch exhausted ", {index})',
						"+runtime_pass_failed",
						".fail",
					),
				),
			)
			lines.append("")
		return lines

	def _render_runtime_query_backtrack_plans(
		self,
		*,
		goal_name: str,
		source_index: int,
	) -> List[str]:
		lines: List[str] = []
		for target_index in range(source_index, 0, -1):
			lines.append(
				f"+!{goal_name} : "
				f"pipeline.choose_runtime_choice({source_index}, {target_index}, CHOICE) <-",
			)
			lines.extend(
				self._indent_body(
					self._runtime_query_backtrack_body(
						source_index=source_index,
						target_index=target_index,
					),
				),
			)
			lines.append("")
			for choice_predicate in self._runtime_query_choice_predicates():
				lines.append(
					f"+!{goal_name} : {choice_predicate}({target_index}, CHOICE) "
					f"& not runtime_backtracked_choice({target_index}, CHOICE) <-",
				)
				lines.extend(
					self._indent_body(
						self._runtime_query_backtrack_body(
							source_index=source_index,
							target_index=target_index,
						),
					),
				)
				lines.append("")
		lines.append(f"+!{goal_name} : true <-")
		lines.extend(
			self._indent_body(
				(
					f'.print("runtime query branch exhausted ", {source_index})',
					"+runtime_pass_failed",
					".fail",
				),
			),
		)
		lines.append("")
		return lines

	def _runtime_query_backtrack_body(
		self,
		*,
		source_index: int,
		target_index: int,
	) -> Tuple[str, ...]:
		checkpoint = f"runtime_query_checkpoint({target_index})"
		return (
			f'.print("runtime query backtrack ", {source_index}, " -> ", {target_index}, " ", CHOICE)',
			f"runtime_restore({checkpoint})",
			".perceive",
			f"+runtime_backtracked_choice({target_index}, CHOICE)",
			"+blocked_runtime_choice(CHOICE)",
			"!runtime_clear_local_repair_state",
			f"!runtime_clear_query_progress_from_{target_index}",
			f"!runtime_execute_from_{target_index}",
			"!finish_or_retry_0",
			".stopMAS",
		)

	def _render_runtime_local_repair_cleanup_plans(self) -> List[str]:
		lines = ["/* Runtime Local Repair Cleanup */"]
		cleanup_statements = [
			".abolish(runtime_reported_failure(_))",
		]
		for predicate in (
			"blocked_runtime_method",
			"active_runtime_method",
			"runtime_current_method",
		):
			for arity in range(3, 12):
				args = ", ".join("_" for _index in range(arity))
				cleanup_statements.append(f".abolish({predicate}({args}))")
		cleanup_statements.append(".abolish(runtime_current_call(_, _, _, _, _, _, _))")
		cleanup_statements.append(".abolish(runtime_method_choice_point(_, _, _, _))")
		cleanup_statements.append(".abolish(runtime_latest_method_choice_point(_, _, _, _))")
		lines.append("+!runtime_clear_local_repair_state : true <-")
		lines.extend(self._indent_body(cleanup_statements))
		lines.append("")
		return lines

	def _render_runtime_query_cleanup_plans(self, query_goal_count: int) -> List[str]:
		lines: List[str] = ["/* Runtime Query Cleanup */"]
		for start_index in range(1, query_goal_count + 1):
			lines.append(f"+!runtime_clear_query_progress_from_{start_index} : true <-")
			lines.extend(
				self._indent_body(
					tuple(
						f"!runtime_clear_query_goal_{index}"
						for index in range(start_index, query_goal_count + 1)
					),
				),
			)
			lines.append("")
		for index in range(1, query_goal_count + 1):
			goal_name = f"runtime_clear_query_goal_{index}"
			lines.append(f"+!{goal_name} : true <-")
			lines.extend(
				self._indent_body(
					(
						f".abolish(runtime_query_goal_completed({index}))",
						f".abolish(runtime_active_query_goal({index}))",
						f".abolish(runtime_last_query_choice_frame({index}, _, _))",
						f".abolish(runtime_query_choice_frame({index}, _, _))",
						f".abolish(runtime_last_query_choice({index}, _))",
						f".abolish(runtime_query_choice({index}, _))",
					),
				),
			)
			lines.append("")
		return lines

	@staticmethod
	def _runtime_query_choice_predicates() -> Tuple[str, ...]:
		return (
			"runtime_last_query_choice",
			"runtime_query_choice",
		)

	def _query_goal_completion_contexts(
		self,
		query_goals: Sequence[Any],
		*,
		goal_facts: Sequence[str] = (),
		method_library: HTNMethodLibrary | None = None,
		action_schemas: Sequence[Dict[str, Any]] = (),
	) -> Tuple[Tuple[str, ...], ...]:
		query_specs = tuple(self._query_goal_specs(query_goals))
		goal_atoms = tuple(
			atom
			for atom in (self._hddl_fact_to_atom(fact) for fact in goal_facts)
			if atom
		)
		semantic_summaries = self._task_completion_summary_contexts(
			method_library=method_library,
			action_schemas=action_schemas,
		)
		contexts: List[Tuple[str, ...]] = []
		for task_name, args in query_specs:
			argument_set = {
				self._canonical_runtime_token(arg)
				for arg in args
				if self._canonical_runtime_token(arg)
			}
			matched_goal_atoms = tuple(
				atom
				for atom in goal_atoms
				if self._runtime_atom_args_are_subset(atom, argument_set)
			)
			if matched_goal_atoms:
				contexts.append((self._join_context_atoms(matched_goal_atoms),))
				continue

			summary_key = self._sanitize_name(task_name)
			summary_alternatives = semantic_summaries.get(summary_key, ())
			contexts.append(
				self._ground_query_summary_contexts(
					summary_alternatives,
					task_name=task_name,
					args=args,
					method_library=method_library,
				),
			)
		return tuple(contexts)

	def _task_completion_summary_contexts(
		self,
		*,
		method_library: HTNMethodLibrary | None,
		action_schemas: Sequence[Dict[str, Any]],
	) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
		if method_library is None:
			return {}

		action_lookup = self._action_schema_lookup(action_schemas)
		task_lookup = {
			self._sanitize_name(getattr(task, "name", "")): task
			for task in (
				*(getattr(method_library, "compound_tasks", ()) or ()),
				*(getattr(method_library, "primitive_tasks", ()) or ()),
			)
			if str(getattr(task, "name", "") or "").strip()
		}
		task_source_names = {
			self._sanitize_name(getattr(task, "name", "")): str(getattr(task, "name", "") or "").strip()
			for task in (
				*(getattr(method_library, "compound_tasks", ()) or ()),
				*(getattr(method_library, "primitive_tasks", ()) or ()),
			)
			if str(getattr(task, "name", "") or "").strip()
		}
		memo: Dict[str, Tuple[Tuple[str, ...], ...]] = {}

		def summarize(task_name: str, stack: Tuple[str, ...] = ()) -> Tuple[Tuple[str, ...], ...]:
			task_key = self._sanitize_name(task_name)
			if task_key in memo:
				return memo[task_key]
			if task_key in stack:
				return ()

			task = task_lookup.get(task_key)
			task_parameters = tuple(str(item) for item in (getattr(task, "parameters", ()) or ()))
			task_formals = {
				self._canonical_runtime_token(parameter)
				for parameter in task_parameters
				if self._canonical_runtime_token(parameter).startswith("?")
			}
			alternatives: List[Tuple[str, ...]] = []
			for method in method_library.methods_for_task(task_source_names.get(task_key, task_name)):
				bindings = self._method_task_bindings(method, task_parameters)
				body_atoms: List[str] = []
				for step in getattr(method, "subtasks", ()) or ():
					step_kind = str(getattr(getattr(step, "kind", ""), "value", getattr(step, "kind", "")) or "")
					if step_kind == "primitive":
						body_atoms.extend(
							self._primitive_step_summary_atoms(
								step,
								bindings=bindings,
								task_formals=task_formals,
								action_lookup=action_lookup,
							),
						)
						continue
					child_name = str(getattr(step, "task_name", "") or "").strip()
					if not child_name:
						continue
					child_task = task_lookup.get(self._sanitize_name(child_name))
					child_parameters = tuple(
						str(item) for item in (getattr(child_task, "parameters", ()) or ())
					)
					child_bindings = {
						self._canonical_runtime_token(parameter): self._resolve_summary_token(arg, bindings)
						for parameter, arg in zip(child_parameters, getattr(step, "args", ()) or ())
					}
					for child_atoms in summarize(child_name, (*stack, task_key)):
						body_atoms.extend(
							self._ground_summary_atoms(
								child_atoms,
								bindings=child_bindings,
								task_formals=task_formals,
							),
						)
				selected = self._select_task_completion_atoms(body_atoms, task_formals)
				if not selected:
					selected = self._select_task_completion_atoms(
						self._method_context_summary_atoms(
							method,
							bindings=bindings,
							task_formals=task_formals,
						),
						task_formals,
					)
				if selected:
					alternatives.append(selected)
			deduped = self._dedupe_context_alternatives(alternatives)
			memo[task_key] = deduped
			return deduped

		for task_name in tuple(task_lookup):
			summarize(task_name)
		return {task_name: alternatives for task_name, alternatives in memo.items() if alternatives}

	def _ground_query_summary_contexts(
		self,
		alternatives: Sequence[Sequence[str]],
		*,
		task_name: str,
		args: Sequence[str],
		method_library: HTNMethodLibrary | None,
	) -> Tuple[str, ...]:
		if method_library is None:
			return ()
		task_parameters = self._task_parameters_for_name(method_library, task_name)
		if not task_parameters:
			return ()
		bindings = {
			self._canonical_runtime_token(parameter): self._canonical_runtime_token(arg)
			for parameter, arg in zip(task_parameters, args)
		}
		contexts: List[str] = []
		for atoms in alternatives:
			grounded = self._ground_summary_atoms(
				atoms,
				bindings=bindings,
				task_formals=set(),
				render_runtime_terms=True,
			)
			if grounded:
				contexts.append(self._join_context_atoms(grounded))
		return tuple(dict.fromkeys(contexts))

	def _task_parameters_for_name(
		self,
		method_library: HTNMethodLibrary,
		task_name: str,
	) -> Tuple[str, ...]:
		task = method_library.task_for_name(task_name)
		if task is None:
			sanitized = self._sanitize_name(task_name)
			for candidate in (
				*(getattr(method_library, "compound_tasks", ()) or ()),
				*(getattr(method_library, "primitive_tasks", ()) or ()),
			):
				if self._sanitize_name(getattr(candidate, "name", "")) == sanitized:
					task = candidate
					break
		return tuple(str(item) for item in (getattr(task, "parameters", ()) or ()))

	def _method_task_bindings(
		self,
		method: Any,
		task_parameters: Sequence[str],
	) -> Dict[str, str]:
		bindings: Dict[str, str] = {}
		for task_arg, task_parameter in zip(getattr(method, "task_args", ()) or (), task_parameters):
			arg_token = self._canonical_runtime_token(str(task_arg))
			parameter_token = self._canonical_runtime_token(str(task_parameter))
			if arg_token:
				bindings[arg_token] = parameter_token
				if arg_token.startswith("?"):
					bindings[arg_token[1:]] = parameter_token
		return bindings

	def _primitive_step_summary_atoms(
		self,
		step: Any,
		*,
		bindings: Dict[str, str],
		task_formals: Set[str],
		action_lookup: Dict[str, Dict[str, Any]],
	) -> Tuple[str, ...]:
		action_name = str(getattr(step, "action_name", None) or getattr(step, "task_name", "") or "")
		schema = action_lookup.get(action_name) or action_lookup.get(self._sanitize_name(action_name))
		if schema is None:
			return ()
		step_bindings = dict(bindings)
		for parameter, arg in zip(schema.get("parameters") or (), getattr(step, "args", ()) or ()):
			resolved = self._resolve_summary_token(str(arg), bindings)
			parameter_token = self._canonical_runtime_token(str(parameter))
			step_bindings[parameter_token] = resolved
			if parameter_token.startswith("?"):
				step_bindings[parameter_token[1:]] = resolved

		effect_atoms = self._summary_atoms_from_patterns(
			schema.get("effects") or (),
			bindings=step_bindings,
			task_formals=task_formals,
		)
		if effect_atoms:
			return effect_atoms
		if schema.get("effects"):
			return ()
		return self._summary_atoms_from_patterns(
			schema.get("preconditions") or (),
			bindings=step_bindings,
			task_formals=task_formals,
		)

	def _method_context_summary_atoms(
		self,
		method: Any,
		*,
		bindings: Dict[str, str],
		task_formals: Set[str],
	) -> Tuple[str, ...]:
		patterns = (
			{
				"predicate": getattr(literal, "predicate", ""),
				"args": tuple(getattr(literal, "args", ()) or ()),
				"is_positive": getattr(literal, "is_positive", True),
			}
			for literal in (getattr(method, "context", ()) or ())
		)
		return self._summary_atoms_from_patterns(
			tuple(patterns),
			bindings=bindings,
			task_formals=task_formals,
		)

	def _summary_atoms_from_patterns(
		self,
		patterns: Sequence[Dict[str, Any]],
		*,
		bindings: Dict[str, str],
		task_formals: Set[str],
	) -> Tuple[str, ...]:
		atoms: List[str] = []
		for pattern in patterns:
			if not bool(pattern.get("is_positive", True)):
				continue
			predicate = str(pattern.get("predicate", "") or "").strip()
			if not predicate or predicate == "=":
				continue
			resolved_args = tuple(
				self._resolve_summary_token(str(arg), bindings)
				for arg in (pattern.get("args") or ())
			)
			if not resolved_args:
				continue
			if any(self._is_unbound_summary_variable(arg, task_formals) for arg in resolved_args):
				continue
			atoms.append(self._call(predicate, resolved_args))
		return tuple(atoms)

	def _ground_summary_atoms(
		self,
		atoms: Sequence[str],
		*,
		bindings: Dict[str, str],
		task_formals: Set[str],
		render_runtime_terms: bool = False,
	) -> Tuple[str, ...]:
		grounded: List[str] = []
		for atom in atoms:
			parsed = self._parse_runtime_atom(atom)
			if parsed is None:
				continue
			predicate, args = parsed
			resolved_args = tuple(self._resolve_summary_token(arg, bindings) for arg in args)
			if any(self._is_unbound_summary_variable(arg, task_formals) for arg in resolved_args):
				continue
			rendered_args = (
				tuple(self._runtime_atom_term(arg) for arg in resolved_args)
				if render_runtime_terms
				else resolved_args
			)
			grounded.append(self._call(predicate, rendered_args))
		return tuple(grounded)

	def _select_task_completion_atoms(
		self,
		atoms: Sequence[str],
		task_formals: Set[str],
	) -> Tuple[str, ...]:
		scored: List[Tuple[int, str]] = []
		for atom in atoms:
			parsed = self._parse_runtime_atom(atom)
			if parsed is None:
				continue
			_, args = parsed
			coverage = len({
				self._canonical_runtime_token(arg)
				for arg in args
				if self._canonical_runtime_token(arg) in task_formals
			})
			if coverage <= 0:
				continue
			scored.append((coverage, atom))
		if not scored:
			return ()
		best_coverage = max(score for score, _atom in scored)
		return tuple(
			atom
			for _score, atom in self._dedupe_scored_atoms(
				(score, atom) for score, atom in scored if score == best_coverage
			)
		)

	@staticmethod
	def _dedupe_scored_atoms(scored_atoms: Iterable[Tuple[int, str]]) -> Tuple[Tuple[int, str], ...]:
		seen: Set[str] = set()
		deduped: List[Tuple[int, str]] = []
		for score, atom in scored_atoms:
			if atom in seen:
				continue
			seen.add(atom)
			deduped.append((score, atom))
		return tuple(deduped)

	def _dedupe_context_alternatives(
		self,
		alternatives: Sequence[Sequence[str]],
	) -> Tuple[Tuple[str, ...], ...]:
		seen: Set[Tuple[str, ...]] = set()
		deduped: List[Tuple[str, ...]] = []
		for atoms in alternatives:
			normalised = tuple(dict.fromkeys(atom for atom in atoms if atom))
			if not normalised or normalised in seen:
				continue
			seen.add(normalised)
			deduped.append(normalised)
		return tuple(deduped)

	@classmethod
	def _query_goal_specs(cls, query_goals: Sequence[Any]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
		specs: List[Tuple[str, Tuple[str, ...]]] = []
		for goal in query_goals:
			if isinstance(goal, dict):
				task_name = str(goal.get("task_name") or "").strip()
				args = tuple(
					str(arg).strip()
					for arg in (goal.get("args") or ())
					if str(arg).strip()
				)
			else:
				task_name = str(getattr(goal, "task_name", "") or "").strip()
				args = tuple(
					str(arg).strip()
					for arg in (getattr(goal, "args", ()) or ())
					if str(arg).strip()
				)
			if task_name:
				specs.append((task_name, args))
		return tuple(specs)

	def _runtime_atom_args_are_subset(self, atom: str, argument_set: Set[str]) -> bool:
		parsed = self._parse_runtime_atom(atom)
		if parsed is None:
			return False
		_, args = parsed
		if not args:
			return False
		return all(self._canonical_runtime_token(arg) in argument_set for arg in args)

	@staticmethod
	def _join_context_atoms(atoms: Sequence[str]) -> str:
		return " & ".join(dict.fromkeys(atom for atom in atoms if atom))

	def _resolve_summary_token(self, token: str, bindings: Dict[str, str]) -> str:
		canonical = self._canonical_runtime_token(token)
		if canonical in bindings:
			return bindings[canonical]
		if canonical.startswith("?") and canonical[1:] in bindings:
			return bindings[canonical[1:]]
		return canonical

	@staticmethod
	def _is_unbound_summary_variable(token: str, task_formals: Set[str]) -> bool:
		canonical = JasonRunner._canonical_runtime_token(token)
		return canonical.startswith("?") and canonical not in task_formals

	def _parse_runtime_atom(self, atom: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
		text = str(atom or "").strip()
		if not text:
			return None
		if "(" not in text:
			return self._sanitize_name(text), ()
		if not text.endswith(")"):
			return None
		predicate, args_text = text.split("(", 1)
		predicate = self._sanitize_name(predicate)
		args = self._split_asl_arguments(args_text[:-1])
		return predicate, tuple(self._canonical_runtime_token(arg) for arg in args)

	def _render_retry_query_goal_calls(self, query_goals: Sequence[Any]) -> Tuple[str, ...]:
		return tuple(
			f"!runtime_query_goal_{index}"
			for index, _goal in enumerate(self._render_query_goal_calls(query_goals), start=1)
		)

	@staticmethod
	def _failure_repair_enabled() -> bool:
		raw_value = os.getenv("JASON_RUNTIME_FAILURE_REPAIR", "").strip().lower()
		if not raw_value:
			return True
		if raw_value in {"0", "false", "no", "off"}:
			return False
		return raw_value in {"1", "true", "yes", "on"}

	def _render_goal_fact_context(self, goal_facts: Sequence[str]) -> str:
		goal_atoms: List[str] = []
		seen: set[str] = set()
		for fact in goal_facts:
			atom = self._hddl_fact_to_atom(fact)
			if not atom or atom in seen:
				continue
			seen.add(atom)
			goal_atoms.append(atom)
		return " & ".join(goal_atoms)

	def _render_chained_goal_plans(
		self,
		goal_prefix: str,
		statements: Sequence[str],
		final_statements: Sequence[str],
		*,
		chunk_size: int = 128,
	) -> List[str]:
		if chunk_size < 1:
			raise ValueError("chunk_size must be positive")

		chunks = [
			list(statements[index:index + chunk_size])
			for index in range(0, len(statements), chunk_size)
		] or [[]]
		lines: List[str] = []
		for index, chunk in enumerate(chunks, start=1):
			goal_name = f"{goal_prefix}_{index}"
			is_last = index == len(chunks)
			body_lines = list(chunk)
			if is_last:
				body_lines.extend(final_statements)
			else:
				body_lines.append(f"!{goal_prefix}_{index + 1}")
			lines.append(f"+!{goal_name} : true <-")
			lines.extend(self._indent_body(body_lines))
			lines.append("")
		return lines

	def _inject_runtime_object_beliefs(
		self,
		agentspeak_code: str,
		*,
		seed_facts: Sequence[str],
		runtime_objects: Sequence[str],
		object_types: Dict[str, str],
		type_parent_map: Dict[str, Optional[str]],
	) -> str:
		_ = seed_facts
		if not runtime_objects:
			return agentspeak_code

		start_marker = "/* Initial Beliefs */"
		end_marker = "/* Primitive Action Plans */"
		start_index = agentspeak_code.find(start_marker)
		end_index = agentspeak_code.find(end_marker)
		if start_index == -1 or end_index == -1 or end_index <= start_index:
			return agentspeak_code

		prefix = agentspeak_code[:start_index]
		section = agentspeak_code[start_index:end_index]
		suffix = agentspeak_code[end_index:]
		section_lines = section.splitlines()
		if not section_lines:
			return agentspeak_code

		header = section_lines[0]
		body_lines = [line for line in section_lines[1:] if line.strip()]
		existing = {line.strip() for line in body_lines}
		inserted: List[str] = []

		for obj in runtime_objects:
			object_line = f"{self._call('object', (self._runtime_atom_term(obj),))}."
			if object_line not in existing:
				existing.add(object_line)
				inserted.append(object_line)
			for type_name in self._type_closure(object_types.get(str(obj)), type_parent_map):
				type_line = (
					f"{self._call('object_type', (self._runtime_atom_term(obj), self._type_atom(type_name)))}."
				)
				if type_line in existing:
					continue
				existing.add(type_line)
				inserted.append(type_line)

		if not inserted:
			return agentspeak_code

		injected_section = "\n".join([header, *inserted, *body_lines]).rstrip() + "\n\n"
		return f"{prefix}{injected_section}{suffix}"

	def _render_query_goal_calls(self, query_goals: Sequence[Any]) -> Tuple[str, ...]:
		goal_calls: List[str] = []
		for goal in query_goals:
			if isinstance(goal, dict):
				task_name = str(goal.get("task_name") or "").strip()
				args = tuple(
					str(arg).strip()
					for arg in (goal.get("args") or ())
					if str(arg).strip()
				)
			else:
				task_name = str(getattr(goal, "task_name", "") or "").strip()
				args = tuple(
					str(arg).strip()
					for arg in (getattr(goal, "args", ()) or ())
					if str(arg).strip()
				)
			if not task_name:
				continue
			goal_calls.append(f"!{self._runtime_call(task_name, args)}")
		return tuple(goal_calls)

	def _extract_action_path(self, stdout: str) -> List[str]:
		action_pattern = re.compile(r"^runtime env action success (.+?)\s*$")
		snapshot_pattern = re.compile(r"^runtime env snapshot (.+?)\s*$")
		restore_pattern = re.compile(r"^runtime env restore (.+?)\s*$")
		commit_pattern = re.compile(r"^runtime env commit (.+?)\s*$")
		actions: List[str] = []
		snapshots: List[Tuple[str, int]] = []
		for raw_line in stdout.splitlines():
			line = raw_line.strip()
			if match := snapshot_pattern.match(line):
				snapshots.append((match.group(1).strip(), len(actions)))
				continue
			if match := restore_pattern.match(line):
				key = match.group(1).strip()
				for index in range(len(snapshots) - 1, -1, -1):
					snapshot_key, action_count = snapshots[index]
					if snapshot_key != key:
						continue
					del actions[action_count:]
					del snapshots[index:]
					break
				continue
			if match := commit_pattern.match(line):
				key = match.group(1).strip()
				for index in range(len(snapshots) - 1, -1, -1):
					if snapshots[index][0] == key:
						del snapshots[index]
						break
				continue
			if match := action_pattern.match(line):
				actions.append(match.group(1).strip())
		return actions

	@staticmethod
	def _extract_goal_repair_pass_count(stdout: str) -> int:
		passes = [
			int(match.group(1))
			for match in re.finditer(r"runtime query pass\s+([0-9]+)", str(stdout or ""))
		]
		return max(passes) if passes else 0

	def _extract_method_trace(self, stdout: str) -> List[Dict[str, Any]]:
		flat_pattern = re.compile(r"runtime trace method flat\s+(.+?)\s*$")
		legacy_pattern = re.compile(r"runtime trace method\s+trace_method\((.*)\)\s*$")
		trace: List[Dict[str, Any]] = []
		for raw_line in stdout.splitlines():
			line = raw_line.strip()
			flat_match = flat_pattern.search(line)
			if flat_match is not None:
				payload = flat_match.group(1).strip()
				parts = [part.strip() for part in payload.split("|")]
				if not parts or not parts[0]:
					continue
				trace.append(
					{
						"method_name": parts[0],
						"task_args": [
							part
							for part in parts[1:]
							if part
						],
					},
				)
				continue

			legacy_match = legacy_pattern.search(line)
			if legacy_match is None:
				continue
			payload = legacy_match.group(1).strip()
			if not payload:
				continue
			parts = [part.strip() for part in payload.split(",")]
			if not parts or not parts[0]:
				continue
			trace.append(
				{
					"method_name": self._strip_quoted_atom(parts[0]),
					"task_args": [
						self._strip_quoted_atom(part)
						for part in parts[1:]
						if part
					],
				},
			)
		return trace

	def _cap_method_trace_records(
		self,
		method_trace: Sequence[Dict[str, Any]],
	) -> Tuple[List[Dict[str, Any]], int, bool]:
		records = [dict(item) for item in (method_trace or ())]
		original_count = len(records)
		limit = max(1, int(self.method_trace_record_limit))
		if original_count <= limit:
			return records, original_count, False
		return records[:limit], original_count, True

	def _bounded_runtime_output_artifact(self, text: str) -> Tuple[str, bool]:
		value = str(text or "")
		limit = max(1, int(self.runtime_output_artifact_limit_chars))
		if len(value) <= limit:
			return value, False
		prefix = (
			f"[truncated runtime output: original_chars={len(value)}, "
			f"kept_tail_chars={limit}, "
			f"sha256={hashlib.sha256(value.encode('utf-8')).hexdigest()}]\n"
		)
		return f"{prefix}{value[-limit:]}", True

	def _extract_panda_method_trace(self, plan_text: str) -> List[Dict[str, Any]]:
		lines = [
			line.strip()
			for line in str(plan_text or "").splitlines()
			if line.strip() and line.strip() != "==>"
		]
		if not lines:
			return []

		method_nodes: Dict[int, Dict[str, Any]] = {}
		primitive_node_ids: set[int] = set()
		root_ids: List[int] = []
		for line in lines:
			if line.startswith("root "):
				root_ids.extend(self._parse_plan_node_ids(line.split()[1:]))
				continue
			parts = line.split()
			if not parts:
				continue
			try:
				node_id = int(parts[0])
			except ValueError:
				continue
			if "->" not in parts:
				primitive_node_ids.add(node_id)
				continue
			arrow_index = parts.index("->")
			if arrow_index < 2 or arrow_index + 1 >= len(parts):
				continue
			method_nodes[node_id] = {
				"method_name": parts[arrow_index + 1],
				"task_args": parts[2:arrow_index],
				"children": self._parse_plan_node_ids(parts[arrow_index + 2:]),
			}

		trace: List[Dict[str, Any]] = []
		visited: set[int] = set()
		first_primitive_cache: Dict[int, int] = {}

		def first_primitive_id(node_id: int) -> int:
			if node_id in first_primitive_cache:
				return first_primitive_cache[node_id]
			if node_id in primitive_node_ids:
				first_primitive_cache[node_id] = node_id
				return node_id
			node = method_nodes.get(node_id)
			if node is None:
				first_primitive_cache[node_id] = node_id
				return node_id
			child_order = [first_primitive_id(child_id) for child_id in node["children"]]
			first_primitive_cache[node_id] = min(child_order) if child_order else node_id
			return first_primitive_cache[node_id]

		def visit(node_id: int) -> None:
			if node_id in visited:
				return
			node = method_nodes.get(node_id)
			if node is None:
				return
			visited.add(node_id)
			trace.append(
				{
					"method_name": node["method_name"],
					"task_args": list(node["task_args"]),
				},
			)
			for child_id in sorted(
				node["children"],
				key=lambda child_id: (first_primitive_id(child_id), child_id),
			):
				visit(child_id)

		for node_id in sorted(root_ids, key=lambda node_id: (first_primitive_id(node_id), node_id)):
			visit(node_id)
		for node_id in sorted(method_nodes, key=lambda node_id: (first_primitive_id(node_id), node_id)):
			visit(node_id)
		return trace

	@staticmethod
	def _parse_plan_node_ids(tokens: Sequence[str]) -> List[int]:
		node_ids: List[int] = []
		for token in tokens:
			try:
				node_ids.append(int(str(token)))
			except ValueError:
				continue
		return node_ids

	def _augment_method_trace_with_query_root_bridges(
		self,
		*,
		method_trace: Sequence[Dict[str, Any]],
		method_library: HTNMethodLibrary | None,
		problem_file: str | Path | None,
	) -> List[Dict[str, Any]]:
		"""Insert missing official-root bridge entries for transition-native traces."""
		trace = [dict(item) for item in (method_trace or ())]
		if not trace or method_library is None or problem_file is None:
			return trace

		bridge_specs = self._query_root_bridge_trace_specs(
			method_library=method_library,
			problem_file=problem_file,
		)
		if not bridge_specs:
			return trace

		method_by_name = {
			str(method.method_name).strip(): method
			for method in method_library.methods
			if str(method.method_name).strip()
		}
		bridge_task_names = {
			str(spec.get("bridge_task_name", "")).strip()
			for spec in bridge_specs
			if str(spec.get("bridge_task_name", "")).strip()
		}
		if any(
			str(getattr(method_by_name.get(str(entry.get("method_name", "")).strip()), "task_name", "")).strip()
			in bridge_task_names
			for entry in trace
		):
			return trace
		remaining_specs = list(bridge_specs)
		augmented_trace: List[Dict[str, Any]] = []
		changed = False

		for entry in trace:
			method_name = str(entry.get("method_name", "")).strip()
			method = method_by_name.get(method_name)
			if method is None:
				augmented_trace.append(entry)
				continue

			existing_bridge_index = self._first_bridge_spec_index(
				remaining_specs,
				key="bridge_method_name",
				value=method_name,
			)
			if existing_bridge_index is not None:
				remaining_specs.pop(existing_bridge_index)
				augmented_trace.append(entry)
				continue

			child_bridge_index = self._first_bridge_spec_index(
				remaining_specs,
				key="child_task_name",
				value=str(method.task_name).strip(),
			)
			if child_bridge_index is not None:
				spec = remaining_specs.pop(child_bridge_index)
				augmented_trace.append({
					"method_name": spec["bridge_method_name"],
					"task_args": list(spec["root_task_args"]),
				})
				changed = True
			augmented_trace.append(entry)

		return augmented_trace if changed else trace

	@staticmethod
	def _first_bridge_spec_index(
		bridge_specs: Sequence[Dict[str, Any]],
		*,
		key: str,
		value: str,
	) -> Optional[int]:
		for index, spec in enumerate(bridge_specs):
			if str(spec.get(key, "")).strip() == value:
				return index
		return None

	def _query_root_bridge_trace_specs(
		self,
		*,
		method_library: HTNMethodLibrary,
		problem_file: str | Path,
	) -> List[Dict[str, Any]]:
		try:
			from utils.hddl_parser import HDDLParser

			problem = HDDLParser.parse_problem(str(problem_file))
		except Exception:
			return []

		root_tasks = list(getattr(problem, "htn_tasks", ()) or ())
		if not root_tasks:
			return []

		bridge_methods_by_task: Dict[str, List[Any]] = {}
		for method in method_library.methods:
			ordered_subtasks = tuple(getattr(method, "subtasks", ()) or ())
			if len(ordered_subtasks) != 1:
				continue
			child = ordered_subtasks[0]
			if getattr(child, "kind", "") != "compound":
				continue
			bridge_methods_by_task.setdefault(str(method.task_name).strip(), []).append(method)

		bridge_tasks_by_source: Dict[str, List[str]] = {}
		for task in method_library.compound_tasks:
			source_name = str(getattr(task, "source_name", "") or "").strip()
			task_name = str(getattr(task, "name", "") or "").strip()
			if not source_name or not task_name or source_name == task_name:
				continue
			if task_name not in bridge_methods_by_task:
				continue
			bridge_tasks_by_source.setdefault(source_name, []).append(task_name)

		occurrence_counts: Dict[str, int] = {}
		bridge_specs: List[Dict[str, Any]] = []
		for root_index, root_task in enumerate(root_tasks, start=1):
			source_name = str(getattr(root_task, "task_name", "") or "").strip()
			if not source_name:
				continue
			candidates = bridge_tasks_by_source.get(source_name) or ()
			expected_bridge_task_name = query_root_alias_task_name(root_index, source_name)
			if expected_bridge_task_name in candidates:
				bridge_task_name = expected_bridge_task_name
				occurrence_counts[source_name] = occurrence_counts.get(source_name, 0) + 1
				for bridge_method in bridge_methods_by_task.get(bridge_task_name, ()):
					child = tuple(getattr(bridge_method, "subtasks", ()) or ())[0]
					bridge_specs.append({
						"bridge_task_name": bridge_task_name,
						"bridge_method_name": str(bridge_method.method_name).strip(),
						"child_task_name": str(getattr(child, "task_name", "")).strip(),
						"root_task_args": [
							str(arg).strip()
							for arg in (getattr(root_task, "args", ()) or ())
						],
					})
				continue
			occurrence_index = occurrence_counts.get(source_name, 0)
			occurrence_counts[source_name] = occurrence_index + 1
			if occurrence_index >= len(candidates):
				continue
			bridge_task_name = candidates[occurrence_index]
			for bridge_method in bridge_methods_by_task.get(bridge_task_name, ()):
				child = tuple(getattr(bridge_method, "subtasks", ()) or ())[0]
				bridge_specs.append({
					"bridge_task_name": bridge_task_name,
					"bridge_method_name": str(bridge_method.method_name).strip(),
					"child_task_name": str(getattr(child, "task_name", "")).strip(),
					"root_task_args": [
						str(arg).strip()
						for arg in (getattr(root_task, "args", ()) or ())
					],
				})
		return bridge_specs

	def _extract_failed_goals(self, stdout: str) -> List[str]:
		pattern = re.compile(r"runtime goal failed\s+fail_goal\((.*)\)\s*$")
		failed: List[str] = []
		seen: Set[str] = set()
		truncated_count = 0
		for raw_line in stdout.splitlines():
			line = raw_line.strip()
			match = pattern.search(line)
			if match is None:
				continue
			payload = match.group(1).strip()
			if not payload or payload in seen:
				continue
			seen.add(payload)
			if len(failed) < self.failed_goal_record_limit:
				failed.append(payload)
			else:
				truncated_count += 1
		if truncated_count:
			failed.append(f"... truncated {truncated_count} additional failed goals")
		return failed

	def _render_action_path(self, action_path: Sequence[str]) -> str:
		if not action_path:
			return ""
		return "\n".join(action_path) + "\n"

	@staticmethod
	def _call(name: str, args: Sequence[str] = ()) -> str:
		functor = JasonRunner._sanitize_name(name)
		if not args:
			return functor
		return f"{functor}({', '.join(args)})"

	@classmethod
	def _runtime_call(cls, name: str, args: Sequence[str] = ()) -> str:
		functor = cls._sanitize_name(name)
		if not args:
			return functor
		rendered_args = [cls._runtime_atom_term(arg) for arg in args]
		return f"{functor}({', '.join(rendered_args)})"

	@staticmethod
	def _type_atom(type_name: str) -> str:
		return JasonRunner._sanitize_name(str(type_name or "object")).lower() or "object"

	@staticmethod
	def _sanitize_name(name: str) -> str:
		return re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip()).strip("_") or "term"

	@staticmethod
	def _asl_string(text: str) -> str:
		return json.dumps(str(text))

	@classmethod
	def _asl_atom_or_string(cls, text: str) -> str:
		token = str(text).strip()
		if re.fullmatch(r"[a-z][a-z0-9_]*", token):
			return token
		return cls._asl_string(token)

	@classmethod
	def _runtime_atom_term(cls, text: str) -> str:
		token = str(text).strip()
		if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
			return token
		return cls._asl_atom_or_string(token)

	@classmethod
	def _type_closure(
		cls,
		type_name: Optional[str],
		type_parent_map: Dict[str, Optional[str]],
	) -> Tuple[str, ...]:
		if not type_name:
			return ()

		closure: List[str] = []
		visited: set[str] = set()
		cursor: Optional[str] = str(type_name).strip()
		while cursor and cursor not in visited:
			visited.add(cursor)
			if cursor != "object":
				closure.append(cursor)
			cursor = type_parent_map.get(cursor)
		return tuple(closure)

	@staticmethod
	def _split_plan_head(head_line: str) -> Optional[Tuple[str, str, str]]:
		match = re.match(r"^(\s*\+![^\s(:]+(?:\([^)]*\))?\s*:\s*)(.*?)(\s*<-\s*)$", head_line)
		if match is None:
			return None
		return match.group(1), match.group(2).strip(), match.group(3)

	@staticmethod
	def _split_method_trigger_head(
		head_line: str,
	) -> Optional[Tuple[str, str, Tuple[str, ...], str, str]]:
		match = re.match(
			r"^(\s*\+!([^\s(:]+)(?:\(([^)]*)\))?\s*:\s*)(.*?)(\s*<-\s*)$",
			head_line,
		)
		if match is None:
			return None
		args_text = (match.group(3) or "").strip()
		args = tuple(part.strip() for part in args_text.split(",") if part.strip())
		return match.group(1), match.group(2).strip(), args, match.group(4).strip(), match.group(5)

	@staticmethod
	def _parse_asl_goal_statement(line: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
		statement = str(line).strip().rstrip(";.")
		match = re.match(r"^!([^\s(;]+)(?:\(([^)]*)\))?$", statement)
		if match is None:
			return None
		args_text = (match.group(2) or "").strip()
		args = tuple(part.strip() for part in args_text.split(",") if part.strip())
		return match.group(1).strip(), args

	@staticmethod
	def _combine_contexts(*contexts: str) -> str:
		parts: List[str] = []
		seen: set[str] = set()
		for context in contexts:
			for part in str(context or "").split(" & "):
				cleaned = part.strip()
				if not cleaned or cleaned == "true" or cleaned in seen:
					continue
				seen.add(cleaned)
				parts.append(cleaned)
		return " & ".join(parts) if parts else "true"

	@staticmethod
	def _strip_quoted_atom(text: str) -> str:
		token = str(text).strip()
		if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
			return token[1:-1]
		return token

	@staticmethod
	def _failure_handler_args(
		parameters: Sequence[str],
		*,
		prefix: str = "",
	) -> Tuple[str, ...]:
		used_counts: Dict[str, int] = {}
		rendered_args: List[str] = []
		for index, parameter in enumerate(parameters, start=1):
			token = re.sub(r"^[?]+", "", str(parameter).strip())
			token = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").upper()
			if not token:
				token = f"ARG{index}"
			prefix_token = re.sub(r"[^A-Za-z0-9]+", "_", str(prefix).strip()).strip("_").upper()
			if prefix_token:
				token = f"{prefix_token}_{token}"
			if not token[0].isalpha():
				token = f"ARG_{token}"
			count = used_counts.get(token, 0) + 1
			used_counts[token] = count
			rendered_args.append(token if count == 1 else f"{token}{count}")
		return tuple(rendered_args)

	@staticmethod
	def _strip_type_annotation(parameter: str) -> str:
		text = str(parameter or "").strip()
		if ":" in text:
			return text.split(":", 1)[0].strip()
		return text

	def _render_failure_handlers(
		self,
		method_library: HTNMethodLibrary | None,
		*,
		plan_library: PlanLibrary | None = None,
		action_schemas: Sequence[Dict[str, Any]] = (),
		allow_repair: bool = False,
		query_goal_count: int = 0,
	) -> List[str]:
		lines = ["/* Failure Handlers */"]
		seen_triggers: set[str] = set()
		for task_name, parameters in self._failure_handler_signatures(
			method_library=method_library,
			plan_library=plan_library,
			action_schemas=action_schemas,
		):
			handler_args = self._failure_handler_args(parameters)
			trigger = self._call(self._sanitize_name(task_name), handler_args)
			if trigger in seen_triggers:
				continue
			seen_triggers.add(trigger)
			fail_term = self._call("fail_goal", (self._asl_atom_or_string(task_name), *handler_args))
			active_method = self._call(
				"active_runtime_method",
				(
					"METHOD",
					self._asl_atom_or_string(self._sanitize_name(task_name)),
					*handler_args,
					"BINDING",
				),
			)
			method_snapshot = self._call(
				"runtime_method_snapshot",
				(
					"METHOD",
					self._asl_atom_or_string(self._sanitize_name(task_name)),
					*handler_args,
					"BINDING",
				),
			)
			blocked_method = self._call(
				"blocked_runtime_method",
				(
					"METHOD",
					self._asl_atom_or_string(self._sanitize_name(task_name)),
					*handler_args,
					"BINDING",
				),
			)
			current_method = self._call(
				"runtime_current_method",
				(
					"METHOD",
					self._asl_atom_or_string(self._sanitize_name(task_name)),
					*handler_args,
					"BINDING",
				),
			)
			active_handler_context = (
				f"{active_method} & not runtime_pass_failed"
				if allow_repair
				else active_method
			)
			choice_point = self._call(
				"runtime_latest_method_choice_point",
				(
					"CHOICE",
					self._asl_atom_or_string(self._sanitize_name(task_name)),
					self._call("runtime_args", handler_args),
					"SNAPSHOT",
				),
			)
			if allow_repair:
				lines.append(f"-!{trigger} : {choice_point} & not runtime_pass_failed <-")
				lines.extend(
					self._indent_body(
						[
							f'.print("runtime method choice failed ", {fail_term})',
							"runtime_restore(SNAPSHOT)",
							".perceive",
							"+blocked_runtime_choice(CHOICE)",
							"runtime_pop_method_choice(CHOICE)",
							f"!{trigger}",
						],
					),
				)
				lines.append("")
			active_handler_body = [
				f'.print("runtime goal branch failed ", {fail_term})',
				f"runtime_restore({method_snapshot})",
				".perceive",
				f"runtime_commit({method_snapshot})",
				f"+{blocked_method}",
				f"-{active_method}",
			]
			if allow_repair:
				active_handler_body.append(f"-{current_method}")
				active_handler_body.append(f"!{trigger}")
			else:
				active_handler_body.append(".fail")
			lines.append(f"-!{trigger} : {active_handler_context} <-")
			lines.extend(self._indent_body(active_handler_body))
			lines.append("")
			if allow_repair:
				for parent_task_name, parent_parameters in self._runtime_caller_signatures(
					failed_task_name=task_name,
					method_library=method_library,
					plan_library=plan_library,
				):
					parent_args = self._failure_handler_args(
						parent_parameters,
						prefix="PARENT",
					)
					parent_trigger = self._call(self._sanitize_name(parent_task_name), parent_args)
					parent_task_atom = self._asl_atom_or_string(self._sanitize_name(parent_task_name))
					child_task_atom = self._asl_atom_or_string(self._sanitize_name(task_name))
					parent_call_frame = self._call(
						"runtime_current_call",
						(
							"METHOD",
							parent_task_atom,
							self._call("runtime_args", parent_args),
							"BINDING",
							"SNAPSHOT",
							child_task_atom,
							self._call("runtime_args", handler_args),
						),
					)
					parent_choice = self._call(
						"runtime_method_choice",
						(
							"METHOD",
							parent_task_atom,
							self._call("runtime_args", parent_args),
							"BINDING",
						),
					)
					parent_current_method = self._call(
						"runtime_current_method",
						("METHOD", parent_task_atom, *parent_args, "BINDING"),
					)
					parent_active_method = self._call(
						"active_runtime_method",
						("METHOD", parent_task_atom, *parent_args, "BINDING"),
					)
					parent_snapshot = self._call(
						"runtime_method_snapshot",
						("METHOD", parent_task_atom, *parent_args, "BINDING"),
					)
					parent_blocked_method = self._call(
						"blocked_runtime_method",
						("METHOD", parent_task_atom, *parent_args, "BINDING"),
					)
					lines.append(
						f"-!{trigger} : {parent_call_frame} & {parent_active_method} "
						"& not runtime_pass_failed <-",
					)
					lines.extend(
						self._indent_body(
							[
								f'.print("runtime caller frame failed ", {fail_term})',
								"runtime_restore(SNAPSHOT)",
								".perceive",
								f"+blocked_runtime_choice({parent_choice})",
								f"+{parent_blocked_method}",
								f"-{parent_call_frame}",
								f"!{parent_trigger}",
								"runtime_commit(SNAPSHOT)",
								f"-{parent_current_method}",
								f"-{parent_active_method}",
								f".succeed_goal({parent_trigger})",
							],
						),
					)
					lines.append("")
					lines.append(
						f"-!{trigger} : {parent_current_method} & {parent_active_method} "
						"& not runtime_pass_failed <-",
					)
					lines.extend(
						self._indent_body(
							[
								f'.print("runtime caller branch failed ", {fail_term})',
								f"runtime_restore({parent_snapshot})",
								".perceive",
								f"+{parent_blocked_method}",
								f"!{parent_trigger}",
								f"runtime_commit({parent_snapshot})",
								f"-{parent_current_method}",
								f"-{parent_active_method}",
								f".succeed_goal({parent_trigger})",
							],
						),
					)
					lines.append("")
			if allow_repair:
				reported_failure = self._call("runtime_reported_failure", (fail_term,))
				for query_goal_index in range(max(0, query_goal_count), 0, -1):
					lines.append(
						f"-!{trigger} : runtime_active_query_goal({query_goal_index}) <-",
					)
					lines.extend(
						self._indent_body(
							[
								(
									f"if (not {reported_failure}) "
									f"{{ +{reported_failure}; "
									f'.print("runtime goal failed ", {fail_term}) }}'
								),
								f".fail_goal(runtime_execute_from_{query_goal_index})",
							],
						),
					)
					lines.append("")
				lines.append(f"-!{trigger} : true <-")
				lines.extend(
					self._indent_body(
						[
							(
								f"if (not {reported_failure}) "
								f"{{ +{reported_failure}; "
								f'.print("runtime goal failed ", {fail_term}) }}'
							),
							".fail",
						],
					),
				)
				lines.append("")
			else:
				lines.append(f"-!{trigger} : true <-")
				lines.extend(
					self._indent_body(
						[
							f'.print("runtime goal failed ", {fail_term})',
							".fail",
						],
					),
				)
				lines.append("")
		return lines

	def _runtime_caller_signatures(
		self,
		*,
		failed_task_name: str,
		method_library: HTNMethodLibrary | None,
		plan_library: PlanLibrary | None,
	) -> List[Tuple[str, Tuple[str, ...]]]:
		signatures: List[Tuple[str, Tuple[str, ...]]] = []
		seen: set[str] = set()
		failed_key = self._sanitize_name(str(failed_task_name or "").strip())

		def register(name: Any, parameters: Sequence[str]) -> None:
			task_name = str(name or "").strip()
			if not task_name or task_name in seen:
				return
			seen.add(task_name)
			signatures.append(
				(
					task_name,
					tuple(str(parameter).strip() for parameter in parameters if str(parameter).strip()),
				),
			)

		def maybe_register(
			child_name: Any,
			parent_name: Any,
			parent_parameters: Sequence[str],
		) -> None:
			child_key = self._sanitize_name(str(child_name or "").strip())
			if child_key and child_key == failed_key:
				register(parent_name, parent_parameters)

		if plan_library is not None:
			trigger_signatures = {
				str(getattr(plan.trigger, "symbol", "") or "").strip(): tuple(
					self._strip_type_annotation(str(argument).strip())
					for argument in (getattr(plan.trigger, "arguments", ()) or ())
					if str(argument).strip()
				)
				for plan in tuple(plan_library.plans or ())
				if getattr(plan, "trigger", None) is not None
			}
			for plan in tuple(plan_library.plans or ()):
				trigger = getattr(plan, "trigger", None)
				if trigger is None:
					continue
				parent_name = str(getattr(trigger, "symbol", "") or "").strip()
				parent_parameters = trigger_signatures.get(parent_name, ())
				for step in tuple(getattr(plan, "body", ()) or ()):
					step_kind = str(getattr(step, "kind", "") or "").strip()
					if step_kind not in {"subgoal", "action"}:
						continue
					maybe_register(getattr(step, "symbol", ""), parent_name, parent_parameters)
		if method_library is not None:
			compound_task_signatures = {
				str(getattr(task, "name", "") or "").strip(): tuple(
					getattr(task, "parameters", ()) or (),
				)
				for task in tuple(method_library.compound_tasks or ())
			}
			for method in tuple(method_library.methods or ()):
				parent_name = str(getattr(method, "task_name", "") or "").strip()
				parent_parameters = compound_task_signatures.get(
					parent_name,
					tuple(getattr(method, "task_args", ()) or getattr(method, "parameters", ()) or ()),
				)
				for step in tuple(getattr(method, "subtasks", ()) or ()):
					step_kind = str(getattr(step, "kind", "") or "").strip()
					child_name = (
						getattr(step, "action_name", None)
						if step_kind == "primitive"
						else getattr(step, "task_name", "")
					)
					maybe_register(
						child_name or getattr(step, "task_name", ""),
						parent_name,
						parent_parameters,
					)
		return signatures

	def _failure_handler_signatures(
		self,
		*,
		method_library: HTNMethodLibrary | None,
		plan_library: PlanLibrary | None,
		action_schemas: Sequence[Dict[str, Any]],
	) -> List[Tuple[str, Tuple[str, ...]]]:
		signatures: List[Tuple[str, Tuple[str, ...]]] = []
		seen_names: set[str] = set()

		def register(name: Any, parameters: Sequence[str]) -> None:
			task_name = str(name or "").strip()
			if not task_name or task_name in seen_names:
				return
			seen_names.add(task_name)
			signatures.append(
				(
					task_name,
					tuple(str(parameter).strip() for parameter in parameters if str(parameter).strip()),
				),
			)

		if plan_library is not None:
			trigger_signatures = {
				str(getattr(plan.trigger, "symbol", "") or "").strip(): tuple(
					self._strip_type_annotation(str(argument).strip())
					for argument in (getattr(plan.trigger, "arguments", ()) or ())
					if str(argument).strip()
				)
				for plan in tuple(plan_library.plans or ())
				if getattr(plan, "trigger", None) is not None
			}
			for task_name, parameters in trigger_signatures.items():
				register(task_name, parameters)
			for plan in tuple(plan_library.plans or ()):
				for step in tuple(getattr(plan, "body", ()) or ()):
					if str(getattr(step, "kind", "") or "").strip() != "subgoal":
						continue
					register(
						getattr(step, "symbol", ""),
						trigger_signatures.get(
							str(getattr(step, "symbol", "") or "").strip(),
							tuple(getattr(step, "arguments", ()) or ()),
						),
					)

		for schema in action_schemas or ():
			register(
				schema.get("source_name") or schema.get("functor") or "",
				tuple(schema.get("parameters") or ()),
			)

		if method_library is not None:
			for task in tuple(method_library.compound_tasks or ()) + tuple(method_library.primitive_tasks or ()):
				register(getattr(task, "name", ""), tuple(getattr(task, "parameters", ()) or ()))

		return signatures

	def _rewrite_primitive_wrappers_for_environment(self, agentspeak_code: str) -> str:
		start_marker = "/* Primitive Action Plans */"
		end_marker = "/* HTN Method Plans */"
		start_index = agentspeak_code.find(start_marker)
		end_index = agentspeak_code.find(end_marker)
		if start_index == -1 or end_index == -1 or end_index <= start_index:
			return agentspeak_code

		prefix = agentspeak_code[:start_index]
		section = agentspeak_code[start_index:end_index]
		suffix = agentspeak_code[end_index:]
		section_lines = section.splitlines()
		if not section_lines:
			return agentspeak_code

		header = section_lines[0]
		content_lines = section_lines[1:]
		chunks: List[List[str]] = []
		current: List[str] = []
		for line in content_lines:
			if not line.strip():
				if current:
					chunks.append(current)
					current = []
				continue
			current.append(line)
		if current:
			chunks.append(current)

		rewritten_chunks: List[str] = []
		for chunk in chunks:
			head_line = chunk[0]
			body_lines = chunk[1:]
			if not head_line.strip().startswith("+!"):
				rewritten_chunks.append("\n".join(chunk))
				continue
			if not body_lines:
				rewritten_chunks.append("\n".join(chunk))
				continue
			statements = [line.strip().rstrip(";.") for line in body_lines if line.strip()]
			if not statements or not statements[0]:
				rewritten_chunks.append("\n".join(chunk))
				continue
			action_statement = statements[0]
			statements = [action_statement, ".perceive"]
			rewritten_body = [
				f"\t{statement}{'.' if index == len(statements) - 1 else ';'}"
				for index, statement in enumerate(statements)
			]
			rewritten_chunks.append("\n".join([head_line, *rewritten_body]))

		rewritten_section = "\n\n".join([header, *rewritten_chunks]).rstrip() + "\n\n"
		return f"{prefix}{rewritten_section}{suffix}"

	def _rewrite_method_primitive_actions_to_goals(
		self,
		agentspeak_code: str,
		*,
		action_schemas: Sequence[Dict[str, Any]],
	) -> str:
		action_functors = {
			str(schema.get("functor") or "").strip()
			for schema in action_schemas or ()
			if str(schema.get("functor") or "").strip()
		}
		action_functors.update(
			self._sanitize_name(str(schema.get("source_name") or "").strip())
			for schema in action_schemas or ()
			if str(schema.get("source_name") or "").strip()
		)
		action_functors.discard("")
		if not action_functors:
			return agentspeak_code

		start_marker = "/* HTN Method Plans */"
		end_marker = "/* Failure Handlers */"
		start_index = agentspeak_code.find(start_marker)
		end_index = agentspeak_code.find(end_marker)
		if start_index == -1:
			return agentspeak_code
		if end_index == -1 or end_index <= start_index:
			end_index = len(agentspeak_code)

		prefix = agentspeak_code[:start_index]
		section = agentspeak_code[start_index:end_index]
		suffix = agentspeak_code[end_index:]
		rewritten_lines: List[str] = []
		changed = False
		for line in section.splitlines():
			rewritten_line = self._rewrite_method_primitive_action_line_to_goal(
				line,
				action_functors=action_functors,
			)
			if rewritten_line != line:
				changed = True
			rewritten_lines.append(rewritten_line)
		if not changed:
			return agentspeak_code
		rewritten_section = "\n".join(rewritten_lines)
		return f"{prefix}{rewritten_section}\n{suffix}"

	@staticmethod
	def _rewrite_method_primitive_action_line_to_goal(
		line: str,
		*,
		action_functors: Set[str],
	) -> str:
		match = re.match(r"^(\s*)(.*?)([;.])?\s*$", str(line))
		if match is None:
			return line
		indent, statement, suffix = match.groups()
		statement = statement.strip()
		if not statement:
			return line
		if statement.startswith(("+", "-", "!", ".", "?")):
			return line
		if statement == "true":
			return line
		call_match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)(?:\((.*)\))?", statement)
		if call_match is None:
			return line
		functor = call_match.group(1).strip()
		if functor not in action_functors:
			return line
		return f"{indent}!{statement}{suffix or ''}"

	def _ground_local_witness_method_plans(
		self,
		agentspeak_code: str,
		*,
		seed_facts: Sequence[str],
		runtime_objects: Sequence[str],
		object_types: Dict[str, str],
		type_parent_map: Dict[str, Optional[str]],
		enable_blocked_goal_guards: bool = False,
		max_candidates_per_clause: int = 64,
		max_total_specialised_chunks: int = 1024,
	) -> str:
		start_marker = "/* HTN Method Plans */"
		end_marker = "/* Failure Handlers */"
		start_index = agentspeak_code.find(start_marker)
		end_index = agentspeak_code.find(end_marker)
		if start_index == -1:
			return agentspeak_code
		if end_index == -1 or end_index <= start_index:
			end_index = len(agentspeak_code)

		prefix = agentspeak_code[:start_index]
		section = agentspeak_code[start_index:end_index]
		suffix = agentspeak_code[end_index:]
		section_lines = section.splitlines()
		if not section_lines:
			return agentspeak_code

		header = section_lines[0]
		content_lines = section_lines[1:]
		chunks: List[List[str]] = []
		current: List[str] = []
		for line in content_lines:
			if not line.strip():
				if current:
					chunks.append(current)
					current = []
				continue
			current.append(line)
		if current:
			chunks.append(current)

		fact_index, type_domains = self._runtime_fact_index_for_local_witness_grounding(
			seed_facts=seed_facts,
			runtime_objects=runtime_objects,
			object_types=object_types,
			type_parent_map=type_parent_map,
		)
		specialised_chunks: List[str] = []
		changed = False
		total_specialised_chunks = 0
		for chunk in chunks:
			specialised = self._specialise_method_chunk_local_witnesses(
				chunk,
				fact_index=fact_index,
				type_domains=type_domains,
				max_candidates_per_clause=max_candidates_per_clause,
			)
			next_specialised_count = total_specialised_chunks + max(0, len(specialised) - 1)
			if next_specialised_count > max_total_specialised_chunks:
				specialised = ["\n".join(chunk)]
			else:
				total_specialised_chunks = next_specialised_count
			if len(specialised) != 1 or specialised[0] != "\n".join(chunk):
				changed = True
			specialised_chunks.extend(specialised)

		pre_recursive_guard_chunks = list(specialised_chunks)
		specialised_chunks = self._insert_self_recursive_no_ancestor_guards(
			specialised_chunks,
		)
		if specialised_chunks != pre_recursive_guard_chunks:
			changed = True
		pre_guard_promotion_chunks = list(specialised_chunks)
		specialised_chunks = self._promote_body_no_ancestor_guards_to_context(
			specialised_chunks,
		)
		if specialised_chunks != pre_guard_promotion_chunks:
			changed = True
		_ = enable_blocked_goal_guards
		pre_ordered_chunks = list(specialised_chunks)
		specialised_chunks = self._order_runtime_method_plan_chunks(
			specialised_chunks,
			fact_index=fact_index,
		)
		if specialised_chunks != pre_ordered_chunks:
			changed = True

		if not changed:
			return agentspeak_code

		rewritten_section = "\n\n".join([header, *specialised_chunks]).rstrip() + "\n\n"
		return f"{prefix}{rewritten_section}{suffix}"

	def _defer_type_only_local_context_guards(self, agentspeak_code: str) -> str:
		"""
		Remove runtime-only local type guards that would bind variables too early.

		The generated S may retain object_type guards as type metadata. In Jason
		runtime execution, a method-local variable that appears only in an
		object_type guard is better left for the downstream subgoal/action context
		to bind at the point of execution; otherwise a stale type-domain choice can
		make an otherwise valid primitive action fail.
		"""

		start_marker = "/* HTN Method Plans */"
		end_marker = "/* Failure Handlers */"
		start_index = agentspeak_code.find(start_marker)
		end_index = agentspeak_code.find(end_marker)
		if start_index == -1:
			return agentspeak_code
		if end_index == -1 or end_index <= start_index:
			end_index = len(agentspeak_code)

		prefix = agentspeak_code[:start_index]
		section = agentspeak_code[start_index:end_index]
		suffix = agentspeak_code[end_index:]
		section_lines = section.splitlines()
		if not section_lines:
			return agentspeak_code

		header = section_lines[0]
		content_lines = section_lines[1:]
		chunks: List[List[str]] = []
		current: List[str] = []
		for line in content_lines:
			if not line.strip():
				if current:
					chunks.append(current)
					current = []
				continue
			current.append(line)
		if current:
			chunks.append(current)

		rewritten_chunks: List[str] = []
		changed = False
		for chunk in chunks:
			if not chunk:
				continue
			rewritten_head = self._defer_type_only_local_context_guards_for_head(chunk[0])
			if rewritten_head != chunk[0]:
				changed = True
			rewritten_chunks.append("\n".join([rewritten_head, *chunk[1:]]))

		if not changed:
			return agentspeak_code
		rewritten_section = "\n\n".join([header, *rewritten_chunks]).rstrip() + "\n\n"
		return f"{prefix}{rewritten_section}{suffix}"

	def _defer_type_only_local_context_guards_for_head(self, head_line: str) -> str:
		parsed_head = self._parse_asl_method_head(head_line)
		if parsed_head is None:
			return head_line
		_, head_args, context_parts = parsed_head
		if not context_parts:
			return head_line

		trigger_variables = {
			arg
			for arg in head_args
			if self._looks_like_asl_variable(str(arg))
		}
		non_type_context_variables: Set[str] = set()
		parsed_parts: List[Tuple[str, Optional[Dict[str, Any]]]] = []
		for part in context_parts:
			parsed = self._parse_asl_context_conjunct(str(part))
			parsed_parts.append((str(part), parsed))
			if parsed is None:
				continue
			if parsed.get("kind") == "atom" and parsed.get("predicate") == "object_type":
				continue
			non_type_context_variables.update(self._extract_asl_variables(str(part)))

		rewritten_context_parts: List[str] = []
		for part, parsed in parsed_parts:
			if parsed is None:
				rewritten_context_parts.append(part)
				continue
			if parsed.get("kind") != "atom" or parsed.get("predicate") != "object_type":
				rewritten_context_parts.append(part)
				continue
			args = tuple(parsed.get("args") or ())
			if not args:
				rewritten_context_parts.append(part)
				continue
			variable = str(args[0]).strip()
			if not self._looks_like_asl_variable(variable):
				rewritten_context_parts.append(part)
				continue
			if variable in trigger_variables or variable in non_type_context_variables:
				rewritten_context_parts.append(part)

		return self._replace_asl_method_context_parts(head_line, rewritten_context_parts)

	def _strip_seed_fact_beliefs(
		self,
		agentspeak_code: str,
		*,
		seed_facts: Sequence[str],
	) -> str:
		def normalise_belief_line(text: str) -> str:
			return re.sub(r"\s+", "", str(text or "").strip())

		seed_belief_lines = {
			normalise_belief_line(f"{atom}.")
			for atom in (
				self._hddl_fact_to_atom(fact)
				for fact in seed_facts
			)
			if atom
		}
		if not seed_belief_lines:
			return agentspeak_code
		lines = [
			line
			for line in agentspeak_code.splitlines()
			if normalise_belief_line(line) not in seed_belief_lines
		]
		return "\n".join(lines).rstrip() + "\n"

	def _promote_body_no_ancestor_guards_to_context(
		self,
		chunks: Sequence[str],
	) -> List[str]:
		rewritten_chunks: List[str] = []
		for chunk in chunks:
			lines = chunk.splitlines()
			if not lines:
				rewritten_chunks.append(chunk)
				continue
			head_line = lines[0]
			guard_context_parts: List[str] = []
			for line in lines[1:]:
				statement = str(line or "").strip().rstrip(";.")
				if not statement.startswith("pipeline.no_ancestor_goal("):
					continue
				if statement not in guard_context_parts:
					guard_context_parts.append(statement)
			if not guard_context_parts:
				rewritten_chunks.append(chunk)
				continue
			rewritten_head = self._append_asl_method_context_parts(
				head_line,
				guard_context_parts,
			)
			if rewritten_head == head_line:
				rewritten_chunks.append(chunk)
				continue
			rewritten_chunks.append("\n".join([rewritten_head, *lines[1:]]))
		return rewritten_chunks

	def _insert_self_recursive_no_ancestor_guards(
		self,
		chunks: Sequence[str],
	) -> List[str]:
		rewritten_chunks: List[str] = []
		for chunk in chunks:
			lines = chunk.splitlines()
			if not lines:
				rewritten_chunks.append(chunk)
				continue
			parsed_head = self._parse_asl_method_head(lines[0])
			if parsed_head is None:
				rewritten_chunks.append(chunk)
				continue
			task_name, _, _ = parsed_head
			rewritten_lines = [lines[0]]
			changed = False
			for line in lines[1:]:
				goal = self._parse_asl_goal_call(line)
				if goal is None or str(goal[0]).strip() != task_name:
					rewritten_lines.append(line)
					continue
				guard_args = ", ".join((self._asl_atom_or_string(task_name), *goal[1]))
				guard = f"pipeline.no_ancestor_goal({guard_args})"
				indent = re.match(r"^(\s*)", line).group(1) if re.match(r"^(\s*)", line) else ""
				guard_line = f"{indent}{guard};"
				if guard_line not in rewritten_lines:
					rewritten_lines.append(guard_line)
					changed = True
				rewritten_lines.append(line)
			rewritten_chunks.append("\n".join(rewritten_lines) if changed else chunk)
		return rewritten_chunks

	def _promote_body_blocked_goal_guards_to_context(
		self,
		chunks: Sequence[str],
	) -> List[str]:
		rewritten_chunks: List[str] = []
		for chunk in chunks:
			lines = chunk.splitlines()
			if not lines:
				rewritten_chunks.append(chunk)
				continue
			if self._parse_asl_method_head(lines[0]) is None:
				rewritten_chunks.append(chunk)
				continue
			blocked_guards: List[str] = []
			for line in lines[1:]:
				goal = self._parse_asl_goal_call(line)
				if goal is None:
					continue
				goal_task_name, goal_args = goal
				blocked_atom = self._call(
					"blocked_runtime_goal",
					(self._asl_atom_or_string(goal_task_name), *goal_args),
				)
				guard = f"not {blocked_atom}"
				if guard not in blocked_guards:
					blocked_guards.append(guard)
			if not blocked_guards:
				rewritten_chunks.append(chunk)
				continue
			rewritten_head = self._append_asl_method_context_parts(
				lines[0],
				blocked_guards,
			)
			if rewritten_head == lines[0]:
				rewritten_chunks.append(chunk)
				continue
			rewritten_chunks.append("\n".join([rewritten_head, *lines[1:]]))
		return rewritten_chunks

	@staticmethod
	def _append_asl_method_context_parts(
		head_line: str,
		extra_context_parts: Sequence[str],
	) -> str:
		if not extra_context_parts:
			return head_line
		match = re.match(r"^(\s*\+![^\s(:]+(?:\([^)]*\))?\s*:\s*)(.*?)(\s*<-\s*)$", head_line)
		if match is None:
			return head_line
		prefix, context_text, suffix = match.groups()
		context_parts = [
			part.strip()
			for part in str(context_text or "").split("&")
			if part.strip() and part.strip() != "true"
		]
		seen = set(context_parts)
		for part in extra_context_parts:
			rendered = str(part or "").strip()
			if not rendered or rendered in seen:
				continue
			context_parts.append(rendered)
			seen.add(rendered)
		rewritten_context = " & ".join(context_parts) if context_parts else "true"
		return f"{prefix}{rewritten_context}{suffix}"

	@staticmethod
	def _replace_asl_method_context_parts(
		head_line: str,
		context_parts: Sequence[str],
	) -> str:
		match = re.match(r"^(\s*\+![^\s(:]+(?:\([^)]*\))?\s*:\s*)(.*?)(\s*<-\s*)$", head_line)
		if match is None:
			return head_line
		prefix, _context_text, suffix = match.groups()
		filtered_context_parts = [
			str(part).strip()
			for part in context_parts
			if str(part).strip() and str(part).strip() != "true"
		]
		rewritten_context = " & ".join(filtered_context_parts) if filtered_context_parts else "true"
		return f"{prefix}{rewritten_context}{suffix}"

	def _order_runtime_method_plan_chunks(
		self,
		chunks: Sequence[str],
		*,
		fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
	) -> List[str]:
		if not chunks:
			return []

		current_fact_arg_pairs = self._runtime_fact_arg_pair_index(fact_index)
		parsed_chunks: List[Dict[str, Any]] = []
		for index, chunk in enumerate(chunks):
			lines = chunk.splitlines()
			if not lines:
				parsed_chunks.append({"index": index, "chunk": chunk, "task_name": "", "sort_key": (0,)})
				continue
			parsed_head = self._parse_asl_method_head(lines[0])
			if parsed_head is None:
				parsed_chunks.append({"index": index, "chunk": chunk, "task_name": "", "sort_key": (0,)})
				continue
			task_name, head_args, context_parts = parsed_head
			body_lines = list(lines[1:])
			parsed_chunks.append(
				{
					"index": index,
					"chunk": chunk,
					"task_name": task_name,
					"head_args": head_args,
					"context_parts": context_parts,
					"body_lines": body_lines,
				},
			)

		grouped_indexes: Dict[str, List[int]] = {}
		group_order: List[str] = []
		for index, item in enumerate(parsed_chunks):
			task_name = str(item.get("task_name") or f"__raw_{index}")
			if task_name not in grouped_indexes:
				grouped_indexes[task_name] = []
				group_order.append(task_name)
			grouped_indexes[task_name].append(index)

		ordered_chunks: List[str] = []
		for task_name in group_order:
			group_items = [parsed_chunks[index] for index in grouped_indexes[task_name]]
			for item in group_items:
				lines = str(item.get("chunk") or "").splitlines()
				body_lines = list(item.get("body_lines") or ())
				body_goals = [
					goal
					for goal in (
						self._parse_asl_goal_call(line)
						for line in body_lines
					)
					if goal is not None
				]
				body_goal_count = len(body_goals)
				has_self_recursive_goal = any(
					str(goal[0]).strip() == str(item.get("task_name") or "").strip()
					for goal in body_goals
				)
				variable_safe = self._chunk_runtime_variables_are_safe(lines)
				body_current_fact_pair_score = self._body_current_fact_pair_score(
					body_goals,
					current_fact_arg_pairs,
				)
				empty_body_rank = 0 if body_goal_count == 0 else 1
				body_goal_rank = 0 if body_goal_count == 0 else -body_goal_count
				grounded_head_arg_count = sum(
					1
					for arg in tuple(item.get("head_args") or ())
					if not self._looks_like_asl_variable(str(arg))
				)
				head_variable_set = {
					str(arg)
					for arg in tuple(item.get("head_args") or ())
					if self._looks_like_asl_variable(str(arg))
				}
				local_context_variables: Set[str] = set()
				non_type_context_count = sum(
					1
					for part in tuple(item.get("context_parts") or ())
					if not str(part).strip().startswith("object_type(")
				)
				grounded_context_arg_count = 0
				for part in tuple(item.get("context_parts") or ()):
					parsed = self._parse_asl_context_conjunct(str(part))
					if parsed is None:
						continue
					if parsed.get("kind") == "atom":
						if str(parsed.get("predicate") or "").strip() == "object_type":
							continue
						local_context_variables.update(
							str(arg)
							for arg in tuple(parsed.get("args") or ())
							if (
								self._looks_like_asl_variable(str(arg))
								and str(arg) not in head_variable_set
							)
						)
						grounded_context_arg_count += sum(
							1
							for arg in tuple(parsed.get("args") or ())
							if not self._looks_like_asl_variable(str(arg))
						)
					elif parsed.get("kind") == "inequality":
						grounded_context_arg_count += sum(
							1
							for arg in (str(parsed["lhs"]), str(parsed["rhs"]))
							if not self._looks_like_asl_variable(arg)
						)
				item["sort_key"] = (
					0 if variable_safe else 1,
					empty_body_rank,
					0 if not has_self_recursive_goal else 1,
					grounded_context_arg_count,
					len(local_context_variables),
					-non_type_context_count,
					body_goal_rank,
					-grounded_head_arg_count,
					-body_current_fact_pair_score,
					int(item.get("index", 0)),
				)
				item["variable_safe"] = variable_safe
			group_items.sort(key=lambda item: tuple(item.get("sort_key") or ()))
			ordered_chunks.extend(str(item["chunk"]) for item in group_items)

		return ordered_chunks

	def _runtime_fact_arg_pair_index(
		self,
		fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
	) -> Set[Tuple[str, str]]:
		arg_pairs: Set[Tuple[str, str]] = set()
		for (predicate, _), facts in fact_index.items():
			if predicate in {"object", "object_type"}:
				continue
			for fact_args in facts:
				ground_args = [
					self._canonical_runtime_token(str(arg))
					for arg in tuple(fact_args)
					if not self._looks_like_asl_variable(str(arg))
				]
				for left_index, left in enumerate(ground_args):
					for right in ground_args[left_index + 1:]:
						if left == right:
							continue
						arg_pairs.add((left, right))
						arg_pairs.add((right, left))
		return arg_pairs

	def _body_current_fact_pair_score(
		self,
		body_goals: Sequence[Tuple[str, Tuple[str, ...]]],
		current_fact_arg_pairs: Set[Tuple[str, str]],
	) -> int:
		score = 0
		for _, goal_args in body_goals:
			ground_args = [
				self._canonical_runtime_token(str(arg))
				for arg in tuple(goal_args)
				if not self._looks_like_asl_variable(str(arg))
			]
			matched_pairs: Set[Tuple[str, str]] = set()
			for left_index, left in enumerate(ground_args):
				for right in ground_args[left_index + 1:]:
					if left == right:
						continue
					pair = (left, right)
					if pair in current_fact_arg_pairs:
						matched_pairs.add(pair)
			score += len(matched_pairs)
		return score

	def _specialise_method_chunk_local_witnesses(
		self,
		chunk: Sequence[str],
		*,
		fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
		type_domains: Dict[str, Tuple[str, ...]],
		max_candidates_per_clause: int,
	) -> List[str]:
		original = "\n".join(chunk)
		if not chunk:
			return [original]

		parsed_head = self._parse_asl_method_head(chunk[0])
		if parsed_head is None:
			return [original]
		task_name, head_args, context_parts = parsed_head
		chunk_text = "\n".join(chunk)
		trigger_vars = {
			term
			for term in head_args
			if self._looks_like_asl_variable(term)
		}
		all_vars = self._extract_asl_variables(chunk_text)
		local_vars = sorted(all_vars - trigger_vars)
		if not local_vars:
			return [original]

		type_constraints: List[Tuple[str, str]] = []
		inequalities: List[Tuple[str, str]] = []
		binding_atoms: List[Dict[str, Any]] = []
		local_var_set = set(local_vars)
		for part in context_parts:
			parsed = self._parse_asl_context_conjunct(part)
			if parsed is None:
				continue
			kind = parsed.get("kind")
			if kind == "inequality":
				inequalities.append((str(parsed["lhs"]), str(parsed["rhs"])))
				continue
			if kind != "atom":
				continue
			predicate = str(parsed["predicate"])
			args = tuple(str(arg) for arg in parsed["args"])
			if predicate == "object_type" and len(args) == 2:
				type_constraints.append((args[0], args[1]))
				continue
			atom_vars = {
				term
				for term in args
				if self._looks_like_asl_variable(term)
			}
			if atom_vars & local_var_set:
				binding_atoms.append({"predicate": predicate, "args": args})

		binding_atoms.sort(
			key=lambda atom: (
				len(fact_index.get((str(atom["predicate"]), len(tuple(atom["args"]))), ())),
				-len(
					[
						term
						for term in tuple(atom["args"])
						if not self._looks_like_asl_variable(term)
					],
				),
			),
		)

		candidate_bindings = self._candidate_bindings_for_local_witnesses(
			binding_atoms=binding_atoms,
			type_constraints=type_constraints,
			inequalities=inequalities,
			local_vars=local_vars,
			fact_index=fact_index,
			type_domains=type_domains,
			max_candidates_per_clause=max_candidates_per_clause,
		)
		if not candidate_bindings:
			return [original]

		specialised_chunks: List[str] = []
		seen_chunks: set[str] = set()
		for binding in candidate_bindings:
			specialised_chunk = "\n".join(
				self._substitute_asl_bindings(line, binding)
				for line in chunk
			)
			if specialised_chunk == original or specialised_chunk in seen_chunks:
				continue
			seen_chunks.add(specialised_chunk)
			specialised_chunks.append(specialised_chunk)

		if not specialised_chunks:
			return [original]
		if original not in seen_chunks:
			specialised_chunks.append(original)
		return specialised_chunks

	def _chunk_runtime_variables_are_safe(
		self,
		chunk: Sequence[str],
	) -> bool:
		if not chunk:
			return True
		parsed_head = self._parse_asl_method_head(str(chunk[0]))
		if parsed_head is None:
			return True
		task_name, head_args, context_parts = parsed_head
		bound_variables = {
			str(arg)
			for arg in tuple(head_args or ())
			if self._looks_like_asl_variable(str(arg))
		}
		for part in tuple(context_parts or ()):
			parsed = self._parse_asl_context_conjunct(str(part))
			if parsed is None:
				continue
			if parsed.get("kind") == "atom":
				if str(parsed.get("predicate") or "").strip() == "object_type":
					continue
				bound_variables.update(
					str(arg)
					for arg in tuple(parsed.get("args") or ())
					if self._looks_like_asl_variable(str(arg))
				)
				continue
			if parsed.get("kind") == "inequality":
				for token in (str(parsed["lhs"]), str(parsed["rhs"])):
					if self._looks_like_asl_variable(token) and token not in bound_variables:
						return False
		for raw_line in chunk[1:]:
			statement = str(raw_line or "").strip().rstrip(";.")
			if not statement:
				continue
			statement_variables = self._extract_asl_variables(statement)
			if statement.startswith("!"):
				goal = self._parse_asl_goal_call(statement)
				if goal is not None:
					goal_name, _goal_args = goal
					unbound_variables = statement_variables - bound_variables
					if str(goal_name).strip() == str(task_name).strip() and unbound_variables:
						return False
					bound_variables.update(statement_variables)
					continue
			if statement_variables and not statement_variables.issubset(bound_variables):
				return False
			bound_variables.update(statement_variables)
		return True

	def _candidate_bindings_for_local_witnesses(
		self,
		*,
		binding_atoms: Sequence[Dict[str, Any]],
		type_constraints: Sequence[Tuple[str, str]],
		inequalities: Sequence[Tuple[str, str]],
		local_vars: Sequence[str],
		fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
		type_domains: Dict[str, Tuple[str, ...]],
		max_candidates_per_clause: int,
	) -> List[Dict[str, str]]:
		bindings: List[Dict[str, str]] = [{}]
		for atom in binding_atoms:
			next_bindings: List[Dict[str, str]] = []
			facts = fact_index.get((str(atom["predicate"]), len(tuple(atom["args"]))), ())
			if not facts:
				return []
			for binding in bindings:
				for fact_args in facts:
					matched = self._match_grounding_atom(
						tuple(atom["args"]),
						fact_args,
						binding,
					)
					if matched is None:
						continue
					if not self._binding_satisfies_local_witness_filters(
						matched,
						type_constraints=type_constraints,
						type_domains=type_domains,
						inequalities=inequalities,
						local_vars=local_vars,
						require_all_local_bindings=False,
					):
						continue
					next_bindings.append(matched)
					if len(next_bindings) > max_candidates_per_clause:
						return []
			if not next_bindings:
				return []
			bindings = next_bindings

		completed_bindings: List[Dict[str, str]] = []
		def expand_binding(index: int, binding: Dict[str, str]) -> bool:
			if len(completed_bindings) > max_candidates_per_clause:
				return False
			if index >= len(local_vars):
				if not self._binding_satisfies_local_witness_filters(
					binding,
					type_constraints=type_constraints,
					type_domains=type_domains,
					inequalities=inequalities,
					local_vars=local_vars,
					require_all_local_bindings=True,
				):
					return True
				completed_bindings.append(dict(binding))
				return True

			variable = str(local_vars[index])
			if variable in binding:
				return expand_binding(index + 1, binding)

			domain = self._local_witness_type_domain(
				variable,
				type_constraints=type_constraints,
				type_domains=type_domains,
			)
			if domain is None:
				return True
			if not domain:
				return True
			if len(domain) > max_candidates_per_clause:
				return False

			for value in domain:
				binding[variable] = value
				if not self._binding_satisfies_local_witness_filters(
					binding,
					type_constraints=type_constraints,
					type_domains=type_domains,
					inequalities=inequalities,
					local_vars=local_vars,
					require_all_local_bindings=False,
				):
					binding.pop(variable, None)
					continue
				if expand_binding(index + 1, binding) is False:
					binding.pop(variable, None)
					return False
				binding.pop(variable, None)
			return True

		for binding in bindings:
			if expand_binding(0, dict(binding)) is False:
				return []

		unique: Dict[Tuple[Tuple[str, str], ...], Dict[str, str]] = {}
		for binding in completed_bindings:
			signature = tuple(sorted(
				(item, value)
				for item, value in binding.items()
				if item in set(local_vars) or value
			))
			unique.setdefault(signature, binding)
		return list(unique.values())

	def _binding_satisfies_local_witness_filters(
		self,
		binding: Dict[str, str],
		*,
		type_constraints: Sequence[Tuple[str, str]],
		type_domains: Dict[str, Tuple[str, ...]],
		inequalities: Sequence[Tuple[str, str]],
		local_vars: Sequence[str],
		require_all_local_bindings: bool,
	) -> bool:
		if require_all_local_bindings and any(var not in binding for var in local_vars):
			return False

		for term, type_name in type_constraints:
			resolved = self._resolve_local_witness_term(term, binding)
			if resolved is None:
				continue
			domain = type_domains.get(str(type_name))
			if domain is None or resolved not in domain:
				return False

		for lhs, rhs in inequalities:
			left_value = self._resolve_local_witness_term(lhs, binding)
			right_value = self._resolve_local_witness_term(rhs, binding)
			if left_value is None or right_value is None:
				continue
			if self._canonical_runtime_token(left_value) == self._canonical_runtime_token(right_value):
				return False
		return True

	def _local_witness_type_domain(
		self,
		variable: str,
		*,
		type_constraints: Sequence[Tuple[str, str]],
		type_domains: Dict[str, Tuple[str, ...]],
	) -> Optional[Tuple[str, ...]]:
		required_types = [
			str(type_name)
			for term, type_name in type_constraints
			if term == variable
		]
		if not required_types:
			return None

		domain_sets = [
			set(type_domains.get(type_name, ()))
			for type_name in required_types
		]
		if not domain_sets:
			return ()
		domain = set.intersection(*domain_sets)
		return tuple(sorted(domain))

	@staticmethod
	def _match_grounding_atom(
		pattern_args: Sequence[str],
		fact_args: Sequence[str],
		binding: Dict[str, str],
	) -> Optional[Dict[str, str]]:
		if len(pattern_args) != len(fact_args):
			return None

		candidate = dict(binding)
		for pattern_term, fact_term in zip(pattern_args, fact_args):
			if JasonRunner._looks_like_asl_variable(pattern_term):
				existing = candidate.get(pattern_term)
				if existing is not None:
					if JasonRunner._canonical_runtime_token(existing) != (
						JasonRunner._canonical_runtime_token(fact_term)
					):
						return None
					continue
				candidate[pattern_term] = fact_term
				continue
			if JasonRunner._canonical_runtime_token(pattern_term) != (
				JasonRunner._canonical_runtime_token(fact_term)
			):
				return None
		return candidate

	def _runtime_fact_index_for_local_witness_grounding(
		self,
		*,
		seed_facts: Sequence[str],
		runtime_objects: Sequence[str],
		object_types: Dict[str, str],
		type_parent_map: Dict[str, Optional[str]],
	) -> Tuple[
		Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
		Dict[str, Tuple[str, ...]],
	]:
		facts_by_predicate: Dict[Tuple[str, int], set[Tuple[str, ...]]] = {}
		type_domains: Dict[str, set[str]] = {}

		for fact in seed_facts:
			atom = self._hddl_fact_to_atom(fact)
			parsed = self._parse_runtime_fact_atom(atom)
			if parsed is None:
				continue
			predicate, args = parsed
			facts_by_predicate.setdefault((predicate, len(args)), set()).add(args)

		for obj in runtime_objects:
			rendered_object = self._runtime_atom_term(str(obj))
			facts_by_predicate.setdefault(("object", 1), set()).add((rendered_object,))
			for type_name in self._type_closure(object_types.get(str(obj)), type_parent_map):
				type_atom = self._type_atom(type_name)
				facts_by_predicate.setdefault(("object_type", 2), set()).add(
					(rendered_object, type_atom),
				)
				type_domains.setdefault(type_atom, set()).add(rendered_object)

		return (
			{
				key: tuple(sorted(values))
				for key, values in facts_by_predicate.items()
			},
			{
				type_name: tuple(sorted(values))
				for type_name, values in type_domains.items()
			},
		)

	@staticmethod
	def _parse_runtime_fact_atom(atom: Optional[str]) -> Optional[Tuple[str, Tuple[str, ...]]]:
		text = str(atom or "").strip()
		if not text:
			return None
		match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)(?:\((.*)\))?", text)
		if match is None:
			return None
		predicate = match.group(1).strip()
		args_text = (match.group(2) or "").strip()
		if not args_text:
			return predicate, ()
		return predicate, JasonRunner._split_asl_arguments(args_text)

	def _parse_asl_method_head(
		self,
		head_line: str,
	) -> Optional[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]:
		match = re.match(
			r"^\s*\+!([^\s(:]+)(?:\(([^)]*)\))?\s*:\s*(.*?)\s*<-\s*$",
			head_line,
		)
		if match is None:
			return None
		task_name = match.group(1).strip()
		args_text = (match.group(2) or "").strip()
		context_text = (match.group(3) or "").strip()
		head_args = self._split_asl_arguments(args_text)
		context_parts = tuple(
			part.strip()
			for part in context_text.split("&")
			if part.strip()
		)
		return task_name, head_args, context_parts

	@staticmethod
	def _parse_asl_context_conjunct(part: str) -> Optional[Dict[str, Any]]:
		text = str(part or "").strip()
		if not text or text == "true":
			return None
		if "\\==" in text:
			lhs, rhs = text.split("\\==", 1)
			return {"kind": "inequality", "lhs": lhs.strip(), "rhs": rhs.strip()}
		match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)(?:\((.*)\))?", text)
		if match is None:
			return {"kind": "other", "text": text}
		predicate = match.group(1).strip()
		args_text = (match.group(2) or "").strip()
		args = JasonRunner._split_asl_arguments(args_text) if args_text else ()
		return {"kind": "atom", "predicate": predicate, "args": args}

	@staticmethod
	def _parse_asl_goal_call(line: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
		text = str(line or "").strip().rstrip(";.")
		if not text.startswith("!"):
			return None
		match = re.fullmatch(r"!([A-Za-z][A-Za-z0-9_]*)(?:\((.*)\))?", text)
		if match is None:
			return None
		task_name = match.group(1).strip()
		args_text = (match.group(2) or "").strip()
		args = JasonRunner._split_asl_arguments(args_text) if args_text else ()
		return task_name, args

	@staticmethod
	def _extract_asl_variables(text: str) -> Set[str]:
		return {
			token
			for token in re.findall(r"\b[A-Z][A-Z0-9_]*\b", str(text or ""))
			if JasonRunner._looks_like_asl_variable(token)
		}

	@staticmethod
	def _looks_like_asl_variable(token: str) -> bool:
		return re.fullmatch(r"[A-Z][A-Z0-9_]*", str(token or "").strip()) is not None

	@staticmethod
	def _split_asl_arguments(args_text: str) -> Tuple[str, ...]:
		text = str(args_text or "").strip()
		if not text:
			return ()
		parts: List[str] = []
		current: List[str] = []
		depth = 0
		quote: Optional[str] = None
		for character in text:
			if quote is not None:
				current.append(character)
				if character == quote:
					quote = None
				continue
			if character in {'"', "'"}:
				quote = character
				current.append(character)
				continue
			if character == "(":
				depth += 1
				current.append(character)
				continue
			if character == ")":
				depth = max(0, depth - 1)
				current.append(character)
				continue
			if character == "," and depth == 0:
				part = "".join(current).strip()
				if part:
					parts.append(part)
				current = []
				continue
			current.append(character)
		part = "".join(current).strip()
		if part:
			parts.append(part)
		return tuple(parts)

	def _resolve_local_witness_term(
		self,
		term: str,
		binding: Dict[str, str],
	) -> Optional[str]:
		token = str(term or "").strip()
		if self._looks_like_asl_variable(token):
			return binding.get(token)
		return token

	def _substitute_asl_bindings(
		self,
		text: str,
		binding: Dict[str, str],
	) -> str:
		rendered = str(text)
		for variable, value in sorted(binding.items(), key=lambda item: len(item[0]), reverse=True):
			pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(variable)}(?![A-Za-z0-9_])")
			rendered = pattern.sub(value, rendered)
		return rendered

	def _instrument_method_plans(
		self,
		agentspeak_code: str,
		method_library: HTNMethodLibrary | None,
		*,
		plan_library: PlanLibrary | None = None,
	) -> str:
		trace_sources: List[Any] = []
		if plan_library is not None and plan_library.plans:
			trace_sources = list(plan_library.plans)
		elif method_library is not None and method_library.methods:
			trace_sources = list(method_library.methods)
		if not trace_sources:
			return agentspeak_code
		if "runtime trace method" in agentspeak_code:
			return agentspeak_code

		start_marker = "/* HTN Method Plans */"
		end_marker = "/* Failure Handlers */"
		start_index = agentspeak_code.find(start_marker)
		end_index = agentspeak_code.find(end_marker)
		if start_index == -1:
			return agentspeak_code
		if end_index == -1 or end_index <= start_index:
			end_index = len(agentspeak_code)

		prefix = agentspeak_code[:start_index]
		section = agentspeak_code[start_index:end_index]
		suffix = agentspeak_code[end_index:]
		section_lines = section.splitlines()
		if not section_lines:
			return agentspeak_code

		header = section_lines[0]
		content_lines = section_lines[1:]
		chunks: List[List[str]] = []
		current: List[str] = []
		for line in content_lines:
			if not line.strip():
				if current:
					chunks.append(current)
					current = []
				continue
			current.append(line)
		if current:
			chunks.append(current)

		if len(chunks) != len(trace_sources):
			return agentspeak_code

		instrumented_chunks: List[str] = []
		for method_like, chunk in zip(trace_sources, chunks):
			head_line = chunk[0]
			body_lines = chunk[1:]
			parsed_head = self._parse_asl_method_head(head_line)
			active_line = ""
			if parsed_head is not None:
				task_name, trigger_args, _context_parts = parsed_head
				method_identity = self._runtime_method_identity(method_like)
				binding_term = self._runtime_method_binding_term(
					trigger_args=trigger_args,
					context_parts=_context_parts,
				)
				active_method = self._runtime_method_state_atom(
					"active_runtime_method",
					method_identity=method_identity,
					task_name=task_name,
					args=trigger_args,
					binding_term=binding_term,
				)
				method_snapshot = self._runtime_method_state_atom(
					"runtime_method_snapshot",
					method_identity=method_identity,
					task_name=task_name,
					args=trigger_args,
					binding_term=binding_term,
				)
				current_method = self._runtime_method_state_atom(
					"runtime_current_method",
					method_identity=method_identity,
					task_name=task_name,
					args=trigger_args,
					binding_term=binding_term,
				)
				blocked_method = self._runtime_method_state_atom(
					"blocked_runtime_method",
					method_identity=method_identity,
					task_name=task_name,
					args=trigger_args,
					binding_term=binding_term,
				)
				choice_term = self._runtime_method_choice_term(
					method_identity=method_identity,
					task_name=task_name,
					args=trigger_args,
					binding_term=binding_term,
				)
				head_line = self._append_asl_method_context_parts(
					head_line,
					(
						f"not {blocked_method}",
						f"not blocked_runtime_choice({choice_term})",
					),
				)
				active_line = f"\t+{active_method};"
				snapshot_line = f"\truntime_snapshot({method_snapshot});"
				choice_line = f"\truntime_record_query_choice({choice_term});"
				choice_point_line = (
					"\truntime_push_method_choice("
					f"{choice_term}, "
					f"{self._asl_atom_or_string(self._sanitize_name(task_name))}, "
					f"{self._call('runtime_args', trigger_args)}, "
					f"{method_snapshot}"
					");"
				)
				body_lines = self._mark_runtime_current_method_before_calls(
					body_lines,
					current_method,
					method_identity=method_identity,
					task_name=task_name,
					args=trigger_args,
					binding_term=binding_term,
					method_snapshot=method_snapshot,
				)
				body_lines = self._append_success_cleanup_statement(
					body_lines,
					f"runtime_pop_method_choice({choice_term})",
				)
				body_lines = self._append_success_cleanup_statement(
					body_lines,
					f"runtime_commit({method_snapshot})",
				)
				body_lines = self._append_success_cleanup_statement(
					body_lines,
					f"-{current_method}",
				)
				body_lines = self._append_success_cleanup_statement(
					body_lines,
					f"-{active_method}",
				)
			trace_line = self._render_method_trace_statement(method_like, head_line)
			prefix_lines = [head_line, trace_line]
			if active_line:
				prefix_lines.append(active_line)
				prefix_lines.append(snapshot_line)
				prefix_lines.append(choice_line)
				prefix_lines.append(choice_point_line)
			instrumented_chunks.append("\n".join([*prefix_lines, *body_lines]))

		instrumented_section = "\n\n".join([header, *instrumented_chunks]).rstrip() + "\n\n"
		return f"{prefix}{instrumented_section}{suffix}"

	def _mark_runtime_current_method_before_calls(
		self,
		body_lines: Sequence[str],
		current_method: str,
		*,
		method_identity: str,
		task_name: str,
		args: Sequence[str],
		binding_term: str,
		method_snapshot: str,
	) -> List[str]:
		current = str(current_method or "").strip()
		if not current:
			return list(body_lines)
		parent_frame_prefix = (
			self._asl_string(method_identity),
			self._asl_atom_or_string(self._sanitize_name(task_name)),
			self._call("runtime_args", tuple(args)),
			binding_term,
			method_snapshot,
		)
		rewritten: List[str] = []
		for line in body_lines:
			match = re.match(r"^(\s*)(.*?)([;.])\s*$", line)
			if match is None:
				rewritten.append(line)
				continue
			indent, statement, _suffix = match.groups()
			call_signature = self._parse_runtime_current_call_statement(statement)
			if call_signature is not None:
				child_name, child_args = call_signature
				call_frame = self._call(
					"runtime_current_call",
					(
						*parent_frame_prefix,
						self._asl_atom_or_string(self._sanitize_name(child_name)),
						self._call("runtime_args", child_args),
					),
				)
				rewritten.append(f"{indent}-+{current};")
				rewritten.append(f"{indent}-+{call_frame};")
				rewritten.append(f"{indent}{statement};")
				rewritten.append(f"{indent}-{call_frame}{_suffix}")
				continue
			rewritten.append(line)
		return rewritten

	@staticmethod
	def _is_runtime_current_method_call_statement(statement: str) -> bool:
		return JasonRunner._parse_runtime_current_call_statement(statement) is not None

	@staticmethod
	def _parse_runtime_current_call_statement(statement: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
		text = str(statement or "").strip()
		if not text or text == "true":
			return None
		if text.startswith("!"):
			text = text[1:].strip()
		elif text.startswith(("+", "-", "?", ".", "not ")):
			return None
		match = re.fullmatch(r"([a-z][A-Za-z0-9_]*)(?:\((.*)\))?", text)
		if match is None:
			return None
		name = match.group(1).strip()
		args_text = (match.group(2) or "").strip()
		args = JasonRunner._split_asl_arguments(args_text) if args_text else ()
		return name, args

	def _runtime_method_identity(self, method: Any) -> str:
		method_name = str(
			getattr(method, "plan_name", None)
			or getattr(method, "method_name", None)
			or "",
		).strip()
		return method_name or "anonymous_method"

	def _runtime_method_state_atom(
		self,
		predicate_name: str,
		*,
		method_identity: str,
		task_name: str,
		args: Sequence[str],
		binding_term: str,
	) -> str:
		return self._call(
			predicate_name,
			(
				self._asl_string(method_identity),
				self._asl_atom_or_string(self._sanitize_name(task_name)),
				*args,
				binding_term,
			),
		)

	def _runtime_method_choice_term(
		self,
		*,
		method_identity: str,
		task_name: str,
		args: Sequence[str],
		binding_term: str,
	) -> str:
		return self._call(
			"runtime_method_choice",
			(
				self._asl_string(method_identity),
				self._asl_atom_or_string(self._sanitize_name(task_name)),
				self._call("runtime_args", tuple(args)),
				binding_term,
			),
		)

	def _runtime_method_binding_term(
		self,
		*,
		trigger_args: Sequence[str],
		context_parts: Sequence[str],
	) -> str:
		trigger_variables = {
			token
			for argument in trigger_args
			for token in self._extract_asl_variables(argument)
		}
		local_variables: List[str] = []
		seen: set[str] = set()
		for part in context_parts:
			for variable in self._extract_asl_variables(part):
				if variable in trigger_variables or variable in seen:
					continue
				seen.add(variable)
				local_variables.append(variable)
		if not local_variables:
			return "runtime_binding"
		return f"runtime_binding({', '.join(local_variables)})"

	@staticmethod
	def _append_success_cleanup_statement(
		body_lines: Sequence[str],
		cleanup_statement: str,
	) -> List[str]:
		cleanup = str(cleanup_statement or "").strip()
		if not cleanup:
			return list(body_lines)
		if not body_lines:
			return [f"\t{cleanup}."]
		rewritten_lines = list(body_lines)
		last_line = rewritten_lines[-1]
		match = re.match(r"^(\s*)(.*?)([;.])\s*$", last_line)
		if match is None:
			rewritten_lines.append(f"\t{cleanup}.")
			return rewritten_lines
		indent, statement, _suffix = match.groups()
		rewritten_lines[-1] = f"{indent}{statement};"
		rewritten_lines.append(f"{indent}{cleanup}.")
		return rewritten_lines

	@classmethod
	def _context_atom(
		cls,
		predicate_name: str,
		args: Sequence[str],
		*,
		preserve_variables: bool = False,
	) -> str:
		rendered_args = tuple(
			(
				str(arg).strip()
				if preserve_variables and cls._looks_like_asl_variable(str(arg).strip())
				else cls._runtime_atom_term(arg)
			)
			for arg in args
			if str(arg).strip()
		)
		return cls._call(predicate_name, rendered_args)

	def _render_method_trace_statement(self, method: Any, head_line: str) -> str:
		trigger_args = self._extract_trigger_args(head_line)
		method_name = str(
			getattr(method, "method_name", None)
			or getattr(method, "plan_name", None)
			or "",
		).strip()
		if getattr(method, "plan_name", None) is not None:
			method_name = re.sub(r"__variant_\d+$", "", method_name)
		trace_line = self._render_flat_method_trace_statement(method_name, trigger_args)
		return f"\t{trace_line};"

	@classmethod
	def _render_flat_method_trace_statement(
		cls,
		method_name: str,
		rendered_args: Sequence[str],
	) -> str:
		trace_args = [
			cls._asl_string("runtime trace method flat "),
			cls._asl_string(method_name),
		]
		for arg in rendered_args:
			trace_args.extend((cls._asl_string("|"), arg))
		return f".print({', '.join(trace_args)})"

	@staticmethod
	def _extract_trigger_args(head_line: str) -> Tuple[str, ...]:
		match = re.match(r"^\s*\+![^\s(:]+(?:\(([^)]*)\))?\s*:", head_line)
		if match is None:
			return ()
		args_text = (match.group(1) or "").strip()
		if not args_text:
			return ()
		return tuple(part.strip() for part in args_text.split(",") if part.strip())

	def _build_runner_mas2j(self, domain_name: str) -> str:
		sanitized_domain = re.sub(r"[^a-zA-Z0-9_]+", "_", domain_name).strip("_").lower()
		if not sanitized_domain:
			sanitized_domain = "runtime"
		return (
			f"MAS execute_{sanitized_domain} {{\n"
			f"    environment: {self.environment_class_name}\n"
			"    agents: agentspeak_generated;\n"
			"    aslSourcePath: \".\";\n"
			"}\n"
		)

	def _build_no_ancestor_goal_internal_action_source(self) -> str:
		return """
package pipeline;

import jason.asSemantics.DefaultInternalAction;
import jason.asSemantics.Event;
import jason.asSemantics.IntendedMeans;
import jason.asSemantics.Intention;
import jason.asSemantics.TransitionSystem;
import jason.asSemantics.Unifier;
import jason.asSyntax.Literal;
import jason.asSyntax.Term;

public class no_ancestor_goal extends DefaultInternalAction {

	@Override
	public Object execute(TransitionSystem ts, Unifier un, Term[] args) throws Exception {
		if (args.length < 1) {
			return false;
		}

		Intention currentIntention = ts.getC().getSelectedIntention();
		if (currentIntention == null) {
			Event event = ts.getC().getSelectedEvent();
			if (event != null) {
				currentIntention = event.getIntention();
			}
		}
		if (currentIntention == null) {
			return true;
		}

		String requestedFunctor = canonicalTerm(args[0].capply(un));
		if (requestedFunctor.isEmpty()) {
			return false;
		}

		int requestedArity = args.length - 1;
		String[] requestedArgs = new String[requestedArity];
		for (int index = 0; index < requestedArity; index++) {
			requestedArgs[index] = canonicalTerm(args[index + 1].capply(un));
		}

		for (IntendedMeans intendedMeans : currentIntention) {
			Literal literal = intendedMeans.getTrigger().getLiteral();
			if (literal == null) {
				continue;
			}
			if (!requestedFunctor.equals(canonicalText(literal.getFunctor()))) {
				continue;
			}
			if (literal.getArity() != requestedArity) {
				continue;
			}
			boolean sameGoal = true;
			for (int index = 0; index < requestedArity; index++) {
				String ancestorArg = canonicalTerm(literal.getTerm(index).capply(intendedMeans.getUnif()));
				if (!requestedArgs[index].equals(ancestorArg)) {
					sameGoal = false;
					break;
				}
			}
			if (sameGoal) {
				return false;
			}
		}
		return true;
	}

	private String canonicalTerm(Term term) {
		return canonicalText(term == null ? "" : term.toString());
	}

	private String canonicalText(String rawValue) {
		String value = rawValue == null ? "" : rawValue.trim();
		if (value.length() >= 2) {
			boolean quoted =
				(value.startsWith("\\\"") && value.endsWith("\\\""))
				|| (value.startsWith("'") && value.endsWith("'"));
			if (quoted) {
				value = value.substring(1, value.length() - 1);
			}
		}
		return value;
	}
}
""".strip() + "\n"

	def _build_choose_runtime_choice_internal_action_source(self) -> str:
		return """
package pipeline;

import java.util.HashSet;
import java.util.Set;

import jason.asSemantics.DefaultInternalAction;
import jason.asSemantics.TransitionSystem;
import jason.asSemantics.Unifier;
import jason.asSyntax.Literal;
import jason.asSyntax.NumberTerm;
import jason.asSyntax.Term;

public class choose_runtime_choice extends DefaultInternalAction {

	@Override
	public Object execute(TransitionSystem ts, Unifier un, Term[] args) throws Exception {
		if (args.length < 3) {
			return false;
		}

		String targetIndex = canonicalTerm(args[1].capply(un));
		if (targetIndex.isEmpty()) {
			return false;
		}

		Term selected = null;
		long selectedSequence = Long.MIN_VALUE;
		Set<String> blockedChoices = blockedChoicesForTarget(ts, un, targetIndex);

		ts.getAg().getBB().getLock().lock();
		try {
			for (Literal belief : ts.getAg().getBB()) {
				if (!"runtime_query_choice_frame".equals(belief.getFunctor()) || belief.getArity() != 3) {
					continue;
				}
				if (!targetIndex.equals(canonicalTerm(belief.getTerm(0).capply(un)))) {
					continue;
				}
				Term choice = belief.getTerm(2).capply(un);
				if (blockedChoices.contains(choice.toString())) {
					continue;
				}
				long sequence = sequenceValue(belief.getTerm(1).capply(un));
				if (sequence >= selectedSequence) {
					selectedSequence = sequence;
					selected = choice;
				}
			}
			if (selected == null) {
				selected = latestUnblockedLegacyChoice(ts, un, targetIndex, blockedChoices);
			}
		} finally {
			ts.getAg().getBB().getLock().unlock();
		}

		return selected != null && un.unifies(args[2], selected);
	}

	private Set<String> blockedChoicesForTarget(TransitionSystem ts, Unifier un, String targetIndex) {
		Set<String> blockedChoices = new HashSet<>();
		ts.getAg().getBB().getLock().lock();
		try {
			for (Literal belief : ts.getAg().getBB()) {
				if ("blocked_runtime_choice".equals(belief.getFunctor()) && belief.getArity() == 1) {
					blockedChoices.add(belief.getTerm(0).capply(un).toString());
					continue;
				}
				if (!"runtime_backtracked_choice".equals(belief.getFunctor()) || belief.getArity() != 2) {
					continue;
				}
				if (!targetIndex.equals(canonicalTerm(belief.getTerm(0).capply(un)))) {
					continue;
				}
				blockedChoices.add(belief.getTerm(1).capply(un).toString());
			}
		} finally {
			ts.getAg().getBB().getLock().unlock();
		}
		return blockedChoices;
	}

	private Term latestUnblockedLegacyChoice(
		TransitionSystem ts,
		Unifier un,
		String targetIndex,
		Set<String> blockedChoices
	) {
		Term selected = null;
		for (Literal belief : ts.getAg().getBB()) {
			if (!"runtime_query_choice".equals(belief.getFunctor()) || belief.getArity() != 2) {
				continue;
			}
			if (!targetIndex.equals(canonicalTerm(belief.getTerm(0).capply(un)))) {
				continue;
			}
			Term choice = belief.getTerm(1).capply(un);
			if (blockedChoices.contains(choice.toString())) {
				continue;
			}
			selected = choice;
		}
		return selected;
	}

	private long sequenceValue(Term term) throws Exception {
		if (term instanceof NumberTerm) {
			return (long) ((NumberTerm) term).solve();
		}
		String text = canonicalTerm(term);
		if (text.endsWith(".0")) {
			text = text.substring(0, text.length() - 2);
		}
		return Long.parseLong(text);
	}

	private String canonicalTerm(Term term) {
		return canonicalText(term == null ? "" : term.toString());
	}

	private String canonicalText(String rawValue) {
		String value = rawValue == null ? "" : rawValue.trim();
		if (value.length() >= 2) {
			boolean quoted =
				(value.startsWith("\\\"") && value.endsWith("\\\""))
				|| (value.startsWith("'") && value.endsWith("'"));
			if (quoted) {
				value = value.substring(1, value.length() - 1);
			}
		}
		return value;
	}
}
""".strip() + "\n"

	def _build_environment_java_source(
		self,
		*,
		action_schemas: Sequence[Dict[str, Any]],
		seed_facts: Sequence[str],
	) -> str:
		seed_atoms = [
			atom
			for atom in (self._hddl_fact_to_atom(fact) for fact in seed_facts)
			if atom is not None
		]

		action_blocks: List[str] = []
		for schema in action_schemas:
			functor = schema.get("functor")
			if not functor:
				continue
			source_name = str(schema.get("source_name") or functor)
			parameters = [str(item) for item in (schema.get("parameters") or [])]
			preconditions = list(schema.get("preconditions") or [])
			precondition_clauses = list(schema.get("precondition_clauses") or [])
			if not precondition_clauses:
				precondition_clauses = [preconditions] if preconditions else [[]]
			effects = list(self._ordered_runtime_effects(schema.get("effects") or []))
			action_blocks.append(
				"""
		register(new ActionSchema(
			{functor},
			{source_name},
			new String[]{{{parameters}}},
			{precondition_clauses},
			new Pattern[]{{{effects}}}
		));
		""".strip().format(
					functor=self._java_quote(functor),
					source_name=self._java_quote(source_name),
					parameters=", ".join(self._java_quote(item) for item in parameters),
					precondition_clauses=self._render_precondition_clauses_java(
						precondition_clauses,
					),
					effects=", ".join(self._render_pattern_java(item) for item in effects),
				),
			)

		seed_lines = "\n".join(
			f"\t\tworld.add({self._java_quote(atom)});"
			for atom in seed_atoms
		)
		action_lines = "\n\t\t".join(action_blocks)
		if not action_lines:
			action_lines = "// no action schemas"

		return f"""
import jason.asSyntax.Literal;
import jason.asSyntax.Structure;
import jason.environment.Environment;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

public class {self.environment_class_name} extends Environment {{

	private static final class Pattern {{
		final String predicate;
		final boolean positive;
		final String[] args;

		Pattern(String predicate, boolean positive, String[] args) {{
			this.predicate = predicate;
			this.positive = positive;
			this.args = args;
		}}
	}}

	private static final class ActionSchema {{
		final String name;
		final String sourceName;
		final String[] parameters;
		final Pattern[][] preconditionClauses;
		final Pattern[] effects;

		ActionSchema(
			String name,
			String sourceName,
			String[] parameters,
			Pattern[][] preconditionClauses,
			Pattern[] effects
		) {{
			this.name = name;
			this.sourceName = sourceName;
			this.parameters = parameters;
			this.preconditionClauses = preconditionClauses;
			this.effects = effects;
		}}
	}}

	private final Set<String> world = new LinkedHashSet<>();
	private final Map<String, ActionSchema> actions = new HashMap<>();
	private final Map<String, Set<String>> snapshots = new HashMap<>();
	private String activeQueryGoalIndex = null;
	private long runtimeChoiceSequence = 0L;

	@Override
	public synchronized void init(String[] args) {{
		super.init(args);
		seedInitialFacts();
		loadActions();
		syncPercepts();
		System.out.println("runtime env ready");
	}}

	@Override
	public synchronized boolean executeAction(String agName, Structure action) {{
		if ("true".equals(action.getFunctor()) && action.getArity() == 0) {{
			return true;
		}}
		if (handleRuntimeControlAction(action)) {{
			return true;
		}}
		ActionSchema schema = actions.get(action.getFunctor());
		if (schema == null) {{
			System.out.println("runtime env unknown action " + action);
			return false;
		}}
		String tracedAction = renderTraceAction(schema.sourceName, action);
		if (action.getArity() != schema.parameters.length) {{
			System.out.println("runtime env action failed " + tracedAction + " reason=arity");
			return false;
		}}

		Map<String, String> bindings = new HashMap<>();
		for (int i = 0; i < schema.parameters.length; i++) {{
			String parameter = canonical(schema.parameters[i]);
			String value = canonical(action.getTerm(i).toString());
			bindings.put(parameter, value);
			if (parameter.startsWith("?")) {{
				bindings.put(parameter.substring(1), value);
			}}
		}}

		if (!checkPreconditions(schema.preconditionClauses, bindings)) {{
			System.out.println("runtime env action failed " + tracedAction + " reason=precondition");
			return false;
		}}

		applyEffects(schema.effects, bindings);
		syncPercepts();
		System.out.println("runtime env action success " + tracedAction);
		return true;
	}}

	private boolean handleRuntimeControlAction(Structure action) {{
		String functor = action.getFunctor();
		if (
			!"runtime_snapshot".equals(functor)
			&& !"runtime_restore".equals(functor)
			&& !"runtime_commit".equals(functor)
			&& !"runtime_set_active_query_goal".equals(functor)
			&& !"runtime_clear_active_query_goal".equals(functor)
			&& !"runtime_record_query_choice".equals(functor)
			&& !"runtime_push_method_choice".equals(functor)
			&& !"runtime_pop_method_choice".equals(functor)
		) {{
			return false;
		}}
		if ("runtime_set_active_query_goal".equals(functor)) {{
			clearActiveQueryGoalFacts();
			String index = action.getArity() == 0 ? "0" : canonical(action.getTerm(0).toString());
			activeQueryGoalIndex = index;
			world.add("runtime_active_query_goal(" + index + ")");
			syncPercepts();
			System.out.println("runtime env active query goal " + index);
			return true;
		}}
		if ("runtime_clear_active_query_goal".equals(functor)) {{
			clearActiveQueryGoalFacts();
			activeQueryGoalIndex = null;
			syncPercepts();
			String index = action.getArity() == 0 ? "all" : canonical(action.getTerm(0).toString());
			System.out.println("runtime env clear active query goal " + index);
			return true;
		}}
		if ("runtime_record_query_choice".equals(functor)) {{
			recordRuntimeQueryChoice(action);
			return true;
		}}
		if ("runtime_push_method_choice".equals(functor)) {{
			pushRuntimeMethodChoice(action);
			return true;
		}}
		if ("runtime_pop_method_choice".equals(functor)) {{
			popRuntimeMethodChoice(action);
			return true;
		}}
		String key = snapshotKey(action);
		if ("runtime_snapshot".equals(functor)) {{
			snapshots.put(key, new LinkedHashSet<>(world));
			System.out.println("runtime env snapshot " + key);
			return true;
		}}
		if ("runtime_restore".equals(functor)) {{
			Set<String> snapshot = snapshots.get(key);
			if (snapshot == null) {{
				return false;
			}}
			world.clear();
			world.addAll(snapshot);
			syncPercepts();
			System.out.println("runtime env restore " + key);
			return true;
		}}
		snapshots.remove(key);
		System.out.println("runtime env commit " + key);
		return true;
	}}

	private void clearActiveQueryGoalFacts() {{
		world.removeIf(fact -> fact.startsWith("runtime_active_query_goal("));
	}}

	private void recordRuntimeQueryChoice(Structure action) {{
		if (activeQueryGoalIndex == null || action.getArity() == 0) {{
			return;
		}}
		String choice = action.getTerm(0).toString();
		runtimeChoiceSequence += 1L;
		String lastPrefix = "runtime_last_query_choice(" + activeQueryGoalIndex + ",";
		String lastFramePrefix = "runtime_last_query_choice_frame(" + activeQueryGoalIndex + ",";
		world.removeIf(fact -> fact.startsWith(lastPrefix));
		world.removeIf(fact -> fact.startsWith(lastFramePrefix));
		world.add(lastPrefix + choice + ")");
		world.add("runtime_query_choice(" + activeQueryGoalIndex + "," + choice + ")");
		world.add(lastFramePrefix + runtimeChoiceSequence + "," + choice + ")");
		world.add(
			"runtime_query_choice_frame("
			+ activeQueryGoalIndex
			+ ","
			+ runtimeChoiceSequence
			+ ","
			+ choice
			+ ")"
		);
		syncPercepts();
	}}

	private void pushRuntimeMethodChoice(Structure action) {{
		if (action.getArity() < 4) {{
			return;
		}}
		String payload = action.getTerm(0).toString()
			+ "," + action.getTerm(1)
			+ "," + action.getTerm(2)
			+ "," + action.getTerm(3);
		String fact = "runtime_method_choice_point(" + payload + ")";
		world.remove(fact);
		clearLatestRuntimeMethodChoicePoint();
		world.add(fact);
		world.add("runtime_latest_method_choice_point(" + payload + ")");
		syncPercepts();
	}}

	private void popRuntimeMethodChoice(Structure action) {{
		if (action.getArity() == 0) {{
			return;
		}}
		String choice = action.getTerm(0).toString();
		String latestPrefix = "runtime_latest_method_choice_point(" + choice + ",";
		String latest = null;
		for (String fact : world) {{
			if (fact.startsWith(latestPrefix)) {{
				latest = fact;
			}}
		}}
		if (latest != null) {{
			String payload = latest.substring("runtime_latest_method_choice_point(".length(), latest.length() - 1);
			world.remove("runtime_method_choice_point(" + payload + ")");
		}} else {{
			world.removeIf(fact -> fact.startsWith("runtime_method_choice_point(" + choice + ","));
		}}
		clearLatestRuntimeMethodChoicePoint();
		recomputeLatestRuntimeMethodChoicePoint();
		syncPercepts();
	}}

	private void clearLatestRuntimeMethodChoicePoint() {{
		world.removeIf(fact -> fact.startsWith("runtime_latest_method_choice_point("));
	}}

	private void recomputeLatestRuntimeMethodChoicePoint() {{
		String latestPayload = null;
		for (String fact : world) {{
			if (fact.startsWith("runtime_method_choice_point(")) {{
				latestPayload = fact.substring("runtime_method_choice_point(".length(), fact.length() - 1);
			}}
		}}
		if (latestPayload != null) {{
			world.add("runtime_latest_method_choice_point(" + latestPayload + ")");
		}}
	}}

	private String snapshotKey(Structure action) {{
		if (action.getArity() == 0) {{
			return "default";
		}}
		String[] args = new String[action.getArity()];
		for (int i = 0; i < action.getArity(); i++) {{
			args[i] = canonical(action.getTerm(i).toString());
		}}
		return String.join("|", args);
	}}

	private void seedInitialFacts() {{
		world.clear();
		snapshots.clear();
		runtimeChoiceSequence = 0L;
{seed_lines if seed_lines else ""}
	}}

	private void loadActions() {{
		actions.clear();
		{action_lines}
	}}

	private void register(ActionSchema schema) {{
		actions.put(schema.name, schema);
	}}

	private boolean checkPreconditions(Pattern[][] preconditionClauses, Map<String, String> bindings) {{
		if (preconditionClauses.length == 0) {{
			return true;
		}}
		for (Pattern[] clause : preconditionClauses) {{
			if (checkPreconditionClause(clause, bindings)) {{
				return true;
			}}
		}}
		return false;
	}}

	private boolean checkPreconditionClause(Pattern[] preconditions, Map<String, String> bindings) {{
		for (Pattern pattern : preconditions) {{
			if ("=".equals(pattern.predicate) && pattern.args.length == 2) {{
				String left = resolveToken(pattern.args[0], bindings);
				String right = resolveToken(pattern.args[1], bindings);
				boolean equal = left.equals(right);
				if (pattern.positive != equal) {{
					return false;
				}}
				continue;
			}}

			String grounded = ground(pattern.predicate, pattern.args, bindings);
			boolean holds;
			if (pattern.positive) {{
				holds = world.contains(grounded);
			}} else {{
				holds = !world.contains(grounded);
			}}
			if (!holds) {{
				return false;
			}}
		}}
		return true;
	}}

	private void applyEffects(Pattern[] effects, Map<String, String> bindings) {{
		for (Pattern pattern : effects) {{
			if ("=".equals(pattern.predicate)) {{
				continue;
			}}
			if (pattern.positive) {{
				continue;
			}}
			String grounded = ground(pattern.predicate, pattern.args, bindings);
			world.remove(grounded);
		}}
		for (Pattern pattern : effects) {{
			if ("=".equals(pattern.predicate)) {{
				continue;
			}}
			if (!pattern.positive) {{
				continue;
			}}
			String grounded = ground(pattern.predicate, pattern.args, bindings);
			world.add(grounded);
		}}
	}}

	private String ground(String predicate, String[] args, Map<String, String> bindings) {{
		if (args.length == 0) {{
			return predicate;
		}}
		String[] groundedArgs = Arrays.stream(args)
			.map(arg -> renderTerm(resolveToken(arg, bindings)))
			.toArray(String[]::new);
		return predicate + "(" + String.join(",", groundedArgs) + ")";
	}}

	private String resolveToken(String rawToken, Map<String, String> bindings) {{
		String token = canonical(rawToken);
		if (bindings.containsKey(token)) {{
			return bindings.get(token);
		}}
		if (token.startsWith("?")) {{
			String bare = token.substring(1);
			if (bindings.containsKey(bare)) {{
				return bindings.get(bare);
			}}
		}}
		return token;
	}}

	private String canonical(String token) {{
		String value = token == null ? "" : token.trim();
		if (value.length() >= 2) {{
			boolean quoted =
				(value.startsWith("\\\"") && value.endsWith("\\\""))
				|| (value.startsWith("'") && value.endsWith("'"));
			if (quoted) {{
				value = value.substring(1, value.length() - 1);
			}}
		}}
		return value;
	}}

	private String renderTerm(String token) {{
		String value = canonical(token);
		if (value.matches("[a-z][a-z0-9_]*")) {{
			return value;
		}}
		return "\\\"" + value.replace("\\\\", "\\\\\\\\").replace("\\\"", "\\\\\\\"") + "\\\"";
	}}

	private String renderTraceAction(String sourceName, Structure action) {{
		if (action.getArity() == 0) {{
			return sourceName + "()";
		}}
		String[] args = new String[action.getArity()];
		for (int i = 0; i < action.getArity(); i++) {{
			args[i] = canonical(action.getTerm(i).toString());
		}}
		return sourceName + "(" + String.join(",", args) + ")";
	}}

	private void syncPercepts() {{
		clearPercepts();
		for (String atom : world) {{
			addPercept(Literal.parseLiteral(atom));
		}}
		informAgsEnvironmentChanged();
	}}
}}
""".strip() + "\n"

	def _compile_environment_java(
		self,
		*,
		java_bin: str,
		javac_bin: str,
		jason_jar: Path,
		env_java_path: Path,
		output_path: Path,
	) -> None:
		java_home = str(Path(java_bin).resolve().parent.parent)
		env = dict(os.environ)
		env["JAVA_HOME"] = java_home
		env["PATH"] = f"{java_home}/bin:{env.get('PATH', '')}"
		java_sources = sorted(
			str(path.relative_to(output_path))
			for path in output_path.rglob("*.java")
		)
		compile_cmd = [
			javac_bin,
			"-cp",
			str(jason_jar),
			*java_sources,
		]
		result = subprocess.run(
			compile_cmd,
			cwd=output_path,
			text=True,
			capture_output=True,
			check=False,
			env=env,
		)
		if result.returncode == 0:
			return

		raise JasonValidationError(
			"Jason environment Java compilation failed.",
			metadata={
				"java_bin": java_bin,
				"javac_bin": javac_bin,
				"environment_java": str(env_java_path),
				"java_sources": java_sources,
				"stdout": result.stdout,
				"stderr": result.stderr,
				"return_code": result.returncode,
			},
		)

	@staticmethod
	def _java_quote(value: str) -> str:
		escaped = value.replace("\\", "\\\\").replace('"', '\\"')
		return f'"{escaped}"'

	def _render_pattern_java(self, payload: Dict[str, Any]) -> str:
		predicate = self._sanitize_name(str(payload.get("predicate", "")))
		args = [str(item) for item in (payload.get("args") or [])]
		is_positive = bool(payload.get("is_positive", True))
		args_expr = ", ".join(self._java_quote(item) for item in args)
		return (
			f"new Pattern({self._java_quote(predicate)}, "
			f"{str(is_positive).lower()}, new String[]{{{args_expr}}})"
		)

	def _render_precondition_clauses_java(
		self,
		clauses: Sequence[Sequence[Dict[str, Any]]],
	) -> str:
		if not clauses:
			return "new Pattern[][]{}"
		rendered_clauses = []
		for clause in clauses:
			rendered_patterns = ", ".join(self._render_pattern_java(item) for item in clause)
			rendered_clauses.append(f"new Pattern[]{{{rendered_patterns}}}")
		return f"new Pattern[][]{{{', '.join(rendered_clauses)}}}"

	def _resolve_log_config(self) -> Path:
		log_conf = (
			self.jason_src_dir
			/ "jason-interpreter"
			/ "src"
			/ "main"
			/ "resources"
			/ "templates"
			/ "console-info-logging.properties"
		)
		if log_conf.exists():
			return log_conf
		raise JasonValidationError(
			"Jason runtime log configuration file is missing.",
			metadata={"log_conf": str(log_conf)},
		)

	def _ensure_jason_jar(self, java_bin: str) -> Path:
		jar_path = self._find_jason_jar()
		if jar_path is not None:
			return jar_path

		self._build_jason_cli(java_bin)
		jar_path = self._find_jason_jar()
		if jar_path is not None:
			return jar_path

		raise JasonValidationError(
			"Jason CLI jar is unavailable after build.",
			metadata={
				"jason_src_dir": str(self.jason_src_dir),
			},
		)

	def _find_jason_jar(self) -> Optional[Path]:
		bin_dir = self.jason_src_dir / "jason-cli" / "build" / "bin"
		if not bin_dir.exists():
			return None

		jars = sorted(bin_dir.glob("jason-cli-all-*.jar"), key=self._jar_version_key, reverse=True)
		if not jars:
			return None
		return jars[0]

	@staticmethod
	def _jar_version_key(path: Path) -> Tuple[int, ...]:
		match = re.search(r"jason-cli-all-(\d+(?:\.\d+)*)\.jar$", path.name)
		if not match:
			return (0,)
		return tuple(int(item) for item in match.group(1).split("."))

	def _build_jason_cli(self, java_bin: str) -> None:
		gradlew = self.jason_src_dir / "gradlew"
		if not gradlew.exists():
			raise JasonValidationError(
				"Jason source directory is missing gradlew.",
				metadata={"gradlew": str(gradlew)},
			)

		java_home = str(Path(java_bin).resolve().parent.parent)
		env = dict(os.environ)
		env["JAVA_HOME"] = java_home
		env["PATH"] = f"{java_home}/bin:{env.get('PATH', '')}"

		result = subprocess.run(
			[str(gradlew), "config"],
			cwd=self.jason_src_dir,
			text=True,
			capture_output=True,
			check=False,
			timeout=600,
			env=env,
		)
		if result.returncode == 0:
			return

		raise JasonValidationError(
			"Jason build failed while running './gradlew config'.",
			metadata={
				"return_code": result.returncode,
				"stdout": result.stdout,
				"stderr": result.stderr,
				"java_home": java_home,
			},
		)

	def _select_java_binary(self) -> Tuple[str, int]:
		candidate_bins = self._discover_java_candidates()
		supported: List[Tuple[str, int]] = []
		unsupported: Dict[str, Optional[int]] = {}

		for candidate in candidate_bins:
			major = self._probe_java_binary(candidate)
			if major is None:
				unsupported[candidate] = None
				continue
			if self.min_java_major <= major <= self.max_java_major:
				supported.append((candidate, major))
			else:
				unsupported[candidate] = major

		if not supported:
			raise JasonValidationError(
				"No supported Java runtime found for Jason execution (requires Java 23).",
				metadata={"candidates": unsupported},
			)

		supported.sort(key=lambda item: item[1], reverse=True)
		return supported[0]

	def _select_javac_binary(self, java_bin: str) -> str:
		java_home = Path(java_bin).resolve().parent.parent
		candidates = [
			str(java_home / "bin" / "javac"),
			shutil.which("javac") or "",
		]
		for candidate in candidates:
			if not candidate:
				continue
			path = Path(candidate)
			if path.exists() and os.access(path, os.X_OK):
				return str(path)
		raise JasonValidationError(
			"No javac binary found for Jason environment compilation.",
			metadata={"java_bin": java_bin, "candidates": candidates},
		)

	def _discover_java_candidates(self) -> List[str]:
		candidates: List[str] = []
		self._append_candidate(candidates, os.getenv("JASON_RUNTIME_JAVA_BIN"))
		self._append_candidate(candidates, os.getenv("STAGE6_JAVA_BIN"))

		runtime_java_home = os.getenv("JASON_RUNTIME_JAVA_HOME") or os.getenv("STAGE6_JAVA_HOME")
		if runtime_java_home:
			self._append_candidate(candidates, str(Path(runtime_java_home) / "bin" / "java"))

		java_home = os.getenv("JAVA_HOME")
		if java_home:
			self._append_candidate(candidates, str(Path(java_home) / "bin" / "java"))

		which_java = shutil.which("java")
		self._append_candidate(candidates, which_java)

		if os.name == "posix":
			for root in (
				Path.home() / "Library" / "Java" / "JavaVirtualMachines",
				Path("/Library/Java/JavaVirtualMachines"),
			):
				if not root.exists():
					continue
				for jdk_home in sorted(root.glob("*/Contents/Home/bin/java")):
					self._append_candidate(candidates, str(jdk_home))

		return candidates

	@staticmethod
	def _append_candidate(candidates: List[str], candidate: Optional[str]) -> None:
		if not candidate:
			return
		resolved = str(Path(candidate).expanduser())
		if resolved in candidates:
			return
		candidates.append(resolved)

	@staticmethod
	def _probe_java_binary(java_bin: str) -> Optional[int]:
		java_path = Path(java_bin)
		if not java_path.exists():
			return None
		try:
			result = subprocess.run(
				[str(java_path), "-version"],
				text=True,
				capture_output=True,
				check=False,
				timeout=10,
			)
		except Exception:
			return None

		version_text = (result.stderr or "") + "\n" + (result.stdout or "")
		match = re.search(r'version "([^"]+)"', version_text)
		if not match:
			return None
		version = match.group(1)
		return JasonRunner._java_major_from_version(version)

	@staticmethod
	def _java_major_from_version(version: str) -> Optional[int]:
		if not version:
			return None
		parts = version.split(".")
		if parts[0] == "1" and len(parts) > 1:
			try:
				return int(parts[1])
			except ValueError:
				return None
		try:
			return int(parts[0])
		except ValueError:
			return None

	def _is_successful_run(
		self,
		*,
		stdout: str,
		exit_code: Optional[int],
		timed_out: bool,
		environment_result: EnvironmentAdapterResult,
	) -> bool:
		if timed_out:
			return False
		if exit_code is None or exit_code != 0:
			return False
		if self.success_marker not in stdout:
			return False
		if not environment_result.success:
			return False
		return True

	def _failure_reason(
		self,
		stdout: str,
		stderr: str,
		exit_code: Optional[int],
		timed_out: bool,
		environment_result: EnvironmentAdapterResult,
	) -> str:
		if timed_out:
			return f"timeout ({self.timeout_seconds}s)"
		if exit_code is None:
			return "missing process exit code"
		if exit_code != 0:
			return f"process exited with code {exit_code}"
		if self.failure_marker in stdout:
			return "failure marker detected in stdout"
		if self.success_marker not in stdout:
			stderr_hint = stderr.strip().splitlines()[-1] if stderr.strip() else "none"
			return f"success marker missing (stderr tail: {stderr_hint})"
		if not environment_result.success:
			return (
				"environment adapter validation failed: "
				+ (environment_result.error or "unknown adapter error")
			)
		return "unknown validation error"

	def _failure_class(
		self,
		stdout: str,
		exit_code: Optional[int],
		timed_out: bool,
		environment_result: EnvironmentAdapterResult,
	) -> str:
		if timed_out:
			return "timeout"
		if exit_code is None:
			return "missing_exit_code"
		if exit_code != 0:
			return "runtime_process_failed"
		if self.failure_marker in stdout:
			return "runtime_failure_marker"
		if self.success_marker not in stdout:
			return "missing_success_marker"
		if not environment_result.success:
			return "environment_adapter_failure"
		return "validation_failed"

	def _run_consistency_checks(
		self,
		*,
		action_path: Sequence[str],
		method_trace: Sequence[Dict[str, Any]],
		method_library: HTNMethodLibrary | None,
		action_schemas: Sequence[Dict[str, Any]],
		seed_facts: Sequence[str],
		problem_file: str | Path | None,
		skip_method_trace_diagnostics: bool = False,
	) -> Dict[str, Any]:
		action_replay = self._replay_action_path_against_schemas(
			action_path=action_path,
			action_schemas=action_schemas,
			seed_facts=seed_facts,
		)
		if skip_method_trace_diagnostics:
			method_trace_check = {
				"passed": None,
				"failure_class": None,
				"message": "skipped by caller",
			}
		else:
			method_trace_check = self._check_method_trace_reconstruction(
				action_path=action_path,
				method_trace=method_trace,
				method_library=method_library,
				problem_file=problem_file,
			)
		return {
			"success": bool(action_replay.get("passed", False))
			and method_trace_check.get("passed") is not False,
			"action_path_schema_replay": action_replay,
			"method_trace_reconstruction": method_trace_check,
		}

	def _replay_action_path_against_schemas(
		self,
		*,
		action_path: Sequence[str],
		action_schemas: Sequence[Dict[str, Any]],
		seed_facts: Sequence[str],
	) -> Dict[str, Any]:
		world = {
			atom
			for atom in (self._hddl_fact_to_atom(fact) for fact in seed_facts)
			if atom is not None
		}
		schema_lookup = self._action_schema_lookup(action_schemas)

		for index, step in enumerate(action_path):
			parsed_step = self._parse_runtime_action_step(step)
			if parsed_step is None:
				return {
					"passed": False,
					"failure_class": "action_path_malformed_step",
					"message": f"runtime action step #{index + 1} is malformed: {step}",
					"checked_steps": index,
					"world_facts": sorted(world),
				}
			action_name, action_args = parsed_step
			schema = schema_lookup.get(action_name)
			if schema is None:
				return {
					"passed": False,
					"failure_class": "action_path_unknown_action",
					"message": f"runtime action step #{index + 1} references unknown action '{action_name}'",
					"checked_steps": index,
					"world_facts": sorted(world),
				}
			parameters = [str(item) for item in (schema.get("parameters") or [])]
			if len(parameters) != len(action_args):
				return {
					"passed": False,
					"failure_class": "action_path_arity_mismatch",
					"message": (
						f"runtime action step #{index + 1} has arity {len(action_args)} for "
						f"'{action_name}', expected {len(parameters)}"
					),
					"checked_steps": index,
					"world_facts": sorted(world),
				}

			bindings: Dict[str, str] = {}
			for parameter, value in zip(parameters, action_args):
				token = self._canonical_runtime_token(parameter)
				bindings[token] = value
				if token.startswith("?"):
					bindings[token[1:]] = value

			precondition_clauses = list(schema.get("precondition_clauses") or [])
			if not precondition_clauses:
				precondition_clauses = [list(schema.get("preconditions") or [])]
			if not any(
				self._replay_precondition_clause_holds(clause, bindings, world)
				for clause in precondition_clauses
			):
				return {
					"passed": False,
					"failure_class": "action_path_precondition_violation",
					"message": (
						f"runtime action step #{index + 1} violates schema preconditions for "
						f"'{action_name}{self._render_runtime_args(action_args)}'"
					),
					"checked_steps": index,
					"world_facts": sorted(world),
				}

			for effect in self._ordered_runtime_effects(schema.get("effects") or []):
				predicate = str(effect.get("predicate", "")).strip()
				if not predicate or predicate == "=":
					continue
				grounded = self._ground_runtime_pattern(
					predicate,
					effect.get("args") or [],
					bindings,
				)
				if effect.get("is_positive", True):
					world.add(grounded)
				else:
					world.discard(grounded)

		return {
			"passed": True,
			"failure_class": None,
			"message": None,
			"checked_steps": len(action_path),
			"world_facts": sorted(world),
		}

	def _check_method_trace_reconstruction(
		self,
		*,
		action_path: Sequence[str],
		method_trace: Sequence[Dict[str, Any]],
		method_library: HTNMethodLibrary | None,
		problem_file: str | Path | None,
	) -> Dict[str, Any]:
		if problem_file is None:
			return {
				"passed": None,
				"failure_class": None,
				"message": "skipped: no problem_file",
			}
		if method_library is None or not method_library.methods:
			return {
				"passed": False,
				"failure_class": "method_trace_reconstruction_failed",
				"message": "method trace cannot be checked without a non-empty method library",
			}

		from verification.official_plan_verifier import IPCPlanVerifier

		verifier = IPCPlanVerifier()
		try:
			rendered_plan = verifier._render_supported_hierarchical_plan(
				domain_file=problem_file,
				problem_file=problem_file,
				action_path=action_path,
				method_library=method_library,
				method_trace=method_trace,
			)
		except Exception as exc:
			return {
				"passed": False,
				"failure_class": "method_trace_reconstruction_failed",
				"message": str(exc),
			}

		build_warning = getattr(verifier, "_last_hierarchical_build_warning", None)
		if not rendered_plan:
			return {
				"passed": False,
				"failure_class": "method_trace_reconstruction_failed",
				"message": "hierarchical plan reconstruction returned no plan",
			}
		if build_warning:
			return {
				"passed": False,
				"failure_class": "method_trace_partial_reconstruction",
				"message": build_warning,
			}
		return {
			"passed": True,
			"failure_class": None,
			"message": None,
		}

	def _replay_precondition_clause_holds(
		self,
		clause: Sequence[Dict[str, Any]],
		bindings: Dict[str, str],
		world: Set[str],
	) -> bool:
		for pattern in clause:
			predicate = str(pattern.get("predicate", "")).strip()
			args = [str(item) for item in (pattern.get("args") or [])]
			is_positive = bool(pattern.get("is_positive", True))
			if predicate == "=" and len(args) == 2:
				left = self._resolve_runtime_token(args[0], bindings)
				right = self._resolve_runtime_token(args[1], bindings)
				if (left == right) != is_positive:
					return False
				continue
			grounded = self._ground_runtime_pattern(predicate, args, bindings)
			holds = grounded in world if is_positive else grounded not in world
			if not holds:
				return False
		return True

	@staticmethod
	def _ordered_runtime_effects(effects: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
		"""
		Apply delete effects before add effects.

		PDDL state-transition semantics treats a fact that is both deleted and added
		by the same action instance as true in the successor state.
		"""

		normalized_effects = tuple(dict(effect) for effect in effects)
		negative_effects = tuple(
			effect for effect in normalized_effects if not bool(effect.get("is_positive", True))
		)
		positive_effects = tuple(
			effect for effect in normalized_effects if bool(effect.get("is_positive", True))
		)
		return (*negative_effects, *positive_effects)

	@staticmethod
	def _parse_runtime_action_step(step: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
		text = (step or "").strip()
		match = re.fullmatch(r"([A-Za-z0-9_-]+)(?:\((.*)\))?", text)
		if match is None:
			return None
		action_name = match.group(1).strip()
		args_text = (match.group(2) or "").strip()
		if not args_text:
			return action_name, ()
		return action_name, tuple(
			part.strip()
			for part in args_text.split(",")
			if part.strip()
		)

	@staticmethod
	def _canonical_runtime_token(token: str) -> str:
		value = str(token or "").strip()
		if len(value) >= 2 and (
			(value.startswith('"') and value.endswith('"'))
			or (value.startswith("'") and value.endswith("'"))
		):
			return value[1:-1]
		return value

	def _resolve_runtime_token(
		self,
		token: str,
		bindings: Dict[str, str],
	) -> str:
		canonical = self._canonical_runtime_token(token)
		if canonical in bindings:
			return bindings[canonical]
		if canonical.startswith("?") and canonical[1:] in bindings:
			return bindings[canonical[1:]]
		return canonical

	def _ground_runtime_pattern(
		self,
		predicate: str,
		args: Sequence[str],
		bindings: Dict[str, str],
	) -> str:
		functor = self._sanitize_name(predicate)
		if not args:
			return functor
		grounded_args = [
			self._runtime_atom_term(self._resolve_runtime_token(arg, bindings))
			for arg in args
		]
		return f"{functor}({','.join(grounded_args)})"

	@staticmethod
	def _render_runtime_args(args: Sequence[str]) -> str:
		if not args:
			return "()"
		return f"({', '.join(args)})"

	@staticmethod
	@lru_cache(maxsize=131072)
	def _hddl_fact_to_atom(fact: str) -> Optional[str]:
		text = (fact or "").strip()
		if not text.startswith("(") or not text.endswith(")"):
			return None
		inner = text[1:-1].strip()
		if not inner or inner.startswith("not "):
			return None
		tokens = inner.split()
		if not tokens:
			return None
		predicate, args = tokens[0], tokens[1:]
		if predicate == "=":
			return None
		functor = JasonRunner._sanitize_name(predicate)
		if not args:
			return functor
		rendered_args = [JasonRunner._runtime_atom_term(arg) for arg in args]
		return f"{functor}({','.join(rendered_args)})"

	@staticmethod
	def _normalise_process_output(output: str | bytes | None) -> str:
		if output is None:
			return ""
		if isinstance(output, bytes):
			return output.decode("utf-8", errors="replace")
		return output

	@staticmethod
	def _combine_process_output(stdout: str, stderr: str) -> str:
		if not stderr:
			return stdout
		if not stdout:
			return stderr
		separator = "" if stdout.endswith("\n") else "\n"
		return f"{stdout}{separator}{stderr}"

	@staticmethod
	def _indent_body(lines: Iterable[str]) -> List[str]:
		body_lines = list(lines)
		if not body_lines:
			return ["\ttrue."]
		rendered: List[str] = []
		last_index = len(body_lines) - 1
		for index, line in enumerate(body_lines):
			suffix = "." if index == last_index else ";"
			rendered.append(f"\t{line}{suffix}")
		return rendered
