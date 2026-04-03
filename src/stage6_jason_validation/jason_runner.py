"""
Stage 6 Jason runner.

Runs generated AgentSpeak code with Jason (RunLocalMAS), boots a real Jason
`Environment` implementation for domain action semantics, and returns structured
validation metadata for pipeline logging.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from stage3_method_synthesis.htn_schema import HTNLiteral, HTNMethodLibrary
from stage6_jason_validation.environment_adapter import (
	EnvironmentAdapterResult,
	Stage6EnvironmentAdapter,
	build_environment_adapter,
)


class JasonValidationError(RuntimeError):
	"""Raised when Stage 6 Jason validation fails."""

	def __init__(self, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
		super().__init__(message)
		self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class JasonValidationResult:
	"""Structured result for Stage 6 validation."""

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
	artifacts: Dict[str, str]

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
		}


class JasonRunner:
	"""Run Stage 5 AgentSpeak output in Jason and validate runtime outcomes."""

	backend_name = "RunLocalMAS"
	success_marker = "execute success"
	failure_marker = "execute failed"
	min_java_major = 17
	max_java_major = 23
	environment_class_name = "Stage6PipelineEnvironment"

	def __init__(
		self,
		*,
		stage6_dir: str | Path | None = None,
		timeout_seconds: int = 120,
		environment_adapter: Stage6EnvironmentAdapter | None = None,
		environment_adapter_name: str | None = None,
	) -> None:
		base_dir = (
			Path(stage6_dir).resolve()
			if stage6_dir is not None
			else Path(__file__).resolve().parent
		)
		self.stage6_dir = base_dir
		self.jason_src_dir = self.stage6_dir / "jason_src"
		self.timeout_seconds = timeout_seconds
		adapter_name = environment_adapter_name or os.getenv("STAGE6_ENV_ADAPTER")
		self.environment_adapter = environment_adapter or build_environment_adapter(adapter_name)

	def validate(
		self,
		*,
		agentspeak_code: str,
		target_literals: Sequence[HTNLiteral],
		method_library: HTNMethodLibrary | None = None,
		action_schemas: Sequence[Dict[str, Any]],
		seed_facts: Sequence[str] = (),
		runtime_objects: Sequence[str] = (),
		object_types: Optional[Dict[str, str]] = None,
		type_parent_map: Optional[Dict[str, Optional[str]]] = None,
		domain_name: str,
		problem_file: str | Path | None = None,
		output_dir: str | Path,
		completion_mode: str = "target_literals",
		ordered_query_sequence: bool = True,
		planning_domain: Any | None = None,
		guided_action_path: Sequence[str] = (),
		guided_method_trace: Sequence[Dict[str, Any]] = (),
		skip_method_trace_reconstruction: bool = False,
	) -> JasonValidationResult:
		"""Execute Jason validation and return a structured result."""

		if not action_schemas:
			raise JasonValidationError(
				"Stage 6 requires action schemas for real environment execution.",
				metadata={"action_schema_count": 0},
			)

		output_path = Path(output_dir).resolve()
		output_path.mkdir(parents=True, exist_ok=True)

		java_bin, java_major = self._select_java_binary()
		javac_bin = self._select_javac_binary(java_bin)
		jason_jar = self._ensure_jason_jar(java_bin)
		log_conf = self._resolve_log_config()

		runner_asl_path = output_path / "agentspeak_generated.asl"
		runner_mas2j_path = output_path / "jason_runner.mas2j"
		env_java_path = output_path / f"{self.environment_class_name}.java"
		env_class_path = output_path / f"{self.environment_class_name}.class"
		stdout_path = output_path / "jason_stdout.txt"
		stderr_path = output_path / "jason_stderr.txt"
		action_path_path = output_path / "action_path.txt"
		method_trace_path = output_path / "method_trace.json"
		validation_json_path = output_path / "jason_validation.json"

		unordered_target_order_ids: List[str] = []
		unordered_target_order_signatures: List[str] = []
		effective_agentspeak_code = agentspeak_code
		if (
			not guided_action_path
			and not ordered_query_sequence
			and completion_mode == "target_literals"
			and method_library is not None
			and planning_domain is not None
		):
			ordering = self._infer_unordered_target_execution_order(
				target_literals=target_literals,
				method_library=method_library,
				action_schemas=action_schemas,
				seed_facts=seed_facts,
				runtime_objects=runtime_objects,
				object_types=object_types or {},
				planning_domain=planning_domain,
				output_path=output_path,
			)
			unordered_target_order_ids = list(ordering.get("target_ids") or [])
			unordered_target_order_signatures = list(ordering.get("target_signatures") or [])
			if unordered_target_order_ids:
				effective_agentspeak_code = self._reorder_unordered_control_plan_blocks(
					agentspeak_code,
					unordered_target_order_ids,
				)

		runner_asl = self._build_runner_asl(
			effective_agentspeak_code,
			target_literals,
			method_library=method_library,
			runtime_objects=runtime_objects,
			object_types=object_types or {},
			type_parent_map=type_parent_map or {},
			completion_mode=completion_mode,
			ordered_query_sequence=ordered_query_sequence,
			guided_action_path=guided_action_path,
			guided_method_trace=guided_method_trace,
		)
		runner_mas2j = self._build_runner_mas2j(domain_name)
		env_source = self._build_environment_java_source(
			action_schemas=action_schemas,
			seed_facts=seed_facts,
			target_literals=target_literals,
		)
		runner_asl_path.write_text(runner_asl)
		runner_mas2j_path.write_text(runner_mas2j)
		env_java_path.write_text(env_source)
		self._compile_environment_java(
			java_bin=java_bin,
			javac_bin=javac_bin,
			jason_jar=jason_jar,
			env_java_path=env_java_path,
			output_path=output_path,
		)
		if not env_class_path.exists():
			raise JasonValidationError(
				"Stage 6 environment class compilation completed but class file is missing.",
				metadata={
					"environment_java": str(env_java_path),
					"environment_class": str(env_class_path),
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
		except subprocess.TimeoutExpired as exc:
			timed_out = True
			raw_stdout = exc.stdout or ""
			raw_stderr = exc.stderr or ""

		stdout_text = self._normalise_process_output(raw_stdout)
		stderr_text = self._normalise_process_output(raw_stderr)
		stdout = self._combine_process_output(stdout_text, stderr_text)
		stderr = stderr_text
		action_path = self._extract_action_path(stdout)
		extracted_method_trace = self._extract_method_trace(stdout)
		method_trace = list(guided_method_trace) if guided_method_trace else extracted_method_trace
		failed_goals = self._extract_failed_goals(stdout)

		stdout_path.write_text(stdout)
		stderr_path.write_text(stderr)
		action_path_path.write_text(self._render_action_path(action_path))
		method_trace_path.write_text(json.dumps(method_trace, indent=2))

		artifacts = {
			"agentspeak_generated": str(runner_asl_path),
			"jason_runner_mas2j": str(runner_mas2j_path),
			"stage6_environment_java": str(env_java_path),
			"stage6_environment_class": str(env_class_path),
			"jason_stdout": str(stdout_path),
			"jason_stderr": str(stderr_path),
			"action_path": str(action_path_path),
			"method_trace": str(method_trace_path),
			"jason_validation": str(validation_json_path),
		}
		if unordered_target_order_ids:
			artifacts["unordered_target_order_ids"] = list(unordered_target_order_ids)
			artifacts["unordered_target_order_signatures"] = list(unordered_target_order_signatures)
		environment_result = self.environment_adapter.validate(stdout=stdout, stderr=stderr)
		consistency_checks = self._run_consistency_checks(
			action_path=action_path,
			method_trace=method_trace,
			method_library=method_library,
			action_schemas=action_schemas,
			seed_facts=seed_facts,
			problem_file=problem_file,
			skip_method_trace_reconstruction=skip_method_trace_reconstruction,
		)
		is_success = self._is_successful_run(
			stdout=stdout,
			exit_code=exit_code,
			timed_out=timed_out,
			environment_result=environment_result,
			consistency_checks=consistency_checks,
		)
		status = "success" if is_success else "failed"
		failure_class = None if is_success else self._failure_class(
			stdout,
			exit_code,
			timed_out,
			environment_result,
			consistency_checks,
		)
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
		)
		validation_json_path.write_text(json.dumps(result_payload.to_dict(), indent=2))

		if not is_success:
			failure_reason = self._failure_reason(
				stdout,
				stderr,
				exit_code,
				timed_out,
				environment_result,
				consistency_checks,
			)
			raise JasonValidationError(
				f"Stage 6 Jason validation failed: {failure_reason}",
				metadata=result_payload.to_dict(),
			)

		return result_payload

	def _infer_unordered_target_execution_order(
		self,
		*,
		target_literals: Sequence[HTNLiteral],
		method_library: HTNMethodLibrary,
		action_schemas: Sequence[Dict[str, Any]],
		seed_facts: Sequence[str],
		runtime_objects: Sequence[str],
		object_types: Dict[str, str],
		planning_domain: Any,
		output_path: Path,
	) -> Dict[str, List[str]]:
		from stage4_panda_planning.panda_planner import PANDAPlanner

		if len(target_literals) <= 1:
			return {
				"target_ids": [f"t{index}" for index, _ in enumerate(target_literals, start=1)],
				"target_signatures": [literal.to_signature() for literal in target_literals],
			}

		planner = PANDAPlanner(workspace=output_path / "unordered_target_ordering")
		if not planner.toolchain_available():
			return {"target_ids": [], "target_signatures": []}

		schema_lookup = self._action_schema_lookup(action_schemas)
		predicate_name_map = self._runtime_predicate_name_map(
			action_schemas=action_schemas,
			predicate_names=[
				str(getattr(predicate, "name", "")).strip()
				for predicate in getattr(planning_domain, "predicates", ()) or ()
				if str(getattr(predicate, "name", "")).strip()
			],
		)
		runtime_world = {
			atom
			for atom in (self._hddl_fact_to_atom(fact) for fact in seed_facts)
			if atom is not None
		}
		typed_objects = tuple(
			(str(obj), object_types[str(obj)])
			for obj in runtime_objects
			if str(obj) in object_types
		)
		remaining = [
			(f"t{index}", literal, index)
			for index, literal in enumerate(target_literals, start=1)
		]
		plan_cache: Dict[Tuple[Tuple[str, ...], str], Any] = {}
		dead_end_cache: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()

		def plan_for_target(
			current_world: Set[str],
			target_id: str,
			literal: HTNLiteral,
			depth: int,
		) -> Any | None:
			task_name = method_library.task_name_for_literal(literal)
			if not task_name:
				return None

			world_key = tuple(sorted(current_world))
			cache_key = (world_key, target_id)
			if cache_key in plan_cache:
				return plan_cache[cache_key]

			try:
				plan = planner.plan(
					domain=planning_domain,
					method_library=method_library,
					objects=tuple(str(obj) for obj in runtime_objects),
					typed_objects=typed_objects,
					target_literal=literal,
					task_name=task_name,
					transition_name=(
						f"stage6_order_depth_{depth}_{target_id}_"
						f"{self._sanitize_name(literal.predicate)}"
					),
					initial_facts=self._runtime_world_to_hddl_facts(
						current_world,
						predicate_name_map=predicate_name_map,
					),
					allow_empty_plan=False,
				)
			except Exception:
				plan = None

			plan_cache[cache_key] = plan
			return plan

		def search(
			current_world: Set[str],
			remaining_targets: Sequence[Tuple[str, HTNLiteral, int]],
			depth: int,
		) -> Optional[List[Tuple[str, str]]]:
			if not remaining_targets:
				return []

			state_key = (
				tuple(sorted(current_world)),
				tuple(target_id for target_id, _, _ in remaining_targets),
			)
			if state_key in dead_end_cache:
				return None

			candidates: List[Tuple[int, int, str, str, HTNLiteral, Any]] = []
			for target_id, literal, original_index in remaining_targets:
				plan = plan_for_target(current_world, target_id, literal, depth)
				if plan is None:
					continue
				candidates.append(
					(
						original_index,
						len(plan.steps),
						literal.to_signature(),
						target_id,
						literal,
						plan,
					),
				)

			if not candidates:
				dead_end_cache.add(state_key)
				return None

			if depth == 0:
				candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
			else:
				candidates.sort(key=lambda item: (item[1], item[0], item[2], item[3]))
			for _, _, signature, target_id, literal, plan in candidates:
				next_world = self._replay_plan_steps_into_world(
					world=current_world,
					steps=plan.steps,
					schema_lookup=schema_lookup,
				)
				next_remaining = [
					(item_target_id, item_literal, item_original_index)
					for item_target_id, item_literal, item_original_index in remaining_targets
					if item_target_id != target_id
				]
				suffix = search(next_world, next_remaining, depth + 1)
				if suffix is not None:
					return [(target_id, signature), *suffix]

			dead_end_cache.add(state_key)
			return None

		search_result = search(runtime_world, remaining, 0) or []
		ordered_target_ids = [target_id for target_id, _ in search_result]
		ordered_signatures = [signature for _, signature in search_result]

		return {
			"target_ids": ordered_target_ids,
			"target_signatures": ordered_signatures,
		}

	def _reorder_unordered_control_plan_blocks(
		self,
		agentspeak_code: str,
		preferred_target_ids: Sequence[str],
	) -> str:
		start_marker = "/* DFA Control Plans */"
		start_index = agentspeak_code.find(start_marker)
		if start_index == -1:
			return agentspeak_code

		prefix = agentspeak_code[:start_index]
		control_section = agentspeak_code[start_index:]
		blocks = [
			block.strip()
			for block in re.split(r"\n\s*\n", control_section.strip())
			if block.strip()
		]
		if not blocks:
			return agentspeak_code

		header = blocks[0]
		success_blocks: List[str] = []
		clear_blocks: List[str] = []
		fallback_blocks: List[str] = []
		pair_lookup: Dict[str, List[str]] = {}
		original_pair_order: List[str] = []
		pending_run_target_id: Optional[str] = None

		for block in blocks[1:]:
			target_id = self._unordered_control_target_id(block)
			header_line = block.splitlines()[0].strip()
			if header_line.startswith("+!run_dfa : target_seen("):
				success_blocks.append(block)
				continue
			if header_line.startswith("+!clear_blocked_targets"):
				clear_blocks.append(block)
				continue
			if header_line.startswith("+!run_dfa : true <-"):
				fallback_blocks.append(block)
				continue
			if header_line.startswith("+!run_dfa : not target_seen(") and target_id is not None:
				pending_run_target_id = target_id
				pair_lookup.setdefault(target_id, []).append(block)
				if target_id not in original_pair_order:
					original_pair_order.append(target_id)
				continue
			if header_line.startswith("-!dfa_step_") and target_id is not None:
				pair_target_id = target_id if pending_run_target_id is None else pending_run_target_id
				pair_lookup.setdefault(pair_target_id, []).append(block)
				if pair_target_id not in original_pair_order:
					original_pair_order.append(pair_target_id)
				pending_run_target_id = None
				continue

		ordered_target_ids = [
			target_id
			for target_id in preferred_target_ids
			if target_id in pair_lookup
		]
		ordered_target_ids.extend(
			target_id
			for target_id in original_pair_order
			if target_id not in ordered_target_ids
		)

		reordered_blocks: List[str] = [header]
		reordered_blocks.extend(success_blocks)
		reordered_blocks.extend(clear_blocks)
		for target_id in ordered_target_ids:
			reordered_blocks.extend(pair_lookup.get(target_id, ()))
		reordered_blocks.extend(fallback_blocks)
		reordered_section = "\n\n".join(reordered_blocks).rstrip() + "\n"
		return prefix + reordered_section

	@staticmethod
	def _unordered_control_target_id(block: str) -> Optional[str]:
		match = re.search(r"target_seen\((t\d+)\)", block)
		if match:
			return match.group(1)
		return None

	def _action_schema_lookup(
		self,
		action_schemas: Sequence[Dict[str, Any]],
	) -> Dict[str, Dict[str, Any]]:
		schema_lookup: Dict[str, Dict[str, Any]] = {}
		for schema in action_schemas:
			functor = str(schema.get("functor", "")).strip()
			source_name = str(schema.get("source_name", "")).strip()
			if functor:
				schema_lookup.setdefault(functor, schema)
			if source_name:
				schema_lookup.setdefault(source_name, schema)
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

			for effect in schema.get("effects") or []:
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
		"""Return whether Java+Jason requirements are available for Stage 6."""

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
		target_literals: Sequence[HTNLiteral],
		*,
		method_library: HTNMethodLibrary | None = None,
		runtime_objects: Sequence[str] = (),
		object_types: Optional[Dict[str, str]] = None,
		type_parent_map: Optional[Dict[str, Optional[str]]] = None,
		completion_mode: str = "target_literals",
		ordered_query_sequence: bool = True,
		guided_action_path: Sequence[str] = (),
		guided_method_trace: Sequence[Dict[str, Any]] = (),
	) -> str:
		runtime_ready_code = self._inject_runtime_object_beliefs(
			agentspeak_code,
			runtime_objects=runtime_objects,
			object_types=object_types or {},
			type_parent_map=type_parent_map or {},
		)
		environment_ready_code = self._rewrite_primitive_wrappers_for_environment(runtime_ready_code)
		trace_ready_code = self._instrument_method_plans(
			environment_ready_code,
			method_library,
		)
		if guided_action_path:
			return self._build_guided_runner_asl(
				trace_ready_code,
				method_library=method_library,
				ordered_query_sequence=ordered_query_sequence,
				guided_action_path=guided_action_path,
				guided_method_trace=guided_method_trace,
			)
		target_observations = self._extract_transition_target_observations(
			trace_ready_code,
			target_literals,
		)
		observation_ready_code = self._instrument_transition_wrappers_for_target_observations(
			trace_ready_code,
			target_observations,
		)
		target_context = (
			self._observed_target_context_expression(target_observations)
			or self._target_context_expression(target_literals)
		)
		initial_dfa_state = self._extract_initial_dfa_state(trace_ready_code) or "q1"
		max_execution_passes = max(2, len(target_literals) + 1)
		lines = [
			observation_ready_code.rstrip(),
			"",
			*self._render_target_observation_plans(target_observations),
			"",
			*self._render_failure_handlers(
				method_library,
				ordered_query_sequence=ordered_query_sequence,
			),
			"",
			"/* Execution Entry */",
			"!execute.",
			"",
		]
		if completion_mode == "accepting_state":
			lines.append("+!execute : true <-")
			lines.extend(
				self._indent_body(
					[
						'.print("execute start")',
						"!run_dfa",
						"?dfa_state(FINAL_STATE)",
						"?accepting_state(FINAL_STATE)",
						'.print("execute success")',
						".stopMAS",
					],
				),
			)
			lines.append("")
			lines.append("-!execute : true <-")
			lines.extend(self._indent_body(['.print("execute failed")', ".stopMAS"]))
			lines.append("")
			return "\n".join(lines)

		if completion_mode != "target_literals":
			raise JasonValidationError(
				f"Unsupported Stage 6 completion mode: {completion_mode}",
				metadata={"completion_mode": completion_mode},
			)

		lines.append(f"+!verify_targets : {target_context} <-")
		lines.extend(self._indent_body(["true"]))
		lines.append("")
		if not ordered_query_sequence:
			lines.append("+!execute : true <-")
			lines.extend(
				self._indent_body(
					[
						'.print("execute start")',
						"!run_dfa",
						"!verify_targets",
						'.print("execute success")',
						".stopMAS",
					],
				),
			)
			lines.append("")
			lines.append("-!execute : true <-")
			lines.extend(self._indent_body(['.print("execute failed")', ".stopMAS"]))
			lines.append("")
			return "\n".join(lines)

		lines.append("+!reset_execution_state : dfa_state(CURRENT_STATE) <-")
		lines.extend(
			self._indent_body(
				[
					"-dfa_state(CURRENT_STATE)",
					f"+dfa_state({initial_dfa_state})",
				],
			),
		)
		lines.append("")
		lines.append("+!execute : true <-")
		lines.extend(
			self._indent_body(
				[
					'.print("execute start")',
					"!execute_round_1",
				],
			),
		)
		lines.append("")
		for round_index in range(1, max_execution_passes + 1):
			goal_name = f"execute_round_{round_index}"
			next_goal_name = f"execute_round_{round_index + 1}"
			lines.append(f"+!{goal_name} : {target_context} <-")
			lines.extend(
				self._indent_body(
					[
						'.print("execute success")',
						".stopMAS",
					],
				),
			)
			lines.append("")
			body_lines: List[str] = [
				"!run_dfa",
				"?dfa_state(FINAL_STATE)",
				"?accepting_state(FINAL_STATE)",
			]
			if round_index < max_execution_passes:
				body_lines.extend(
					[
						"!reset_execution_state",
						f"!{next_goal_name}",
					],
				)
			else:
				body_lines.append(f"!{next_goal_name}")
			lines.append(f"+!{goal_name} : true <-")
			lines.extend(self._indent_body(body_lines))
			lines.append("")

		final_goal_name = f"execute_round_{max_execution_passes + 1}"
		lines.append(f"+!{final_goal_name} : {target_context} <-")
		lines.extend(
			self._indent_body(
				[
					'.print("execute success")',
					".stopMAS",
				],
			),
		)
		lines.append("")
		lines.append(f"+!{final_goal_name} : true <-")
		lines.extend(self._indent_body(['.print("execute failed")', ".stopMAS"]))
		lines.append("")
		lines.append("-!execute : true <-")
		lines.extend(self._indent_body(['.print("execute failed")', ".stopMAS"]))
		lines.append("")
		return "\n".join(lines)

	def _build_guided_runner_asl(
		self,
		agentspeak_code: str,
		*,
		method_library: HTNMethodLibrary | None,
		ordered_query_sequence: bool,
		guided_action_path: Sequence[str],
		guided_method_trace: Sequence[Dict[str, Any]],
	) -> str:
		lines = [
			agentspeak_code.rstrip(),
			"",
			*self._render_failure_handlers(
				method_library,
				ordered_query_sequence=ordered_query_sequence,
			),
			"",
			"/* Execution Entry */",
			"!execute.",
			"",
			"+!execute : true <-",
		]
		body_lines: List[str] = ['.print("execute start")']
		body_lines.extend(
			self._render_guided_method_trace_statement(entry)
			for entry in guided_method_trace
		)
		body_lines.extend(
			f"!{action_step}"
			for action_step in guided_action_path
		)
		body_lines.extend(
			[
				'.print("execute success")',
				".stopMAS",
			],
		)
		lines.extend(self._indent_body(body_lines))
		lines.append("")
		lines.append("-!execute : true <-")
		lines.extend(self._indent_body(['.print("execute failed")', ".stopMAS"]))
		lines.append("")
		return "\n".join(lines)

	def _render_guided_method_trace_statement(self, entry: Dict[str, Any]) -> str:
		method_name = self._asl_atom_or_string(str(entry.get("method_name", "") or ""))
		task_args = tuple(
			self._runtime_atom_term(str(arg))
			for arg in (entry.get("task_args") or ())
		)
		trace_term = self._call("trace_method", (method_name, *task_args))
		return f'.print("runtime trace method ", {trace_term})'

	def _inject_runtime_object_beliefs(
		self,
		agentspeak_code: str,
		*,
		runtime_objects: Sequence[str],
		object_types: Dict[str, str],
		type_parent_map: Dict[str, Optional[str]],
	) -> str:
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

	def _extract_transition_target_observations(
		self,
		agentspeak_code: str,
		target_literals: Sequence[HTNLiteral],
	) -> List[Dict[str, Any]]:
		if not target_literals:
			return []
		targets_by_signature = {
			literal.to_signature(): literal
			for literal in target_literals
		}
		observations: List[Dict[str, Any]] = []
		seen_transitions: set[str] = set()
		for match in re.finditer(
			r'dfa_edge_label\(([^,]+),\s*"([^"]*)"\)\.',
			agentspeak_code,
		):
			transition_name = match.group(1).strip()
			label = match.group(2).strip()
			target_literal = targets_by_signature.get(label)
			if target_literal is None or transition_name in seen_transitions:
				continue
			seen_transitions.add(transition_name)
			observations.append(
				{
					"transition_name": transition_name,
					"target_id": f"t{len(observations) + 1}",
					"literal": target_literal,
				},
			)
		return observations

	def _instrument_transition_wrappers_for_target_observations(
		self,
		agentspeak_code: str,
		target_observations: Sequence[Dict[str, Any]],
	) -> str:
		if not target_observations:
			return agentspeak_code

		observation_by_transition = {
			str(item["transition_name"]): str(item["target_id"])
			for item in target_observations
		}
		start_marker = "/* DFA Transition Wrappers */"
		end_marker = "/* DFA Control Plans */"
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

		instrumented_chunks: List[str] = []
		for chunk in chunks:
			head_line = chunk[0]
			body_lines = list(chunk[1:])
			match = re.match(r"^\s*\+!([^\s(:]+)", head_line)
			if match is None or not body_lines:
				instrumented_chunks.append("\n".join(chunk))
				continue
			transition_name = match.group(1).strip()
			target_id = observation_by_transition.get(transition_name)
			if target_id is None:
				instrumented_chunks.append("\n".join(chunk))
				continue
			insert_at = 1
			for index, line in enumerate(body_lines, start=1):
				stripped = line.strip()
				if not stripped.startswith("!"):
					continue
				if stripped.startswith("!mark_target_"):
					continue
				insert_at = index
			instrumented_body = list(body_lines)
			instrumented_body.insert(insert_at, f"\t!mark_target_{target_id};")
			normalised_body = [
				line.strip().rstrip(";.")
				for line in instrumented_body
				if line.strip()
			]
			instrumented_chunks.append(
				"\n".join([head_line, *self._indent_body(normalised_body)]),
			)

		instrumented_section = "\n\n".join([header, *instrumented_chunks]).rstrip() + "\n\n"
		return f"{prefix}{instrumented_section}{suffix}"

	def _render_target_observation_plans(
		self,
		target_observations: Sequence[Dict[str, Any]],
	) -> List[str]:
		if not target_observations:
			return []

		lines = ["/* Target Observation Plans */"]
		for observation in target_observations:
			target_id = str(observation["target_id"])
			literal = observation["literal"]
			context = self._literal_to_context_expression(literal)
			protection_atom = self._target_protection_atom(literal)
			lines.append(f"+!mark_target_{target_id} : target_seen({target_id}) <-")
			lines.extend(self._indent_body(["true"]))
			lines.append("")
			if protection_atom:
				lines.append(
					f"+!mark_target_{target_id} : {context} & {protection_atom} <-"
				)
				lines.extend(self._indent_body([f"+target_seen({target_id})"]))
				lines.append("")
				lines.append(
					f"+!mark_target_{target_id} : {context} & not {protection_atom} <-"
				)
				lines.extend(
					self._indent_body(
						[
							f"+target_seen({target_id})",
							f"+{protection_atom}",
						],
					),
				)
			else:
				lines.append(f"+!mark_target_{target_id} : {context} <-")
				lines.extend(self._indent_body([f"+target_seen({target_id})"]))
			lines.append("")
		return lines

	def _observed_target_context_expression(
		self,
		target_observations: Sequence[Dict[str, Any]],
	) -> str:
		if not target_observations:
			return ""
		return " & ".join(
			f"target_seen({item['target_id']})"
			for item in target_observations
		)

	def _extract_initial_dfa_state(self, agentspeak_code: str) -> Optional[str]:
		match = re.search(r"^\s*dfa_state\(([^)]+)\)\.\s*$", agentspeak_code, re.MULTILINE)
		if match is None:
			return None
		return match.group(1).strip()

	def _extract_action_path(self, stdout: str) -> List[str]:
		pattern = re.compile(r"^runtime env action success (.+?)\s*$")
		return [
			match.group(1).strip()
			for line in stdout.splitlines()
			if (match := pattern.match(line.strip())) is not None
		]

	def _extract_method_trace(self, stdout: str) -> List[Dict[str, Any]]:
		pattern = re.compile(r"runtime trace method\s+trace_method\((.*)\)\s*$")
		trace: List[Dict[str, Any]] = []
		for raw_line in stdout.splitlines():
			line = raw_line.strip()
			match = pattern.search(line)
			if match is None:
				continue
			payload = match.group(1).strip()
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

	def _extract_failed_goals(self, stdout: str) -> List[str]:
		pattern = re.compile(r"runtime goal failed\s+fail_goal\((.*)\)\s*$")
		failed: List[str] = []
		for raw_line in stdout.splitlines():
			line = raw_line.strip()
			match = pattern.search(line)
			if match is None:
				continue
			payload = match.group(1).strip()
			if payload:
				failed.append(payload)
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
	def _strip_quoted_atom(text: str) -> str:
		token = str(text).strip()
		if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
			return token[1:-1]
		return token

	@staticmethod
	def _failure_handler_args(parameters: Sequence[str]) -> Tuple[str, ...]:
		used_counts: Dict[str, int] = {}
		rendered_args: List[str] = []
		for index, parameter in enumerate(parameters, start=1):
			token = re.sub(r"^[?]+", "", str(parameter).strip())
			token = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").upper()
			if not token:
				token = f"ARG{index}"
			if not token[0].isalpha():
				token = f"ARG_{token}"
			count = used_counts.get(token, 0) + 1
			used_counts[token] = count
			rendered_args.append(token if count == 1 else f"{token}{count}")
		return tuple(rendered_args)

	def _render_failure_handlers(
		self,
		method_library: HTNMethodLibrary | None,
		*,
		ordered_query_sequence: bool = True,
	) -> List[str]:
		lines = ["/* Failure Handlers */"]
		lines.append("-!run_dfa : true <-")
		lines.extend(self._indent_body(['.print("runtime goal failed ", fail_goal(run_dfa))', ".fail"]))
		lines.append("")
		lines.append("-!verify_targets : true <-")
		lines.extend(
			self._indent_body(['.print("runtime goal failed ", fail_goal(verify_targets))', ".fail"]),
		)
		lines.append("")
		if method_library is None or not ordered_query_sequence:
			return lines
		for task in method_library.compound_tasks:
			handler_args = self._failure_handler_args(task.parameters)
			trigger = self._call(self._sanitize_name(task.name), handler_args)
			fail_term = self._call("fail_goal", (self._asl_atom_or_string(task.name), *handler_args))
			lines.append(f"-!{trigger} : true <-")
			lines.extend(self._indent_body([f'.print("runtime goal failed ", {fail_term})', ".fail"]))
			lines.append("")
		return lines

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
			first_statement = body_lines[0].strip().rstrip(";.")
			if not first_statement:
				rewritten_chunks.append("\n".join(chunk))
				continue
			rewritten_chunks.append("\n".join([head_line, f"\t{first_statement}."]))

		rewritten_section = "\n\n".join([header, *rewritten_chunks]).rstrip() + "\n\n"
		return f"{prefix}{rewritten_section}{suffix}"

	def _instrument_method_plans(
		self,
		agentspeak_code: str,
		method_library: HTNMethodLibrary | None,
	) -> str:
		if method_library is None or not method_library.methods:
			return agentspeak_code
		if "runtime trace method" in agentspeak_code:
			return agentspeak_code

		start_marker = "/* HTN Method Plans */"
		end_marker = "/* DFA Transition Wrappers */"
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

		if len(chunks) != len(method_library.methods):
			return agentspeak_code

		instrumented_chunks: List[str] = []
		for method, chunk in zip(method_library.methods, chunks):
			head_line = chunk[0]
			body_lines = chunk[1:]
			trace_line = self._render_method_trace_statement(method, head_line)
			instrumented_chunks.append("\n".join([head_line, trace_line, *body_lines]))

		instrumented_section = "\n\n".join([header, *instrumented_chunks]).rstrip() + "\n\n"
		return f"{prefix}{instrumented_section}{suffix}"

	def _render_method_trace_statement(self, method: Any, head_line: str) -> str:
		trigger_args = self._extract_trigger_args(head_line)
		trace_term = self._call(
			"trace_method",
			(self._asl_atom_or_string(method.method_name), *trigger_args),
		)
		return f'\t.print("runtime trace method ", {trace_term});'

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

	def _build_environment_java_source(
		self,
		*,
		action_schemas: Sequence[Dict[str, Any]],
		seed_facts: Sequence[str],
		target_literals: Sequence[HTNLiteral],
	) -> str:
		_ = target_literals
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
			effects = list(schema.get("effects") or [])
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

	private void seedInitialFacts() {{
		world.clear();
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
			String grounded = ground(pattern.predicate, pattern.args, bindings);
			if (pattern.positive) {{
				world.add(grounded);
			}} else {{
				world.remove(grounded);
			}}
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
		compile_cmd = [
			javac_bin,
			"-cp",
			str(jason_jar),
			env_java_path.name,
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
			"Stage 6 environment Java compilation failed.",
			metadata={
				"java_bin": java_bin,
				"javac_bin": javac_bin,
				"environment_java": str(env_java_path),
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

	def _target_context_expression(self, target_literals: Sequence[HTNLiteral]) -> str:
		rendered_literals = [
			self._literal_to_context_expression(literal)
			for literal in target_literals
		]
		rendered_literals = [item for item in rendered_literals if item]
		if not rendered_literals:
			return "true"
		return " & ".join(rendered_literals)

	@classmethod
	def _literal_to_context_expression(cls, literal: HTNLiteral) -> str:
		if literal.is_equality and len(literal.args) == 2:
			operator = "==" if literal.is_positive else "\\=="
			return (
				f"{cls._runtime_atom_term(literal.args[0])} {operator} "
				f"{cls._runtime_atom_term(literal.args[1])}"
			)
		predicate = cls._sanitize_name(literal.predicate)
		atom = (
			f"{predicate}({', '.join(cls._runtime_atom_term(arg) for arg in literal.args)})"
			if literal.args
			else predicate
		)
		if literal.is_positive:
			return atom
		return f"not {atom}"

	@classmethod
	def _target_protection_atom(cls, literal: HTNLiteral) -> str:
		if literal.is_equality:
			return ""
		family = "protected_target" if literal.is_positive else "protected_absence"
		predicate = cls._sanitize_name(literal.predicate)
		args = tuple(cls._runtime_atom_term(arg) for arg in literal.args)
		return cls._call(f"{family}_{predicate}", args)

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
			"Stage 6 log configuration file is missing.",
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
				"Stage 6 Jason source directory is missing gradlew.",
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
				"No supported Java runtime found for Stage 6 (requires Java 17-23).",
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
			"No javac binary found for Stage 6 environment compilation.",
			metadata={"java_bin": java_bin, "candidates": candidates},
		)

	def _discover_java_candidates(self) -> List[str]:
		candidates: List[str] = []
		self._append_candidate(candidates, os.getenv("STAGE6_JAVA_BIN"))

		stage6_java_home = os.getenv("STAGE6_JAVA_HOME")
		if stage6_java_home:
			self._append_candidate(candidates, str(Path(stage6_java_home) / "bin" / "java"))

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
		consistency_checks: Dict[str, Any],
	) -> bool:
		if timed_out:
			return False
		if exit_code is None or exit_code != 0:
			return False
		if self.success_marker not in stdout:
			return False
		if self.failure_marker in stdout:
			return False
		if not environment_result.success:
			return False
		if not bool(consistency_checks.get("success", True)):
			return False
		return True

	def _failure_reason(
		self,
		stdout: str,
		stderr: str,
		exit_code: Optional[int],
		timed_out: bool,
		environment_result: EnvironmentAdapterResult,
		consistency_checks: Dict[str, Any],
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
		if not bool(consistency_checks.get("success", True)):
			for check_name in ("action_path_schema_replay", "method_trace_reconstruction"):
				check_payload = consistency_checks.get(check_name) or {}
				if check_payload.get("passed") is False:
					return (
						f"{check_name} failed: "
						f"{check_payload.get('message') or check_payload.get('failure_class') or 'unknown'}"
					)
		return "unknown validation error"

	def _failure_class(
		self,
		stdout: str,
		exit_code: Optional[int],
		timed_out: bool,
		environment_result: EnvironmentAdapterResult,
		consistency_checks: Dict[str, Any],
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
		for check_name in ("action_path_schema_replay", "method_trace_reconstruction"):
			check_payload = consistency_checks.get(check_name) or {}
			if check_payload.get("passed") is False:
				return str(check_payload.get("failure_class") or f"{check_name}_failed")
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
		skip_method_trace_reconstruction: bool = False,
	) -> Dict[str, Any]:
		action_replay = self._replay_action_path_against_schemas(
			action_path=action_path,
			action_schemas=action_schemas,
			seed_facts=seed_facts,
		)
		if skip_method_trace_reconstruction:
			method_trace_check = {
				"passed": None,
				"failure_class": None,
				"message": "skipped: authoritative guided hierarchical plan available",
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
		schema_lookup: Dict[str, Dict[str, Any]] = {}
		for schema in action_schemas:
			functor = str(schema.get("functor", "")).strip()
			source_name = str(schema.get("source_name", "")).strip()
			if functor:
				schema_lookup.setdefault(functor, schema)
			if source_name:
				schema_lookup.setdefault(source_name, schema)

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

			for effect in schema.get("effects") or []:
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

		from utils.ipc_plan_verifier import IPCPlanVerifier

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

	@classmethod
	def _hddl_fact_to_atom(cls, fact: str) -> Optional[str]:
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
		functor = cls._sanitize_name(predicate)
		if not args:
			return functor
		rendered_args = [cls._runtime_atom_term(arg) for arg in args]
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
