"""
True Jadex runtime runner.

This backend does not implement plan search in Python. It generates a small
Jadex BDIV3 Java agent from the AgentSpeak(L) plan library, starts a real Jadex
platform, lets Jadex select/retry plans, and then reads the runtime trace that
the generated agent writes to disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from method_library.synthesis.schema import HTNMethodLibrary
from plan_library.models import AgentSpeakBodyStep, AgentSpeakPlan, PlanLibrary


class JadexValidationError(RuntimeError):
	"""Raised when the real Jadex runtime fails."""

	def __init__(self, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
		super().__init__(message)
		self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class JadexValidationResult:
	"""Structured result for a true Jadex runtime execution."""

	status: str
	backend: str
	java_path: Optional[str]
	java_version: Optional[int]
	exit_code: Optional[int]
	timed_out: bool
	action_path: Tuple[str, ...]
	method_trace: Tuple[Dict[str, Any], ...]
	failed_goals: Tuple[str, ...]
	failure_class: Optional[str]
	consistency_checks: Dict[str, Any]
	artifacts: Dict[str, Any]
	timing_profile: Dict[str, Any]

	def to_compact_dict(self) -> Dict[str, Any]:
		return {
			"status": self.status,
			"backend": self.backend,
			"java_path": self.java_path,
			"java_version": self.java_version,
			"exit_code": self.exit_code,
			"timed_out": self.timed_out,
			"action_path_count": len(self.action_path),
			"method_trace_count": len(self.method_trace),
			"failed_goals": list(self.failed_goals),
			"failure_class": self.failure_class,
			"consistency_checks": dict(self.consistency_checks),
			"artifacts": dict(self.artifacts),
			"timing_profile": dict(self.timing_profile),
		}


class JadexBDIRunner:
	"""Generate and run a real Jadex BDIV3 agent for one evaluation query."""

	backend_name = "JadexBDIV3"
	jadex_version = "4.0.267"
	jna_version = "5.18.1"
	min_java_major = 17
	default_java_home = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
	stdout_limit_chars = 80_000

	def __init__(self, *, timeout_seconds: int = 1800) -> None:
		self.timeout_seconds = timeout_seconds

	def validate(
		self,
		*,
		action_schemas: Sequence[Dict[str, Any]],
		method_library: HTNMethodLibrary | None = None,
		plan_library: PlanLibrary | None = None,
		seed_facts: Sequence[str] = (),
		goal_facts: Sequence[str] = (),
		runtime_objects: Sequence[str] = (),
		object_types: Optional[Dict[str, str]] = None,
		type_parent_map: Optional[Dict[str, Optional[str]]] = None,
		query_goals: Sequence[Any] = (),
		output_dir: str | Path,
		**_: Any,
	) -> JadexValidationResult:
		"""Run the generated plan library in a real Jadex runtime."""

		del method_library
		start = time.perf_counter()
		output_path = Path(output_dir).resolve()
		output_path.mkdir(parents=True, exist_ok=True)
		if plan_library is None or not plan_library.plans:
			raise JadexValidationError(
				"Jadex runtime requires a non-empty plan library.",
				metadata={"failure_class": "missing_plan_library"},
			)

		java_home = self._resolve_java_home()
		java_bin = Path(java_home) / "bin" / "java" if java_home else shutil.which("java")
		mvn_bin = shutil.which("mvn")
		if java_bin is None or not Path(java_bin).exists():
			raise JadexValidationError(
				"Java runtime not found for Jadex.",
				metadata={"failure_class": "jadex_java_missing"},
			)
		if mvn_bin is None:
			raise JadexValidationError(
				"Maven not found for Jadex bridge compilation.",
				metadata={"failure_class": "jadex_maven_missing"},
			)
		java_major = self._java_major(str(java_bin))
		if java_major is not None and java_major < self.min_java_major:
			raise JadexValidationError(
				f"Jadex bridge requires Java {self.min_java_major}+; found {java_major}.",
				metadata={"failure_class": "jadex_java_too_old"},
			)

		project_dir = output_path / "jadex_project"
		src_dir = project_dir / "src" / "main" / "java" / "pipeline"
		src_dir.mkdir(parents=True, exist_ok=True)
		payload = self._build_payload(
			action_schemas=action_schemas,
			plan_library=plan_library,
			seed_facts=seed_facts,
			goal_facts=goal_facts,
			runtime_objects=runtime_objects,
			object_types=dict(object_types or {}),
			type_parent_map=dict(type_parent_map or {}),
			query_goals=query_goals,
		)
		payload_path = output_path / "jadex_payload.json"
		payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
		(project_dir / "pom.xml").write_text(self._render_pom(), encoding="utf-8")
		(src_dir / "PipelineBDI.java").write_text(
			self._render_agent_source(payload),
			encoding="utf-8",
		)
		(src_dir / "PipelineMain.java").write_text(self._render_main_source(), encoding="utf-8")
		(src_dir / "PipelineEnhanceMain.java").write_text(
			self._render_enhance_main_source(),
			encoding="utf-8",
		)

		stdout_path = output_path / "jadex_stdout.txt"
		stderr_path = output_path / "jadex_stderr.txt"
		action_path_path = output_path / "action_path.txt"
		method_trace_path = output_path / "method_trace.json"
		final_world_path = output_path / "final_world_facts.txt"
		validation_path = output_path / "jadex_validation.json"
		cp_path = project_dir / "classpath.txt"

		env = dict(os.environ)
		if java_home:
			env["JAVA_HOME"] = java_home
			env["PATH"] = f"{Path(java_home) / 'bin'}:{env.get('PATH', '')}"

		compile_cmd = [
			mvn_bin,
			"-q",
			"-f",
			str(project_dir / "pom.xml"),
			"clean",
			"compile",
			"dependency:build-classpath",
			f"-Dmdep.outputFile={cp_path}",
		]
		enhance_cmd = [
			str(java_bin),
			"--add-opens",
			"java.base/java.lang=ALL-UNNAMED",
			"-cp",
			f"{project_dir / 'target' / 'classes'}:{cp_path.read_text(encoding='utf-8') if cp_path.exists() else ''}",
			"pipeline.PipelineEnhanceMain",
			str(project_dir / "target" / "classes"),
		]
		run_cmd = [
			str(java_bin),
			"--add-opens",
			"java.base/java.lang=ALL-UNNAMED",
			"-cp",
			f"{project_dir / 'target' / 'classes'}:{cp_path.read_text(encoding='utf-8') if cp_path.exists() else ''}",
			f"-Dpipeline.output.dir={output_path}",
			f"-Dpipeline.payload.path={payload_path}",
			"-Dpipeline.protect.goal.facts=true",
			"-Dpipeline.query.order=",
			"pipeline.PipelineMain",
		]
		protect_goal_flag_index = 7
		query_order_flag_index = 8

		try:
			compile_proc = subprocess.run(
				compile_cmd,
				cwd=project_dir,
				env=env,
				text=True,
				capture_output=True,
				timeout=min(300, max(60, self.timeout_seconds)),
				check=False,
			)
			if compile_proc.returncode != 0:
				self._write_text_artifact(stdout_path, compile_proc.stdout)
				self._write_text_artifact(stderr_path, compile_proc.stderr)
				metadata = self._failure_metadata(
					output_path=output_path,
					validation_path=validation_path,
					failure_class="jadex_bridge_compile_failed",
					exit_code=compile_proc.returncode,
					timed_out=False,
					stdout=compile_proc.stdout,
					stderr=compile_proc.stderr,
					timing_profile={"total_seconds": time.perf_counter() - start},
				)
				raise JadexValidationError("Jadex bridge compilation failed.", metadata=metadata)

			classpath = (
				f"{project_dir / 'target' / 'classes'}:"
				f"{cp_path.read_text(encoding='utf-8')}"
			)
			enhance_cmd[4] = classpath
			enhance_proc = subprocess.run(
				enhance_cmd,
				cwd=project_dir,
				env=env,
				text=True,
				capture_output=True,
				timeout=min(300, max(60, self.timeout_seconds)),
				check=False,
			)
			if enhance_proc.returncode != 0:
				self._write_text_artifact(stdout_path, enhance_proc.stdout)
				self._write_text_artifact(stderr_path, enhance_proc.stderr)
				metadata = self._failure_metadata(
					output_path=output_path,
					validation_path=validation_path,
					failure_class="jadex_bridge_enhancement_failed",
					exit_code=enhance_proc.returncode,
					timed_out=False,
					stdout=enhance_proc.stdout,
					stderr=enhance_proc.stderr,
					timing_profile={"total_seconds": time.perf_counter() - start},
				)
				raise JadexValidationError("Jadex BDI bytecode enhancement failed.", metadata=metadata)

			run_cmd[4] = (
				f"{project_dir / 'target' / 'classes'}:"
				f"{cp_path.read_text(encoding='utf-8')}"
			)
			proc: subprocess.CompletedProcess[str] | None = None
			fallback_used = False
			attempt_records: List[Dict[str, Any]] = []
			for attempt_index, order in enumerate(self._query_order_candidates(payload)):
				attempt_cmd = list(run_cmd)
				attempt_cmd[query_order_flag_index] = (
					"-Dpipeline.query.order=" + ",".join(str(item) for item in order)
				)
				attempt_proc = self._run_streamed_process(
					attempt_cmd,
					cwd=project_dir,
					env=env,
					stdout_path=stdout_path,
					stderr_path=stderr_path,
					timeout_seconds=max(1, self.timeout_seconds),
				)
				attempt_fallback_used = False
				if (
					(attempt_proc.returncode != 0 or "execute success" not in attempt_proc.stdout)
					and "protected-goal delete rejected" in attempt_proc.stdout
				):
					fallback_cmd = list(attempt_cmd)
					fallback_cmd[protect_goal_flag_index] = "-Dpipeline.protect.goal.facts=false"
					fallback_proc = self._run_streamed_process(
						fallback_cmd,
						cwd=project_dir,
						env=env,
						stdout_path=stdout_path,
						stderr_path=stderr_path,
						timeout_seconds=max(1, self.timeout_seconds),
					)
					attempt_proc = subprocess.CompletedProcess(
						args=fallback_proc.args,
						returncode=fallback_proc.returncode,
						stdout=(
							attempt_proc.stdout
							+ "\n--- retry without protected final fact guard ---\n"
							+ fallback_proc.stdout
						),
						stderr=(
							attempt_proc.stderr
							+ "\n--- retry without protected final fact guard ---\n"
							+ fallback_proc.stderr
						),
					)
					attempt_fallback_used = True
				attempt_success = (
					attempt_proc.returncode == 0
					and "execute success" in attempt_proc.stdout
				)
				attempt_records.append(
					{
						"attempt": attempt_index,
						"query_order": list(order),
						"success": attempt_success,
						"exit_code": attempt_proc.returncode,
						"protected_goal_fallback_used": attempt_fallback_used,
						"stdout_chars": len(attempt_proc.stdout or ""),
						"stderr_chars": len(attempt_proc.stderr or ""),
					},
				)
				proc = attempt_proc
				fallback_used = attempt_fallback_used
				if attempt_success:
					break
			(output_path / "jadex_attempts.json").write_text(
				json.dumps(attempt_records, indent=2),
				encoding="utf-8",
			)
			timed_out = False
		except subprocess.TimeoutExpired as exc:
			proc = None
			timed_out = True
			fallback_used = False
			stdout = self._read_text_tail(stdout_path) or str(exc.stdout or "")
			stderr = self._read_text_tail(stderr_path) or str(exc.stderr or "")
			self._write_text_artifact(stdout_path, stdout)
			self._write_text_artifact(stderr_path, stderr)
			metadata = self._failure_metadata(
				output_path=output_path,
				validation_path=validation_path,
				failure_class="jadex_runtime_timeout",
				exit_code=None,
				timed_out=True,
				stdout=stdout,
				stderr=stderr,
				timing_profile={"total_seconds": time.perf_counter() - start},
			)
			raise JadexValidationError("Jadex runtime timed out.", metadata=metadata) from exc

		stdout = proc.stdout if proc is not None else ""
		stderr = proc.stderr if proc is not None else ""
		self._write_text_artifact(stdout_path, stdout)
		self._write_text_artifact(stderr_path, stderr)
		action_path = self._read_lines(action_path_path)
		method_trace = self._read_method_trace(method_trace_path)
		failed_goals = tuple(
			line.split("runtime goal failed ", 1)[1].strip()
			for line in stdout.splitlines()
			if "runtime goal failed " in line
		)
		final_world_facts = self._read_lines(final_world_path)
		timing_profile = {"total_seconds": time.perf_counter() - start}
		artifacts = self._artifact_payload(
			output_path=output_path,
			project_dir=project_dir,
			payload_path=payload_path,
			stdout_path=stdout_path,
			stderr_path=stderr_path,
			action_path_path=action_path_path,
			method_trace_path=method_trace_path,
			final_world_path=final_world_path,
			validation_path=validation_path,
			stdout=stdout,
			stderr=stderr,
		)
		if proc is None or proc.returncode != 0 or "execute success" not in stdout:
			failure_class = "jadex_runtime_failure_marker"
			metadata = {
				"status": "failed",
				"backend": self.backend_name,
				"java_path": str(java_bin),
				"java_version": java_major,
				"exit_code": proc.returncode if proc is not None else None,
				"timed_out": timed_out,
				"action_path_count": len(action_path),
				"method_trace_count": len(method_trace),
				"failed_goals": list(failed_goals),
				"failure_class": failure_class,
				"consistency_checks": {
					"success": False,
					"runtime_semantics": "jadex_bdiv3_framework",
					"protected_goal_fallback_used": fallback_used,
					"action_path_schema_replay": {"world_facts": list(final_world_facts)},
				},
				"artifacts": artifacts,
				"timing_profile": timing_profile,
			}
			validation_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
			raise JadexValidationError("Jadex runtime execution failed.", metadata=metadata)

		result = JadexValidationResult(
			status="success",
			backend=self.backend_name,
			java_path=str(java_bin),
			java_version=java_major,
			exit_code=proc.returncode,
			timed_out=False,
			action_path=tuple(action_path),
			method_trace=tuple(method_trace),
			failed_goals=failed_goals,
			failure_class=None,
			consistency_checks={
				"success": True,
				"runtime_semantics": "jadex_bdiv3_framework",
				"protected_goal_fallback_used": fallback_used,
				"action_path_schema_replay": {"world_facts": list(final_world_facts)},
			},
			artifacts=artifacts,
			timing_profile=timing_profile,
		)
		validation_path.write_text(json.dumps(result.to_compact_dict(), indent=2), encoding="utf-8")
		return result

	def _build_payload(
		self,
		*,
		action_schemas: Sequence[Dict[str, Any]],
		plan_library: PlanLibrary,
		seed_facts: Sequence[str],
		goal_facts: Sequence[str],
		runtime_objects: Sequence[str],
		object_types: Dict[str, str],
		type_parent_map: Dict[str, Optional[str]],
		query_goals: Sequence[Any],
	) -> Dict[str, Any]:
		actions = [self._normalise_action_schema(schema) for schema in action_schemas]
		plans = [self._normalise_plan(plan) for plan in plan_library.plans]
		return {
			"seed_facts": [
				atom
				for fact in seed_facts
				if (atom := self._hddl_fact_to_atom(fact)) is not None
			],
			"goal_facts": [
				atom
				for fact in goal_facts
				if (atom := self._hddl_fact_to_atom(fact)) is not None
			],
			"runtime_objects": [str(item) for item in runtime_objects],
			"object_types": {str(key): str(value) for key, value in object_types.items()},
			"type_parent_map": {
				str(key): (None if value is None else str(value))
				for key, value in type_parent_map.items()
			},
			"actions": actions,
			"plans": self._order_plans_for_jadex_runtime(plans, actions),
			"query_goals": [self._normalise_query_goal(goal) for goal in query_goals],
		}

	def _order_plans_for_jadex_runtime(
		self,
		plans: Sequence[Dict[str, Any]],
		actions: Sequence[Dict[str, Any]],
	) -> List[Dict[str, Any]]:
		action_by_name: Dict[str, Dict[str, Any]] = {}
		for action in actions:
			action_by_name[str(action["functor"])] = action
			action_by_name[self._normalise_name(str(action["source_name"]))] = action
		return sorted(
			plans,
			key=lambda plan: self._plan_runtime_preference_key(plan, action_by_name),
		)

	def _plan_runtime_preference_key(
		self,
		plan: Dict[str, Any],
		action_by_name: Dict[str, Dict[str, Any]],
	) -> Tuple[int, int, int, str]:
		destructive_effects = 0
		action_steps = 0
		body = list(plan.get("body") or [])
		for step in body:
			if step.get("kind") != "action":
				continue
			action_steps += 1
			action = action_by_name.get(str(step.get("symbol") or ""))
			if not action:
				continue
			destructive_effects += sum(
				1
				for effect in (action.get("effects") or ())
				if not bool(effect.get("positive", True))
			)
		return (
			destructive_effects,
			len(body),
			action_steps,
			str(plan.get("plan_name") or ""),
		)

	def _query_order_candidates(self, payload: Dict[str, Any]) -> List[Tuple[int, ...]]:
		query_goals = list(payload.get("query_goals") or [])
		goal_count = len(query_goals)
		if goal_count <= 1:
			return [tuple(range(goal_count))]
		original = tuple(range(goal_count))
		if (
			goal_count > 4
			or len(payload.get("seed_facts") or []) > 250
			or len(payload.get("plans") or []) > 80
			or len(payload.get("object_types") or {}) > 120
		):
			return [original]
		mrv = tuple(
			sorted(
				range(goal_count),
				key=lambda index: (
					self._static_goal_candidate_count(payload, query_goals[index]),
					index,
				),
			),
		)
		candidates: List[Tuple[int, ...]] = []
		for candidate in (
			original,
			mrv,
			tuple(reversed(mrv)),
			tuple(reversed(original)),
		):
			self._append_unique_order(candidates, candidate)
		for index in mrv:
			self._append_unique_order(
				candidates,
				(index,) + tuple(item for item in mrv if item != index),
			)
		for offset in range(goal_count):
			rotation = mrv[offset:] + mrv[:offset]
			self._append_unique_order(candidates, rotation)
		if goal_count > 20:
			max_attempts = 2
		elif goal_count > 10:
			max_attempts = 4
		else:
			max_attempts = 16
		return candidates[: max(1, min(max_attempts, goal_count * 2 + 4))]

	@staticmethod
	def _append_unique_order(
		candidates: List[Tuple[int, ...]],
		candidate: Tuple[int, ...],
	) -> None:
		if candidate not in candidates:
			candidates.append(candidate)

	def _static_goal_candidate_count(
		self,
		payload: Dict[str, Any],
		goal: Dict[str, Any],
	) -> int:
		count = 0
		for plan in payload.get("plans") or []:
			if str(plan.get("trigger") or "") != str(goal.get("task_name") or ""):
				continue
			bindings: Dict[str, str] = {}
			trigger_args = list(plan.get("trigger_arguments") or [])
			goal_args = list(goal.get("args") or [])
			if len(trigger_args) != len(goal_args):
				continue
			if not all(
				self._py_unify(pattern, value, bindings, payload)
				for pattern, value in zip(trigger_args, goal_args)
			):
				continue
			count += len(
				self._py_satisfy(
					list(plan.get("context") or []),
					0,
					bindings,
					payload,
					limit=100 - count,
				),
			)
			if count >= 100:
				return count
		return count

	def _py_satisfy(
		self,
		literals: Sequence[Dict[str, Any]],
		index: int,
		bindings: Dict[str, str],
		payload: Dict[str, Any],
		*,
		limit: int,
	) -> List[Dict[str, str]]:
		if limit <= 0:
			return []
		if index >= len(literals):
			return [dict(bindings)]
		output: List[Dict[str, str]] = []
		for next_bindings in self._py_satisfy_one(literals[index], bindings, payload):
			output.extend(
				self._py_satisfy(
					literals,
					index + 1,
					next_bindings,
					payload,
					limit=limit - len(output),
				),
			)
			if len(output) >= limit:
				break
		return output

	def _py_satisfy_one(
		self,
		literal: Dict[str, Any],
		bindings: Dict[str, str],
		payload: Dict[str, Any],
	) -> List[Dict[str, str]]:
		predicate = str(literal.get("predicate") or "")
		args = [str(arg) for arg in (literal.get("args") or [])]
		positive = bool(literal.get("positive", True))
		if predicate == "=" and len(args) == 2:
			next_bindings = dict(bindings)
			if self._py_unify(args[0], args[1], next_bindings, payload) == positive:
				return [next_bindings]
			return []
		if predicate == "object_type" and len(args) == 2:
			value = self._py_resolve(args[0], bindings)
			expected_type = self._normalise_name(args[1])
			if self._py_is_variable(value, payload):
				output = []
				for object_name, actual_type in (payload.get("object_types") or {}).items():
					if self._py_is_subtype(str(actual_type), expected_type, payload):
						next_bindings = dict(bindings)
						if self._py_unify(args[0], str(object_name), next_bindings, payload):
							output.append(next_bindings)
				return output
			if self._py_is_subtype(
				str((payload.get("object_types") or {}).get(value) or ""),
				expected_type,
				payload,
			):
				return [dict(bindings)]
			return []
		resolved_args = [self._py_resolve(arg, bindings) for arg in args]
		if not positive:
			if any(self._py_is_variable(arg, payload) for arg in resolved_args):
				return []
			return (
				[]
				if (predicate, tuple(resolved_args)) in self._payload_fact_set(payload)
				else [dict(bindings)]
			)
		output = []
		for fact_predicate, fact_args in self._payload_fact_set(payload):
			if fact_predicate != predicate or len(fact_args) != len(args):
				continue
			next_bindings = dict(bindings)
			if all(
				self._py_unify(pattern, value, next_bindings, payload)
				for pattern, value in zip(args, fact_args)
			):
				output.append(next_bindings)
		return output

	def _payload_fact_set(self, payload: Dict[str, Any]) -> set[Tuple[str, Tuple[str, ...]]]:
		facts: set[Tuple[str, Tuple[str, ...]]] = set()
		for item in payload.get("seed_facts") or []:
			if isinstance(item, dict):
				facts.add(
					(
						str(item.get("predicate") or ""),
						tuple(str(arg) for arg in (item.get("args") or ())),
					),
				)
			elif item:
				values = list(item)
				facts.add((str(values[0]), tuple(str(arg) for arg in values[1:])))
		return facts

	def _py_unify(
		self,
		pattern: str,
		value: str,
		bindings: Dict[str, str],
		payload: Dict[str, Any],
	) -> bool:
		left = self._py_resolve(pattern, bindings)
		right = self._py_resolve(value, bindings)
		if self._py_is_variable(left, payload):
			bindings[self._py_var_key(left)] = self._py_strip(right)
			return True
		if self._py_is_variable(right, payload):
			bindings[self._py_var_key(right)] = self._py_strip(left)
			return True
		return self._py_strip(left) == self._py_strip(right)

	def _py_resolve(self, value: str, bindings: Dict[str, str]) -> str:
		current = self._py_strip(value)
		seen: set[str] = set()
		while self._py_is_variable_token(current):
			key = self._py_var_key(current)
			if key in seen or key not in bindings:
				break
			seen.add(key)
			current = self._py_strip(bindings[key])
		return current

	def _py_is_variable(self, value: str, payload: Dict[str, Any]) -> bool:
		text = self._py_strip(value)
		if text in (payload.get("object_types") or {}):
			return False
		return self._py_is_variable_token(text)

	@staticmethod
	def _py_is_variable_token(value: str) -> bool:
		text = str(value or "").strip()
		return text.startswith("?") or bool(re.match(r"^[A-Z][A-Za-z0-9_]*$", text))

	@staticmethod
	def _py_var_key(value: str) -> str:
		text = str(value or "").strip()
		return text[1:] if text.startswith("?") else text

	@staticmethod
	def _py_strip(value: str) -> str:
		return str(value or "").strip().replace('"', "")

	def _py_is_subtype(self, actual: str, expected: str, payload: Dict[str, Any]) -> bool:
		if not actual or not expected:
			return False
		cursor = self._normalise_name(actual)
		target = self._normalise_name(expected)
		parents = payload.get("type_parent_map") or {}
		seen: set[str] = set()
		while cursor and cursor not in seen:
			if cursor == target:
				return True
			seen.add(cursor)
			cursor = self._normalise_name(str(parents.get(cursor) or ""))
		return target == "object"

	def _normalise_action_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
		clauses = schema.get("precondition_clauses") or []
		if not clauses:
			clauses = [schema.get("preconditions") or []]
		return {
			"functor": self._normalise_name(str(schema.get("functor") or "")),
			"source_name": str(schema.get("source_name") or schema.get("functor") or ""),
			"parameters": [str(item) for item in (schema.get("parameters") or ())],
			"precondition_clauses": [
				[self._normalise_literal(item) for item in clause]
				for clause in clauses
			],
			"effects": [self._normalise_literal(item) for item in (schema.get("effects") or ())],
		}

	def _normalise_plan(self, plan: AgentSpeakPlan) -> Dict[str, Any]:
		return {
			"plan_name": plan.plan_name,
			"trigger": self._normalise_name(plan.trigger.symbol),
			"trigger_arguments": [
				self._trigger_variable(argument)
				for argument in plan.trigger.arguments
			],
			"context": [
				literal
				for text in plan.context
				if (literal := self._parse_context_literal(text)) is not None
			],
			"body": [self._normalise_body_step(step) for step in plan.body],
		}

	def _normalise_body_step(self, step: AgentSpeakBodyStep) -> Dict[str, Any]:
		return {
			"kind": step.kind,
			"symbol": self._normalise_name(step.symbol),
			"source_symbol": step.symbol,
			"arguments": [str(argument) for argument in step.arguments],
		}

	def _normalise_literal(self, literal: Dict[str, Any]) -> Dict[str, Any]:
		return {
			"predicate": self._normalise_name(str(literal.get("predicate") or "")),
			"args": [str(arg) for arg in (literal.get("args") or ())],
			"positive": bool(literal.get("is_positive", True)),
		}

	def _normalise_query_goal(self, goal: Any) -> Dict[str, Any]:
		if isinstance(goal, dict):
			return {
				"task_name": self._normalise_name(str(goal.get("task_name") or "")),
				"args": [str(arg) for arg in (goal.get("args") or ())],
			}
		return {
			"task_name": self._normalise_name(str(getattr(goal, "task_name", "") or "")),
			"args": [str(arg) for arg in (getattr(goal, "args", ()) or ())],
		}

	def _render_pom(self) -> str:
		return textwrap.dedent(
			f"""\
			<project xmlns="http://maven.apache.org/POM/4.0.0"
				xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
				xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
				<modelVersion>4.0.0</modelVersion>
				<groupId>pipeline</groupId>
				<artifactId>jadex-runtime-bridge</artifactId>
				<version>1.0-SNAPSHOT</version>
				<properties>
					<maven.compiler.release>8</maven.compiler.release>
				</properties>
				<dependencies>
					<dependency>
						<groupId>org.activecomponents.jadex</groupId>
						<artifactId>jadex-distribution-minimal</artifactId>
						<version>{self.jadex_version}</version>
					</dependency>
					<dependency>
						<groupId>org.activecomponents.jadex</groupId>
						<artifactId>jadex-kernel-bdiv3</artifactId>
						<version>{self.jadex_version}</version>
					</dependency>
					<dependency>
						<groupId>org.activecomponents.jadex</groupId>
						<artifactId>jadex-kernel-application</artifactId>
						<version>{self.jadex_version}</version>
					</dependency>
					<dependency>
						<groupId>net.java.dev.jna</groupId>
						<artifactId>jna</artifactId>
						<version>{self.jna_version}</version>
					</dependency>
					<dependency>
						<groupId>net.java.dev.jna</groupId>
						<artifactId>jna-platform</artifactId>
						<version>{self.jna_version}</version>
					</dependency>
					<dependency>
						<groupId>com.fasterxml.jackson.core</groupId>
						<artifactId>jackson-databind</artifactId>
						<version>2.17.2</version>
					</dependency>
				</dependencies>
			</project>
			""",
		)

	def _render_enhance_main_source(self) -> str:
		return textwrap.dedent(
			"""\
			package pipeline;

			import jadex.bdiv3.BDIEnhancer;

			public class PipelineEnhanceMain {
				public static void main(String[] args) {
					BDIEnhancer.enhanceBDIClasses(args[0], null);
				}
			}
			""",
		)

	def _render_main_source(self) -> str:
		return textwrap.dedent(
			"""\
			package pipeline;

			import jadex.base.Starter;
			import jadex.bdiv3.BDIAgentFactory;
			import jadex.bridge.IExternalAccess;
			import jadex.bridge.service.types.clock.IClockService;
			import jadex.bridge.service.types.execution.IExecutionService;
			import jadex.commons.Tuple2;
			import java.nio.file.Files;
			import java.nio.file.Path;
			import java.nio.file.Paths;

			public class PipelineMain {
				public static void main(String[] args) throws Exception {
					Path output = Paths.get(System.getProperty("pipeline.output.dir"));
					Path done = output.resolve("jadex_done.txt");
					Files.deleteIfExists(done);
					Tuple2<IExecutionService, IClockService> services = Starter.createServices();
					BDIAgentFactory factory = new BDIAgentFactory("pipeline");
					factory.setFeatures(BDIAgentFactory.NOPLATFORM_DEFAULT_FEATURES);
					IExternalAccess agent = Starter.createAgent(
						"pipeline.PipelineBDI.class",
						factory,
						services.getFirstEntity(),
						services.getSecondEntity()
					).get(30000);
					long deadline = System.currentTimeMillis() + 2147483647L;
					while(!Files.exists(done) && System.currentTimeMillis() < deadline) {
						Thread.sleep(100L);
					}
					if(!Files.exists(done)) {
						try {
							agent.killComponent().get(30000);
						} catch(Exception ignored) {
						}
						throw new RuntimeException("Jadex agent did not write completion marker.");
					}
				}
			}
			""",
		)

	def _render_agent_source(self, payload: Dict[str, Any]) -> str:
		plans = payload["plans"]
		plan_classes = "\n".join(
			self._render_plan_class(index)
			for index, _plan in enumerate(plans)
		)
		return (
			self._agent_source_prefix()
			+ "\n"
			+ self._render_payload_initializers(payload)
			+ "\n"
			+ plan_classes
			+ "\n"
			+ self._agent_source_suffix()
		)

	def _render_plan_class(self, index: int) -> str:
		return textwrap.dedent(
			f"""\
				@Plan(trigger=@Trigger(goals=TaskGoal.class))
				public class Plan_{index} {{
					@PlanAPI
					protected IPlan rplan;

					@PlanReason
					protected Object reason;

					@PlanPrecondition
					public boolean precondition() {{
						return PipelineBDI.this.isApplicable({index}, (TaskGoal)reason);
					}}

					@PlanBody
					public IFuture<Void> body() {{
						try {{
							PipelineBDI.this.executePlan({index}, (TaskGoal)reason, rplan);
							return IFuture.DONE;
						}} catch(Exception ex) {{
							return new Future<Void>(new PlanFailureException());
						}}
					}}
				}}
			""",
		)

	def _render_payload_initializers(self, payload: Dict[str, Any]) -> str:
		del payload
		return textwrap.dedent(
			"""\
				private static String[][] INITIAL_FACTS = new String[0][0];
				private static String[][] GOAL_FACTS = new String[0][0];
				private static String[][] QUERY_GOALS = new String[0][0];
				private static String[][] OBJECT_TYPES = new String[0][0];
				private static String[][] TYPE_PARENTS = new String[0][0];
				private static ActionSchema[] ACTIONS = new ActionSchema[0];
				private static PlanSpec[] PLANS = new PlanSpec[0];
				private static boolean PAYLOAD_LOADED = false;
			""",
		)

	def _agent_source_prefix(self) -> str:
		return textwrap.dedent(
			"""\
			package pipeline;

			import java.nio.charset.StandardCharsets;
			import java.nio.file.Files;
			import java.nio.file.Path;
			import java.nio.file.Paths;
			import java.util.ArrayList;
			import java.util.Arrays;
			import java.util.Collections;
			import java.util.Comparator;
			import java.util.HashMap;
			import java.util.HashSet;
			import java.util.LinkedHashMap;
			import java.util.List;
			import java.util.Map;
			import java.util.Set;
			import java.util.concurrent.CountDownLatch;
			import jadex.bdiv3.annotation.ExcludeMode;
			import jadex.bdiv3.annotation.Goal;
			import jadex.bdiv3.annotation.Plan;
			import jadex.bdiv3.annotation.PlanAPI;
			import jadex.bdiv3.annotation.PlanBody;
			import jadex.bdiv3.annotation.PlanPrecondition;
			import jadex.bdiv3.annotation.PlanReason;
			import jadex.bdiv3.annotation.Trigger;
			import jadex.bdiv3.features.IBDIAgentFeature;
			import jadex.bdiv3.runtime.IPlan;
			import jadex.bdiv3.runtime.impl.PlanFailureException;
			import jadex.bridge.IInternalAccess;
			import jadex.commons.Boolean3;
			import jadex.commons.future.Future;
			import jadex.commons.future.IFuture;
			import jadex.micro.annotation.Agent;
			import jadex.micro.annotation.AgentBody;
			import com.fasterxml.jackson.databind.JsonNode;
			import com.fasterxml.jackson.databind.ObjectMapper;

			@Agent(type="bdi", keepalive=Boolean3.FALSE)
			public class PipelineBDI {
				public static final CountDownLatch done = new CountDownLatch(1);

				@Agent
				protected IInternalAccess agent;

				@Goal(retry=true, excludemode=ExcludeMode.WhenTried)
				public static class TaskGoal {
					public final String name;
					public final String[] args;
					public TaskGoal(String name, String[] args) {
						this.name = norm(name);
						this.args = args == null ? new String[0] : args;
					}
				}

				static class Literal {
					final String predicate;
					final String[] args;
					final boolean positive;
					Literal(String predicate, String[] args, boolean positive) {
						this.predicate = norm(predicate);
						this.args = args == null ? new String[0] : args;
						this.positive = positive;
					}
				}

				static class Step {
					final String kind;
					final String symbol;
					final String sourceSymbol;
					final String[] args;
					Step(String kind, String symbol, String sourceSymbol, String[] args) {
						this.kind = kind;
						this.symbol = norm(symbol);
						this.sourceSymbol = sourceSymbol;
						this.args = args == null ? new String[0] : args;
					}
				}

				static class PlanSpec {
					final String name;
					final String trigger;
					final String[] triggerArgs;
					final Literal[] context;
					final Step[] body;
					PlanSpec(String name, String trigger, String[] triggerArgs, Literal[] context, Step[] body) {
						this.name = name;
						this.trigger = norm(trigger);
						this.triggerArgs = triggerArgs == null ? new String[0] : triggerArgs;
						this.context = context == null ? new Literal[0] : context;
						this.body = body == null ? new Step[0] : body;
					}
				}

				static class ActionSchema {
					final String functor;
					final String sourceName;
					final String[] parameters;
					final Literal[][] clauses;
					final Literal[] effects;
					ActionSchema(String functor, String sourceName, String[] parameters, Literal[][] clauses, Literal[] effects) {
						this.functor = norm(functor);
						this.sourceName = sourceName;
						this.parameters = parameters == null ? new String[0] : parameters;
						this.clauses = clauses == null ? new Literal[0][0] : clauses;
						this.effects = effects == null ? new Literal[0] : effects;
					}
				}

				static class RuntimeSnapshot {
					final Set<String> facts;
					final Set<String> protectedFacts;
					final int actionPathSize;
					final int methodTraceSize;
					final int failedGoalsSize;
					RuntimeSnapshot(
						Set<String> facts,
						Set<String> protectedFacts,
						int actionPathSize,
						int methodTraceSize,
						int failedGoalsSize
					) {
						this.facts = facts;
						this.protectedFacts = protectedFacts;
						this.actionPathSize = actionPathSize;
						this.methodTraceSize = methodTraceSize;
						this.failedGoalsSize = failedGoalsSize;
					}
				}
			""",
		)

	def _agent_source_suffix(self) -> str:
		return textwrap.dedent(
			"""\
				private final Set<String> facts = new HashSet<String>();
				private final Set<String> goalFacts = new HashSet<String>();
				private final Set<String> protectedFacts = new HashSet<String>();
				private final boolean protectGoalFacts = Boolean.parseBoolean(
					System.getProperty("pipeline.protect.goal.facts", "true")
				);
				private final Map<String, String> objectTypes = new HashMap<String, String>();
				private final Map<String, String> typeParents = new HashMap<String, String>();
				private final List<String> actionPath = new ArrayList<String>();
				private final List<String[]> methodTrace = new ArrayList<String[]>();
				private final List<String> failedGoals = new ArrayList<String>();
				private final int maxTopLevelSearchAttempts = Integer.parseInt(
					System.getProperty("pipeline.max.top.level.attempts", "5000")
				);
				private int topLevelSearchAttempts = 0;

				@AgentBody
				public void body() {
					try {
						initState();
						if(!executeQueryGoals(agent.getFeature(IBDIAgentFeature.class))) {
							failedGoals.add("top_level_search_failed");
							throw new PlanFailureException();
						}
						writeArtifacts("success");
						System.out.println("execute success");
					} catch(Exception ex) {
						try {
							writeArtifacts("failed");
						} catch(Exception ignored) {
						}
						System.out.println("execute failed");
						ex.printStackTrace();
					} finally {
						try {
							Path output = Paths.get(System.getProperty("pipeline.output.dir"));
							Files.createDirectories(output);
							Files.write(output.resolve("jadex_done.txt"), "done".getBytes(StandardCharsets.UTF_8));
						} catch(Exception ignored) {
						}
						done.countDown();
						agent.killComponent();
					}
				}

				private boolean executeQueryGoals(IBDIAgentFeature bdi) {
					List<TaskGoal> goals = new ArrayList<TaskGoal>();
					for(String[] item: QUERY_GOALS) {
						String[] args = Arrays.copyOfRange(item, 1, item.length);
						goals.add(new TaskGoal(item[0], args));
					}
					List<Integer> explicitOrder = explicitGoalOrder(goals.size());
					if(!explicitOrder.isEmpty()) {
						return executeGoalOrder(goals, explicitOrder, bdi);
					}
					return executeGoalSearch(goals, bdi);
				}

				private boolean executeGoalOrder(
					List<TaskGoal> goals,
					List<Integer> order,
					IBDIAgentFeature bdi
				) {
					for(Integer indexValue: order) {
						int index = indexValue.intValue();
						if(index < 0 || index >= goals.size()) {
							return false;
						}
						try {
							TaskGoal goal = goals.get(index);
							bdi.dispatchTopLevelGoal(new TaskGoal(goal.name, goal.args)).get();
						} catch(Exception ex) {
							return false;
						}
					}
					return true;
				}

				private List<Integer> explicitGoalOrder(int goalCount) {
					List<Integer> order = new ArrayList<Integer>();
					String raw = System.getProperty("pipeline.query.order", "").trim();
					if(raw.isEmpty()) {
						return order;
					}
					Set<Integer> seen = new HashSet<Integer>();
					for(String item: raw.split(",")) {
						if(item.trim().isEmpty()) {
							continue;
						}
						int index = Integer.parseInt(item.trim());
						if(index < 0 || index >= goalCount || seen.contains(Integer.valueOf(index))) {
							return new ArrayList<Integer>();
						}
						order.add(Integer.valueOf(index));
						seen.add(Integer.valueOf(index));
					}
					if(order.size() != goalCount) {
						return new ArrayList<Integer>();
					}
					return order;
				}

				private boolean executeGoalSearch(List<TaskGoal> remaining, IBDIAgentFeature bdi) {
					if(remaining.isEmpty()) {
						return true;
					}
					if(topLevelSearchAttempts >= maxTopLevelSearchAttempts) {
						return false;
					}
					for(Integer indexValue: orderedGoalIndexes(remaining)) {
						int index = indexValue.intValue();
						topLevelSearchAttempts += 1;
						RuntimeSnapshot snapshot = snapshotRuntime();
						TaskGoal goal = remaining.get(index);
						try {
							bdi.dispatchTopLevelGoal(new TaskGoal(goal.name, goal.args)).get();
							List<TaskGoal> next = new ArrayList<TaskGoal>(remaining);
							next.remove(index);
							if(executeGoalSearch(next, bdi)) {
								return true;
							}
						} catch(Exception ex) {
						}
						restoreRuntime(snapshot);
						if(topLevelSearchAttempts >= maxTopLevelSearchAttempts) {
							return false;
						}
					}
					return false;
				}

				private List<Integer> orderedGoalIndexes(List<TaskGoal> remaining) {
					List<Integer> indexes = new ArrayList<Integer>();
					for(int index=0; index<remaining.size(); index++) {
						indexes.add(Integer.valueOf(index));
					}
					Collections.sort(indexes, new Comparator<Integer>() {
						public int compare(Integer left, Integer right) {
							int leftCount = countGoalCandidates(remaining.get(left.intValue()));
							int rightCount = countGoalCandidates(remaining.get(right.intValue()));
							if(leftCount != rightCount) {
								return Integer.compare(leftCount, rightCount);
							}
							return Integer.compare(left.intValue(), right.intValue());
						}
					});
					return indexes;
				}

				private int countGoalCandidates(TaskGoal goal) {
					int count = 0;
					for(int index=0; index<PLANS.length; index++) {
						PlanSpec plan = PLANS[index];
						for(Map<String, String> bindings: allBindings(plan, goal)) {
							if(!planWouldDeleteProtectedGoal(plan, bindings)) {
								count += 1;
								if(count >= 100) {
									return count;
								}
							}
						}
					}
					return count;
				}

				private RuntimeSnapshot snapshotRuntime() {
					return new RuntimeSnapshot(
						new HashSet<String>(facts),
						new HashSet<String>(protectedFacts),
						actionPath.size(),
						methodTrace.size(),
						failedGoals.size()
					);
				}

				private void restoreRuntime(RuntimeSnapshot snapshot) {
					facts.clear();
					facts.addAll(snapshot.facts);
					protectedFacts.clear();
					protectedFacts.addAll(snapshot.protectedFacts);
					while(actionPath.size() > snapshot.actionPathSize) {
						actionPath.remove(actionPath.size() - 1);
					}
					while(methodTrace.size() > snapshot.methodTraceSize) {
						methodTrace.remove(methodTrace.size() - 1);
					}
					while(failedGoals.size() > snapshot.failedGoalsSize) {
						failedGoals.remove(failedGoals.size() - 1);
					}
				}

				private static synchronized void loadPayload() throws Exception {
					if(PAYLOAD_LOADED) {
						return;
					}
					Path payloadPath = Paths.get(System.getProperty("pipeline.payload.path"));
					JsonNode root = new ObjectMapper().readTree(payloadPath.toFile());
					INITIAL_FACTS = factRows(root.path("seed_facts"));
					GOAL_FACTS = factRows(root.path("goal_facts"));
					QUERY_GOALS = queryGoalRows(root.path("query_goals"));
					OBJECT_TYPES = mapPairs(root.path("object_types"), false);
					TYPE_PARENTS = mapPairs(root.path("type_parent_map"), true);
					ACTIONS = actionSchemas(root.path("actions"));
					PLANS = planSpecs(root.path("plans"));
					PAYLOAD_LOADED = true;
				}

				private static String[][] factRows(JsonNode items) {
					List<String[]> rows = new ArrayList<String[]>();
					for(JsonNode item: items) {
						rows.add(prepend(text(item, "predicate"), stringArray(item.path("args"))));
					}
					return rows.toArray(new String[rows.size()][]);
				}

				private static String[][] queryGoalRows(JsonNode items) {
					List<String[]> rows = new ArrayList<String[]>();
					for(JsonNode item: items) {
						rows.add(prepend(text(item, "task_name"), stringArray(item.path("args"))));
					}
					return rows.toArray(new String[rows.size()][]);
				}

				private static String[][] mapPairs(JsonNode values, boolean allowNullValue) {
					List<String[]> rows = new ArrayList<String[]>();
					java.util.Iterator<Map.Entry<String, JsonNode>> fields = values.fields();
					while(fields.hasNext()) {
						Map.Entry<String, JsonNode> entry = fields.next();
						if(allowNullValue && entry.getValue().isNull()) {
							rows.add(new String[]{entry.getKey()});
						} else {
							rows.add(new String[]{entry.getKey(), entry.getValue().asText("")});
						}
					}
					return rows.toArray(new String[rows.size()][]);
				}

				private static ActionSchema[] actionSchemas(JsonNode items) {
					List<ActionSchema> output = new ArrayList<ActionSchema>();
					for(JsonNode item: items) {
						output.add(
							new ActionSchema(
								text(item, "functor"),
								text(item, "source_name"),
								stringArray(item.path("parameters")),
								literalClauses(item.path("precondition_clauses")),
								literals(item.path("effects"))
							)
						);
					}
					return output.toArray(new ActionSchema[output.size()]);
				}

				private static PlanSpec[] planSpecs(JsonNode items) {
					List<PlanSpec> output = new ArrayList<PlanSpec>();
					for(JsonNode item: items) {
						output.add(
							new PlanSpec(
								text(item, "plan_name"),
								text(item, "trigger"),
								stringArray(item.path("trigger_arguments")),
								literals(item.path("context")),
								steps(item.path("body"))
							)
						);
					}
					return output.toArray(new PlanSpec[output.size()]);
				}

				private static Literal[][] literalClauses(JsonNode clauses) {
					List<Literal[]> output = new ArrayList<Literal[]>();
					for(JsonNode clause: clauses) {
						output.add(literals(clause));
					}
					return output.toArray(new Literal[output.size()][]);
				}

				private static Literal[] literals(JsonNode items) {
					List<Literal> output = new ArrayList<Literal>();
					for(JsonNode item: items) {
						output.add(
							new Literal(
								text(item, "predicate"),
								stringArray(item.path("args")),
								item.path("positive").asBoolean(true)
							)
						);
					}
					return output.toArray(new Literal[output.size()]);
				}

				private static Step[] steps(JsonNode items) {
					List<Step> output = new ArrayList<Step>();
					for(JsonNode item: items) {
						output.add(
							new Step(
								text(item, "kind"),
								text(item, "symbol"),
								text(item, "source_symbol"),
								stringArray(item.path("arguments"))
							)
						);
					}
					return output.toArray(new Step[output.size()]);
				}

				private static String[] prepend(String head, String[] tail) {
					String[] output = new String[tail.length + 1];
					output[0] = head;
					System.arraycopy(tail, 0, output, 1, tail.length);
					return output;
				}

				private static String[] stringArray(JsonNode items) {
					List<String> output = new ArrayList<String>();
					for(JsonNode item: items) {
						output.add(item.asText(""));
					}
					return output.toArray(new String[output.size()]);
				}

				private static String text(JsonNode node, String field) {
					return node.path(field).asText("");
				}

				private void initState() throws Exception {
					loadPayload();
					for(String[] fact: INITIAL_FACTS) {
						facts.add(atom(fact[0], Arrays.copyOfRange(fact, 1, fact.length)));
					}
					for(String[] fact: GOAL_FACTS) {
						goalFacts.add(atom(fact[0], Arrays.copyOfRange(fact, 1, fact.length)));
					}
					for(String[] pair: OBJECT_TYPES) {
						objectTypes.put(pair[0], pair[1]);
					}
					for(String[] pair: TYPE_PARENTS) {
						typeParents.put(pair[0], pair.length > 1 ? pair[1] : null);
					}
				}

				private boolean isApplicable(int planIndex, TaskGoal goal) {
					PlanSpec plan = PLANS[planIndex];
					for(Map<String, String> bindings: allBindings(plan, goal)) {
						if(!planWouldDeleteProtectedGoal(plan, bindings)) {
							return true;
						}
					}
					return false;
				}

				private void executePlan(int planIndex, TaskGoal goal, IPlan rplan) {
					PlanSpec plan = PLANS[planIndex];
					List<Map<String, String>> candidates = allBindings(plan, goal);
					if(candidates.isEmpty()) {
						recordFailedGoal(goal);
						throw new PlanFailureException();
					}
					for(Map<String, String> bindings: candidates) {
						if(planWouldDeleteProtectedGoal(plan, bindings)) {
							continue;
						}
						Set<String> factSnapshot = new HashSet<String>(facts);
						Set<String> protectedSnapshot = new HashSet<String>(protectedFacts);
						int actionStart = actionPath.size();
						int traceStart = methodTrace.size();
						methodTrace.add(new String[]{plan.name, String.join(",", goal.args)});
						try {
							for(Step step: plan.body) {
								String[] args = resolveArgs(step.args, bindings);
								if("subgoal".equals(step.kind)) {
									try {
										rplan.dispatchSubgoal(new TaskGoal(step.symbol, args)).get();
									} catch(Exception ex) {
										throw new PlanFailureException();
									}
								} else if("action".equals(step.kind)) {
									if(!applyAction(step.symbol, args)) {
										throw new PlanFailureException();
									}
								}
							}
							return;
						} catch(RuntimeException ex) {
							facts.clear();
							facts.addAll(factSnapshot);
							protectedFacts.clear();
							protectedFacts.addAll(protectedSnapshot);
							while(actionPath.size() > actionStart) {
								actionPath.remove(actionPath.size() - 1);
							}
							while(methodTrace.size() > traceStart) {
								methodTrace.remove(methodTrace.size() - 1);
							}
						}
					}
					recordFailedGoal(goal);
					throw new PlanFailureException();
				}

				private List<Map<String, String>> allBindings(PlanSpec plan, TaskGoal goal) {
					if(!plan.trigger.equals(goal.name) || plan.triggerArgs.length != goal.args.length) {
						return Collections.emptyList();
					}
					Map<String, String> bindings = new LinkedHashMap<String, String>();
					for(int i=0; i<plan.triggerArgs.length; i++) {
						if(!unify(plan.triggerArgs[i], goal.args[i], bindings)) {
							return Collections.emptyList();
						}
					}
					return satisfy(plan.context, 0, bindings);
				}

				private boolean applyAction(String actionName, String[] args) {
					ActionSchema schema = findAction(actionName);
					if(schema == null || schema.parameters.length != args.length) {
						return false;
					}
					Map<String, String> base = new LinkedHashMap<String, String>();
					for(int i=0; i<schema.parameters.length; i++) {
						if(!unify(schema.parameters[i], args[i], base)) {
							return false;
						}
					}
					for(Literal[] clause: schema.clauses) {
						List<Map<String, String>> matches = satisfy(clause, 0, base);
						if(matches.isEmpty()) {
							continue;
						}
						for(Map<String, String> bindings: matches) {
							if(wouldDeleteProtectedGoal(schema.effects, bindings)) {
								continue;
							}
							for(Literal effect: schema.effects) {
								String[] effectArgs = resolveArgs(effect.args, bindings);
								if(hasVariable(effectArgs)) {
									continue;
								}
								String fact = atom(effect.predicate, effectArgs);
								if(effect.positive) {
									facts.add(fact);
									if(protectGoalFacts && goalFacts.contains(fact)) {
										protectedFacts.add(fact);
									}
								} else {
									facts.remove(fact);
								}
							}
							String rendered = renderAction(schema.sourceName, resolveArgs(schema.parameters, bindings));
							actionPath.add(rendered);
							return true;
						}
					}
					return false;
				}

				private boolean planWouldDeleteProtectedGoal(PlanSpec plan, Map<String, String> bindings) {
					for(Step step: plan.body) {
						if(!"action".equals(step.kind)) {
							continue;
						}
						String[] args = resolveArgs(step.args, bindings);
						if(hasVariable(args)) {
							continue;
						}
						ActionSchema schema = findAction(step.symbol);
						if(schema == null || schema.parameters.length != args.length) {
							continue;
						}
						Map<String, String> base = new LinkedHashMap<String, String>();
						boolean unified = true;
						for(int i=0; i<schema.parameters.length; i++) {
							unified = unified && unify(schema.parameters[i], args[i], base);
						}
						if(!unified) {
							continue;
						}
						boolean sawExecutableBinding = false;
						boolean sawProtectedDeletion = false;
						for(Literal[] clause: schema.clauses) {
							List<Map<String, String>> matches = satisfy(clause, 0, base);
							for(Map<String, String> match: matches) {
								sawExecutableBinding = true;
								if(wouldDeleteProtectedGoal(schema.effects, match)) {
									sawProtectedDeletion = true;
								} else {
									return false;
								}
							}
						}
						if(sawExecutableBinding && sawProtectedDeletion) {
							return true;
						}
					}
					return false;
				}

				private boolean wouldDeleteProtectedGoal(Literal[] effects, Map<String, String> bindings) {
					if(!protectGoalFacts) {
						return false;
					}
					Set<String> restored = new HashSet<String>();
					Set<String> deleted = new HashSet<String>();
					for(Literal effect: effects) {
						String[] effectArgs = resolveArgs(effect.args, bindings);
						if(hasVariable(effectArgs)) {
							continue;
						}
						String fact = atom(effect.predicate, effectArgs);
						if(effect.positive) {
							restored.add(fact);
						} else {
							deleted.add(fact);
						}
					}
					for(String fact: deleted) {
						if(protectedFacts.contains(fact) && facts.contains(fact) && !restored.contains(fact)) {
							System.out.println("runtime env action protected-goal delete rejected " + renderEncodedFact(fact));
							return true;
						}
					}
					return false;
				}

				private List<Map<String, String>> satisfy(Literal[] literals, int index, Map<String, String> bindings) {
					List<Map<String, String>> output = new ArrayList<Map<String, String>>();
					if(index >= literals.length) {
						output.add(new LinkedHashMap<String, String>(bindings));
						return output;
					}
					Literal literal = literals[index];
					for(Map<String, String> next: satisfyOne(literal, bindings)) {
						output.addAll(satisfy(literals, index + 1, next));
					}
					return output;
				}

				private List<Map<String, String>> satisfyOne(Literal literal, Map<String, String> bindings) {
					List<Map<String, String>> output = new ArrayList<Map<String, String>>();
					if("=".equals(literal.predicate) && literal.args.length == 2) {
						Map<String, String> next = new LinkedHashMap<String, String>(bindings);
						if(unify(literal.args[0], literal.args[1], next) == literal.positive) {
							output.add(next);
						}
						return output;
					}
					if("object_type".equals(literal.predicate) && literal.args.length == 2) {
						String value = resolve(literal.args[0], bindings);
						String type = strip(literal.args[1]);
						if(isVariable(value)) {
							for(Map.Entry<String, String> entry: objectTypes.entrySet()) {
								if(isSubtype(entry.getValue(), type)) {
									Map<String, String> next = new LinkedHashMap<String, String>(bindings);
									if(unify(literal.args[0], entry.getKey(), next)) {
										output.add(next);
									}
								}
							}
						} else if(isSubtype(objectTypes.get(value), type)) {
							output.add(new LinkedHashMap<String, String>(bindings));
						}
						return output;
					}
					String[] args = resolveArgs(literal.args, bindings);
					if(!literal.positive) {
						if(!hasVariable(args) && !facts.contains(atom(literal.predicate, args))) {
							output.add(new LinkedHashMap<String, String>(bindings));
						}
						return output;
					}
					for(String fact: facts) {
						String[] parts = fact.split("\\\\|", -1);
						if(parts.length - 1 != literal.args.length || !parts[0].equals(literal.predicate)) {
							continue;
						}
						Map<String, String> next = new LinkedHashMap<String, String>(bindings);
						boolean ok = true;
						for(int i=0; i<literal.args.length; i++) {
							ok = ok && unify(literal.args[i], parts[i + 1], next);
						}
						if(ok) {
							output.add(next);
						}
					}
					return output;
				}

				private ActionSchema findAction(String actionName) {
					String normalized = norm(actionName);
					for(ActionSchema schema: ACTIONS) {
						if(schema.functor.equals(normalized) || norm(schema.sourceName).equals(normalized)) {
							return schema;
						}
					}
					return null;
				}

				private boolean unify(String pattern, String value, Map<String, String> bindings) {
					String left = resolve(pattern, bindings);
					String right = resolve(value, bindings);
					if(isVariable(left)) {
						bindings.put(varKey(left), strip(right));
						return true;
					}
					if(isVariable(right)) {
						bindings.put(varKey(right), strip(left));
						return true;
					}
					return strip(left).equals(strip(right));
				}

				private String resolve(String value, Map<String, String> bindings) {
					String current = strip(value);
					Set<String> seen = new HashSet<String>();
					while(isVariable(current)) {
						String key = varKey(current);
						if(seen.contains(key) || !bindings.containsKey(key)) {
							break;
						}
						seen.add(key);
						current = strip(bindings.get(key));
					}
					return current;
				}

				private String[] resolveArgs(String[] args, Map<String, String> bindings) {
					String[] output = new String[args.length];
					for(int i=0; i<args.length; i++) {
						output[i] = resolve(args[i], bindings);
					}
					return output;
				}

				private boolean hasVariable(String[] values) {
					for(String value: values) {
						if(isVariable(value)) {
							return true;
						}
					}
					return false;
				}

				private boolean isVariable(String value) {
					String text = strip(value);
					if(objectTypes.containsKey(text)) {
						return false;
					}
					return text.startsWith("?") || text.matches("[A-Z][A-Za-z0-9_]*");
				}

				private String varKey(String value) {
					String text = strip(value);
					return text.startsWith("?") ? text.substring(1) : text;
				}

				private boolean isSubtype(String actual, String expected) {
					if(actual == null || expected == null) {
						return false;
					}
					String cursor = norm(actual);
					String target = norm(expected);
					Set<String> seen = new HashSet<String>();
					while(cursor != null && !seen.contains(cursor)) {
						if(cursor.equals(target)) {
							return true;
						}
						seen.add(cursor);
						cursor = typeParents.get(cursor);
					}
					return "object".equals(target);
				}

				private void recordFailedGoal(TaskGoal goal) {
					String rendered = goal.name + "(" + String.join(",", goal.args) + ")";
					if(!failedGoals.contains(rendered)) {
						failedGoals.add(rendered);
						System.out.println("runtime goal failed " + rendered);
					}
				}

				private void writeArtifacts(String status) throws Exception {
					Path output = Paths.get(System.getProperty("pipeline.output.dir"));
					Files.createDirectories(output);
					Files.write(output.resolve("action_path.txt"), String.join("\\n", actionPath).getBytes(StandardCharsets.UTF_8));
					List<String> traceLines = new ArrayList<String>();
					traceLines.add("[");
					for(int i=0; i<methodTrace.size(); i++) {
						String[] item = methodTrace.get(i);
						String[] args = item.length > 1 && !item[1].isEmpty() ? item[1].split(",", -1) : new String[0];
						traceLines.add("  {\\\"method_name\\\":\\\"" + json(item[0]) + "\\\",\\\"task_args\\\":" + jsonArray(args) + "}" + (i + 1 < methodTrace.size() ? "," : ""));
					}
					traceLines.add("]");
					Files.write(output.resolve("method_trace.json"), String.join("\\n", traceLines).getBytes(StandardCharsets.UTF_8));
					List<String> world = new ArrayList<String>();
					for(String fact: facts) {
						String[] parts = fact.split("\\\\|", -1);
						world.add(renderFact(parts[0], Arrays.copyOfRange(parts, 1, parts.length)));
					}
					java.util.Collections.sort(world);
					Files.write(output.resolve("final_world_facts.txt"), String.join("\\n", world).getBytes(StandardCharsets.UTF_8));
					Files.write(output.resolve("jadex_status.txt"), status.getBytes(StandardCharsets.UTF_8));
				}

				private static String atom(String predicate, String[] args) {
					String name = norm(predicate);
					return args.length == 0 ? name : name + "|" + String.join("|", args);
				}

				private static String renderAction(String name, String[] args) {
					return args.length == 0 ? name : name + "(" + String.join(", ", args) + ")";
				}

				private static String renderFact(String predicate, String[] args) {
					return args.length == 0 ? predicate : predicate + "(" + String.join(", ", args) + ")";
				}

				private static String renderEncodedFact(String fact) {
					String[] parts = fact.split("\\\\|", -1);
					return renderFact(parts[0], Arrays.copyOfRange(parts, 1, parts.length));
				}

				private static String strip(String value) {
					String text = value == null ? "" : value.trim();
					if(text.length() >= 2 && ((text.startsWith("\\\"") && text.endsWith("\\\"")) || (text.startsWith("'") && text.endsWith("'")))) {
						return text.substring(1, text.length() - 1);
					}
					return text;
				}

				private static String norm(String value) {
					return strip(value).replaceAll("[^A-Za-z0-9_]+", "_").replaceAll("^_+|_+$", "");
				}

				private static String json(String value) {
					return value.replace("\\\\", "\\\\\\\\").replace("\\\"", "\\\\\\\"");
				}

				private static String jsonArray(String[] values) {
					List<String> encoded = new ArrayList<String>();
					for(String value: values) {
						encoded.add("\\\"" + json(value) + "\\\"");
					}
					return "[" + String.join(",", encoded) + "]";
				}
			}
			""",
		)

	def _java_actions(self, actions: Sequence[Dict[str, Any]]) -> str:
		items = [
			(
				"new ActionSchema("
				f"{self._j(action['functor'])}, {self._j(action['source_name'])}, "
				f"{self._java_string_array(action['parameters'])}, "
				f"{self._java_literal_matrix(action['precondition_clauses'])}, "
				f"{self._java_literals(action['effects'])})"
			)
			for action in actions
		]
		return "new ActionSchema[]{" + ", ".join(items) + "}"

	def _java_plans(self, plans: Sequence[Dict[str, Any]]) -> str:
		items = [
			(
				"new PlanSpec("
				f"{self._j(plan['plan_name'])}, {self._j(plan['trigger'])}, "
				f"{self._java_string_array(plan['trigger_arguments'])}, "
				f"{self._java_literals(plan['context'])}, "
				f"{self._java_steps(plan['body'])})"
			)
			for plan in plans
		]
		return "new PlanSpec[]{" + ", ".join(items) + "}"

	def _java_steps(self, steps: Sequence[Dict[str, Any]]) -> str:
		items = [
			(
				"new Step("
				f"{self._j(step['kind'])}, {self._j(step['symbol'])}, "
				f"{self._j(step['source_symbol'])}, {self._java_string_array(step['arguments'])})"
			)
			for step in steps
		]
		return "new Step[]{" + ", ".join(items) + "}"

	def _java_literal_matrix(self, clauses: Sequence[Sequence[Dict[str, Any]]]) -> str:
		return "new Literal[][]{" + ", ".join(self._java_literals(clause) for clause in clauses) + "}"

	def _java_literals(self, literals: Sequence[Dict[str, Any]]) -> str:
		items = [
			(
				"new Literal("
				f"{self._j(literal['predicate'])}, "
				f"{self._java_string_array(literal['args'])}, "
				f"{str(bool(literal['positive'])).lower()})"
			)
			for literal in literals
		]
		return "new Literal[]{" + ", ".join(items) + "}"

	def _java_string_matrix(self, facts: Sequence[Dict[str, Any]]) -> str:
		rows = [
			self._java_string_array([fact["predicate"], *fact["args"]])
			for fact in facts
		]
		return "new String[][]{" + ", ".join(rows) + "}"

	def _java_goal_matrix(self, goals: Sequence[Dict[str, Any]]) -> str:
		rows = [
			self._java_string_array([goal["task_name"], *goal["args"]])
			for goal in goals
		]
		return "new String[][]{" + ", ".join(rows) + "}"

	def _java_map_pairs(self, values: Dict[str, str]) -> str:
		return "new String[][]{" + ", ".join(
			self._java_string_array([key, value])
			for key, value in values.items()
		) + "}"

	def _java_nullable_map_pairs(self, values: Dict[str, Optional[str]]) -> str:
		return "new String[][]{" + ", ".join(
			self._java_string_array([key] if value is None else [key, value])
			for key, value in values.items()
		) + "}"

	def _java_string_array(self, values: Sequence[str]) -> str:
		return "new String[]{" + ", ".join(self._j(value) for value in values) + "}"

	def _j(self, value: Any) -> str:
		return json.dumps(str(value))

	def _parse_context_literal(self, text: str) -> Optional[Dict[str, Any]]:
		value = str(text or "").strip()
		if not value or value.lower() == "true":
			return None
		positive = True
		if value.startswith("not "):
			positive = False
			value = value[4:].strip()
		if value.startswith("!"):
			positive = False
			value = value[1:].strip()
		if "!=" in value:
			left, right = value.split("!=", 1)
			return {"predicate": "=", "args": [left.strip(), right.strip()], "positive": False}
		if "\\==" in value:
			left, right = value.split("\\==", 1)
			return {"predicate": "=", "args": [left.strip(), right.strip()], "positive": False}
		if "==" in value:
			left, right = value.split("==", 1)
			return {"predicate": "=", "args": [left.strip(), right.strip()], "positive": True}
		match = re.fullmatch(r"([A-Za-z0-9_-]+)(?:\((.*)\))?", value)
		if match is None:
			return None
		args_text = (match.group(2) or "").strip()
		return {
			"predicate": self._normalise_name(match.group(1)),
			"args": [part.strip() for part in args_text.split(",") if part.strip()],
			"positive": positive,
		}

	def _hddl_fact_to_atom(self, fact: str) -> Optional[Dict[str, Any]]:
		text = str(fact or "").strip()
		if not text.startswith("(") or not text.endswith(")"):
			return None
		tokens = text[1:-1].strip().split()
		if not tokens or tokens[0].lower() == "not":
			return None
		return {"predicate": self._normalise_name(tokens[0]), "args": tokens[1:]}

	def _trigger_variable(self, pattern: str) -> str:
		value = str(pattern or "").strip()
		if ":" in value:
			value = value.split(":", 1)[0].strip()
		return value

	def _artifact_payload(
		self,
		*,
		output_path: Path,
		project_dir: Path,
		payload_path: Path,
		stdout_path: Path,
		stderr_path: Path,
		action_path_path: Path,
		method_trace_path: Path,
		final_world_path: Path,
		validation_path: Path,
		stdout: str,
		stderr: str,
	) -> Dict[str, Any]:
		return {
			"runtime_backend": "jadex",
			"runtime_semantics": "jadex_bdiv3_framework",
			"jadex_version": self.jadex_version,
			"jadex_project": str(project_dir),
			"jadex_payload": str(payload_path),
			"jadex_stdout": str(stdout_path),
			"jadex_stderr": str(stderr_path),
			"action_path": str(action_path_path),
			"method_trace": str(method_trace_path),
			"final_world_facts": str(final_world_path),
			"jadex_validation": str(validation_path),
			"jadex_attempts": str(output_path / "jadex_attempts.json"),
			"stdout_chars": len(stdout),
			"stderr_chars": len(stderr),
			"stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
			"stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
		}

	def _failure_metadata(
		self,
		*,
		output_path: Path,
		validation_path: Path,
		failure_class: str,
		exit_code: Optional[int],
		timed_out: bool,
		stdout: str,
		stderr: str,
		timing_profile: Dict[str, Any],
	) -> Dict[str, Any]:
		artifacts = {
			"runtime_backend": "jadex",
			"runtime_semantics": "jadex_bdiv3_framework",
			"jadex_version": self.jadex_version,
			"jadex_stdout": str(output_path / "jadex_stdout.txt"),
			"jadex_stderr": str(output_path / "jadex_stderr.txt"),
			"jadex_validation": str(validation_path),
			"stdout_chars": len(stdout),
			"stderr_chars": len(stderr),
		}
		metadata = {
			"status": "failed",
			"backend": self.backend_name,
			"exit_code": exit_code,
			"timed_out": timed_out,
			"action_path_count": 0,
			"method_trace_count": 0,
			"failed_goals": [],
			"failure_class": failure_class,
			"consistency_checks": {
				"success": False,
				"runtime_semantics": "jadex_bdiv3_framework",
			},
			"artifacts": artifacts,
			"timing_profile": dict(timing_profile),
		}
		validation_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
		return metadata

	def _run_streamed_process(
		self,
		cmd: Sequence[str],
		*,
		cwd: Path,
		env: Dict[str, str],
		stdout_path: Path,
		stderr_path: Path,
		timeout_seconds: int,
	) -> subprocess.CompletedProcess[str]:
		stdout_tail: Dict[str, str] = {}
		stderr_tail: Dict[str, str] = {}
		proc = subprocess.Popen(
			list(cmd),
			cwd=cwd,
			env=env,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
		)
		stdout_thread = threading.Thread(
			target=self._consume_limited_stream,
			args=(proc.stdout, stdout_path, stdout_tail),
			daemon=True,
		)
		stderr_thread = threading.Thread(
			target=self._consume_limited_stream,
			args=(proc.stderr, stderr_path, stderr_tail),
			daemon=True,
		)
		stdout_thread.start()
		stderr_thread.start()
		try:
			return_code = proc.wait(timeout=timeout_seconds)
		except subprocess.TimeoutExpired as exc:
			proc.kill()
			return_code = proc.wait()
			stdout_thread.join()
			stderr_thread.join()
			raise subprocess.TimeoutExpired(
				cmd=exc.cmd,
				timeout=exc.timeout,
				output=stdout_tail.get("text", ""),
				stderr=stderr_tail.get("text", ""),
			) from exc
		stdout_thread.join()
		stderr_thread.join()
		return subprocess.CompletedProcess(
			args=cmd,
			returncode=return_code,
			stdout=stdout_tail.get("text", ""),
			stderr=stderr_tail.get("text", ""),
		)

	def _consume_limited_stream(
		self,
		stream: Any,
		path: Path,
		result: Dict[str, str],
	) -> None:
		limit = max(1, int(self.stdout_limit_chars))
		tail = bytearray()
		total = 0
		try:
			while True:
				chunk = stream.read(8192)
				if not chunk:
					break
				total += len(chunk)
				tail.extend(chunk)
				if len(tail) > limit:
					del tail[: len(tail) - limit]
		finally:
			try:
				stream.close()
			except Exception:
				pass
		text = tail.decode("utf-8", errors="replace")
		if total > limit:
			prefix = f"[truncated: captured last {limit} bytes of {total} bytes]\n"
			text = prefix + text[-max(0, limit - len(prefix)) :]
		self._write_text_artifact(path, text)
		result["text"] = text

	def _write_text_artifact(self, path: Path, text: str) -> None:
		value = str(text or "")
		if len(value) > self.stdout_limit_chars:
			value = value[-self.stdout_limit_chars :]
		path.write_text(value, encoding="utf-8")

	def _read_text_tail(self, path: Path) -> str:
		if not path.exists():
			return ""
		size = path.stat().st_size
		with path.open("rb") as handle:
			if size > self.stdout_limit_chars:
				handle.seek(-self.stdout_limit_chars, os.SEEK_END)
			return handle.read().decode("utf-8", errors="replace")

	def _read_lines(self, path: Path) -> Tuple[str, ...]:
		if not path.exists():
			return ()
		return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

	def _read_method_trace(self, path: Path) -> Tuple[Dict[str, Any], ...]:
		if not path.exists():
			return ()
		payload = json.loads(path.read_text(encoding="utf-8") or "[]")
		return tuple(dict(item) for item in payload if isinstance(item, dict))

	def _resolve_java_home(self) -> Optional[str]:
		for key in ("JADEX_JAVA_HOME", "JAVA17_HOME"):
			value = os.getenv(key)
			if value and (Path(value) / "bin" / "java").exists():
				return value
		if (Path(self.default_java_home) / "bin" / "java").exists():
			return self.default_java_home
		return os.getenv("JAVA_HOME")

	def _java_major(self, java_bin: str) -> Optional[int]:
		try:
			proc = subprocess.run(
				[java_bin, "-version"],
				text=True,
				capture_output=True,
				timeout=10,
				check=False,
			)
		except Exception:
			return None
		output = f"{proc.stdout}\n{proc.stderr}"
		match = re.search(r'version "(\d+)(?:\.(\d+))?', output)
		if match is None:
			return None
		first = int(match.group(1))
		if first == 1 and match.group(2):
			return int(match.group(2))
		return first

	@staticmethod
	def _normalise_name(value: str) -> str:
		return re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_")
