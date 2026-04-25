"""
Jadex-style BDI execution runner.

This backend evaluates the generated AgentSpeak(L) plan library as a set of
Jadex-style achievement-goal plans: failed plan candidates are excluded and the
goal is retried with alternative applicable candidates. It does not consume an
HTN planner trace; it executes the plan library directly over the benchmark
state transition model.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from evaluation.jason_runtime.runner import JasonRunner
from method_library.synthesis.schema import HTNMethodLibrary
from plan_library.models import AgentSpeakBodyStep, AgentSpeakPlan, PlanLibrary


Atom = Tuple[str, Tuple[str, ...]]
Bindings = Dict[str, str]
World = frozenset[Atom]


class JadexValidationError(RuntimeError):
	"""Raised when Jadex-style BDI execution fails."""

	def __init__(self, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
		super().__init__(message)
		self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class JadexValidationResult:
	"""Structured result for Jadex-style BDI execution."""

	status: str
	backend: str
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
			"action_path_count": len(self.action_path),
			"method_trace_count": len(self.method_trace),
			"failed_goals": list(self.failed_goals),
			"failure_class": self.failure_class,
			"consistency_checks": dict(self.consistency_checks),
			"artifacts": dict(self.artifacts),
			"timing_profile": dict(self.timing_profile),
		}


@dataclass(frozen=True)
class _SearchState:
	world: World
	bindings: Bindings
	action_path: Tuple[str, ...]
	method_trace: Tuple[Dict[str, Any], ...]


class JadexBDIRunner:
	"""Execute a plan library with Jadex-style retry and candidate exclusion."""

	backend_name = "JadexBDIRetry"
	runtime_output_artifact_limit_chars = 500_000
	max_search_nodes = 50_000_000
	max_goal_depth = 4_000
	failed_goal_record_limit = 64
	candidate_retry_limit = 1_000_000
	candidate_rejection_log_limit = 64
	max_debug_lines = 4_096
	max_debug_chars = 200_000

	def __init__(self, *, timeout_seconds: int = 1800) -> None:
		self.timeout_seconds = timeout_seconds
		self._helper = JasonRunner(timeout_seconds=timeout_seconds)
		self._schema_by_name: Dict[str, Dict[str, Any]] = {}
		self._objects_by_type: Dict[str, Tuple[str, ...]] = {}
		self._object_types: Dict[str, str] = {}
		self._type_parent_map: Dict[str, Optional[str]] = {}
		self._nodes_expanded = 0
		self._deadline = 0.0
		self._debug_lines: List[str] = []
		self._debug_chars = 0
		self._debug_truncated = False
		self._failed_goals: List[str] = []

	def validate(
		self,
		*,
		action_schemas: Sequence[Dict[str, Any]],
		method_library: HTNMethodLibrary | None = None,
		plan_library: PlanLibrary | None = None,
		seed_facts: Sequence[str] = (),
		runtime_objects: Sequence[str] = (),
		object_types: Optional[Dict[str, str]] = None,
		type_parent_map: Optional[Dict[str, Optional[str]]] = None,
		query_goals: Sequence[Any] = (),
		output_dir: str | Path,
		accept_candidate: Optional[
			Callable[[Tuple[str, ...], Tuple[Dict[str, Any], ...], Dict[str, Any]], bool]
		] = None,
		candidate_limit: Optional[int] = None,
	) -> JadexValidationResult:
		"""Execute query goals directly against the plan library."""

		total_start = time.perf_counter()
		output_path = Path(output_dir).resolve()
		output_path.mkdir(parents=True, exist_ok=True)
		self._deadline = total_start + max(1, int(self.timeout_seconds))
		self._nodes_expanded = 0
		self._debug_lines = []
		self._debug_chars = 0
		self._debug_truncated = False
		self._append_debug_line("jadex runtime ready")
		self._append_debug_line("execute start")
		self._failed_goals = []
		self._schema_by_name = self._build_schema_lookup(action_schemas)
		self._object_types = {
			str(key): str(value)
			for key, value in dict(object_types or {}).items()
			if str(key).strip() and str(value).strip()
		}
		self._type_parent_map = dict(type_parent_map or {})
		self._objects_by_type = self._build_objects_by_type(
			runtime_objects=runtime_objects,
			object_types=self._object_types,
		)

		if plan_library is None or not plan_library.plans:
			raise JadexValidationError(
				"Jadex runtime requires a non-empty plan library.",
				metadata={"failure_class": "missing_plan_library"},
			)

		world = frozenset(
			atom
			for fact in seed_facts
			if (atom := self._hddl_fact_to_atom(fact)) is not None
		)
		initial_state = _SearchState(
			world=world,
			bindings={},
			action_path=(),
			method_trace=(),
		)

		timing_profile: Dict[str, Any] = {}
		search_start = time.perf_counter()
		result_state: Optional[_SearchState] = None
		candidates_considered = 0
		candidates_rejected = 0
		candidate_limit_value = max(1, int(candidate_limit or self.candidate_retry_limit))
		try:
			for candidate_state in self._execute_goal_sequence(
				goals=tuple(self._normalise_query_goal(goal) for goal in query_goals),
				index=0,
				state=initial_state,
				plan_library=plan_library,
				goal_stack=(),
			):
				candidates_considered += 1
				candidate_metadata = {
					"candidate_index": candidates_considered,
					"candidate_limit": candidate_limit_value,
					"world_facts": list(self._render_world_facts(candidate_state.world)),
					"nodes_expanded": self._nodes_expanded,
				}
				if accept_candidate is None or accept_candidate(
					candidate_state.action_path,
					candidate_state.method_trace,
					candidate_metadata,
				):
					result_state = candidate_state
					break
				candidates_rejected += 1
				if candidates_rejected <= self.candidate_rejection_log_limit:
					self._append_debug_line(
						f"runtime candidate rejected candidate={candidates_considered}",
					)
				elif candidates_rejected == self.candidate_rejection_log_limit + 1:
					self._append_debug_line("runtime candidate rejection log limit reached")
				if candidates_considered >= candidate_limit_value:
					break
		except TimeoutError as exc:
			metadata = self._failure_metadata(
				status="failed",
				failure_class="jadex_runtime_timeout",
				output_path=output_path,
				timing_profile={
					"search_seconds": time.perf_counter() - search_start,
					"total_seconds": time.perf_counter() - total_start,
				},
				candidates_considered=candidates_considered,
				candidates_rejected=candidates_rejected,
			)
			raise JadexValidationError(str(exc), metadata=metadata) from exc
		timing_profile["search_seconds"] = time.perf_counter() - search_start
		timing_profile["total_seconds"] = time.perf_counter() - total_start

		if result_state is None:
			failure_class = (
				"jadex_candidate_rejected"
				if candidates_considered
				else "jadex_goal_failure"
			)
			metadata = self._failure_metadata(
				status="failed",
				failure_class=failure_class,
				output_path=output_path,
				timing_profile=timing_profile,
				candidates_considered=candidates_considered,
				candidates_rejected=candidates_rejected,
			)
			raise JadexValidationError(
				"Jadex-style BDI execution failed: no applicable retry path satisfied all goals.",
				metadata=metadata,
			)

		self._append_debug_line("execute success", force=True)
		artifacts = self._write_artifacts(
			output_path=output_path,
			stdout="\n".join(self._debug_lines) + "\n",
			stderr="",
			action_path=result_state.action_path,
			method_trace=result_state.method_trace,
			status="success",
			failure_class=None,
			timing_profile=timing_profile,
			candidates_considered=candidates_considered,
			candidates_rejected=candidates_rejected,
		)
		final_world_facts = tuple(self._render_world_facts(result_state.world))
		return JadexValidationResult(
			status="success",
			backend=self.backend_name,
			action_path=result_state.action_path,
			method_trace=result_state.method_trace,
			failed_goals=tuple(self._failed_goals),
			failure_class=None,
			consistency_checks={
				"success": True,
				"runtime_semantics": "jadex_retry_exclude",
				"nodes_expanded": self._nodes_expanded,
				"candidates_considered": candidates_considered,
				"candidates_rejected": candidates_rejected,
				"action_path_schema_replay": {
					"world_facts": list(final_world_facts),
				},
			},
			artifacts=artifacts,
			timing_profile=timing_profile,
		)

	def _failure_metadata(
		self,
		*,
		status: str,
		failure_class: str,
		output_path: Path,
		timing_profile: Dict[str, Any],
		candidates_considered: int = 0,
		candidates_rejected: int = 0,
	) -> Dict[str, Any]:
		self._append_debug_line("execute failed", force=True)
		artifacts = self._write_artifacts(
			output_path=output_path,
			stdout="\n".join(self._debug_lines) + "\n",
			stderr="",
			action_path=(),
			method_trace=(),
			status=status,
			failure_class=failure_class,
			timing_profile=timing_profile,
			candidates_considered=candidates_considered,
			candidates_rejected=candidates_rejected,
		)
		return {
			"status": status,
			"backend": self.backend_name,
			"action_path_count": 0,
			"method_trace_count": 0,
			"failed_goals": list(self._failed_goals[: self.failed_goal_record_limit]),
			"failure_class": failure_class,
			"consistency_checks": {
				"success": False,
				"runtime_semantics": "jadex_retry_exclude",
				"nodes_expanded": self._nodes_expanded,
				"candidates_considered": candidates_considered,
				"candidates_rejected": candidates_rejected,
			},
			"artifacts": artifacts,
			"timing_profile": dict(timing_profile),
		}

	def _write_artifacts(
		self,
		*,
		output_path: Path,
		stdout: str,
		stderr: str,
		action_path: Sequence[str],
		method_trace: Sequence[Dict[str, Any]],
		status: str,
		failure_class: Optional[str],
		timing_profile: Dict[str, Any],
		candidates_considered: int = 0,
		candidates_rejected: int = 0,
	) -> Dict[str, Any]:
		stdout_path = output_path / "jadex_stdout.txt"
		stderr_path = output_path / "jadex_stderr.txt"
		action_path_path = output_path / "action_path.txt"
		method_trace_path = output_path / "method_trace.json"
		validation_path = output_path / "jadex_validation.json"
		stdout_artifact, stdout_truncated = self._bounded_text(stdout)
		stderr_artifact, stderr_truncated = self._bounded_text(stderr)
		stdout_path.write_text(stdout_artifact)
		stderr_path.write_text(stderr_artifact)
		action_path_path.write_text("\n".join(action_path) + ("\n" if action_path else ""))
		method_trace_path.write_text(json.dumps(list(method_trace), indent=2))
		artifacts = {
			"runtime_backend": "jadex",
			"runtime_semantics": "jadex_retry_exclude",
			"jadex_stdout": str(stdout_path),
			"jadex_stderr": str(stderr_path),
			"action_path": str(action_path_path),
			"method_trace": str(method_trace_path),
			"jadex_validation": str(validation_path),
			"stdout_artifact_truncated": stdout_truncated,
			"stderr_artifact_truncated": stderr_truncated,
			"stdout_chars": len(stdout),
			"stderr_chars": len(stderr),
			"stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
			"stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
			"debug_log_truncated": self._debug_truncated,
			"debug_log_lines": len(self._debug_lines),
			"debug_log_chars": sum(len(line) + 1 for line in self._debug_lines),
			"nodes_expanded": self._nodes_expanded,
			"candidates_considered": candidates_considered,
			"candidates_rejected": candidates_rejected,
		}
		validation_path.write_text(
			json.dumps(
				{
					"status": status,
					"backend": self.backend_name,
					"failure_class": failure_class,
					"action_path_count": len(action_path),
					"method_trace_count": len(method_trace),
					"failed_goals": list(self._failed_goals[: self.failed_goal_record_limit]),
					"candidates_considered": candidates_considered,
					"candidates_rejected": candidates_rejected,
					"artifacts": artifacts,
					"timing_profile": dict(timing_profile),
				},
				indent=2,
			),
		)
		return artifacts

	def _execute_goal_sequence(
		self,
		*,
		goals: Sequence[Tuple[str, Tuple[str, ...]]],
		index: int,
		state: _SearchState,
		plan_library: PlanLibrary,
		goal_stack: Tuple[Tuple[str, Tuple[str, ...]], ...],
	) -> Iterator[_SearchState]:
		if index >= len(goals):
			yield state
			return
		task_name, args = goals[index]
		for next_state in self._execute_goal(
			task_name=task_name,
			args=args,
			state=state,
			plan_library=plan_library,
			goal_stack=goal_stack,
		):
			yield from self._execute_goal_sequence(
				goals=goals,
				index=index + 1,
				state=next_state,
				plan_library=plan_library,
				goal_stack=goal_stack,
			)

	def _execute_goal(
		self,
		*,
		task_name: str,
		args: Sequence[str],
		state: _SearchState,
		plan_library: PlanLibrary,
		goal_stack: Tuple[Tuple[str, Tuple[str, ...]], ...],
	) -> Iterator[_SearchState]:
		self._check_budget()
		resolved_args = tuple(self._resolve_value(arg, state.bindings) for arg in args)
		goal_key = (task_name, resolved_args)
		if goal_key in goal_stack:
			self._record_failed_goal(task_name, resolved_args)
			return
		applicable_plans = [
			(index, plan)
			for index, plan in enumerate(plan_library.plans)
			if self._normalise_name(plan.trigger.symbol) == self._normalise_name(task_name)
		]
		for _, plan in sorted(applicable_plans, key=self._plan_retry_priority):
			scope_prefix = f"?scope_{self._nodes_expanded}_{self._normalise_name(plan.plan_name)}_"
			trigger_arguments = tuple(
				self._scope_trigger_argument(argument, scope_prefix)
				for argument in plan.trigger.arguments
			)
			context = tuple(
				self._scope_context_literal(text, scope_prefix)
				for text in plan.context
			)
			body = tuple(self._scope_body_step(step, scope_prefix) for step in plan.body)
			for plan_bindings in self._bind_trigger_arguments(
				trigger_arguments,
				resolved_args,
				state.bindings,
			):
				for context_bindings in self._satisfy_context(context, plan_bindings, state.world):
					grounded_trace_args = tuple(
						self._resolve_value(arg, context_bindings)
						for arg in resolved_args
					)
					if any(self._is_variable(arg) for arg in grounded_trace_args):
						continue
					trace_entry = {
						"method_name": plan.plan_name,
						"task_args": list(grounded_trace_args),
					}
					self._append_debug_line(
						"runtime trace method flat "
						+ "|".join([plan.plan_name, *grounded_trace_args]),
					)
					next_state = _SearchState(
						world=state.world,
						bindings=context_bindings,
						action_path=state.action_path,
						method_trace=(*state.method_trace, trace_entry),
					)
					yield from self._execute_plan_body(
						body=body,
						index=0,
						state=next_state,
						plan_library=plan_library,
						goal_stack=(*goal_stack, goal_key),
					)
		self._record_failed_goal(task_name, resolved_args)

	def _execute_plan_body(
		self,
		*,
		body: Sequence[AgentSpeakBodyStep],
		index: int,
		state: _SearchState,
		plan_library: PlanLibrary,
		goal_stack: Tuple[Tuple[str, Tuple[str, ...]], ...],
	) -> Iterator[_SearchState]:
		if index >= len(body):
			yield state
			return
		step = body[index]
		step_args = tuple(self._resolve_value(arg, state.bindings) for arg in step.arguments)
		if step.kind == "subgoal":
			candidates = self._execute_goal(
				task_name=step.symbol,
				args=step_args,
				state=state,
				plan_library=plan_library,
				goal_stack=goal_stack,
			)
		elif step.kind == "action":
			candidates = self._execute_action(
				action_name=step.symbol,
				args=step_args,
				state=state,
			)
		else:
			candidates = iter(())
		for next_state in candidates:
			yield from self._execute_plan_body(
				body=body,
				index=index + 1,
				state=next_state,
				plan_library=plan_library,
				goal_stack=goal_stack,
			)

	def _execute_action(
		self,
		*,
		action_name: str,
		args: Sequence[str],
		state: _SearchState,
	) -> Iterator[_SearchState]:
		self._check_budget()
		schema = self._schema_by_name.get(self._normalise_name(action_name))
		if schema is None:
			return
		parameters = tuple(str(parameter) for parameter in (schema.get("parameters") or ()))
		local_parameter_keys = {
			self._variable_key(parameter)
			for parameter in parameters
			if self._is_variable(parameter)
		}
		base_bindings = {
			key: value
			for key, value in state.bindings.items()
			if key not in local_parameter_keys
		}
		for parameter, arg in zip(parameters, args):
			if not self._unify_term(str(parameter), str(arg), base_bindings):
				return
		clauses = schema.get("precondition_clauses") or []
		if not clauses:
			clauses = [schema.get("preconditions") or []]
		for clause in clauses:
			for action_bindings in self._satisfy_literals(clause, base_bindings, state.world):
				ground_args = tuple(
					self._resolve_value(parameter, action_bindings)
					for parameter in parameters[: len(args)]
				)
				if any(self._is_variable(arg) for arg in ground_args):
					continue
				next_world = set(state.world)
				for effect in self._helper._ordered_runtime_effects(schema.get("effects") or []):
					atom = self._ground_literal_atom(effect, action_bindings)
					if atom is None:
						continue
					if bool(effect.get("is_positive", True)):
						next_world.add(atom)
					else:
						next_world.discard(atom)
				source_name = str(schema.get("source_name") or action_name)
				rendered_action = self._render_action_call(source_name, ground_args)
				self._append_debug_line(f"runtime env action success {rendered_action}")
				yield _SearchState(
					world=frozenset(next_world),
					bindings={
						key: value
						for key, value in action_bindings.items()
						if key not in local_parameter_keys
					},
					action_path=(*state.action_path, rendered_action),
					method_trace=state.method_trace,
				)

	def _satisfy_context(
		self,
		context: Sequence[str],
		bindings: Bindings,
		world: World,
	) -> Iterator[Bindings]:
		literals = [literal for text in context if (literal := self._parse_context_literal(text))]
		yield from self._satisfy_literals(self._order_literals(literals), bindings, world)

	def _satisfy_literals(
		self,
		literals: Sequence[Dict[str, Any]],
		bindings: Bindings,
		world: World,
	) -> Iterator[Bindings]:
		self._check_deadline()
		if not literals:
			yield dict(bindings)
			return
		first, rest = literals[0], literals[1:]
		for next_bindings in self._satisfy_one_literal(first, bindings, world):
			yield from self._satisfy_literals(rest, next_bindings, world)

	def _satisfy_one_literal(
		self,
		literal: Dict[str, Any],
		bindings: Bindings,
		world: World,
	) -> Iterator[Bindings]:
		self._check_deadline()
		predicate = self._normalise_name(str(literal.get("predicate") or ""))
		args = tuple(str(arg) for arg in (literal.get("args") or ()))
		is_positive = bool(literal.get("is_positive", True))
		if predicate == "=" and len(args) == 2:
			next_bindings = dict(bindings)
			if self._unify_term(args[0], args[1], next_bindings):
				yield next_bindings
			return
		if predicate == "object_type" and len(args) == 2:
			yield from self._satisfy_object_type(args[0], args[1], bindings)
			return
		if not is_positive:
			if all(not self._is_variable(self._resolve_value(arg, bindings)) for arg in args):
				atom = (predicate, tuple(self._resolve_value(arg, bindings) for arg in args))
				if atom not in world:
					yield dict(bindings)
			return
		for fact_predicate, fact_args in world:
			self._check_deadline()
			if fact_predicate != predicate or len(fact_args) != len(args):
				continue
			next_bindings = dict(bindings)
			if all(self._unify_term(pattern, value, next_bindings) for pattern, value in zip(args, fact_args)):
				yield next_bindings

	def _satisfy_object_type(
		self,
		value_token: str,
		type_token: str,
		bindings: Bindings,
	) -> Iterator[Bindings]:
		expected_type = self._normalise_name(self._strip_quotes(type_token))
		resolved_value = self._resolve_value(value_token, bindings)
		if not self._is_variable(resolved_value):
			actual_type = self._object_types.get(self._strip_quotes(resolved_value))
			if actual_type is not None and self._is_subtype(actual_type, expected_type):
				yield dict(bindings)
			return
		for candidate in self._objects_by_type.get(expected_type, ()):
			self._check_deadline()
			next_bindings = dict(bindings)
			if self._unify_term(value_token, candidate, next_bindings):
				yield next_bindings

	def _bind_trigger_arguments(
		self,
		trigger_arguments: Sequence[str],
		args: Sequence[str],
		bindings: Bindings,
	) -> Iterator[Bindings]:
		if len(trigger_arguments) != len(args):
			return
		next_bindings = dict(bindings)
		for pattern, value in zip(trigger_arguments, args):
			if not self._unify_term(pattern, value, next_bindings):
				return
		yield next_bindings

	def _scope_body_step(
		self,
		step: AgentSpeakBodyStep,
		scope_prefix: str,
	) -> AgentSpeakBodyStep:
		return AgentSpeakBodyStep(
			kind=step.kind,
			symbol=step.symbol,
			arguments=tuple(self._scope_value(arg, scope_prefix) for arg in step.arguments),
		)

	def _scope_context_literal(self, text: str, scope_prefix: str) -> str:
		literal = self._parse_context_literal(text)
		if literal is None:
			return text
		predicate = str(literal.get("predicate") or "")
		args = [self._scope_value(str(arg), scope_prefix) for arg in (literal.get("args") or ())]
		if predicate == "=" and len(args) == 2:
			operator = "==" if bool(literal.get("is_positive", True)) else "\\=="
			return f"{args[0]} {operator} {args[1]}"
		prefix = "" if bool(literal.get("is_positive", True)) else "not "
		return f"{prefix}{predicate}({', '.join(args)})"

	def _scope_trigger_argument(self, pattern: str, scope_prefix: str) -> str:
		variable = self._trigger_variable(pattern)
		return self._scope_value(variable, scope_prefix)

	def _scope_value(self, value: str, scope_prefix: str) -> str:
		text = str(value or "").strip()
		if not self._is_variable(text):
			return text
		return f"{scope_prefix}{self._variable_key(text)}"

	def _ground_literal_atom(
		self,
		literal: Dict[str, Any],
		bindings: Bindings,
	) -> Optional[Atom]:
		predicate = self._normalise_name(str(literal.get("predicate") or ""))
		if not predicate or predicate == "=":
			return None
		args = tuple(self._resolve_value(str(arg), bindings) for arg in (literal.get("args") or ()))
		if any(self._is_variable(arg) for arg in args):
			return None
		return predicate, args

	def _parse_context_literal(self, text: str) -> Optional[Dict[str, Any]]:
		value = str(text or "").strip()
		if not value or value.lower() == "true":
			return None
		is_positive = True
		if value.startswith("not "):
			is_positive = False
			value = value[4:].strip()
		if value.startswith("!"):
			is_positive = False
			value = value[1:].strip()
		if "!=" in value:
			left, right = value.split("!=", 1)
			return {"predicate": "=", "args": [left.strip(), right.strip()], "is_positive": False}
		if "\\==" in value:
			left, right = value.split("\\==", 1)
			return {"predicate": "=", "args": [left.strip(), right.strip()], "is_positive": False}
		if "==" in value:
			left, right = value.split("==", 1)
			return {"predicate": "=", "args": [left.strip(), right.strip()], "is_positive": True}
		match = re.fullmatch(r"([A-Za-z0-9_-]+)(?:\((.*)\))?", value)
		if match is None:
			return None
		predicate = match.group(1)
		args_text = (match.group(2) or "").strip()
		args = self._split_args(args_text) if args_text else []
		return {"predicate": predicate, "args": args, "is_positive": is_positive}

	@staticmethod
	def _split_args(args_text: str) -> List[str]:
		return [part.strip() for part in str(args_text or "").split(",") if part.strip()]

	def _order_literals(self, literals: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
		def rank(literal: Dict[str, Any]) -> Tuple[int, int]:
			predicate = self._normalise_name(str(literal.get("predicate") or ""))
			if not bool(literal.get("is_positive", True)):
				return 3, 0
			if predicate == "object_type":
				return 2, 0
			if predicate == "=":
				return 1, 0
			constants = sum(
				0 if self._is_variable(str(arg)) else 1
				for arg in (literal.get("args") or ())
			)
			return 0, -constants
		return tuple(sorted((dict(item) for item in literals), key=rank))

	def _plan_retry_priority(
		self,
		indexed_plan: Tuple[int, AgentSpeakPlan],
	) -> Tuple[int, int, int, int, int, int]:
		index, plan = indexed_plan
		non_noop_actions = sum(
			1
			for step in plan.body
			if step.kind == "action" and self._normalise_name(step.symbol) != "nop"
		)
		subgoal_count = sum(1 for step in plan.body if step.kind == "subgoal")
		action_count = sum(1 for step in plan.body if step.kind == "action")
		negative_context_count = sum(
			1
			for text in plan.context
			if (
				(literal := self._parse_context_literal(text)) is not None
				and not bool(literal.get("is_positive", True))
			)
		)
		return (
			non_noop_actions,
			subgoal_count,
			action_count,
			negative_context_count,
			len(plan.context),
			index,
		)

	def _build_schema_lookup(
		self,
		action_schemas: Sequence[Dict[str, Any]],
	) -> Dict[str, Dict[str, Any]]:
		lookup: Dict[str, Dict[str, Any]] = {}
		for schema in action_schemas:
			for key in (schema.get("functor"), schema.get("source_name")):
				name = self._normalise_name(str(key or ""))
				if name:
					lookup.setdefault(name, dict(schema))
		return lookup

	def _build_objects_by_type(
		self,
		*,
		runtime_objects: Sequence[str],
		object_types: Dict[str, str],
	) -> Dict[str, Tuple[str, ...]]:
		accumulator: Dict[str, List[str]] = {}
		for object_name in runtime_objects:
			name = str(object_name).strip()
			if not name:
				continue
			for type_name in self._type_closure(object_types.get(name)):
				accumulator.setdefault(type_name, []).append(name)
		return {
			type_name: tuple(values)
			for type_name, values in accumulator.items()
		}

	def _type_closure(self, type_name: Optional[str]) -> Tuple[str, ...]:
		if not type_name:
			return ("object",)
		closure: List[str] = []
		seen: set[str] = set()
		cursor: Optional[str] = str(type_name).strip()
		while cursor and cursor not in seen:
			seen.add(cursor)
			closure.append(self._normalise_name(cursor))
			cursor = self._type_parent_map.get(cursor)
		if "object" not in closure:
			closure.append("object")
		return tuple(closure)

	def _is_subtype(self, actual_type: str, expected_type: str) -> bool:
		return expected_type in self._type_closure(actual_type)

	def _hddl_fact_to_atom(self, fact: str) -> Optional[Atom]:
		text = str(fact or "").strip()
		if not text.startswith("(") or not text.endswith(")"):
			return None
		inner = text[1:-1].strip()
		if not inner or inner.startswith("not "):
			return None
		tokens = inner.split()
		if not tokens:
			return None
		predicate = self._normalise_name(tokens[0])
		if predicate == "=":
			return None
		return predicate, tuple(tokens[1:])

	def _render_world_facts(self, world: World) -> Tuple[str, ...]:
		return tuple(
			f"{predicate}({', '.join(args)})" if args else predicate
			for predicate, args in sorted(world)
		)

	def _normalise_query_goal(self, goal: Any) -> Tuple[str, Tuple[str, ...]]:
		if isinstance(goal, dict):
			task_name = str(goal.get("task_name") or "").strip()
			args = tuple(str(arg).strip() for arg in (goal.get("args") or ()) if str(arg).strip())
		else:
			task_name = str(getattr(goal, "task_name", "") or "").strip()
			args = tuple(str(arg).strip() for arg in (getattr(goal, "args", ()) or ()) if str(arg).strip())
		return task_name, args

	@staticmethod
	def _trigger_variable(pattern: str) -> str:
		value = str(pattern or "").strip()
		if ":" in value:
			value = value.split(":", 1)[0].strip()
		return value

	def _unify_term(self, pattern: str, value: str, bindings: Bindings) -> bool:
		left = self._resolve_value(pattern, bindings)
		right = self._resolve_value(value, bindings)
		if self._is_variable(left):
			bindings[self._variable_key(left)] = self._strip_quotes(right)
			return True
		if self._is_variable(right):
			bindings[self._variable_key(right)] = self._strip_quotes(left)
			return True
		return self._strip_quotes(left) == self._strip_quotes(right)

	def _resolve_value(self, value: str, bindings: Bindings) -> str:
		text = self._strip_quotes(str(value or "").strip())
		seen: set[str] = set()
		while self._is_variable(text):
			key = self._variable_key(text)
			if key in seen or key not in bindings:
				break
			seen.add(key)
			text = self._strip_quotes(bindings[key])
		return text

	def _is_variable(self, value: str) -> bool:
		text = self._strip_quotes(str(value or "").strip())
		if text in self._object_types:
			return False
		return bool(text) and (
			text.startswith("?")
			or re.fullmatch(r"[A-Z][A-Za-z0-9_]*", text) is not None
		)

	@staticmethod
	def _variable_key(value: str) -> str:
		text = str(value or "").strip()
		return text[1:] if text.startswith("?") else text

	@staticmethod
	def _strip_quotes(value: str) -> str:
		text = str(value or "").strip()
		if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
			return text[1:-1]
		return text

	@staticmethod
	def _normalise_name(value: str) -> str:
		return re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_")

	@staticmethod
	def _render_action_call(action_name: str, args: Sequence[str]) -> str:
		if not args:
			return str(action_name)
		return f"{action_name}({', '.join(args)})"

	def _record_failed_goal(self, task_name: str, args: Sequence[str]) -> None:
		if len(self._failed_goals) >= self.failed_goal_record_limit:
			return
		payload = ",".join([str(task_name), *(str(arg) for arg in args)])
		if payload in self._failed_goals:
			return
		self._failed_goals.append(payload)
		self._append_debug_line(f"runtime goal failed fail_goal({payload})")

	def _check_budget(self) -> None:
		self._check_deadline()
		self._nodes_expanded += 1
		if self._nodes_expanded > self.max_search_nodes:
			raise TimeoutError(
				f"Jadex-style BDI execution exceeded node budget {self.max_search_nodes}.",
			)

	def _check_deadline(self) -> None:
		if time.perf_counter() >= self._deadline:
			raise TimeoutError("Jadex-style BDI execution timed out.")

	def _bounded_text(self, text: str) -> Tuple[str, bool]:
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

	def _append_debug_line(self, line: str, *, force: bool = False) -> None:
		text = str(line)
		line_chars = len(text) + 1
		if force:
			if len(self._debug_lines) < self.max_debug_lines:
				self._debug_lines.append(text)
			elif self._debug_lines:
				self._debug_lines[-1] = text
			else:
				self._debug_lines.append(text)
			self._debug_chars = sum(len(item) + 1 for item in self._debug_lines)
			return
		if self._debug_truncated:
			return
		if (
			len(self._debug_lines) >= self.max_debug_lines
			or self._debug_chars + line_chars > self.max_debug_chars
		):
			notice = (
				"runtime debug log limit reached "
				f"lines={len(self._debug_lines)} chars={self._debug_chars}"
			)
			if len(self._debug_lines) < self.max_debug_lines:
				self._debug_lines.append(notice)
			elif self._debug_lines:
				self._debug_lines[-1] = notice
			else:
				self._debug_lines.append(notice)
			self._debug_truncated = True
			self._debug_chars = sum(len(item) + 1 for item in self._debug_lines)
			return
		self._debug_lines.append(text)
		self._debug_chars += line_chars
