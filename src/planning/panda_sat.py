"""
PANDA SAT wrapper for the primary HTN evaluation baseline.

This module exports HDDL inputs, runs the PANDA parser, grounder, and SAT engine,
then returns the hierarchical plan artifact consumed by the IPC verifier.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from method_library.synthesis.schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
)
from planning.process_capture import read_full_process_output, run_subprocess_to_files
from planning.problem_encoding import PANDAProblemBuilder
from planning.plan_models import PANDAPlanResult, PANDAPlanStep


class PANDAPlanningError(RuntimeError):
	"""Raised when the PANDA planning backend fails."""

	def __init__(
		self,
		message: str,
		*,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		super().__init__(message)
		self.metadata = dict(metadata or {})


class PANDAPlanner:
	"""Invoke the SAT path of the PANDA PI toolchain on an HDDL problem."""

	_LINEARIZED_TOKEN_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
		("BAR_", "|"),
		("SEM_", ";"),
		("COM_", ","),
		("PLUS_", "+"),
		("MINUS_", "-"),
		("EXCLAMATION_", "!"),
		("US_", "_"),
	)

	def __init__(
		self,
		workspace: Optional[str | Path] = None,
		parser_cmd: str = "pandaPIparser",
		grounder_cmd: str = "pandaPIgrounder",
		engine_cmd: str = "pandaPIengine",
		problem_builder: Optional[PANDAProblemBuilder] = None,
	) -> None:
		self.workspace = Path(workspace).resolve() if workspace else None
		self.parser_cmd = parser_cmd
		self.grounder_cmd = grounder_cmd
		self.engine_cmd = engine_cmd
		self.problem_builder = problem_builder or PANDAProblemBuilder()

	def toolchain_available(self) -> bool:
		return all(
			self._resolve_command_head(command) is not None
			for command in (self.parser_cmd, self.grounder_cmd, self.engine_cmd)
		)

	def plan(
		self,
		domain: Any,
		method_library: HTNMethodLibrary,
		objects: Sequence[str],
		target_literal: Optional[HTNLiteral],
		task_name: str,
		transition_name: str,
		*,
		typed_objects: Optional[Sequence[Tuple[str, str]]] = None,
		htn_parameters: Optional[Sequence[Tuple[str, str]]] = None,
		task_args: Optional[Sequence[str]] = None,
		task_network: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
		task_network_ordered: bool = True,
		ordering_edges: Optional[Sequence[Tuple[str, str]]] = None,
		allow_empty_plan: bool = False,
		initial_facts: Optional[Sequence[str]] = None,
		goal_facts: Optional[Sequence[str]] = None,
		timeout_seconds: Optional[float] = None,
		solver_configs: Optional[Sequence[Dict[str, Any]]] = None,
		collect_all_candidates: bool = False,
	) -> PANDAPlanResult:
		self._require_toolchain()
		total_start = time.perf_counter()
		timing_profile: Dict[str, Any] = {}

		task_args_tuple = tuple(
			task_args if task_args is not None else (target_literal.args if target_literal else ())
		)
		task_network_entries = tuple(
			(str(network_task), tuple(str(arg) for arg in network_args))
			for network_task, network_args in (task_network or ())
		)
		domain_name = f"{domain.name}_{transition_name}"
		work_dir = self._resolve_work_dir(transition_name)
		work_dir.mkdir(parents=True, exist_ok=True)

		export_start = time.perf_counter()
		domain_hddl = self._build_domain_hddl(
			domain,
			method_library,
			domain_name,
			export_source_names=False,
		)
		problem_hddl = self.problem_builder.build_problem_hddl(
			domain=domain,
			domain_name=domain_name,
			objects=objects,
			typed_objects=typed_objects,
			htn_parameters=htn_parameters,
			task_name=task_name,
			task_args=task_args_tuple,
			task_network=task_network_entries,
			task_network_ordered=task_network_ordered,
			ordering_edges=ordering_edges,
			initial_facts=initial_facts,
			goal_facts=goal_facts,
		)
		timing_profile["hddl_export_seconds"] = time.perf_counter() - export_start

		return self._plan_from_hddl_texts(
			domain=domain,
			domain_hddl=domain_hddl,
			problem_hddl=problem_hddl,
			task_name=task_name,
			task_args_tuple=task_args_tuple,
			target_literal=target_literal,
			transition_name=transition_name,
			allow_empty_plan=allow_empty_plan,
			timeout_seconds=timeout_seconds,
			total_start=total_start,
			timing_profile=timing_profile,
			solver_configs=solver_configs,
			collect_all_candidates=collect_all_candidates,
		)

	def plan_hddl_files(
		self,
		*,
		domain: Any,
		domain_file: str | Path,
		problem_file: str | Path,
		task_name: str,
		transition_name: str,
		task_args: Optional[Sequence[str]] = None,
		target_literal: Optional[HTNLiteral] = None,
		allow_empty_plan: bool = False,
		timeout_seconds: Optional[float] = None,
		solver_configs: Optional[Sequence[Dict[str, Any]]] = None,
		collect_all_candidates: bool = False,
	) -> PANDAPlanResult:
		self._require_toolchain()
		total_start = time.perf_counter()
		timing_profile: Dict[str, Any] = {}
		domain_path = Path(domain_file).resolve()
		problem_path = Path(problem_file).resolve()
		read_start = time.perf_counter()
		domain_hddl = domain_path.read_text()
		problem_hddl = problem_path.read_text()
		timing_profile["read_input_files_seconds"] = time.perf_counter() - read_start
		task_args_tuple = tuple(task_args or ())
		return self._plan_from_hddl_texts(
			domain=domain,
			domain_hddl=domain_hddl,
			problem_hddl=problem_hddl,
			task_name=task_name,
			task_args_tuple=task_args_tuple,
			target_literal=target_literal,
			transition_name=transition_name,
			allow_empty_plan=allow_empty_plan,
			timeout_seconds=timeout_seconds,
			total_start=total_start,
			timing_profile=timing_profile,
			solver_configs=solver_configs,
			collect_all_candidates=collect_all_candidates,
		)

	def _plan_from_hddl_texts(
		self,
		*,
		domain: Any,
		domain_hddl: str,
		problem_hddl: str,
		task_name: str,
		task_args_tuple: Tuple[str, ...],
		target_literal: Optional[HTNLiteral],
		transition_name: str,
		allow_empty_plan: bool,
		timeout_seconds: Optional[float],
		total_start: float,
		timing_profile: Dict[str, Any],
		solver_configs: Optional[Sequence[Dict[str, Any]]],
		collect_all_candidates: bool,
	) -> PANDAPlanResult:
		work_dir = self._resolve_work_dir(transition_name)
		work_dir.mkdir(parents=True, exist_ok=True)

		domain_path = work_dir / "domain.hddl"
		problem_path = work_dir / "problem.hddl"
		parsed_path = work_dir / "problem.psas"
		grounded_path = work_dir / "problem.psas.grounded"
		raw_plan_path = work_dir / "plan.original"
		actual_plan_path = work_dir / "plan.actual"

		write_files_start = time.perf_counter()
		domain_path.write_text(domain_hddl)
		problem_path.write_text(problem_hddl)
		timing_profile["write_input_files_seconds"] = time.perf_counter() - write_files_start

		parser_start = time.perf_counter()
		parser_run = self._run_command(
			self._build_command(self.parser_cmd, str(domain_path), str(problem_path), str(parsed_path)),
			"parser",
			work_dir,
			timeout_seconds=timeout_seconds,
			output_label="parser",
		)
		timing_profile["parser_seconds"] = time.perf_counter() - parser_start
		grounder_start = time.perf_counter()
		grounder_run = self._run_command(
			self._build_command(self.grounder_cmd, str(parsed_path), str(grounded_path)),
			"grounder",
			work_dir,
			timeout_seconds=timeout_seconds,
			output_label="grounder",
		)
		timing_profile["grounder_seconds"] = time.perf_counter() - grounder_start
		engine_start = time.perf_counter()
		engine_attempts: List[Dict[str, Any]] = []
		conversion_total_seconds = 0.0
		parse_total_seconds = 0.0
		selected_engine_mode: Optional[str] = None
		selected_solver_id: Optional[str] = None
		engine_run: Optional[Dict[str, str]] = None
		raw_plan_text = ""
		actual_plan_text = ""
		conversion_stdout = ""
		conversion_stderr = ""
		steps: List[PANDAPlanStep] = []
		configured_solvers = tuple(
			dict(config)
			for config in (solver_configs or self.default_solver_configs())
		)
		for solver_config in configured_solvers:
			solver_id = str(solver_config.get("solver_id") or "").strip()
			engine_mode = str(solver_config.get("engine_mode") or "").strip() or None
			if not solver_id:
				raise ValueError("Each PANDA solver configuration requires a non-empty solver_id")
			attempt_start = time.perf_counter()
			attempt_timeout = self._effective_timeout_seconds(
				total_start=total_start,
				total_timeout_seconds=timeout_seconds,
				attempt_start=attempt_start,
				attempt_timeout_seconds=self._coerce_timeout_seconds(
					solver_config.get("timeout_seconds"),
				),
			)
			if attempt_timeout is not None and attempt_timeout <= 0:
				engine_attempts.append(
					{
						"solver_id": solver_id,
						"mode": engine_mode,
						"status": "skipped",
						"reason": "no_time_remaining",
					},
				)
				continue
			prepared_solver = self._prepare_solver_config(solver_config)
			if prepared_solver.get("status") == "skipped":
				engine_attempts.append(
					{
						"solver_id": solver_id,
						"mode": engine_mode,
						"status": "skipped",
						"reason": prepared_solver.get("reason"),
						"required_binary": prepared_solver.get("required_binary"),
					},
				)
				continue

			try:
				engine_command = self._build_engine_command(
					str(grounded_path),
					prepared_solver,
				)
				candidate_engine_run = self._run_command(
					engine_command,
					"engine",
					work_dir,
					timeout_seconds=attempt_timeout,
					output_label=f"engine_{solver_id}",
				)
			except PANDAPlanningError as exc:
				engine_attempts.append(
					{
						"solver_id": solver_id,
						"mode": engine_mode,
						"status": "failed",
						"seconds": time.perf_counter() - attempt_start,
						"error": str(exc),
						"command": engine_command,
						"metadata": dict(exc.metadata or {}),
					},
				)
				continue

			candidate_raw_plan_text = read_full_process_output(candidate_engine_run["stdout_path"])
			candidate_raw_plan_path = work_dir / f"plan.{solver_id}.raw.txt"
			candidate_actual_plan_path = work_dir / f"plan.{solver_id}.actual.txt"
			candidate_raw_plan_path.write_text(candidate_raw_plan_text)
			try:
				candidate_actual_plan_text, candidate_conversion_stdout, candidate_conversion_stderr = (
					self._convert_plan_output(
						raw_plan_path=candidate_raw_plan_path,
						actual_plan_path=candidate_actual_plan_path,
						work_dir=work_dir,
						timeout_seconds=self._effective_timeout_seconds(
							total_start=total_start,
							total_timeout_seconds=timeout_seconds,
							attempt_start=attempt_start,
							attempt_timeout_seconds=self._coerce_timeout_seconds(
								solver_config.get("timeout_seconds"),
							),
						),
					)
				)
			except PANDAPlanningError as exc:
				engine_attempts.append(
					{
						"solver_id": solver_id,
						"mode": engine_mode,
						"status": "failed",
						"seconds": time.perf_counter() - attempt_start,
						"error": str(exc),
						"command": engine_command,
						"metadata": dict(exc.metadata or {}),
						"raw_plan_path": str(candidate_raw_plan_path),
					},
				)
				continue

			candidate_conversion_seconds = time.perf_counter() - attempt_start
			conversion_total_seconds += candidate_conversion_seconds
			candidate_actual_plan_text = self._decode_linearized_plan_tokens(
				candidate_actual_plan_text or candidate_raw_plan_text,
			)
			candidate_actual_plan_path.write_text(candidate_actual_plan_text)
			parse_plan_start = time.perf_counter()
			candidate_steps = self._parse_plan_steps(
				candidate_actual_plan_text,
				domain,
			)
			parse_seconds = time.perf_counter() - parse_plan_start
			parse_total_seconds += parse_seconds
			candidate_has_hierarchical_trace = "->" in candidate_actual_plan_text
			has_executable_plan = bool(candidate_steps) or allow_empty_plan or candidate_has_hierarchical_trace
			candidate_action_path = [
				(
					f"{step.action_name}({', '.join(step.args)})"
					if step.args
					else str(step.action_name)
				)
				for step in candidate_steps
			]
			candidate_record = {
				"solver_id": solver_id,
				"mode": engine_mode,
				"status": "success" if has_executable_plan else "no_plan",
				"seconds": time.perf_counter() - attempt_start,
				"step_count": len(candidate_steps),
				"has_hierarchical_trace": candidate_has_hierarchical_trace,
				"stdout_head": "\n".join(candidate_engine_run["stdout"].splitlines()[:12]),
				"stderr_head": "\n".join(candidate_engine_run["stderr"].splitlines()[:12]),
				"command": engine_command,
				"raw_plan_path": str(candidate_raw_plan_path),
				"actual_plan_path": str(candidate_actual_plan_path),
				"action_path": candidate_action_path,
				"steps": [step.to_dict() for step in candidate_steps],
				"engine_stdout": candidate_engine_run["stdout"],
				"engine_stderr": candidate_engine_run["stderr"],
				"engine_stdout_path": candidate_engine_run["stdout_path"],
				"engine_stderr_path": candidate_engine_run["stderr_path"],
			}
			engine_attempts.append(candidate_record)
			if not has_executable_plan:
				continue

			if selected_engine_mode is None:
				selected_engine_mode = engine_mode
				selected_solver_id = solver_id
				engine_run = candidate_engine_run
				raw_plan_text = candidate_raw_plan_text
				actual_plan_text = candidate_actual_plan_text or candidate_raw_plan_text
				conversion_stdout = candidate_conversion_stdout
				conversion_stderr = candidate_conversion_stderr
				steps = candidate_steps
			if not collect_all_candidates:
				break

		timing_profile["engine_seconds"] = time.perf_counter() - engine_start
		timing_profile["conversion_seconds"] = conversion_total_seconds
		timing_profile["parse_plan_seconds"] = parse_total_seconds
		timing_profile["engine_mode_attempts"] = [
			{
				"solver_id": attempt.get("solver_id"),
				"mode": attempt.get("mode"),
				"status": attempt.get("status"),
				"seconds": attempt.get("seconds"),
				"step_count": attempt.get("step_count"),
				"reason": attempt.get("reason"),
			}
			for attempt in engine_attempts
		]
		timing_profile["engine_mode"] = selected_engine_mode
		timing_profile["solver_id"] = selected_solver_id

		if selected_engine_mode is None or engine_run is None:
			raise PANDAPlanningError(
				f"PANDA returned no executable primitive plan for {task_name}({', '.join(task_args_tuple)})",
				metadata={
					"backend": "pandaPI",
					"transition_name": transition_name,
					"task_name": task_name,
					"task_args": list(task_args_tuple),
					"work_dir": str(work_dir),
					"domain_hddl": domain_hddl,
					"problem_hddl": problem_hddl,
					"parser_stdout": parser_run["stdout"],
					"parser_stderr": parser_run["stderr"],
					"parser_stdout_path": parser_run["stdout_path"],
					"parser_stderr_path": parser_run["stderr_path"],
					"grounder_stdout": grounder_run["stdout"],
					"grounder_stderr": grounder_run["stderr"],
					"grounder_stdout_path": grounder_run["stdout_path"],
					"grounder_stderr_path": grounder_run["stderr_path"],
					"engine_attempts": list(engine_attempts),
				},
			)
		raw_plan_path.write_text(raw_plan_text)
		actual_plan_path.write_text(actual_plan_text or raw_plan_text)

		timing_profile["total_seconds"] = time.perf_counter() - total_start
		return PANDAPlanResult(
			task_name=task_name,
			task_args=task_args_tuple,
			target_literal=target_literal,
			engine_mode=selected_engine_mode,
			solver_id=selected_solver_id,
			steps=steps,
			domain_hddl=domain_hddl,
			problem_hddl=problem_hddl,
			parser_stdout=parser_run["stdout"] + conversion_stdout,
			parser_stderr=parser_run["stderr"] + conversion_stderr,
			grounder_stdout=grounder_run["stdout"],
			grounder_stderr=grounder_run["stderr"],
			engine_stdout=engine_run["stdout"],
			engine_stderr=engine_run["stderr"],
			parser_stdout_path=parser_run["stdout_path"],
			parser_stderr_path=parser_run["stderr_path"],
			grounder_stdout_path=grounder_run["stdout_path"],
			grounder_stderr_path=grounder_run["stderr_path"],
			engine_stdout_path=engine_run["stdout_path"],
			engine_stderr_path=engine_run["stderr_path"],
			raw_plan=raw_plan_text,
			actual_plan=actual_plan_text or raw_plan_text,
			work_dir=str(work_dir),
			timing_profile=timing_profile,
			solver_candidates=list(engine_attempts),
		)

	def default_solver_configs(self) -> Tuple[Dict[str, Any], ...]:
		"""Return the single SAT engine configuration used by lifted_panda_sat."""

		return (self._solver_config_by_id("sat"),)

	def _solver_config_by_id(self, solver_id: str) -> Dict[str, Any]:
		configs = {
			"sat": {
				"solver_id": "sat",
				"engine_mode": "sat",
				"engine_args": ("-s",),
				"timeout_seconds": 90.0,
			},
		}
		try:
			return dict(configs[solver_id])
		except KeyError as exc:
			raise ValueError(f"Unknown PANDA solver configuration '{solver_id}'") from exc

	def _prepare_solver_config(self, solver_config: Dict[str, Any]) -> Dict[str, Any]:
		prepared = dict(solver_config)
		engine_args = tuple(str(value) for value in (prepared.get("engine_args") or ()))
		engine_cmd = str(prepared.get("engine_cmd") or self.engine_cmd)
		prepared["engine_args"] = engine_args
		prepared["engine_cmd"] = engine_cmd
		prepared["command"] = self._build_command(
			engine_cmd,
			*engine_args,
			"{grounded_path}",
		)
		return prepared

	def _build_engine_command(
		self,
		grounded_path: str,
		solver_config: Dict[str, Any],
	) -> List[str]:
		engine_args = tuple(str(value) for value in (solver_config.get("engine_args") or ()))
		engine_cmd = str(solver_config.get("engine_cmd") or self.engine_cmd)
		return self._build_command(
			engine_cmd,
			*engine_args,
			grounded_path,
		)

	@staticmethod
	def _remaining_timeout_seconds(
		total_start: float,
		timeout_seconds: Optional[float],
	) -> Optional[float]:
		if timeout_seconds is None:
			return None
		elapsed = time.perf_counter() - total_start
		return max(timeout_seconds - elapsed, 0.0)

	@staticmethod
	def _coerce_timeout_seconds(value: Any) -> Optional[float]:
		if value in (None, ""):
			return None
		return max(float(value), 0.0)

	@classmethod
	def _effective_timeout_seconds(
		cls,
		*,
		total_start: float,
		total_timeout_seconds: Optional[float],
		attempt_start: float,
		attempt_timeout_seconds: Optional[float],
	) -> Optional[float]:
		total_remaining = cls._remaining_timeout_seconds(total_start, total_timeout_seconds)
		attempt_remaining = cls._remaining_timeout_seconds(attempt_start, attempt_timeout_seconds)
		if total_remaining is None:
			return attempt_remaining
		if attempt_remaining is None:
			return total_remaining
		return min(total_remaining, attempt_remaining)

	def _convert_plan_output(
		self,
		*,
		raw_plan_path: Path,
		actual_plan_path: Path,
		work_dir: Path,
		timeout_seconds: Optional[float],
	) -> Tuple[str, str, str]:
		actual_plan_text = ""
		conversion_stdout = ""
		conversion_stderr = ""
		conversion_command = self._build_command(
			self.parser_cmd,
			"-c",
			str(raw_plan_path),
			str(actual_plan_path),
		)
		try:
			conversion_result = self._run_subprocess(
				conversion_command,
				work_dir,
				timeout_seconds=timeout_seconds,
				output_label=f"convert_{raw_plan_path.stem}",
			)
			conversion_stdout = conversion_result["stdout"]
			conversion_stderr = conversion_result["stderr"]
		except PANDAPlanningError as exc:
			if exc.metadata.get("stage") != "subprocess_timeout":
				raise
			raise PANDAPlanningError(
				"PANDA plan conversion timed out",
				metadata={
					"backend": "pandaPI",
					"stage": "conversion",
					"command": list(conversion_command),
					"stdout": exc.metadata.get("stdout", ""),
					"stderr": exc.metadata.get("stderr", ""),
					"stdout_path": exc.metadata.get("stdout_path"),
					"stderr_path": exc.metadata.get("stderr_path"),
					"work_dir": str(work_dir),
					"timeout_seconds": timeout_seconds,
				},
			) from exc
		if conversion_result["returncode"] == 0 and actual_plan_path.exists():
			actual_plan_text = actual_plan_path.read_text()
		elif raw_plan_path.exists():
			actual_plan_text = raw_plan_path.read_text()
		return actual_plan_text, conversion_stdout, conversion_stderr

	def extract_method_trace(self, plan_text: str) -> List[Dict[str, Any]]:
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
			child_order = [
				first_primitive_id(child_id)
				for child_id in node["children"]
			]
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

	def _resolve_work_dir(self, transition_name: str) -> Path:
		if self.workspace is None:
			return Path(tempfile.mkdtemp(prefix="panda_method_synthesis_")) / transition_name
		return self.workspace / "panda" / transition_name

	def _require_toolchain(self) -> None:
		missing = [
			command
			for command in (self.parser_cmd, self.grounder_cmd, self.engine_cmd)
			if self._resolve_command_head(command) is None
		]
		if not missing:
			return

		raise PANDAPlanningError(
			"PANDA toolchain is not available on PATH. Missing commands: "
			+ ", ".join(missing),
			metadata={
				"backend": "pandaPI",
				"missing_commands": missing,
			},
		)

	def _build_domain_hddl(
		self,
		domain: Any,
		method_library: HTNMethodLibrary,
		domain_name: str,
		*,
		export_source_names: bool = False,
	) -> str:
		task_type_map = self._infer_task_type_map(domain, method_library)
		requirements = list(domain.requirements or [])
		if ":hierarchy" not in requirements:
			requirements.append(":hierarchy")

		lines = [f"(define (domain {domain_name})"]
		if requirements:
			lines.append(f"  (:requirements {' '.join(requirements)})")
		if domain.types:
			lines.append(f"  (:types {' '.join(domain.types)})")
		lines.append("")
		lines.append("  (:predicates")
		for predicate in domain.predicates:
			lines.append(f"    ({predicate.name}{self._render_signature_parameters(predicate.parameters)})")
		lines.append("  )")
		lines.append("")

		rendered_task_names: set[tuple[str, tuple[str, ...]]] = set()
		for task in method_library.compound_tasks:
			exported_task_name = self._export_task_name(task, export_source_names)
			task_signature = task_type_map.get(task.name) or task_type_map.get(task.source_name or "")
			rendered_key = (exported_task_name, tuple(task_signature or ()))
			if rendered_key in rendered_task_names:
				continue
			rendered_task_names.add(rendered_key)
			lines.append(f"  (:task {exported_task_name}")
			lines.append(
				f"    :parameters ({self._render_typed_variables(task.parameters, domain.types, task_signature)})"
			)
			lines.append("  )")
			lines.append("")

		action_name_map = self._action_export_name_map(domain, method_library)
		task_lookup = {task.name: task for task in method_library.compound_tasks}
		for method in method_library.methods:
			lines.extend(
				self._render_method(
					method,
					task_lookup,
					action_name_map,
					domain.types,
					task_type_map,
					self._predicate_type_map(domain),
					self._action_type_map(domain),
					export_source_names=export_source_names,
				)
			)
			lines.append("")

		for action in domain.actions:
			lines.append(f"  (:action {action.name}")
			lines.append(f"    :parameters ({self._render_signature_parameters(action.parameters).strip()})")
			lines.append(f"    :precondition {action.preconditions}")
			lines.append(f"    :effect {action.effects}")
			lines.append("  )")
			lines.append("")

		if lines[-1] == "":
			lines.pop()
		lines.append(")")
		return "\n".join(lines) + "\n"

	def _render_method(
		self,
		method: HTNMethod,
		task_lookup: Dict[str, Any],
		action_name_map: Dict[str, str],
		domain_types: Sequence[str],
		task_type_map: Dict[str, Tuple[str, ...]],
		predicate_types: Dict[str, Tuple[str, ...]],
		action_types: Dict[str, Tuple[str, ...]],
		*,
		export_source_names: bool = False,
	) -> List[str]:
		method_parameters = self._method_parameter_order(method, task_lookup.get(method.task_name))
		task_signature = task_type_map.get(method.task_name, ())
		method_parameter_types = self._infer_method_parameter_types(
			method,
			method_parameters,
			task_signature,
			predicate_types,
			action_types,
			task_type_map,
			self._default_parameter_type(domain_types),
		)
		variable_map = {
			name: self._render_variable(name)
			for name in method_parameters
		}
		task_schema = task_lookup.get(method.task_name)
		task_args = method.task_args or self._default_method_task_args(method, task_schema)

		lines = [f"  (:method {method.method_name}"]
		lines.append(
			f"    :parameters ({self._render_typed_variables(method_parameters, domain_types, method_parameter_types)})"
		)
		lines.append(
			f"    :task ({self._export_task_name(task_schema, export_source_names, fallback=method.task_name)}"
			f"{self._render_invocation_tokens(task_args, variable_map)})"
		)
		lines.append(
			f"    :precondition {self._render_literal_conjunction(method.context, variable_map)}"
		)
		subtasks_keyword = ":subtasks" if method.ordering else ":ordered-subtasks"
		lines.append(f"    {subtasks_keyword} (and")
		for step in method.subtasks:
			step_name = self._render_method_step_name(step, task_lookup, action_name_map)
			if step.kind == "compound":
				step_name = self._render_method_step_name(
					step,
					task_lookup,
					action_name_map,
					export_source_names=export_source_names,
				)
			lines.append(
				f"      ({step.step_id} ({step_name}"
				f"{self._render_invocation_tokens(step.args, variable_map)}))"
			)
		lines.append("    )")
		if method.ordering:
			lines.append("    :ordering (and")
			for before, after in method.ordering:
				lines.append(f"      (< {before} {after})")
			lines.append("    )")
		lines.append("  )")
		return lines

	@staticmethod
	def _default_method_task_args(
		method: HTNMethod,
		task_schema: Any,
	) -> Tuple[str, ...]:
		if method.task_args:
			return tuple(method.task_args)
		if task_schema is None:
			return tuple(method.parameters)
		declared_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
		if not declared_parameters:
			return ()
		leading_parameters = tuple(method.parameters[: len(declared_parameters)])
		if len(leading_parameters) == len(declared_parameters):
			return leading_parameters
		return declared_parameters

	def _render_method_step_name(
		self,
		step: Any,
		task_lookup: Dict[str, Any],
		action_name_map: Dict[str, str],
		*,
		export_source_names: bool = False,
	) -> str:
		if getattr(step, "kind", None) == "primitive":
			action_name = str(
				getattr(step, "action_name", None) or getattr(step, "task_name", "") or "",
			).strip()
			if not action_name:
				return ""
			if action_name in action_name_map:
				return action_name_map[action_name]
			sanitized_name = self._sanitize_name(action_name)
			if sanitized_name in action_name_map:
				return action_name_map[sanitized_name]
			return action_name.replace("_", "-")
		task_schema = task_lookup.get(step.task_name)
		return self._export_task_name(task_schema, export_source_names, fallback=step.task_name)

	def _action_export_name_map(
		self,
		domain: Any,
		method_library: HTNMethodLibrary,
	) -> Dict[str, str]:
		mapping: Dict[str, str] = {}

		def register(alias: str, export_name: str) -> None:
			alias_text = str(alias or "").strip()
			export_text = str(export_name or "").strip()
			if not alias_text or not export_text:
				return
			mapping.setdefault(alias_text, export_text)

		for action in domain.actions:
			action_name = str(getattr(action, "name", "") or "").strip()
			if not action_name:
				continue
			register(action_name, action_name)
			register(self._sanitize_name(action_name), action_name)

		for task in getattr(method_library, "primitive_tasks", ()) or ():
			task_name = str(getattr(task, "name", "") or "").strip()
			source_name = str(getattr(task, "source_name", "") or "").strip()
			export_name = (
				mapping.get(source_name)
				or mapping.get(task_name)
				or mapping.get(self._sanitize_name(task_name))
				or source_name
				or task_name.replace("_", "-")
			)
			register(task_name, export_name)
			register(self._sanitize_name(task_name), export_name)
			if source_name:
				register(source_name, export_name)
				register(self._sanitize_name(source_name), export_name)

		return mapping

	@staticmethod
	def _export_task_name(task_schema: Any, export_source_names: bool, *, fallback: str = "") -> str:
		if task_schema is None:
			return fallback
		if export_source_names and getattr(task_schema, "source_name", None):
			return getattr(task_schema, "source_name")
		return getattr(task_schema, "name", None) or fallback

	def _parse_plan_steps(self, plan_text: str, domain: Any) -> List[PANDAPlanStep]:
		action_names = {action.name for action in domain.actions}
		steps: List[PANDAPlanStep] = []

		for line in plan_text.splitlines():
			plain_match = re.fullmatch(
				r"\s*\d+\s+([a-zA-Z][a-zA-Z0-9_-]*)(?:\s+(.*?))?\s*",
				line,
			)
			if plain_match and "->" not in line:
				action_name = plain_match.group(1)
				if action_name not in action_names:
					continue
				args_blob = plain_match.group(2) or ""
				args = tuple(token for token in args_blob.strip().split() if token)
				steps.append(
					PANDAPlanStep(
						task_name=self._sanitize_name(action_name),
						action_name=action_name,
						args=args,
						source_line=line.strip(),
					)
				)
				continue

			matches = re.findall(r"\(([a-zA-Z][a-zA-Z0-9_-]*)([^)]*)\)", line)
			for action_name, args_blob in matches:
				if action_name not in action_names:
					continue
				args = tuple(token for token in args_blob.strip().split() if token)
				steps.append(
					PANDAPlanStep(
						task_name=self._sanitize_name(action_name),
						action_name=action_name,
						args=args,
						source_line=line.strip(),
					)
				)
				break

		return steps

	@classmethod
	def _decode_linearized_plan_tokens(cls, plan_text: str) -> str:
		"""Decode PANDA linearizer-safe identifiers before parsing or verifying plans."""
		decoded = str(plan_text or "")
		for source, target in cls._LINEARIZED_TOKEN_REPLACEMENTS:
			decoded = decoded.replace(source, target)
		return decoded

	def _run_command(
		self,
		command: Sequence[str],
		stage: str,
		work_dir: Path,
		*,
		timeout_seconds: Optional[float] = None,
		output_label: Optional[str] = None,
	) -> Dict[str, Any]:
		try:
			result = self._run_subprocess(
				command,
				work_dir,
				timeout_seconds=timeout_seconds,
				output_label=output_label or stage,
			)
		except PANDAPlanningError as exc:
			if exc.metadata.get("stage") != "subprocess_timeout":
				raise
			raise PANDAPlanningError(
				f"PANDA {stage} step timed out",
				metadata={
					"backend": "pandaPI",
					"stage": stage,
					"command": list(command),
					"stdout": exc.metadata.get("stdout", ""),
					"stderr": exc.metadata.get("stderr", ""),
					"stdout_path": exc.metadata.get("stdout_path"),
					"stderr_path": exc.metadata.get("stderr_path"),
					"work_dir": str(work_dir),
					"timeout_seconds": timeout_seconds,
				},
			) from exc
		if result["returncode"] == 0:
			return dict(result)

		raise PANDAPlanningError(
			f"PANDA {stage} step failed with exit code {result['returncode']}",
			metadata={
				"backend": "pandaPI",
				"stage": stage,
				"command": list(command),
				"stdout": result["stdout"],
				"stderr": result["stderr"],
				"stdout_path": result["stdout_path"],
				"stderr_path": result["stderr_path"],
				"work_dir": str(work_dir),
			},
		)

	def _run_subprocess(
		self,
		command: Sequence[str],
		work_dir: Path,
		*,
		timeout_seconds: Optional[float] = None,
		output_label: str,
	) -> Dict[str, Any]:
		result = run_subprocess_to_files(
			command,
			work_dir=work_dir,
			output_label=output_label,
			timeout_seconds=timeout_seconds,
		)
		if result["timed_out"]:
			raise PANDAPlanningError(
				"PANDA subprocess timed out",
				metadata={
					"stage": "subprocess_timeout",
					"command": list(command),
					"stdout": result["stdout"],
					"stderr": result["stderr"],
					"stdout_path": result["stdout_path"],
					"stderr_path": result["stderr_path"],
					"work_dir": str(work_dir),
					"timeout_seconds": timeout_seconds,
				},
			)
		return result

	@staticmethod
	def _parse_plan_node_ids(tokens: Sequence[str]) -> List[int]:
		node_ids: List[int] = []
		for token in tokens:
			try:
				node_ids.append(int(str(token)))
			except ValueError:
				continue
		return node_ids

	@staticmethod
	def _sanitize_name(name: str) -> str:
		return name.replace("-", "_")

	@staticmethod
	def _command_head(command: str) -> str:
		return shlex.split(command)[0]

	def _build_command(self, command: str, *args: str) -> List[str]:
		parts = shlex.split(command)
		resolved = self._resolve_command_head(command)
		if resolved:
			parts[0] = resolved
		return [*parts, *args]

	def _resolve_command_head(self, command: str) -> Optional[str]:
		head = self._command_head(command)
		if os.path.sep in head:
			if Path(head).is_file() and os.access(head, os.X_OK):
				return head
			return None

		for directory in self._default_command_dirs():
			candidate = directory / head
			if candidate.is_file() and os.access(candidate, os.X_OK):
				return str(candidate)

		resolved = shutil.which(head)
		if resolved:
			return resolved

		return None

	def _default_command_dirs(self) -> Tuple[Path, ...]:
		directories: List[Path] = []
		panda_home = os.getenv("PANDA_PI_HOME")
		if panda_home:
			directories.append(Path(panda_home) / "bin")
		panda_bin = os.getenv("PANDA_PI_BIN")
		if panda_bin:
			directories.append(Path(panda_bin))
		directories.append(Path.home() / ".local" / "pandaPI-full" / "bin")
		directories.append(Path.home() / ".local" / "pandaPI" / "bin")

		unique: List[Path] = []
		for directory in directories:
			if directory not in unique:
				unique.append(directory)
		return tuple(unique)

	@staticmethod
	def _render_signature_parameters(parameters: Iterable[str]) -> str:
		text = " ".join(str(item) for item in parameters if str(item).strip())
		return f" {text}" if text else ""

	def _method_parameter_order(
		self,
		method: HTNMethod,
		task_schema: Any,
	) -> Tuple[str, ...]:
		ordered: List[str] = []
		seen: set[str] = set()

		def add(token: str) -> None:
			if not self._is_variable_token(token):
				return
			canonical = token.lstrip("?")
			if canonical in seen:
				return
			seen.add(canonical)
			ordered.append(canonical)

		def add_schema_token(token: str) -> None:
			if not token:
				return
			canonical = token.lstrip("?")
			if canonical in seen:
				return
			seen.add(canonical)
			ordered.append(canonical)

		task_binding_tokens: Sequence[str] = ()
		if task_schema is not None:
			task_binding_tokens = self._task_binding_parameters(
				method,
				len(tuple(getattr(task_schema, "parameters", ()) or ())),
			)
		for token in task_binding_tokens:
			add_schema_token(token)
		for token in method.parameters:
			add_schema_token(token)
		for literal in method.context:
			for token in literal.args:
				add(token)
		for step in method.subtasks:
			for token in step.args:
				add(token)
			if step.literal:
				for token in step.literal.args:
					add(token)
			for literal in step.preconditions:
				for token in literal.args:
					add(token)
			for literal in step.effects:
				for token in literal.args:
					add(token)

		return tuple(ordered)

	def _render_typed_variables(
		self,
		parameters: Sequence[str],
		domain_types: Sequence[str],
		type_signature: Optional[Sequence[Optional[str]]] = None,
	) -> str:
		default_type = self._default_parameter_type(domain_types)
		return " ".join(
			f"{self._render_variable(parameter)} - "
			f"{self._signature_type_at(type_signature, index, default_type)}"
			for index, parameter in enumerate(parameters)
		)

	def _infer_task_type_map(
		self,
		domain: Any,
		method_library: HTNMethodLibrary,
	) -> Dict[str, Tuple[str, ...]]:
		predicate_types = self._predicate_type_map(domain)
		action_types = self._action_type_map(domain)
		declared_task_types = self._declared_task_type_map(domain)
		default_type = self._default_parameter_type(getattr(domain, "types", ()))
		methods_by_task: Dict[str, List[HTNMethod]] = {}
		task_aliases: Dict[str, str] = {}
		inferred: Dict[str, List[Optional[str]]] = {}

		for task in method_library.compound_tasks:
			task_aliases[task.name] = task.name
			if task.source_name:
				task_aliases[task.source_name] = task.name
			inferred[task.name] = [None] * len(task.parameters)
			declared_signature = declared_task_types.get(task.source_name or task.name)
			if declared_signature and len(declared_signature) == len(task.parameters):
				inferred[task.name] = list(declared_signature)
				continue
			if len(task.source_predicates) == 1:
				predicate_signature = predicate_types.get(task.source_predicates[0])
				if predicate_signature and len(predicate_signature) == len(task.parameters):
					inferred[task.name] = list(predicate_signature)

		for method in method_library.methods:
			methods_by_task.setdefault(method.task_name, []).append(method)

		changed = True
		while changed:
			changed = False
			for task in method_library.compound_tasks:
				task_signature = inferred.get(task.name, [])
				for method in methods_by_task.get(task.name, ()):
					for index, parameter in enumerate(
						self._task_binding_parameters(method, len(task.parameters))
					):
						if index >= len(task_signature) or task_signature[index] is not None:
							continue
						candidates = self._variable_type_candidates(
							method,
							parameter,
							predicate_types,
							action_types,
							inferred,
							task_aliases,
						)
						if not candidates:
							continue
						task_signature[index] = candidates[0]
						changed = True

		resolved: Dict[str, Tuple[str, ...]] = {}
		for task in method_library.compound_tasks:
			signature = tuple(type_name or default_type for type_name in inferred.get(task.name, ()))
			resolved[task.name] = signature
			if task.source_name:
				resolved[task.source_name] = signature
		return resolved

	def _infer_method_parameter_types(
		self,
		method: HTNMethod,
		ordered_parameters: Sequence[str],
		task_signature: Sequence[str],
		predicate_types: Dict[str, Tuple[str, ...]],
		action_types: Dict[str, Tuple[str, ...]],
		task_type_map: Dict[str, Tuple[str, ...]],
		default_type: str,
	) -> Tuple[str, ...]:
		variable_types: Dict[str, str] = {}
		for index, parameter in enumerate(self._task_binding_parameters(method, len(task_signature))):
			if index >= len(task_signature):
				continue
			variable_types[self._canonical_symbol(parameter)] = task_signature[index]

		for parameter in ordered_parameters:
			canonical = self._canonical_symbol(parameter)
			if canonical in variable_types:
				continue
			candidates = self._variable_type_candidates(
				method,
				parameter,
				predicate_types,
				action_types,
				task_type_map,
				{},
			)
			variable_types[canonical] = candidates[0] if candidates else default_type

		return tuple(variable_types.get(self._canonical_symbol(parameter), default_type) for parameter in ordered_parameters)

	def _variable_type_candidates(
		self,
		method: HTNMethod,
		variable: str,
		predicate_types: Dict[str, Tuple[str, ...]],
		action_types: Dict[str, Tuple[str, ...]],
		task_type_map: Dict[str, Sequence[Optional[str]]],
		task_aliases: Dict[str, str],
	) -> List[str]:
		canonical_variable = self._canonical_symbol(variable)
		candidates: List[str] = []

		def add_candidate(candidate: Optional[str]) -> None:
			if candidate and candidate not in candidates:
				candidates.append(candidate)

		def collect_from_literal(literal: HTNLiteral) -> None:
			signature = predicate_types.get(literal.predicate)
			if not signature:
				return
			for index, arg in enumerate(literal.args):
				if self._canonical_symbol(arg) == canonical_variable and index < len(signature):
					add_candidate(signature[index])

		for literal in method.context:
			collect_from_literal(literal)

		for step in method.subtasks:
			if step.kind == "primitive":
				step_signature = action_types.get(step.action_name or "") or action_types.get(step.task_name)
			else:
				internal_name = task_aliases.get(step.task_name, step.task_name)
				step_signature = task_type_map.get(internal_name) or task_type_map.get(step.task_name)
			if step_signature:
				for index, arg in enumerate(step.args):
					if self._canonical_symbol(arg) == canonical_variable and index < len(step_signature):
						add_candidate(step_signature[index])
			for literal in (step.literal, *step.preconditions, *step.effects):
				if literal is not None:
					collect_from_literal(literal)

		return candidates

	def _predicate_type_map(self, domain: Any) -> Dict[str, Tuple[str, ...]]:
		mapping: Dict[str, Tuple[str, ...]] = {}
		for predicate in getattr(domain, "predicates", []) or ():
			mapping[predicate.name] = tuple(
				self._parameter_type(parameter)
				for parameter in getattr(predicate, "parameters", ()) or ()
			)
		return mapping

	def _action_type_map(self, domain: Any) -> Dict[str, Tuple[str, ...]]:
		mapping: Dict[str, Tuple[str, ...]] = {}
		for action in getattr(domain, "actions", []) or ():
			type_signature = tuple(
				self._parameter_type(parameter)
				for parameter in getattr(action, "parameters", ()) or ()
			)
			mapping[action.name] = type_signature
			mapping[self._sanitize_name(action.name)] = type_signature
		return mapping

	def _declared_task_type_map(self, domain: Any) -> Dict[str, Tuple[str, ...]]:
		mapping: Dict[str, Tuple[str, ...]] = {}
		for task in getattr(domain, "tasks", []) or ():
			mapping[str(task.name)] = tuple(
				self._parameter_type(parameter)
				for parameter in getattr(task, "parameters", ()) or ()
			)
		return mapping

	@staticmethod
	def _task_binding_parameters(
		method: HTNMethod,
		task_arity: int,
	) -> Tuple[str, ...]:
		task_binding_parameters = tuple(method.task_args or ())
		if len(task_binding_parameters) == task_arity:
			return task_binding_parameters
		leading_parameters = tuple(method.parameters[:task_arity])
		if len(leading_parameters) == task_arity:
			return leading_parameters
		return task_binding_parameters

	@staticmethod
	def _parameter_type(parameter: Any) -> str:
		text = str(parameter or "").strip()
		if "-" not in text:
			return "object"
		_, type_name = text.split("-", 1)
		resolved = type_name.strip()
		return resolved or "object"

	@staticmethod
	def _canonical_symbol(token: str) -> str:
		return str(token or "").strip().lstrip("?")

	@staticmethod
	def _default_parameter_type(domain_types: Sequence[str]) -> str:
		if "object" in domain_types:
			return "object"
		return domain_types[0] if domain_types else "object"

	@staticmethod
	def _signature_type_at(
		type_signature: Optional[Sequence[Optional[str]]],
		index: int,
		default_type: str,
	) -> str:
		if type_signature is None or index >= len(type_signature):
			return default_type
		return type_signature[index] or default_type

	def _render_literal_conjunction(
		self,
		literals: Sequence[HTNLiteral],
		variable_map: Dict[str, str],
	) -> str:
		if not literals:
			return "(and)"

		rendered = []
		for literal in literals:
			if literal.is_equality and len(literal.args) == 2:
				left = self._render_symbol_token(literal.args[0], variable_map)
				right = self._render_symbol_token(literal.args[1], variable_map)
				base = f"(= {left} {right})"
			else:
				base = f"({literal.predicate}{self._render_invocation_tokens(literal.args, variable_map)})"
			if literal.is_positive:
				rendered.append(base)
			else:
				rendered.append(f"(not {base})")
		return f"(and {' '.join(rendered)})"

	def _render_invocation_tokens(
		self,
		args: Sequence[str],
		variable_map: Dict[str, str],
	) -> str:
		if not args:
			return ""
		rendered = " ".join(self._render_symbol_token(arg, variable_map) for arg in args)
		return f" {rendered}"

	@staticmethod
	def _render_variable(name: str) -> str:
		canonical = name.lstrip("?")
		return f"?{canonical.lower()}"

	@staticmethod
	def _is_variable_token(token: str) -> bool:
		if not token:
			return False
		return token.startswith("?") or token[0].isupper()

	def _render_symbol_token(
		self,
		token: str,
		variable_map: Dict[str, str],
	) -> str:
		canonical = token.lstrip("?")
		if canonical in variable_map:
			return variable_map[canonical]
		if self._is_variable_token(token):
			return self._render_variable(token)
		return token
