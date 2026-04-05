"""
PANDA-backed Stage 4 planner.

This module exports the Stage 3 HTN method library into temporary HDDL files,
invokes the PANDA toolchain, and parses the resulting primitive plan.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
)
from stage4_panda_planning.problem_builder import PANDAProblemBuilder
from stage4_panda_planning.panda_schema import PANDAPlanResult, PANDAPlanStep


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
	"""Invoke the PANDA PI toolchain on an exported HDDL planning problem."""

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
		task_args: Optional[Sequence[str]] = None,
		task_network: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
		task_network_ordered: bool = True,
		allow_empty_plan: bool = False,
		initial_facts: Optional[Sequence[str]] = None,
		goal_facts: Optional[Sequence[str]] = None,
		timeout_seconds: Optional[float] = None,
	) -> PANDAPlanResult:
		self._require_toolchain()
		total_start = time.perf_counter()
		timing_profile: Dict[str, float] = {}

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
		)
		problem_hddl = self.problem_builder.build_problem_hddl(
			domain=domain,
			domain_name=domain_name,
			objects=objects,
			typed_objects=typed_objects,
			task_name=task_name,
			task_args=task_args_tuple,
			task_network=task_network_entries,
			task_network_ordered=task_network_ordered,
			initial_facts=initial_facts,
			goal_facts=goal_facts,
		)
		timing_profile["hddl_export_seconds"] = time.perf_counter() - export_start

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
		)
		timing_profile["parser_seconds"] = time.perf_counter() - parser_start
		grounder_start = time.perf_counter()
		grounder_run = self._run_command(
			self._build_command(self.grounder_cmd, str(parsed_path), str(grounded_path)),
			"grounder",
			work_dir,
			timeout_seconds=timeout_seconds,
		)
		timing_profile["grounder_seconds"] = time.perf_counter() - grounder_start
		engine_start = time.perf_counter()
		engine_run = self._run_command(
			self._build_command(self.engine_cmd, str(grounded_path)),
			"engine",
			work_dir,
			timeout_seconds=timeout_seconds,
		)
		timing_profile["engine_seconds"] = time.perf_counter() - engine_start
		raw_plan_text = engine_run["stdout"]
		raw_plan_path.write_text(raw_plan_text)

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
			conversion_start = time.perf_counter()
			conversion_result = self._run_subprocess(
				conversion_command,
				work_dir,
				timeout_seconds=timeout_seconds,
			)
			timing_profile["conversion_seconds"] = time.perf_counter() - conversion_start
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
					"work_dir": str(work_dir),
					"timeout_seconds": timeout_seconds,
				},
			) from exc
		if conversion_result["returncode"] == 0 and actual_plan_path.exists():
			actual_plan_text = actual_plan_path.read_text()
		elif raw_plan_path.exists():
			actual_plan_text = raw_plan_path.read_text()

		parse_plan_start = time.perf_counter()
		steps = self._parse_plan_steps(actual_plan_text or raw_plan_text, domain)
		timing_profile["parse_plan_seconds"] = time.perf_counter() - parse_plan_start
		if not steps and not allow_empty_plan and "->" not in actual_plan_text:
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
					"grounder_stdout": grounder_run["stdout"],
					"grounder_stderr": grounder_run["stderr"],
					"engine_stdout": engine_run["stdout"],
					"engine_stderr": engine_run["stderr"],
					"raw_plan": raw_plan_text,
					"actual_plan": actual_plan_text,
					"conversion_stdout": conversion_stdout,
					"conversion_stderr": conversion_stderr,
				},
			)

		timing_profile["total_seconds"] = time.perf_counter() - total_start
		return PANDAPlanResult(
			task_name=task_name,
			task_args=task_args_tuple,
			target_literal=target_literal,
			steps=steps,
			domain_hddl=domain_hddl,
			problem_hddl=problem_hddl,
			parser_stdout=parser_run["stdout"] + conversion_stdout,
			parser_stderr=parser_run["stderr"] + conversion_stderr,
			grounder_stdout=grounder_run["stdout"],
			grounder_stderr=grounder_run["stderr"],
			engine_stdout=engine_run["stdout"],
			engine_stderr=engine_run["stderr"],
			raw_plan=raw_plan_text,
			actual_plan=actual_plan_text or raw_plan_text,
			work_dir=str(work_dir),
			timing_profile=timing_profile,
		)

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
			return Path.cwd() / ".panda_stage3" / transition_name
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

		for task in method_library.compound_tasks:
			task_signature = task_type_map.get(task.name) or task_type_map.get(task.source_name or "")
			lines.append(f"  (:task {task.source_name or task.name}")
			lines.append(
				f"    :parameters ({self._render_typed_variables(task.parameters, domain.types, task_signature)})"
			)
			lines.append("  )")
			lines.append("")

		task_lookup = {task.name: task for task in method_library.compound_tasks}
		for method in method_library.methods:
			lines.extend(
				self._render_method(
					method,
					task_lookup,
					domain.types,
					task_type_map,
					self._predicate_type_map(domain),
					self._action_type_map(domain),
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
		domain_types: Sequence[str],
		task_type_map: Dict[str, Tuple[str, ...]],
		predicate_types: Dict[str, Tuple[str, ...]],
		action_types: Dict[str, Tuple[str, ...]],
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
			f"    :task ({getattr(task_schema, 'source_name', None) or method.task_name}"
			f"{self._render_invocation_tokens(task_args, variable_map)})"
		)
		lines.append(
			f"    :precondition {self._render_literal_conjunction(method.context, variable_map)}"
		)
		subtasks_keyword = ":subtasks" if method.ordering else ":ordered-subtasks"
		lines.append(f"    {subtasks_keyword} (and")
		for step in method.subtasks:
			step_name = self._render_method_step_name(step, task_lookup)
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
			return tuple(method.parameters)
		leading_parameters = tuple(method.parameters[: len(declared_parameters)])
		if len(leading_parameters) == len(declared_parameters):
			return leading_parameters
		return declared_parameters

	def _render_method_step_name(self, step: Any, task_lookup: Dict[str, Any]) -> str:
		if getattr(step, "kind", None) == "primitive":
			if getattr(step, "action_name", None):
				return step.action_name
			return step.task_name.replace("_", "-")
		task_schema = task_lookup.get(step.task_name)
		return getattr(task_schema, "source_name", None) or step.task_name

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

	def _run_command(
		self,
		command: Sequence[str],
		stage: str,
		work_dir: Path,
		*,
		timeout_seconds: Optional[float] = None,
	) -> Dict[str, str]:
		try:
			result = self._run_subprocess(
				command,
				work_dir,
				timeout_seconds=timeout_seconds,
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
					"work_dir": str(work_dir),
					"timeout_seconds": timeout_seconds,
				},
			) from exc
		if result["returncode"] == 0:
			return {
				"stdout": result["stdout"],
				"stderr": result["stderr"],
			}

		raise PANDAPlanningError(
			f"PANDA {stage} step failed with exit code {result['returncode']}",
			metadata={
				"backend": "pandaPI",
				"stage": stage,
				"command": list(command),
				"stdout": result["stdout"],
				"stderr": result["stderr"],
				"work_dir": str(work_dir),
			},
		)

	def _run_subprocess(
		self,
		command: Sequence[str],
		work_dir: Path,
		*,
		timeout_seconds: Optional[float] = None,
	) -> Dict[str, Any]:
		process = subprocess.Popen(
			command,
			cwd=work_dir,
			text=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			start_new_session=True,
		)
		try:
			stdout, stderr = process.communicate(timeout=timeout_seconds)
		except subprocess.TimeoutExpired as exc:
			self._terminate_process_group(process)
			stdout, stderr = process.communicate()
			raise PANDAPlanningError(
				"PANDA subprocess timed out",
				metadata={
					"stage": "subprocess_timeout",
					"command": list(command),
					"stdout": exc.stdout or stdout or "",
					"stderr": exc.stderr or stderr or "",
					"work_dir": str(work_dir),
					"timeout_seconds": timeout_seconds,
				},
			) from exc
		return {
			"returncode": process.returncode,
			"stdout": stdout or "",
			"stderr": stderr or "",
		}

	@staticmethod
	def _terminate_process_group(process: subprocess.Popen) -> None:
		try:
			os.killpg(os.getpgid(process.pid), signal.SIGKILL)
		except Exception:
			try:
				process.kill()
			except Exception:
				return

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

		resolved = shutil.which(head)
		if resolved:
			return resolved

		for directory in self._default_command_dirs():
			candidate = directory / head
			if candidate.is_file() and os.access(candidate, os.X_OK):
				return str(candidate)

		return None

	def _default_command_dirs(self) -> Tuple[Path, ...]:
		directories: List[Path] = []
		panda_home = os.getenv("PANDA_PI_HOME")
		if panda_home:
			directories.append(Path(panda_home) / "bin")
		panda_bin = os.getenv("PANDA_PI_BIN")
		if panda_bin:
			directories.append(Path(panda_bin))
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
