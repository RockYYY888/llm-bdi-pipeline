"""
PANDA-backed Stage 4 planner.

This module exports the Stage 3 HTN method library into temporary HDDL files,
invokes the PANDA toolchain, and parses the resulting primitive plan.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
)
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
	) -> None:
		self.workspace = Path(workspace) if workspace else None
		self.parser_cmd = parser_cmd
		self.grounder_cmd = grounder_cmd
		self.engine_cmd = engine_cmd

	def toolchain_available(self) -> bool:
		return all(
			shutil.which(self._command_head(command)) is not None
			for command in (self.parser_cmd, self.grounder_cmd, self.engine_cmd)
		)

	def plan(
		self,
		domain: Any,
		method_library: HTNMethodLibrary,
		objects: Sequence[str],
		target_literal: HTNLiteral,
		transition_name: str,
	) -> PANDAPlanResult:
		self._require_toolchain()

		task_name = self._task_name_for_literal(target_literal)
		task_args = tuple(target_literal.args)
		domain_name = f"{domain.name}_{transition_name}"
		work_dir = self._resolve_work_dir(transition_name)
		work_dir.mkdir(parents=True, exist_ok=True)

		domain_hddl = self._build_domain_hddl(domain, method_library, domain_name)
		problem_hddl = self._build_problem_hddl(
			domain=domain,
			domain_name=domain_name,
			objects=objects,
			task_name=task_name,
			task_args=task_args,
		)

		domain_path = work_dir / "domain.hddl"
		problem_path = work_dir / "problem.hddl"
		parsed_path = work_dir / "problem.psas"
		grounded_path = work_dir / "problem.psas.grounded"
		raw_plan_path = work_dir / "plan.original"
		actual_plan_path = work_dir / "plan.actual"

		domain_path.write_text(domain_hddl)
		problem_path.write_text(problem_hddl)

		parser_run = self._run_command(
			self._build_command(self.parser_cmd, str(domain_path), str(problem_path), str(parsed_path)),
			"parser",
			work_dir,
		)
		grounder_run = self._run_command(
			self._build_command(self.grounder_cmd, str(parsed_path), str(grounded_path)),
			"grounder",
			work_dir,
		)
		engine_run = self._run_command(
			self._build_command(self.engine_cmd, str(grounded_path), str(raw_plan_path)),
			"engine",
			work_dir,
		)

		actual_plan_text = ""
		conversion_stdout = ""
		conversion_stderr = ""
		conversion_command = self._build_command(
			self.parser_cmd,
			"-c",
			str(raw_plan_path),
			str(actual_plan_path),
		)
		conversion_result = subprocess.run(
			conversion_command,
			cwd=work_dir,
			text=True,
			capture_output=True,
			check=False,
		)
		conversion_stdout = conversion_result.stdout
		conversion_stderr = conversion_result.stderr
		if conversion_result.returncode == 0 and actual_plan_path.exists():
			actual_plan_text = actual_plan_path.read_text()
		elif raw_plan_path.exists():
			actual_plan_text = raw_plan_path.read_text()

		raw_plan_text = raw_plan_path.read_text() if raw_plan_path.exists() else ""
		steps = self._parse_plan_steps(actual_plan_text or raw_plan_text, domain)
		if not steps:
			raise PANDAPlanningError(
				f"PANDA returned no executable primitive plan for {task_name}({', '.join(task_args)})",
				metadata={
					"backend": "pandaPI",
					"transition_name": transition_name,
					"task_name": task_name,
					"task_args": list(task_args),
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

		return PANDAPlanResult(
			task_name=task_name,
			task_args=task_args,
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
		)

	def _resolve_work_dir(self, transition_name: str) -> Path:
		if self.workspace is None:
			return Path.cwd() / ".panda_stage3" / transition_name
		return self.workspace / "panda" / transition_name

	def _require_toolchain(self) -> None:
		missing = [
			command
			for command in (self.parser_cmd, self.grounder_cmd, self.engine_cmd)
			if shutil.which(self._command_head(command)) is None
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
			lines.append(f"  (:task {task.name}")
			lines.append(
				f"    :parameters ({self._render_typed_variables(task.parameters, domain.types)})"
			)
			lines.append("  )")
			lines.append("")

		task_lookup = {task.name: task for task in method_library.compound_tasks}
		for method in method_library.methods:
			lines.extend(self._render_method(method, task_lookup, domain.types))
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
	) -> List[str]:
		method_parameters = self._method_parameter_order(method, task_lookup.get(method.task_name))
		variable_map = {
			name: self._render_variable(name)
			for name in method_parameters
		}
		task_schema = task_lookup.get(method.task_name)
		task_args = task_schema.parameters if task_schema else method.parameters

		lines = [f"  (:method {method.method_name}"]
		lines.append(
			f"    :parameters ({self._render_typed_variables(method_parameters, domain_types)})"
		)
		lines.append(
			f"    :task ({method.task_name}{self._render_invocation_tokens(task_args, variable_map)})"
		)
		lines.append(
			f"    :precondition {self._render_literal_conjunction(method.context, variable_map)}"
		)
		lines.append("    :ordered-subtasks (and")
		for step in method.subtasks:
			lines.append(
				f"      ({step.step_id} ({step.task_name}"
				f"{self._render_invocation_tokens(step.args, variable_map)}))"
			)
		lines.append("    )")
		lines.append("  )")
		return lines

	def _build_problem_hddl(
		self,
		domain: Any,
		domain_name: str,
		objects: Sequence[str],
		task_name: str,
		task_args: Sequence[str],
	) -> str:
		object_list = list(dict.fromkeys(objects or task_args))
		object_type = domain.types[0] if domain.types else "object"
		lines = [f"(define (problem {domain_name}_problem)"]
		lines.append(f"  (:domain {domain_name})")
		if object_list:
			lines.append(f"  (:objects {' '.join(object_list)} - {object_type})")
		lines.append("  (:htn")
		lines.append("    :parameters ()")
		lines.append(
			f"    :ordered-subtasks (and (t1 ({task_name}{self._render_problem_args(task_args)})))"
		)
		lines.append("  )")
		lines.append("  (:init")
		for fact in self._build_initial_facts(domain, object_list):
			lines.append(f"    {fact}")
		lines.append("  )")
		lines.append("  (:goal (and))")
		lines.append(")")
		return "\n".join(lines) + "\n"

	def _build_initial_facts(self, domain: Any, objects: Sequence[str]) -> List[str]:
		predicates = {predicate.name: len(predicate.parameters) for predicate in domain.predicates}
		facts: List[str] = []

		if "handempty" in predicates and predicates["handempty"] == 0:
			facts.append("(handempty)")
		if "ontable" in predicates and predicates["ontable"] == 1:
			for obj in objects:
				facts.append(f"(ontable {obj})")
		if "clear" in predicates and predicates["clear"] == 1:
			for obj in objects:
				facts.append(f"(clear {obj})")

		return facts

	def _parse_plan_steps(self, plan_text: str, domain: Any) -> List[PANDAPlanStep]:
		action_names = {action.name for action in domain.actions}
		steps: List[PANDAPlanStep] = []

		for line in plan_text.splitlines():
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
	) -> Dict[str, str]:
		result = subprocess.run(
			command,
			cwd=work_dir,
			text=True,
			capture_output=True,
			check=False,
		)
		if result.returncode == 0:
			return {
				"stdout": result.stdout,
				"stderr": result.stderr,
			}

		raise PANDAPlanningError(
			f"PANDA {stage} step failed with exit code {result.returncode}",
			metadata={
				"backend": "pandaPI",
				"stage": stage,
				"command": list(command),
				"stdout": result.stdout,
				"stderr": result.stderr,
				"work_dir": str(work_dir),
			},
		)

	@staticmethod
	def _task_name_for_literal(literal: HTNLiteral) -> str:
		predicate = literal.predicate.replace("-", "_")
		if literal.is_positive:
			return f"achieve_{predicate}"
		return f"maintain_not_{predicate}"

	@staticmethod
	def _sanitize_name(name: str) -> str:
		return name.replace("-", "_")

	@staticmethod
	def _command_head(command: str) -> str:
		return shlex.split(command)[0]

	@staticmethod
	def _build_command(command: str, *args: str) -> List[str]:
		return [*shlex.split(command), *args]

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

		for token in (task_schema.parameters if task_schema else ()):
			add(token)
		for token in method.parameters:
			add(token)
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
	) -> str:
		default_type = domain_types[0] if domain_types else "object"
		return " ".join(f"{self._render_variable(parameter)} - {default_type}" for parameter in parameters)

	def _render_literal_conjunction(
		self,
		literals: Sequence[HTNLiteral],
		variable_map: Dict[str, str],
	) -> str:
		if not literals:
			return "(and)"

		rendered = []
		for literal in literals:
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
	def _render_problem_args(args: Sequence[str]) -> str:
		if not args:
			return ""
		return f" {' '.join(args)}"

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
