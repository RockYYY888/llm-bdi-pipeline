from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from method_library.synthesis.schema import HTNMethod, HTNMethodLibrary
from method_library.synthesis.naming import query_root_alias_task_name, sanitize_identifier
from utils.hddl_parser import HDDLParser


_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
_TEXT_PREVIEW_LIMIT = 2000


def _compact_text_preview(text: str) -> str:
	payload = str(text or "")
	if len(payload) <= _TEXT_PREVIEW_LIMIT:
		return payload
	head_limit = _TEXT_PREVIEW_LIMIT // 2
	tail_limit = _TEXT_PREVIEW_LIMIT - head_limit
	omitted = len(payload) - head_limit - tail_limit
	return (
		f"{payload[:head_limit]}\n"
		f"...[truncated {omitted} chars; full text in output_file]...\n"
		f"{payload[-tail_limit:]}"
	)


@dataclass(frozen=True)
class IPCPrimitivePlanVerificationResult:
	tool_available: bool
	command: List[str]
	plan_file: str
	output_file: str
	stdout: str
	stderr: str
	primitive_plan_only: Optional[bool]
	primitive_plan_executable: Optional[bool]
	verification_result: Optional[bool]
	reached_goal_state: Optional[bool]
	plan_kind: str = "primitive_only"
	build_warning: Optional[str] = None
	error: Optional[str] = None

	def to_dict(self) -> Dict[str, object]:
		return {
			"tool_available": self.tool_available,
			"command": list(self.command),
			"plan_file": self.plan_file,
			"output_file": self.output_file,
			"stdout_preview": _compact_text_preview(self.stdout),
			"stdout_chars": len(self.stdout or ""),
			"stderr_preview": _compact_text_preview(self.stderr),
			"stderr_chars": len(self.stderr or ""),
			"primitive_plan_only": self.primitive_plan_only,
			"primitive_plan_executable": self.primitive_plan_executable,
			"verification_result": self.verification_result,
			"reached_goal_state": self.reached_goal_state,
			"plan_kind": self.plan_kind,
			"build_warning": self.build_warning,
			"error": self.error,
		}


@dataclass
class _PrimitiveNode:
	name: str
	args: Tuple[str, ...]
	node_id: Optional[int] = None


@dataclass
class _AbstractNode:
	task_name: str
	args: Tuple[str, ...]
	method_name: str
	children: List[Union["_PrimitiveNode", "_AbstractNode"]]
	node_id: Optional[int] = None


_PlanNode = Union[_PrimitiveNode, _AbstractNode]
_ActionStep = Tuple[str, Tuple[str, ...]]


@dataclass(frozen=True)
class _MethodTraceEntry:
	method_name: str
	task_args: Tuple[str, ...]


class IPCPlanVerifier:
	"""Run the PANDA HTN verifier with best-effort hierarchical plan export."""

	def __init__(
		self,
		parser_cmd: str = "pandaPIparser",
		grounder_cmd: str = "pandaPIgrounder",
		engine_cmd: str = "pandaPIengine",
	) -> None:
		self.parser_cmd = parser_cmd
		self.grounder_cmd = grounder_cmd
		self.engine_cmd = engine_cmd
		self._last_hierarchical_build_warning: Optional[str] = None

	def tool_available(self) -> bool:
		return self._resolve_command_head(self.parser_cmd) is not None

	def planning_toolchain_available(self) -> bool:
		return all(
			self._resolve_command_head(command) is not None
			for command in (self.parser_cmd, self.grounder_cmd, self.engine_cmd)
		)

	def verify_plan(
		self,
		*,
		domain_file: str | Path,
		problem_file: str | Path,
		action_path: Sequence[str],
		method_library: HTNMethodLibrary | None = None,
		method_trace: Sequence[Dict[str, Any]] | None = None,
		output_dir: str | Path,
		plan_filename: str = "ipc_official_plan.txt",
		output_filename: str = "ipc_official_verifier.txt",
		json_filename: str = "ipc_official_verification.json",
	) -> IPCPrimitivePlanVerificationResult:
		return self._verify(
			domain_file=domain_file,
			problem_file=problem_file,
			action_path=action_path,
			method_library=method_library,
			method_trace=method_trace,
			output_dir=output_dir,
			plan_filename=plan_filename,
			output_filename=output_filename,
			json_filename=json_filename,
			prefer_hierarchical=True,
		)

	def verify_plan_text(
		self,
		*,
		domain_file: str | Path,
		problem_file: str | Path,
		plan_text: str,
		output_dir: str | Path,
		plan_kind: str = "hierarchical",
		build_warning: Optional[str] = None,
		plan_filename: str = "ipc_official_plan.txt",
		output_filename: str = "ipc_official_verifier.txt",
		json_filename: str = "ipc_official_verification.json",
	) -> IPCPrimitivePlanVerificationResult:
		output_path = Path(output_dir).resolve()
		output_path.mkdir(parents=True, exist_ok=True)
		plan_path = output_path / plan_filename
		output_text_path = output_path / output_filename
		output_json_path = output_path / json_filename

		plan_path.write_text(str(plan_text))
		resolved_parser = self._resolve_command_head(self.parser_cmd) or self.parser_cmd
		command = [
			resolved_parser,
			"-v",
			str(Path(domain_file).resolve()),
			str(Path(problem_file).resolve()),
			str(plan_path),
		]

		if not self.tool_available():
			result = IPCPrimitivePlanVerificationResult(
				tool_available=False,
				command=command,
				plan_file=str(plan_path),
				output_file=str(output_text_path),
				stdout="",
				stderr="",
				primitive_plan_only=None,
				primitive_plan_executable=None,
				verification_result=None,
				reached_goal_state=None,
				plan_kind=plan_kind,
				build_warning=build_warning,
				error=f"{self.parser_cmd} is not available on PATH",
			)
			output_json_path.write_text(json.dumps(result.to_dict(), indent=2))
			return result

		completed = subprocess.run(
			command,
			text=True,
			capture_output=True,
			check=False,
		)
		stdout = self.strip_ansi(completed.stdout)
		stderr = self.strip_ansi(completed.stderr)
		combined = self._combine_output(stdout, stderr)
		output_text_path.write_text(combined)

		result = IPCPrimitivePlanVerificationResult(
			tool_available=True,
			command=command,
			plan_file=str(plan_path),
			output_file=str(output_text_path),
			stdout=stdout,
			stderr=stderr,
			primitive_plan_only="Primitive plan only" in combined,
			primitive_plan_executable=self._extract_executability(combined),
			verification_result=self._extract_bool(
				combined,
				"Plan verification result",
			),
			reached_goal_state=self._infer_goal_reached(combined),
			plan_kind=plan_kind,
			build_warning=build_warning,
			error=None if completed.returncode == 0 else f"verifier exited with code {completed.returncode}",
		)
		output_json_path.write_text(json.dumps(result.to_dict(), indent=2))
		return result

	def verify_primitive_plan(
		self,
		*,
		domain_file: str | Path,
		problem_file: str | Path,
		action_path: Sequence[str],
		output_dir: str | Path,
		plan_filename: str = "ipc_official_plan.txt",
		output_filename: str = "ipc_official_verifier.txt",
		json_filename: str = "ipc_official_verification.json",
	) -> IPCPrimitivePlanVerificationResult:
		return self._verify(
			domain_file=domain_file,
			problem_file=problem_file,
			action_path=action_path,
			method_library=None,
			method_trace=None,
			output_dir=output_dir,
			plan_filename=plan_filename,
			output_filename=output_filename,
			json_filename=json_filename,
			prefer_hierarchical=False,
		)

	def verify_planned_hierarchical_plan(
		self,
		*,
		domain_file: str | Path,
		problem_file: str | Path,
		output_dir: str | Path,
		plan_filename: str = "ipc_official_plan.txt",
		output_filename: str = "ipc_official_verifier.txt",
		json_filename: str = "ipc_official_verification.json",
		raw_plan_filename: str = "ipc_official_plan.raw.txt",
	) -> IPCPrimitivePlanVerificationResult:
		output_path = Path(output_dir).resolve()
		output_path.mkdir(parents=True, exist_ok=True)
		plan_path = output_path / plan_filename
		raw_plan_path = output_path / raw_plan_filename
		output_text_path = output_path / output_filename
		output_json_path = output_path / json_filename
		parsed_problem_path = output_path / "ipc_official_problem.psas"
		grounded_problem_path = output_path / "ipc_official_problem.psas.grounded"

		parser_head = self._resolve_command_head(self.parser_cmd)
		grounder_head = self._resolve_command_head(self.grounder_cmd)
		engine_head = self._resolve_command_head(self.engine_cmd)
		verify_command = [
			parser_head or self.parser_cmd,
			"-v",
			str(Path(domain_file).resolve()),
			str(Path(problem_file).resolve()),
			str(plan_path),
		]
		if not all((parser_head, grounder_head, engine_head)):
			result = IPCPrimitivePlanVerificationResult(
				tool_available=False,
				command=verify_command,
				plan_file=str(plan_path),
				output_file=str(output_text_path),
				stdout="",
				stderr="",
				primitive_plan_only=None,
				primitive_plan_executable=None,
				verification_result=None,
				reached_goal_state=None,
				plan_kind="hierarchical",
				build_warning="planner toolchain unavailable",
				error=(
					"planner toolchain unavailable: "
					f"{self.parser_cmd}, {self.grounder_cmd}, or {self.engine_cmd}"
				),
			)
			output_json_path.write_text(json.dumps(result.to_dict(), indent=2))
			return result

		build_warning: Optional[str] = None
		try:
			subprocess.run(
				[
					parser_head,
					str(Path(domain_file).resolve()),
					str(Path(problem_file).resolve()),
					str(parsed_problem_path),
				],
				text=True,
				capture_output=True,
				check=True,
			)
			subprocess.run(
				[
					grounder_head,
					str(parsed_problem_path),
					str(grounded_problem_path),
				],
				text=True,
				capture_output=True,
				check=True,
			)
			engine_run = subprocess.run(
				[
					engine_head,
					str(grounded_problem_path),
				],
				text=True,
				capture_output=True,
				check=True,
			)
		except subprocess.CalledProcessError as exc:
			combined = self._combine_output(
				self.strip_ansi(exc.stdout or ""),
				self.strip_ansi(exc.stderr or ""),
			)
			output_text_path.write_text(combined)
			result = IPCPrimitivePlanVerificationResult(
				tool_available=True,
				command=verify_command,
				plan_file=str(plan_path),
				output_file=str(output_text_path),
				stdout=self.strip_ansi(exc.stdout or ""),
				stderr=self.strip_ansi(exc.stderr or ""),
				primitive_plan_only=None,
				primitive_plan_executable=None,
				verification_result=None,
				reached_goal_state=None,
				plan_kind="hierarchical",
				build_warning="planner failed before verification",
				error=f"planner exited with code {exc.returncode}",
			)
			output_json_path.write_text(json.dumps(result.to_dict(), indent=2))
			return result

		raw_plan_text = self.strip_ansi(engine_run.stdout)
		raw_plan_path.write_text(raw_plan_text)

		conversion_result = subprocess.run(
			[
				parser_head,
				"-c",
				str(raw_plan_path),
				str(plan_path),
			],
			text=True,
			capture_output=True,
			check=False,
		)
		if conversion_result.returncode != 0 or not plan_path.exists():
			plan_path.write_text(raw_plan_text)
			build_warning = "planner plan conversion failed; verifying raw PANDA plan"

		completed = subprocess.run(
			verify_command,
			text=True,
			capture_output=True,
			check=False,
		)
		stdout = self.strip_ansi(completed.stdout)
		stderr = self.strip_ansi(completed.stderr)
		combined = self._combine_output(stdout, stderr)
		output_text_path.write_text(combined)

		result = IPCPrimitivePlanVerificationResult(
			tool_available=True,
			command=verify_command,
			plan_file=str(plan_path),
			output_file=str(output_text_path),
			stdout=stdout,
			stderr=stderr,
			primitive_plan_only="Primitive plan only" in combined,
			primitive_plan_executable=self._extract_executability(combined),
			verification_result=self._extract_bool(
				combined,
				"Plan verification result",
			),
			reached_goal_state=self._infer_goal_reached(combined),
			plan_kind="hierarchical",
			build_warning=build_warning,
			error=None if completed.returncode == 0 else f"verifier exited with code {completed.returncode}",
		)
		output_json_path.write_text(json.dumps(result.to_dict(), indent=2))
		return result

	def _verify(
		self,
		*,
		domain_file: str | Path,
		problem_file: str | Path,
		action_path: Sequence[str],
		method_library: HTNMethodLibrary | None,
		method_trace: Sequence[Dict[str, Any]] | None,
		output_dir: str | Path,
		plan_filename: str,
		output_filename: str,
		json_filename: str,
		prefer_hierarchical: bool,
	) -> IPCPrimitivePlanVerificationResult:
		output_path = Path(output_dir).resolve()
		output_path.mkdir(parents=True, exist_ok=True)
		plan_path = output_path / plan_filename
		output_text_path = output_path / output_filename
		output_json_path = output_path / json_filename

		build_warning = None
		plan_kind = "primitive_only"
		plan_text = self.render_primitive_only_plan(action_path)
		if prefer_hierarchical:
			self._last_hierarchical_build_warning = None
			try:
				rendered = self._render_supported_hierarchical_plan(
					domain_file=domain_file,
					problem_file=problem_file,
					action_path=action_path,
					method_library=method_library,
					method_trace=method_trace,
				)
			except Exception as exc:
				rendered = None
				build_warning = str(exc)
			if rendered is not None:
				plan_text = rendered
				plan_kind = "hierarchical"
				if build_warning is None:
					build_warning = self._last_hierarchical_build_warning
		plan_path.write_text(plan_text)

		resolved_parser = self._resolve_command_head(self.parser_cmd) or self.parser_cmd
		command = [
			resolved_parser,
			"-v",
			str(Path(domain_file).resolve()),
			str(Path(problem_file).resolve()),
			str(plan_path),
		]

		if not self.tool_available():
			result = IPCPrimitivePlanVerificationResult(
				tool_available=False,
				command=command,
				plan_file=str(plan_path),
				output_file=str(output_text_path),
				stdout="",
				stderr="",
				primitive_plan_only=None,
				primitive_plan_executable=None,
				verification_result=None,
				reached_goal_state=None,
				plan_kind=plan_kind,
				build_warning=build_warning,
				error=f"{self.parser_cmd} is not available on PATH",
			)
			output_json_path.write_text(json.dumps(result.to_dict(), indent=2))
			return result

		completed = subprocess.run(
			command,
			text=True,
			capture_output=True,
			check=False,
		)
		stdout = self.strip_ansi(completed.stdout)
		stderr = self.strip_ansi(completed.stderr)
		combined = self._combine_output(stdout, stderr)
		output_text_path.write_text(combined)

		result = IPCPrimitivePlanVerificationResult(
			tool_available=True,
			command=command,
			plan_file=str(plan_path),
			output_file=str(output_text_path),
			stdout=stdout,
			stderr=stderr,
			primitive_plan_only="Primitive plan only" in combined,
			primitive_plan_executable=self._extract_executability(combined),
			verification_result=self._extract_bool(
				combined,
				"Plan verification result",
			),
			reached_goal_state=self._infer_goal_reached(combined),
			plan_kind=plan_kind,
			build_warning=build_warning,
			error=None if completed.returncode == 0 else f"verifier exited with code {completed.returncode}",
		)
		output_json_path.write_text(json.dumps(result.to_dict(), indent=2))
		return result

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
	def _command_head(command: str) -> str:
		parts = str(command).strip().split()
		return parts[0] if parts else str(command)

	@staticmethod
	def render_primitive_only_plan(action_path: Sequence[str]) -> str:
		lines = ["==>"]
		for index, action_step in enumerate(action_path):
			name, args = IPCPlanVerifier._parse_action_step(action_step)
			lines.append(" ".join([str(index), name, *args]).rstrip())
		lines.append("root")
		return "\n".join(lines) + "\n"

	def _render_supported_hierarchical_plan(
		self,
		*,
		domain_file: str | Path,
		problem_file: str | Path,
		action_path: Sequence[str],
		method_library: HTNMethodLibrary | None,
		method_trace: Sequence[Dict[str, Any]] | None,
	) -> Optional[str]:
		problem = HDDLParser.parse_problem(str(problem_file))
		if method_library is None or not method_library.methods:
			return None
		if method_trace is None:
			return None
		if not problem.htn_tasks:
			return None

		actions = [self._parse_action_step(step) for step in action_path]
		trace_entries = self._normalise_method_trace(method_trace)
		root_tasks = list(problem.htn_tasks)
		root_tasks_ordered = bool(problem.htn_ordered)
		if root_tasks_ordered and self._root_tasks_use_query_root_aliases(root_tasks, method_library):
			root_tasks_ordered = False
		root_nodes, action_index, trace_index = self._reconstruct_hierarchy(
			method_library=method_library,
			root_tasks=root_tasks,
			actions=actions,
			trace_entries=trace_entries,
			root_tasks_ordered=root_tasks_ordered,
		)
		warnings: List[str] = []
		if action_index != len(actions):
			warnings.append(
				"hierarchical exporter used only the runtime action prefix "
				f"({action_index}/{len(actions)} steps)"
			)
		if trace_index != len(trace_entries):
			warnings.append(
				"hierarchical exporter used only the method-trace prefix "
				f"({trace_index}/{len(trace_entries)} entries)"
			)
		self._last_hierarchical_build_warning = "; ".join(warnings) if warnings else None
		if warnings:
			return None
		return self._render_hierarchical_plan(root_nodes)

	@staticmethod
	def _all_query_root_alias_tasks(root_tasks: Sequence[Any]) -> bool:
		if not root_tasks:
			return False
		for task in tuple(root_tasks):
			task_name = str(
				getattr(task, "task_name", None) or getattr(task, "name", "") or "",
			)
			if re.fullmatch(r"query_root_\d+_[A-Za-z0-9_]+", task_name) is None:
				return False
		return True

	@classmethod
	def _root_tasks_use_query_root_aliases(
		cls,
		root_tasks: Sequence[Any],
		method_library: HTNMethodLibrary,
	) -> bool:
		query_root_sources = {
			str(getattr(task, "source_name", None) or getattr(task, "name", "") or "")
			for task in tuple(method_library.compound_tasks or ()) + tuple(method_library.primitive_tasks or ())
			if cls._all_query_root_alias_tasks((task,))
		}
		if not query_root_sources:
			return False
		for task in tuple(root_tasks or ()):
			source_task_name = str(getattr(task, "task_name", "") or "")
			if source_task_name not in query_root_sources:
				return False
		return True

	def _reconstruct_hierarchy(
		self,
		*,
		method_library: HTNMethodLibrary,
		root_tasks: Sequence[Any],
		actions: Sequence[_ActionStep],
		trace_entries: Sequence[_MethodTraceEntry],
		root_tasks_ordered: bool,
	) -> Tuple[List[_AbstractNode], int, int]:
		task_lookup = {
			task.name: task
			for task in method_library.compound_tasks + method_library.primitive_tasks
		}
		internal_tasks_by_source: Dict[str, List[str]] = {}
		source_task_by_internal: Dict[str, str] = {}
		for task in method_library.compound_tasks + method_library.primitive_tasks:
			source_name = getattr(task, "source_name", None) or task.name
			internal_tasks_by_source.setdefault(source_name, [])
			if task.name not in internal_tasks_by_source[source_name]:
				internal_tasks_by_source[source_name].append(task.name)
			internal_tasks_by_source.setdefault(task.name, [])
			if task.name not in internal_tasks_by_source[task.name]:
				internal_tasks_by_source[task.name].append(task.name)
			source_task_by_internal[task.name] = source_name

		method_lookup = {
			method.method_name: method
			for method in method_library.methods
		}
		if len(method_lookup) != len(method_library.methods):
			raise ValueError("method library contains duplicate method names")

		def reconstruct_task(
			task_name: str,
			task_args: Tuple[str, ...],
			source_task_name: str,
			action_index: int,
			trace_index: int,
		) -> Tuple[_AbstractNode, int, int]:
			next_action_index = action_index
			next_trace_index = trace_index
			while next_trace_index < len(trace_entries):
				entry = trace_entries[next_trace_index]
				method = method_lookup.get(entry.method_name)
				if method is None:
					raise ValueError(f"method trace references unknown method '{entry.method_name}'")
				if self._method_trace_entry_matches_task(
					method=method,
					entry=entry,
					task_name=task_name,
					task_args=task_args,
				):
					break
				if method.task_name != task_name:
					raise ValueError(
						f"expected method for task '{task_name}' but trace selected '{method.task_name}'",
					)
				raise ValueError(
					f"method trace arguments for '{entry.method_name}' do not match task "
					f"{source_task_name}{task_args}: got {entry.task_args}",
				)

			if next_trace_index >= len(trace_entries):
				raise ValueError(
					f"method trace ended before task {source_task_name}{task_args} could be reconstructed",
				)

			entry = trace_entries[next_trace_index]
			method = method_lookup[entry.method_name]
			grounded_task_args = self._resolve_task_args_from_trace(
				task_args=task_args,
				trace_args=entry.task_args,
			)
			bindings = self._seed_method_bindings(method, grounded_task_args, task_lookup)

			next_trace_index += 1
			child_nodes: List[_PlanNode] = []
			remaining_steps = list(self._ordered_method_steps(method))
			while remaining_steps:
				step = remaining_steps[0]
				if step.kind == "primitive":
					expected_action_name = step.action_name or step.task_name
					if next_action_index >= len(actions):
						raise ValueError(
							f"runtime action path ended before primitive step '{expected_action_name}'",
						)
					actual_action_name, actual_action_args = actions[next_action_index]
					if not self._action_names_match(expected_action_name, actual_action_name):
						raise ValueError(
							f"primitive step mismatch for '{expected_action_name}': "
							f"expected action '{expected_action_name}', got '{actual_action_name}'",
						)
					bindings = self._unify_arguments(
						step.args,
						actual_action_args,
						bindings,
						method.parameters,
					)
					child_nodes.append(
						_PrimitiveNode(
							name=actual_action_name,
							args=actual_action_args,
						),
					)
					next_action_index += 1
					remaining_steps.pop(0)
					continue

				reorderable_prefix_end = 0
				while (
					reorderable_prefix_end < len(remaining_steps)
					and remaining_steps[reorderable_prefix_end].kind == "compound"
				):
					reorderable_prefix_end += 1
				prefix_steps = remaining_steps[:reorderable_prefix_end]
				prefix_nodes: List[Optional[_PlanNode]] = [None] * len(prefix_steps)
				unmatched_prefix_indices = list(range(len(prefix_steps)))

				while unmatched_prefix_indices:
					if next_trace_index >= len(trace_entries):
						raise ValueError(
							f"method trace ended before child task '{prefix_steps[0].task_name}' could be reconstructed",
						)
					child_entry = trace_entries[next_trace_index]
					child_method = method_lookup.get(child_entry.method_name)
					selected_step_index = unmatched_prefix_indices[0]
					if child_method is not None:
						for candidate_index in unmatched_prefix_indices:
							candidate_step = prefix_steps[candidate_index]
							if child_method.task_name != candidate_step.task_name:
								continue
							try:
								self._unify_arguments(
									candidate_step.args,
									child_entry.task_args,
									dict(bindings),
									method.parameters,
								)
							except ValueError:
								continue
							selected_step_index = candidate_index
							break

					step = prefix_steps[selected_step_index]
					child_internal_name = step.task_name
					child_task_args = child_entry.task_args
					bindings = self._unify_arguments(
						step.args,
						child_task_args,
						bindings,
						method.parameters,
					)
					child_source_name = source_task_by_internal.get(
						child_internal_name,
						child_internal_name,
					)
					child_node, next_action_index, next_trace_index = reconstruct_task(
						child_internal_name,
						child_task_args,
						child_source_name,
						next_action_index,
						next_trace_index,
					)
					prefix_nodes[selected_step_index] = child_node
					unmatched_prefix_indices.remove(selected_step_index)

				child_nodes.extend(
					node
					for node in prefix_nodes
					if node is not None
				)
				remaining_steps = remaining_steps[reorderable_prefix_end:]

			refined_task_args = self._refine_task_args_from_method_bindings(
				task_args=grounded_task_args,
				method=method,
				task_lookup=task_lookup,
				bindings=bindings,
			)

			return (
				_AbstractNode(
					task_name=source_task_name,
					args=refined_task_args,
					method_name=method.source_method_name or method.method_name,
					children=child_nodes,
				),
				next_action_index,
				next_trace_index,
			)

		def candidate_internal_task_names(
			source_task_name: str,
			occurrence_index: int = 0,
			root_index: Optional[int] = None,
		) -> List[str]:
			candidates = list(
				internal_tasks_by_source.get(source_task_name, [source_task_name])
			)
			if not candidates:
				return [source_task_name]
			query_root_candidates = [
				name
				for name in candidates
				if self._all_query_root_alias_tasks((type("TaskRef", (), {"name": name})(),))
			]
			non_query_root_candidates = [
				name
				for name in candidates
				if name not in query_root_candidates
			]
			if query_root_candidates:
				if root_index is not None:
					expected_query_root = query_root_alias_task_name(root_index, source_task_name)
					if expected_query_root in query_root_candidates:
						query_root_candidates = [expected_query_root] + [
							name
							for name in query_root_candidates
							if name != expected_query_root
						]
				elif 0 <= occurrence_index < len(query_root_candidates):
					preferred = query_root_candidates[occurrence_index]
					query_root_candidates = [preferred] + [
						name
						for name in query_root_candidates
						if name != preferred
					]
				return query_root_candidates + non_query_root_candidates
			if root_index is not None:
				expected_query_root = query_root_alias_task_name(root_index, source_task_name)
				if expected_query_root in candidates:
					return [expected_query_root] + [
						name
						for name in candidates
						if name != expected_query_root
					]
			if len(candidates) == 1:
				return candidates
			if 0 <= occurrence_index < len(candidates):
				preferred = candidates[occurrence_index]
				return [preferred] + [name for name in candidates if name != preferred]
			return candidates

		root_nodes: List[_AbstractNode] = []
		next_action_index = 0
		next_trace_index = 0
		if root_tasks_ordered:
			source_occurrence_counts: Dict[str, int] = {}
			for root_index, task in enumerate(root_tasks, start=1):
				source_task_name = getattr(task, "task_name")
				task_args = tuple(getattr(task, "args"))
				occurrence_index = source_occurrence_counts.get(source_task_name, 0)
				source_occurrence_counts[source_task_name] = occurrence_index + 1
				last_error: Optional[Exception] = None
				matched = False
				for internal_task_name in candidate_internal_task_names(
					source_task_name,
					occurrence_index,
					root_index=root_index,
				):
					try:
						root_node, candidate_action_index, candidate_trace_index = reconstruct_task(
							internal_task_name,
							task_args,
							source_task_name,
							next_action_index,
							next_trace_index,
						)
					except Exception as exc:
						last_error = exc
						continue
					root_nodes.append(root_node)
					next_action_index = candidate_action_index
					next_trace_index = candidate_trace_index
					matched = True
					break
				if not matched:
					if last_error is not None:
						raise last_error
					raise ValueError(
						f"failed to match ordered root task {source_task_name}{task_args}",
					)
			return root_nodes, next_action_index, next_trace_index

		remaining_root_tasks = list(root_tasks)
		while remaining_root_tasks:
			current_entry = trace_entries[next_trace_index] if next_trace_index < len(trace_entries) else None
			prioritised_tasks: List[Any] = []
			other_tasks: List[Any] = []
			for task in remaining_root_tasks:
				source_task_name = getattr(task, "task_name")
				task_args = tuple(getattr(task, "args"))
				method = method_lookup.get(current_entry.method_name) if current_entry is not None else None
				candidate_names = candidate_internal_task_names(source_task_name)
				if method is not None and any(
					self._method_trace_entry_matches_task(
						method=method,
						entry=current_entry,
						task_name=internal_task_name,
						task_args=task_args,
					)
					for internal_task_name in candidate_names
				):
					prioritised_tasks.append(task)
				else:
					other_tasks.append(task)

			candidate_tasks = prioritised_tasks + other_tasks
			last_error: Optional[Exception] = None
			matched_task = None
			for task in candidate_tasks:
				source_task_name = getattr(task, "task_name")
				task_args = tuple(getattr(task, "args"))
				for internal_task_name in candidate_internal_task_names(source_task_name):
					try:
						root_node, candidate_action_index, candidate_trace_index = reconstruct_task(
							internal_task_name,
							task_args,
							source_task_name,
							next_action_index,
							next_trace_index,
						)
					except Exception as exc:
						last_error = exc
						continue
					root_nodes.append(root_node)
					next_action_index = candidate_action_index
					next_trace_index = candidate_trace_index
					matched_task = task
					break
				if matched_task is not None:
					break

			if matched_task is None:
				if last_error is not None:
					raise last_error
				raise ValueError("failed to match unordered root task from method trace")
			remaining_root_tasks.remove(matched_task)

		return root_nodes, next_action_index, next_trace_index

	def _method_trace_entry_matches_task(
		self,
		*,
		method: HTNMethod,
		entry: _MethodTraceEntry,
		task_name: str,
		task_args: Sequence[str],
	) -> bool:
		if method.task_name != task_name:
			return False
		try:
			self._resolve_task_args_from_trace(
				task_args=task_args,
				trace_args=entry.task_args,
			)
		except ValueError:
			return False
		return True

	@staticmethod
	def _normalise_method_trace(
		method_trace: Sequence[Dict[str, Any]],
	) -> List[_MethodTraceEntry]:
		trace_entries: List[_MethodTraceEntry] = []
		for item in method_trace:
			method_name = str(item.get("method_name", "")).strip()
			if not method_name:
				continue
			task_args = tuple(
				str(value).strip()
				for value in (item.get("task_args") or item.get("bindings") or [])
			)
			trace_entries.append(
				_MethodTraceEntry(
					method_name=method_name,
					task_args=task_args,
				),
			)
		return trace_entries

	@staticmethod
	def _resolve_task_args_from_trace(
		task_args: Sequence[str],
		trace_args: Sequence[str],
	) -> Tuple[str, ...]:
		bindings = IPCPlanVerifier._unify_arguments(
			task_args,
			trace_args,
			{},
			(),
		)
		return IPCPlanVerifier._ground_arguments(task_args, bindings)

	@staticmethod
	def _ground_arguments(args: Sequence[str], bindings: Dict[str, str]) -> Tuple[str, ...]:
		return tuple(bindings.get(arg, arg) for arg in args)

	@staticmethod
	def _seed_method_bindings(
		method: HTNMethod,
		task_args: Tuple[str, ...],
		task_lookup: Dict[str, Any],
	) -> Dict[str, str]:
		pattern = method.task_args or IPCPlanVerifier._default_task_args(method, task_lookup)
		return IPCPlanVerifier._unify_arguments(
			pattern,
			task_args,
			{},
			method.parameters,
		)

	@staticmethod
	def _refine_task_args_from_method_bindings(
		task_args: Sequence[str],
		method: HTNMethod,
		task_lookup: Dict[str, Any],
		bindings: Dict[str, str],
	) -> Tuple[str, ...]:
		pattern = method.task_args or IPCPlanVerifier._default_task_args(method, task_lookup)
		refined: List[str] = []
		for current_arg, pattern_arg in zip(task_args, pattern):
			if IPCPlanVerifier._looks_like_variable(current_arg):
				bound_value = bindings.get(pattern_arg)
				if bound_value and not IPCPlanVerifier._looks_like_variable(bound_value):
					refined.append(bound_value)
					continue
			refined.append(current_arg)
		return tuple(refined)

	@staticmethod
	def _unify_arguments(
		pattern_args: Sequence[str],
		grounded_args: Sequence[str],
		bindings: Dict[str, str],
		known_variables: Sequence[str],
	) -> Dict[str, str]:
		if len(pattern_args) != len(grounded_args):
			raise ValueError(
				f"argument arity mismatch: expected {len(pattern_args)}, got {len(grounded_args)}",
			)

		next_bindings = dict(bindings)
		known_variable_set = set(known_variables)
		for pattern_arg, grounded_arg in zip(pattern_args, grounded_args):
			if pattern_arg in known_variable_set or IPCPlanVerifier._looks_like_variable(pattern_arg):
				existing = next_bindings.get(pattern_arg)
				if existing is not None and existing != grounded_arg:
					if (
						IPCPlanVerifier._looks_like_variable(existing)
						and not IPCPlanVerifier._looks_like_variable(grounded_arg)
					):
						next_bindings[pattern_arg] = grounded_arg
						continue
					if (
						not IPCPlanVerifier._looks_like_variable(existing)
						and IPCPlanVerifier._looks_like_variable(grounded_arg)
					):
						continue
					raise ValueError(
						f"variable binding mismatch for '{pattern_arg}': "
						f"expected '{existing}', got '{grounded_arg}'",
					)
				next_bindings[pattern_arg] = grounded_arg
				continue
			if pattern_arg != grounded_arg:
				raise ValueError(
					f"constant mismatch: expected '{pattern_arg}', got '{grounded_arg}'",
				)
		return next_bindings

	@staticmethod
	def _looks_like_variable(token: str) -> bool:
		return bool(token) and (token[0].isupper() or token[0] == "?")

	@staticmethod
	def _default_task_args(
		method: HTNMethod,
		task_lookup: Dict[str, Any],
	) -> Tuple[str, ...]:
		if method.task_args:
			return tuple(method.task_args)
		task_schema = task_lookup.get(method.task_name)
		if task_schema is None:
			return tuple(method.parameters)
		declared_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
		leading_parameters = tuple(method.parameters[: len(declared_parameters)])
		if len(leading_parameters) == len(declared_parameters):
			return leading_parameters
		return declared_parameters

	@staticmethod
	def _ordered_method_steps(method: HTNMethod) -> List[Any]:
		if len(method.subtasks) <= 1 or not method.ordering:
			return list(method.subtasks)

		step_lookup = {
			step.step_id: step
			for step in method.subtasks
		}
		dependents: Dict[str, List[str]] = {
			step.step_id: []
			for step in method.subtasks
		}
		in_degree: Dict[str, int] = {
			step.step_id: 0
			for step in method.subtasks
		}

		for before, after in method.ordering:
			if before not in step_lookup or after not in step_lookup:
				return list(method.subtasks)
			dependents[before].append(after)
			in_degree[after] += 1

		ordered_steps: List[Any] = []
		ready = [
			step.step_id
			for step in method.subtasks
			if in_degree[step.step_id] == 0
		]
		while ready:
			current_id = ready.pop(0)
			ordered_steps.append(step_lookup[current_id])
			for next_id in dependents[current_id]:
				in_degree[next_id] -= 1
				if in_degree[next_id] == 0:
					ready.append(next_id)

		if len(ordered_steps) != len(method.subtasks):
			return list(method.subtasks)
		return ordered_steps

	@staticmethod
	def _parse_action_step(action_step: str) -> _ActionStep:
		bare = action_step.strip()
		if bare and "(" not in bare and ")" not in bare:
			return bare, ()
		match = re.fullmatch(r"([^\s(]+)\((.*)\)", action_step.strip())
		if match is None:
			raise ValueError(f"Invalid action_path step for IPC verifier: {action_step}")
		args = tuple(arg.strip() for arg in match.group(2).split(",") if arg.strip())
		return match.group(1), args

	@staticmethod
	def _action_names_match(expected_action_name: str, actual_action_name: str) -> bool:
		expected = str(expected_action_name or "").strip()
		actual = str(actual_action_name or "").strip()
		if expected == actual:
			return True
		return sanitize_identifier(expected) == sanitize_identifier(actual)

	@staticmethod
	def _render_hierarchical_plan(root_nodes: Sequence[_AbstractNode]) -> str:
		primitive_nodes: List[_PrimitiveNode] = []
		abstract_nodes: List[_AbstractNode] = []

		def visit(node: _PlanNode) -> None:
			if isinstance(node, _PrimitiveNode):
				primitive_nodes.append(node)
				return
			abstract_nodes.append(node)
			for child in node.children:
				visit(child)

		for root_node in root_nodes:
			visit(root_node)

		for index, node in enumerate(primitive_nodes):
			node.node_id = index
		for index, node in enumerate(abstract_nodes, start=len(primitive_nodes)):
			node.node_id = index

		lines = ["==>"]
		for node in primitive_nodes:
			lines.append(" ".join([str(node.node_id), node.name, *node.args]).rstrip())
		lines.append("root " + " ".join(str(node.node_id) for node in root_nodes))
		for node in abstract_nodes:
			child_ids = [str(child.node_id) for child in node.children]
			lines.append(
				" ".join(
					[
						str(node.node_id),
						node.task_name,
						*node.args,
						"->",
						node.method_name,
						*child_ids,
					],
				).rstrip(),
			)
		return "\n".join(lines) + "\n"

	@staticmethod
	def strip_ansi(text: str) -> str:
		return _ANSI_ESCAPE_PATTERN.sub("", text or "")

	@staticmethod
	def _combine_output(stdout: str, stderr: str) -> str:
		if stdout and stderr:
			return f"{stdout.rstrip()}\n{stderr.rstrip()}\n"
		if stdout:
			return stdout if stdout.endswith("\n") else f"{stdout}\n"
		if stderr:
			return stderr if stderr.endswith("\n") else f"{stderr}\n"
		return ""

	@staticmethod
	def _extract_executability(text: str) -> Optional[bool]:
		primitive_only_result = IPCPlanVerifier._extract_bool(
			text,
			"Primitive plan alone executable",
		)
		if primitive_only_result is not None:
			return primitive_only_result
		return IPCPlanVerifier._extract_bool(text, "Plan is executable")

	@staticmethod
	def _extract_bool(text: str, label: str) -> Optional[bool]:
		match = re.search(rf"{re.escape(label)}:\s*(true|false)", text, re.IGNORECASE)
		if match is None:
			return None
		return match.group(1).lower() == "true"

	@staticmethod
	def _infer_goal_reached(text: str) -> Optional[bool]:
		if "Primitive plan does not reach the goal state" in text:
			return False
		if IPCPlanVerifier._extract_bool(text, "Plan verification result") is True:
			return True
		primitive_plan_executable = IPCPlanVerifier._extract_executability(text)
		if primitive_plan_executable is None:
			return None
		return primitive_plan_executable
