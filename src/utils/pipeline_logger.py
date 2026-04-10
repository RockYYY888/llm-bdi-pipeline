"""
Pipeline logger for the LTLf -> DFA -> HTN synthesis -> PANDA -> AgentSpeak pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PipelineRecord:
	"""Complete record of a pipeline execution."""

	timestamp: str
	natural_language: str
	success: bool
	mode: str = "dfa_agentspeak"
	run_origin: str = "src"
	logs_root: str = "logs"
	domain_name: Optional[str] = None
	problem_name: Optional[str] = None

	stage1_status: str = "pending"
	stage1_ltlf_spec: Optional[Dict[str, Any]] = None
	stage1_error: Optional[str] = None
	stage1_used_llm: bool = False
	stage1_model: Optional[str] = None
	stage1_llm_prompt: Optional[Dict[str, str]] = None
	stage1_llm_response: Optional[str] = None

	stage2_status: str = "pending"
	stage2_dfa_result: Optional[Dict[str, Any]] = None
	stage2_formula: Optional[str] = None
	stage2_num_states: int = 0
	stage2_num_transitions: int = 0
	stage2_error: Optional[str] = None

	stage3_status: str = "pending"
	stage3_method_library: Optional[Dict[str, Any]] = None
	stage3_error: Optional[str] = None
	stage3_used_llm: bool = False
	stage3_model: Optional[str] = None
	stage3_llm_prompt: Optional[Dict[str, str]] = None
	stage3_llm_response: Optional[str] = None
	stage3_metadata: Optional[Dict[str, Any]] = None
	negation_resolution: Optional[Dict[str, Any]] = None

	stage4_status: str = "pending"
	stage4_backend: str = "pandaPI"
	stage4_error: Optional[str] = None
	stage4_metadata: Optional[Dict[str, Any]] = None
	stage4_artifacts: Optional[Dict[str, Any]] = None

	stage5_status: str = "pending"
	stage5_agentspeak: Optional[str] = None
	stage5_error: Optional[str] = None
	stage5_code_size_chars: int = 0
	stage5_metadata: Optional[Dict[str, Any]] = None

	stage6_status: str = "pending"
	stage6_backend: str = "RunLocalMAS"
	stage6_error: Optional[str] = None
	stage6_metadata: Optional[Dict[str, Any]] = None
	stage6_artifacts: Optional[Dict[str, Any]] = None

	stage7_status: str = "pending"
	stage7_backend: str = "pandaPIparser"
	stage7_error: Optional[str] = None
	stage7_metadata: Optional[Dict[str, Any]] = None
	stage7_artifacts: Optional[Dict[str, Any]] = None

	domain_file: str = ""
	problem_file: Optional[str] = None
	output_dir: str = "output"
	execution_time_seconds: float = 0.0
	timing_profile: Dict[str, Any] = field(default_factory=dict)


class PipelineLogger:
	"""Save structured JSON and readable logs after each pipeline stage."""

	def __init__(self, logs_dir: str = "logs", run_origin: str = "src") -> None:
		self.logs_dir = Path(logs_dir)
		self.logs_dir.mkdir(parents=True, exist_ok=True)
		self.run_origin = run_origin
		self.current_record: Optional[PipelineRecord] = None
		self.start_time: Optional[datetime] = None
		self.current_log_dir: Optional[Path] = None

	def start_pipeline(
		self,
		natural_language: str,
		mode: str = "dfa_agentspeak",
		domain_file: str = "",
		problem_file: str | None = None,
		domain_name: str | None = None,
		problem_name: str | None = None,
		output_dir: str = "output",
		timestamp: str | None = None,
	) -> None:
		self.start_time = datetime.now()
		if timestamp is None:
			timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

		resolved_domain_name = self._slug_component(domain_name or Path(domain_file).stem)
		resolved_problem_name = self._slug_component(problem_name) if problem_name else None
		dir_parts = [timestamp, resolved_domain_name]
		if resolved_problem_name:
			dir_parts.append(resolved_problem_name)
		dir_name = "_".join(part for part in dir_parts if part)
		self.current_log_dir = self.logs_dir / dir_name
		self.current_log_dir.mkdir(parents=True, exist_ok=True)

		self.current_record = PipelineRecord(
			timestamp=timestamp,
			natural_language=natural_language,
			success=False,
			mode=mode,
			run_origin=self.run_origin,
			logs_root=str(self.logs_dir.resolve()),
			domain_name=domain_name,
			problem_name=problem_name,
			domain_file=domain_file,
			problem_file=problem_file,
			output_dir=output_dir,
		)

	@staticmethod
	def _slug_component(value: str | None) -> str:
		if not value:
			return "unknown"
		return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"

	def _relative_artifact_path(self, path: str | Path | None) -> Optional[str]:
		if not path:
			return None
		candidate = Path(path)
		if self.current_log_dir is not None:
			try:
				return str(candidate.resolve().relative_to(self.current_log_dir.resolve()))
			except Exception:
				try:
					return str(candidate.relative_to(self.current_log_dir))
				except Exception:
					pass
		return candidate.name or str(candidate)

	def _relative_artifact_reference(self, value: Any) -> Any:
		if value is None:
			return None
		if isinstance(value, (str, Path)):
			return self._relative_artifact_path(value)
		if isinstance(value, (list, tuple)):
			items = [
				self._relative_artifact_reference(item)
				for item in value
			]
			items = [item for item in items if item is not None]
			return items or None
		if isinstance(value, dict):
			items = {
				str(key): self._relative_artifact_reference(item)
				for key, item in value.items()
			}
			items = {
				key: item
				for key, item in items.items()
				if item is not None
			}
			return items or None
		return None

	@staticmethod
	def _normalise_timing_breakdown(breakdown: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
		if not breakdown:
			return None
		normalised: Dict[str, Any] = {}
		for key, value in breakdown.items():
			if value is None:
				continue
			try:
				normalised[str(key)] = round(float(value), 6)
			except (TypeError, ValueError):
				continue
		return normalised or None

	def record_stage_timing(
		self,
		stage_name: str,
		duration_seconds: float,
		*,
		breakdown: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		if not self.current_record:
			return

		stage_timing: Dict[str, Any] = {
			"duration_seconds": round(float(duration_seconds), 6),
		}
		normalised_breakdown = self._normalise_timing_breakdown(breakdown)
		if normalised_breakdown:
			stage_timing["breakdown_seconds"] = normalised_breakdown
		if metadata:
			stage_timing["metadata"] = dict(metadata)

		self.current_record.timing_profile = dict(self.current_record.timing_profile or {})
		self.current_record.timing_profile[str(stage_name)] = stage_timing
		self._save_current_state()

	def log_stage1_success(
		self,
		ltl_spec: Dict[str, Any],
		used_llm: bool = False,
		model: Optional[str] = None,
		llm_prompt: Optional[Dict[str, str]] = None,
		llm_response: Optional[str] = None,
	) -> None:
		if not self.current_record:
			return

		self.current_record.stage1_status = "success"
		self.current_record.stage1_ltlf_spec = self._sanitise_stage1_spec(ltl_spec)
		self.current_record.stage1_used_llm = used_llm
		self.current_record.stage1_model = model
		self.current_record.stage1_llm_prompt = llm_prompt
		self.current_record.stage1_llm_response = llm_response

		if ltl_spec and "grounding_map" in ltl_spec and self.current_log_dir:
			grounding_map_path = self.current_log_dir / "grounding_map.json"
			grounding_map_path.write_text(json.dumps(ltl_spec["grounding_map"], indent=2))

		self._save_current_state()

	@staticmethod
	def _sanitise_stage1_spec(ltl_spec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
		if not isinstance(ltl_spec, dict):
			return ltl_spec
		sanitised = dict(ltl_spec)
		# The recursive formula tree is redundant once formulas_string is present and
		# makes execution logs disproportionately large on longer benchmark queries.
		if sanitised.get("formulas_string"):
			sanitised.pop("formulas", None)
		return sanitised

	def log_stage1_error(self, error: str) -> None:
		if not self.current_record:
			return

		self.current_record.stage1_status = "failed"
		self.current_record.stage1_error = str(error)
		self._save_current_state()

	def log_stage1(
		self,
		nl_input: str,
		ltl_spec: Any,
		status: str,
		error: str | None = None,
		model: str | None = None,
		llm_prompt: Dict[str, str] | None = None,
		llm_response: str | None = None,
	) -> None:
		if status == "Success" and ltl_spec:
			self.log_stage1_success(
				ltl_spec.to_dict() if hasattr(ltl_spec, "to_dict") else ltl_spec,
				used_llm=True,
				model=model,
				llm_prompt=llm_prompt,
				llm_response=llm_response,
			)
		elif error:
			self.log_stage1_error(error)

	def log_stage2_dfas(self, ltl_spec: Any, dfa_result: Any, status: str, error: str | None = None) -> None:
		if not self.current_record:
			return

		if status == "Success" and dfa_result:
			self.current_record.stage2_status = "success"
			self.current_record.stage2_dfa_result = self._sanitise_stage2_dfa_result(dfa_result)
			self.current_record.stage2_formula = dfa_result.get("formula", "N/A")
			self.current_record.stage2_num_states = dfa_result.get("num_states", 0)
			self.current_record.stage2_num_transitions = dfa_result.get("num_transitions", 0)

			if self.current_log_dir:
				if "dfa_dot" in dfa_result:
					(self.current_log_dir / "dfa.dot").write_text(dfa_result["dfa_dot"])
		elif error:
			self.current_record.stage2_status = "failed"
			self.current_record.stage2_error = str(error)

		self._save_current_state()

	@staticmethod
	def _sanitise_stage2_dfa_result(dfa_result: Any) -> Any:
		if not isinstance(dfa_result, dict):
			return dfa_result
		sanitised = dict(dfa_result)
		sanitised.pop("dfa_dot", None)
		if "dfa_path" not in sanitised:
			sanitised["dfa_path"] = "dfa.dot"
		return sanitised

	def log_stage3_method_synthesis(
		self,
		method_library: Optional[Dict[str, Any]],
		status: str,
		*,
		error: str | None = None,
		model: str | None = None,
		llm_prompt: Dict[str, str] | None = None,
		llm_response: str | None = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		if not self.current_record:
			return

		self.current_record.stage3_metadata = dict(metadata or {})
		if self.current_record.stage3_metadata.get("negation_resolution"):
			self.current_record.negation_resolution = dict(
				self.current_record.stage3_metadata["negation_resolution"]
			)
		if model:
			self.current_record.stage3_used_llm = True
			self.current_record.stage3_model = model
		if llm_prompt:
			self.current_record.stage3_llm_prompt = llm_prompt
		if llm_response:
			self.current_record.stage3_llm_response = llm_response

		if status == "Success" and method_library is not None:
			self.current_record.stage3_status = "success"
			self.current_record.stage3_metadata = self._build_stage3_summary(
				method_library,
				self.current_record.stage3_metadata,
			)
			if self.current_log_dir:
				(self.current_log_dir / "htn_method_library.json").write_text(
					json.dumps(method_library, indent=2)
				)
			self.current_record.stage3_method_library = {
				"artifact_path": "htn_method_library.json",
			}
		elif error:
			self.current_record.stage3_status = "failed"
			self.current_record.stage3_error = str(error)

		self._save_current_state()

	def _build_stage3_summary(
		self,
		method_library: Dict[str, Any],
		base_metadata: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		summary = dict(base_metadata or {})
		compound_tasks = method_library.get("compound_tasks", [])
		primitive_tasks = method_library.get("primitive_tasks", [])
		methods = method_library.get("methods", [])
		target_literals = method_library.get("target_literals", [])
		target_task_bindings = method_library.get("target_task_bindings", [])

		method_counts_by_task: Dict[str, int] = {}
		for method in methods:
			task_name = method.get("task_name")
			if not task_name:
				continue
			method_counts_by_task[task_name] = method_counts_by_task.get(task_name, 0) + 1

		summary.update(
			{
				"target_literals": [
					self._literal_signature(item)
					for item in target_literals
				],
				"target_task_bindings": target_task_bindings,
				"target_task_binding_count": len(target_task_bindings),
				"compound_tasks": len(compound_tasks),
				"compound_task_names": [task.get("name") for task in compound_tasks if task.get("name")],
				"primitive_tasks": len(primitive_tasks),
				"primitive_task_names": [task.get("name") for task in primitive_tasks if task.get("name")],
				"methods": len(methods),
				"method_counts_by_task": method_counts_by_task,
			}
		)
		return summary

	@staticmethod
	def _literal_signature(item: Dict[str, Any]) -> str:
		predicate = item.get("predicate", "")
		args = item.get("args", [])
		is_positive = bool(item.get("is_positive", True))
		inner = f"{predicate}({', '.join(args)})" if args else predicate
		if is_positive:
			return inner
		return f"!{inner}"

	def log_stage4_panda_planning(
		self,
		transitions: Optional[Dict[str, Any]],
		status: str,
		*,
		error: str | None = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		if not self.current_record:
			return

		self.current_record.stage4_metadata = metadata
		if status == "Success" and transitions is not None:
			self.current_record.stage4_status = "success"
			self.current_record.stage4_artifacts = self._sanitise_stage4_artifacts(transitions)
			if self.current_log_dir:
				(self.current_log_dir / "panda_transitions.json").write_text(
					json.dumps(transitions, indent=2)
				)
		elif error:
			self.current_record.stage4_status = "failed"
			self.current_record.stage4_error = str(error)

		self._save_current_state()

	@staticmethod
	def _sanitise_stage4_artifacts(transitions: Any) -> Any:
		if not isinstance(transitions, dict):
			return transitions
		sanitised = dict(transitions)
		transition_items = list(sanitised.pop("transitions", ()) or ())
		sanitised["transition_count"] = len(transition_items)
		if "transitions_path" not in sanitised:
			sanitised["transitions_path"] = "panda_transitions.json"
		return sanitised

	def log_stage5_agentspeak_rendering(
		self,
		agentspeak_code: Optional[str],
		status: str,
		*,
		error: str | None = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		if not self.current_record:
			return

		self.current_record.stage5_metadata = metadata
		if status == "Success" and agentspeak_code is not None:
			self.current_record.stage5_status = "success"
			self.current_record.stage5_agentspeak = "agentspeak_generated.asl"
			self.current_record.stage5_code_size_chars = len(agentspeak_code)
		elif error:
			self.current_record.stage5_status = "failed"
			self.current_record.stage5_error = str(error)

		self._save_current_state()

	def log_stage6_jason_validation(
		self,
		artifacts: Optional[Dict[str, Any]],
		status: str,
		*,
		error: str | None = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		if not self.current_record:
			return

		self.current_record.stage6_metadata = metadata
		backend = None
		if isinstance(metadata, dict):
			backend = metadata.get("backend")
		if not backend and isinstance(artifacts, dict):
			backend = artifacts.get("backend")
		if backend:
			self.current_record.stage6_backend = str(backend)
		if status == "Success" and artifacts is not None:
			self.current_record.stage6_status = "success"
			self.current_record.stage6_artifacts = self._sanitise_stage6_artifacts(artifacts)
		elif error:
			self.current_record.stage6_status = "failed"
			self.current_record.stage6_error = str(error)
			self.current_record.stage6_artifacts = self._sanitise_stage6_artifacts(artifacts)

		self._save_current_state()

	def _sanitise_stage6_artifacts(self, artifacts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
		if not isinstance(artifacts, dict):
			return artifacts
		sanitised = {
			"status": artifacts.get("status"),
			"backend": artifacts.get("backend"),
			"java_path": artifacts.get("java_path"),
			"java_version": artifacts.get("java_version"),
			"javac_path": artifacts.get("javac_path"),
			"jason_jar": artifacts.get("jason_jar"),
			"exit_code": artifacts.get("exit_code"),
			"timed_out": artifacts.get("timed_out"),
			"failed_goals": list(artifacts.get("failed_goals") or ()),
			"failed_goal_count": len(artifacts.get("failed_goals") or ()),
			"environment_adapter": artifacts.get("environment_adapter"),
			"failure_class": artifacts.get("failure_class"),
			"consistency_checks": artifacts.get("consistency_checks"),
		}
		artifact_paths = {
			key: self._relative_artifact_reference(value)
			for key, value in dict(artifacts.get("artifacts") or {}).items()
			if self._relative_artifact_reference(value) is not None
		}
		if artifact_paths:
			sanitised["artifacts"] = artifact_paths
			action_path_path = artifact_paths.get("action_path")
			if action_path_path:
				sanitised["action_path_path"] = action_path_path
			method_trace_path = artifact_paths.get("method_trace")
			if method_trace_path:
				sanitised["method_trace_path"] = method_trace_path
			stdout_path = artifact_paths.get("jason_stdout")
			if stdout_path:
				sanitised["stdout_path"] = stdout_path
			stderr_path = artifact_paths.get("jason_stderr")
			if stderr_path:
				sanitised["stderr_path"] = stderr_path
		guided_plan_text = artifacts.get("guided_hierarchical_plan_text")
		if guided_plan_text and self.current_log_dir is not None:
			guided_plan_path = self.current_log_dir / "guided_hierarchical_plan.txt"
			guided_plan_path.write_text(str(guided_plan_text))
			sanitised["guided_hierarchical_plan_path"] = guided_plan_path.name
		if artifacts.get("guided_hierarchical_plan_source") is not None:
			sanitised["guided_hierarchical_plan_source"] = artifacts.get(
				"guided_hierarchical_plan_source",
			)
		return sanitised

	def log_stage7_official_verification(
		self,
		artifacts: Optional[Dict[str, Any]],
		status: str,
		*,
		error: str | None = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		if not self.current_record:
			return

		self.current_record.stage7_metadata = metadata
		backend = None
		if isinstance(metadata, dict):
			backend = metadata.get("backend")
		if not backend and isinstance(artifacts, dict):
			backend = artifacts.get("backend")
		if backend:
			self.current_record.stage7_backend = str(backend)
		if status == "Success" and artifacts is not None:
			self.current_record.stage7_status = "success"
			self.current_record.stage7_artifacts = self._sanitise_stage7_artifacts(artifacts)
		elif status == "Skipped":
			self.current_record.stage7_status = "skipped"
			self.current_record.stage7_artifacts = self._sanitise_stage7_artifacts(artifacts)
		elif error:
			self.current_record.stage7_status = "failed"
			self.current_record.stage7_error = str(error)
			self.current_record.stage7_artifacts = self._sanitise_stage7_artifacts(artifacts)

		self._save_current_state()

	def _sanitise_stage7_artifacts(self, artifacts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
		if not isinstance(artifacts, dict):
			return artifacts
		sanitised = dict(artifacts)
		sanitised.pop("stdout", None)
		sanitised.pop("stderr", None)
		command = sanitised.get("command")
		if command is not None:
			sanitised["command"] = list(command)
		if "plan_file" in sanitised:
			sanitised["plan_file"] = self._relative_artifact_path(sanitised.get("plan_file"))
		if "output_file" in sanitised:
			sanitised["output_file"] = self._relative_artifact_path(sanitised.get("output_file"))
		return sanitised

	def _save_current_state(self) -> None:
		if not self.current_record or not self.current_log_dir:
			return

		if self.start_time:
			self.current_record.execution_time_seconds = (
				datetime.now() - self.start_time
			).total_seconds()

		record_dict = asdict(self.current_record)
		(self.current_log_dir / "execution.json").write_text(json.dumps(record_dict, indent=2))
		self._save_readable_format(self.current_log_dir / "execution.txt", record_dict)

	def end_pipeline(self, success: bool = True) -> Path:
		if not self.current_record or not self.start_time or not self.current_log_dir:
			raise RuntimeError("No active pipeline record to end")

		self.current_record.success = success
		self._save_current_state()
		return self.current_log_dir / "execution.json"

	def _save_readable_format(self, filepath: Path, record: Dict[str, Any]) -> None:
		with filepath.open("w") as handle:
			handle.write("=" * 80 + "\n")
			handle.write("LTL PIPELINE EXECUTION RECORD\n")
			handle.write("=" * 80 + "\n\n")

			handle.write(f"Timestamp: {record['timestamp']}\n")
			handle.write(f"Execution Time: {record['execution_time_seconds']:.2f} seconds\n")
			handle.write(f"Overall Status: {'✓ SUCCESS' if record['success'] else '✗ FAILED'}\n")
			handle.write(f"Run Origin: {record['run_origin']}\n")
			handle.write(f"Logs Root: {record['logs_root']}\n")
			if record.get("domain_name"):
				handle.write(f"Domain Name: {record['domain_name']}\n")
			if record.get("problem_name"):
				handle.write(f"Problem Name: {record['problem_name']}\n")
			handle.write(f"Domain: {record['domain_file']}\n")
			if record.get("problem_file"):
				handle.write(f"Problem: {record['problem_file']}\n")
			handle.write(f"Output Directory: {record['output_dir']}\n\n")

			handle.write("-" * 80 + "\n")
			handle.write("INPUT\n")
			handle.write("-" * 80 + "\n")
			handle.write(f"\"{record['natural_language']}\"\n\n")

			self._write_timing_profile(handle, record)

			self._write_stage1(handle, record)
			self._write_stage2(handle, record)
			self._write_stage3(handle, record)
			self._write_stage4(handle, record)
			self._write_stage5(handle, record)
			self._write_stage6(handle, record)
			self._write_stage7(handle, record)

			handle.write("=" * 80 + "\n")
			handle.write("END OF RECORD\n")
			handle.write("=" * 80 + "\n")

	def _write_timing_profile(self, handle: Any, record: Dict[str, Any]) -> None:
		timing_profile = dict(record.get("timing_profile") or {})
		if not timing_profile:
			return

		handle.write("-" * 80 + "\n")
		handle.write("TIMING PROFILE\n")
		handle.write("-" * 80 + "\n")
		for stage_name in (
			"stage1",
			"stage2",
			"stage3",
			"stage4",
			"stage5",
			"stage6",
			"stage7",
		):
			stage_timing = timing_profile.get(stage_name)
			if not isinstance(stage_timing, dict):
				continue
			duration_seconds = stage_timing.get("duration_seconds")
			if duration_seconds is None:
				continue
			handle.write(f"{stage_name}: {duration_seconds:.3f}s\n")
			for key, value in dict(stage_timing.get("breakdown_seconds") or {}).items():
				handle.write(f"  - {key}: {value:.3f}s\n")
		handle.write("\n")

	def _write_stage_timing(self, handle: Any, record: Dict[str, Any], stage_name: str) -> None:
		stage_timing = dict((record.get("timing_profile") or {}).get(stage_name) or {})
		duration_seconds = stage_timing.get("duration_seconds")
		if duration_seconds is None:
			return
		handle.write(f"Duration: {duration_seconds:.3f} seconds\n")
		breakdown = dict(stage_timing.get("breakdown_seconds") or {})
		if not breakdown:
			return
		handle.write("Breakdown:\n")
		for key, value in breakdown.items():
			handle.write(f"  - {key}: {value:.3f} seconds\n")

	def _write_stage1(self, handle: Any, record: Dict[str, Any]) -> None:
		handle.write("-" * 80 + "\n")
		handle.write("STAGE 1: Natural Language → LTL Specification\n")
		handle.write("-" * 80 + "\n")
		handle.write(f"Status: {record['stage1_status'].upper()}\n")
		self._write_stage_timing(handle, record, "stage1")
		if record["stage1_used_llm"]:
			handle.write(f"Parser: LLM ({record['stage1_model']})\n")
		else:
			handle.write("Parser: Not configured\n")

		if record["stage1_used_llm"] and record["stage1_llm_prompt"]:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("System Prompt\n")
			handle.write("~" * 40 + "\n")
			handle.write(record["stage1_llm_prompt"].get("system", "N/A"))
			handle.write("\n\n" + "~" * 40 + "\n")
			handle.write("User Prompt\n")
			handle.write("~" * 40 + "\n")
			handle.write(record["stage1_llm_prompt"].get("user", "N/A"))
			handle.write("\n")

		if record["stage1_used_llm"] and record["stage1_llm_response"]:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("LLM RESPONSE (Stage 1)\n")
			handle.write("~" * 40 + "\n")
			handle.write(record["stage1_llm_response"])
			handle.write("\n")

		if record["stage1_status"] == "success" and record["stage1_ltlf_spec"]:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("PARSED OUTPUT (Stage 1)\n")
			handle.write("~" * 40 + "\n")
			ltlf = record["stage1_ltlf_spec"]
			handle.write(f"Objects: {ltlf.get('objects', [])}\n")
			handle.write(
				"\nLTLf Formulas (goal semantics only; Stage 4 later instantiates planning state):\n"
			)
			for index, formula_str in enumerate(ltlf.get("formulas_string", []), start=1):
				handle.write(f"  {index}. {formula_str}\n")
		elif record["stage1_error"]:
			handle.write(f"\nError: {record['stage1_error']}\n")

		handle.write("\n")

	def _write_stage2(self, handle: Any, record: Dict[str, Any]) -> None:
		handle.write("-" * 80 + "\n")
		handle.write("STAGE 2: LTL Specification → DFA Generation\n")
		handle.write("-" * 80 + "\n")
		handle.write(f"Status: {record['stage2_status'].upper()}\n")
		self._write_stage_timing(handle, record, "stage2")

		if record["stage2_status"] == "success" and record["stage2_dfa_result"]:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("DFA GENERATION RESULT\n")
			handle.write("~" * 40 + "\n")
			handle.write(f"Formula: {record.get('stage2_formula', 'N/A')}\n\n")
			handle.write("Raw DFA:\n")
			handle.write(f"  States: {record.get('stage2_num_states', 0)}\n")
			handle.write(f"  Transitions: {record.get('stage2_num_transitions', 0)}\n")
			handle.write("  File: dfa.dot\n\n")
			handle.write(
				"Full DFA bodies are stored in dfa.dot; "
				"execution artifacts keep only metadata to avoid pathological log bloat.\n\n"
			)
		elif record["stage2_error"]:
			handle.write(f"\nError: {record['stage2_error']}\n\n")
		else:
			handle.write("\n")

	def _write_stage3(self, handle: Any, record: Dict[str, Any]) -> None:
		handle.write("-" * 80 + "\n")
		handle.write("STAGE 3: DFA → HTN Method Synthesis\n")
		handle.write("-" * 80 + "\n")
		handle.write(f"Status: {record['stage3_status'].upper()}\n")
		self._write_stage_timing(handle, record, "stage3")
		if record["stage3_used_llm"]:
			handle.write(f"Generator: LLM ({record['stage3_model']})\n")
		else:
			handle.write("Generator: Not configured\n")

		if record["stage3_used_llm"] and record["stage3_llm_prompt"]:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("System Prompt\n")
			handle.write("~" * 40 + "\n")
			handle.write(record["stage3_llm_prompt"].get("system", "N/A"))
			handle.write("\n\n" + "~" * 40 + "\n")
			handle.write("User Prompt\n")
			handle.write("~" * 40 + "\n")
			handle.write(record["stage3_llm_prompt"].get("user", "N/A"))
			handle.write("\n")

		if record["stage3_used_llm"] and record["stage3_llm_response"]:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("LLM RESPONSE (Stage 3)\n")
			handle.write("~" * 40 + "\n")
			handle.write(record["stage3_llm_response"])
			handle.write("\n")

		if record["stage3_metadata"]:
			handle.write("\n" + "~" * 40 + "\n")
			title = "HTN METHOD SYNTHESIS SUMMARY" if record["stage3_status"] == "success" else "HTN METHOD SYNTHESIS DIAGNOSTICS"
			handle.write(f"{title}\n")
			handle.write("~" * 40 + "\n")
			for key, value in record["stage3_metadata"].items():
				handle.write(f"{key}: {value}\n")
			handle.write("\n")

		if record.get("negation_resolution"):
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("NEGATION RESOLUTION\n")
			handle.write("~" * 40 + "\n")
			handle.write(json.dumps(record["negation_resolution"], indent=2))
			handle.write("\n")

		if record["stage3_status"] == "success" and record["stage3_method_library"] is not None:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("HTN METHOD LIBRARY (Stage 3)\n")
			handle.write("~" * 40 + "\n")
			handle.write(
				f"Artifact path: {record['stage3_method_library'].get('artifact_path', 'htn_method_library.json')}\n"
			)
		elif record["stage3_error"]:
			handle.write(f"\nError: {record['stage3_error']}\n")

		handle.write("\n")

	def _write_stage4(self, handle: Any, record: Dict[str, Any]) -> None:
		handle.write("-" * 80 + "\n")
		handle.write("STAGE 4: HTN Method Library → PANDA Planning\n")
		handle.write("-" * 80 + "\n")
		handle.write(f"Status: {record['stage4_status'].upper()}\n")
		self._write_stage_timing(handle, record, "stage4")
		handle.write(f"Backend: {record['stage4_backend']}\n")

		if record["stage4_metadata"]:
			handle.write("\n" + "~" * 40 + "\n")
			title = "PANDA PLANNING SUMMARY" if record["stage4_status"] == "success" else "PANDA PLANNING DIAGNOSTICS"
			handle.write(f"{title}\n")
			handle.write("~" * 40 + "\n")
			for key, value in record["stage4_metadata"].items():
				handle.write(f"{key}: {value}\n")
			handle.write("\n")

		if record["stage4_status"] == "success" and record["stage4_artifacts"] is not None:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("PANDA PLAN ARTIFACTS (Stage 4)\n")
			handle.write("~" * 40 + "\n")
			handle.write(
				f"Artifact path: {record['stage4_artifacts'].get('transitions_path', 'panda_transitions.json')}\n"
			)
			handle.write(json.dumps(record["stage4_artifacts"], indent=2))
			handle.write("\n")
		elif record["stage4_error"]:
			handle.write(f"\nError: {record['stage4_error']}\n")

		handle.write("\n")

	def _write_stage5(self, handle: Any, record: Dict[str, Any]) -> None:
		handle.write("-" * 80 + "\n")
		handle.write("STAGE 5: HTN Methods + DFA Wrappers → AgentSpeak Rendering\n")
		handle.write("-" * 80 + "\n")
		handle.write(f"Status: {record['stage5_status'].upper()}\n")
		self._write_stage_timing(handle, record, "stage5")

		if record["stage5_metadata"]:
			handle.write("\n" + "~" * 40 + "\n")
			title = "AGENTSPEAK RENDERING SUMMARY" if record["stage5_status"] == "success" else "AGENTSPEAK RENDERING DIAGNOSTICS"
			handle.write(f"{title}\n")
			handle.write("~" * 40 + "\n")
			for key, value in record["stage5_metadata"].items():
				handle.write(f"{key}: {value}\n")
			handle.write("\n")

		if record["stage5_status"] == "success" and record["stage5_agentspeak"]:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("GENERATED AGENTSPEAK CODE (Stage 5)\n")
			handle.write("~" * 40 + "\n")
			handle.write(f"Artifact path: {record['stage5_agentspeak']}\n")
		elif record["stage5_error"]:
			handle.write(f"\nError: {record['stage5_error']}\n")

		handle.write("\n")

	def _write_stage6(self, handle: Any, record: Dict[str, Any]) -> None:
		handle.write("-" * 80 + "\n")
		handle.write("STAGE 6: AgentSpeak → Jason Runtime Validation\n")
		handle.write("-" * 80 + "\n")
		handle.write(f"Status: {record['stage6_status'].upper()}\n")
		self._write_stage_timing(handle, record, "stage6")
		handle.write(f"Backend: {record['stage6_backend']}\n")

		if record["stage6_metadata"]:
			handle.write("\n" + "~" * 40 + "\n")
			title = (
				"JASON VALIDATION SUMMARY"
				if record["stage6_status"] == "success"
				else "JASON VALIDATION DIAGNOSTICS"
			)
			handle.write(f"{title}\n")
			handle.write("~" * 40 + "\n")
			for key, value in record["stage6_metadata"].items():
				handle.write(f"{key}: {value}\n")
			handle.write("\n")

		if record["stage6_status"] == "success" and record["stage6_artifacts"] is not None:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("JASON ARTIFACTS (Stage 6)\n")
			handle.write("~" * 40 + "\n")
			handle.write(json.dumps(record["stage6_artifacts"], indent=2))
			handle.write("\n")
		elif record["stage6_error"]:
			handle.write(f"\nError: {record['stage6_error']}\n")

		handle.write("\n")

	def _write_stage7(self, handle: Any, record: Dict[str, Any]) -> None:
		handle.write("-" * 80 + "\n")
		handle.write("STAGE 7: Official IPC HTN Plan Verification\n")
		handle.write("-" * 80 + "\n")
		handle.write(f"Status: {record['stage7_status'].upper()}\n")
		self._write_stage_timing(handle, record, "stage7")
		handle.write(f"Backend: {record['stage7_backend']}\n")

		if record["stage7_metadata"]:
			handle.write("\n" + "~" * 40 + "\n")
			title = (
				"OFFICIAL VERIFICATION SUMMARY"
				if record["stage7_status"] == "success"
				else "OFFICIAL VERIFICATION DIAGNOSTICS"
			)
			handle.write(f"{title}\n")
			handle.write("~" * 40 + "\n")
			for key, value in record["stage7_metadata"].items():
				handle.write(f"{key}: {value}\n")
			handle.write("\n")

		if record["stage7_status"] == "success" and record["stage7_artifacts"] is not None:
			handle.write("\n" + "~" * 40 + "\n")
			handle.write("OFFICIAL VERIFICATION ARTIFACTS (Stage 7)\n")
			handle.write("~" * 40 + "\n")
			handle.write(json.dumps(record["stage7_artifacts"], indent=2))
			handle.write("\n")
		elif record["stage7_error"]:
			handle.write(f"\nError: {record['stage7_error']}\n")

		handle.write("\n")
