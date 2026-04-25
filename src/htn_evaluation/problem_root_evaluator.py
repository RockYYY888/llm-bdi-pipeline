"""
Problem-root Hierarchical Task Network evaluation orchestration.

This module owns the planner-based benchmark evaluation tracks for official and
generated domains. It is a reference and diagnostic layer, not the deployed
evaluation runtime.
"""

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import queue
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from planning.linearization import LiftedLinearPlanner
from planning.panda_sat import PANDAPlanningError
from planning.primary_planner import (
	PrimaryPlannerTask,
	default_primary_planners,
	expand_primary_planner_tasks_for_representations,
)
from planning.representations import PlanningRepresentation, RepresentationBuildResult
from .problem_root_runtime import official_problem_root_planning_task_worker
from .result_tables import (
	PRIMARY_HTN_PLANNER_ID,
	PRIMARY_PLANNER_SELECTION_RULE,
	SINGLE_PLANNER_MODE,
	validate_evaluation_mode,
	validate_planner_id,
)


PRIMARY_PLANNER_RESULT_MESSAGE = "primary_planner_attempt"
HTN_ATTEMPT_TEXT_PREVIEW_CHARS = 4096


def _attempt_text_preview(value: Any) -> str:
	text = str(value or "")
	if len(text) <= HTN_ATTEMPT_TEXT_PREVIEW_CHARS:
		return text
	head_limit = HTN_ATTEMPT_TEXT_PREVIEW_CHARS // 2
	tail_limit = HTN_ATTEMPT_TEXT_PREVIEW_CHARS - head_limit
	return (
		text[:head_limit]
		+ f"\n...[truncated {len(text) - HTN_ATTEMPT_TEXT_PREVIEW_CHARS} chars]...\n"
		+ text[-tail_limit:]
	)


def _planner_attempt_summary(attempt: Dict[str, Any]) -> Dict[str, Any]:
	plan_solve_summary = dict((attempt.get("plan_solve_data") or {}).get("summary") or {})
	plan_verification_summary = dict(
		(attempt.get("plan_verification_data") or {}).get("summary") or {},
	)
	stdout = str(attempt.get("stdout") or "")
	stderr = str(attempt.get("stderr") or "")
	return {
		"planner_id": str(attempt.get("planner_id") or "unknown"),
		"task_id": str(attempt.get("task_id") or "unknown"),
		"representation_id": str(attempt.get("representation_id") or "unknown"),
		"success": bool(attempt.get("success")),
		"selected_bucket": attempt.get("selected_bucket"),
		"resource_profile": dict(attempt.get("resource_profile") or {}),
		"plan_solve_status": plan_solve_summary.get("status"),
		"plan_verification_status": plan_verification_summary.get("status"),
		"plan_solve_failure_bucket": plan_solve_summary.get("failure_bucket"),
		"plan_verification_failure_bucket": plan_verification_summary.get("failure_bucket"),
		"total_seconds": attempt.get("total_seconds"),
		"plan_solve_seconds": attempt.get("plan_solve_seconds"),
		"plan_verification_seconds": attempt.get("plan_verification_seconds"),
		"output_dir": attempt.get("output_dir"),
		"stdout_preview": _attempt_text_preview(stdout),
		"stderr_preview": _attempt_text_preview(stderr),
		"stdout_chars": len(stdout),
		"stderr_chars": len(stderr),
	}


class HTNProblemRootEvaluator:
	"""Run Hierarchical Task Network planner-based benchmark evaluation."""

	def __init__(self, pipeline_context: Any) -> None:
		self.context = pipeline_context

	def planning_tasks(
		self,
		timeout_seconds: Optional[float] = None,
		*,
		planner_id: Optional[str] = None,
	) -> tuple[PrimaryPlannerTask, ...]:
		build_result = self.context._build_problem_representations(
			timeout_seconds=timeout_seconds,
		)
		selected_planners = tuple(
			planner
			for planner in default_primary_planners(workspace=str(self.context.output_dir))
			if planner_id is None or planner.planner_id == planner_id
		)
		if planner_id is not None and not selected_planners:
			raise ValueError(f"Unknown primary HTN planner '{planner_id}'.")
		tasks = expand_primary_planner_tasks_for_representations(
			representations=build_result.representations,
			planners=selected_planners,
		)
		if tasks:
			return tasks
		if not self._selected_planners_require_linearized_input(selected_planners):
			return tasks
		return expand_primary_planner_tasks_for_representations(
			representations=(
				*self._representations_without_duplicate_linearized(build_result),
				self._build_linearized_representation_for_primary_planner(
					build_result=build_result,
					timeout_seconds=timeout_seconds,
				),
			),
			planners=selected_planners,
		)

	@staticmethod
	def _selected_planners_require_linearized_input(
		selected_planners: Sequence[Any],
	) -> bool:
		return any(
			"linearized" in set(getattr(planner, "supported_sources", frozenset()))
			for planner in selected_planners
		)

	@staticmethod
	def _representations_without_duplicate_linearized(
		build_result: RepresentationBuildResult,
	) -> tuple[PlanningRepresentation, ...]:
		return tuple(
			representation
			for representation in build_result.representations
			if representation.representation_id != "linearized_total_order"
		)

	def _build_linearized_representation_for_primary_planner(
		self,
		*,
		build_result: RepresentationBuildResult,
		timeout_seconds: Optional[float],
	) -> PlanningRepresentation:
		if self.context.output_dir is None:
			raise ValueError("Linearized primary-planner representation requires an output directory.")
		workspace = Path(self.context.output_dir).resolve() / "representations"
		linearizer = LiftedLinearPlanner(workspace=workspace)
		artifacts = linearizer.linearize_hddl_files(
			domain_file=self.context.domain_file,
			problem_file=self.context.problem_file,
			transition_name="official_problem_root_primary_planner_linearization",
			timeout_seconds=timeout_seconds,
		)
		metadata = dict(artifacts)
		metadata.update(
			{
				"forced_for_primary_planner_capability": True,
				"original_requires_linearization": bool(
					build_result.structure.requires_linearization,
				),
			},
		)
		return PlanningRepresentation(
			representation_id="linearized_total_order",
			representation_source="linearized",
			ordering_kind="total_order",
			domain_file=str(Path(artifacts["linearized_domain_file"]).resolve()),
			problem_file=str(Path(artifacts["linearized_problem_file"]).resolve()),
			compilation_profile="semantics_preserving_linearization",
			metadata=metadata,
		)

	def run_primary_planner_evaluation(
		self,
		*,
		evaluation_mode: str = SINGLE_PLANNER_MODE,
		planner_id: Optional[str] = PRIMARY_HTN_PLANNER_ID,
	) -> Dict[str, Any]:
		if self.context.output_dir is None:
			raise ValueError("Official problem-root planner evaluation requires an output directory.")

		mode = validate_evaluation_mode(evaluation_mode)
		normalized_planner_id = validate_planner_id(
			planner_id,
			evaluation_mode=mode,
		)
		planning_timeout_seconds = self.context._official_problem_root_planning_timeout_seconds()
		planner_root = Path(self.context.output_dir) / "primary_planner"
		planner_root.mkdir(parents=True, exist_ok=True)
		representation_build_start = time.perf_counter()
		try:
			planning_tasks = self.planning_tasks(
				timeout_seconds=planning_timeout_seconds,
				planner_id=normalized_planner_id,
			)
		except Exception as exc:
			representation_build_seconds = time.perf_counter() - representation_build_start
			selected_attempt = self.representation_build_failure_attempt(
				exc,
				planner_root=planner_root,
				planner_id=normalized_planner_id,
				total_seconds=representation_build_seconds,
			)
			return {
				"evaluation_mode": mode,
				"requested_planner_id": normalized_planner_id,
				"planning_tasks": [],
				"attempts": [selected_attempt],
				"selected_attempt": selected_attempt,
				"representation_build_seconds": representation_build_seconds,
				"planner_wallclock_seconds": 0.0,
			}
		representation_build_seconds = time.perf_counter() - representation_build_start
		context = multiprocessing.get_context("spawn")
		attempts: List[Dict[str, Any]] = []
		planner_start = time.perf_counter()
		selected_attempt: Optional[Dict[str, Any]] = None

		def incomplete_attempt(
			planning_task: PrimaryPlannerTask,
			*,
			failure_reason: str,
			total_seconds: float,
		) -> Dict[str, Any]:
			return {
				"message_type": PRIMARY_PLANNER_RESULT_MESSAGE,
				"planner_id": planning_task.planner_id,
				"task_id": planning_task.task_id,
				"representation_id": planning_task.representation.representation_id,
				"output_dir": str((planner_root / planning_task.task_id).resolve()),
				"plan_solve_data": {
					"summary": {
						"planner_id": planning_task.planner_id,
						"status": "failed",
						"planning_mode": "official_problem_root",
						"failure_bucket": "no_plan_from_solver",
						"representation_id": planning_task.representation.representation_id,
					},
					"artifacts": {
						"planner_id": planning_task.planner_id,
						"status": "failed",
						"planning_mode": "official_problem_root",
						"failure_bucket": "no_plan_from_solver",
						"planning_representation": planning_task.representation.to_dict(),
						"failure_reason": failure_reason,
					},
				},
				"plan_verification_data": {
					"summary": {
						"verifier_id": "pandaPIparser",
						"status": "failed",
						"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
						"failure_bucket": "no_plan_from_solver",
					},
					"artifacts": {
						"planner_id": planning_task.planner_id,
						"status": "failed",
						"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
						"failure_bucket": "no_plan_from_solver",
						"failure_reason": failure_reason,
					},
				},
				"plan_solve_seconds": None,
				"plan_verification_seconds": None,
				"total_seconds": total_seconds,
				"success": False,
				"selected_bucket": "no_plan_from_solver",
				"stdout": "",
				"stderr": "",
			}

		def run_single_task(planning_task: PrimaryPlannerTask) -> Dict[str, Any]:
			raw_remaining_timeout = planning_timeout_seconds - (time.perf_counter() - planner_start)
			if raw_remaining_timeout <= 0.0:
				return incomplete_attempt(
					planning_task,
					failure_reason="planner_timeout_budget_exhausted_before_primary_planner_launch",
					total_seconds=planning_timeout_seconds,
				)
			remaining_timeout = max(raw_remaining_timeout, 1.0)
			attempt_output_dir = planner_root / planning_task.task_id
			attempt_output_dir.mkdir(parents=True, exist_ok=True)
			result_queue = context.Queue()
			process = context.Process(
				target=official_problem_root_planning_task_worker,
				kwargs={
					"result_queue": result_queue,
					"domain_file": str(Path(self.context.domain_file).resolve()),
					"problem_file": str(Path(self.context.problem_file).resolve()),
					"output_dir": str(attempt_output_dir.resolve()),
					"task_payload": planning_task.to_dict(),
					"planning_timeout_seconds": remaining_timeout,
				},
			)
			attempt: Optional[Dict[str, Any]] = None
			try:
				process.start()
				deadline = time.perf_counter() + remaining_timeout + 5.0
				try:
					while True:
						try:
							wait_seconds = max(min(deadline - time.perf_counter(), 5.0), 0.1)
							message = result_queue.get(timeout=wait_seconds)
							attempt = dict(message)
							break
						except queue.Empty:
							if time.perf_counter() >= deadline:
								break
							if not process.is_alive():
								break
				finally:
					process.join(timeout=1.0)
					if process.is_alive():
						self.terminate_planner_process(process)
						process.join(timeout=1.0)
			finally:
				self.close_planner_queue(result_queue)
			if attempt is not None:
				return attempt
			return incomplete_attempt(
				planning_task,
				failure_reason="planner_attempt_incomplete_before_deadline",
				total_seconds=planning_timeout_seconds,
			)

		for planning_task in planning_tasks:
			attempt = run_single_task(planning_task)
			attempts.append(attempt)
			if bool(attempt.get("success")):
				selected_attempt = dict(attempt)
				break

		if selected_attempt is None:
			selected_attempt = self.select_planner_attempt(attempts)

		return {
			"evaluation_mode": mode,
			"requested_planner_id": normalized_planner_id,
			"planning_tasks": [task.to_dict() for task in planning_tasks],
			"attempts": attempts,
			"selected_attempt": selected_attempt,
			"representation_build_seconds": representation_build_seconds,
			"planner_wallclock_seconds": time.perf_counter() - planner_start,
		}

	def representation_build_failure_attempt(
		self,
		exc: Exception,
		*,
		planner_root: Path,
		planner_id: Optional[str],
		total_seconds: float,
	) -> Dict[str, Any]:
		"""Convert representation build failures into one checkpointable query result."""
		failure_metadata = dict(getattr(exc, "metadata", {}) or {})
		planner_id = str(planner_id or failure_metadata.get("planner_id") or "unknown")
		representation_id = (
			"linearized_total_order"
			if planner_id == "lifted_panda_sat"
			or "linearizer" in failure_metadata
			or isinstance(exc, PANDAPlanningError)
			else "representation_build"
		)
		task_id = f"{planner_id}@{representation_id}"
		output_dir = planner_root / "representation_build_failure"
		output_dir.mkdir(parents=True, exist_ok=True)
		failure_reason = f"representation_build_failed: {exc}"
		if failure_metadata:
			(output_dir / "representation_build_failure_metadata.json").write_text(
				self._safe_json_dump(failure_metadata),
			)
		return {
			"message_type": PRIMARY_PLANNER_RESULT_MESSAGE,
			"planner_id": planner_id,
			"task_id": task_id,
			"representation_id": representation_id,
			"output_dir": str(output_dir.resolve()),
			"plan_solve_data": {
				"summary": {
					"planner_id": planner_id,
					"status": "failed",
					"planning_mode": "official_problem_root",
					"failure_bucket": "no_plan_from_solver",
					"failure_stage": "representation_build",
					"failure_reason": failure_reason,
					"solver_id": planner_id,
					"representation_id": representation_id,
				},
				"artifacts": {
					"planner_id": planner_id,
					"status": "failed",
					"planning_mode": "official_problem_root",
					"failure_bucket": "no_plan_from_solver",
					"failure_stage": "representation_build",
					"failure_reason": failure_reason,
					"failure_metadata": failure_metadata,
					"solver_candidates": list(failure_metadata.get("engine_attempts") or ()),
				},
			},
			"plan_verification_data": {
				"summary": {
					"verifier_id": "pandaPIparser",
					"status": "failed",
					"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
					"failure_bucket": "no_plan_from_solver",
					"failure_stage": "representation_build",
					"failure_reason": failure_reason,
				},
				"artifacts": {
					"planner_id": planner_id,
					"status": "failed",
					"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
					"failure_bucket": "no_plan_from_solver",
					"failure_stage": "representation_build",
					"failure_reason": failure_reason,
					"selected_solver_id": planner_id,
					"selected_planner_id": planner_id,
					"selected_representation_id": representation_id,
					"selected_bucket": "no_plan_from_solver",
					"failure_metadata": failure_metadata,
					"solver_candidates": list(failure_metadata.get("engine_attempts") or ()),
				},
			},
			"plan_solve_seconds": 0.0,
			"plan_verification_seconds": 0.0,
			"total_seconds": total_seconds,
			"success": False,
			"selected_bucket": "no_plan_from_solver",
			"resource_profile": {},
			"stdout": str(failure_metadata.get("linearizer_stdout") or ""),
			"stderr": str(failure_metadata.get("linearizer_stderr") or ""),
		}

	@staticmethod
	def _safe_json_dump(payload: Dict[str, Any]) -> str:
		return json.dumps(payload, indent=2, default=str)

	def run_primary_planner(self) -> Dict[str, Any]:
		return self.run_primary_planner_evaluation(
			evaluation_mode=SINGLE_PLANNER_MODE,
			planner_id=PRIMARY_HTN_PLANNER_ID,
		)

	def execute_problem_root_evaluation(
		self,
		method_library=None,
		*,
		evaluation_mode: str = SINGLE_PLANNER_MODE,
		planner_id: Optional[str] = PRIMARY_HTN_PLANNER_ID,
	) -> Dict[str, Any]:
		mode = validate_evaluation_mode(evaluation_mode)
		normalized_planner_id = validate_planner_id(
			planner_id,
			evaluation_mode=mode,
		)
		print("\n[PLAN SOLVE]")
		print("-" * 80)
		planner_result = self.run_primary_planner_evaluation(
			evaluation_mode=mode,
			planner_id=normalized_planner_id,
		)
		planning_tasks = list(planner_result.get("planning_tasks") or ())
		task_labels = [
			(
				f"{task.get('planner_id')}@"
				f"{((task.get('representation') or {}).get('representation_id') or 'unknown')}"
			)
			for task in planning_tasks
		]
		print(f"• Running official planning tasks sequentially: {', '.join(task_labels)}")
		attempts = list(planner_result.get("attempts") or ())
		selected_attempt = dict(planner_result.get("selected_attempt") or {})
		selected_output_dir = Path(str(selected_attempt.get("output_dir") or self.context.output_dir)).resolve()
		self.context._merge_primary_planner_output_dir(selected_output_dir)

		plan_solve_data = copy.deepcopy(selected_attempt.get("plan_solve_data") or {})
		plan_verification_data = copy.deepcopy(selected_attempt.get("plan_verification_data") or {})
		plan_solve_data = self.context._rewrite_artifact_root_paths(
			plan_solve_data,
			selected_output_dir,
			Path(self.context.output_dir).resolve(),
		)
		plan_verification_data = self.context._rewrite_artifact_root_paths(
			plan_verification_data,
			selected_output_dir,
			Path(self.context.output_dir).resolve(),
		)

		attempt_summaries = [
			_planner_attempt_summary(dict(attempt))
			for attempt in attempts
		]

		plan_solve_summary = dict((plan_solve_data.get("summary") or {}))
		plan_solve_artifacts = dict((plan_solve_data.get("artifacts") or {}))
		plan_verification_summary = dict((plan_verification_data.get("summary") or {}))
		plan_verification_artifacts = dict((plan_verification_data.get("artifacts") or {}))
		plan_solve_summary.update(
			{
				"primary_planner_strategy": "lifted_panda_sat_primary",
				"evaluation_mode": mode,
				"requested_planner_id": normalized_planner_id,
				"planner_attempts": attempt_summaries,
				"selected_solver_id": str(
					plan_solve_summary.get("solver_id")
					or selected_attempt.get("planner_id")
					or "unknown"
				),
				"selected_planner_id": str(selected_attempt.get("planner_id") or "unknown"),
				"selected_representation_id": str(selected_attempt.get("representation_id") or "unknown"),
				"representation_build_seconds": planner_result.get("representation_build_seconds"),
				"planner_wallclock_seconds": planner_result.get("planner_wallclock_seconds"),
				"official_resource_profile": dict(selected_attempt.get("resource_profile") or {}),
			}
		)
		plan_solve_artifacts.update(
			{
				"primary_planner_strategy": "lifted_panda_sat_primary",
				"evaluation_mode": mode,
				"requested_planner_id": normalized_planner_id,
				"planner_attempts": attempt_summaries,
				"selected_solver_id": str(plan_solve_summary.get("selected_solver_id") or ""),
				"selected_planner_id": str(selected_attempt.get("planner_id") or "unknown"),
				"selected_representation_id": str(selected_attempt.get("representation_id") or "unknown"),
				"selected_bucket": selected_attempt.get("selected_bucket"),
				"official_resource_profile": dict(selected_attempt.get("resource_profile") or {}),
			}
		)
		plan_verification_summary.update(
			{
				"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
				"evaluation_mode": mode,
				"requested_planner_id": normalized_planner_id,
				"selected_solver_id": str(plan_solve_summary.get("selected_solver_id") or ""),
				"selected_planner_id": str(selected_attempt.get("planner_id") or "unknown"),
				"selected_representation_id": str(selected_attempt.get("representation_id") or "unknown"),
				"representation_build_seconds": planner_result.get("representation_build_seconds"),
				"planner_wallclock_seconds": planner_result.get("planner_wallclock_seconds"),
				"official_resource_profile": dict(selected_attempt.get("resource_profile") or {}),
			}
		)
		plan_verification_artifacts.update(
			{
				"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
				"evaluation_mode": mode,
				"requested_planner_id": normalized_planner_id,
				"selected_solver_id": str(plan_solve_summary.get("selected_solver_id") or ""),
				"selected_planner_id": str(selected_attempt.get("planner_id") or "unknown"),
				"selected_representation_id": str(selected_attempt.get("representation_id") or "unknown"),
				"planner_attempts": attempt_summaries,
				"official_resource_profile": dict(selected_attempt.get("resource_profile") or {}),
			}
		)
		plan_solve_data = {
			"summary": plan_solve_summary,
			"artifacts": plan_solve_artifacts,
		}
		plan_verification_data = {
			"summary": plan_verification_summary,
			"artifacts": plan_verification_artifacts,
		}

		selected_planner_id = str(selected_attempt.get("planner_id") or "unknown")
		selected_representation_id = str(selected_attempt.get("representation_id") or "unknown")
		plan_solve_status_label = (
			"Success" if plan_solve_summary.get("status") == "success" else "Failed"
		)
		self.context.logger.log_plan_solve(
			plan_solve_artifacts,
			plan_solve_status_label,
			error=(
				None
				if plan_solve_status_label == "Success"
				else "official primary-planner evaluation failed"
			),
			metadata=plan_solve_summary,
		)
		self.context.logger.record_step_timing(
			"plan_solve",
			float(selected_attempt.get("plan_solve_seconds") or 0.0)
			+ float(planner_result.get("representation_build_seconds") or 0.0),
			metadata={
				"selected_planner_id": selected_planner_id,
				"selected_representation_id": selected_representation_id,
				"representation_build_seconds": round(
					float(planner_result.get("representation_build_seconds") or 0.0),
					6,
				),
				"planner_wallclock_seconds": round(
					float(planner_result.get("planner_wallclock_seconds") or 0.0),
					6,
				),
				"planner_attempt_count": len(attempt_summaries),
			},
		)
		if plan_solve_summary.get("status") == "success":
			print(
				f"✓ Planner returned: {selected_planner_id} "
				f"on {selected_representation_id}"
			)
		else:
			print(
				"✗ Plan solve failed across all primary-planner tasks "
				f"(selected failure: {selected_planner_id} on {selected_representation_id})"
			)

		print("\n[OFFICIAL VERIFICATION]")
		print("-" * 80)
		plan_verification_status_label = (
			"Success" if plan_verification_summary.get("status") == "success" else "Failed"
		)
		self.context.logger.log_official_verification(
			plan_verification_artifacts,
			plan_verification_status_label,
			error=(
				None
				if plan_verification_status_label == "Success"
				else "all official planning tasks failed verification"
			),
			metadata=plan_verification_summary,
		)
		self.context.logger.record_step_timing(
			"plan_verification",
			float(selected_attempt.get("plan_verification_seconds") or 0.0),
			metadata={
				"selected_planner_id": selected_planner_id,
				"selected_representation_id": selected_representation_id,
				"planner_wallclock_seconds": round(
					float(planner_result.get("planner_wallclock_seconds") or 0.0),
					6,
				),
				"planner_attempt_count": len(attempt_summaries),
			},
		)
		if plan_verification_summary.get("status") == "success":
			print("✓ Official IPC verification complete")
			print(f"  Selected planner: {selected_planner_id}")
			print(f"  Selected representation: {selected_representation_id}")
			print(
				f"  Verification result: "
				f"{plan_verification_artifacts.get('verification_result')}"
			)
		else:
			print("✗ Official verification failed: all primary-planner tasks failed")
			print(f"  Selected failure planner: {selected_planner_id}")
			print(f"  Selected failure representation: {selected_representation_id}")
		return {
			"plan_solve": plan_solve_data,
			"plan_verification": plan_verification_data,
		}

	def execute_primary_planner(
		self,
		method_library=None,
	) -> Dict[str, Any]:
		return self.execute_problem_root_evaluation(
			method_library=method_library,
			evaluation_mode=SINGLE_PLANNER_MODE,
			planner_id=PRIMARY_HTN_PLANNER_ID,
		)

	@staticmethod
	def official_problem_root_failure_rank(attempt: Dict[str, Any]) -> tuple[int, float]:
		if attempt.get("success"):
			return (0, float(attempt.get("total_seconds") or 0.0))
		bucket = str(attempt.get("selected_bucket") or "unknown_failure")
		priority = {
			"hierarchical_plan_verified": 0,
			"primitive_plan_valid_but_hierarchical_rejected": 1,
			"primitive_plan_invalid": 2,
			"no_plan_from_solver": 3,
			"worker_exception": 4,
		}.get(bucket, 5)
		return (priority, float(attempt.get("total_seconds") or 0.0))

	def select_planner_attempt(
		self,
		attempts: Sequence[Dict[str, Any]],
	) -> Dict[str, Any]:
		successful = [
			attempt
			for attempt in attempts
			if bool(attempt.get("success"))
		]
		if successful:
			return min(
				successful,
				key=lambda attempt: float(attempt.get("total_seconds") or 0.0),
			)
		if not attempts:
			raise ValueError("No official problem-root primary-planner attempts were provided.")
		return min(attempts, key=self.official_problem_root_failure_rank)

	@staticmethod
	def close_planner_queue(result_queue: Any) -> None:
		close_fn = getattr(result_queue, "close", None)
		if callable(close_fn):
			close_fn()
		join_thread_fn = getattr(result_queue, "join_thread", None)
		if callable(join_thread_fn):
			join_thread_fn()

	@staticmethod
	def terminate_planner_process(process: multiprocessing.Process) -> None:
		if not process.is_alive():
			return
		try:
			if os.name == "posix":
				process_group_id = os.getpgid(process.pid)
				os.killpg(process_group_id, signal.SIGTERM)
			else:
				process.terminate()
		except Exception:
			process.terminate()
		process.join(timeout=1.0)
		if process.is_alive():
			try:
				if os.name == "posix":
					process_group_id = os.getpgid(process.pid)
					os.killpg(process_group_id, signal.SIGKILL)
				else:
					process.kill()
			except Exception:
				process.kill()
