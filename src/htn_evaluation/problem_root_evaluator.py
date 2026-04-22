"""
Problem-root Hierarchical Task Network evaluation orchestration.

This module owns the planner-based benchmark evaluation tracks for official and
generated domains. It is a reference and diagnostic layer, not the deployed
online runtime.
"""

from __future__ import annotations

import copy
import multiprocessing
import os
import queue
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from planning.backends import PlanningBackendTask, default_official_backends, expand_backend_tasks_for_representations
from planning.linearization import LiftedLinearPlanner
from planning.official_benchmark import OFFICIAL_BACKEND_SELECTION_RULE
from planning.representations import PlanningRepresentation, RepresentationBuildResult
from .problem_root_runtime import official_problem_root_planning_task_worker
from .result_tables import (
	PLANNER_OR_RACE_MODE,
	validate_evaluation_mode,
	validate_planner_id,
)


BACKEND_RESULT_MESSAGE = "backend_attempt"


class HTNProblemRootEvaluator:
	"""Run Hierarchical Task Network planner-based benchmark evaluation."""

	def __init__(self, pipeline_context: Any) -> None:
		self.context = pipeline_context

	def planning_tasks(
		self,
		timeout_seconds: Optional[float] = None,
		*,
		planner_id: Optional[str] = None,
	) -> tuple[PlanningBackendTask, ...]:
		build_result = self.context._build_problem_representations(
			timeout_seconds=timeout_seconds,
		)
		selected_backends = tuple(
			backend
			for backend in default_official_backends(workspace=str(self.context.output_dir))
			if planner_id is None or backend.backend_name == planner_id
		)
		if planner_id is not None and not selected_backends:
			raise ValueError(f"Unknown official planning backend '{planner_id}'.")
		tasks = expand_backend_tasks_for_representations(
			representations=build_result.representations,
			backends=selected_backends,
		)
		if tasks:
			return tasks
		if not self._selected_backends_require_linearized_input(selected_backends):
			return tasks
		return expand_backend_tasks_for_representations(
			representations=(
				*self._representations_without_duplicate_linearized(build_result),
				self._build_linearized_representation_for_backend(
					build_result=build_result,
					timeout_seconds=timeout_seconds,
				),
			),
			backends=selected_backends,
		)

	@staticmethod
	def _selected_backends_require_linearized_input(
		selected_backends: Sequence[Any],
	) -> bool:
		return any(
			"linearized" in set(getattr(backend, "supported_sources", frozenset()))
			for backend in selected_backends
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

	def _build_linearized_representation_for_backend(
		self,
		*,
		build_result: RepresentationBuildResult,
		timeout_seconds: Optional[float],
	) -> PlanningRepresentation:
		if self.context.output_dir is None:
			raise ValueError("Linearized backend representation requires an output directory.")
		workspace = Path(self.context.output_dir).resolve() / "representations"
		linearizer = LiftedLinearPlanner(workspace=workspace)
		artifacts = linearizer.linearize_hddl_files(
			domain_file=self.context.domain_file,
			problem_file=self.context.problem_file,
			transition_name="official_problem_root_backend_linearization",
			timeout_seconds=timeout_seconds,
		)
		metadata = dict(artifacts)
		metadata.update(
			{
				"forced_for_backend_capability": True,
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

	def run_backend_evaluation(
		self,
		*,
		evaluation_mode: str = PLANNER_OR_RACE_MODE,
		planner_id: Optional[str] = None,
	) -> Dict[str, Any]:
		if self.context.output_dir is None:
			raise ValueError("Official problem-root backend race requires an output directory.")

		mode = validate_evaluation_mode(evaluation_mode)
		normalized_planner_id = validate_planner_id(
			planner_id,
			evaluation_mode=mode,
		)
		planning_timeout_seconds = self.context._official_problem_root_planning_timeout_seconds()
		representation_build_start = time.perf_counter()
		planning_tasks = self.planning_tasks(
			timeout_seconds=planning_timeout_seconds,
			planner_id=normalized_planner_id,
		)
		representation_build_seconds = time.perf_counter() - representation_build_start
		backend_root = Path(self.context.output_dir) / "backend_race"
		backend_root.mkdir(parents=True, exist_ok=True)
		context = multiprocessing.get_context("spawn")
		attempts: List[Dict[str, Any]] = []
		race_start = time.perf_counter()
		selected_attempt: Optional[Dict[str, Any]] = None
		stop_on_success = mode == PLANNER_OR_RACE_MODE

		def incomplete_attempt(
			planning_task: PlanningBackendTask,
			*,
			failure_reason: str,
			total_seconds: float,
		) -> Dict[str, Any]:
			return {
				"message_type": BACKEND_RESULT_MESSAGE,
				"backend_name": planning_task.backend_name,
				"task_id": planning_task.task_id,
				"representation_id": planning_task.representation.representation_id,
				"output_dir": str((backend_root / planning_task.task_id).resolve()),
				"plan_solve_data": {
					"summary": {
						"backend": planning_task.backend_name,
						"status": "failed",
						"planning_mode": "official_problem_root",
						"failure_bucket": "no_plan_from_solver",
						"representation_id": planning_task.representation.representation_id,
					},
					"artifacts": {
						"backend": planning_task.backend_name,
						"status": "failed",
						"planning_mode": "official_problem_root",
						"failure_bucket": "no_plan_from_solver",
						"planning_representation": planning_task.representation.to_dict(),
						"failure_reason": failure_reason,
					},
				},
				"plan_verification_data": {
					"summary": {
						"backend": "pandaPIparser",
						"status": "failed",
						"selection_rule": "first_hierarchical_verification_success",
						"failure_bucket": "no_plan_from_solver",
					},
					"artifacts": {
						"backend": planning_task.backend_name,
						"status": "failed",
						"selection_rule": "first_hierarchical_verification_success",
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

		def run_single_task(planning_task: PlanningBackendTask) -> Dict[str, Any]:
			raw_remaining_timeout = planning_timeout_seconds - (time.perf_counter() - race_start)
			if raw_remaining_timeout <= 0.0:
				return incomplete_attempt(
					planning_task,
					failure_reason="planner_timeout_budget_exhausted_before_backend_launch",
					total_seconds=planning_timeout_seconds,
				)
			remaining_timeout = max(raw_remaining_timeout, 1.0)
			attempt_output_dir = backend_root / planning_task.task_id
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
						self.terminate_backend_process(process)
						process.join(timeout=1.0)
			finally:
				self.close_backend_race_queue(result_queue)
			if attempt is not None:
				return attempt
			return incomplete_attempt(
				planning_task,
				failure_reason="backend_attempt_incomplete_before_deadline",
				total_seconds=planning_timeout_seconds,
			)

		for planning_task in planning_tasks:
			attempt = run_single_task(planning_task)
			attempts.append(attempt)
			if stop_on_success and bool(attempt.get("success")):
				selected_attempt = dict(attempt)
				break

		if selected_attempt is None:
			selected_attempt = self.select_backend_attempt(attempts)

		return {
			"evaluation_mode": mode,
			"requested_planner_id": normalized_planner_id,
			"planning_tasks": [task.to_dict() for task in planning_tasks],
			"attempts": attempts,
			"selected_attempt": selected_attempt,
			"representation_build_seconds": representation_build_seconds,
			"race_wallclock_seconds": time.perf_counter() - race_start,
		}

	def run_backend_race(self) -> Dict[str, Any]:
		return self.run_backend_evaluation(
			evaluation_mode=PLANNER_OR_RACE_MODE,
			planner_id=None,
		)

	def execute_problem_root_evaluation(
		self,
		method_library=None,
		*,
		evaluation_mode: str = PLANNER_OR_RACE_MODE,
		planner_id: Optional[str] = None,
	) -> Dict[str, Any]:
		mode = validate_evaluation_mode(evaluation_mode)
		normalized_planner_id = validate_planner_id(
			planner_id,
			evaluation_mode=mode,
		)
		print("\n[PLAN SOLVE]")
		print("-" * 80)
		race_result = self.run_backend_evaluation(
			evaluation_mode=mode,
			planner_id=normalized_planner_id,
		)
		planning_tasks = list(race_result.get("planning_tasks") or ())
		task_labels = [
			(
				f"{task.get('backend_name')}@"
				f"{((task.get('representation') or {}).get('representation_id') or 'unknown')}"
			)
			for task in planning_tasks
		]
		print(f"• Running official planning tasks sequentially: {', '.join(task_labels)}")
		attempts = list(race_result.get("attempts") or ())
		selected_attempt = dict(race_result.get("selected_attempt") or {})
		selected_output_dir = Path(str(selected_attempt.get("output_dir") or self.context.output_dir)).resolve()
		self.context._merge_official_backend_output_dir(selected_output_dir)

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
			{
				"backend_name": str(attempt.get("backend_name") or "unknown"),
				"task_id": str(attempt.get("task_id") or "unknown"),
				"representation_id": str(attempt.get("representation_id") or "unknown"),
				"success": bool(attempt.get("success")),
				"selected_bucket": attempt.get("selected_bucket"),
				"resource_profile": dict(attempt.get("resource_profile") or {}),
				"plan_solve_status": ((attempt.get("plan_solve_data") or {}).get("summary") or {}).get("status"),
				"plan_verification_status": ((attempt.get("plan_verification_data") or {}).get("summary") or {}).get("status"),
				"total_seconds": attempt.get("total_seconds"),
				"stdout": attempt.get("stdout"),
				"stderr": attempt.get("stderr"),
			}
			for attempt in attempts
		]

		plan_solve_summary = dict((plan_solve_data.get("summary") or {}))
		plan_solve_artifacts = dict((plan_solve_data.get("artifacts") or {}))
		plan_verification_summary = dict((plan_verification_data.get("summary") or {}))
		plan_verification_artifacts = dict((plan_verification_data.get("artifacts") or {}))
		plan_solve_summary.update(
			{
				"solver_race_strategy": (
					OFFICIAL_BACKEND_SELECTION_RULE
					if mode == PLANNER_OR_RACE_MODE
					else "single_planner_best_attempt"
				),
				"evaluation_mode": mode,
				"requested_planner_id": normalized_planner_id,
				"solver_attempts": attempt_summaries,
				"selected_solver_id": str(plan_solve_summary.get("solver_id") or selected_attempt.get("backend_name") or "unknown"),
				"selected_backend_name": str(selected_attempt.get("backend_name") or "unknown"),
				"selected_representation_id": str(selected_attempt.get("representation_id") or "unknown"),
				"representation_build_seconds": race_result.get("representation_build_seconds"),
				"race_wallclock_seconds": race_result.get("race_wallclock_seconds"),
				"official_resource_profile": dict(selected_attempt.get("resource_profile") or {}),
			}
		)
		plan_solve_artifacts.update(
			{
				"solver_race_strategy": (
					OFFICIAL_BACKEND_SELECTION_RULE
					if mode == PLANNER_OR_RACE_MODE
					else "single_planner_best_attempt"
				),
				"evaluation_mode": mode,
				"requested_planner_id": normalized_planner_id,
				"solver_attempts": attempt_summaries,
				"selected_solver_id": str(plan_solve_summary.get("selected_solver_id") or ""),
				"selected_backend_name": str(selected_attempt.get("backend_name") or "unknown"),
				"selected_representation_id": str(selected_attempt.get("representation_id") or "unknown"),
				"selected_bucket": selected_attempt.get("selected_bucket"),
				"official_resource_profile": dict(selected_attempt.get("resource_profile") or {}),
			}
		)
		plan_verification_summary.update(
			{
				"selection_rule": (
					OFFICIAL_BACKEND_SELECTION_RULE
					if mode == PLANNER_OR_RACE_MODE
					else "single_planner_best_attempt"
				),
				"evaluation_mode": mode,
				"requested_planner_id": normalized_planner_id,
				"selected_solver_id": str(plan_solve_summary.get("selected_solver_id") or ""),
				"selected_backend_name": str(selected_attempt.get("backend_name") or "unknown"),
				"selected_representation_id": str(selected_attempt.get("representation_id") or "unknown"),
				"representation_build_seconds": race_result.get("representation_build_seconds"),
				"race_wallclock_seconds": race_result.get("race_wallclock_seconds"),
				"official_resource_profile": dict(selected_attempt.get("resource_profile") or {}),
			}
		)
		plan_verification_artifacts.update(
			{
				"selection_rule": (
					OFFICIAL_BACKEND_SELECTION_RULE
					if mode == PLANNER_OR_RACE_MODE
					else "single_planner_best_attempt"
				),
				"evaluation_mode": mode,
				"requested_planner_id": normalized_planner_id,
				"selected_solver_id": str(plan_solve_summary.get("selected_solver_id") or ""),
				"selected_backend_name": str(selected_attempt.get("backend_name") or "unknown"),
				"selected_representation_id": str(selected_attempt.get("representation_id") or "unknown"),
				"solver_attempts": attempt_summaries,
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

		selected_backend_name = str(selected_attempt.get("backend_name") or "unknown")
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
				else "official backend evaluation failed"
			),
			metadata=plan_solve_summary,
		)
		self.context.logger.record_step_timing(
			"plan_solve",
			float(selected_attempt.get("plan_solve_seconds") or 0.0)
			+ float(race_result.get("representation_build_seconds") or 0.0),
			metadata={
				"selected_backend_name": selected_backend_name,
				"selected_representation_id": selected_representation_id,
				"representation_build_seconds": round(
					float(race_result.get("representation_build_seconds") or 0.0),
					6,
				),
				"race_wallclock_seconds": round(
					float(race_result.get("race_wallclock_seconds") or 0.0),
					6,
				),
				"backend_attempt_count": len(attempt_summaries),
			},
		)
		if plan_solve_summary.get("status") == "success":
			print(
				f"✓ Planner returned via backend: {selected_backend_name} "
				f"on {selected_representation_id}"
			)
		else:
			print(
				"✗ Plan solve failed across all planning tasks "
				f"(selected failure: {selected_backend_name} on {selected_representation_id})"
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
				"selected_backend_name": selected_backend_name,
				"selected_representation_id": selected_representation_id,
				"race_wallclock_seconds": round(
					float(race_result.get("race_wallclock_seconds") or 0.0),
					6,
				),
				"backend_attempt_count": len(attempt_summaries),
			},
		)
		if plan_verification_summary.get("status") == "success":
			print("✓ Official IPC verification complete")
			print(f"  Selected backend: {selected_backend_name}")
			print(f"  Selected representation: {selected_representation_id}")
			print(
				f"  Verification result: "
				f"{plan_verification_artifacts.get('verification_result')}"
			)
		else:
			print("✗ Official verification failed: all planning tasks failed")
			print(f"  Selected failure backend: {selected_backend_name}")
			print(f"  Selected failure representation: {selected_representation_id}")
		return {
			"plan_solve": plan_solve_data,
			"plan_verification": plan_verification_data,
		}

	def execute_parallel_solver_race(
		self,
		method_library=None,
	) -> Dict[str, Any]:
		return self.execute_problem_root_evaluation(
			method_library=method_library,
			evaluation_mode=PLANNER_OR_RACE_MODE,
			planner_id=None,
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

	def select_backend_attempt(
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
			raise ValueError("No official problem-root backend attempts were provided.")
		return min(attempts, key=self.official_problem_root_failure_rank)

	@staticmethod
	def close_backend_race_queue(result_queue: Any) -> None:
		close_fn = getattr(result_queue, "close", None)
		if callable(close_fn):
			close_fn()
		join_thread_fn = getattr(result_queue, "join_thread", None)
		if callable(join_thread_fn):
			join_thread_fn()

	@staticmethod
	def terminate_backend_process(process: multiprocessing.Process) -> None:
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
