"""
Problem-root HTN evaluation runtime helpers.

This module contains the worker-compatible planning and official-verification
logic for the HTN evaluation track. It depends on planning/ and verification/
only, not on the legacy compatibility façade.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import signal
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

from planning.backends import PlanningBackendTask, backend_by_name
from planning.panda_portfolio import PANDAPlanner
from verification.official_plan_verifier import IPCPlanVerifier

from .context import HTNEvaluationContext


class _NullExecutionLogger:
	"""Minimal logger for isolated worker attempts."""

	current_record = None

	def record_step_timing(self, *args, **kwargs) -> None:
		return None

	def log_plan_solve(self, *args, **kwargs) -> None:
		return None

	def log_official_verification(self, *args, **kwargs) -> None:
		return None


def _sanitize_identifier(value: str) -> str:
	text = str(value or "").strip().lower()
	text = re.sub(r"[^a-z0-9_]+", "_", text)
	text = re.sub(r"_+", "_", text)
	return text.strip("_") or "item"


def solve_problem_root_backend_task(
	context: HTNEvaluationContext,
	*,
	planning_task: PlanningBackendTask,
	timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
	"""Solve one compiled official representation with one backend."""
	print("\n[PLAN SOLVE]")
	print("-" * 80)
	stage_start = time.perf_counter()
	effective_timeout_seconds = context._official_problem_root_planning_timeout_seconds(
		timeout_seconds,
	)
	if context.output_dir is None:
		raise ValueError("Problem-root planning requires an output directory.")
	backend = backend_by_name(
		planning_task.backend_name,
		workspace=str(context.output_dir),
	)

	try:
		if not context.problem.htn_tasks:
			raise ValueError("Problem file contains no root HTN tasks.")
		if not backend.toolchain_available():
			raise ValueError(
				f"Planning backend '{planning_task.backend_name}' is unavailable on PATH.",
			)

		task_network = tuple(
			(str(task.task_name), tuple(str(arg) for arg in (task.args or ())))
			for task in (context.problem.htn_tasks or ())
		)
		ordering_edges = context._problem_root_task_network_ordering_edges()
		task_network_ordered = context._problem_root_task_network_is_totally_ordered()
		primary_task_name, primary_task_args = task_network[0]
		plan = backend.solve(
			domain=context.domain,
			task_name=str(primary_task_name),
			representation=planning_task.representation,
			task_args=tuple(primary_task_args),
			timeout_seconds=effective_timeout_seconds,
		)
		action_path = [
			(
				f"{step.action_name}({', '.join(step.args)})"
				if step.args
				else str(step.action_name)
			)
			for step in plan.steps
		]
		method_trace = PANDAPlanner(workspace=str(context.output_dir)).extract_method_trace(
			plan.actual_plan,
		)
		timing_profile = dict(plan.timing_profile or {})
		work_dir = Path(str(plan.work_dir)).resolve() if plan.work_dir else None

		action_path_file = context.output_dir / "plan_solve_action_path.txt"
		action_path_file.write_text("".join(f"{step}\n" for step in action_path))
		method_trace_file = context.output_dir / "plan_solve_method_trace.json"
		method_trace_file.write_text(json.dumps(method_trace, indent=2))
		combined_plan_file = context.output_dir / "plan_solve_hierarchical_plan.txt"
		combined_plan_file.write_text(str(plan.actual_plan or ""))

		artifacts = {
			"backend": planning_task.backend_name,
			"status": "success",
			"planning_mode": "official_problem_root",
			"engine_mode": plan.engine_mode,
			"solver_id": plan.solver_id,
			"planning_representation": planning_task.representation.to_dict(),
			"task_network": [
				{"task_name": task_name, "args": list(task_args)}
				for task_name, task_args in task_network
			],
			"task_network_ordered": task_network_ordered,
			"ordering_edges": [
				{"before": before, "after": after}
				for before, after in ordering_edges
			],
			"step_count": len(action_path),
			"action_path": action_path,
			"method_trace": method_trace,
			"guided_hierarchical_plan_text": plan.actual_plan or "",
			"guided_hierarchical_plan_source": backend.plan_source_label,
			"timing_profile": timing_profile,
			"solver_candidates": list(plan.solver_candidates or ()),
			"artifacts": {
				"domain_hddl": planning_task.representation.domain_file,
				"problem_hddl": planning_task.representation.problem_file,
				"parsed_problem": str(work_dir / "problem.psas") if work_dir else None,
				"grounded_problem": str(work_dir / "problem.psas.grounded") if work_dir else None,
				"raw_plan": str(work_dir / "plan.original") if work_dir else None,
				"actual_plan": str(combined_plan_file),
				"action_path": str(action_path_file),
				"method_trace": str(method_trace_file),
			},
		}
		summary = {
			"backend": planning_task.backend_name,
			"status": "success",
			"planning_mode": "official_problem_root",
			"engine_mode": plan.engine_mode,
			"solver_id": plan.solver_id,
			"representation_id": planning_task.representation.representation_id,
			"representation_source": planning_task.representation.representation_source,
			"representation_ordering_kind": planning_task.representation.ordering_kind,
			"task_count": len(task_network),
			"precedence_edge_count": len(ordering_edges),
			"step_count": len(action_path),
			"solver_candidate_count": len(plan.solver_candidates or ()),
		}
		context.logger.log_plan_solve(
			artifacts,
			"Success",
			metadata=summary,
		)
		context._record_step_timing(
			"plan_solve",
			stage_start,
			breakdown=context._timing_breakdown_without_total(timing_profile),
			metadata={
				"task_count": len(task_network),
				"step_count": len(action_path),
				"planning_mode": "official_problem_root",
				"solver_id": plan.solver_id,
				"backend": planning_task.backend_name,
				"representation_id": planning_task.representation.representation_id,
			},
		)
		print(f"✓ Planner returned {len(action_path)} primitive steps")
		print(f"  Problem root task count: {len(task_network)}")
		return {"summary": summary, "artifacts": artifacts}
	except Exception as exc:
		failure_metadata = dict(getattr(exc, "metadata", {}) or {})
		failure_artifacts = {
			"backend": planning_task.backend_name,
			"status": "failed",
			"planning_mode": "official_problem_root",
			"planning_representation": planning_task.representation.to_dict(),
			"task_network": [
				{
					"task_name": str(task.task_name),
					"args": list(task.args or ()),
				}
				for task in (context.problem.htn_tasks or ())
			],
			"task_network_ordered": context._problem_root_task_network_is_totally_ordered(),
			"ordering_edges": [
				{"before": before, "after": after}
				for before, after in context._problem_root_task_network_ordering_edges()
			],
			"step_count": 0,
			"failure_bucket": "no_plan_from_solver",
			"solver_candidates": list(failure_metadata.get("engine_attempts") or ()),
			"failure_metadata": failure_metadata,
		}
		summary = {
			"backend": planning_task.backend_name,
			"status": "failed",
			"planning_mode": "official_problem_root",
			"engine_mode": failure_metadata.get("engine_mode"),
			"failure_bucket": "no_plan_from_solver",
			"representation_id": planning_task.representation.representation_id,
			"representation_source": planning_task.representation.representation_source,
			"representation_ordering_kind": planning_task.representation.ordering_kind,
			"solver_candidate_count": len(failure_metadata.get("engine_attempts") or ()),
		}
		context.logger.log_plan_solve(
			failure_artifacts,
			"Failed",
			error=str(exc),
			metadata=summary,
		)
		context._record_step_timing("plan_solve", stage_start)
		print(f"✗ Plan solve failed: {exc}")
		return {"summary": summary, "artifacts": failure_artifacts}


def verify_problem_root_solver_race(
	context: HTNEvaluationContext,
	*,
	verifier: IPCPlanVerifier,
	plan_solve_data: Dict[str, Any],
	plan_solve_artifacts: Dict[str, Any],
	stage_start: float,
) -> Dict[str, Any]:
	verification_domain_file = Path(context.domain_file).resolve()
	plan_solve_summary = dict(plan_solve_data.get("summary") or {})
	solver_candidates = list(plan_solve_artifacts.get("solver_candidates") or ())

	if plan_solve_summary.get("status") != "success" or not solver_candidates:
		summary = {
			"backend": "pandaPIparser",
			"status": "failed",
			"selection_rule": "first_hierarchical_verification_success",
			"failure_bucket": "no_plan_from_solver",
			"solver_candidate_count": len(solver_candidates),
			"verification_domain_file": str(verification_domain_file),
			"verification_problem_file": str(Path(context.problem_file).resolve()),
		}
		artifacts = {
			"tool_available": verifier.tool_available(),
			"plan_kind": None,
			"verification_result": False,
			"primitive_plan_executable": None,
			"reached_goal_state": None,
			"selected_solver_id": None,
			"selected_bucket": "no_plan_from_solver",
			"selection_rule": "first_hierarchical_verification_success",
			"solver_candidates": solver_candidates,
		}
		context.logger.log_official_verification(
			artifacts,
			"Failed",
			error="Plan solve produced no executable solver candidate for the official baseline",
			metadata=summary,
		)
		context._record_step_timing(
			"plan_verification",
			stage_start,
			metadata={"failure_bucket": "no_plan_from_solver"},
		)
		print("✗ Official verification failed: no executable solver candidate was available")
		return {"summary": summary, "artifacts": artifacts}

	selected_success: Optional[Dict[str, Any]] = None
	selected_primitive_valid: Optional[Dict[str, Any]] = None
	verified_candidates: List[Dict[str, Any]] = []

	for candidate in solver_candidates:
		solver_id = str(candidate.get("solver_id") or candidate.get("mode") or "unknown").strip()
		solver_mode = str(candidate.get("mode") or "").strip() or None
		candidate_status = str(candidate.get("status") or "").strip() or "unknown"
		if candidate_status != "success":
			verified_candidates.append(
				{
					"solver_id": solver_id,
					"mode": solver_mode,
					"status": candidate_status,
					"bucket": "no_plan_from_solver",
					"command": list(candidate.get("command") or ()),
					"reason": candidate.get("reason"),
					"step_count": candidate.get("step_count"),
				},
			)
			continue

		if context.output_dir is None:
			raise ValueError("Official verification requires an output directory.")
		candidate_output_dir = context.output_dir / "solver_portfolio" / _sanitize_identifier(solver_id)
		candidate_output_dir.mkdir(parents=True, exist_ok=True)
		action_path = list(candidate.get("action_path") or ())
		primitive_result = verifier.verify_primitive_plan(
			domain_file=verification_domain_file,
			problem_file=context.problem_file,
			action_path=action_path,
			output_dir=candidate_output_dir,
			plan_filename="primitive_plan.txt",
			output_filename="primitive_verifier.txt",
			json_filename="primitive_verification.json",
		)

		candidate_actual_plan_text = ""
		candidate_actual_plan_path = candidate.get("actual_plan_path")
		if candidate_actual_plan_path:
			actual_plan_path = Path(str(candidate_actual_plan_path))
			if actual_plan_path.exists():
				candidate_actual_plan_text = actual_plan_path.read_text()
		hierarchical_result = verifier.verify_plan_text(
			domain_file=verification_domain_file,
			problem_file=context.problem_file,
			plan_text=candidate_actual_plan_text,
			output_dir=candidate_output_dir,
			plan_kind="hierarchical",
			build_warning=None,
			plan_filename="hierarchical_plan.txt",
			output_filename="hierarchical_verifier.txt",
			json_filename="hierarchical_verification.json",
		)

		primitive_valid = (
			primitive_result.tool_available is True
			and primitive_result.primitive_plan_executable is True
			and primitive_result.verification_result is True
			and primitive_result.reached_goal_state is True
		)
		hierarchical_valid = (
			hierarchical_result.tool_available is True
			and hierarchical_result.plan_kind == "hierarchical"
			and hierarchical_result.verification_result is True
		)
		if hierarchical_valid:
			bucket = "hierarchical_plan_verified"
		elif primitive_valid:
			bucket = "primitive_plan_valid_but_hierarchical_rejected"
		else:
			bucket = "primitive_plan_invalid"

		record = {
			"solver_id": solver_id,
			"mode": solver_mode,
			"status": candidate_status,
			"bucket": bucket,
			"command": list(candidate.get("command") or ()),
			"step_count": candidate.get("step_count"),
			"action_path": action_path,
			"raw_plan_path": candidate.get("raw_plan_path"),
			"actual_plan_path": candidate_actual_plan_path,
			"verification_output_dir": str(candidate_output_dir),
			"primitive_verification": primitive_result.to_dict(),
			"hierarchical_verification": hierarchical_result.to_dict(),
		}
		verified_candidates.append(record)
		if bucket == "hierarchical_plan_verified" and selected_success is None:
			selected_success = record
		if bucket == "primitive_plan_valid_but_hierarchical_rejected" and selected_primitive_valid is None:
			selected_primitive_valid = record

	selected_record = (
		selected_success
		or selected_primitive_valid
		or (verified_candidates[0] if verified_candidates else None)
	)
	selected_bucket = (
		str(selected_record.get("bucket"))
		if isinstance(selected_record, dict)
		else "no_plan_from_solver"
	)
	selected_solver_id = (
		str(selected_record.get("solver_id"))
		if isinstance(selected_record, dict) and selected_record.get("solver_id") is not None
		else None
	)

	if selected_success is not None:
		selected_hierarchical = dict(selected_success.get("hierarchical_verification") or {})
		selected_primitive = dict(selected_success.get("primitive_verification") or {})
		composite_artifacts = {
			**selected_hierarchical,
			"primitive_plan_executable": selected_primitive.get("primitive_plan_executable"),
			"reached_goal_state": selected_primitive.get("reached_goal_state"),
		}
	elif selected_primitive_valid is not None:
		selected_hierarchical = dict(selected_primitive_valid.get("hierarchical_verification") or {})
		selected_primitive = dict(selected_primitive_valid.get("primitive_verification") or {})
		composite_artifacts = {
			**selected_hierarchical,
			"tool_available": selected_hierarchical.get(
				"tool_available",
				selected_primitive.get("tool_available"),
			),
			"primitive_plan_executable": selected_primitive.get("primitive_plan_executable"),
			"reached_goal_state": selected_primitive.get("reached_goal_state"),
			"verification_result": selected_hierarchical.get("verification_result"),
		}
	else:
		composite_artifacts = {
			"tool_available": verifier.tool_available(),
			"plan_kind": None,
			"verification_result": False,
			"primitive_plan_executable": None,
			"reached_goal_state": None,
		}

	selected_verification_dir = None
	selected_json_path = None
	if isinstance(selected_record, dict):
		selected_verification_dir = selected_record.get("verification_output_dir")
	if selected_verification_dir:
		selected_verification_dir = str(selected_verification_dir)
		selected_json_path = str(
			Path(selected_verification_dir)
			/ (
				"hierarchical_verification.json"
				if selected_record and selected_record.get("status") == "success"
				else "primitive_verification.json"
			),
		)
	official_plan_file = context.output_dir / "ipc_official_plan.txt"
	official_output_file = context.output_dir / "ipc_official_verifier.txt"
	official_json_file = context.output_dir / "ipc_official_verification.json"
	source_plan_file = composite_artifacts.get("plan_file")
	source_output_file = composite_artifacts.get("output_file")
	if source_plan_file and Path(str(source_plan_file)).exists():
		shutil.copyfile(Path(str(source_plan_file)), official_plan_file)
		composite_artifacts["plan_file"] = str(official_plan_file)
	if source_output_file and Path(str(source_output_file)).exists():
		shutil.copyfile(Path(str(source_output_file)), official_output_file)
		composite_artifacts["output_file"] = str(official_output_file)
	if selected_json_path and Path(str(selected_json_path)).exists():
		shutil.copyfile(Path(str(selected_json_path)), official_json_file)
		composite_artifacts["json_file"] = str(official_json_file)

	artifacts = {
		**composite_artifacts,
		"selected_solver_id": selected_solver_id,
		"selected_bucket": selected_bucket,
		"selection_rule": "first_hierarchical_verification_success",
		"solver_candidates": verified_candidates,
	}
	summary = {
		"backend": "pandaPIparser",
		"status": "success" if selected_success is not None else "failed",
		"selection_rule": "first_hierarchical_verification_success",
		"failure_bucket": None if selected_success is not None else selected_bucket,
		"selected_solver_id": selected_solver_id,
		"selected_bucket": selected_bucket,
		"solver_candidate_count": len(verified_candidates),
		"verified_success_count": sum(
			1
			for candidate in verified_candidates
			if candidate.get("bucket") == "hierarchical_plan_verified"
		),
		"primitive_valid_failure_count": sum(
			1
			for candidate in verified_candidates
			if candidate.get("bucket") == "primitive_plan_valid_but_hierarchical_rejected"
		),
		"verification_domain_file": str(verification_domain_file),
		"verification_problem_file": str(Path(context.problem_file).resolve()),
	}

	status_label = "Success" if selected_success is not None else "Failed"
	error = None
	if selected_success is None:
		error = (
			"Official IPC verifier found no hierarchically verified solver candidate; "
			f"best bucket={selected_bucket}"
		)
	context.logger.log_official_verification(
		artifacts,
		status_label,
		error=error,
		metadata=summary,
	)
	context._record_step_timing(
		"plan_verification",
		stage_start,
		metadata={
			"selected_solver_id": selected_solver_id,
			"selected_bucket": selected_bucket,
			"solver_candidate_count": len(verified_candidates),
		},
	)
	if selected_success is None:
		print(
			"✗ Official verification failed: no solver candidate passed hierarchical verification "
			f"(best bucket: {selected_bucket})"
		)
	else:
		print("✓ Official IPC verification complete")
		print(f"  Selected solver: {selected_solver_id}")
		print(f"  Verification result: {artifacts.get('verification_result')}")
	return {"summary": summary, "artifacts": artifacts}


def verify_plan_officially(
	context: HTNEvaluationContext,
	plan_solve_data: Dict[str, Any],
) -> Dict[str, Any]:
	"""Run official verification for one official problem-root plan-solve result."""
	print("\n[OFFICIAL VERIFICATION]")
	print("-" * 80)
	stage_start = time.perf_counter()

	verifier = IPCPlanVerifier()
	if not verifier.tool_available():
		error = "pandaPIparser is not available on PATH for official IPC verification"
		context.logger.log_official_verification(
			None,
			"Failed",
			error=error,
			metadata={
				"backend": "pandaPIparser",
				"status": "failed",
			},
		)
		context._record_step_timing("plan_verification", stage_start)
		print(f"✗ Official verification failed: {error}")
		return {
			"summary": {
				"backend": "pandaPIparser",
				"status": "failed",
			},
			"artifacts": {},
		}

	plan_solve_artifacts = dict(plan_solve_data.get("artifacts") or {})
	planning_mode = str(plan_solve_artifacts.get("planning_mode") or "")
	if planning_mode != "official_problem_root":
		raise ValueError(
			f"Unsupported HTN evaluation planning mode '{planning_mode or 'unknown'}'.",
		)
	return verify_problem_root_solver_race(
		context,
		verifier=verifier,
		plan_solve_data=plan_solve_data,
		plan_solve_artifacts=plan_solve_artifacts,
		stage_start=stage_start,
	)


def official_problem_root_planning_task_worker(
	result_queue,
	*,
	domain_file: str,
	problem_file: str,
	output_dir: str,
	task_payload: Dict[str, Any],
	planning_timeout_seconds: float,
) -> None:
	"""Spawn-safe worker for one backend attempt."""
	plan_solve_seconds = 0.0
	plan_verification_seconds = 0.0
	total_start = time.perf_counter()
	captured_stdout = io.StringIO()
	captured_stderr = io.StringIO()
	planning_task = PlanningBackendTask.from_dict(dict(task_payload))
	try:
		if hasattr(os, "setsid"):
			os.setsid()
		context = HTNEvaluationContext(
			domain_file=domain_file,
			problem_file=problem_file,
		)
		context.logger = _NullExecutionLogger()
		context.output_dir = Path(output_dir).resolve()
		context.output_dir.mkdir(parents=True, exist_ok=True)

		with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
			plan_solve_start = time.perf_counter()
			plan_solve_data = solve_problem_root_backend_task(
				context,
				planning_task=planning_task,
				timeout_seconds=planning_timeout_seconds,
			)
			plan_solve_seconds = time.perf_counter() - plan_solve_start

			plan_verification_start = time.perf_counter()
			plan_verification_data = verify_plan_officially(context, plan_solve_data)
			plan_verification_seconds = time.perf_counter() - plan_verification_start

		plan_solve_summary = dict((plan_solve_data or {}).get("summary") or {})
		plan_verification_summary = dict((plan_verification_data or {}).get("summary") or {})
		result_queue.put(
			{
				"message_type": "backend_attempt",
				"backend_name": planning_task.backend_name,
				"task_id": planning_task.task_id,
				"representation_id": planning_task.representation.representation_id,
				"output_dir": str(Path(output_dir).resolve()),
				"plan_solve_data": plan_solve_data,
				"plan_verification_data": plan_verification_data,
				"plan_solve_seconds": plan_solve_seconds,
				"plan_verification_seconds": plan_verification_seconds,
				"total_seconds": time.perf_counter() - total_start,
				"success": (
					plan_solve_summary.get("status") == "success"
					and plan_verification_summary.get("status") == "success"
				),
				"selected_bucket": (
					plan_verification_summary.get("selected_bucket")
					or plan_verification_summary.get("failure_bucket")
					or plan_solve_summary.get("failure_bucket")
					or "unknown_failure"
				),
				"stdout": captured_stdout.getvalue(),
				"stderr": captured_stderr.getvalue(),
			},
		)
	except Exception as exc:
		result_queue.put(
			{
				"message_type": "backend_attempt",
				"backend_name": planning_task.backend_name,
				"task_id": planning_task.task_id,
				"representation_id": planning_task.representation.representation_id,
				"output_dir": str(Path(output_dir).resolve()),
				"plan_solve_data": {
					"summary": {
						"backend": planning_task.backend_name,
						"status": "failed",
						"planning_mode": "official_problem_root",
						"failure_bucket": "worker_exception",
						"representation_id": planning_task.representation.representation_id,
					},
					"artifacts": {
						"backend": planning_task.backend_name,
						"status": "failed",
						"planning_mode": "official_problem_root",
						"failure_bucket": "worker_exception",
						"planning_representation": planning_task.representation.to_dict(),
					},
				},
				"plan_verification_data": {
					"summary": {
						"backend": "pandaPIparser",
						"status": "failed",
						"selection_rule": "first_hierarchical_verification_success",
						"failure_bucket": "worker_exception",
					},
					"artifacts": {
						"backend": planning_task.backend_name,
						"status": "failed",
						"selection_rule": "first_hierarchical_verification_success",
						"failure_bucket": "worker_exception",
						"error": str(exc),
					},
				},
				"plan_solve_seconds": plan_solve_seconds,
				"plan_verification_seconds": plan_verification_seconds,
				"total_seconds": time.perf_counter() - total_start,
				"success": False,
				"selected_bucket": "worker_exception",
				"stdout": captured_stdout.getvalue(),
				"stderr": captured_stderr.getvalue(),
			},
		)
