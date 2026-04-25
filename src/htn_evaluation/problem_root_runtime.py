"""
Problem-root HTN evaluation runtime helpers.

This module contains the worker-compatible planning and official-verification
logic for the HTN evaluation track. It depends on planning/ and verification/
only, not on broader pipeline orchestration.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import signal
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
	import resource
except ImportError:  # pragma: no cover - not expected on Unix CI, but keep runtime-safe.
	resource = None  # type: ignore[assignment]

from planning.primary_planner import PrimaryPlannerTask, primary_planner_by_id
from planning.panda_sat import PANDAPlanner
from planning.official_benchmark import (
	OFFICIAL_BENCHMARK_CPU_COUNT,
	OFFICIAL_BENCHMARK_MEMORY_LIMIT_MIB,
)
from verification.official_plan_verifier import IPCPlanVerifier

from .context import HTNEvaluationContext
from .result_tables import PRIMARY_PLANNER_SELECTION_RULE


class _NullExecutionLogger:
	"""Minimal logger for isolated worker attempts."""

	current_record = None

	def record_step_timing(self, *args, **kwargs) -> None:
		return None

	def log_plan_solve(self, *args, **kwargs) -> None:
		return None

	def log_official_verification(self, *args, **kwargs) -> None:
		return None


def _apply_official_resource_profile(
	*,
	memory_limit_mib: int,
	cpu_count: int,
) -> Dict[str, Any]:
	"""Apply IPC-style resource bounds to the current worker process."""
	profile: Dict[str, Any] = {
		"requested_memory_limit_mib": int(memory_limit_mib),
		"requested_cpu_count": int(cpu_count),
		"memory_limit_enforced": False,
		"cpu_affinity_enforced": False,
		"memory_limit_strategy": None,
		"cpu_affinity_strategy": None,
		"platform": sys.platform,
	}

	if not sys.platform.startswith("linux"):
		profile["memory_limit_strategy"] = "linux_only_not_applied"
		profile["cpu_affinity_strategy"] = "linux_only_not_applied"
		return profile

	memory_limit_bytes = max(int(memory_limit_mib), 1) * 1024 * 1024
	if resource is not None:
		for limit_name in ("RLIMIT_AS", "RLIMIT_DATA"):
			limit_key = getattr(resource, limit_name, None)
			if limit_key is None:
				continue
			try:
				resource.setrlimit(limit_key, (memory_limit_bytes, memory_limit_bytes))
				profile["memory_limit_enforced"] = True
				profile["memory_limit_strategy"] = limit_name
				break
			except (OSError, ValueError):
				continue
	else:
		profile["memory_limit_strategy"] = "resource_unavailable"

	if int(cpu_count) == 1:
		sched_setaffinity = getattr(os, "sched_setaffinity", None)
		if callable(sched_setaffinity):
			try:
				existing_affinity = os.sched_getaffinity(0)
				target_cpu = min(existing_affinity) if existing_affinity else 0
				sched_setaffinity(0, {target_cpu})
				profile["cpu_affinity_enforced"] = True
				profile["cpu_affinity_strategy"] = "sched_setaffinity"
			except (AttributeError, OSError, ValueError):
				profile["cpu_affinity_strategy"] = "sched_setaffinity_failed"
		else:
			profile["cpu_affinity_strategy"] = "sched_setaffinity_unavailable"
	else:
		profile["cpu_affinity_strategy"] = "not_requested"

	return profile

def _sanitize_identifier(value: str) -> str:
	text = str(value or "").strip().lower()
	text = re.sub(r"[^a-z0-9_]+", "_", text)
	text = re.sub(r"_+", "_", text)
	return text.strip("_") or "item"

HTN_EVIDENCE_TEXT_PREVIEW_CHARS = 4096
HTN_HEAVY_ARTIFACT_NAMES = frozenset(
	{
		"output.sas",
		"problem.psas",
		"problem.psas.grounded",
	}
)


def _text_preview(value: Any, *, limit: int = HTN_EVIDENCE_TEXT_PREVIEW_CHARS) -> str:
	text = str(value or "")
	if len(text) <= limit:
		return text
	head_limit = max(limit // 2, 1)
	tail_limit = max(limit - head_limit, 1)
	return (
		text[:head_limit]
		+ f"\n...[truncated {len(text) - limit} chars]...\n"
		+ text[-tail_limit:]
	)


def _path_size(path_value: Any) -> Optional[int]:
	if not path_value:
		return None
	try:
		path = Path(str(path_value))
		if path.exists():
			return int(path.stat().st_size)
	except OSError:
		return None
	return None


def _rewrite_text_file_as_preview(path_value: Any, text: str) -> Optional[int]:
	if not path_value:
		return None
	path = Path(str(path_value))
	if not path.exists():
		return None
	original_size = int(path.stat().st_size)
	preview = _text_preview(text)
	if len(preview.encode("utf-8", errors="replace")) >= original_size:
		return original_size
	path.write_text(
		(
			f"[HTN evaluation compact evidence]\n"
			f"original_bytes={original_size}\n\n"
			f"{preview}"
		),
		encoding="utf-8",
	)
	return original_size


def _compact_verification_result(
	result: Any,
	*,
	json_filename: str,
) -> Dict[str, Any]:
	payload = dict(result.to_dict())
	stdout = str(payload.pop("stdout", "") or "")
	stderr = str(payload.pop("stderr", "") or "")
	output_file = payload.get("output_file")
	if output_file:
		original_output_bytes = _rewrite_text_file_as_preview(
			output_file,
			IPCPlanVerifier._combine_output(stdout, stderr),
		)
		if original_output_bytes is not None:
			payload["output_original_bytes"] = original_output_bytes
	payload["stdout_preview"] = _text_preview(stdout)
	payload["stderr_preview"] = _text_preview(stderr)
	payload["stdout_chars"] = len(stdout)
	payload["stderr_chars"] = len(stderr)
	json_path = Path(str(payload.get("output_file") or "")).with_name(json_filename)
	if json_path.parent.exists():
		json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	return payload


def _compact_solver_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
	compact: Dict[str, Any] = {}
	for key in (
		"solver_id",
		"mode",
		"status",
		"bucket",
		"reason",
		"error",
		"seconds",
		"step_count",
		"has_hierarchical_trace",
		"raw_plan_path",
		"actual_plan_path",
		"verification_output_dir",
		"engine_stdout_path",
		"engine_stderr_path",
	):
		if candidate.get(key) is not None:
			compact[key] = candidate.get(key)
	for path_key in (
		"raw_plan_path",
		"actual_plan_path",
		"engine_stdout_path",
		"engine_stderr_path",
	):
		if compact.get(path_key):
			compact[f"{path_key}_bytes"] = _path_size(compact[path_key])
	action_path = candidate.get("action_path")
	if isinstance(action_path, list):
		compact["action_path_count"] = len(action_path)
	if candidate.get("command"):
		compact["command"] = list(candidate.get("command") or ())
	if candidate.get("stdout_head"):
		compact["stdout_preview"] = _text_preview(candidate.get("stdout_head"))
	if candidate.get("stderr_head"):
		compact["stderr_preview"] = _text_preview(candidate.get("stderr_head"))
	if isinstance(candidate.get("primitive_verification"), dict):
		compact["primitive_verification"] = _compact_verification_payload(
			dict(candidate["primitive_verification"]),
		)
	if isinstance(candidate.get("hierarchical_verification"), dict):
		compact["hierarchical_verification"] = _compact_verification_payload(
			dict(candidate["hierarchical_verification"]),
		)
	return compact


def _compact_verification_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
	compact = {
		key: payload.get(key)
		for key in (
			"tool_available",
			"command",
			"plan_file",
			"output_file",
			"output_original_bytes",
			"primitive_plan_only",
			"primitive_plan_executable",
			"verification_result",
			"reached_goal_state",
			"plan_kind",
			"build_warning",
			"error",
			"stdout_chars",
			"stderr_chars",
			"stdout_preview",
			"stderr_preview",
		)
		if payload.get(key) is not None
	}
	return compact


def _compact_plan_solve_data_for_parent(plan_solve_data: Dict[str, Any]) -> Dict[str, Any]:
	summary = dict(plan_solve_data.get("summary") or {})
	artifacts = dict(plan_solve_data.get("artifacts") or {})
	nested_artifacts = dict(artifacts.get("artifacts") or {})
	compact_artifacts = {
		key: artifacts.get(key)
		for key in (
			"planner_id",
			"status",
			"planning_mode",
			"engine_mode",
			"solver_id",
			"task_network_ordered",
			"step_count",
			"guided_hierarchical_plan_source",
			"failure_bucket",
		)
		if artifacts.get(key) is not None
	}
	if artifacts.get("planning_representation"):
		representation = dict(artifacts.get("planning_representation") or {})
		compact_artifacts["planning_representation"] = {
			key: representation.get(key)
			for key in (
				"representation_id",
				"representation_source",
				"ordering_kind",
				"domain_file",
				"problem_file",
				"compilation_profile",
			)
			if representation.get(key) is not None
		}
	if isinstance(artifacts.get("task_network"), list):
		compact_artifacts["task_network_count"] = len(artifacts["task_network"])
	if isinstance(artifacts.get("ordering_edges"), list):
		compact_artifacts["ordering_edge_count"] = len(artifacts["ordering_edges"])
	if nested_artifacts:
		compact_artifacts["artifact_files"] = {
			key: value
			for key, value in nested_artifacts.items()
			if value is not None
		}
		compact_artifacts["artifact_file_bytes"] = {
			key: _path_size(value)
			for key, value in nested_artifacts.items()
			if value is not None
		}
	if isinstance(artifacts.get("solver_candidates"), list):
		compact_artifacts["solver_candidates"] = [
			_compact_solver_candidate(dict(candidate))
			for candidate in artifacts["solver_candidates"]
		]
	if artifacts.get("failure_metadata"):
		failure_metadata = dict(artifacts.get("failure_metadata") or {})
		compact_artifacts["failure_metadata"] = {
			key: value
			for key, value in failure_metadata.items()
			if key not in {"domain_hddl", "problem_hddl", "engine_attempts"}
		}
		if isinstance(failure_metadata.get("engine_attempts"), list):
			compact_artifacts["failure_solver_candidates"] = [
				_compact_solver_candidate(dict(candidate))
				for candidate in failure_metadata["engine_attempts"]
			]
	return {"summary": summary, "artifacts": compact_artifacts}


def _compact_plan_verification_data_for_parent(
	plan_verification_data: Dict[str, Any],
) -> Dict[str, Any]:
	summary = dict(plan_verification_data.get("summary") or {})
	artifacts = dict(plan_verification_data.get("artifacts") or {})
	compact_artifacts = {
		key: artifacts.get(key)
		for key in (
			"tool_available",
			"plan_kind",
			"verification_result",
			"primitive_plan_executable",
			"reached_goal_state",
			"selected_solver_id",
			"selected_bucket",
			"selection_rule",
			"plan_file",
			"output_file",
			"json_file",
			"output_original_bytes",
			"stdout_chars",
			"stderr_chars",
			"stdout_preview",
			"stderr_preview",
		)
		if artifacts.get(key) is not None
	}
	if isinstance(artifacts.get("solver_candidates"), list):
		compact_artifacts["solver_candidates"] = [
			_compact_solver_candidate(dict(candidate))
			for candidate in artifacts["solver_candidates"]
		]
	return {"summary": summary, "artifacts": compact_artifacts}


def _prune_heavy_planner_artifacts(output_dir: str | Path) -> None:
	if os.environ.get("HTN_EVAL_KEEP_RAW_PLANNER_ARTIFACTS") == "1":
		return
	root = Path(output_dir)
	if not root.exists():
		return
	for path in root.rglob("*"):
		if path.is_file() and path.name in HTN_HEAVY_ARTIFACT_NAMES:
			try:
				original_size = int(path.stat().st_size)
				path.write_text(
					(
						"[HTN evaluation compact evidence]\n"
						f"original_bytes={original_size}\n"
						"raw planner intermediate omitted; "
						"problem_results.json and execution.json retain the audit summary.\n"
					),
					encoding="utf-8",
				)
			except OSError:
				continue


def solve_problem_root_primary_planner_task(
	context: HTNEvaluationContext,
	*,
	planning_task: PrimaryPlannerTask,
	timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
	"""Solve one compiled official representation with the primary HTN planner."""
	print("\n[PLAN SOLVE]")
	print("-" * 80)
	stage_start = time.perf_counter()
	effective_timeout_seconds = context._official_problem_root_planning_timeout_seconds(
		timeout_seconds,
	)
	if context.output_dir is None:
		raise ValueError("Problem-root planning requires an output directory.")
	planner = primary_planner_by_id(
		planning_task.planner_id,
		workspace=str(context.output_dir),
	)

	try:
		if not context.problem.htn_tasks:
			raise ValueError("Problem file contains no root HTN tasks.")
		if not planner.toolchain_available():
			raise ValueError(
				f"Primary HTN planner '{planning_task.planner_id}' is unavailable on PATH.",
			)

		task_network = tuple(
			(str(task.task_name), tuple(str(arg) for arg in (task.args or ())))
			for task in (context.problem.htn_tasks or ())
		)
		ordering_edges = context._problem_root_task_network_ordering_edges()
		task_network_ordered = context._problem_root_task_network_is_totally_ordered()
		primary_task_name, primary_task_args = task_network[0]
		plan = planner.solve(
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
			"planner_id": planning_task.planner_id,
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
			"guided_hierarchical_plan_source": planner.plan_source_label,
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
			"planner_id": planning_task.planner_id,
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
				"planner_id": planning_task.planner_id,
				"representation_id": planning_task.representation.representation_id,
			},
		)
		print(f"✓ Planner returned {len(action_path)} primitive steps")
		print(f"  Problem root task count: {len(task_network)}")
		return {"summary": summary, "artifacts": artifacts}
	except Exception as exc:
		failure_metadata = dict(getattr(exc, "metadata", {}) or {})
		failure_artifacts = {
			"planner_id": planning_task.planner_id,
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
			"planner_id": planning_task.planner_id,
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


def verify_primary_planner_solution(
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
			"verifier_id": "pandaPIparser",
			"status": "failed",
			"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
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
			"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
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
		candidate_output_dir = (
			context.output_dir
			/ "primary_planner_verification"
			/ _sanitize_identifier(solver_id)
		)
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
		primitive_payload = _compact_verification_result(
			primitive_result,
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
		hierarchical_payload = _compact_verification_result(
			hierarchical_result,
			json_filename="hierarchical_verification.json",
		)

		primitive_valid = (
			primitive_result.tool_available is True
			and primitive_result.primitive_plan_executable is True
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
			"primitive_verification": primitive_payload,
			"hierarchical_verification": hierarchical_payload,
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
		"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
		"solver_candidates": verified_candidates,
	}
	summary = {
		"verifier_id": "pandaPIparser",
		"status": "success" if selected_success is not None else "failed",
		"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
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
				"verifier_id": "pandaPIparser",
				"status": "failed",
			},
		)
		context._record_step_timing("plan_verification", stage_start)
		print(f"✗ Official verification failed: {error}")
		return {
			"summary": {
				"verifier_id": "pandaPIparser",
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
	return verify_primary_planner_solution(
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
	"""Spawn-safe worker for one primary planner attempt."""
	plan_solve_seconds = 0.0
	plan_verification_seconds = 0.0
	total_start = time.perf_counter()
	captured_stdout = io.StringIO()
	captured_stderr = io.StringIO()
	planning_task = PrimaryPlannerTask.from_dict(dict(task_payload))
	try:
		if hasattr(os, "setsid"):
			os.setsid()
		resource_profile = _apply_official_resource_profile(
			memory_limit_mib=OFFICIAL_BENCHMARK_MEMORY_LIMIT_MIB,
			cpu_count=OFFICIAL_BENCHMARK_CPU_COUNT,
		)
		context = HTNEvaluationContext(
			domain_file=domain_file,
			problem_file=problem_file,
		)
		context.logger = _NullExecutionLogger()
		context.output_dir = Path(output_dir).resolve()
		context.output_dir.mkdir(parents=True, exist_ok=True)

		with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
			plan_solve_start = time.perf_counter()
			plan_solve_data = solve_problem_root_primary_planner_task(
				context,
				planning_task=planning_task,
				timeout_seconds=planning_timeout_seconds,
			)
			plan_solve_seconds = time.perf_counter() - plan_solve_start

			plan_verification_start = time.perf_counter()
			plan_verification_data = verify_plan_officially(context, plan_solve_data)
			plan_verification_seconds = time.perf_counter() - plan_verification_start

		plan_solve_data = _compact_plan_solve_data_for_parent(plan_solve_data)
		plan_verification_data = _compact_plan_verification_data_for_parent(
			plan_verification_data,
		)
		_prune_heavy_planner_artifacts(output_dir)
		plan_solve_summary = dict((plan_solve_data or {}).get("summary") or {})
		plan_verification_summary = dict((plan_verification_data or {}).get("summary") or {})
		result_queue.put(
			{
				"message_type": "primary_planner_attempt",
				"planner_id": planning_task.planner_id,
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
				"resource_profile": resource_profile,
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
				"message_type": "primary_planner_attempt",
				"planner_id": planning_task.planner_id,
				"task_id": planning_task.task_id,
				"representation_id": planning_task.representation.representation_id,
				"output_dir": str(Path(output_dir).resolve()),
				"plan_solve_data": {
					"summary": {
						"planner_id": planning_task.planner_id,
						"status": "failed",
						"planning_mode": "official_problem_root",
						"failure_bucket": "worker_exception",
						"representation_id": planning_task.representation.representation_id,
					},
					"artifacts": {
						"planner_id": planning_task.planner_id,
						"status": "failed",
						"planning_mode": "official_problem_root",
						"failure_bucket": "worker_exception",
						"planning_representation": planning_task.representation.to_dict(),
					},
				},
				"plan_verification_data": {
					"summary": {
						"verifier_id": "pandaPIparser",
						"status": "failed",
						"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
						"failure_bucket": "worker_exception",
					},
					"artifacts": {
						"planner_id": planning_task.planner_id,
						"status": "failed",
						"selection_rule": PRIMARY_PLANNER_SELECTION_RULE,
						"failure_bucket": "worker_exception",
						"error": str(exc),
					},
				},
				"plan_solve_seconds": plan_solve_seconds,
				"plan_verification_seconds": plan_verification_seconds,
				"total_seconds": time.perf_counter() - total_start,
				"success": False,
				"resource_profile": {
					"requested_memory_limit_mib": OFFICIAL_BENCHMARK_MEMORY_LIMIT_MIB,
					"requested_cpu_count": OFFICIAL_BENCHMARK_CPU_COUNT,
					"memory_limit_enforced": False,
					"cpu_affinity_enforced": False,
					"memory_limit_strategy": "worker_exception_before_context",
					"cpu_affinity_strategy": "worker_exception_before_context",
					"platform": sys.platform,
				},
				"selected_bucket": "worker_exception",
				"stdout": captured_stdout.getvalue(),
				"stderr": captured_stderr.getvalue(),
			},
		)
