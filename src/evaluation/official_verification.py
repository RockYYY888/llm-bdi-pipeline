"""
Official hierarchical verification helpers for plan-library evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from method_library.synthesis.schema import HTNMethodLibrary
from evaluation.domain_selection import (
	EVALUATION_DOMAIN_SOURCE_BENCHMARK,
	EvaluationDomainContext,
)
from evaluation.failure_signature import infer_missing_goal_facts
from planning.panda_sat import PANDAPlanner
from verification.official_plan_verifier import IPCPlanVerifier


@dataclass(frozen=True)
class EvaluationVerificationOutcome:
	"""Structured official-verifier outcome for plan-library evaluation."""

	success: bool
	data: Optional[Dict[str, Any]]
	error: Optional[str]
	timing_breakdown: Dict[str, float]


def render_supported_hierarchical_plan(
	*,
	action_path: Sequence[str],
	method_library: HTNMethodLibrary,
	method_trace: Sequence[Dict[str, Any]],
	domain_file: str | Path,
	problem_file: str | Path,
) -> Optional[str]:
	"""Render Jason action and method traces as verifier-supported hierarchical text."""

	verifier = IPCPlanVerifier()
	try:
		return verifier._render_supported_hierarchical_plan(
			domain_file=str(Path(domain_file).resolve()),
			problem_file=str(Path(problem_file).resolve()),
			action_path=action_path,
			method_library=method_library,
			method_trace=method_trace,
		)
	except Exception:
		return None


def rewrite_hierarchical_plan_source_names(
	plan_text: str,
	method_library: HTNMethodLibrary,
) -> str:
	"""Rewrite internal task names in hierarchical plan text to benchmark source names."""

	task_name_map = {
		task.name: str(task.source_name).strip()
		for task in getattr(method_library, "compound_tasks", ())
		if str(getattr(task, "source_name", "") or "").strip()
	}
	if not task_name_map:
		return plan_text

	trailing_newline = str(plan_text).endswith("\n")
	rewritten_lines = []
	for raw_line in str(plan_text).splitlines():
		stripped = raw_line.strip()
		if "->" not in stripped:
			rewritten_lines.append(raw_line)
			continue
		before, _, after = stripped.partition("->")
		head_tokens = before.split()
		if len(head_tokens) < 2 or not head_tokens[0].isdigit():
			rewritten_lines.append(raw_line)
			continue
		source_name = task_name_map.get(head_tokens[1])
		if not source_name:
			rewritten_lines.append(raw_line)
			continue
		head_tokens[1] = source_name
		leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
		rewritten_lines.append(f"{leading}{' '.join(head_tokens)} -> {after.strip()}".rstrip())

	rewritten = "\n".join(rewritten_lines)
	if trailing_newline:
		rewritten += "\n"
	return rewritten


def resolve_verification_domain_file(
	*,
	method_library: HTNMethodLibrary,
	evaluation_domain: EvaluationDomainContext,
	output_dir: str | Path,
) -> Tuple[Path, float]:
	"""Resolve the HDDL domain file used by official hierarchical verification."""

	if evaluation_domain.source == EVALUATION_DOMAIN_SOURCE_BENCHMARK:
		return Path(evaluation_domain.domain_file).resolve(), 0.0

	domain_build_start = time.perf_counter()
	verification_domain_file = build_verification_domain(
		method_library=method_library,
		evaluation_domain=evaluation_domain,
		output_dir=output_dir,
	)
	return verification_domain_file, time.perf_counter() - domain_build_start


def build_verification_domain(
	*,
	method_library: HTNMethodLibrary,
	evaluation_domain: EvaluationDomainContext,
	output_dir: str | Path,
) -> Path:
	"""Build a verifier-facing generated domain when the evaluation source is generated."""

	planner = PANDAPlanner()
	verification_domain_hddl = planner._build_domain_hddl(
		evaluation_domain.domain,
		method_library,
		evaluation_domain.domain.name,
		export_source_names=True,
	)
	verification_domain_path = Path(output_dir).resolve() / "ipc_verification_domain.hddl"
	verification_domain_path.write_text(verification_domain_hddl)
	return verification_domain_path


def verify_jason_hierarchical_plan(
	*,
	method_library: HTNMethodLibrary,
	plan_solve_data: Dict[str, Any],
	evaluation_domain: EvaluationDomainContext,
	problem_file: Optional[str | Path],
	output_dir: str | Path,
) -> EvaluationVerificationOutcome:
	"""Run official hierarchical verification for the Jason-produced plan."""

	if problem_file is None:
		summary = {
			"backend": "pandaPIparser",
			"status": "skipped",
			"reason": "No problem_file was provided",
		}
		artifacts = {
			"tool_available": None,
			"plan_kind": None,
			"verification_result": None,
			"primitive_plan_executable": None,
			"reached_goal_state": None,
		}
		return EvaluationVerificationOutcome(
			success=True,
			data={"summary": summary, "artifacts": artifacts},
			error=None,
			timing_breakdown={},
		)

	verifier = IPCPlanVerifier()
	if not verifier.tool_available():
		error = "pandaPIparser is not available on PATH for official IPC verification"
		return EvaluationVerificationOutcome(
			success=False,
			data=None,
			error=error,
			timing_breakdown={},
		)

	plan_solve_artifacts = plan_solve_data.get("artifacts") or {}
	verification_problem_file = str(
		plan_solve_artifacts.get("verification_problem_file")
		or Path(problem_file).resolve()
	)
	verification_mode = str(plan_solve_artifacts.get("verification_mode") or "original_problem")
	hierarchical_plan_text = plan_solve_artifacts.get("hierarchical_plan_text")
	verification_domain_file, domain_build_seconds = resolve_verification_domain_file(
		method_library=method_library,
		evaluation_domain=evaluation_domain,
		output_dir=output_dir,
	)

	verifier_start = time.perf_counter()
	hierarchical_verifier_result = None
	primitive_verifier_result = None
	if hierarchical_plan_text:
		hierarchical_plan_text = rewrite_hierarchical_plan_source_names(
			str(hierarchical_plan_text),
			method_library,
		)
		if hierarchical_plan_text and not hierarchical_plan_text.endswith("\n"):
			hierarchical_plan_text = f"{hierarchical_plan_text}\n"
		hierarchical_verifier_result = verifier.verify_plan_text(
			domain_file=verification_domain_file,
			problem_file=verification_problem_file,
			plan_text=hierarchical_plan_text,
			output_dir=output_dir,
			plan_kind="hierarchical",
			build_warning=None,
		)
		verifier_result = hierarchical_verifier_result
		if hierarchical_verifier_result.verification_result is not True:
			primitive_verifier_result = verifier.verify_primitive_plan(
				domain_file=verification_domain_file,
				problem_file=verification_problem_file,
				action_path=plan_solve_artifacts.get("action_path") or [],
				output_dir=output_dir,
				plan_filename="ipc_official_primitive_plan.txt",
				output_filename="ipc_official_primitive_verifier.txt",
				json_filename="ipc_official_primitive_verification.json",
			)
			verifier_result = primitive_verifier_result
	else:
		primitive_verifier_result = verifier.verify_primitive_plan(
			domain_file=verification_domain_file,
			problem_file=verification_problem_file,
			action_path=plan_solve_artifacts.get("action_path") or [],
			output_dir=output_dir,
		)
		verifier_result = primitive_verifier_result
	verifier_seconds = time.perf_counter() - verifier_start

	artifacts = verifier_result.to_dict()
	if hierarchical_verifier_result is not None:
		artifacts["hierarchical_verification"] = hierarchical_verifier_result.to_dict()
	if primitive_verifier_result is not None and primitive_verifier_result is not verifier_result:
		artifacts["primitive_runtime_verification"] = primitive_verifier_result.to_dict()
	replayed_world_facts = (
		(((plan_solve_artifacts.get("consistency_checks") or {}).get("action_path_schema_replay") or {}).get(
			"world_facts",
		))
		or ()
	)
	missing_goal_facts = infer_missing_goal_facts(
		problem_file=verification_problem_file,
		world_facts=replayed_world_facts,
	)
	if missing_goal_facts:
		artifacts["missing_goal_facts"] = list(missing_goal_facts)
	runtime_goal_reached = bool(replayed_world_facts) and not missing_goal_facts
	hierarchical_success = (
		verifier_result.tool_available
		and verifier_result.plan_kind == "hierarchical"
		and verifier_result.verification_result is True
	)
	primitive_runtime_success = (
		verifier_result.tool_available
		and verifier_result.plan_kind == "primitive_only"
		and verifier_result.primitive_plan_executable is True
		and runtime_goal_reached
	)
	success = hierarchical_success or primitive_runtime_success
	summary = {
		"backend": "pandaPIparser",
		"status": "success" if success else "failed",
		"tool_available": verifier_result.tool_available,
		"plan_kind": verifier_result.plan_kind,
		"verification_result": verifier_result.verification_result,
		"primitive_plan_executable": verifier_result.primitive_plan_executable,
		"reached_goal_state": verifier_result.reached_goal_state,
		"runtime_goal_reached": runtime_goal_reached,
		"build_warning": verifier_result.build_warning,
		"verification_mode": verification_mode,
		"evaluation_domain_source": evaluation_domain.source,
		"evaluation_domain_file": evaluation_domain.domain_file,
		"verification_domain_file": str(verification_domain_file),
		"verification_problem_file": verification_problem_file,
		"missing_goal_facts": list(missing_goal_facts),
		"hierarchical_verification_result": (
			hierarchical_verifier_result.verification_result
			if hierarchical_verifier_result is not None
			else None
		),
		"primitive_runtime_fallback_used": (
			hierarchical_verifier_result is not None
			and primitive_verifier_result is verifier_result
		),
	}

	error = None
	if not success:
		error = (
			"Official IPC verifier rejected the generated plan: "
			f"plan_kind={verifier_result.plan_kind}, "
			f"verification_result={verifier_result.verification_result}, "
			f"primitive_plan_executable={verifier_result.primitive_plan_executable}, "
			f"runtime_goal_reached={runtime_goal_reached}"
		)

	return EvaluationVerificationOutcome(
		success=success,
		data={"summary": summary, "artifacts": artifacts},
		error=error,
		timing_breakdown={
			"verification_domain_build_seconds": domain_build_seconds,
			"official_verifier_seconds": verifier_seconds,
		},
	)
