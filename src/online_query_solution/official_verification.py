"""
Official hierarchical verification helpers for online Jason execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from offline_method_generation.method_synthesis.schema import HTNMethodLibrary
from online_query_solution.domain_selection import (
	ONLINE_DOMAIN_SOURCE_BENCHMARK,
	OnlineDomainContext,
)
from planning.panda_portfolio import PANDAPlanner
from verification.official_plan_verifier import IPCPlanVerifier


@dataclass(frozen=True)
class OnlineVerificationOutcome:
	"""Structured official-verifier outcome for online query execution."""

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


def rewrite_guided_plan_source_names(
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
	online_domain: OnlineDomainContext,
	output_dir: str | Path,
) -> Tuple[Path, float]:
	"""Resolve the HDDL domain file used by official hierarchical verification."""

	if online_domain.source == ONLINE_DOMAIN_SOURCE_BENCHMARK:
		return Path(online_domain.domain_file).resolve(), 0.0

	domain_build_start = time.perf_counter()
	verification_domain_file = build_verification_domain(
		method_library=method_library,
		online_domain=online_domain,
		output_dir=output_dir,
	)
	return verification_domain_file, time.perf_counter() - domain_build_start


def build_verification_domain(
	*,
	method_library: HTNMethodLibrary,
	online_domain: OnlineDomainContext,
	output_dir: str | Path,
) -> Path:
	"""Build a verifier-facing generated domain when the online source is generated."""

	planner = PANDAPlanner()
	verification_domain_hddl = planner._build_domain_hddl(
		online_domain.domain,
		method_library,
		online_domain.domain.name,
		export_source_names=True,
	)
	verification_domain_path = Path(output_dir).resolve() / "ipc_verification_domain.hddl"
	verification_domain_path.write_text(verification_domain_hddl)
	return verification_domain_path


def verify_jason_hierarchical_plan(
	*,
	method_library: HTNMethodLibrary,
	plan_solve_data: Dict[str, Any],
	online_domain: OnlineDomainContext,
	problem_file: Optional[str | Path],
	output_dir: str | Path,
) -> OnlineVerificationOutcome:
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
		return OnlineVerificationOutcome(
			success=True,
			data={"summary": summary, "artifacts": artifacts},
			error=None,
			timing_breakdown={},
		)

	verifier = IPCPlanVerifier()
	if not verifier.tool_available():
		error = "pandaPIparser is not available on PATH for official IPC verification"
		return OnlineVerificationOutcome(
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
	guided_plan_text = plan_solve_artifacts.get("guided_hierarchical_plan_text")
	verification_domain_file, domain_build_seconds = resolve_verification_domain_file(
		method_library=method_library,
		online_domain=online_domain,
		output_dir=output_dir,
	)

	verifier_start = time.perf_counter()
	if guided_plan_text:
		guided_plan_text = rewrite_guided_plan_source_names(
			str(guided_plan_text),
			method_library,
		)
		if guided_plan_text and not guided_plan_text.endswith("\n"):
			guided_plan_text = f"{guided_plan_text}\n"
		verifier_result = verifier.verify_plan_text(
			domain_file=verification_domain_file,
			problem_file=verification_problem_file,
			plan_text=guided_plan_text,
			output_dir=output_dir,
			plan_kind="hierarchical",
			build_warning=None,
		)
	else:
		verifier_result = verifier.verify_plan(
			domain_file=verification_domain_file,
			problem_file=verification_problem_file,
			action_path=plan_solve_artifacts.get("action_path") or [],
			method_library=method_library,
			method_trace=plan_solve_artifacts.get("method_trace") or [],
			output_dir=output_dir,
		)
	verifier_seconds = time.perf_counter() - verifier_start

	artifacts = verifier_result.to_dict()
	summary = {
		"backend": "pandaPIparser",
		"status": "success" if verifier_result.verification_result is True else "failed",
		"tool_available": verifier_result.tool_available,
		"plan_kind": verifier_result.plan_kind,
		"verification_result": verifier_result.verification_result,
		"primitive_plan_executable": verifier_result.primitive_plan_executable,
		"reached_goal_state": verifier_result.reached_goal_state,
		"build_warning": verifier_result.build_warning,
		"verification_mode": verification_mode,
		"online_domain_source": online_domain.source,
		"online_domain_file": online_domain.domain_file,
		"verification_domain_file": str(verification_domain_file),
		"verification_problem_file": verification_problem_file,
	}

	success = (
		verifier_result.tool_available
		and verifier_result.plan_kind == "hierarchical"
		and verifier_result.verification_result is True
	)
	error = None
	if not success:
		error = (
			"Official IPC verifier rejected the generated hierarchical plan: "
			f"plan_kind={verifier_result.plan_kind}, "
			f"verification_result={verifier_result.verification_result}"
		)

	return OnlineVerificationOutcome(
		success=success,
		data={"summary": summary, "artifacts": artifacts},
		error=error,
		timing_breakdown={
			"verification_domain_build_seconds": domain_build_seconds,
			"official_verifier_seconds": verifier_seconds,
		},
	)
