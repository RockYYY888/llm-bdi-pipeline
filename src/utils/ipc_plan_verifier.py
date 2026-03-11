from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


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
	error: Optional[str] = None

	def to_dict(self) -> Dict[str, object]:
		return {
			"tool_available": self.tool_available,
			"command": list(self.command),
			"plan_file": self.plan_file,
			"output_file": self.output_file,
			"stdout": self.stdout,
			"stderr": self.stderr,
			"primitive_plan_only": self.primitive_plan_only,
			"primitive_plan_executable": self.primitive_plan_executable,
			"verification_result": self.verification_result,
			"reached_goal_state": self.reached_goal_state,
			"error": self.error,
		}


class IPCPlanVerifier:
	"""Run the official PANDA HTN verifier on primitive-only plans."""

	def __init__(self, parser_cmd: str = "pandaPIparser") -> None:
		self.parser_cmd = parser_cmd

	def tool_available(self) -> bool:
		return shutil.which(self.parser_cmd) is not None

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
		output_path = Path(output_dir).resolve()
		output_path.mkdir(parents=True, exist_ok=True)
		plan_path = output_path / plan_filename
		output_text_path = output_path / output_filename
		output_json_path = output_path / json_filename
		plan_text = self.render_primitive_only_plan(action_path)
		plan_path.write_text(plan_text)

		command = [
			self.parser_cmd,
			"-V",
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
			primitive_plan_executable=self._extract_bool(
				combined,
				"Primitive plan alone executable",
			),
			verification_result=self._extract_bool(
				combined,
				"Plan verification result",
			),
			reached_goal_state=self._infer_goal_reached(combined),
			error=None if completed.returncode == 0 else f"verifier exited with code {completed.returncode}",
		)
		output_json_path.write_text(json.dumps(result.to_dict(), indent=2))
		return result

	@staticmethod
	def render_primitive_only_plan(action_path: Sequence[str]) -> str:
		lines = ["==>"]
		for index, action_step in enumerate(action_path):
			match = re.fullmatch(r"([^\s(]+)\((.*)\)", action_step.strip())
			if match is None:
				raise ValueError(f"Invalid action_path step for IPC verifier: {action_step}")
			args = [arg.strip() for arg in match.group(2).split(",") if arg.strip()]
			lines.append(" ".join([str(index), match.group(1), *args]))
		lines.append("root")
		return "\n".join(lines)

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
	def _extract_bool(text: str, label: str) -> Optional[bool]:
		match = re.search(rf"{re.escape(label)}:\s*(true|false)", text, re.IGNORECASE)
		if match is None:
			return None
		return match.group(1).lower() == "true"

	@staticmethod
	def _infer_goal_reached(text: str) -> Optional[bool]:
		if "Primitive plan does not reach the goal state" in text:
			return False
		primitive_plan_executable = IPCPlanVerifier._extract_bool(
			text,
			"Primitive plan alone executable",
		)
		if primitive_plan_executable is None:
			return None
		return primitive_plan_executable
