"""Environment-adapter interfaces for real Jason runtime validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class EnvironmentAdapterResult:
	"""Result of post-run Jason environment validation."""

	success: bool
	adapter_name: str
	mode: str
	error: str | None = None
	details: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		payload = {
			"success": self.success,
			"adapter": self.adapter_name,
			"mode": self.mode,
			"details": dict(self.details),
		}
		if self.error is not None:
			payload["error"] = self.error
		return payload


class Stage6EnvironmentAdapter(ABC):
	"""Validate Jason runtime traces against real environment expectations."""

	name: str = "unknown"
	mode: str = "unknown"

	@abstractmethod
	def validate(self, *, stdout: str, stderr: str) -> EnvironmentAdapterResult:
		"""Validate runtime traces and return pass/fail metadata."""


class JasonEnvironmentRuntimeAdapter(Stage6EnvironmentAdapter):
	"""Validate that a real Jason environment was booted and actions did not fail."""

	name = "jason_environment_runtime"
	mode = "real"

	def validate(self, *, stdout: str, stderr: str) -> EnvironmentAdapterResult:
		has_ready_marker = "runtime env ready" in stdout
		has_action_failure = "runtime env action failed" in stdout
		has_unknown_action = "runtime env unknown action" in stdout
		has_compile_error = "runtime env compile failed" in stdout

		details = {
			"has_ready_marker": has_ready_marker,
			"has_action_failure": has_action_failure,
			"has_unknown_action": has_unknown_action,
			"has_compile_error": has_compile_error,
		}
		if not has_ready_marker:
			return EnvironmentAdapterResult(
				success=False,
				adapter_name=self.name,
				mode=self.mode,
				error="environment ready marker not found in runtime output",
				details=details,
			)
		if has_action_failure or has_unknown_action or has_compile_error:
			return EnvironmentAdapterResult(
				success=False,
				adapter_name=self.name,
				mode=self.mode,
				error="environment reported action/runtime failure",
				details=details,
			)
		return EnvironmentAdapterResult(
			success=True,
			adapter_name=self.name,
			mode=self.mode,
			details=details,
		)


def build_environment_adapter(name: str | None = None) -> Stage6EnvironmentAdapter:
	"""Create a Jason runtime environment adapter by name."""

	adapter_name = (name or "jason_environment_runtime").strip().lower()
	if adapter_name in {"jason_environment_runtime", "default", "runtime"}:
		return JasonEnvironmentRuntimeAdapter()
	raise ValueError(f"Unknown Jason runtime environment adapter: {name}")
