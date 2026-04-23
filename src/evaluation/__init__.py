"""
Chapter 5 evaluation exports.

Keep package import side effects minimal so evaluation submodules can import
each other without forcing the full evaluation pipeline during package init.
"""

from __future__ import annotations

from typing import Any

__all__ = [
	"EnvironmentAdapterResult",
	"JasonRunner",
	"JasonValidationError",
	"JasonValidationResult",
	"PlanLibraryEvaluationPipeline",
	"Stage6EnvironmentAdapter",
	"build_environment_adapter",
]


def __getattr__(name: str) -> Any:
	if name == "PlanLibraryEvaluationPipeline":
		from .pipeline import PlanLibraryEvaluationPipeline

		return PlanLibraryEvaluationPipeline
	if name in {
		"EnvironmentAdapterResult",
		"JasonRunner",
		"JasonValidationError",
		"JasonValidationResult",
		"Stage6EnvironmentAdapter",
		"build_environment_adapter",
	}:
		from .runtime import (
			EnvironmentAdapterResult,
			JasonRunner,
			JasonValidationError,
			JasonValidationResult,
			Stage6EnvironmentAdapter,
			build_environment_adapter,
		)

		return {
			"EnvironmentAdapterResult": EnvironmentAdapterResult,
			"JasonRunner": JasonRunner,
			"JasonValidationError": JasonValidationError,
			"JasonValidationResult": JasonValidationResult,
			"Stage6EnvironmentAdapter": Stage6EnvironmentAdapter,
			"build_environment_adapter": build_environment_adapter,
		}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
