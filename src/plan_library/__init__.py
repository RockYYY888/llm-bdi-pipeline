"""
Plan-library exports for the dissertation workflow.

Keep package import side effects minimal so stage modules can import
plan-library types without triggering the full generation pipeline.
"""

from __future__ import annotations

from typing import Any

__all__ = [
	"AgentSpeakBodyStep",
	"AgentSpeakPlan",
	"AgentSpeakTrigger",
	"LibraryValidationRecord",
	"PlanLibrary",
	"PlanLibraryArtifactBundle",
	"PlanLibraryGenerationPipeline",
	"PlanLibrarySetResult",
	"TranslationCoverage",
	"build_plan_library",
	"build_library_validation_record",
	"deduplicate_plan_library",
	"load_plan_library_artifact_bundle",
	"persist_plan_library_artifact_bundle",
	"plan_fingerprint",
	"render_plan_library_asl",
]


def __getattr__(name: str) -> Any:
	if name in {
		"PlanLibraryArtifactBundle",
		"load_plan_library_artifact_bundle",
		"persist_plan_library_artifact_bundle",
	}:
		from .artifacts import (
			PlanLibraryArtifactBundle,
			load_plan_library_artifact_bundle,
			persist_plan_library_artifact_bundle,
		)

		return {
			"PlanLibraryArtifactBundle": PlanLibraryArtifactBundle,
			"load_plan_library_artifact_bundle": load_plan_library_artifact_bundle,
			"persist_plan_library_artifact_bundle": persist_plan_library_artifact_bundle,
		}[name]
	if name in {
		"AgentSpeakBodyStep",
		"AgentSpeakPlan",
		"AgentSpeakTrigger",
		"LibraryValidationRecord",
		"PlanLibrary",
		"TranslationCoverage",
	}:
		from .models import (
			AgentSpeakBodyStep,
			AgentSpeakPlan,
			AgentSpeakTrigger,
			LibraryValidationRecord,
			PlanLibrary,
			TranslationCoverage,
		)

		return {
			"AgentSpeakBodyStep": AgentSpeakBodyStep,
			"AgentSpeakPlan": AgentSpeakPlan,
			"AgentSpeakTrigger": AgentSpeakTrigger,
			"LibraryValidationRecord": LibraryValidationRecord,
			"PlanLibrary": PlanLibrary,
			"TranslationCoverage": TranslationCoverage,
		}[name]
	if name == "PlanLibraryGenerationPipeline":
		from .pipeline import PlanLibraryGenerationPipeline

		return PlanLibraryGenerationPipeline
	if name == "render_plan_library_asl":
		from .rendering import render_plan_library_asl

		return render_plan_library_asl
	if name == "build_plan_library":
		from .translation import build_plan_library

		return build_plan_library
	if name == "build_library_validation_record":
		from .validation import build_library_validation_record

		return build_library_validation_record
	if name in {"PlanLibrarySetResult", "deduplicate_plan_library", "plan_fingerprint"}:
		from .set_semantics import (
			PlanLibrarySetResult,
			deduplicate_plan_library,
			plan_fingerprint,
		)

		return {
			"PlanLibrarySetResult": PlanLibrarySetResult,
			"deduplicate_plan_library": deduplicate_plan_library,
			"plan_fingerprint": plan_fingerprint,
		}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
