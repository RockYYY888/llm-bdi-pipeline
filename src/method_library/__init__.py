"""
Method-library exports for the plan-library workflow.

Keep package import side effects minimal so subprocess spawn can import
`method_library.synthesis.schema` without pulling in the full method-library synthesis
stack and creating circular dependencies.
"""

from __future__ import annotations

from typing import Any

from .synthesis.schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
	HTNTargetTaskBinding,
)

__all__ = [
	"HTNLiteral",
	"HTNMethod",
	"HTNMethodLibrary",
	"HTNMethodStep",
	"HTNTask",
	"HTNTargetTaskBinding",
	"HTNMethodSynthesizer",
	"MethodLibraryValidator",
	"build_domain_htn_system_prompt",
	"build_domain_htn_user_prompt",
	"build_domain_prompt_analysis_payload",
	"validate_decomposition_admissibility",
	"validate_domain_complete_coverage",
	"validate_minimal_library",
	"validate_signature_conformance",
	"validate_typed_structural_soundness",
]


def __getattr__(name: str) -> Any:
	if name == "HTNMethodSynthesizer":
		from .synthesis.synthesizer import HTNMethodSynthesizer

		return HTNMethodSynthesizer
	if name in {
		"build_domain_htn_system_prompt",
		"build_domain_htn_user_prompt",
		"build_domain_prompt_analysis_payload",
	}:
		from .synthesis.domain_prompts import (
			build_domain_htn_system_prompt,
			build_domain_htn_user_prompt,
			build_domain_prompt_analysis_payload,
		)

		return {
			"build_domain_htn_system_prompt": build_domain_htn_system_prompt,
			"build_domain_htn_user_prompt": build_domain_htn_user_prompt,
			"build_domain_prompt_analysis_payload": build_domain_prompt_analysis_payload,
		}[name]
	if name == "MethodLibraryValidator":
		from .validation.validator import MethodLibraryValidator

		return MethodLibraryValidator
	if name in {
		"validate_decomposition_admissibility",
		"validate_domain_complete_coverage",
		"validate_minimal_library",
		"validate_signature_conformance",
		"validate_typed_structural_soundness",
	}:
		from .validation.minimal_validation import (
			validate_decomposition_admissibility,
			validate_domain_complete_coverage,
			validate_minimal_library,
			validate_signature_conformance,
			validate_typed_structural_soundness,
		)

		return {
			"validate_decomposition_admissibility": validate_decomposition_admissibility,
			"validate_domain_complete_coverage": validate_domain_complete_coverage,
			"validate_minimal_library": validate_minimal_library,
			"validate_signature_conformance": validate_signature_conformance,
			"validate_typed_structural_soundness": validate_typed_structural_soundness,
		}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
