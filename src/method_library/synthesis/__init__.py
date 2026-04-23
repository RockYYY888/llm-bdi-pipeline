"""
Stage-owned synthesis exports for method-library construction.

Package import must stay lightweight because spawned HTN evaluation workers may
only need schema types; eager imports here can create circular dependencies.
"""

from __future__ import annotations

from typing import Any

from .schema import (
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
	"build_domain_htn_system_prompt",
	"build_domain_htn_user_prompt",
	"build_domain_prompt_analysis_payload",
	"render_generated_domain_text",
	"strip_methods_from_domain_text",
	"write_generated_domain_file",
	"write_masked_domain_file",
]


def __getattr__(name: str) -> Any:
	if name == "HTNMethodSynthesizer":
		from .synthesizer import HTNMethodSynthesizer

		return HTNMethodSynthesizer
	if name in {
		"build_domain_htn_system_prompt",
		"build_domain_htn_user_prompt",
		"build_domain_prompt_analysis_payload",
	}:
		from .domain_prompts import (
			build_domain_htn_system_prompt,
			build_domain_htn_user_prompt,
			build_domain_prompt_analysis_payload,
		)

		return {
			"build_domain_htn_system_prompt": build_domain_htn_system_prompt,
			"build_domain_htn_user_prompt": build_domain_htn_user_prompt,
			"build_domain_prompt_analysis_payload": build_domain_prompt_analysis_payload,
		}[name]
	if name in {
		"render_generated_domain_text",
		"strip_methods_from_domain_text",
		"write_generated_domain_file",
		"write_masked_domain_file",
	}:
		from domain_model.materialization import (
			render_generated_domain_text,
			strip_methods_from_domain_text,
			write_generated_domain_file,
			write_masked_domain_file,
		)

		return {
			"render_generated_domain_text": render_generated_domain_text,
			"strip_methods_from_domain_text": strip_methods_from_domain_text,
			"write_generated_domain_file": write_generated_domain_file,
			"write_masked_domain_file": write_masked_domain_file,
		}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
