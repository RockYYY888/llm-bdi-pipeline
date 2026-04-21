"""
Offline domain-synthesis orchestration.

This module owns the domain-only masked-method generation workflow. It does not
solve benchmark problem instances and does not participate in the online Jason
runtime.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from offline_method_generation.artifacts import (
	DomainLibraryArtifact,
	persist_domain_library_artifact,
)
from offline_method_generation.context import OfflineSynthesisContext
from offline_method_generation.domain_gate.validator import OfflineDomainGateValidator
from offline_method_generation.method_synthesis.domain_materialization import (
	write_generated_domain_file,
	write_masked_domain_file,
)
from offline_method_generation.method_synthesis.minimal_validation import (
	validate_typed_structural_soundness,
)
from offline_method_generation.method_synthesis.naming import sanitize_identifier
from offline_method_generation.method_synthesis.synthesizer import HTNMethodSynthesizer


class OfflineDomainSynthesisOrchestrator:
	"""Build one reusable generated domain artifact from one official domain file."""

	def __init__(self, pipeline_context: OfflineSynthesisContext) -> None:
		self.context = pipeline_context
		self.domain_gate_validator = OfflineDomainGateValidator(pipeline_context)

	def build_domain_library(
		self,
		*,
		output_root: Optional[str] = None,
	) -> Dict[str, Any]:
		self.context.logger.start_pipeline(
			f"Build domain-complete HTN library for {self.context.domain.name}",
			mode="offline_method_generation",
			domain_file=self.context.domain_file,
			problem_file=None,
			domain_name=self.context.domain.name,
			problem_name=None,
			output_dir="artifacts/runs",
		)
		self.context.output_dir = self.context.logger.current_log_dir
		if self.context.logger.current_record is not None and self.context.output_dir is not None:
			self.context.logger.current_record.output_dir = str(self.context.output_dir)
			self.context.logger._save_current_state()

		domain_build = self.build_generated_domain_library_artifact(
			output_root=output_root,
		)
		if not domain_build.get("success", False):
			log_filepath = self.context.logger.end_pipeline(success=False)
			return {
				"success": False,
				"step": str(domain_build.get("step") or "domain_build"),
				"error": str(domain_build.get("error") or "Domain build failed"),
				"log_path": str(log_filepath),
			}
		log_filepath = self.context.logger.end_pipeline(success=True)
		return {
			"success": True,
			"domain_name": self.context.domain.name,
			"artifact": domain_build["artifact"].to_dict(),
			"artifact_paths": domain_build["artifact_paths"],
			"log_path": str(log_filepath),
		}

	def build_generated_domain_library_artifact(
		self,
		*,
		output_root: Optional[str] = None,
	) -> Dict[str, Any]:
		masked_domain_inputs = self.prepare_masked_domain_build_inputs()
		synthesis_domain = masked_domain_inputs["masked_domain"]
		method_library, method_synthesis_data = self.synthesise_domain_methods(
			synthesis_domain=synthesis_domain,
			source_domain_kind="masked_official",
			masked_domain_file=str(masked_domain_inputs["masked_domain_file"]),
			original_method_count=int(masked_domain_inputs["original_method_count"]),
		)
		if not method_library:
			return {
				"success": False,
				"step": "method_synthesis",
				"error": "Domain HTN synthesis failed",
			}

		generated_domain_outputs = self.materialize_generated_domain_build_output(
			masked_domain_text=str(masked_domain_inputs["masked_domain_text"]),
			method_library=method_library,
		)
		domain_gate_data = self.validate_domain_library(
			method_library,
			generated_domain_file=str(generated_domain_outputs["generated_domain_file"]),
		)
		if domain_gate_data is None:
			return {
				"success": False,
				"step": "domain_gate",
				"error": "Domain gate failed",
			}

		persisted_artifact = DomainLibraryArtifact(
			domain_name=self.context.domain.name,
			method_library=method_library,
			method_synthesis_metadata=method_synthesis_data,
			domain_gate=domain_gate_data,
			source_domain_kind="masked_official",
		)
		artifact_paths = self.persist_domain_library_artifact(
			persisted_artifact,
			output_root=output_root,
			masked_domain_text=str(masked_domain_inputs["masked_domain_text"]),
			generated_domain_text=str(generated_domain_outputs["generated_domain_text"]),
		)
		artifact = DomainLibraryArtifact(
			domain_name=self.context.domain.name,
			method_library=method_library,
			method_synthesis_metadata=method_synthesis_data,
			domain_gate=domain_gate_data,
			source_domain_kind="masked_official",
			artifact_root=str(Path(artifact_paths["method_library"]).resolve().parent),
			masked_domain_file=(
				str(artifact_paths.get("masked_domain"))
				if artifact_paths.get("masked_domain") is not None
				else None
			),
			generated_domain_file=(
				str(artifact_paths.get("generated_domain"))
				if artifact_paths.get("generated_domain") is not None
				else None
			),
		)
		return {
			"success": True,
			"artifact": artifact,
			"artifact_paths": artifact_paths,
			"masked_domain": masked_domain_inputs,
			"generated_domain": generated_domain_outputs,
		}

	def stable_domain_artifact_root(self, output_root: Optional[str] = None) -> Path:
		if output_root:
			return Path(output_root).expanduser().resolve()
		if getattr(self.context.logger, "run_origin", "") == "tests" and self.context.output_dir is not None:
			return (
				Path(self.context.output_dir)
				/ "domain_build_artifacts"
				/ sanitize_identifier(self.context.domain.name)
			)
		return (
			self.context.project_root
			/ "artifacts"
			/ "domain_builds"
			/ sanitize_identifier(self.context.domain.name)
		)

	def persist_domain_library_artifact(
		self,
		artifact: DomainLibraryArtifact,
		*,
		output_root: Optional[str] = None,
		masked_domain_text: Optional[str] = None,
		generated_domain_text: Optional[str] = None,
	) -> Dict[str, str]:
		return persist_domain_library_artifact(
			artifact_root=self.stable_domain_artifact_root(output_root),
			artifact=artifact,
			masked_domain_text=masked_domain_text,
			generated_domain_text=generated_domain_text,
		)

	def prepare_masked_domain_build_inputs(self) -> Dict[str, Any]:
		if self.context.output_dir is None:
			raise ValueError("Masked domain preparation requires an active output directory.")
		masked_domain_path = Path(self.context.output_dir) / "masked_domain.hddl"
		return write_masked_domain_file(
			official_domain_file=self.context.domain_file,
			output_path=masked_domain_path,
		)

	def materialize_generated_domain_build_output(
		self,
		*,
		masked_domain_text: str,
		method_library,
	) -> Dict[str, Any]:
		if self.context.output_dir is None:
			raise ValueError("Generated domain materialization requires an active output directory.")
		generated_domain_path = Path(self.context.output_dir) / "generated_domain.hddl"
		return write_generated_domain_file(
			masked_domain_text=masked_domain_text,
			domain=self.context.domain,
			method_library=method_library,
			output_path=generated_domain_path,
		)

	def synthesise_domain_methods(
		self,
		*,
		synthesis_domain: Optional[Any] = None,
		source_domain_kind: str = "official",
		masked_domain_file: Optional[str] = None,
		original_method_count: Optional[int] = None,
	):
		print("\n[METHOD SYNTHESIS]")
		print("-"*80)
		stage_start = time.perf_counter()
		domain_for_synthesis = synthesis_domain or self.context.domain
		method_library = None
		synthesis_meta: Dict[str, Any] = {}

		try:
			synthesizer = HTNMethodSynthesizer(
				api_key=self.context.config.offline_openai_api_key,
				model=self.context.config.method_synthesis_model,
				base_url=self.context.config.openai_base_url,
				timeout=float(self.context.config.method_synthesis_timeout),
				max_tokens=int(self.context.config.method_synthesis_max_tokens),
			)
			synthesis_start = time.perf_counter()
			method_library, synthesis_meta = synthesizer.synthesize_domain_complete(
				domain=domain_for_synthesis,
			)
			synthesis_seconds = time.perf_counter() - synthesis_start
			validation_start = time.perf_counter()
			validate_typed_structural_soundness(domain_for_synthesis, method_library)
			typing_validation_seconds = time.perf_counter() - validation_start
			summary = {
				"used_llm": synthesis_meta["used_llm"],
				"llm_attempted": synthesis_meta["llm_prompt"] is not None,
				"llm_finish_reason": synthesis_meta.get("llm_finish_reason"),
				"llm_request_id": synthesis_meta.get("llm_request_id"),
				"llm_request_profile": synthesis_meta.get("llm_request_profile"),
				"llm_response_mode": synthesis_meta.get("llm_response_mode"),
				"llm_first_chunk_seconds": synthesis_meta.get("llm_first_chunk_seconds"),
				"llm_first_chunk_timeout_seconds": synthesis_meta.get(
					"llm_first_chunk_timeout_seconds",
				),
				"llm_complete_json_seconds": synthesis_meta.get("llm_complete_json_seconds"),
				"llm_reasoning_preview": synthesis_meta.get("llm_reasoning_preview"),
				"llm_reasoning_characters": synthesis_meta.get("llm_reasoning_characters"),
				"llm_attempts": synthesis_meta.get("llm_attempts"),
				"llm_response_time_seconds": synthesis_meta.get("llm_response_time_seconds"),
				"llm_attempt_durations_seconds": synthesis_meta.get(
					"llm_attempt_durations_seconds",
				),
				"domain_task_contracts": synthesis_meta.get("domain_task_contracts", []),
				"action_analysis": synthesis_meta.get("action_analysis", {}),
				"derived_analysis": synthesis_meta.get("derived_analysis", {}),
				"prompt_strategy": synthesis_meta.get("prompt_strategy"),
				"prompt_declared_task_count": synthesis_meta.get("prompt_declared_task_count"),
				"prompt_domain_task_contract_count": synthesis_meta.get(
					"prompt_domain_task_contract_count",
				),
				"prompt_reusable_dynamic_resource_count": synthesis_meta.get(
					"prompt_reusable_dynamic_resource_count",
				),
				"llm_request_count": synthesis_meta.get("llm_request_count"),
				"failure_class": synthesis_meta.get("failure_class"),
				"declared_compound_tasks": synthesis_meta.get("declared_compound_tasks", []),
				"compound_tasks": synthesis_meta["compound_tasks"],
				"primitive_tasks": synthesis_meta["primitive_tasks"],
				"methods": synthesis_meta["methods"],
				"model": synthesis_meta.get("model"),
				"source_domain_kind": source_domain_kind,
				"masked_domain_file": masked_domain_file,
				"original_method_count": original_method_count,
				"synthesis_domain_name": getattr(domain_for_synthesis, "name", self.context.domain.name),
			}
			self.context._latest_transition_prompt_analysis = {}
			self.context._record_step_timing(
				"method_synthesis",
				stage_start,
				breakdown={
					"synthesis_seconds": synthesis_seconds,
					"typing_validation_seconds": typing_validation_seconds,
					"llm_response_seconds": synthesis_meta.get("llm_response_time_seconds"),
				},
				metadata={
					"used_llm": synthesis_meta.get("used_llm"),
					"llm_attempted": synthesis_meta.get("llm_prompt") is not None,
					"domain_complete": True,
					"source_domain_kind": source_domain_kind,
				},
			)
			self.context.logger.log_method_synthesis(
				method_library.to_dict(),
				"Success",
				model=synthesis_meta["model"] if synthesis_meta["llm_prompt"] is not None else None,
				llm_prompt=synthesis_meta["llm_prompt"],
				llm_response=synthesis_meta["llm_response"],
				metadata=summary,
			)

			print("✓ Domain-complete HTN method synthesis complete")
			print(f"  Attempted LLM synthesis: {summary['llm_attempted']}")
			print(f"  Accepted LLM output: {summary['used_llm']}")
			print(f"  Declared compound tasks: {len(summary['declared_compound_tasks'])}")
			print(f"  Synthesised compound tasks: {summary['compound_tasks']}")
			print(f"  Primitive tasks: {summary['primitive_tasks']}")
			print(f"  Methods: {summary['methods']}")
			return method_library, summary
		except Exception as exc:
			self.context._latest_transition_specs = ()
			self.context._latest_transition_prompt_analysis = {}
			self.context._record_step_timing("method_synthesis", stage_start)
			error_metadata = dict(getattr(exc, "metadata", {}) or {})
			self.context.logger.log_method_synthesis(
				None,
				"Failed",
				error=str(exc),
				model=error_metadata.get("model"),
				llm_prompt=error_metadata.get("llm_prompt"),
				llm_response=error_metadata.get("llm_response"),
				metadata=error_metadata or None,
			)
			print(f"✗ Method synthesis failed: {exc}")
			return None, {}

	def validate_domain_library(self, method_library, *, generated_domain_file: Optional[str] = None):
		return self.domain_gate_validator.validate(
			method_library,
			generated_domain_file=generated_domain_file,
		)

	def validate_domain_library_cases(self, method_library):
		return self.domain_gate_validator.build_cases(method_library)
