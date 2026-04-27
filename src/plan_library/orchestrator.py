"""
Method-library synthesis orchestration for the plan-library workflow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from domain_model.materialization import write_masked_domain_file
from method_library.context import MethodLibrarySynthesisContext
from method_library.validation.minimal_validation import validate_typed_structural_soundness
from method_library.synthesis.synthesizer import HTNMethodSynthesizer
from method_library.validation import MethodLibraryValidator


class PlanLibraryGenerationOrchestrator:
	"""Own the generation steps up to method-library validation."""

	def __init__(self, pipeline_context: MethodLibrarySynthesisContext) -> None:
		self.context = pipeline_context
		self.method_library_validator = MethodLibraryValidator(pipeline_context)

	def prepare_masked_domain_build_inputs(self) -> Dict[str, Any]:
		if self.context.output_dir is None:
			raise ValueError("Masked domain preparation requires an active output directory.")
		masked_domain_path = Path(self.context.output_dir) / "masked_domain.hddl"
		return write_masked_domain_file(
			official_domain_file=self.context.domain_file,
			output_path=masked_domain_path,
		)

	def synthesise_domain_methods(
		self,
		*,
		synthesis_domain: Optional[Any] = None,
		source_domain_kind: str = "official",
		masked_domain_file: Optional[str] = None,
		original_method_count: Optional[int] = None,
		query_sequence: Optional[Tuple[Dict[str, Any], ...]] = None,
		temporal_specifications: Optional[Tuple[Dict[str, Any], ...]] = None,
	):
		print("\n[METHOD SYNTHESIS]")
		print("-" * 80)
		stage_start = time.perf_counter()
		domain_for_synthesis = synthesis_domain or self.context.domain

		try:
			synthesizer = HTNMethodSynthesizer(
				api_key=self.context.config.method_synthesis_api_key,
				model=self.context.config.method_synthesis_model,
				base_url=self.context.config.method_synthesis_base_url,
				timeout=float(self.context.config.method_synthesis_timeout),
				max_tokens=int(self.context.config.method_synthesis_max_tokens),
				session_id=self.context.config.method_synthesis_session_id,
			)
			synthesis_start = time.perf_counter()
			method_library, synthesis_meta = synthesizer.synthesize_domain_complete(
				domain=domain_for_synthesis,
				query_sequence=query_sequence,
				temporal_specifications=temporal_specifications,
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
				"llm_request_max_tokens": synthesis_meta.get("llm_request_max_tokens"),
				"llm_completion_max_tokens": synthesis_meta.get("llm_completion_max_tokens"),
				"llm_thinking_type": synthesis_meta.get("llm_thinking_type"),
				"llm_reasoning_effort": synthesis_meta.get("llm_reasoning_effort"),
				"llm_response_mode": synthesis_meta.get("llm_response_mode"),
				"llm_stream_handshake_seconds": synthesis_meta.get("llm_stream_handshake_seconds"),
				"llm_first_stream_chunk_seconds": synthesis_meta.get("llm_first_stream_chunk_seconds"),
				"llm_first_chunk_seconds": synthesis_meta.get("llm_first_chunk_seconds"),
				"llm_first_content_chunk_seconds": synthesis_meta.get(
					"llm_first_content_chunk_seconds",
				),
				"llm_first_reasoning_chunk_seconds": synthesis_meta.get(
					"llm_first_reasoning_chunk_seconds",
				),
				"llm_first_chunk_timeout_seconds": synthesis_meta.get(
					"llm_first_chunk_timeout_seconds",
				),
				"llm_complete_json_seconds": synthesis_meta.get("llm_complete_json_seconds"),
				"llm_reasoning_chunks_ignored": synthesis_meta.get("llm_reasoning_chunks_ignored"),
				"llm_reasoning_excluded": synthesis_meta.get("llm_reasoning_excluded"),
				"llm_max_tokens_policy": synthesis_meta.get("llm_max_tokens_policy"),
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
				"query_sequence_count": len(tuple(query_sequence or ())),
				"temporal_specification_count": len(tuple(temporal_specifications or ())),
			}
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
					"source_domain_kind": source_domain_kind,
				},
			)
			self.context.logger.log_method_synthesis(
				None,
				"Success",
				model=synthesis_meta["model"] if synthesis_meta["llm_prompt"] is not None else None,
				llm_prompt=synthesis_meta["llm_prompt"],
				llm_response=synthesis_meta["llm_response"],
				metadata=summary,
			)
			print("✓ Method-library synthesis complete")
			print(f"  Attempted LLM synthesis: {summary['llm_attempted']}")
			print(f"  Accepted LLM output: {summary['used_llm']}")
			print(f"  Declared compound tasks: {len(summary['declared_compound_tasks'])}")
			print(f"  Synthesised compound tasks: {summary['compound_tasks']}")
			print(f"  Primitive tasks: {summary['primitive_tasks']}")
			print(f"  Methods: {summary['methods']}")
			return method_library, summary
		except Exception as exc:
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

	def validate_method_library(
		self,
		method_library,
		*,
		materialized_domain_file: Optional[str] = None,
	):
		return self.method_library_validator.validate(
			method_library,
			generated_domain_file=materialized_domain_file,
		)
