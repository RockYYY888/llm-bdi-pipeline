"""
Plan-library generation pipeline aligned with Chapter 4.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from domain_model import infer_query_domain, load_query_sequence_records
from method_library.context import MethodLibrarySynthesisContext
from execution_logging.execution_logger import ExecutionLogger
from plan_library.orchestrator import PlanLibraryGenerationOrchestrator

from .artifacts import (
	PlanLibraryArtifactBundle,
	persist_plan_library_artifact_bundle,
)
from .rendering import render_plan_library_asl
from .translation import build_plan_library
from .validation import build_library_validation_record


class PlanLibraryGenerationPipeline:
	"""Generate method library M and AgentSpeak(L) plan library S from Chapter 4 inputs."""

	def __init__(
		self,
		*,
		domain_file: str,
		query_dataset: str | None = None,
		query_domain: str | None = None,
	) -> None:
		self._context = MethodLibrarySynthesisContext(domain_file=domain_file)
		self._orchestrator = PlanLibraryGenerationOrchestrator(self._context)
		self._query_dataset = query_dataset
		self._query_domain = query_domain

	@property
	def context(self) -> MethodLibrarySynthesisContext:
		return self._context

	@property
	def logger(self) -> ExecutionLogger:
		return self._context.logger

	@logger.setter
	def logger(self, value: ExecutionLogger) -> None:
		self._context.logger = value

	def build_library_bundle(
		self,
		*,
		output_root: Optional[str] = None,
	) -> Dict[str, Any]:
		query_sequence, temporal_specifications = load_query_sequence_records(
			domain_file=self._context.domain_file,
			dataset_path=self._query_dataset,
			query_domain=self._query_domain,
		)
		query_domain = infer_query_domain(
			domain_file=self._context.domain_file,
			explicit_domain=self._query_domain,
		)
		self._context.logger.start_pipeline(
			f"Generate AgentSpeak(L) plan library for {self._context.domain.name}",
			mode="plan_library_generation",
			domain_file=self._context.domain_file,
			problem_file=None,
			domain_name=self._context.domain.name,
			problem_name=None,
			output_dir="artifacts/runs",
		)
		self._context.output_dir = self._context.logger.current_log_dir
		if self._context.logger.current_record is not None and self._context.output_dir is not None:
			self._context.logger.current_record.output_dir = str(self._context.output_dir)
			self._context.logger._save_current_state()

		try:
			masked_domain_inputs = self._orchestrator.prepare_masked_domain_build_inputs()
			method_library, method_synthesis_metadata = self._orchestrator.synthesise_domain_methods(
				synthesis_domain=masked_domain_inputs["masked_domain"],
				source_domain_kind="masked_official",
				masked_domain_file=str(masked_domain_inputs["masked_domain_file"]),
				original_method_count=int(masked_domain_inputs["original_method_count"]),
				query_sequence=tuple(record.to_dict() for record in query_sequence),
				temporal_specifications=tuple(
					record.to_dict() for record in temporal_specifications
				),
			)
			if method_library is None:
				raise RuntimeError("Method-library synthesis failed.")

			translation_start = time.perf_counter()
			plan_library, translation_coverage = build_plan_library(
				domain=self._context.domain,
				method_library=method_library,
			)
			self._context._record_step_timing(
				"plan_library_translation",
				translation_start,
				metadata={
					"methods_considered": translation_coverage.methods_considered,
					"plans_generated": translation_coverage.plans_generated,
				},
			)
			render_start = time.perf_counter()
			plan_library_asl = render_plan_library_asl(plan_library)
			self._context._record_step_timing(
				"plan_library_rendering",
				render_start,
				metadata={"plan_count": len(plan_library.plans)},
			)

			method_validation = self._orchestrator.validate_method_library(method_library)
			library_validation = build_library_validation_record(
				domain_name=self._context.domain.name,
				domain=self._context.domain,
				method_library=method_library,
				plan_library=plan_library,
				translation_coverage=translation_coverage,
				method_validation=method_validation,
			)
			if not library_validation.passed:
				raise RuntimeError(library_validation.failure_reason or "Library validation failed.")

			artifact_root = (
				Path(output_root).expanduser().resolve()
				if output_root is not None
				else (
					self._context.project_root
					/ "artifacts"
					/ "plan_library"
					/ query_domain
				)
			)
			bundle = PlanLibraryArtifactBundle(
				domain_name=self._context.domain.name,
				query_sequence=query_sequence,
				temporal_specifications=temporal_specifications,
				method_library=method_library,
				plan_library=plan_library,
				translation_coverage=translation_coverage,
				library_validation=library_validation,
				method_synthesis_metadata=dict(method_synthesis_metadata or {}),
				artifact_root=str(artifact_root),
				masked_domain_file=str(masked_domain_inputs["masked_domain_file"]),
				plan_library_asl_file=str(artifact_root / "plan_library.asl"),
			)
			artifact_paths = persist_plan_library_artifact_bundle(
				artifact_root=artifact_root,
				artifact=bundle,
				masked_domain_text=str(masked_domain_inputs["masked_domain_text"]),
				plan_library_asl_text=plan_library_asl,
			)
			log_path = self._context.logger.end_pipeline(success=True)
			return {
				"success": True,
				"artifact": bundle.to_dict(),
				"artifact_paths": artifact_paths,
				"log_path": str(log_path),
			}
		except Exception as exc:
			log_path = self._context.logger.end_pipeline(success=False)
			return {
				"success": False,
				"error": str(exc),
				"log_path": str(log_path),
			}
