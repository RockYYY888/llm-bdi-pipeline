"""
Plan-library generation pipeline for the dissertation workflow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from domain_model import infer_query_domain, load_query_sequence_records
from domain_model.materialization import write_generated_domain_file
from method_library.context import MethodLibrarySynthesisContext
from execution_logging.execution_logger import ExecutionLogger
from plan_library.orchestrator import PlanLibraryGenerationOrchestrator

from .artifacts import (
	PlanLibraryArtifactBundle,
	persist_plan_library_artifact_bundle,
)
from .models import TranslationCoverage
from .rendering import render_plan_library_asl
from .set_semantics import deduplicate_plan_library
from .translation import build_plan_library
from .validation import build_library_validation_record


class PlanLibraryGenerationPipeline:
	"""Generate method library M and AgentSpeak(L) plan library S from workflow inputs."""

	def __init__(
		self,
		*,
		domain_file: str,
		query_dataset: str | None = None,
		query_domain: str | None = None,
		query_ids: Sequence[str] | None = None,
	) -> None:
		self._context = MethodLibrarySynthesisContext(domain_file=domain_file)
		self._orchestrator = PlanLibraryGenerationOrchestrator(self._context)
		self._query_dataset = query_dataset
		self._query_domain = query_domain
		self._query_ids = tuple(
			query_id_text
			for query_id_text in (str(query_id or "").strip() for query_id in (query_ids or ()))
			if query_id_text
		)

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
			query_ids=self._query_ids,
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
			set_result = deduplicate_plan_library(plan_library)
			plan_library = set_result.plan_library
			if set_result.removed_duplicate_plans:
				translation_coverage = TranslationCoverage(
					domain_name=translation_coverage.domain_name,
					methods_considered=translation_coverage.methods_considered,
					plans_generated=len(plan_library.plans),
					accepted_translation=translation_coverage.accepted_translation,
					unsupported_buckets=dict(translation_coverage.unsupported_buckets),
					unsupported_methods=tuple(translation_coverage.unsupported_methods),
				)
			method_synthesis_metadata = dict(method_synthesis_metadata or {})
			method_synthesis_metadata["plan_set_normalisation"] = set_result.to_dict()
			self._context._record_step_timing(
				"plan_library_translation",
				translation_start,
				metadata={
					"methods_considered": translation_coverage.methods_considered,
					"plans_generated": translation_coverage.plans_generated,
					"removed_duplicate_plans": set_result.removed_duplicate_plans,
					"renamed_plans": set_result.renamed_plans,
				},
			)
			render_start = time.perf_counter()
			plan_library_asl = render_plan_library_asl(plan_library)
			self._context._record_step_timing(
				"plan_library_rendering",
				render_start,
				metadata={"plan_count": len(plan_library.plans)},
			)
			self._context.logger.log_agentspeak_rendering(
				None,
				"Success",
				metadata={
					"plan_count": len(plan_library.plans),
					"rendered_asl_bytes": len(plan_library_asl.encode("utf-8")),
				},
			)

			artifact_root = (
				Path(output_root).expanduser().resolve()
				if output_root is not None
				else self._default_artifact_root(query_domain)
			)
			generated_domain_outputs = write_generated_domain_file(
				masked_domain_text=str(masked_domain_inputs["masked_domain_text"]),
				domain=masked_domain_inputs["masked_domain"],
				method_library=method_library,
				output_path=artifact_root / "generated_domain.hddl",
			)
			method_validation = self._orchestrator.validate_method_library(
				method_library,
				materialized_domain_file=str(generated_domain_outputs["generated_domain_file"]),
			)
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

			bundle = PlanLibraryArtifactBundle(
				domain_name=self._context.domain.name,
				query_sequence=query_sequence,
				temporal_specifications=temporal_specifications,
				method_library=method_library,
				plan_library=plan_library,
				translation_coverage=translation_coverage,
				library_validation=library_validation,
				method_synthesis_metadata=method_synthesis_metadata,
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
			artifact_paths["generated_domain"] = str(
				generated_domain_outputs["generated_domain_file"],
			)
			self._context.logger.update_step_artifacts(
				"method_synthesis",
				{
					"method_library_file": artifact_paths["method_library"],
					"method_synthesis_metadata_file": artifact_paths[
						"method_synthesis_metadata"
					],
					"generated_domain_file": artifact_paths["generated_domain"],
				},
			)
			self._context.logger.update_step_artifacts(
				"agentspeak_rendering",
				{
					"plan_library_file": artifact_paths["plan_library"],
					"plan_library_asl_file": artifact_paths["plan_library_asl"],
					"translation_coverage_file": artifact_paths["translation_coverage"],
					"library_validation_file": artifact_paths["library_validation"],
				},
			)
			artifact_summary = _artifact_summary(bundle, artifact_paths)
			log_path = self._context.logger.end_pipeline(success=True)
			return {
				"success": True,
				"artifact": artifact_summary,
				"artifact_summary": artifact_summary,
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

	def _default_artifact_root(self, query_domain: str) -> Path:
		artifact_root = (
			self._context.project_root
			/ "artifacts"
			/ "plan_library"
			/ query_domain
		)
		if self._query_ids:
			return artifact_root / self._query_selection_artifact_slug()
		return artifact_root

	def _query_selection_artifact_slug(self) -> str:
		if not self._query_ids:
			return ""
		if len(self._query_ids) == 1:
			return _artifact_slug(self._query_ids[0])
		selection_key = "_".join(_artifact_slug(query_id) for query_id in self._query_ids[:5])
		if len(self._query_ids) > 5:
			selection_key = f"{selection_key}_plus_{len(self._query_ids) - 5}"
		return selection_key or "selected_queries"


def _artifact_slug(value: str) -> str:
	text = str(value or "").strip().lower()
	buffer = [character if character.isalnum() else "_" for character in text]
	slug = "".join(buffer).strip("_")
	while "__" in slug:
		slug = slug.replace("__", "_")
	return slug or "query"


def _artifact_summary(
	bundle: PlanLibraryArtifactBundle,
	artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
	return {
		"domain_name": bundle.domain_name,
		"query_count": len(tuple(bundle.query_sequence or ())),
		"temporal_specification_count": len(tuple(bundle.temporal_specifications or ())),
		"compound_task_count": len(tuple(bundle.method_library.compound_tasks or ())),
		"primitive_task_count": len(tuple(bundle.method_library.primitive_tasks or ())),
		"method_count": len(tuple(bundle.method_library.methods or ())),
		"plan_count": len(tuple(bundle.plan_library.plans or ())),
		"translation_coverage": bundle.translation_coverage.to_dict(),
		"library_validation": bundle.library_validation.to_dict(),
		"artifact_root": bundle.artifact_root,
		"method_library_file": artifact_paths.get("method_library"),
		"generated_domain_file": artifact_paths.get("generated_domain"),
		"plan_library_file": artifact_paths.get("plan_library"),
		"plan_library_asl_file": artifact_paths.get("plan_library_asl"),
		"method_synthesis_metadata_file": artifact_paths.get("method_synthesis_metadata"),
	}
