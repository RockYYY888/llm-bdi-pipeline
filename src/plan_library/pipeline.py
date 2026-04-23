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
from .models import LibraryValidationRecord
from .rendering import render_plan_library_asl
from .translation import build_plan_library


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
			library_validation = _build_library_validation_record(
				domain_name=self._context.domain.name,
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


def _build_library_validation_record(
	*,
	domain_name: str,
	method_library,
	plan_library,
	translation_coverage,
	method_validation,
) -> LibraryValidationRecord:
	layer_results = dict((method_validation or {}).get("layers") or {})
	checked_layers = {
		"signature_conformance": bool(
			((layer_results.get("signature_conformance") or {}).get("passed"))
		),
		"typed_structure": bool(
			((layer_results.get("typed_structural_soundness") or {}).get("passed"))
		),
		"body_symbol_validity": bool(
			((layer_results.get("decomposition_admissibility") or {}).get("passed"))
		),
		"groundability_precheck": bool(method_validation is not None),
	}
	warnings = []
	for layer_name in (
		"signature_conformance",
		"typed_structural_soundness",
		"decomposition_admissibility",
		"materialized_parseability",
	):
		for warning in ((layer_results.get(layer_name) or {}).get("warnings") or ()):
			warning_text = str(warning).strip()
			if warning_text:
				warnings.append(warning_text)
	if translation_coverage.unsupported_buckets:
		warnings.append(
			"Unsupported method constructs were excluded from the generated plan library: "
			+ ", ".join(
				f"{bucket}={count}"
				for bucket, count in sorted(translation_coverage.unsupported_buckets.items())
			),
		)
	failure_reason = None
	if not all(checked_layers.values()):
		for layer_name, passed in checked_layers.items():
			if passed:
				continue
			failure_reason = f"{layer_name} failed"
			break
	elif translation_coverage.accepted_translation <= 0:
		failure_reason = "No HTN methods were accepted by the AgentSpeak(L) translation layer."
	return LibraryValidationRecord(
		library_id=domain_name,
		passed=all(checked_layers.values()) and translation_coverage.accepted_translation > 0,
		method_count=len(method_library.methods),
		plan_count=len(plan_library.plans),
		checked_layers=checked_layers,
		warnings=tuple(dict.fromkeys(warnings)),
		failure_reason=failure_reason,
	)
