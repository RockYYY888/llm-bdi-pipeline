"""
Artifact persistence for generated plan-library bundles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from method_library.synthesis.schema import HTNMethodLibrary
from temporal_specification import QueryInstructionRecord, TemporalSpecificationRecord

from .models import LibraryValidationRecord, PlanLibrary, TranslationCoverage


@dataclass(frozen=True)
class PlanLibraryArtifactBundle:
	"""Persisted artifact bundle for one generated plan library."""

	domain_name: str
	query_sequence: Sequence[QueryInstructionRecord]
	temporal_specifications: Sequence[TemporalSpecificationRecord]
	method_library: HTNMethodLibrary
	plan_library: PlanLibrary
	translation_coverage: TranslationCoverage
	library_validation: LibraryValidationRecord
	method_synthesis_metadata: Dict[str, Any]
	artifact_root: str | None = None
	masked_domain_file: str | None = None
	plan_library_asl_file: str | None = None

	def to_dict(self) -> Dict[str, Any]:
		return {
			"domain_name": self.domain_name,
			"query_sequence": [record.to_dict() for record in self.query_sequence],
			"temporal_specifications": [record.to_dict() for record in self.temporal_specifications],
			"method_library": self.method_library.to_dict(),
			"plan_library": self.plan_library.to_dict(),
			"translation_coverage": self.translation_coverage.to_dict(),
			"library_validation": self.library_validation.to_dict(),
			"method_synthesis_metadata": dict(self.method_synthesis_metadata),
			"artifact_root": self.artifact_root,
			"masked_domain_file": self.masked_domain_file,
			"plan_library_asl_file": self.plan_library_asl_file,
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "PlanLibraryArtifactBundle":
		return cls(
			domain_name=str(payload.get("domain_name") or "").strip(),
			query_sequence=tuple(
				QueryInstructionRecord.from_dict(item)
				for item in (payload.get("query_sequence") or ())
				if isinstance(item, dict)
			),
			temporal_specifications=tuple(
				TemporalSpecificationRecord.from_dict(item)
				for item in (payload.get("temporal_specifications") or ())
				if isinstance(item, dict)
			),
			method_library=HTNMethodLibrary.from_dict(dict(payload.get("method_library") or {})),
			plan_library=PlanLibrary.from_dict(dict(payload.get("plan_library") or {})),
			translation_coverage=TranslationCoverage.from_dict(
				dict(payload.get("translation_coverage") or {}),
			),
			library_validation=LibraryValidationRecord.from_dict(
				dict(payload.get("library_validation") or {}),
			),
			method_synthesis_metadata=dict(payload.get("method_synthesis_metadata") or {}),
			artifact_root=(
				str(payload.get("artifact_root")).strip()
				if payload.get("artifact_root") is not None
				else None
			),
			masked_domain_file=(
				str(payload.get("masked_domain_file")).strip()
				if payload.get("masked_domain_file") is not None
				else None
			),
			plan_library_asl_file=(
				str(payload.get("plan_library_asl_file")).strip()
				if payload.get("plan_library_asl_file") is not None
				else None
			),
		)

def persist_plan_library_artifact_bundle(
	*,
	artifact_root: str | Path,
	artifact: PlanLibraryArtifactBundle,
	masked_domain_text: str | None = None,
	plan_library_asl_text: str | None = None,
) -> Dict[str, str]:
	"""Persist a plan-library bundle under one stable artifact root."""

	root = Path(artifact_root).expanduser().resolve()
	root.mkdir(parents=True, exist_ok=True)

	artifact_metadata_path = root / "artifact_metadata.json"
	masked_domain_path = root / "masked_domain.hddl"
	query_sequence_path = root / "query_sequence.json"
	temporal_specifications_path = root / "temporal_specifications.json"
	method_library_path = root / "method_library.json"
	plan_library_path = root / "plan_library.json"
	plan_library_asl_path = root / "plan_library.asl"
	translation_coverage_path = root / "translation_coverage.json"
	library_validation_path = root / "library_validation.json"
	method_synthesis_metadata_path = root / "method_synthesis_metadata.json"

	artifact_metadata_path.write_text(
		json.dumps(
			{
				"domain_name": artifact.domain_name,
				"query_count": len(tuple(artifact.query_sequence or ())),
				"temporal_specification_count": len(tuple(artifact.temporal_specifications or ())),
			},
			indent=2,
		),
		encoding="utf-8",
	)
	query_sequence_path.write_text(
		json.dumps([record.to_dict() for record in artifact.query_sequence], indent=2),
		encoding="utf-8",
	)
	temporal_specifications_path.write_text(
		json.dumps([record.to_dict() for record in artifact.temporal_specifications], indent=2),
		encoding="utf-8",
	)
	method_library_path.write_text(
		json.dumps(artifact.method_library.to_dict(), indent=2),
		encoding="utf-8",
	)
	plan_library_path.write_text(
		json.dumps(artifact.plan_library.to_dict(), indent=2),
		encoding="utf-8",
	)
	translation_coverage_path.write_text(
		json.dumps(artifact.translation_coverage.to_dict(), indent=2),
		encoding="utf-8",
	)
	library_validation_path.write_text(
		json.dumps(artifact.library_validation.to_dict(), indent=2),
		encoding="utf-8",
	)
	method_synthesis_metadata_path.write_text(
		json.dumps(artifact.method_synthesis_metadata, indent=2),
		encoding="utf-8",
	)
	if masked_domain_text is not None:
		masked_domain_path.write_text(str(masked_domain_text), encoding="utf-8")
	if plan_library_asl_text is not None:
		plan_library_asl_path.write_text(str(plan_library_asl_text), encoding="utf-8")

	paths = {
		"artifact_metadata": str(artifact_metadata_path),
		"query_sequence": str(query_sequence_path),
		"temporal_specifications": str(temporal_specifications_path),
		"method_library": str(method_library_path),
		"plan_library": str(plan_library_path),
		"translation_coverage": str(translation_coverage_path),
		"library_validation": str(library_validation_path),
		"method_synthesis_metadata": str(method_synthesis_metadata_path),
	}
	if masked_domain_text is not None:
		paths["masked_domain"] = str(masked_domain_path)
	if plan_library_asl_text is not None:
		paths["plan_library_asl"] = str(plan_library_asl_path)
	return paths


def load_plan_library_artifact_bundle(
	library_artifact: str | Path | Dict[str, Any] | PlanLibraryArtifactBundle | HTNMethodLibrary,
) -> PlanLibraryArtifactBundle:
	"""Load a plan-library artifact bundle from disk or memory."""

	if isinstance(library_artifact, PlanLibraryArtifactBundle):
		return library_artifact
	if isinstance(library_artifact, HTNMethodLibrary):
		return PlanLibraryArtifactBundle(
			domain_name="",
			query_sequence=(),
			temporal_specifications=(),
			method_library=library_artifact,
			plan_library=PlanLibrary(domain_name="", plans=()),
			translation_coverage=TranslationCoverage(
				domain_name="",
				methods_considered=len(tuple(library_artifact.methods or ())),
				plans_generated=0,
				accepted_translation=0,
			),
			library_validation=LibraryValidationRecord(
				library_id="",
				passed=True,
				method_count=len(tuple(library_artifact.methods or ())),
				plan_count=0,
				checked_layers={},
			),
			method_synthesis_metadata={},
		)
	if isinstance(library_artifact, dict):
		if "method_library" not in library_artifact:
			return load_plan_library_artifact_bundle(
				HTNMethodLibrary.from_dict(dict(library_artifact)),
			)
		return PlanLibraryArtifactBundle.from_dict(library_artifact)

	artifact_root = Path(library_artifact).expanduser().resolve()
	if artifact_root.is_file():
		artifact_root = artifact_root.parent

	query_sequence_payload = json.loads((artifact_root / "query_sequence.json").read_text())
	temporal_specifications_payload = json.loads(
		(artifact_root / "temporal_specifications.json").read_text(),
	)
	method_library_payload = json.loads((artifact_root / "method_library.json").read_text())
	plan_library_payload = json.loads((artifact_root / "plan_library.json").read_text())
	translation_coverage_payload = json.loads((artifact_root / "translation_coverage.json").read_text())
	library_validation_payload = json.loads((artifact_root / "library_validation.json").read_text())
	metadata_path = artifact_root / "method_synthesis_metadata.json"
	method_synthesis_metadata = (
		json.loads(metadata_path.read_text())
		if metadata_path.exists()
		else {}
	)
	artifact_metadata_path = artifact_root / "artifact_metadata.json"
	artifact_metadata = (
		json.loads(artifact_metadata_path.read_text())
		if artifact_metadata_path.exists()
		else {}
	)
	masked_domain_path = artifact_root / "masked_domain.hddl"
	plan_library_asl_path = artifact_root / "plan_library.asl"
	return PlanLibraryArtifactBundle(
		domain_name=str(artifact_metadata.get("domain_name") or artifact_root.name),
		query_sequence=tuple(
			QueryInstructionRecord.from_dict(item)
			for item in query_sequence_payload
			if isinstance(item, dict)
		),
		temporal_specifications=tuple(
			TemporalSpecificationRecord.from_dict(item)
			for item in temporal_specifications_payload
			if isinstance(item, dict)
		),
		method_library=HTNMethodLibrary.from_dict(dict(method_library_payload)),
		plan_library=PlanLibrary.from_dict(dict(plan_library_payload)),
		translation_coverage=TranslationCoverage.from_dict(dict(translation_coverage_payload)),
		library_validation=LibraryValidationRecord.from_dict(dict(library_validation_payload)),
		method_synthesis_metadata=dict(method_synthesis_metadata),
		artifact_root=str(artifact_root),
		masked_domain_file=str(masked_domain_path) if masked_domain_path.exists() else None,
		plan_library_asl_file=(
			str(plan_library_asl_path) if plan_library_asl_path.exists() else None
		),
	)
