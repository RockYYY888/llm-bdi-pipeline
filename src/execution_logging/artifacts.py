"""
Persisted artifact structures for domain builds and evaluation execution.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from method_library.synthesis.schema import (
	HTNMethodLibrary,
)


INLINE_TEXT_PREVIEW_CHARS = 2_000
INLINE_SEQUENCE_PREVIEW_ITEMS = 40


def _compact_text_payload(key: str, value: str) -> Dict[str, Any]:
	text = str(value or "")
	if len(text) <= INLINE_TEXT_PREVIEW_CHARS:
		return {key: text}
	return {
		key: text[:INLINE_TEXT_PREVIEW_CHARS],
		f"{key}_truncated": True,
		f"{key}_chars": len(text),
		f"{key}_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
	}


def _compact_sequence_payload(values: Sequence[Any]) -> Dict[str, Any]:
	items = list(values)
	if len(items) <= INLINE_SEQUENCE_PREVIEW_ITEMS:
		return {"items": items, "count": len(items), "truncated": False}
	edge_count = INLINE_SEQUENCE_PREVIEW_ITEMS // 2
	return {
		"items": items[:edge_count] + items[-edge_count:],
		"count": len(items),
		"truncated": True,
		"omitted_middle_count": len(items) - (edge_count * 2),
	}


@dataclass(frozen=True)
class GroundedSubgoal:
	"""One grounded evaluation subgoal emitted by temporal goal grounding."""

	subgoal_id: str
	task_name: str
	args: Tuple[str, ...] = ()
	argument_types: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.subgoal_id,
			"task_name": self.task_name,
			"args": list(self.args),
			"argument_types": list(self.argument_types),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "GroundedSubgoal":
		return cls(
			subgoal_id=str(payload.get("id") or payload.get("subgoal_id") or ""),
			task_name=str(payload.get("task_name") or ""),
			args=tuple(str(arg) for arg in (payload.get("args") or ())),
			argument_types=tuple(
				str(type_name)
				for type_name in (payload.get("argument_types") or ())
			),
		)

@dataclass(frozen=True)
class TemporalGroundingResult:
	"""Validated evaluation semantic grounding output."""

	query_text: str
	ltlf_formula: str
	subgoals: Tuple[GroundedSubgoal, ...]
	typed_objects: Dict[str, str] = field(default_factory=dict)
	query_object_inventory: Tuple[Dict[str, Any], ...] = ()
	diagnostics: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"query_text": self.query_text,
			"ltlf_formula": self.ltlf_formula,
			"subgoals": [subgoal.to_dict() for subgoal in self.subgoals],
			"typed_objects": dict(self.typed_objects),
			"query_object_inventory": [dict(entry) for entry in self.query_object_inventory],
			"diagnostics": list(self.diagnostics),
		}

	def to_log_dict(self) -> Dict[str, Any]:
		"""Return bounded grounding metadata for execution logs."""

		subgoals = [subgoal.to_dict() for subgoal in self.subgoals]
		query_object_inventory = [
			dict(entry)
			for entry in self.query_object_inventory
		]
		return {
			**_compact_text_payload("query_text", self.query_text),
			**_compact_text_payload("ltlf_formula", self.ltlf_formula),
			"subgoals": _compact_sequence_payload(subgoals),
			"typed_object_count": len(self.typed_objects),
			"query_object_inventory": _compact_sequence_payload(query_object_inventory),
			"diagnostics": list(self.diagnostics),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "TemporalGroundingResult":
		return cls(
			query_text=str(payload.get("query_text") or ""),
			ltlf_formula=str(payload.get("ltlf_formula") or ""),
			subgoals=tuple(
				GroundedSubgoal.from_dict(item)
				for item in (payload.get("subgoals") or ())
				if isinstance(item, dict)
			),
			typed_objects={
				str(key): str(value)
				for key, value in dict(payload.get("typed_objects") or {}).items()
			},
			query_object_inventory=tuple(
				dict(entry)
				for entry in (payload.get("query_object_inventory") or ())
				if isinstance(entry, dict)
			),
			diagnostics=tuple(
				str(message).strip()
				for message in (payload.get("diagnostics") or ())
				if str(message).strip()
			),
		)


@dataclass(frozen=True)
class JasonExecutionResult:
	"""Structured Jason runtime outcome for one evaluation query."""

	query_text: str
	ltlf_formula: str
	action_path: Tuple[str, ...]
	method_trace: Tuple[Dict[str, Any], ...]
	hierarchical_plan_text: Optional[str] = None
	verification_problem_file: Optional[str] = None
	verification_mode: str = "original_problem"
	failed_goals: Tuple[str, ...] = ()
	failure_class: Optional[str] = None
	consistency_checks: Dict[str, Any] = field(default_factory=dict)
	artifacts: Dict[str, Any] = field(default_factory=dict)
	timing_profile: Dict[str, Any] = field(default_factory=dict)
	diagnostics: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"query_text": self.query_text,
			"ltlf_formula": self.ltlf_formula,
			"action_path": list(self.action_path),
			"method_trace": [dict(item) for item in self.method_trace],
			"hierarchical_plan_text": self.hierarchical_plan_text,
			"verification_problem_file": self.verification_problem_file,
			"verification_mode": self.verification_mode,
			"failed_goals": list(self.failed_goals),
			"failure_class": self.failure_class,
			"consistency_checks": dict(self.consistency_checks),
			"artifacts": dict(self.artifacts),
			"timing_profile": dict(self.timing_profile),
			"diagnostics": list(self.diagnostics),
		}

	def to_log_dict(self) -> Dict[str, Any]:
		"""Return bounded runtime metadata for execution logs."""

		hierarchical_plan_text = str(self.hierarchical_plan_text or "")
		return {
			**_compact_text_payload("query_text", self.query_text),
			**_compact_text_payload("ltlf_formula", self.ltlf_formula),
			"action_path_count": len(self.action_path),
			"method_trace_count": len(self.method_trace),
			"hierarchical_plan_text_chars": len(hierarchical_plan_text),
			"verification_problem_file": self.verification_problem_file,
			"verification_mode": self.verification_mode,
			"failed_goals": list(self.failed_goals),
			"failure_class": self.failure_class,
			"consistency_checks": dict(self.consistency_checks),
			"artifacts": dict(self.artifacts),
			"timing_profile": dict(self.timing_profile),
			"diagnostics": list(self.diagnostics),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "JasonExecutionResult":
		return cls(
			query_text=str(payload.get("query_text") or ""),
			ltlf_formula=str(payload.get("ltlf_formula") or ""),
			action_path=tuple(
				str(item).strip()
				for item in (payload.get("action_path") or ())
				if str(item).strip()
			),
			method_trace=tuple(
				dict(item)
				for item in (payload.get("method_trace") or ())
				if isinstance(item, dict)
			),
			hierarchical_plan_text=payload.get("hierarchical_plan_text"),
			verification_problem_file=payload.get("verification_problem_file"),
			verification_mode=str(payload.get("verification_mode") or "original_problem"),
			failed_goals=tuple(
				str(goal).strip()
				for goal in (payload.get("failed_goals") or ())
				if str(goal).strip()
			),
			failure_class=(
				str(payload.get("failure_class")).strip()
				if payload.get("failure_class") is not None
				else None
			),
			consistency_checks=dict(payload.get("consistency_checks") or {}),
			artifacts=dict(payload.get("artifacts") or {}),
			timing_profile=dict(payload.get("timing_profile") or {}),
			diagnostics=tuple(
				str(message).strip()
				for message in (payload.get("diagnostics") or ())
				if str(message).strip()
			),
		)


@dataclass(frozen=True)
class DomainLibraryArtifact:
	"""Stable persisted artifact for one domain-complete library build."""

	domain_name: str
	method_library: HTNMethodLibrary
	method_synthesis_metadata: Dict[str, Any]
	domain_gate: Dict[str, Any]
	source_domain_kind: str = "generated"
	artifact_root: Optional[str] = None
	masked_domain_file: Optional[str] = None
	generated_domain_file: Optional[str] = None

	def to_dict(self) -> Dict[str, Any]:
		return {
			"domain_name": self.domain_name,
			"method_library": self.method_library.to_dict(),
			"method_synthesis_metadata": dict(self.method_synthesis_metadata),
			"domain_gate": dict(self.domain_gate),
			"source_domain_kind": self.source_domain_kind,
			"artifact_root": self.artifact_root,
			"masked_domain_file": self.masked_domain_file,
			"generated_domain_file": self.generated_domain_file,
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "DomainLibraryArtifact":
		return cls(
			domain_name=str(payload.get("domain_name") or ""),
			method_library=HTNMethodLibrary.from_dict(
				dict(payload.get("method_library") or {}),
			),
			method_synthesis_metadata=dict(payload.get("method_synthesis_metadata") or {}),
			domain_gate=dict(payload.get("domain_gate") or {}),
			source_domain_kind=str(payload.get("source_domain_kind") or "generated"),
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
			generated_domain_file=(
				str(payload.get("generated_domain_file")).strip()
				if payload.get("generated_domain_file") is not None
				else None
			),
		)


def persist_domain_library_artifact(
	*,
	artifact_root: Path,
	artifact: DomainLibraryArtifact,
	masked_domain_text: Optional[str] = None,
	generated_domain_text: Optional[str] = None,
) -> Dict[str, str]:
	"""Persist the domain build artifact under a stable cache root."""

	artifact_root.mkdir(parents=True, exist_ok=True)
	artifact_metadata_path = artifact_root / "artifact_metadata.json"
	method_library_path = artifact_root / "method_library.json"
	method_synthesis_metadata_path = artifact_root / "method_synthesis_metadata.json"
	domain_gate_path = artifact_root / "domain_gate.json"
	masked_domain_path = artifact_root / "masked_domain.hddl"
	generated_domain_path = artifact_root / "generated_domain.hddl"

	artifact_metadata_path.write_text(
		json.dumps(
			{
				"domain_name": artifact.domain_name,
				"source_domain_kind": artifact.source_domain_kind,
			},
			indent=2,
		),
	)
	method_library_path.write_text(json.dumps(artifact.method_library.to_dict(), indent=2))
	method_synthesis_metadata_path.write_text(
		json.dumps(artifact.method_synthesis_metadata, indent=2),
	)
	domain_gate_path.write_text(json.dumps(artifact.domain_gate, indent=2))
	if masked_domain_text is not None:
		masked_domain_path.write_text(str(masked_domain_text))
	if generated_domain_text is not None:
		generated_domain_path.write_text(str(generated_domain_text))

	paths = {
		"artifact_metadata": str(artifact_metadata_path),
		"method_library": str(method_library_path),
		"method_synthesis_metadata": str(method_synthesis_metadata_path),
		"domain_gate": str(domain_gate_path),
	}
	if masked_domain_text is not None:
		paths["masked_domain"] = str(masked_domain_path)
	if generated_domain_text is not None:
		paths["generated_domain"] = str(generated_domain_path)
	return paths


def load_domain_library_artifact(
	library_artifact: str | Path | Dict[str, Any] | DomainLibraryArtifact | HTNMethodLibrary,
) -> DomainLibraryArtifact:
	"""Load a domain library artifact from disk or in-memory payloads."""

	if isinstance(library_artifact, DomainLibraryArtifact):
		return library_artifact
	if isinstance(library_artifact, HTNMethodLibrary):
		return DomainLibraryArtifact(
			domain_name="",
			method_library=library_artifact,
			method_synthesis_metadata={},
			domain_gate={},
			source_domain_kind="generated",
		)
	if isinstance(library_artifact, dict):
		if "method_library" in library_artifact:
			return DomainLibraryArtifact.from_dict(library_artifact)
		return DomainLibraryArtifact(
			domain_name="",
			method_library=HTNMethodLibrary.from_dict(library_artifact),
			method_synthesis_metadata={},
			domain_gate={},
			source_domain_kind="generated",
		)

	artifact_path = Path(library_artifact).expanduser().resolve()
	if artifact_path.is_dir():
		method_library_payload = json.loads((artifact_path / "method_library.json").read_text())
		artifact_metadata_path = artifact_path / "artifact_metadata.json"
		method_synthesis_metadata_path = artifact_path / "method_synthesis_metadata.json"
		domain_gate_path = artifact_path / "domain_gate.json"
		artifact_metadata = (
			json.loads(artifact_metadata_path.read_text())
			if artifact_metadata_path.exists()
			else {}
		)
		method_synthesis_metadata = json.loads(method_synthesis_metadata_path.read_text())
		domain_gate = json.loads(domain_gate_path.read_text())
		masked_domain_path = artifact_path / "masked_domain.hddl"
		generated_domain_path = artifact_path / "generated_domain.hddl"
		return DomainLibraryArtifact(
			domain_name=str(artifact_metadata.get("domain_name") or artifact_path.name),
			method_library=HTNMethodLibrary.from_dict(method_library_payload),
			method_synthesis_metadata=dict(method_synthesis_metadata),
			domain_gate=dict(domain_gate),
			source_domain_kind=str(
				artifact_metadata.get("source_domain_kind")
				or method_synthesis_metadata.get("source_domain_kind")
				or "generated"
			),
			artifact_root=str(artifact_path),
			masked_domain_file=str(masked_domain_path) if masked_domain_path.exists() else None,
			generated_domain_file=(
				str(generated_domain_path) if generated_domain_path.exists() else None
			),
		)

	method_library_payload = json.loads(artifact_path.read_text())
	artifact_root = str(artifact_path.parent)
	masked_domain_path = artifact_path.parent / "masked_domain.hddl"
	generated_domain_path = artifact_path.parent / "generated_domain.hddl"
	return DomainLibraryArtifact(
		domain_name=artifact_path.stem,
		method_library=HTNMethodLibrary.from_dict(method_library_payload),
		method_synthesis_metadata={},
		domain_gate={},
		source_domain_kind="generated",
		artifact_root=artifact_root,
		masked_domain_file=str(masked_domain_path) if masked_domain_path.exists() else None,
		generated_domain_file=(
			str(generated_domain_path) if generated_domain_path.exists() else None
		),
	)
__all__ = [
	"DomainLibraryArtifact",
	"GroundedSubgoal",
	"JasonExecutionResult",
	"TemporalGroundingResult",
	"load_domain_library_artifact",
	"persist_domain_library_artifact",
]
