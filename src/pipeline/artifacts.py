"""
Persisted artifact structures for domain builds and query execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from domain_build.method_synthesis.schema import (
	HTNLiteral,
	HTNMethodLibrary,
	HTNTargetTaskBinding,
)


def _literal_to_dict(literal: HTNLiteral) -> Dict[str, Any]:
	return {
		"predicate": literal.predicate,
		"args": list(literal.args),
		"is_positive": literal.is_positive,
		"source_symbol": literal.source_symbol,
		"is_equality": literal.is_equality,
	}


def _load_literal(payload: Dict[str, Any]) -> HTNLiteral:
	return HTNLiteral(
		predicate=str(payload.get("predicate") or ""),
		args=tuple(str(arg) for arg in (payload.get("args") or ())),
		is_positive=bool(payload.get("is_positive", True)),
		source_symbol=payload.get("source_symbol"),
	)


@dataclass(frozen=True)
class GoalRequestNode:
	"""Grounded query-time task node."""

	node_id: str
	task_name: str
	args: Tuple[str, ...] = ()
	argument_types: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"node_id": self.node_id,
			"task_name": self.task_name,
			"args": list(self.args),
			"argument_types": list(self.argument_types),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "GoalRequestNode":
		return cls(
			node_id=str(payload.get("node_id") or ""),
			task_name=str(payload.get("task_name") or ""),
			args=tuple(str(arg) for arg in (payload.get("args") or ())),
			argument_types=tuple(
				str(type_name)
				for type_name in (payload.get("argument_types") or ())
			),
		)


@dataclass(frozen=True)
class GoalRequest:
	"""Grounded query-time request graph."""

	query_text: str
	nodes: Tuple[GoalRequestNode, ...]
	precedence_edges: Tuple[Tuple[str, str], ...] = ()
	query_object_inventory: Tuple[Dict[str, Any], ...] = ()
	typed_objects: Dict[str, str] = field(default_factory=dict)
	diagnostics: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"query_text": self.query_text,
			"nodes": [node.to_dict() for node in self.nodes],
			"precedence_edges": [
				{"before": before, "after": after}
				for before, after in self.precedence_edges
			],
			"query_object_inventory": [dict(entry) for entry in self.query_object_inventory],
			"typed_objects": dict(self.typed_objects),
			"diagnostics": list(self.diagnostics),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "GoalRequest":
		return cls(
			query_text=str(payload.get("query_text") or ""),
			nodes=tuple(
				GoalRequestNode.from_dict(node_payload)
				for node_payload in (payload.get("nodes") or ())
				if isinstance(node_payload, dict)
			),
			precedence_edges=tuple(
				(
					str(edge_payload.get("before") or ""),
					str(edge_payload.get("after") or ""),
				)
				for edge_payload in (payload.get("precedence_edges") or ())
				if isinstance(edge_payload, dict)
			),
			query_object_inventory=tuple(
				dict(entry)
				for entry in (payload.get("query_object_inventory") or ())
				if isinstance(entry, dict)
			),
			typed_objects={
				str(key): str(value)
				for key, value in dict(payload.get("typed_objects") or {}).items()
			},
			diagnostics=tuple(
				str(message).strip()
				for message in (payload.get("diagnostics") or ())
				if str(message).strip()
			),
		)


@dataclass(frozen=True)
class PlanningRequest:
	"""Planner-oriented query request context."""

	query_text: str
	goal_request: GoalRequest
	problem_objects: Tuple[str, ...] = ()
	typed_objects: Dict[str, str] = field(default_factory=dict)
	task_network: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
	task_network_ordered: bool = True
	ordering_edges: Tuple[Tuple[str, str], ...] = ()
	diagnostics: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"query_text": self.query_text,
			"goal_request": self.goal_request.to_dict(),
			"problem_objects": list(self.problem_objects),
			"typed_objects": dict(self.typed_objects),
			"task_network": [
				{"task_name": task_name, "args": list(args)}
				for task_name, args in self.task_network
			],
			"task_network_ordered": self.task_network_ordered,
			"ordering_edges": [
				{"before": before, "after": after}
				for before, after in self.ordering_edges
			],
			"diagnostics": list(self.diagnostics),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "PlanningRequest":
		goal_request_payload = dict(payload.get("goal_request") or {})
		return cls(
			query_text=str(payload.get("query_text") or ""),
			goal_request=GoalRequest.from_dict(goal_request_payload),
			problem_objects=tuple(
				str(value).strip()
				for value in (payload.get("problem_objects") or ())
				if str(value).strip()
			),
			typed_objects={
				str(key): str(value)
				for key, value in dict(payload.get("typed_objects") or {}).items()
			},
			task_network=tuple(
				(
					str(entry.get("task_name") or ""),
					tuple(str(arg) for arg in (entry.get("args") or ())),
				)
				for entry in (payload.get("task_network") or ())
				if isinstance(entry, dict)
			),
			task_network_ordered=bool(payload.get("task_network_ordered", True)),
			ordering_edges=tuple(
				(
					str(edge_payload.get("before") or ""),
					str(edge_payload.get("after") or ""),
				)
				for edge_payload in (payload.get("ordering_edges") or ())
				if isinstance(edge_payload, dict)
			),
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

	def to_dict(self) -> Dict[str, Any]:
		return {
			"domain_name": self.domain_name,
			"method_library": self.method_library.to_dict(),
			"method_synthesis_metadata": dict(self.method_synthesis_metadata),
			"domain_gate": dict(self.domain_gate),
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
		)


def persist_domain_library_artifact(
	*,
	artifact_root: Path,
	artifact: DomainLibraryArtifact,
) -> Dict[str, str]:
	"""Persist the domain build artifact under a stable cache root."""

	artifact_root.mkdir(parents=True, exist_ok=True)
	method_library_path = artifact_root / "method_library.json"
	method_synthesis_metadata_path = artifact_root / "method_synthesis_metadata.json"
	domain_gate_path = artifact_root / "domain_gate.json"

	method_library_path.write_text(json.dumps(artifact.method_library.to_dict(), indent=2))
	method_synthesis_metadata_path.write_text(
		json.dumps(artifact.method_synthesis_metadata, indent=2),
	)
	domain_gate_path.write_text(json.dumps(artifact.domain_gate, indent=2))

	return {
		"method_library": str(method_library_path),
		"method_synthesis_metadata": str(method_synthesis_metadata_path),
		"domain_gate": str(domain_gate_path),
	}


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
		)
	if isinstance(library_artifact, dict):
		if "method_library" in library_artifact:
			return DomainLibraryArtifact.from_dict(library_artifact)
		return DomainLibraryArtifact(
			domain_name="",
			method_library=HTNMethodLibrary.from_dict(library_artifact),
			method_synthesis_metadata={},
			domain_gate={},
		)

	artifact_path = Path(library_artifact).expanduser().resolve()
	if artifact_path.is_dir():
		method_library_payload = json.loads((artifact_path / "method_library.json").read_text())
		method_synthesis_metadata_path = artifact_path / "method_synthesis_metadata.json"
		domain_gate_path = artifact_path / "domain_gate.json"
		method_synthesis_metadata = json.loads(method_synthesis_metadata_path.read_text())
		domain_gate = json.loads(domain_gate_path.read_text())
		return DomainLibraryArtifact(
			domain_name=artifact_path.name,
			method_library=HTNMethodLibrary.from_dict(method_library_payload),
			method_synthesis_metadata=dict(method_synthesis_metadata),
			domain_gate=dict(domain_gate),
		)

	method_library_payload = json.loads(artifact_path.read_text())
	return DomainLibraryArtifact(
		domain_name=artifact_path.stem,
		method_library=HTNMethodLibrary.from_dict(method_library_payload),
		method_synthesis_metadata={},
		domain_gate={},
	)


def query_bound_method_library(
	method_library: HTNMethodLibrary,
	*,
	target_literals: Sequence[HTNLiteral],
	target_task_bindings: Sequence[HTNTargetTaskBinding],
) -> HTNMethodLibrary:
	"""Create a transient query-bound overlay without mutating the persisted artifact."""

	return HTNMethodLibrary(
		compound_tasks=list(method_library.compound_tasks),
		primitive_tasks=list(method_library.primitive_tasks),
		methods=list(method_library.methods),
		target_literals=list(target_literals),
		target_task_bindings=list(target_task_bindings),
	)


__all__ = [
	"DomainLibraryArtifact",
	"GoalRequest",
	"GoalRequestNode",
	"PlanningRequest",
	"load_domain_library_artifact",
	"persist_domain_library_artifact",
	"query_bound_method_library",
]
