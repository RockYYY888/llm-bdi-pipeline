"""
Pipeline artifact structures for domain builds and query execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from stage3_method_synthesis.htn_schema import (
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
class TemporallyExtendedGoalNode:
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
	def from_dict(cls, payload: Dict[str, Any]) -> "TemporallyExtendedGoalNode":
		return cls(
			node_id=str(payload.get("node_id") or ""),
			task_name=str(payload.get("task_name") or ""),
			args=tuple(str(arg) for arg in (payload.get("args") or ())),
			argument_types=tuple(
				str(type_name) for type_name in (payload.get("argument_types") or ())
			),
		)


@dataclass(frozen=True)
class TemporallyExtendedGoal:
	"""Query-time temporally extended goal graph."""

	query_text: str
	nodes: Tuple[TemporallyExtendedGoalNode, ...]
	precedence_edges: Tuple[Tuple[str, str], ...] = ()
	query_object_inventory: Tuple[Dict[str, Any], ...] = ()
	typed_objects: Dict[str, str] = field(default_factory=dict)
	diagnostics: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"query_text": self.query_text,
			"nodes": [node.to_dict() for node in self.nodes],
			"precedence_edges": [
				{
					"before": before,
					"after": after,
				}
				for before, after in self.precedence_edges
			],
			"query_object_inventory": [dict(entry) for entry in self.query_object_inventory],
			"typed_objects": dict(self.typed_objects),
			"diagnostics": list(self.diagnostics),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "TemporallyExtendedGoal":
		return cls(
			query_text=str(payload.get("query_text") or ""),
			nodes=tuple(
				TemporallyExtendedGoalNode.from_dict(node_payload)
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
class PlanningRequestContext:
	"""Planner-oriented query request context."""

	query_text: str
	temporally_extended_goal: TemporallyExtendedGoal
	problem_objects: Tuple[str, ...] = ()
	typed_objects: Dict[str, str] = field(default_factory=dict)
	task_network: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
	task_network_ordered: bool = True
	ordering_edges: Tuple[Tuple[str, str], ...] = ()
	diagnostics: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"query_text": self.query_text,
			"temporally_extended_goal": self.temporally_extended_goal.to_dict(),
			"problem_objects": list(self.problem_objects),
			"typed_objects": dict(self.typed_objects),
			"task_network": [
				{
					"task_name": task_name,
					"args": list(args),
				}
				for task_name, args in self.task_network
			],
			"task_network_ordered": self.task_network_ordered,
			"ordering_edges": [
				{
					"before": before,
					"after": after,
				}
				for before, after in self.ordering_edges
			],
			"diagnostics": list(self.diagnostics),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "PlanningRequestContext":
		teg_payload = dict(payload.get("temporally_extended_goal") or {})
		return cls(
			query_text=str(payload.get("query_text") or ""),
			temporally_extended_goal=TemporallyExtendedGoal.from_dict(teg_payload),
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
class QueryExecutionContext:
	"""Query-time context built after Stage 1 and Stage 2."""

	query_text: str
	ordered_query_sequence: bool
	query_task_anchors: Tuple[Dict[str, Any], ...] = ()
	query_task_network: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
	query_task_name_map: Dict[str, str] = field(default_factory=dict)
	target_literals: Tuple[HTNLiteral, ...] = ()
	target_task_bindings: Tuple[HTNTargetTaskBinding, ...] = ()
	literal_signatures: Tuple[str, ...] = ()
	query_object_inventory: Tuple[Dict[str, Any], ...] = ()
	query_objects: Tuple[str, ...] = ()
	typed_objects: Dict[str, str] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"query_text": self.query_text,
			"ordered_query_sequence": self.ordered_query_sequence,
			"query_task_anchors": [dict(anchor) for anchor in self.query_task_anchors],
			"query_task_network": [
				{
					"task_name": task_name,
					"args": list(args),
				}
				for task_name, args in self.query_task_network
			],
			"query_task_name_map": dict(self.query_task_name_map),
			"target_literals": [_literal_to_dict(literal) for literal in self.target_literals],
			"target_task_bindings": [
				binding.to_dict()
				for binding in self.target_task_bindings
			],
			"literal_signatures": list(self.literal_signatures),
			"query_object_inventory": [dict(entry) for entry in self.query_object_inventory],
			"query_objects": list(self.query_objects),
			"typed_objects": dict(self.typed_objects),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "QueryExecutionContext":
		return cls(
			query_text=str(payload.get("query_text") or ""),
			ordered_query_sequence=bool(payload.get("ordered_query_sequence", False)),
			query_task_anchors=tuple(
				dict(anchor)
				for anchor in (payload.get("query_task_anchors") or ())
				if isinstance(anchor, dict)
			),
			query_task_network=tuple(
				(
					str(entry.get("task_name") or ""),
					tuple(str(arg) for arg in (entry.get("args") or ())),
				)
				for entry in (payload.get("query_task_network") or ())
				if isinstance(entry, dict)
			),
			query_task_name_map={
				str(key): str(value)
				for key, value in dict(payload.get("query_task_name_map") or {}).items()
			},
			target_literals=tuple(
				_load_literal(item)
				for item in (payload.get("target_literals") or ())
				if isinstance(item, dict)
			),
			target_task_bindings=tuple(
				HTNTargetTaskBinding(
					target_literal=str(item.get("target_literal") or ""),
					task_name=str(item.get("task_name") or ""),
				)
				for item in (payload.get("target_task_bindings") or ())
				if isinstance(item, dict)
			),
			literal_signatures=tuple(
				str(value).strip()
				for value in (payload.get("literal_signatures") or ())
				if str(value).strip()
			),
			query_object_inventory=tuple(
				dict(entry)
				for entry in (payload.get("query_object_inventory") or ())
				if isinstance(entry, dict)
			),
			query_objects=tuple(
				str(value).strip()
				for value in (payload.get("query_objects") or ())
				if str(value).strip()
			),
			typed_objects={
				str(key): str(value)
				for key, value in dict(payload.get("typed_objects") or {}).items()
			},
		)


@dataclass(frozen=True)
class DomainBuildArtifact:
	"""Stable persisted artifact for one domain-complete method library build."""

	domain_name: str
	method_library: HTNMethodLibrary
	stage3_metadata: Dict[str, Any]
	stage4_domain_gate: Dict[str, Any]

	def to_dict(self) -> Dict[str, Any]:
		return {
			"domain_name": self.domain_name,
			"method_library": self.method_library.to_dict(),
			"stage3_metadata": dict(self.stage3_metadata),
			"stage4_domain_gate": dict(self.stage4_domain_gate),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "DomainBuildArtifact":
		return cls(
			domain_name=str(payload.get("domain_name") or ""),
			method_library=HTNMethodLibrary.from_dict(
				dict(payload.get("method_library") or {}),
			),
			stage3_metadata=dict(payload.get("stage3_metadata") or {}),
			stage4_domain_gate=dict(payload.get("stage4_domain_gate") or {}),
		)


def persist_domain_build_artifact(
	*,
	artifact_root: Path,
	artifact: DomainBuildArtifact,
) -> Dict[str, str]:
	"""Persist the domain build artifact under a stable cache root."""

	artifact_root.mkdir(parents=True, exist_ok=True)
	method_library_path = artifact_root / "method_library.json"
	stage3_metadata_path = artifact_root / "stage3_metadata.json"
	stage4_domain_gate_path = artifact_root / "stage4_domain_gate.json"

	method_library_path.write_text(
		__import__("json").dumps(artifact.method_library.to_dict(), indent=2),
	)
	stage3_metadata_path.write_text(
		__import__("json").dumps(artifact.stage3_metadata, indent=2),
	)
	stage4_domain_gate_path.write_text(
		__import__("json").dumps(artifact.stage4_domain_gate, indent=2),
	)

	return {
		"method_library": str(method_library_path),
		"stage3_metadata": str(stage3_metadata_path),
		"stage4_domain_gate": str(stage4_domain_gate_path),
	}


def load_domain_build_artifact(
	library_artifact: str | Path | Dict[str, Any] | DomainBuildArtifact | HTNMethodLibrary,
) -> DomainBuildArtifact:
	"""Load a domain build artifact from disk or in-memory payloads."""

	if isinstance(library_artifact, DomainBuildArtifact):
		return library_artifact
	if isinstance(library_artifact, HTNMethodLibrary):
		return DomainBuildArtifact(
			domain_name="",
			method_library=library_artifact,
			stage3_metadata={},
			stage4_domain_gate={},
		)
	if isinstance(library_artifact, dict):
		if "method_library" in library_artifact:
			return DomainBuildArtifact.from_dict(library_artifact)
		return DomainBuildArtifact(
			domain_name="",
			method_library=HTNMethodLibrary.from_dict(library_artifact),
			stage3_metadata={},
			stage4_domain_gate={},
		)

	artifact_path = Path(library_artifact).expanduser().resolve()
	if artifact_path.is_dir():
		method_library_payload = __import__("json").loads(
			(artifact_path / "method_library.json").read_text(),
		)
		stage3_metadata = __import__("json").loads(
			(artifact_path / "stage3_metadata.json").read_text(),
		)
		stage4_domain_gate = __import__("json").loads(
			(artifact_path / "stage4_domain_gate.json").read_text(),
		)
		return DomainBuildArtifact(
			domain_name=artifact_path.name,
			method_library=HTNMethodLibrary.from_dict(method_library_payload),
			stage3_metadata=dict(stage3_metadata),
			stage4_domain_gate=dict(stage4_domain_gate),
		)

	method_library_payload = __import__("json").loads(artifact_path.read_text())
	return DomainBuildArtifact(
		domain_name=artifact_path.stem,
		method_library=HTNMethodLibrary.from_dict(method_library_payload),
		stage3_metadata={},
		stage4_domain_gate={},
	)


def query_bound_method_library(
	method_library: HTNMethodLibrary,
	*,
	target_literals: Sequence[HTNLiteral],
	target_task_bindings: Sequence[HTNTargetTaskBinding],
) -> HTNMethodLibrary:
	"""Create a transient query-bound overlay without mutating the domain artifact."""

	return HTNMethodLibrary(
		compound_tasks=list(method_library.compound_tasks),
		primitive_tasks=list(method_library.primitive_tasks),
		methods=list(method_library.methods),
		target_literals=list(target_literals),
		target_task_bindings=list(target_task_bindings),
	)
