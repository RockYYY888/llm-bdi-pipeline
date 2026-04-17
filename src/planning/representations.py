"""
Planning representation construction for official Hierarchical Task Network benchmarks.

This module is the compilation layer. It transforms the original instance into one or
more solver-ready representations without making solver-selection decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from planning.problem_structure import ProblemStructure, ProblemStructureAnalyzer
from planning.linearization import LiftedLinearPlanner


@dataclass(frozen=True)
class PlanningRepresentation:
	"""A solver-ready representation derived from one official instance."""

	representation_id: str
	representation_source: str
	ordering_kind: str
	domain_file: str
	problem_file: str
	compilation_profile: str
	metadata: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"representation_id": self.representation_id,
			"representation_source": self.representation_source,
			"ordering_kind": self.ordering_kind,
			"domain_file": self.domain_file,
			"problem_file": self.problem_file,
			"compilation_profile": self.compilation_profile,
			"metadata": dict(self.metadata),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "PlanningRepresentation":
		return cls(
			representation_id=str(payload["representation_id"]),
			representation_source=str(payload["representation_source"]),
			ordering_kind=str(payload["ordering_kind"]),
			domain_file=str(payload["domain_file"]),
			problem_file=str(payload["problem_file"]),
			compilation_profile=str(payload["compilation_profile"]),
			metadata=dict(payload.get("metadata") or {}),
		)


@dataclass(frozen=True)
class RepresentationBuildResult:
	"""Compilation-layer output for one instance."""

	structure: ProblemStructure
	representations: Tuple[PlanningRepresentation, ...]


class PlanningRepresentationBuilder:
	"""Build original and optionally linearized representations for official planning."""

	def __init__(
		self,
		workspace: Optional[str | Path] = None,
		analyzer: Optional[ProblemStructureAnalyzer] = None,
		linearizer: Optional[LiftedLinearPlanner] = None,
	) -> None:
		self.workspace = Path(workspace).resolve() if workspace else None
		self.analyzer = analyzer or ProblemStructureAnalyzer()
		self.linearizer = linearizer or LiftedLinearPlanner(workspace=self.workspace)

	def build(
		self,
		*,
		domain: Any,
		problem: Any,
		domain_file: str | Path,
		problem_file: str | Path,
		timeout_seconds: Optional[float] = None,
	) -> RepresentationBuildResult:
		structure = self.analyzer.analyze(domain=domain, problem=problem)
		representations = [self._build_original_representation(domain_file, problem_file, structure)]
		if structure.requires_linearization:
			representations.append(
				self._build_linearized_representation(
					domain_file=domain_file,
					problem_file=problem_file,
					timeout_seconds=timeout_seconds,
				),
			)
		return RepresentationBuildResult(
			structure=structure,
			representations=tuple(representations),
		)

	def _build_original_representation(
		self,
		domain_file: str | Path,
		problem_file: str | Path,
		structure: ProblemStructure,
	) -> PlanningRepresentation:
		return PlanningRepresentation(
			representation_id=(
				"original_total_order"
				if structure.is_total_order
				else "original_partial_order"
			),
			representation_source="original",
			ordering_kind="total_order" if structure.is_total_order else "partial_order",
			domain_file=str(Path(domain_file).resolve()),
			problem_file=str(Path(problem_file).resolve()),
			compilation_profile="identity",
			metadata={
				"requires_linearization": structure.requires_linearization,
			},
		)

	def _build_linearized_representation(
		self,
		*,
		domain_file: str | Path,
		problem_file: str | Path,
		timeout_seconds: Optional[float],
	) -> PlanningRepresentation:
		artifacts = self.linearizer.linearize_hddl_files(
			domain_file=domain_file,
			problem_file=problem_file,
			transition_name="official_problem_root_compilation",
			timeout_seconds=timeout_seconds,
		)
		return PlanningRepresentation(
			representation_id="linearized_total_order",
			representation_source="linearized",
			ordering_kind="total_order",
			domain_file=str(Path(artifacts["linearized_domain_file"]).resolve()),
			problem_file=str(Path(artifacts["linearized_problem_file"]).resolve()),
			compilation_profile="semantics_preserving_linearization",
			metadata=dict(artifacts),
		)
