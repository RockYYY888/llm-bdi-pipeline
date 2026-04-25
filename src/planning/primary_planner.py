"""
Primary planner capability declaration and dispatch for HTN evaluation.

The dissertation baseline intentionally exposes one primary planner configuration:
lifted_panda_sat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from planning.linearization import LiftedLinearPlanner
from planning.official_benchmark import (
	OFFICIAL_LIFTED_LINEAR_INNER_SOLVER_ID,
)
from planning.plan_models import PANDAPlanResult
from planning.representations import PlanningRepresentation


def _solver_config_with_planner_budget(solver_config: Dict[str, Any]) -> Dict[str, Any]:
	"""
	The primary lifted_panda_sat baseline should inherit the full benchmark
	timeout instead of the lower standalone SAT wrapper default.
	"""
	prepared = dict(solver_config)
	prepared.pop("timeout_seconds", None)
	return prepared


@dataclass(frozen=True)
class PrimaryPlannerTask:
	"""One primary planner invocation over one compiled representation."""

	task_id: str
	planner_id: str
	representation: PlanningRepresentation

	def to_dict(self) -> Dict[str, Any]:
		return {
			"task_id": self.task_id,
			"planner_id": self.planner_id,
			"representation": self.representation.to_dict(),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "PrimaryPlannerTask":
		return cls(
			task_id=str(payload["task_id"]),
			planner_id=str(payload["planner_id"]),
			representation=PlanningRepresentation.from_dict(dict(payload["representation"])),
		)


class PrimaryHTNPlanner(ABC):
	"""Common interface for the supported primary HTN planner."""

	planner_id: str
	plan_source_label: str
	supported_sources: frozenset[str]
	supported_ordering_kinds: frozenset[str]

	def supports(self, representation: PlanningRepresentation) -> bool:
		return (
			representation.representation_source in self.supported_sources
			and representation.ordering_kind in self.supported_ordering_kinds
		)

	@abstractmethod
	def toolchain_available(self) -> bool:
		"""Whether the planner can run in the current environment."""

	@abstractmethod
	def solve(
		self,
		*,
		domain: Any,
		representation: PlanningRepresentation,
		task_name: str,
		task_args: Sequence[str],
		timeout_seconds: Optional[float],
	) -> PANDAPlanResult:
		"""Run the planner on one representation."""


class LiftedPandaSatPlanner(PrimaryHTNPlanner):
	"""Lifted PANDA SAT over a linearized total-order representation."""

	def __init__(
		self,
		workspace: Optional[str | Path] = None,
	) -> None:
		self.planner_id = "lifted_panda_sat"
		self.plan_source_label = "lifted_panda_plan"
		self.supported_sources = frozenset({"linearized"})
		self.supported_ordering_kinds = frozenset({"total_order"})
		self.planner = LiftedLinearPlanner(workspace=workspace)

	def toolchain_available(self) -> bool:
		return self.planner.panda_planner.toolchain_available()

	def solve(
		self,
		*,
		domain: Any,
		representation: PlanningRepresentation,
		task_name: str,
		task_args: Sequence[str],
		timeout_seconds: Optional[float],
	) -> PANDAPlanResult:
		solver_config = _solver_config_with_planner_budget(
			self.planner.panda_planner._solver_config_by_id(
				OFFICIAL_LIFTED_LINEAR_INNER_SOLVER_ID,
			),
		)
		return self.planner.plan_linearized_hddl_files(
			domain=domain,
			linearized_domain_file=representation.domain_file,
			linearized_problem_file=representation.problem_file,
			task_name=str(task_name),
			transition_name=representation.representation_id,
			task_args=tuple(task_args),
			timeout_seconds=timeout_seconds,
			solver_configs=(solver_config,),
			reported_solver_id=self.planner_id,
			reported_engine_mode=OFFICIAL_LIFTED_LINEAR_INNER_SOLVER_ID,
			linearization_metadata=dict(representation.metadata),
		)


def default_primary_planners(
	workspace: Optional[str | Path] = None,
) -> Tuple[PrimaryHTNPlanner, ...]:
	"""Primary HTN planner baseline used by dissertation evaluation."""

	return (
		LiftedPandaSatPlanner(workspace=workspace),
	)


def primary_planner_by_id(
	planner_id: str,
	workspace: Optional[str | Path] = None,
) -> PrimaryHTNPlanner:
	"""Instantiate the primary planner by its stable identifier."""

	for planner in default_primary_planners(workspace=workspace):
		if planner.planner_id == planner_id:
			return planner
	raise ValueError(f"Unknown primary HTN planner '{planner_id}'")


def expand_primary_planner_tasks_for_representations(
	representations: Iterable[PlanningRepresentation],
	planners: Iterable[PrimaryHTNPlanner],
) -> Tuple[PrimaryPlannerTask, ...]:
	"""Cross product between representations and applicable primary planner capabilities."""

	tasks: List[PrimaryPlannerTask] = []
	for representation in representations:
		for planner in planners:
			if not planner.supports(representation):
				continue
			task_id = _task_identifier(
				planner_id=planner.planner_id,
				representation_id=representation.representation_id,
			)
			tasks.append(
				PrimaryPlannerTask(
					task_id=task_id,
					planner_id=planner.planner_id,
					representation=representation,
				),
			)
	return tuple(tasks)


def _task_identifier(
	*,
	planner_id: str,
	representation_id: str,
) -> str:
	return (
		f"{planner_id}__{representation_id}"
		.replace("/", "_")
		.replace(" ", "_")
		.replace(":", "_")
	)
