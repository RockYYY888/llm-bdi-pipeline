"""
Backend capability declarations and solver dispatch for official HTN planning.

This module is the solving layer. Backends declare which compiled representations they
can solve, and the caller is free to run all applicable backends in parallel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from planning.linearization import LiftedLinearPlanner
from planning.panda_portfolio import PANDAPlanner
from planning.plan_models import PANDAPlanResult
from planning.representations import PlanningRepresentation


def _solver_config_with_backend_budget(solver_config: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Standalone backends should inherit the full backend budget.

	Per-attempt caps are useful inside the sequential PANDA portfolio, but they
	artificially truncate standalone official backends such as PandaDealer and the
	linearized Lifted-PANDA solver.
	"""
	prepared = dict(solver_config)
	prepared.pop("timeout_seconds", None)
	return prepared


@dataclass(frozen=True)
class PlanningBackendTask:
	"""One backend invocation over one compiled representation."""

	task_id: str
	backend_name: str
	representation: PlanningRepresentation

	def to_dict(self) -> Dict[str, Any]:
		return {
			"task_id": self.task_id,
			"backend_name": self.backend_name,
			"representation": self.representation.to_dict(),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "PlanningBackendTask":
		return cls(
			task_id=str(payload["task_id"]),
			backend_name=str(payload["backend_name"]),
			representation=PlanningRepresentation.from_dict(dict(payload["representation"])),
		)


class HierarchicalPlanningBackend(ABC):
	"""Common interface for solver backends."""

	backend_name: str
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
		"""Whether the backend can run in the current environment."""

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
		"""Run the backend on one representation."""


class PandaPortfolioBackend(HierarchicalPlanningBackend):
	"""PANDA PI fixed official solver matrix."""

	def __init__(
		self,
		workspace: Optional[str | Path] = None,
	) -> None:
		self.backend_name = "panda_pi_portfolio"
		self.plan_source_label = "panda_plan"
		self.supported_sources = frozenset({"original", "linearized"})
		self.supported_ordering_kinds = frozenset({"total_order", "partial_order"})
		self.planner = PANDAPlanner(workspace=workspace)

	def toolchain_available(self) -> bool:
		return self.planner.toolchain_available()

	def solve(
		self,
		*,
		domain: Any,
		representation: PlanningRepresentation,
		task_name: str,
		task_args: Sequence[str],
		timeout_seconds: Optional[float],
	) -> PANDAPlanResult:
		return self.planner.plan_hddl_files(
			domain=domain,
			domain_file=representation.domain_file,
			problem_file=representation.problem_file,
			task_name=str(task_name),
			transition_name=representation.representation_id,
			task_args=tuple(task_args),
			target_literal=None,
			allow_empty_plan=False,
			timeout_seconds=timeout_seconds,
			solver_configs=self.planner.official_solver_portfolio_for_backend(self.backend_name),
			collect_all_candidates=True,
		)


class PandaDealerBackend(HierarchicalPlanningBackend):
	"""PandaDealer agile-lama official total-order backend."""

	def __init__(
		self,
		workspace: Optional[str | Path] = None,
	) -> None:
		self.backend_name = "pandadealer_agile_lama"
		self.plan_source_label = "panda_plan"
		self.supported_sources = frozenset({"original", "linearized"})
		self.supported_ordering_kinds = frozenset({"total_order"})
		self.planner = PANDAPlanner(workspace=workspace)

	def toolchain_available(self) -> bool:
		return self.planner.toolchain_available()

	def solve(
		self,
		*,
		domain: Any,
		representation: PlanningRepresentation,
		task_name: str,
		task_args: Sequence[str],
		timeout_seconds: Optional[float],
	) -> PANDAPlanResult:
		solver_config = _solver_config_with_backend_budget(
			self.planner._solver_config_by_id("pandadealer_agile_lama"),
		)
		return self.planner.plan_hddl_files(
			domain=domain,
			domain_file=representation.domain_file,
			problem_file=representation.problem_file,
			task_name=str(task_name),
			transition_name=representation.representation_id,
			task_args=tuple(task_args),
			target_literal=None,
			allow_empty_plan=False,
			timeout_seconds=timeout_seconds,
			solver_configs=(solver_config,),
			collect_all_candidates=True,
		)


class LiftedPandaBackend(HierarchicalPlanningBackend):
	"""Lifted PANDA SAT over a linearized total-order representation."""

	def __init__(
		self,
		workspace: Optional[str | Path] = None,
	) -> None:
		self.backend_name = "lifted_panda_sat"
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
		solver_config = _solver_config_with_backend_budget(
			self.planner.panda_planner._solver_config_by_id(
				LiftedLinearPlanner.INNER_SOLVER_ID,
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
			reported_solver_id=self.backend_name,
			reported_engine_mode=LiftedLinearPlanner.INNER_SOLVER_ID,
			linearization_metadata=dict(representation.metadata),
		)


def default_official_backends(
	workspace: Optional[str | Path] = None,
) -> Tuple[HierarchicalPlanningBackend, ...]:
	"""Backends used by the official baseline by default."""

	return (
		PandaPortfolioBackend(workspace=workspace),
		PandaDealerBackend(workspace=workspace),
		LiftedPandaBackend(workspace=workspace),
	)


def backend_by_name(
	backend_name: str,
	workspace: Optional[str | Path] = None,
) -> HierarchicalPlanningBackend:
	"""Instantiate one backend by its stable identifier."""

	for backend in default_official_backends(workspace=workspace):
		if backend.backend_name == backend_name:
			return backend
	raise ValueError(f"Unknown official planning backend '{backend_name}'")


def expand_backend_tasks_for_representations(
	representations: Iterable[PlanningRepresentation],
	backends: Iterable[HierarchicalPlanningBackend],
) -> Tuple[PlanningBackendTask, ...]:
	"""Cross product between representations and applicable backend capabilities."""

	tasks: List[PlanningBackendTask] = []
	for representation in representations:
		for backend in backends:
			if not backend.supports(representation):
				continue
			task_id = _task_identifier(
				backend_name=backend.backend_name,
				representation_id=representation.representation_id,
			)
			tasks.append(
				PlanningBackendTask(
					task_id=task_id,
					backend_name=backend.backend_name,
					representation=representation,
				),
			)
	return tuple(tasks)


def _task_identifier(
	*,
	backend_name: str,
	representation_id: str,
) -> str:
	return (
		f"{backend_name}__{representation_id}"
		.replace("/", "_")
		.replace(" ", "_")
		.replace(":", "_")
	)
