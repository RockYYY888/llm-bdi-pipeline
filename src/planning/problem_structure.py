"""
Structure analysis for official Hierarchical Task Network problem-root planning.

This module is the semantic layer. It only inspects the original HDDL task-network
structure and does not make any planner-specific decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class TaskNetworkStructure:
	"""Structural facts for one task network."""

	network_id: str
	node_ids: Tuple[str, ...]
	ordering_edges: Tuple[Tuple[str, str], ...]
	declared_ordered: bool
	is_total_order: bool


@dataclass(frozen=True)
class ProblemStructure:
	"""Semantic structure classification for one official HTN instance."""

	root_task_network: TaskNetworkStructure
	method_task_networks: Tuple[TaskNetworkStructure, ...]
	is_total_order: bool
	requires_linearization: bool


class ProblemStructureAnalyzer:
	"""Analyze whether an HTN instance is structurally total-order or partial-order."""

	def analyze(
		self,
		*,
		domain: Any,
		problem: Any,
	) -> ProblemStructure:
		root_network = self._analyze_root_task_network(problem)
		method_networks = tuple(self._analyze_method_task_network(method) for method in self._methods(domain))
		is_total_order = root_network.is_total_order and all(
			network.is_total_order
			for network in method_networks
		)
		return ProblemStructure(
			root_task_network=root_network,
			method_task_networks=method_networks,
			is_total_order=is_total_order,
			requires_linearization=not is_total_order,
		)

	def root_ordering_edges(
		self,
		problem: Any,
	) -> Tuple[Tuple[str, str], ...]:
		return self._analyze_root_task_network(problem).ordering_edges

	def _analyze_root_task_network(self, problem: Any) -> TaskNetworkStructure:
		tasks = tuple(getattr(problem, "htn_tasks", ()) or ())
		node_ids = tuple(f"t{index}" for index in range(1, len(tasks) + 1))
		return TaskNetworkStructure(
			network_id="problem_root",
			node_ids=node_ids,
			ordering_edges=self._resolve_root_ordering_edges(problem),
			declared_ordered=bool(getattr(problem, "htn_ordered", False)),
			is_total_order=self._task_network_is_total_order(
				node_ids=node_ids,
				ordering_edges=self._resolve_root_ordering_edges(problem),
				declared_ordered=bool(getattr(problem, "htn_ordered", False)),
			),
		)

	def _analyze_method_task_network(self, method: Any) -> TaskNetworkStructure:
		subtasks = tuple(getattr(method, "subtasks", ()) or ())
		node_ids = tuple(
			str(getattr(subtask, "label", "") or "").strip()
			for subtask in subtasks
			if str(getattr(subtask, "label", "") or "").strip()
		)
		is_total_order = (
			len(node_ids) == len(subtasks)
			and self._task_network_is_total_order(
				node_ids=node_ids,
				ordering_edges=tuple(getattr(method, "ordering", ()) or ()),
				declared_ordered=False,
			)
		)
		return TaskNetworkStructure(
			network_id=str(getattr(method, "name", "unknown_method")),
			node_ids=node_ids,
			ordering_edges=tuple(getattr(method, "ordering", ()) or ()),
			declared_ordered=False,
			is_total_order=is_total_order,
		)

	@staticmethod
	def _methods(domain: Any) -> Tuple[Any, ...]:
		return tuple(getattr(domain, "methods", ()) or ())

	@staticmethod
	def _resolve_root_ordering_edges(problem: Any) -> Tuple[Tuple[str, str], ...]:
		ordering_edges = tuple(getattr(problem, "htn_ordering", ()) or ())
		if not ordering_edges:
			return ()

		label_to_runtime_id: Dict[str, str] = {}
		for index, task in enumerate(getattr(problem, "htn_tasks", ()) or (), start=1):
			label = str(getattr(task, "label", None) or f"t{index}").strip()
			if label:
				label_to_runtime_id[label] = f"t{index}"

		resolved_edges: List[Tuple[str, str]] = []
		for before, after in ordering_edges:
			before_id = label_to_runtime_id.get(str(before).strip())
			after_id = label_to_runtime_id.get(str(after).strip())
			if before_id and after_id:
				resolved_edges.append((before_id, after_id))
		return tuple(resolved_edges)

	@staticmethod
	def _task_network_is_total_order(
		*,
		node_ids: Sequence[str],
		ordering_edges: Sequence[Tuple[str, str]],
		declared_ordered: bool = False,
	) -> bool:
		canonical_node_ids = tuple(
			str(node_id).strip()
			for node_id in (node_ids or ())
			if str(node_id).strip()
		)
		if len(canonical_node_ids) <= 1:
			return True
		if declared_ordered:
			return True
		if not ordering_edges:
			return False

		reachable: Dict[str, set[str]] = {
			node_id: set()
			for node_id in canonical_node_ids
		}
		for before, after in ordering_edges:
			before_id = str(before).strip()
			after_id = str(after).strip()
			if before_id not in reachable or after_id not in reachable or before_id == after_id:
				return False
			reachable[before_id].add(after_id)

		changed = True
		while changed:
			changed = False
			for node_id in canonical_node_ids:
				expanded = set(reachable[node_id])
				for child_id in tuple(reachable[node_id]):
					expanded.update(reachable.get(child_id, set()))
				if expanded != reachable[node_id]:
					reachable[node_id] = expanded
					changed = True

		for node_id in canonical_node_ids:
			if node_id in reachable[node_id]:
				return False

		for index, left_id in enumerate(canonical_node_ids):
			for right_id in canonical_node_ids[index + 1:]:
				left_before_right = right_id in reachable[left_id]
				right_before_left = left_id in reachable[right_id]
				if left_before_right == right_before_left:
					return False
		return True
