"""
HDDL problem builder for planner-facing problem instances.

This module owns transient problem-instance construction so the PANDA planner
can stay focused on invoking the external toolchain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PANDAProblemBuilderConfig:
	"""Explicit planner-side problem-construction settings."""

	object_type: Optional[str] = None
	initial_facts: Optional[Tuple[str, ...]] = None
	global_initial_predicates: Tuple[str, ...] = ()
	per_object_initial_predicates: Tuple[str, ...] = ()


@dataclass
class PANDAProblemBuilder:
	"""Build temporary HDDL problem files for PANDA planning."""

	config: PANDAProblemBuilderConfig = field(default_factory=PANDAProblemBuilderConfig)

	def build_problem_hddl(
		self,
		*,
		domain: Any,
		domain_name: str,
		objects: Sequence[str],
		typed_objects: Optional[Sequence[Tuple[str, str]]] = None,
		htn_parameters: Optional[Sequence[Tuple[str, str]]] = None,
		task_name: str,
		task_args: Sequence[str],
		task_network: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
		task_network_ordered: bool = True,
		ordering_edges: Optional[Sequence[Tuple[str, str]]] = None,
		initial_facts: Optional[Sequence[str]] = None,
		goal_facts: Optional[Sequence[str]] = None,
	) -> str:
		task_network_entries = tuple(task_network or ())
		ordering_constraints = tuple(ordering_edges or ())
		object_list = self._select_objects(
			objects,
			task_args,
			task_network=task_network_entries,
		)
		object_type = self._resolve_object_type(domain)
		typed_object_list = self._select_typed_objects(
			typed_objects,
			object_list,
			object_type,
		)

		lines = [f"(define (problem {domain_name}_problem)"]
		lines.append(f"  (:domain {domain_name})")
		if typed_object_list:
			lines.append(f"  (:objects {self._render_typed_objects(typed_object_list)})")
		lines.append("  (:htn")
		lines.append(
			f"    :parameters {self._render_htn_parameters(htn_parameters or ())}"
		)
		if task_network_entries:
			task_lines = [
				f"(t{index} ({network_task}{self._render_problem_args(network_args)}))"
				for index, (network_task, network_args) in enumerate(task_network_entries, start=1)
			]
			if ordering_constraints:
				lines.append(f"    :subtasks (and {' '.join(task_lines)})")
				lines.append("    :ordering (and")
				for before, after in ordering_constraints:
					lines.append(f"      (< {before} {after})")
				lines.append("    )")
			else:
				subtask_keyword = ":ordered-subtasks" if task_network_ordered else ":subtasks"
				lines.append(f"    {subtask_keyword} (and {' '.join(task_lines)})")
		else:
			lines.append(
				f"    :ordered-subtasks (and (t1 ({task_name}{self._render_problem_args(task_args)})))"
			)
		lines.append("  )")
		lines.append("  (:init")
		for fact in self._build_initial_facts(domain, object_list, initial_facts):
			lines.append(f"    {fact}")
		lines.append("  )")
		lines.extend(self._render_goal_block(goal_facts))
		lines.append(")")
		return "\n".join(lines) + "\n"

	def _select_objects(
		self,
		objects: Sequence[str],
		task_args: Sequence[str],
		*,
		task_network: Sequence[Tuple[str, Sequence[str]]] = (),
	) -> List[str]:
		candidates = [
			str(obj)
			for obj in (objects or task_args)
			if str(obj).strip() and not str(obj).strip().startswith("?")
		]
		for _, network_args in task_network:
			candidates.extend(
				str(arg)
				for arg in network_args
				if str(arg).strip() and not str(arg).strip().startswith("?")
			)
		return list(dict.fromkeys(candidates))

	def _select_typed_objects(
		self,
		typed_objects: Optional[Sequence[Tuple[str, str]]],
		object_names: Sequence[str],
		default_type: str,
	) -> List[Tuple[str, str]]:
		if not typed_objects:
			if default_type == "__ambiguous__":
				raise ValueError(
					"Typed objects are required for PANDA problem export in multi-type domains. "
					"The planner request must provide explicit object->type assignments.",
				)
			return [(name, default_type) for name in object_names]

		type_lookup = {
			name: type_name
			for name, type_name in typed_objects
		}
		entries: List[Tuple[str, str]] = []
		for name in object_names:
			if name not in type_lookup:
				raise ValueError(
					f"Missing explicit type assignment for object '{name}' in PANDA export.",
				)
			entries.append((name, type_lookup[name]))
		return entries

	def _resolve_object_type(self, domain: Any) -> str:
		if self.config.object_type:
			return self.config.object_type
		raw_types = list(getattr(domain, "types", None) or [])
		normalized_types = [
			token.strip()
			for token in raw_types
			if token and token.strip() and token.strip() != "-"
		]
		unique_types = list(dict.fromkeys(normalized_types))
		if len(unique_types) == 1:
			return unique_types[0]
		if len(unique_types) > 1:
			return "__ambiguous__"
		return "object"

	def _build_initial_facts(
		self,
		domain: Any,
		objects: Sequence[str],
		initial_facts: Optional[Sequence[str]] = None,
	) -> List[str]:
		if initial_facts is not None:
			return list(initial_facts)
		if self.config.initial_facts is not None:
			return list(self.config.initial_facts)

		predicate_arity = {
			predicate.name: len(predicate.parameters)
			for predicate in getattr(domain, "predicates", [])
		}
		facts: List[str] = []

		for predicate_name in self.config.global_initial_predicates:
			if predicate_arity.get(predicate_name) == 0:
				facts.append(f"({predicate_name})")

		for predicate_name in self.config.per_object_initial_predicates:
			if predicate_arity.get(predicate_name) != 1:
				continue
			for obj in objects:
				facts.append(f"({predicate_name} {obj})")

		return facts

	@staticmethod
	def _render_goal_block(goal_facts: Optional[Sequence[str]]) -> List[str]:
		facts = [str(fact).strip() for fact in (goal_facts or ()) if str(fact).strip()]
		if not facts:
			return ["  (:goal (and))"]

		lines = ["  (:goal (and"]
		for fact in facts:
			lines.append(f"    {fact}")
		lines.append("  ))")
		return lines

	@staticmethod
	def _render_typed_objects(typed_objects: Sequence[Tuple[str, str]]) -> str:
		grouped: dict[str, List[str]] = {}
		for name, type_name in typed_objects:
			grouped.setdefault(type_name, []).append(name)

		chunks: List[str] = []
		for type_name, names in grouped.items():
			chunks.append(f"{' '.join(names)} - {type_name}")
		return " ".join(chunks)

	@staticmethod
	def _render_htn_parameters(parameters: Sequence[Tuple[str, str]]) -> str:
		if not parameters:
			return "()"
		parts = []
		for name, type_name in parameters:
			parts.append(f"{name} - {type_name}")
		return f"({' '.join(parts)})"

	@staticmethod
	def _render_problem_args(args: Iterable[str]) -> str:
		values = [str(arg) for arg in args if str(arg).strip()]
		if not values:
			return ""
		return f" {' '.join(values)}"
