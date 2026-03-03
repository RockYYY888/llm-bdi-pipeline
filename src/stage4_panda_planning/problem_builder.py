"""
Stage 4 HDDL problem builder.

This module owns the Stage 4 problem-instance construction logic so the PANDA
planner can stay focused on invoking the external toolchain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PANDAProblemBuilderConfig:
	"""Explicit Stage 4 problem-construction settings."""

	object_type: Optional[str] = None
	initial_facts: Optional[Tuple[str, ...]] = None
	global_initial_predicates: Tuple[str, ...] = ()
	per_object_initial_predicates: Tuple[str, ...] = ()


@dataclass
class PANDAProblemBuilder:
	"""Build temporary HDDL problem files for Stage 4 PANDA planning."""

	config: PANDAProblemBuilderConfig = field(default_factory=PANDAProblemBuilderConfig)

	def build_problem_hddl(
		self,
		*,
		domain: Any,
		domain_name: str,
		objects: Sequence[str],
		typed_objects: Optional[Sequence[Tuple[str, str]]] = None,
		task_name: str,
		task_args: Sequence[str],
		initial_facts: Optional[Sequence[str]] = None,
	) -> str:
		object_list = self._select_objects(objects, task_args)
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
		lines.append("    :parameters ()")
		lines.append(
			f"    :ordered-subtasks (and (t1 ({task_name}{self._render_problem_args(task_args)})))"
		)
		lines.append("  )")
		lines.append("  (:init")
		for fact in self._build_initial_facts(domain, object_list, initial_facts):
			lines.append(f"    {fact}")
		lines.append("  )")
		lines.append("  (:goal (and))")
		lines.append(")")
		return "\n".join(lines) + "\n"

	def _select_objects(
		self,
		objects: Sequence[str],
		task_args: Sequence[str],
	) -> List[str]:
		return list(dict.fromkeys(objects or task_args))

	def _select_typed_objects(
		self,
		typed_objects: Optional[Sequence[Tuple[str, str]]],
		object_names: Sequence[str],
		default_type: str,
	) -> List[Tuple[str, str]]:
		if not typed_objects:
			return [(name, default_type) for name in object_names]

		type_lookup = {
			name: type_name
			for name, type_name in typed_objects
		}
		entries: List[Tuple[str, str]] = []
		for name in object_names:
			entries.append((name, type_lookup.get(name, default_type)))
		return entries

	def _resolve_object_type(self, domain: Any) -> str:
		if self.config.object_type:
			return self.config.object_type
		if getattr(domain, "types", None):
			return domain.types[0]
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
	def _render_typed_objects(typed_objects: Sequence[Tuple[str, str]]) -> str:
		grouped: dict[str, List[str]] = {}
		for name, type_name in typed_objects:
			grouped.setdefault(type_name, []).append(name)

		chunks: List[str] = []
		for type_name, names in grouped.items():
			chunks.append(f"{' '.join(names)} - {type_name}")
		return " ".join(chunks)

	@staticmethod
	def _render_problem_args(args: Iterable[str]) -> str:
		values = [str(arg) for arg in args if str(arg).strip()]
		if not values:
			return ""
		return f" {' '.join(values)}"
