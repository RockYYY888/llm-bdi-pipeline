"""
Minimal structural validation for domain-complete method synthesis.

This module intentionally avoids speculative semantic repair rules. It keeps only
the checks that protect parser stability, planner input well-formedness, and
obviously degenerate recursive structures.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

from offline_method_generation.method_synthesis.schema import HTNLiteral, HTNMethodLibrary


def validate_domain_complete_coverage(domain: Any, library: HTNMethodLibrary) -> None:
	declared_tasks = {
		str(getattr(task, "name", "")).replace("-", "_")
		for task in getattr(domain, "tasks", [])
		if str(getattr(task, "name", "")).strip()
	}
	library_tasks = {str(task.name).strip() for task in library.compound_tasks if str(task.name).strip()}
	missing_tasks = sorted(declared_tasks - library_tasks)
	extra_tasks = sorted(library_tasks - declared_tasks)
	if missing_tasks:
		raise ValueError(
			f"Generated library omitted declared compound tasks: {', '.join(missing_tasks)}"
		)
	if extra_tasks:
		raise ValueError(
			f"Generated library introduced undeclared compound tasks: {', '.join(extra_tasks)}"
		)


def validate_minimal_library(
	library: HTNMethodLibrary,
	domain: Any,
) -> None:
	task_lookup = {
		str(task.name).strip(): task
		for task in library.compound_tasks
		if str(getattr(task, "name", "")).strip()
	}
	primitive_names = {
		str(getattr(action, "name", "")).replace("-", "_")
		for action in getattr(domain, "actions", [])
		if str(getattr(action, "name", "")).strip()
	}
	predicate_arities = {
		str(getattr(predicate, "name", "")).strip(): len(getattr(predicate, "parameters", ()) or ())
		for predicate in getattr(domain, "predicates", [])
		if str(getattr(predicate, "name", "")).strip()
	}
	seen_method_names: set[str] = set()
	for task_name, task in task_lookup.items():
		if task_name in seen_method_names:
			raise ValueError(f"Duplicate compound task declaration '{task_name}'.")
		seen_method_names.add(task_name)
	for method in library.methods:
		method_name = str(method.method_name).strip()
		if not method_name:
			raise ValueError("Every method must have a non-empty method_name.")
		if method_name in seen_method_names:
			raise ValueError(f"Duplicate method identifier '{method_name}'.")
		seen_method_names.add(method_name)
		if method.task_name not in task_lookup:
			raise ValueError(
				f"Method '{method_name}' targets unknown compound task '{method.task_name}'."
			)
		task = task_lookup[method.task_name]
		if method.task_args and len(method.task_args) != len(task.parameters):
			raise ValueError(
				f"Method '{method_name}' binds {len(method.task_args)} task args for "
				f"task '{method.task_name}' with arity {len(task.parameters)}."
			)
		_validate_literal_collection(
			method.context,
			predicate_arities=predicate_arities,
			context_label=f"method '{method_name}' context",
		)
		step_ids: set[str] = set()
		for step in method.subtasks:
			step_id = str(step.step_id).strip()
			if not step_id:
				raise ValueError(f"Method '{method_name}' contains a step without step_id.")
			if step_id in step_ids:
				raise ValueError(
					f"Method '{method_name}' contains duplicate step_id '{step_id}'."
				)
			step_ids.add(step_id)
			step_name = str(step.task_name).strip()
			if not step_name:
				raise ValueError(
					f"Method '{method_name}' step '{step_id}' is missing its task_name."
				)
			if step.kind == "primitive":
				if step_name not in primitive_names:
					raise ValueError(
						f"Method '{method_name}' step '{step_id}' references unknown primitive "
						f"'{step_name}'."
					)
			elif step.kind == "compound":
				if step_name not in task_lookup:
					raise ValueError(
						f"Method '{method_name}' step '{step_id}' references unknown compound "
						f"'{step_name}'."
					)
				if step_name == method.task_name and tuple(step.args) == tuple(method.task_args):
					raise ValueError(
						f"Method '{method_name}' contains immediate same-argument recursion via "
						f"step '{step_id}'."
					)
			_validate_literal_collection(
				step.preconditions,
				predicate_arities=predicate_arities,
				context_label=f"method '{method_name}' step '{step_id}' preconditions",
			)
			_validate_literal_collection(
				step.effects,
				predicate_arities=predicate_arities,
				context_label=f"method '{method_name}' step '{step_id}' effects",
			)
			if step.literal is not None:
				_validate_literal_collection(
					(step.literal,),
					predicate_arities=predicate_arities,
					context_label=f"method '{method_name}' step '{step_id}' literal",
				)
		for before_step_id, after_step_id in method.ordering:
			if before_step_id not in step_ids or after_step_id not in step_ids:
				raise ValueError(
					f"Method '{method_name}' ordering references missing step ids "
					f"'{before_step_id}' -> '{after_step_id}'."
				)
			if before_step_id == after_step_id:
				raise ValueError(
					f"Method '{method_name}' contains self-ordering edge on '{before_step_id}'."
				)


def _validate_literal_collection(
	literals: Iterable[HTNLiteral],
	*,
	predicate_arities: Dict[str, int],
	context_label: str,
) -> None:
	for literal in literals:
		predicate = str(literal.predicate).strip()
		if not predicate:
			raise ValueError(f"{context_label} contains a literal without predicate name.")
		if predicate == "=":
			if len(literal.args) != 2:
				raise ValueError(
					f"{context_label} contains equality literal with arity {len(literal.args)}."
				)
			continue
		expected_arity = predicate_arities.get(predicate)
		if expected_arity is None:
			raise ValueError(
				f"{context_label} contains unknown predicate '{predicate}'."
			)
		if len(literal.args) != expected_arity:
			raise ValueError(
				f"{context_label} contains predicate '{predicate}' with arity "
				f"{len(literal.args)} but expected {expected_arity}."
			)
