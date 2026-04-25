"""
Method-family taxonomy for official HDDL methods and prompt blueprints.

The taxonomy is intentionally structural. It classifies concrete methods by the
shape of their decomposition rather than by any benchmark-specific task name.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, Mapping, Sequence


ALREADY_SATISFIED_GUARD = "already_satisfied_guard"
DIRECT_LEAF = "direct_leaf"
SUPPORT_THEN_LEAF = "support_then_leaf"
RECURSIVE_REFINEMENT = "recursive_refinement"
HIERARCHICAL_ORCHESTRATION = "hierarchical_orchestration"


@dataclass(frozen=True)
class MethodFamilyClassification:
	method_name: str
	task_name: str
	archetype: str
	coordination_pattern: str
	final_step_name: str | None
	final_step_kind: str
	support_profile: str
	subtask_count: int
	uses_ordering: bool
	uses_auxiliary_witness: bool
	uses_self_recursion: bool
	is_already_satisfied_branch: bool


def classify_domain_methods(domain: Any) -> tuple[MethodFamilyClassification, ...]:
	task_names = {
		str(getattr(task, "name", "")).strip()
		for task in getattr(domain, "tasks", ()) or ()
		if str(getattr(task, "name", "")).strip()
	}
	action_names = {
		str(getattr(action, "name", "")).strip()
		for action in getattr(domain, "actions", ()) or ()
		if str(getattr(action, "name", "")).strip()
	}
	return tuple(
		classify_method_family(
			method,
			task_names=task_names,
			action_names=action_names,
		)
		for method in getattr(domain, "methods", ()) or ()
	)


def classify_method_family(
	method: Any,
	*,
	task_names: Iterable[str],
	action_names: Iterable[str],
) -> MethodFamilyClassification:
	task_name = str(getattr(method, "task_name", "") or "").strip()
	task_args = tuple(str(arg).strip() for arg in (getattr(method, "task_args", ()) or ()))
	task_arg_set = set(task_args)
	task_name_aliases = _name_aliases(task_name)
	task_name_pool = _expand_name_pool(task_names)
	action_name_pool = _expand_name_pool(action_names)

	subtasks = list(getattr(method, "subtasks", ()) or ())
	subtask_count = len(subtasks)
	ordered = bool(tuple(getattr(method, "ordering", ()) or ()))
	uses_self_recursion = any(
		_subtask_kind(step, task_name_pool=task_name_pool, action_name_pool=action_name_pool) == "compound"
		and _name_aliases(str(getattr(step, "task_name", "") or "").strip()) & task_name_aliases
		and tuple(str(arg).strip() for arg in (getattr(step, "args", ()) or ())) != task_args
		for step in subtasks
	)
	auxiliary_variables = {
		variable
		for variable in _variables_in_method(method)
		if variable not in task_arg_set
	}
	uses_auxiliary_witness = bool(auxiliary_variables)

	final_step = subtasks[-1] if subtasks else None
	final_step_name = (
		str(getattr(final_step, "task_name", "") or "").strip()
		if final_step is not None
		else None
	)
	final_step_kind = _subtask_kind(
		final_step,
		task_name_pool=task_name_pool,
		action_name_pool=action_name_pool,
	) if final_step is not None else "none"
	support_subtasks = subtasks[:-1] if subtasks else []
	support_kinds = [
		_subtask_kind(
			step,
			task_name_pool=task_name_pool,
			action_name_pool=action_name_pool,
		)
		for step in support_subtasks
	]
	support_profile = _support_profile(support_kinds)
	is_already_satisfied_branch = subtask_count == 0

	if is_already_satisfied_branch:
		archetype = ALREADY_SATISFIED_GUARD
	elif uses_self_recursion:
		archetype = RECURSIVE_REFINEMENT
	elif final_step_kind == "primitive" and not support_subtasks:
		archetype = DIRECT_LEAF
	elif final_step_kind == "primitive":
		archetype = SUPPORT_THEN_LEAF
	else:
		archetype = HIERARCHICAL_ORCHESTRATION

	if archetype == ALREADY_SATISFIED_GUARD:
		coordination_pattern = "already_satisfied"
	elif archetype == RECURSIVE_REFINEMENT:
		coordination_pattern = "recursive_via_witness"
	elif archetype == DIRECT_LEAF:
		coordination_pattern = "terminal_leaf"
	elif archetype == SUPPORT_THEN_LEAF and uses_auxiliary_witness:
		coordination_pattern = "via_intermediate_support_chain"
	elif archetype == SUPPORT_THEN_LEAF:
		coordination_pattern = "linear_support_chain"
	else:
		coordination_pattern = "compound_orchestration"

	return MethodFamilyClassification(
		method_name=str(getattr(method, "name", "") or "").strip(),
		task_name=task_name,
		archetype=archetype,
		coordination_pattern=coordination_pattern,
		final_step_name=final_step_name,
		final_step_kind=final_step_kind,
		support_profile=support_profile,
		subtask_count=subtask_count,
		uses_ordering=ordered,
		uses_auxiliary_witness=uses_auxiliary_witness,
		uses_self_recursion=uses_self_recursion,
		is_already_satisfied_branch=is_already_satisfied_branch,
	)


def infer_blueprint_family_archetypes(blueprint: Mapping[str, Any]) -> tuple[str, ...]:
	headline_candidates = [
		str(item).strip()
		for item in (blueprint.get("headline_candidates") or ())
		if str(item).strip()
	]
	helper_only = headline_candidates == ["helper_only"]
	headline_support_tasks = _non_none_values(
		blueprint.get("headline_support_tasks") or (),
	)
	support_calls = _non_none_values(
		blueprint.get("support_call_palette") or (),
	)
	uncovered_families = _non_none_values(
		blueprint.get("uncovered_prerequisite_families") or (),
	)
	direct_primitives = _non_none_values(
		blueprint.get("direct_primitive_achievers") or (),
	)
	method_family_schemas = [
		family
		for family in (blueprint.get("method_family_schemas") or ())
		if isinstance(family, dict)
	]
	recursive_families = [
		family
		for family in method_family_schemas
		if list(family.get("recursive_support_calls") or ())
	]
	task_signature = str(
		blueprint.get("typed_task_signature")
		or blueprint.get("task_signature")
		or "",
	).strip()

	archetypes: list[str] = []
	if not helper_only:
		archetypes.append(ALREADY_SATISFIED_GUARD)
	if recursive_families:
		archetypes.append(RECURSIVE_REFINEMENT)
	if headline_support_tasks and not helper_only:
		archetypes.append(HIERARCHICAL_ORCHESTRATION)
	if direct_primitives:
		if helper_only and not support_calls and not uncovered_families:
			archetypes.append(DIRECT_LEAF)
		elif (
			not headline_support_tasks
			and (support_calls or uncovered_families)
			and not _looks_like_composite_task_signature(task_signature)
		):
			archetypes.append(SUPPORT_THEN_LEAF)
		elif not headline_support_tasks and not (support_calls or uncovered_families):
			archetypes.append(DIRECT_LEAF)
	if support_calls:
		if _looks_like_composite_task_signature(task_signature):
			archetypes.append(HIERARCHICAL_ORCHESTRATION)
		elif HIERARCHICAL_ORCHESTRATION not in archetypes:
			archetypes.append(SUPPORT_THEN_LEAF)
	if uncovered_families and method_family_schemas and not direct_primitives:
		if _looks_like_composite_task_signature(task_signature):
			archetypes.append(HIERARCHICAL_ORCHESTRATION)
		elif HIERARCHICAL_ORCHESTRATION not in archetypes:
			archetypes.append(SUPPORT_THEN_LEAF)
	if method_family_schemas and not recursive_families and not direct_primitives:
		if _looks_like_composite_task_signature(task_signature):
			archetypes.append(HIERARCHICAL_ORCHESTRATION)
	return tuple(_unique_preserve_order(archetypes))


def _looks_like_composite_task_signature(task_signature: str) -> bool:
	name = task_signature.split("(", 1)[0].strip()
	tokens = _name_tokens(name)
	if "deliver" in tokens or "activate" in tokens:
		return True
	return "get" in tokens and any(
		token in {"soil", "rock", "image", "data"}
		for token in tokens
	)


def _non_none_values(values: Iterable[object]) -> list[str]:
	return [
		str(value).strip()
		for value in values
		if str(value).strip() and str(value).strip() != "none"
	]


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
	ordered: list[str] = []
	seen: set[str] = set()
	for value in values:
		text = str(value).strip()
		if not text or text in seen:
			continue
		seen.add(text)
		ordered.append(text)
	return ordered


def _variables_in_method(method: Any) -> tuple[str, ...]:
	variables: list[str] = []
	for parameter in getattr(method, "parameters", ()) or ():
		token = _parameter_token(parameter)
		if token:
			variables.append(token)
	for step in getattr(method, "subtasks", ()) or ():
		for arg in getattr(step, "args", ()) or ():
			text = str(arg).strip()
			if text.startswith("?"):
				variables.append(text)
	precondition = str(getattr(method, "precondition", "") or "").strip()
	for token in re.findall(r"\?[A-Za-z0-9_\\-]+", precondition):
		variables.append(token)
	return tuple(_unique_preserve_order(variables))


def _subtask_kind(
	step: Any,
	*,
	task_name_pool: set[str],
	action_name_pool: set[str],
) -> str:
	if step is None:
		return "none"
	name = str(getattr(step, "task_name", "") or "").strip()
	aliases = _name_aliases(name)
	if aliases & action_name_pool:
		return "primitive"
	if aliases & task_name_pool:
		return "compound"
	return "unknown"


def _support_profile(kinds: Sequence[str]) -> str:
	kind_set = {kind for kind in kinds if kind not in {"", "none"}}
	if not kind_set:
		return "none"
	if kind_set == {"primitive"}:
		return "primitive_only"
	if kind_set == {"compound"}:
		return "compound_only"
	return "mixed"


def _expand_name_pool(names: Iterable[str]) -> set[str]:
	pool: set[str] = set()
	for name in names:
		pool.update(_name_aliases(str(name or "").strip()))
	return pool


def _name_aliases(name: str) -> set[str]:
	text = str(name or "").strip()
	if not text:
		return set()
	sanitized = text.replace("-", "_")
	return {
		text,
		sanitized,
		text.lower(),
		sanitized.lower(),
	}


def _parameter_token(parameter: str) -> str:
	return str(parameter).split("-", 1)[0].strip()


def _name_tokens(name: str) -> tuple[str, ...]:
	return tuple(
		token
		for token in re.findall(r"[a-z0-9]+", str(name or "").replace("-", "_").lower())
		if token
	)
