"""
HTN method-library to AgentSpeak(L) plan-library translation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from method_library.synthesis.schema import HTNMethod, HTNMethodLibrary

from .models import (
	AgentSpeakBodyStep,
	AgentSpeakPlan,
	AgentSpeakTrigger,
	PlanLibrary,
	TranslationCoverage,
)


def build_plan_library(
	*,
	domain: Any,
	method_library: HTNMethodLibrary,
) -> Tuple[PlanLibrary, TranslationCoverage]:
	"""Translate HTN methods into structured AgentSpeak(L) plans."""

	task_type_map = _task_type_map_for_domain(domain)
	task_lookup = {
		task.name: task
		for task in [*list(method_library.compound_tasks), *list(method_library.primitive_tasks)]
	}
	plans: List[AgentSpeakPlan] = []
	accepted_methods = 0
	unsupported_buckets: Dict[str, int] = defaultdict(int)
	unsupported_methods: List[Dict[str, Any]] = []

	for method in method_library.methods:
		ordered_step_variants, unsupported_reason = _ordered_method_steps(method)
		if unsupported_reason is not None:
			unsupported_buckets[unsupported_reason] += 1
			unsupported_methods.append(
				{
					"method_name": method.method_name,
					"task_name": method.task_name,
					"reason": unsupported_reason,
				},
			)
			continue
		accepted_methods += 1
		task_schema = task_lookup.get(method.task_name)
		task_parameter_types = task_type_map.get(method.task_name, ())
		trigger_arguments = _typed_trigger_arguments(
			method=method,
			task_schema=task_schema,
			task_parameter_types=task_parameter_types,
		)
		plan_name = str(method.method_name).strip()
		for variant_index, ordered_steps in enumerate(ordered_step_variants, start=1):
			body = tuple(_translate_step(step) for step in ordered_steps)
			variant_plan_name = plan_name
			if len(ordered_step_variants) > 1:
				variant_plan_name = f"{plan_name}__variant_{variant_index}"
			plans.append(
				AgentSpeakPlan(
					plan_name=variant_plan_name,
					trigger=AgentSpeakTrigger(
						event_type="achievement_goal",
						symbol=str(method.task_name).strip(),
						arguments=trigger_arguments,
					),
					context=tuple(
						literal.to_signature()
						for literal in tuple(getattr(method, "context", ()) or ())
					),
					body=body,
					source_instruction_ids=tuple(
						str(value).strip()
						for value in tuple(getattr(method, "source_instruction_ids", ()) or ())
						if str(value).strip()
					),
				),
			)

	coverage = TranslationCoverage(
		domain_name=str(getattr(domain, "name", "") or ""),
		methods_considered=len(method_library.methods),
		plans_generated=len(plans),
		accepted_translation=accepted_methods,
		unsupported_buckets=dict(unsupported_buckets),
		unsupported_methods=tuple(unsupported_methods),
	)
	return PlanLibrary(
		domain_name=str(getattr(domain, "name", "") or ""),
		plans=tuple(plans),
	), coverage


def _task_type_map_for_domain(domain: Any) -> Dict[str, Tuple[str, ...]]:
	mapping: Dict[str, Tuple[str, ...]] = {}
	for task in getattr(domain, "tasks", ()) or ():
		task_name = str(getattr(task, "name", "") or "").strip()
		if not task_name:
			continue
		mapping[task_name] = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(task, "parameters", ()) or ())
		)
	return mapping


def _parameter_type(parameter: str) -> str:
	text = str(parameter or "").strip()
	if ":" in text:
		return text.split(":", 1)[1].strip() or "object"
	if "-" in text:
		return text.split("-", 1)[1].strip() or "object"
	return "object"


def _typed_trigger_arguments(
	*,
	method: HTNMethod,
	task_schema: Any | None,
	task_parameter_types: Sequence[str],
) -> Tuple[str, ...]:
	raw_arguments = tuple(getattr(method, "task_args", ()) or ())
	if not raw_arguments and task_schema is not None:
		raw_arguments = tuple(getattr(task_schema, "parameters", ()) or ())
	if not raw_arguments:
		raw_arguments = tuple(getattr(method, "parameters", ()) or ())
	typed_arguments: List[str] = []
	parameter_types = list(task_parameter_types)
	if len(parameter_types) < len(raw_arguments):
		parameter_types.extend(["object"] * (len(raw_arguments) - len(parameter_types)))
	for index, raw_argument in enumerate(raw_arguments):
		argument_name = str(raw_argument).strip().lstrip("?") or f"ARG{index + 1}"
		argument_name = argument_name.upper() if argument_name[0].isalpha() else f"ARG{index + 1}"
		type_name = parameter_types[index] if index < len(parameter_types) else "object"
		typed_arguments.append(f"{argument_name}:{type_name}")
	return tuple(typed_arguments)


def _translate_step(step: Any) -> AgentSpeakBodyStep:
	is_primitive = str(getattr(step, "kind", "") or "").strip() == "primitive"
	symbol = str(
		getattr(step, "action_name", None)
		if is_primitive and str(getattr(step, "action_name", "") or "").strip()
		else getattr(step, "task_name", "")
	).strip()
	return AgentSpeakBodyStep(
		kind="action" if is_primitive else "subgoal",
		symbol=symbol,
		arguments=tuple(
			str(argument).strip()
			for argument in (getattr(step, "args", ()) or ())
			if str(argument).strip()
		),
	)


def _ordered_method_steps(method: HTNMethod) -> Tuple[Tuple[Tuple[Any, ...], ...], str | None]:
	steps = tuple(getattr(method, "subtasks", ()) or ())
	if not steps:
		return ((),), None
	step_index = {
		str(getattr(step, "step_id", "") or "").strip(): index
		for index, step in enumerate(steps)
		if str(getattr(step, "step_id", "") or "").strip()
	}
	if len(step_index) != len(steps):
		return (), "missing_step_identifier"
	ordering = tuple(getattr(method, "ordering", ()) or ())
	if not ordering:
		return (steps,), None

	successors: Dict[str, set[str]] = {step_id: set() for step_id in step_index}
	indegree: Dict[str, int] = {step_id: 0 for step_id in step_index}
	for before_step, after_step in ordering:
		before_id = str(before_step or "").strip()
		after_id = str(after_step or "").strip()
		if before_id not in step_index or after_id not in step_index:
			return (), "ordering_references_unknown_step"
		if before_id == after_id:
			return (), "ordering_cycle"
		if after_id in successors[before_id]:
			continue
		successors[before_id].add(after_id)
		indegree[after_id] += 1

	initial_ready = tuple(
		sorted(
			(step_id for step_id, degree in indegree.items() if degree == 0),
			key=lambda step_id: step_index[step_id],
		),
	)
	ordered_step_ids = _topological_linearizations(
		ready=initial_ready,
		indegree=indegree,
		successors=successors,
		step_index=step_index,
		prefix=(),
		total_steps=len(steps),
	)
	if not ordered_step_ids:
		return (), "ordering_cycle"
	return tuple(
		tuple(steps[step_index[step_id]] for step_id in step_ids)
		for step_ids in ordered_step_ids
	), None


def _topological_linearizations(
	*,
	ready: Tuple[str, ...],
	indegree: Dict[str, int],
	successors: Dict[str, set[str]],
	step_index: Dict[str, int],
	prefix: Tuple[str, ...],
	total_steps: int,
) -> Tuple[Tuple[str, ...], ...]:
	if len(prefix) == total_steps:
		return (prefix,)
	if not ready:
		return ()

	linearizations: List[Tuple[str, ...]] = []
	for position, current in enumerate(ready):
		next_indegree = dict(indegree)
		next_ready = list(ready[:position] + ready[position + 1 :])
		for successor_id in sorted(successors[current], key=lambda step_id: step_index[step_id]):
			next_indegree[successor_id] -= 1
			if next_indegree[successor_id] == 0:
				next_ready.append(successor_id)
		next_ready = sorted(next_ready, key=lambda step_id: step_index[step_id])
		linearizations.extend(
			_topological_linearizations(
				ready=tuple(next_ready),
				indegree=next_indegree,
				successors=successors,
				step_index=step_index,
				prefix=prefix + (current,),
				total_steps=total_steps,
			)
		)
	return tuple(linearizations)
