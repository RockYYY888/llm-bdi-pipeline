"""
HTN method-library to AgentSpeak(L) plan-library translation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from method_library.synthesis.naming import sanitize_identifier
from method_library.synthesis.schema import HTNLiteral, HTNMethod, HTNMethodLibrary
from utils.hddl_condition_parser import HDDLConditionParser

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
	action_type_map = _action_type_map_for_domain(domain)
	predicate_type_map = _predicate_type_map_for_domain(domain)
	domain_method_type_map = _domain_method_parameter_type_map(domain)
	action_semantics_map = _action_semantics_map_for_domain(domain)
	task_lookup = {
		task.name: task
		for task in [*list(method_library.compound_tasks), *list(method_library.primitive_tasks)]
	}
	methods_by_task: Dict[str, List[HTNMethod]] = defaultdict(list)
	for candidate_method in method_library.methods:
		methods_by_task[str(candidate_method.task_name).strip()].append(candidate_method)
	mutable_predicates = _mutable_predicates(action_semantics_map)
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
		variable_map = _method_variable_map(
			method=method,
			task_schema=task_schema,
		)
		method_variable_types = _method_variable_type_map(
			method=method,
			task_schema=task_schema,
			task_parameter_types=task_parameter_types,
			domain_method_type_map=domain_method_type_map,
			task_type_map=task_type_map,
			action_type_map=action_type_map,
			predicate_type_map=predicate_type_map,
		)
		context_literals = _translated_context_literals(
			method=method,
			task_schema=task_schema,
			variable_map=variable_map,
			method_variable_types=method_variable_types,
			methods_by_task=methods_by_task,
			mutable_predicates=mutable_predicates,
			action_semantics_map=action_semantics_map,
		)
		plan_name = str(method.method_name).strip()
		for variant_index, ordered_steps in enumerate(ordered_step_variants, start=1):
			body = tuple(_translate_step(step, variable_map=variable_map) for step in ordered_steps)
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
					context=context_literals,
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
		task_types = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(task, "parameters", ()) or ())
		)
		mapping[task_name] = task_types
		mapping.setdefault(sanitize_identifier(task_name), task_types)
	return mapping


def _action_type_map_for_domain(domain: Any) -> Dict[str, Tuple[str, ...]]:
	mapping: Dict[str, Tuple[str, ...]] = {}
	for action in getattr(domain, "actions", ()) or ():
		action_name = str(getattr(action, "name", "") or "").strip()
		if not action_name:
			continue
		action_types = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(action, "parameters", ()) or ())
		)
		mapping[action_name] = action_types
		mapping.setdefault(sanitize_identifier(action_name), action_types)
	return mapping


def _predicate_type_map_for_domain(domain: Any) -> Dict[str, Tuple[str, ...]]:
	mapping: Dict[str, Tuple[str, ...]] = {}
	for predicate in getattr(domain, "predicates", ()) or ():
		predicate_name = str(getattr(predicate, "name", "") or "").strip()
		if not predicate_name:
			continue
		predicate_types = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(predicate, "parameters", ()) or ())
		)
		mapping[predicate_name] = predicate_types
		mapping.setdefault(sanitize_identifier(predicate_name), predicate_types)
	return mapping


def _domain_method_parameter_type_map(domain: Any) -> Dict[str, Dict[str, str]]:
	mapping: Dict[str, Dict[str, str]] = {}
	for method in getattr(domain, "methods", ()) or ():
		method_name = str(getattr(method, "name", "") or "").strip()
		if not method_name:
			continue
		typed_parameters: Dict[str, str] = {}
		for parameter in getattr(method, "parameters", ()) or ():
			for variable_name, type_name in _parameter_name_type_pairs(parameter):
				typed_parameters[variable_name] = type_name
		mapping[method_name] = typed_parameters
		mapping.setdefault(sanitize_identifier(method_name), typed_parameters)
	return mapping


def _action_semantics_map_for_domain(domain: Any) -> Dict[str, Dict[str, Any]]:
	parser = HDDLConditionParser()
	mapping: Dict[str, Dict[str, Any]] = {}
	for action in getattr(domain, "actions", ()) or ():
		action_name = str(getattr(action, "name", "") or "").strip()
		if not action_name:
			continue
		if not hasattr(action, "preconditions") or not hasattr(action, "effects"):
			continue
		try:
			parsed = parser.parse_action(action)
		except Exception:
			continue
		entry = {
			"preconditions": parsed.preconditions,
			"effects": parsed.effects,
			"parameters": parsed.parameters,
		}
		mapping[action_name] = entry
		mapping.setdefault(sanitize_identifier(action_name), entry)
	return mapping


def _mutable_predicates(action_semantics_map: Dict[str, Dict[str, Any]]) -> set[str]:
	predicates: set[str] = set()
	for action_entry in action_semantics_map.values():
		for effect in tuple(action_entry.get("effects") or ()):
			predicate = str(getattr(effect, "predicate", "") or "").strip()
			if predicate and predicate != "=":
				predicates.add(predicate)
	return predicates


def _parameter_type(parameter: str) -> str:
	text = str(parameter or "").strip()
	if ":" in text:
		return text.split(":", 1)[1].strip() or "object"
	if "-" in text:
		return text.split("-", 1)[1].strip() or "object"
	return "object"


def _parameter_name_type_pairs(parameter: Any) -> Tuple[Tuple[str, str], ...]:
	text = str(parameter or "").strip()
	if not text:
		return ()
	if " - " in text:
		raw_names, raw_type = text.split(" - ", 1)
		type_name = raw_type.strip() or "object"
		return tuple(
			(_symbol_token(name), type_name)
			for name in raw_names.split()
			if _symbol_token(name)
		)
	if ":" in text:
		raw_name, raw_type = text.split(":", 1)
		name = _symbol_token(raw_name)
		return ((name, raw_type.strip() or "object"),) if name else ()
	name = _symbol_token(text)
	return ((name, "object"),) if name else ()


def _typed_trigger_arguments(
	*,
	method: HTNMethod,
	task_schema: Any | None,
	task_parameter_types: Sequence[str],
) -> Tuple[str, ...]:
	raw_arguments = _trigger_argument_tokens(method=method, task_schema=task_schema)
	if not raw_arguments and task_schema is not None:
		raw_arguments = tuple(getattr(task_schema, "parameters", ()) or ())
	if not raw_arguments:
		raw_arguments = tuple(getattr(method, "parameters", ()) or ())
	typed_arguments: List[str] = []
	parameter_types = list(task_parameter_types)
	if len(parameter_types) < len(raw_arguments):
		parameter_types.extend(["object"] * (len(raw_arguments) - len(parameter_types)))
	used_names: set[str] = set()
	for index, raw_argument in enumerate(raw_arguments):
		token = _symbol_token(raw_argument)
		if _looks_like_variable(token):
			argument_name = _canonical_symbol_name(
				token,
				used_names=used_names,
				fallback=f"ARG{index + 1}",
			)
		else:
			argument_name = token
		type_name = parameter_types[index] if index < len(parameter_types) else "object"
		typed_arguments.append(f"{argument_name}:{type_name}")
	return tuple(typed_arguments)


def _method_variable_map(
	*,
	method: HTNMethod,
	task_schema: Any | None,
) -> Dict[str, str]:
	mapping: Dict[str, str] = {}
	used_names: set[str] = set()
	for raw_argument in _trigger_argument_tokens(method=method, task_schema=task_schema):
		token = _symbol_token(raw_argument)
		if not _looks_like_variable(token):
			continue
		mapping[token] = _canonical_symbol_name(token, used_names=used_names)

	for raw_parameter in tuple(getattr(method, "parameters", ()) or ()):
		token = _symbol_token(raw_parameter)
		if not _looks_like_variable(token) or token in mapping:
			continue
		mapping[token] = _canonical_symbol_name(token, used_names=used_names)

	for token in _method_variable_tokens(method):
		if token in mapping:
			continue
		mapping[token] = _canonical_symbol_name(token, used_names=used_names)

	return mapping


def _method_variable_type_map(
	*,
	method: HTNMethod,
	task_schema: Any | None,
	task_parameter_types: Sequence[str],
	domain_method_type_map: Dict[str, Dict[str, str]],
	task_type_map: Dict[str, Tuple[str, ...]],
	action_type_map: Dict[str, Tuple[str, ...]],
	predicate_type_map: Dict[str, Tuple[str, ...]],
) -> Dict[str, str]:
	variable_types: Dict[str, str] = {}

	def remember(variable: Any, type_name: Any) -> None:
		token = _symbol_token(variable)
		if not _looks_like_variable(token):
			return
		normalised_type = str(type_name or "").strip() or "object"
		if not normalised_type or normalised_type == "object":
			variable_types.setdefault(token, "object")
			return
		current_type = variable_types.get(token)
		if current_type is None or current_type == "object":
			variable_types[token] = normalised_type

	for raw_parameter in tuple(getattr(method, "parameters", ()) or ()):
		for variable_name, type_name in _parameter_name_type_pairs(raw_parameter):
			remember(variable_name, type_name)

	for method_key in (
		getattr(method, "source_method_name", None),
		getattr(method, "method_name", None),
		sanitize_identifier(str(getattr(method, "method_name", "") or "")),
	):
		typed_parameters = domain_method_type_map.get(str(method_key or "").strip())
		if not typed_parameters:
			continue
		for variable_name, type_name in typed_parameters.items():
			remember(variable_name, type_name)

	for argument, type_name in zip(
		_trigger_argument_tokens(method=method, task_schema=task_schema),
		task_parameter_types,
	):
		remember(argument, type_name)

	for literal in tuple(getattr(method, "context", ()) or ()):
		predicate_types = predicate_type_map.get(str(getattr(literal, "predicate", "") or ""))
		if predicate_types is None:
			continue
		for argument, type_name in zip(getattr(literal, "args", ()) or (), predicate_types):
			remember(argument, type_name)

	for step in tuple(getattr(method, "subtasks", ()) or ()):
		step_kind = str(getattr(step, "kind", "") or "").strip()
		if step_kind == "primitive":
			symbol = str(
				getattr(step, "action_name", None)
				or getattr(step, "task_name", "")
				or "",
			).strip()
			signature = action_type_map.get(symbol) or action_type_map.get(sanitize_identifier(symbol))
		else:
			symbol = str(getattr(step, "task_name", "") or "").strip()
			signature = task_type_map.get(symbol) or task_type_map.get(sanitize_identifier(symbol))
		for argument, type_name in zip(getattr(step, "args", ()) or (), signature or ()):
			remember(argument, type_name)
		for step_literal in (
			*(tuple(getattr(step, "preconditions", ()) or ())),
			*(tuple(getattr(step, "effects", ()) or ())),
		):
			predicate_types = predicate_type_map.get(
				str(getattr(step_literal, "predicate", "") or ""),
			)
			if predicate_types is None:
				continue
			for argument, type_name in zip(
				getattr(step_literal, "args", ()) or (),
				predicate_types,
			):
				remember(argument, type_name)

	return variable_types


def _translate_step(
	step: Any,
	*,
	variable_map: Dict[str, str],
) -> AgentSpeakBodyStep:
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
			_translate_term(argument, variable_map=variable_map)
			for argument in (getattr(step, "args", ()) or ())
			if str(argument).strip()
		),
	)


def _translate_context_literal(
	literal: Any,
	*,
	variable_map: Dict[str, str],
) -> str:
	predicate = str(getattr(literal, "predicate", "") or "").strip()
	args = tuple(
		_translate_term(argument, variable_map=variable_map)
		for argument in (getattr(literal, "args", ()) or ())
	)
	is_positive = bool(getattr(literal, "is_positive", True))
	if predicate == "=" and len(args) == 2:
		operator = "==" if is_positive else "!="
		return f"{args[0]} {operator} {args[1]}"
	base = predicate
	if args:
		base = f"{base}({', '.join(args)})"
	if is_positive:
		return base
	return f"!{base}"


def _translated_context_literals(
	*,
	method: HTNMethod,
	task_schema: Any | None,
	variable_map: Dict[str, str],
	method_variable_types: Dict[str, str],
	methods_by_task: Dict[str, List[HTNMethod]],
	mutable_predicates: set[str],
	action_semantics_map: Dict[str, Dict[str, Any]],
) -> Tuple[str, ...]:
	trigger_variables = {
		_symbol_token(argument)
		for argument in _trigger_argument_tokens(method=method, task_schema=task_schema)
		if _looks_like_variable(_symbol_token(argument))
	}
	local_variables = {
		token
		for token in _method_variable_tokens(method)
		if token not in trigger_variables
	}
	context_bound_variables = set(trigger_variables)
	context_bound_variables.update(_positive_context_variable_tokens(method))
	context_guard_variables = set(trigger_variables)
	context_literals: List[str] = []
	for literal in tuple(getattr(method, "context", ()) or ()):
		if _is_positive_predicate_literal(literal):
			context_guard_variables.update(_literal_variable_tokens(getattr(literal, "args", ()) or ()))
		context_literals.append(_translate_context_literal(literal, variable_map=variable_map))
	for literal in _local_witness_binding_literals(
		method=method,
		local_variables=local_variables,
		initial_bound_variables=context_bound_variables,
		methods_by_task=methods_by_task,
		mutable_predicates=mutable_predicates,
		action_semantics_map=action_semantics_map,
	):
		if _is_positive_predicate_literal(literal):
			context_guard_variables.update(_literal_variable_tokens(getattr(literal, "args", ()) or ()))
		rendered = _translate_context_literal(literal, variable_map=variable_map)
		if rendered not in context_literals:
			context_literals.append(rendered)
	type_guard_literals = _type_guard_context_literals(
		method_variable_types=method_variable_types,
		guard_variables=context_guard_variables,
		variable_map=variable_map,
	)
	context_literals = [*type_guard_literals, *context_literals]
	return tuple(sorted(context_literals, key=_context_literal_order_key))


def _is_positive_predicate_literal(literal: Any) -> bool:
	if not bool(getattr(literal, "is_positive", True)):
		return False
	predicate = str(getattr(literal, "predicate", "") or "").strip()
	return bool(predicate and predicate != "=")


def _type_guard_context_literals(
	*,
	method_variable_types: Dict[str, str],
	guard_variables: set[str],
	variable_map: Dict[str, str],
) -> List[str]:
	context_literals: List[str] = []
	seen: set[Tuple[str, str]] = set()
	for variable, type_name in method_variable_types.items():
		canonical_variable = _translate_term(variable, variable_map=variable_map)
		if variable not in guard_variables and canonical_variable not in guard_variables:
			continue
		if not _looks_like_variable(variable):
			continue
		if not type_name or type_name == "object":
			continue
		key = (canonical_variable, type_name)
		if key in seen:
			continue
		seen.add(key)
		context_literals.append(f"object_type({canonical_variable}, {type_name})")
	return context_literals


def _context_literal_order_key(literal: str) -> int:
	"""Order context literals so Jason binds variables before negation checks."""

	text = str(literal or "").strip()
	if not text:
		return 3
	if text.startswith("!") or text.lower().startswith("not ") or "!=" in text:
		return 2
	if "==" in text:
		return 1
	return 0


def _local_witness_binding_literals(
	*,
	method: HTNMethod,
	local_variables: set[str],
	initial_bound_variables: set[str],
	methods_by_task: Dict[str, List[HTNMethod]],
	mutable_predicates: set[str],
	action_semantics_map: Dict[str, Dict[str, Any]],
) -> Tuple[Any, ...]:
	binding_literals: List[Any] = []
	seen_signatures: set[str] = set()
	produced_positive_signatures: set[str] = set()
	previous_compound_arg_tuples: set[Tuple[str, ...]] = set()
	bound_variables = set(initial_bound_variables)
	prior_compound_step_seen = False
	for step in tuple(getattr(method, "subtasks", ()) or ()):
		step_kind = str(getattr(step, "kind", "") or "").strip()
		if step_kind == "compound":
			canonical_step_args = tuple(
				_lifted_child_argument(argument)
				for argument in (getattr(step, "args", ()) or ())
			)
			current_step_variables = _literal_variable_tokens(canonical_step_args)
			for precondition in _single_child_method_precondition_literals(
				step=step,
				methods_by_task=methods_by_task,
				action_semantics_map=action_semantics_map,
			):
				predicate = str(getattr(precondition, "predicate", "") or "").strip()
				if not predicate or predicate == "=":
					continue
				bound_args = tuple(str(arg) for arg in (getattr(precondition, "args", ()) or ()))
				signature = _canonical_literal_signature(predicate, bound_args)
				if signature in produced_positive_signatures:
					continue
				bound_arg_variables = _literal_variable_tokens(bound_args)
				if (
					prior_compound_step_seen
					and predicate in mutable_predicates
					and not bound_arg_variables <= current_step_variables
				):
					continue
				if (
					bound_arg_variables
					and bound_arg_variables <= bound_variables
					and _covered_by_previous_compound_subtask(
						bound_arg_variables,
						previous_compound_arg_tuples,
					)
				):
					continue
				if signature in seen_signatures:
					continue
				seen_signatures.add(signature)
				bound_variables.update(bound_arg_variables)
				binding_literals.append(precondition)
			prior_compound_step_seen = True
			previous_compound_arg_tuples.add(
				canonical_step_args,
			)
			bound_variables.update(
				_literal_variable_tokens(canonical_step_args),
			)
			produced_positive_signatures.update(
				_compound_step_positive_effect_signatures(
					step=step,
					methods_by_task=methods_by_task,
					action_semantics_map=action_semantics_map,
					stack=(),
				),
			)
			continue
		if step_kind != "primitive":
			continue
		action_name = str(
			getattr(step, "action_name", None)
			or getattr(step, "task_name", "")
			or "",
		).strip()
		action_entry = action_semantics_map.get(action_name) or action_semantics_map.get(
			sanitize_identifier(action_name),
		)
		if action_entry is None:
			continue
		action_parameters = tuple(action_entry.get("parameters") or ())
		step_args = tuple(getattr(step, "args", ()) or ())
		bindings = {
			str(parameter): str(argument)
			for parameter, argument in zip(action_parameters, step_args)
		}
		for precondition in tuple(action_entry.get("preconditions") or ()):
			if not bool(getattr(precondition, "is_positive", True)):
				continue
			predicate = str(getattr(precondition, "predicate", "") or "").strip()
			if not predicate or predicate == "=":
				continue
			if prior_compound_step_seen and predicate in mutable_predicates:
				continue
			bound_args = tuple(
				bindings.get(str(argument), str(argument))
				for argument in (getattr(precondition, "args", ()) or ())
			)
			signature = _canonical_literal_signature(predicate, bound_args)
			if signature in produced_positive_signatures:
				continue
			bound_arg_variables = _literal_variable_tokens(bound_args)
			if (
				bound_arg_variables
				and bound_arg_variables <= bound_variables
				and _covered_by_previous_compound_subtask(
					bound_arg_variables,
					previous_compound_arg_tuples,
				)
			):
				continue
			if signature in seen_signatures:
				continue
			seen_signatures.add(signature)
			bound_variables.update(bound_arg_variables)
			binding_literals.append(
				HTNLiteral(
					predicate=predicate,
					args=bound_args,
					is_positive=True,
				),
			)
		for effect in tuple(action_entry.get("effects") or ()):
			if not bool(getattr(effect, "is_positive", True)):
				continue
			predicate = str(getattr(effect, "predicate", "") or "").strip()
			if not predicate or predicate == "=":
				continue
			bound_args = tuple(
				bindings.get(str(argument), str(argument))
				for argument in (getattr(effect, "args", ()) or ())
			)
			produced_positive_signatures.add(_canonical_literal_signature(predicate, bound_args))
	return tuple(binding_literals)


def _single_child_method_precondition_literals(
	*,
	step: Any,
	methods_by_task: Dict[str, List[HTNMethod]],
	action_semantics_map: Dict[str, Dict[str, Any]],
) -> Tuple[Any, ...]:
	task_name = str(getattr(step, "task_name", "") or "").strip()
	child_methods = tuple(methods_by_task.get(task_name, ()) or ())
	if len(child_methods) != 1:
		return ()
	child_method = child_methods[0]
	child_task_args = _trigger_argument_tokens(method=child_method, task_schema=None)
	step_args = tuple(str(argument) for argument in (getattr(step, "args", ()) or ()))
	child_bindings = {
		str(parameter): str(argument)
		for parameter, argument in zip(child_task_args, step_args)
	}
	literals: List[Any] = []
	seen_signatures: set[str] = set()
	produced_positive_signatures: set[str] = set()
	for literal in tuple(getattr(child_method, "context", ()) or ()):
		if not _is_positive_predicate_literal(literal):
			continue
		bound_args = _parent_bound_child_args(
			getattr(literal, "args", ()) or (),
			child_bindings=child_bindings,
		)
		if bound_args is None:
			continue
		signature = _canonical_literal_signature(str(literal.predicate), bound_args)
		if signature in seen_signatures:
			continue
		seen_signatures.add(signature)
		literals.append(
			HTNLiteral(
				predicate=str(literal.predicate),
				args=bound_args,
				is_positive=True,
			),
		)
	for child_step in tuple(getattr(child_method, "subtasks", ()) or ()):
		if str(getattr(child_step, "kind", "") or "").strip() != "primitive":
			continue
		action_name = str(
			getattr(child_step, "action_name", None)
			or getattr(child_step, "task_name", "")
			or "",
		).strip()
		action_entry = action_semantics_map.get(action_name) or action_semantics_map.get(
			sanitize_identifier(action_name),
		)
		if action_entry is None:
			continue
		action_parameters = tuple(action_entry.get("parameters") or ())
		step_arguments = tuple(getattr(child_step, "args", ()) or ())
		action_bindings = {
			str(parameter): str(argument)
			for parameter, argument in zip(action_parameters, step_arguments)
		}
		for precondition in tuple(action_entry.get("preconditions") or ()):
			if not _is_positive_predicate_literal(precondition):
				continue
			bound_args = _parent_bound_child_args(
				(
					action_bindings.get(str(argument), str(argument))
					for argument in (getattr(precondition, "args", ()) or ())
				),
				child_bindings=child_bindings,
			)
			if bound_args is None:
				continue
			signature = _canonical_literal_signature(str(precondition.predicate), bound_args)
			if signature in produced_positive_signatures or signature in seen_signatures:
				continue
			seen_signatures.add(signature)
			literals.append(
				HTNLiteral(
					predicate=str(precondition.predicate),
					args=bound_args,
					is_positive=True,
				),
			)
		for effect in tuple(action_entry.get("effects") or ()):
			if not _is_positive_predicate_literal(effect):
				continue
			bound_args = _parent_bound_child_args(
				(
					action_bindings.get(str(argument), str(argument))
					for argument in (getattr(effect, "args", ()) or ())
				),
				child_bindings=child_bindings,
			)
			if bound_args is None:
				continue
			produced_positive_signatures.add(
				_canonical_literal_signature(str(effect.predicate), bound_args),
			)
	return tuple(literals)


def _parent_bound_child_args(
	raw_args: Iterable[Any],
	*,
	child_bindings: Dict[str, str],
) -> Tuple[str, ...] | None:
	bound_args: List[str] = []
	for argument in raw_args:
		token = _symbol_token(argument)
		if _looks_like_variable(token) and token not in child_bindings:
			return None
		bound_args.append(_lifted_child_argument(child_bindings.get(token, token)))
	return tuple(bound_args)


def _lifted_child_argument(argument: Any) -> str:
	token = _symbol_token(argument)
	if token.startswith("?"):
		return token.lstrip("?").upper()
	return token


def _canonical_literal_signature(predicate: str, args: Iterable[Any]) -> str:
	return f"{predicate}({','.join(_lifted_child_argument(argument) for argument in args)})"


def _compound_step_positive_effect_signatures(
	*,
	step: Any,
	methods_by_task: Dict[str, List[HTNMethod]],
	action_semantics_map: Dict[str, Dict[str, Any]],
	stack: Tuple[str, ...],
) -> set[str]:
	task_name = str(getattr(step, "task_name", "") or "").strip()
	if not task_name:
		return set()
	step_args = tuple(str(argument) for argument in (getattr(step, "args", ()) or ()))
	signatures: set[str] = set()
	for child_method in tuple(methods_by_task.get(task_name, ()) or ()):
		method_key = str(getattr(child_method, "method_name", "") or "").strip()
		if method_key and method_key in stack:
			continue
		child_effects = _method_positive_effect_literals(
			method=child_method,
			methods_by_task=methods_by_task,
			action_semantics_map=action_semantics_map,
			stack=stack + ((method_key or task_name),),
		)
		child_task_args = _trigger_argument_tokens(method=child_method, task_schema=None)
		bindings = {
			str(parameter): str(argument)
			for parameter, argument in zip(child_task_args, step_args)
		}
		for effect in child_effects:
			bound_args = tuple(
				_lifted_child_argument(bindings.get(str(argument), str(argument)))
				for argument in (getattr(effect, "args", ()) or ())
			)
			signatures.add(f"{effect.predicate}({','.join(bound_args)})")
	return signatures


def _method_positive_effect_literals(
	*,
	method: HTNMethod,
	methods_by_task: Dict[str, List[HTNMethod]],
	action_semantics_map: Dict[str, Dict[str, Any]],
	stack: Tuple[str, ...],
) -> Tuple[Any, ...]:
	effects: List[Any] = []
	for step in tuple(getattr(method, "subtasks", ()) or ()):
		step_kind = str(getattr(step, "kind", "") or "").strip()
		if step_kind == "primitive":
			action_name = str(
				getattr(step, "action_name", None)
				or getattr(step, "task_name", "")
				or "",
			).strip()
			action_entry = action_semantics_map.get(action_name) or action_semantics_map.get(
				sanitize_identifier(action_name),
			)
			if action_entry is None:
				continue
			action_parameters = tuple(action_entry.get("parameters") or ())
			step_args = tuple(getattr(step, "args", ()) or ())
			bindings = {
				str(parameter): str(argument)
				for parameter, argument in zip(action_parameters, step_args)
			}
			for effect in tuple(action_entry.get("effects") or ()):
				if not bool(getattr(effect, "is_positive", True)):
					continue
				predicate = str(getattr(effect, "predicate", "") or "").strip()
				if not predicate or predicate == "=":
					continue
				bound_args = tuple(
					_lifted_child_argument(bindings.get(str(argument), str(argument)))
					for argument in (getattr(effect, "args", ()) or ())
				)
				effects.append(
					HTNLiteral(
						predicate=predicate,
						args=bound_args,
						is_positive=True,
					),
				)
			continue
		if step_kind == "compound":
			for signature in _compound_step_positive_effect_signatures(
				step=step,
				methods_by_task=methods_by_task,
				action_semantics_map=action_semantics_map,
				stack=stack,
			):
				predicate, raw_args = signature.split("(", 1)
				args = tuple(arg for arg in raw_args[:-1].split(",") if arg)
				effects.append(
					HTNLiteral(
						predicate=predicate,
						args=args,
						is_positive=True,
					),
				)
	return tuple(effects)


def _positive_context_variable_tokens(method: HTNMethod) -> set[str]:
	tokens: set[str] = set()
	for literal in tuple(getattr(method, "context", ()) or ()):
		if not bool(getattr(literal, "is_positive", True)):
			continue
		predicate = str(getattr(literal, "predicate", "") or "").strip()
		if not predicate or predicate == "=":
			continue
		tokens.update(_literal_variable_tokens(getattr(literal, "args", ()) or ()))
	return tokens


def _literal_variable_tokens(values: Iterable[Any]) -> set[str]:
	return {
		token
		for token in (_symbol_token(value) for value in values)
		if _looks_like_variable(token)
	}


def _covered_by_previous_compound_subtask(
	variable_tokens: set[str],
	previous_compound_arg_tuples: set[Tuple[str, ...]],
) -> bool:
	if not variable_tokens:
		return False
	for args in previous_compound_arg_tuples:
		if variable_tokens <= _literal_variable_tokens(args):
			return True
	return False


def _translate_term(
	raw_term: Any,
	*,
	variable_map: Dict[str, str],
) -> str:
	token = _symbol_token(raw_term)
	if not _looks_like_variable(token):
		return token
	return variable_map.get(token, token)


def _trigger_argument_tokens(
	*,
	method: HTNMethod,
	task_schema: Any | None,
) -> Tuple[str, ...]:
	raw_arguments = tuple(getattr(method, "task_args", ()) or ())
	if raw_arguments:
		return tuple(_symbol_token(argument) for argument in raw_arguments if _symbol_token(argument))
	if task_schema is not None:
		schema_arguments = tuple(getattr(task_schema, "parameters", ()) or ())
		if schema_arguments:
			return tuple(
				_symbol_token(argument)
				for argument in schema_arguments
				if _symbol_token(argument)
			)
	return tuple(
		_symbol_token(argument)
		for argument in (getattr(method, "parameters", ()) or ())
		if _symbol_token(argument)
	)


def _method_variable_tokens(method: HTNMethod) -> Tuple[str, ...]:
	tokens: List[str] = []

	def collect(values: Iterable[Any]) -> None:
		for value in values:
			token = _symbol_token(value)
			if _looks_like_variable(token) and token not in tokens:
				tokens.append(token)

	collect(getattr(method, "parameters", ()) or ())
	for literal in tuple(getattr(method, "context", ()) or ()):
		collect(getattr(literal, "args", ()) or ())
	for step in tuple(getattr(method, "subtasks", ()) or ()):
		collect(getattr(step, "args", ()) or ())
		step_literal = getattr(step, "literal", None)
		if step_literal is not None:
			collect(getattr(step_literal, "args", ()) or ())
		for literal in tuple(getattr(step, "preconditions", ()) or ()):
			collect(getattr(literal, "args", ()) or ())
		for literal in tuple(getattr(step, "effects", ()) or ()):
			collect(getattr(literal, "args", ()) or ())
	return tuple(tokens)


def _symbol_token(raw_value: Any) -> str:
	text = str(raw_value or "").strip()
	if not text:
		return ""
	if text.startswith("?") and " - " in text:
		return text.split(" - ", 1)[0].strip()
	if text.startswith("?") and ":" in text:
		return text.split(":", 1)[0].strip()
	return text


def _looks_like_variable(symbol: str) -> bool:
	text = str(symbol or "").strip()
	if not text:
		return False
	if text.startswith("?"):
		return len(text) > 1 and text[1].isalpha()
	return text[0].isupper()


def _canonical_symbol_name(
	raw_symbol: str,
	*,
	used_names: set[str],
	fallback: str | None = None,
) -> str:
	token = _symbol_token(raw_symbol).lstrip("?")
	if token and token[0].isalpha():
		base_name = token.upper()
	else:
		base_name = fallback or "VAR"
	candidate = base_name
	index = 2
	while candidate in used_names:
		candidate = f"{base_name}_{index}"
		index += 1
	used_names.add(candidate)
	return candidate


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
