"""
Plan-library structural validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from method_library.synthesis.naming import sanitize_identifier
from method_library.synthesis.schema import HTNMethodLibrary
from utils.hddl_condition_parser import HDDLConditionParser

from .models import LibraryValidationRecord, PlanLibrary, TranslationCoverage


@dataclass(frozen=True)
class PlanLibraryStructuralValidation:
	"""Structured validation outcome for one generated AgentSpeak(L) plan library."""

	checked_layers: Dict[str, bool]
	warnings: Tuple[str, ...] = ()


def build_library_validation_record(
	*,
	domain_name: str,
	domain: Any,
	method_library: HTNMethodLibrary,
	plan_library: PlanLibrary,
	translation_coverage: TranslationCoverage,
	method_validation: Dict[str, Any] | None,
) -> LibraryValidationRecord:
	"""Build the validation record for a generated plan library."""

	plan_validation = validate_plan_library_structure(
		domain=domain,
		method_library=method_library,
		plan_library=plan_library,
		translation_coverage=translation_coverage,
		method_validation=method_validation,
	)
	checked_layers = dict(plan_validation.checked_layers)
	failure_reason = None
	if not all(checked_layers.values()):
		for layer_name, passed in checked_layers.items():
			if passed:
				continue
			failure_reason = f"{layer_name} failed"
			break
	elif translation_coverage.accepted_translation <= 0:
		failure_reason = "No HTN methods were accepted by the AgentSpeak(L) translation layer."

	return LibraryValidationRecord(
		library_id=domain_name,
		passed=all(checked_layers.values()) and translation_coverage.accepted_translation > 0,
		method_count=len(tuple(method_library.methods or ())),
		plan_count=len(tuple(plan_library.plans or ())),
		checked_layers=checked_layers,
		warnings=tuple(dict.fromkeys(plan_validation.warnings)),
		failure_reason=failure_reason,
	)


def validate_plan_library_structure(
	*,
	domain: Any,
	method_library: HTNMethodLibrary,
	plan_library: PlanLibrary,
	translation_coverage: TranslationCoverage,
	method_validation: Dict[str, Any] | None,
) -> PlanLibraryStructuralValidation:
	"""Validate the generated structured plan library against the workflow contract."""

	task_signatures = _symbol_signature_map(
		getattr(domain, "tasks", ()) or (),
		getattr(method_library, "compound_tasks", ()) or (),
	)
	action_signatures = _symbol_signature_map(
		getattr(domain, "actions", ()) or (),
		getattr(method_library, "primitive_tasks", ()) or (),
	)
	action_semantics_map = _action_semantics_map_for_validation(domain)
	predicate_signatures = _symbol_signature_map(getattr(domain, "predicates", ()) or (), ())
	predicate_signatures.setdefault("object_type", ("object", "object"))
	layer_results = dict((method_validation or {}).get("layers") or {})

	plan_names = [
		str(getattr(plan, "plan_name", "") or "").strip()
		for plan in tuple(plan_library.plans or ())
		if str(getattr(plan, "plan_name", "") or "").strip()
	]
	unique_plan_names = len(set(plan_names)) == len(plan_names)
	plan_checks = [
		_validate_plan(
			plan=plan,
			task_signatures=task_signatures,
			action_signatures=action_signatures,
			action_semantics_map=action_semantics_map,
			predicate_signatures=predicate_signatures,
		)
		for plan in tuple(plan_library.plans or ())
	]
	jason_functor_collisions = _jason_functor_collisions(plan_library)
	has_jason_functor_collision = bool(jason_functor_collisions)
	has_body_functor_collision = any(
		collision["category"] in {"action", "task"}
		for collision in jason_functor_collisions
	)

	signature_conformance = bool(
		((layer_results.get("signature_conformance") or {}).get("passed", True))
	) and unique_plan_names and not has_jason_functor_collision and all(
		check["signature_conformance"] for check in plan_checks
	)
	typed_structure = bool(
		((layer_results.get("typed_structural_soundness") or {}).get("passed", True))
	) and all(check["typed_structure"] for check in plan_checks)
	body_symbol_validity = bool(
		((layer_results.get("decomposition_admissibility") or {}).get("passed", True))
	) and not has_body_functor_collision and all(
		check["body_symbol_validity"] for check in plan_checks
	)
	groundability_precheck = not has_jason_functor_collision and all(
		check["groundability_precheck"] for check in plan_checks
	)

	warnings: List[str] = []
	for layer_name in (
		"signature_conformance",
		"typed_structural_soundness",
		"decomposition_admissibility",
		"materialized_parseability",
	):
		for warning in ((layer_results.get(layer_name) or {}).get("warnings") or ()):
			warning_text = str(warning).strip()
			if warning_text:
				warnings.append(warning_text)

	if not unique_plan_names:
		warnings.append("Generated plan names are not unique.")

	for check in plan_checks:
		warnings.extend(check["warnings"])
	for collision in jason_functor_collisions:
		warnings.append(
			"Jason functor collision in "
			f"{collision['category']} symbols after AgentSpeak(L) rendering: "
			f"{collision['functor']} <- {', '.join(collision['symbols'])}.",
		)

	if translation_coverage.unsupported_buckets:
		warnings.append(
			"Unsupported method constructs were excluded from the generated plan library: "
			+ ", ".join(
				f"{bucket}={count}"
				for bucket, count in sorted(translation_coverage.unsupported_buckets.items())
			),
		)
	if translation_coverage.plans_generated > translation_coverage.accepted_translation:
		warnings.append(
			"Partial-order methods were expanded into sequential plan variants in the generated "
			"AgentSpeak(L) library to preserve ordering within the supported fragment.",
		)
	auxiliary_step_semantics = _count_auxiliary_step_semantics(
		method_library=method_library,
		translation_coverage=translation_coverage,
	)
	if auxiliary_step_semantics > 0:
		warnings.append(
			"HTN step literal/precondition/effect annotations remain auxiliary in the method "
			"library; the AgentSpeak(L) translation preserves method head, context, subtasks, "
			"and ordering as specified by the workflow contract.",
		)

	return PlanLibraryStructuralValidation(
		checked_layers={
			"signature_conformance": signature_conformance,
			"typed_structure": typed_structure,
			"body_symbol_validity": body_symbol_validity,
			"groundability_precheck": groundability_precheck,
		},
		warnings=tuple(dict.fromkeys(warnings)),
	)


def _validate_plan(
	*,
	plan: Any,
	task_signatures: Dict[str, Tuple[str, ...]],
	action_signatures: Dict[str, Tuple[str, ...]],
	action_semantics_map: Dict[str, Dict[str, Any]],
	predicate_signatures: Dict[str, Tuple[str, ...]],
) -> Dict[str, Any]:
	plan_name = str(getattr(plan, "plan_name", "") or "").strip() or "<unnamed-plan>"
	warnings: List[str] = []
	signature_conformance = True
	typed_structure = True
	body_symbol_validity = True
	groundability_precheck = True

	trigger = getattr(plan, "trigger", None)
	trigger_symbol = str(getattr(trigger, "symbol", "") or "").strip()
	trigger_event_type = str(getattr(trigger, "event_type", "") or "").strip()
	trigger_arguments = tuple(getattr(trigger, "arguments", ()) or ())
	if trigger_event_type != "achievement_goal":
		signature_conformance = False
		warnings.append(f"Plan '{plan_name}' uses unsupported trigger event type '{trigger_event_type}'.")
	trigger_signature = task_signatures.get(trigger_symbol)
	if trigger_signature is None:
		signature_conformance = False
		warnings.append(f"Plan '{plan_name}' trigger '{trigger_symbol}' is not a declared task.")

	variable_types: Dict[str, set[str]] = {}
	if trigger_signature is not None and len(trigger_signature) != len(trigger_arguments):
		typed_structure = False
		warnings.append(
			f"Plan '{plan_name}' trigger '{trigger_symbol}' has arity {len(trigger_arguments)} but "
			f"the declared task signature requires {len(trigger_signature)} arguments.",
		)
	for index, raw_argument in enumerate(trigger_arguments):
		argument_name, type_name = _split_typed_argument(raw_argument)
		if not _is_agentspeak_variable(argument_name):
			typed_structure = False
			warnings.append(
				f"Plan '{plan_name}' trigger argument '{argument_name}' is not in canonical "
				"AgentSpeak(L) variable form.",
			)
		if type_name is None:
			typed_structure = False
			warnings.append(
				f"Plan '{plan_name}' trigger argument '{argument_name}' is missing a type annotation.",
			)
			continue
		variable_types.setdefault(argument_name, set()).add(type_name)
		if trigger_signature is not None and index < len(trigger_signature):
			expected_type = trigger_signature[index]
			if expected_type and type_name != expected_type:
				typed_structure = False
				warnings.append(
					f"Plan '{plan_name}' trigger argument '{argument_name}' is typed as '{type_name}' "
					f"but task '{trigger_symbol}' expects '{expected_type}'.",
				)

	bound_variables = {
		_split_typed_argument(raw_argument)[0]
		for raw_argument in trigger_arguments
		if _is_agentspeak_variable(_split_typed_argument(raw_argument)[0])
	}
	for raw_literal in tuple(getattr(plan, "context", ()) or ()):
		literal = _parse_plan_context_literal(raw_literal)
		if literal is None:
			typed_structure = False
			groundability_precheck = False
			warnings.append(f"Plan '{plan_name}' context literal '{raw_literal}' is not parseable.")
			continue
		if literal["kind"] == "equality":
			literal_variables = _context_literal_variables(literal)
			unbound_variables = sorted(literal_variables - bound_variables)
			for variable in unbound_variables:
				groundability_precheck = False
				warnings.append(
					f"Plan '{plan_name}' context equality '{raw_literal}' uses unbound "
					f"variable '{variable}'.",
				)
			for token in literal["args"]:
				if _looks_like_variable(token) and not _is_agentspeak_variable(token):
					typed_structure = False
					warnings.append(
						f"Plan '{plan_name}' context variable '{token}' is not in canonical "
						"AgentSpeak(L) variable form.",
					)
			continue
		predicate_signature = predicate_signatures.get(literal["symbol"])
		if predicate_signature is None:
			signature_conformance = False
			groundability_precheck = False
			warnings.append(
				f"Plan '{plan_name}' context predicate '{literal['symbol']}' is not declared in the domain.",
			)
			continue
		if len(predicate_signature) != len(literal["args"]):
			typed_structure = False
			groundability_precheck = False
			warnings.append(
				f"Plan '{plan_name}' context predicate '{literal['symbol']}' has arity {len(literal['args'])} "
				f"but the declared predicate signature requires {len(predicate_signature)} arguments.",
			)
			continue
		for argument, expected_type in zip(literal["args"], predicate_signature):
			if _looks_like_variable(argument):
				if not _is_agentspeak_variable(argument):
					typed_structure = False
					warnings.append(
						f"Plan '{plan_name}' context variable '{argument}' is not in canonical "
						"AgentSpeak(L) variable form.",
					)
				variable_types.setdefault(argument, set()).add(expected_type)
		literal_variables = _context_literal_variables(literal)
		if bool(literal.get("positive", True)):
			if literal["symbol"] != "object_type":
				bound_variables.update(literal_variables)
		else:
			unbound_variables = sorted(literal_variables - bound_variables)
			for variable in unbound_variables:
				groundability_precheck = False
				warnings.append(
					f"Plan '{plan_name}' negative context literal '{raw_literal}' uses unbound "
					f"variable '{variable}'.",
				)

	for step in tuple(getattr(plan, "body", ()) or ()):
		step_kind = str(getattr(step, "kind", "") or "").strip()
		step_symbol = str(getattr(step, "symbol", "") or "").strip()
		step_arguments = tuple(getattr(step, "arguments", ()) or ())
		if step_kind == "action":
			step_signature = action_signatures.get(step_symbol)
			if step_signature is None:
				body_symbol_validity = False
				signature_conformance = False
				groundability_precheck = False
				warnings.append(
					f"Plan '{plan_name}' action step '{step_symbol}' is not a declared primitive action.",
				)
				continue
		elif step_kind == "subgoal":
			step_signature = task_signatures.get(step_symbol)
			if step_signature is None:
				body_symbol_validity = False
				signature_conformance = False
				groundability_precheck = False
				warnings.append(
					f"Plan '{plan_name}' subgoal step '{step_symbol}' is not a declared compound task.",
				)
				continue
		else:
			body_symbol_validity = False
			signature_conformance = False
			groundability_precheck = False
			warnings.append(
				f"Plan '{plan_name}' step '{step_symbol}' uses unsupported kind '{step_kind}'.",
			)
			continue
		if len(step_signature) != len(step_arguments):
			typed_structure = False
			groundability_precheck = False
			warnings.append(
				f"Plan '{plan_name}' step '{step_symbol}' has arity {len(step_arguments)} but the "
				f"declared signature requires {len(step_signature)} arguments.",
			)
			continue
		for argument, expected_type in zip(step_arguments, step_signature):
			if _looks_like_variable(argument):
				if not _is_agentspeak_variable(argument):
					typed_structure = False
					warnings.append(
						f"Plan '{plan_name}' body variable '{argument}' is not in canonical "
						"AgentSpeak(L) variable form.",
					)
				variable_types.setdefault(argument, set()).add(expected_type)
		step_variables = _argument_variables(step_arguments)
		unbound_step_variables = sorted(step_variables - bound_variables)
		if step_kind == "subgoal":
			bound_variables.update(step_variables)
		else:
			action_precondition_bindings = _action_precondition_bindable_variables(
				step_symbol=step_symbol,
				step_arguments=step_arguments,
				action_semantics_map=action_semantics_map,
			)
			unsafe_unbound_variables = sorted(
				variable
				for variable in unbound_step_variables
				if variable not in action_precondition_bindings
			)
			for variable in unsafe_unbound_variables:
				groundability_precheck = False
				warnings.append(
					f"Plan '{plan_name}' {step_kind} step '{step_symbol}' uses unbound "
					f"variable '{variable}'.",
				)
			if not unsafe_unbound_variables:
				bound_variables.update(step_variables)

	for variable_name, inferred_types in variable_types.items():
		if not inferred_types:
			groundability_precheck = False
			warnings.append(
				f"Plan '{plan_name}' variable '{variable_name}' could not be assigned a domain type.",
			)

	return {
		"signature_conformance": signature_conformance,
		"typed_structure": typed_structure,
		"body_symbol_validity": body_symbol_validity,
		"groundability_precheck": groundability_precheck,
		"warnings": tuple(warnings),
	}


def _count_auxiliary_step_semantics(
	*,
	method_library: HTNMethodLibrary,
	translation_coverage: TranslationCoverage,
) -> int:
	unsupported_methods = {
		str(item.get("method_name") or "").strip()
		for item in tuple(translation_coverage.unsupported_methods or ())
		if str(item.get("method_name") or "").strip()
	}
	count = 0
	for method in tuple(method_library.methods or ()):
		method_name = str(getattr(method, "method_name", "") or "").strip()
		if method_name in unsupported_methods:
			continue
		for step in tuple(getattr(method, "subtasks", ()) or ()):
			if getattr(step, "literal", None) is not None:
				count += 1
			if tuple(getattr(step, "preconditions", ()) or ()):
				count += 1
			if tuple(getattr(step, "effects", ()) or ()):
				count += 1
	return count


def _symbol_signature_map(
	primary_symbols: Sequence[Any],
	fallback_symbols: Sequence[Any],
) -> Dict[str, Tuple[str, ...]]:
	signatures: Dict[str, Tuple[str, ...]] = {}
	for symbol in primary_symbols:
		name = str(getattr(symbol, "name", "") or "").strip()
		if not name:
			continue
		signature = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(symbol, "parameters", ()) or ())
		)
		signatures[name] = signature
		signatures.setdefault(sanitize_identifier(name), signature)
	for symbol in fallback_symbols:
		name = str(getattr(symbol, "name", "") or "").strip()
		if not name:
			continue
		signature = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(symbol, "parameters", ()) or ())
		)
		signatures.setdefault(name, signature)
		signatures.setdefault(sanitize_identifier(name), signature)
	return signatures


def _action_semantics_map_for_validation(domain: Any) -> Dict[str, Dict[str, Any]]:
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
			"parameters": parsed.parameters,
			"preconditions": parsed.preconditions,
		}
		mapping[action_name] = entry
		mapping.setdefault(sanitize_identifier(action_name), entry)
	return mapping


def _action_precondition_bindable_variables(
	*,
	step_symbol: str,
	step_arguments: Sequence[Any],
	action_semantics_map: Dict[str, Dict[str, Any]],
) -> set[str]:
	action_name = str(step_symbol or "").strip()
	action_entry = action_semantics_map.get(action_name) or action_semantics_map.get(
		sanitize_identifier(action_name),
	)
	if action_entry is None:
		return set()
	action_parameters = tuple(
		str(parameter)
		for parameter in tuple(action_entry.get("parameters") or ())
	)
	action_arguments = tuple(str(argument) for argument in tuple(step_arguments or ()))
	action_bindings: Dict[str, str] = {}
	for parameter, argument in zip(action_parameters, action_arguments):
		action_bindings[parameter] = argument
		action_bindings[_symbol_token(parameter)] = argument
	step_variables = _argument_variables(action_arguments)
	bindable_variables: set[str] = set()
	for precondition in tuple(action_entry.get("preconditions") or ()):
		if not bool(getattr(precondition, "is_positive", True)):
			continue
		predicate = str(getattr(precondition, "predicate", "") or "").strip()
		if not predicate or predicate in {"=", "object_type"}:
			continue
		bound_args = tuple(
			action_bindings.get(str(argument), str(argument))
			for argument in (getattr(precondition, "args", ()) or ())
		)
		bindable_variables.update(_argument_variables(bound_args) & step_variables)
	return bindable_variables


def _jason_functor_collisions(plan_library: PlanLibrary) -> Tuple[Dict[str, Any], ...]:
	seen_by_category: Dict[str, Dict[str, set[str]]] = {
		"task": {},
		"action": {},
		"predicate": {},
	}

	def remember(category: str, symbol: Any) -> None:
		text = str(symbol or "").strip()
		if not text:
			return
		functor = sanitize_identifier(text)
		seen_by_category.setdefault(category, {}).setdefault(functor, set()).add(text)

	for plan in tuple(plan_library.plans or ()):
		trigger = getattr(plan, "trigger", None)
		remember("task", getattr(trigger, "symbol", ""))
		for raw_literal in tuple(getattr(plan, "context", ()) or ()):
			literal = _parse_plan_context_literal(raw_literal)
			if literal is None or literal["kind"] == "equality":
				continue
			remember("predicate", literal["symbol"])
		for step in tuple(getattr(plan, "body", ()) or ()):
			step_kind = str(getattr(step, "kind", "") or "").strip()
			if step_kind == "action":
				remember("action", getattr(step, "symbol", ""))
			elif step_kind == "subgoal":
				remember("task", getattr(step, "symbol", ""))

	collisions: List[Dict[str, Any]] = []
	for category, functor_map in seen_by_category.items():
		for functor, symbols in sorted(functor_map.items()):
			if len(symbols) <= 1:
				continue
			collisions.append(
				{
					"category": category,
					"functor": functor,
					"symbols": tuple(sorted(symbols)),
				},
			)
	return tuple(collisions)


def _split_typed_argument(raw_argument: Any) -> Tuple[str, str | None]:
	text = str(raw_argument or "").strip()
	if ":" not in text:
		return text, None
	name, type_name = text.split(":", 1)
	return name.strip(), type_name.strip() or None


def _parse_plan_context_literal(raw_literal: Any) -> Dict[str, Any] | None:
	text = str(raw_literal or "").strip()
	if not text:
		return None
	positive = True
	if text.startswith("!"):
		positive = False
		text = text[1:].strip()
	if text.lower().startswith("not "):
		positive = False
		text = text[4:].strip()
	equality_match = re.fullmatch(r"(.+?)(==|!=)(.+)", text)
	if equality_match is not None:
		return {
			"kind": "equality",
			"args": (
				equality_match.group(1).strip(),
				equality_match.group(3).strip(),
			),
			"positive": equality_match.group(2) == "==" and positive,
		}
	if not text:
		return None
	if "(" not in text:
		return {"kind": "predicate", "symbol": text, "args": (), "positive": positive}
	if not text.endswith(")"):
		return None
	symbol, args_text = text.split("(", 1)
	args = tuple(
		part.strip()
		for part in args_text[:-1].split(",")
		if part.strip()
	)
	return {
		"kind": "predicate",
		"symbol": symbol.strip(),
		"args": args,
		"positive": positive,
	}


def _context_literal_variables(literal: Dict[str, Any]) -> set[str]:
	return _argument_variables(tuple(literal.get("args") or ()))


def _argument_variables(arguments: Sequence[Any]) -> set[str]:
	return {
		str(argument).strip()
		for argument in tuple(arguments or ())
		if _looks_like_variable(str(argument).strip())
	}


def _parameter_type(parameter: Any) -> str:
	text = str(parameter or "").strip()
	if not text:
		return "object"
	if ":" in text:
		return text.split(":", 1)[1].strip() or "object"
	if " - " in text:
		return text.split(" - ", 1)[1].strip() or "object"
	if "-" in text and text.startswith("?"):
		return text.split("-", 1)[1].strip() or "object"
	return "object"


def _symbol_token(raw_value: Any) -> str:
	text = str(raw_value or "").strip()
	if not text:
		return ""
	if text.startswith("?") and " - " in text:
		return text.split(" - ", 1)[0].strip()
	if text.startswith("?") and ":" in text:
		return text.split(":", 1)[0].strip()
	return text


def _looks_like_variable(token: str) -> bool:
	text = str(token or "").strip()
	if not text:
		return False
	if text.startswith("?"):
		return len(text) > 1 and text[1].isalpha()
	return text[0].isupper()


def _is_agentspeak_variable(token: str) -> bool:
	return re.fullmatch(r"[A-Z][A-Za-z0-9_]*", str(token or "").strip()) is not None
