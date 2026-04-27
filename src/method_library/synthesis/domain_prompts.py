"""
Domain-complete HTN prompt builders.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any, Dict, Optional, Sequence

from utils.hddl_condition_parser import HDDLConditionParser

from .method_family_taxonomy import infer_blueprint_family_archetypes
from .prompt_support import (
	_aligned_task_parameter_labels_for_predicate,
	_constructive_template_summary_for_task,
	_declared_task_schema_map,
	_dynamic_support_candidate_map,
	_dynamic_support_hint_lines,
	_extend_mapping_with_action_parameters,
	_fallback_action_template_summaries_for_task,
	_format_tagged_block,
	_limited_unique,
	_literal_pattern_signature,
	_name_tokens,
	_normalise_action_analysis,
	_parse_literal_signature,
	_parameter_token,
	_parameter_type,
	_render_signature_with_mapping,
	_render_positive_dynamic_requirements,
	_render_positive_static_requirements,
	_reusable_dynamic_resource_payloads,
	_same_arity_declared_task_candidates,
	_sanitize_name,
	_shared_dynamic_requirements_for_predicate,
	_task_headline_candidate_map,
	_task_invocation_signature,
	_typed_task_invocation_signature,
	_signature_types_can_biject,
)

def build_domain_prompt_analysis_payload(
	domain: Any,
	*,
	action_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Build compact domain-complete contracts for all declared compound tasks."""

	analysis = _normalise_action_analysis(domain, action_analysis)
	task_headline_candidates = _task_headline_candidate_map(domain, analysis)
	shared_dynamic_prerequisites_by_task: Dict[str, list[str]] = {}
	producer_consumer_templates_by_task: Dict[str, list[str]] = {}
	domain_task_contracts: list[Dict[str, Any]] = []

	for task in getattr(domain, "tasks", []):
		task_name = str(getattr(task, "name", "")).strip()
		if not task_name:
			continue
		sanitized_task_name = _sanitize_name(task_name)
		task_parameters = tuple(_parameter_token(parameter) for parameter in task.parameters)
		task_parameter_types = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(task, "parameters", ()) or ())
		)
		headline_candidates = tuple(
			dict.fromkeys(
				str(predicate_name).strip()
				for predicate_name in (
					getattr(task, "source_predicates", ())
					or task_headline_candidates.get(sanitized_task_name, ())
				)
				if str(predicate_name).strip()
			)
		)
		shared_requirements: list[str] = []
		template_summaries: list[str] = []
		used_action_fallback_templates = False
		for predicate_name in headline_candidates:
			for requirement in _shared_dynamic_requirements_for_predicate(
				predicate_name,
				task_parameters,
				analysis,
				predicate_arg_types=task_parameter_types,
			):
				if requirement not in shared_requirements:
					shared_requirements.append(requirement)
			constructive_template = _constructive_template_summary_for_task(
				task_name,
				task_parameters,
				predicate_name,
				analysis,
				task_parameter_types=task_parameter_types,
			)
			if constructive_template and not (
				not shared_requirements and "; " in constructive_template
			) and constructive_template not in template_summaries:
				template_summaries.append(constructive_template)
		if not template_summaries:
			template_summaries = _fallback_action_template_summaries_for_task(
				task_name,
				task_parameters,
				task_parameter_types,
				analysis,
			)
			used_action_fallback_templates = bool(template_summaries)
		if used_action_fallback_templates:
			# When we only recover a task through action-level fallback, the
			# inferred predicate headline is too weak to be treated as a real
			# task headline. Keep the prompt helper-oriented instead of forcing
			# a misleading predicate such as at(...).
			headline_candidates = ()
			shared_requirements = []
		shared_requirements = _limited_unique(shared_requirements, limit=4)
		template_summaries = _limited_unique(template_summaries, limit=2)
		shared_dynamic_prerequisites_by_task[sanitized_task_name] = shared_requirements
		producer_consumer_templates_by_task[sanitized_task_name] = template_summaries
		domain_task_contracts.append(
			{
				"task_name": task_name,
				"task_signature": _task_invocation_signature(task_name, task_parameters),
				"typed_task_signature": _typed_task_invocation_signature(
					task_name,
					getattr(task, "parameters", ()) or (),
				),
				"parameters": list(task_parameters),
				"parameter_types": list(task_parameter_types),
				"headline_candidates": list(headline_candidates),
				"shared_dynamic_prerequisites": list(shared_requirements),
				"producer_consumer_templates": list(template_summaries),
				"used_action_fallback_templates": used_action_fallback_templates,
				"composition_support_tasks": [],
				"recursive_support_predicates": [],
				"prerequisite_acquisition_templates": [],
			}
		)

	support_task_palette = [
		(
			str(contract.get("task_name") or "").strip(),
			str(contract.get("task_signature") or "").strip(),
			list(contract.get("headline_candidates") or ()),
		)
		for contract in domain_task_contracts
		if list(contract.get("headline_candidates") or ())
	]
	task_schemas = _declared_task_schema_map(domain)
	generic_resource_predicates = {
		str(payload.get("predicate") or "").strip()
		for payload in _reusable_dynamic_resource_payloads(analysis)
		if str(payload.get("predicate") or "").strip()
	}
	palette_eligible_generic_predicates = {
		predicate_name
		for predicate_name in generic_resource_predicates
		if any(
			predicate_name in candidate_headlines
			for _, _, candidate_headlines in support_task_palette
		)
	}
	task_signature_by_name = {
		str(contract.get("task_name") or "").strip(): str(contract.get("task_signature") or "").strip()
		for contract in domain_task_contracts
		if str(contract.get("task_name") or "").strip()
	}
	task_headlines_by_name = {
		str(contract.get("task_name") or "").strip(): list(contract.get("headline_candidates") or ())
		for contract in domain_task_contracts
		if str(contract.get("task_name") or "").strip()
	}
	support_candidates_by_predicate = _dynamic_support_candidate_map(domain, analysis)
	for contract in domain_task_contracts:
		task_name = str(contract.get("task_name") or "").strip()
		task_signature = str(contract.get("task_signature") or "").strip()
		task_schema = task_schemas.get(task_name)
		task_parameter_types = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(task_schema, "parameters", ()) or ())
		)
		relevant_predicates = {
			str(predicate_name).strip()
			for predicate_name in list(contract.get("headline_candidates") or ())
			if str(predicate_name).strip()
		}
		relevant_predicates.update(
			str(requirement).split("(", 1)[0].strip()
			for requirement in list(contract.get("shared_dynamic_prerequisites") or ())
			if str(requirement).strip()
			and "(" in str(requirement)
			and (
				str(requirement).split("(", 1)[0].strip() not in generic_resource_predicates
				or str(requirement).split("(", 1)[0].strip() in palette_eligible_generic_predicates
			)
		)
		composition_support_tasks: list[str] = []
		recursive_support_predicates: list[str] = []
		for candidate_name, candidate_signature, candidate_headlines in support_task_palette:
			if not candidate_signature or not candidate_headlines:
				continue
			candidate_schema = task_schemas.get(candidate_name)
			candidate_parameter_types = tuple(
				_parameter_type(parameter)
				for parameter in (getattr(candidate_schema, "parameters", ()) or ())
			)
			if (
				len(task_parameter_types) == len(candidate_parameter_types)
				and task_parameter_types
				and candidate_parameter_types
				and not _signature_types_can_biject(
					task_parameter_types,
					candidate_parameter_types,
					analysis.get("type_parent_map", {}) or {},
				)
				and not _signature_types_can_biject(
					candidate_parameter_types,
					task_parameter_types,
					analysis.get("type_parent_map", {}) or {},
				)
			):
				continue
			headline_overlap = [
				headline
				for headline in candidate_headlines
				if headline in relevant_predicates
			]
			if not headline_overlap:
				continue
			if candidate_signature == task_signature:
				entry = (
					f"{candidate_signature} stabilizes {', '.join(headline_overlap)} "
					"(recursive reuse allowed)"
				)
			else:
				entry = f"{candidate_signature} stabilizes {', '.join(headline_overlap)}"
			if entry not in composition_support_tasks:
				composition_support_tasks.append(entry)
			if candidate_signature == task_signature:
				for headline in headline_overlap:
					if headline not in recursive_support_predicates:
						recursive_support_predicates.append(headline)
		for predicate_name in sorted(relevant_predicates):
			for candidate_name in support_candidates_by_predicate.get(predicate_name, []):
				candidate_signature = task_signature_by_name.get(candidate_name)
				candidate_headlines = task_headlines_by_name.get(candidate_name) or []
				candidate_schema = task_schemas.get(candidate_name)
				candidate_parameter_types = tuple(
					_parameter_type(parameter)
					for parameter in (getattr(candidate_schema, "parameters", ()) or ())
				)
				if not candidate_signature:
					continue
				if predicate_name not in candidate_headlines:
					continue
				if (
					len(task_parameter_types) == len(candidate_parameter_types)
					and task_parameter_types
					and candidate_parameter_types
					and not _signature_types_can_biject(
						task_parameter_types,
						candidate_parameter_types,
						analysis.get("type_parent_map", {}) or {},
					)
					and not _signature_types_can_biject(
						candidate_parameter_types,
						task_parameter_types,
						analysis.get("type_parent_map", {}) or {},
					)
				):
					continue
				if candidate_signature == task_signature:
					entry = (
						f"{candidate_signature} stabilizes {predicate_name} "
						"(recursive reuse allowed)"
					)
				else:
					entry = f"{candidate_signature} stabilizes {predicate_name}"
				if entry not in composition_support_tasks:
					composition_support_tasks.append(entry)
				if candidate_signature == task_signature and predicate_name not in recursive_support_predicates:
					recursive_support_predicates.append(predicate_name)
		if not composition_support_tasks and task_name:
			task_schema = task_schemas.get(task_name)
			task_parameter_types = {
				_parameter_type(parameter)
				for parameter in (getattr(task_schema, "parameters", ()) or ())
			}
			for candidate_name in _same_arity_declared_task_candidates(
				domain,
				task_name,
				task_schemas,
			):
				candidate_signature = task_signature_by_name.get(candidate_name)
				candidate_headlines = task_headlines_by_name.get(candidate_name) or []
				if not candidate_signature or not candidate_headlines:
					continue
				entry = f"{candidate_signature} stabilizes {', '.join(candidate_headlines)}"
				if entry not in composition_support_tasks:
					composition_support_tasks.append(entry)
			for candidate_name, candidate_signature, candidate_headlines in support_task_palette:
				if candidate_name == task_name or not candidate_headlines:
					continue
				candidate_schema = task_schemas.get(candidate_name)
				candidate_parameter_types = {
					_parameter_type(parameter)
					for parameter in (getattr(candidate_schema, "parameters", ()) or ())
				}
				if not candidate_parameter_types or not candidate_parameter_types.issubset(task_parameter_types):
					continue
				entry = f"{candidate_signature} stabilizes {', '.join(candidate_headlines)}"
				if entry not in composition_support_tasks:
					composition_support_tasks.append(entry)
		contract["composition_support_tasks"] = (
			_limited_unique(composition_support_tasks, limit=6) or ["none"]
		)
		contract["recursive_support_predicates"] = recursive_support_predicates or ["none"]
		contract["prerequisite_acquisition_templates"] = (
			_render_prerequisite_acquisition_templates(
				list(contract.get("shared_dynamic_prerequisites") or ()),
				analysis,
			)
			or ["none"]
		)
	method_blueprints = _build_method_blueprints(
		action_analysis=analysis,
		domain_task_contracts=domain_task_contracts,
		task_schemas=task_schemas,
		support_candidates_by_predicate=support_candidates_by_predicate,
		domain=domain,
	)

	return {
		"declared_compound_tasks": [
			_task_invocation_signature(
				str(getattr(task, "name", "")).strip(),
				tuple(_parameter_token(parameter) for parameter in task.parameters),
			)
			for task in getattr(domain, "tasks", [])
			if str(getattr(task, "name", "")).strip()
		],
		"typed_declared_compound_tasks": [
			_typed_task_invocation_signature(
				str(getattr(task, "name", "")).strip(),
				tuple(getattr(task, "parameters", ()) or ()),
			)
			for task in getattr(domain, "tasks", [])
			if str(getattr(task, "name", "")).strip()
		],
		"task_headline_candidates": task_headline_candidates,
		"shared_dynamic_prerequisites_by_task": shared_dynamic_prerequisites_by_task,
		"producer_consumer_templates_by_task": producer_consumer_templates_by_task,
		"reusable_dynamic_resource_predicates": _reusable_dynamic_resource_payloads(analysis),
		"domain_task_contracts": domain_task_contracts,
		"method_blueprints": method_blueprints,
	}


def _type_can_fill_slot(parent_type: str, child_type: str) -> bool:
	parent = str(parent_type or "object").strip().lower()
	child = str(child_type or "object").strip().lower()
	return _signature_types_can_biject((parent,), (child,), {})


def _enumerate_aligned_support_calls(
	parent_parameters: Sequence[str],
	parent_parameter_types: Sequence[str],
	child_signature: str,
	child_parameter_types: Sequence[str],
	*,
	limit: int = 4,
) -> list[str]:
	if not child_parameter_types:
		return [child_signature]
	if len(child_parameter_types) > len(parent_parameter_types):
		return []

	aligned_calls: list[str] = []
	parent_positions = range(len(parent_parameters))
	for indices in combinations(parent_positions, len(child_parameter_types)):
		candidate_args = [parent_parameters[index] for index in indices]
		candidate_types = [parent_parameter_types[index] for index in indices]
		if not all(
			_type_can_fill_slot(parent_type, child_type)
			for parent_type, child_type in zip(candidate_types, child_parameter_types)
		):
			continue
		aligned_calls.append(_task_invocation_signature(child_signature, tuple(candidate_args)))
		if len(aligned_calls) >= limit:
			break
	return aligned_calls


def _build_method_blueprints(
	*,
	action_analysis: Dict[str, Any],
	domain_task_contracts: Sequence[Dict[str, Any]],
	task_schemas: Dict[str, Any],
	support_candidates_by_predicate: Dict[str, list[str]],
	domain: Any,
) -> list[Dict[str, Any]]:
	_ = support_candidates_by_predicate
	contracts_by_name = {
		str(contract.get("task_name") or "").strip(): dict(contract)
		for contract in domain_task_contracts
		if str(contract.get("task_name") or "").strip()
	}
	blueprints: list[Dict[str, Any]] = []
	for task_name, contract in contracts_by_name.items():
		task_parameters = list(contract.get("parameters") or ())
		task_parameter_types = list(contract.get("parameter_types") or ())
		headline_candidates = [
			str(item).strip()
			for item in (contract.get("headline_candidates") or ())
			if str(item).strip() and str(item).strip() != "none"
		]
		shared_dynamic_prerequisites = [
			str(item).strip()
			for item in (contract.get("shared_dynamic_prerequisites") or ())
			if str(item).strip() and str(item).strip() != "none"
		]
		producer_templates = [
			str(item).strip()
			for item in (contract.get("producer_consumer_templates") or ())
			if str(item).strip() and str(item).strip() != "none"
		]
		recursive_support_predicates = [
			str(item).strip()
			for item in (contract.get("recursive_support_predicates") or ())
			if str(item).strip() and str(item).strip() != "none"
		]

		support_call_palette: list[str] = []
		uncovered_prerequisite_families: list[str] = []
		for requirement in shared_dynamic_prerequisites:
			parsed_requirement = _parse_literal_signature(requirement)
			if parsed_requirement is None:
				continue
			predicate_name, _, is_positive = parsed_requirement
			if not is_positive:
				continue
			candidate_calls: list[str] = []
			for candidate_name, candidate_contract in contracts_by_name.items():
				candidate_headlines = {
					str(item).strip()
					for item in (candidate_contract.get("headline_candidates") or ())
					if str(item).strip() and str(item).strip() != "none"
				}
				if predicate_name not in candidate_headlines:
					continue
				candidate_schema = task_schemas.get(candidate_name)
				if candidate_schema is None:
					continue
				candidate_signature = _sanitize_name(candidate_name)
				child_parameter_types = [
					_parameter_type(parameter)
					for parameter in (getattr(candidate_schema, "parameters", ()) or ())
				]
				candidate_calls.extend(
					_enumerate_aligned_support_calls(
						task_parameters,
						task_parameter_types,
						candidate_signature,
						child_parameter_types,
					),
				)
			candidate_calls = _limited_unique(candidate_calls, limit=4)
			if candidate_calls:
				support_call_palette.extend(candidate_calls)
			family_lines = [
				str(item).strip()
				for item in (contract.get("prerequisite_acquisition_templates") or ())
				if str(item).strip().startswith(f"{requirement} via ")
			]
			if not candidate_calls and family_lines:
				uncovered_prerequisite_families.extend(family_lines)
		method_family_schemas = _render_method_family_schemas(
			action_analysis=action_analysis,
			headline_candidates=headline_candidates,
			recursive_support_predicates=recursive_support_predicates,
			task_name=task_name,
			task_parameters=task_parameters,
			task_parameter_types=task_parameter_types,
		)
		if bool(contract.get("used_action_fallback_templates")):
			method_family_schemas = []
		method_family_schemas = _prune_method_family_schemas(
			method_family_schemas,
			task_parameters=task_parameters,
			shared_dynamic_prerequisites=shared_dynamic_prerequisites,
		)
		if (
			not shared_dynamic_prerequisites
			and method_family_schemas
			and all(
				isinstance(family, dict)
				and str(family.get("family_role") or "") == "recursive_blocker_removal"
				for family in method_family_schemas
			)
		):
			producer_templates = []
		support_task_hints = [
			str(item).strip()
			for item in (contract.get("composition_support_tasks") or ())
			if str(item).strip()
			and str(item).strip() != "none"
			and not str(item).strip().startswith(f"{contract.get('task_signature')} ")
		]
		task_name_tokens = set(_name_tokens(task_name))
		headline_named_task = any(
			set(_name_tokens(headline)).issubset(task_name_tokens)
			for headline in headline_candidates
			if str(headline).strip()
		)
		headline_support_tasks = [
			entry
			for entry in support_task_hints
			if any(
				f" stabilizes {headline}" in entry
				for headline in headline_candidates
			)
		] if headline_named_task else []
		if not headline_candidates:
			support_task_hints = []
			headline_support_tasks = []
		support_call_palette = _prune_support_call_palette(
			support_call_palette,
			task_signature=str(contract.get("task_signature") or "").strip(),
		)
		preferred_family_shape = _classify_preferred_family_shape(
			support_call_palette=support_call_palette,
			support_task_hints=support_task_hints,
			uncovered_prerequisite_families=uncovered_prerequisite_families,
			direct_primitive_achievers=producer_templates,
			method_family_schemas=method_family_schemas,
		)
		witness_binding_required = _payload_contains_auxiliary_witnesses(
			producer_templates,
			method_family_schemas,
		)
		blueprint_direct_primitive_achievers = (
			[]
			if method_family_schemas
			else producer_templates
		)

		blueprints.append(
			{
				"task_name": task_name,
				"task_signature": str(contract.get("task_signature") or "").strip(),
				"typed_task_signature": str(contract.get("typed_task_signature") or "").strip(),
				"headline_candidates": headline_candidates or ["helper_only"],
				"preferred_family_shape": preferred_family_shape,
				"witness_binding_required": witness_binding_required,
				"headline_support_tasks": _limited_unique(headline_support_tasks, limit=4) or ["none"],
				"support_call_palette": _limited_unique(support_call_palette, limit=6) or ["none"],
				"support_task_hints": _limited_unique(support_task_hints, limit=4) or ["none"],
				"direct_primitive_achievers": blueprint_direct_primitive_achievers or ["none"],
				"uncovered_prerequisite_families": _limited_unique(
					uncovered_prerequisite_families,
					limit=4,
				) or ["none"],
				"method_family_schemas": method_family_schemas or ["none"],
			},
		)
		blueprints[-1]["family_archetypes"] = list(
			infer_blueprint_family_archetypes(blueprints[-1]),
		) or [preferred_family_shape]
	return blueprints


def _payload_contains_auxiliary_witnesses(*payloads: object) -> bool:
	def _contains_aux(value: object) -> bool:
		if isinstance(value, dict):
			return any(_contains_aux(item) for item in value.values())
		if isinstance(value, (list, tuple, set)):
			return any(_contains_aux(item) for item in value)
		return "AUX_" in str(value or "")

	return any(_contains_aux(payload) for payload in payloads)


def _prune_support_call_palette(
	support_call_palette: Sequence[str],
	*,
	task_signature: str,
) -> list[str]:
	normalised_task_signature = _normalise_task_call_signature(task_signature)
	return [
		call
		for call in _limited_unique(support_call_palette, limit=6)
		if str(call).strip()
		and str(call).strip() != "none"
		and _normalise_task_call_signature(call) != normalised_task_signature
	]


def _normalise_task_call_signature(signature: object) -> str:
	text = str(signature or "").strip()
	match = re.fullmatch(r"([^(]+)\((.*)\)", text)
	if match is None:
		return _sanitize_name(text)
	name = _sanitize_name(match.group(1).strip())
	args = ",".join(str(arg).strip() for arg in match.group(2).split(","))
	return f"{name}({args})"


def _classify_preferred_family_shape(
	*,
	support_call_palette: Sequence[str],
	support_task_hints: Sequence[str],
	uncovered_prerequisite_families: Sequence[str],
	direct_primitive_achievers: Sequence[str],
	method_family_schemas: Sequence[object],
) -> str:
	recursive_families = [
		family
		for family in method_family_schemas
		if isinstance(family, dict)
		and str(family.get("family_role") or "") == "recursive_blocker_removal"
	]
	if recursive_families and len(recursive_families) == len(
		[item for item in method_family_schemas if isinstance(item, dict)],
	):
		return "recursive_blocker_removal"
	has_support_calls = any(str(item).strip() and str(item).strip() != "none" for item in support_call_palette)
	has_support_hints = any(str(item).strip() and str(item).strip() != "none" for item in support_task_hints)
	has_uncovered_families = any(
		str(item).strip() and str(item).strip() != "none"
		for item in uncovered_prerequisite_families
	)
	if has_support_calls or has_support_hints or has_uncovered_families:
		return "support_then_final"
	return "direct_leaf"


def _prune_method_family_schemas(
	families: Sequence[Dict[str, Any]],
	*,
	task_parameters: Sequence[str],
	shared_dynamic_prerequisites: Sequence[str],
) -> list[Dict[str, Any]]:
	if shared_dynamic_prerequisites:
		return [dict(family) for family in families if isinstance(family, dict)]
	recursive_families = [
		dict(family)
		for family in families
		if isinstance(family, dict)
		and str(family.get("family_role") or "") == "recursive_blocker_removal"
	]
	if not recursive_families:
		return [dict(family) for family in families if isinstance(family, dict)]
	task_parameter_set = {str(parameter).strip() for parameter in task_parameters if str(parameter).strip()}
	relational_recursive_families = [
		family
		for family in recursive_families
		if any(
			(
				(parsed_literal := _parse_literal_signature(str(signature))) is not None
				and parsed_literal[2]
				and len(parsed_literal[1]) >= 2
				and any(arg in task_parameter_set for arg in parsed_literal[1])
				and any(arg not in task_parameter_set for arg in parsed_literal[1])
			)
			for signature in (family.get("context") or ())
		)
	]
	if relational_recursive_families:
		return relational_recursive_families
	return recursive_families


def _render_method_blueprint_blocks(method_blueprints: Sequence[Dict[str, Any]]) -> str:
	serializable_payloads: list[Dict[str, Any]] = []
	for payload in method_blueprints:
		headline_candidates = list(payload["headline_candidates"])
		serializable: Dict[str, Any] = {
			"task": payload["typed_task_signature"] or payload["task_signature"],
			"preferred_family_shape": str(payload.get("preferred_family_shape") or "direct_leaf"),
		}
		if bool(payload.get("witness_binding_required")):
			serializable["witness_binding_required"] = True
		archetypes = [
			str(item).strip()
			for item in (payload.get("family_archetypes") or ())
			if str(item).strip()
		]
		if archetypes:
			serializable["family_archetypes"] = archetypes
		if headline_candidates == ["helper_only"]:
			pass
		elif len(headline_candidates) == 1:
			serializable["headline"] = headline_candidates[0]
		else:
			serializable["headlines"] = headline_candidates
		for key in (
			"headline_support_tasks",
			"uncovered_prerequisite_families",
		):
			if key == "uncovered_prerequisite_families":
				values = []
				for item in (payload.get(key) or ()):
					compact_value = _compact_uncovered_prerequisite_family(item)
					if compact_value is not None:
						values.append(compact_value)
			else:
				values = [
					str(item).strip()
					for item in (payload.get(key) or ())
					if str(item).strip() and str(item).strip() != "none"
				]
			if values:
				serializable[key] = values
		family_schemas = [
			_compact_method_family_schema(family)
			for family in (payload.get("method_family_schemas") or ())
			if isinstance(family, dict)
		]
		if family_schemas:
			serializable["method_family_schemas"] = family_schemas
		else:
			values = []
			for item in (payload.get("direct_primitive_achievers") or ()):
				compact_value = _compact_direct_primitive_achiever(item)
				if compact_value is not None:
					values.append(compact_value)
			if values:
				serializable["direct_primitive_achievers"] = values
		serializable_payloads.append(serializable)
	return json.dumps(serializable_payloads, separators=(",", ":"))


def _compact_uncovered_prerequisite_family(item: object) -> Dict[str, str] | None:
	text = str(item or "").strip()
	if not text or text == "none":
		return None
	match = re.fullmatch(r"(.+?) via (.+?)(?: \[needs .+\])?", text)
	if match is None:
		return {"need": text}
	return {
		"need": match.group(1).strip(),
		"primitive_action": match.group(2).strip(),
		"support_kind": "primitive_support",
	}


def _compact_direct_primitive_achiever(item: object) -> Dict[str, str] | None:
	text = str(item or "").strip()
	if not text or text == "none":
		return None
	match = re.fullmatch(r"(.+?)(?: \[needs (.+)\])?", text)
	if match is None:
		return {"primitive_action": text, "support_kind": "primitive_leaf"}
	serializable = {
		"primitive_action": match.group(1).strip(),
		"support_kind": "primitive_leaf",
	}
	needs = str(match.group(2) or "").strip()
	if needs:
		serializable["needs"] = needs
	return serializable


def _compact_method_family_schema(family: Dict[str, Any]) -> Dict[str, Any]:
	serializable = dict(family)
	if str(serializable.get("family_role") or "").strip() == "direct_achiever":
		serializable.pop("family_role", None)
	if serializable.get("redundant_if_already_satisfied") is False:
		serializable.pop("redundant_if_already_satisfied", None)
	return serializable


def _render_cleanup_steps_for_family(
	*,
	action_analysis: Dict[str, Any],
	headline_signature: str,
	pattern: Dict[str, Any],
	task_parameter_set: set[str],
	token_mapping: Dict[str, str],
) -> list[str]:
	cleanup_candidates: list[tuple[int, str]] = []
	for effect_signature in (pattern.get("positive_effect_signatures") or ()):
		rendered_effect = _render_signature_with_mapping(effect_signature, token_mapping)
		if rendered_effect == headline_signature:
			continue
		parsed_effect = _parse_literal_signature(rendered_effect)
		if parsed_effect is None or not parsed_effect[2]:
			continue
		predicate_name, effect_args, _ = parsed_effect
		if not effect_args or not any(arg not in task_parameter_set for arg in effect_args):
			continue
		for consumer_pattern in (
			action_analysis.get("consumer_patterns_by_predicate", {}).get(predicate_name, ())
		):
			precondition_args = list(consumer_pattern.get("precondition_args") or ())
			if len(precondition_args) != len(effect_args):
				continue
			consumer_mapping = {
				token: arg
				for token, arg in zip(precondition_args, effect_args)
			}
			rendered_action_args = _extend_mapping_with_action_parameters(
				consumer_mapping,
				consumer_pattern.get("action_parameters") or (),
				action_parameter_types=consumer_pattern.get("action_parameter_types") or (),
			)
			other_needs = [
				_render_signature_with_mapping(signature, consumer_mapping)
				for signature in (consumer_pattern.get("other_dynamic_precondition_signatures") or ())
				if str(signature).strip() and not str(signature).startswith("not ")
			]
			cleanup_candidates.append(
				(
					len(other_needs),
					_task_invocation_signature(
						str(consumer_pattern.get("action_name") or "").strip(),
						rendered_action_args,
					),
				),
			)
	cleanup_candidates.sort(key=lambda item: (item[0], item[1]))
	return _limited_unique((call for _, call in cleanup_candidates), limit=2)


def _render_method_family_schemas(
	*,
	action_analysis: Dict[str, Any],
	headline_candidates: Sequence[str],
	recursive_support_predicates: Sequence[str],
	task_name: str,
	task_parameters: Sequence[str],
	task_parameter_types: Sequence[str],
) -> list[Dict[str, Any]]:
	task_parameter_set = set(task_parameters)
	recursive_predicates = set(recursive_support_predicates)
	families: list[Dict[str, Any]] = []
	seen_payloads: set[str] = set()
	for predicate_name in headline_candidates:
		for pattern in (action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, ())):
			effect_args = list(pattern.get("effect_args") or ())
			if not effect_args or len(effect_args) > len(task_parameters):
				continue
			aligned_task_parameters = _aligned_task_parameter_labels_for_predicate(
				predicate_name,
				task_parameters,
				task_parameter_types,
				action_analysis,
				producer_pattern=pattern,
			)
			if aligned_task_parameters is None:
				continue
			token_mapping = {
				token: task_parameter
				for token, task_parameter in zip(effect_args, aligned_task_parameters)
			}
			rendered_action_args = _extend_mapping_with_action_parameters(
				token_mapping,
				pattern.get("action_parameters") or (),
				action_parameter_types=pattern.get("action_parameter_types") or (),
			)
			rendered_symbol_types: Dict[str, str] = {
				str(task_parameter): str(task_parameter_type or "object")
				for task_parameter, task_parameter_type in zip(task_parameters, task_parameter_types)
			}
			for rendered_arg, action_parameter_type in zip(
				rendered_action_args,
				pattern.get("action_parameter_types") or (),
			):
				rendered_symbol_types.setdefault(
					str(rendered_arg),
					str(action_parameter_type or "object"),
				)
			final_step = _task_invocation_signature(
				str(pattern.get("action_name") or "").strip(),
				rendered_action_args,
			)
			headline_signature = _task_invocation_signature(
				predicate_name,
				tuple(aligned_task_parameters),
			)
			context = _limited_unique(
				[
					*_render_positive_dynamic_requirements(pattern, token_mapping),
					*_render_positive_static_requirements(pattern, token_mapping),
				],
				limit=8,
			)
			recursive_support_calls: list[str] = []
			for requirement in _render_positive_dynamic_requirements(pattern, token_mapping):
				parsed_requirement = _parse_literal_signature(requirement)
				if parsed_requirement is None or not parsed_requirement[2]:
					continue
				requirement_predicate, requirement_args, _ = parsed_requirement
				if requirement_predicate not in recursive_predicates:
					continue
				if set(requirement_args).issubset(task_parameter_set):
					continue
				if len(requirement_args) != len(task_parameters):
					continue
				if any(
					not _signature_types_can_biject(
						(str(rendered_symbol_types.get(str(requirement_arg), "object")),),
						(str(task_parameter_type or "object"),),
						action_analysis.get("type_parent_map", {}) or {},
					)
					for requirement_arg, task_parameter_type in zip(requirement_args, task_parameter_types)
				):
					continue
				recursive_support_calls.append(
					_task_invocation_signature(task_name, requirement_args),
				)
			family_payload: Dict[str, Any] = {
				"family_role": (
					"recursive_blocker_removal"
					if recursive_support_calls
					else "direct_achiever"
				),
				"final_step": final_step,
				"context": context,
					"redundant_if_already_satisfied": headline_signature in context,
			}
			if recursive_support_calls:
				family_payload["recursive_support_calls"] = _limited_unique(
					recursive_support_calls,
					limit=2,
				)
			cleanup_steps = _render_cleanup_steps_for_family(
				action_analysis=action_analysis,
				headline_signature=headline_signature,
				pattern=pattern,
				task_parameter_set=task_parameter_set,
				token_mapping=dict(token_mapping),
			)
			if recursive_support_calls and cleanup_steps:
				family_payload["cleanup_steps"] = cleanup_steps
			serialized = json.dumps(family_payload, sort_keys=True)
			if serialized in seen_payloads:
				continue
			seen_payloads.add(serialized)
			families.append(family_payload)
	return families


def _render_prerequisite_acquisition_templates(
	shared_dynamic_prerequisites: Sequence[str],
	action_analysis: Dict[str, Any],
) -> list[str]:
	lines: list[str] = []
	per_requirement_lines: list[list[str]] = []
	for requirement in shared_dynamic_prerequisites:
		requirement_lines: list[str] = []
		parsed_requirement = _parse_literal_signature(str(requirement))
		if parsed_requirement is None:
			continue
		predicate_name, requirement_args, is_positive = parsed_requirement
		if not is_positive:
			continue
		for producer_pattern in (
			action_analysis.get("producer_patterns_by_predicate", {}).get(predicate_name, ())
		):
			effect_signature = str(producer_pattern.get("effect_signature") or "").strip()
			parsed_effect = _parse_literal_signature(effect_signature)
			if parsed_effect is None:
				continue
			_, effect_args, effect_positive = parsed_effect
			if not effect_positive or len(effect_args) != len(requirement_args):
				continue
			token_mapping = {
				source_arg: target_arg
				for source_arg, target_arg in zip(effect_args, requirement_args)
			}
			rendered_action_args = _extend_mapping_with_action_parameters(
				token_mapping,
				tuple(producer_pattern.get("action_parameters") or ()),
				action_parameter_types=tuple(producer_pattern.get("action_parameter_types") or ()),
			)
			action_signature = _task_invocation_signature(
				str(producer_pattern.get("source_action_name") or producer_pattern.get("action_name") or "").strip(),
				rendered_action_args,
			)
			dynamic_needs = [
				_render_signature_with_mapping(str(signature), token_mapping)
				for signature in (producer_pattern.get("dynamic_precondition_signatures") or ())
				if str(signature).strip() and not str(signature).startswith("not ")
			]
			if requirement in dynamic_needs:
				continue
			requirement_lines.append(
				f"{requirement} via {action_signature} "
				f"[needs {', '.join(dynamic_needs) if dynamic_needs else 'none'}]"
			)
		per_requirement_lines.append(
			_limited_unique(requirement_lines, limit=2),
		)
	for requirement_lines in per_requirement_lines:
		lines.extend(requirement_lines)
	return _limited_unique(
		lines,
		limit=max(6, len(shared_dynamic_prerequisites) * 2),
	)
def _render_domain_action_schema_blocks(domain: Any) -> str:
	parser = HDDLConditionParser()
	blocks: list[str] = []
	for action in getattr(domain, "actions", []):
		parsed_action = parser.parse_action(action)
		parameter_signature = ", ".join(
			f"{_parameter_token(parameter)}:{_parameter_type(parameter)}"
			for parameter in getattr(action, "parameters", ())
		)
		preconditions = [
			_literal_pattern_signature(pattern)
			for pattern in parsed_action.preconditions
			if pattern.predicate != "="
		]
		add_effects = [
			_literal_pattern_signature(pattern)
			for pattern in parsed_action.effects
			if pattern.predicate != "=" and pattern.is_positive
		]
		delete_effects = [
			_literal_pattern_signature(pattern)
			for pattern in parsed_action.effects
			if pattern.predicate != "=" and not pattern.is_positive
		]
		blocks.append(
			f"{parsed_action.name}({parameter_signature}) "
			f"pre:[{', '.join(preconditions) if preconditions else 'true'}] "
			f"add:[{', '.join(add_effects) if add_effects else 'none'}] "
			f"del:[{', '.join(delete_effects) if delete_effects else 'none'}]"
		)
	return "\n\n".join(blocks).strip()


def _render_declared_compound_task_blocks(domain: Any) -> str:
	blocks: list[str] = []
	for task in getattr(domain, "tasks", []):
		task_name = str(getattr(task, "name", "")).strip()
		if not task_name:
			continue
		parameter_signature = ", ".join(
			f"{_parameter_token(parameter)}:{_parameter_type(parameter)}"
			for parameter in getattr(task, "parameters", ())
		)
		blocks.append(f"{task_name}({parameter_signature})")
	return "\n".join(blocks).strip()


def _render_structural_contract_block(domain: Any) -> str:
	required_task_names = [
		str(getattr(task, "name", "")).strip()
		for task in getattr(domain, "tasks", [])
		if str(getattr(task, "name", "")).strip()
	]
	return "\n".join(
		[
			"method_coverage:",
			f"- required_method_task_names: {json.dumps(required_task_names)}",
			"- output compound_tasks must define exactly these task names with their "
			"declared parameter types.",
			"- methods.task_name must cover every required name at least once; do not "
			"drop tasks absent from temporal_specifications.",
			"ordering_integrity:",
			"- local_step_ids are the step_id values inside the same method.subtasks array.",
			"- each ordering edge [before, after] must use two distinct local_step_ids.",
			"- if a method has fewer than two subtasks, ordering must be [].",
		]
	)


def _render_causal_executability_contract_block() -> str:
	return "\n".join(
		[
			"primitive_support:",
			"- For every constructive method, each primitive subtask must be executable "
			"at its position.",
			"- Every positive dynamic precondition of a primitive subtask must be "
			"guaranteed by method context or by an earlier subtask that establishes "
			"the same literal under the same variable mapping.",
			"- If a required primitive precondition is not guaranteed, insert an "
			"earlier compound support subgoal when one exists; otherwise make it an "
			"explicit method context guard.",
			"headline_alignment:",
			"- A direct primitive leaf is valid only if its positive add effects "
			"establish the compound task headline literal.",
			"- Bind task parameters according to action effect argument positions; "
			"use fresh witness variables for non-task action parameters.",
			"- Do not require mutually exclusive fluent values in one context; use "
			"transition subtasks instead.",
		]
	)


def build_domain_htn_system_prompt() -> str:
	"""System prompt for one-shot domain-complete method synthesis."""

	return (
		"ROLE:\n"
		"Generate a typed symbolic Hierarchical Task Network method library M for "
		"AgentSpeak(L) plan-library construction for the dissertation workflow.\n"
		"\n"
		"OBJECTIVE:\n"
		"Infer reusable domain-level decompositions over the masked HDDL domain D^- "
		"and the supplied task requirements. M will be structurally checked and then "
		"translated into AgentSpeak(L) plan library S; generation must not depend on "
		"benchmark problem state or original HDDL methods.\n"
		"\n"
		"CONTRACT:\n"
		"- Use only supplied compound tasks, primitive actions, predicates, types, and instruction identifiers.\n"
		"- Preserve the compound/primitive boundary: method.task_name and compound subtasks must be declared compound tasks; primitive subtasks must be declared primitive actions.\n"
		"- Actions are operators, not predicates. Contexts, preconditions, effects, and negated literals may use only predicates or equality.\n"
		"- Variables are typed symbolic parameters. Give each variable one declared type and use fresh variables for distinct typed roles or witness roles.\n"
		"- Methods are reusable schemas aligned with the temporal specifications, not one grounded plan trace.\n"
		"- M is domain-complete: output compound tasks must match declared compound "
		"tasks, and methods must cover every declared compound task.\n"
		"- Do not copy object constants from temporal specifications into M; task_args and subtask args should be method variables unless a declared schema requires a constant.\n"
		"- Use explicit step objects and local pairwise ordering edges only: "
		"[[\"s1\", \"s2\"]]. Every endpoint must be a step_id in the same method.\n"
		"- If a method has zero or one subtask, ordering must be [].\n"
		"- Empty subtasks are allowed only for already-satisfied guard methods; constructive methods must contain real subtasks.\n"
		"- Constructive methods must be causally executable: each primitive precondition must be supported by context or earlier subtasks.\n"
		"- Recursive methods must make progress by changing at least one witness argument or state-support step before recursion.\n"
		"- All JSON scalar values for names, variables, constants, literals, step ids, and instruction ids must be quoted strings.\n"
		"\n"
		"OUTPUT:\n"
		"Return exactly one minified JSON object with top-level keys compound_tasks and methods. "
		"Every method must include source_instruction_ids. Do not emit primitive_tasks, "
		"markdown, commentary, or reasoning text.\n"
	)

def build_domain_htn_user_prompt(
	domain: Any,
	schema_hint: str,
	*,
	action_analysis: Optional[Dict[str, Any]] = None,
	derived_analysis: Optional[Dict[str, Any]] = None,
	query_sequence: Optional[Sequence[Dict[str, Any]]] = None,
	temporal_specifications: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
	"""User prompt for one-shot domain-complete method synthesis synthesis."""

	analysis = _normalise_action_analysis(domain, action_analysis)
	prompt_analysis = dict(
		derived_analysis
		or build_domain_prompt_analysis_payload(
			domain,
			action_analysis=analysis,
		)
	)
	method_blueprints = list(prompt_analysis.get("method_blueprints") or ())
	action_schema_block = _render_domain_action_schema_blocks(domain)
	declared_task_block = _render_declared_compound_task_blocks(domain)
	structural_contract_block = _render_structural_contract_block(domain)
	causal_contract_block = _render_causal_executability_contract_block()
	method_blueprint_block = _render_method_blueprint_blocks(method_blueprints)
	temporal_specifications = tuple(temporal_specifications or ())
	domain_summary_block = "\n".join(
		[
			f"domain: {domain.name}",
			"declared_compound_tasks:",
			declared_task_block or "none",
			"primitive_action_schemas:",
			action_schema_block or "none",
		]
	)
	instructions_block = "\n".join(
		[
			"1. Define exactly the declared compound tasks and a domain-complete set of methods.",
			"2. Use method_blueprints as compact decomposition evidence, not as permission to invent new symbols.",
			"3. direct_leaf means one primitive achiever whose positive add effect establishes the task headline; support_then_leaf means support steps before the final primitive; hierarchical_orchestration means compound delegation or task composition.",
			"4. Primitive action names from primitive_action_schemas, direct_primitive_achievers, or uncovered_prerequisite_families may appear only in subtasks with kind=primitive.",
			"5. Compound task names may appear only as method.task_name or subtasks with kind=compound.",
			"6. Preserve distinct AUX witness roles; bind required witnesses in parameters, context, subtasks, and ordering, but not in task_args.",
			"7. Methods that are not already satisfied must contain real subtasks; primitive leaf methods must include the primitive action itself.",
			"8. Use temporal_specifications as the only task-level supervision while keeping methods reusable and variable-parameterized.",
			"9. Every method must cite one or more source_instruction_ids from temporal_specifications.",
			"10. Enforce causal_executability: primitive preconditions must be context-supported or established by earlier subtasks.",
			"11. Bind task parameters from headline/effect positions; use fresh typed witnesses for non-task action parameters.",
			"12. Enforce structural_contract exactly; it is part of the output specification.",
		]
	)
	gate_check_block = "\n".join(
		[
			"Before emitting JSON, check that:",
			"- every symbol is declared in domain_summary;",
			"- methods.task_name covers every required_method_task_names entry;",
			"- primitive and compound subtasks are not swapped;",
			"- no method is specialized to concrete temporal-specification objects such as benchmark block names;",
			"- contexts and step annotations contain predicates/equality only, never action names;",
			"- each variable has one declared type and compatible arity everywhere;",
			"- constructive methods have subtasks and every ordering edge references two distinct local step ids.",
			"- methods with fewer than two subtasks have empty ordering.",
			"- every primitive action precondition is supported by context or earlier subtasks;",
			"- each direct primitive leaf establishes the compound task headline;",
			"- support subgoals precede primitive actions that need their effects;",
			"- contexts do not require mutually exclusive fluent values.",
		]
	)
	temporal_specifications_block = "\n".join(
		(
			f"- {str(item.get('instruction_id') or '').strip()}: "
			f"ltlf={str(item.get('ltlf_formula') or '').strip()} | "
			f"referenced_events="
			f"{', '.join(str(event.get('event') or '').strip() for event in (item.get('referenced_events') or ()) if str(event.get('event') or '').strip()) or 'none'}"
		)
		for item in temporal_specifications
		if str(item.get("instruction_id") or "").strip()
		and str(item.get("ltlf_formula") or "").strip()
	)
	sections = [
		_format_tagged_block(
			"task",
			"Generate one executable JSON HTN library for the whole domain. "
			"Use only the provided temporal specifications as the task-level requirements that the reusable library should support.",
		),
		_format_tagged_block("domain_summary", domain_summary_block),
		_format_tagged_block(
			"temporal_specifications",
			temporal_specifications_block or "none",
		),
		_format_tagged_block("structural_contract", structural_contract_block),
		_format_tagged_block("causal_executability", causal_contract_block),
		_format_tagged_block(
			"method_blueprints",
			method_blueprint_block,
		),
		_format_tagged_block("instructions", instructions_block),
		_format_tagged_block("gate_checklist", gate_check_block),
		_format_tagged_block(
			"output_schema",
			"Emit one JSON object with keys compound_tasks and methods.\n"
			"compound_tasks entries use: name, parameters, optional source_predicates, optional source_name.\n"
			"methods entries use: method_name, task_name, parameters, task_args, context, subtasks, ordering, source_instruction_ids.\n"
			"ordering must be an array of two-element step-id arrays such as [[\"s1\", \"s2\"]].\n"
			"Each subtask entry uses: step_id, task_name, args, kind.\n"
			"All names, args, context literals, ordering step ids, and source_instruction_ids are strings; never emit bare tokens.\n"
				"Use empty subtasks plus empty ordering only for already-satisfied guarded methods.\n"
			f"schema_hint: {schema_hint}",
		),
	]
	return "\n\n".join(section for section in sections if section).strip() + "\n"
