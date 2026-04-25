"""
Set semantics for AgentSpeak(L) plan libraries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Dict

from method_library.synthesis.naming import sanitize_identifier

from .models import AgentSpeakBodyStep, AgentSpeakPlan, PlanLibrary


@dataclass(frozen=True)
class PlanLibrarySetResult:
	"""Structured S after set-normalisation."""

	plan_library: PlanLibrary
	removed_duplicate_plans: int
	renamed_plans: int

	def to_dict(self) -> Dict[str, Any]:
		return {
			"plan_count": len(self.plan_library.plans),
			"removed_duplicate_plans": self.removed_duplicate_plans,
			"renamed_plans": self.renamed_plans,
		}


def deduplicate_plan_library(plan_library: PlanLibrary) -> PlanLibrarySetResult:
	"""Enforce S as a finite set of unique AgentSpeak(L) plan schemas."""

	plans: list[AgentSpeakPlan] = []
	index_by_fingerprint: Dict[str, int] = {}
	used_plan_names: set[str] = set()
	removed_duplicates = 0
	renamed_plans = 0
	for plan in plan_library.plans:
		fingerprint = plan_fingerprint(plan)
		existing_index = index_by_fingerprint.get(fingerprint)
		if existing_index is not None:
			removed_duplicates += 1
			plans[existing_index] = _merge_plan_metadata(plans[existing_index], plan)
			continue
		plan_name = str(plan.plan_name or "").strip() or f"plan_{len(plans) + 1}"
		unique_plan_name = _unique_name(plan_name, used_plan_names)
		if unique_plan_name != plan_name:
			renamed_plans += 1
			plan = replace(plan, plan_name=unique_plan_name)
		used_plan_names.add(unique_plan_name)
		index_by_fingerprint[fingerprint] = len(plans)
		plans.append(plan)
	return PlanLibrarySetResult(
		plan_library=PlanLibrary(domain_name=plan_library.domain_name, plans=tuple(plans)),
		removed_duplicate_plans=removed_duplicates,
		renamed_plans=renamed_plans,
	)


def plan_fingerprint(plan: AgentSpeakPlan) -> str:
	"""Return a stable semantic fingerprint for a plan, excluding metadata."""

	payload = {
		"trigger": {
			"event_type": plan.trigger.event_type,
			"symbol": plan.trigger.symbol,
			"arguments": list(plan.trigger.arguments),
		},
		"context": sorted(str(literal) for literal in plan.context),
		"body": [
			_body_step_fingerprint(step)
			for step in plan.body
		],
	}
	return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _merge_plan_metadata(base: AgentSpeakPlan, patch: AgentSpeakPlan) -> AgentSpeakPlan:
	source_ids = tuple(
		dict.fromkeys(
			[
				*list(base.source_instruction_ids),
				*list(patch.source_instruction_ids),
			],
		)
	)
	binding_certificate = tuple(
		_unique_dicts(
			[
				*list(base.binding_certificate),
				*list(patch.binding_certificate),
			],
		)
	)
	return replace(
		base,
		source_instruction_ids=source_ids,
		binding_certificate=binding_certificate,
	)


def _unique_dicts(items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
	seen: set[str] = set()
	result: list[Dict[str, Any]] = []
	for item in items:
		key = json.dumps(item, sort_keys=True, separators=(",", ":"))
		if key in seen:
			continue
		seen.add(key)
		result.append(dict(item))
	return result


def _unique_name(name: str, used_names: set[str]) -> str:
	base = sanitize_identifier(name) or "item"
	if base not in used_names:
		return base
	index = 2
	while f"{base}__{index}" in used_names:
		index += 1
	return f"{base}__{index}"


def _body_step_fingerprint(step: AgentSpeakBodyStep) -> Dict[str, Any]:
	return {
		"kind": step.kind,
		"symbol": step.symbol,
		"arguments": list(step.arguments),
	}
