"""
Structured AgentSpeak(L) plan-library models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class AgentSpeakTrigger:
	"""Structured trigger for one AgentSpeak(L) plan."""

	event_type: str
	symbol: str
	arguments: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"event_type": self.event_type,
			"symbol": self.symbol,
			"arguments": list(self.arguments),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "AgentSpeakTrigger":
		return cls(
			event_type=str(payload.get("event_type") or "").strip(),
			symbol=str(payload.get("symbol") or "").strip(),
			arguments=tuple(
				str(value).strip()
				for value in (payload.get("arguments") or ())
				if str(value).strip()
			),
		)


@dataclass(frozen=True)
class AgentSpeakBodyStep:
	"""Structured body step for one AgentSpeak(L) plan."""

	kind: str
	symbol: str
	arguments: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"kind": self.kind,
			"symbol": self.symbol,
			"arguments": list(self.arguments),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "AgentSpeakBodyStep":
		return cls(
			kind=str(payload.get("kind") or "").strip(),
			symbol=str(payload.get("symbol") or "").strip(),
			arguments=tuple(
				str(value).strip()
				for value in (payload.get("arguments") or ())
				if str(value).strip()
			),
		)


@dataclass(frozen=True)
class AgentSpeakPlan:
	"""Structured AgentSpeak(L) plan entry."""

	plan_name: str
	trigger: AgentSpeakTrigger
	context: Tuple[str, ...] = ()
	body: Tuple[AgentSpeakBodyStep, ...] = ()
	source_instruction_ids: Tuple[str, ...] = ()
	binding_certificate: Tuple[Dict[str, Any], ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"plan_name": self.plan_name,
			"trigger": self.trigger.to_dict(),
			"context": list(self.context),
			"body": [step.to_dict() for step in self.body],
			"source_instruction_ids": list(self.source_instruction_ids),
			"binding_certificate": [dict(item) for item in self.binding_certificate],
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "AgentSpeakPlan":
		return cls(
			plan_name=str(payload.get("plan_name") or "").strip(),
			trigger=AgentSpeakTrigger.from_dict(dict(payload.get("trigger") or {})),
			context=tuple(
				str(value).strip()
				for value in (payload.get("context") or ())
				if str(value).strip()
			),
			body=tuple(
				AgentSpeakBodyStep.from_dict(item)
				for item in (payload.get("body") or ())
				if isinstance(item, dict)
			),
			source_instruction_ids=tuple(
				str(value).strip()
				for value in (payload.get("source_instruction_ids") or ())
				if str(value).strip()
			),
			binding_certificate=tuple(
				dict(item)
				for item in (payload.get("binding_certificate") or ())
				if isinstance(item, dict)
			),
		)


@dataclass(frozen=True)
class PlanLibrary:
	"""Generated AgentSpeak(L) plan library S."""

	domain_name: str
	plans: Tuple[AgentSpeakPlan, ...]

	def to_dict(self) -> Dict[str, Any]:
		return {
			"domain_name": self.domain_name,
			"plans": [plan.to_dict() for plan in self.plans],
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "PlanLibrary":
		return cls(
			domain_name=str(payload.get("domain_name") or "").strip(),
			plans=tuple(
				AgentSpeakPlan.from_dict(item)
				for item in (payload.get("plans") or ())
				if isinstance(item, dict)
			),
		)


@dataclass(frozen=True)
class TranslationCoverage:
	"""Summary of HTN-method to AgentSpeak(L) translation acceptance."""

	domain_name: str
	methods_considered: int
	plans_generated: int
	accepted_translation: int
	unsupported_buckets: Dict[str, int] = field(default_factory=dict)
	unsupported_methods: Tuple[Dict[str, Any], ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"domain_name": self.domain_name,
			"methods_considered": self.methods_considered,
			"plans_generated": self.plans_generated,
			"accepted_translation": self.accepted_translation,
			"unsupported_buckets": dict(self.unsupported_buckets),
			"unsupported_methods": [dict(item) for item in self.unsupported_methods],
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "TranslationCoverage":
		return cls(
			domain_name=str(payload.get("domain_name") or "").strip(),
			methods_considered=int(payload.get("methods_considered") or 0),
			plans_generated=int(payload.get("plans_generated") or 0),
			accepted_translation=int(payload.get("accepted_translation") or 0),
			unsupported_buckets={
				str(key): int(value)
				for key, value in dict(payload.get("unsupported_buckets") or {}).items()
			},
			unsupported_methods=tuple(
				dict(item)
				for item in (payload.get("unsupported_methods") or ())
				if isinstance(item, dict)
			),
		)


@dataclass(frozen=True)
class LibraryValidationRecord:
	"""Validation summary for one generated plan library bundle."""

	library_id: str
	passed: bool
	method_count: int
	plan_count: int
	checked_layers: Dict[str, bool]
	warnings: Tuple[str, ...] = ()
	failure_reason: str | None = None

	def to_dict(self) -> Dict[str, Any]:
		return {
			"library_id": self.library_id,
			"passed": self.passed,
			"method_count": self.method_count,
			"plan_count": self.plan_count,
			"checked_layers": dict(self.checked_layers),
			"warnings": list(self.warnings),
			"failure_reason": self.failure_reason,
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "LibraryValidationRecord":
		return cls(
			library_id=str(payload.get("library_id") or "").strip(),
			passed=bool(payload.get("passed", False)),
			method_count=int(payload.get("method_count") or 0),
			plan_count=int(payload.get("plan_count") or 0),
			checked_layers={
				str(key): bool(value)
				for key, value in dict(payload.get("checked_layers") or {}).items()
			},
			warnings=tuple(
				str(value).strip()
				for value in (payload.get("warnings") or ())
				if str(value).strip()
			),
			failure_reason=(
				str(payload.get("failure_reason")).strip()
				if payload.get("failure_reason") is not None
				else None
			),
		)
