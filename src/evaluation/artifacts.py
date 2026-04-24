"""
Evaluation artifact models for task grounding, runtime execution, and verification evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


INLINE_TEXT_PREVIEW_CHARS = 2_000
INLINE_SEQUENCE_PREVIEW_ITEMS = 40


def _compact_text_payload(key: str, value: str) -> Dict[str, Any]:
	text = str(value or "")
	if len(text) <= INLINE_TEXT_PREVIEW_CHARS:
		return {key: text}
	return {
		key: text[:INLINE_TEXT_PREVIEW_CHARS],
		f"{key}_truncated": True,
		f"{key}_chars": len(text),
		f"{key}_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
	}


def _compact_sequence_payload(values: Sequence[Any]) -> Dict[str, Any]:
	items = list(values)
	if len(items) <= INLINE_SEQUENCE_PREVIEW_ITEMS:
		return {"items": items, "count": len(items), "truncated": False}
	edge_count = INLINE_SEQUENCE_PREVIEW_ITEMS // 2
	return {
		"items": items[:edge_count] + items[-edge_count:],
		"count": len(items),
		"truncated": True,
		"omitted_middle_count": len(items) - (edge_count * 2),
	}


@dataclass(frozen=True)
class GroundedSubgoal:
	"""One grounded task event emitted by temporal-specification grounding."""

	subgoal_id: str
	task_name: str
	args: Tuple[str, ...] = ()
	argument_types: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.subgoal_id,
			"task_name": self.task_name,
			"args": list(self.args),
			"argument_types": list(self.argument_types),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "GroundedSubgoal":
		return cls(
			subgoal_id=str(payload.get("id") or payload.get("subgoal_id") or ""),
			task_name=str(payload.get("task_name") or ""),
			args=tuple(str(arg) for arg in (payload.get("args") or ())),
			argument_types=tuple(
				str(type_name)
				for type_name in (payload.get("argument_types") or ())
			),
		)


@dataclass(frozen=True)
class TemporalGroundingResult:
	"""Validated temporal-specification grounding output."""

	query_text: str
	ltlf_formula: str
	subgoals: Tuple[GroundedSubgoal, ...]
	typed_objects: Dict[str, str] = field(default_factory=dict)
	query_object_inventory: Tuple[Dict[str, Any], ...] = ()
	diagnostics: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"query_text": self.query_text,
			"ltlf_formula": self.ltlf_formula,
			"subgoals": [subgoal.to_dict() for subgoal in self.subgoals],
			"typed_objects": dict(self.typed_objects),
			"query_object_inventory": [dict(entry) for entry in self.query_object_inventory],
			"diagnostics": list(self.diagnostics),
		}

	def to_log_dict(self) -> Dict[str, Any]:
		"""Return bounded grounding metadata for execution logs."""

		subgoals = [subgoal.to_dict() for subgoal in self.subgoals]
		query_object_inventory = [
			dict(entry)
			for entry in self.query_object_inventory
		]
		return {
			**_compact_text_payload("query_text", self.query_text),
			**_compact_text_payload("ltlf_formula", self.ltlf_formula),
			"subgoals": _compact_sequence_payload(subgoals),
			"typed_object_count": len(self.typed_objects),
			"query_object_inventory": _compact_sequence_payload(query_object_inventory),
			"diagnostics": list(self.diagnostics),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "TemporalGroundingResult":
		return cls(
			query_text=str(payload.get("query_text") or ""),
			ltlf_formula=str(payload.get("ltlf_formula") or ""),
			subgoals=tuple(
				GroundedSubgoal.from_dict(item)
				for item in (payload.get("subgoals") or ())
				if isinstance(item, dict)
			),
			typed_objects={
				str(key): str(value)
				for key, value in dict(payload.get("typed_objects") or {}).items()
			},
			query_object_inventory=tuple(
				dict(entry)
				for entry in (payload.get("query_object_inventory") or ())
				if isinstance(entry, dict)
			),
			diagnostics=tuple(
				str(message).strip()
				for message in (payload.get("diagnostics") or ())
				if str(message).strip()
			),
		)


@dataclass(frozen=True)
class JasonExecutionResult:
	"""Structured Jason runtime outcome for one evaluation query."""

	query_text: str
	ltlf_formula: str
	action_path: Tuple[str, ...]
	method_trace: Tuple[Dict[str, Any], ...]
	hierarchical_plan_text: Optional[str] = None
	verification_problem_file: Optional[str] = None
	verification_mode: str = "original_problem"
	failed_goals: Tuple[str, ...] = ()
	failure_class: Optional[str] = None
	consistency_checks: Dict[str, Any] = field(default_factory=dict)
	artifacts: Dict[str, Any] = field(default_factory=dict)
	timing_profile: Dict[str, Any] = field(default_factory=dict)
	diagnostics: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"query_text": self.query_text,
			"ltlf_formula": self.ltlf_formula,
			"action_path": list(self.action_path),
			"method_trace": [dict(item) for item in self.method_trace],
			"hierarchical_plan_text": self.hierarchical_plan_text,
			"verification_problem_file": self.verification_problem_file,
			"verification_mode": self.verification_mode,
			"failed_goals": list(self.failed_goals),
			"failure_class": self.failure_class,
			"consistency_checks": dict(self.consistency_checks),
			"artifacts": dict(self.artifacts),
			"timing_profile": dict(self.timing_profile),
			"diagnostics": list(self.diagnostics),
		}

	def to_log_dict(self) -> Dict[str, Any]:
		"""Return bounded runtime metadata for execution logs."""

		hierarchical_plan_text = str(self.hierarchical_plan_text or "")
		return {
			**_compact_text_payload("query_text", self.query_text),
			**_compact_text_payload("ltlf_formula", self.ltlf_formula),
			"action_path_count": len(self.action_path),
			"method_trace_count": len(self.method_trace),
			"hierarchical_plan_text_chars": len(hierarchical_plan_text),
			"verification_problem_file": self.verification_problem_file,
			"verification_mode": self.verification_mode,
			"failed_goals": list(self.failed_goals),
			"failure_class": self.failure_class,
			"consistency_checks": dict(self.consistency_checks),
			"artifacts": dict(self.artifacts),
			"timing_profile": dict(self.timing_profile),
			"diagnostics": list(self.diagnostics),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "JasonExecutionResult":
		return cls(
			query_text=str(payload.get("query_text") or ""),
			ltlf_formula=str(payload.get("ltlf_formula") or ""),
			action_path=tuple(
				str(item).strip()
				for item in (payload.get("action_path") or ())
				if str(item).strip()
			),
			method_trace=tuple(
				dict(item)
				for item in (payload.get("method_trace") or ())
				if isinstance(item, dict)
			),
			hierarchical_plan_text=payload.get("hierarchical_plan_text"),
			verification_problem_file=payload.get("verification_problem_file"),
			verification_mode=str(payload.get("verification_mode") or "original_problem"),
			failed_goals=tuple(
				str(goal).strip()
				for goal in (payload.get("failed_goals") or ())
				if str(goal).strip()
			),
			failure_class=(
				str(payload.get("failure_class")).strip()
				if payload.get("failure_class") is not None
				else None
			),
			consistency_checks=dict(payload.get("consistency_checks") or {}),
			artifacts=dict(payload.get("artifacts") or {}),
			timing_profile=dict(payload.get("timing_profile") or {}),
			diagnostics=tuple(
				str(message).strip()
				for message in (payload.get("diagnostics") or ())
				if str(message).strip()
			),
		)


__all__ = [
	"GroundedSubgoal",
	"JasonExecutionResult",
	"TemporalGroundingResult",
]
