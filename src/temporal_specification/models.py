"""
Structured temporal-specification records used by the plan-library workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class QueryInstructionRecord:
	"""One single-sentence natural-language task query from the query sequence."""

	instruction_id: str
	source_text: str
	problem_file: str | None = None

	def to_dict(self) -> Dict[str, Any]:
		return {
			"instruction_id": self.instruction_id,
			"source_text": self.source_text,
			"problem_file": self.problem_file,
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "QueryInstructionRecord":
		return cls(
			instruction_id=str(payload.get("instruction_id") or "").strip(),
			source_text=str(
				payload.get("source_text")
				or payload.get("instruction")
				or payload.get("query_text")
				or ""
			).strip(),
			problem_file=(
				str(payload.get("problem_file")).strip()
				if payload.get("problem_file") is not None
				else None
			),
		)


@dataclass(frozen=True)
class ReferencedEvent:
	"""One domain-relevant event referenced by a temporal specification."""

	event: str
	arguments: Tuple[str, ...] = ()

	def to_dict(self) -> Dict[str, Any]:
		return {
			"event": self.event,
			"arguments": list(self.arguments),
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "ReferencedEvent":
		return cls(
			event=str(payload.get("event") or "").strip(),
			arguments=tuple(
				str(value).strip()
				for value in (payload.get("arguments") or ())
				if str(value).strip()
			),
		)


@dataclass(frozen=True)
class TemporalSpecificationRecord:
	"""Validated temporal specification derived from one query sentence."""

	instruction_id: str
	source_text: str
	ltlf_formula: str
	referenced_events: Tuple[ReferencedEvent, ...]
	diagnostics: Tuple[str, ...] = field(default_factory=tuple)
	problem_file: str | None = None

	def to_dict(self) -> Dict[str, Any]:
		return {
			"instruction_id": self.instruction_id,
			"source_text": self.source_text,
			"ltlf_formula": self.ltlf_formula,
			"referenced_events": [event.to_dict() for event in self.referenced_events],
			"diagnostics": list(self.diagnostics),
			"problem_file": self.problem_file,
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, Any]) -> "TemporalSpecificationRecord":
		return cls(
			instruction_id=str(payload.get("instruction_id") or "").strip(),
			source_text=str(
				payload.get("source_text")
				or payload.get("instruction")
				or payload.get("query_text")
				or ""
			).strip(),
			ltlf_formula=str(payload.get("ltlf_formula") or "").strip(),
			referenced_events=tuple(
				ReferencedEvent.from_dict(item)
				for item in (payload.get("referenced_events") or ())
				if isinstance(item, dict)
			),
			diagnostics=tuple(
				str(value).strip()
				for value in (payload.get("diagnostics") or ())
				if str(value).strip()
			),
			problem_file=(
				str(payload.get("problem_file")).strip()
				if payload.get("problem_file") is not None
				else None
			),
		)
