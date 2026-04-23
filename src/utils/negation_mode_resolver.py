"""
Resolve negative literal semantics across method synthesis and planning.

Current policy is intentionally strict and simple: all negative literals are treated
as NAF (`not p(...)`) across the whole pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, Tuple

if TYPE_CHECKING:
	from method_library.synthesis.schema import HTNLiteral


NegationMode = Literal["naf"]


@dataclass(frozen=True)
class NegationResolutionEntry:
	"""Resolved negation mode for one predicate signature."""

	predicate: str
	arity: int
	mode: NegationMode
	evidence: Tuple[str, ...]

	@property
	def key(self) -> str:
		return f"{self.predicate}/{self.arity}"

	def to_dict(self) -> Dict[str, Any]:
		return {
			"predicate": self.predicate,
			"arity": self.arity,
			"mode": self.mode,
			"evidence": list(self.evidence),
		}


@dataclass(frozen=True)
class NegationResolution:
	"""Resolved negation modes and diagnostics for negative predicates."""

	entries: Tuple[NegationResolutionEntry, ...]

	def mode_for(self, predicate: str, arity: int) -> NegationMode:
		_ = (predicate, arity)
		return "naf"

	def apply(self, literal: "HTNLiteral") -> "HTNLiteral":
		if literal.is_positive or literal.is_equality:
			return literal
		if literal.negation_mode == "naf":
			return literal
		from method_library.synthesis.schema import HTNLiteral

		return HTNLiteral(
			predicate=literal.predicate,
			args=literal.args,
			is_positive=literal.is_positive,
			negation_mode="naf",
			source_symbol=literal.source_symbol,
		)

	def to_dict(self) -> Dict[str, Any]:
		entries = [entry.to_dict() for entry in self.entries]
		mode_by_predicate = {
			entry.key: entry.mode
			for entry in self.entries
		}
		return {
			"policy": "all_naf",
			"predicates": entries,
			"mode_by_predicate": mode_by_predicate,
		}


def extract_query_strong_negation_markers(query_text: Optional[str]) -> list[str]:
	"""Strong-negation hints are disabled under the all-NAF policy."""

	_ = query_text
	return []


def resolve_negation_modes(
	domain: Any,
	target_literals: Sequence["HTNLiteral"],
	*,
	query_text: Optional[str] = None,
	goal_grounding_hints: Optional[Dict[str, Any]] = None,
) -> NegationResolution:
	"""Resolve all negative target predicates to NAF."""

	_ = (domain, query_text, goal_grounding_hints)
	entries: list[NegationResolutionEntry] = []
	seen: set[tuple[str, int]] = set()
	for literal in target_literals:
		if literal.is_positive or literal.is_equality:
			continue
		key = (literal.predicate, len(literal.args))
		if key in seen:
			continue
		seen.add(key)
		entries.append(
			NegationResolutionEntry(
				predicate=literal.predicate,
				arity=len(literal.args),
				mode="naf",
				evidence=("policy: all negative predicates use NAF",),
			)
		)
	return NegationResolution(entries=tuple(entries))
