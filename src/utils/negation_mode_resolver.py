"""
Resolve negative literal semantics (NAF vs strong negation) for Stage 3-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

from utils.hddl_condition_parser import HDDLConditionParser

if TYPE_CHECKING:
	from stage3_method_synthesis.htn_schema import HTNLiteral


NegationMode = Literal["naf", "strong"]


_STRONG_HINT_PHRASES: Tuple[str, ...] = (
	"explicitly false",
	"definitely false",
	"known false",
	"strictly false",
	"explicitly not",
	"definitely not",
	"明确为假",
	"显式为假",
	"已知为假",
	"明确不是",
	"显式不是",
	"明确不成立",
	"显式不成立",
)


_NEGATION_PREFIXES: Tuple[str, ...] = ("not_", "no_", "non_", "un_")


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
		for entry in self.entries:
			if entry.predicate == predicate and entry.arity == arity:
				return entry.mode
		return "naf"

	def apply(self, literal: "HTNLiteral") -> "HTNLiteral":
		if literal.is_positive or literal.is_equality:
			return literal
		mode = self.mode_for(literal.predicate, len(literal.args))
		if literal.negation_mode == mode:
			return literal
		from stage3_method_synthesis.htn_schema import HTNLiteral

		return HTNLiteral(
			predicate=literal.predicate,
			args=literal.args,
			is_positive=literal.is_positive,
			negation_mode=mode,
			source_symbol=literal.source_symbol,
		)

	def to_dict(self) -> Dict[str, Any]:
		entries = [entry.to_dict() for entry in self.entries]
		mode_by_predicate = {
			entry.key: entry.mode
			for entry in self.entries
		}
		return {
			"predicates": entries,
			"mode_by_predicate": mode_by_predicate,
		}


@dataclass(frozen=True)
class _ComplementInfo:
	name: str
	arity: int
	complements: Tuple[str, ...]



def extract_query_strong_negation_markers(query_text: Optional[str]) -> List[str]:
	"""Extract strong-negation textual hints from the natural-language query."""

	if not query_text:
		return []
	text = query_text.lower()
	markers = [phrase for phrase in _STRONG_HINT_PHRASES if phrase in text]
	return sorted(set(markers))



def resolve_negation_modes(
	domain: Any,
	target_literals: Sequence["HTNLiteral"],
	*,
	query_text: Optional[str] = None,
	stage1_hints: Optional[Dict[str, Any]] = None,
) -> NegationResolution:
	"""Resolve `naf` vs `strong` for each negative target predicate."""

	negative_targets: Dict[Tuple[str, int], "HTNLiteral"] = {}
	for literal in target_literals:
		if literal.is_positive or literal.is_equality:
			continue
		negative_targets[(literal.predicate, len(literal.args))] = literal

	if not negative_targets:
		return NegationResolution(entries=())

	query_markers = set(extract_query_strong_negation_markers(query_text))
	if stage1_hints:
		query_markers.update(
			str(item)
			for item in stage1_hints.get("strong_negation_markers", [])
			if str(item).strip()
		)

	predicate_arities = {
		predicate.name: len(getattr(predicate, "parameters", []))
		for predicate in getattr(domain, "predicates", [])
	}
	complements = _discover_complements(predicate_arities)
	effect_couplings = _discover_effect_couplings(domain, complements)

	entries: List[NegationResolutionEntry] = []
	for predicate, arity in sorted(negative_targets):
		evidence: List[str] = []
		is_strong = False
		key = f"{predicate}/{arity}"

		if query_markers:
			markers = sorted(query_markers)
			evidence.append(
				"query_hint: matched strong-negation phrase(s): "
				+ ", ".join(markers)
			)
			is_strong = True

		complement_info = complements.get(key)
		if complement_info and complement_info.complements:
			evidence.append(
				"complement_predicate: found naming complement(s): "
				+ ", ".join(complement_info.complements)
			)
			is_strong = True

		if key in effect_couplings:
			evidence.append(
				"action_effect_coupling: paired add/delete detected in action(s): "
				+ ", ".join(effect_couplings[key])
			)
			is_strong = True

		mode: NegationMode = "strong" if is_strong else "naf"
		if not evidence:
			evidence.append("default_fallback: no strong-negation evidence found")

		entries.append(
			NegationResolutionEntry(
				predicate=predicate,
				arity=arity,
				mode=mode,
				evidence=tuple(evidence),
			)
		)

	return NegationResolution(entries=tuple(entries))



def _discover_complements(predicate_arities: Dict[str, int]) -> Dict[str, _ComplementInfo]:
	grouped: Dict[int, List[str]] = {}
	for name, arity in predicate_arities.items():
		grouped.setdefault(arity, []).append(name)

	by_key: Dict[str, _ComplementInfo] = {}
	for arity, names in grouped.items():
		name_set = set(names)
		for name in sorted(name_set):
			complements: List[str] = []
			base = _strip_negation_prefix(name)
			if base != name and base in name_set:
				complements.append(base)
			if base == name:
				for prefix in _NEGATION_PREFIXES:
					candidate = f"{prefix}{name}"
					if candidate in name_set:
						complements.append(candidate)
			complements = sorted(set(complements))
			by_key[f"{name}/{arity}"] = _ComplementInfo(
				name=name,
				arity=arity,
				complements=tuple(complements),
			)
	return by_key



def _discover_effect_couplings(
	domain: Any,
	complements: Dict[str, _ComplementInfo],
) -> Dict[str, Tuple[str, ...]]:
	parser = HDDLConditionParser()
	couplings: Dict[str, set[str]] = {}

	for action in getattr(domain, "actions", []):
		parsed = parser.parse_action(action)
		positive = {
			(literal.predicate, len(literal.args))
			for literal in parsed.positive_effects
			if literal.predicate != "="
		}
		negative = {
			(literal.predicate, len(literal.args))
			for literal in parsed.negative_effects
			if literal.predicate != "="
		}
		if not positive and not negative:
			continue

		for positive_predicate, arity in positive:
			key = f"{positive_predicate}/{arity}"
			complement_info = complements.get(key)
			if not complement_info:
				continue
			for complement in complement_info.complements:
				if (complement, arity) not in negative:
					continue
				couplings.setdefault(key, set()).add(action.name)
				couplings.setdefault(f"{complement}/{arity}", set()).add(action.name)

		for negative_predicate, arity in negative:
			key = f"{negative_predicate}/{arity}"
			complement_info = complements.get(key)
			if not complement_info:
				continue
			for complement in complement_info.complements:
				if (complement, arity) not in positive:
					continue
				couplings.setdefault(key, set()).add(action.name)
				couplings.setdefault(f"{complement}/{arity}", set()).add(action.name)

	return {
		key: tuple(sorted(values))
		for key, values in couplings.items()
	}



def _strip_negation_prefix(name: str) -> str:
	for prefix in _NEGATION_PREFIXES:
		if name.startswith(prefix) and len(name) > len(prefix):
			return name[len(prefix):]
	return name
