"""
Shared failure-signature helpers for evaluation benchmark execution artifacts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from evaluation.runtime_context import render_problem_fact
from utils.hddl_parser import HDDLParser


_FORMULA_ATOM_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\([^()]*\)")
_FORMULA_OPERATOR_PATTERNS = {
	"F": re.compile(r"(?<![A-Za-z0-9_])F(?=\s*\()"),
	"G": re.compile(r"(?<![A-Za-z0-9_])G(?=\s*\()"),
	"X": re.compile(r"(?<![A-Za-z0-9_])X(?=\s*\()"),
	"WX": re.compile(r"(?<![A-Za-z0-9_])WX(?=\s*\()"),
	"U": re.compile(r"(?<![A-Za-z0-9_])U(?![A-Za-z0-9_])"),
	"R": re.compile(r"(?<![A-Za-z0-9_])R(?![A-Za-z0-9_])"),
	"!": re.compile(r"!"),
	"&": re.compile(r"&"),
	"|": re.compile(r"(?<!\|)\|(?!\|)"),
	"->": re.compile(r"->"),
	"<->": re.compile(r"<->"),
	"last": re.compile(r"(?<![A-Za-z0-9_])last(?![A-Za-z0-9_])"),
}


def ltlf_atom_count(ltlf_formula: str | None) -> int:
	"""Count grounded task-event atoms in one LTLf formula string."""

	return len(_FORMULA_ATOM_PATTERN.findall(str(ltlf_formula or "").strip()))


def ltlf_operator_counts(ltlf_formula: str | None) -> Dict[str, int]:
	"""Count supported LTLf operators in one formula string."""

	text = str(ltlf_formula or "").strip()
	return {
		operator: len(pattern.findall(text))
		for operator, pattern in _FORMULA_OPERATOR_PATTERNS.items()
	}


def infer_missing_goal_facts(
	*,
	problem_file: str | Path | None,
	world_facts: Sequence[str] | None,
) -> Tuple[str, ...]:
	"""Infer which benchmark goal facts are still absent from a replayed final world state."""

	if problem_file is None or not world_facts:
		return ()
	problem = HDDLParser.parse_problem(str(Path(problem_file).resolve()))
	world_fact_set = {
		canonical_fact
		for fact in (world_facts or ())
		if (canonical_fact := _canonical_positive_fact_atom(str(fact).strip()))
	}
	missing_goal_facts = [
		render_problem_fact(fact)
		for fact in tuple(getattr(problem, "goal_facts", ()) or ())
	]
	return tuple(
		sorted(
			fact
			for fact in missing_goal_facts
			if fact and _canonical_positive_fact_atom(fact) not in world_fact_set
		),
	)


def _canonical_positive_fact_atom(fact: str) -> Optional[str]:
	text = str(fact or "").strip()
	if not text:
		return None
	if text.startswith("(") and text.endswith(")"):
		inner = text[1:-1].strip()
		if not inner or inner.startswith("not "):
			return None
		tokens = inner.split()
		if not tokens:
			return None
		return _canonical_fact_parts(tokens[0], tokens[1:])
	match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_-]*)\((.*)\)", text)
	if match is not None:
		args = [
			_strip_fact_argument_quotes(part.strip())
			for part in match.group(2).split(",")
			if part.strip()
		]
		return _canonical_fact_parts(match.group(1), args)
	tokens = text.split()
	if tokens:
		return _canonical_fact_parts(tokens[0], tokens[1:])
	return None


def _canonical_fact_parts(predicate: str, args: Sequence[str]) -> Optional[str]:
	functor = re.sub(r"[^A-Za-z0-9_]+", "_", str(predicate).strip()).strip("_")
	if not functor or functor == "_":
		return None
	if functor == "=":
		return None
	canonical_args = tuple(_strip_fact_argument_quotes(arg) for arg in args)
	return f"{functor}({','.join(canonical_args)})" if canonical_args else functor


def _strip_fact_argument_quotes(argument: str) -> str:
	token = str(argument).strip()
	if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
		return token[1:-1]
	return token


def build_failure_signature(
	*,
	ltlf_formula: str | None,
	jason_failure_class: str | None = None,
	failed_goals: Sequence[str] | None = None,
	verifier_missing_goal_facts: Sequence[str] | None = None,
) -> Dict[str, Any]:
	"""Build the stable failure-signature payload stored in execution and checkpoints."""

	return {
		"ltlf_formula": str(ltlf_formula or "").strip() or None,
		"ltlf_atom_count": ltlf_atom_count(ltlf_formula),
		"ltlf_operator_counts": ltlf_operator_counts(ltlf_formula),
		"jason_failure_class": _optional_failure_text(jason_failure_class),
		"failed_goals": [
			str(goal).strip()
			for goal in (failed_goals or ())
			if str(goal).strip()
		],
		"verifier_missing_goal_facts": [
			str(fact).strip()
			for fact in (verifier_missing_goal_facts or ())
			if str(fact).strip()
		],
	}


def _optional_failure_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	if not text or text.lower() == "none":
		return None
	return text
