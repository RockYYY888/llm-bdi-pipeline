"""Tests for negative literal mode resolution (NAF vs strong negation)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage3_method_synthesis.htn_schema import HTNLiteral
from utils.negation_mode_resolver import resolve_negation_modes


def _predicate(name: str, arity: int):
	return SimpleNamespace(
		name=name,
		parameters=[f"?x{index + 1} - object" for index in range(arity)],
	)


def _action(name: str, preconditions: str, effects: str):
	return SimpleNamespace(
		name=name,
		parameters=["?x - object"],
		preconditions=preconditions,
		effects=effects,
	)


def test_query_strong_hint_resolves_negative_literal_to_strong_mode():
	domain = SimpleNamespace(
		predicates=[_predicate("on", 2)],
		actions=[_action("noop", "(and)", "(and)")],
	)
	resolution = resolve_negation_modes(
		domain,
		[
			HTNLiteral(predicate="on", args=("a", "b"), is_positive=False, source_symbol=None),
		],
		query_text="Make on(a,b) explicitly false.",
	)

	entry = resolution.entries[0]
	assert entry.predicate == "on"
	assert entry.arity == 2
	assert entry.mode == "strong"
	assert any("query_hint" in evidence for evidence in entry.evidence)


def test_complement_predicate_and_effect_coupling_resolve_to_strong_mode():
	domain = SimpleNamespace(
		predicates=[
			_predicate("door_open", 1),
			_predicate("not_door_open", 1),
		],
		actions=[
			_action(
				"close_door",
				"(door_open ?x)",
				"(and (not (door_open ?x)) (not_door_open ?x))",
			),
		],
	)
	resolution = resolve_negation_modes(
		domain,
		[
			HTNLiteral(
				predicate="door_open",
				args=("d1",),
				is_positive=False,
				source_symbol=None,
			),
		],
	)

	entry = resolution.entries[0]
	assert entry.mode == "strong"
	assert any("complement_predicate" in evidence for evidence in entry.evidence)
	assert any("action_effect_coupling" in evidence for evidence in entry.evidence)


def test_missing_evidence_defaults_to_naf_mode():
	domain = SimpleNamespace(
		predicates=[_predicate("clear", 1)],
		actions=[
			_action(
				"drop",
				"(and)",
				"(and (clear ?x))",
			),
		],
	)
	resolution = resolve_negation_modes(
		domain,
		[
			HTNLiteral(predicate="clear", args=("a",), is_positive=False, source_symbol=None),
		],
	)

	entry = resolution.entries[0]
	assert entry.mode == "naf"
	assert entry.evidence == ("default_fallback: no strong-negation evidence found",)
