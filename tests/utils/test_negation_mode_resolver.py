"""Tests for all-NAF negative literal mode resolution."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from method_library.synthesis.schema import HTNLiteral
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


def test_policy_forces_all_negative_literals_to_naf_even_with_query_hints():
	domain = SimpleNamespace(
		predicates=[_predicate("on", 2)],
		actions=[_action("wait", "(and)", "(and)")],
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
	assert entry.mode == "naf"
	assert entry.evidence == ("policy: all negative predicates use NAF",)


def test_policy_forces_all_negative_literals_to_naf_even_with_complements():
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
	assert entry.mode == "naf"
	assert entry.evidence == ("policy: all negative predicates use NAF",)


def test_policy_keeps_default_naf_mode():
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
	assert entry.evidence == ("policy: all negative predicates use NAF",)
	assert resolution.to_dict()["policy"] == "all_naf"
