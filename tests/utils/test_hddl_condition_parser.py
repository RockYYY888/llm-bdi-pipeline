"""
Focused tests for symbolic HDDL condition parsing.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

import pytest

from utils.hddl_condition_parser import HDDLConditionParser, UnsupportedHDDLConstructError


def test_condition_parser_keeps_equality_and_disequality_literals():
	parser = HDDLConditionParser()

	literals = parser.parse_literals("(and (= ?x ?y) (not (= ?x ?z)))")

	assert len(literals) == 2
	assert literals[0].predicate == "="
	assert literals[0].args == ("?x", "?y")
	assert literals[0].is_positive is True
	assert literals[1].predicate == "="
	assert literals[1].args == ("?x", "?z")
	assert literals[1].is_positive is False


def test_condition_parser_supports_or_preconditions_as_dnf_clauses():
	parser = HDDLConditionParser()
	clauses = parser.parse_dnf("(or (clear ?x) (holding ?x))", action_name="demo_action")
	required = parser.parse_literals("(or (clear ?x) (holding ?x))", action_name="demo_action")

	assert len(clauses) == 2
	assert clauses[0][0].predicate == "clear"
	assert clauses[1][0].predicate == "holding"
	assert required == ()


def test_condition_parser_supports_imply_by_lowering_to_or_not():
	parser = HDDLConditionParser()
	clauses = parser.parse_dnf(
		"(imply (clear ?x) (holding ?x))",
		action_name="demo_action",
	)

	assert len(clauses) == 2
	assert clauses[0][0].predicate == "clear"
	assert clauses[0][0].is_positive is False
	assert clauses[1][0].predicate == "holding"
	assert clauses[1][0].is_positive is True


def test_condition_parser_fails_fast_on_unsupported_when_construct():
	parser = HDDLConditionParser()

	with pytest.raises(UnsupportedHDDLConstructError, match="Unsupported HDDL construct 'when'"):
		parser.parse_literals("(when (clear ?x) (holding ?x))", action_name="demo_action")


def test_condition_parser_fails_fast_on_disjunctive_effects():
	parser = HDDLConditionParser()
	action = type(
		"ActionStub",
		(),
		{
			"name": "nondet_effect",
			"parameters": ["?x - object"],
			"preconditions": "(and)",
			"effects": "(or (p ?x) (q ?x))",
		},
	)()

	with pytest.raises(UnsupportedHDDLConstructError, match="disjunctive_effect"):
		parser.parse_action(action)
