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


def test_condition_parser_fails_fast_on_unsupported_or_construct():
	parser = HDDLConditionParser()

	with pytest.raises(UnsupportedHDDLConstructError, match="Unsupported HDDL construct 'or'"):
		parser.parse_literals("(or (clear ?x) (holding ?x))", action_name="demo_action")
