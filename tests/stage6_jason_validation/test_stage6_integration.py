from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage3_method_synthesis.htn_schema import HTNLiteral
from stage6_jason_validation.jason_runner import JasonRunner
from utils.hddl_condition_parser import HDDLConditionParser, UnsupportedHDDLConstructError
from utils.hddl_parser import HDDLParser


def _stage6_ready() -> bool:
	return JasonRunner().toolchain_available()


def _load_blocksworld_action_schemas():
	domain_path = Path("src/domains/blocksworld/domain.hddl")
	if not domain_path.exists():
		pytest.skip(f"Blocksworld domain missing: {domain_path}")
	domain = HDDLParser.parse_domain(str(domain_path))
	parser = HDDLConditionParser()
	schemas = []
	for action in domain.actions:
		parsed = parser.parse_action(action)
		schemas.append(
			{
				"functor": action.name.replace("-", "_"),
				"parameters": list(parsed.parameters),
				"preconditions": [
					{
						"predicate": literal.predicate,
						"args": list(literal.args),
						"is_positive": literal.is_positive,
					}
					for literal in parsed.preconditions
				],
				"precondition_clauses": [
					[
						{
							"predicate": literal.predicate,
							"args": list(literal.args),
							"is_positive": literal.is_positive,
						}
						for literal in clause
					]
					for clause in parsed.precondition_clauses
				],
				"effects": [
					{
						"predicate": literal.predicate,
						"args": list(literal.args),
						"is_positive": literal.is_positive,
					}
					for literal in parsed.effects
				],
			},
			)
	return schemas


def _parse_positive_hddl_fact(fact: str):
	text = (fact or "").strip()
	if not text.startswith("(") or not text.endswith(")"):
		return None
	inner = text[1:-1].strip()
	if not inner or inner.startswith("not "):
		return None
	tokens = inner.split()
	if not tokens:
		return None
	return tokens[0], tuple(tokens[1:])


def _load_seed_facts(log_dir: Path, target_literals):
	panda_json = log_dir / "panda_transitions.json"
	if not panda_json.exists():
		return ()
	payload = json.loads(panda_json.read_text())
	negative_targets = {
		(literal.predicate, tuple(literal.args))
		for literal in target_literals
		if not literal.is_positive and not literal.is_equality
	}
	facts = []
	seen = set()
	for transition in payload.get("transitions", []):
		for fact in transition.get("initial_facts", []):
			parsed = _parse_positive_hddl_fact(fact)
			if parsed is not None and parsed in negative_targets:
				continue
			if fact in seen:
				continue
			seen.add(fact)
			facts.append(fact)
	return tuple(facts)


def test_stage6_runs_query2_sample_agentspeak(tmp_path):
	if not _stage6_ready():
		pytest.skip("Stage 6 integration requires Java 17-23 and Jason runtime toolchain")

	sample_asl = Path("tests/logs/20260304_123250_dfa_agentspeak/agentspeak_generated.asl")
	if not sample_asl.exists():
		pytest.skip(f"Sample ASL missing: {sample_asl}")

	target_literals = [
		HTNLiteral(predicate="on", args=("a", "b"), is_positive=False, source_symbol=None),
	]
	runner = JasonRunner(timeout_seconds=60)
	result = runner.validate(
		agentspeak_code=sample_asl.read_text(),
		target_literals=target_literals,
		action_schemas=_load_blocksworld_action_schemas(),
		seed_facts=_load_seed_facts(sample_asl.parent, target_literals),
		domain_name="blocksworld",
		output_dir=tmp_path,
	)

	assert result.status == "success"
	assert "execute success" in result.stdout
	assert "execute failed" not in result.stdout


def test_stage6_runs_mixed_goal_sample_agentspeak(tmp_path):
	if not _stage6_ready():
		pytest.skip("Stage 6 integration requires Java 17-23 and Jason runtime toolchain")

	sample_asl = Path("tests/logs/20260304_123745_dfa_agentspeak/agentspeak_generated.asl")
	if not sample_asl.exists():
		pytest.skip(f"Sample ASL missing: {sample_asl}")

	target_literals = [
		HTNLiteral(predicate="clear", args=("c",), is_positive=True, source_symbol=None),
		HTNLiteral(predicate="clear", args=("d",), is_positive=False, source_symbol=None),
		HTNLiteral(predicate="on", args=("a", "b"), is_positive=False, source_symbol=None),
		HTNLiteral(predicate="on", args=("c", "d"), is_positive=True, source_symbol=None),
	]
	runner = JasonRunner(timeout_seconds=60)
	result = runner.validate(
		agentspeak_code=sample_asl.read_text(),
		target_literals=target_literals,
		action_schemas=_load_blocksworld_action_schemas(),
		seed_facts=_load_seed_facts(sample_asl.parent, target_literals),
		domain_name="blocksworld",
		output_dir=tmp_path,
	)

	assert result.status == "success"
	assert "execute success" in result.stdout
	assert "execute failed" not in result.stdout


def test_stage6_runs_negative_target_case_with_naf(tmp_path):
	if not _stage6_ready():
		pytest.skip("Stage 6 integration requires Java 17-23 and Jason runtime toolchain")

	agentspeak_code = """/* Initial Beliefs */
domain(test).
object(a).
dfa_state(q1).
accepting_state(q2).
dfa_edge_label(dfa_step_q1_q2_not_clear_a, "!clear(a)").

/* Primitive Action Plans */
+!seal(BLOCK) : not clear(BLOCK) <-
\tseal(BLOCK);
\t-clear(BLOCK).

/* HTN Method Plans */
+!keep_not_clear(BLOCK) : true <-
\t!seal(BLOCK).

/* DFA Transition Wrappers */
+!dfa_step_q1_q2_not_clear_a : dfa_state(q1) <-
\t!keep_not_clear(a);
\t-dfa_state(q1);
\t+dfa_state(q2).

/* DFA Control Plans */
+!run_dfa : dfa_state(q2) & accepting_state(q2) <-
\ttrue.

+!run_dfa : dfa_state(q1) <-
\t!dfa_step_q1_q2_not_clear_a;
\t!run_dfa.
"""

	runner = JasonRunner(timeout_seconds=60)
	result = runner.validate(
		agentspeak_code=agentspeak_code,
		target_literals=[
			HTNLiteral(
				predicate="clear",
				args=("a",),
				is_positive=False,
				source_symbol=None,
			),
		],
		action_schemas=[
			{
				"functor": "seal",
				"parameters": ["?x"],
				"preconditions": [
					{
						"predicate": "clear",
						"args": ["?x"],
						"is_positive": False,
					},
				],
				"precondition_clauses": [
					[
						{
							"predicate": "clear",
							"args": ["?x"],
							"is_positive": False,
						},
					],
				],
				"effects": [
					{
						"predicate": "clear",
						"args": ["?x"],
						"is_positive": False,
					},
				],
			},
		],
		seed_facts=(),
		domain_name="naf_negation_demo",
		output_dir=tmp_path,
	)

	assert result.status == "success"
	assert "execute success" in result.stdout
	assert "~clear(a)" not in (tmp_path / "agentspeak_generated.asl").read_text()


def test_stage6_runs_or_precondition_case(tmp_path):
	if not _stage6_ready():
		pytest.skip("Stage 6 integration requires Java 17-23 and Jason runtime toolchain")

	agentspeak_code = """/* Initial Beliefs */
domain(test).
object(a).
dfa_state(q1).
accepting_state(q2).
dfa_edge_label(dfa_step_q1_q2_checked_a, "checked(a)").

/* Primitive Action Plans */
+!probe(BLOCK) : clear(BLOCK) | holding(BLOCK) <-
\tprobe(BLOCK);
\t+checked(BLOCK).

/* HTN Method Plans */
+!make_checked(BLOCK) : true <-
\t!probe(BLOCK).

/* DFA Transition Wrappers */
+!dfa_step_q1_q2_checked_a : dfa_state(q1) <-
\t!make_checked(a);
\t-dfa_state(q1);
\t+dfa_state(q2).

/* DFA Control Plans */
+!run_dfa : dfa_state(q2) & accepting_state(q2) <-
\ttrue.

+!run_dfa : dfa_state(q1) <-
\t!dfa_step_q1_q2_checked_a;
\t!run_dfa.
"""

	runner = JasonRunner(timeout_seconds=60)
	result = runner.validate(
		agentspeak_code=agentspeak_code,
		target_literals=[
			HTNLiteral(
				predicate="checked",
				args=("a",),
				is_positive=True,
				source_symbol=None,
			),
		],
		action_schemas=[
			{
				"functor": "probe",
				"parameters": ["?x"],
				"precondition_clauses": [
					[
						{
							"predicate": "clear",
							"args": ["?x"],
							"is_positive": True,
						},
					],
					[
						{
							"predicate": "holding",
							"args": ["?x"],
							"is_positive": True,
						},
					],
				],
				"effects": [
					{
						"predicate": "checked",
						"args": ["?x"],
						"is_positive": True,
					},
				],
			},
		],
		seed_facts=("(clear a)",),
		domain_name="or_precondition_demo",
		output_dir=tmp_path,
	)

	assert result.status == "success"
	assert "execute success" in result.stdout
	assert "execute failed" not in result.stdout


def test_stage6_fail_fast_on_unsupported_hddl_constructs():
	parser = HDDLConditionParser()
	action = type(
		"ActionStub",
		(),
		{
			"name": "unsupported_when",
			"parameters": ["?x - object"],
			"preconditions": "(when (p ?x) (q ?x))",
			"effects": "(and)",
		},
	)()

	with pytest.raises(UnsupportedHDDLConstructError, match="unsupported_when"):
		parser.parse_action(action)
