"""
Focused tests for Stage 5 AgentSpeak rendering.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage3_method_synthesis.htn_schema import HTNLiteral
from stage4_panda_planning.panda_schema import PANDAPlanResult, PANDAPlanStep
from stage5_agentspeak_rendering.agentspeak_renderer import AgentSpeakRenderer
from utils.hddl_parser import HDDLParser


def _domain():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "src"
		/ "domains"
		/ "blocksworld"
		/ "domain.hddl"
	)
	return HDDLParser.parse_domain(str(domain_path))


def test_renderer_emits_transition_goal_and_primitive_wrappers():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		plan_records=[
			{
				"transition_name": "transition_1",
				"label": "on(a, b)",
				"plan": PANDAPlanResult(
					task_name="place_on",
					task_args=("a", "b"),
					target_literal=HTNLiteral("on", ("a", "b"), True, "on_a_b"),
					steps=[
						PANDAPlanStep("put_on_block", "put-on-block", ("a", "b")),
					],
				),
			},
		],
	)

	assert "/* PANDA Goal Plans */" in code
	assert "+!place_on(a, b) : true <-" in code
	assert "\t!put_on_block(a, b)." in code
	assert "+!transition_1 : true <-" in code
	assert "+!put_on_block(X1, X2) :" in code


def test_renderer_accepts_zero_step_panda_plans():
	renderer = AgentSpeakRenderer()
	code = renderer.generate(
		domain=_domain(),
		objects=("a", "b"),
		plan_records=[
			{
				"transition_name": "transition_1",
				"label": "!on(a, b)",
				"plan": PANDAPlanResult(
					task_name="keep_apart",
					task_args=("a", "b"),
					target_literal=HTNLiteral("on", ("a", "b"), False, "on_a_b"),
					steps=[],
				),
			},
		],
	)

	assert "+!keep_apart(a, b) : true <-" in code
	assert "\ttrue." in code
