"""
Focused live integration tests for Stage 4 PANDA planning.
"""

import sys
from pathlib import Path

import pytest

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator
from stage3_method_synthesis.htn_method_synthesis import HTNMethodSynthesizer
from stage4_panda_planning.panda_planner import PANDAPlanner
from utils.config import get_config
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


def _live_stage3_kwargs() -> dict:
	config = get_config()
	if not config.validate():
		pytest.skip("Stage 4 live tests require a valid OPENAI_API_KEY")

	return {
		"api_key": config.openai_api_key,
		"model": config.openai_model,
		"base_url": config.openai_base_url,
		"timeout": float(config.openai_timeout),
	}


def _require_panda_toolchain() -> None:
	if not PANDAPlanner().toolchain_available():
		pytest.skip("Stage 4 live tests require pandaPIparser, pandaPIgrounder, and pandaPIengine")


def _eventually_on_spec() -> LTLSpecification:
	spec = LTLSpecification()
	spec.objects = ["a", "b"]
	spec.grounding_map = GroundingMap()
	spec.grounding_map.add_atom("on_a_b", "on", ["a", "b"])
	spec.formulas = [
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[
				LTLFormula(
					operator=None,
					predicate={"on": ["a", "b"]},
					sub_formulas=[],
					logical_op=None,
				)
			],
			logical_op=None,
		),
	]
	return spec


def _dfa_result_for_labels(*labels: str) -> dict:
	edges = "\n".join(f'  0 -> 1 [label="{label}"];' for label in labels)
	return {
		"dfa_dot": (
			"digraph {\n"
			"  0 [shape=circle];\n"
			"  1 [shape=doublecircle];\n"
			f"{edges}\n"
			"}\n"
		),
	}


def test_stage4_planner_generates_live_plan_from_synthesised_library(tmp_path):
	_require_panda_toolchain()
	domain = _domain()
	spec = _eventually_on_spec()
	method_library, metadata = HTNMethodSynthesizer(
		**_live_stage3_kwargs(),
	).synthesize(
		domain=domain,
		grounding_map=spec.grounding_map,
		dfa_result=_dfa_result_for_labels("on_a_b"),
	)

	assert metadata["used_llm"] is True

	planner = PANDAPlanner(workspace=str(tmp_path))
	plan = planner.plan(
		domain=domain,
		method_library=method_library,
		objects=spec.objects,
		target_literal=method_library.target_literals[0],
		transition_name="transition_1",
	)

	assert plan.task_name == "achieve_on"
	assert plan.task_args == ("a", "b")
	assert plan.steps
	assert any(step.task_name == "put_on_block" for step in plan.steps)
	assert "(t1 (achieve_on a b))" in plan.problem_hddl
