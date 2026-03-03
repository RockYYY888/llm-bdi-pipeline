"""
Focused live tests for the HTN-based Stage 3 pipeline.
"""

import sys
from pathlib import Path

import pytest

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import (
    LTLFormula,
    LTLSpecification,
    LogicalOperator,
    TemporalOperator,
)
from stage3_code_generation.htn_method_synthesis import HTNMethodSynthesizer
from stage3_code_generation.htn_planner_generator import HTNPlannerGenerator
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
        pytest.skip("Stage 3 live tests require a valid OPENAI_API_KEY")

    return {
        "api_key": config.openai_api_key,
        "model": config.openai_model,
        "base_url": config.openai_base_url,
        "timeout": float(config.openai_timeout),
    }


def _atomic_formula(predicate: str, args: list[str]) -> LTLFormula:
    return LTLFormula(
        operator=None,
        predicate={predicate: args},
        sub_formulas=[],
        logical_op=None,
    )


def _eventually_on_spec() -> LTLSpecification:
    spec = LTLSpecification()
    spec.objects = ["a", "b"]
    spec.grounding_map = GroundingMap()
    spec.grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    spec.formulas = [
        LTLFormula(
            operator=TemporalOperator.FINALLY,
            predicate=None,
            sub_formulas=[_atomic_formula("on", ["a", "b"])],
            logical_op=None,
        ),
    ]
    return spec


def _globally_not_on_spec() -> LTLSpecification:
    spec = LTLSpecification()
    spec.objects = ["a", "b"]
    spec.grounding_map = GroundingMap()
    spec.grounding_map.add_atom("on_a_b", "on", ["a", "b"])

    atom = _atomic_formula("on", ["a", "b"])
    negation = LTLFormula(
        operator=None,
        predicate=None,
        sub_formulas=[atom],
        logical_op=LogicalOperator.NOT,
    )
    spec.formulas = [
        LTLFormula(
            operator=TemporalOperator.GLOBALLY,
            predicate=None,
            sub_formulas=[negation],
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


def test_method_synthesizer_uses_live_llm_output():
    domain = _domain()
    spec = _eventually_on_spec()
    dfa_result = _dfa_result_for_labels("on_a_b")

    library, metadata = HTNMethodSynthesizer(
        **_live_stage3_kwargs(),
    ).synthesize(
        domain=domain,
        grounding_map=spec.grounding_map,
        dfa_result=dfa_result,
    )

    assert metadata["used_llm"] is True
    assert metadata["llm_prompt"] is not None
    assert metadata["llm_response"]
    assert metadata["target_literals"] == ["on(a, b)"]
    assert metadata["compound_tasks"] >= 1
    assert metadata["methods"] >= 1
    assert library.compound_tasks
    assert library.methods

    primitive_task_names = {task.name for task in library.primitive_tasks}
    assert primitive_task_names == {
        "pick_up",
        "pick_up_from_table",
        "put_on_block",
        "put_down",
    }

    compound_task_names = {task.name for task in library.compound_tasks}
    assert all(method.task_name in compound_task_names for method in library.methods)


def test_generator_emits_specialised_transition_and_primitive_plans():
    domain = _domain()
    spec = _eventually_on_spec()
    dfa_result = _dfa_result_for_labels("on_a_b")
    ltl_dict = {
        "objects": spec.objects,
        "formulas_string": [formula.to_string() for formula in spec.formulas],
        "grounding_map": spec.grounding_map,
    }

    code, artifacts = HTNPlannerGenerator(
        domain,
        spec.grounding_map,
        **_live_stage3_kwargs(),
    ).generate(ltl_dict, dfa_result)

    assert artifacts["summary"]["method"] == "htn"
    assert artifacts["summary"]["used_llm"] is True
    assert artifacts["summary"]["transition_count"] == 1
    assert artifacts["llm"]["response"]
    assert "+!achieve_on(a, b)" in code
    assert "\t!put_on_block(a, b)." in code
    assert "+!put_on_block(X1, X2)" in code
    assert "\tput_on_block(X1, X2);" in code
    assert "_physical(" not in code
    assert "+!transition_1" in code
    assert any(item["label"] == "on(a, b)" for item in artifacts["transitions"])


def test_negative_goal_generates_guard_plan():
    domain = _domain()
    spec = _globally_not_on_spec()
    dfa_result = _dfa_result_for_labels("!on_a_b")
    ltl_dict = {
        "objects": spec.objects,
        "formulas_string": [formula.to_string() for formula in spec.formulas],
        "grounding_map": spec.grounding_map,
    }

    code, artifacts = HTNPlannerGenerator(
        domain,
        spec.grounding_map,
        **_live_stage3_kwargs(),
    ).generate(ltl_dict, dfa_result)

    assert artifacts["summary"]["transition_count"] >= 1
    assert artifacts["summary"]["used_llm"] is True
    assert "+!maintain_not_on(a, b)" in code
    assert ": not on(a, b) <-" in code
