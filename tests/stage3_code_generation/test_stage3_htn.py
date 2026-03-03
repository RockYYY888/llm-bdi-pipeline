"""
Focused tests for the HTN-based Stage 3 pipeline.
"""

import sys
from pathlib import Path

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
from stage2_dfa_generation.dfa_builder import DFABuilder
from stage3_code_generation.htn_method_synthesis import HTNMethodSynthesizer
from stage3_code_generation.htn_planner_generator import HTNPlannerGenerator
from utils.pddl_parser import PDDLParser


def _domain():
    domain_path = Path(__file__).parent.parent.parent / "src" / "domains" / "blocksworld" / "domain.pddl"
    return PDDLParser.parse_domain(str(domain_path))


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
        )
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
        )
    ]
    return spec


def test_method_synthesizer_creates_goal_and_support_tasks():
    domain = _domain()
    spec = _eventually_on_spec()
    dfa_result = DFABuilder().build(spec)

    library, metadata = HTNMethodSynthesizer().synthesize(
        domain=domain,
        grounding_map=spec.grounding_map,
        dfa_result=dfa_result,
    )

    compound_task_names = {task.name for task in library.compound_tasks}
    method_names = {method.method_name for method in library.methods}

    assert metadata["used_llm"] is False
    assert "achieve_on" in compound_task_names
    assert "achieve_holding" in compound_task_names
    assert "achieve_on__via_put_on_block" in method_names


def test_generator_emits_specialised_transition_and_primitive_plans():
    domain = _domain()
    spec = _eventually_on_spec()
    dfa_result = DFABuilder().build(spec)
    ltl_dict = {
        "objects": spec.objects,
        "formulas_string": [formula.to_string() for formula in spec.formulas],
        "grounding_map": spec.grounding_map,
    }

    code, artifacts = HTNPlannerGenerator(domain, spec.grounding_map).generate(ltl_dict, dfa_result)

    assert artifacts["summary"]["method"] == "htn"
    assert "+!put_on_block(X1, X2)" in code
    assert "\tput_on_block(X1, X2);" in code
    assert "_physical(" not in code
    assert "+!transition_1" in code
    assert "+!achieve_on(a, b)" in code
    assert any(item["label"] == "on(a, b)" for item in artifacts["transitions"])


def test_negative_goal_generates_guard_plan():
    domain = _domain()
    spec = _globally_not_on_spec()
    dfa_result = DFABuilder().build(spec)
    ltl_dict = {
        "objects": spec.objects,
        "formulas_string": [formula.to_string() for formula in spec.formulas],
        "grounding_map": spec.grounding_map,
    }

    code, artifacts = HTNPlannerGenerator(domain, spec.grounding_map).generate(ltl_dict, dfa_result)

    assert artifacts["summary"]["transition_count"] >= 1
    assert "+!maintain_not_on(a, b)" in code
    assert ": not on(a, b) <-" in code
