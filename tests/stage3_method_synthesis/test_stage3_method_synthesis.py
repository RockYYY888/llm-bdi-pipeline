"""
Focused tests for Stage 3 HTN method synthesis.
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
    TemporalOperator,
)
from stage3_method_synthesis.htn_method_synthesis import HTNMethodSynthesizer
from stage3_method_synthesis.htn_prompts import (
    build_htn_system_prompt,
    build_htn_user_prompt,
)
from stage3_method_synthesis.htn_schema import HTNLiteral, HTNMethod, HTNMethodLibrary, HTNTask
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


def test_method_synthesizer_requires_top_level_task_for_each_target_literal():
    domain = _domain()
    synthesizer = HTNMethodSynthesizer()
    library = HTNMethodLibrary(
        compound_tasks=[
            HTNTask("achieve_holding", ("B",), False, ("holding",)),
        ],
        primitive_tasks=synthesizer._build_primitive_tasks(domain),
        methods=[
            HTNMethod(
                method_name="achieve_holding__guard",
                task_name="achieve_holding",
                parameters=("B",),
                context=(),
                subtasks=(),
                ordering=(),
                origin="llm",
            ),
        ],
        target_literals=[
            HTNLiteral("on", ("a", "b"), True, "on_a_b"),
        ],
    )

    with pytest.raises(ValueError, match="missing the top-level compound task 'achieve_on'"):
        synthesizer._validate_library(library, domain)


def test_stage3_prompts_make_naming_and_ordering_rules_explicit():
    system_prompt = build_htn_system_prompt()
    user_prompt = build_htn_user_prompt(
        _domain(),
        ["on(a, b)", "!clear(a)"],
        '{"compound_tasks": [], "methods": []}',
    )

    assert "Non-guard method names must follow exactly: {task_name}__via_{strategy}." in system_prompt
    assert "Guard method names must follow exactly: {task_name}__guard." in system_prompt
    assert "context is for method-level preconditions checked before decomposition." in system_prompt
    assert "ordering must be a list of edges [from_step_id, to_step_id]." in system_prompt

    assert "REQUIRED COMPOUND TASK NAMES:" in user_prompt
    assert "REQUIRED compound task name for on(a, b): achieve_on" in user_prompt
    assert "REQUIRED compound task name for !clear(a): maintain_not_clear" in user_prompt
    assert "If a helper subtask corresponds to a positive predicate P, name it achieve_P." in user_prompt
    assert "For every non-guard method, method_name must be exactly task_name + '__via_'" in user_prompt
    assert "For every guard method, method_name must end exactly in '__guard'." in user_prompt
