import sys
from pathlib import Path
from types import SimpleNamespace

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


def _atomic_formula(predicate: str, args: list[str]) -> LTLFormula:
    return LTLFormula(
        operator=None,
        predicate={predicate: args},
        sub_formulas=[],
        logical_op=None,
    )


def _finally_formula(predicate: str, args: list[str]) -> LTLFormula:
    return LTLFormula(
        operator=TemporalOperator.FINALLY,
        predicate=None,
        sub_formulas=[_atomic_formula(predicate, args)],
        logical_op=None,
    )


def test_ltl_specification_combined_formula_conjoins_multiple_top_level_formulas():
    spec = LTLSpecification()
    spec.add_formula(_finally_formula("on", ["a", "b"]))
    spec.add_formula(_finally_formula("on", ["b", "c"]))

    combined = spec.combined_formula()

    assert combined.logical_op == LogicalOperator.AND
    assert len(combined.sub_formulas) == 2
    assert combined.to_string() == "(F(on(a, b)) & F(on(b, c)))"


def test_dfa_builder_build_uses_combined_formula_string_for_multi_formula_specs():
    spec = LTLSpecification()
    spec.add_formula(_finally_formula("on", ["a", "b"]))
    spec.add_formula(_finally_formula("on", ["b", "c"]))
    spec.grounding_map = GroundingMap()

    original_dfa = """
digraph G {
  node [shape = doublecircle]; 2;
  node [shape = circle]; 1;
  init -> 1;
  1 -> 2 [label="true"];
}
""".strip()
    simplified_dfa = """
digraph G {
  node [shape = doublecircle]; 2;
  node [shape = circle]; 1;
  init -> 1;
  1 -> 2 [label="true"];
}
""".strip()

    builder = DFABuilder()
    builder.converter.convert = lambda _: (original_dfa, {"original_formula": "stub"})
    builder.simplifier.simplify = lambda dot, grounding: SimpleNamespace(
        simplified_dot=simplified_dfa,
        stats={"method": "stub"},
    )

    result = builder.build(spec)

    assert result["formula"] == "(F(on(a, b)) & F(on(b, c)))"
    assert result["original_num_states"] == 2
    assert result["original_num_transitions"] == 1
    assert result["num_states"] == 2
    assert result["num_transitions"] == 1
