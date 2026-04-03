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
from stage2_dfa_generation.ltlf_to_dfa import LTLfToDFA


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


def test_ltlf_to_dfa_uses_fast_path_for_independent_eventually_conjunctions():
    spec = LTLSpecification()
    spec.add_formula(_finally_formula("on", ["a", "b"]))
    spec.add_formula(_finally_formula("clear", ["a"]))
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    grounding_map.add_atom("clear_a", "clear", ["a"])
    spec.grounding_map = grounding_map

    converter = LTLfToDFA()
    converter.ltlf_parser = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("generic ltlf2dfa path should not be used"),
    )

    dfa_dot, metadata = converter.convert(spec)

    assert metadata["construction"] == "independent_eventually_atomic_fast_path"
    assert metadata["num_states"] == 4
    assert set(metadata["alphabet"]) == {"on_a_b", "clear_a"}
    assert '1 -> 2 [label="on_a_b"]' in dfa_dot
    assert '1 -> 3 [label="clear_a"]' in dfa_dot
    assert '2 -> 4 [label="clear_a"]' in dfa_dot
    assert '3 -> 4 [label="on_a_b"]' in dfa_dot


def test_dfa_builder_skips_bdd_simplification_for_independent_eventually_atomic_fast_path():
    spec = LTLSpecification()
    spec.add_formula(_finally_formula("on", ["a", "b"]))
    spec.add_formula(_finally_formula("clear", ["a"]))
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    grounding_map.add_atom("clear_a", "clear", ["a"])
    spec.grounding_map = grounding_map

    builder = DFABuilder()
    builder.simplifier.simplify = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("BDD simplifier should be skipped for the atomic fast path"),
    )

    result = builder.build(spec)

    assert result["original_num_states"] == 4
    assert result["num_states"] == 4
    assert result["original_num_transitions"] == 5
    assert result["num_transitions"] == 5
    assert result["simplification_stats"]["method"] == "independent_eventually_atomic_fast_path"
    assert result["simplification_stats"]["skipped_simplifier"] is True


def test_ltlf_to_dfa_falls_back_to_generic_converter_for_large_independent_eventually_sets():
    spec = LTLSpecification()
    grounding_map = GroundingMap()
    for index in range(13):
        predicate = f"goal_{index}"
        spec.add_formula(_finally_formula(predicate, []))
        grounding_map.add_atom(predicate, predicate, [])
    spec.grounding_map = grounding_map

    parser_calls: list[str] = []

    class StubFormula:
        def to_dfa(self):
            return """
digraph MONA_DFA {
 node [shape = doublecircle]; 2;
 node [shape = circle]; 1;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 2 [label="goal_0"];
}
""".strip()

    converter = LTLfToDFA()
    converter.ltlf_parser = lambda formula: parser_calls.append(formula) or StubFormula()

    dfa_dot, metadata = converter.convert(spec)

    assert parser_calls
    assert metadata.get("construction") != "independent_eventually_atomic_fast_path"
    assert metadata["num_transitions"] == 1
    assert '1 -> 2 [label="goal_0"]' in dfa_dot


def test_dfa_builder_uses_generic_path_for_large_independent_eventually_sets():
    spec = LTLSpecification()
    grounding_map = GroundingMap()
    for index in range(13):
        predicate = f"goal_{index}"
        spec.add_formula(_finally_formula(predicate, []))
        grounding_map.add_atom(predicate, predicate, [])
    spec.grounding_map = grounding_map

    original_dfa = """
digraph MONA_DFA {
  node [shape = doublecircle]; 2;
  node [shape = circle]; 1;
  init -> 1;
  1 -> 2 [label="goal_0"];
}
""".strip()
    simplified_dfa = original_dfa

    builder = DFABuilder()
    builder.converter.ltlf_parser = lambda *_args, **_kwargs: SimpleNamespace(to_dfa=lambda: original_dfa)
    builder.simplifier.simplify = lambda dot, grounding: SimpleNamespace(
        simplified_dot=simplified_dfa,
        stats={"method": "stub"},
    )

    result = builder.build(spec)

    assert result["original_num_transitions"] == 1
    assert result["num_transitions"] == 1
    assert result["simplification_stats"]["method"] == "stub"


def test_ltlf_to_dfa_uses_fast_path_for_ordered_eventually_sequences():
    spec = LTLSpecification()
    spec.formulas = [
        LTLFormula(
            operator=TemporalOperator.FINALLY,
            predicate=None,
            sub_formulas=[
                LTLFormula(
                    operator=None,
                    predicate=None,
                    logical_op=LogicalOperator.AND,
                    sub_formulas=[
                        _atomic_formula("on", ["a", "b"]),
                        LTLFormula(
                            operator=TemporalOperator.FINALLY,
                            predicate=None,
                            sub_formulas=[
                                LTLFormula(
                                    operator=None,
                                    predicate=None,
                                    logical_op=LogicalOperator.AND,
                                    sub_formulas=[
                                        _atomic_formula("clear", ["a"]),
                                        _finally_formula("holding", ["a"]),
                                    ],
                                ),
                            ],
                            logical_op=None,
                        ),
                    ],
                ),
            ],
            logical_op=None,
        ),
    ]
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    grounding_map.add_atom("clear_a", "clear", ["a"])
    grounding_map.add_atom("holding_a", "holding", ["a"])
    spec.grounding_map = grounding_map

    converter = LTLfToDFA()
    converter.ltlf_parser = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("generic ltlf2dfa path should not be used for ordered atomic sequences"),
    )

    dfa_dot, metadata = converter.convert(spec)

    assert metadata["construction"] == "ordered_eventually_atomic_fast_path"
    assert metadata["num_states"] == 4
    assert '1 -> 2 [label="on_a_b"]' in dfa_dot
    assert '2 -> 3 [label="clear_a"]' in dfa_dot
    assert '3 -> 4 [label="holding_a"]' in dfa_dot


def test_dfa_builder_skips_bdd_simplification_for_ordered_eventually_atomic_fast_path():
    spec = LTLSpecification()
    spec.formulas = [_finally_formula("goal", ["a"])]
    spec.query_task_sequence_is_ordered = True
    spec.query_task_literal_signatures = ["goal(a)", "goal(a)", "finish(a)"]
    grounding_map = GroundingMap()
    grounding_map.add_atom("goal_a", "goal", ["a"])
    grounding_map.add_atom("finish_a", "finish", ["a"])
    spec.grounding_map = grounding_map

    builder = DFABuilder()
    builder.simplifier.simplify = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("BDD simplifier should be skipped for ordered atomic fast path"),
    )

    result = builder.build(spec)

    assert result["original_num_states"] == 4
    assert result["num_states"] == 4
    assert result["original_num_transitions"] == 3
    assert result["num_transitions"] == 3
    assert result["simplification_stats"]["method"] == "ordered_eventually_atomic_fast_path"
    assert result["simplification_stats"]["skipped_simplifier"] is True


def test_dfa_builder_count_transitions_ignores_init_and_handles_multiple_edges_per_line():
    builder = DFABuilder()
    dfa_dot = """
digraph G {
  init -> 1;
  1 -> 2 [label="a"]; 1 -> 3 [label="b"];
  2 -> 4 [label="c"];
}
""".strip()

    assert builder._count_transitions(dfa_dot) == 3
