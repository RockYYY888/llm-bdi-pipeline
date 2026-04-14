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
    builder = DFABuilder()
    builder.converter.convert = lambda _: (original_dfa, {"original_formula": "stub"})

    result = builder.build(spec)

    assert result["formula"] == "(F(on(a, b)) & F(on(b, c)))"
    assert result["num_states"] == 2
    assert result["num_transitions"] == 1
    assert result["construction"] == "generic_ltlf2dfa"
    assert result["dfa_path"] == "dfa.dot"
    assert "timing_profile" in result
    assert result["timing_profile"]["convert_seconds"] >= 0.0
    assert result["timing_profile"]["total_seconds"] >= 0.0


def test_ltlf_to_dfa_uses_generic_converter_for_independent_eventually_conjunctions():
    spec = LTLSpecification()
    spec.add_formula(_finally_formula("on", ["a", "b"]))
    spec.add_formula(_finally_formula("clear", ["a"]))
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    grounding_map.add_atom("clear_a", "clear", ["a"])
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
 1 -> 2 [label="on_a_b"];
}
""".strip()

    converter = LTLfToDFA()
    converter.ltlf_parser = lambda formula: parser_calls.append(formula) or StubFormula()

    dfa_dot, metadata = converter.convert(spec)

    assert parser_calls
    assert "on_a_b" in parser_calls[0]
    assert "clear_a" in parser_calls[0]
    assert "construction" not in metadata
    assert metadata["num_transitions"] == 1
    assert '1 -> 2 [label="on_a_b"]' in dfa_dot


def test_dfa_builder_keeps_generic_ltlf2dfa_output_for_independent_eventually_specs():
    spec = LTLSpecification()
    spec.add_formula(_finally_formula("on", ["a", "b"]))
    spec.add_formula(_finally_formula("clear", ["a"]))
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    grounding_map.add_atom("clear_a", "clear", ["a"])
    spec.grounding_map = grounding_map

    original_dfa = """
digraph MONA_DFA {
 node [shape = doublecircle]; 2;
 node [shape = circle]; 1;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 2 [label="on_a_b"];
}
""".strip()

    builder = DFABuilder()
    builder.converter.ltlf_parser = lambda *_args, **_kwargs: type(
        "StubFormula",
        (),
        {"to_dfa": staticmethod(lambda: original_dfa)},
    )()

    result = builder.build(spec)

    assert result["num_transitions"] == 1
    assert result["construction"] == "generic_ltlf2dfa"


def test_ltlf_to_dfa_uses_generic_converter_for_large_unordered_query_task_sets():
    spec = LTLSpecification()
    grounding_map = GroundingMap()
    signatures = []
    for index in range(17):
        predicate = f"goal_{index}"
        spec.add_formula(_finally_formula(predicate, []))
        grounding_map.add_atom(predicate, predicate, [])
        signatures.append(predicate)
    spec.grounding_map = grounding_map
    spec.query_task_literal_signatures = signatures
    spec.query_task_sequence_is_ordered = False

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
    assert "construction" not in metadata
    assert metadata["num_transitions"] == 1
    assert '1 -> 2 [label="goal_0"]' in dfa_dot


def test_dfa_builder_uses_generic_path_for_large_independent_eventually_sets():
    spec = LTLSpecification()
    grounding_map = GroundingMap()
    for index in range(16):
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
    builder = DFABuilder()
    builder.converter.ltlf_parser = lambda *_args, **_kwargs: type(
        "StubFormula",
        (),
        {"to_dfa": staticmethod(lambda: original_dfa)},
    )()

    result = builder.build(spec)

    assert result["num_transitions"] == 1
    assert result["construction"] == "generic_ltlf2dfa"


def test_ltlf_to_dfa_uses_generic_converter_for_ordered_eventually_sequences():
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
    spec.query_task_sequence_is_ordered = True
    spec.query_task_literal_signatures = ["on(a, b)", "clear(a)", "holding(a)"]

    parser_calls: list[str] = []

    class StubFormula:
        def to_dfa(self):
            return """
digraph MONA_DFA {
 node [shape = doublecircle]; 2;
 node [shape = circle]; 1;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 2 [label="on_a_b"];
}
""".strip()

    converter = LTLfToDFA()
    converter.ltlf_parser = lambda formula: parser_calls.append(formula) or StubFormula()

    dfa_dot, metadata = converter.convert(spec)

    assert parser_calls
    assert "construction" not in metadata
    assert metadata["num_transitions"] == 1
    assert '1 -> 2 [label="on_a_b"]' in dfa_dot


def test_dfa_builder_keeps_generic_ltlf2dfa_output_for_ordered_eventually_specs():
    spec = LTLSpecification()
    spec.formulas = [_finally_formula("goal", ["a"])]
    spec.query_task_sequence_is_ordered = True
    spec.query_task_literal_signatures = ["goal(a)", "goal(a)", "finish(a)"]
    grounding_map = GroundingMap()
    grounding_map.add_atom("goal_a", "goal", ["a"])
    grounding_map.add_atom("finish_a", "finish", ["a"])
    spec.grounding_map = grounding_map

    original_dfa = """
digraph MONA_DFA {
  node [shape = doublecircle]; 2;
  node [shape = circle]; 1;
  init -> 1;
  1 -> 2 [label="goal_a"];
}
""".strip()
    builder = DFABuilder()
    builder.converter.ltlf_parser = lambda *_args, **_kwargs: type(
        "StubFormula",
        (),
        {"to_dfa": staticmethod(lambda: original_dfa)},
    )()

    result = builder.build(spec)

    assert result["num_transitions"] == 1
    assert result["construction"] == "generic_ltlf2dfa"


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
