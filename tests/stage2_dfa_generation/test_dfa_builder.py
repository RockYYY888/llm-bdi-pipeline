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


def test_ltl_specification_combined_formula_balances_large_top_level_conjunctions():
    spec = LTLSpecification()
    spec.add_formula(_finally_formula("goal", ["a"]))
    spec.add_formula(_finally_formula("goal", ["b"]))
    spec.add_formula(_finally_formula("goal", ["c"]))
    spec.add_formula(_finally_formula("goal", ["d"]))

    combined = spec.combined_formula()

    assert combined.logical_op == LogicalOperator.AND
    assert len(combined.sub_formulas) == 2
    assert combined.to_string() == "((F(goal(a)) & F(goal(b))) & (F(goal(c)) & F(goal(d))))"


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


def test_dfa_builder_uses_exact_symbolic_monitor_for_unordered_query_step_conjunctions():
    spec = LTLSpecification()
    spec.query_task_sequence_is_ordered = False
    for index in range(1, 21):
        spec.add_formula(_finally_formula(f"query_step_{index}", []))

    result = DFABuilder().build(spec)

    assert result["construction"] == "exact_symbolic_query_step_conjunction"
    assert "dfa_dot" not in result
    assert result["num_states"] == 2 ** 20
    assert result["num_transitions"] == 20 * (2 ** 19)
    assert result["num_predicates"] == 20
    assert result["symbolic_query_step_monitor"] == {
        "mode": "unordered_eventually_conjunction",
        "query_step_indices": list(range(1, 21)),
        "initial_state": "q0",
        "accepting_states": [],
        "num_states": 2 ** 20,
        "num_transitions": 20 * (2 ** 19),
    }


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


def test_ltlf_to_dfa_parses_raw_mona_output_into_grouped_dot_edges():
    spec = LTLSpecification()
    spec.add_formula(_finally_formula("goal", ["a"]))
    grounding_map = GroundingMap()
    grounding_map.add_atom("goal_a", "goal", ["a"])
    spec.grounding_map = grounding_map

    raw_mona_output = """
DFA for formula with free variables: GOAL_A OTHER
Initial state: 0
Accepting states: 2
Rejecting states: 0 1

Automaton has 3 states and 4 BDD-nodes
Transitions:
State 0: XX -> state 1
State 1: 10 -> state 2
State 1: 1X -> state 2
State 1: 00 -> state 1
State 1: 01 -> state 1
State 2: XX -> state 2
""".strip()

    class StubFormula:
        def to_mona(self):
            return "true"

        def find_labels(self):
            return ("goal_a", "other")

    converter = LTLfToDFA()
    converter.ltlf_parser = lambda *_args, **_kwargs: StubFormula()
    converter._invoke_mona_directly = lambda *_args, **_kwargs: raw_mona_output

    dfa_dot, metadata = converter.convert(spec)

    assert 'init -> 1;' in dfa_dot
    assert 'node [shape = doublecircle]; 2;' in dfa_dot
    assert '1 -> 2 [label="(goal_a & ~other) | (goal_a)"];' in dfa_dot
    assert '1 -> 1 [label="(~goal_a & ~other) | (~goal_a & other)"];' in dfa_dot
    assert metadata["num_states"] == 2
    assert metadata["num_transitions"] == 3


def test_ltlf_to_dfa_uses_placeholder_for_large_raw_mona_guard_groups():
    spec = LTLSpecification()
    spec.add_formula(_finally_formula("goal", ["a"]))
    grounding_map = GroundingMap()
    grounding_map.add_atom("goal_a", "goal", ["a"])
    spec.grounding_map = grounding_map

    repetitive_guards = "\n".join(
        f"State 1: {format(i, '04b')} -> state 2"
        for i in range(7, 16)
    )
    raw_mona_output = f"""
DFA for formula with free variables: GOAL_A X1 X2 X3
Initial state: 0
Accepting states: 2
Rejecting states: 0 1

Automaton has 3 states and 16 BDD-nodes
Transitions:
State 0: XXXX -> state 1
{repetitive_guards}
State 1: 0000 -> state 1
State 2: XXXX -> state 2
""".strip()

    class StubFormula:
        def to_mona(self):
            return "true"

        def find_labels(self):
            return ("goal_a", "x1", "x2", "x3")

    converter = LTLfToDFA()
    converter.ltlf_parser = lambda *_args, **_kwargs: StubFormula()
    converter._invoke_mona_directly = lambda *_args, **_kwargs: raw_mona_output

    dfa_dot, metadata = converter.convert(spec)

    assert '1 -> 2 [label="guard_group_9"];' in dfa_dot
    assert metadata["num_transitions"] == 3


def test_ltlf_to_dfa_renders_query_step_guards_from_raw_mona_output_without_rewrite():
    spec = LTLSpecification()
    spec.add_formula(_finally_formula("query_step_1", []))
    grounding_map = GroundingMap()
    grounding_map.add_atom("query_step_1", "query_step_1", [])
    grounding_map.add_atom("query_step_2", "query_step_2", [])
    spec.grounding_map = grounding_map

    raw_mona_output = """
DFA for formula with free variables: QUERY_STEP_2 QUERY_STEP_1
Initial state: 0
Accepting states: 2
Rejecting states: 0 1

Automaton has 3 states and 4 BDD-nodes
Transitions:
State 0: XX -> state 1
State 1: 10 -> state 2
State 1: 01 -> state 2
State 1: 00 -> state 1
State 1: 11 -> state 1
State 2: XX -> state 2
""".strip()

    class StubFormula:
        def to_mona(self):
            return "true"

        def find_labels(self):
            return ("query_step_2", "query_step_1")

    converter = LTLfToDFA()
    converter.ltlf_parser = lambda *_args, **_kwargs: StubFormula()
    converter._invoke_mona_directly = lambda *_args, **_kwargs: raw_mona_output

    dfa_dot, metadata = converter.convert(spec)

    assert '1 -> 2 [label="(query_step_2 & ~query_step_1) | (~query_step_2 & query_step_1)"];' in dfa_dot
    assert '1 -> 1 [label="(~query_step_2 & ~query_step_1) | (query_step_2 & query_step_1)"];' in dfa_dot
    assert metadata["num_states"] == 2
    assert metadata["num_transitions"] == 3
