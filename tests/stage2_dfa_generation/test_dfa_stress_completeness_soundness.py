"""
Stress tests for DFA simplification: Completeness and Soundness

This test suite validates that the BDD Shannon Expansion implementation is:
1. COMPLETE: Handles all possible input combinations correctly
2. SOUND: Produces DFAs that are semantically equivalent to the input

Test Categories:
- Large formula complexity (many atoms, deep nesting)
- Edge cases in boolean logic (tautologies, contradictions)
- Multiple transitions with overlapping conditions
- Stress testing with 4+ predicates
- Pathological cases (all conjunctions, all disjunctions)
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stage2_dfa_generation.dfa_simplifier import DFASimplifier
from src.stage1_interpretation.grounding_map import GroundingMap
from tests.stage2_dfa_generation.test_dfa_equivalence_verification import verify_equivalence


def test_stress_4_predicates_complex_formula():
    """
    Stress test: 4 predicates with complex nested formula

    Formula: (a&b) | (c&d)
    This creates 2^4 = 16 possible valuations
    """
    print("=" * 80)
    print("STRESS TEST: 4 Predicates - (a&b)|(c&d)")
    print("=" * 80)
    print()

    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="(a&b)|(c&d)"];
    1 -> 1 [label="~((a&b)|(c&d))"];
    2 -> 2 [label="true"];
}
    """

    gmap = GroundingMap()
    gmap.add_atom("a", "a", [])
    gmap.add_atom("b", "b", [])
    gmap.add_atom("c", "c", [])
    gmap.add_atom("d", "d", [])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap, verify=False)

    print("Original DFA:")
    print(original_dfa)
    print("\nSimplified DFA:")
    print(result.simplified_dot)
    print()

    predicates = ['a', 'b', 'c', 'd']
    is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, predicates)

    assert is_equiv, f"4-predicate formula failed with {len(counterexamples)} counterexamples"
    print("✓ Test passed: 4-predicate complex formula is equivalent\n")


def test_stress_5_predicates_disjunction_chain():
    """
    Stress test: 5 predicates with disjunction chain

    Formula: a|b|c|d|e (at least one must be true)
    This creates 2^5 = 32 possible valuations
    """
    print("=" * 80)
    print("STRESS TEST: 5 Predicates - a|b|c|d|e")
    print("=" * 80)
    print()

    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="a|b|c|d|e"];
    1 -> 1 [label="~(a|b|c|d|e)"];
    2 -> 2 [label="true"];
}
    """

    gmap = GroundingMap()
    for pred in ['a', 'b', 'c', 'd', 'e']:
        gmap.add_atom(pred, pred, [])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap, verify=False)

    print("Original DFA:")
    print(original_dfa)
    print("\nSimplified DFA:")
    print(result.simplified_dot)
    print()

    predicates = ['a', 'b', 'c', 'd', 'e']
    is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, predicates)

    assert is_equiv, f"5-predicate disjunction failed with {len(counterexamples)} counterexamples"
    print("✓ Test passed: 5-predicate disjunction chain is equivalent\n")


def test_stress_conjunction_chain():
    """
    Stress test: Multiple conjunctions

    Formula: a&b&c&d (all must be true)
    Only the all-true valuation should accept
    """
    print("=" * 80)
    print("STRESS TEST: Conjunction Chain - a&b&c&d")
    print("=" * 80)
    print()

    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="a&b&c&d"];
    1 -> 1 [label="~(a&b&c&d)"];
    2 -> 2 [label="true"];
}
    """

    gmap = GroundingMap()
    for pred in ['a', 'b', 'c', 'd']:
        gmap.add_atom(pred, pred, [])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap, verify=False)

    print("Original DFA:")
    print(original_dfa)
    print("\nSimplified DFA:")
    print(result.simplified_dot)
    print()

    predicates = ['a', 'b', 'c', 'd']
    is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, predicates)

    assert is_equiv, f"Conjunction chain failed with {len(counterexamples)} counterexamples"
    print("✓ Test passed: Conjunction chain is equivalent\n")


def test_stress_xor_pattern():
    """
    Stress test: XOR pattern (exactly one true)

    Formula: (a&~b) | (~a&b)
    This is XOR(a,b) - exactly one must be true
    """
    print("=" * 80)
    print("STRESS TEST: XOR Pattern - (a&~b)|(~a&b)")
    print("=" * 80)
    print()

    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="(a&~b)|(~a&b)"];
    1 -> 1 [label="~((a&~b)|(~a&b))"];
    2 -> 2 [label="true"];
}
    """

    gmap = GroundingMap()
    gmap.add_atom("a", "a", [])
    gmap.add_atom("b", "b", [])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap, verify=False)

    print("Original DFA:")
    print(original_dfa)
    print("\nSimplified DFA:")
    print(result.simplified_dot)
    print()

    predicates = ['a', 'b']
    is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, predicates)

    assert is_equiv, f"XOR pattern failed with {len(counterexamples)} counterexamples"
    print("✓ Test passed: XOR pattern is equivalent\n")


def test_stress_3way_xor():
    """
    Stress test: 3-way XOR (exactly one of three true)

    Formula: (a&~b&~c) | (~a&b&~c) | (~a&~b&c)
    """
    print("=" * 80)
    print("STRESS TEST: 3-way XOR - exactly one of {a,b,c} true")
    print("=" * 80)
    print()

    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="(a&~b&~c)|(~a&b&~c)|(~a&~b&c)"];
    1 -> 1 [label="~((a&~b&~c)|(~a&b&~c)|(~a&~b&c))"];
    2 -> 2 [label="true"];
}
    """

    gmap = GroundingMap()
    for pred in ['a', 'b', 'c']:
        gmap.add_atom(pred, pred, [])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap, verify=False)

    print("Original DFA:")
    print(original_dfa)
    print("\nSimplified DFA:")
    print(result.simplified_dot)
    print()

    predicates = ['a', 'b', 'c']
    is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, predicates)

    assert is_equiv, f"3-way XOR failed with {len(counterexamples)} counterexamples"
    print("✓ Test passed: 3-way XOR is equivalent\n")


def test_stress_implication_pattern():
    """
    Stress test: Logical implication

    Formula: a -> b (equivalent to ~a | b)
    If a is true, then b must be true
    """
    print("=" * 80)
    print("STRESS TEST: Implication - a->b (~a|b)")
    print("=" * 80)
    print()

    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="~a|b"];
    1 -> 1 [label="~(~a|b)"];
    2 -> 2 [label="true"];
}
    """

    gmap = GroundingMap()
    gmap.add_atom("a", "a", [])
    gmap.add_atom("b", "b", [])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap, verify=False)

    print("Original DFA:")
    print(original_dfa)
    print("\nSimplified DFA:")
    print(result.simplified_dot)
    print()

    predicates = ['a', 'b']
    is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, predicates)

    assert is_equiv, f"Implication pattern failed with {len(counterexamples)} counterexamples"
    print("✓ Test passed: Implication pattern is equivalent\n")


def test_stress_at_least_2_of_4():
    """
    Stress test: At least 2 out of 4 predicates must be true

    This tests counting logic - more complex than simple AND/OR
    """
    print("=" * 80)
    print("STRESS TEST: At least 2 of {a,b,c,d} must be true")
    print("=" * 80)
    print()

    # Formula: All combinations where at least 2 are true
    # (a&b)|(a&c)|(a&d)|(b&c)|(b&d)|(c&d)
    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="(a&b)|(a&c)|(a&d)|(b&c)|(b&d)|(c&d)"];
    1 -> 1 [label="~((a&b)|(a&c)|(a&d)|(b&c)|(b&d)|(c&d))"];
    2 -> 2 [label="true"];
}
    """

    gmap = GroundingMap()
    for pred in ['a', 'b', 'c', 'd']:
        gmap.add_atom(pred, pred, [])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap, verify=False)

    print("Original DFA:")
    print(original_dfa)
    print("\nSimplified DFA:")
    print(result.simplified_dot)
    print()

    predicates = ['a', 'b', 'c', 'd']
    is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, predicates)

    assert is_equiv, f"At-least-2 counting failed with {len(counterexamples)} counterexamples"
    print("✓ Test passed: At-least-2 counting is equivalent\n")


def test_stress_multi_state_complex_transitions():
    """
    Stress test: Multiple states with complex transition conditions

    This tests that the algorithm handles multiple source states correctly
    """
    print("=" * 80)
    print("STRESS TEST: Multi-state DFA with complex transitions")
    print("=" * 80)
    print()

    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 3;
    node [shape = circle]; 1 2;
    init -> 1;
    1 -> 2 [label="a&b"];
    1 -> 1 [label="~(a&b)"];
    2 -> 3 [label="c|d"];
    2 -> 1 [label="~(c|d)"];
    3 -> 3 [label="true"];
}
    """

    gmap = GroundingMap()
    for pred in ['a', 'b', 'c', 'd']:
        gmap.add_atom(pred, pred, [])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap, verify=False)

    print("Original DFA:")
    print(original_dfa)
    print("\nSimplified DFA:")
    print(result.simplified_dot)
    print()

    predicates = ['a', 'b', 'c', 'd']
    is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, predicates)

    assert is_equiv, f"Multi-state complex transitions failed with {len(counterexamples)} counterexamples"
    print("✓ Test passed: Multi-state complex transitions is equivalent\n")


def test_stress_tautology():
    """
    Soundness test: Formula that's always true (tautology)

    Formula: a | ~a (always true)
    """
    print("=" * 80)
    print("SOUNDNESS TEST: Tautology - a|~a")
    print("=" * 80)
    print()

    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="a|~a"];
    1 -> 1 [label="~(a|~a)"];
    2 -> 2 [label="true"];
}
    """

    gmap = GroundingMap()
    gmap.add_atom("a", "a", [])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap, verify=False)

    print("Original DFA:")
    print(original_dfa)
    print("\nSimplified DFA:")
    print(result.simplified_dot)
    print()

    predicates = ['a']
    is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, predicates)

    assert is_equiv, f"Tautology handling failed with {len(counterexamples)} counterexamples"
    print("✓ Test passed: Tautology is equivalent\n")


def test_stress_contradiction():
    """
    Soundness test: Formula that's always false (contradiction)

    Formula: a & ~a (always false)
    """
    print("=" * 80)
    print("SOUNDNESS TEST: Contradiction - a&~a")
    print("=" * 80)
    print()

    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="a&~a"];
    1 -> 1 [label="~(a&~a)"];
    2 -> 2 [label="true"];
}
    """

    gmap = GroundingMap()
    gmap.add_atom("a", "a", [])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap, verify=False)

    print("Original DFA:")
    print(original_dfa)
    print("\nSimplified DFA:")
    print(result.simplified_dot)
    print()

    predicates = ['a']
    is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, predicates)

    assert is_equiv, f"Contradiction handling failed with {len(counterexamples)} counterexamples"
    print("✓ Test passed: Contradiction is equivalent\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
