#!/usr/bin/env python3
"""
DFA Equivalence Verification

Systematically verify that simplified DFA is equivalent to original DFA
by testing all possible input valuations.

For a DFA over atomic propositions P = {p1, p2, ..., pn}, we need to test
all 2^n possible valuations and verify both DFAs give the same accept/reject decision.
"""

import sys
from pathlib import Path
from typing import Dict, Set, Tuple, List
import re
from itertools import product

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage2_dfa_generation.dfa_simplifier import DFASimplifier


class DFAEvaluator:
    """
    Evaluates DFA on input valuations

    A valuation is a set of atoms that are true.
    For example: {"on_a_b", "clear_c"} means on_a_b=True, clear_c=True, on_d_e=False
    """

    def __init__(self, dfa_dot: str, all_atoms: List[str]):
        """
        Initialize evaluator

        Args:
            dfa_dot: DFA in DOT format
            all_atoms: Complete list of all atoms in the domain
        """
        self.all_atoms = sorted(all_atoms)
        self.transitions = self._parse_transitions(dfa_dot)
        self.accepting_states = self._parse_accepting_states(dfa_dot)
        self.initial_state = self._parse_initial_state(dfa_dot)

    def _parse_transitions(self, dfa_dot: str) -> List[Tuple[str, str, str]]:
        """Parse transitions from DOT format"""
        transitions = []
        for line in dfa_dot.split('\n'):
            match = re.match(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]', line.strip())
            if match:
                from_state, to_state, label = match.groups()
                if from_state not in ['init', '__start']:
                    transitions.append((from_state, to_state, label))
        return transitions

    def _parse_accepting_states(self, dfa_dot: str) -> Set[str]:
        """Parse accepting states from DOT format"""
        accepting = set()
        for line in dfa_dot.split('\n'):
            # Look for: node [shape = doublecircle]; state1 state2 ...
            match = re.search(r'node\s*\[shape\s*=\s*doublecircle\];\s*([^;]+)', line)
            if match:
                states = match.group(1).split()
                accepting.update(states)
        return accepting

    def _parse_initial_state(self, dfa_dot: str) -> str:
        """Parse initial state from DOT format"""
        for line in dfa_dot.split('\n'):
            match = re.match(r'(?:init|__start)\s*->\s*(\w+)', line.strip())
            if match:
                return match.group(1)
        return "1"  # Default to state 1

    def _eval_label(self, label: str, valuation: Set[str]) -> bool:
        """
        Evaluate if a transition label is satisfied by the valuation

        Args:
            label: Boolean expression (e.g., "on_a_b & clear_c", "!on_d_e", "true")
            valuation: Set of atoms that are true

        Returns:
            True if label is satisfied, False otherwise
        """
        # Special cases
        if label == "true":
            return True
        if label == "false":
            return False

        # Build evaluation context
        context = {atom: (atom in valuation) for atom in self.all_atoms}

        # Normalize label
        expr = label.replace('~', ' not ')
        expr = expr.replace('&', ' and ')
        expr = expr.replace('|', ' or ')
        expr = expr.replace('!', ' not ')

        # Evaluate
        try:
            result = eval(expr, {"__builtins__": {}}, context)
            return bool(result)
        except Exception as e:
            print(f"Warning: Could not evaluate label '{label}': {e}")
            return False

    def evaluate(self, valuation: Set[str]) -> bool:
        """
        Run DFA on input valuation

        Args:
            valuation: Set of atoms that are true

        Returns:
            True if valuation is accepted, False otherwise
        """
        current_state = self.initial_state
        visited_states = set()
        max_steps = len(self.transitions) * 2  # Prevent infinite loops

        steps = 0
        while steps < max_steps:
            steps += 1

            # Prevent infinite loops by tracking visited states
            state_signature = (current_state, frozenset(valuation))
            if state_signature in visited_states:
                # We've been in this state with this valuation before - stop
                break
            visited_states.add(state_signature)

            # Find all transitions from current state that are enabled
            enabled_transitions = []
            for from_s, to_s, label in self.transitions:
                if from_s == current_state and self._eval_label(label, valuation):
                    enabled_transitions.append((to_s, label))

            if not enabled_transitions:
                # No enabled transition - stop here
                break

            # Take the first enabled transition (deterministic)
            next_state, taken_label = enabled_transitions[0]

            current_state = next_state

            # If we took a "true" transition to an accepting state, we're done
            if taken_label == "true" and current_state in self.accepting_states:
                break

        # Check if final state is accepting
        return current_state in self.accepting_states


def verify_equivalence(original_dfa: str, simplified_dfa: str,
                       all_atoms: List[str]) -> Tuple[bool, List[Dict]]:
    """
    Verify that two DFAs are equivalent by testing all possible input valuations

    Args:
        original_dfa: Original DFA in DOT format
        simplified_dfa: Simplified DFA in DOT format
        all_atoms: Complete list of all atoms in the domain

    Returns:
        (is_equivalent, counterexamples)
        - is_equivalent: True if DFAs are equivalent
        - counterexamples: List of valuations where DFAs disagree
    """
    eval_original = DFAEvaluator(original_dfa, all_atoms)
    eval_simplified = DFAEvaluator(simplified_dfa, all_atoms)

    counterexamples = []

    # Generate all possible valuations (2^n combinations)
    n = len(all_atoms)
    print(f"Testing all 2^{n} = {2**n} possible valuations...")

    for i, combo in enumerate(product([False, True], repeat=n)):
        # Build valuation set
        valuation = {all_atoms[j] for j in range(n) if combo[j]}

        # Evaluate both DFAs
        result_original = eval_original.evaluate(valuation)
        result_simplified = eval_simplified.evaluate(valuation)

        # Check if results match
        if result_original != result_simplified:
            counterexamples.append({
                'valuation': valuation,
                'original_result': result_original,
                'simplified_result': result_simplified
            })
            print(f"  ❌ Counterexample {len(counterexamples)}: {valuation}")
            print(f"     Original: {result_original}, Simplified: {result_simplified}")

    if not counterexamples:
        print(f"✅ All {2**n} valuations agree - DFAs are equivalent!")
    else:
        print(f"❌ Found {len(counterexamples)} counterexamples - DFAs are NOT equivalent")

    return (len(counterexamples) == 0, counterexamples)


def test_equivalence_simple_case():
    """Test equivalence on a simple example"""
    print("=" * 80)
    print("TEST: Simple Equivalence Verification")
    print("=" * 80)
    print()

    # Simple DFA: accepts if on_a_b is true
    original_dfa = """
digraph G {
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="on_a_b"];
    1 -> 1 [label="!on_a_b"];
    2 -> 2 [label="true"];
}
"""

    # Simplified version (should be equivalent)
    simplified_dfa = """
digraph G {
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="on_a_b"];
    2 -> 2 [label="true"];
}
"""

    all_atoms = ["on_a_b"]

    is_equiv, counterexamples = verify_equivalence(original_dfa, simplified_dfa, all_atoms)

    if is_equiv:
        print("✓ Test passed: DFAs are equivalent\n")
    else:
        print(f"✗ Test failed: Found {len(counterexamples)} counterexamples\n")
        assert False, "DFAs should be equivalent"


def test_equivalence_with_simplifier():
    """Test equivalence using actual simplifier"""
    print("=" * 80)
    print("TEST: Equivalence with DFA Simplifier")
    print("=" * 80)
    print()

    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 1 [label="~on_a_b"];
    1 -> 2 [label="on_a_b"];
    2 -> 2 [label="true"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap)

    all_atoms = ["on_a_b"]

    print("Original DFA:")
    print(original_dfa)
    print()
    print("Simplified DFA:")
    print(result.simplified_dot)
    print()

    is_equiv, counterexamples = verify_equivalence(
        original_dfa,
        result.simplified_dot,
        all_atoms
    )

    if is_equiv:
        print("✓ Test passed: Simplified DFA is equivalent to original\n")
    else:
        print(f"✗ Test failed: Simplified DFA is NOT equivalent\n")
        for ce in counterexamples:
            print(f"  Counterexample: {ce['valuation']}")
            print(f"    Original: {ce['original_result']}, Simplified: {ce['simplified_result']}")
        assert False, "Simplified DFA must be equivalent to original"


def test_complex_formula_3_atoms():
    """Test with 3 atoms and complex nested formula"""
    print("=" * 80)
    print("TEST: Complex Formula with 3 Atoms")
    print("=" * 80)
    print()

    # Formula: (on_a_b & clear_c) | holding_d
    # This requires testing 2^3 = 8 valuations
    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="(on_a_b&clear_c)|holding_d"];
    1 -> 1 [label="~((on_a_b&clear_c)|holding_d)"];
    2 -> 2 [label="true"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    gmap.add_atom("holding_d", "holding", ["d"])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap)

    all_atoms = ["on_a_b", "clear_c", "holding_d"]

    print("Original DFA:")
    print(original_dfa)
    print()
    print("Simplified DFA:")
    print(result.simplified_dot)
    print()

    is_equiv, counterexamples = verify_equivalence(
        original_dfa,
        result.simplified_dot,
        all_atoms
    )

    if is_equiv:
        print("✓ Test passed: 3-atom complex formula is equivalent\n")
    else:
        print(f"✗ Test failed: Found {len(counterexamples)} counterexamples\n")
        for ce in counterexamples[:3]:
            print(f"  Counterexample: {ce['valuation']}")
            print(f"    Original: {ce['original_result']}, Simplified: {ce['simplified_result']}")
        assert False, "3-atom formula equivalence failed"


def test_deeply_nested_expression():
    """Test with deeply nested boolean expressions"""
    print("=" * 80)
    print("TEST: Deeply Nested Expression")
    print("=" * 80)
    print()

    # Formula: (a & (b | (c & d)))
    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="a&(b|(c&d))"];
    1 -> 1 [label="~(a&(b|(c&d)))"];
    2 -> 2 [label="true"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("a", "on", ["a", "b"])
    gmap.add_atom("b", "clear", ["c"])
    gmap.add_atom("c", "holding", ["d"])
    gmap.add_atom("d", "ontable", ["e"])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap)

    all_atoms = ["a", "b", "c", "d"]

    print("Original DFA:")
    print(original_dfa)
    print()
    print("Simplified DFA:")
    print(result.simplified_dot)
    print()

    is_equiv, counterexamples = verify_equivalence(
        original_dfa,
        result.simplified_dot,
        all_atoms
    )

    if is_equiv:
        print("✓ Test passed: Deeply nested expression is equivalent\n")
    else:
        print(f"✗ Test failed: Found {len(counterexamples)} counterexamples\n")
        for ce in counterexamples[:3]:
            print(f"  Counterexample: {ce['valuation']}")
            print(f"    Original: {ce['original_result']}, Simplified: {ce['simplified_result']}")
        assert False, "Deeply nested expression equivalence failed"


def test_mixed_conjunction_disjunction():
    """Test with mixed AND/OR operations"""
    print("=" * 80)
    print("TEST: Mixed Conjunction and Disjunction")
    print("=" * 80)
    print()

    # Formula: (a & b) | (c & d) - DNF form
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
    gmap.add_atom("a", "on", ["a", "b"])
    gmap.add_atom("b", "clear", ["c"])
    gmap.add_atom("c", "holding", ["d"])
    gmap.add_atom("d", "ontable", ["e"])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap)

    all_atoms = ["a", "b", "c", "d"]

    print("Original DFA:")
    print(original_dfa)
    print()
    print("Simplified DFA:")
    print(result.simplified_dot)
    print()

    is_equiv, counterexamples = verify_equivalence(
        original_dfa,
        result.simplified_dot,
        all_atoms
    )

    if is_equiv:
        print("✓ Test passed: Mixed AND/OR formula is equivalent\n")
    else:
        print(f"✗ Test failed: Found {len(counterexamples)} counterexamples\n")
        for ce in counterexamples[:3]:
            print(f"  Counterexample: {ce['valuation']}")
            print(f"    Original: {ce['original_result']}, Simplified: {ce['simplified_result']}")
        assert False, "Mixed AND/OR equivalence failed"


def test_edge_case_empty_dfa():
    """Test edge case: DFA with no predicates"""
    print("=" * 80)
    print("TEST: Edge Case - Empty DFA (no predicates)")
    print("=" * 80)
    print()

    # DFA that accepts everything (no predicates to check)
    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 1;
    init -> 1;
    1 -> 1 [label="true"];
}
"""

    gmap = GroundingMap()

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap)

    print("Original DFA:")
    print(original_dfa)
    print()
    print("Simplified DFA:")
    print(result.simplified_dot)
    print()

    # Should handle empty atom list gracefully
    all_atoms = []

    is_equiv, counterexamples = verify_equivalence(
        original_dfa,
        result.simplified_dot,
        all_atoms
    )

    if is_equiv:
        print("✓ Test passed: Empty DFA handled correctly\n")
    else:
        print(f"✗ Test failed: Empty DFA equivalence failed\n")
        assert False, "Empty DFA equivalence failed"


def test_edge_case_single_state_reject():
    """Test edge case: Single state DFA that rejects everything"""
    print("=" * 80)
    print("TEST: Edge Case - Single State Rejecting DFA")
    print("=" * 80)
    print()

    # DFA with single non-accepting state
    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 1 [label="true"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap)

    all_atoms = ["on_a_b"]

    print("Original DFA:")
    print(original_dfa)
    print()
    print("Simplified DFA:")
    print(result.simplified_dot)
    print()

    is_equiv, counterexamples = verify_equivalence(
        original_dfa,
        result.simplified_dot,
        all_atoms
    )

    if is_equiv:
        print("✓ Test passed: Single-state rejecting DFA is equivalent\n")
    else:
        print(f"✗ Test failed: Single-state rejecting DFA failed\n")
        for ce in counterexamples[:3]:
            print(f"  Counterexample: {ce['valuation']}")
            print(f"    Original: {ce['original_result']}, Simplified: {ce['simplified_result']}")
        assert False, "Single-state rejecting DFA equivalence failed"


def test_edge_case_negation_only():
    """Test edge case: Formula with only negations"""
    print("=" * 80)
    print("TEST: Edge Case - Negation Only Formula")
    print("=" * 80)
    print()

    # Formula: ~a & ~b (both atoms must be false)
    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="~a&~b"];
    1 -> 1 [label="a|b"];
    2 -> 2 [label="true"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("a", "on", ["a", "b"])
    gmap.add_atom("b", "clear", ["c"])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap)

    all_atoms = ["a", "b"]

    print("Original DFA:")
    print(original_dfa)
    print()
    print("Simplified DFA:")
    print(result.simplified_dot)
    print()

    is_equiv, counterexamples = verify_equivalence(
        original_dfa,
        result.simplified_dot,
        all_atoms
    )

    if is_equiv:
        print("✓ Test passed: Negation-only formula is equivalent\n")
    else:
        print(f"✗ Test failed: Negation-only formula failed\n")
        for ce in counterexamples[:3]:
            print(f"  Counterexample: {ce['valuation']}")
            print(f"    Original: {ce['original_result']}, Simplified: {ce['simplified_result']}")
        assert False, "Negation-only formula equivalence failed"


def test_regression_negated_bdd_nodes():
    """
    Regression test for negated BDD node handling

    This tests the critical fix in dfa_simplifier.py:288-301 where
    negated BDD nodes have inverted high/low semantics.

    When a BDD node has negated=True:
    - high branch means variable is FALSE
    - low branch means variable is TRUE
    """
    print("=" * 80)
    print("TEST: Regression - Negated BDD Node Semantics")
    print("=" * 80)
    print()

    # Formula that will trigger negated BDD nodes: ~(a & b) = ~a | ~b
    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="~(a&b)"];
    1 -> 1 [label="a&b"];
    2 -> 2 [label="true"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("a", "on", ["a", "b"])
    gmap.add_atom("b", "clear", ["c"])

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap)

    all_atoms = ["a", "b"]

    print("Original DFA:")
    print(original_dfa)
    print()
    print("Simplified DFA:")
    print(result.simplified_dot)
    print()

    is_equiv, counterexamples = verify_equivalence(
        original_dfa,
        result.simplified_dot,
        all_atoms
    )

    if is_equiv:
        print("✓ Test passed: Negated BDD nodes handled correctly\n")
    else:
        print(f"✗ Test failed: Negated BDD node semantics broken!\n")
        print("This is a CRITICAL regression - the fix at dfa_simplifier.py:288-301 may be broken")
        for ce in counterexamples[:3]:
            print(f"  Counterexample: {ce['valuation']}")
            print(f"    Original: {ce['original_result']}, Simplified: {ce['simplified_result']}")
        assert False, "CRITICAL: Negated BDD node semantics regression"


def test_regression_double_negation():
    """Regression test for double negation: ~~a should equal a"""
    print("=" * 80)
    print("TEST: Regression - Double Negation")
    print("=" * 80)
    print()

    # Formula: ~~a (should be equivalent to just 'a')
    original_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="a"];
    1 -> 1 [label="~a"];
    2 -> 2 [label="true"];
}
"""

    # Simplified version with double negation (should still be equivalent)
    double_neg_dfa = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="~~a"];
    1 -> 1 [label="~(~~a)"];
    2 -> 2 [label="true"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("a", "on", ["a", "b"])

    simplifier = DFASimplifier()
    result = simplifier.simplify(double_neg_dfa, gmap)

    all_atoms = ["a"]

    print("Original DFA (single a):")
    print(original_dfa)
    print()
    print("Double negation DFA (~~a):")
    print(double_neg_dfa)
    print()
    print("Simplified DFA:")
    print(result.simplified_dot)
    print()

    # Both should accept when a=True and reject when a=False
    is_equiv, counterexamples = verify_equivalence(
        original_dfa,
        result.simplified_dot,
        all_atoms
    )

    if is_equiv:
        print("✓ Test passed: Double negation handled correctly\n")
    else:
        print(f"✗ Test failed: Double negation equivalence broken!\n")
        for ce in counterexamples[:3]:
            print(f"  Counterexample: {ce['valuation']}")
            print(f"    Original: {ce['original_result']}, Simplified: {ce['simplified_result']}")
        assert False, "Double negation regression"


def test_regression_de_morgan_laws():
    """Regression test for De Morgan's laws: ~(a | b) = ~a & ~b"""
    print("=" * 80)
    print("TEST: Regression - De Morgan's Laws")
    print("=" * 80)
    print()

    # Two equivalent formulas via De Morgan's law
    dfa1 = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="~(a|b)"];
    1 -> 1 [label="a|b"];
    2 -> 2 [label="true"];
}
"""

    dfa2 = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="~a&~b"];
    1 -> 1 [label="a|b"];
    2 -> 2 [label="true"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("a", "on", ["a", "b"])
    gmap.add_atom("b", "clear", ["c"])

    simplifier = DFASimplifier()
    result1 = simplifier.simplify(dfa1, gmap)
    result2 = simplifier.simplify(dfa2, gmap)

    all_atoms = ["a", "b"]

    print("DFA1: ~(a|b)")
    print(dfa1)
    print()
    print("DFA2: ~a&~b (De Morgan equivalent)")
    print(dfa2)
    print()

    # Both simplified DFAs should be equivalent to each other
    is_equiv, counterexamples = verify_equivalence(
        result1.simplified_dot,
        result2.simplified_dot,
        all_atoms
    )

    if is_equiv:
        print("✓ Test passed: De Morgan's laws preserved\n")
    else:
        print(f"✗ Test failed: De Morgan's laws broken!\n")
        for ce in counterexamples[:3]:
            print(f"  Counterexample: {ce['valuation']}")
            print(f"    DFA1: {ce['original_result']}, DFA2: {ce['simplified_result']}")
        assert False, "De Morgan's laws regression"


def run_all_tests():
    """Run all equivalence verification tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "DFA EQUIVALENCE VERIFICATION" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    tests = [
        test_equivalence_simple_case,
        test_equivalence_with_simplifier,
        test_complex_formula_3_atoms,
        test_deeply_nested_expression,
        test_mixed_conjunction_disjunction,
        test_edge_case_empty_dfa,
        test_edge_case_single_state_reject,
        test_edge_case_negation_only,
        test_regression_negated_bdd_nodes,
        test_regression_double_negation,
        test_regression_de_morgan_laws,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
