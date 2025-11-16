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

        # For DFAs with atomic labels, we process atoms one by one
        # For each atom in the valuation, we try to take a transition
        remaining_atoms = set(valuation)

        while remaining_atoms or current_state != self.initial_state:
            # Find all transitions from current state that are enabled
            enabled_transitions = []
            for from_s, to_s, label in self.transitions:
                if from_s == current_state and self._eval_label(label, valuation):
                    enabled_transitions.append((to_s, label))

            if not enabled_transitions:
                # No enabled transition - check if we're in accepting state
                break

            # Take the first enabled transition (deterministic)
            next_state, taken_label = enabled_transitions[0]

            # Remove atoms from taken_label from remaining_atoms
            if taken_label in self.all_atoms:
                remaining_atoms.discard(taken_label)
            elif taken_label == "true":
                # Unconditional transition - consume all remaining atoms
                remaining_atoms.clear()

            current_state = next_state

            # If we took a "true" transition, we're done processing
            if taken_label == "true":
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
