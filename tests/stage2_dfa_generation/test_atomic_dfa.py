#!/usr/bin/env python3
"""
Test Atomic DFA Simplifier

Tests that DFA simplifier converts complex boolean expressions
to atomic-only transitions (single positive atoms per transition).
"""

import sys
from pathlib import Path

# Add src to path
_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage2_dfa_generation.dfa_simplifier import DFASimplifier


def test_atomic_only_transitions():
    """Test that all transitions have only positive atoms"""
    print("=" * 80)
    print("TEST: Atomic-only transitions")
    print("=" * 80)

    dfa_dot = """
digraph G {
    s0 [label="0"];
    s1 [label="1"];
    s2 [label="2", shape=doublecircle];

    init -> s0;
    s0 -> s1 [label="on_a_b | clear_c"];
    s1 -> s2 [label="on_a_b & ~clear_c"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])

    simplifier = DFASimplifier()
    result = simplifier.simplify(dfa_dot, gmap)

    print(f"\nResults:")
    print(f"  Method: {result.stats['method']}")
    print(f"  Original states: {result.stats['num_original_states']}")
    print(f"  New states: {result.stats['num_new_states']}")
    print(f"  Original transitions: {result.stats['num_original_transitions']}")
    print(f"  New transitions: {result.stats['num_new_transitions']}")

    # Extract all labels from simplified DFA
    import re
    labels = set()
    for line in result.simplified_dot.split('\n'):
        match = re.search(r'\[label="([^"]+)"\]', line)
        if match and 'init' not in line:
            labels.add(match.group(1))

    print(f"\nUnique labels in simplified DFA:")
    for label in sorted(labels):
        print(f"  - {label}")

    # Verify: no negations, no conjunctions, no disjunctions
    valid_atoms = {"on_a_b", "clear_c", "true"}
    for label in labels:
        assert label in valid_atoms, f"Invalid label: {label} (expected only: {valid_atoms})"

    print("\n✅ All transitions are atomic (single positive atoms only)")
    print("✓ Test passed\n")


def test_empty_dfa():
    """Test DFA with no predicates"""
    print("=" * 80)
    print("TEST: Empty DFA (no predicates)")
    print("=" * 80)

    dfa_dot = """
digraph G {
    s0 -> s1 [label="true"];
}
"""

    gmap = GroundingMap()
    simplifier = DFASimplifier()
    result = simplifier.simplify(dfa_dot, gmap)

    print(f"  Predicates: {result.stats['num_predicates']}")
    assert result.stats['num_predicates'] == 0
    print("✓ Test passed\n")


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "ATOMIC DFA TESTS" + " " * 37 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    tests = [
        test_atomic_only_transitions,
        test_empty_dfa,
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
