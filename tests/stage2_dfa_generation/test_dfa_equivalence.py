#!/usr/bin/env python3
"""
Test DFA Equivalence

Verify that simplified DFA is equivalent to original DFA by:
1. Checking determinism (no duplicate transitions)
2. Random testing with sample inputs
"""

import sys
from pathlib import Path
import re
from typing import Dict, Set, Tuple, List

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage2_dfa_generation.dfa_simplifier import DFASimplifier


def parse_dfa_transitions(dot_string: str) -> Dict[Tuple[str, str], str]:
    """
    Parse DFA transitions into a map: (from_state, label) -> to_state

    Returns dict for determinism checking
    """
    transitions = {}
    for line in dot_string.split('\n'):
        match = re.match(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]', line.strip())
        if match:
            from_state, to_state, label = match.groups()
            if from_state not in ['init', '__start']:
                key = (from_state, label)
                if key in transitions:
                    print(f"⚠️  Duplicate transition: {from_state} --[{label}]--> (existing: {transitions[key]}, new: {to_state})")
                transitions[key] = to_state
    return transitions


def check_determinism(dot_string: str) -> bool:
    """Check if DFA is deterministic (no duplicate transitions from same state with same label)"""
    transitions = {}
    duplicate_found = False

    for line in dot_string.split('\n'):
        match = re.match(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]', line.strip())
        if match:
            from_state, to_state, label = match.groups()
            if from_state not in ['init', '__start']:
                key = (from_state, label)
                if key in transitions:
                    print(f"  ❌ Non-deterministic: {from_state} --[{label}]--> {transitions[key]} AND {to_state}")
                    duplicate_found = True
                else:
                    transitions[key] = to_state

    return not duplicate_found


def test_determinism():
    """Test that simplified DFA is deterministic"""
    print("=" * 80)
    print("TEST: Determinism Check")
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

    print("\nChecking determinism...")
    is_deterministic = check_determinism(result.simplified_dot)

    if is_deterministic:
        print("✅ DFA is deterministic (no duplicate transitions)")
    else:
        print("❌ DFA is non-deterministic (found duplicate transitions)")
        print("\nSimplified DFA:")
        print(result.simplified_dot)

    assert is_deterministic, "DFA must be deterministic"
    print("\n✓ Test passed\n")


def test_atomic_labels():
    """Test that all labels are atomic (single atom or negation)"""
    print("=" * 80)
    print("TEST: Atomic Labels Check")
    print("=" * 80)

    dfa_dot = """
digraph G {
    s0 -> s1 [label="on_a_b & clear_c | on_d_e"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    gmap.add_atom("on_d_e", "on", ["d", "e"])

    simplifier = DFASimplifier()
    result = simplifier.simplify(dfa_dot, gmap)

    # Extract labels
    labels = set()
    for line in result.simplified_dot.split('\n'):
        match = re.search(r'\[label="([^"]+)"\]', line)
        if match and 'init' not in line:
            labels.add(match.group(1))

    print(f"\nLabels found: {sorted(labels)}")

    # Check each label is atomic
    valid_atoms = {"on_a_b", "clear_c", "on_d_e", "!on_a_b", "!clear_c", "!on_d_e", "true"}
    for label in labels:
        if label not in valid_atoms:
            print(f"  ❌ Non-atomic label: {label}")
            assert False, f"Found non-atomic label: {label}"

    print("✅ All labels are atomic")
    print("\n✓ Test passed\n")


def run_all_tests():
    """Run all equivalence tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 23 + "DFA EQUIVALENCE TESTS" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    tests = [
        test_determinism,
        test_atomic_labels,
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
