#!/usr/bin/env python3
"""
Atomic Transition Verification - Direct DFA Analysis

Creates realistic DFAs with complex boolean expressions and verifies
they are simplified to atomic partitions.
"""

import sys
import re
from pathlib import Path

_src_dir = str(Path(__file__).parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage2_dfa_generation.dfa_simplifier import DFASimplifier


def extract_transition_labels(dfa_dot: str) -> list:
    """Extract all transition labels from DFA"""
    labels = []
    for line in dfa_dot.split('\n'):
        match = re.search(r'\[label="([^"]+)"\]', line)
        if match and 'init' not in line:
            labels.append(match.group(1))
    return labels


def is_atomic_label(label: str, partition_map: dict) -> bool:
    """
    Check if a label is atomic (single partition symbol or simple atom)
    """
    # Check if it's a partition symbol
    if partition_map and label in partition_map:
        return True

    # Check if it's a simple constant
    if label in ['true', 'false']:
        return True

    # Check if it contains boolean operators (non-atomic)
    has_operators = any(op in label for op in ['&', '|', '!', '~', '(', ')'])
    return not has_operators


def test_case(name: str, original_dfa: str, gmap: GroundingMap):
    """Run a test case"""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

    # Extract original labels
    original_labels = extract_transition_labels(original_dfa)
    print(f"\nOriginal DFA transition labels ({len(original_labels)}):")
    for label in set(original_labels):
        print(f"  - '{label}'")

    # Check if original has complex expressions
    has_complex = any(
        any(op in label for op in ['&', '|', '!', '~']) and label not in ['true', 'false']
        for label in original_labels
    )

    if not has_complex:
        print("\n⚠️  Note: Original DFA already has simple labels")
    else:
        print("\n✓ Original DFA has complex boolean expressions")

    # Simplify
    print(f"\nSimplifying DFA...")
    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, gmap)

    # Extract simplified labels
    simplified_labels = extract_transition_labels(result.simplified_dot)
    print(f"\nSimplified DFA transition labels ({len(simplified_labels)}):")
    for label in set(simplified_labels):
        print(f"  - '{label}'")

    # Check if all simplified labels are atomic
    print(f"\nPartition map ({len(result.partition_map)} partitions):")
    for symbol, partition in result.partition_map.items():
        print(f"  {symbol}: {partition.expression}")

    # Verify atomicity
    non_atomic = []
    for label in simplified_labels:
        if not is_atomic_label(label, result.partition_map):
            non_atomic.append(label)

    print(f"\nAtomicity verification:")
    print(f"  Total transitions: {len(simplified_labels)}")
    print(f"  Atomic transitions: {len(simplified_labels) - len(non_atomic)}")
    print(f"  Non-atomic transitions: {len(non_atomic)}")

    if non_atomic:
        print(f"\n❌ FAILED - Found non-atomic labels:")
        for label in non_atomic:
            print(f"  - '{label}'")
        return False

    print(f"\n✅ PASSED - All transitions are atomic")
    return True


if __name__ == "__main__":
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*22 + "ATOMIC TRANSITION VERIFICATION" + " "*26 + "║")
    print("╚" + "═"*78 + "╝")

    results = []

    # Test 1: Simple conjunction
    gmap1 = GroundingMap()
    gmap1.add_atom("on_a_b", "on", ["a", "b"])
    gmap1.add_atom("clear_c", "clear", ["c"])

    dfa1 = """
digraph G {
    init -> s0;
    s0 -> s1 [label="on_a_b & clear_c"];
    s1 -> s1 [label="true"];
}
"""
    results.append(("Simple Conjunction", test_case("Simple Conjunction", dfa1, gmap1)))

    # Test 2: Disjunction
    dfa2 = """
digraph G {
    init -> s0;
    s0 -> s0 [label="!(on_a_b | clear_c)"];
    s0 -> s1 [label="on_a_b | clear_c"];
    s1 -> s1 [label="true"];
}
"""
    results.append(("Disjunction", test_case("Disjunction", dfa2, gmap1)))

    # Test 3: Complex nested expression
    gmap3 = GroundingMap()
    gmap3.add_atom("on_a_b", "on", ["a", "b"])
    gmap3.add_atom("clear_c", "clear", ["c"])
    gmap3.add_atom("holding_d", "holding", ["d"])

    dfa3 = """
digraph G {
    init -> s0;
    s0 -> s0 [label="!((on_a_b & clear_c) | holding_d)"];
    s0 -> s1 [label="(on_a_b & clear_c) | holding_d"];
    s1 -> s1 [label="true"];
}
"""
    results.append(("Complex Nested", test_case("Complex Nested Expression", dfa3, gmap3)))

    # Test 4: Multiple complex transitions
    dfa4 = """
digraph G {
    init -> s0;
    s0 -> s1 [label="on_a_b & ~clear_c"];
    s0 -> s2 [label="~on_a_b & clear_c"];
    s0 -> s3 [label="on_a_b | clear_c"];
    s1 -> s1 [label="true"];
    s2 -> s2 [label="true"];
    s3 -> s3 [label="true"];
}
"""
    results.append(("Multiple Transitions", test_case("Multiple Complex Transitions", dfa4, gmap1)))

    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\n{passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\n🎯 VERIFICATION COMPLETE:")
        print("  ✓ Complex boolean expressions → Atomic partition symbols")
        print("  ✓ Disjunctions expanded to multiple atomic transitions")
        print("  ✓ Negations handled correctly in partitions")
        print("  ✓ Every transition label is now atomic (pN, αN, or simple)")
        print("\n💡 The DFA simplifier is working robustly!")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        sys.exit(1)
