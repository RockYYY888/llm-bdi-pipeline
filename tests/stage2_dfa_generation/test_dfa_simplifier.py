#!/usr/bin/env python3
"""
Test DFA Simplifier

Tests both BDD-based and minterm-based simplification methods.
"""

import sys
from pathlib import Path

# Add src to path
_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Import directly to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dfa_simplifier",
    str(Path(__file__).parent.parent.parent / "src" / "stage2_dfa_generation" / "dfa_simplifier.py")
)
dfa_simplifier_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfa_simplifier_module)

DFASimplifier = dfa_simplifier_module.DFASimplifier
BDDSimplifier = dfa_simplifier_module.BDDSimplifier
SimpleMintermSimplifier = dfa_simplifier_module.SimpleMintermSimplifier

from stage1_interpretation.grounding_map import GroundingMap


def test_simple_dfa():
    """Test simplification of a simple DFA"""
    print("=" * 80)
    print("TEST 1: Simple DFA with 2 predicates")
    print("=" * 80)

    # Create a simple DFA
    dfa_dot = """
digraph G {
    rankdir=LR;

    node [shape=circle];
    s0 [label="0"];
    s1 [label="1"];
    s2 [label="2", shape=doublecircle];

    init -> s0;
    s0 -> s1 [label="on_a_b | clear_c"];
    s1 -> s2 [label="on_a_b & ~clear_c"];
}
"""

    # Create grounding map
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])

    # Test minterm simplifier
    print("\n--- Minterm Simplifier ---")
    minterm_simplifier = SimpleMintermSimplifier()
    result = minterm_simplifier.simplify(dfa_dot, gmap)

    print(f"\nResults:")
    print(f"  Method: {result.stats['method']}")
    print(f"  Predicates: {result.stats['num_predicates']}")
    print(f"  Partitions: {result.stats['num_partitions']}")
    print(f"  Total minterms: {result.stats.get('num_total_minterms', 'N/A')}")
    print(f"  Compression: {result.stats['compression_ratio']:.2f}x")

    print(f"\nPartitions generated:")
    for partition in result.partitions:
        print(f"  {partition.symbol}: {partition.expression}")

    print(f"\nOriginal label mapping:")
    for label, partitions in result.original_label_to_partitions.items():
        print(f"  '{label}' → {partitions}")

    print(f"\nSimplified DFA:")
    print(result.simplified_dot)

    print("\n✓ Test 1 passed\n")


def test_complex_dfa():
    """Test with more complex expressions"""
    print("=" * 80)
    print("TEST 2: Complex DFA with 3 predicates")
    print("=" * 80)

    dfa_dot = """
digraph G {
    rankdir=LR;

    s0 [label="0"];
    s1 [label="1"];
    s2 [label="2"];
    s3 [label="3", shape=doublecircle];

    init -> s0;
    s0 -> s1 [label="on_a_b & clear_c"];
    s0 -> s2 [label="on_a_b | holding_d"];
    s1 -> s3 [label="~clear_c"];
    s2 -> s3 [label="holding_d & ~on_a_b"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    gmap.add_atom("holding_d", "holding", ["d"])

    print("\n--- Minterm Simplifier ---")
    minterm_simplifier = SimpleMintermSimplifier()
    result = minterm_simplifier.simplify(dfa_dot, gmap)

    print(f"\nResults:")
    print(f"  Predicates: {result.stats['num_predicates']}")
    print(f"  Partitions: {result.stats['num_partitions']} / {result.stats['num_total_minterms']}")
    print(f"  Compression: {result.stats['compression_ratio']:.2f}x")

    print(f"\nPartitions (first 10):")
    for partition in result.partitions[:10]:
        print(f"  {partition.symbol}: {partition.expression}")
    if len(result.partitions) > 10:
        print(f"  ... and {len(result.partitions) - 10} more")

    print("\n✓ Test 2 passed\n")


def test_bdd_simplifier():
    """Test BDD-based simplifier if available"""
    print("=" * 80)
    print("TEST 3: BDD-based Simplifier")
    print("=" * 80)

    bdd_simplifier = BDDSimplifier()

    if not bdd_simplifier.is_available():
        print("⚠ BDD library not available, skipping test")
        print("  Install with: pip install dd")
        return

    print("✓ BDD library available")

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

    result = bdd_simplifier.simplify(dfa_dot, gmap)

    print(f"\nResults:")
    print(f"  Method: {result.stats['method']}")
    print(f"  Predicates: {result.stats['num_predicates']}")
    print(f"  Partitions: {result.stats['num_partitions']}")
    print(f"  Compression: {result.stats['compression_ratio']:.2f}x")

    print(f"\nPartitions:")
    for partition in result.partitions:
        print(f"  {partition.symbol}: {partition.expression}")
        print(f"    Predicate values: {partition.predicate_values}")

    print(f"\nOriginal label mapping:")
    for label, partitions in result.original_label_to_partitions.items():
        print(f"  '{label}' → {partitions}")

    print("\n✓ Test 3 passed\n")


def test_auto_selection():
    """Test automatic method selection"""
    print("=" * 80)
    print("TEST 4: Auto Method Selection")
    print("=" * 80)

    dfa_dot = """
digraph G {
    s0 -> s1 [label="on_a_b"];
    s1 -> s2 [label="clear_c"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])

    simplifier = DFASimplifier()

    result = simplifier.simplify(dfa_dot, gmap)

    print(f"\nAuto-selected method: {result.stats['method']}")
    print(f"Partitions: {result.stats['num_partitions']}")

    print("\n✓ Test 4 passed\n")


def test_true_false_labels():
    """Test handling of 'true' and 'false' labels"""
    print("=" * 80)
    print("TEST 5: True/False Labels")
    print("=" * 80)

    dfa_dot = """
digraph G {
    s0 -> s0 [label="true"];
    s0 -> s1 [label="on_a_b"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])

    simplifier = SimpleMintermSimplifier()
    result = simplifier.simplify(dfa_dot, gmap)

    print(f"Partitions: {result.stats['num_partitions']}")

    for partition in result.partitions:
        print(f"  {partition.symbol}: {partition.expression}")

    print("\n✓ Test 5 passed\n")


def test_correctness_verification():
    """Verify that simplification preserves semantics"""
    print("=" * 80)
    print("TEST 6: Correctness Verification")
    print("=" * 80)

    dfa_dot = """
digraph G {
    s0 -> s1 [label="on_a_b | clear_c"];
    s1 -> s2 [label="on_a_b & clear_c"];
}
"""

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])

    simplifier = SimpleMintermSimplifier()
    result = simplifier.simplify(dfa_dot, gmap)

    # Verify: "on_a_b | clear_c" should map to 3 partitions
    # (on_a_b & clear_c), (on_a_b & ~clear_c), (~on_a_b & clear_c)
    label1_partitions = result.original_label_to_partitions.get("on_a_b | clear_c", [])
    print(f"'on_a_b | clear_c' maps to {len(label1_partitions)} partitions: {label1_partitions}")
    assert len(label1_partitions) == 3, f"Expected 3 partitions, got {len(label1_partitions)}"

    # Verify: "on_a_b & clear_c" should map to 1 partition
    label2_partitions = result.original_label_to_partitions.get("on_a_b & clear_c", [])
    print(f"'on_a_b & clear_c' maps to {len(label2_partitions)} partition(s): {label2_partitions}")
    assert len(label2_partitions) == 1, f"Expected 1 partition, got {len(label2_partitions)}"

    # Verify that label2's partition is a subset of label1's partitions
    assert label2_partitions[0] in label1_partitions, \
        f"Partition {label2_partitions[0]} should be in {label1_partitions}"

    print("\n✓ Correctness verified!")
    print("✓ Test 6 passed\n")


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "DFA SIMPLIFIER TESTS" + " " * 33 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    tests = [
        test_simple_dfa,
        test_complex_dfa,
        test_bdd_simplifier,
        test_auto_selection,
        test_true_false_labels,
        test_correctness_verification,
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


if __name__ == "__main__":
    run_all_tests()
