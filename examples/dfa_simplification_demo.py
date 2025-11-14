#!/usr/bin/env python3
"""
DFA Simplification Demo

Demonstrates how to simplify DFA transition labels from complex boolean
expressions to atomic partition symbols.

This example shows:
1. Creating a DFA with complex boolean expressions
2. Simplifying it to atomic partitions
3. Understanding the partition mapping
4. Visualizing the transformation
"""

import sys
from pathlib import Path

# Add src to path
_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Import directly to avoid dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dfa_simplifier",
    str(Path(__file__).parent.parent / "src" / "stage2_dfa_generation" / "dfa_simplifier.py")
)
dfa_simplifier_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfa_simplifier_module)

DFASimplifier = dfa_simplifier_module.DFASimplifier

from stage1_interpretation.grounding_map import GroundingMap


def demo_blocksworld_dfa():
    """Demo: Simplify a Blocksworld DFA with complex goal expressions"""

    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "DFA SIMPLIFICATION DEMO" + " " * 35 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Example: Blocksworld DFA with complex boolean expressions
    # Goal: Stack blocks a, b, c in a tower
    dfa_dot = """
digraph BlocksworldDFA {
    rankdir=LR;

    // States
    node [shape=circle];
    s0 [label="Initial"];
    s1 [label="Partial"];
    s2 [label="Goal", shape=doublecircle];

    // Transitions with complex boolean expressions
    init -> s0;
    s0 -> s1 [label="on_a_b | clear_a"];
    s1 -> s2 [label="(on_a_b & on_b_c) | holding_a"];
    s1 -> s1 [label="~on_a_b & clear_a"];
}
"""

    print("ORIGINAL DFA")
    print("=" * 80)
    print()
    print("Transitions with complex boolean expressions:")
    print("  s0 → s1:  on_a_b | clear_a")
    print("  s1 → s2:  (on_a_b & on_b_c) | holding_a")
    print("  s1 → s1:  ~on_a_b & clear_a")
    print()
    print("Problem: These are complex boolean expressions, not atomic predicates!")
    print()

    # Create grounding map
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("on_b_c", "on", ["b", "c"])
    gmap.add_atom("clear_a", "clear", ["a"])
    gmap.add_atom("holding_a", "holding", ["a"])

    print("PREDICATES")
    print("=" * 80)
    predicates = ["on_a_b", "on_b_c", "clear_a", "holding_a"]
    print(f"Found {len(predicates)} atomic predicates:")
    for pred in predicates:
        print(f"  - {pred}")
    print()
    print(f"Naive approach: Generate all 2^{len(predicates)} = {2**len(predicates)} minterms")
    print("Our approach: Generate only USED minterms (partition refinement)")
    print()

    # Simplify
    print("SIMPLIFICATION")
    print("=" * 80)
    print()

    simplifier = DFASimplifier()
    result = simplifier.simplify(dfa_dot, gmap)

    print(f"✓ Simplification complete!")
    print()
    print(f"Statistics:")
    print(f"  Method: {result.stats['method']}")
    print(f"  Atomic predicates: {result.stats['num_predicates']}")
    print(f"  Partitions generated: {result.stats['num_partitions']}")
    print(f"  Total possible: {2**result.stats['num_predicates']}")
    print(f"  Compression ratio: {result.stats['compression_ratio']:.2f}x")
    print()

    # Show partitions
    print("PARTITIONS GENERATED")
    print("=" * 80)
    print()
    print("Each partition represents a unique combination of predicate values:")
    print()

    for partition in result.partitions:
        print(f"  {partition.symbol}: {partition.expression}")

        # Show truth values
        if partition.predicate_values:
            values_str = ", ".join(
                f"{k}={'T' if v else 'F'}"
                for k, v in sorted(partition.predicate_values.items())
            )
            print(f"         ({values_str})")
        print()

    # Show mapping
    print("LABEL → PARTITION MAPPING")
    print("=" * 80)
    print()
    print("Each original boolean expression maps to one or more partitions:")
    print()

    for label, partition_symbols in result.original_label_to_partitions.items():
        print(f"  '{label}'")
        print(f"    → {', '.join(partition_symbols)}")
        print()

        # Show what each partition means
        for symbol in partition_symbols[:3]:  # Show first 3
            partition = result.partition_map[symbol]
            print(f"       {symbol}: {partition.expression}")

        if len(partition_symbols) > 3:
            print(f"       ... and {len(partition_symbols) - 3} more")
        print()

    # Show simplified DFA
    print("SIMPLIFIED DFA")
    print("=" * 80)
    print()
    print("Transitions now use atomic partition symbols:")
    print()

    # Extract and show transitions from simplified DOT
    import re
    transitions = re.findall(
        r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]',
        result.simplified_dot
    )

    for from_state, to_state, label in transitions:
        if from_state not in ['init', '__start']:
            partition = result.partition_map.get(label)
            if partition:
                print(f"  {from_state} → {to_state}:  {label}")
                print(f"      Meaning: {partition.expression}")
            else:
                print(f"  {from_state} → {to_state}:  {label}")

    print()
    print("BENEFITS")
    print("=" * 80)
    print()
    print("✓ Each transition now has a SINGLE atomic symbol")
    print("✓ Partitions are mutually exclusive (no overlap)")
    print("✓ Complete coverage (all possible inputs)")
    print("✓ Minimal set (no redundant partitions)")
    print("✓ Lossless (can reconstruct original expressions)")
    print()
    print("This enables:")
    print("  • Easier formal verification")
    print("  • Deterministic execution")
    print("  • Clear input-output mapping")
    print("  • Efficient code generation")
    print()


def demo_scalability():
    """Demo: Show scalability with different predicate counts"""

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "SCALABILITY DEMO" + " " * 37 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    test_cases = [
        (2, "Small domain (2 predicates)"),
        (5, "Medium domain (5 predicates)"),
        (10, "Large domain (10 predicates)"),
    ]

    print("Testing simplification with different predicate counts:")
    print()

    for num_preds, description in test_cases:
        # Generate test DFA
        predicates = [f"p{i}" for i in range(num_preds)]

        # Create simple DFA with one complex expression
        expr = " | ".join(predicates[:min(3, num_preds)])

        dfa_dot = f"""
digraph G {{
    s0 -> s1 [label="{expr}"];
}}
"""

        gmap = GroundingMap()
        for pred in predicates:
            gmap.add_atom(pred, pred, [])

        simplifier = DFASimplifier()
        result = simplifier.simplify(dfa_dot, gmap)

        print(f"  {description}:")
        print(f"    Method: {result.stats['method']}")
        print(f"    Partitions: {result.stats['num_partitions']} / {2**num_preds} possible")
        print(f"    Compression: {result.stats['compression_ratio']:.2f}x")
        print()

    print("Note: For >15 predicates, install BDD library for better performance:")
    print("      pip install dd")
    print()


if __name__ == "__main__":
    demo_blocksworld_dfa()
    demo_scalability()

    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 33 + "DEMO COMPLETE" + " " * 33 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    print("For more information, see:")
    print("  • docs/dfa_simplification_design.md")
    print("  • docs/dfa_simplification_usage.md")
    print("  • tests/stage2_dfa_generation/test_dfa_simplifier.py")
    print()
