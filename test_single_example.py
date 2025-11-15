#!/usr/bin/env python3
"""
Test single example: F(on(a, b) & clear(c) | on(d, e))

This test verifies the DFA simplification for a specific complex formula.
"""

import sys
import re
from pathlib import Path

_src_dir = str(Path(__file__).parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator, LogicalOperator
from stage2_dfa_generation.dfa_builder import DFABuilder


def test_complex_disjunction():
    """Test: F(on(a, b) & clear(c) | on(d, e))"""

    print("="*80)
    print("TEST: F(on(a, b) & clear(c) | on(d, e))")
    print("="*80)

    # Create LTL specification
    spec = LTLSpecification()
    spec.objects = ["a", "b", "c", "d", "e"]

    # Create grounding map
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    gmap.add_atom("on_d_e", "on", ["d", "e"])
    spec.grounding_map = gmap

    print("\nGrounding Map:")
    print(f"  on_a_b  → on(a, b)")
    print(f"  clear_c → clear(c)")
    print(f"  on_d_e  → on(d, e)")

    # Build formula: F((on(a, b) & clear(c)) | on(d, e))
    # Use dict format for predicates: {"predicate_name": ["arg1", "arg2", ...]}
    on_a_b = LTLFormula(
        operator=None,
        predicate={"on": ["a", "b"]},
        sub_formulas=[],
        logical_op=None
    )

    clear_c = LTLFormula(
        operator=None,
        predicate={"clear": ["c"]},
        sub_formulas=[],
        logical_op=None
    )

    on_d_e = LTLFormula(
        operator=None,
        predicate={"on": ["d", "e"]},
        sub_formulas=[],
        logical_op=None
    )

    # Conjunction: on(a, b) & clear(c)
    conj = LTLFormula(
        operator=None,
        predicate=None,
        sub_formulas=[on_a_b, clear_c],
        logical_op=LogicalOperator.AND
    )

    # Disjunction: (on(a, b) & clear(c)) | on(d, e)
    disj = LTLFormula(
        operator=None,
        predicate=None,
        sub_formulas=[conj, on_d_e],
        logical_op=LogicalOperator.OR
    )

    # Finally: F((on(a, b) & clear(c)) | on(d, e))
    f_formula = LTLFormula(
        operator=TemporalOperator.FINALLY,
        predicate=None,
        sub_formulas=[disj],
        logical_op=None
    )

    spec.formulas = [f_formula]

    print(f"\nLTL Formula: {f_formula.to_string()}")

    # Build DFA
    print("\n" + "="*80)
    print("Building DFA...")
    print("="*80)

    builder = DFABuilder()
    dfa_result = builder.build(spec)

    print(f"\nDFA Statistics:")
    print(f"  States: {dfa_result['num_states']}")
    print(f"  Transitions: {dfa_result['num_transitions']}")
    print(f"  Simplification method: {dfa_result['simplification_stats']['method']}")
    print(f"  Predicates: {dfa_result['simplification_stats']['num_predicates']}")
    print(f"  Partitions: {dfa_result['simplification_stats']['num_partitions']}")

    # Show partition map
    if dfa_result.get('partition_map'):
        print(f"\n" + "="*80)
        print("Partition Map:")
        print("="*80)
        for symbol, partition in dfa_result['partition_map'].items():
            print(f"\n{symbol}:")
            print(f"  Expression: {partition.expression}")
            print(f"  Values: {partition.predicate_values}")

    # Extract and show transition labels
    print(f"\n" + "="*80)
    print("DFA Transition Labels:")
    print("="*80)

    labels = set()
    for line in dfa_result['dfa_dot'].split('\n'):
        match = re.search(r'\[label="([^"]+)"\]', line)
        if match and 'init' not in line:
            labels.add(match.group(1))

    print(f"\nUnique labels ({len(labels)}):")
    for label in sorted(labels):
        print(f"  - {label}")

    # Show sample transitions
    print(f"\n" + "="*80)
    print("Sample Transitions (first 10):")
    print("="*80)
    count = 0
    for line in dfa_result['dfa_dot'].split('\n'):
        if '->' in line and 'init' not in line and '[label=' in line:
            print(f"  {line.strip()}")
            count += 1
            if count >= 10:
                break

    # Check if labels are partition symbols or grounded atoms
    print(f"\n" + "="*80)
    print("Label Analysis:")
    print("="*80)

    partition_symbols = [label for label in labels if label in dfa_result.get('partition_map', {})]
    grounded_atoms = [label for label in labels if label in ['on_a_b', 'clear_c', 'on_d_e', 'true', 'false']]
    complex_expr = [label for label in labels if label not in partition_symbols and label not in grounded_atoms]

    print(f"\nPartition symbols (e.g., p1, p2): {len(partition_symbols)}")
    if partition_symbols:
        print(f"  {partition_symbols[:5]}")

    print(f"\nGrounded atoms (e.g., on_a_b): {len(grounded_atoms)}")
    if grounded_atoms:
        print(f"  {grounded_atoms}")

    print(f"\nComplex expressions: {len(complex_expr)}")
    if complex_expr:
        print(f"  {complex_expr[:5]}")

    # Issue detection
    print(f"\n" + "="*80)
    print("ISSUE DETECTION:")
    print("="*80)

    if partition_symbols:
        print(f"\n⚠️  Found {len(partition_symbols)} partition symbols (p1, p2, etc.)")
        print(f"    Expected: Grounded atom names (on_a_b, clear_c, on_d_e)")
        print(f"    Current: {partition_symbols[:10]}")
        print(f"\n❌ BUG: DFA labels should use grounded atom names, not partition symbols!")
    else:
        print(f"\n✅ No partition symbols found - using grounded atom names")

    return dfa_result


if __name__ == "__main__":
    result = test_complex_disjunction()

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("\nThe DFA simplifier currently generates partition symbols (p1, p2, ...)")
    print("but should instead use the original grounded proposition names.")
    print("\nExpected behavior:")
    print("  - Each transition should be labeled with a grounded atom (e.g., on_a_b)")
    print("  - NOT with partition symbols (e.g., p1)")
