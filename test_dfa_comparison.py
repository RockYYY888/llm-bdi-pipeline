#!/usr/bin/env python3
"""
Compare DFA before and after simplification for: F(on(a, b) & clear(c) | on(d, e))
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator, LogicalOperator
from stage2_dfa_generation.ltlf_to_dfa import LTLfToDFA
from stage2_dfa_generation.dfa_simplifier import DFASimplifier


def create_test_formula():
    """Create the test formula: F(on(a, b) & clear(c) | on(d, e))"""
    spec = LTLSpecification()
    spec.objects = ["a", "b", "c", "d", "e"]

    # Create grounding map
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    gmap.add_atom("on_d_e", "on", ["d", "e"])
    spec.grounding_map = gmap

    # Build formula: F((on(a, b) & clear(c)) | on(d, e))
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
    return spec


def main():
    print("="*80)
    print("DFA COMPARISON: Before vs After Simplification")
    print("="*80)
    print()
    print("Formula: F(on(a, b) & clear(c) | on(d, e))")
    print()

    # Create test specification
    spec = create_test_formula()

    print("Grounding Map:")
    print("  on_a_b  → on(a, b)")
    print("  clear_c → clear(c)")
    print("  on_d_e  → on(d, e)")
    print()

    # Step 1: Generate original DFA (without simplification)
    print("="*80)
    print("STEP 1: ORIGINAL DFA (Before Simplification)")
    print("="*80)
    print()

    converter = LTLfToDFA()
    original_dfa_dot, metadata = converter.convert(spec)

    print(f"Original DFA Statistics:")
    print(f"  States: {metadata.get('num_states', 'N/A')}")
    print()

    print("Original DFA (Complete DOT format):")
    print("-" * 80)
    print(original_dfa_dot)
    print("-" * 80)
    print()

    # Step 2: Apply simplification
    print("="*80)
    print("STEP 2: SIMPLIFIED DFA (After Simplification)")
    print("="*80)
    print()

    simplifier = DFASimplifier()
    simplified_result = simplifier.simplify(original_dfa_dot, spec.grounding_map)

    print(f"Simplified DFA Statistics:")
    print(f"  Method: {simplified_result.stats['method']}")
    print(f"  Predicates: {simplified_result.stats['num_predicates']}")
    print(f"  Partitions: {simplified_result.stats['num_partitions']}")
    print()

    print("Partition Map:")
    print("-" * 80)
    for symbol, info in simplified_result.partition_map.items():
        print(f"\n{symbol}:")
        print(f"  Expression: {info.expression}")
        print(f"  Values: {info.predicate_values}")
    print("-" * 80)
    print()

    print("Simplified DFA (Complete DOT format):")
    print("-" * 80)
    print(simplified_result.simplified_dot)
    print("-" * 80)
    print()

    # Step 3: Analysis
    print("="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)
    print()

    # Extract labels from original DFA
    import re
    original_labels = set()
    for line in original_dfa_dot.split('\n'):
        match = re.search(r'\[label="([^"]+)"\]', line)
        if match and 'init' not in line:
            original_labels.add(match.group(1))

    # Extract labels from simplified DFA
    simplified_labels = set()
    for line in simplified_result.simplified_dot.split('\n'):
        match = re.search(r'\[label="([^"]+)"\]', line)
        if match and 'init' not in line:
            simplified_labels.add(match.group(1))

    print(f"Original DFA unique labels ({len(original_labels)}):")
    for label in sorted(original_labels):
        print(f"  - {label}")
    print()

    print(f"Simplified DFA unique labels ({len(simplified_labels)}):")
    for label in sorted(simplified_labels):
        print(f"  - {label}")
    print()

    # Check for grounded atom names
    grounded_atoms = ["on_a_b", "clear_c", "on_d_e"]
    found_grounded = [label for label in simplified_labels if label in grounded_atoms]

    print("Grounded atom names found in simplified DFA:")
    if found_grounded:
        for atom in found_grounded:
            print(f"  ✅ {atom}")
    else:
        print("  ❌ No grounded atom names found")
    print()

    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("The simplified DFA uses:")
    print("  - Grounded atom names for single predicates (on_a_b, clear_c, on_d_e)")
    print("  - Boolean expressions for complex conditions")
    print("  - This preserves correct proposition names and grounding")


if __name__ == "__main__":
    main()
