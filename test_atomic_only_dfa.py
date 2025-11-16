#!/usr/bin/env python3
"""
Test atomic-only DFA transitions
Verify that simplified DFA has only atomic labels (single atoms with or without negation)
"""

import sys
from pathlib import Path
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage1_interpretation.ltlf_formula import LTLFormula
from stage1_interpretation.grounding_map import GroundingMap
from stage2_dfa_generation.ltlf_to_dfa import translate_ltlf_to_dfa
from stage2_dfa_generation.dfa_simplifier import DFASimplifier


def create_test_formula():
    """Create the test formula: F(on(a, b) & clear(c) | on(d, e))"""
    # Build formula: F((on(a, b) & clear(c)) | on(d, e))
    on_a_b = LTLFormula(
        operator=None,
        predicate={"on": ["a", "b"]},
        left=None,
        right=None
    )

    clear_c = LTLFormula(
        operator=None,
        predicate={"clear": ["c"]},
        left=None,
        right=None
    )

    on_d_e = LTLFormula(
        operator=None,
        predicate={"on": ["d", "e"]},
        left=None,
        right=None
    )

    # Build: on(a, b) & clear(c)
    and_part = LTLFormula(
        operator="&",
        predicate=None,
        left=on_a_b,
        right=clear_c
    )

    # Build: (on(a, b) & clear(c)) | on(d, e)
    or_part = LTLFormula(
        operator="|",
        predicate=None,
        left=and_part,
        right=on_d_e
    )

    # Build: F(...)
    formula = LTLFormula(
        operator="F",
        predicate=None,
        left=or_part,
        right=None
    )

    return formula


def is_atomic_label(label: str) -> bool:
    """
    Check if a label is atomic (single atom with optional negation)

    Atomic labels:
    - "atom_name" (positive literal)
    - "!atom_name" (negated literal)
    - "true" (special case)

    NOT atomic:
    - "atom1 & atom2"
    - "atom1 | atom2"
    - "(atom1 & atom2) | atom3"
    """
    if label == "true":
        return True

    # Remove leading negation
    if label.startswith("!") or label.startswith("~"):
        label = label[1:]

    # Should be a single identifier (no operators)
    # Check for boolean operators
    if any(op in label for op in ['&', '|', '(', ')', ' ']):
        return False

    # Should be valid identifier
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', label))


def parse_transitions(dfa_dot: str):
    """Parse transitions from DOT format"""
    transitions = []
    for line in dfa_dot.split('\n'):
        match = re.match(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]', line.strip())
        if match:
            from_state, to_state, label = match.groups()
            if from_state not in ['init', '__start']:
                transitions.append((from_state, to_state, label))
    return transitions


def main():
    print("="*80)
    print("ATOMIC DFA VERIFICATION")
    print("="*80)

    # Create formula
    formula = create_test_formula()
    print("\nFormula: F(on(a, b) & clear(c) | on(d, e))")

    # Create grounding map
    grounding_map = GroundingMap()
    grounding_map.add_grounded_atom("on_a_b", {"on": ["a", "b"]})
    grounding_map.add_grounded_atom("clear_c", {"clear": ["c"]})
    grounding_map.add_grounded_atom("on_d_e", {"on": ["d", "e"]})

    print("\nGrounding Map:")
    for atom_name, pred in grounding_map.atoms.items():
        print(f"  {atom_name} → {pred}")

    # Generate original DFA
    print("\n" + "="*80)
    print("STEP 1: Generate Original DFA")
    print("="*80)

    original_dfa = translate_ltlf_to_dfa(formula, grounding_map)

    # Simplify DFA
    print("\n" + "="*80)
    print("STEP 2: Simplify DFA to Atomic Transitions")
    print("="*80)

    simplifier = DFASimplifier()
    result = simplifier.simplify(original_dfa, grounding_map)

    print(f"\nSimplified DFA Stats:")
    print(f"  Method: {result.stats['method']}")
    print(f"  Predicates: {result.stats['num_predicates']}")
    print(f"  Original states: {result.stats['num_original_states']}")
    print(f"  New states: {result.stats['num_new_states']}")
    print(f"  Original transitions: {result.stats['num_original_transitions']}")
    print(f"  New transitions: {result.stats['num_new_transitions']}")

    # Verify atomicity
    print("\n" + "="*80)
    print("STEP 3: Verify Atomic Labels")
    print("="*80)

    transitions = parse_transitions(result.simplified_dot)

    print(f"\nTotal transitions: {len(transitions)}")

    atomic_count = 0
    non_atomic_count = 0

    print("\nTransition Analysis:")
    for from_state, to_state, label in transitions:
        is_atomic = is_atomic_label(label)
        status = "✅ ATOMIC" if is_atomic else "❌ NON-ATOMIC"
        print(f"  {from_state} --[{label}]--> {to_state}  {status}")

        if is_atomic:
            atomic_count += 1
        else:
            non_atomic_count += 1

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n✅ Atomic transitions: {atomic_count}")
    print(f"❌ Non-atomic transitions: {non_atomic_count}")

    if non_atomic_count == 0:
        print("\n🎉 SUCCESS: All transitions are atomic!")
        return 0
    else:
        print("\n❌ FAILURE: Some transitions are not atomic")
        return 1


if __name__ == "__main__":
    sys.exit(main())
