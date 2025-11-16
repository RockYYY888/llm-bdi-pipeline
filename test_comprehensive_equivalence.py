#!/usr/bin/env python3
"""
Comprehensive DFA equivalence test for the complex formula
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator, LogicalOperator
from stage2_dfa_generation.ltlf_to_dfa import LTLfToDFA
from stage2_dfa_generation.dfa_simplifier import DFASimplifier
from tests.stage2_dfa_generation.test_dfa_equivalence_verification import verify_equivalence

# Create formula
on_a_b_pred = ("on", ["a", "b"])
clear_c_pred = ("clear", ["c"])
on_d_e_pred = ("on", ["d", "e"])

on_a_b = LTLFormula(on_a_b_pred, LogicalOperator.NONE, None, None)
clear_c = LTLFormula(clear_c_pred, LogicalOperator.NONE, None, None)
on_d_e = LTLFormula(on_d_e_pred, LogicalOperator.NONE, None, None)

# (on(a, b) & clear(c))
and_expr = LTLFormula(None, LogicalOperator.AND, on_a_b, clear_c)

# (on(a, b) & clear(c)) | on(d, e)
or_expr = LTLFormula(None, LogicalOperator.OR, and_expr, on_d_e)

# F(...)
formula = LTLFormula(None, TemporalOperator.EVENTUALLY, or_expr, None)

# Create specification
spec = LTLSpecification([formula])

# Grounding map
grounding_map = GroundingMap()
grounding_map.add_grounded_atom("on_a_b", {"on": ["a", "b"]})
grounding_map.add_grounded_atom("clear_c", {"clear": ["c"]})
grounding_map.add_grounded_atom("on_d_e", {"on": ["d", "e"]})

# Generate DFA
converter = LTLfToDFA(grounding_map)
original_dfa = converter.generate_dfa(spec)

# Simplify
simplifier = DFASimplifier()
result = simplifier.simplify(original_dfa, grounding_map)

print("="*80)
print("COMPREHENSIVE EQUIVALENCE CHECK")
print("="*80)

# Test all possible valuations
all_atoms = ["clear_c", "on_a_b", "on_d_e"]

is_equiv, counterexamples = verify_equivalence(original_dfa, result.simplified_dot, all_atoms)

if is_equiv:
    print("\n✅ DFAs ARE EQUIVALENT")
    print(f"Tested all {2**len(all_atoms)} valuations - all match!")
else:
    print(f"\n❌ DFAs ARE NOT EQUIVALENT")
    print(f"Found {len(counterexamples)} counterexample(s):")
    for ce in counterexamples[:10]:
        print(f"  Valuation: {ce['valuation']}")
        print(f"    Original: {ce['original']}, Simplified: {ce['simplified']}")

sys.exit(0 if is_equiv else 1)
