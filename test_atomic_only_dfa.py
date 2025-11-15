#!/usr/bin/env python3
"""
Test: Convert DFA to atomic-only transitions

Goal: Each transition should have ONLY ONE positive atom (no negations, no conjunctions/disjunctions)

Example transformation:
  BEFORE: s1 -> s2 [label="on_d_e | (clear_c & on_a_b)"]

  AFTER:  s1 -> s2 [label="on_d_e"]
          s1 -> s_temp1 [label="clear_c"]
          s_temp1 -> s2 [label="on_a_b"]

This requires state splitting to decompose complex boolean expressions.
"""

print("=" * 80)
print("ATOMIC-ONLY DFA TRANSFORMATION")
print("=" * 80)
print()

print("INPUT:")
print("  Original transition: s1 -> s2 [label=\"on_d_e | (clear_c & on_a_b)\"]")
print()

print("EXPECTED OUTPUT:")
print("  s1 -> s2 [label=\"on_d_e\"]")
print("  s1 -> s_temp [label=\"clear_c\"]")
print("  s_temp -> s2 [label=\"on_a_b\"]")
print()

print("ALGORITHM NEEDED:")
print("  1. Parse boolean expression into AST")
print("  2. Convert to decision tree / OBDD")
print("  3. Each decision node becomes a state")
print("  4. Each edge checks exactly one atom")
print()

print("QUESTION:")
print("  Should we preserve the accepting/rejecting semantics?")
print("  For example, what if on_d_e=false, clear_c=false?")
print("  In original DFA, this would take a different transition or stay.")
print("  In atomic-only DFA, we need implicit 'else' transitions.")
print()
