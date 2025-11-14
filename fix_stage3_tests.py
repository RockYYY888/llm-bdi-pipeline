#!/usr/bin/env python3
"""
Fix test_stage3_complete.py to add grounding_map before DFA build

This script adds spec.grounding_map before all builder.build(spec) calls
"""

import re

# Read the file
with open('tests/stage3_code_generation/test_stage3_complete.py', 'r') as f:
    content = f.read()

# Pattern 1: Find cases where grounding_map is created AFTER builder.build
# Example pattern:
#   builder = DFABuilder()
#   dfa_result = builder.build(spec)
#   ...
#   grounding_map = GroundingMap()
#   grounding_map.add_atom(...)

# We need to move grounding_map creation BEFORE builder.build and add to spec

# Let's look for specific patterns and fix them

# Pattern for test_2_1_globally_negation (line 414)
pattern_2_1 = r'''(    spec = LTLSpecification\(\)
    spec\.objects = \["a", "b"\]
    spec\.formulas = \[g_formula\]

    # Build DFA
    builder = DFABuilder\(\)
    dfa_result = builder\.build\(spec\))
    print\(f"  ✓ DFA generated.*?\)

    # Create grounding map
    (grounding_map = GroundingMap\(\)
    grounding_map\.add_atom\("on_a_b", "on", \["a", "b"\]\))'''

replacement_2_1 = r'''    spec = LTLSpecification()
    spec.objects = ["a", "b"]
    spec.formulas = [g_formula]

    # Create grounding map (required for DFA simplification)
    \2
    spec.grounding_map = grounding_map

    # Build DFA
    builder = DFABuilder()
    dfa_result = builder.build(spec)
    print(f"  ✓ DFA generated: {dfa_result['num_states']} states, {dfa_result['num_transitions']} transitions")'''

content = re.sub(pattern_2_1, replacement_2_1, content, flags=re.DOTALL)

# Similar patterns for other tests...
# This is getting complex. Let me do a more targeted approach.

print("Fixed test_stage3_complete.py patterns")
print("Writing updated file...")

with open('tests/stage3_code_generation/test_stage3_complete.py', 'w') as f:
    f.write(content)

print("Done!")
