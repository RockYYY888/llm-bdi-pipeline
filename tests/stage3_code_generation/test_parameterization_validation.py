"""
Analyze generated AgentSpeak code to identify issues
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'src'))

from stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from utils.pddl_parser import PDDLParser
from stage1_interpretation.grounding_map import GroundingMap, GroundedAtom

# Load domain
domain_file = Path('src/domains/blocksworld/domain.pddl')
domain = PDDLParser.parse_domain(str(domain_file))

# Create grounding map
gmap = GroundingMap()
gmap.add_atom("on_a_b", "on", ["a", "b"])
gmap.add_atom("clear_a", "clear", ["a"])

# Create generator
generator = BackwardPlannerGenerator(domain, gmap)

# Simple test case
ltl_dict = {
    'objects': ['a', 'b'],
    'formulas_string': ['F(on_a_b)']
}

dfa_result = {
    'formula': 'F(on_a_b)',
    'dfa_dot': '''digraph G {
    state0 [label="state0"];
    state1 [label="state1" shape=doublecircle];
    state0 -> state1 [label="on_a_b"];
}'''
}

# Generate code
print("Generating AgentSpeak code...")
code, truncated = generator.generate(ltl_dict, dfa_result)

print("\n" + "=" * 80)
print("ANALYSIS OF GENERATED CODE")
print("=" * 80)

lines = code.split('\n')

print(f"\nTotal lines: {len(lines)}")
print()

# Find goal achievement plans
print("=" * 80)
print("GOAL ACHIEVEMENT PLANS")
print("=" * 80)

in_goal_section = False
goal_lines = []

for i, line in enumerate(lines):
    if '!achieve' in line or '!on' in line or '!clear' in line:
        print(f"Line {i+1}: {line}")
        goal_lines.append((i+1, line))

print(f"\nFound {len(goal_lines)} goal-related lines")

# Check for parameterization
print("\n" + "=" * 80)
print("PARAMETERIZATION ANALYSIS")
print("=" * 80)

has_concrete = False
has_parameterized = False

for i, line in goal_lines:
    if '!achieve_on_a_b' in line or '!achieve_clear_a' in line:
        has_concrete = True
        print(f"✗ CONCRETE (line {i}): {line.strip()}")
    if '!achieve_on(X, Y)' in line or '!on(X, Y)' in line:
        has_parameterized = True
        print(f"✓ PARAMETERIZED (line {i}): {line.strip()}")

print()
if has_concrete and not has_parameterized:
    print("❌ ISSUE: Only concrete goals found!")
    print("   Generated plans are object-specific and won't work for arbitrary objects")
elif has_parameterized:
    print("✅ GOOD: Parameterized goals found")
else:
    print("⚠️  WARNING: No clear goal patterns detected")

# Check action plans
print("\n" + "=" * 80)
print("ACTION PLAN ANALYSIS")
print("=" * 80)

action_lines = []
for i, line in enumerate(lines):
    if '+!pick' in line or '+!put' in line:
        action_lines.append((i+1, line))

for i, line in action_lines[:5]:  # Show first 5
    print(f"Line {i}: {line}")

# Check if using variables
uses_variables = any('(B' in line or '(X' in line or '(Y' in line for _, line in action_lines)
print(f"\nUses AgentSpeak variables: {'✅ YES' if uses_variables else '❌ NO'}")

# Show initial beliefs
print("\n" + "=" * 80)
print("INITIAL BELIEFS")
print("=" * 80)

for i, line in enumerate(lines):
    if i < 50 and ('ontable(' in line or 'clear(' in line or 'handempty' in line):
        if line.strip() and not line.strip().startswith('/*'):
            print(f"Line {i+1}: {line.strip()}")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

findings = []

if has_concrete:
    findings.append("❌ Generates object-specific goals (e.g., !achieve_on_a_b)")
    findings.append("   → Won't work for different object pairs")
    findings.append("   → Needs parameterization: !achieve_on(X, Y)")

if uses_variables:
    findings.append("✅ Action plans use AgentSpeak variables correctly")
else:
    findings.append("❌ Action plans don't use variables")

# Check initial beliefs
concrete_beliefs = [l for i, l in enumerate(lines) if 'ontable(a)' in l or 'clear(b)' in l]
if concrete_beliefs:
    findings.append("✅ Initial beliefs are concrete (correct for problem instance)")

for finding in findings:
    print(finding)

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("""
1. Goal Achievement Plans should be PARAMETERIZED:
   WRONG: +!achieve_on_a_b : ...
   RIGHT: +!achieve_on(X, Y) : ...

2. Goal should be triggered with CONCRETE arguments:
   !achieve_on(a, b)

3. This allows the same plan to work for ANY object pair:
   !achieve_on(a, b)  → uses same plan
   !achieve_on(c, d)  → uses same plan
   !achieve_on(x, y)  → uses same plan

4. Type checking should ensure X, Y are blocks (or correct type)
""")
