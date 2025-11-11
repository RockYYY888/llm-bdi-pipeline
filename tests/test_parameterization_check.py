"""
Quick test to check if goal plans are parameterized or object-specific
"""
import sys
from pathlib import Path

_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from stage1_interpretation.grounding_map import GroundingMap
from utils.pddl_parser import PDDLParser

# Load domain
domain_file = Path(__file__).parent.parent / "src" / "domains" / "blocksworld" / "domain.pddl"
domain = PDDLParser.parse_domain(str(domain_file))

# Create grounding map
gmap = GroundingMap()
gmap.add_atom("on_a_b", "on", ["a", "b"])

# Create simple DFA
dfa_dot = """
digraph G {
    __start [shape=none];
    s0 [shape=circle];
    s1 [shape=doublecircle];
    __start -> s0;
    s0 -> s1 [label="on_a_b"];
}
"""

# Generate code
gen = BackwardPlannerGenerator(domain, gmap)
code, _ = gen.generate(
    {'objects': ['a', 'b']},
    {'dfa_dot': dfa_dot, 'grounding_map': gmap.to_dict()}
)

print("="*80)
print("PARAMETERIZATION CHECK")
print("="*80)
print()

# Look for goal achievement plans
lines = code.split('\n')
for i, line in enumerate(lines):
    # Look for goal plan definitions
    if '+!on(' in line or '-!on(' in line:
        # Show context (5 lines before and after)
        start = max(0, i-2)
        end = min(len(lines), i+3)
        print(f"Found goal plan at line {i+1}:")
        for j in range(start, end):
            marker = ">>> " if j == i else "    "
            print(f"{marker}{lines[j]}")
        print()

# Check for specific patterns
has_parameterized = "on(V0, V1)" in code or "on(X, Y)" in code or "on(Arg0, Arg1)" in code
has_object_specific = "+!on(a, b)" in code or "-!on(a, b)" in code

print("="*80)
print("RESULT:")
print("="*80)
if has_parameterized:
    print("✓ Found PARAMETERIZED goal plans (CORRECT)")
    print("  Pattern: on(X, Y) or on(V0, V1) or on(Arg0, Arg1)")
elif has_object_specific:
    print("✗ Found OBJECT-SPECIFIC goal plans (WRONG)")
    print("  Pattern: on(a, b)")
else:
    print("? Could not determine - please check manually")

print()
print("First 100 lines of generated code:")
print("="*80)
for i, line in enumerate(lines[:100], 1):
    print(f"{i:3}: {line}")
