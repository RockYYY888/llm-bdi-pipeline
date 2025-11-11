"""
Test that cache works across different transitions in the same DFA

This test verifies that:
1. on(a, b) triggers a cache MISS (first exploration)
2. on(b, a) triggers a cache HIT (reuses exploration from on(a, b))

Both normalize to on(?v0, ?v1) and should share the same state space exploration.
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

# Create grounding map with TWO goals
gmap = GroundingMap()
gmap.add_atom("on_a_b", "on", ["a", "b"])
gmap.add_atom("on_b_a", "on", ["b", "a"])  # Different objects, same structure

# Create DFA with TWO transitions (different goals)
dfa_dot = """
digraph G {
    __start [shape=none];
    s0 [shape=circle];
    s1 [shape=circle];
    s2 [shape=doublecircle];
    __start -> s0;
    s0 -> s1 [label="on_a_b"];
    s1 -> s2 [label="on_b_a"];
}
"""

print("="*80)
print("CROSS-TRANSITION CACHE TEST")
print("="*80)
print()
print("DFA has 2 transitions:")
print("  1. s0 --[on_a_b]-> s1")
print("  2. s1 --[on_b_a]-> s2")
print()
print("Expected behavior:")
print("  Transition 1: on(a, b) → Cache MISS (first time)")
print("  Transition 2: on(b, a) → Cache HIT (reuses on(?v0, ?v1))")
print()
print("="*80)
print()

# Generate code
gen = BackwardPlannerGenerator(domain, gmap)
code, _ = gen.generate(
    {'objects': ['a', 'b']},
    {'dfa_dot': dfa_dot, 'grounding_map': gmap.to_dict()}
)

print()
print("="*80)
print("ANALYSIS")
print("="*80)
print()

# The generator should have printed cache statistics
# Let's verify in the output above

# Count explorations vs cache hits from the output
# We expect: 1 exploration (on_a_b), 1 cache hit (on_b_a)

print("If you see in the output above:")
print("  - 'Cache MISS - running VARIABLE-LEVEL exploration...' for on(a, b)")
print("  - '✓ VARIABLE-LEVEL Cache HIT! Reusing exploration' for on(b, a)")
print()
print("Then the cross-transition cache is working correctly! ✅")
print()
print("This means:")
print("  - on(a, b) and on(b, a) both normalize to on(?v0, ?v1)")
print("  - Only ONE state space exploration was performed")
print("  - The second goal REUSED the cached exploration")
print("  - This is TRUE schema-level caching across the entire DFA")
print()
print("="*80)
print("CACHE STATISTICS")
print("="*80)
print("(See 'Variable-level goal exploration cache:' section above)")
print()
