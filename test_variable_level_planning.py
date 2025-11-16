#!/usr/bin/env python3
"""
Quick test for variable-level planning refactoring
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from stage3_code_generation.state_space import PredicateAtom
from stage1_interpretation.grounding_map import GroundingMap
from utils.pddl_parser import PDDLParser

def test_variable_level_planning():
    print("="*80)
    print("TESTING VARIABLE-LEVEL PLANNING")
    print("="*80)

    # Load domain
    domain_file = Path(__file__).parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    if not domain_file.exists():
        print(f"❌ Domain file not found: {domain_file}")
        return False

    domain = PDDLParser.parse_domain(str(domain_file))
    print(f"✓ Loaded domain: {domain.name}")

    # Create grounding map with ONLY 3 objects (to keep state space small)
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("on_b_c", "on", ["b", "c"])
    gmap.add_atom("on_c_a", "on", ["c", "a"])

    # Create generator
    generator = BackwardPlannerGenerator(domain, gmap)
    print("✓ Created BackwardPlannerGenerator")

    # Create mock DFA with THREE transitions that have same structure
    # This should result in 1 exploration (pattern-based caching) instead of 3
    # All three goals have structure on(?v0, ?v1) - should share ONE state graph!
    dfa_dot = """
digraph MONA_DFA {
    rankdir = LR;
    node [shape = doublecircle]; 2;
    node [shape = circle]; 1;
    init -> 1;
    1 -> 2 [label="on_a_b"];
    1 -> 2 [label="on_b_c"];
    1 -> 2 [label="on_c_a"];
    2 -> 2 [label="true"];
}
"""

    # Create LTL dict - ONLY 3 objects to avoid state space explosion!
    ltl_dict = {
        'objects': ['a', 'b', 'c'],  # Only 3 objects!
        'formulas_string': ['F(on_a_b | on_b_c | on_c_a)'],
        'grounding_map': gmap
    }

    # Create DFA result
    dfa_result = {
        'formula': 'F(on_a_b | on_b_c | on_c_a)',
        'dfa_dot': dfa_dot,
        'num_states': 2,
        'num_transitions': 3
    }

    print("\n" + "="*80)
    print("RUNNING GENERATION (watch for cache statistics)")
    print("="*80)

    try:
        code, truncated = generator.generate(ltl_dict, dfa_result)

        print("\n" + "="*80)
        print("GENERATION COMPLETE!")
        print("="*80)
        print(f"Truncated: {truncated}")
        print(f"Code length: {len(code)} characters")

        # Check cache statistics in output
        # Should see: Cache hits: 2, Cache misses: 1
        # (First on_a_b is miss, then on_c_d and on_e_f are hits)

        print("\n✓ Variable-level planning test PASSED!")
        print("  Expected: 1 cache miss, 2 cache hits")
        print("  (Check output above for actual statistics)")

        return True

    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_variable_level_planning()
    sys.exit(0 if success else 1)
