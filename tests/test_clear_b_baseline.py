"""
Test lifted planning with clear(b) goal

This test is specifically for measuring the improvement from the weakness fixes.
According to CODE_WEAKNESSES_ANALYSIS.md:
- Before fixes: 10,854 states for clear(b)
- Expected after fixes: ~2,000-3,000 states (72-81% reduction)
"""

import sys
from pathlib import Path

# Add parent directory to path
_parent = str(Path(__file__).parent.parent / "src")
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from stage3_code_generation.lifted_planner import LiftedPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser


def test_clear_b():
    """
    Test lifted planning with clear(b) goal
    """
    print("="*80)
    print("LIFTED PLANNING - clear(b) Baseline Test")
    print("="*80)

    # Load domain
    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))
    print(f"Domain: {domain.name}")
    print(f"Actions: {[a.name for a in domain.actions]}\n")

    # Goal: clear(b)
    goal_preds = [
        PredicateAtom("clear", ["b"])
    ]

    planner = LiftedPlanner(domain)
    print(f"Goal: {[str(p) for p in goal_preds]}")
    print("Starting exploration...\n")

    try:
        result = planner.explore_from_goal(goal_preds, max_states=15000)
        print(f"\nRESULT:")
        print(f"  Abstract states: {len(result['states']):,}")
        print(f"  Transitions: {len(result['transitions']):,}")

        # Analyze depth distribution
        depth_counts = {}
        for state in result['states']:
            depth = state.depth
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        print(f"\nDepth distribution:")
        for depth in sorted(depth_counts.keys()):
            count = depth_counts[depth]
            pct = (count / len(result['states'])) * 100
            print(f"  Depth {depth}: {count:,} states ({pct:.1f}%)")

        print(f"\nSample abstract states from depth 0-2:")
        sample_states = [s for s in result['states'] if s.depth <= 2][:5]
        for i, state in enumerate(sample_states):
            print(f"  State {i+1} (depth {state.depth}): {state}")

        print(f"\nComparison with weakness analysis:")
        print(f"  Before fixes: 10,854 states")
        print(f"  Current: {len(result['states']):,} states")
        if len(result['states']) < 10854:
            reduction = ((10854 - len(result['states'])) / 10854) * 100
            print(f"  Reduction: {reduction:.1f}%")
        else:
            increase = ((len(result['states']) - 10854) / 10854) * 100
            print(f"  INCREASE: {increase:.1f}% (worse than baseline!)")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    print("="*80)


if __name__ == "__main__":
    test_clear_b()
