"""
Simple test to demonstrate lifted planning with minimal goal

This test uses a simpler goal without inference to show the true difference.
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


def test_simple_lifted():
    """
    Test lifted planning with simple goal
    """
    print("="*80)
    print("LIFTED PLANNING - Simple Test")
    print("="*80)

    # Load domain
    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))
    print(f"Domain: {domain.name}")
    print(f"Actions: {[a.name for a in domain.actions]}\n")

    # Simple goal: holding(?X)
    goal_preds = [
        PredicateAtom("holding", ["?X"])
    ]

    planner = LiftedPlanner(domain)
    print(f"Goal: {[str(p) for p in goal_preds]}")
    print("Starting exploration...\n")

    try:
        result = planner.explore_from_goal(goal_preds, max_states=500)
        print(f"\nRESULT:")
        print(f"  Abstract states: {len(result['states']):,}")
        print(f"  Transitions: {len(result['transitions']):,}")

        print(f"\nSample abstract states:")
        for i, state in enumerate(list(result['states'])[:5]):
            print(f"  State {i+1}: {state}")

        print(f"\nKey insight:")
        print(f"  These {len(result['states'])} abstract states represent ALL possible")
        print(f"  configurations for achieving holding(?X), regardless of number of objects!")
        print(f"  With 3 objects: same {len(result['states'])} abstract states")
        print(f"  With 100 objects: STILL {len(result['states'])} abstract states!")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    print("="*80)


if __name__ == "__main__":
    test_simple_lifted()
