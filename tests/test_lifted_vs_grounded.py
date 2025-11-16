"""
Test to demonstrate TRUE lifted planning vs PSEUDO-lifted (grounded with variables)

This test shows the fundamental difference:
- Grounded with variables: Enumerates all variable combinations (state explosion)
- True lifted: Explores abstract state space (minimal states)
"""

import sys
from pathlib import Path

# Add parent directory to path
_parent = str(Path(__file__).parent.parent / "src")
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from stage3_code_generation.forward_planner import ForwardStatePlanner
from stage3_code_generation.lifted_planner import LiftedPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser


def test_comparison():
    """
    Compare lifted vs grounded planning on same goal

    Goal: on(?X, ?Y)  (or on(a, b) for grounded)
    Domain: blocksworld
    Objects: 3 objects [a, b, c] or [?v0, ?v1, ?v2]
    """
    print("="*80)
    print("COMPARISON: True Lifted Planning vs Grounded-with-Variables")
    print("="*80)

    # Load domain
    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))
    print(f"Domain: {domain.name}")
    print(f"Actions: {[a.name for a in domain.actions]}")

    # Define goal
    print("\n" + "="*80)
    print("TEST 1: Grounded Planning with Variables (PSEUDO-lifted)")
    print("="*80)
    print("Using: [?v0, ?v1, ?v2] as 'objects'")
    print("This will still ENUMERATE all combinations in itertools.product!")

    # Grounded with variables
    variables = ["?v0", "?v1", "?v2"]
    goal_preds_grounded = [PredicateAtom("on", ["?v0", "?v1"])]

    planner_grounded = ForwardStatePlanner(domain, variables, use_variables=True)
    print(f"\nGoal: {[str(p) for p in goal_preds_grounded]}")
    print("Starting exploration...")

    try:
        graph_grounded = planner_grounded.explore_from_goal(
            goal_preds_grounded,
            max_states=1000  # Limit to prevent explosion
        )
        print(f"\nRESULT (Grounded with Variables):")
        print(f"  States: {len(graph_grounded.states):,}")
        print(f"  Transitions: {len(graph_grounded.transitions):,}")
        grounded_state_count = len(graph_grounded.states)
    except Exception as e:
        print(f"\nERROR: {e}")
        grounded_state_count = "ERROR"

    # True lifted planning
    print("\n" + "="*80)
    print("TEST 2: True Lifted Planning")
    print("="*80)
    print("Using: Abstract variables through unification")
    print("This will NOT enumerate - uses unification to match!")

    goal_preds_lifted = [PredicateAtom("on", ["?X", "?Y"])]

    planner_lifted = LiftedPlanner(domain)
    print(f"\nGoal: {[str(p) for p in goal_preds_lifted]}")
    print("Starting exploration...")

    try:
        result_lifted = planner_lifted.explore_from_goal(
            goal_preds_lifted,
            max_states=1000
        )
        print(f"\nRESULT (True Lifted):")
        print(f"  Abstract states: {len(result_lifted['states']):,}")
        print(f"  Transitions: {len(result_lifted['transitions']):,}")
        lifted_state_count = len(result_lifted['states'])
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        lifted_state_count = "ERROR"

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Grounded with Variables (PSEUDO):  {grounded_state_count} states")
    print(f"True Lifted Planning:              {lifted_state_count} states")

    if isinstance(grounded_state_count, int) and isinstance(lifted_state_count, int):
        ratio = grounded_state_count / max(lifted_state_count, 1)
        print(f"Ratio: {ratio:.1f}x more states with pseudo-lifted!")
        print()
        print("This demonstrates the KEY INSIGHT:")
        print("  - Pseudo-lifted (grounded with vars) still enumerates all combinations")
        print("  - True lifted planning explores ABSTRACT state space")
        print("  - State count is INDEPENDENT of number of domain objects!")

    print("="*80)


if __name__ == "__main__":
    test_comparison()
