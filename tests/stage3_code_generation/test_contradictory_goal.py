#!/usr/bin/env python3
"""
Test how contradictory goals like on(a,b) & on(b,a) are handled
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.state_space import PredicateAtom
from stage3_code_generation.legacy.forward_planner import ForwardStatePlanner
from utils.pddl_parser import PDDLParser


def test_contradictory_goal():
    """Test contradictory goal: on(a,b) & on(b,a)"""
    print("=" * 80)
    print("Test: Contradictory Goal - on(a,b) & on(b,a)")
    print("=" * 80)

    # Load domain
    domain_file = Path(__file__).parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))
    print(f"✓ Loaded domain: {domain.name}\n")

    # Create contradictory goal
    objects = ["a", "b"]
    goal_predicates = [
        PredicateAtom("on", ["a", "b"]),
        PredicateAtom("on", ["b", "a"])
    ]

    print("Goal predicates:")
    for pred in goal_predicates:
        print(f"  - {pred}")
    print()

    # Create planner
    planner = ForwardStatePlanner(domain, objects, use_variables=False)

    # Step 1: Infer complete goal state
    print("Step 1: Infer complete goal state")
    print("-" * 80)
    complete_goal = planner.infer_complete_goal_state(goal_predicates)
    print(f"Complete goal state has {len(complete_goal)} predicates:")
    for pred in sorted(complete_goal, key=str):
        print(f"  - {pred}")
    print()

    # Check for contradictions
    print("Step 2: Check for logical contradictions")
    print("-" * 80)

    # Check if both on(a,b) and on(b,a) are in complete goal
    has_on_ab = PredicateAtom("on", ["a", "b"]) in complete_goal
    has_on_ba = PredicateAtom("on", ["b", "a"]) in complete_goal

    print(f"  on(a, b) in goal: {has_on_ab}")
    print(f"  on(b, a) in goal: {has_on_ba}")

    if has_on_ab and has_on_ba:
        print(f"  ⚠️  CONTRADICTION DETECTED!")
        print(f"  ⚠️  Both on(a,b) and on(b,a) are in the goal state")
        print(f"  ⚠️  This is physically impossible in blocksworld")
    print()

    # Step 3: Try to explore from this goal
    print("Step 3: Explore from contradictory goal state")
    print("-" * 80)
    try:
        state_graph = planner.explore_from_goal(goal_predicates, max_states=100)
        print(f"\nExploration completed:")
        print(f"  States: {len(state_graph.states)}")
        print(f"  Transitions: {len(state_graph.transitions)}")
        print(f"  Truncated: {state_graph.truncated}")

        # Check if goal state has any outgoing transitions
        leaf_states = state_graph.get_leaf_states()
        is_goal_leaf = state_graph.goal_state in leaf_states

        print(f"\n  Goal state is a leaf (no transitions): {is_goal_leaf}")

        if is_goal_leaf:
            print(f"  ⚠️  No actions applicable from contradictory goal state!")
            print(f"  ⚠️  This goal is UNREACHABLE from any valid state")

    except Exception as e:
        print(f"  ✗ Exploration failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Analysis:")
    print("=" * 80)
    print("Current behavior:")
    print("  1. System accepts contradictory goals without validation")
    print("  2. Creates a goal state with both contradictory predicates")
    print("  3. No actions can achieve this state (it's unreachable)")
    print("  4. Results in a leaf state with no transitions")
    print()
    print("Implications:")
    print("  - Plans generated will be empty or minimal")
    print("  - AgentSpeak code will have no useful plans for this goal")
    print("  - No runtime error, but semantically incorrect")
    print("=" * 80)


if __name__ == "__main__":
    test_contradictory_goal()
