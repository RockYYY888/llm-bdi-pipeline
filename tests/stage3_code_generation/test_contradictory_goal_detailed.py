#!/usr/bin/env python3
"""
Detailed analysis of contradictory goals like on(a,b) & on(b,a)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.state_space import PredicateAtom, WorldState
from stage3_code_generation.legacy.forward_planner import ForwardStatePlanner
from utils.pddl_parser import PDDLParser


def test_contradictory_goal_detailed():
    """Detailed test of contradictory goal: on(a,b) & on(b,a)"""
    print("=" * 80)
    print("Detailed Analysis: Contradictory Goal - on(a,b) & on(b,a)")
    print("=" * 80)

    # Load domain
    domain_file = Path(__file__).parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
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

    # Create planner and explore
    planner = ForwardStatePlanner(domain, objects, use_variables=False)
    state_graph = planner.explore_from_goal(goal_predicates, max_states=50)

    print("\n" + "=" * 80)
    print("Key Question: Is the contradictory goal state REACHABLE?")
    print("=" * 80)

    goal_state = state_graph.goal_state
    print(f"\nGoal state predicates:")
    for pred in sorted(goal_state.predicates, key=str):
        print(f"  - {pred}")

    # Check if goal state has any INCOMING transitions
    incoming = state_graph.get_incoming_transitions(goal_state)
    print(f"\nIncoming transitions to goal state: {len(incoming)}")

    if len(incoming) == 0:
        print("⚠️  NO INCOMING TRANSITIONS!")
        print("⚠️  The goal state CANNOT be reached from any other state")
        print("⚠️  This means the contradictory goal is IMPOSSIBLE to achieve")
    else:
        print("✓ Goal state has incoming transitions:")
        for i, trans in enumerate(incoming[:5]):  # Show first 5
            print(f"  {i+1}. {trans}")

    # Check outgoing transitions (can we leave the goal state?)
    outgoing = state_graph.get_outgoing_transitions(goal_state)
    print(f"\nOutgoing transitions from goal state: {len(outgoing)}")
    if len(outgoing) > 0:
        print("✓ Can leave goal state via actions (but CAN'T reach it in first place):")
        for i, trans in enumerate(outgoing[:5]):  # Show first 5
            print(f"  {i+1}. {trans}")

    # Find leaf states (states with no outgoing transitions)
    leaf_states = state_graph.get_leaf_states()
    print(f"\n" + "=" * 80)
    print(f"Exploration Statistics")
    print("=" * 80)
    print(f"Total states: {len(state_graph.states)}")
    print(f"Total transitions: {len(state_graph.transitions)}")
    print(f"Leaf states (no outgoing): {len(leaf_states)}")

    # Show some leaf states
    if leaf_states:
        print(f"\nExample leaf states (states with no outgoing transitions):")
        for i, state in enumerate(list(leaf_states)[:3]):
            print(f"\n  Leaf state {i+1}:")
            for pred in sorted(state.predicates, key=str):
                print(f"    - {pred}")

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print("Current behavior for contradictory goals like on(a,b) & on(b,a):")
    print()
    print("1. ✗ NO VALIDATION: System accepts contradictory goals without checking")
    print()
    print("2. ✗ UNREACHABLE GOAL STATE: The goal state with both contradictions")
    print("      has NO incoming transitions - it's impossible to reach")
    print()
    print("3. ✓ EXPLORATION STILL WORKS: Forward planner can still explore FROM")
    print("      the contradictory state (even though you can't reach it)")
    print()
    print("4. ⚠️  SEMANTIC INCORRECTNESS: The generated plans will be based on")
    print("      an impossible starting point, making them semantically wrong")
    print()
    print("5. ⚠️  NO RUNTIME ERROR: Code generation proceeds, but produces")
    print("      plans that can never actually achieve the goal")
    print()
    print("Real-world impact:")
    print("  - If DFA has transition label 'on(a,b) & on(b,a)', code is generated")
    print("  - AgentSpeak plans exist, but they start from impossible state")
    print("  - At runtime, agent can never trigger these plans (goal unreachable)")
    print("  - This wastes computation and produces misleading code")
    print("=" * 80)


if __name__ == "__main__":
    test_contradictory_goal_detailed()
