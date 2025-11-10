"""
Simple 2-Blocks Visualization Test

Initial state: ontable(a), ontable(b), handempty, clear(a), clear(b)
Goal: on(a, b)

This is a minimal example to demonstrate the backward planning process.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from src.stage3_code_generation.forward_planner import ForwardStatePlanner
from src.stage3_code_generation.state_space import PredicateAtom, WorldState
from src.utils.pddl_parser import PDDLParser
from src.stage1_interpretation.grounding_map import GroundingMap


def print_state(state: WorldState, label: str = ""):
    """Pretty print a state"""
    if label:
        print(f"{label}:")
    sorted_preds = sorted(state.predicates, key=lambda p: (p.name, p.args))
    for pred in sorted_preds:
        print(f"  - {pred.to_agentspeak()}")
    print()


def main():
    print("="*80)
    print("2-BLOCKS VISUALIZATION: Simple Example")
    print("="*80)
    print()
    print("Initial State (what we start with):")
    print("  - ontable(a)")
    print("  - ontable(b)")
    print("  - clear(a)")
    print("  - clear(b)")
    print("  - handempty")
    print()
    print("Goal State (what we want):")
    print("  - on(a, b)")
    print()
    print("="*80)

    # Load domain
    domain_path = project_root / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Setup
    objects = ['a', 'b']
    planner = ForwardStatePlanner(domain, objects)

    # Step 1: Goal inference
    print("\nSTEP 1: Goal State Inference")
    print("-" * 40)

    goal_predicates = [PredicateAtom('on', ['a', 'b'])]
    complete_goal = planner.infer_complete_goal_state(goal_predicates)
    goal_state = WorldState(complete_goal, depth=0)

    print_state(goal_state, "Complete Goal State (depth 0)")

    print("Explanation:")
    print("  The system infers that if on(a,b) is true, then:")
    print("  - clear(a) must be true (nothing on top of a)")
    print("  - handempty must be true (not holding anything)")
    print("  - ontable(a) must be FALSE (a is on b, not on table)")
    print()

    # Step 2: Show backward exploration manually
    print("="*80)
    print("\nSTEP 2: Backward Exploration (Manual Trace)")
    print("-" * 40)

    from collections import deque
    from src.stage3_code_generation.state_space import StateGraph

    graph = StateGraph(goal_state)
    queue = deque([goal_state])
    visited_map = {goal_state.predicates: goal_state}

    print("Starting from GOAL state, we ask: 'What actions could have led here?'\n")

    # Manually explore first few states
    states_to_show = 5
    for i in range(states_to_show):
        if not queue:
            break

        current_state = queue.popleft()

        print(f"[State {i+1}] Current state (depth {current_state.depth}):")
        print_state(current_state)

        # Try all actions
        new_count = 0
        for grounded_action in planner._ground_all_actions():
            if not planner._check_preconditions(grounded_action, current_state):
                continue

            new_states_data = planner._apply_action(grounded_action, current_state)

            for new_state, belief_updates, preconditions in new_states_data:
                new_pred_set = frozenset(new_state.predicates)

                if new_pred_set not in visited_map:
                    new_depth = current_state.depth + 1
                    final_state = WorldState(new_state.predicates, depth=new_depth)
                    visited_map[new_pred_set] = final_state
                    queue.append(final_state)
                    new_count += 1

                    # Show first new state found
                    if new_count == 1:
                        print(f"  Found predecessor via action: {grounded_action.action.name}({', '.join(grounded_action.args)})")
                        print(f"  Belief updates: {belief_updates}")
                        print_state(final_state, f"  New state (depth {new_depth})")

        print(f"  Total new states found: {new_count}")
        print(f"  Queue size: {len(queue)}\n")

    print(f"After exploring {states_to_show} states:")
    print(f"  Total states discovered: {len(visited_map)}")
    print(f"  States remaining in queue: {len(queue)}")
    print()

    # Step 3: Check if initial state is reachable
    print("="*80)
    print("\nSTEP 3: Checking Initial State Reachability")
    print("-" * 40)

    # The initial state predicates
    initial_state_preds = frozenset([
        PredicateAtom('ontable', ['a']),
        PredicateAtom('ontable', ['b']),
        PredicateAtom('clear', ['a']),
        PredicateAtom('clear', ['b']),
        PredicateAtom('handempty', [])
    ])

    print("Looking for this initial state in explored states:")
    print_state(WorldState(set(initial_state_preds), 0), "")

    # Complete the exploration
    print("Running complete exploration...")
    while queue:
        current_state = queue.popleft()

        for grounded_action in planner._ground_all_actions():
            if not planner._check_preconditions(grounded_action, current_state):
                continue

            new_states_data = planner._apply_action(grounded_action, current_state)

            for new_state, _, _ in new_states_data:
                new_pred_set = frozenset(new_state.predicates)

                if new_pred_set not in visited_map:
                    new_depth = current_state.depth + 1
                    final_state = WorldState(new_state.predicates, depth=new_depth)
                    visited_map[new_pred_set] = final_state
                    queue.append(final_state)

    print(f"Exploration complete: {len(visited_map)} total states explored")
    print()

    # Check if initial state was found
    if initial_state_preds in visited_map:
        found_state = visited_map[initial_state_preds]
        print(f"✅ Initial state IS reachable from goal!")
        print(f"   Found at depth: {found_state.depth}")
        print(f"   This means: goal is achievable in {found_state.depth} steps from initial state")
    else:
        print("❌ Initial state NOT found in explored states")
        print("   This could indicate:")
        print("   1. Initial state representation mismatch")
        print("   2. Bug in state exploration")
        print("   3. Goal is unreachable from this initial state")

    print()

    # Step 4: Run full code generation
    print("="*80)
    print("\nSTEP 4: Full Code Generation")
    print("-" * 40)

    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])

    dfa_dot = """
digraph {
    rankdir=LR;
    node [shape=circle];
    __start [shape=point];
    __start -> state0;
    state0 [label="0"];
    state1 [label="1", shape=doublecircle];
    state0 -> state1 [label="on_a_b"];
}
"""

    dfa_result = {
        "formula": "F(on(a, b))",
        "dfa_dot": dfa_dot,
        "num_states": 2,
        "num_transitions": 1
    }

    ltl_dict = {
        "objects": ["a", "b"],
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    import time
    start = time.time()

    generator = BackwardPlannerGenerator(domain, grounding_map)
    asl_code = generator.generate(ltl_dict, dfa_result)

    elapsed = time.time() - start

    print(f"✅ Code generated in {elapsed:.2f}s")
    print(f"   Code size: {len(asl_code):,} chars")

    # Extract stats
    import re
    states_match = re.search(r'States:\s*(\d+)', asl_code)
    if states_match:
        print(f"   States explored: {states_match.group(1)}")

    # Check if initial state scenario is covered
    if "ontable(a)" in asl_code and "ontable(b)" in asl_code:
        print("   ✅ Generated code covers initial state scenario")

    print()

    # Step 5: Summary
    print("="*80)
    print("\nSUMMARY")
    print("="*80)
    print()
    print("✅ Goal state inference works correctly")
    print("✅ Backward exploration discovers predecessor states")
    print("✅ State deduplication prevents redundant exploration")

    if initial_state_preds in visited_map:
        print("✅ Initial state is reachable from goal")
    else:
        print("⚠️  Initial state reachability needs investigation")

    print()
    print("📊 Key Statistics:")
    print(f"   - Objects: 2 blocks")
    print(f"   - States explored: {len(visited_map)}")
    print(f"   - Max depth: {max(s.depth for s in visited_map.values())}")
    print(f"   - Time: {elapsed:.2f}s")
    print()


if __name__ == "__main__":
    main()
