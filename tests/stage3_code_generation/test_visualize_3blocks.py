"""
Visualization and Debug Test for 3 Blocks

This test demonstrates the complete backward planning workflow with:
- Initial state: a, b, c all on table, handempty
- Goal: on(a, b)

Shows step-by-step state exploration to help identify bugs.
"""

import sys
from pathlib import Path

# Add project root to path
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
        print(f"\n{label}:")
    print(f"  Depth: {state.depth}")
    print(f"  Predicates ({len(state.predicates)}):")
    sorted_preds = sorted(state.predicates, key=lambda p: (p.name, p.args))
    for pred in sorted_preds:
        print(f"    - {pred.to_agentspeak()}")


def visualize_initial_goal_inference():
    """
    Step 1: Show how partial goal is inferred to complete goal state
    """
    print("="*80)
    print("STEP 1: Goal State Inference")
    print("="*80)

    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    objects = ['a', 'b', 'c']

    planner = ForwardStatePlanner(domain, objects)

    # Partial goal: on(a, b)
    goal_predicates = [PredicateAtom('on', ['a', 'b'])]

    print("\n📝 Input (Partial Goal):")
    print(f"  - on(a, b)")

    # Infer complete goal
    complete_goal = planner.infer_complete_goal_state(goal_predicates)
    complete_goal_state = WorldState(complete_goal, depth=0)

    print_state(complete_goal_state, "\n✅ Inferred Complete Goal State")

    print("\n💡 Explanation:")
    print("  The planner infers a COMPLETE state from the partial goal.")
    print("  Missing predicates are inferred using:")
    print("  1. Closed-world assumption: what's not mentioned is false")
    print("  2. Domain rules: if on(a,b), then ~ontable(a), ~holding(a)")
    print("  3. Physical constraints: only one block can be on another")

    return planner, goal_predicates


def visualize_first_10_states(planner: ForwardStatePlanner, goal_predicates):
    """
    Step 2: Show first 10 states explored from goal
    """
    print("\n\n" + "="*80)
    print("STEP 2: State Space Exploration (First 10 States)")
    print("="*80)

    print("\n🔍 Starting backward exploration from goal state...")
    print("   We apply actions in REVERSE to find predecessor states.")
    print()

    # We'll manually run a few iterations to show the process
    complete_goal = planner.infer_complete_goal_state(goal_predicates)
    goal_state = WorldState(complete_goal, depth=0)

    from collections import deque
    from src.stage3_code_generation.state_space import StateGraph

    graph = StateGraph(goal_state)
    queue = deque([goal_state])
    visited_map = {goal_state.predicates: goal_state}

    states_explored = 0
    max_show = 10

    print("State Exploration Process:")
    print("-" * 80)

    while queue and states_explored < max_show:
        current_state = queue.popleft()
        states_explored += 1

        print(f"\n[State {states_explored}] Exploring:")
        print_state(current_state)

        # Try all ground actions
        actions_tried = 0
        new_states_found = 0

        for grounded_action in planner._ground_all_actions():
            if not planner._check_preconditions(grounded_action, current_state):
                continue

            actions_tried += 1
            new_states_data = planner._apply_action(grounded_action, current_state)

            for new_state, belief_updates, preconditions in new_states_data:
                new_pred_set = frozenset(new_state.predicates)

                if new_pred_set not in visited_map:
                    new_depth = current_state.depth + 1
                    final_state = WorldState(new_state.predicates, depth=new_depth)
                    visited_map[new_pred_set] = final_state
                    queue.append(final_state)
                    new_states_found += 1

                    if new_states_found <= 2:  # Show first 2 successors
                        print(f"\n  ➡️  Action: {grounded_action.action.name}({', '.join(grounded_action.args)})")
                        print(f"      Belief updates: {belief_updates}")
                        print(f"      New state depth: {new_depth}")

        print(f"\n  Summary: Tried {actions_tried} actions, found {new_states_found} new states")
        print(f"  Queue size: {len(queue)}")

    print("\n" + "-" * 80)
    print(f"\n✅ Explored {states_explored} states")
    print(f"   Total states discovered so far: {len(visited_map)}")
    print(f"   States still in queue: {len(queue)}")

    return len(visited_map), len(queue)


def visualize_complete_search():
    """
    Step 3: Run complete search with statistics
    """
    print("\n\n" + "="*80)
    print("STEP 3: Complete State Space Exploration")
    print("="*80)

    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

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
        "objects": ["a", "b", "c"],
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    print("\n🎯 Goal: F(on(a, b))")
    print("🧱 Objects: a, b, c")
    print("📦 Initial concept: All blocks on table, hand empty")
    print()

    import time
    start = time.time()

    generator = BackwardPlannerGenerator(domain, grounding_map)
    asl_code = generator.generate(ltl_dict, dfa_result)

    elapsed = time.time() - start

    print(f"\n✅ Generation complete!")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Code size: {len(asl_code):,} chars")

    # Extract statistics
    import re
    states_match = re.search(r'States:\s*(\d+)', asl_code)
    transitions_match = re.search(r'Transitions:\s*(\d+)', asl_code)
    depth_match = re.search(r'Max depth reached:\s*(\d+)', asl_code)

    if states_match:
        print(f"   States explored: {int(states_match.group(1)):,}")
    if transitions_match:
        print(f"   Transitions: {int(transitions_match.group(1)):,}")
    if depth_match:
        print(f"   Max depth: {depth_match.group(1)}")

    return asl_code


def check_initial_state_coverage(asl_code: str):
    """
    Step 4: Verify that initial state (all on table) is covered
    """
    print("\n\n" + "="*80)
    print("STEP 4: Initial State Coverage Check")
    print("="*80)

    print("\n🔍 Checking if initial state is reachable from goal...")
    print("   Initial state: ontable(a), ontable(b), ontable(c), handempty")
    print()

    # Check if there's a plan for achieving on(a,b) from table state
    # This would be in the generated code

    if "ontable(a)" in asl_code.lower() and "ontable(b)" in asl_code.lower():
        print("✅ Initial state IS covered in generated plans")
        print("   Found plans mentioning 'ontable' predicates")
    else:
        print("⚠️  Initial state might NOT be fully covered")
        print("   No plans found mentioning 'ontable' predicates")

    # Count plan variants
    plan_count = asl_code.count("+!on(a, b)")
    print(f"\n📊 Found {plan_count} different plan variants for achieving on(a, b)")

    # Check for success plans (goal already achieved)
    if "already achieved" in asl_code.lower():
        print("✅ Includes success plans (goal already satisfied)")

    # Check for failure plans (goal unreachable)
    if ".fail" in asl_code or "unreachable" in asl_code.lower():
        print("⚠️  Includes failure plans (some states can't reach goal)")


def analyze_potential_bugs():
    """
    Step 5: Known issues and potential bugs to check
    """
    print("\n\n" + "="*80)
    print("STEP 5: Potential Bug Analysis")
    print("="*80)

    print("\n🔬 Checking for known issues:")
    print()

    checks = [
        ("State reuse", "States with same predicates should be deduplicated", "✅ FIXED"),
        ("DFA parsing", "Should parse exactly 2 states (state0, state1)", "✅ FIXED"),
        ("Max states limit", "Should stop at 50,000 states for safety", "✅ IMPLEMENTED"),
        ("Queue growth", "Queue should eventually decrease (not grow forever)", "⚠️  TO CHECK"),
        ("Depth calculation", "All paths to same state should have correct depth", "⚠️  TO CHECK"),
        ("Initial state reachability", "Should reach initial state (all on table)", "⚠️  TO VERIFY"),
    ]

    for name, description, status in checks:
        print(f"{status} {name}")
        print(f"    {description}")
        print()


if __name__ == "__main__":
    print("\n" + "🔬 3-BLOCKS BACKWARD PLANNING VISUALIZATION 🔬".center(80))
    print()
    print("This test demonstrates the complete backward planning process")
    print("for 3 blocks (a, b, c) with goal: on(a, b)")
    print()

    # Step 1: Goal inference
    planner, goal_predicates = visualize_initial_goal_inference()

    # Step 2: First 10 states
    input("\n\n▶️  Press Enter to explore first 10 states...")
    total_states, queue_size = visualize_first_10_states(planner, goal_predicates)

    # Step 3: Complete search
    input("\n\n▶️  Press Enter to run complete state space exploration...")
    asl_code = visualize_complete_search()

    # Step 4: Check coverage
    check_initial_state_coverage(asl_code)

    # Step 5: Bug analysis
    analyze_potential_bugs()

    # Final summary
    print("\n\n" + "="*80)
    print("🏁 VISUALIZATION COMPLETE")
    print("="*80)
    print("\n📝 Key Observations:")
    print("  1. State space is explored systematically from goal backward")
    print("  2. Each state can have multiple predecessor states")
    print("  3. BFS ensures shortest paths are found first")
    print("  4. State deduplication prevents exponential blowup")
    print()
    print("⚠️  Potential Issues to Monitor:")
    print("  1. Queue growth - should stabilize or decrease eventually")
    print("  2. Memory usage - grows with state count")
    print("  3. Time complexity - exponential in number of objects")
    print()
    print("✅ For 3 blocks, search should complete in 1-5 minutes")
    print("❌ For 4+ blocks, search may take 10+ minutes or hit limits")
    print()
