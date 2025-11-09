"""
Debug: Why does depth 2 produce 1093 states?

This test investigates the state space explosion to understand
why a simple 2-block problem generates so many states.
"""

import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.forward_planner import ForwardStatePlanner
from src.stage3_code_generation.state_space import PredicateAtom, WorldState
from src.utils.pddl_parser import PDDLParser


def analyze_state_space():
    print("="*80)
    print("DEBUG: State Space Explosion Analysis")
    print("="*80)
    print()

    # Load domain
    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # 2 blocks
    objects = ['a', 'b']
    planner = ForwardStatePlanner(domain, objects)

    # Goal: on(a, b)
    goal_predicates = [PredicateAtom('on', ['a', 'b'])]
    complete_goal = planner.infer_complete_goal_state(goal_predicates)
    goal_state = WorldState(complete_goal, depth=0)

    print("Goal state:")
    for pred in sorted(goal_state.predicates, key=lambda p: (p.name, p.args)):
        print(f"  - {pred.to_agentspeak()}")
    print()

    # Count actions
    all_actions = list(planner._ground_all_actions())
    print(f"Total ground actions: {len(all_actions)}")

    action_counts = defaultdict(int)
    for ga in all_actions:
        action_counts[ga.action.name] += 1

    print("\nGround actions by type:")
    for action_name, count in sorted(action_counts.items()):
        print(f"  {action_name}: {count}")
    print()

    # Manual BFS with detailed tracking
    from collections import deque

    queue = deque([goal_state])
    visited_map = {goal_state.predicates: goal_state}

    # Track states by depth
    states_by_depth = defaultdict(list)
    states_by_depth[0].append(goal_state)

    # Track states by predicate count
    states_by_pred_count = defaultdict(int)
    states_by_pred_count[len(goal_state.predicates)] += 1

    # Track states by predicate types
    predicate_combinations = defaultdict(int)

    print("Starting exploration...")
    print()

    max_depth_to_analyze = 2

    while queue:
        current_state = queue.popleft()

        if current_state.depth > max_depth_to_analyze:
            continue

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

                    # Track statistics
                    states_by_depth[new_depth].append(final_state)
                    states_by_pred_count[len(final_state.predicates)] += 1

                    # Track predicate type combination
                    pred_types = tuple(sorted(set(p.name for p in final_state.predicates)))
                    predicate_combinations[pred_types] += 1

    # Analysis
    print("="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    print()

    print(f"Total states explored (depth 0-{max_depth_to_analyze}): {len(visited_map)}")
    print()

    print("States by depth:")
    for depth in sorted(states_by_depth.keys()):
        count = len(states_by_depth[depth])
        print(f"  Depth {depth}: {count} states")
    print()

    print("States by number of predicates:")
    for pred_count in sorted(states_by_pred_count.keys()):
        count = states_by_pred_count[pred_count]
        print(f"  {pred_count} predicates: {count} states")
    print()

    print("Top 10 predicate type combinations:")
    sorted_combos = sorted(predicate_combinations.items(), key=lambda x: x[1], reverse=True)
    for combo, count in sorted_combos[:10]:
        print(f"  {combo}: {count} states")
    print()

    # Sample states at different depths
    print("="*80)
    print("SAMPLE STATES AT EACH DEPTH")
    print("="*80)
    print()

    for depth in [0, 1, 2]:
        if depth not in states_by_depth:
            continue

        print(f"\n--- Depth {depth} (showing 5 random states) ---")

        import random
        samples = random.sample(states_by_depth[depth], min(5, len(states_by_depth[depth])))

        for i, state in enumerate(samples, 1):
            print(f"\nSample {i}:")
            for pred in sorted(state.predicates, key=lambda p: (p.name, p.args)):
                print(f"  {pred.to_agentspeak()}")

    # Key insight
    print("\n\n" + "="*80)
    print("KEY INSIGHT: Why so many states?")
    print("="*80)
    print()

    print("You expected: ~7^2 = 49 states (thinking of action sequences)")
    print(f"Actual result: {len(visited_map)} states")
    print()
    print("The difference is because:")
    print()
    print("1. We're counting STATES, not action sequences")
    print("   - Same action sequence can reach different states depending on context")
    print()
    print("2. States are predicate combinations, not just 'what happened'")
    print("   - Example states at depth 1:")
    print("     * holding(a), on(a, b)  [different from...]")
    print("     * holding(a), clear(b)  [different from...]")
    print("     * clear(a), clear(b), handempty")
    print()
    print("3. Blocksworld allows many 'irrelevant' predicate combinations")
    print("   - on(a, a), on(b, b), etc. (self-loops, though invalid)")
    print("   - holding(a), holding(b) (impossible but explored)")
    print()
    print("4. Each block can be in multiple configurations:")
    print("   For 2 blocks {a, b}, predicates include:")
    print("   - on(a, b), on(b, a), on(a, a), on(b, b)")
    print("   - ontable(a), ontable(b)")
    print("   - holding(a), holding(b)")
    print("   - clear(a), clear(b)")
    print("   - handempty")
    print()
    print("   Different subsets of these = different states!")
    print()

    # Count unique action sequences
    print("="*80)
    print("ACTION SEQUENCES vs STATES")
    print("="*80)
    print()

    # Trace back paths from initial state
    initial_preds = frozenset([
        PredicateAtom('ontable', ['a']),
        PredicateAtom('ontable', ['b']),
        PredicateAtom('clear', ['a']),
        PredicateAtom('clear', ['b']),
        PredicateAtom('handempty', [])
    ])

    if initial_preds in visited_map:
        initial_state = visited_map[initial_preds]
        print(f"Initial state found at depth: {initial_state.depth}")
        print()
        print("This means:")
        print(f"  - Shortest path from initial to goal: {initial_state.depth} steps")
        print(f"  - But total states explored: {len(visited_map)}")
        print(f"  - Ratio: {len(visited_map) / max(1, initial_state.depth):.0f} states per depth level")
        print()
        print("Why the difference?")
        print("  - Each step can branch into MULTIPLE states (not just one next state)")
        print("  - Different actions produce different resulting states")
        print("  - States with same depth but different predicates are counted separately")
    else:
        print("Initial state not found in exploration (depth limit)")

    print()


if __name__ == "__main__":
    analyze_state_space()
