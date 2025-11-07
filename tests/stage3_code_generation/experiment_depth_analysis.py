"""
Experiment: Analyze the effect of max_depth on state space exploration

This experiment tests how different max_depth values affect:
1. Number of states explored
2. Number of transitions generated
3. Number of goal plans generated
4. Code generation time
"""

import sys
from pathlib import Path
from typing import List
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.pddl_parser import PDDLParser
from src.stage3_code_generation.state_space import PredicateAtom
from src.stage3_code_generation.forward_planner import ForwardStatePlanner


def run_depth_experiment(goal_predicates: List[str], depths: List[int], objects: List[str] = None):
    """Run exploration with different max_depth values"""

    # Load PDDL domain
    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Default objects if not provided
    if objects is None:
        objects = ["a", "b"]

    print("=" * 80)
    print("DEPTH ANALYSIS EXPERIMENT")
    print("=" * 80)
    print(f"\nGoal: {', '.join(goal_predicates)}")
    print(f"Objects: {', '.join(objects)}")
    print(f"Domain: {domain.name}")
    print(f"  Actions: {len(domain.actions)}")
    print(f"  Predicates: {len(domain.predicates)}")
    print()

    # Convert goal predicates to PredicateAtom objects
    goal_atoms = []
    for pred_str in goal_predicates:
        # Parse "on(a, b)" into name and args
        name = pred_str[:pred_str.index('(')]
        args_str = pred_str[pred_str.index('(')+1:pred_str.index(')')]
        args = [arg.strip() for arg in args_str.split(',')]
        goal_atoms.append(PredicateAtom(name, tuple(args)))

    results = []

    for depth in depths:
        print(f"\n{'=' * 80}")
        print(f"Testing depth = {depth}")
        print(f"{'=' * 80}")

        planner = ForwardStatePlanner(domain, objects)

        start_time = time.time()
        state_graph = planner.explore_from_goal(goal_atoms, max_depth=depth)
        elapsed_time = time.time() - start_time

        # Count non-trivial paths (states that can reach goal with distance > 0)
        paths = state_graph.find_shortest_paths_to_goal()
        non_trivial_paths = sum(
            1 for state, path in paths.items()
            if len(path) > 0  # States that need at least one step to reach goal
        )

        result = {
            'depth': depth,
            'states': len(state_graph.states),
            'transitions': len(state_graph.transitions),
            'leaf_states': len(state_graph.get_leaf_states()),
            'non_trivial_paths': non_trivial_paths,
            'time': elapsed_time
        }
        results.append(result)

        print(f"\nResults:")
        print(f"  States explored:    {result['states']:>6}")
        print(f"  Transitions:        {result['transitions']:>6}")
        print(f"  Leaf states:        {result['leaf_states']:>6}")
        print(f"  Non-trivial paths:  {result['non_trivial_paths']:>6} (potential goal plans)")
        print(f"  Time:               {result['time']:>6.3f}s")

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(f"{'Depth':>6} | {'States':>8} | {'Trans':>8} | {'Paths':>8} | {'Time':>8}")
    print(f"{'-' * 6}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}")

    for r in results:
        print(f"{r['depth']:>6} | {r['states']:>8} | {r['transitions']:>8} | {r['non_trivial_paths']:>8} | {r['time']:>6.2f}s")

    # Calculate growth rates
    print(f"\n{'=' * 80}")
    print("GROWTH ANALYSIS")
    print(f"{'=' * 80}")

    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]

        state_growth = (curr['states'] - prev['states']) / prev['states'] * 100
        trans_growth = (curr['transitions'] - prev['transitions']) / prev['transitions'] * 100
        path_growth = (curr['non_trivial_paths'] - prev['non_trivial_paths']) / prev['non_trivial_paths'] * 100 if prev['non_trivial_paths'] > 0 else 0

        print(f"\nDepth {prev['depth']} → {curr['depth']}:")
        print(f"  States:  +{curr['states'] - prev['states']:>4} ({state_growth:>+6.1f}%)")
        print(f"  Trans:   +{curr['transitions'] - prev['transitions']:>4} ({trans_growth:>+6.1f}%)")
        print(f"  Paths:   +{curr['non_trivial_paths'] - prev['non_trivial_paths']:>4} ({path_growth:>+6.1f}%)")

    # Analysis
    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print(f"{'=' * 80}")

    # Check if growth is leveling off
    if len(results) >= 3:
        last_growth = results[-1]['states'] - results[-2]['states']
        second_last_growth = results[-2]['states'] - results[-3]['states']

        print(f"\nState growth trend:")
        print(f"  Depth {results[-3]['depth']}→{results[-2]['depth']}: +{second_last_growth}")
        print(f"  Depth {results[-2]['depth']}→{results[-1]['depth']}: +{last_growth}")

        if last_growth < second_last_growth * 0.5:
            print(f"  ⚠️  Growth is slowing significantly")
        elif last_growth == 0:
            print(f"  ✅ State space is fully explored (no new states)")
        else:
            print(f"  ⏳ State space is still expanding")

    # Check completeness at different depths
    print(f"\nCompleteness assessment:")
    if results[-1]['leaf_states'] == 0:
        print(f"  ✅ No leaf states at depth {results[-1]['depth']} - exploration is 'complete'")
        print(f"     (All states can transition further)")
    else:
        print(f"  ⚠️  {results[-1]['leaf_states']} leaf states at depth {results[-1]['depth']}")
        print(f"     (Some states cannot transition further)")

    print()
    return results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Simple goal on(a, b)")
    print("=" * 80)

    # Test with the simple goal from our integration tests
    goal = ["on(a, b)"]
    depths = [1, 2, 3, 4, 5]

    results1 = run_depth_experiment(goal, depths)

    print("\n\n" + "=" * 80)
    print("EXPERIMENT 2: Complex goal on(a, b) with more objects")
    print("=" * 80)

    # This would require modifying the domain to have more objects
    # For now, let's test with a different goal
    goal2 = ["clear(a)"]
    results2 = run_depth_experiment(goal2, depths)

    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("\nFor goal on(a, b):")
    print(f"  Depth 2 finds: {results1[1]['non_trivial_paths']} goal plans")
    print(f"  Depth 5 finds: {results1[4]['non_trivial_paths']} goal plans")
    print(f"  Difference: {results1[4]['non_trivial_paths'] - results1[1]['non_trivial_paths']} additional plans")

    print("\nFor goal clear(a):")
    print(f"  Depth 2 finds: {results2[1]['non_trivial_paths']} goal plans")
    print(f"  Depth 5 finds: {results2[4]['non_trivial_paths']} goal plans")
    print(f"  Difference: {results2[4]['non_trivial_paths'] - results2[1]['non_trivial_paths']} additional plans")

    print("\n✅ Experiment complete!")
