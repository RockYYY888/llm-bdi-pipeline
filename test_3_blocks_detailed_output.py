"""
Detailed output test for 3 blocks case - save actual states to file
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from utils.pddl_parser import PDDLParser
from stage3_code_generation.backward_search_refactored import BackwardSearchPlanner
from stage3_code_generation.state_space import PredicateAtom

def test_3_blocks_detailed():
    """Run search and save detailed output"""

    domain = PDDLParser.parse_domain('src/domains/blocksworld/domain.pddl')
    planner = BackwardSearchPlanner(domain)

    # Goal: ~on(a, b)
    goal = [PredicateAtom('on', ['a', 'b'], negated=True)]

    print(f"Running search with goal: {[str(g) for g in goal]}")
    print(f"Objects: a, b, c (3 blocks)")
    print(f"max_objects: 3")
    print(f"max_states: 500")
    print(f"Saving detailed output to test_3_blocks_output.txt...")

    # Run search
    graph = planner.search(goal, max_states=500, max_objects=3)

    # Write detailed output to file
    with open('test_3_blocks_output.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("3 BLOCKS DETAILED STATE OUTPUT\n")
        f.write("="*80 + "\n")
        f.write(f"\nGoal: {[str(g) for g in goal]}\n")
        f.write(f"Objects: a, b, c\n")
        f.write(f"max_objects: 3\n")
        f.write(f"max_states: 500\n\n")

        f.write(f"Results:\n")
        f.write(f"  States explored: {len(graph.states)}\n")
        f.write(f"  Was truncated: {graph.truncated}\n\n")

        # Group states by depth
        states_by_depth = {}
        for state in graph.states:
            depth = state.depth
            if depth not in states_by_depth:
                states_by_depth[depth] = []
            states_by_depth[depth].append(state)

        f.write(f"States by depth:\n")
        for depth in sorted(states_by_depth.keys()):
            f.write(f"  Depth {depth}: {len(states_by_depth[depth])} states\n")

        # Sample states from each depth
        f.write(f"\n" + "="*80 + "\n")
        f.write("DETAILED STATE SAMPLES (10 samples per depth, first 5 depths)\n")
        f.write("="*80 + "\n\n")

        for depth in sorted(states_by_depth.keys())[:5]:  # First 5 depths
            f.write(f"\n{'='*80}\n")
            f.write(f"DEPTH {depth} (showing 10/{len(states_by_depth[depth])} states)\n")
            f.write(f"{'='*80}\n\n")

            for i, state in enumerate(states_by_depth[depth][:10]):  # 10 samples per depth
                # Count variables
                unique_vars = set()
                grounded_args = set()
                for pred in state.predicates:
                    for arg in pred.args:
                        if arg.startswith('?v'):
                            unique_vars.add(arg)
                        else:
                            grounded_args.add(arg)

                f.write(f"--- State {i+1} (depth {depth}) ---\n")
                f.write(f"Predicates ({len(state.predicates)}):\n")
                for pred in sorted(state.predicates, key=lambda p: (p.name, str(p.args))):
                    neg_str = "~" if pred.negated else ""
                    f.write(f"  {neg_str}{pred.name}({', '.join(pred.args)})\n")

                # Get constraints if available
                if hasattr(state, 'constraints') and state.constraints:
                    f.write(f"Constraints ({len(state.constraints)}):\n")
                    for c in sorted(state.constraints, key=lambda x: (x.var1, x.var2)):
                        f.write(f"  {c.var1} ≠ {c.var2}\n")

                f.write(f"Variables: {len(unique_vars)} - {sorted(unique_vars)}\n")
                f.write(f"Grounded: {sorted(grounded_args)}\n")
                f.write(f"\n")

        # Analyze patterns
        f.write(f"\n{'='*80}\n")
        f.write("PATTERN ANALYSIS\n")
        f.write(f"{'='*80}\n\n")

        # Count predicate combinations
        f.write("Most common predicate names:\n")
        predicate_counts = {}
        for state in graph.states:
            for pred in state.predicates:
                name = pred.name
                predicate_counts[name] = predicate_counts.get(name, 0) + 1

        for name, count in sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {name}: {count} occurrences\n")

        # States by complexity
        f.write(f"\nStates by number of predicates:\n")
        complexity_counts = {}
        for state in graph.states:
            num_preds = len(state.predicates)
            complexity_counts[num_preds] = complexity_counts.get(num_preds, 0) + 1

        for num_preds in sorted(complexity_counts.keys()):
            f.write(f"  {num_preds} predicates: {complexity_counts[num_preds]} states\n")

        # Variable usage distribution
        f.write(f"\nVariable usage distribution:\n")
        var_usage_counts = {}
        for state in graph.states:
            unique_vars = set()
            for pred in state.predicates:
                for arg in pred.args:
                    if arg.startswith('?v'):
                        unique_vars.add(arg)
            num_vars = len(unique_vars)
            var_usage_counts[num_vars] = var_usage_counts.get(num_vars, 0) + 1

        for num_vars in sorted(var_usage_counts.keys()):
            f.write(f"  {num_vars} variables: {var_usage_counts[num_vars]} states ({var_usage_counts[num_vars]/len(graph.states)*100:.1f}%)\n")

        # Sample some states with 3 variables to see patterns
        f.write(f"\n{'='*80}\n")
        f.write("SAMPLE: States with 3 variables (max allowed)\n")
        f.write(f"{'='*80}\n\n")

        three_var_states = []
        for state in graph.states:
            unique_vars = set()
            for pred in state.predicates:
                for arg in pred.args:
                    if arg.startswith('?v'):
                        unique_vars.add(arg)
            if len(unique_vars) == 3:
                three_var_states.append(state)

        f.write(f"Found {len(three_var_states)} states with exactly 3 variables\n\n")

        for i, state in enumerate(three_var_states[:20]):  # Show 20 examples
            f.write(f"Example {i+1} (depth {state.depth}):\n")
            for pred in sorted(state.predicates, key=lambda p: (p.name, str(p.args))):
                neg_str = "~" if pred.negated else ""
                f.write(f"  {neg_str}{pred.name}({', '.join(pred.args)})\n")
            f.write(f"\n")

        f.write(f"\n{'='*80}\n")
        f.write("END OF DETAILED OUTPUT\n")
        f.write(f"{'='*80}\n")

    print(f"✓ Detailed output saved to test_3_blocks_output.txt")
    print(f"  Total states: {len(graph.states)}")
    print(f"  States with 3 variables: {len(three_var_states)}")


if __name__ == "__main__":
    test_3_blocks_detailed()
