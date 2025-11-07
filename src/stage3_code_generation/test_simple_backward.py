"""
Simple test for backward planning with depth=1
"""
from src.stage3_code_generation.forward_planner import ForwardStatePlanner
from src.stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser

# Load domain
domain = PDDLParser.parse_domain('src/legacy/fond/domains/blocksworld/domain.pddl')

# Simple goal: on(a, b)
goal_preds = [PredicateAtom('on', ('a', 'b'))]
objects = ['a', 'b']

# Create planner with depth=1
planner = ForwardStatePlanner(domain, objects)
graph = planner.explore_from_goal(goal_preds, max_depth=1)

print(f'\nStates: {len(graph.states)}')
print(f'Transitions: {len(graph.transitions)}')
print(f'Leaf states: {len(graph.get_leaf_states())}')

# Print some example states
print('\nExample states:')
for i, state in enumerate(list(graph.states)[:5]):
    preds_str = ', '.join(str(p) for p in sorted(state.predicates, key=str)[:3])
    print(f'  State {i}: {preds_str}...')

# Print paths
paths = graph.find_shortest_paths_to_goal()
print(f'\nStates with paths to goal: {len(paths)}')

# Check one example path
for state, path in list(paths.items())[:3]:
    if state != graph.goal_state and path:
        print(f'\nExample path from state:')
        preds_str = ', '.join(str(p) for p in sorted(state.predicates, key=str)[:3])
        print(f'  State: {preds_str}')
        print(f'  Path length: {len(path)}')
        print(f'  First action: {path[0].action.name} with args {path[0].action_args}')
        break
