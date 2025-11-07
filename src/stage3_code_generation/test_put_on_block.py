from src.stage3_code_generation.forward_planner import ForwardStatePlanner
from src.stage3_code_generation.state_space import PredicateAtom
from src.utils.pddl_parser import PDDLParser
from src.stage3_code_generation.pddl_condition_parser import PDDLConditionParser

domain = PDDLParser.parse_domain('src/legacy/fond/domains/blocksworld/domain.pddl')
goal_preds = [PredicateAtom('on', ('a', 'b'))]
objects = ['a', 'b']

planner = ForwardStatePlanner(domain, objects)
graph = planner.explore_from_goal(goal_preds, max_depth=1)

# Find the state with holding(a), clear(b)
target_state = None
for state in graph.states:
    if state.depth == 1:
        preds_str = {str(p) for p in state.predicates}
        if 'holding(a)' in preds_str and 'clear(b)' in preds_str:
            target_state = state
            break

if target_state:
    preds_list = ', '.join(str(p) for p in sorted(target_state.predicates, key=str))
    print(f'Found target state: {preds_list}')
    print(f'Depth: {target_state.depth}')

    # Check put-on-block action
    put_on_block = next((a for a in domain.actions if a.name == 'put-on-block'), None)
    if put_on_block:
        print(f'\nput-on-block action found')
        print(f'  Parameters: {put_on_block.parameters}')
        print(f'  Preconditions: {put_on_block.preconditions[:100]}...')

        # Try to ground it with (a, b)
        bindings = {'?b1': 'a', '?b2': 'b'}
        print(f'\n  Testing with bindings: {bindings}')

        # Check preconditions
        parser = PDDLConditionParser()
        try:
            preconds = parser.parse(put_on_block.preconditions, bindings)
            print(f'  Parsed preconditions:')
            for p in preconds:
                if p in target_state.predicates:
                    print(f'    ✓ {p}')
                else:
                    print(f'    ✗ {p} (NOT in state)')
        except Exception as e:
            print(f'  Error: {e}')
else:
    print('Target state not found')
