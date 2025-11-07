from src.stage3_code_generation.forward_planner import ForwardStatePlanner
from src.stage3_code_generation.state_space import PredicateAtom, WorldState
from src.utils.pddl_parser import PDDLParser
from src.stage3_code_generation.pddl_condition_parser import PDDLConditionParser, PDDLEffectParser

domain = PDDLParser.parse_domain('src/legacy/fond/domains/blocksworld/domain.pddl')
goal_preds = [PredicateAtom('on', ('a', 'b'))]
objects = ['a', 'b']

planner = ForwardStatePlanner(domain, objects)
graph = planner.explore_from_goal(goal_preds, max_depth=1)

# Find state with holding(a), clear(b)
target_state = None
for state in graph.states:
    if state.depth == 1:
        preds_str = {str(p) for p in state.predicates}
        if 'holding(a)' in preds_str and 'clear(b)' in preds_str:
            target_state = state
            break

if target_state:
    print('Target state predicates:')
    for p in sorted(target_state.predicates, key=str):
        print(f'  - {p}')

    # Apply put-on-block(a, b)
    put_on_block = next((a for a in domain.actions if a.name == 'put-on-block'), None)
    bindings = {'?b1': 'a', '?b2': 'b'}

    effect_parser = PDDLEffectParser()
    effects = effect_parser.parse(put_on_block.effects, bindings)

    print(f'\nput-on-block(a, b) effects ({len(effects)} branches):')
    for i, branch in enumerate(effects):
        print(f'\nBranch {i+1}:')
        new_preds = set(target_state.predicates)
        for eff in branch:
            symbol = '+' if eff.is_add else '-'
            print(f'  {symbol}{eff.predicate}')
            if eff.is_add:
                new_preds.add(eff.predicate)
            else:
                new_preds.discard(eff.predicate)

        print(f'  Result state predicates:')
        for p in sorted(new_preds, key=str):
            print(f'    - {p}')

        # Check if this is goal state
        goal_predicates = set(goal_preds)
        if goal_predicates == set([p for p in new_preds if str(p) == 'on(a, b)']):
            print(f'    → This would be goal state!')

        # Check if on(a,b) is in result
        has_goal = any(str(p) == 'on(a, b)' for p in new_preds)
        print(f'    Contains on(a, b): {has_goal}')
