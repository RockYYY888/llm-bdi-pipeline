from src.stage3_code_generation.forward_planner import ForwardStatePlanner
from src.stage3_code_generation.state_space import PredicateAtom, WorldState
from src.utils.pddl_parser import PDDLParser
from src.stage3_code_generation.pddl_condition_parser import PDDLEffectParser

domain = PDDLParser.parse_domain('src/legacy/fond/domains/blocksworld/domain.pddl')

# Find pickup action
pickup = next((a for a in domain.actions if a.name == 'pick-up'), None)

if pickup:
    print('pick-up action:')
    print(f'  Parameters: {pickup.parameters}')
    print(f'  Preconditions: {pickup.preconditions}')
    print(f'  Effects: {pickup.effects}')

    # Ground with (a, b)
    bindings = {'?b1': 'a', '?b2': 'b'}
    print(f'\nGround with {bindings}:')

    effect_parser = PDDLEffectParser()
    effects = effect_parser.parse(pickup.effects, bindings)

    print(f'\nParsed effects ({len(effects)} branches):')
    for i, branch in enumerate(effects):
        print(f'\nBranch {i+1}:')
        for eff in branch:
            symbol = '+' if eff.is_add else '-'
            print(f'  {symbol}{eff.predicate}')

    # Apply to goal state
    goal_state = WorldState(frozenset([PredicateAtom('on', ('a', 'b'))]))
    print(f'\nApplying to goal state: {{on(a, b)}}')

    for i, branch in enumerate(effects):
        new_preds = set(goal_state.predicates)
        print(f'\nBranch {i+1} application:')
        for eff in branch:
            if eff.is_add:
                print(f'  Adding: {eff.predicate}')
                new_preds.add(eff.predicate)
            else:
                print(f'  Removing: {eff.predicate}')
                new_preds.discard(eff.predicate)

        print(f'  Result: {{{", ".join(str(p) for p in sorted(new_preds, key=str))}}}')
        has_on_ab = any(str(p) == 'on(a, b)' for p in new_preds)
        print(f'  Still has on(a, b)? {has_on_ab}')
