"""
解释为什么 backward planning 中 goal predicate 会持续存在
"""
from src.utils.pddl_parser import PDDLParser
from src.stage3_code_generation.pddl_condition_parser import PDDLConditionParser, PDDLEffectParser

domain = PDDLParser.parse_domain('src/legacy/fond/domains/blocksworld/domain.pddl')

print("=" * 80)
print("分析: 哪些 actions 可以删除 on(a, b)?")
print("=" * 80)

goal = "on(a, b)"
objects = ['a', 'b', 'c']

condition_parser = PDDLConditionParser()
effect_parser = PDDLEffectParser()

print(f"\nGoal: {goal}")
print(f"Objects: {objects}")

actions_that_delete_goal = []

for action in domain.actions:
    print(f"\n{'='*60}")
    print(f"Action: {action.name}")
    print(f"Parameters: {action.parameters}")

    # Ground with a, b
    if len(action.parameters) == 2:
        bindings = {'?b1': 'a', '?b2': 'b'}
        args = ('a', 'b')
    elif len(action.parameters) == 1:
        bindings = {'?b': 'a'}
        args = ('a',)
    elif len(action.parameters) == 3:
        bindings = {'?b1': 'a', '?b2': 'b', '?b3': 'c'}
        args = ('a', 'b', 'c')
    else:
        continue

    print(f"Grounded as: {action.name}{args}")
    print(f"Bindings: {bindings}")

    # Parse preconditions
    try:
        preconditions = condition_parser.parse(action.preconditions, bindings)
        print(f"\nPreconditions:")
        has_goal_precond = False
        for p in preconditions:
            print(f"  - {p}")
            if str(p) == goal or (p.name == 'on' and p.args == ('a', 'b')):
                has_goal_precond = True
                print(f"    ^^^ 需要 {goal} 作为前置条件!")
    except Exception as e:
        print(f"  Error: {e}")
        preconditions = []
        has_goal_precond = False

    # Parse effects
    try:
        effect_branches = effect_parser.parse(action.effects, bindings)
        print(f"\nEffects ({len(effect_branches)} branches):")
        deletes_goal = False

        for i, branch in enumerate(effect_branches):
            print(f"  Branch {i+1}:")
            for eff in branch:
                symbol = "+" if eff.is_add else "-"
                print(f"    {symbol}{eff.predicate}")
                if not eff.is_add and eff.predicate.name == 'on' and eff.predicate.args == ('a', 'b'):
                    deletes_goal = True
                    print(f"      ^^^ 删除 {goal}!")

        if deletes_goal:
            actions_that_delete_goal.append({
                'action': action.name,
                'args': args,
                'needs_goal_precond': has_goal_precond,
                'preconditions': [str(p) for p in preconditions]
            })

    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 80)
print("总结: Actions that delete on(a, b)")
print("=" * 80)

if not actions_that_delete_goal:
    print("\n没有 action 可以删除 on(a, b)!")
else:
    for item in actions_that_delete_goal:
        print(f"\n{item['action']}{item['args']}:")
        print(f"  需要 on(a,b) 作为前置条件: {item['needs_goal_precond']}")
        if item['needs_goal_precond']:
            print(f"  ⚠️ 问题: 要删除 on(a,b), 必须先有 on(a,b)!")
            print(f"  → backward planning 会保留 on(a,b) 在 predecessor state")
        print(f"  所有前置条件:")
        for p in item['preconditions']:
            print(f"    - {p}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("""
在 blocksworld domain 中:
1. 只有 pick-up(?b1, ?b2) 可以删除 on(?b1, ?b2)
2. 但 pick-up 需要 on(?b1, ?b2) 作为前置条件!
3. 因此在 backward planning 中:
   - 从 {on(a,b)} 开始
   - 应用 pick-up(a,b) backward
   - 必须添加 precondition on(a,b) 到 predecessor state
   - 结果: predecessor state 仍然包含 on(a,b)

这就是为什么所有 explored states 都包含 goal predicate!

这是 regression-based backward planning 的正确行为，
但对于生成 AgentSpeak forward execution plans 来说不合适。
""")
