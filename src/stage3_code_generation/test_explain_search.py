"""
详细展示backward planning搜索过程
"""
from src.stage3_code_generation.forward_planner import ForwardStatePlanner
from src.stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser

# Load domain
domain = PDDLParser.parse_domain('src/legacy/fond/domains/blocksworld/domain.pddl')

print("=" * 80)
print("PDDL Domain Actions:")
print("=" * 80)
for action in domain.actions[:2]:  # 只显示前2个
    print(f"\nAction: {action.name}")
    print(f"  Parameters: {action.parameters}")
    print(f"  Preconditions: {action.preconditions[:100]}...")
    print(f"  Effects: {action.effects[:100]}...")

print("\n" + "=" * 80)
print("Grounding Example:")
print("=" * 80)

# 展示 grounding
action = domain.actions[0]  # pick-up
objects = ['a', 'b']

print(f"\nAction: {action.name}")
print(f"Parameters: {action.parameters}")
print(f"Objects: {objects}")

planner = ForwardStatePlanner(domain, objects)
grounded = planner._ground_action(action)

print(f"\nGrounded instances: {len(grounded)}")
for i, g in enumerate(grounded[:4]):
    print(f"  {i+1}. {action.name}{tuple(g.args)}")
    print(f"     Bindings: {g.bindings}")

print("\n" + "=" * 80)
print("Backward State Transition Example:")
print("=" * 80)

# 从 goal state 开始
goal_state_predicates = frozenset([PredicateAtom('on', ('a', 'b'))])
print(f"\nGoal State: {{{', '.join(str(p) for p in goal_state_predicates)}}}")

# 选择一个 grounded action
test_action = grounded[0]  # 比如 pick-up(a, b)
print(f"\nApplying action (backward): {action.name}{tuple(test_action.args)}")
print(f"  Bindings: {test_action.bindings}")

# 手动展示 effect 解析
from src.stage3_code_generation.pddl_condition_parser import PDDLConditionParser, PDDLEffectParser

condition_parser = PDDLConditionParser()
effect_parser = PDDLEffectParser()

# Parse preconditions
try:
    preconditions = condition_parser.parse(action.preconditions, test_action.bindings)
    print(f"\n  Parsed Preconditions:")
    for p in preconditions:
        print(f"    - {p}")
except Exception as e:
    print(f"  Error parsing preconditions: {e}")
    preconditions = []

# Parse effects
try:
    effect_branches = effect_parser.parse(action.effects, test_action.bindings)
    print(f"\n  Parsed Effects (branches: {len(effect_branches)}):")
    for i, branch in enumerate(effect_branches):
        print(f"    Branch {i+1}:")
        for eff in branch:
            symbol = "+" if eff.is_add else "-"
            print(f"      {symbol}{eff.predicate}")
except Exception as e:
    print(f"  Error parsing effects: {e}")
    effect_branches = []

# 应用 backward effects
print(f"\n  Applying Backward Effects:")
print(f"    Starting state: {{{', '.join(str(p) for p in goal_state_predicates)}}}")

for i, branch in enumerate(effect_branches[:2]):  # 只显示前2个分支
    new_predicates = set(goal_state_predicates)
    print(f"\n    Branch {i+1} transformation:")

    for eff in branch:
        if eff.is_add:
            print(f"      - Effect: +{eff.predicate} (forward)")
            print(f"        → Backward: REMOVE {eff.predicate}")
            new_predicates.discard(eff.predicate)
        else:
            print(f"      - Effect: -{eff.predicate} (forward)")
            print(f"        → Backward: ADD {eff.predicate}")
            new_predicates.add(eff.predicate)

    # Add preconditions
    print(f"      - Adding preconditions to predecessor state:")
    for precond in preconditions:
        if not precond.negated:
            print(f"        + {precond}")
            new_predicates.add(precond)

    print(f"\n    Resulting predecessor state:")
    print(f"      {{{', '.join(str(p) for p in sorted(new_predicates, key=str))}}}")

print("\n" + "=" * 80)
print("Complete Search Example (depth=1):")
print("=" * 80)

# 完整搜索
goal_preds = [PredicateAtom('on', ('a', 'b'))]
graph = planner.explore_from_goal(goal_preds, max_depth=1)

print(f"\nExplored: {len(graph.states)} states, {len(graph.transitions)} transitions")

# 显示几个例子状态
print(f"\nExample predecessor states:")
count = 0
for state in graph.states:
    if state != graph.goal_state:
        preds_str = ', '.join(str(p) for p in sorted(state.predicates, key=str))
        print(f"  {count+1}. {{{preds_str}}}")
        count += 1
        if count >= 3:
            break

# 检查是否有不包含 goal 的状态
print(f"\n检查关键问题: 有多少状态不包含 on(a,b)?")
goal_pred_set = graph.goal_state.predicates
states_without_goal = 0
for state in graph.states:
    if state != graph.goal_state and not goal_pred_set.issubset(state.predicates):
        states_without_goal += 1
        if states_without_goal <= 3:
            preds_str = ', '.join(str(p) for p in sorted(state.predicates, key=str))
            print(f"  Found: {{{preds_str}}}")

if states_without_goal == 0:
    print(f"  结果: 0 个状态不包含 on(a,b)")
    print(f"  原因: backward planning 会保留 goal predicates 当它们是 preconditions")
else:
    print(f"  结果: {states_without_goal} 个状态不包含 on(a,b)")
