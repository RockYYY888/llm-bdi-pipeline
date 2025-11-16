# 基于一阶谓词逻辑的Lifted Planning

## 第一性原理

### 核心问题

**问题：** 如何实现真正domain-independent的lifted planning？

**错误方向：** Domain-specific macros（如clear-block, deliver-all）
- ❌ 需要为每个domain手工定义
- ❌ 无法泛化到新domains
- ❌ 维护成本高

**正确方向：** 一阶谓词逻辑（First-Order Logic）
- ✅ 数学基础扎实
- ✅ 完全domain-independent
- ✅ 可自动化
- ✅ 适用于任何PDDL domain

## 一阶谓词逻辑基础

### Quantifiers（量词）

**Existential (∃)**: "存在某个"
```
∃?X. P(?X)  ≡  "存在某个?X满足P(?X)"
```

**Universal (∀)**: "对所有"
```
∀?X. P(?X)  ≡  "对所有?X都满足P(?X)"
```

### 为什么Quantifiers解决问题

**场景：** 要实现on(a, b)，但b上有c, d, e三个blocks

**Grounded方法（枚举）：**
```
State: {on(c, b), on(d, b), on(e, b)}
Actions to apply:
  pick-up(c, b)  → State1
  pick-up(d, b)  → State2
  pick-up(e, b)  → State3

❌ 3个transitions！枚举了每个blocker
```

**Quantified方法（不枚举）：**
```
State: {∃?Z. on(?Z, b)}
Action to apply:
  ∃?Z. pick-up(?Z, b)  → State_cleared

✅ 1个abstract transition！不枚举blockers
```

**关键区别：**
- Grounded: State包含3个具体predicates → 3个transitions
- Quantified: State包含1个quantified predicate → 1个transition
- State space从O(n)变成O(1)！

## Domain-Independent算法

### 1. Quantified State Representation

```python
class QuantifiedPredicate:
    """
    ∃?X. P(?X) 或 ∀?X. P(?X)

    完全domain-independent - 适用于任何predicate P
    """
    quantifier: Quantifier  # EXISTS or FORALL
    variables: List[str]    # 量化的变量
    formula: Formula        # 可以是PredicateAtom或复合formula
    constraints: ConstraintSet

class AbstractState:
    """
    混合concrete和quantified predicates
    """
    concrete: Set[PredicateAtom]  # 我们关心具体值的predicates
    quantified: Set[QuantifiedPredicate]  # 不关心具体值的predicates
    constraints: ConstraintSet
```

**例子（Blocksworld）：**
```python
# Goal: on(a, b)
# b上有未知数量的blockers

State = AbstractState(
    concrete = {
        on(a, table),  # a的位置我们关心
        clear(a)       # a的状态我们关心
    },
    quantified = {
        ∃?Z. on(?Z, b),  # b上有某些blockers（不关心具体哪些）
        ∀?W. (clear(?W) ∧ ?W != a) → ontable(?W)  # 其他clear的都在table
    },
    constraints = {}
)

# ✅ Domain-independent: 同样的结构适用于任何domain
```

**例子（Logistics）：**
```python
# Goal: at(pkg1, location5)
# depot里有未知数量的packages

State = AbstractState(
    concrete = {
        at(pkg1, depot),
        at(truck1, depot)
    },
    quantified = {
        ∃?P. (at(?P, depot) ∧ ?P != pkg1),  # depot里有其他packages
        ∀?L. reachable(truck1, ?L)  # 所有location都可达
    },
    constraints = {}
)

# ✅ 同样的quantified representation！
```

### 2. Quantifier Detection（自动化）

**核心idea：** 当多个predicates可以unify同一个action precondition时，合并为quantified predicate

```python
def detect_quantification_opportunity(state, action_precond):
    """
    Domain-independent规则：

    如果state中有多个predicates {P1, P2, ..., Pn}都可以unify action的precondition，
    并且这些predicates只在某些变量上不同，
    则可以合并为 ∃?X. P(?X)
    """

    # 找到所有可以unify的predicates
    matching_preds = []
    for pred in state.concrete:
        if can_unify(pred, action_precond):
            matching_preds.append(pred)

    if len(matching_preds) <= 1:
        return None  # 没有quantification机会

    # 检查是否可以抽象
    if can_abstract_to_pattern(matching_preds):
        # 创建quantified predicate
        pattern, quantified_vars = extract_pattern(matching_preds)
        return QuantifiedPredicate(
            quantifier=EXISTS,
            variables=quantified_vars,
            formula=pattern,
            constraints=extract_constraints(matching_preds)
        )

    return None

# 例子：
# matching_preds = [on(c, b), on(d, b), on(e, b)]
# pattern = on(?Z, b)
# quantified_vars = [?Z]
# → ∃?Z. on(?Z, b)
```

**Domain-independent:**
- 不需要知道predicate名称（on, at, etc）
- 不需要知道domain类型（blocksworld, logistics）
- 纯粹基于结构模式匹配

### 3. Non-Enumerating Action Application

```python
def apply_action_non_enumerating(action, state):
    """
    关键：不为每个可能的unification生成transition
    """

    # 检查是否可以用quantifier
    quantification = detect_quantification_opportunity(
        state,
        action.preconditions
    )

    if quantification:
        # 生成一个quantified transition
        new_state = apply_with_quantifier(action, state, quantification)
        return [(new_state, quantified_substitution)]
    else:
        # 正常处理（只有1个match或无法quantify）
        matches = find_all_unifications(action.preconditions, state)
        results = []
        for match in matches:
            new_state = apply_concrete(action, state, match)
            results.append((new_state, match))
        return results

# 例子：
# State: {on(c,b), on(d,b), on(e,b)}
# Action: pick-up(?X, ?Y)
#
# 检测到3个matches可以quantify
# → 生成1个transition with ∃?Z. pick-up(?Z, b)
# ✅ 而不是3个transitions！
```

### 4. Quantifier Propagation

```python
def apply_with_quantifier(action, state, quantification):
    """
    应用quantified action，保持quantified形式
    """

    new_concrete = set(state.concrete)
    new_quantified = set(state.quantified)

    # 应用effects
    for effect in action.effects:
        if effect.is_add:
            # 如果effect涉及quantified variable，保持quantified
            if involves_quantified_var(effect, quantification):
                quantified_effect = apply_quantifier_to_effect(
                    effect,
                    quantification
                )
                new_quantified.add(quantified_effect)
            else:
                new_concrete.add(effect.predicate)
        else:
            # Delete effect
            if involves_quantified_var(effect, quantification):
                # 从quantified set中移除
                new_quantified = remove_quantified(
                    new_quantified,
                    effect,
                    quantification
                )
            else:
                new_concrete.discard(effect.predicate)

    return AbstractState(new_concrete, new_quantified, state.constraints)

# 例子：
# State: {∃?Z. on(?Z, b)}
# Action: pick-up(?X, ?Y) with effect: -on(?X, ?Y), +holding(?X)
#
# Apply:
#   - Remove: ∃?Z. on(?Z, b)  (deleted by effect)
#   + Add: ∃?Z. holding(?Z)  (added by effect, still quantified)
#
# Result: {∃?Z. holding(?Z)}
# ✅ 仍然是quantified，没有具体化！
```

### 5. Plan Instantiation（最后一步）

```python
def instantiate_plan(abstract_plan, concrete_world_state):
    """
    将abstract plan（带quantifiers）转换为concrete plan

    这一步才具体化quantified variables
    """

    concrete_plan = []

    for abstract_action in abstract_plan:
        if has_quantifiers(abstract_action):
            # 消除quantifiers
            concrete_actions = eliminate_quantifiers(
                abstract_action,
                concrete_world_state
            )
            concrete_plan.extend(concrete_actions)
        else:
            # 已经是concrete的
            concrete_plan.append(abstract_action)

    return concrete_plan

# 例子：
# Abstract plan: [∃?Z. pick-up(?Z, b), put-on(a, b)]
# Concrete world: {on(c, b), on(d, b), on(e, b)}
#
# Instantiation:
#   ∃?Z. pick-up(?Z, b)  → [pick-up(c, b), pick-up(d, b), pick-up(e, b)]
#   put-on(a, b)         → [put-on(a, b)]
#
# Concrete plan: [pick-up(c, b), pick-up(d, b), pick-up(e, b), put-on(a, b)]
```

## Domain-Independent性能分析

### State Space复杂度

**Grounded (即使用变量):**
```
n个blockers on b
→ n个predicates: on(?v0, b), on(?v1, b), ..., on(?vn-1, b)
→ n个possible actions to apply
→ State space: O(n)
```

**Quantified:**
```
n个blockers on b
→ 1个quantified predicate: ∃?Z. on(?Z, b)
→ 1个quantified action to apply
→ State space: O(1)
```

**对比：**
| Blockers | Grounded States | Quantified States | Reduction |
|----------|----------------|-------------------|-----------|
| 3 | 3 | 1 | 3x |
| 10 | 10 | 1 | 10x |
| 100 | 100 | 1 | 100x |
| n | n | 1 | **nx** |

### 适用于任何Domain

**Blocksworld:** ∃?Z. on(?Z, b)
**Logistics:** ∃?P. at(?P, location)
**Rovers:** ∃?R. available(?R)
**Satellite:** ∃?I. calibrated(?I, target)

**完全相同的算法！** 不需要domain knowledge！

## 实现路线图

### Phase 1: 基础设施 (2-3 days)

1. **定义Quantifier类型**
   ```python
   class Quantifier(Enum):
       EXISTS = "∃"
       FORALL = "∀"
   ```

2. **实现QuantifiedPredicate**
   ```python
   @dataclass(frozen=True)
   class QuantifiedPredicate:
       quantifier: Quantifier
       variables: List[str]
       formula: PredicateAtom
       constraints: ConstraintSet
   ```

3. **更新AbstractState**
   ```python
   class AbstractState:
       concrete: FrozenSet[PredicateAtom]
       quantified: FrozenSet[QuantifiedPredicate]  # NEW
       constraints: ConstraintSet
   ```

### Phase 2: Quantifier Detection (3-4 days)

1. **实现pattern matching**
   - 检测多个predicates可以合并
   - 提取共同pattern
   - 识别varying variables

2. **实现quantifier creation**
   - 从matches创建∃或∀
   - 提取constraints
   - Validation

3. **测试**
   - 多个domains (blocksworld, logistics, rovers)
   - 各种patterns

### Phase 3: Non-Enumerating Exploration (5-7 days)

1. **修改_apply_abstract_action**
   - 调用quantifier detection
   - 生成quantified transitions
   - 不为每个match生成transition

2. **实现quantifier propagation**
   - Through action effects
   - Constraint maintenance
   - Simplification rules

3. **测试**
   - 验证state space reduction
   - 10个blockers → 1 transition (not 10)

### Phase 4: Plan Instantiation (3-4 days)

1. **实现quantifier elimination**
   - ∃?X. P(?X) → concrete instances
   - ∀?X. P(?X) → all instances

2. **Variable binding**
   - Propagate bindings
   - Handle dependencies

3. **测试**
   - Abstract → concrete correctness
   - Multiple instantiations

## 验证标准

### 正确性

```python
# 对于任何domain和任何goal：
abstract_plan = lifted_planner.plan(goal)
concrete_plan = instantiate(abstract_plan, world_state)

# Concrete plan必须valid
assert is_valid_plan(concrete_plan, initial_state, goal)

# Abstract plan的state space必须远小于grounded
assert len(abstract_states) << len(grounded_states)
```

### Domain-Independence

```python
# 同样的代码应该work on所有domains
for domain in [blocksworld, logistics, rovers, satellite, ...]:
    planner = LiftedPlanner(domain)
    plan = planner.plan(goal)
    assert plan is not None

    # 不需要domain-specific code
    assert no_domain_specific_checks(planner)
```

### 性能

```python
# State space应该独立于object数量
for n_objects in [10, 100, 1000]:
    states = planner.explore(goal, n_objects)

    # Quantified states数量应该稳定
    assert len(states) < 1000  # 不随n_objects增长
```

## 总结

**核心原则：**
1. ✅ 基于一阶谓词逻辑（数学基础）
2. ✅ 完全domain-independent（适用任何PDDL domain）
3. ✅ 自动化（不需要人工定义macros）
4. ✅ State space O(1) for clearing operations

**不是：**
- ❌ Domain-specific macros
- ❌ Hand-coded optimizations
- ❌ Ad-hoc solutions

**这才是从第一性原理出发的正确方向！**
