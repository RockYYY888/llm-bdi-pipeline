# 状态数增加分析 (1093 → 1152)

## 用户问题

> "为什么之前2 blocks只需要1096个states explored，现在加入了pruning以后反而变成了1152个states？"

## 实验验证

### 测试结果

| Mode | 2 Blocks States | Transitions | Complete Goal |
|------|----------------|-------------|---------------|
| **Grounded** | 1,093 | 63,394 | `['clear(a)', 'on(a, b)', 'handempty']` |
| **Variable** | 1,152 | 66,816 | `['on(?arg0, ?arg1)']` |
| **Difference** | **+59 (+5.4%)** | +3,422 | **2 fewer predicates** |

## 根本原因

### 不是 Pruning 的问题

**误解**: 加入 variable abstraction 后会"prune"更多状态，减少探索

**实际情况**: Variable abstraction 的目的是**cache sharing**，不是 pruning

### 真正的原因：Goal Inference 失败

**问题**: Variable mode 下，goal inference 不工作

#### Grounded Mode (正常工作)

```python
# Input goal
goal = [PredicateAtom('on', ['a', 'b'])]

# Inference process
for action in actions:
    for grounded_action in ground_action(action, objects=['a', 'b']):
        # Example: put-on-block(a, b)
        effects = parse_effects(action.effects, bindings={'?b1': 'a', '?b2': 'b'})
        # Effects: [on(a, b), handempty, clear(a), ~holding(a), ~clear(b)]

        # Check if adds goal predicate
        if 'on(a, b)' in add_effects:
            # Include ALL add effects in complete goal
            complete_goal.add(on(a, b))
            complete_goal.add(handempty)
            complete_goal.add(clear(a))  # ← 推断出来的

# Result
complete_goal = ['clear(a)', 'on(a, b)', 'handempty']  ✓ 3 predicates
```

#### Variable Mode (失败)

```python
# Input goal
goal = [PredicateAtom('on', ['?arg0', '?arg1'])]

# Inference process
for action in actions:
    for grounded_action in ground_action(action, objects=['?v0', '?v1']):
        # Example: put-on-block(?v0, ?v1)
        effects = parse_effects(action.effects, bindings={'?b1': '?v0', '?b2': '?v1'})
        # Effects: [on(?v0, ?v1), handempty, clear(?v0), ~holding(?v0), ~clear(?v1)]

        # Check if adds goal predicate
        # 问题：on(?v0, ?v1) ≠ on(?arg0, ?arg1)  ❌
        # 变量名不匹配！Inference 失败

# Result
complete_goal = ['on(?arg0, ?arg1)']  ❌ 只有 1 predicate (inference 失败)
```

### 为什么导致更多状态？

**Less Constrained Goal → More States Explored**

| Aspect | Grounded Mode (3 predicates) | Variable Mode (1 predicate) |
|--------|------------------------------|----------------------------|
| **Goal State** | `{on(a,b), clear(a), handempty}` | `{on(?arg0, ?arg1)}` |
| **Constraints** | 非常specific | 更宽松 |
| **Backward Search** | 只探索导致完整goal的状态 | 探索更多"可能"导致goal的状态 |
| **States** | 1,093 | 1,152 (+5.4%) |

**示例**:

Grounded mode的goal state明确要求：
- ✓ `on(a, b)` 必须为真
- ✓ `clear(a)` 必须为真
- ✓ `handempty` 必须为真

Variable mode的goal state只要求：
- ✓ `on(?arg0, ?arg1)` 必须为真
- ❌ `clear(?arg0)` 不要求（因为inference失败）
- ❌ `handempty` 不要求（因为inference失败）

结果：更多的中间状态被认为是"valid"，导致探索更多状态。

---

## 详细分析

### Goal Inference 的重要性

**设计决策 #3** (from design docs):
> "Complete goal state = NOT just predicates in original goal - include all relevant world state"

**Why it matters**:

Without inference:
```
Goal: on(a, b)
Valid predecessor: any state where we can apply an action leading to on(a, b)
→ MANY states qualify
```

With inference:
```
Goal: {on(a, b), clear(a), handempty}
Valid predecessor: only states where we can achieve ALL three
→ FEWER states qualify
```

### Variable Name Mismatch

**Root cause**: 变量命名不一致

- Normalization uses: `?arg0, ?arg1, ...`
- Grounding uses: `?v0, ?v1, ...`
- Predicates don't match → Inference fails

**Why different names?**

1. **Normalization** (`variable_normalizer.py`):
   ```python
   obj_to_var[arg] = f"?arg{var_counter}"  # ← ?arg0, ?arg1
   ```

2. **Forward Planner** (`forward_planner.py`):
   ```python
   self.objects = variables  # ['?v0', '?v1']
   ```

3. **Grounding**:
   ```python
   grounded_action.bindings = {'?b1': '?v0', '?b2': '?v1'}
   effects = [on(?v0, ?v1), clear(?v0), ...]
   ```

4. **Matching**:
   ```python
   goal_pred = on(?arg0, ?arg1)
   effect_pred = on(?v0, ?v1)
   goal_pred == effect_pred  # False! ❌
   ```

---

## 解决方案

### Option 1: 统一变量命名 (推荐)

**改变**: 让 normalizer 使用 `?v0, ?v1, ...` 而不是 `?arg0, ?arg1, ...`

```python
# variable_normalizer.py
obj_to_var[arg] = f"?v{var_counter}"  # 改为 ?v0, ?v1
```

**优点**:
- ✓ Simple fix
- ✓ Goal inference 就能工作
- ✓ Variable mode 也会有 1,093 states (和 grounded mode 一致)

**缺点**:
- ❌ Schema cache keys 会改变（需要重新build cache）

### Option 2: 改进 Goal Inference 的匹配逻辑

**改变**: 让 predicate 比较忽略变量名，只比较结构

```python
def _predicates_match_structurally(pred1, pred2):
    """Check if predicates match ignoring variable names"""
    if pred1.name != pred2.name or len(pred1.args) != len(pred2.args):
        return False

    for arg1, arg2 in zip(pred1.args, pred2.args):
        # Both variables → match
        if arg1.startswith('?') and arg2.startswith('?'):
            continue
        # Both same concrete value → match
        if arg1 == arg2:
            continue
        # Otherwise → don't match
        return False

    return True
```

**优点**:
- ✓ 更鲁棒
- ✓ 不需要改变命名convention

**缺点**:
- ❌ 更复杂
- ❌ 需要修改多处代码

### Option 3: Variable Mode 专门的 Goal Inference

**改变**: 为 variable mode 实现单独的 inference 逻辑

```python
def infer_complete_goal_state_variable_mode(self, goal_predicates):
    """Goal inference for variable mode using unification"""
    complete_goal = set(goal_predicates)

    for goal_pred in goal_predicates:
        for action in self.domain.actions:
            # Generate schema-level action (not grounded)
            schema_effects = self._get_schema_effects(action)

            # Try to unify goal_pred with each effect
            for effect_pred in schema_effects:
                substitution = self._unify(goal_pred, effect_pred)
                if substitution:
                    # Apply substitution to all effects
                    for eff in schema_effects:
                        instantiated = self._apply_substitution(eff, substitution)
                        complete_goal.add(instantiated)

    return complete_goal
```

**优点**:
- ✓ 正确处理 variable mode
- ✓ 不影响 grounded mode

**缺点**:
- ❌ 需要实现 unification
- ❌ 代码复杂度高

---

## 推荐方案

### 短期修复 (Option 1)

统一变量命名为 `?v{i}`:

```python
# src/stage3_code_generation/variable_normalizer.py:191
# 改变
obj_to_var[arg] = f"?arg{var_counter}"
# 为
obj_to_var[arg] = f"?v{var_counter}"
```

**预期结果**:
- 2 blocks variable mode: 1,152 → **1,093 states** (和 grounded mode 一致)
- Goal inference 正常工作
- Complete goal: `['on(?v0, ?v1)', 'clear(?v0)', 'handempty']` ✓

### 长期优化 (Option 2)

实现structural matching for predicates，使系统更鲁棒。

---

## 结论

### 回答用户问题

**Q**: "为什么加入了pruning以后反而变成了1152个states？"

**A**:
1. **不是pruning的问题** - Variable abstraction 的目的是 cache sharing，不是 pruning
2. **Goal inference 失败** - Variable mode下，因为变量名不匹配，inference 没有工作
3. **Less constrained goal** - 只有1个predicate vs 3个predicates，导致探索更多状态
4. **预期行为** - 修复 inference 后，应该回到 1,093 states

### 关键洞察

**Variable Abstraction ≠ State Space Pruning**

Variable abstraction 是为了：
- ✓ Schema-level caching (cache hits)
- ✓ Reuse exploration results
- ✓ Reduce redundant explorations

**不是为了**:
- ❌ Reduce state space size
- ❌ Prune states
- ❌ Find shorter plans

**正确的期望**:
- ✅ Variable mode 应该探索**相同数量**的 states
- ✅ 但可以 **reuse** 这些 explorations across different object instantiations
- ✅ 整体pipeline更快（因为cache hits），但单个exploration不变

---

## 测试验证

### Before Fix (当前)

```bash
$ python test_2blocks.py
Grounded mode: 1,093 states
Variable mode: 1,152 states  ← 错误：更多states
Complete goal (grounded): 3 predicates
Complete goal (variable): 1 predicate  ← inference失败
```

### After Fix (预期)

```bash
$ python test_2blocks.py  # 修复后
Grounded mode: 1,093 states
Variable mode: 1,093 states  ← 正确：相同states
Complete goal (grounded): 3 predicates
Complete goal (variable): 3 predicates  ← inference成功
```

---

**文档创建**: 2025-11-10
**Status**: Bug identified - Goal inference incompatible with variable mode
**Fix**: Option 1 (统一变量命名) 推荐作为immediate fix
