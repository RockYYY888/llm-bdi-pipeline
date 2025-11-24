# Backward Search Refactoring Summary

## 问题描述

在后向搜索（Backward Search）过程中，发现了以下问题：

1. ❌ **Self-Reference Predicates**: 生成了非法谓词如 `on(?v0, ?v0)`, `on(a, a)`
2. ❌ **变量命名混乱**: 混用 `?v0, ?v1` 和 `?1, ?2` 两种格式
3. ❌ **Planning策略错误**: 过早使用变量化目标进行搜索

## 解决方案

### ✅ 1. 采用策略A：Grounded Search

**正确的Planning流程**：

```
Input: Grounded Goal
  例如: on(a, b)  ← 使用实际对象

  ↓

Backward Search with GROUNDED predicates
  - 使用实际对象进行搜索
  - 仅在无法完全绑定时生成变量
  - 示例1: put-on-block(a, b) → holding(a) ∧ clear(b)
  - 示例2: pick-up(a, ?v0) → 第二个参数未知，生成变量?v0

  ↓

State Graph (包含grounded predicates)
  - 状态包含具体对象和按需生成的变量

  ↓

Normalization for Caching
  - on(a, b) → on(?v0, ?v1)
  - on(c, d) → on(?v0, ?v1)  ✓ 相同pattern，共享cache

  ↓

Code Generation
  - 使用var_mapping: {a: ?v0, b: ?v1}
  - 转换为AgentSpeak: ?v0 → V0, ?v1 → V1
```

### ✅ 2. 统一变量命名：`?v0, ?v1, ?v2, ...`

**为什么选择这个格式**：
- ✅ AgentSpeak兼容：`?v0` → `V0` (首字母大写)
- ✅ 从0开始符合编程习惯
- ✅ 与PDDL的 `?b1, ?b2` 格式区分

**示例**：
```
Goal: on(a, b)
Action: put-on-block(?b1, ?b2)
Binding: {?b1: a, ?b2: b}
Next Goal: holding(a) ∧ clear(b)

Goal: holding(a)
Action: pick-up(?b1, ?b2)
Binding: {?b1: a}  ← 不完全绑定
Complete Binding: {?b1: a, ?b2: ?v0}  ← 生成变量?v0
Next Goal: handempty ∧ clear(a) ∧ on(a, ?v0) ∧ (a ≠ ?v0)
```

### ✅ 3. 基于约束的验证（非硬编码）

**❌ 错误方法：硬编码规则**
```python
# 错误：假设所有2参数谓词都不能self-reference
if len(pred.args) >= 2:
    if pred.args[0] == pred.args[1]:
        return False  # 拒绝 on(?v0, ?v0)
```

**✅ 正确方法：检查不等式约束**
```python
# 正确：检查PDDL中定义的不等式约束
for constraint in constraints:
    if constraint.var1 == constraint.var2:
        # 约束 ?v0 ≠ ?v0 永远为假
        return False
```

**为什么这样做**：
1. 不是所有谓词都禁止self-reference
2. 应该依赖PDDL domain中的语义定义
3. 如果PDDL定义了 `(not (= ?b1 ?b2))`，系统会自动添加约束 `?b1 ≠ ?b2`
4. 约束验证会自动拒绝违反约束的状态

### ✅ 4. PDDL Domain修复

**添加了缺失的不等式约束**：

```pddl
(:action put-on-block
  :parameters (?b1 ?b2 - block)
  :precondition (and
    (not (= ?b1 ?b2))  ← 新增：防止 on(a, a)
    (holding ?b1)
    (clear ?b2))
  :effect (and (on ?b1 ?b2) (handempty) (clear ?b1)
               (not (holding ?b1)) (not (clear ?b2)))
)

(:action put-tower-on-block
  :parameters (?b1 ?b2 ?b3 - block)
  :precondition (and
    (not (= ?b1 ?b2))  ← 新增
    (not (= ?b2 ?b3))  ← 新增
    (not (= ?b1 ?b3))  ← 新增
    (holding ?b2)
    (on ?b1 ?b2)
    (clear ?b3))
  :effect (and (on ?b2 ?b3) (handempty)
               (not (holding ?b2)) (not (clear ?b3)))
)
```

## 关键代码修改

### 1. `backward_search_refactored.py`

**变量生成（`_complete_binding`）**：
```python
# 从 parent_max_var + 1 开始生成
# 格式：?v0, ?v1, ?v2, ...
for param in parameters:
    if param not in complete_binding:
        new_var = f"?v{next_var_num}"
        # 确保不与已有变量冲突
        while new_var in used_vars:
            next_var_num += 1
            new_var = f"?v{next_var_num}"
        complete_binding[param] = new_var
        used_vars.add(new_var)
        next_var_num += 1
```

**约束验证（`_validate_constraints`）**：
```python
def _validate_constraints(self, predicates, constraints):
    for constraint in constraints:
        if constraint.var1 == constraint.var2:
            # 约束 ?v0 ≠ ?v0 永远为假
            return False
    return True
```

### 2. `backward_planner_generator.py`

**使用grounded goals搜索**：
```python
# ✓ CORRECT: 使用grounded predicates搜索
state_graph = planner.search(
    goal_predicates=list(goal_predicates),  # on(a, b)
    max_states=200000,
    max_objects=len(objects)
)

# 搜索后normalize用于caching
normalized_preds, var_mapping = self.normalizer.normalize_predicates(goal_predicates)
pattern_key = self.normalizer.serialize_goal(normalized_preds)
goal_cache[pattern_key] = (state_graph, var_mapping)
```

## 测试验证

```bash
$ python test_self_reference_fix.py

✓ All actions have proper inequality constraints
✓ Backward search with grounded goal: on(a, b)
✓ Explored 1,000 states
✓ Generated 14,904 unique states
✓ NO self-referencing predicates found
✓ TEST PASSED
```

## 关键Insights

1. **Grounded Search是正确的**：
   - 使用实际对象搜索更直观
   - 符合经典planning原理
   - 变量仅在必要时生成

2. **约束驱动验证**：
   - 不要硬编码语义规则
   - 依赖PDDL domain中的约束定义
   - 自动传播和检查约束

3. **Normalization延迟到搜索后**：
   - 搜索时用grounded predicates
   - 搜索完成后normalize
   - Normalization仅用于caching和codegen

4. **PDDL Domain的完整性很重要**：
   - 缺失约束会导致非法状态
   - 添加适当的 `(not (= ?x ?y))` 约束
   - 约束是语义的一部分

## 文件修改清单

- ✅ `src/domains/blocksworld/domain.pddl` - 添加不等式约束
- ✅ `src/stage3_code_generation/backward_search_refactored.py` - Grounded Search
- ✅ `src/stage3_code_generation/backward_planner_generator.py` - 使用grounded goals
- ✅ `test_self_reference_fix.py` - 验证修复效果

---

**最终结论**：通过采用Grounded Search策略、统一变量格式、基于约束的验证，以及修复PDDL Domain，成功解决了self-reference问题，并建立了更健壮的backward planning系统。
