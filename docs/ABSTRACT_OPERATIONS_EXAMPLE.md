# 抽象操作示例：为什么当前实现仍不够抽象

## 场景

**初始状态：**
```
Stack 1: [a, b, c]  (c on b on a on table)
Stack 2: [d, e, f]  (f on e on d on table)
```

**目标：** `on(a, e)`

**需要做什么：**
1. 移除c (on c上的所有blocks) - 这里是空
2. 移除b (on b上的所有blocks) - 这里是c
3. 移除a上的阻碍（没有）
4. 移除e上的所有blocks - 这里是f
5. 把a放到e上

## 当前Lifted Planning的行为

### 问题：仍然枚举每个阻碍物

```python
# 当前实现会生成：
State0: {on(b,a), on(c,b), on(e,d), on(f,e), ...}

# 要clear(b)，会生成：
State0 --[pick-up(?V0, b)]-> State1  where ?V0=c
  # 内部：unify(on(?V0, b), on(c, b)) = {?V0/c}

# 要clear(e)，会生成：
State_n --[pick-up(?V1, e)]-> State_n+1  where ?V1=f
  # 内部：unify(on(?V1, e), on(f, e)) = {?V1/f}
```

**问题所在：**
- 虽然用了变量?V0, ?V1，但**仍然为每个阻碍物生成一个transition**
- 如果b上有10个blocks，会生成10个transitions
- State space仍然随着阻碍物数量增长！

### 更糟糕的场景：塔状结构

```
Initial: on(h, g), on(g, f), on(f, e), on(e, d), on(d, c), on(c, b), on(b, a)
Goal: on(a, table)
```

**当前实现：**
```python
# 需要移除所有7个blocks (h, g, f, e, d, c, b)
# 会生成：
State0 --[pick-up(?V0, g)]-> State1  where ?V0=h
State1 --[pick-up(?V1, f)]-> State2  where ?V1=g
State2 --[pick-up(?V2, e)]-> State3  where ?V2=f
State3 --[pick-up(?V3, d)]-> State4  where ?V3=e
State4 --[pick-up(?V4, c)]-> State5  where ?V4=d
State5 --[pick-up(?V5, b)]-> State6  where ?V5=c
State6 --[pick-up(?V6, a)]-> State7  where ?V6=b

# 7个states！每个阻碍物一个！
```

**State space仍然是 O(n)** where n = 阻碍物数量

## 真正的抽象应该做什么

### 期望行为：抽象宏操作

```python
# 应该是单个抽象操作：
State0 --[clear-tower(a)]-> State1
  # 内部语义: "移除a上所有blocks，递归处理"
  # 不枚举每个block
  # 不生成7个intermediate states
```

**State space: O(1)** - 独立于塔高度！

### 表示方法：Existential Quantification

```python
# 当前状态表示：
AbstractState({
    on(?V0, ?V1),  # 具体的某个on关系
    on(?V1, ?V2),
    on(?V2, ?V3),
    # ... 每个都具体化
})

# 期望的抽象表示：
AbstractState({
    on(?X, ?Y),          # 我们关心的主要关系
    exists(?Z): on(?Z, ?Y)  # ?Y上"存在"某些阻碍，但不枚举
})
```

### 抽象Action表示

```python
# 当前：只有primitive actions
pick-up(?X, ?Y): {
    precondition: on(?X, ?Y) ∧ clear(?X) ∧ handempty,
    effect: holding(?X) ∧ ¬on(?X, ?Y) ∧ ¬handempty ∧ clear(?Y)
}

# 期望：支持macro actions
clear-tower(?X): {
    precondition: True,
    abstract_effect: clear(?X),
    expansion: "∀?Z. on(?Z, ?X) → pick-up(?Z, ?X); clear-tower(?Z)",
    # 递归定义，不立即展开
}
```

## 具体例子对比

### 例子1：移除3个blocks

**Grounded Planning:**
```
States: 移除c, 移除b时的c位置, 移除a时的b和c位置, ...
Total states: ~hundreds (组合爆炸)
```

**Current Lifted Planning:**
```
States:
  S0 --[pick-up(?V0, b)]-> S1  (?V0=c)
  S1 --[pick-up(?V1, a)]-> S2  (?V1=b)
  S2 --[pick-up(?V2, e)]-> S3  (?V2=f)
Total abstract states: 3 (每个阻碍物一个)
```

**True Abstract Planning (期望):**
```
States:
  S0 --[clear-block(b)]-> S1  (抽象操作)
  S1 --[clear-block(e)]-> S2
  S2 --[put-on(a, e)]-> S3
Total abstract states: 3 (每个高层操作一个)
```

看起来数量一样？**关键区别：**
- Current: 3 states因为有3个**具体**blocks需要移除
- Expected: 3 states因为有3个**抽象**操作（clear两个blocks + 最终动作）
- **如果有100个阻碍物：**
  - Current: 100个states
  - Expected: **仍然3个states**（两个clear + 一个put）

## 为什么这很重要

### 可扩展性

```
场景：超高塔 (100 blocks stacked)
Goal: on(block_0, table)

Current Lifted:
  - 100个states (每个block一个)
  - 100个transitions
  - State space: O(n)

True Abstract:
  - 1个state: clear-tower(block_0) -> goal
  - 1个transition
  - State space: O(1)
```

### 真正独立于Objects数量

```
Current Lifted:
  - 3 objects, 3 blockers → 3 states
  - 10 objects, 10 blockers → 10 states
  - 仍然随着具体objects增长！

True Abstract:
  - 任意objects, 任意blockers → 1 state (clear-tower)
  - 真正独立！
```

## 实现路径

### 短期：Existential Variables

```python
class AbstractState:
    predicates: FrozenSet[PredicateAtom]
    constraints: ConstraintSet
    existentials: Set[ExistentialFormula]  # NEW

class ExistentialFormula:
    var: str  # e.g., "?Z"
    formula: PredicateAtom  # e.g., on(?Z, ?Y)
    # 表示: ∃?Z. on(?Z, ?Y)，不具体化?Z
```

### 中期：Macro Actions

```python
class MacroAction:
    name: str
    params: List[str]
    abstract_precondition: Formula
    abstract_effect: Formula
    expansion: Optional[Callable]  # 延迟展开

# 定义
clear_block = MacroAction(
    name="clear-block",
    params=["?X"],
    abstract_effect=PredicateAtom("clear", ["?X"]),
    expansion=lambda x: recursive_clear(x)
)
```

### 长期：Hierarchical Planning

```python
# 多层抽象
Level 2 (High): achieve(on(a, e))
  └─> Level 1 (Macro): clear-block(e), move-block(a, e)
      └─> Level 0 (Primitive): pick-up(f, e), put-down(f), ...
```

## 总结

**当前实现的成就：**
- ✅ 使用unification而不是itertools.product
- ✅ State数量不随domain objects总数增长
- ✅ 支持任意数量的参数

**仍然缺少：**
- ❌ Existential quantification - 不枚举阻碍物
- ❌ Macro operations - 抽象高层操作
- ❌ State数量仍随**relevant阻碍物**数量增长
- ❌ 不是真正的"抽象操作"，只是"变量化的具体操作"

**下一步目标：**
1. 实现existential variables
2. 定义常用macro actions (clear-block等)
3. 修改exploration不为每个existential binding生成state
4. 真正实现O(1) state space for clearing operations
