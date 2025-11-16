# DFA Boolean Operators: Formal Methods for Simplification

## 问题定义

给定一个DFA，其transition labels包含复杂的boolean表达式：
- 例如：`on_d_e | (clear_c & on_a_b)`
- 目标：转换为等价的DFA，每个transition label只包含单个atom（atomic predicate）

## 理论基础

### 1. Symbolic Automata (符号自动机)

**定义：** Symbolic Finite Automaton (SFA) 是一个五元组 `(Q, Σ, Ψ, δ, F)`：
- `Q`: 有限状态集
- `Σ`: 可能无限的字母表（alphabet）
- `Ψ`: 有效布尔代数（effective Boolean algebra），定义在Σ上
- `δ: Q × Ψ → Q`: 转移函数（transition function）
- `F ⊆ Q`: 接受状态集

**关键特性：** Transition labels是**predicates（谓词）**而不是单个符号。

### 2. Atomic Predicates (原子谓词)

**定义：** 给定一个SFA中使用的所有predicates集合 `P = {φ₁, φ₂, ..., φₙ}`，
atomic predicates是对输入空间的**最小分割（minimal partition）**。

**构造方法：**

对于n个predicates，atomic predicates由所有可能的assignments构成：
```
AP = {α₁, α₂, ..., α₂ₙ}
```
其中每个 αᵢ 是形如 `±φ₁ ∧ ±φ₂ ∧ ... ∧ ±φₙ` 的合取式（conjunction），
`±φ` 表示 `φ` 或 `¬φ`。

**示例：**
- Predicates: `P = {on_a_b, clear_c, on_d_e}`
- Atomic predicates (8个)：
  1. `on_a_b & clear_c & on_d_e`
  2. `on_a_b & clear_c & ¬on_d_e`
  3. `on_a_b & ¬clear_c & on_d_e`
  4. `on_a_b & ¬clear_c & ¬on_d_e`
  5. `¬on_a_b & clear_c & on_d_e`
  6. `¬on_a_b & clear_c & ¬on_d_e`
  7. `¬on_a_b & ¬clear_c & on_d_e`
  8. `¬on_a_b & ¬clear_c & ¬on_d_e`

**性质：**
- 任何predicate φ ∈ Ψ 可以表示为atomic predicates的析取：`φ = ∨ᵢ αᵢ`
- Atomic predicates是**互斥的（mutually exclusive）**和**完备的（complete）**

### 3. Boolean Operators的转换方法

#### Method 1: Minterm Expansion (最小项展开)

**原理：** 任何boolean函数可以表示为minterms的析取。

**步骤：**
1. 识别所有atoms: `{p₁, p₂, ..., pₙ}`
2. 对于复杂表达式 `φ`，生成所有minterms
3. 评估每个minterm是否满足 `φ`
4. 收集满足的minterms

**示例：**
```
φ = on_d_e | (clear_c & on_a_b)

Minterms (3 atoms → 8 minterms):
m₁ = on_a_b & clear_c & on_d_e    → φ(m₁) = TRUE
m₂ = on_a_b & clear_c & ¬on_d_e   → φ(m₂) = TRUE
m₃ = on_a_b & ¬clear_c & on_d_e   → φ(m₃) = TRUE
m₄ = on_a_b & ¬clear_c & ¬on_d_e  → φ(m₄) = FALSE
m₅ = ¬on_a_b & clear_c & on_d_e   → φ(m₅) = TRUE
m₆ = ¬on_a_b & clear_c & ¬on_d_e  → φ(m₆) = FALSE
m₇ = ¬on_a_b & ¬clear_c & on_d_e  → φ(m₇) = TRUE
m₈ = ¬on_a_b & ¬clear_c & ¬on_d_e → φ(m₈) = FALSE

Result: φ = m₁ ∨ m₂ ∨ m₃ ∨ m₅ ∨ m₇
```

**DFA构造：**
- 对于transition `q₁ --[φ]--> q₂`
- 替换为多个transitions: `q₁ --[mᵢ]--> q₂` (对所有满足φ的mᵢ)

**问题：**
- 指数级别的transitions（2ⁿ个minterms）
- 每个minterm包含所有atoms的conjunction（不是atomic）

#### Method 2: BDD-based Shannon Expansion (Shannon展开)

**原理：** Shannon Expansion Theorem
```
f = (x ∧ f|ₓ₌₁) ∨ (¬x ∧ f|ₓ₌₀)
```

其中：
- `f|ₓ₌₁`: cofactor when x=1 (x为true时的简化)
- `f|ₓ₌₀`: cofactor when x=0 (x为false时的简化)

**BDD表示：**
```
     [x]
    /   \
  f|₁   f|₀
```
- 高分支（high branch）: x=true
- 低分支（low branch）: x=false

**DFA构造算法：**

```
Input: transition q₁ --[φ]--> q₂, BDD(φ)
Output: set of atomic transitions

Algorithm BuildAtomicDFA(current_state, target_state, bdd_node):
    if bdd_node == TRUE:
        return {current_state --[true]--> target_state}

    if bdd_node == FALSE:
        return ∅  // no transition

    // bdd_node tests variable x
    x = variable(bdd_node)
    high = high_branch(bdd_node)
    low = low_branch(bdd_node)

    transitions = ∅

    // Process high branch (x=true)
    if high != FALSE:
        state_high = new_state(high)
        transitions += {current_state --[x]--> state_high}
        transitions += BuildAtomicDFA(state_high, target_state, high)

    // Process low branch (x=false)
    if low != FALSE:
        state_low = new_state(low)
        transitions += {current_state --[¬x]--> state_low}
        transitions += BuildAtomicDFA(state_low, target_state, low)

    return transitions
```

**关键特性：**
- BDD node sharing: 相同的子表达式共享同一个node
- 保证等价性：BDD是canonical representation
- 每个transition检查一个atom（x或¬x）

**问题：**
- 仍然包含negations（¬x）
- 需要进一步转换为positive-only atoms

#### Method 3: Positive Atoms Only (仅正原子)

**挑战：** 如何表示negation without using `¬`？

**方法A：Product Construction (乘积构造)**

构造一个新的DFA，其状态空间是 `Q × 2^P`，其中：
- `Q`: 原DFA状态
- `2^P`: 所有atoms的子集（表示已确定为true的atoms）

**Transition规则：**
```
(q, V) --[p]--> (q', V ∪ {p})   if ∃ transition q --[φ]--> q' where φ[V∪{p}]=true
```

**方法B：Decision Tree Expansion (决策树展开)**

**原理：** 按固定顺序测试每个atom，构建完整的decision tree。

**步骤：**
1. 选择atom顺序：`p₁, p₂, ..., pₙ`
2. 对每个state，构建decision tree
3. 每层测试一个atom
4. 叶节点决定目标状态

**示例：** 对于 `φ = on_d_e | (clear_c & on_a_b)`

```
Decision tree (顺序: clear_c, on_a_b, on_d_e):

                [clear_c?]
               /          \
          YES /            \ NO
             /              \
        [on_a_b?]        [on_d_e?]
        /      \         /        \
    YES/    NO /     YES/      NO \
      /        \       /            \
  [on_d_e?] [on_d_e?] ACCEPT      REJECT
   /    \     /    \
 YES  NO  YES  NO
  |    |   |    |
 ACC  ACC ACC  REJ
```

**DFA构造：** 每个decision node成为一个state，每个分支成为一个transition。

**方法C：Disjunctive Normal Form (DNF) 展开**

**如果要求positive atoms only:**

**原理：** 对于包含negations的formula，使用DNF展开后选择性创建transitions。

**关键观察：**
- `¬x` 在DFA中可以表示为"implicit rejection"
- 只为positive atoms创建explicit transitions
- 缺少transition = 隐式reject

**示例：**
```
Original: q₁ --[on_d_e | (clear_c & on_a_b)]--> q₂

Positive-only transitions:
q₁ --[on_d_e]--> q₂
q₁ --[clear_c]--> q_temp
q_temp --[on_a_b]--> q₂
```

**解释：**
- 如果`on_d_e=true`：直接到q₂（第一个transition）
- 如果`on_d_e=false, clear_c=true`：到q_temp（第二个transition）
- 如果在q_temp且`on_a_b=true`：到q₂（第三个transition）

**注意：** 这实际上创建了一个**NFA (Non-deterministic)**结构！
- 从q₁有两个transitions：`on_d_e`和`clear_c`
- 如果输入同时满足两者，存在non-determinism

## 等价性保证

### Theorem 1: Minterm Expansion保持等价性

**证明：**
- 任何boolean函数f的minterm expansion是唯一的
- Minterms是互斥的
- 覆盖所有满足f的输入
- ∴ DFA接受相同的语言

### Theorem 2: BDD Construction保持等价性

**证明：**
- BDD是boolean函数的canonical representation
- Shannon Expansion是sound和complete的
- BDD node sharing保证structural sharing
- Traversal覆盖所有路径到terminal nodes
- ∴ 构造的DFA等价于原DFA

### Theorem 3: Positive-only Atoms需要特殊处理

**问题：**
- 简单的"只创建positive transitions"**不保证等价性**
- 可能遗漏某些accept paths

**正确方法：**
需要使用**Product Construction**或**完整Decision Tree**来保证：
1. Determinism（确定性）
2. Completeness（完备性）
3. Equivalence（等价性）

## 推荐方案

### 方案A：使用Atomic Predicates (带negations)

**优点：**
- 理论foundation坚实
- 保证等价性
- BDD-based实现高效

**缺点：**
- Transition labels包含negations（¬x）

**适用场景：** 当negations可接受时

### 方案B：使用Minterms (Positive-only via enumeration)

**方法：**
1. 生成所有atomic predicates（minterms）
2. 只保留完全positive的minterms
3. 为每个positive minterm创建transition

**示例：**
```
φ = on_d_e | (clear_c & on_a_b)

Positive minterms satisfying φ:
- on_a_b & clear_c & on_d_e
- on_a_b & clear_c  (when on_d_e可true可false → split)
```

**问题：** 指数级explosion

### 方案C：Hybrid BDD + Determinization

**步骤：**
1. 使用BDD Shannon Expansion构建初始DFA (with negations)
2. 对于每个negation transition `¬x`：
   - 使用subset construction
   - 或者使用decision tree完整展开
3. 确保determinism

**优点：** 理论正确，保证等价性

**缺点：** 复杂度高

## 结论

**当前实现的问题：**
1. ❌ 使用BDD traversal但没有正确处理shared nodes
2. ❌ 创建duplicate transitions导致non-determinism
3. ❌ Positive-only策略不完整，无法保证equivalence

**建议修复方案：**
1. **Option 1 (Simple):** 保留negations，修复BDD traversal logic
   - 使用`±atom`作为transition labels
   - 确保global node mapping
   - 保证determinism

2. **Option 2 (Complete):** 使用Decision Tree Construction
   - 构建完整的decision tree
   - 每层测试一个atom（按固定顺序）
   - Positive-only but deterministic and complete

3. **Option 3 (Theoretical):** 使用Product Construction
   - State space: `Q × 2^P`
   - 每个transition只测试一个positive atom
   - 保证等价性via product construction theorem

**推荐：** Start with Option 1（保留negations），验证等价性后再考虑Option 2/3。
