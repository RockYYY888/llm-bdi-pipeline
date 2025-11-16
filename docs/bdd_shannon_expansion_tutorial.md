# BDD Shannon Expansion 详解

## 目录
1. Shannon Expansion基本原理
2. BDD (Binary Decision Diagram) 简介
3. 从Boolean Formula到BDD
4. 如何用BDD构建DFA
5. 实例演示

---

## 1. Shannon Expansion 基本原理

### 1.1 Shannon Expansion Theorem (布尔展开定理)

**定理：** 任何布尔函数 f 都可以关于变量 x 展开为：

```
f = (x ∧ f|ₓ₌₁) ∨ (¬x ∧ f|ₓ₌₀)
```

**符号说明：**
- `f|ₓ₌₁`: **cofactor** (余因子) - 当x=1时，f的简化形式
- `f|ₓ₌₀`: **cofactor** - 当x=0时，f的简化形式

**直观理解：**
- 如果 x=true，则 f 等于 `f|ₓ₌₁`
- 如果 x=false，则 f 等于 `f|ₓ₌₀`

### 1.2 简单例子

**例1：** `f = x ∨ y`

关于变量 x 展开：
```
f|ₓ₌₁ = 1 ∨ y = 1          (x=1时，整个式子为true)
f|ₓ₌₀ = 0 ∨ y = y          (x=0时，f取决于y)

∴ f = (x ∧ 1) ∨ (¬x ∧ y)
    = x ∨ (¬x ∧ y)
    = x ∨ y  ✓ (正确!)
```

**例2：** `f = x ∧ y`

关于变量 x 展开：
```
f|ₓ₌₁ = 1 ∧ y = y          (x=1时，f取决于y)
f|ₓ₌₀ = 0 ∧ y = 0          (x=0时，整个式子为false)

∴ f = (x ∧ y) ∨ (¬x ∧ 0)
    = x ∧ y  ✓ (正确!)
```

### 1.3 递归应用

Shannon Expansion可以**递归应用**到所有变量：

**例3：** `f = (a ∧ b) ∨ c`

**Step 1:** 关于变量 a 展开
```
f|ₐ₌₁ = (1 ∧ b) ∨ c = b ∨ c
f|ₐ₌₀ = (0 ∧ b) ∨ c = c

f = (a ∧ (b ∨ c)) ∨ (¬a ∧ c)
```

**Step 2:** 继续展开 `f|ₐ₌₁ = b ∨ c` 关于变量 b
```
(b ∨ c)|ᵦ₌₁ = 1 ∨ c = 1
(b ∨ c)|ᵦ₌₀ = 0 ∨ c = c

b ∨ c = (b ∧ 1) ∨ (¬b ∧ c)
```

**Step 3:** 继续展开 `c` 关于变量 c
```
c|꜀₌₁ = 1
c|꜀₌₀ = 0

c = (c ∧ 1) ∨ (¬c ∧ 0) = c
```

**最终结果：** 得到一个完整的decision tree!

---

## 2. BDD (Binary Decision Diagram) 简介

### 2.1 什么是BDD？

**BDD** 是布尔函数的**图形表示（graphical representation）**：
- 每个**内部节点（internal node）** 代表一个变量
- 每个节点有两条边：
  - **High edge (实线)**: 变量=1时走的路径
  - **Low edge (虚线)**: 变量=0时走的路径
- 两个**终端节点（terminal nodes）**：
  - **1-terminal**: 表示TRUE
  - **0-terminal**: 表示FALSE

### 2.2 BDD的图形表示

**例：** `f = (a ∧ b) ∨ c`

```
         [a]
        /   \
       /     \
      /       \
   HIGH       LOW
   (a=1)     (a=0)
    /           \
 [b]            [c]
 / \            / \
H   L          H   L
|   |          |   |
[c] [c]       @1  @0
/ \ / \
H  L H  L
|  | |  |
@1 @1 @1 @0

说明：
- @1 = TRUE terminal
- @0 = FALSE terminal
- H = High branch (变量=true)
- L = Low branch (变量=false)
```

更清晰的表示：
```
              [a]
             /   \
          H /     \ L
           /       \
        [b]        [c]
        / \        / \
     H /   \ L  H /   \ L
      /     \   /     \
    [c]    [c] @1     @0
    / \    / \
 H /   \L H/  \L
  /     \ /    \
 @1     @1     @0
```

### 2.3 BDD的关键特性

#### 特性1：Canonical Representation (规范表示)

**重要！** 对于给定的：
- 变量顺序（variable ordering）
- 布尔函数 f

BDD是**唯一的（unique）**！

这意味着：
- 两个不同的布尔表达式，如果等价，会得到**相同的BDD**
- 可以用BDD来判断两个表达式是否等价

**例：**
```
f₁ = (a ∧ b) ∨ (a ∧ c)
f₂ = a ∧ (b ∨ c)

这两个表达式逻辑等价，会产生相同的BDD！
```

#### 特性2：Node Sharing (节点共享)

BDD会自动**共享相同的子图**：

**例：** `f = (a ∧ b) ∨ (¬a ∧ b)`

```
        [a]
       /   \
      /     \
    [b]     [b]  ← 这两个[b]节点可以合并!
    / \     / \
   1   0   1   0
```

优化后（with sharing）：
```
        [a]
       /   \
      /     \___
     /           \
    [b] ←────────┘  (共享同一个[b]节点)
    / \
   @1  @0
```

**这个特性非常重要！** 它会导致我们当前实现的bug。

---

## 3. 从Boolean Formula到BDD

### 3.1 构建过程

**输入：** 布尔表达式 `f = on_d_e | (clear_c & on_a_b)`

**变量：** `{clear_c, on_a_b, on_d_e}`

**变量顺序：** `clear_c < on_a_b < on_d_e` (字母序)

### 3.2 Step-by-step构建

**Step 1:** 最顶层变量是 `clear_c`

应用Shannon Expansion：
```
f = (clear_c ∧ f|꜀₌₁) ∨ (¬clear_c ∧ f|꜀₌₀)
```

计算cofactors：
```
f|clear_c=1 = on_d_e | (1 & on_a_b) = on_d_e | on_a_b
f|clear_c=0 = on_d_e | (0 & on_a_b) = on_d_e
```

**Step 2:** 处理 high branch: `on_d_e | on_a_b`

下一个变量是 `on_a_b`：
```
(on_d_e | on_a_b)|on_a_b=1 = on_d_e | 1 = 1
(on_d_e | on_a_b)|on_a_b=0 = on_d_e | 0 = on_d_e
```

**Step 3:** 处理 low branch: `on_d_e`

下一个变量是 `on_a_b`（即使不在表达式中也要测试）：
```
on_d_e|on_a_b=1 = on_d_e
on_d_e|on_a_b=0 = on_d_e
```

两个cofactor相同！说明 `on_a_b` 对结果无影响，可以跳过。

直接到下一个变量 `on_d_e`：
```
on_d_e|on_d_e=1 = 1
on_d_e|on_d_e=0 = 0
```

### 3.3 最终BDD结构

```
                [clear_c]
               /         \
             H /           \ L
              /             \
         [on_a_b]        [on_d_e] ←┐
          /     \          /    \   │
        H/      L\       H/     L\  │
        /         \      /        \ │
      @1      [on_d_e]←─┘         @0
               /    \
             H/     L\
             /        \
           @1         @0
```

**注意节点共享：**
- `[on_d_e]` 节点出现了两次，但实际上是**同一个节点**！
- Low branch of `clear_c` 直接指向这个共享的 `[on_d_e]` 节点
- Low branch of `on_a_b` 也指向同一个 `[on_d_e]` 节点

**这就是我们当前bug的根源！**

---

## 4. 如何用BDD构建DFA

### 4.1 基本思想

**核心思路：**
- BDD的每个**内部节点** → DFA的一个**状态（state）**
- BDD的**边（edge）** → DFA的**转移（transition）**
  - High edge → transition labeled with **positive atom**
  - Low edge → transition labeled with **negated atom**

### 4.2 算法伪代码

```python
def BDD_to_DFA(bdd_node, current_dfa_state, target_dfa_state):
    """
    将BDD转换为DFA transitions

    参数:
        bdd_node: 当前BDD节点
        current_dfa_state: 当前DFA状态
        target_dfa_state: 目标DFA状态
    """
    # Terminal cases
    if bdd_node == TRUE:
        return [(current_dfa_state, target_dfa_state, "true")]

    if bdd_node == FALSE:
        return []  # no transition

    # Get variable at this node
    var = variable(bdd_node)

    # Get branches
    high_child = high_branch(bdd_node)
    low_child = low_branch(bdd_node)

    transitions = []

    # Process HIGH branch (var=true)
    if high_child != FALSE:
        high_state = get_or_create_state(high_child)
        transitions.append((current_dfa_state, high_state, var))
        transitions += BDD_to_DFA(high_child, high_state, target_dfa_state)

    # Process LOW branch (var=false)
    if low_child != FALSE:
        low_state = get_or_create_state(low_child)
        transitions.append((current_dfa_state, low_state, f"!{var}"))
        transitions += BDD_to_DFA(low_child, low_state, target_dfa_state)

    return transitions
```

### 4.3 关键：State Mapping

**问题：** 如何为BDD节点分配DFA状态名？

**方法1（错误）：** 每次遇到BDD节点时创建新状态
```python
state_counter = 0
def create_state():
    state_counter += 1
    return f"s{state_counter}"
```
❌ **问题：** 共享的BDD节点会被创建多次！

**方法2（正确）：** 使用node ID作为key，全局映射
```python
state_map = {}  # node_id -> state_name

def get_or_create_state(bdd_node):
    node_id = id(bdd_node)
    if node_id not in state_map:
        state_map[node_id] = f"s{len(state_map)}"
    return state_map[node_id]
```
✓ **优点：** 共享的BDD节点会映射到同一个DFA状态

---

## 5. 实例演示

### 5.1 完整例子

**原始DFA transition:**
```
state_1 --[on_d_e | (clear_c & on_a_b)]--> state_2
```

**目标：** 转换为atomic transitions

### 5.2 Step-by-Step转换

**Step 1: 构建BDD**

BDD结构（带node IDs）：
```
                [clear_c]  (id=100)
               /         \
             H /           \ L
              /             \
         [on_a_b]        [on_d_e] ←─┐
         (id=200)        (id=300)   │
          /     \          /    \   │
        H/      L\       H/     L\  │
        /         \      /        \ │
      @1      [on_d_e]←─┘         @0
             (same: id=300)
               /    \
             H/     L\
             /        \
           @1         @0
```

**Step 2: 遍历BDD，创建state mapping**

```python
# 遍历开始
current_state = "1"  (原DFA的state_1)
target_state = "2"   (原DFA的state_2)
bdd_root = node(id=100)  ([clear_c])

# 处理 [clear_c] node
state_map[100] = "1"  (root对应原DFA的state_1)

# HIGH branch: clear_c = true
high_child = node(id=200)  ([on_a_b])
state_map[200] = "s1"  (新创建)
transitions += [("1", "s1", "clear_c")]

# LOW branch: clear_c = false
low_child = node(id=300)  ([on_d_e])
state_map[300] = "s2"  (新创建)
transitions += [("1", "s2", "!clear_c")]
```

**Step 3: 递归处理 s1 ([on_a_b] node)**

```python
# 从s1出发，处理[on_a_b]节点
current = "s1"
bdd_node = node(id=200)

# HIGH branch: on_a_b = true
high_child = TRUE
transitions += [("s1", "2", "on_a_b")]  # 直接到target

# LOW branch: on_a_b = false
low_child = node(id=300)  ([on_d_e])
# ⚠️ 注意：这个node已经在state_map中！
state_map[300] = "s2"  (已存在，复用)
transitions += [("s1", "s2", "!on_a_b")]
```

**Step 4: 处理 s2 ([on_d_e] node)**

```python
# ⚠️ 关键：这个节点会被处理两次！
# 第一次：来自clear_c的LOW branch
# 第二次：来自on_a_b的LOW branch

current = "s2"
bdd_node = node(id=300)

# HIGH branch: on_d_e = true
transitions += [("s2", "2", "on_d_e")]

# LOW branch: on_d_e = false
transitions += []  # FALSE，no transition
```

**Step 5: 最终结果**

```
Transitions:
1. ("1", "s1", "clear_c")
2. ("1", "s2", "!clear_c")
3. ("s1", "2", "on_a_b")
4. ("s1", "s2", "!on_a_b")
5. ("s2", "2", "on_d_e")
```

**DFA可视化：**
```
      1
     / \
    /   \
  clear_c  !clear_c
  /         \
 s1          s2
 / \          |
on_a_b !on_a_b  on_d_e
|      |      |
2      s2     2
```

### 5.3 验证正确性

测试几个输入：

**输入1: {clear_c, on_a_b}**
```
1 --[clear_c]--> s1 --[on_a_b]--> 2  ✓ Accept
```

**输入2: {on_d_e}**
```
1 --[!clear_c]--> s2 --[on_d_e]--> 2  ✓ Accept
```

**输入3: {clear_c}** (没有on_a_b)
```
1 --[clear_c]--> s1 --[!on_a_b]--> s2 (没有on_d_e)
s2 没有到2的transition  ✗ Reject
```

**输入4: {clear_c, on_a_b, on_d_e}**
```
1 --[clear_c]--> s1 --[on_a_b]--> 2  ✓ Accept
```

---

## 6. 当前实现的Bug

### 6.1 Bug原因

**问题代码：**
```python
def _bdd_to_atomic_transitions(self, ...):
    # ❌ 错误：每次调用都重置！
    self.state_map = {}
    self.state_counter = 0

    trans, acc = self._traverse_bdd(...)
    return trans, acc
```

**后果：**
- 每处理一个原DFA transition，都重置state_map
- 如果原DFA有多个transitions，它们会创建重复的状态名
- 例如：第一个transition创建"s1"，第二个transition也创建"s1"（但是不同的BDD node）

### 6.2 正确做法

**修复：** 使用全局的state_map
```python
def simplify(self, ...):
    # ✓ 正确：全局初始化一次
    self.state_map = {}
    self.state_counter = 0

    for transition in all_transitions:
        trans, acc = self._bdd_to_atomic_transitions(...)
        # 不重置state_map！
```

### 6.3 另一个Bug：Processed Nodes

**问题：** 共享的BDD节点被遍历多次，创建duplicate transitions

**当前代码：**
```python
def _traverse_bdd(self, current_state, target_state, bdd_node):
    # 检查是否处理过
    key = (id(bdd_node), target_state)
    if key in processed_nodes:
        return []  # 已处理，跳过
```

**问题分析：**
- 这个检查基于 `(node_id, target_state)`
- 但是同一个BDD node可以从**不同的current_state**到达
- 例如：
  ```
  s1 --[!on_a_b]--> s2 (node_300)
  从这里处理node_300，创建: s2 --[on_d_e]--> 2

  后来：
  1 --[!clear_c]--> s2 (same node_300)
  再次处理node_300，又创建: s2 --[on_d_e]--> 2

  结果：duplicate transition!
  ```

**正确做法：**
- 每个BDD node只处理一次
- 使用 `state_map[node_id]` 来复用已创建的DFA state
- 不重复创建transitions

---

## 总结

**BDD Shannon Expansion的核心思想：**

1. **Shannon Expansion**: 任何布尔函数可以关于某个变量x分解为两部分
   - x=true时的情况
   - x=false时的情况

2. **BDD**: 用图形表示Shannon Expansion的递归应用
   - 每个节点测试一个变量
   - High/Low边表示变量的true/false分支
   - **Node sharing**: 相同的子表达式共享同一个节点

3. **DFA构造**:
   - BDD节点 → DFA状态
   - BDD边 → DFA转移
   - **关键**: 使用全局state_map确保共享的BDD节点映射到同一个DFA状态

4. **等价性保证**:
   - BDD是c
   - 完整遍历BDD覆盖所有paths
   - ∴ 构造的DFA等价于原表达式

**下一步：修复代码**
- 移除state_map的重置
- 修复processed_nodes逻辑
- 确保每个BDD node只创建一次DFA transitions
