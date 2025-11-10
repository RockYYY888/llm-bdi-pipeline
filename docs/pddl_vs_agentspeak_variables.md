# PDDL vs AgentSpeak 变量概念澄清

## 核心混淆点

用户指出：
> "AgentSpeak的语法里面，第一个Char大写的应该都是变量，但是此变量不是彼变量，我们的objects是objects，而agentspeak的syntax支持赋值之类的，但是赋值不一定是一个object。"

这揭示了三个不同的"变量"概念需要区分：
1. **PDDL Variables** (规划变量)
2. **PDDL Objects** (问题对象)
3. **AgentSpeak Variables** (逻辑变量)

---

## 三种"变量"的区别

### 1. PDDL Variables (规划变量)

**语法**: 以 `?` 开头

**作用**: 在 action schemas 和 domain 定义中表示参数占位符

**示例**:
```pddl
(:action pick-up
  :parameters (?b1 ?b2 - block)  ; ← PDDL variables
  :precondition (and (on ?b1 ?b2) (clear ?b1))
  :effect (holding ?b1)
)
```

**特点**:
- 定义在 domain level
- 通过 grounding 绑定到具体 objects
- 不出现在 grounded predicates 中

---

### 2. PDDL Objects (问题对象)

**语法**: 普通标识符（无特殊前缀）

**作用**: Problem instance 中的具体实体

**示例**:
```pddl
(:objects a b c - block)  ; ← PDDL objects
```

**Grounded predicates**:
```
on(a, b)        ; a 和 b 是 objects
clear(c)        ; c 是 object
```

**特点**:
- 定义在 problem level
- 是具体的实体，不是占位符
- **这是我们要抽象的目标**

---

### 3. AgentSpeak Variables (逻辑变量)

**语法**: **首字母大写**

**作用**: 在 AgentSpeak 中用于模式匹配、统一(unification)、赋值

**示例**:
```asl
// AgentSpeak 代码
+on(X, Y) : clear(X)  // X, Y 是 AgentSpeak variables
  <- pick_up(X, Y).

+!stack(Block1, Block2)  // Block1, Block2 是 AgentSpeak variables
  <- put_on(Block1, Block2).
```

**特点**:
- **首字母大写**是 AgentSpeak 语法要求
- 用于模式匹配和统一
- 可以在运行时绑定到具体值
- **不一定对应 PDDL objects**

---

## 当前实现的情况

### 我们在做什么？

我们在 Stage 3 中：
1. 接收 **grounded PDDL predicates**（包含 PDDL objects）
2. 将 PDDL objects 抽象为 **schema variables**（形如 `?arg0`, `?arg1`）
3. 最终生成 **AgentSpeak 代码**

### 当前的变量格式

**Variable Normalization 使用**:
```
on(a, b)  →  on(?arg0, ?arg1)  ; 使用 PDDL variable syntax (?开头)
```

**但这是中间表示**，不是最终的 AgentSpeak 代码！

### 问题：AgentSpeak 代码生成

**当前代码生成**:
```python
# agentspeak_codegen.py
def to_agentspeak(self) -> str:
    if not self.args:
        return self.name
    args_str = ", ".join(self.args)  # ← 直接使用参数
    pred_str = f"{self.name}({args_str})"
    return f"~{pred_str}" if self.negated else pred_str
```

**生成的代码**:
```asl
on(?arg0, ?arg1)  ; ← 错误！AgentSpeak 不支持 ? 前缀
```

**应该生成**:
```asl
on(Arg0, Arg1)    ; ← 正确：首字母大写，无 ? 前缀
// 或者
on(X, Y)          ; ← 更简洁的命名
```

---

## 概念映射表

| 概念 | PDDL Domain | PDDL Problem | Our Schema | AgentSpeak Code |
|------|------------|--------------|------------|-----------------|
| **Action Parameters** | `?b1, ?b2` | - | - | `Block1, Block2` |
| **Problem Objects** | - | `a, b, c` | `?arg0, ?arg1` | `a, b, c` (atoms) |
| **Schema Variables** | - | - | `?arg0, ?arg1` | `Arg0, Arg1` (vars) |
| **Logic Variables** | - | - | - | `X, Y, Block` (vars) |

---

## 正确的转换流程

### Step 1: PDDL Objects → Schema Variables (Normalization)

```python
# Input: grounded PDDL predicates
on(a, b)
clear(a)

# Normalize (abstract objects → schema variables)
# 使用 PDDL 风格 (? 前缀) 作为中间表示
on(?arg0, ?arg1)
clear(?arg0)
```

**用途**:
- Schema-level caching
- Variable-level planning
- **仅作为中间表示**

### Step 2: Schema Variables → AgentSpeak Variables (Code Generation)

```python
# Input: schema predicates with PDDL-style variables
on(?arg0, ?arg1)
clear(?arg0)

# Convert to AgentSpeak syntax (capitalize, remove ?)
on(Arg0, Arg1)
clear(Arg0)
```

**转换规则**:
```
?arg0  →  Arg0
?arg1  →  Arg1
?v0    →  V0
?x     →  X
```

### Step 3: 实例化 (Instantiation)

当需要具体对象时：
```python
# Mapping: {?arg0: a, ?arg1: b}

# AgentSpeak code with concrete objects (atoms)
on(a, b)    ; a, b 是 atoms (小写)
clear(a)

# 或者保持变量形式（用于模式匹配）
on(Arg0, Arg1)  ; Arg0, Arg1 是 variables (大写)
clear(Arg0)
```

---

## AgentSpeak 变量的两种用途

### 用途 1: 模式匹配和统一

```asl
// Plan template - 使用变量进行匹配
+!achieve(on(X, Y)) : ontable(X) & clear(Y)
  <- pick_up(X);
     put_on(X, Y).

// 运行时: X=a, Y=b (通过统一绑定)
```

**变量 X, Y**:
- 是AgentSpeak 逻辑变量
- **不是** PDDL objects
- 可以绑定到任何符合条件的值

### 用途 2: 参数传递

```asl
+!move_robot(Robot, Distance, Direction)  // 参数
  <- move(Robot, Distance, Direction).

// 调用: !move_robot(robot1, -2, left)
// Robot=robot1, Distance=-2, Direction=left
```

**变量 Robot, Distance, Direction**:
- 是形式参数
- 运行时绑定到实际值
- 实际值可能是 objects (robot1)，也可能是 literals (-2, left)

---

## 我们的生成策略

### 当前问题

生成的代码使用 PDDL-style variables (`?arg0`)，但 AgentSpeak 不支持。

### 解决方案

**需要实现**: PDDL variable → AgentSpeak variable 转换

**位置**: `agentspeak_codegen.py`

**转换函数**:
```python
def _pddl_var_to_agentspeak(self, arg: str) -> str:
    """
    Convert PDDL-style variable to AgentSpeak-style variable

    Examples:
        ?arg0 → Arg0
        ?arg1 → Arg1
        ?v0   → V0
        ?x    → X
        a     → a (atoms unchanged)
    """
    if arg.startswith('?'):
        # Remove ? and capitalize first letter
        var_name = arg[1:]  # Remove ?
        return var_name[0].upper() + var_name[1:]  # Capitalize
    return arg  # Not a variable, return as-is
```

**使用**:
```python
def to_agentspeak(self, convert_vars=True) -> str:
    """Generate AgentSpeak code"""
    if convert_vars:
        # Convert PDDL variables to AgentSpeak variables
        args = [self._pddl_var_to_agentspeak(a) for a in self.args]
    else:
        args = self.args

    args_str = ", ".join(args)
    pred_str = f"{self.name}({args_str})"
    return f"~{pred_str}" if self.negated else pred_str
```

**效果**:
```python
pred = PredicateAtom("on", ["?arg0", "?arg1"])
pred.to_agentspeak()
# Output: "on(Arg0, Arg1)"  ✓ 正确的 AgentSpeak 语法
```

---

## 完整示例

### 输入 (Natural Language)

```
"Stack block a on block b"
```

### Stage 1: LTL Parsing

```json
{
  "objects": ["a", "b"],
  "formulas": ["F(on_a_b)"]
}
```

### Stage 2: DFA Generation

```
state0 --[on_a_b]--> state1
```

### Stage 3: Backward Planning

#### 3.1 Goal Extraction & Normalization

```python
# Grounded goal
goal = [PredicateAtom("on", ["a", "b"])]

# Normalize (abstract objects)
normalized = [PredicateAtom("on", ["?arg0", "?arg1"])]
mapping = {"a": "?arg0", "b": "?arg1"}
```

#### 3.2 Variable-Level Planning

```python
# Plan with schema variables (PDDL-style)
# State graph uses ?arg0, ?arg1, etc.
```

#### 3.3 AgentSpeak Code Generation

**错误方式** (当前):
```asl
+!achieve_on_a_b
  <- ?pick_up(?arg0, ?arg1).  // ❌ 语法错误!
```

**正确方式** (应该生成):
```asl
// Option 1: 使用具体对象 (instantiated)
+!achieve_on_a_b
  <- pick_up(a, b).  // ✓ a, b 是 atoms

// Option 2: 使用变量 (parameterized)
+!achieve_on(Arg0, Arg1)
  <- pick_up(Arg0, Arg1).  // ✓ Arg0, Arg1 是 variables
```

---

## 总结

### 三种"变量"

1. **PDDL Variables** (`?x`): Domain-level 占位符
2. **PDDL Objects** (`a`): Problem-level 具体实体
3. **AgentSpeak Variables** (`X`): 逻辑变量，用于匹配和统一

### 我们的抽象

- 抽象的是 **PDDL Objects** → Schema variables
- Schema variables 使用 PDDL syntax (`?arg0`) 作为**中间表示**
- 最终生成 AgentSpeak 时，需要转换为 AgentSpeak syntax (`Arg0`)

### 需要的修复

**当前**: `on(?arg0, ?arg1)` (PDDL-style, AgentSpeak 不支持)
**应该**: `on(Arg0, Arg1)` (AgentSpeak-style, 首字母大写)

### 实现位置

`src/stage3_code_generation/agentspeak_codegen.py`:
- 添加 `_pddl_var_to_agentspeak()` 方法
- 修改 `to_agentspeak()` 进行变量转换

---

**文档创建**: 2025-11-10
**关键洞察**: AgentSpeak variables (大写) ≠ PDDL objects
