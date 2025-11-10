# Constants vs Variables 区分逻辑分析

## 问题背景

用户质疑："请详解我们是怎么区分constant和variable的，你有仔细考虑过PDDL以及AgentSpeak的语法吗？"

用户举例：`move(a, -2, Dir)`

## 关键发现：实现中的严重缺陷

### 缺陷 1: 没有使用 `object_list` 的语义信息

**当前实现**（错误）：
```python
def _is_constant(self, arg: str) -> bool:
    # 基于字符串模式匹配
    if arg.startswith('?'):
        return True
    if is_number(arg):
        return True
    if arg.isupper() and arg.isalpha():
        return True  # ← 启发式规则！
    ...
    return False  # ← 完全没有检查 self.object_list！
```

**正确实现**应该是：
```python
def _should_abstract(self, arg: str) -> bool:
    """是否应该抽象此参数？"""
    # 核心判断：是否在 object_list 中
    if arg in self.object_list:
        return True  # 是问题实例中的对象 → 抽象
    else:
        return False  # 不是对象 → 保留（可能是domain constant或字面量）
```

### 缺陷 2: 混淆了PDDL和AgentSpeak的语法

#### PDDL 语法

**在 PDDL domain 定义中**：
```pddl
(:action move
  :parameters (?robot - robot ?distance - number ?direction - direction)
  :precondition ...
  :effect ...
)

(:constants LEFT RIGHT UP DOWN - direction)
```

- **Variables**: 以 `?` 开头（如 `?robot`, `?b1`）
- **Constants**: 在 `(:constants ...)` 中定义（如 `LEFT`, `RIGHT`）
- **Objects**: 在 problem 的 `(:objects ...)` 中定义（如 `a`, `b`, `robot1`）

**在 grounded predicate 中**：
- 不包含 `?` 变量
- 只包含 objects 和 constants 的具体值
- 例如：`move(robot1, -2, LEFT)` 或 `on(a, b)`

#### AgentSpeak 语法

**在 AgentSpeak 中**：
- **Variables**: **大写字母开头**（如 `X`, `Block`, `Dir`）
- **Constants/Atoms**: **小写字母开头**（如 `a`, `left`, `block1`）
- **Numbers**: 直接使用（如 `-2`, `3.14`）

### 缺陷 3: 用户例子的正确解读

用户说：`move(a, -2, Dir)`

**如果这是 PDDL grounded predicate**：
- `a`: 对象名（object）→ 应该抽象
- `-2`: 数字字面量（literal）→ 应该保留
- `Dir`: 对象名或常量名（看它在哪里定义）
  - 如果 `Dir` 在 problem 的 objects 中 → 应该抽象
  - 如果 `Dir` 在 domain 的 constants 中 → 应该保留

**如果这是 AgentSpeak syntax**：
- `a`: 常量（小写开头）
- `-2`: 数字
- `Dir`: **变量**（大写开头）！

### 缺陷 4: 当前实现的错误案例

#### 错误案例 1: 大写对象名被误判
```python
# 假设 objects = ["ROBOT1", "ROBOT2"]
pred = PredicateAtom("at", ["ROBOT1"])

# 当前实现（错误）：
_is_constant("ROBOT1")  # → True（因为全大写！）
# 结果：at(ROBOT1) → at(ROBOT1)  # 没有被抽象！

# 正确行为：
# ROBOT1 在 object_list 中 → 应该抽象
# 结果：at(ROBOT1) → at(?arg0)
```

#### 错误案例 2: 小写常量被误判
```python
# 假设 constants = ["left", "right"]，objects = ["a", "b"]
pred = PredicateAtom("move", ["a", "left"])

# 当前实现（错误）：
_is_constant("a")     # → False（小写且不在特殊模式中）
_is_constant("left")  # → False（小写且不在特殊模式中）
# 结果：move(a, left) → move(?arg0, ?arg1)  # 都被抽象了！

# 正确行为：
# "a" 在 object_list 中 → 抽象
# "left" 不在 object_list 中 → 保留
# 结果：move(a, left) → move(?arg0, left)
```

## 语义信息的缺失

### PDDLDomain 不包含 constants 信息

```python
@dataclass
class PDDLDomain:
    name: str
    types: List[str]
    predicates: List[PDDLPredicate]
    actions: List[PDDLAction]
    # ← 缺少 constants 字段！
```

**问题**：
- PDDL domain 可以定义 `(:constants ...)`
- 但我们的 parser 没有解析它
- 导致无法区分 domain constants 和 objects

## 当前实现能"侥幸工作"的原因

### 为什么测试通过了？

在 **blocksworld** domain 中：
1. **没有定义 constants**
2. **所有参数都是 objects**（类型为 block）
3. **objects 都是小写字母**（a, b, c）
4. **没有数字或特殊字面量**

所以当前的启发式规则"恰好"能工作：
- 小写字母 → 被判定为需要抽象（虽然逻辑错误，但结果对）
- 数字 → 被正确保留
- 大写 → 被保留（但blocksworld没有大写objects）

**但这完全依赖于命名约定的巧合，不是基于语义！**

## 正确的实现方案

### 方案 1: 基于 object_list（推荐）

```python
def _should_abstract(self, arg: str) -> bool:
    """
    判断参数是否应该被抽象为schema变量

    核心原则：只抽象 problem instance 中的 objects，
            保留 domain constants 和字面量

    Args:
        arg: 参数字符串

    Returns:
        True 如果应该抽象（是problem object）
        False 如果应该保留（是domain constant或字面量）
    """
    # 1. 如果已经是变量形式（?开头），保持不变
    if arg.startswith('?'):
        return False  # 已经是变量，不需要再抽象

    # 2. 核心判断：检查是否在 object_list 中
    if arg in self.object_list:
        return True  # 是problem中的对象 → 抽象

    # 3. 不在 object_list 中 → 是constant或literal → 保留
    return False
```

**优点**：
- ✅ 基于语义，不依赖命名约定
- ✅ 适用于任何命名风格（大写、小写、数字）
- ✅ 清晰的语义：objects 抽象，非objects 保留

**缺点**：
- ⚠️ 要求 object_list 必须完整准确
- ⚠️ 无法处理 object_list 中没有的对象

### 方案 2: 混合方法（更鲁棒）

```python
def _should_abstract(self, arg: str) -> bool:
    """判断参数是否应该被抽象"""
    # 1. 已经是变量 → 保持
    if arg.startswith('?'):
        return False

    # 2. 明确的字面量 → 保留
    if self._is_literal(arg):
        return False  # 数字、字符串字面量

    # 3. 在 object_list 中 → 抽象
    if arg in self.object_list:
        return True

    # 4. 不在 object_list 中的标识符 → 假设是domain constant → 保留
    return False

def _is_literal(self, arg: str) -> bool:
    """判断是否是字面量（数字、字符串）"""
    # 数字
    try:
        float(arg)
        return True
    except ValueError:
        pass

    # 字符串字面量（带引号）
    if (arg.startswith("'") and arg.endswith("'")) or \
       (arg.startswith('"') and arg.endswith('"')):
        return True

    # Boolean literals (如果支持)
    if arg.lower() in ['true', 'false']:
        return True

    return False
```

### 方案 3: 完整解析 PDDL constants（最准确）

**需要修改 PDDLParser**：
```python
@dataclass
class PDDLDomain:
    name: str
    types: List[str]
    constants: List[str]  # ← 新增
    predicates: List[PDDLPredicate]
    actions: List[PDDLAction]
```

然后：
```python
def _should_abstract(self, arg: str) -> bool:
    # 1. 在 domain.constants 中 → 保留
    if arg in self.domain.constants:
        return False

    # 2. 在 object_list 中 → 抽象
    if arg in self.object_list:
        return True

    # 3. 是字面量 → 保留
    if self._is_literal(arg):
        return False

    # 4. 默认：保留（保守策略）
    return False
```

## AgentSpeak 代码生成的考虑

在生成 AgentSpeak 代码时：
- PDDL variables (?x) → AgentSpeak variables (X)
- PDDL objects/constants → AgentSpeak atoms (小写)

当前的 `to_agentspeak()` 方法：
```python
def to_agentspeak(self) -> str:
    if not self.args:
        return self.name
    args_str = ", ".join(self.args)
    pred_str = f"{self.name}({args_str})"
    return f"~{pred_str}" if self.negated else pred_str
```

**问题**：没有转换变量格式！
- PDDL: `on(?arg0, ?arg1)`
- 应该生成 AgentSpeak: `on(Arg0, Arg1)` 或 `on(X, Y)`

但当前生成的是：`on(?arg0, ?arg1)` （保留了?）

这在 AgentSpeak 中是**不合法**的语法！

## 建议的修正步骤

### 1. 立即修正：使用 object_list

```python
def _should_abstract(self, arg: str) -> bool:
    """判断是否应该抽象此参数（重命名，语义更清晰）"""
    if arg.startswith('?'):
        return False  # 已经是变量

    if self._is_literal(arg):
        return False  # 字面量

    return arg in self.object_list  # 核心判断
```

### 2. 修正 normalize_predicates

```python
def normalize_predicates(self, predicates):
    obj_to_var = {}
    var_counter = 0

    # 第一遍：收集需要抽象的对象
    for pred in predicates:
        for arg in pred.args:
            if self._should_abstract(arg) and arg not in obj_to_var:
                obj_to_var[arg] = f"?arg{var_counter}"
                var_counter += 1

    # 第二遍：生成normalized predicates
    ...
```

### 3. 修正 AgentSpeak 代码生成

```python
def to_agentspeak(self, convert_vars=True) -> str:
    """
    转换为 AgentSpeak 格式

    Args:
        convert_vars: 是否转换 PDDL 变量为 AgentSpeak 变量
                      ?arg0 → Arg0 或 X
    """
    if convert_vars:
        args = [self._pddl_var_to_agentspeak(a) for a in self.args]
    else:
        args = self.args
    ...

def _pddl_var_to_agentspeak(self, arg: str) -> str:
    """PDDL变量转AgentSpeak变量：?arg0 → Arg0"""
    if arg.startswith('?'):
        var_name = arg[1:]  # 去掉 ?
        return var_name.capitalize()  # 首字母大写
    return arg
```

## 测试用例需要更新

当前测试只覆盖了 blocksworld 的简单情况。需要增加：

1. **大写对象名测试**：
   ```python
   objects = ["ROBOT1", "ROBOT2"]
   pred = PredicateAtom("at", ["ROBOT1"])
   # 应该：at(ROBOT1) → at(?arg0)
   ```

2. **小写常量测试**：
   ```python
   objects = ["a", "b"]
   constants = ["left", "right"]  # 需要支持
   pred = PredicateAtom("move", ["a", "left"])
   # 应该：move(a, left) → move(?arg0, left)
   ```

3. **混合命名测试**：
   ```python
   objects = ["robot_1", "Block2", "obj-A"]
   pred = PredicateAtom("holding", ["robot_1"])
   # 应该：holding(robot_1) → holding(?arg0)
   ```

## 结论

**当前实现的核心问题**：
1. ❌ 使用启发式字符串匹配，而非语义信息（object_list）
2. ❌ 术语混淆（_is_constant 实际表达 _should_preserve）
3. ❌ 对 ?开头参数的处理逻辑不清晰
4. ❌ 没有正确区分 PDDL 和 AgentSpeak 语法
5. ❌ PDDLParser 不解析 constants
6. ❌ AgentSpeak 代码生成不转换变量格式

**为什么测试通过了**：
- ✓ 侥幸！blocksworld 的命名约定恰好符合启发式规则
- ✓ 没有测试边界情况（大写对象、小写常量、混合命名）

**正确的做法**：
- ✅ 基于 `object_list` 进行语义判断
- ✅ 支持任意命名风格
- ✅ 区分 domain constants 和 problem objects
- ✅ 正确转换 PDDL ↔ AgentSpeak 变量格式
- ✅ 增强 PDDLParser 支持 constants
