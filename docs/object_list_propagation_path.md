# object_list 传递路径文档

## 核心原则

**object_list 是唯一可靠来源** - 用于区分 objects 和 constants

严格检查：
- ✅ 在 object_list 中 → 是 problem object → 抽象为 schema variable
- ✅ 不在 object_list 中 → 不是 object → 保留原样

**不使用任何启发式规则**（大小写、命名约定等）

---

## 完整传递路径

### Stage 1: 自然语言 → LTLf (LLM 生成)

**文件**: `src/stage1_interpretation/ltlf_generator.py`

**关键代码** (Line 152):
```python
# Build LTL specification
spec = LTLSpecification()
spec.objects = result["objects"]  # ← 从 LLM JSON 输出提取
```

**来源**: LLM 分析自然语言，识别并提取对象列表

**示例**:
```json
{
  "objects": ["a", "b", "c"],
  "formulas": [...]
}
```

---

### Stage 1.5: LTLSpecification 数据结构

**文件**: `src/stage1_interpretation/ltlf_formula.py`

**类定义** (Line 212-230):
```python
class LTLSpecification:
    def __init__(self):
        self.formulas: List[LTLFormula] = []
        self.objects: List[str] = []  # ← 对象列表存储在这里
        self.initial_state: List[Dict[str, List[str]]] = []
        self.grounding_map: Optional['GroundingMap'] = None
```

---

### Stage 2: 管道传递

**文件**: `src/ltl_bdi_pipeline.py`

**传递给 Stage 3** (Line 218-224):
```python
# Convert ltl_spec to dict format
ltl_dict = {
    "objects": ltl_spec.objects,  # ← 从 LTLSpecification 提取
    "formulas_string": [f.to_string() for f in ltl_spec.formulas]
}
```

---

### Stage 3: Backward Planning (代码生成)

#### 3.1 接收 objects

**文件**: `src/stage3_code_generation/backward_planner_generator.py`

**接收** (Line 96):
```python
def generate(self, ltl_dict: Dict[str, Any], dfa_result: Dict[str, Any]) -> str:
    # Extract objects
    objects = ltl_dict['objects']  # ← 从 ltl_dict 提取
    print(f"Objects: {objects}")
```

#### 3.2 创建 VariableNormalizer

**初始化 Normalizer** (Line 105):
```python
# VARIABLE ABSTRACTION: Create variable normalizer
# This enables variable-level caching instead of object-level caching
normalizer = VariableNormalizer(self.domain, objects)  # ← 传入 objects
print(f"Variable abstraction enabled: using {len(objects)} variables")
```

#### 3.3 使用 object_list 进行判断

**文件**: `src/stage3_code_generation/variable_normalizer.py`

**初始化** (Line 65-74):
```python
def __init__(self, domain: PDDLDomain, objects: List[str]):
    self.domain = domain
    self.object_list = sorted(objects)  # ← 存储为 object_list
    self.object_types = self._infer_object_types()
```

**核心判断逻辑** (Line 138-141):
```python
# 4. CORE CHECK: 唯一真实来源 - object_list
#    在 object_list 中 → abstract (return False)
#    不在 object_list 中 → preserve (return True)
return arg not in self.object_list
```

---

## 数据流图

```
自然语言指令
    ↓
┌─────────────────────────────────────────┐
│ Stage 1: LLM 解析                        │
│ (ltlf_generator.py)                     │
│                                         │
│ LLM → JSON {objects: [...]} →           │
│       spec.objects = result["objects"]  │ ← 提取点
└─────────────────────────────────────────┘
    ↓
    objects = ["a", "b", "c"]  # 示例
    ↓
┌─────────────────────────────────────────┐
│ Stage 2: LTLSpecification               │
│ (ltlf_formula.py)                       │
│                                         │
│ spec.objects                            │ ← 存储
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Pipeline: 传递                          │
│ (ltl_bdi_pipeline.py)                   │
│                                         │
│ ltl_dict = {"objects": spec.objects}    │ ← 打包
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Stage 3: Backward Planning              │
│ (backward_planner_generator.py)         │
│                                         │
│ objects = ltl_dict['objects']           │ ← 解包
│ normalizer = VariableNormalizer(        │
│     domain, objects                     │
│ )                                       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Variable Normalization                  │
│ (variable_normalizer.py)                │
│                                         │
│ self.object_list = sorted(objects)      │ ← 最终使用点
│                                         │
│ _should_preserve(arg):                  │
│   return arg not in self.object_list    │ ← 判断逻辑
└─────────────────────────────────────────┘
```

---

## 可靠性保证

### 1. 单一来源（Single Source of Truth）

- ✅ objects 来自 Stage 1 的 LLM 解析
- ✅ 在整个 pipeline 中只有一个 objects 列表
- ✅ 不从其他地方推断或猜测

### 2. 完整传递

**传递链**:
```
LLM JSON → spec.objects → ltl_dict['objects'] → objects → self.object_list
```

每一步都是**显式传递**，没有隐式转换或推断。

### 3. 无启发式规则

❌ **不依赖**:
- 命名约定（大小写、下划线等）
- 字符串模式匹配
- 类型推断

✅ **只依赖**:
- object_list 的**明确**membership check

---

## 示例验证

### 示例 1: 标准情况

```
自然语言: "Stack block a on block b"
  ↓
Stage 1 LLM: {"objects": ["a", "b"], ...}
  ↓
Stage 3: objects = ["a", "b"]
  ↓
Normalizer: object_list = ["a", "b"]
  ↓
on(a, b) → on(?arg0, ?arg1)  ✓ 两个都在 object_list 中，都被抽象
```

### 示例 2: 混合对象和常量

```
自然语言: "Move robot1 left by 2 units"
  ↓
Stage 1 LLM: {"objects": ["robot1"], ...}  # LLM 识别 robot1 是对象
  ↓
Stage 3: objects = ["robot1"]
  ↓
Normalizer: object_list = ["robot1"]
  ↓
move(robot1, -2, LEFT)
  ↓
- robot1: 在 object_list 中 → 抽象为 ?arg0
- -2: 数字字面量 → 保留
- LEFT: 不在 object_list 中 → 保留
  ↓
结果: move(?arg0, -2, LEFT)  ✓ 正确
```

### 示例 3: 大写对象名（之前的BUG）

```
自然语言: "ROBOT picks BLOCK"
  ↓
Stage 1 LLM: {"objects": ["ROBOT", "BLOCK"], ...}
  ↓
Stage 3: objects = ["ROBOT", "BLOCK"]
  ↓
Normalizer: object_list = ["ROBOT", "BLOCK"]
  ↓
holding(ROBOT)
  ↓
- ROBOT: 在 object_list 中 → 抽象为 ?arg0
  ↓
结果: holding(?arg0)  ✓ 正确（之前会错误保留ROBOT）
```

---

## 潜在风险点（需要注意）

### ⚠️ 风险 1: LLM 识别错误

**场景**: LLM 没有正确识别对象

**示例**:
```
指令: "Move robot LEFT"
LLM 错误输出: {"objects": ["robot", "LEFT"]}  # LEFT 被误认为对象
```

**影响**:
- `move(robot, LEFT)` 会被错误地抽象为 `move(?arg0, ?arg1)`
- 应该是 `move(?arg0, LEFT)`

**缓解措施**:
- 提升 Stage 1 LLM prompt 质量
- 添加对象识别的 validation
- 使用 domain knowledge (types, constants) 进行校验

### ⚠️ 风险 2: objects 列表不完整

**场景**: objects 列表遗漏了某些对象

**示例**:
```
指令: "Put a on b and c on table"
LLM 输出: {"objects": ["a", "b"]}  # 漏掉了 c
```

**影响**:
- `on(c, table)` 中的 c 不会被抽象
- 导致不同实例无法 share schema

**缓解措施**:
- 改进 Stage 1 提示词，要求完整识别
- 添加完整性检查（扫描 formulas 中提到的所有对象）

### ⚠️ 风险 3: 对象名和常量名冲突

**场景**: domain constant 和 problem object 同名

**示例**:
```
Domain: (:constants LEFT RIGHT - direction)
Problem: objects = ["LEFT"]  # 有个对象恰好叫 LEFT
```

**当前行为**:
- LEFT 在 object_list 中 → 会被抽象
- 即使它也是 domain constant

**正确性分析**:
- 这是**正确的**！
- 如果 "LEFT" 在 problem objects 中，说明它在这个问题实例中是对象
- Domain constant "LEFT" 和 problem object "LEFT" 是不同的实体
- 但这种命名很容易混淆，应该避免

---

## 未来改进方向

### 1. 添加 Domain Constants 支持

**当前**: PDDLParser 不解析 `(:constants ...)` 部分

**改进**:
```python
@dataclass
class PDDLDomain:
    name: str
    types: List[str]
    constants: List[str]  # ← 新增
    predicates: List[PDDLPredicate]
    actions: List[PDDLAction]
```

**好处**:
- 可以显式区分 domain constants
- 更好的 validation（检查命名冲突）

### 2. 对象识别 Validation

在 Stage 1 后添加验证：
```python
def validate_objects(spec: LTLSpecification, domain: PDDLDomain):
    """验证识别的对象是否合理"""
    # 1. 检查是否有对象名与 domain constants 冲突
    # 2. 检查formulas 中是否有未列入 objects 的标识符
    # 3. 检查对象是否符合 domain types
```

### 3. 完整性检查

自动扫描 formulas 和 grounding_map，确保所有对象都在 objects 列表中：
```python
def check_object_completeness(spec: LTLSpecification):
    """检查 objects 列表是否完整"""
    mentioned_objects = set()
    for atom in spec.grounding_map.atoms.values():
        mentioned_objects.update(atom.args)

    missing = mentioned_objects - set(spec.objects)
    if missing:
        raise ValueError(f"Objects {missing} mentioned but not in object list")
```

---

## 测试覆盖

### 现有测试

✅ `tests/test_constant_handling.py`: 测试混合类型处理
✅ `tests/test_semantic_constant_detection.py`: 测试基于 object_list 的判断
✅ 大写对象名测试
✅ 混合命名约定测试

### 需要添加的测试

❌ End-to-end 测试：从自然语言到代码生成，验证 objects 正确传递
❌ LLM 识别错误的 recovery 测试
❌ objects 列表不完整的检测测试

---

## 总结

### 核心设计

**object_list 是唯一可靠来源** for abstraction decisions:
1. ✅ 来自 Stage 1 LLM 的明确识别
2. ✅ 在整个 pipeline 中完整传递
3. ✅ Stage 3 严格基于 membership check

### 不依赖

❌ 启发式规则（大小写、命名约定）
❌ 模式匹配
❌ 类型推断

### 依赖

✅ **唯一真实来源**: `self.object_list`
✅ **严格判断**: `arg in self.object_list`

---

**文档维护**: 2025-11-10
**最后验证**: Variable normalization 已重构为完全基于 object_list
