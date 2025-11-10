# Stage 3: Variable Abstraction for Backward Planning

## Motivation

在当前的实现中，backward planning对每个具体的goal实例（如`on(a, b)`, `on(c, d)`）都分别进行状态空间探索。但实际上，这些goals的结构是相同的，只是对象实例不同。

### 关键洞察

在PDDL中，predicates和actions都是用**变量**定义的：
- `(on ?b1 ?b2 - block)` - predicate定义
- `(:action pick-up :parameters (?b1 ?b2 - block) ...)` - action定义

这意味着：
- `on(a, b)` 和 `on(c, d)` 本质上是**同一个模式** `on(?b1, ?b2)` 的不同实例
- 它们应该**共用同一套backward planning结果**
- 只需在最后生成代码时，将变量绑定到具体对象

## 设计原则

### 1. 变量级别的抽象（Variable-Level Abstraction）

**核心思想：** 在backward planning中使用变量化的predicates，而不是grounded predicates。

**示例：**
```
当前方法：
  goal: on(a, b)  → 探索 → state graph with on(a, b), holding(a), clear(b), ...
  goal: on(c, d)  → 探索 → state graph with on(c, d), holding(c), clear(d), ...

新方法：
  goal: on(?v0, ?v1) → 探索一次 → state graph with on(?v0, ?v1), holding(?v0), clear(?v1), ...

  然后复用：
  - on(a, b) → 绑定 {?v0: a, ?v1: b}
  - on(c, d) → 绑定 {?v0: c, ?v1: d}
```

### 2. 类型感知的变量分配（Type-Aware Variable Assignment）

根据PDDL类型系统为对象分配变量：
- 相同类型的不同对象使用不同变量
- 例如：3个block对象 → `?v0_block`, `?v1_block`, `?v2_block`

### 3. 一致性保证（Consistency）

同一个对象在不同predicates中必须映射到同一个变量：
```
on(a, b) & clear(a)
→ on(?v0, ?v1) & clear(?v0)  ✓ (a映射到?v0)
→ on(?v0, ?v1) & clear(?v2)  ✗ (a不能同时映射到?v0和?v2)
```

## 实现方案

### 组件1: VariableNormalizer

负责将grounded predicates标准化为变量化的形式。

```python
class VariableNormalizer:
    """
    将grounded predicates标准化为变量化形式

    示例：
        输入: [on(a, b), clear(a), ontable(c)]
        输出: [on(?v0, ?v1), clear(?v0), ontable(?v2)]
        变量映射: {a: ?v0, b: ?v1, c: ?v2}
    """

    def normalize_predicates(self, predicates: List[PredicateAtom]) -> Tuple[List[PredicateAtom], Dict[str, str]]:
        """
        标准化predicates

        Returns:
            (变量化的predicates, 对象到变量的映射)
        """
        pass

    def normalize_goal(self, goal: List[PredicateAtom]) -> str:
        """
        将goal标准化为cache key

        示例:
            on(a, b) → "on(?v0, ?v1)"
            on(c, d) → "on(?v0, ?v1)"  (相同的key!)
        """
        pass
```

### 组件2: 修改PredicateAtom

添加对变量的支持：

```python
@dataclass(frozen=True)
class PredicateAtom:
    name: str
    args: Tuple[str, ...]
    negated: bool = False

    def is_variable_arg(self, arg: str) -> bool:
        """检查参数是否是变量（以?开头）"""
        return arg.startswith('?')

    def is_grounded(self) -> bool:
        """检查predicate是否完全grounded（没有变量）"""
        return all(not self.is_variable_arg(arg) for arg in self.args)

    def is_variable_predicate(self) -> bool:
        """检查predicate是否使用变量"""
        return any(self.is_variable_arg(arg) for arg in self.args)
```

### 组件3: 修改ForwardStatePlanner

使用变量化的planning：

```python
class ForwardStatePlanner:
    def __init__(self, domain: PDDLDomain, num_objects: int, object_type: str = "block"):
        """
        不再接收具体的objects列表，而是接收对象数量和类型

        Args:
            domain: PDDL domain
            num_objects: 对象数量（例如：3个blocks）
            object_type: 对象类型（例如："block"）
        """
        self.domain = domain
        self.num_objects = num_objects
        self.object_type = object_type

        # 生成变量列表: [?v0, ?v1, ?v2, ...]
        self.variables = [f"?v{i}" for i in range(num_objects)]

        # 使用变量而不是具体对象来ground actions
        self._cached_variable_actions = self._ground_all_actions()

    def explore_from_goal(self, goal_predicates: List[PredicateAtom]) -> StateGraph:
        """
        探索使用变量化的predicates

        Args:
            goal_predicates: 变量化的goal predicates (例如: [on(?v0, ?v1)])
        """
        # 状态空间中的所有predicates都使用变量
        pass
```

### 组件4: 修改BackwardPlannerGenerator

在generator层面使用变量标准化：

```python
class BackwardPlannerGenerator:
    def generate(self, ltl_dict, dfa_result):
        # 1. 提取对象信息
        objects = ltl_dict['objects']
        num_objects = len(objects)

        # 2. 创建normalizer
        normalizer = VariableNormalizer(objects)

        # 3. 变量级别的cache
        variable_goal_cache = {}  # 变量化的goal → state_graph

        for transition in dfa_info.transitions:
            # 4. 解析goal
            goal_predicates = self._parse_transition_label(transition.label)

            # 5. 标准化goal（转换为变量形式）
            normalized_goal, obj_to_var_map = normalizer.normalize_predicates(goal_predicates)
            goal_key = normalizer.serialize_goal(normalized_goal)

            # 6. 检查cache（变量级别）
            if goal_key in variable_goal_cache:
                state_graph = variable_goal_cache[goal_key]
                # 7. 使用obj_to_var_map来instantiate code
            else:
                # 8. 用变量进行planning
                planner = ForwardStatePlanner(domain, num_objects)
                state_graph = planner.explore_from_goal(normalized_goal)
                variable_goal_cache[goal_key] = state_graph
```

### 组件5: 修改AgentSpeakCodeGenerator

在code generation时实例化变量：

```python
class AgentSpeakCodeGenerator:
    def __init__(self, state_graph, goal_name, domain, var_to_obj_map):
        """
        Args:
            var_to_obj_map: 变量到对象的映射 (例如: {?v0: a, ?v1: b})
        """
        self.var_to_obj_map = var_to_obj_map

    def instantiate_predicate(self, pred: PredicateAtom) -> PredicateAtom:
        """
        将变量化的predicate实例化为具体对象

        例如:
            on(?v0, ?v1) + {?v0: a, ?v1: b} → on(a, b)
        """
        new_args = [self.var_to_obj_map.get(arg, arg) for arg in pred.args]
        return PredicateAtom(pred.name, new_args, pred.negated)
```

## 优化效果

### Before (Object-Level Planning)
```
Goal: on(a, b) → 探索1093 states → generate code
Goal: on(b, a) → 探索1093 states → generate code
Goal: on(c, d) → 探索1093 states → generate code

Total: 3 × 1093 = 3279 states explored
```

### After (Variable-Level Planning)
```
Goal pattern: on(?v0, ?v1) → 探索1093 states一次 → state graph

实例化:
- on(a, b) → 复用 state graph + 绑定 {?v0: a, ?v1: b}
- on(b, a) → 复用 state graph + 绑定 {?v0: b, ?v1: a}
- on(c, d) → 复用 state graph + 绑定 {?v0: c, ?v1: d}

Total: 1 × 1093 = 1093 states explored
Reduction: 67% less exploration
```

## 实现顺序

1. ✅ 创建 `variable_normalizer.py` 模块
2. ✅ 更新 `PredicateAtom` 添加变量支持
3. ✅ 修改 `ForwardStatePlanner` 使用变量
4. ✅ 更新 `backward_planner_generator.py`
5. ✅ 更新 `AgentSpeakCodeGenerator`
6. ✅ 测试和验证

## 注意事项

1. **向后兼容性**: 保持对现有测试的支持
2. **类型推断**: 正确处理PDDL类型系统
3. **变量命名**: 使用一致的变量命名规则
4. **调试输出**: 清晰地显示变量和对象的映射关系

## 测试计划

1. 单元测试: VariableNormalizer
2. 集成测试: 验证相同模式的goals共用state graph
3. 性能测试: 测量探索次数的减少
4. 正确性测试: 验证生成的代码功能正确
