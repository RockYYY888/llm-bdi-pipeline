# Stage 3 Optimization Opportunities

## 发现的重复和冗余

### 🔴 Critical: Ground Actions在每个state重复计算

**问题位置**: `forward_planner.py:167-177`

```python
while queue:  # For each state in BFS queue
    current_state = queue.popleft()

    for grounded_action in self._ground_all_actions():  # ← 每次都重新ground!
        # Check preconditions...
```

**具体问题**:
- `_ground_all_actions()` 在**每个state**都被调用一次
- 对于2 blocks: 探索1093个states → 调用1093次
- 每次都重新计算 `itertools.product(objects, repeat=n)`
- 实际上ground actions对于固定的domain和objects是**完全不变的**！

**性能影响**:
```
2 blocks, 1093 states:
- Current: 1093 × 32 ground actions = 34,976次grounding计算
- Optimal: 1次 × 32 ground actions = 32次grounding计算
- 浪费: 1093x = 99.9%的重复计算
```

**优化方案**:
```python
class ForwardStatePlanner:
    def __init__(self, domain, objects):
        self.domain = domain
        self.objects = objects
        self.grounded_actions = self._ground_all_actions()  # ← 只计算一次

    def explore_from_goal(...):
        while queue:
            current_state = queue.popleft()

            for grounded_action in self.grounded_actions:  # ← 直接使用缓存
                # Check preconditions...
```

**预期提升**:
- 2 blocks: 减少99.9%的grounding计算
- 3 blocks: 减少99.98%的grounding计算（假设50k states）
- 时间节省估计: **20-30%总体速度提升**

---

### 🟠 Important: 相同Goal的重复探索

**问题位置**: `backward_planner_generator.py:105-145`

```python
all_code_sections = []

for transition in dfa_transitions:
    goal = parse_label(transition.label)  # "on_a_b"

    # 每个transition都独立探索，即使goal相同
    state_graph = planner.explore_from_goal(goal)  # ← 重复探索
    code = codegen.generate(state_graph)
    all_code_sections.append(code)
```

**具体问题**:
如果DFA中有多个transitions使用相同的label，会重复探索：

```
DFA example:
state0 --[on_a_b]--> state1
state2 --[on_a_b]--> state3  ← 相同的label!
```

当前实现会对 `on_a_b` 探索**两次**，生成**两次**完全相同的state graph和code。

**性能影响**:
```
如果DFA有N个transitions使用相同goal:
- Current: N次完整探索（N × 1093 states）
- Optimal: 1次探索 + (N-1)次代码复用
- 浪费: (N-1) × 100%的重复探索
```

**优化方案**:
```python
def generate(self, ltl_dict, dfa_result):
    # Cache for goal → state_graph mapping
    goal_cache = {}

    all_code_sections = []

    for transition in dfa_transitions:
        goal = parse_label(transition.label)
        goal_key = self._serialize_goal(goal)  # Convert to hashable key

        if goal_key in goal_cache:
            # Reuse cached state graph
            state_graph = goal_cache[goal_key]
            print(f"  ✓ Reusing cached exploration for {goal_key}")
        else:
            # First time seeing this goal
            state_graph = planner.explore_from_goal(goal)
            goal_cache[goal_key] = state_graph

        code = codegen.generate(state_graph)
        all_code_sections.append(code)
```

**预期提升**:
- DFA with duplicate goals: 减少50-90%的探索时间（取决于重复度）

---

### 🟡 Moderate: Initial Beliefs和Action Plans的重复生成

**问题位置**: `agentspeak_codegen.py:77-86`

```python
def generate(self):
    sections = []

    initial_beliefs = self._generate_initial_beliefs()  # ← 每个transition都生成
    action_plans = self._generate_action_plans()       # ← 每个transition都生成
    goal_plans = self._generate_goal_plans()           # Only this differs!

    sections.append(initial_beliefs)
    sections.append(action_plans)
    sections.append(goal_plans)

    return "\n\n".join(sections)
```

**具体问题**:
每个transition (code section) 都会生成：
1. **Initial Beliefs**: `ontable(a). clear(a). ...` （完全相同）
2. **Action Plans**: `+!pick_up(B) : ... <- ...` （完全相同）
3. Goal Plans: `+!on(a, b) : ... <- ...` （**唯一**不同的部分）

对于2个transitions：
- Initial beliefs生成**2次**（浪费1次）
- Action plans生成**2次**（浪费1次）

**最终AgentSpeak文件**:
```agentspeak
/* ========== Goal: on(a, b) ========== */
/* Initial Beliefs */        ← 重复1
ontable(a). clear(a). ...

/* Action Plans */            ← 重复2
+!pick_up(B) : ... <- ...
+!put_on_block(B1, B2) : ... <- ...

/* Goal Plans */
+!on(a, b) : ... <- ...

/* ========== Next Goal ========== */

/* ========== Goal: clear(a) ========== */
/* Initial Beliefs */        ← 重复1（相同！）
ontable(a). clear(a). ...

/* Action Plans */            ← 重复2（相同！）
+!pick_up(B) : ... <- ...
+!put_on_block(B1, B2) : ... <- ...

/* Goal Plans */
+!clear(a) : ... <- ...
```

**优化方案**:

**方案A**: 只生成一次共享部分
```agentspeak
/* Main Header */

/* ========== Shared Components ========== */

/* Initial Beliefs */
ontable(a). clear(a). handempty.

/* Action Plans */
+!pick_up(B1, B2) : ... <- ...
+!put_on_block(B1, B2) : ... <- ...
... (all action plans)

/* ========== Goal-Specific Plans ========== */

/* Goal: on(a, b) */
+!on(a, b) : ... <- ...

/* Goal: clear(a) */
+!clear(a) : ... <- ...
```

**方案B**: 修改code generation结构
```python
class BackwardPlannerGenerator:
    def generate(self, ltl_dict, dfa_result):
        # Generate shared parts ONCE
        shared_initial_beliefs = self._generate_shared_initial_beliefs()
        shared_action_plans = self._generate_shared_action_plans()

        # Generate goal-specific parts for each transition
        all_goal_plans = []
        for transition in dfa_transitions:
            goal_plans = self._generate_goal_plans_only(transition)
            all_goal_plans.append(goal_plans)

        # Combine
        final_code = header + shared_initial_beliefs + shared_action_plans + \
                    "\n\n".join(all_goal_plans)
        return final_code
```

**性能影响**:
- Code generation时间: 减少30-50%
- 生成的代码大小: 减少20-40%（对于多个transitions）
- **更重要**: 代码更清晰，避免重复定义

---

### 🟢 Nice-to-have: 对称性优化（Symmetry Reduction）

**问题描述**:
在blocksworld中，很多states是**对称的**：

```
on(a, b) 和 on(b, a) 的state space结构是对称的
只需要把a和b交换即可
```

**例子**:
```
Goal 1: on(a, b)
- 探索1093个states
- 生成26个plans

Goal 2: on(b, a)  ← 参数只是交换了
- 又探索1093个states
- 生成26个plans（结构相同，只是参数不同）
```

**优化方案** (复杂度高):
1. 检测goal之间的对称性
2. 只探索canonical form (如 on(x, y) where x < y)
3. 通过参数重命名生成symmetric goals的代码

**预期提升**:
- 理论上可以减少50%的探索（对于对称domain）
- **但实现复杂度极高**，可能不值得

**建议**: 暂时不实现，除非有明确的对称性需求

---

## 优化优先级和实施建议

### Priority 1: 🔴 Critical (必须修复)

**优化1: Cache ground actions**
- **实施难度**: 极低（5分钟）
- **性能提升**: 20-30%总体速度
- **风险**: 极低
- **建议**: 立即实施

### Priority 2: 🟠 Important (强烈建议)

**优化2: Cache goal exploration results**
- **实施难度**: 低（30分钟）
- **性能提升**: 取决于DFA重复度（0-90%）
- **风险**: 低（需要正确的goal serialization）
- **建议**: 尽快实施

### Priority 3: 🟡 Moderate (建议优化)

**优化3: 重构code generation避免重复**
- **实施难度**: 中等（1-2小时）
- **性能提升**: 30-50% code generation时间
- **风险**: 中等（需要修改AgentSpeak文件结构）
- **建议**: 在完成Priority 1-2后实施

### Priority 4: 🟢 Nice-to-have (可选)

**优化4: Symmetry reduction**
- **实施难度**: 极高（数周）
- **性能提升**: 理论50%（实际可能更低）
- **风险**: 高（容易引入bugs）
- **建议**: 暂不实施，除非有明确需求

---

## 实施计划

### Phase 1: Quick Wins (1小时)
1. ✅ Cache ground actions in ForwardStatePlanner
2. ✅ Add performance metrics logging

### Phase 2: Medium Impact (2-3小时)
3. ✅ Implement goal exploration caching
4. ✅ Add cache hit/miss statistics

### Phase 3: Code Quality (2-3小时)
5. ✅ Refactor AgentSpeak code generation
6. ✅ Update tests to verify new structure
7. ✅ Update documentation

### Phase 4: Future Work (暂不实施)
- Symmetry reduction (research project level)
- Heuristic search (A* with delete relaxation)
- Partial-order reduction

---

## 预期总体提升

**当前性能** (2 blocks, 2 transitions with same goal):
- Grounding: 34,976次计算
- Exploration: 2次 × 1093 states = 2186 states
- Code generation: 2次完整生成

**优化后性能** (实施Phase 1-3):
- Grounding: 32次计算 (减少99.9%)
- Exploration: 1次 × 1093 states (减少50%)
- Code generation: 1次共享部分 + 2次goal-specific部分 (减少40%)

**总体预期提升**:
- **50-70% 总体速度提升**
- **30-40% 内存使用减少**
- **更清晰的代码结构**

---

## 测试计划

对于每个优化，需要验证：
1. ✅ 功能正确性（生成的plans相同）
2. ✅ 性能提升（测量实际时间）
3. ✅ 边界情况（empty DFA, single transition, etc.）
4. ✅ 向后兼容（现有tests仍然通过）
