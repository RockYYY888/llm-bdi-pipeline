# DNF Disjunct 与 Caching 机制深度分析

## 核心发现

### ✅ DNF转换：算法正确但产生大量disjuncts

**机制**：
```
DFA Transition Label → Parse → DNF (Disjunctive Normal Form)

例: ~on_d_e & (~clear_c | ~on_a_b)
  → Disjunct 1: [not on(d,e), not clear(c)]
  → Disjunct 2: [not on(d,e), not on(a,b)]
```

**转换算法** (src/stage3_code_generation/boolean_expression_parser.py):
1. **消除蕴含**: `A → B` ⇒ `~A | B`
2. **推入否定**: `~(A & B)` ⇒ `~A | ~B` (De Morgan)
3. **提取DNF**: 使用递归笛卡尔积
   - `OR`: 连接所有子DNF
   - `AND`: 笛卡尔积所有子DNF
   - 例: `(A|B) & (C|D)` ⇒ `[AC, AD, BC, BD]` (4个disjuncts)

**验证结果**: 19/19 复杂测试全部通过 ✓

---

### ❌ Caching机制：粒度过粗导致完全失效

**当前策略** (src/stage3_code_generation/backward_planner_generator.py:112-160):
```python
variable_goal_cache = {}  # 完整goal组合 → state_graph

# Cache key = 整个goal的序列化
cache_key = "not clear(?v2)|not on(?v0, ?v1)"  # 排序后的完整组合
```

**问题**：不同disjuncts几乎永远无法匹配相同的cache key！

---

## 性能瓶颈剖析

### TEST 3 实际运行数据

| Disjunct | Goal Predicates | Normalized | Cache Key | States | Time |
|----------|----------------|------------|-----------|--------|------|
| T1-D1 | `not on(d,e)`, `not clear(c)` | `not on(?v0,?v1)`, `not clear(?v2)` | `"not clear(?v2)\|not on(?v0,?v1)"` | 19,173 | ~190s |
| T1-D2 | `not on(d,e)`, `not on(a,b)` | `not on(?v0,?v1)`, `not on(?v2,?v3)` | `"not on(?v0,?v1)\|not on(?v2,?v3)"` | 2,934 | ~30s |
| T2-D1 | `on(d,e)` | `on(?v0,?v1)` | `"on(?v0,?v1)"` | **50,000+** | **450s+** |
| T2-D2 | `clear(c)`, `on(a,b)` | `clear(?v0)`, `on(?v1,?v2)` | `"clear(?v0)\|on(?v1,?v2)"` | 1 | <1s |

**Cache命中率**: 0.0% (0 hits, 4 misses)

### 问题根源

**两个disjuncts共享谓词却无法复用探索**：
```
T1-D1: [not on(?v0,?v1), not clear(?v2)]  ← 探索了 19,173 个状态
T1-D2: [not on(?v0,?v1), not on(?v2,?v3)]  ← 又探索了 2,934 个状态
        ^^^^^^^^^^^^^^
        这部分完全相同！但因cache key不同而被浪费
```

**对比TEST 2.4** (3 objects vs 5 objects):
```
TEST 2.4: on(?v0,?v1) → 525 states    (3 objects, 84 actions)
TEST 3:   on(?v0,?v1) → 50,000+ states (5 objects, 330 actions)

状态数量 ≈ O(actions^depth) = O(objects^param_count)^depth
```

---

## 为什么当前Cache完全失效？

### 1. **DNF产生组合爆炸**
```
复杂表达式 → DNF转换 → N个disjuncts
每个disjunct = 不同的谓词组合
→ N个不同的cache keys
→ N次独立的完整探索
```

### 2. **Cache Key过于严格**
```python
# 当前: 必须完全匹配整个goal组合
"not clear(?v2)|not on(?v0, ?v1)" ≠ "not on(?v0, ?v1)|not on(?v2, ?v3)"
                   ^^^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^^^
                   即使包含相同子goal，也无法复用！
```

### 3. **跨Transition也无法复用**
即使不同transitions有相同的variable pattern（如多个`on(?v0,?v1)`），因为objects数量不同导致：
- 不同的变量列表: `['?v0', '?v1', '?v2']` vs `['?v0', '?v1', '?v2', '?v3', '?v4']`
- 不同的action数量: 84 vs 330
- 完全独立的state graphs

---

## 量化影响

### TEST 3 性能分解
```
总时间: 676.62s

T1-D1: 19,173 states × ~10ms  ≈ 190s
T1-D2:  2,934 states × ~10ms  ≈  30s
T2-D1: 50,000 states × ~9ms   ≈ 450s
T2-D2:  1 state              ≈  <1s
                    Total     ≈ 670s ✓

如果能复用 not on(?v0,?v1) 的探索：
  理论节省: ~30s (T1-D2的重复部分)
  实际节省可能更多（如果T2-D1也能部分复用）
```

### 否定谓词的状态空间
```
肯定目标 on(d,e):
  完整状态 = {on(d,e), clear(d), handempty}
  约束强 → 状态空间相对小

否定目标 not on(d,e):
  "d不在e上" = d可以在{table, a, b, c上, 或被拿着}
  约束弱 → 状态空间巨大 (TEST 3: 19,173 states!)
```

---

## 优化方案

### 方案A: Predicate级别的缓存 ⭐ 推荐
```python
predicate_cache = {
    "not on(?v0, ?v1)": partial_state_graph_1,
    "not clear(?v2)": partial_state_graph_2
}

# 组合时合并子图
full_graph = merge_graphs([
    predicate_cache["not on(?v0, ?v1)"],
    predicate_cache["not clear(?v2)"]
])
```

**优点**:
- 最大化复用（单个谓词级别）
- 跨disjunct共享
- 实现相对简单

**挑战**:
- 需要实现graph合并逻辑
- 处理变量重命名
- 保证合并后的一致性

### 方案B: Sub-goal提取
```python
# 从已探索的完整goal中提取子集
cache["not on(?v0,?v1)|not clear(?v2)"] → StateGraph(19,173 states)

# 提取子集
extract_subgraph(
    full_graph=cache["not on(?v0,?v1)|not clear(?v2)"],
    sub_goal=["not on(?v0,?v1)"]
) → StateGraph(subset of 19,173 states)
```

**优点**:
- 利用已有探索结果
- 无需修改探索算法

**挑战**:
- 提取算法复杂
- 可能遗漏某些路径

### 方案C: 深度限制 + 启发式剪枝
```python
max_depth = 3  # 限制探索深度
max_states = 10_000  # 降低状态限制

# 针对否定目标使用更激进的剪枝
if has_negation(goal):
    use_heuristic_pruning()
```

**优点**:
- 立即减少计算量
- 实现简单

**缺点**:
- 可能丢失有效路径
- 影响plan完整性

---

## 结论

1. **DNF转换算法正确** ✓
   - 通过19个复杂测试验证
   - 正确处理嵌套、De Morgan、蕴含等

2. **性能瓶颈是Cache粒度问题** ✗
   - 当前: Goal级别缓存 → 0% 命中率
   - DNF产生大量disjuncts → 每个都需要完整探索
   - 共享的sub-goals被完全浪费

3. **影响巨大**:
   - TEST 3: 676秒（>11分钟）
   - 如果有效缓存: 预计可减少30-50%时间
   - 更大问题: 5 objects已触发截断，无法生成完整plan

4. **推荐方案**: Predicate级别缓存
   - 需要实现graph合并机制
   - 复用潜力最大
   - 适用于DNF场景

---

## 下一步行动

1. **立即**: 实现深度限制作为临时缓解
2. **短期**: 原型实现predicate级别缓存
3. **长期**: 研究symbolic/BDD表示减少状态数量
