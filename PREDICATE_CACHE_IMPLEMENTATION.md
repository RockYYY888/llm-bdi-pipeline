# Predicate-Level Caching Implementation

## 问题回顾

DNF转换会产生大量disjuncts，原有的goal-level缓存导致：
- Cache命中率0%
- 共享的单个predicates被重复探索
- TEST 3耗时676秒且被截断

## 解决方案：双层缓存

### 架构设计

```python
# Tier 1: Single predicate cache
predicate_cache = {}
# Key: (predicate_pattern, num_objects)
# Example: ("on(?v0, ?v1)", 5) → StateGraph

# Tier 2: Full goal cache
full_goal_cache = {}
# Key: serialized full goal
# Example: "not clear(?v2)|not on(?v0, ?v1)" → StateGraph
```

### 缓存策略

**查询逻辑**：
1. 如果goal只有1个predicate → 查predicate_cache
2. 否则 → 查full_goal_cache
3. 两层都miss → 探索并缓存

**更新逻辑**：
1. 单predicate goal → 缓存到predicate_cache
2. 多predicate goal → 缓存到full_goal_cache

### Cache Key设计

关键insight：同一predicate pattern在不同objects数量下会产生不同的state space！

```
"on(?v0, ?v1)" + 3 objects → 525 states
"on(?v0, ?v1)" + 5 objects → 50,000+ states
```

因此：**Cache key = (predicate_pattern, num_objects)**

## 实现细节

### 修改文件
- `src/stage3_code_generation/backward_planner_generator.py`

### 关键代码段

```python
# Two-tier cache lookup
if len(normalized_goal) == 1:
    single_pred = normalized_goal[0]
    single_pred_key = (single_pred.to_agentspeak(), len(objects))

    if single_pred_key in predicate_cache:
        # Predicate cache HIT!
        state_graph, _ = predicate_cache[single_pred_key]
        predicate_cache_hits += 1
```

### 统计输出

新的缓存统计包含：
- Tier 1 (Predicate cache): hits/misses/hit rate
- Tier 2 (Full-goal cache): hits/misses/hit rate
- Overall: total hits/misses/hit rate

## 预期效果

### 场景1：相同单个predicate在不同disjuncts中出现

**Before**:
```
D1: [on(?v0,?v1)] → Explore 50K states
D2: [on(?v0,?v1)] → Explore 50K states again! ✗
```

**After**:
```
D1: [on(?v0,?v1)] → Explore 50K states
D2: [on(?v0,?v1)] → Cache HIT! ✓ Reuse
```

**节省**: 100% 的重复探索时间

### 场景2：跨Transition复用

如果Transition 1和Transition 2都包含相同的单个predicate goal，也能复用！

### 场景3：多predicate goals

保持原有行为，使用full-goal cache，不受影响。

## 测试验证

### 单元测试
```bash
python test_predicate_cache.py
```

验证：
- ✓ Cache key正确生成
- ✓ 相同predicate + 相同objects数量 = 相同key
- ✓ Cache hit/miss逻辑正确

### 集成测试

预期在以下场景看到改进：
1. TEST 3中如果有重复的单个predicate goals
2. 跨多个test cases的相同predicates
3. 未来更复杂的DNF表达式

## 限制与未来优化

### 当前限制
1. **不合并多predicate goals**:
   - [on(?v0,?v1), clear(?v2)] 探索后，不会分别缓存单个predicates
   - 原因：单个predicate的complete goal state ≠ 组合的complete goal state

2. **无法部分复用**:
   - [on(?v0,?v1), clear(?v2)] 不能复用已缓存的 [on(?v0,?v1)]
   - 需要实现graph合并/提取机制

### 未来优化方向

#### Option A: Sub-graph Extraction
```python
# 从已探索的完整goal中提取子集
full_graph = cache["on(?v0,?v1)|clear(?v2)"]  # 19K states
sub_graph = extract_subgraph(full_graph, ["on(?v0,?v1)"])  # subset
```

#### Option B: Compositional Caching
```python
# 合并多个单predicate graphs
graph1 = predicate_cache["on(?v0,?v1)"]
graph2 = predicate_cache["clear(?v2)"]
combined = merge_graphs([graph1, graph2])  # 需要复杂的合并逻辑
```

#### Option C: Incremental Exploration
```python
# 如果部分predicates已缓存，只探索增量部分
if "on(?v0,?v1)" in predicate_cache:
    base_graph = predicate_cache["on(?v0,?v1)"]
    incremental_explore(base_graph, additional_predicates=["clear(?v2)"])
```

## 总结

✅ **已实现**: 双层predicate-level + full-goal-level缓存
✅ **已验证**: Cache key逻辑正确
⏳ **待测试**: 实际性能改进（需要完整test suite）
🔮 **未来**: Graph合并/提取机制以进一步提升复用率
