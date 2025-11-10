# 🎯 Stage 3 项目文档导航指南

**更新日期**: 2025-11-11
**当前状态**: Variable Abstraction & Schema-Level Caching 已完成，存在设计问题需要修复

---

## 📋 快速状态总览

### ✅ 已完成的功能
- **Backward Planning 核心实现** - Stage 3 从 LLM 迁移到 Backward Planning
- **Ground Actions Caching** - 99.9% 冗余计算消除 (Priority 1)
- **Goal Exploration Caching** - 缓存系统实现 (Priority 2)
- **Code Structure Optimization** - 共享组件生成优化 (Priority 3)
- **Variable-Level Planning** - 变量抽象基础设施 (Phase 1)
- **Schema-Level Abstraction** - 真正的模式级抽象 (完全实现)
- **Constants Handling** - 正确处理常量与对象的区分
- **Scalability Testing** - 2/3 blocks 测试验证通过

### ⚠️ 已知问题（需要修复）
- **Object-Specific Goal Plans** - 生成的 goal plans 是 object-specific 而非 parameterized
- **Type System Incomplete** - 类型推断不完整，所有对象分配到第一个类型
- **Variable Naming Inconsistency** - 归一化使用 `?arg0` 但规划器使用 `?v0`

### 🔴 生产环境限制（长期改进方向）
- **State Space Explosion** - 4+ blocks 状态爆炸，需要启发式搜索
- **No Heuristic Search** - 当前仅 BFS，需要 A* 和 landmarks
- **Domain-Specific Hardcoding** - blocksworld 假设，难以泛化到其他领域

---

## 🚀 三种阅读路径

### 路径 1: 快速了解（15分钟）- 项目概览

**目标**: 了解 Stage 3 是什么，当前做了什么，有什么限制

1. **README.md** (根目录)
   - 第 14-15 行: Stage 3 从 LLM 迁移到 Backward Planning
   - 第 129-149 行: Stage 3 架构图
   - 第 155-167 行: Backward Planning 关键特性
   - 第 559-597 行: 当前实现状态和已知限制

2. **docs/stage3_backward_planning_design.md** (前 50 行)
   - 背景和动机
   - 核心设计决策概览
   - 关键创新点

3. **docs/stage3_production_limitations.md** (浏览表格)
   - State Space Explosion 章节
   - 2 blocks vs 3 blocks vs N blocks 状态数
   - 生产环境就绪度评估表

4. **运行核心测试**
   ```bash
   python tests/stage3_code_generation/test_integration_backward_planner.py
   ```
   - 期望: 所有测试通过 ✅
   - 验证: Variable-level planning 工作正常

**✅ 得到**: 知道 Stage 3 做什么、怎么做、限制是什么

---

### 路径 2: 深入理解（1小时）- 设计与实现

**目标**: 完全理解设计思路、实现方式、优化策略

#### 第一步: 核心设计（20分钟）
1. **docs/stage3_backward_planning_design.md** (完整阅读)
   - 16个核心设计决策（这是精华！）
   - Q&A 记录
   - 技术架构
   - 重点关注:
     - Decision 1-3: DFA语义、搜索方向、状态表示
     - Decision 7-8: 如何处理多个转换、布尔表达式
     - Decision 12: Belief Updates 处理

#### 第二步: 变量抽象实现（20分钟）
2. **docs/stage3_schema_level_abstraction.md**
   - **STATUS: FULLY IMPLEMENTED ✅**
   - Position-Based Normalization 算法
   - 性能结果: 8 goals → 3 explorations (62.5% cache hit rate)
   - 真正的模式级抽象实现

3. **docs/stage3_variable_abstraction_summary.md**
   - Phase 1 完成状态
   - 当前行为说明
   - 已实现的优势
   - 未来增强方向

#### 第三步: 优化与限制（20分钟）
4. **docs/stage3_optimization_opportunities.md**
   - ✅ Priority 1-3: 已完成
     - Ground actions caching: 99.9% 减少
     - Goal exploration caching: 工作中
     - Code structure optimization: 共享组件
   - ⏳ Priority 4: 未实现（Symmetry reduction）

5. **docs/stage3_production_limitations.md**
   - 7个关键限制详解
   - 为什么 blocksworld 能工作但大规模场景不行
   - 生产环境就绪度评估
   - 推荐解决方案

6. **docs/stage3_technical_debt.md**
   - 6个类别的技术债务
   - ✅ 已解决: Redundant Action Grounding
   - 🔧 当前问题: No Heuristic, Dead-End Detection等
   - 优先级路线图

**✅ 得到**: 完全理解设计、实现、优化和限制

---

### 路径 3: 问题诊断与修复（按需）- 解决具体问题

**目标**: 修复已知设计问题

#### 🔴 Critical Issue: Object-Specific vs Parameterized Plans

**必读文档**:
1. **docs/CRITICAL_DESIGN_ISSUES.md**
   - **STATUS: CRITICAL - Requires immediate refactoring**
   - Issue 1: Object-Specific Goal Plans ❌ CRITICAL
     - 当前生成: `+!on(a, b) : on(a, b) <- ...`
     - 应该生成: `+!on(X, Y) : on(X, Y) <- ...`
   - Issue 2: Incomplete Type System ❌ HIGH PRIORITY
   - Issue 3: Variable Naming Inconsistency
   - 修复优先级和实施建议

**相关分析文档**:
2. **docs/constant_variable_distinction_analysis.md**
   - 深入分析常量与变量的区分逻辑
   - 当前实现的缺陷分析
   - 为什么测试能通过（侥幸！）
   - 正确的实现方案

#### ✅ Resolved Issues (参考历史)

如果想了解已经解决的问题:

1. **docs/issue_ab_resolution.md**
   - Issue A (Constants Handling): ✅ FIXED
   - Issue B (Scalability Behavior): ✅ VERIFIED AS CORRECT
   - 包含详细的修复方案和测试结果

2. **docs/variable_abstraction_soundness_analysis.md**
   - 最初发现的 soundness 问题
   - 问题 A 和 B 的详细分析
   - （已被 issue_ab_resolution.md 解决）

---

## 📚 所有文档分类索引

### 🎯 核心设计文档（必读）
- **stage3_backward_planning_design.md** (43KB, 最重要)
  - 完整设计规范
  - 16个核心设计决策
  - Q&A 记录
  - 技术架构

### ✅ 实现状态文档
- **stage3_schema_level_abstraction.md** - Schema-Level 抽象完成状态
- **stage3_variable_abstraction_summary.md** - 变量抽象实现总结
- **stage3_optimization_opportunities.md** - 优化机会（Priority 1-3 已完成）

### 🔴 问题与限制文档
- **CRITICAL_DESIGN_ISSUES.md** ⚠️ - 需要修复的关键设计问题
- **stage3_production_limitations.md** - 生产环境限制（长期）
- **stage3_technical_debt.md** - 技术债务追踪

### ✅ 问题解决记录（历史参考）
- **issue_ab_resolution.md** - Issue A & B 解决报告
- **variable_abstraction_soundness_analysis.md** - 最初的 soundness 分析
- **constant_variable_distinction_analysis.md** - 常量变量区分分析

### 📊 其他分析文档
- **state_count_analysis.md** - 状态数分析
- **object_list_propagation_path.md** - object_list 传播路径
- **pddl_vs_agentspeak_variables.md** - PDDL vs AgentSpeak 变量对比
- **stage3_variable_abstraction_design.md** - 变量抽象设计（原始）

### 🚫 不需要查看的文档
- **nl_instruction_template.md** - LTL 指令模板（Stage 1 相关）

---

## 🧭 针对不同角色的阅读建议

### 对于新加入的开发者
**目标**: 快速上手，了解系统
1. 先走 **路径 1: 快速了解（15分钟）**
2. 运行测试验证环境
3. 阅读 **stage3_backward_planning_design.md** 前半部分
4. 查看 **CRITICAL_DESIGN_ISSUES.md** 了解当前需要修复的问题

### 对于准备修复问题的开发者
**目标**: 修复 object-specific plans 问题
1. 详细阅读 **CRITICAL_DESIGN_ISSUES.md**
2. 阅读 **constant_variable_distinction_analysis.md** 理解背景
3. 查看 **issue_ab_resolution.md** 了解已解决问题的方法
4. 开始实施 Priority 1 修复

### 对于优化性能的开发者
**目标**: 进一步优化系统
1. 阅读 **stage3_optimization_opportunities.md** 了解已完成和待做的优化
2. 阅读 **stage3_technical_debt.md** 了解技术债务
3. 阅读 **stage3_production_limitations.md** 了解长期限制
4. 选择合适的优化方向（Priority 4 或 Heuristic Search）

### 对于准备发表论文的研究者
**目标**: 理解系统、准备材料、说明限制
1. 完整阅读 **stage3_backward_planning_design.md**
2. 阅读 **stage3_schema_level_abstraction.md** 了解核心创新
3. 阅读 **stage3_production_limitations.md** 准备 Limitations 章节
4. 阅读 **stage3_optimization_opportunities.md** 准备 Future Work 章节
5. 查看 **CRITICAL_DESIGN_ISSUES.md** 并在论文中说明已知问题

---

## 🔧 当前开发重点

### 立即需要做的（Critical）
1. **修复 Object-Specific Goal Plans**
   - 参考: `CRITICAL_DESIGN_ISSUES.md` Issue 1
   - 目标: 生成 parameterized goal plans
   - 估计时间: 2-3 天

2. **实现真正的类型系统**
   - 参考: `CRITICAL_DESIGN_ISSUES.md` Issue 2
   - 目标: 正确的类型推断和验证
   - 估计时间: 1-2 天

3. **修复变量命名不一致**
   - 参考: `CRITICAL_DESIGN_ISSUES.md` Issue 3
   - 目标: 统一使用 `?v{i}` 或实现结构匹配
   - 估计时间: 半天

### 中期改进（Important）
1. **增强测试覆盖**
   - 多类型 domain 测试
   - 大写/小写对象混合测试
   - 常量处理边界情况

2. **代码清理**
   - 移除未使用的 legacy code
   - 更新过时的注释
   - 统一代码风格

### 长期优化（Nice-to-have）
1. **Heuristic Search** (A* with delete relaxation)
2. **Symmetry Reduction**
3. **多 domain 支持**
4. **与 Fast Downward 集成**

---

## 🎓 学习路径建议

### 第一周: 理解系统
- [ ] 完成路径 1: 快速了解
- [ ] 运行所有测试，理解输出
- [ ] 阅读 stage3_backward_planning_design.md
- [ ] 浏览源码: backward_planner_generator.py, forward_planner.py

### 第二周: 深入实现
- [ ] 完成路径 2: 深入理解
- [ ] 阅读所有优化和限制文档
- [ ] 理解 variable abstraction 实现
- [ ] 理解 schema-level caching 机制

### 第三周: 问题诊断
- [ ] 阅读 CRITICAL_DESIGN_ISSUES.md
- [ ] 理解 object-specific vs parameterized 问题
- [ ] 研究类型系统问题
- [ ] 准备修复方案

### 第四周: 开始贡献
- [ ] 选择一个 Priority 1 问题
- [ ] 实施修复
- [ ] 编写测试
- [ ] 提交 Pull Request

---

## 📝 文档维护说明

### 文档更新规则
1. **重大功能完成时**: 更新对应的实现状态文档
2. **发现新问题时**: 在 CRITICAL_DESIGN_ISSUES.md 或 stage3_technical_debt.md 记录
3. **问题解决后**: 在对应文档标记 ✅ RESOLVED，考虑创建 resolution 文档
4. **性能优化后**: 更新 stage3_optimization_opportunities.md

### 文档清理建议
考虑归档或删除以下历史分析文档（已被 resolution 文档取代）:
- `variable_abstraction_soundness_analysis.md` (被 issue_ab_resolution.md 取代)
- 但保留以供参考历史问题的分析过程

---

## 🔗 快速链接

### 最重要的5个文档
1. `stage3_backward_planning_design.md` - 设计圣经
2. `CRITICAL_DESIGN_ISSUES.md` - 当前问题
3. `stage3_schema_level_abstraction.md` - 核心创新
4. `stage3_production_limitations.md` - 已知限制
5. `stage3_optimization_opportunities.md` - 优化状态

### 测试文件
- `tests/stage3_code_generation/test_integration_backward_planner.py` - 集成测试
- `tests/stage3_code_generation/test_scalability.py` - 可扩展性测试
- `tests/test_constant_handling.py` - 常量处理测试

---

**最后更新**: 2025-11-11
**维护者**: 项目团队
**反馈**: 发现文档问题请更新此文件
