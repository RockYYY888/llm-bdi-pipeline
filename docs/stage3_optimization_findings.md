# Stage 3 Optimization Findings

**Date**: November 7, 2025
**Branch**: `claude/stage3-backward-planning-codegen-011CUrcEeLPqznLN6dTUFiD2`

## Summary

During optimization and strict verification against the design specification, several critical issues were discovered and fixed. However, a fundamental architectural issue remains that requires a different approach.

---

## Fixes Implemented

### 1. **Transition Direction Reversal** (forward_planner.py:146-148)

**Issue**: Transitions were stored as `current_state -> new_state` (exploration direction), but for plan generation we need `new_state -> goal_state` (execution direction).

**Fix**: Reversed transition storage:
```python
transition = StateTransition(
    from_state=final_state,      # Reversed
    to_state=current_state,       # Reversed
    ...
)
```

**Impact**: Path finding now correctly discovers 660 states with paths to goal (vs 0 before).

---

### 2. **Backward Effect Application** (forward_planner.py:304-324)

**Issue**: Effects were being applied FORWARD, but for backward planning (regression), they must be applied in REVERSE.

**Fix**:
```python
for effect_atom in effect_branch:
    if effect_atom.is_add:
        # Add effect in forward = REMOVE in backward
        new_predicates.discard(effect_atom.predicate)
    else:
        # Delete effect in forward = ADD in backward
        new_predicates.add(effect_atom.predicate)

# Add preconditions to new state (required before action)
for precond in preconditions:
    if not precond.negated:
        new_predicates.add(precond)
```

**Impact**: Correct regression-based predecessor state generation.

---

### 3. **Goal State Filtering** (agentspeak_codegen.py:431-436)

**Issue**: Many explored states contain the goal predicate (from regression), leading to nonsensical plans like "if goal already true, achieve goal".

**Fix**:
```python
# Skip states where goal is already satisfied
goal_predicates = self.graph.goal_state.predicates
if goal_predicates.issubset(state.predicates):
    continue  # Don't generate plan for this state
```

**Impact**: Filters out states where goal is trivially satisfied.

---

###4. **Removed Problematic Precondition Subgoals** (agentspeak_codegen.py:484-491)

**Issue**: Generated circular subgoals and impossible preconditions (e.g., `!on(a,b)` as subgoal when trying to achieve `on(a,b)`).

**Fix**: Removed precondition subgoal generation entirely. With complete path finding, preconditions are guaranteed to be met by the path.

```python
# Build plan body
body_lines = []
body_lines.append(action_goal)           # Action goal invocation
body_lines.append(f"!{self.goal_name}")  # Recursive goal check
```

**Impact**: Simpler, cleaner plans without circular dependencies.

---

## Critical Finding: Fundamental Architectural Issue

### The Problem

After implementing all fixes, testing revealed that **nearly all explored states (31/31 at depth=1) still contain the goal predicate `on(a,b)`**, causing them all to be filtered out and resulting in no plans being generated.

### Root Cause

This is actually **CORRECT** behavior for regression-based backward planning:

1. **Goal state**: `{on(a, b)}`

2. **Regression with `pick-up(a, b)`**:
   - Preconditions: `handempty`, `clear(a)`, `on(a, b)`
   - Effects: `+holding(a)`, `+clear(b)`, `-handempty`, `-clear(a)`, `-on(a, b)`
   - Predecessor state: remove adds, add deletes, add preconditions
   - Result: `{on(a, b), handempty, clear(a)}`

3. **The predicate `on(a, b)` remains** because it's a **precondition** of the action that deletes it!

4. To execute `pick-up(a, b)` and delete `on(a, b)`, you need `on(a, b)` to be true BEFORE the action.

5. Therefore, the predecessor state must also contain `on(a, b)`.

### Why This Breaks Plan Generation

- Backward exploration from goal naturally produces states that contain the goal predicate
- These states represent valid predecessors from which the goal can be maintained/reached
- But for AgentSpeak plan generation, we want states where the goal is NOT yet achieved
- Filtering out states with goal predicates leaves almost nothing to generate plans from

### Comparison: What We Have vs What We Need

**Current Approach (Regression-based state-space exploration)**:
```
Goal: {on(a,b)}
    ↓ regress with pick-up(a,b)
{on(a,b), handempty, clear(a)}  ← Still has goal!
    ↓ regress with other actions
{on(a,b), handempty, clear(a), ...}  ← Still has goal!
```

**What We Actually Need (Goal-directed forward planning)**:
```
Initial: {ontable(a), ontable(b), clear(a), clear(b), handempty}
    ↓ find action to achieve on(a,b)
Action: put-on-block(a, b)
    ↓ what are preconditions?
Subgoals: holding(a), clear(b)
    ↓ how to achieve holding(a)?
Action: pick-up-from-table(a)
    ↓ generate plan
Plan: !pick_up_from_table(a); !put_on_block(a,b); !on(a,b).
```

---

## Proposed Solutions

### Option 1: True Goal Regression (Recommended)

Instead of exploring full state space backward, use **goal regression tree**:

1. **Start with goal**: `on(a, b)`
2. **Find achieving actions**: Actions with `+on(a, b)` in effects → `put-on-block(a, b)`
3. **Extract preconditions**: `holding(a)`, `clear(b)`
4. **Recursively regress**: Treat preconditions as subgoals
5. **Generate plans**: Direct mapping from regression tree to AgentSpeak

**Advantages**:
- Simpler algorithm
- Natural mapping to AgentSpeak goal hierarchy
- No state-space explosion
- Plans only for states where goal NOT achieved

**Implementation**: ~200 lines, replace `forward_planner.py`

---

### Option 2: Forward Planning from Initial State

Use traditional forward search with heuristics:

1. Define initial state (e.g., all blocks on table)
2. Forward search toward goal using A* or best-first
3. Generate plans from found paths

**Advantages**:
- Well-understood algorithm
- Guaranteed correct plans

**Disadvantages**:
- Requires initial state specification
- Still faces state-space explosion
- Doesn't match "backward planning" design concept

---

### Option 3: Hybrid Approach

Combine backward goal regression with forward verification:

1. Use goal regression to identify relevant actions and subgoals
2. Build dependency graph
3. Generate AgentSpeak plans from dependency graph
4. Optionally validate with forward simulation

**Advantages**:
- Best of both worlds
- Correct and efficient

**Disadvantages**:
- More complex implementation

---

## Recommendation

Implement **Option 1: True Goal Regression** because:

1. ✅ Matches the spirit of "backward planning" in design
2. ✅ Avoids state-space explosion
3. ✅ Natural mapping to AgentSpeak's goal-oriented structure
4. ✅ Simpler than current approach (~200 lines vs ~650 lines)
5. ✅ Generates only meaningful plans (goal not yet achieved)

---

## Code Changes Summary

| File | Lines Changed | Status |
|------|---------------|--------|
| `forward_planner.py` | ~30 | ✅ Fixed (transition reversal, backward effects) |
| `agentspeak_codegen.py` | ~20 | ✅ Fixed (goal filtering, removed precond subgoals) |
| `backward_planner_generator.py` | 0 | ⚠️ Works but produces empty output due to filtering |

---

## Test Results

### Test 1: Simple Goal with depth=1
- **Explored**: 32 states, 58 transitions
- **States with paths to goal**: 32
- **States without goal predicate**: 0 ❌
- **Generated plans**: 0 (all filtered out)

### Test 2: Simple Goal with depth=2
- **Explored**: 661 states, 6966 transitions
- **States with paths to goal**: 660
- **States without goal predicate**: ~0 (estimated)
- **Generated plans**: 0 (all filtered out)

### Conclusion

The fixes are correct for regression-based backward planning, but the approach itself doesn't suit our use case. We need goal regression, not state-space regression.

---

## Next Steps

1. ✅ Commit current optimizations with documentation
2. ⬜ Implement Option 1 (Goal Regression) as new approach
3. ⬜ Test with blocksworld examples
4. ⬜ Integrate with main pipeline
5. ⬜ Update design document to reflect new approach

---

**Author**: Claude (Anthropic)
**Review Status**: Pending user approval for proposed solution
