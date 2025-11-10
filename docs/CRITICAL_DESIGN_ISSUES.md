# Critical Design Issues - System Analysis

## Date: 2025-11-10
## Status: CRITICAL - Requires immediate refactoring

---

## Executive Summary

The current system has **fundamental design flaws** that prevent it from generating correct, reusable AgentSpeak code. The code is **object-specific** rather than **parameterized**, meaning it cannot handle arbitrary objects of the correct type.

### Key Finding

**Generated (WRONG)**:
```asl
+!on(a, b) : on(a, b) <- .print("Goal achieved").
```

**Should Generate (CORRECT)**:
```asl
+!on(X, Y) : on(X, Y) <- .print("Goal achieved for ", X, " and ", Y).
```

**Impact**: Plans only work for specific object instances mentioned during generation, not for arbitrary objects.

---

## Issue 1: Object-Specific Goal Plans ❌ CRITICAL

### Current Behavior

```asl
/* Goal Achievement Plans for: on(a, b) */
+!on(a, b) : on(a, b) <-
    .print("Goal on(a, b) achieved").

-!on(a, b) : true <-
    .print("Failed to achieve on(a, b)").
```

### Problems

1. **Not Reusable**: Only works for objects `a` and `b`
2. **Not Generic**: Cannot handle `on(c, d)` or any other object pair
3. **Violates User Requirement**: "arbitrary objects that fit the type"
4. **Inconsistent**: Action plans USE variables (`+!pick_up(B1, B2)`), but goal plans DON'T

### Root Cause

**File**: `src/stage3_code_generation/agentspeak_codegen.py`

Goal generation uses:
```python
# In _generate_success_plan():
goal_preds = [self._instantiate_predicate(pred) for pred in self.graph.goal_state.predicates]
# ← instantiate converts ?arg0 → a, ?arg1 → b
```

This **instantiates** variables to concrete objects, creating object-specific plans.

### Correct Approach

Goal plans should be **param**eterized:

```asl
/* Generic goal achievement - works for ANY blocks */
+!on(X, Y) : on(X, Y) <-
    .print("Goal on(", X, ", ", Y, ") achieved").

-!on(X, Y) : true <-
    .print("Failed to achieve on(", X, ", ", Y, ")").
```

Then **invoke** with concrete arguments:
```asl
!on(a, b)  // Uses the generic +!on(X, Y) plan with X=a, Y=b
!on(c, d)  // Uses the SAME plan with X=c, Y=d
```

---

## Issue 2: Incomplete Type System ❌ HIGH PRIORITY

### Current Implementation

**File**: `src/stage3_code_generation/variable_normalizer.py:81-97`

```python
def _infer_object_types(self) -> Dict[str, str]:
    """Infer object types from domain"""
    # Simple implementation: assign all objects to first domain type
    if self.domain.types:
        default_type = self.domain.types[0]
    else:
        default_type = "object"

    return {obj: default_type for obj in self.object_list}
```

### Problems

1. **No Real Type Inference**: All objects assigned to first type
2. **Ignores PDDL Type Information**: Domain has `(:types block)` but not used properly
3. **No Type Validation**: Cannot verify objects match predicate signatures
4. **No Multi-Type Support**: Cannot handle domains with multiple types

### Example Domain with Multiple Types

```pddl
(:types
    block location robot - object
)

(:predicates
    (at ?r - robot ?l - location)
    (on ?b1 ?b2 - block)
    (holding ?r - robot ?b - block)
)
```

**Current system would fail** because:
- Cannot distinguish robot vs block vs location
- Would assign all objects to first type (block)
- Cannot validate `at(robot1, loc1)` requires robot + location

### Required Fix

1. **Parse object-type declarations** from problem file
2. **Validate predicate arguments** against type signatures
3. **Type-check during normalization**
4. **Generate type guards** in AgentSpeak code

---

## Issue 3: Variable Naming Inconsistency

### The Problem

**Normalization**: Uses `?arg0, ?arg1, ...`
**Forward Planner**: Uses `?v0, ?v1, ...`

**Impact**:
- Goal inference fails (predicates don't match)
- Variable mode explores MORE states than grounded mode (1152 vs 1093)
- Less constrained goals → broader exploration

### Evidence

```python
# Normalization
on(a, b) → on(?arg0, ?arg1)

# Forward planning
grounded_action('put-on-block', ['?v0', '?v1'])
effects = [on(?v0, ?v1), ...]  # Uses ?v0, ?v1

# Matching
on(?arg0, ?arg1) == on(?v0, ?v1)  # FALSE! ❌
```

### Fix Options

**Option A** (Quick): Standardize on `?v{i}`
```python
obj_to_var[arg] = f"?v{var_counter}"  # Change ?arg → ?v
```

**Option B** (Proper): Implement structural matching that ignores variable names

---

## Issue 4: AgentSpeak Variable Format ❌ MEDIUM

### Current Output

```asl
+!pick_up(B1, B2) : handempty & clear(B1) & on(B1, B2) <-
```

This is **correct** - uses uppercase AgentSpeak variables.

### But Goal Plans Output

```asl
+!on(a, b) : on(a, b) <-
```

This uses **concrete objects** instead of **variables**.

### Inconsistency

- Action plans: Parameterized ✅
- Goal plans: Concrete ❌

Both should be parameterized!

---

## Issue 5: Lack of Type Guards

### Missing Feature

Generated plans have NO type checking:

```asl
+!on(X, Y) : on(X, Y) <- ...
```

**Problem**: Nothing prevents calling `!on(robot1, location1)` even though both must be blocks.

### Should Generate

```asl
+!on(X, Y) : on(X, Y) & is_block(X) & is_block(Y) <-
    .print("Achieving on for blocks ", X, " and ", Y).

+!on(X, Y) : true <-
    .print("ERROR: on/2 requires both arguments to be blocks");
    .print("  Got: X=", X, ", Y=", Y);
    .fail.
```

Or use Jason's built-in type system if available.

---

## Issue 6: object_list Completeness Not Verified

### Current Assumption

`object_list` from Stage 1 LLM is assumed complete and correct.

### Risk

If LLM misses objects or misidentifies them:
- Normalization will be incorrect
- Some objects won't be abstracted
- Schema caching won't work properly

### Mitigation Needed

**Stage 1 Post-Processing**:
```python
def validate_and_complete_objects(spec: LTLSpecification, domain: PDDLDomain):
    """Ensure object_list is complete"""
    # Extract all mentioned objects from formulas
    mentioned_objects = extract_objects_from_formulas(spec.formulas)

    # Extract from grounding map
    mapped_objects = extract_objects_from_grounding_map(spec.grounding_map)

    # Check completeness
    all_objects = mentioned_objects | mapped_objects
    missing = all_objects - set(spec.objects)

    if missing:
        logger.warning(f"Objects mentioned but not in list: {missing}")
        # Auto-add or prompt user
        spec.objects.extend(missing)

    return spec
```

---

## Issue 7: Legacy Code Mixed with New Code

### Problem

Legacy FOND planning code still exists alongside new backward planning code.

**Location**: `src/legacy/fond/`

### Risks

1. **Confusion**: Which code is actually used?
2. **Maintenance Burden**: Need to maintain both
3. **Potential Bugs**: Might accidentally use wrong code path

### Files to Review

```
src/legacy/fond/
  ├── domains/
  ├── planner/
  └── ...
```

**Action Required**: Identify which legacy components are still needed, remove the rest.

---

## System-Wide Recommendations

### Priority 1: Parameterized Goal Plans (CRITICAL)

**Change**: Generate parameterized goal achievement plans

**File**: `src/stage3_code_generation/agentspeak_codegen.py`

**Before**:
```python
def _generate_success_plan(self):
    # Instantiate to concrete objects
    goal_preds = [self._instantiate_predicate(pred) for pred in ...]
    return f"+!{goal_name} : {context} <- ..."
```

**After**:
```python
def _generate_success_plan(self):
    # Keep as variables
    goal_preds = self.graph.goal_state.predicates  # With variables
    # Convert PDDL vars (?arg0) to AgentSpeak vars (Arg0)
    as_goal_preds = [self._pddl_to_agentspeak_vars(p) for p in goal_preds]
    return f"+!{goal_pattern} : {context} <- ..."
```

**Result**:
```asl
+!on(X, Y) : on(X, Y) <- .print("Goal achieved").
```

### Priority 2: Implement Real Type System

**Required Components**:

1. **Type Parser** - Extract types from domain
2. **Type Validator** - Check object-predicate compatibility
3. **Type Guards** - Generate type checks in AgentSpeak
4. **Multi-Type Support** - Handle domains with multiple types

### Priority 3: Fix Variable Naming

**Quick Fix**: Standardize on `?v{i}` everywhere

**Location**: `variable_normalizer.py:186`
```python
obj_to_var[arg] = f"?v{var_counter}"  # Was: f"?arg{var_counter}"
```

### Priority 4: Add Completeness Checks

**Stage 1 Enhancement**: Validate object_list completeness
**Stage 3 Enhancement**: Verify all objects in predicates are in object_list

### Priority 5: Clean Up Legacy

**Action**: Remove unused legacy code after verification

---

## Testing Requirements

### New Tests Needed

1. **Parameterized Plan Test**
   - Generate code with parameterized goals
   - Test with different object sets
   - Verify same plan works for all

2. **Type System Test**
   - Multi-type domain
   - Validate type checking
   - Ensure type guards work

3. **Completeness Test**
   - Incomplete object_list
   - Verify detection and correction

4. **Integration Test**
   - End-to-end from NL to AgentSpeak
   - Multiple object sets
   - Verify reusability

---

## Migration Path

### Phase 1: Analysis (Current)
- ✅ Identify all issues
- ✅ Document problems
- ✅ Prioritize fixes

### Phase 2: Critical Fixes (Next)
- ⏳ Implement parameterized goal plans
- ⏳ Fix variable naming
- ⏳ Add basic type checking

### Phase 3: Type System (Following)
- ⏳ Full type inference
- ⏳ Type validation
- ⏳ Type guards

### Phase 4: Cleanup (Final)
- ⏳ Remove legacy code
- ⏳ Comprehensive testing
- ⏳ Documentation update

---

## Conclusion

The current system has fundamental design flaws that prevent it from generating correct, reusable AgentSpeak code. **Immediate refactoring is required** to address:

1. **Object-specific vs Parameterized plans** (CRITICAL)
2. **Type system incompleteness** (HIGH)
3. **Variable naming inconsistency** (MEDIUM)

Without these fixes, the generated code will **not work for arbitrary objects**, violating the core requirement.

**Estimated Effort**: 2-3 days for Phase 2 critical fixes
**Risk**: HIGH if not addressed - system produces incorrect code

---

**Document Status**: Draft for review and implementation planning
**Next Step**: Begin Phase 2 implementation
