# Critical Design Issues - System Analysis

## Date: 2025-11-10
## Last Updated: 2025-11-11
## Status: ISSUE 1 RESOLVED ✅ - Other issues remain

---

## Executive Summary

**UPDATE (2025-11-11)**: Issue 1 has been RESOLVED ✅. The system now correctly generates parameterized AgentSpeak code.

### Issue Status

1. ✅ **Object-Specific Goal Plans** - RESOLVED
   - System generates parameterized plans with AgentSpeak variables (V0, V1, etc.)
   - Plans work for arbitrary objects of correct type
   - Verified by tests: test_parameterization_check.py, test_parameterization_validation.py

2. ❌ **Incomplete Type System** - HIGH PRIORITY (still exists)
   - All objects assigned to first domain type
   - Cannot handle multi-type domains

3. ⚠️ **Variable Naming Inconsistency** - MEDIUM (needs verification)
   - Normalization uses `?arg0` vs planner uses `?v0`
   - May impact goal inference

### Key Finding (UPDATED)

**Currently Generates (CORRECT)** ✅:
```asl
+!clear_V0_and_handempty_and_on_V0_V1 : clear(V0) & handempty & on(V0, V1) <-
    .print("Goal clear_V0_and_handempty_and_on_V0_V1 already achieved!").
```

**Uses AgentSpeak variables (V0, V1)** - works for ANY objects of correct type!

---

## Issue 1: Object-Specific Goal Plans ✅ RESOLVED

### Resolution Date: 2025-11-11

### Resolution Summary

**The system now correctly generates PARAMETERIZED goal plans using AgentSpeak variables.**

### Current Behavior (CORRECT)

```asl
/* Goal Achievement Plans for: clear_V0_and_handempty_and_on_V0_V1 */
+!clear_V0_and_handempty_and_on_V0_V1 : holding(V0) & on(V0, V1) <-
    !clear(V1);
    !put_on_block(V0, V1);
    !clear_V0_and_handempty_and_on_V0_V1.

+!clear_V0_and_handempty_and_on_V0_V1 : clear(V0) & handempty & on(V0, V1) <-
    .print("Goal clear_V0_and_handempty_and_on_V0_V1 already achieved!").

-!clear_V0_and_handempty_and_on_V0_V1 : true <-
    .print("Failed to achieve goal clear_V0_and_handempty_and_on_V0_V1");
    .fail.
```

### Verification

Tests confirm parameterization works correctly:
- ✅ `tests/test_parameterization_check.py` - Verifies V0, V1 variables used
- ✅ `tests/stage3_code_generation/test_parameterization_validation.py` - Comprehensive analysis
- ✅ All goal plans use AgentSpeak variables (V0, V1, B1, B2, etc.)
- ✅ Action plans use variables correctly
- ✅ Initial beliefs remain concrete (as expected)

### How It Works

**File**: `src/stage3_code_generation/agentspeak_codegen.py`

Goal generation uses:
```python
# In _get_parameterized_goal_pattern() (line 105-128):
goal_preds = list(self.graph.goal_state.predicates)  # Keep variables
return goal_preds[0].to_agentspeak(convert_vars=True)  # Convert PDDL vars to AgentSpeak

# In _generate_success_plan() (line 738-754):
param_goal_pattern = self._get_parameterized_goal_pattern()
context = self.graph.goal_state.to_agentspeak_context(convert_vars=True)
# Uses parameterized pattern with AgentSpeak variables
```

Key method in `state_space.py` (line 57-99):
```python
def to_agentspeak(self, convert_vars: bool = False) -> str:
    if convert_vars:
        args_str = ", ".join(self._pddl_var_to_agentspeak(arg) for arg in self.args)
    # ?v0 → V0, ?v1 → V1
```

### ~~Previous Issues~~ (NO LONGER EXIST)

~~1. **Not Reusable**: Only works for objects `a` and `b`~~
~~2. **Not Generic**: Cannot handle `on(c, d)` or any other object pair~~
~~3. **Violates User Requirement**: "arbitrary objects that fit the type"~~
~~4. **Inconsistent**: Action plans USE variables, but goal plans DON'T~~

**ALL FIXED** ✅

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

### Priority 1: Parameterized Goal Plans ✅ COMPLETED

**Status**: RESOLVED (2025-11-11)

**Implementation**: Already correct in current codebase

**File**: `src/stage3_code_generation/agentspeak_codegen.py`

**Current Code** (CORRECT):
```python
def _generate_success_plan(self):
    # Get parameterized goal pattern (with AgentSpeak variables)
    param_goal_pattern = self._get_parameterized_goal_pattern()
    # Get context condition (also with AgentSpeak variables)
    context = self.graph.goal_state.to_agentspeak_context(convert_vars=True)
    return f"+!{param_goal_pattern} : {context} <- ..."
```

**Result**:
```asl
+!clear_V0_and_handempty_and_on_V0_V1 : clear(V0) & handempty & on(V0, V1) <-
    .print("Goal clear_V0_and_handempty_and_on_V0_V1 already achieved!").
```

**Verification**: Tests pass, confirmed parameterized generation works.

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

### Phase 2: Critical Fixes (In Progress)
- ✅ Implement parameterized goal plans (COMPLETED 2025-11-11)
- ⏳ Fix variable naming (needs verification)
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

**UPDATE (2025-11-11)**: Critical Issue 1 has been RESOLVED ✅

### Current Status

1. ✅ **Object-specific vs Parameterized plans** - RESOLVED
   - System now generates parameterized AgentSpeak code correctly
   - Verified by comprehensive tests
   - Plans work for arbitrary objects of correct type

2. ❌ **Type system incompleteness** (HIGH) - Still needs work
   - All objects assigned to first domain type
   - Cannot handle multi-type domains properly

3. ⚠️ **Variable naming inconsistency** (MEDIUM) - Needs verification
   - May affect goal inference in variable mode

### Remaining Work

**Estimated Effort**: 1-2 days for remaining issues
**Risk**: MEDIUM - System works for single-type domains, needs enhancement for multi-type

### Code Quality

The generated code is now **correct and reusable** for single-type domains (like blocksworld).
For multi-type domains, additional work is needed on the type system.

---

**Document Status**: Issue 1 resolved, remaining issues documented
**Next Step**: Verify variable naming issue, then implement proper type system
