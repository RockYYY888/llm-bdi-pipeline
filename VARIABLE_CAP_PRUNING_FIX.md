# Variable Cap Pruning Fix

## Problem Identified

States with too many variables were being generated despite `max_objects` cap:

```
State(depth=4, preds=[clear(?v3), clear(?v5), ...])  # 5 variables when max_objects=3
```

## Root Cause

The pruning logic was using `next_var_number` (the counter) instead of counting **actual unique variables** in the state:

```python
# OLD CODE - WRONG
if next_var_number > self.max_objects:
    return []  # Prune
```

Problem: If variables are `?v1, ?v3, ?v5`, then `next_var_number=5` but we only have **3 unique variables**.

## Solution

Count **actual unique variables** in the state's predicates:

```python
# NEW CODE - CORRECT
if self.max_objects is not None:
    # Count ACTUAL unique variables in the new state (not just max var number)
    unique_vars = set()
    for pred in new_predicates:
        for arg in pred.args:
            if arg.startswith('?v'):
                unique_vars.add(arg)

    if len(unique_vars) > self.max_objects:
        # Too many variables for available objects - PRUNE
        return []
```

## Verification

### Test 1: Variable Cap Pruning Test

```
Goal: on(a, b)
max_objects: 3

Results:
✓ ALL states respect the variable cap (max 3 variables)

Max variables by depth:
  Depth 0: max 0 variables ✓
  Depth 1: max 1 variables ✓
  Depth 2: max 2 variables ✓
  Depth 3: max 3 variables ✓
```

### Test 2: Complete Stage 3 Test

```
Test: test_1_simple_goal_2_blocks
Status: PASSED in 6.57 seconds ✓
```

### Test 3: Termination Analysis

Before fix:
- **Massive state explosion** with states having 5+ variables

After fix:
- **319 states** generated (much reduced from before)
- All states respect variable cap
- State space significantly pruned

## Impact

1. **Correctness**: Variable cap now enforced correctly
2. **Performance**: Significant reduction in state space
3. **Test Success**: Complete stage 3 test now passes
4. **Scalability**: Search terminates in reasonable time

## Modified File

- `src/stage3_code_generation/backward_search_refactored.py` (lines 616-628)

## Status

✅ **FIXED**: Variable cap pruning now working correctly
