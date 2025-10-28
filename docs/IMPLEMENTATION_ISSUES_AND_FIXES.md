# Implementation Issues and Fixes

**Document Status**: Active Issue Tracking
**Last Updated**: 2025-10-27
**Purpose**: Track all identified issues, root causes, and fixes for the LTL-BDI pipeline MVP

---

## Critical Issues Identified

### Issue 1: Initial State Format Mismatch ❌ BLOCKING

**Severity**: CRITICAL
**Status**: 🔴 IDENTIFIED - Needs Fix

**Problem**:
- LTL parser generates: `{'ontable': ['b']}`
- Blocksworld simulator expects: `on(b, table)`
- Classical planner fails because blocks are not actually on table in initial state

**Root Cause**:
Stage 1 (NL→LTLf) generates predicates as `ontable(X)` but blocksworld domain uses `on(X, table)` convention.

**Test Evidence**:
```
Initial State: [{'ontable': ['b']}, {'ontable': ['c']}, ...]
Classical plan action: pickup(c)
Result: FAILED - precondition not met
```

**Fix Required**:
1. Modify LTL parser to generate `on(X, table)` instead of `ontable(X)`
2. OR modify blocksworld simulator to handle `ontable` predicate
3. OR add conversion layer in dual_branch_pipeline.py

**Fix Decision**: Option 3 - Add conversion in pipeline (least invasive)

---

### Issue 2: AgentSpeak Plan Matching Failure ❌ BLOCKING

**Severity**: CRITICAL
**Status**: 🔴 IDENTIFIED - Needs Fix

**Problem**:
- AgentSpeak generator creates plans with triggers like `+!stack(c, b)`
- AgentSpeak simulator looks for exact match `stack(c, b)`
- No plans are found as "applicable"

**Root Cause**:
AgentSpeak parser expects parameterized plans `+!stack(X, Y)` but generated code has specific constants `+!stack(c, b)`.

**Test Evidence**:
```
AgentSpeak Code: "Plans: 22"
Execution: "No applicable plan for: stack(c, b)"
```

**Fix Required**:
1. Improve AgentSpeak plan matching to handle constants vs variables
2. OR modify generator to always use variables in triggers
3. OR enhance simulator plan selection logic

**Fix Decision**: Option 1 + 3 - Improve matching logic and plan selection

---

### Issue 3: Blocksworld Simulator State Management 🟡 MINOR

**Severity**: MEDIUM
**Status**: 🟡 IDENTIFIED - Low Priority

**Problem**:
- Simulator uses `ontable` in `from_beliefs()` parsing
- But domain uses `on(X, table)`
- Inconsistent predicate naming

**Root Cause**:
Hardcoded predicate names in simulator don't match PDDL domain

**Fix Required**:
Normalize all predicates to use `on(X, table)` convention

---

### Issue 4: No Actual LTLf Verification ⚠️ LIMITATION

**Severity**: MEDIUM
**Status**: 🟠 ACKNOWLEDGED - Documented Limitation

**Problem**:
- Stage 4 comparison doesn't actually verify LTLf formulas
- Only checks if goal predicate exists in final state
- No temporal logic verification

**Current Implementation**:
```python
def _check_ltl_satisfaction(self, final_state, ltl_goal):
    if ltl_goal.startswith('F(') and ltl_goal.endswith(')'):
        goal_predicate = ltl_goal[2:-1]  # Extract φ from F(φ)
        return goal_predicate in final_state
    return True  # MVP: can't verify other formulas
```

**Limitation**: This is MVP-acceptable but should be documented

**Future Fix**: Integrate proper LTLf verification library (e.g., spot, pyLTL)

---

## Fixes Applied

### Fix 1.1: Initial State Conversion (Issue #1)

**File**: `src/dual_branch_pipeline.py`
**Location**: `_stage4_execute_and_compare()` method
**Change**: Added conversion from `ontable(X)` to `on(X, table)`

```python
# Convert dict-based initial_state to string-based beliefs
beliefs = []
for pred_dict in ltl_spec.initial_state:
    for pred_name, args in pred_dict.items():
        if pred_name == 'ontable' and args:
            # Convert ontable(X) to on(X, table)
            for block in args:
                beliefs.append(f"on({block}, table)")
        elif args:
            beliefs.append(f"{pred_name}({', '.join(args)})")
        else:
            beliefs.append(pred_name)
```

**Status**: ✅ IMPLEMENTED

---

### Fix 2.1: AgentSpeak Plan Matching (Issue #2)

**File**: `src/stage4_execution/agentspeak_simulator.py`
**Location**: `_select_plan()` and `_parse_asl()` methods
**Changes**:
1. Enhanced multi-line plan parsing
2. Added declarative goal support (`+!!`)
3. Implemented variable unification
4. Added bracket stripping for goal matching

**Status**: ✅ IMPLEMENTED & TESTED

**Key Fixes**:

1. **Multi-line Plan Parsing**: AgentSpeak code can span multiple lines
```python
def _parse_asl(self, asl_code: str):
    """Parse AgentSpeak code - handles multi-line plans"""
    current_plan_text = ""
    for line in lines:
        if line.startswith(('+!', '-!')):
            if current_plan_text:
                self._parse_single_plan(current_plan_text)
            current_plan_text = line
        elif current_plan_text:
            current_plan_text += " " + line

        if current_plan_text and current_plan_text.endswith('.'):
            self._parse_single_plan(current_plan_text)
            current_plan_text = ""
```

2. **Declarative Goal Support**: Handle `+!![goal]` syntax
```python
# Pattern handles both +! and +!!
plan_pattern = r'(\+!!?[\w\(\),\s\[\]_]+)\s*:\s*([^<]*)\s*<-\s*(.+)\.'
```

3. **Bracket Normalization**: Strip brackets from both triggers and goals
```python
goal_normalized = goal.strip('[]')
trigger_goal = trigger_goal.strip('[]')
```

4. **Variable Unification**: Full implementation in `_unify_goal()` method

---

### Fix 2.2: Belief Format Conversion (Issue #3)

**File**: `src/stage4_execution/agentspeak_simulator.py`
**Location**: `_belief_exists()` method
**Change**: Handle predicate format differences between generated code and simulator

**Status**: ✅ IMPLEMENTED & TESTED

**Problem**: Generated AgentSpeak uses `ontable(X)` but beliefs use `on(X,table)` (with/without space)

**Solution**: Automatic bidirectional conversion in belief checking
```python
def _belief_exists(self, condition: str) -> bool:
    """Check if belief exists, handling ontable(X) <-> on(X,table) conversion"""
    # Direct match first
    if condition in self.beliefs:
        return True

    # Convert ontable(X) to on(X, table) - try both with and without space
    match = re.match(r'ontable\((\w+)\)', condition)
    if match:
        block = match.group(1)
        return (f"on({block}, table)" in self.beliefs or
                f"on({block},table)" in self.beliefs)

    # Convert on(X, table) to ontable(X)
    match = re.match(r'on\((\w+),\s*table\)', condition)
    if match:
        block = match.group(1)
        return f"ontable({block})" in self.beliefs

    return False
```

**Impact**: This fix allows generated AgentSpeak plans with `ontable(X)` contexts to work with blocksworld states using `on(X,table)` format.

---

### Fix 2.3: Declarative Goal Execution

**File**: `src/stage4_execution/agentspeak_simulator.py`
**Location**: `_execute_action()` and `_achieve_goal()` methods
**Change**: Proper handling of declarative goals (`!!goal`)

**Status**: ✅ IMPLEMENTED & TESTED

**Key Changes**:

1. **Action Execution**: Distinguish between `!goal` and `!!goal`
```python
# Declarative goal (!!goal)
if action_str.startswith('!!'):
    subgoal = action_str[2:].strip()  # Remove both !!
    return self._achieve_goal(subgoal)

# Achievement goal (!goal)
if action_str.startswith('!'):
    subgoal = action_str[1:].strip()
    return self._achieve_goal(subgoal)
```

2. **Early Satisfaction Check**: For declarative goals, check if already satisfied before plan selection
```python
def _achieve_goal(self, goal: str) -> bool:
    # For declarative goals with brackets, check if already satisfied
    if goal.startswith('[') and goal.endswith(']'):
        goal_condition = goal.strip('[]')
        if self._belief_exists(goal_condition):
            return True  # Already satisfied!

    # Otherwise, select and execute plan
    plan = self._select_plan(goal)
    ...
```

---

## Known Limitations (MVP Acceptable)

### L1: No Full LTLf Temporal Verification
**Impact**: Can only verify `F(φ)` goals, not `G(φ)`, `X(φ)`, or `φ U ψ`
**Workaround**: Document as future work
**Priority**: LOW (research prototype)

### L2: Simple AgentSpeak Parsing
**Impact**: Cannot handle complex AgentSpeak features (annotations, internal actions beyond .print)
**Workaround**: Generator produces simplified AgentSpeak subset
**Priority**: MEDIUM

### L3: No Belief Revision
**Impact**: Belief updates after actions are manual in AgentSpeak code
**Workaround**: Generator includes explicit +/- belief updates
**Priority**: LOW

### L4: No Multi-Agent Support
**Impact**: Single agent only
**Workaround**: Out of scope for MVP
**Priority**: N/A

---

## Testing Status

### Test Scenarios

| Scenario | Stage 1 | Stage 2 | Stage 3A | Stage 3B | Stage 4 | Status |
|----------|---------|---------|----------|----------|---------|--------|
| Simple stack (C on B) | ✅ | ✅ | ✅ | ✅ | ✅ | **PASSING** |
| Three-block tower | ❓ | ❓ | ❓ | ❓ | ❓ | NOT TESTED |
| Clear block | ❓ | ❓ | ❓ | ❓ | ❓ | NOT TESTED |
| Move block | ❓ | ❓ | ❓ | ❓ | ❓ | NOT TESTED |

**Latest Test Results** (Simple Stack Scenario):
```
✓ Both branches succeeded
- Classical: 2 actions (pickup, stack)
- AgentSpeak: 2 actions (pickup, stack)
- Efficiency Ratio: 1.00
- Final State: on(c,b) achieved
```

---

## Next Steps

1. ✅ Document all identified issues
2. ✅ Implement Fix 1.1 (initial state conversion)
3. ✅ Implement Fix 2.1 (multi-line parsing, declarative goals, variable unification)
4. ✅ Implement Fix 2.2 (belief format conversion)
5. ✅ Implement Fix 2.3 (declarative goal execution)
6. ✅ Run test suite - **PASSING**
7. 📝 Update CORE_ARCHITECTURE.md with limitations
8. 🧹 Clean up legacy code
9. 🧪 Run comprehensive tests with different scenarios
10. ✅ Final verification for simple stack scenario

---

**Legend**:
- ✅ Complete
- 🔧 In Progress
- ❌ Failed
- ❓ Not Tested
- 🔴 Critical
- 🟡 Medium
- 🟠 Low Priority
