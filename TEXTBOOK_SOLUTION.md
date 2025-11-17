# The Correct Textbook Solution

## Problem Statement

**Input**: DFA with complex boolean formulas on transitions
```
State 1 -> State 2 [label="(a&b)|c"]
State 1 -> State 1 [label="~((a&b)|c)"]
```

**Goal**: Equivalent DFA with atomic literals only
```
State 1 -> Intermediate [label="a"]
Intermediate -> State 2 [label="b"]
Intermediate -> ... [label="!b"]
...
```

## The Textbook Solution: Shannon Expansion with Parallel Restriction

This is the **standard approach** from:
- Symbolic model checking literature (Bryant 1986, BDD papers)
- MONA tool implementation
- Automata-theoretic verification

### Core Algorithm

```python
def expand_state_transitions(source_state, transitions, var_index=0):
    """
    transitions = [
        (bdd_formula1, target_state1, is_accepting1),
        (bdd_formula2, target_state2, is_accepting2),
        ...
    ]
    """

    # Base case: all variables have been assigned
    if var_index >= len(variables):
        # Find which formula evaluates to TRUE
        for bdd, target, is_accepting in transitions:
            if bdd == BDD.TRUE:
                # Direct transition to this target
                add_transition(source_state, target, "true")
                if is_accepting:
                    mark_accepting(target)
                return
        # No formula true means no transition (or error)
        return

    # Get current variable
    var = variables[var_index]

    # === KEY STEP: Restrict ALL formulas in parallel ===

    # When var=TRUE
    high_transitions = [
        (bdd.let({var: True}), target, is_accepting)
        for bdd, target, is_accepting in transitions
    ]

    # When var=FALSE
    low_transitions = [
        (bdd.let({var: False}), target, is_accepting)
        for bdd, target, is_accepting in transitions
    ]

    # Determine where each branch leads
    high_target = find_unique_target(high_transitions)
    low_target = find_unique_target(low_transitions)

    # Create high branch
    if needs_expansion(high_transitions):
        intermediate_high = create_new_state()
        add_transition(source_state, intermediate_high, var)
        expand_state_transitions(intermediate_high, high_transitions, var_index + 1)
    else:
        add_transition(source_state, high_target, var)
        if high_is_accepting:
            mark_accepting(high_target)

    # Create low branch
    if needs_expansion(low_transitions):
        intermediate_low = create_new_state()
        add_transition(source_state, intermediate_low, f"!{var}")
        expand_state_transitions(intermediate_low, low_transitions, var_index + 1)
    else:
        add_transition(source_state, low_target, f"!{var}")
        if low_is_accepting:
            mark_accepting(low_target)

def find_unique_target(transitions):
    """
    Find which target state this set of restricted formulas leads to.

    For a deterministic DFA, exactly ONE formula should be TRUE
    for any variable assignment.
    """
    for bdd, target, is_accepting in transitions:
        if bdd == BDD.TRUE:
            return (target, is_accepting)

    # If no formula is TRUE yet, need more expansion
    # Return None to indicate need for intermediate state
    return None

def needs_expansion(transitions):
    """Check if we've reached a terminal state or need more variable tests"""
    for bdd, target, is_accepting in transitions:
        if bdd == BDD.TRUE:
            return False  # Found terminal, no more expansion needed
    return True  # Still have formulas to evaluate
```

### Why This Works

1. **Uses BDD `let()` method**: Properly restricts formulas by assigning variables
   - `bdd.let({var: True})` returns the cofactor of bdd with var=true
   - This is the standard BDD operation for Shannon Expansion

2. **Processes formulas in parallel**: All formulas are restricted together
   - Maintains the relationship between formulas
   - Ensures determinism (each branch has one target)

3. **Guaranteed correct**:
   - Every variable assignment is considered (complete)
   - Each branch leads to exactly one state (deterministic)
   - Preserves the semantics of original formulas (sound)

### Example Walkthrough

Original:
```
State 1 -> State 2 [~(a&b)]
State 1 -> State 1 [a&b]
```

**Step 1**: var_index=0, var='a'
- Restrict with a=True:
  - ~(True&b) = ~b → State 2
  - True&b = b → State 1
- Restrict with a=False:
  - ~(False&b) = ~False = True → State 2
  - False&b = False → State 1

High branch (a=True): Both formulas need 'b', create intermediate s1
Low branch (a=False): ~(a&b)=True, go directly to State 2

Transitions so far:
```
State 1 -> s1 [a]
State 1 -> State 2 [!a]
```

**Step 2**: At s1, var_index=1, var='b'
- Restrict with b=True:
  - ~b = False → (no transition)
  - b = True → State 1
- Restrict with b=False:
  - ~b = True → State 2
  - b = False → (no transition)

Transitions:
```
s1 -> State 1 [b]
s1 -> State 2 [!b]
```

**Final DFA**:
```
State 1 -> s1 [a]
State 1 -> State 2 [!a]
s1 -> State 1 [b]
s1 -> State 2 [!b]
```

This is **deterministic** and **equivalent** to the original!

## Key Differences from Failed Attempts

### ❌ Failed Approach 1: Independent BDD Expansion
```python
for formula, target in transitions:
    expand_bdd(formula)  # Each formula processed separately
    # Creates conflicting transitions from same state!
```

### ❌ Failed Approach 2: Skip Duplicate States
```python
if state already has transitions:
    skip this formula  # Loses valid transitions!
```

### ✅ Correct Approach: Parallel Restriction
```python
# Restrict ALL formulas together at each variable
high_formulas = [f.let({var: True}) for f in formulas]
low_formulas = [f.let({var: False}) for f in formulas]
# Process restricted formulas recursively
```

## Implementation Requirements

1. **Use `bdd.let()` method** from dd library for restriction
2. **Process all outgoing transitions from same state together**
3. **Variable ordering**: Use BDD's natural variable ordering or specify consistent order
4. **Terminal detection**: Check when bdd == BDD.TRUE to stop recursion
5. **Target determination**: For deterministic DFA, exactly one formula should be TRUE

## References

- Bryant, R. E. (1986). "Graph-Based Algorithms for Boolean Function Manipulation"
- Klarlund, N., & Møller, A. (2001). "MONA Implementation Secrets"
- Symbolic model checking textbooks (Clarke, Grumberg, Peled)
