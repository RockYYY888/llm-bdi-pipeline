# Backward Search Termination Issue Analysis

## Problem Confirmed

Running backward search with goal `on(a, b)` and `max_states=50`:
- **States explored**: 50 (hit limit)
- **Unique states generated**: 520
- **Search truncated**: YES
- **Average branching factor**: 5.43 (exponential growth!)

### State Growth by Depth
```
Depth 0: 1 state
Depth 1: 2 states
Depth 2: 16 states
Depth 3: 150 states  
Depth 4: 351 states
```

**Conclusion**: Search will NEVER terminate naturally for non-trivial goals.

---

## Root Cause Analysis

### 1. Variable-Level Planning Creates Huge State Space

Example expansion from `holding(a)`:
```
Goal: holding(a)

Actions found:
- pick-up(?b1=a, ?b2=?v1)  → State: [clear(a), handempty, on(a, ?v1)]
- pick-up(?b1=a, ?b2=?v2)  → State: [clear(a), handempty, on(a, ?v2)]  
- pick-up(?b1=a, ?b2=?v3)  → State: [clear(a), handempty, on(a, ?v3)]
... (infinite!)
```

Each unbound parameter creates a NEW variable, leading to exponentially many "different" states that are essentially the same at the grounded level.

### 2. No Subsumption Checking

States that should be considered equivalent are treated as different:
- `on(a, ?v1)` vs `on(a, ?v2)` → Same meaning, different variable names
- `clear(?v1) ∧ clear(?v2)` vs `clear(?v3) ∧ clear(?v4)` → Same structure

### 3. No Reachability Analysis

No detection of:
- States that can't possibly be reached from any initial state
- States that require contradictory preconditions
- States with unsatisfiable variable bindings

---

## Current Termination Conditions

✓ **Working**:
1. `states_explored >= max_states` (hard limit)
2. `current_state.is_goal_achieved()` (empty state)
3. Contradiction detection (P and ~P pruned)
4. Constraint violation (e.g., ?v1 = ?v1 when ?v1 ≠ ?v1 required)

✗ **Missing**:
1. Queue becoming empty naturally (never happens)
2. Subsumption checking (state generalization)
3. Reachability analysis
4. Loop detection (revisiting equivalent states)

---

## Proposed Solutions

### Solution 1: Depth-Limited Search (SIMPLE)
**Idea**: Add max_depth parameter to limit search depth

```python
def search(self, goal_predicates: List[PredicateAtom],
           max_states: int = 200000,
           max_depth: int = 10,  # NEW
           max_objects: Optional[int] = None) -> StateGraph:
    ...
    while queue and states_explored < max_states:
        current_state = queue.popleft()
        
        # NEW: Skip states that are too deep
        if current_state.depth >= max_depth:
            continue
        
        states_explored += 1
        ...
```

**Pros**: Simple to implement, prevents infinite depth
**Cons**: Arbitrary cutoff, may miss valid plans

---

### Solution 2: Variable Normalization (BETTER)
**Idea**: Normalize variable names to detect equivalent states

```python
def _normalize_state(self, predicates: Set[PredicateAtom]) -> FrozenSet[PredicateAtom]:
    """
    Normalize variable names in predicates
    
    Example:
        on(a, ?v2) ∧ clear(?v5) → on(a, ?v1) ∧ clear(?v2)
        on(a, ?v7) ∧ clear(?v3) → on(a, ?v1) ∧ clear(?v2)  (SAME!)
    """
    # Collect all variables in order of first appearance
    var_mapping = {}
    next_var = 1
    
    normalized = []
    for pred in sorted(predicates, key=lambda p: (p.name, p.args)):
        new_args = []
        for arg in pred.args:
            if arg.startswith('?v'):
                if arg not in var_mapping:
                    var_mapping[arg] = f'?v{next_var}'
                    next_var += 1
                new_args.append(var_mapping[arg])
            else:
                new_args.append(arg)
        normalized.append(PredicateAtom(pred.name, new_args, pred.negated))
    
    return frozenset(normalized)
```

Then in `_state_key()`:
```python
def _state_key(self, state: BackwardState) -> Tuple:
    # Use normalized predicates for state key
    normalized_preds = self._normalize_state(state.predicates)
    normalized_constraints = self._normalize_constraints(state.constraints)
    return (normalized_preds, normalized_constraints)
```

**Pros**: Detects equivalent states with different variable names
**Cons**: More complex, need to normalize constraints too

---

### Solution 3: Grounded Search with Object Limit (CURRENT APPROACH)
**Idea**: Use `max_objects` parameter to cap variable generation

```python
# Already implemented in _complete_binding()
# But not enforced strictly enough
```

**Current issue**: We removed the hard cap to allow abstract planning, but this causes explosion.

**Fix**: Re-enable max_objects cap, but track it per-state:
```python
if self.max_objects is not None:
    if next_var_num > self.max_objects:
        # PRUNE: Too many variables for this object limit
        return []  # Skip this state
```

---

### Solution 4: Reachability Heuristic (ADVANCED)
**Idea**: Estimate if a state is reachable from likely initial states

```python
def _is_potentially_reachable(self, predicates: Set[PredicateAtom]) -> bool:
    """
    Heuristic check if state might be reachable
    
    Prune if:
    - Too many holding() predicates (can only hold 1 block)
    - Cyclic on() relationships
    - etc.
    """
    # Count holding predicates
    holding_count = sum(1 for p in predicates if p.name == 'holding')
    if holding_count > 1:
        return False  # Can't hold multiple blocks
    
    # More heuristics...
    return True
```

---

## Recommended Approach

**Combine multiple solutions**:

1. **Depth limit** (max_depth=15): Prevent infinite depth
2. **Object limit** (max_objects): Cap variable generation strictly  
3. **Variable normalization**: Detect equivalent states
4. **Simple reachability heuristics**: Prune obviously unreachable states

This multi-pronged approach should:
- Terminate naturally for most goals
- Significantly reduce state space
- Still find valid plans when they exist

---

## Implementation Priority

1. **HIGH**: Add depth limit (quick win)
2. **HIGH**: Strictly enforce max_objects cap
3. **MEDIUM**: Add variable normalization
4. **LOW**: Add reachability heuristics (domain-specific)

---

## Testing Strategy

Test on goals of increasing complexity:
1. `on(a, b)` - Simple, should terminate quickly
2. `on(a, b) ∧ on(b, c)` - Medium, should still terminate
3. `~on(a, b)` - Negative goal, test deletion effects
4. Complex goals - Monitor state count and depth

Track metrics:
- States explored before termination
- Max depth reached
- Queue size over time
- Pruned states count
