# Variable Explosion Analysis

## 🔴 Problem

In Test 3 with **5 objects** (a, b, c, d, e), backward search generates variables up to **`?16`**, which leads to AgentSpeak code with `V16`.

Example output:
```agentspeak
+!~on(V15, V16) : clear(V16) & holding(V0) & holding(V6) & holding(V15) ...
```

**Question**: Why do we need 16 variables for only 5 objects?

## 📊 Current Implementation

### Variable Numbering Strategy

1. **Initial state** (from VariableNormalizer):
   - Goal: `on(a, b)` → `on(?v0, ?v1)`
   - `max_var_number = 1`

2. **Each layer** (from BackwardSearchPlanner):
   - When action has unbound parameters, generate `?{max_var + 1}`
   - Update state's `max_var_number = next_var_num - 1`

3. **Linear growth**:
   ```
   Depth 0: on(?v0, ?v1)                    max_var=1
   Depth 1: holding(?v0) ∧ clear(?v1)       max_var=1 (no new vars)
   Depth 2: clear(?v0) ∧ on(?v0, ?2)        max_var=2 (added ?2)
   Depth 3: ...on(?2, ?3) ∧ on(?3, ?4)      max_var=4 (added ?3, ?4)
   Depth 4: ...on(?4, ?5) ∧ on(?5, ?6)      max_var=6 (added ?5, ?6)
   Depth 5: ...                              max_var=16 (many new vars)
   ```

### Why Variables Grow

**Scenario**: Goal is `on(?v0, ?v1)` (2 objects needed)

**Step 1**: Find action `put-on-block(?b1, ?b2)`
- Bindings: `{?b1: ?v0, ?b2: ?v1}` (fully bound)
- New predicates: `holding(?v0) ∧ clear(?v1)`
- max_var = 1 (unchanged)

**Step 2**: Process `holding(?v0)`, find action `pick-up(?b1, ?b2)`
- Try to match `holding(?b1)` with `holding(?v0)`
- Bindings: `{?b1: ?v0}` (partial)
- **Missing**: `?b2` → generate `?2`
- New predicates: `handempty ∧ clear(?v0) ∧ on(?v0, ?2)`
- max_var = 2

**Step 3**: Process `on(?v0, ?2)`, find action `put-on-block(?b1, ?b2)`
- Bindings: `{?b1: ?v0, ?b2: ?2}` (fully bound)
- But preconditions add more predicates with new vars...
- max_var = 3 or 4

**Continues...**

After 5 depth levels with multiple branches, max_var can reach 15-16.

## 🤔 Analysis

### Question 1: Is this correct behavior?

**Arguments FOR current behavior**:
- Represents "any object" abstractly
- Allows planning without knowing specific objects
- Variables are placeholders, not bound to specific object count

**Arguments AGAINST**:
- With 5 objects, max 5 variables should suffice
- Variables beyond object count have no valid instantiation
- Generated plans will have redundant variables

### Question 2: What does "variable-level planning" mean?

**Interpretation A**: Variables as object placeholders
- 5 objects → max 5 variables needed: `?v0, ?v1, ?v2, ?v3, ?v4`
- New unbound parameters should reuse from this pool
- Example: `pick-up(?v0, ?v2)` instead of `pick-up(?v0, ?5)`

**Interpretation B**: Variables as abstract slots
- Variables represent "some object" without knowing which one
- Can have unlimited variables: `?1, ?2, ?3, ..., ?100`
- During instantiation, all bind to actual objects (a, b, c, d, e)

**Current implementation**: Follows Interpretation B

## 🔧 Possible Solutions

### Solution 1: Cap max_var_number by object count

```python
def _complete_binding(...):
    # Get number of objects from normalizer
    max_allowed_var = len(self.domain_objects)

    for param in parameters:
        if param not in complete_binding:
            # Reuse variables cyclically or fail if exceeded
            if next_var_num >= max_allowed_var:
                # Option A: Cycle back to ?v0
                new_var = f"?v{next_var_num % max_allowed_var}"
            else:
                new_var = f"?{next_var_num}"
            complete_binding[param] = new_var
            next_var_num += 1
```

**Problem**: BackwardSearchPlanner doesn't know domain object count!

### Solution 2: Pass object count to planner

```python
class BackwardSearchPlanner:
    def __init__(self, domain: PDDLDomain, num_objects: int):
        self.domain = domain
        self.num_objects = num_objects  # Cap for variables
```

Then in `_complete_binding`, reuse variables when exceeding object count.

### Solution 3: Smarter variable reuse

Instead of always generating new variables, try to:
1. Check if existing variables in parent state can be reused
2. Only generate new variable if semantically different slot needed
3. Maintain variable pool and reuse from it

**Challenge**: How to determine when reuse is safe?

### Solution 4: Accept current behavior

Arguments:
- Planning is abstract, variables represent "any object"
- During code generation/execution, all variables get bound to actual objects
- Extra variables don't affect correctness, only code readability
- Real issue is code bloat, not logical incorrectness

## 📋 Recommendation

Need clarification on intended behavior:

**Option A**: Variables should be bounded by object count
- Modify planner to know object count
- Implement variable pooling/reuse strategy
- Cap max_var_number at object count

**Option B**: Current behavior is acceptable
- Document that variables are abstract
- Optimize AgentSpeak code generation to compact variable names
- Accept that planning uses more variables than objects

## 🎯 Next Steps

1. **Clarify user intent**: Which interpretation is correct?
2. **Decide on solution**: Based on clarification
3. **Implement fix**: Modify `_complete_binding` or accept current behavior
4. **Test thoroughly**: Ensure no regression in planning correctness

## 📝 Related Code

- `backward_search_refactored.py:511` - `_complete_binding()` method
- `backward_search_refactored.py:258` - `_create_initial_goal_state()`
- `agentspeak_codegen.py:77` - `_build_unified_variable_map()`
- `variable_normalizer.py:43` - Position-based normalization
