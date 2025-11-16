# dd.autoref BDD 正确遍历算法

## 关键发现

dd.autoref BDD库使用**节点符号（node sign）**来表示negation：

```python
x:    node=+2, negated=False
~x:   node=-2, negated=True
```

### Negation的语义

**当 `negated=True` 时，high/low branches的语义是INVERTED的：**

```python
# For normal node (negated=False):
x.high  → variable is TRUE
x.low   → variable is FALSE

# For negated node (negated=True):
(~x).high  → variable is FALSE  (inverted!)
(~x).low   → variable is TRUE   (inverted!)
```

## 正确的遍历算法

### 算法伪代码

```python
def traverse_bdd(node):
    """
    Correct BDD traversal that handles negation
    """
    # Get base properties
    is_negated = node.negated
    var = node.var  # variable name
    high_child = node.high
    low_child = node.low

    # Determine TRUE and FALSE branches based on negation
    if not is_negated:
        # Normal: high=var_true, low=var_false
        true_branch = high_child
        false_branch = low_child
        true_label = var          # "x"
        false_label = f"!{var}"   # "!x"
    else:
        # Negated: INVERTED semantics
        true_branch = low_child   # ← swapped!
        false_branch = high_child # ← swapped!
        true_label = f"!{var}"    # "!x"
        false_label = var         # "x"

    # Create transitions
    if true_branch:
        create_transition(current_state, next_state, true_label)
    if false_branch:
        create_transition(current_state, next_state, false_label)
```

### 具体示例

**Example 1: 正常节点 `x`**
```python
node.negated = False
node.var = 'x'
node.high = TRUE
node.low = FALSE

→ Transition: state --[x]--> target  (when x=true)
```

**Example 2: 否定节点 `~x`**
```python
node.negated = True
node.var = 'x'
node.high = TRUE  ← 但这表示x=FALSE时的结果！
node.low = FALSE  ← 但这表示x=TRUE时的结果！

正确解释:
- When x=TRUE: follow LOW → FALSE → reject
- When x=FALSE: follow HIGH → TRUE → accept

→ Transition: state --[!x]--> target  (when x=false)
```

## 完整的DFA构建算法

```python
def bdd_to_dfa(bdd_node, start_state, target_state):
    """
    Convert BDD to DFA transitions with correct negation handling
    """
    transitions = []

    # Terminal cases
    if bdd_node == bdd.true:
        transitions.append((start_state, target_state, "true"))
        return transitions

    if bdd_node == bdd.false:
        return []  # No transition

    # Get node properties
    is_negated = bdd_node.negated
    var_name = bdd_node.var
    high_child = bdd_node.high
    low_child = bdd_node.low

    # Map to state
    node_key = abs(bdd_node.node)  # Use absolute value as key
    current_state = state_map.get(node_key, start_state)

    # Determine which branch corresponds to var=TRUE and var=FALSE
    if not is_negated:
        # Normal node: high=TRUE, low=FALSE
        true_child = high_child
        false_child = low_child
        true_label = var_name
        false_label = f"!{var_name}"
    else:
        # Negated node: semantics are inverted!
        true_child = low_child   # When var=TRUE, follow LOW
        false_child = high_child # When var=FALSE, follow HIGH
        true_label = f"!{var_name}"  # Inverted label
        false_label = var_name       # Inverted label

    # Process TRUE branch (var=TRUE)
    if true_child and true_child != bdd.false:
        if true_child == bdd.true:
            transitions.append((current_state, target_state, true_label))
        else:
            # Handle child negation
            child_key = abs(true_child.node)
            if child_key not in state_map:
                state_map[child_key] = create_new_state()
            next_state = state_map[child_key]
            transitions.append((current_state, next_state, true_label))
            transitions.extend(bdd_to_dfa(true_child, next_state, target_state))

    # Process FALSE branch (var=FALSE)
    if false_child and false_child != bdd.false:
        if false_child == bdd.true:
            transitions.append((current_state, target_state, false_label))
        else:
            child_key = abs(false_child.node)
            if child_key not in state_map:
                state_map[child_key] = create_new_state()
            next_state = state_map[child_key]
            transitions.append((current_state, next_state, false_label))
            transitions.extend(bdd_to_dfa(false_child, next_state, target_state))

    return transitions
```

## 关键点总结

1. **使用 `abs(node.node)` 作为state mapping的key**
   - 因为 `x` (node=2) 和 `~x` (node=-2) 共享相同的BDD结构
   - abs value去除符号影响

2. **检查 `node.negated` 来决定分支语义**
   - `negated=False`: high=var_true, low=var_false
   - `negated=True`: high=var_false, low=var_true (inverted!)

3. **Transition labels根据negation调整**
   - Normal: true_label=var, false_label=!var
   - Negated: true_label=!var, false_label=var

4. **递归处理children时也要检查它们的negation**
   - 每个child可能也是negated的
   - 不能假设child的语义

## 测试验证

```python
# Test case: ~on_a_b
bdd_node.negated = True
bdd_node.var = 'on_a_b'
bdd_node.high = TRUE
bdd_node.low = FALSE

# Correct interpretation:
#   When on_a_b=TRUE: follow LOW → FALSE → no transition
#   When on_a_b=FALSE: follow HIGH → TRUE → create transition

# Expected output:
#   state_1 --[!on_a_b]--> state_2
```

这个算法保证了正确处理dd.autoref的BDD negation机制。
