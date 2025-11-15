# DFA Transition Label Simplification Design

## Problem Statement

Transform a DFA with complex boolean expressions on transition labels into an equivalent DFA where each transition has a single, atomic predicate label.

### Requirements
1. **Scalability**: Support large domains with 100+ predicates
2. **Lossless**: Preserve all semantic information from original labels
3. **Minimality**: Generate the minimum number of partition symbols needed
4. **Equivalence**: Resulting DFA must accept exactly the same language

## Solution: BDD-based Partition Refinement

### Core Idea

Instead of generating all 2^n minterms, we generate the **minimal partition** that:
- Distinguishes all unique behaviors in the original DFA
- Is mutually exclusive (at any moment, exactly one partition is true)
- Can perfectly reconstruct all original transition labels

### Algorithm Overview

```
Input: DFA with boolean expressions on transitions
Output: Equivalent DFA with atomic partition symbols

1. Collect all atomic predicates P = {p1, p2, ..., pn}
2. Build BDD for each transition label
3. Compute partition refinement:
   - Start with initial partition: {TRUE, FALSE}
   - For each transition label BDD:
     - Split existing partitions based on this BDD
     - Keep only reachable partitions
4. Assign unique symbols to each partition (α1, α2, ...)
5. Rebuild DFA with partition symbols
6. Maintain mapping: partition_symbol → original_expression
```

### Partition Refinement Example

```
Original transitions:
  s0 --[on_a_b | clear_c]--> s1
  s0 --[on_a_b & ~clear_c]--> s2
  s1 --[clear_c]--> s3

Step 1: Collect predicates
  P = {on_a_b, clear_c}

Step 2: Build BDDs
  BDD1: on_a_b ∨ clear_c
  BDD2: on_a_b ∧ ¬clear_c
  BDD3: clear_c

Step 3: Partition refinement
  Initial: {all_true, all_false}

  After BDD1 (on_a_b | clear_c):
    - Partition A: (on_a_b ∧ clear_c)     [satisfies BDD1]
    - Partition B: (on_a_b ∧ ¬clear_c)    [satisfies BDD1]
    - Partition C: (¬on_a_b ∧ clear_c)    [satisfies BDD1]
    - Partition D: (¬on_a_b ∧ ¬clear_c)   [doesn't satisfy BDD1]

  After BDD2 (on_a_b & ~clear_c):
    - No further splitting needed (B already represents this)

  After BDD3 (clear_c):
    - No further splitting needed (A and C already represent this)

Step 4: Final minimal partitions (only used ones)
  α1 = (on_a_b ∧ clear_c)      // Partition A
  α2 = (on_a_b ∧ ¬clear_c)     // Partition B
  α3 = (¬on_a_b ∧ clear_c)     // Partition C
  // Note: Partition D is unused, so we don't generate it

Step 5: Simplified DFA
  s0 --[α1]--> s1
  s0 --[α2]--> s1
  s0 --[α3]--> s1
  s0 --[α2]--> s2
  s1 --[α1]--> s3
  s1 --[α3]--> s3
```

### Why BDD Instead of Naive Minterm Enumeration?

| Aspect | Naive Minterm | BDD-based |
|--------|---------------|-----------|
| Partitions generated | 2^n (all possible) | k (only used) |
| Scalability | Fails at n>15 | Works for n>100 |
| Memory | Exponential | Polynomial (symbolic) |
| Minimality | No (generates unused) | Yes (only reachable) |
| Lossless | Yes | Yes |

### Implementation Strategy

#### Phase 1: BDD Library Integration
- Use `dd` (Decision Diagrams) library for Python
- Supports both Binary Decision Diagrams (BDD) and Zero-suppressed Decision Diagrams (ZDD)
- Mature, actively maintained, efficient

#### Phase 2: Core Components

```python
class DFASimplifier:
    """Simplifies DFA using BDD-based partition refinement"""

    def __init__(self, use_bdd: bool = True):
        self.use_bdd = use_bdd
        self.bdd_manager = None  # Will be dd.autoref.BDD()

    def simplify(self, dfa_dot: str) -> SimplifiedDFA:
        """
        Main entry point

        Returns:
            SimplifiedDFA with:
              - simplified_dot: DFA with atomic partition labels
              - partition_map: {partition_symbol: boolean_expression}
              - stats: {num_predicates, num_partitions, compression_ratio}
        """

    def _collect_predicates(self, transitions) -> List[str]:
        """Extract all atomic predicates from all labels"""

    def _build_label_bdds(self, transitions) -> Dict[str, BDD]:
        """Build BDD for each unique transition label"""

    def _compute_partition_refinement(self, label_bdds) -> List[BDD]:
        """Core algorithm: compute minimal partition using BDD operations"""

    def _bdd_to_expression(self, bdd) -> str:
        """Convert BDD to human-readable expression"""

    def _rebuild_dfa(self, original_dfa, partitions) -> str:
        """Rebuild DFA DOT with partition symbols"""
```

#### Phase 3: Fallback Strategy

For systems without BDD library or very simple cases:

```python
class SimpleMintermSimplifier:
    """Fallback: explicit minterm enumeration for small n (<10)"""

    def simplify(self, dfa_dot: str, max_predicates: int = 10):
        if num_predicates > max_predicates:
            raise ValueError("Use BDD-based simplifier for large domains")
        # Enumerate all 2^n minterms
        # Filter to only used ones
```

### Integration with Existing Pipeline

```
Stage 2: DFA Generation
  ↓
[NEW] DFA Simplification (optional, configurable)
  ├─ If enabled: simplify_dfa(dfa_dot, use_bdd=True)
  └─ If disabled: pass through original DFA
  ↓
Stage 3: Code Generation (existing)
```

### Configuration

Add to pipeline config:

```python
DFA_SIMPLIFICATION = {
    'enabled': True,
    'method': 'bdd',  # 'bdd' | 'minterm' | 'none'
    'keep_original_labels': True,  # Maintain mapping for debugging
    'max_partitions': 1000,  # Safety limit
}
```

## Benefits

1. **Scalability**: BDD can handle 100+ predicates (tested in verification tools)
2. **Optimality**: Generates minimum number of partitions
3. **Lossless**: Perfect semantic preservation
4. **Debuggable**: Maintains partition → expression mapping
5. **Optional**: Can be disabled for simple domains

## Testing Strategy

1. **Correctness**: Verify simplified DFA accepts same language
2. **Minimality**: Check partition count ≤ 2^n
3. **Scalability**: Test with 10, 20, 50, 100 predicates
4. **Regression**: All existing tests still pass

## Future Extensions

- **BDD variable ordering optimization**: Auto-tune for specific domains
- **Incremental simplification**: Update partitions when DFA changes
- **Partition visualization**: Show partition structure as decision tree
