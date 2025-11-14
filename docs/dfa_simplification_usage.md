# DFA Simplification - Usage Guide

## Overview

The DFA simplifier transforms a DFA with complex boolean expressions on transition labels into an equivalent DFA where each transition has a single, atomic partition symbol.

**Benefits:**
- ✅ **Lossless**: Preserves all semantic information
- ✅ **Minimal**: Generates only necessary partitions (not all 2^n minterms)
- ✅ **Scalable**: BDD-based method supports 100+ predicates
- ✅ **Correct**: Guaranteed semantic equivalence

## Quick Start

### Installation

For optimal performance with large domains, install the BDD library:

```bash
pip install dd
```

If BDD is not available, the system automatically falls back to minterm enumeration (suitable for domains with <12 predicates).

### Basic Usage

```python
from stage2_dfa_generation.dfa_simplifier import DFASimplifier
from stage1_interpretation.grounding_map import GroundingMap

# Create grounding map (maps propositional symbols to predicates)
gmap = GroundingMap()
gmap.add_atom("on_a_b", "on", ["a", "b"])
gmap.add_atom("clear_c", "clear", ["c"])

# Your DFA in DOT format
dfa_dot = """
digraph G {
    s0 -> s1 [label="on_a_b | clear_c"];
    s1 -> s2 [label="on_a_b & ~clear_c"];
}
"""

# Simplify
simplifier = DFASimplifier()
result = simplifier.simplify(dfa_dot, gmap)

# Results
print(f"Method used: {result.stats['method']}")
print(f"Partitions: {result.stats['num_partitions']}")
print(f"Compression: {result.stats['compression_ratio']:.2f}x")

# View partitions
for partition in result.partitions:
    print(f"{partition.symbol}: {partition.expression}")

# View mapping from original labels to partitions
for label, partitions in result.original_label_to_partitions.items():
    print(f"'{label}' → {partitions}")

# Get simplified DFA
simplified_dfa_dot = result.simplified_dot
```

### Example Output

```
Method used: minterm
Partitions: 3
Compression: 1.33x

p1: on_a_b & ~clear_c
p2: ~on_a_b & clear_c
p3: on_a_b & clear_c

'on_a_b | clear_c' → ['p1', 'p2', 'p3']
'on_a_b & ~clear_c' → ['p1']
```

**Original DFA:**
```dot
s0 -> s1 [label="on_a_b | clear_c"];
s1 -> s2 [label="on_a_b & ~clear_c"];
```

**Simplified DFA:**
```dot
s0 -> s1 [label="p1"];
s0 -> s1 [label="p2"];
s0 -> s1 [label="p3"];
s1 -> s2 [label="p1"];
```

## Advanced Usage

### Force Specific Method

```python
# Force BDD method
result = simplifier.simplify(dfa_dot, gmap, method='bdd')

# Force minterm method
result = simplifier.simplify(dfa_dot, gmap, method='minterm')
```

### Check BDD Availability

```python
from stage2_dfa_generation.dfa_simplifier import BDDSimplifier

bdd_simplifier = BDDSimplifier()
if bdd_simplifier.is_available():
    print("BDD available - can handle large domains")
else:
    print("BDD not available - limited to small domains")
```

### Access Detailed Partition Information

```python
# Each partition has:
for partition in result.partitions:
    print(f"Symbol: {partition.symbol}")
    print(f"Expression: {partition.expression}")
    print(f"Predicate values: {partition.predicate_values}")
    # predicate_values is a dict: {'on_a_b': True, 'clear_c': False}
```

## Integration with Pipeline

### Option 1: Post-process DFA (Recommended)

Add simplification after DFA generation:

```python
from stage2_dfa_generation import DFABuilder
from stage2_dfa_generation.dfa_simplifier import DFASimplifier

# Generate DFA
dfa_builder = DFABuilder()
dfa_result = dfa_builder.build(ltl_spec)

# Simplify (optional)
simplifier = DFASimplifier()
simplified = simplifier.simplify(
    dfa_result['dfa_dot'],
    ltl_spec.grounding_map
)

# Use simplified DFA
dfa_result['dfa_dot'] = simplified.simplified_dot
dfa_result['partition_map'] = simplified.partition_map
```

### Option 2: Integrate into DFABuilder

Modify `dfa_builder.py` to optionally simplify:

```python
class DFABuilder:
    def __init__(self, simplify_transitions: bool = False):
        self.simplify_transitions = simplify_transitions
        if simplify_transitions:
            from .dfa_simplifier import DFASimplifier
            self.simplifier = DFASimplifier()

    def build(self, ltl_spec):
        # ... existing code ...

        if self.simplify_transitions:
            simplified = self.simplifier.simplify(
                dfa_dot,
                ltl_spec.grounding_map
            )
            dfa_dot = simplified.simplified_dot
            # Store partition map for later use
            result['partition_map'] = simplified.partition_map

        return result
```

### Option 3: Pipeline Configuration

Add to your pipeline config:

```python
PIPELINE_CONFIG = {
    'dfa_simplification': {
        'enabled': True,
        'method': 'auto',  # 'auto', 'bdd', 'minterm'
        'max_predicates_for_minterm': 12,
    }
}
```

## When to Use DFA Simplification?

### Use simplification when:
- ✅ You need "one predicate per transition" for downstream processing
- ✅ You want to enumerate all distinct input conditions
- ✅ You're building formal verification tools
- ✅ You need to analyze DFA structure programmatically

### Skip simplification when:
- ❌ Boolean expressions are already simple (single predicates)
- ❌ You have very large domains (>50 predicates without BDD)
- ❌ Downstream tools can handle boolean expressions directly

## Performance Characteristics

| Predicates | Method | Partitions (worst case) | Time | Memory |
|------------|--------|-------------------------|------|--------|
| 1-10 | Minterm | 2^n | <1s | <10MB |
| 10-15 | Minterm | 2^n | 1-5s | 10-100MB |
| 15+ | BDD | Typically << 2^n | 1-10s | 100MB-1GB |
| 50+ | BDD | Depends on structure | 10-60s | 1-10GB |
| 100+ | BDD | Depends on structure | Variable | Variable |

**Note**: BDD performance heavily depends on:
- Variable ordering (automatic in `dd` library)
- Expression structure (some formulas compress better)
- Available memory

## Debugging

### Enable Verbose Output

The simplifier prints progress information:

```
[DFA Simplifier] Using method: minterm
[Simple Simplifier] Using explicit minterm enumeration
  Predicates (3): ['clear_c', 'holding_d', 'on_a_b']
  Will generate up to 8 minterms
  Generated 8 total minterms
  Used minterms: 7 out of 8
```

### Verify Correctness

Check that simplification preserves semantics:

```python
# Original label should be union of its partitions
label = "on_a_b | clear_c"
partitions = result.original_label_to_partitions[label]

print(f"'{label}' maps to {len(partitions)} partitions: {partitions}")

# Verify each partition expression
for symbol in partitions:
    partition = result.partition_map[symbol]
    print(f"  {symbol}: {partition.expression}")
    print(f"    Values: {partition.predicate_values}")
```

## Troubleshooting

### Error: "Too many predicates for simple simplifier"

**Solution**: Install BDD library:
```bash
pip install dd
```

### Error: "Failed to build BDD for label"

**Cause**: Complex expression syntax not supported

**Solution**: Check label syntax. Supported operators:
- `&`, `&&`: AND
- `|`, `||`: OR
- `!`, `~`: NOT
- Parentheses: `(`, `)`
- Constants: `true`, `false`

### High memory usage

**Cause**: Too many predicates without BDD

**Solution**:
1. Install BDD: `pip install dd`
2. Reduce number of predicates if possible
3. Simplify boolean expressions before DFA generation

## Examples

See `tests/stage2_dfa_generation/test_dfa_simplifier.py` for comprehensive examples covering:
- Simple 2-predicate DFA
- Complex 3-predicate DFA
- BDD-based simplification
- Auto method selection
- True/false label handling
- Correctness verification

## API Reference

### DFASimplifier

Main entry point with automatic method selection.

```python
simplifier = DFASimplifier(
    prefer_bdd=True,  # Prefer BDD if available
    max_predicates_for_minterm=12  # Max for minterm method
)

result = simplifier.simplify(
    dfa_dot,  # DFA in DOT format (str)
    grounding_map,  # GroundingMap instance
    method=None  # 'bdd', 'minterm', or None for auto
)
```

### SimplifiedDFA

Result object containing:

```python
@dataclass
class SimplifiedDFA:
    simplified_dot: str  # Simplified DFA in DOT format
    partitions: List[PartitionInfo]  # List of partitions
    partition_map: Dict[str, PartitionInfo]  # Symbol → PartitionInfo
    original_label_to_partitions: Dict[str, List[str]]  # Label → symbols
    stats: Dict[str, Any]  # Statistics
```

### PartitionInfo

Information about each partition:

```python
@dataclass
class PartitionInfo:
    symbol: str  # e.g., "p1", "α1"
    expression: str  # e.g., "on_a_b & ~clear_c"
    minterm: Optional[str]  # Complete minterm (minterm method)
    predicate_values: Dict[str, bool]  # Predicate assignments
```

## Future Enhancements

Planned features:

- [ ] Variable ordering optimization for BDD
- [ ] Incremental simplification (update existing simplified DFA)
- [ ] Partition visualization as decision tree
- [ ] Support for temporal operators (X, U, etc.)
- [ ] Integration with DFA minimization

## References

- Design document: `docs/dfa_simplification_design.md`
- Implementation: `src/stage2_dfa_generation/dfa_simplifier.py`
- Tests: `tests/stage2_dfa_generation/test_dfa_simplifier.py`
- BDD library: https://github.com/tulip-control/dd
