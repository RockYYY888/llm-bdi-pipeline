# LTLf2DFA Integration

## Overview

This project now includes **ltlf2dfa** integration for converting LTLf specifications to Deterministic Finite Automata (DFA).

## Installation

```bash
uv pip install ltlf2dfa
```

## What was Added

### 1. Stage 1.5: LTLf to DFA Conversion
- **Location**: `src/stage1_5_dfa/ltlf_to_dfa.py`
- **Purpose**: Converts predicate-based LTLf formulas to DFA
- **Key Feature**: Automatic predicate-to-proposition encoding

### 2. Predicate Encoding
ltlf2dfa uses propositional variables, but our LTLf formulas use predicates:

```
on(a, b)    → on_a_b      (predicate → proposition)
clear(a)    → clear_a
holding(x)  → holding_x
handempty   → handempty   (already propositional)
```

### 3. Test Scripts
- `temp/test_ltlf2dfa.py`: Basic ltlf2dfa functionality tests
- All tests passing ✓

## Pipeline Architecture (with DFA)

```
Natural Language ("Stack block A on block B")
    ↓
Stage 1: NL → LTLf
    ↓  (e.g., F(on(a, b)))
Stage 1.5: LTLf → DFA [NEW]
    ↓  (Propositional encoding: F(on_a_b) → DFA)
Stage 2: LTLf → AgentSpeak Code
    ↓
AgentSpeak Program (.asl)
```

## Usage Example

```python
from stage1_5_dfa.ltlf_to_dfa import LTLfToDFA

converter = LTLfToDFA()
dfa_dot, metadata = converter.convert(ltl_spec)

# Outputs:
# - dfa_dot: DFA in DOT format (for visualization)
# - metadata: {
#     "original_formula": "F(on(a, b))",
#     "propositional_formula": "F(on_a_b)",
#     "predicate_to_prop_mapping": {"on(a, b)": "on_a_b"},
#     ...
#   }
```

## Testing

```bash
# Test ltlf2dfa basic functionality
cd temp
uv run python test_ltlf2dfa.py

# Test DFA converter
uv run python src/stage1_5_dfa/ltlf_to_dfa.py
```

## Benefits

1. **Formal Verification**: DFA representation enables formal verification of LTLf goals
2. **Visualization**: DOT format can be rendered with Graphviz
3. **Alternative Planning**: DFA can be used for automata-based planning
4. **Research Baseline**: Compare LLM-generated code against DFA-based approaches

## Future Integration

The DFA conversion is currently standalone. Future work:
- Integrate DFA generation into main pipeline
- Use DFA for plan verification
- Compare DFA-based vs LLM-based approaches
- Visualize DFA in execution logs

## References

- ltlf2dfa: https://github.com/whitemech/LTLf2DFA
- Online tool: http://ltlf2dfa.diag.uniroma1.it
- MONA: http://www.brics.dk/mona/
