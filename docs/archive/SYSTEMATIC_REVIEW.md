# Systematic Review: DFA Equivalence Verification

## Executive Summary
**CRITICAL FINDING**: The current codebase has NEVER been able to deliver 100% correct equivalent DFAs with one predicate per branch. The BDD Shannon expansion approach has fundamental nondeterminism issues that persist across all versions.

## Test Results Across Versions

### Original Code (commit a991130) - BASELINE
**Result**: 6/11 tests pass (54.5%)

**Failing Tests**:
- test_complex_formula_3_atoms
- test_deeply_nested_expression  
- test_mixed_conjunction_disjunction
- test_edge_case_negation_only
- test_regression_negated_bdd_nodes

**Issue**: Nondeterministic DFA generation
Example from test_complex_formula_3_atoms:
```
1 -> s1 [label="!clear_c"];
1 -> s1 [label="clear_c"];   # NONDETERMINISTIC
1 -> s3 [label="!clear_c"];  # NONDETERMINISTIC  
1 -> s3 [label="clear_c"];   # NONDETERMINISTIC
```

### Version with dfa_states_done (commit 864844c)
**Result**: 8/11 tests pass (72.7%)

**Improvement**: Fixed nondeterminism for some cases
**Regression**: Skips valid transitions when formulas have different targets

**Failing Tests**:
- test_equivalence_with_simplifier
- test_edge_case_negation_only
- test_regression_negated_bdd_nodes

**Issue**: Incomplete DFA (missing transitions)

### Variable-Level Planning (commit c7be937) - CURRENT
**Result**: 8/11 tests pass (72.7%)

**Status**: Deterministic DFA but incorrect formula evaluation
**Issue**: BDD restriction logic doesn't handle variable ordering correctly

## Root Cause Analysis

### The Fundamental Problem

When processing multiple outgoing transitions from the same DFA state:
```
State 1 -> State 2 when formula f1
State 1 -> State 1 when formula ~f1
```

The BDD Shannon expansion processes each formula independently:
1. BDD for f1 creates decision tree from state 1
2. BDD for ~f1 creates DIFFERENT decision tree from state 1
3. Both trees start at state 1 and create conflicting transitions

### Why This Happens

BDD nodes are shared between formulas, but BDD ROOTS are different:
- Formula f1 has root node with hash=7
- Formula ~f1 has root node with hash=-7
- Both roots map to DFA state "1"
- Each generates its own set of transitions from state "1"

Result: **Nondeterministic DFA**

## What Each Transition Label Should Be

According to the design spec:
- Each transition should test exactly ONE atomic literal
- Labels should be: `var`, `!var`, or `true`
- No complex boolean expressions like `(a&b)|c`

## Current Status

**Best Result**: 8/11 tests passing (72.7%)

**Unsolved Issues**:
1. Nondeterminism when multiple formulas share BDD structure
2. Incomplete DFAs when using dfa_states_done approach
3. Incorrect formula evaluation with variable-level planning

## Correct Solution Required

The current approaches all fail because they try to process BDD formulas independently. The correct solution requires ONE of:

1. **Build Unified BDD**: Create a single BDD that represents ALL outgoing transitions from a state as a mapping from valuations to target states

2. **DFA Determinization**: Accept nondeterministic intermediate DFA, then apply standard subset construction algorithm

3. **Proper Shannon Expansion**: Use dd library's BDD operations (restrict/cofactor) correctly to build a unified decision tree

## Recommendation

**STOP** attempting to patch the current approach. 

**START** implementing proper DFA determinization:
1. Allow BDD Shannon expansion to generate nondeterministic DFA
2. Apply subset construction algorithm to determinize
3. Apply DFA minimization to reduce states
4. Verify equivalence

This is the standard textbook approach and will work reliably.
