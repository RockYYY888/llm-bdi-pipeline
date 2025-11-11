# Stage 3 Test Suite

## Overview

Stage 3 testing is consolidated into a single comprehensive integration test that validates all functionality of the backward planning pipeline.

## Running Tests

```bash
# Run the complete Stage 3 integration test (RECOMMENDED)
python tests/stage3_code_generation/test_stage3_complete.py
```

**Runtime**: ~2.9 seconds
**Coverage**: All Stage 3 functionality

## What is Tested

The `test_stage3_complete.py` file contains 5 comprehensive test cases:

### Test 1: Simple Goal with 2 Blocks
- **Goal**: `F(on(a, b))`
- **Validates**:
  - End-to-end pipeline (LTLf → DFA → AgentSpeak)
  - Basic backward planning
  - State consistency (100% valid states)
  - Code generation quality
- **Assertions**:
  - All code validations pass
  - Zero invalid states
  - Code not truncated

### Test 2: Scalability with 3 Blocks
- **Goal**: `F(on(a, b))` with 3 blocks
- **Validates**:
  - Scaling to larger state spaces
  - Performance degradation is graceful
  - State reuse is effective
  - Memory usage is reasonable
- **Assertions**:
  - All code validations pass
  - Zero invalid states
  - Performance < 60 seconds

### Test 3: Variable Abstraction & Caching
- **Validates**:
  - Schema-level caching works
  - Cache hit rate is measured
  - Variable normalization is correct
  - Constants are properly detected
- **Assertions**:
  - Generated code uses parameterized plans
  - Plans use variables (V0, V1, etc.) not specific objects

### Test 4: Multi-Transition DFA Handling
- **Goal**: Sequential goals `F(on(a,b) & F(clear(a)))`
- **Validates**:
  - Multiple DFA transitions handled correctly
  - Goals are processed in sequence
  - Code merging works properly
- **Assertions**:
  - Both goals present in generated code
  - All validations pass

### Test 5: State Consistency Guarantee
- **Validates**:
  - 100% of generated states are physically valid
  - No circular dependencies
  - No contradictions
  - All 7 consistency checks pass:
    1. Hand contradictions (handempty vs holding)
    2. Multiple holdings
    3. Self-loops (on(a,a))
    4. Multiple locations (on(a,b) & on(a,c))
    5. Circular on-relationships (on(a,b) & on(b,a))
    6. Location contradictions (ontable & on)
    7. Clear contradictions (clear(x) & on(y,x))
- **Assertions**:
  - Zero invalid states for both 2 and 3 blocks

## Test Output

Expected output format:

```
================================================================================
STAGE 3 COMPLETE INTEGRATION TEST SUITE
================================================================================

TEST 1: Simple Goal with 2 Blocks - F(on(a, b))
...
✅ TEST 1 PASSED

TEST 2: Scalability with 3 Blocks - F(on(a, b))
...
✅ TEST 2 PASSED

TEST 3: Variable Abstraction & Schema-Level Caching
...
✅ TEST 3 PASSED

TEST 4: Multi-Transition DFA - Sequential Goals
...
✅ TEST 4 PASSED

TEST 5: State Consistency Guarantee (100% Valid States)
...
✅ TEST 5 PASSED - 100% Valid States Guaranteed

================================================================================
TEST SUITE SUMMARY
================================================================================
Test 1 - Simple Goal (2 blocks):           ✅ PASS
Test 2 - Scalability (3 blocks):           ✅ PASS
Test 3 - Variable Abstraction & Caching:   ✅ PASS
Test 4 - Multi-Transition DFA:             ✅ PASS
Test 5 - State Consistency (100% valid):   ✅ PASS

Total time: 2.91s
================================================================================

✅ ALL TESTS PASSED - Stage 3 is working correctly!
```

## Utility Files

### `agentspeak_validator.py`

Provides code validation utilities used by the test suite:
- Syntax checking
- Semantic validation
- Code structure analysis

This utility is used internally by `test_stage3_complete.py` and is not intended to be run directly.

## Test Philosophy

Rather than maintaining 18+ separate test files, we consolidate all functionality into one comprehensive test that:
1. **Validates real-world usage**: Tests the complete E2E pipeline
2. **Catches regressions**: All critical functionality tested in one run
3. **Easy to maintain**: Single file to update when code changes
4. **Fast execution**: ~2.9s to validate everything
5. **Clear output**: Easy to identify which functionality failed

## Historical Note

Previous test suite contained 18 separate test files covering:
- Integration tests
- Stress tests
- Multi-transition tests
- Variable abstraction tests
- Constant handling tests
- Scalability tests
- State consistency tests
- Parameterization tests
- Validator tests
- Simple 2-block tests
- Logger integration tests

All functionality from these tests has been consolidated into `test_stage3_complete.py` for better maintainability and easier validation.
