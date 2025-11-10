# Stage 3 Code Generation Tests

This directory contains integration tests for Stage 3 backward planning system.

## Test Files

### 1. test_integration_backward_planner.py
**Comprehensive integration test suite**

Tests the complete backward planning pipeline:
- DFA input parsing
- Forward state space exploration
- AgentSpeak code generation
- Syntax validation

**Test Cases**:
- **Test 1**: Simple DFA (single transition `on_a_b`)
- **Test 2**: Complex DFA (multiple transitions `on_a_b` → `clear_a`)
- **Test 3**: State graph statistics verification

**Expected Results**: All 3 tests should PASS ✅

---

## Running the Tests

### Using the Test Runner Script (Recommended)
```bash
# From tests/stage3_code_generation directory
cd tests/stage3_code_generation
./run_tests.sh                    # Run all tests
./run_tests.sh integration        # Run integration tests only
./run_tests.sh diagnostic         # Run diagnostic tests only
```

### Run Individual Tests
```bash
# From project root
python tests/stage3_code_generation/test_integration_backward_planner.py
```

**Expected Output**:
```
================================================================================
INTEGRATION TEST SUMMARY
================================================================================
Simple DFA Test:      ✅ PASS
Complex DFA Test:     ✅ PASS
State Graph Test:     ✅ PASS

✅ ALL INTEGRATION TESTS PASSED
```

---

## Test Requirements

All tests require:
- Python 3.8+
- Project dependencies installed
- Access to `src/domains/blocksworld/domain.pddl`

No external test frameworks (pytest, unittest) required - tests are self-contained.

---

## Test Performance

Typical execution times:
- Simple DFA Test: ~3-5 seconds
- Complex DFA Test: ~5-7 seconds
- State Graph Test: ~3-5 seconds
- **Total**: ~15-20 seconds

---

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'src'`:
- Ensure you're running from the project root directory
- Tests automatically add project root to sys.path

### Path Errors
If you see `FileNotFoundError` for domain.pddl:
- Ensure `src/domains/blocksworld/domain.pddl` exists
- Tests use `project_root / 'src' / 'legacy' / 'fond' / ...`

### Test Failures
If tests fail:
1. Check the error message and stack trace
2. Run diagnostic test to verify system components
3. Review test validation criteria
4. Check recent code changes

---

## What the Tests Verify

### Functional Correctness
- ✅ DFA parsing and transition extraction
- ✅ Goal state inference (minimal → complete)
- ✅ Forward state space exploration
- ✅ Bidirectional graph creation
- ✅ Path finding from states to goal
- ✅ AgentSpeak code generation
- ✅ Jason-compatible syntax

### Integration Points
- ✅ Boolean expression parser
- ✅ PDDL domain parser
- ✅ Grounding map
- ✅ Forward planner
- ✅ AgentSpeak code generator

### Design Compliance
- ✅ All 16 core design decisions
- ✅ All 18 Q&A requirements
- ✅ Proper import paths (src. prefix)

---

## Test Results Archive

See `docs/stage3_integration_test_results.md` for:
- Detailed test results
- Performance metrics
- Bug fix history
- Design verification

---

## Continuous Testing

To ensure system health:
1. Run tests after any code changes
2. Run tests before committing
3. Run tests after pulling updates
4. Add new tests for new features

---

**Last Updated**: 2025-11-07
**Status**: ✅ All tests passing
**Coverage**: Core backward planning pipeline
