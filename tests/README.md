# Tests Directory

This directory contains all tests for the LTL-BDI pipeline, organized to mirror the `src/` directory structure.

## Directory Structure

```
tests/
├── README.md                           # This file
├── test_integration_pipeline.py        # End-to-end integration tests
├── stage1_interpretation/              # Tests for Stage 1 (NL -> LTLf)
│   └── __init__.py
├── stage2_dfa_generation/              # Tests for Stage 2 (LTLf -> DFA)
│   ├── __init__.py
│   └── test_ltlf2dfa.py               # ltlf2dfa integration tests
└── stage3_code_generation/             # Tests for Stage 3 (DFA -> AgentSpeak)
    └── __init__.py
```

## Test Categories

### Integration Tests (Root Level)
- **`test_integration_pipeline.py`**: End-to-end tests of the complete pipeline
  - Tests all stages working together
  - Based on FOND benchmark problems (bw_5_1, bw_5_3, bw_5_5)
  - Validates NL -> LTLf -> DFA -> AgentSpeak flow

### Stage 1: Natural Language to LTLf (`stage1_interpretation/`)
- Tests for NL parsing and LTL formula generation
- Domain-specific prompt testing
- LLM response validation

### Stage 2: LTLf to DFA Conversion (`stage2_dfa_generation/`)
- **`test_ltlf2dfa.py`**: Tests ltlf2dfa library integration
  - Simple temporal operators (F, G, X, U)
  - Blocksworld-specific formulas
  - Complex nested formulas
  - DFA DOT format validation

### Stage 3: AgentSpeak Code Generation (`stage3_code_generation/`)
- Tests for LLM-based AgentSpeak generation
- Prompt validation
- Generated code syntax checking

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Integration Tests Only
```bash
python tests/test_integration_pipeline.py
```

### Run Stage-Specific Tests
```bash
# Stage 2 tests
python tests/stage2_dfa_generation/test_ltlf2dfa.py

# Or with pytest
pytest tests/stage2_dfa_generation/
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Naming Conventions

- **Integration tests**: `test_integration_*.py` (root level)
- **Unit tests**: `test_<module_name>.py` (in stage-specific folders)
- **Test functions**: `test_<functionality>()`
- **Test classes**: `Test<Component>`

## Adding New Tests

1. **Unit tests**: Add to the appropriate stage folder matching the source module
2. **Integration tests**: Add to root `tests/` directory
3. **Always include docstrings** explaining what the test validates
4. **Follow pytest conventions** for test discovery

## Notes

- All tests use pytest framework
- Tests should be independent and not rely on execution order
- Use fixtures for common setup/teardown
- Mock external API calls (OpenAI) where appropriate
- Integration tests may take longer due to LLM API calls
