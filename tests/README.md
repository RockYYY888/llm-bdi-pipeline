# Test Suite

This repository keeps the tests focused on the active pipeline.

## Current Coverage

- `tests/stage2_dfa_generation/`
  - DFA conversion and simplification checks
- `tests/stage3_code_generation/test_stage3_panda.py`
  - HTN method synthesis
  - PANDA HDDL export + PANDA-backed planning
  - AgentSpeak rendering for positive and negative literals
- `tests/test_pipeline.py`
  - Canonical blocksworld example integration test
  - Calls `pipeline.execute()` across Stage 1, Stage 2, and Stage 3
  - Verifies the logger recorded each stage's inputs, outputs, and saved artifacts

## Recommended Commands

```bash
./.venv/bin/pytest -q tests/stage2_dfa_generation/test_ltlf2dfa.py
./.venv/bin/pytest -q tests/stage3_code_generation/test_stage3_panda.py
./.venv/bin/pytest -q tests/test_pipeline.py
```

## Notes

- Stage 3 live planning also requires `pandaPIparser`, `pandaPIgrounder`, and `pandaPIengine`.
- `tests/test_pipeline.py` is the default example test to run for acceptance.
