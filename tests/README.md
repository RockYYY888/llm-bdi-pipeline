# Test Suite

This repository keeps the tests focused on the active pipeline.

## Current Coverage

- `tests/stage2_dfa_generation/`
  - DFA conversion and simplification checks
- `tests/stage3_code_generation/test_stage3_htn.py`
  - HTN method synthesis
  - HTN decomposition + preferred specialisation
  - AgentSpeak rendering for positive and negative literals
- `tests/test_pipeline_stage3_htn.py`
  - Pipeline-level Stage 3 integration without calling Stage 1

## Recommended Commands

```bash
./.venv/bin/pytest -q tests/stage2_dfa_generation/test_ltlf2dfa.py
./.venv/bin/pytest -q tests/stage3_code_generation/test_stage3_htn.py
./.venv/bin/pytest -q tests/test_pipeline_stage3_htn.py
```

## Notes

- Stage 1 full end-to-end tests still require an external LLM, so the committed tests avoid that dependency.
