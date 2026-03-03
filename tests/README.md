# Test Suite

This repository keeps the tests focused on the active pipeline.

## Current Coverage

- `tests/stage2_dfa_generation/`
  - DFA conversion and simplification checks
- `tests/stage3_method_synthesis/`
  - HTN method synthesis
  - HTN library validation checks
- `tests/stage4_panda_planning/`
  - PANDA HDDL export + plan parsing
  - Live PANDA planning using the synthesised HTN library
- `tests/stage5_agentspeak_rendering/`
  - AgentSpeak rendering from PANDA plan records
- `tests/test_pipeline.py`
  - Canonical blocksworld example integration test
  - Calls `pipeline.execute()` across Stage 1, Stage 2, Stage 3, Stage 4, and Stage 5
  - Verifies the logger recorded each stage's inputs, outputs, and saved artifacts

## Recommended Commands

```bash
./.venv/bin/pytest -q tests/stage2_dfa_generation/test_ltlf2dfa.py
./.venv/bin/pytest -q tests/stage3_method_synthesis/test_stage3_method_synthesis.py
./.venv/bin/pytest -q tests/stage4_panda_planning/test_panda_planner.py
./.venv/bin/pytest -q tests/stage4_panda_planning/test_stage4_panda_planning.py
./.venv/bin/pytest -q tests/stage5_agentspeak_rendering/test_agentspeak_renderer.py
./.venv/bin/pytest -q tests/test_pipeline.py
```

## Notes

- Stage 4 live planning requires `pandaPIparser`, `pandaPIgrounder`, and `pandaPIengine`.
- `tests/test_pipeline.py` is the default example test to run for acceptance.
