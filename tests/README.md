# Test Suite

This directory contains the active regression and benchmark-acceptance checks for the current
pipeline mainline.

## Coverage Overview

- `tests/stage2_dfa_generation/`
  - DFA conversion and simplification
- `tests/stage3_method_synthesis/`
  - HTN library synthesis rules and validation gates
- `tests/stage4_panda_planning/`
  - PANDA export, problem building, and live planning checks
- `tests/stage5_agentspeak_rendering/`
  - AgentSpeak rendering from validated HTN structures
- `tests/test_pipeline.py`
  - end-to-end benchmark acceptance harness
  - reverse-generates one single-sentence query per problem
  - currently covers Blocksworld `p01`-`p03` and Marsrover `pfile01`-`pfile03`
  - drives Stage 1 -> Stage 7 and checks saved artifacts

## Recommended Commands

```bash
./.venv/bin/pytest -q tests/stage2_dfa_generation/test_ltlf2dfa.py
./.venv/bin/pytest -q tests/stage3_method_synthesis/test_stage3_method_synthesis.py
./.venv/bin/pytest -q tests/stage4_panda_planning/test_panda_planner.py
./.venv/bin/pytest -q tests/stage4_panda_planning/test_stage4_panda_planning.py
./.venv/bin/pytest -q tests/stage5_agentspeak_rendering/test_agentspeak_renderer.py
./.venv/bin/python tests/test_pipeline.py query_1
```

Run the full live acceptance sweep only when doing final validation:

```bash
PIPELINE_TEST_ALL=1 ./.venv/bin/pytest -q tests/test_pipeline.py
```

## Notes

- Stage 1 and Stage 3 tests that hit a live model require API access.
- Stage 4 live planning requires `pandaPIparser`, `pandaPIgrounder`, and `pandaPIengine`.
- Stage 6 requires Java 17-23 plus the Jason runtime toolchain.
- `tests/test_pipeline.py` is the canonical benchmark-backed acceptance harness.
