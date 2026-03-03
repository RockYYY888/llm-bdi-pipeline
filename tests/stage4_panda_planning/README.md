# Stage 4 PANDA Tests

Stage 4 now consumes the Stage 3 HTN method library and runs PANDA planning.

## Main Test Files

```bash
./.venv/bin/pytest -q tests/stage4_panda_planning/test_panda_planner.py
./.venv/bin/pytest -q tests/stage4_panda_planning/test_stage4_panda_planning.py
```

## What It Verifies

- The PANDA exporter builds a temporary HDDL domain/problem pair
- The planner uses the PANDA PI toolchain for the executable plan
- The live planning path accepts a Stage 3 synthesised HTN library
- PANDA returns executable primitive actions for the target task

## Why These Tests Stay Small

The goal is to keep Stage 4 verification fast and non-redundant:

- no state-space explosion inside the test harness
- real LLM API only where Stage 3 synthesis is a prerequisite
- PANDA-backed live planning when the toolchain is installed
