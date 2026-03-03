# Stage 3 PANDA Tests

Stage 3 now uses HTN method synthesis, PANDA planning,
and AgentSpeak rendering.

## Main Test File

```bash
./.venv/bin/pytest -q tests/stage3_code_generation/test_stage3_panda.py
```

## What It Verifies

- Stage 3A synthesizes an HTN library from the blocksworld HDDL domain
- The synthesizer creates both goal tasks and support tasks
- The PANDA exporter builds a temporary HDDL domain/problem pair
- The planner uses the PANDA PI toolchain for the executable plan
- The AgentSpeak renderer emits:
  - primitive action wrappers
  - PANDA-backed HTN goal plans
  - transition dispatch plans
- Negative DFA literals generate guard-style maintenance plans

## Why This Test Is Small

The goal is to keep Stage 3 verification fast and non-redundant:

- no state-space explosion
- real LLM API for Stage 3A
- PANDA-backed live planning when the toolchain is installed
