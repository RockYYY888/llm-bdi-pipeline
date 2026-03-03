# Stage 3 HTN Tests

Stage 3 now uses HTN method synthesis, decomposition, preferred specialisation,
and AgentSpeak rendering.

## Main Test File

```bash
./.venv/bin/pytest -q tests/stage3_code_generation/test_stage3_htn.py
```

## What It Verifies

- Stage 3A synthesizes an HTN library from the blocksworld PDDL domain
- The synthesizer creates both goal tasks and support tasks
- The planner builds a decomposition trace for target literals
- Preferred specialisation retains the relevant abstract cut
- The AgentSpeak renderer emits:
  - primitive action wrappers
  - specialised HTN goal plans
  - transition dispatch plans
- Negative DFA literals generate guard-style maintenance plans

## Why This Test Is Small

The goal is to keep Stage 3 verification fast and non-redundant:

- no state-space explosion
- no dependency on external APIs
