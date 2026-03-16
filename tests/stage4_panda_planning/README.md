# Stage 4 PANDA Tests

This directory covers the current Stage 4 contract: export the generated HTN library to temporary
HDDL artifacts, build the concrete planning problem, and ask PANDA for executability witnesses.

## Main Test Files

```bash
./.venv/bin/pytest -q tests/stage4_panda_planning/test_panda_planner.py
./.venv/bin/pytest -q tests/stage4_panda_planning/test_stage4_panda_planning.py
```

## What These Tests Verify

- temporary HDDL domain/problem export from the generated HTN library
- explicit initial-fact construction by the Stage 4 problem builder
- PANDA invocation and plan parsing
- live planning against the true Stage 3 library structure
- hard-fail witness validation for the current query-specific structure

## What They Do Not Claim

- global state-space completeness
- synthetic wrapper generation
- synthetic guard completion
- post-synthesis method repair

## Notes

- Live tests require `pandaPIparser`, `pandaPIgrounder`, and `pandaPIengine`.
- The default local install path under `$HOME/.local/pandaPI/bin` is auto-discovered when
  available.
