# TO-DO LIST

## Current Objective

- Keep the repository tree aligned with the semantic mainline:
  - offline domain build
  - online query execution
- Keep the official ground-truth baseline as the first acceptance gate.
- Re-run the full official ground-truth sweep now that the official benchmark
  profile is fixed in code and the previous `transport` tail cases have flipped
  to verified success.

## Completed

- [x] Replaced the numbered-stage orchestrator and helpers with semantic modules under:
  - `src/pipeline/`
  - `src/domain_build/`
  - `src/query_execution/`
  - `src/planning/`
  - `src/verification/`
- [x] Removed retired numbered-stage runtime paths from tracked source and tests.
- [x] Moved retired code into local-only `archive/legacy_runtime/`.
- [x] Updated the logger and artifact schema to use semantic step names.
- [x] Fixed root-log merging so large planner intermediates stay only under `backend_race/`.
- [x] Completed the semantic official sweep:
  - `4/4` domain preflights passed
  - `115/115` official problem-root runs completed
  - `113/115` verified successes
  - `2/115` solver no-plan failures
- [x] Identified the remaining failures as:
  - `transport pfile39.hddl`
  - `transport pfile40.hddl`
- [x] Cleaned the repository structure further by removing retired documentation, empty legacy folders, and unused compatibility utilities.
- [x] Pushed the semantic cleanup and standalone backend timeout fix to `main`.
- [x] Centralized the official benchmark profile in code:
  - official planning time budget
  - official backend identifiers
  - official PandaDealer and Lifted Linear command profiles
- [x] Re-ran `transport` `query_38`, `query_39`, and `query_40` through the
  file-backed harness.
- [x] Confirmed all three targeted `transport` cases now succeed under the
  fixed official profile:
  - `query_38` -> `lifted_panda_sat`
  - `query_39` -> `lifted_panda_sat`
  - `query_40` -> `lifted_panda_sat`

## Verified

- [x] Semantic regression suite passes for:
  - `tests/pipeline/test_execution_logger.py`
  - `tests/pipeline/test_domain_complete_pipeline.py`
  - `tests/pipeline/test_ground_truth_baseline_units.py`
  - `tests/pipeline/test_ground_truth_baseline.py`
  - `tests/utils/test_config.py`
  - `tests/utils/test_symbol_normalizer.py`
  - `tests/utils/test_negation_mode_resolver.py`

## Remaining

- [x] Compare `transport` `pfile38` with `pfile39` and `pfile40` at the instance-structure level.
- [x] Decide whether the previous remaining failures were structural benchmark failures.
- [ ] Re-run the full `115` official problem-root sweep under the fixed official profile.
- [ ] Confirm the semantic summary updates from `113/115` to the new full-sweep result.
