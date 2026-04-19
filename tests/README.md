# Test Suite

This directory contains the active regression and benchmark-acceptance checks for the current
domain-complete pipeline mainline.

## Coverage Overview

- `tests/online_query_solution/test_execution_logger.py`
  - semantic execution logger schema and active-step visibility
- `tests/online_query_solution/test_structure.py`
  - cleanup guards against retired imports and stage-numbered artifact keys
- `tests/official_benchmark/test_ground_truth_baseline_units.py`
  - problem-structure and official method-library unit coverage
- `tests/official_benchmark/test_ground_truth_baseline.py`
  - official domain preflight and official problem-root smoke coverage
- `tests/run_official_problem_root_baseline.py`
  - parallel four-domain full sweep harness for the `115` official problem-root cases

## Recommended Commands

```bash
./.venv/bin/pytest -q tests/online_query_solution/test_execution_logger.py
./.venv/bin/pytest -q tests/online_query_solution/test_structure.py
./.venv/bin/pytest -q tests/official_benchmark/test_ground_truth_baseline_units.py
./.venv/bin/pytest -q tests/official_benchmark/test_ground_truth_baseline.py -k smoke
./.venv/bin/python tests/run_official_problem_root_baseline.py --domain blocksworld --run-dir tests/generated/tmp
```

Run the full live acceptance sweep only when doing final validation:

```bash
./.venv/bin/python tests/run_official_problem_root_baseline.py
```

## Notes

- Goal-grounding and method-synthesis tests that hit a live model require API access.
- Official planning requires `pandaPIparser`, `pandaPIgrounder`, and `pandaPIengine`.
- `tests/run_official_problem_root_baseline.py` is the canonical benchmark-backed acceptance harness.
