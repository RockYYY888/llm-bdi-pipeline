# Test Suite

This directory contains the active regression and benchmark-acceptance checks for the current
domain-complete pipeline mainline.

## Test Command

From the repository root, run:

```bash
uv run pytest
```

The default suite is designed to pass from a clean checkout with only Python
dependencies installed. Tests that require unavailable external planner
toolchains are skipped.

## Coverage Overview

- `tests/evaluation/test_execution_logger.py`
  - semantic execution logger schema and active-step visibility
- `tests/evaluation/test_structure.py`
  - cleanup guards against retired imports and stage-numbered artifact keys
- `tests/official_benchmark/test_ground_truth_baseline_units.py`
  - problem-structure and official method-library unit coverage
- `tests/official_benchmark/test_ground_truth_baseline.py`
  - official domain preflight and official problem-root smoke coverage
- `tests/run_official_problem_root_baseline.py`
  - parallel four-domain full sweep harness for the `115` official problem-root cases

## Recommended Commands

```bash
uv run pytest -q tests/evaluation/test_execution_logger.py
uv run pytest -q tests/evaluation/test_structure.py
uv run pytest -q tests/official_benchmark/test_ground_truth_baseline_units.py
uv run pytest -q tests/official_benchmark/test_ground_truth_baseline.py -k smoke
uv run python tests/run_official_problem_root_baseline.py --domain blocksworld --run-dir tests/generated/tmp
```

Run the full live acceptance sweep only when doing final validation:

```bash
uv run python tests/run_official_problem_root_baseline.py
```

## Notes

- Goal-grounding and method-synthesis tests that hit a live model require API access.
- Official planning requires `pandaPIparser`, `pandaPIgrounder`, and `pandaPIengine`.
- `tests/run_official_problem_root_baseline.py` is the canonical benchmark-backed acceptance harness.
