# TO-DO LIST

## Current Objective

- Keep the dissertation Chapter 4 methodology as the repository source of
  truth:
  - `D^- + L_s -> Φ_s -> M -> S`
- Keep the codebase organized by stage-owned packages only:
  - `src/domain_model/`
  - `src/temporal_specification/`
  - `src/method_library/`
  - `src/plan_library/`
  - `src/evaluation/`
  - `src/compat/`
- Keep generation and evaluation separated:
  - generation builds the plan-library bundle
  - evaluation consumes an existing bundle and records evidence
- Do not reintroduce retired architecture boundaries or terminology into the
  mainline code path.

## Verified Current State

- [x] Repository mainline is aligned with the Chapter 4 package structure.
- [x] The retired method-generation and query-runtime roots have been removed.
- [x] The public command-line interface now uses:
  - `generate-library`
  - `evaluate-library`
- [x] The persisted generation artifact is the plan-library bundle:
  - `artifact_metadata.json`
  - `masked_domain.hddl`
  - `query_sequence.json`
  - `temporal_specifications.json`
  - `method_library.json`
  - `plan_library.json`
  - `plan_library.asl`
  - `translation_coverage.json`
  - `library_validation.json`
- [x] Shared execution logging has been moved to
  `src/execution_logging/` to avoid package-name collisions.
- [x] Package-level imports for `method_library`, `plan_library`, and
  `evaluation` were reduced to lightweight exports to avoid spawned-process
  circular-import failures.
- [x] HTN evaluation worker import failures caused by outdated method-library
  validation imports were repaired.
- [x] `README.md` and `docs/query_protocol.md` now describe the current
  repository structure and query contract.
- [x] Targeted regression coverage currently passes:
  - `215 passed, 2 warnings`
  - command:
    - `PYTHONPATH=src:. uv run pytest -q tests/temporal_specification/test_validation.py tests/plan_library/test_translation.py tests/plan_library/test_pipeline.py tests/evaluation/test_pipeline.py tests/evaluation/test_runtime_units.py tests/evaluation/test_execution_logger.py tests/evaluation/test_structure.py tests/evaluation/test_benchmark_harness_units.py tests/utils/test_config.py tests/method_library/test_generated_domain_build_units.py tests/method_library/test_method_library_quality_gate.py tests/official_benchmark/test_ground_truth_baseline_units.py -k "not full_benchmark"`

## Open Follow-Ups

- [ ] Re-run only the previously polluted HTN evaluation queries after the
  import-path fixes, and do not reuse any result produced during the broken
  worker-import window.
- [ ] Decide whether `src/compat/` should remain as a temporary transition layer
  or be removed after all callers are migrated.
- [ ] Reconfirm that Chapter 5 additions, when finalized, do not break the
  Chapter 4 artifact boundary.
- [ ] If more benchmark evidence is generated, keep it under ignored generated
  output roots rather than committing run artifacts into the repository.

## Guardrails

- Keep the repository domain-agnostic.
- Do not leak benchmark-instance specifics into generation prompts or schemas.
- Do not make query-independent claims about a pipeline that now explicitly
  consumes `L_s` and `Φ_s`.
- Treat grounding, Jason execution, planner runs, and official verification as
  evaluation evidence, not as the primary generation architecture.
