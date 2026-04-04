# Onion Verification Status

This document records the current benchmark-validation status of the pipeline,
the hard project constraints that every stage must respect, and the key generic
repairs that were required to make the non-Stage3 pipeline robust.

Use this file as the first handoff document when opening a new thread.

## Hard Constraints

- `Stage3` must remain single-shot and single-pass.
- `Stage3` must not use a second round, retry repair, projection, runtime-guided repair, or code-side method completion.
- The pipeline must remain domain-agnostic.
- No `blocksworld`-specific, `marsrover`-specific, `satellite`-specific, `transport`-specific, or other domain-specific hardcoding is allowed in the core logic.
- All rules must be derived generically from declared tasks, predicates, primitive actions, and the benchmark query.
- `problem.hddl` must not enter the semantic input of `Stage1` or `Stage3`.
- `problem.hddl` is only allowed in `Stage6` runtime initialisation and `Stage7` official verification.
- Each benchmark problem must correspond to exactly one single-sentence natural-language query.
- That query must be reverse-derived from the root HTN task network in `problem.hddl`.
- The query must explicitly mention task invocation.
- The query must not leak `:init` or `:goal` semantics.
- `tests/test_pipeline.py` is the benchmark end-to-end acceptance harness.
- `Stage4` is a hard-fail gate.
- `nop` must be preserved.
- `Stage7` must align with the IPC HTN `pandaPIparser -verify/-vverify` standard.
- The core contribution remains: generate a general method library without methods.
- That contribution must not be replaced by projection, manual completion, runtime patching, or domain templates.
- Git pushes should use:
  - `http_proxy=http://127.0.0.1:10808`
  - `https_proxy=http://127.0.0.1:10808`

## Current Onion Status

### Layer A: Ground-Truth `Stage1` + Ground-Truth `Stage3`

This layer masks both `Stage1` and `Stage3`, then validates the rest of the
pipeline end to end.

Current status:

- `blocksworld`: `30/30` passed
- `marsrover`: `20/20` passed
- `satellite`: `25/25` passed
- `transport`: `40/40` passed

Total:

- `115/115` passed

Interpretation:

- `Stage2` through `Stage7` are validated end to end under oracle semantic input.
- The remaining research uncertainty is not in the downstream execution chain.

### Layer B: `Stage1` Live + Ground-Truth `Stage3`

This layer uses the real benchmark natural-language input, runs the live
`Stage1` model, masks only `Stage3` with the official HDDL methods, and then
validates `Stage2` through `Stage7` end to end.

Current status:

- `blocksworld`: `30/30` passed
- `marsrover`: `20/20` passed
- `satellite`: `25/25` passed
- `transport`: `40/40` passed

Total:

- `115/115` passed

Interpretation:

- Under the assumption that `Stage3` always returns the correct method library,
  the entire remaining pipeline is currently correct for all four benchmark
  domains and all benchmark problems.
- The main unresolved research target is now the real unmasked `Stage3`.

## Benchmark Query Protocol Status

Benchmark natural-language inputs are now materialised and versioned.

Primary artifacts:

- [official_problem_queries.json](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/benchmark_data/official_problem_queries.json)
- [benchmark_query_manifest.py](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/utils/benchmark_query_manifest.py)
- [nl_instruction_template.md](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/docs/nl_instruction_template.md)

Protocol properties:

- Query generation is deterministic and versioned.
- Queries are reverse-derived from the official root task network.
- Queries expose only task invocation and typed object inventory needed by the
  query.
- Queries do not expose `:init` or `:goal`.

## Key Generic Repairs

The following changes were necessary to make the pipeline robust while staying
within the constraints above.

### Stage 1

- Materialised and versioned benchmark queries.
  - Commit: `f95b140`
- Hardened benchmark query generation and provider interaction.
  - Commits: `a628b30`, `4c02897`
- Compressed large task-grounded benchmark outputs to reduce drift and
  truncation.
  - Commit: `6ae4ec6`
- Minimized benchmark query object inventories to only query-referenced objects
  when grounding is explicit.
  - Commit: `8460f43`
- Salvaged malformed Stage1 JSON prefixes.
  - Commit: `12ec716`
- Lowered the skeletal-output threshold for large explicit task lists.
  - Commit: `ad95007`
- Normalised task-grounded object inventories so placeholder tokens such as
  `ROVER` do not leak into later stages.
  - Commit: `6cd6569`
- Accepted a narrow set of equivalent top-level schema-key aliases such as
  `formulas -> ltl_formulas` and `semantic_objects -> objects`.
  - Commit: `3650d87`

### Stage 2

- Replaced brittle large unordered-eventually handling with generic guarded
  logic.
- Exact fast path now uses a transition-budget criterion rather than a too-small
  state-count criterion.
- Very large unordered benchmark cases can use a symbolic surrogate when the
  query-task literal contract already determines the semantics needed by the
  downstream benchmark harness.
  - Commit: `7d0ad6c`

### Stage 5 to Stage 7

- Hardened masked-oracle downstream execution generically.
  - Commits: `112f3dd`, `aff232b`, `be0fee6`
- Preserved original HDDL fact spellings across chunked replay.
  - Commit: `aff232b`
- Stabilised guided execution and hierarchical-plan handoff to official
  verification.
  - Commit: `bdc6f1b`

## Why These Repairs Matter

The main failures encountered during onion validation were not domain bugs.
They were generic robustness failures:

- Stage1 contract drift on large queries
- Stage1 placeholder-object contamination
- Stage1 malformed or aliased JSON outputs
- Stage2 blow-up on large unordered eventually sets
- Stage6 and Stage7 guided-execution bookkeeping under large schedules

Each fix above was designed to eliminate one of those classes without adding any
domain-specific branch.

## Reproduction

### Full `Stage1` Live + `Stage3` Masked Sweep

Representative harness entry point:

- [test_pipeline.py](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/tests/test_pipeline.py)

Representative invocation pattern:

```bash
PYTHONPATH=src:. ./.venv/bin/python - <<'PY'
import io, json, time, contextlib
from pytest import MonkeyPatch
from tests.test_pipeline import (
    BLOCKSWORLD_PROBLEM_DIR,
    _load_problem_query_cases,
    _run_domain_query_case_with_official_stage3_mask,
)

query_cases = _load_problem_query_cases(BLOCKSWORLD_PROBLEM_DIR, limit=10000)
for idx in range(1, len(query_cases) + 1):
    query_id = f"query_{idx}"
    mp = MonkeyPatch()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report = _run_domain_query_case_with_official_stage3_mask(
                "blocksworld",
                query_id,
                mp,
                query_cases=query_cases,
            )
        print(json.dumps({
            "query_id": query_id,
            "success": bool(report["result"].get("success")),
            "has_bug": bool(report["has_bug"]),
            "log_dir": str(report["log_dir"]),
        }))
        if report["has_bug"]:
            break
    finally:
        mp.undo()
PY
```

Replace the domain directory and domain name for `marsrover`, `satellite`, and
`transport`.

## Current Research Conclusion

As of the current `main` branch:

- The non-Stage3 pipeline is benchmark-complete under oracle `Stage3`.
- The same remains true even when `Stage1` is live and only `Stage3` is oracle-masked.
- The current bottleneck for the paper is therefore the real single-shot,
  domain-agnostic `Stage3` synthesis itself, not the downstream execution chain.

## Latest Relevant Commits

- `3650d87` `fix: accept stage1 schema key aliases`
- `6cd6569` `fix: normalise task-grounded stage1 object inventories`
- `ad95007` `fix: lower skeletal stage1 threshold for large task queries`
- `12ec716` `fix: salvage malformed stage1 json prefixes`
- `7d0ad6c` `fix: harden stage2 large unordered task queries`
- `8460f43` `refactor: minimise benchmark query object inventories`
- `6ae4ec6` `fix: compress stage1 task-grounded benchmark outputs`
- `f95b140` `feat: materialize benchmark query manifest`
- `be0fee6` `fix: harden domain-agnostic oracle-masked pipeline`
- `aff232b` `fix: preserve hddl facts across stage6 chunked replay`
- `112f3dd` `fix: harden domain-agnostic masked pipeline stages`
- `bdc6f1b` `fix: harden masked stage6 guided execution`

## Next Work

The next thread should start from:

- real `Stage3`, with the rest of the pipeline assumed validated under the
  benchmark protocol above
- strict preservation of the project constraints listed at the top of this file
