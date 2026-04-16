# TO-DO LIST

## Current Focus

- Finish the domain-complete refactor so each domain builds one cached Stage 3/Stage 4 artifact
  and benchmark queries only run Stage 1/2/5/6/7 against that artifact.
- Keep `Stage2 = LTLf -> equivalent DFA`, `Stage3 = domain-only method library`, and keep all
  query grounding and target binding in the query-side execution context.
- Keep Stage 6 free of query-side Panda helper records; any remaining runtime guidance must come
  from Stage 5 lowering or Jason-executable structure.
- Convert the benchmark acceptance harness from the compatibility wrapper to explicit
  `4 domain builds + 115 query executions`, with separate reporting for domain builds and query runs.
- Keep generated outputs isolated under `artifacts/` and `tests/generated/` so source, tests, and
  runtime artifacts stay clearly separated.

## Recent Completed Milestones

- [x] Milestone 26: Split the pipeline into explicit domain-build and query-execution public
  entrypoints, add persisted domain-build artifacts plus query execution context, move official
  Stage 3 mask runs onto `build once + execute with cached library`, and separate generated test
  outputs into `tests/generated`
- [x] Milestone 25: Align Stage 6 guided-prefix target tracking with literal-based transition IDs,
  render observation plans for completed targets, and relax transition-native support gating so
  auxiliary-role requirements (for example calibration) are supported; masked marsrover `query_1`
  runs through end-to-end again
- [x] Milestone 24: Keep benchmark queries as single reverse-generated sentences from official
  `problem.hddl` root tasks plus available typed objects, update NL instruction guidance, and fix
  Stage 3 target-binding normalisation plus constructive-sibling pruning so official blocksworld
  `p01`-`p03` again pass together under one live `tests/test_pipeline.py all` run
- [x] Milestone 21: Remove blocksworld-specific IPC hierarchy repair logic, replace it with
  generic method-trace-based hierarchical plan export, and make Stage 4 method validation
  respect equality-bound runtime task arguments so official blocksworld `p01`-`p03` all pass
  with `Has Bug: False` and official hierarchical verification `True`
- [x] Milestone 22: Promote the official IPC verifier into formal Stage 7, rename log directories
  to `{timestamp}_{domain}_{problem}`, keep `problem.hddl` out of synthesis, and re-accept official
  blocksworld `p01`-`p03` with end-to-end `Has Bug: False`
- [x] Milestone 23: Make Stage 4 query-relevant method-validation failures hard-fail the pipeline
  instead of remaining diagnostic-only, and re-accept official blocksworld `p01`-`p03`
