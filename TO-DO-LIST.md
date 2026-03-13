# TO-DO LIST

## Current Focus

- Keep the default Stage 1-7 path green with Jason runtime execution plus official IPC verification.
- Keep negation semantics unified to all-NAF (`not`) across Stage 3/5/6.
- Keep HDDL parsing on the sound subset only (`and/or/not/imply`; fail-fast on unsupported constructs).
- Keep the benchmark boundary clean: compile from `domain + user query`, use `problem` only for
  Stage 6 runtime init and Stage 7 official verification.
- Expand live coverage beyond blocksworld, starting with the official Mars Rover `pfile01`-`pfile03`
  problems.
- Keep an explicit, code-faithful assumptions boundary document in sync with README.

## Recent Completed Milestones

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
