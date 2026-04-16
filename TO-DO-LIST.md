# TO-DO LIST

## Current Focus

- Finish the domain-complete refactor so each domain builds one cached Stage 3/Stage 4 artifact
  and benchmark queries run the new `Stage1 -> Temporally Extended Goal -> Stage5 planner ->
  Stage7 verifier` path against that artifact.
- Replace the mainline `QueryExecutionContext` path with `TemporallyExtendedGoal` and
  `PlanningRequestContext`, while keeping Stage 3 and Stage 4 completely domain-only.
- Make the default benchmark and command-line entrypoints use the planner-backed query path rather
  than `Deterministic Finite Automaton / AgentSpeak / Jason`.
- Convert the benchmark acceptance harness to explicit `4 domain builds + 115 query executions`,
  with separate reporting for domain builds, grounding, planner solving, and official verification.
- Keep generated outputs isolated under `artifacts/` and `tests/generated/` so source, tests, and
  runtime artifacts stay clearly separated.

## Active Refactor Notes

- `plan.md` is now the authoritative architecture note for the
  Domain-Complete Hierarchical Task Network refactor with Temporally Extended Goal support.
- `Temporally Extended Goal` means a query-time task graph with grounded task nodes and precedence
  edges, not a return to `Linear Temporal Logic on finite traces`.
- `Belief-Desire-Intention / Jason` is no longer the benchmark-time solver; if retained, it is a
  downstream export target only.

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
