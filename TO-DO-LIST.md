# TO-DO LIST

## Current Objective

- Reorganize `src/` and `tests/` around two primary tracks:
  - offline method generation
  - online query solution
- Reduce reliance on oversized legacy files, especially:
  - `src/pipeline/domain_complete_pipeline.py`
  - `src/offline_method_generation/method_synthesis/synthesizer.py`
  - `src/offline_method_generation/method_synthesis/schema.py`
- Keep the Jason online runtime working during the refactor.
- Use the official benchmark HDDL files as the testing standard while the new
  structure is being stabilized.

## In Progress

- [x] Create an isolated git worktree on `codex/offline-online-reorg`.
- [x] Sync the already-validated Jason online work into the new worktree.
- [x] Sync the local offline method-generation refactor files into the new
  worktree.
- [ ] Replace the giant mixed pipeline orchestrator with smaller offline and
  online entry modules.
- [ ] Rename top-level source and test folders so the structure reflects the
  offline and online split.
- [ ] Remove retired paths and compatibility leftovers that no longer serve the
  active pipeline.

## Verified So Far

- [x] The Jason online runtime can execute a real query end to end in the
  current implementation.
- [x] Query `Put block b2 on the table.` reaches:
  - temporal grounding
  - LTLf to DFA compilation
  - AgentSpeak rendering
  - Jason execution
  - official hierarchical verification

## Pending Validation

- [ ] Re-run the focused regression suite after the structural rename.
- [ ] Re-run the real Jason query after the new module layout is in place.
- [ ] Confirm no stage-number paths or labels remain in active code or tests.
