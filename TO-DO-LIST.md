# TO-DO LIST

## Current Focus

- Keep the default Stage 1-6 path green with real Jason runtime validation.
- Keep negation semantics unified to all-NAF (`not`) across Stage 3/5/6.
- Keep HDDL parsing on the sound subset only (`and/or/not/imply`; fail-fast on unsupported constructs).
- Expand live coverage beyond blocksworld with official MarsRover HDDL queries and track failures.
- Keep an explicit, code-faithful assumptions boundary document in sync with README.

## Milestones

- [x] Milestone 1: Split Stage 3/4/5 pipeline modules and PANDA integration
- [x] Milestone 2: Custom planner removal and test suite refresh
- [x] Milestone 3: Documentation cleanup and final verification
- [x] Milestone 4: HDDL-only domain migration
- [x] Milestone 5: PANDA toolchain setup, problem-builder extraction, and live acceptance
- [x] Milestone 6: Stage 3 prompt contract tightening for naming and ordering
- [x] Milestone 7: Stage 3 validator tightening and no-silent-sanitisation
- [x] Milestone 8: Semantic task naming, query_i live harness, and negative-goal validation fixes
- [x] Milestone 9: DFA progress-edge wrappers and Stage 5 HTN method-library rendering
- [x] Milestone 10: Free-variable binding validation and context hoisting for Stage 5 methods
- [x] Milestone 11: Stable type-based AgentSpeak variables, Stage 3 CoT-style prompt hardening,
  and semantic target-binding validation
- [x] Milestone 12: Default per-sibling PANDA validation, typed witness-object support,
  and branch-specific method-validation initial-fact construction
- [x] Milestone 13: Stage 3 target-guard completion, direct self-recursive sibling pruning,
  and full live revalidation for query_1 through query_6
- [x] Milestone 14: Root-method validation wrapper fix, equality/disequality preservation,
  and full live revalidation for query_1 through query_6
- [x] Milestone 15: Validation-task closure from target bindings, generic Stage 4 builder defaults,
  and Stage 3 truncation-safe live synthesis
- [x] Milestone 16: Stage 6 symbolic-environment hardening, `not/~` end-to-end support,
  negation-mode auto-resolution diagnostics, and HDDL unsupported-construct fail-fast
- [x] Milestone 17: Publish Stage 1-6 assumption boundary document and link it from README
- [x] Milestone 18: Export Stage 6 executed primitive trace as `action_path.txt`
  and record `src/tests` run origin in pipeline logs
- [x] Milestone 19: Seed Stage 6 from explicit `problem.hddl :init` when provided,
  make `agentspeak_generated.asl` the only runtime ASL artifact, and rename runtime entrypoints
  to neutral `execute/verify_targets`
