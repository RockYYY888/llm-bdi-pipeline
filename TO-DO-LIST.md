# TO-DO LIST

## Current Objective

- Stabilize the **offline masked-domain method-generation** pipeline so that all
  four benchmark domains can reliably produce:
  - `masked_domain.hddl`
  - generated method library JSON
  - `generated_domain.hddl`
  - passing domain gate output
- Keep the design domain-agnostic:
  - no domain-specific hard coding
  - no benchmark-instance leakage into offline generation
  - no query-conditioned offline synthesis
- Use logger artifacts and real planner / verifier behavior as the main
  debugging signal, not heavyweight semantic validator layers.
- Keep the architectural boundary explicit:
  - offline = domain-only synthesis plus lightweight preflight
  - Hierarchical Task Network planners = evaluation / diagnosis only
  - online Jason runtime = frozen natural-language execution path

## Active Design

- Offline generation path:
  - official `domain.hddl`
  - strip official methods
  - large-model method synthesis from the masked domain only
  - lightweight normalization and minimal structural checks
  - domain gate
  - persist generated artifact
- Online query solution remains compiled and intact, but it is **not** the
  active milestone right now.
- Method-synthesis prompt is method-centric and HDDL-aligned:
  - `compound_tasks`
  - `methods`
  - compact `method_family_schemas`
  - domain-level, query-independent constraints

## Completed This Round

- [x] Refactored the code boundary into three explicit tracks:
  - `src/offline_method_generation/` now owns offline orchestration
  - `src/htn_evaluation/` now owns planner-based problem-root evaluation
  - `src/online_query_solution/` remains the frozen runtime path
- [x] Reduced `DomainCompletePipeline` to a compatibility façade that dispatches
  to:
  - `OfflineDomainSynthesisOrchestrator`
  - `HTNProblemRootEvaluator`
- [x] Removed the remaining offline runtime dependency on
  `DomainCompletePipeline`:
  - `OfflineMethodGenerationPipeline` now builds an
    `OfflineSynthesisContext` directly
  - offline tests and support helpers no longer import the legacy façade
  - offline-only context state no longer carries problem-instance or transition
    runtime fields
- [x] Split mixed benchmark support into:
  - `tests/support/offline_generation_support.py`
  - `tests/support/htn_evaluation_support.py`
  - `tests/support/ground_truth_baseline_support.py` as compatibility re-export
- [x] Added façade-boundary regression tests for:
  - offline build delegation
  - official parallel solver race delegation
  - dedicated offline pipeline orchestration
- [x] Removed the oversized legacy semantic-rule layer from offline generation.
- [x] Split method-synthesis logic into smaller files under
  `src/offline_method_generation/method_synthesis/`.
- [x] Added request diagnostics to method synthesis:
  - request id
  - response mode
  - first chunk timing
  - complete JSON timing
  - reasoning preview when no usable JSON arrives
- [x] Fixed the MiniMax request policy so offline method synthesis:
  - constrains reasoning budget directly
  - does not send an application-side total completion `max_tokens` ceiling
  - uses first-chunk deadlines instead of completion truncation as the main
    latency control
- [x] Fixed `MiniMax` model pinning to `minimax/minimax-m2.7`.
- [x] Replaced the old multi-attempt MiniMax retry ladder with a single-pass
  policy:
  - no provider fallback
  - no transport retry ladder
  - single request only, hard fail on the first unsuccessful call
- [x] Rebased MiniMax reasoning allocation on the routed model context instead of
  tiny fixed reasoning budgets:
  - OpenRouter-routed `minimax/minimax-m2.7` now uses `196608` total context as
    the active budgeting baseline
  - reserve the largest observed visible answer budget
  - reserve about `10%` global context slack
  - reserve prompt-estimate and transport overhead
  - assign `70%` of the remaining headroom to provider-side reasoning
- [x] Added a stable OpenRouter session label for offline synthesis:
  - `session_id = offline-method-generation`
- [x] Confirmed prompt support analysis no longer drops action-pattern
  structure.
- [x] Added domain-gate progress output so long-running synthetic gate cases are
  observable.
- [x] Fixed method-library postprocessing for:
  - duplicate/self ordering edges
  - typed parameter normalization
  - auxiliary `task_args` truncation
  - trivial `true` literals in method context
- [x] Fixed report propagation so generated-build reports now surface the model
  name again.
- [x] Repaired the offline-generation regression suite on the real current test
  tree.
- [x] Collapsed the offline domain gate into a legality-only preflight:
  - declared compound task coverage
  - compound-child closure
  - typed argument/object-scope case recording
  - no planner portfolio inside offline gate
- [x] Added an explicit official-method taxonomy module:
  - `noop_guard`
  - `direct_leaf`
  - `support_then_leaf`
  - `recursive_refinement`
  - `hierarchical_orchestration`
- [x] Re-aligned prompt blueprints to that taxonomy instead of relying on loose
  predicate-headline guesses alone.
- [x] Reduced same-headline support noise in prompt analysis:
  - same-arity declared support tasks now require typed-slot compatibility
  - `headline_support_tasks` now only become strong prompt signals for
    state-goal tasks whose name actually reflects the headline state

## Verified Current State

- [x] Regression suite currently passes:
  - `46 passed, 8 skipped`
  - command:
    - `PYTHONPATH=src:. uv run pytest -q tests/offline_method_generation/test_method_family_taxonomy.py tests/offline_method_generation/test_generated_domain_build_units.py tests/offline_method_generation/test_generated_domain_baseline.py`
- [x] Real sequential live generated build succeeded for `blocksworld`:
  - log: `tests/generated/logs/20260419_110517_BLOCKS/execution.json`
  - request id recorded
  - method synthesis succeeded
  - domain gate succeeded
- [x] Real sequential live generated build succeeded for `marsrover`:
  - log: `tests/generated/logs/20260419_111650_rover/execution.json`
  - request id recorded
  - method synthesis succeeded
  - domain gate succeeded
- [x] Real sequential live generated build succeeded for `satellite`:
  - log: `tests/generated/logs/20260419_115540_satellite2/execution.json`
  - request id recorded
  - method synthesis succeeded
  - domain gate succeeded
- [x] Real sequential live generated build succeeded for `blocksworld` under
  the uncapped-completion MiniMax request policy:
  - log: `tests/generated/logs/20260419_175744_BLOCKS/execution.json`
  - `llm_request_max_tokens = null`
  - first chunk arrived in about `31.8s`
  - method synthesis succeeded
  - domain gate succeeded
- [x] Real sequential live generated build succeeded for `marsrover` under the
  uncapped-completion MiniMax request policy:
  - log: `tests/generated/logs/20260419_175937_rover/execution.json`
  - `llm_request_max_tokens = null`
  - first chunk arrived in about `32.7s`
  - method synthesis succeeded
  - domain gate succeeded
- [x] Real sequential live generated build succeeded for `transport` under the
  uncapped-completion MiniMax request policy:
  - log: `tests/generated/logs/20260419_181204_transport/execution.json`
  - `llm_request_max_tokens = null`
  - first chunk arrived in about `105.7s`
  - method synthesis succeeded
  - domain gate succeeded
- [x] Real sequential live generated build succeeded for `satellite` under the
  uncapped-completion MiniMax request policy:
  - log: `tests/generated/logs/20260419_181413_satellite2/execution.json`
  - `llm_request_max_tokens = null`
  - first chunk arrived in about `218.6s` under the old retry ladder
  - method synthesis succeeded
  - domain gate succeeded

## Current Failures / Open Problems

- [x] `transport` prompt-analysis is now helper-only for `load` / `unload`:
  - no fake `at(...)` headline
  - no fake support wrappers on those two tasks
  - direct leaf primitive families only
- [x] OpenRouter streaming transport now has a raw HTTP fallback path instead of
  relying only on the OpenAI SDK wrapper.
- [ ] Sequential live generation can now succeed on all four domains, but it is
  not yet repetition-stable:
  - the old multi-attempt ladder has now been removed
  - repetition stability must be re-measured under the new single-pass policy
  - provider-side first-chunk variance is still the main instability source

## Important Findings

- Prompt length alone is **not** the main issue:
  - `blocksworld` prompt is about `5661` characters and can succeed
  - `marsrover` prompt is about `12156` characters and can also succeed
  - `satellite` prompt is only about `5401` characters, yet still stalls
- The old timeout-thread approach was unsound:
  - timed-out requests could continue running in the background
  - this caused misleading late progress output and risked unnecessary memory
    growth
- The current transport no longer shows that duplicated late-output behavior in
  the successful / stalled reruns after the refactor.
- For `minimax/minimax-m2.7` offline method synthesis, the current preferred
  policy is:
  - keep total completion uncapped at the application layer
  - constrain reasoning instead
  - compute reasoning from routed context headroom instead of fixed `1/0` values
  - use a single request with a conservative first-chunk deadline, not retries
- `satellite` required two additional real fixes beyond prompt compression:
  - allow streaming responses with raw JSON-like text to continue into the
    existing parse / salvage path
  - accept `ordering` edge objects using `first` / `second` keys
- Prompt-analysis quality matters more than raw prompt length:
  - helper tasks like `auto_calibrate`, `load`, and `unload` needed better
    action-aligned fallback blueprints
  - generic noisy cleanup suggestions had to be reduced
- Official methods across the four domains are not one flat pattern:
  - some tasks are true leaf wrappers
  - some are guarded state-achievement tasks
  - some are recursive relocation / blocker-removal tasks
  - some are mission-level orchestration tasks
- The prompt should therefore start from decomposition family, not from
  predicate headline alone.
- A declared task that shares a headline predicate is **not** automatically a
  good support task:
  - typed-slot compatibility matters
  - state-goal tasks and operational tasks should not be mixed just because
    they affect the same predicate
- The final remaining instability is not explained by prompt length alone:
  - `satellite` can still spend two full request profiles with no first chunk
  - `transport` can be slow to first chunk even when it eventually succeeds
  - this points to provider-side latency / routing variance rather than a local
    parser or domain-gate failure

## Next Steps

- [ ] Keep sequential live generation for now; do **not** parallelize the
  provider-facing generation requests.
- [ ] Re-run sequential live generation to measure repetition stability instead
  of single-success feasibility:
  - first target: `satellite`
  - second target: `transport`
- [ ] Use the new official-method taxonomy as the prompt-design baseline:
  - `do_put_on`-like tasks should bias toward guarded hierarchical delegation
  - `do_move` / `send_soil_data`-like tasks should bias toward support-then-leaf
  - `get-to` / `do_clear`-like tasks should bias toward recursive refinement
  - `deliver` / `get_*_data`-like tasks should bias toward orchestration
- [ ] Preserve the current prompt constraints while reducing response-latency
  risk:
  - no benchmark query leakage
  - no official method leakage
  - no domain-specific special cases
- [ ] After all four domains can generate and pass domain gate reliably in
  sequential live runs, re-run the generated smoke suite.
- [ ] Only after sequential stability is demonstrated, re-test the broader
  generated benchmark sweep behavior.
