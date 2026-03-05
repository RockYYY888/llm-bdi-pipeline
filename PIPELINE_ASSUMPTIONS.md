# Pipeline Assumptions and Validation Boundary

This document states the assumptions required for the current Stage 1-6
pipeline to behave as designed. It is intentionally faithful to the code path
in `src/main.py -> LTL_BDI_Pipeline.execute(mode="dfa_agentspeak")`.

## 1. Runtime and Toolchain Assumptions

1. A domain file must be explicitly provided (`--domain-file`) and must exist.
2. Live LLM access is required (`OPENAI_API_KEY`) because Stage 1 and Stage 3
   both depend on LLM output.
3. Stage 2 requires `ltlf2dfa` + MONA and the BDD library (`dd`).
4. Stage 4 requires PANDA PI binaries:
   `pandaPIparser`, `pandaPIgrounder`, `pandaPIengine`.
5. Stage 6 requires Java 17-23, `javac`, and a buildable local Jason source
   tree at `src/stage6_jason_validation/jason_src`.

## 2. HDDL Language Subset Assumptions

The action-condition parser is sound only for a boolean subset:

1. Supported in preconditions/effects: `and`, `or`, `not`, `imply`, and
   equality/disequality (`=`).
2. Explicitly unsupported and fail-fast: `when`, `forall`, `exists`.
3. Disjunctive effects are rejected (`(or ...)` in `:effect` is fail-fast).
4. Stage 6 environment semantics are boolean symbolic state updates
   (`add/remove` facts), not numeric fluents, temporal durations, or stochastic
   effects.

## 3. Query-to-Domain Semantic Alignment Assumptions

1. User goals must compile to literals over predicates declared in the domain
   (plus `=` constraints).
2. Predicate arity must match domain signatures.
3. Query intents outside predicate-level symbolic semantics (for example,
   "better", "safer", "more efficient" without formal predicates) are not
   guaranteed to compile.

## 4. Typing Assumptions (Strict and Fail-Fast)

1. Every relevant object/variable must resolve to a unique legal type.
2. Type evidence is unified from predicate signatures, action signatures, task
   signatures, and method/task bindings.
3. Unknown types, conflicting constraints, missing evidence, and ambiguous
   leaf-type candidates are all hard errors.
4. Multi-type export to Stage 4 requires explicit object->type assignments for
   objects used in the generated problem.

## 5. Identifier and Symbol Conventions

1. Stage 3 expects AgentSpeak-safe identifiers (`[a-z][a-z0-9_]*`) for task,
   method, and subtask names.
2. Variable detection is uppercase-first in Stage 3/4/5 code paths. Naming that
   violates this convention may be interpreted incorrectly.
3. Primitive action aliases are normalized from HDDL names by replacing `-`
   with `_` at runtime.

## 6. Negation Assumptions (Current Policy)

1. Current policy is `all_naf`: negative literals are interpreted as
   `not p(...)` (negation-as-failure).
2. Strong negation (`~p`) is not active in the current mainline.

## 7. Validation Boundary Assumptions

1. Stage 4 is existence-witness validation for query-relevant structure, not a
   full state-space completeness proof.
2. Stage 6 is a real Jason execution gate with runtime markers and environment
   adapter checks, but still a run-level witness check, not exhaustive
   verification.
3. Stage 4 method-validation failures are recorded in artifacts; they are not
   currently promoted to a hard Stage 4 failure by default.

## 8. Known Interpretation Caveats

1. Stage 2 summary counters (`num_states`, `num_transitions`) may under-report
   real DFA size in some graphs due to legacy counting heuristics. Structural
   truth should be read from `dfa_simplified.dot` and
   `simplification_stats`.
2. Stage 5 may generate multiple `+!run_dfa` plans for one source state when
   multiple outgoing wrappers exist; Jason handles this as normal BDI plan
   selection, not explicit deterministic branching encoded by us.

## 9. Practical Definition of "General" for This Repository

Under the assumptions above, the pipeline is general for typed symbolic HDDL
domains within the supported subset and does not rely on blocksworld-specific
hardcoding in core Stage 1-6 logic.
