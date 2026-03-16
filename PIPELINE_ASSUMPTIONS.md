# Pipeline Assumptions and Validation Boundary

This document states the assumptions under which the default
`Stage 1 -> Stage 7` mainline is intended to operate. It describes the current
runtime contract, not historical variants.

## 1. Mainline Scope

The active path is `src/main.py -> LTL_BDI_Pipeline.execute(mode="dfa_agentspeak")`.
Under that path:

- Stage 1 interprets one natural-language query into predicate-grounded LTLf.
- Stage 3 performs one single-call LLM synthesis of a query-specific HTN method library.
- No stage is allowed to project missing methods from the gold domain, run runtime-guided repair,
  or inject code-authored replacement methods after synthesis.

## 2. Runtime and Toolchain Assumptions

1. A domain file must be provided explicitly with `--domain-file`.
2. Live LLM access is required because Stage 1 and Stage 3 both depend on model output.
3. Stage 2 requires `ltlf2dfa`, MONA, and the `dd` BDD library.
4. Stage 4 requires `pandaPIparser`, `pandaPIgrounder`, and `pandaPIengine`.
5. Stage 6 requires Java 17-23 plus a buildable Jason CLI tree under
   `src/stage6_jason_validation/jason_src`.

## 3. Supported HDDL Subset

The current parser and runtime are designed for a typed symbolic subset of HDDL:

1. Preconditions/effects may use `and`, `or`, `not`, `imply`, and equality/disequality.
2. `when`, `forall`, and `exists` are unsupported and fail fast.
3. Disjunctive effects (`(or ...)` inside `:effect`) are unsupported and fail fast.
4. Runtime semantics are boolean symbolic updates over facts. Numeric fluents, durations,
   and stochastic effects are out of scope.

## 4. Information-Flow Boundary

The pipeline intentionally separates domain semantics from problem-instance semantics.

1. Stage 1 consumes the natural-language query plus domain signatures.
2. Stage 3 consumes the query, Stage 2 targets, and domain signatures.
3. `domain.methods` are not semantic input to Stage 1 or Stage 3.
4. `problem.hddl` is not semantic input to Stage 1 or Stage 3.
5. `problem.hddl` is used only for:
   - Stage 6 runtime initialisation when available
   - Stage 7 official IPC verification

## 5. Query Assumptions

1. Queries must be expressible over predicates declared in the domain.
2. Predicate arity must match domain signatures.
3. For benchmark-backed acceptance, each `problem.hddl` instance is paired with exactly one
   reverse-generated single-sentence query built from:
   - the problem's root HTN tasks
   - the typed object inventory available to execution
4. Query text may mention declared task invocations, but Stage 1 must still output
   predicate-grounded LTLf rather than task atoms.

## 6. Typing and Identifier Assumptions

1. Relevant objects and variables must resolve to legal domain types.
2. Type evidence is unified from predicates, actions, tasks, and method bindings.
3. Conflicting or ambiguous type assignments are hard errors.
4. Stage 3 task and method identifiers must remain AgentSpeak-safe:
   `[a-z][a-z0-9_]*`.
5. Primitive action aliases normalise HDDL names by replacing `-` with `_`.
6. Variable detection remains uppercase-first in the Stage 3/4/5 export path.

## 7. Validation Boundary

1. Stage 4 is a hard-fail existence-witness gate over the generated library and the current
   query structure. It is not a full state-space completeness proof.
2. Stage 4 validates the real generated library only. It does not add synthetic wrappers,
   synthetic guards, or synthetic bridge methods.
3. Stage 6 is a hard-fail Jason runtime execution gate for the generated AgentSpeak program.
4. Stage 7 is a hard-fail official IPC HTN verification gate using `pandaPIparser`.

## 8. Negation Policy

1. The current policy is `all_naf`.
2. Negative literals are rendered as `not p(...)`.
3. A strong-negation track (`~p`) is not active in the current mainline.

## 9. Practical Meaning of "General"

Within the boundary above, the repository targets typed symbolic HDDL domains in the supported
subset without domain-specific hardcoding in the core Stage 1 -> Stage 7 logic.
