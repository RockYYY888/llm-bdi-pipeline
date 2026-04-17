# Domain-Complete Pipeline Assumptions

This file records the current runtime contract of the semantic mainline.

## Scope

The active repository path is:

- offline domain build
  - method synthesis
  - domain gate
- online query execution
  - goal grounding
  - plan solve
  - official verification

The official ground-truth baseline bypasses goal grounding and method synthesis:

- official `domain.hddl`
- official `problem.hddl`
- official root task network
- plan solve
- official verification

## Information Boundaries

1. `domain.hddl` is the only semantic input to offline method synthesis.
2. `problem.hddl` does not enter offline method synthesis or domain gate.
3. Online goal grounding may use typed problem objects from `problem.hddl`.
4. Official verification always checks plans against the original official domain and problem files.

## Planning Contract

1. Problem structure analysis decides whether an instance is total-order or partial-order.
2. Representation construction may add one semantics-preserving linearized representation for partial-order instances.
3. Multiple planner backends may race in parallel across the available representations.
4. A run is successful only if one backend reaches official hierarchical verification success.
5. Finding a primitive plan or a textual hierarchical plan is not sufficient on its own.

## Supported HDDL Boundary

1. The parser targets typed symbolic HDDL used by the benchmark domains in this repository.
2. Unsupported constructs fail fast rather than being approximated silently.
3. Numeric fluents, durations, and stochastic effects are out of scope.

## Naming and Artifact Stability

1. Task and method identifiers must remain lowercase underscore-separated identifiers:
   `[a-z][a-z0-9_]*`
2. Active run artifacts use semantic sections only:
   - `goal_grounding`
   - `method_synthesis`
   - `domain_gate`
   - `plan_solve`
   - `plan_verification`
3. Inactive sections are omitted instead of being recorded as numbered pending stages.

## Verification Boundary

1. Domain build acceptance is decided by domain gate success.
2. Official baseline acceptance is decided only by official verification.
3. Generated-query acceptance is decided only by official verification.

## Current Practical Meaning of Generality

Within the assumptions above, the repository targets domain-complete Hierarchical Task Network planning for the four official benchmark domains without domain-specific hardcoding in the mainline orchestration.
