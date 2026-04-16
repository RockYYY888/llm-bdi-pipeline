# plan.md — Domain-Complete Hierarchical Task Network Refactor with Temporally Extended Goal Support

## Summary

This refactor replaces the current `natural language -> Linear Temporal Logic on finite traces -> Deterministic Finite Automaton -> AgentSpeak -> Jason` execution pipeline with a two-pipeline architecture centered on:

1. **Offline domain build**
   - `domain.hddl -> Stage3 domain-complete method synthesis -> Stage4 domain gate -> cached method library`

2. **Online query execution**
   - `natural language query + problem.hddl + cached method library -> Stage1 goal grounding -> Temporally Extended Goal construction -> Stage5 Hierarchical Task Network solve -> Stage7 official verification`

The system contribution becomes:

- automatic generation of a domain-complete Hierarchical Task Network method library from `domain.hddl`
- natural-language grounding into a typed, domain-valid **Temporally Extended Goal** request
- solving with a standard Hierarchical Task Network planner, not with a Belief-Desire-Intention runtime solver

The `Temporally Extended Goal` layer replaces the old role previously played by `Linear Temporal Logic on finite traces / Deterministic Finite Automaton` for structured query semantics. It preserves future support for sequencing and partial order, while keeping the main solver path entirely within the Hierarchical Task Network planning framework.

## Key Changes

### 1. Remove `Linear Temporal Logic on finite traces / Deterministic Finite Automaton / AgentSpeak / Jason` from the primary execution path

The primary benchmark path will no longer use:
- `Stage2 Deterministic Finite Automaton generation`
- AgentSpeak rendering
- Jason runtime solving

They may remain temporarily in a clearly isolated experimental area during migration, but:
- they must not be used by default
- they must not participate in benchmark acceptance
- they must not define the main public interfaces
- once the new masked and live baselines are green, the remaining active mainline code for them should be physically deleted

### 2. Promote `Temporally Extended Goal` to the main query-time semantic representation

Introduce a new first-class type:
- `TemporallyExtendedGoal`
- or `GoalGroundingContext` containing a `TemporallyExtendedGoal`

Recommended semantics:
- nodes represent grounded top-level task invocations, not arbitrary logical literals
- edges represent precedence constraints
- the structure is a directed acyclic request graph, not a temporal logic formula

Minimum fields:
- original query text
- grounded task nodes
- typed arguments per node
- precedence edges
- optional unordered groups or independent components
- grounding diagnostics
- typed object inventory from `problem.hddl`

Architectural decision:
- the representation should support **partial order from day one**
- benchmark v1 may mostly exercise single-node and simple ordered cases, but the interface must not be restricted to a single node only

### 3. Keep Stage3/Stage4 domain-only

`build_domain_library(domain_file, output_root, ...)` remains the offline entrypoint.

Persisted domain artifact contains only:
- `compound_tasks`
- `primitive_tasks`
- `methods`
- `stage3_metadata`
- `stage4_domain_gate`

It must not persist:
- query text
- grounded objects
- target literals
- target-task bindings
- any query-specific validation data

Stage3 remains:
- single-shot large language model synthesis
- driven only by `domain.hddl`
- based on the Hybrid Graph-to-Contracts internal analysis

Stage4 remains:
- domain gate only
- validates coverage, child-task closure, and planner acceptability under synthetic lifted witnesses
- emits no query-time replay or ordering artifacts

### 4. Replace `QueryExecutionContext` with a planner-oriented request model

Retire the current mainline `QueryExecutionContext`, because it is shaped around the removed temporal and AgentSpeak path.

Add:
- `TemporallyExtendedGoal`
- `PlanningRequestContext`

`PlanningRequestContext` should include:
- original query text
- typed problem object inventory
- grounded `TemporallyExtendedGoal`
- compiled planner request task network
- grounding warnings or ambiguity diagnostics

It should not include:
- `target_literals`
- `target_task_bindings`
- `Deterministic Finite Automaton` states or transitions
- AgentSpeak or Jason execution metadata

### 5. Make `Stage5` the Hierarchical Task Network solver stage

`execute_query_with_library(...)` should become:

1. load cached domain artifact
2. run Stage1 natural-language grounding
3. build `TemporallyExtendedGoal`
4. compile `TemporallyExtendedGoal` into a planner request network
5. run `PANDAPlanner` directly on:
   - original domain
   - original problem initial state
   - generated method library
   - compiled request network
6. return the hierarchical plan and planning metadata
7. run official verification

Compilation rule:
- single-node `TemporallyExtendedGoal` becomes a single top-level task request
- multi-node ordered `TemporallyExtendedGoal` becomes an ordered synthetic task network
- multi-node partially ordered `TemporallyExtendedGoal` becomes a precedence-constrained synthetic task network when supported by the planner/export path

### 6. Redefine the stage story

Primary numbered pipeline becomes:

- `Stage1`: natural-language goal grounding
- `Stage3`: domain-complete method synthesis
- `Stage4`: domain gate
- `Stage5`: Hierarchical Task Network solve
- `Stage7`: official verification

`Stage2 Deterministic Finite Automaton generation` is retired from the mainline.

`Belief-Desire-Intention` becomes optional downstream export only:
- not used in benchmark solving
- not required for baseline recovery
- may later consume generated methods or solved hierarchical plans as an export target

### 7. Logger, profiler, and repository cleanup

Logger requirements:
- distinguish `domain_build` runs from `query_execution` runs
- report timings only for active mainline stages
- remove stale `Deterministic Finite Automaton / AgentSpeak / Jason` payload expectations from primary logs
- keep artifacts under:
  - `artifacts/domain_builds/...`
  - `artifacts/runs/...`
  - `tests/generated/...`

Profiler requirements:
- domain build timings:
  - Stage3 total
  - Stage4 total
- query execution timings:
  - Stage1 grounding
  - `Temporally Extended Goal` construction
  - request compilation
  - Stage5 planning
  - Stage7 verification

Repository cleanup requirements:
- default tests must stop importing the removed primary-path modules
- old temporal / AgentSpeak / Jason tests must be moved out of the default benchmark suite or deleted once their code is removed
- public command-line help, docs, and execution summaries must describe the new planning path, not the old solver chain

## Public Interfaces and Type Changes

### Keep
- `build_domain_library(domain_file, output_root, ...)`

### Redefine
- `execute_query_with_library(domain_file, problem_file, nl_query, library_artifact, ...)`
  - old meaning: `natural language -> Deterministic Finite Automaton -> AgentSpeak -> Jason`
  - new meaning: `natural language -> Temporally Extended Goal grounding -> Hierarchical Task Network solve`

### Add
- `TemporallyExtendedGoal`
- `PlanningRequestContext`
- planner solve result payload containing:
  - grounded request summary
  - `TemporallyExtendedGoal` summary
  - planner task-network artifact path
  - hierarchical plan artifact path
  - verification summary

### Retire from the mainline
- `QueryExecutionContext` in its current shape
- `dfa_result` in query execution outputs
- AgentSpeak artifacts in query execution outputs
- Jason validation artifacts in query execution outputs

## Test Plan

### 1. Domain build tests
- domain-complete prompt contains all declared compound tasks and no query text
- generated library contains no query-bound fields
- Stage3 metadata records domain contracts only
- Stage4 domain gate runs without query input
- Stage4 rejects missing task coverage and unresolved child tasks
- Stage4 emits no query-specific runtime or replay records

### 2. `Temporally Extended Goal` grounding tests
- natural language grounding resolves only to declared domain tasks
- grounding uses `problem.hddl` objects and types only at query time
- single-task queries produce a one-node `TemporallyExtendedGoal`
- ordered natural-language queries produce precedence edges
- ambiguous grounding produces explicit diagnostics instead of silent guessing

### 3. Planner request compilation tests
- one-node `TemporallyExtendedGoal` compiles to one top-level task request
- ordered multi-node `TemporallyExtendedGoal` compiles to an ordered task network
- partially ordered `TemporallyExtendedGoal` preserves precedence constraints in the planner request representation
- compiled requests use generated methods without requiring any `Deterministic Finite Automaton` or AgentSpeak intermediary

### 4. Planner execution tests
- cached domain library plus compiled request runs through `PANDAPlanner`
- returned hierarchical plan is structurally valid
- official verification accepts valid benchmark cases
- query execution logs no longer mention:
  - `Linear Temporal Logic on finite traces`
  - `Deterministic Finite Automaton`
  - AgentSpeak
  - Jason

### 5. Benchmark acceptance
Primary acceptance is split into two suites:

1. **Masked baseline**
- `4` domain builds with official masked generation
- `115` real query executions with:
  - real natural-language query
  - real `problem.hddl`
  - cached generated method library
  - `Temporally Extended Goal` grounding
  - planner solve
  - official verification

2. **Live baseline**
- one live generation per domain
- then all `115` query executions against cached live artifacts

Acceptance reporting must separate:
- domain-build pass/fail
- grounding pass/fail
- `Temporally Extended Goal` compilation pass/fail
- planner-solve pass/fail
- official-verification pass/fail

## Assumptions and Defaults

- `TEG` is interpreted as **temporally extended goal**
- `Temporally Extended Goal` is the new query-time semantic layer replacing the old temporal compilation role
- `domain.hddl` is the only semantic input to method generation
- `problem.hddl` is not used in Stage3 or Stage4
- natural language remains in the system only as the user-facing goal input
- `Linear Temporal Logic on finite traces` is not part of the primary pipeline
- `Deterministic Finite Automaton` is removed from the mainline rather than preserved as a co-equal execution mode
- `Belief-Desire-Intention` is a downstream export target, not the benchmark solver
- the architecture supports partial-order `Temporally Extended Goal` from the start, even if the first recovered baselines mostly exercise one-node or simple ordered cases
- once the new masked and live baselines are green, the remaining mainline `Deterministic Finite Automaton / AgentSpeak / Jason` code should be physically deleted
