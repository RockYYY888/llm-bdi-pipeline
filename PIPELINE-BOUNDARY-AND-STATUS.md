# Pipeline Boundary And Status

Last updated: 2026-04-19

## 1. Purpose of this document

This document records the current structure of the whole pipeline after the
offline versus online boundary refactor.

The repository now has three explicit tracks:

1. Offline domain synthesis
2. Hierarchical Task Network evaluation
3. Online Jason runtime

The crucial architectural rule is:

- Offline is domain-only synthesis.
- Hierarchical Task Network planners are evaluation and diagnostic instruments.
- Online Jason runtime is the deployed natural-language execution path.

This document records:

- which modules own each track
- what each track is allowed to read
- what each track must not do
- what each track produces
- which tests and harnesses belong to each track
- what has already been verified
- what is still pending and should be treated as `PENDING`

## 2. High-level architecture

### 2.1 Track overview

#### Track A: Offline domain synthesis

Input:

- one official `domain.hddl`

Output:

- one `masked_domain.hddl`
- one generated method library
- one `generated_domain.hddl`
- one domain-gate preflight result

Responsibility:

- generate a reusable domain-level plan library
- do not solve benchmark problem instances
- do not read benchmark query text
- do not read `problem.hddl`

#### Track B: Hierarchical Task Network evaluation

Input families:

- official benchmark pair:
  - official `domain.hddl`
  - official `problem.hddl`
- generated benchmark pair:
  - generated `domain.hddl`
  - official `problem.hddl`

Output:

- planner race result
- official verification result
- benchmark summary and failure buckets

Responsibility:

- provide a reference baseline
- probe generated-domain quality
- diagnose whether failure comes from offline synthesis quality or from the
  online runtime

Non-responsibility:

- not the deployed runtime solver
- not part of the natural-language execution chain

#### Track C: Online Jason runtime

Input:

- natural-language query
- one benchmark `problem.hddl`
- one library artifact or generated-domain-backed library

Runtime chain:

- natural language
- one Linear Temporal Logic on finite traces formula
- deterministic finite automaton
- AgentSpeak
- Jason
- hierarchical plan text
- benchmark verification

Responsibility:

- execute benchmark-derived natural-language tasks with the Belief-Desire-Intention runtime

Non-responsibility:

- no Hierarchical Task Network planner solving in the deployed runtime path

## 3. Source tree ownership

### 3.1 Offline domain synthesis ownership

Primary package:

- `src/offline_method_generation/`

Current owner modules:

- `src/offline_method_generation/pipeline.py`
  - public offline entrypoint
  - class: `OfflineMethodGenerationPipeline`
- `src/offline_method_generation/context.py`
  - offline-only synthesis context
  - owns domain parsing, type maps, logger wiring, and lightweight typing checks
  - must not import `DomainCompletePipeline`
- `src/offline_method_generation/orchestrator.py`
  - main offline orchestrator
  - class: `OfflineDomainSynthesisOrchestrator`
- `src/offline_method_generation/method_synthesis/`
  - prompt construction
  - prompt-analysis payload building
  - large-language-model transport
  - schema parsing
  - postprocessing
  - minimal validation
- `src/offline_method_generation/domain_gate/validator.py`
  - domain-only planner-acceptability preflight
  - class: `OfflineDomainGateValidator`
- `src/offline_method_generation/artifacts.py`
  - persisted offline artifact structure

### 3.2 Hierarchical Task Network evaluation ownership

Primary package:

- `src/htn_evaluation/`

Current owner modules:

- `src/htn_evaluation/problem_root_evaluator.py`
  - worker launch
  - solver race
  - attempt selection
  - official verification integration
  - selected-attempt artifact promotion
  - class: `HTNProblemRootEvaluator`

Shared lower-level planning and verification modules used by this track:

- `src/planning/problem_structure.py`
- `src/planning/representations.py`
- `src/planning/backends.py`
- `src/planning/linearization.py`
- `src/planning/problem_encoding.py`
- `src/planning/panda_portfolio.py`
- `src/planning/plan_models.py`
- `src/planning/process_capture.py`
- `src/planning/official_benchmark.py`
- `src/verification/official_plan_verifier.py`

### 3.3 Online Jason runtime ownership

Primary package:

- `src/online_query_solution/`

Current owner modules:

- `src/online_query_solution/pipeline.py`
  - public online entrypoint
  - class: `OnlineQuerySolutionPipeline`
- `src/online_query_solution/goal_grounding/`
- `src/online_query_solution/temporal_compilation/`
- `src/online_query_solution/agentspeak/`
- `src/online_query_solution/jason_runtime/`
- `src/online_query_solution/domain_selection.py`

Current rule:

- this path is frozen semantically during the current refactor

### 3.4 Temporary compatibility layer

Compatibility facade:

- `src/pipeline/domain_complete_pipeline.py`

Current role:

- still the central object that many tests and harnesses instantiate
- no longer the architectural source of truth
- now dispatches to:
  - `OfflineDomainSynthesisOrchestrator`
  - `HTNProblemRootEvaluator`
  - the frozen online runtime methods

Offline boundary rule:

- `src/offline_method_generation/` must not import or instantiate
  `DomainCompletePipeline`
- compatibility support for legacy façade calls belongs outside the offline
  package

### 3.5 Command-line entrypoints

Primary command-line entrypoint:

- `src/main.py`

Current dispatch rule:

- `--build-domain-library`:
  - goes to `OfflineMethodGenerationPipeline`
- otherwise:
  - goes to `OnlineQuerySolutionPipeline`

Current note:

- planner-based benchmark evaluation is not exposed as a first-class production
