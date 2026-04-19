# Domain-Complete Hierarchical Task Network Pipeline

This repository now has two active paths:

- offline masked-domain method generation
- online natural-language query execution through `LTLf -> Deterministic Finite Automaton -> AgentSpeak -> Jason`
- generated-domain benchmark evaluation on official `problem.hddl` files

The current mainline is:

- offline domain build:
  `official domain.hddl -> strip official methods -> domain-complete method synthesis -> domain gate -> generated domain artifact`
- online query execution:
  `natural-language query -> large-language-model grounded LTLf + grounded subgoals -> Deterministic Finite Automaton -> AgentSpeak -> Jason -> official hierarchical verification`
- generated benchmark evaluation:
  `generated domain artifact + official problem.hddl root task network -> plan solve -> official verification`

## Current Milestone Rules

- all generation code must remain domain-agnostic
- no domain-specific hard-coded shortcuts or projections
- offline method generation is domain-level only
- online query grounding is large-language-model driven; code only validates, compiles, executes, and verifies
- online query grounding must emit grounded `subgoal_*` atoms only
- one full generated sweep should issue exactly four large-model requests:
  one per benchmark domain
- benchmark query text in
  [`src/benchmark_data/benchmark_queries.json`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/benchmark_data/benchmark_queries.json)
  is used by the Jason online benchmark harness, not offline synthesis
- generated benchmark failures are expected to drive prompt engineering, not
  ad hoc domain-specific patches

## Repository Layout

```text
.
├── src/
│   ├── offline_method_generation/
│   │   ├── method_synthesis/
│   │   └── domain_gate/
│   ├── online_query_solution/
│   │   ├── goal_grounding/
│   │   ├── temporal_compilation/
│   │   ├── agentspeak/
│   │   └── jason_runtime/
│   ├── planning/
│   ├── pipeline/
│   ├── verification/
│   ├── domains/
│   ├── benchmark_data/
│   └── utils/
├── tests/
│   ├── offline_method_generation/
│   ├── online_query_solution/
│   ├── official_benchmark/
│   ├── support/
│   └── utils/
└── TO-DO-LIST.md
```

## Main Entry Points

- command-line entry point:
  [`src/main.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/main.py)
- offline entrypoint:
  [`src/offline_method_generation/pipeline.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev-reorg/src/offline_method_generation/pipeline.py)
- online entrypoint:
  [`src/online_query_solution/pipeline.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev-reorg/src/online_query_solution/pipeline.py)
- compatibility orchestrator:
  [`src/pipeline/domain_complete_pipeline.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/pipeline/domain_complete_pipeline.py)
- official verifier wrapper:
  [`src/verification/official_plan_verifier.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/verification/official_plan_verifier.py)
- official ground-truth sweep:
  [`tests/run_official_problem_root_baseline.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/tests/run_official_problem_root_baseline.py)
- generated-domain sweep:
  [`tests/offline_method_generation/run_generated_problem_root_baseline.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/tests/offline_method_generation/run_generated_problem_root_baseline.py)

## Offline Build Artifacts

`build_domain_library(...)` now persists:

- `method_library.json`
- `method_synthesis_metadata.json`
- `domain_gate.json`
- `masked_domain.hddl`
- `generated_domain.hddl`

The persisted artifact records whether the source domain was:

- `official`
- `masked_official`
- `generated`

For the active masked-generation path, the source kind is `masked_official`.

## Prompting Strategy

The active domain-complete synthesis prompt follows a constrained symbolic
interface rather than free-form completion:

- compact domain-wide task contracts instead of a full combinatorial tree
- explicit output schema with one task entry per declared compound task
- domain-summary blocks limited to task signatures, relevant primitive actions,
  and reusable dynamic resources
- instructions that enforce query independence, symbolic discipline, and
  reusable decomposition structure

This keeps the prompt interpretable, domain-agnostic, and small enough to stay
inside a single request per domain.

## Official Ground-Truth Baseline

The official benchmark profile is pinned in code. The latest complete official
ground-truth summary is here:

- [`tests/generated/official_ground_truth_full/20260418_000457/summary.json`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/tests/generated/official_ground_truth_full/20260418_000457/summary.json)

## Quick Start

Create the project environment with `uv`:

```bash
uv venv
uv sync
```

Prepare `.env`:

```bash
cp .env.example .env
```

Minimum configuration:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
METHOD_SYNTHESIS_MODEL=minimax/minimax-m2.7
GOAL_GROUNDING_MODEL=deepseek/deepseek-chat-v3-0324
ONLINE_DOMAIN_SOURCE=benchmark
```

Run one masked offline domain build:

```bash
uv run python src/main.py \
  --build-domain-library \
  --domain-file ./src/domains/blocksworld/domain.hddl
```

Run one online Jason query:

```bash
uv run python src/main.py \
  "First put block b4 on block b2, then put block b1 on block b4." \
  --domain-file ./src/domains/blocksworld/domain.hddl \
  --problem-file ./src/domains/blocksworld/problems/p01.hddl
```

Run the official ground-truth sweep:

```bash
uv run python tests/run_official_problem_root_baseline.py
```

Run the generated-domain sweep:

```bash
uv run python tests/offline_method_generation/run_generated_problem_root_baseline.py
```

## Toolchains

The active planning path expects:

- `pandaPIparser`
- `pandaPIgrounder`
- `pandaPIengine`
- `mona`
- Java 17 to 23 for Jason runtime execution

Optional local toolchains used by the current backend race can live under
`.external/`. That directory is treated as local-only and is ignored by git.
