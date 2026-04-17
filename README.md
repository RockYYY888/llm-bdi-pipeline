# Domain-Complete Hierarchical Task Network Pipeline

This repository now exposes one semantic mainline with two pipelines:

- offline domain build:
  `domain.hddl -> method synthesis -> domain gate -> cached domain library`
- online query execution:
  `natural language query + problem.hddl + cached domain library -> goal grounding -> plan solve -> official verification`

The first acceptance target is the official ground-truth baseline:

- one domain preflight per benchmark domain
- one official problem-root execution per benchmark problem
- success decided only by the official hierarchical verifier

The active benchmark domains are:

- `blocksworld`
- `marsrover`
- `satellite`
- `transport`

## Repository Layout

```text
.
├── src/
│   ├── domain_build/
│   │   ├── method_synthesis/
│   │   └── domain_gate/
│   ├── query_execution/
│   │   └── goal_grounding/
│   ├── planning/
│   ├── pipeline/
│   ├── verification/
│   ├── domains/
│   ├── benchmark_data/
│   └── utils/
├── tests/
│   ├── pipeline/
│   ├── support/
│   └── utils/
└── TO-DO-LIST.md
```

## Main Entry Points

- command-line entry point:
  [`src/main.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/main.py)
- orchestrator:
  [`src/pipeline/domain_complete_pipeline.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/pipeline/domain_complete_pipeline.py)
- official verifier wrapper:
  [`src/verification/official_plan_verifier.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/verification/official_plan_verifier.py)
- official baseline harness:
  [`tests/run_official_problem_root_baseline.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/tests/run_official_problem_root_baseline.py)

## Planning Design

The planning subsystem is organized in three semantic layers:

1. problem structure analysis
   - classify whether the official instance is total-order or partial-order
2. representation construction
   - keep the original representation
   - build one linearized representation when the instance is partial-order
3. backend race with official verification
   - total-order instances run original-representation backends
   - partial-order instances run both original and linearized backends
   - the first backend that reaches official hierarchical verification success wins

The currently active backends are:

- `PandaDealer`
- `PANDA` solver portfolio
- `Lifted-PANDA` on the linearized representation

## Ground-Truth Baseline Status

The current semantic sweep result is:

- `4/4` official domain preflights passed
- `115/115` official problem-root runs completed
- `113/115` verified successes
- `2/115` solver no-plan failures
- `0` primitive-invalid failures
- `0` hierarchical-rejection failures

The latest summary is in:

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

The minimum configuration is:

```bash
OPENAI_API_KEY=...
GOAL_GROUNDING_MODEL=deepseek/deepseek-chat-v3-0324
METHOD_SYNTHESIS_MODEL=minimax/minimax-m2
```

Run an offline domain build:

```bash
uv run python src/main.py \
  --build-domain-library \
  --domain-file ./src/domains/blocksworld/domain.hddl
```

Run an online query against a cached library:

```bash
uv run python src/main.py \
  "Stack block C on block B" \
  --domain-file ./src/domains/blocksworld/domain.hddl \
  --problem-file ./src/domains/blocksworld/problems/p01.hddl \
  --library-artifact ./artifacts/domain_builds/blocksworld
```

Run the official ground-truth sweep:

```bash
uv run python tests/run_official_problem_root_baseline.py
```

## Toolchains

The active planning path expects:

- `pandaPIparser`
- `pandaPIgrounder`
- `pandaPIengine`

Optional local toolchains used by the current backend race can live under
`.external/`. That directory is treated as local-only and is ignored by git.
