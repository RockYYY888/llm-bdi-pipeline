# LLM-BDI Pipeline

This repository contains the implementation for the dissertation's end-to-end
LLM-to-BDI plan-library study. It builds reusable BDI plan libraries from
masked HDDL domains and natural-language task instructions, then evaluates
those libraries through structural validation, grounding, Jason execution,
direct plan-generation baselines, and HTN planner reference runs across the
supported IPC 2020 benchmark domains.

The core generated-library flow is:

```text
D^- + L_s -> Phi_s -> M -> S
```

- `D^-`: masked official HDDL domain with methods removed
- `L_s`: stored domain-specific query sequence
- `Phi_s`: validated temporal specifications from `queries_LTLf.json`
- `M`: synthesized Hierarchical Task Network method library
- `S`: translated AgentSpeak(L) plan library

Around that flow, the repository also implements the benchmark query protocol,
language-model transport, generated-library validation, execution logging,
Jason runtime evaluation, direct final-plan generation baseline, lifted PANDA
SAT planner reference baseline, and IPC verifier integration used by the
dissertation evaluation.

## What Is Included

The repository includes the Python implementation, regression tests, supported
IPC 2020 benchmark domains and problem files, stored benchmark query records,
stored LTLf records, the Jason runtime source used by the evaluator, and the
`uv.lock` dependency lock file. The vendored Jason runtime is pruned to the
files needed for building and running the evaluator; upstream demos, examples,
site docs, and upstream test fixtures are not included.

The repository intentionally does not include local `.env` files, API keys,
generated run outputs under `artifacts/` or `tests/generated/`, or external
planner/verifier executables. Those outputs are reproducible from the commands
below, while the external executables must be installed on the local machine
for full planning and verification experiments.

## Repository Layout

```text
.
├── src/
│   ├── domain_model/
│   ├── temporal_specification/
│   ├── method_library/
│   ├── plan_library/
│   ├── evaluation/
│   ├── htn_evaluation/
│   ├── language_model/
│   ├── planning/
│   ├── execution_logging/
│   ├── verification/
│   ├── domains/
│   ├── benchmark_data/
│   └── utils/
└── tests/
    ├── temporal_specification/
    ├── plan_library/
    ├── evaluation/
    ├── method_library/
    ├── official_benchmark/
    ├── support/
    └── utils/
```

Generated run outputs are intentionally local-only. The repository does not
track `artifacts/`, `tests/generated/`, `tests/method_library/generated/`,
`tmp/`, local thesis material, or environment files.

## Setup

Prerequisites:

- Python 3.12
- `uv`

Install dependencies with `uv`:

```bash
uv sync
```

The offline regression suite does not require an API key or external planner
tools:

```bash
uv run pytest
uv run python src/main.py --help
```

Prepare API configuration only when running live language-model generation:

```bash
cp .env.example .env
```

Minimum live language-model configuration:

```bash
LANGUAGE_MODEL_API_KEY=...
LANGUAGE_MODEL_BASE_URL=https://api.deepseek.com
LANGUAGE_MODEL_MODEL=deepseek-v4-pro
```

All live model calls use the shared OpenAI-compatible JSON Chat Completion
transport in `src/language_model/openai_compatible.py`. Stage-specific
environment variables such as `METHOD_SYNTHESIS_MODEL` remain optional
overrides for experiments.

## Quick Start

Use these commands to check the repository from a fresh clone:

```bash
uv sync
uv run pytest
```

This validates the checked-in code and data without requiring generated
outputs, API access, or external planning tools. Tests that need unavailable
full toolchains are skipped automatically.

To run one live generation path, configure `.env` first, then run:

```bash
uv run python src/main.py generate-library \
  --domain-file ./src/domains/blocksworld/domain.hddl \
  --query-id query_1 \
  --output-root ./artifacts/plan_library/blocksworld
```

After that artifact exists, evaluate the same stored benchmark case:

```bash
uv run python src/main.py evaluate-library \
  --library-artifact ./artifacts/plan_library/blocksworld \
  --domain-file ./src/domains/blocksworld/domain.hddl \
  --query-id query_1
```

## Main Commands

Generate or refresh stored LTLf specifications:

```bash
uv run python src/main.py generate-ltlf-dataset \
  --source-query-dataset ./src/benchmark_data/benchmark_queries.json \
  --output-dataset ./src/benchmark_data/queries_LTLf.json
```

Generate a plan-library bundle:

```bash
uv run python src/main.py generate-library \
  --domain-file ./src/domains/blocksworld/domain.hddl
```

Evaluate a stored benchmark case after generating a library artifact:

```bash
uv run python src/main.py evaluate-library \
  --library-artifact ./artifacts/plan_library/blocksworld \
  --domain-file ./src/domains/blocksworld/domain.hddl \
  --query-id query_1
```

Evaluate an ad hoc instruction with an explicit formula:

```bash
uv run python src/main.py evaluate-library \
  --library-artifact ./artifacts/plan_library/blocksworld \
  --domain-file ./src/domains/blocksworld/domain.hddl \
  --problem-file ./src/domains/blocksworld/problems/p01.hddl \
  --instruction "Put block b4 on block b2." \
  --ltlf-formula "do_put_on(b4, b2)"
```

## Experiment Scripts

Longer experiments use standalone Python scripts:

- `tests/run_plan_library_evaluation_benchmark.py`
- `tests/run_official_problem_root_baseline.py`
- `tests/run_direct_plan_generation_baseline.py`
- `tests/run_direct_plan_generation_api_sweep.py`
- `tests/method_library/run_generated_domain_build_sweep.py`
- `tests/method_library/run_generated_problem_root_baseline.py`

Example:

```bash
uv run python tests/run_direct_plan_generation_api_sweep.py \
  --domain blocksworld \
  --query-id query_1 \
  --skip-verifier
```

## Toolchains

Unit tests and prompt generation run with only the Python dependencies above.
Full planning and verification experiments also need these runtime tools:

- `pandaPIparser`
- `pandaPIgrounder`
- `pandaPIengine`
- `mona`
- Java 23 for Jason runtime execution

Optional local toolchains can live under `.external/`, which is ignored by git.

### PATH Setup

Add the directories that contain the required binaries to `PATH`. Replace the
placeholder paths with your local install locations.

```bash
export JAVA_HOME="/path/to/jdk-23"
export PATH="$JAVA_HOME/bin:/path/to/pandaPIparser/bin:/path/to/pandaPIgrounder/bin:/path/to/pandaPIengine/bin:/path/to/mona/bin:$PATH"
```

For zsh, put these two lines in `~/.zshrc` and run `source ~/.zshrc`.

Verify the setup before running full experiments:

```bash
command -v pandaPIparser
command -v pandaPIgrounder
command -v pandaPIengine
command -v mona
java -version
```

Each `command -v` call should print a path, and `java -version` should report
Java 23. If anything is missing, update `PATH` and reload your shell.
