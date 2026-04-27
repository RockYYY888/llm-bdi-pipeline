# LLM-BDI Pipeline

This repository implements the Chapter 4 plan-library pipeline:

```text
D^- + L_s -> Phi_s -> M -> S
```

- `D^-`: masked official HDDL domain with methods removed
- `L_s`: stored domain-specific query sequence
- `Phi_s`: validated temporal specifications from `queries_LTLf.json`
- `M`: synthesized Hierarchical Task Network method library
- `S`: translated AgentSpeak(L) plan library

Grounding, Jason execution, planner runs, and verifier checks are evaluation
evidence built on top of `S`.

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

Install dependencies with `uv`:

```bash
uv sync
```

Prepare API configuration:

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

Evaluate a stored benchmark case:

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
