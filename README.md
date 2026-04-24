# Chapter 4 Plan-Library Pipeline

This repository now treats the dissertation methodology as the primary implementation contract:

`D^- + L_s -> Φ_s -> M -> S`

- `D^-`: masked official HDDL domain with methods removed
- `L_s`: stored domain-specific query sequence
- `Φ_s`: validated temporal specifications from `queries_LTLf.json`
- `M`: synthesized HTN method library
- `S`: translated AgentSpeak(L) plan library

Grounding, Jason execution, planner runs, and verifier checks are evaluation evidence built on top of `S`. They are no longer the primary generation architecture.

## Repository Layout

```text
.
├── src/
│   ├── domain_model/
│   ├── temporal_specification/
│   ├── method_library/
│   ├── plan_library/
│   ├── evaluation/
│   ├── compat/
│   ├── planning/
│   ├── execution_logging/
│   ├── verification/
│   ├── domains/
│   ├── benchmark_data/
│   └── utils/
├── tests/
│   ├── temporal_specification/
│   ├── plan_library/
│   ├── evaluation/
│   ├── support/
│   └── utils/
└── TO-DO-LIST.md
```

## Main Entry Points

- command-line entry point:
  [`src/main.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/main.py)
- Chapter 4 generation pipeline:
  [`src/plan_library/pipeline.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/plan_library/pipeline.py)
- evaluation pipeline:
  [`src/evaluation/pipeline.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/evaluation/pipeline.py)
- compatibility helpers:
  [`src/compat/__init__.py`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/compat/__init__.py)

## Persisted Generation Artifacts

`generate-library` persists one paper-aligned bundle:

- `artifact_metadata.json`
- `masked_domain.hddl`
- `query_sequence.json`
- `temporal_specifications.json`
- `method_library.json`
- `plan_library.json`
- `plan_library.asl`
- `translation_coverage.json`
- `library_validation.json`
- `method_synthesis_metadata.json`

`generated_domain.hddl` is no longer a core generation artifact. It is only materialized inside evaluation flows when a legacy planner path requires an HDDL adapter.

## Command-Line Interface

Create the project environment with `uv`:

```bash
uv venv
uv sync
```

Prepare `.env`:

```bash
cp .env.example .env
```

Minimum configuration for optional LTLf generation and method synthesis:

```bash
LTLF_GENERATION_API_KEY=...
LTLF_GENERATION_BASE_URL=https://openrouter.ai/api/v1
LTLF_GENERATION_MODEL=moonshotai/kimi-k2.6
LTLF_GENERATION_SESSION_ID=ltlf-generation
METHOD_SYNTHESIS_API_KEY=...
METHOD_SYNTHESIS_BASE_URL=https://openrouter.ai/api/v1
METHOD_SYNTHESIS_MODEL=moonshotai/kimi-k2.6
METHOD_SYNTHESIS_SESSION_ID=method-synthesis
```

Generate or refresh the stored LTLf dataset only when `queries_LTLf.json` is absent or
needs regeneration:

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

Evaluate a stored benchmark case from `queries_LTLf.json`:

```bash
uv run python src/main.py evaluate-library \
  --library-artifact ./artifacts/plan_library/blocksworld \
  --domain-file ./src/domains/blocksworld/domain.hddl \
  --query-id query_1
```

Evaluate an ad hoc instruction with an explicit LTLf formula:

```bash
uv run python src/main.py evaluate-library \
  --library-artifact ./artifacts/plan_library/blocksworld \
  --domain-file ./src/domains/blocksworld/domain.hddl \
  --problem-file ./src/domains/blocksworld/problems/p01.hddl \
  --instruction "Put block b4 on block b2." \
  --ltlf-formula "do_put_on(b4, b2)"
```

## Query Dataset

The default stored temporal-specification dataset is:

- [`src/benchmark_data/queries_LTLf.json`](/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/benchmark_data/queries_LTLf.json)

Generation uses this dataset by default, filtered to the selected domain. Stored benchmark evaluation also uses it directly, rather than rerunning live grounding. The `generate-ltlf-dataset` command is an explicit preparation step, not an implicit part of `generate-library`.

## Toolchains

The evaluation path expects:

- `pandaPIparser`
- `pandaPIgrounder`
- `pandaPIengine`
- `mona`
- Java 17 to 23 for Jason runtime execution

Optional local toolchains can live under `.external/`. That directory is treated as local-only and ignored by git.
