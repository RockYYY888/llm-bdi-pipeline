# LTLf to PANDA-Backed BDI Pipeline

This repository generates AgentSpeak plan libraries from natural-language goals.
The pipeline now uses **HTN method synthesis + PANDA planning**
for Stage 3.
The only actively maintained planning domain in this repository is **blocksworld**.

## Current Architecture

1. **Stage 1: Natural Language -> LTLf**
   - `src/stage1_interpretation/`
   - Uses an LLM to convert user instructions into `LTLSpecification`
   - Produces a `GroundingMap` that links propositional symbols back to domain predicates

2. **Stage 2: LTLf -> DFA**
   - `src/stage2_dfa_generation/`
   - Uses `ltlf2dfa`
   - Simplifies DFA labels into atomic literals

3. **Stage 3: DFA -> AgentSpeak**
   - `src/stage3_code_generation/`
   - **Stage 3A**: HTN method synthesis
     - Uses an LLM to synthesize the HTN method library for the current DFA targets
   - **Stage 3B**: PANDA planning
     - Exports a temporary HDDL planning problem and solves it with the PANDA PI toolchain
   - **Stage 3C**: AgentSpeak rendering
     - Emits primitive action wrappers and PANDA-backed HTN goal plans

4. **Stage 4 Assets: Jason Validation**
   - `src/stage4_jason_validation/`
   - Contains Jason source and a local validation project
   - This is not part of the default Stage 1-3 generation path

## Important Design Choices

- Stage 3 is implemented through HTN method synthesis and PANDA-backed planning.
- The codebase contains a single Stage 3 path.
- Stage 3A is **LLM-only** and rejects missing or malformed live model output.
- Stage 3 works from DFA transition literals, symbolic HDDL action schemas, and a PANDA-generated primitive plan.
- The generated AgentSpeak is static, domain-specific, and specialised to the current goal set.
- The full end-to-end pipeline still requires an API key because Stage 1 is LLM-only.
- The full Stage 3 path also requires the PANDA PI toolchain (`pandaPIparser`, `pandaPIgrounder`,
  `pandaPIengine`) to be available on `PATH`.

## Repository Layout

```text
.
├── src/
│   ├── domains/
│   ├── external/
│   ├── stage1_interpretation/
│   ├── stage2_dfa_generation/
│   ├── stage3_code_generation/
│   │   ├── htn_schema.py
│   │   ├── htn_method_synthesis.py
│   │   ├── panda_planner.py
│   │   ├── panda_planner_generator.py
│   │   └── agentspeak_codegen.py
│   ├── stage4_jason_validation/
│   └── utils/
├── tests/
│   ├── stage1_interpretation/
│   ├── stage2_dfa_generation/
│   ├── stage3_code_generation/
│   │   └── test_stage3_panda.py
│   └── test_pipeline.py
└── TO-DO-LIST.md
```

## Quick Start

From a fresh clone to a full pipeline run:

1. Clone the repository and enter it:

```bash
git clone https://github.com/RockYYY888/llm-bdi-pipeline.git
cd llm-bdi-pipeline
```

2. Create the Python environment and install dependencies:

```bash
uv venv
uv sync
```

3. Prepare your API configuration:

```bash
cp .env.example .env
```

Edit `.env` so it follows this format:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=

# Optional: Model selection
OPENAI_MODEL=deepseek-chat

# Optional: Base URL (for DeepSeek or other custom endpoints)
OPENAI_BASE_URL=https://api.deepseek.com

# Optional: API timeout (seconds)
OPENAI_TIMEOUT=120
```

4. Ensure MONA is available for `ltlf2dfa`:

The repository already includes a local MONA build under `src/external/mona-1.4/`.
If it is missing or needs to be rebuilt, run:

```bash
cd src/external/mona-1.4
./configure --prefix=$(pwd)/mona-install --disable-shared --enable-static
make
make install-strip
cd ../../..
```

5. Run the canonical acceptance test:

```bash
./.venv/bin/pytest -q tests/test_pipeline.py
```

This test now also requires the PANDA PI toolchain to be installed and on `PATH`.

6. Run the full pipeline on a real blocksworld instruction:

```bash
./.venv/bin/python src/main.py "Using blocks a and b, arrange them so that a is on b."
```

7. Inspect the generated artifacts:

- The pipeline writes a timestamped directory under `logs/`
- Each successful run contains:
  - `execution.json`
  - `execution.txt`
  - `grounding_map.json`
  - `dfa_original.dot`
  - `dfa_simplified.dot`
  - `dfa.json`
  - `agentspeak_generated.asl`
  - `htn_method_library.json`
  - `panda_transitions.json`

## Running the Pipeline

The default entry point is:

```bash
python src/main.py "Put block a on block b"
```

Notes:

- Stage 1 requires an LLM API key.
- Stage 1 and Stage 3 both read the same `OPENAI_*` configuration.
- Stage 3 also requires `pandaPIparser`, `pandaPIgrounder`, and `pandaPIengine` on `PATH`.
- The maintained domain is `src/domains/blocksworld/`.
- Generated outputs are written to `logs/<timestamp>_dfa_agentspeak/`.

## Running Tests

Run the focused Stage 3 tests:

```bash
./.venv/bin/pytest -q tests/stage3_code_generation/test_stage3_panda.py
./.venv/bin/pytest -q tests/test_pipeline.py
```

The canonical example acceptance test is:

```bash
./.venv/bin/pytest -q tests/test_pipeline.py
```

It runs a fixed blocksworld example through `pipeline.execute()` and checks that the logger
captured each stage's live inputs, outputs, and persisted artifacts. The test uses the real
Stage 1/2/3 execution path (no stubbed Stage 1 response) and writes its run records under
`tests/logs/`.

Run the Stage 2 DFA tests:

```bash
./.venv/bin/pytest -q tests/stage2_dfa_generation/test_ltlf2dfa.py
```

The pipeline-level integration test and the Stage 1 tests depend on external LLM access, so they
are not part of the default fast regression loop in environments without API access.

## Stage 3 Outputs

A successful Stage 3 run writes:

- `agentspeak_generated.asl`
- `htn_method_library.json`
- `panda_transitions.json`

The logger also records the PANDA planning metadata inside the run log.

## Current Benchmarks

The active benchmark surface is:

- the blocksworld HDDL domain in `src/domains/blocksworld/`
- the Stage 2 formula regression cases in `tests/stage2_dfa_generation/test_ltlf2dfa.py`
- the Stage 3 PANDA regression cases in `tests/stage3_code_generation/test_stage3_panda.py`
- the pipeline-level integration check in `tests/test_pipeline.py`
