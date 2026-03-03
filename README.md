# LTLf to HTN-Specialised BDI Pipeline

This repository generates AgentSpeak plan libraries from natural-language goals.
The pipeline now uses **HTN method synthesis + decomposition + preferred specialisation**
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
     - Builds a deterministic baseline HTN method library from PDDL
     - Can optionally ask an LLM to replace or augment the library
     - Falls back to the deterministic library if the LLM output is invalid
   - **Stage 3B**: HTN decomposition
     - Expands each target literal into a decomposition trace
   - **Stage 3C**: Preferred specialisation
     - Prunes irrelevant leaves and retains the most abstract valid cut
   - **Stage 3D**: AgentSpeak rendering
     - Emits primitive action wrappers and specialised HTN goal plans

4. **Stage 4 Assets: Jason Validation**
   - `src/stage4_jason_validation/`
   - Contains Jason source and a local validation project
   - This is not part of the default Stage 1-3 generation path

## Important Design Choices

- Stage 3 is implemented entirely through HTN method synthesis, decomposition, and specialisation.
- The codebase contains a single Stage 3 path.
- Stage 3A is **LLM-backed but not LLM-dependent**:
  - If Stage 3 has an API client, it can synthesize an HTN library through a strict JSON schema.
  - If not, it uses the deterministic heuristic synthesizer.
- Stage 3 works from DFA transition literals and symbolic PDDL action schemas.
- The generated AgentSpeak is static, domain-specific, and specialised to the current goal set.
- The full end-to-end pipeline still requires an API key because Stage 1 is LLM-only.
- Stage 3 itself can still complete without Stage 3 LLM output because the HTN synthesizer has
  a deterministic fallback.

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
│   │   ├── htn_planner.py
│   │   ├── htn_specialiser.py
│   │   ├── htn_planner_generator.py
│   │   └── agentspeak_codegen.py
│   ├── stage4_jason_validation/
│   └── utils/
├── tests/
│   ├── stage1_interpretation/
│   ├── stage2_dfa_generation/
│   ├── stage3_code_generation/
│   │   └── test_stage3_htn.py
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
  - `htn_transitions.json`

## Running the Pipeline

The default entry point is:

```bash
python src/main.py "Put block a on block b"
```

Notes:

- Stage 1 requires an LLM API key.
- Stage 1 and Stage 3 both read the same `OPENAI_*` configuration.
- The Stage 3 synthesizer still falls back to deterministic HTN synthesis when no valid Stage 3
  LLM output is available.
- The maintained domain is `src/domains/blocksworld/`.
- Generated outputs are written to `logs/<timestamp>_dfa_agentspeak/`.

## Running Tests

Run the focused HTN Stage 3 tests:

```bash
./.venv/bin/pytest -q tests/stage3_code_generation/test_stage3_htn.py
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
- `htn_transitions.json`

The logger also records the HTN metadata inside the run log.

## Current Benchmarks

The active benchmark surface is:

- the blocksworld PDDL domain in `src/domains/blocksworld/`
- the Stage 2 formula regression cases in `tests/stage2_dfa_generation/test_ltlf2dfa.py`
- the Stage 3 HTN regression cases in `tests/stage3_code_generation/test_stage3_htn.py`
- the pipeline-level integration check in `tests/test_pipeline.py`
