# LTLf-HTN-PANDA-AgentSpeak-Jason Pipeline

This repository generates AgentSpeak plan libraries from natural-language goals.
The default runtime mainline is **Stage 1 -> Stage 6** in
`src/main.py` -> `LTL_BDI_Pipeline.execute(mode="dfa_agentspeak")`.
The only actively maintained planning domain in this repository is **blocksworld**.

## Current Architecture

1. **Stage 1: Natural Language -> LTLf**
   - `src/stage1_interpretation/`
   - Input: natural-language instruction + HDDL domain signatures
   - Does: uses an LLM to convert the instruction into an `LTLSpecification`
   - Output: `LTLSpecification` + `GroundingMap` that links propositional symbols back to domain predicates

2. **Stage 2: LTLf -> DFA**
   - `src/stage2_dfa_generation/`
   - Input: `LTLSpecification`
   - Does: uses `ltlf2dfa` and simplifies DFA labels into atomic literals
   - Output: simplified DFA metadata + DOT artifacts

3. **Stage 3: DFA -> HTN Method Synthesis**
   - `src/stage3_method_synthesis/`
   - Input: simplified DFA + `GroundingMap` + HDDL domain signatures
   - Does: uses an LLM to synthesize the HTN method library for the current DFA targets
   - Output: `HTNMethodLibrary`

4. **Stage 4: HTN Method Library -> PANDA Validation**
   - `src/stage4_panda_planning/`
   - Input: `HTNMethodLibrary` + Stage 3 transition specs + concrete object set
   - Does: builds temporary HDDL domain/problem files and uses PANDA PI to produce
     existence witnesses for (a) transition-bound tasks and (b) query-relevant sibling methods
   - Output: PANDA validation records (`PANDAPlanResult`) + temporary HDDL / plan artifacts

5. **Stage 5: HTN Methods + DFA Wrappers -> AgentSpeak Rendering**
   - `src/stage5_agentspeak_rendering/`
   - Input: `HTNMethodLibrary` + validated transition records + HDDL domain
   - Does: emits primitive action wrappers, the full HTN method library as AgentSpeak plans,
     and DFA state-aware transition wrappers
   - Output: `agentspeak_generated.asl`

6. **Stage 6: AgentSpeak -> Jason Runtime Validation**
   - `src/stage6_jason_validation/`
   - Input: Stage 5 `agentspeak_generated.asl` + Stage 3 target literals
   - Does: appends a Stage 6 execution wrapper, launches Jason via
     `jason.infra.local.RunLocalMAS`, and validates runtime markers
   - Output: `jason_runner_agent.asl`, `jason_runner.mas2j`, runtime stdout/stderr,
     and `jason_validation.json`

## Important Design Choices

- Stage 3 is **LLM-only** and rejects missing or malformed live model output.
- Stage 4 uses the PANDA PI toolchain on temporary HDDL domain/problem files.
- Stage 4 keeps the problem-instance builder separate from the planner entrypoint. The default
  builder is explicit and configurable instead of hard-coding initial-state facts inside the planner.
- Stage 4 is an existence-witness validation step for the current query structure, not a global
  completeness proof over the entire domain state space.
- Stage 5 renders static, domain-specific AgentSpeak from the HTN method library, while Stage 4
  PANDA outputs remain validation artifacts plus DFA-edge witnesses.
- Stage 6 is enabled by default and is a hard gate: if Jason validation fails, the full pipeline
  run fails.
- The generated AgentSpeak is static, domain-specific, and specialised to the current goal set.
- The full end-to-end pipeline requires an API key because Stage 1 and Stage 3 are LLM-backed.
- The full Stage 4 path requires the PANDA PI toolchain (`pandaPIparser`, `pandaPIgrounder`,
  `pandaPIengine`). The runtime auto-detects a local install under `$HOME/.local/pandaPI/bin`,
  or you can expose it through `PATH`, `PANDA_PI_HOME`, or `PANDA_PI_BIN`.
- Stage 6 requires Java 17-23 and a Jason CLI build from
  `src/stage6_jason_validation/jason_src`.

## Assumptions and Validation Boundary

The Stage 1-6 pipeline is general under an explicit set of runtime, HDDL
subset, typing, and semantic assumptions. The full, code-faithful assumption
list is maintained in:

- `PIPELINE_ASSUMPTIONS.md`

Read this before claiming domain-level generality or interpreting validation
results as global proofs.

## Repository Layout

```text
.
├── src/
│   ├── domains/
│   ├── external/
│   ├── stage1_interpretation/
│   ├── stage2_dfa_generation/
│   ├── stage3_method_synthesis/
│   │   ├── htn_method_synthesis.py
│   │   ├── htn_prompts.py
│   │   └── htn_schema.py
│   ├── stage4_panda_planning/
│   │   ├── panda_planner.py
│   │   ├── problem_builder.py
│   │   └── panda_schema.py
│   ├── stage5_agentspeak_rendering/
│   │   └── agentspeak_renderer.py
│   ├── stage6_jason_validation/
│   └── utils/
├── tests/
│   ├── stage1_interpretation/
│   ├── stage2_dfa_generation/
│   ├── stage3_method_synthesis/
│   │   └── test_stage3_method_synthesis.py
│   ├── stage4_panda_planning/
│   │   ├── test_panda_planner.py
│   │   └── test_stage4_panda_planning.py
│   ├── stage5_agentspeak_rendering/
│   │   └── test_agentspeak_renderer.py
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

5. Install the PANDA PI toolchain:

The repository does not ship a bundled PANDA binary. The setup below is the
tested path used for this project on macOS.

```bash
brew install gengetopt bison flex pkgconf

git clone https://github.com/panda-planner-dev/pandaPIparser.git ~/.local/src/pandaPIparser
git clone https://github.com/panda-planner-dev/pandaPIgrounder.git ~/.local/src/pandaPIgrounder
git clone https://github.com/panda-planner-dev/pandaPIengine.git ~/.local/src/pandaPIengine

# parser
cd ~/.local/src/pandaPIparser
make -j4 BISON=/opt/homebrew/opt/bison/bin/bison FLEX=/opt/homebrew/opt/flex/bin/flex

# grounder
cd ~/.local/src/pandaPIgrounder
git submodule update --init --recursive
git -C cpddl/third-party/boruvka apply 0001-Removed-non-macos-call-in-unused-function.patch || true
git -C cpddl/third-party/boruvka apply 0001-boruvka-endian.patch || true
make boruvka opts bliss lpsolve
make
cd src
make -j4 CXX=g++ CC=gcc

# engine
cd ~/.local/src/pandaPIengine
mkdir -p build
cd build
cmake ../src
make -j4

# install binaries to the default auto-detected location
install -d ~/.local/pandaPI/bin
install ~/.local/src/pandaPIparser/pandaPIparser ~/.local/pandaPI/bin/pandaPIparser
install ~/.local/src/pandaPIgrounder/pandaPIgrounder ~/.local/pandaPI/bin/pandaPIgrounder
install ~/.local/src/pandaPIengine/build/pandaPIengine ~/.local/pandaPI/bin/pandaPIengine
```

Optional shell setup:

```bash
export PATH="$HOME/.local/pandaPI/bin:$PATH"
```

If you do not add that `PATH` export, the Stage 4 runtime still auto-discovers
the default install directory above.

6. Ensure Java 17-23 is available and build Jason CLI:

```bash
# Example (macOS Corretto 23)
export STAGE6_JAVA_HOME=/Users/lyw/Library/Java/JavaVirtualMachines/corretto-23.0.2/Contents/Home
export JAVA_HOME="$STAGE6_JAVA_HOME"

cd src/stage6_jason_validation/jason_src
./gradlew config
cd ../../..
```

7. Run the canonical development acceptance test (single query):

```bash
./.venv/bin/python tests/test_pipeline.py query_1
```

This test requires:
- a valid `OPENAI_API_KEY` in `.env`
- a working PANDA PI toolchain
- Stage 6 Java 17-23 + Jason runtime toolchain
- the default local PANDA install path (`$HOME/.local/pandaPI/bin`) or an equivalent
  `PATH` / `PANDA_PI_*` configuration

8. Run the full pipeline acceptance sweep only at final validation:

```bash
PIPELINE_TEST_ALL=1 ./.venv/bin/pytest -q tests/test_pipeline.py
```

9. Run the full pipeline on a real blocksworld instruction:

```bash
./.venv/bin/python src/main.py \
  "Using blocks a and b, arrange them so that a is on b." \
  --domain-file ./src/domains/blocksworld/domain.hddl
```

10. Inspect the generated artifacts:

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
  - `jason_runner_agent.asl`
  - `jason_runner.mas2j`
  - `jason_stdout.txt`
  - `jason_stderr.txt`
  - `jason_validation.json`

## Running the Pipeline

The default entry point is:

```bash
python src/main.py \
  "Put block a on block b" \
  --domain-file ./src/domains/blocksworld/domain.hddl
```

Notes:

- Stage 1 requires an LLM API key.
- Stage 1 and Stage 3 both read the same `OPENAI_*` configuration.
- `--domain-file` is required. The pipeline does not use an implicit default domain.
- Stage 4 looks for `pandaPIparser`, `pandaPIgrounder`, and `pandaPIengine` in this order:
  `PATH`, `PANDA_PI_HOME/bin`, `PANDA_PI_BIN`, `$HOME/.local/pandaPI/bin`.
- Stage 6 looks for a supported Java runtime (17-23) in this order:
  `STAGE6_JAVA_BIN`, `STAGE6_JAVA_HOME/bin/java`, `JAVA_HOME/bin/java`, `PATH`, and
  macOS JVM directories.
- The maintained domain is `src/domains/blocksworld/`.
- Generated outputs are written to `logs/<timestamp>_dfa_agentspeak/`.

## Running Tests

Run the focused stage tests:

```bash
./.venv/bin/pytest -q tests/stage3_method_synthesis/test_stage3_method_synthesis.py
./.venv/bin/pytest -q tests/stage4_panda_planning/test_panda_planner.py
./.venv/bin/pytest -q tests/stage4_panda_planning/test_stage4_panda_planning.py
./.venv/bin/pytest -q tests/stage5_agentspeak_rendering/test_agentspeak_renderer.py
./.venv/bin/pytest -q tests/stage6_jason_validation/test_jason_runner.py
./.venv/bin/pytest -q tests/stage6_jason_validation/test_stage6_integration.py
```

The canonical development acceptance check is a single live query:

```bash
./.venv/bin/python tests/test_pipeline.py query_1
```

For final full acceptance, run the full live sweep explicitly:

```bash
PIPELINE_TEST_ALL=1 ./.venv/bin/pytest -q tests/test_pipeline.py
```

`tests/test_pipeline.py` defaults to one query in pytest mode and only runs all queries when
`PIPELINE_TEST_ALL=1` is provided. You can also select a specific query with
`PIPELINE_TEST_QUERY=query_3`.

Run the Stage 2 DFA tests:

```bash
./.venv/bin/pytest -q tests/stage2_dfa_generation/test_ltlf2dfa.py
```

The pipeline-level integration test and the Stage 1 tests depend on external LLM access, so they
are not part of the default fast regression loop in environments without API access.

## Pipeline Outputs

A successful Stage 1-6 run writes:

- `execution.json`
- `execution.txt`
- `grounding_map.json`
- `dfa_original.dot`
- `dfa_simplified.dot`
- `dfa.json`
- `agentspeak_generated.asl`
- `htn_method_library.json`
- `panda_transitions.json`
- `jason_runner_agent.asl`
- `jason_runner.mas2j`
- `jason_stdout.txt`
- `jason_stderr.txt`
- `jason_validation.json`

The logger also records the Stage 3 synthesis metadata, Stage 4 PANDA metadata,
and Stage 5 rendering metadata inside the run log.

## Current Benchmarks

The active benchmark surface is:

- the blocksworld HDDL domain in `src/domains/blocksworld/`
- the Stage 2 formula regression cases in `tests/stage2_dfa_generation/test_ltlf2dfa.py`
- the Stage 3 synthesis regression cases in `tests/stage3_method_synthesis/test_stage3_method_synthesis.py`
- the Stage 4 PANDA regression cases in `tests/stage4_panda_planning/test_stage4_panda_planning.py`
- the Stage 5 rendering regression cases in `tests/stage5_agentspeak_rendering/test_agentspeak_renderer.py`
- the pipeline-level integration check in `tests/test_pipeline.py`
