# LTLf to HTN-Specialised BDI Pipeline

This repository generates AgentSpeak plan libraries from natural-language goals.
The pipeline now uses **HTN method synthesis + decomposition + preferred specialisation**
for Stage 3. The previous backward-planning implementation has been removed.

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

## Important Design Choices

- Stage 3 no longer performs backward state-space search.
- There is no backward-planning fallback path in the codebase.
- Stage 3A is **LLM-backed but not LLM-dependent**:
  - If Stage 3 has an API client, it can synthesize an HTN library through a strict JSON schema.
  - If not, it uses the deterministic heuristic synthesizer.
- Stage 3 works from DFA transition literals and symbolic PDDL action schemas.
- The generated AgentSpeak is static, domain-specific, and specialised to the current goal set.

## Repository Layout

```text
.
├── src/
│   ├── domains/
│   ├── stage1_interpretation/
│   ├── stage2_dfa_generation/
│   ├── stage3_code_generation/
│   │   ├── htn_schema.py
│   │   ├── htn_method_synthesis.py
│   │   ├── htn_planner.py
│   │   ├── htn_specialiser.py
│   │   ├── htn_planner_generator.py
│   │   └── agentspeak_codegen.py
│   └── utils/
├── tests/
│   ├── stage2_dfa_generation/
│   ├── stage3_code_generation/
│   │   └── test_stage3_htn.py
│   └── test_pipeline_stage3_htn.py
└── TO-DO-LIST.md
```

## Running the Pipeline

The default entry point is:

```bash
python src/main.py "Put block a on block b"
```

Notes:

- Stage 1 still requires an LLM API key.
- Stage 3 can run without a Stage 3 API key because it has a deterministic synthesizer.
- Generated outputs are written to `logs/<timestamp>_dfa_agentspeak/`.

## Running Tests

Run the focused HTN Stage 3 tests:

```bash
./.venv/bin/pytest -q tests/stage3_code_generation/test_stage3_htn.py
./.venv/bin/pytest -q tests/test_pipeline_stage3_htn.py
```

Run the Stage 2 DFA tests:

```bash
./.venv/bin/pytest -q tests/stage2_dfa_generation/test_ltlf2dfa.py
```

## Stage 3 Outputs

A successful Stage 3 run writes:

- `agentspeak_generated.asl`
- `htn_method_library.json`
- `htn_transitions.json`

The logger also records the HTN metadata inside the run log.

## What Was Removed

The refactor intentionally removed:

- backward state-space planning
- lifted mutex extraction used only by backward search
- pruning diagnostics for the old planner
- the old Stage 3 integration suite tied to backward planning

If you see references to the old planner in archived notes, treat them as historical context only.
