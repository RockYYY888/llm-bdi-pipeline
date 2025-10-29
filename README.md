# LTL-BDI Dual-Branch Pipeline

Comparative Evaluation: LLM AgentSpeak Generation vs. FOND Planning

---

## Project Overview

A research pipeline comparing two approaches to BDI agent planning from temporal goals:
- **Branch A (llm_agentspeak)**: LLM directly generates AgentSpeak code from LTLf specifications (Baseline)
- **Branch B (fond)**: FOND planner generates policies from PDDL problem formulations (Research)

The pipeline converts natural language instructions to LTLf (Linear Temporal Logic on Finite Traces) specifications, then evaluates two distinct approaches for generating executable agent plans.

---

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install openai python-dotenv

# Install Docker (required for PR2 FOND planner)
# macOS: Install Docker Desktop from https://www.docker.com/products/docker-desktop
# Linux: sudo apt-get install docker.io
```

### Configuration

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

### Run Demo

```bash
# Run both branches (default) - comparison mode
python src/main.py "Stack block C on block B"

# Run LLM AgentSpeak baseline only
python src/main.py "Stack block C on block B" --mode llm_agentspeak

# Run FOND planning only
python src/main.py "Stack block C on block B" --mode fond
```

---

## System Architecture

```
Natural Language Input ("Stack block C on block B")
         |
         v
+------------------------------------------------+
|  STAGE 1: NL -> LTLf                           |
|  ltl_parser.py (LLM-based)                     |
|  Output: F(on(c,b))                            |
+------------------------+-----------------------+
                         |
       +-----------------+-----------------+
       |                                   |
       v                                   v
+---------------------------+  +---------------------------+
|  BRANCH A (Baseline)      |  |  BRANCH B (Research)      |
|  LLM AgentSpeak           |  |  FOND Planning            |
+---------------------------+  +---------------------------+
       |                                   |
       v                                   v
+---------------------------+  +---------------------------+
| STAGE 2A: LLM AgentSpeak  |  | STAGE 2B: FOND Planning   |
| Generation                |  |                           |
|                           |  | Step 1: LTLf -> PDDL      |
| Input: F(on(c,b))         |  | Step 2: PR2 FOND Planner  |
| Output:                   |  |                           |
|  AgentSpeak code (.asl)   |  | Input: F(on(c,b))         |
|  with plans and beliefs   |  | Output:                   |
|                           |  |  [pick-up-from-table(c),  |
|                           |  |   put-on-block(c,b)]      |
+---------------------------+  +---------------------------+
       |                                   |
       +----------------+------------------+
                        |
                        v
        +-------------------------------+
        |  STAGE 3: Execution &         |
        |  Comparison (if mode=both)    |
        |                               |
        |  - Goal satisfaction          |
        |  - Efficiency (action count)  |
        |  - Success rate               |
        |  - Execution traces           |
        +-------------------------------+
```

---

## Execution Modes

The pipeline supports three execution modes via the `--mode` flag:

1. **both** (default): Run both branches and compare results
2. **llm_agentspeak**: Run only LLM AgentSpeak baseline branch
3. **fond**: Run only FOND planning branch

---

## Key Features

### Branch A: LLM AgentSpeak Generation (Baseline)
- Complete AgentSpeak (.asl) program generation
- LLM directly generates BDI plans from LTLf goals
- Plan libraries with context-sensitive plans
- Declarative goal handling
- End-to-end LLM-based approach

### Branch B: FOND Planning (Research)
- Formal FOND (Fully Observable Non-Deterministic) planning
- LTLf goals converted to PDDL problem formulations
- PR2/PRP planner generates strong cyclic policies
- Handles non-deterministic action effects
- Classical planning approach

### Comparative Evaluation
- Goal satisfaction verification against LTLf specifications
- Efficiency comparison (action count)
- Success rate tracking
- Detailed execution traces
- JSON and text log outputs

---

## Example Usage

### Simple Goal: "Stack block C on block B"

```bash
$ python src/main.py "Stack block C on block B"

================================================================================
LTL-BDI PIPELINE - DUAL BRANCH COMPARISON
================================================================================

Natural Language Instruction: "Stack block C on block B"

--------------------------------------------------------------------------------
[STAGE 1] Natural Language -> LTLf Specification
--------------------------------------------------------------------------------
✓ LTLf Formula: ['F(on(c, b))']
  Objects: ['b', 'c']
  Initial State: [{'ontable': ['b']}, {'ontable': ['c']}, ...]

--------------------------------------------------------------------------------
[STAGE 2A] BRANCH A: LLM AgentSpeak Generation (Baseline)
--------------------------------------------------------------------------------
✓ AgentSpeak Code Generated

--------------------------------------------------------------------------------
[STAGE 2B] BRANCH B: FOND Planning (PR2)
--------------------------------------------------------------------------------
✓ PDDL Problem Generated
✓ FOND Plan Generated (2 actions)
  1. pick-up-from-table(c)
  2. put-on-block(c, b)

--------------------------------------------------------------------------------
[STAGE 3] Execution & Comparative Evaluation
--------------------------------------------------------------------------------
...
```

---

## Project Structure

```
.
├── src/
│   ├── main.py                          # Entry point with mode selection
│   ├── config.py                        # Configuration management
│   ├── dual_branch_pipeline.py          # Main pipeline orchestration
│   ├── pipeline_logger.py               # Logging utilities
│   ├── stage1_interpretation/
│   │   └── ltl_parser.py                # NL -> LTLf conversion (LLM)
│   ├── stage2_pddl/
│   │   └── pddl_problem_generator.py    # LTLf -> PDDL conversion
│   ├── stage3_codegen/
│   │   ├── agentspeak_generator.py      # LLM AgentSpeak generation
│   │   └── pr2_planner.py               # PR2 FOND planner wrapper
│   └── stage4_execution/
│       ├── blocksworld_simulator.py     # Environment simulation
│       ├── agentspeak_simulator.py      # AgentSpeak execution
│       └── comparative_evaluator.py     # Branch comparison
├── domains/
│   └── blocksworld/
│       └── domain.pddl                  # Blocksworld PDDL domain
├── pr2/                                 # PR2 FOND planner files
├── logs/                                # Execution logs (timestamped)
├── output/                              # Generated plans
└── tests/                               # Test suites
```

---

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Test Individual Components

```bash
# Test LTL parser
python src/stage1_interpretation/ltl_parser.py

# Test AgentSpeak generator
python src/stage3_codegen/agentspeak_generator.py

# Test PDDL problem generator
python src/stage2_pddl/pddl_problem_generator.py
```

---

## Pipeline Stages

### Stage 1: Natural Language -> LTLf
- **Input**: Natural language instruction (e.g., "Stack block C on block B")
- **Process**: LLM-based parser extracts objects, initial state, and goals
- **Output**: LTLf specification with formula(s), objects, initial state

### Stage 2A: LLM AgentSpeak Generation (Branch A)
- **Input**: LTLf specification
- **Process**: LLM generates complete AgentSpeak program
- **Output**: AgentSpeak code (.asl) with plans and beliefs

### Stage 2B: FOND Planning (Branch B)
- **Step 1**: LTLf -> PDDL problem conversion
- **Step 2**: PR2 FOND planner execution (Docker-based)
- **Output**: Action sequence policy

### Stage 3: Execution & Comparison
- Execute both branches in blocksworld simulator
- Verify LTLf goal satisfaction
- Compare efficiency and success rates
- Generate detailed logs

---

## Implementation Notes

### LTL-BDI Integration
The pipeline demonstrates two approaches to LLM integration in BDI systems:
1. **Direct Generation (Branch A)**: LLM generates full AgentSpeak programs
2. **Formal Planning (Branch B)**: Classical FOND planning from formal specifications

### FOND Planning
- Uses PR2 (Planner for Relevant Policies) via Docker
- Generates strong cyclic policies for non-deterministic domains
- Handles uncertainty in action outcomes
- Requires PDDL domain with `:non-deterministic` requirements

### Blocksworld Domain
The blocksworld domain provides a testbed for:
- Goal dependency analysis (stacking order)
- State space exploration (block configurations)
- Action precondition/effect reasoning

---

## Current Status

### Implemented Features
- ✓ Dual-branch architecture (LLM AgentSpeak vs FOND)
- ✓ LTLf goal specification from natural language
- ✓ PDDL problem generation from LTLf
- ✓ PR2 FOND planner integration (Docker)
- ✓ AgentSpeak code generation via LLM
- ✓ Blocksworld simulator
- ✓ Comparative evaluation framework
- ✓ JSON and text logging

### Known Limitations
1. **Single Domain**: Currently supports blocksworld only
2. **Action Name Mapping**: FOND uses hyphens, simulator expects underscores
3. **LTL Operators**: Currently supports F (eventually) operator primarily

---

## Future Work

### High Priority
1. **Extended Domain Support**
   - Mars Rover domain
   - Logistics domain
   - Additional IPC benchmark domains

2. **Runtime LTL Monitoring**
   - Dynamic goal satisfaction checking
   - Replanning on violations

3. **Jason Integration**
   - Full BDI platform integration
   - Multi-agent scenarios

### Medium Priority
4. **Performance Optimization**
   - Parallel branch execution
   - LLM response caching
   - Incremental plan generation

5. **Enhanced LTL Support**
   - G (always), U (until), X (next) operators
   - Complex temporal formulas

---

## Citation

If you use this work, please cite:

```bibtex
@software{ltl_bdi_pipeline,
  title={LTL-BDI Dual-Branch Pipeline: LLM AgentSpeak vs FOND Planning},
  author={Yiwei LI},
  year={2025},
  institution={University of Nottingham Ningbo China}
}
```

---

## License

This project is developed for academic research purposes.

---

## Acknowledgments

Research supervised by faculty at University of Nottingham Ningbo China.

Key references:
- Rao & Georgeff: BDI architecture foundations
- Bordini et al.: AgentSpeak semantics and implementation
- De Giacomo & Vardi: LTLf specifications
- Geffner & Bonet: FOND planning
