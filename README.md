# LTL-BDI Pipeline: LLM AgentSpeak Generation

Research pipeline for generating BDI agent code from natural language using LLMs and Linear Temporal Logic (LTLf).

---

## Project Overview

This pipeline converts natural language instructions into executable AgentSpeak code through a two-stage process:

1. **Stage 1**: Natural Language → LTLf (Linear Temporal Logic on Finite Traces) Specification
2. **Stage 2**: LTLf → AgentSpeak Code Generation (via LLM)

**Note**: This project previously supported a dual-branch comparison with FOND planning (Branch B). That functionality has been moved to `src/legacy/fond/` to focus development on LLM-based AgentSpeak generation.

---

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install openai python-dotenv
```

### Configuration

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=deepseek-chat  # or gpt-4o-mini
```

### Run Demo

```bash
# Run the pipeline
python src/main.py "Stack block C on block B"
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
+------------------------------------------------+
         |
         v
+------------------------------------------------+
|  STAGE 2: LTLf -> AgentSpeak Code              |
|  agentspeak_generator.py (LLM-based)           |
|                                                |
|  Input: F(on(c,b))                            |
|  Output: Complete AgentSpeak (.asl) program   |
|          with plans and beliefs                |
+------------------------------------------------+
```

---

## Key Features

### LLM-Based AgentSpeak Generation
- Complete AgentSpeak (.asl) program generation from LTLf goals
- Automatic plan library creation with context-sensitive plans
- Declarative goal handling using LTL temporal operators
- Support for F (eventually), G (always), U (until), X (next) operators
- Domain-aware code generation with action preconditions

### Comprehensive Logging
- Timestamped execution logs in JSON and human-readable formats
- Full LLM prompts and responses recorded for reproducibility
- Separate log directories for each execution
- Complete trace of NL → LTLf → AgentSpeak transformation

---

## Example Usage

```bash
$ python src/main.py "Stack block C on block B"

================================================================================
LTL-BDI PIPELINE - LLM AGENTSPEAK GENERATION
================================================================================

Natural Language Instruction: "Stack block C on block B"
Output directory: logs/20251030_123456_llm_agentspeak

--------------------------------------------------------------------------------
[STAGE 1] Natural Language -> LTLf Specification
--------------------------------------------------------------------------------
✓ LTLf Formula: ['F(on(c, b))']
  Objects: ['b', 'c']
  Initial State: [{'ontable': ['b']}, {'ontable': ['c']}, ...]

--------------------------------------------------------------------------------
[STAGE 2] LLM AgentSpeak Generation
--------------------------------------------------------------------------------
✓ AgentSpeak Code Generated
  First few lines:
    // Main goal from LTLf F formula: F(on(c, b))
    +!achieve_on_c_b : true <-
        .print("Starting to achieve on(c,b)");
        !![on(c,b)].

  Saved to: logs/20251030_123456_llm_agentspeak/agentspeak_generated.asl

================================================================================
STAGES 1-2 COMPLETED SUCCESSFULLY
================================================================================

Execution log saved to: logs/20251030_123456_llm_agentspeak/execution.json
```

---

## Project Structure

```
.
├── src/
│   ├── main.py                          # Entry point
│   ├── config.py                        # Configuration management
│   ├── ltl_bdi_pipeline.py              # Main pipeline orchestration
│   ├── dual_branch_pipeline.py          # DEPRECATED: Backward compatibility wrapper
│   ├── pipeline_logger.py               # Logging utilities
│   ├── stage1_interpretation/
│   │   └── ltl_parser.py                # Stage 1: NL -> LTLf conversion (LLM)
│   ├── stage2_planning/
│   │   └── agentspeak_generator.py      # Stage 2: LTLf -> AgentSpeak (LLM)
│   └── legacy/
│       ├── fond/                        # Legacy FOND planning (Branch B)
│       │   ├── README.md                # Instructions for restoring FOND functionality
│       │   ├── stage2_planning/         # PDDL and PR2 planner
│       │   ├── stage2_translation/      # LTLf to PDDL conversion
│       │   ├── domains/blocksworld/     # PDDL domain files
│       │   └── external/pr2/            # PR2 FOND planner (Docker)
│       └── stage4_execution/            # Future: Execution & evaluation
├── domains/
│   └── blocksworld/                     # (empty - PDDL domain moved to legacy)
├── logs/                                # Execution logs (timestamped JSON + TXT)
└── tests/                               # Test suites
    └── test_complex_cases.py            # Complex FOND benchmark tests
```

---

## Development

### Running Tests

Run the comprehensive test suite with 3 complex scenarios based on FOND benchmarks:

```bash
# Run all complex test cases (bw_5_1, bw_5_3, bw_5_5)
python tests/test_complex_cases.py

# Run with output logging
python tests/test_complex_cases.py 2>&1 | tee tests/test_results.log
```

### Test Individual Components

```bash
# Test LTL parser
python src/stage1_interpretation/ltl_parser.py

# Test AgentSpeak generator
python src/stage2_planning/agentspeak_generator.py
```

---

## Pipeline Stages

### Stage 1: Natural Language → LTLf
- **Input**: Natural language instruction (e.g., "Stack block C on block B")
- **Process**: LLM-based parser extracts objects, initial state, and temporal goals
- **Output**: LTLf specification with formula(s), objects, initial state
- **LLM Model**: Configured via `OPENAI_MODEL` in `.env` (e.g., `deepseek-chat`, `gpt-4o-mini`)

### Stage 2: LTLf → AgentSpeak Code
- **Input**: LTLf specification from Stage 1
- **Process**: LLM generates complete AgentSpeak program with:
  - Initial beliefs (from initial state)
  - Goal-achieving plans (from LTLf formulas)
  - Context-sensitive plan alternatives
  - Belief updates and action sequences
- **Output**: AgentSpeak code (.asl) ready for Jason/JAdex execution
- **LLM Model**: Same as Stage 1

---

## Implementation Notes

### LTL-BDI Integration
The pipeline demonstrates LLM-based generation of BDI agent code from declarative temporal goals:
- **Declarative Goals**: LTLf formulas specify *what* should be achieved, not *how*
- **Plan Generation**: LLM translates temporal goals into procedural AgentSpeak plans
- **Domain Knowledge**: LLM incorporates domain actions and predicates into generated code

### Blocksworld Domain
The blocksworld domain provides a testbed for:
- Goal dependency analysis (stacking order)
- State space exploration (block configurations)
- Action precondition/effect reasoning

**Domain Actions**: pickup, putdown, stack, unstack
**Domain Predicates**: on(X, Y), ontable(X), clear(X), holding(X), handempty

---

## Current Status

### Implemented Features (Stages 1-2)
- ✓ **Stage 1**: Natural language → LTLf specification (LLM-based)
- ✓ **Stage 2**: LTLf → LLM AgentSpeak code generation
- ✓ Blocksworld domain support
- ✓ Comprehensive JSON + text execution logs with full LLM prompts
- ✓ Support for F, G, U, X temporal operators and nested formulas
- ✓ AgentSpeak code output (.asl files)

### Not Yet Implemented
- ⏳ **Stage 3**: Execution & Comparative Evaluation
  - Execution of AgentSpeak code in Jason/JAdex
  - Goal satisfaction verification
  - Performance metrics
  - (Code exists in `legacy/stage4_execution/` for future development)

### Known Limitations
1. **Single Domain**: Currently supports blocksworld only
2. **No Execution**: Pipeline stops after code generation (Stage 2)

---

## Future Work

### High Priority
1. **Extended Domain Support**
   - Mars Rover domain
   - Logistics domain
   - Additional IPC benchmark domains

2. **Jason/JAdex Integration**
   - Full BDI platform integration
   - Actual code execution and verification
   - Multi-agent scenarios

3. **Runtime LTL Monitoring**
   - Dynamic goal satisfaction checking during execution
   - Replanning on goal violations

### Medium Priority
4. **Performance Optimization**
   - LLM response caching
   - Incremental plan generation
   - Parallel LLM calls for independent subgoals

5. **Enhanced LTL Support**
   - Complex nested temporal formulas
   - Quantified LTL (QLTL)
   - Probabilistic temporal logic

---

## Legacy: FOND Planning (Branch B)

The project originally included a **Branch B** that used FOND (Fully Observable Non-Deterministic) planning with the PR2 planner. This has been **moved to `src/legacy/fond/`** to focus development on LLM-based approaches.

### Why Moved to Legacy?
- Direct LLM-to-BDI code generation is more aligned with modern LLM capabilities
- Simpler architecture without requiring classical planners and PDDL engineering
- Easier to extend to multiple domains
- More suitable for pure LLM-based research

### Restoring FOND Functionality
See `src/legacy/fond/README.md` for detailed instructions on restoring the FOND planning branch if needed.

---

## Citation

If you use this work, please cite:

```bibtex
@software{ltl_bdi_pipeline,
  title={LTL-BDI Pipeline: LLM-Based AgentSpeak Generation from Natural Language},
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
- Bordini et al.: AgentSpeak semantics and implementation (Jason)
- De Giacomo & Vardi: LTLf specifications
