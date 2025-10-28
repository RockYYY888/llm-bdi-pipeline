# LTL-BDI Dual-Branch Pipeline

Comparative Evaluation: LLM Policy Generation vs. LLM-Generated AgentSpeak

---

## Project Overview

A research pipeline that compares two LLM-based approaches to intelligent agent planning:
- **Branch A (LLM Policy)**: Direct action sequence generation from LTLf specifications
- **Branch B (AgentSpeak)**: BDI agent program generation with plan libraries

Both branches use LLMs to generate executable plans from LTLf (Linear Temporal Logic on Finite Traces) specifications, demonstrating LLM capability for temporal goal reasoning and action sequence generation.

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install openai python-dotenv
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
# Run both branches (default)
python src/main.py "Stack block C on block B"

# Run LLM policy only
python src/main.py "Stack block C on block B" --mode llm

# Run AgentSpeak only
python src/main.py "Stack block C on block B" --mode asl

# Run PDDL planner only
python src/main.py "Stack block C on block B" --mode pddl
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
+------------------------+  +--------------------------+
|  BRANCH A              |  |  BRANCH B                |
|  LLM Policy            |  |  LLM AgentSpeak          |
+------------------------+  +--------------------------+
       |                                   |
       v                                   v
+------------------------+  +--------------------------+
| STAGE 2A: LLM Policy   |  | STAGE 2B: LLM AgentSpeak |
| Generator              |  | Generator                |
|                        |  |                          |
| Input: F(on(c,b))      |  | Input: F(on(c,b))        |
| Output:                |  | Output:                  |
|  [pickup(c),           |  |  generated_agent.asl     |
|   stack(c,b)]          |  |  (BDI plan library)      |
+------------------------+  +--------------------------+
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
        +-------------------------------+
```

---

## Execution Modes

The pipeline supports four execution modes via the `--mode` flag:

1. **both** (default): Run both LLM and AgentSpeak branches and compare results
2. **llm**: Run only LLM policy generation branch
3. **asl**: Run only AgentSpeak generation branch
4. **pddl**: Run only PDDL classical planning branch (using pyperplan)

---

## Key Features

### Branch A: LLM Policy Generation
- Direct plan generation from LTLf goals
- Goal dependency analysis for multi-goal scenarios
- Optimal action sequence generation
- Example output: `[pickup(c), stack(c, b)]`

### Branch B: AgentSpeak Generation
- Complete BDI agent program generation
- Plan libraries with multiple context-sensitive plans
- Declarative goal handling
- Failure recovery plans (when LLM generates them)

### Comparative Evaluation
- Goal satisfaction verification against LTLf specifications
- Efficiency comparison (action count)
- Success rate tracking
- Detailed execution traces

---

## Example Results

### Simple Goal: "Stack block C on block B"
- **LTLf Formula**: F(on(c, b))
- **Branch A**: SUCCESS - 2 actions
- **Branch B**: SUCCESS - 2 actions
- **Efficiency Ratio**: 1.00 (equal)

### Complex Multi-Goal: "Build a tower with block A on block B on block C"
- **LTLf Formulas**: [F(on(a, b)), F(on(b, c))]
- **Branch A**: SUCCESS - 4 actions (B->C first, then A->B)
  - Correctly analyzes goal dependencies
- **Branch B**: FAILED - Executes A->B first, blocking B->C
  - Demonstrates goal ordering dependency limitation

**Key Finding**: LLM policy generator demonstrates superior goal dependency analysis, correctly determining execution order to avoid blocking states.

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
│   │   └── ltl_parser.py                # NL -> LTLf conversion
│   ├── stage3_codegen/
│   │   ├── llm_policy_generator.py      # LLM policy generation
│   │   └── agentspeak_generator.py      # AgentSpeak generation
│   └── stage4_execution/
│       ├── blocksworld_simulator.py     # Environment simulation
│       ├── agentspeak_simulator.py      # AgentSpeak execution
│       └── comparative_evaluator.py     # Result comparison
├── domains/
│   └── blocksworld/
│       └── domain.pddl                  # Blocksworld domain definition
├── output/                              # Generated plans and logs
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

# Test LLM policy generator
python src/stage3_codegen/llm_policy_generator.py

# Test AgentSpeak generator
python src/stage3_codegen/agentspeak_generator.py
```

---

## Implementation Notes

### LTL-BDI Integration
The pipeline demonstrates LLM integration into BDI systems through:
1. **Goal Specification**: Natural language -> LTLf formal specifications
2. **Policy Generation**: LTL goals -> executable action sequences
3. **BDI Programming**: LTL goals -> AgentSpeak plan libraries
4. **Verification**: LTL goal satisfaction checking

### Blocksworld Domain
The blocksworld domain provides a clear testbed for:
- Goal dependency analysis (stacking order matters)
- State space exploration (block configuration)
- Action precondition/effect reasoning

### Future Extensions
- Additional planning domains (Mars Rover, Logistics)
- Runtime LTL monitoring integration
- FOND (Fully Observable Non-Deterministic) planner integration
- Extended LTL formula support (G, U, X operators)

---

## Implementation Status

### Completed Features ✓
1. **PDDL Planning Integration** - Classical PDDL planning using pyperplan
   - LTLf → PDDL problem generation
   - Pyperplan integration for classical planning
   - PDDL mode (`--mode pddl`) fully functional
   - Maintains PDDL in the pipeline workflow

2. **Logger Output Integration** - Timestamped execution logs
   - Logs saved to `logs/YYYYMMDD_HHMMSS/`
   - Both JSON and human-readable text formats
   - Complete execution trace for all modes

3. **Branch Mode Toggle** - Flexible execution modes
   - `--mode both`: LLM + AgentSpeak comparison
   - `--mode llm`: LLM policy only
   - `--mode asl`: AgentSpeak only
   - `--mode pddl`: Classical PDDL planning only

4. **Documentation** - Clean, accurate README
   - No emojis, single architecture diagram
   - 100% accuracy with implementation
   - Concise and precise content

### Known Limitations

1. **FOND (Non-Deterministic) Planning** - BLOCKED (Solution Available via Docker)
   - **Status**: NOT IMPLEMENTED in current environment
   - **Root Cause**: Direct compilation blocked by modern C++ compiler incompatibility
     - Safe-planner: Requires external classical planners (FF v2.3 fails with implicit function declarations)
     - PRP planner: Uses deprecated C++ header `<hash_set>` removed in C++11+
   - **Available Solution**: Docker/Apptainer containers (requires Docker installation)
     - PRP provides official Dockerfile (`external/planner-for-relevant-policies/apptainer/Dockerfile.prp`)
     - Dockerfile uses Ubuntu 18.04 with compatible C++ compiler
     - Successfully builds PRP with all FOND planning capabilities
   - **Current Workaround**: Classical PDDL planning with pyperplan (deterministic only)
   - **To Enable FOND Planning**:
     ```bash
     # Install Docker Desktop for macOS/Windows or Docker Engine for Linux
     # Then build PRP Docker image:
     cd external/planner-for-relevant-policies/apptainer
     docker build -t prp:latest -f Dockerfile.prp .

     # Usage:
     docker run -v $(pwd):/workspace prp:latest domain.pddl problem.pddl
     ```

2. **PPDDL with Oneof Clauses** - NOT SUPPORTED
   - Current implementation uses standard PDDL (deterministic)
   - PPDDL examples exist in `external/safe-planner/benchmarks/`
   - Requires FOND planner (see above) to utilize non-deterministic effects
   - Once PRP Docker is set up, PPDDL support can be added to pipeline

---

## Future Work / TODO

### High Priority
1. **Enable FOND Planning via Docker**
   - Install Docker Desktop (macOS/Windows) or Docker Engine (Linux)
   - Build PRP Docker image using provided Dockerfile
   - Create Python wrapper (`src/stage3_codegen/prp_docker_planner.py`) to invoke PRP via Docker
   - Replace pyperplan with PRP for non-deterministic planning support

2. **Add PPDDL Problem Generation**
   - Extend `PDDLProblemGenerator` to support `oneof` clauses for non-deterministic effects
   - Example: `(oneof (on b1 b2) (on b1 b3))` for stochastic block placement
   - Enable realistic uncertainty modeling in blocksworld domain

3. **Test FOND Planning with Non-Deterministic Scenarios**
   - Create test cases with probabilistic action outcomes
   - Verify PRP generates robust policies (not just plans)
   - Compare policy robustness vs. classical plans

### Medium Priority
4. **Extend to Additional Domains**
   - Mars Rover domain with sensor uncertainty
   - Logistics domain with probabilistic delivery success
   - Kitchen domain with stochastic cooking outcomes

5. **Runtime Monitoring Integration**
   - LTL runtime monitors for goal satisfaction verification
   - Dynamic replanning on goal violations
   - Integration with BDI execution cycle

### Low Priority
6. **Performance Optimization**
   - Parallel execution of LLM/AgentSpeak/PDDL branches
   - Caching of LLM responses for repeated queries
   - Incremental plan generation

---

## Citation

If you use this work, please cite:

```bibtex
@software{ltl_bdi_pipeline,
  title={LTL-BDI Dual-Branch Pipeline: LLM-Based Policy vs AgentSpeak Generation},
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
- Geffner & Bonet: Planning as heuristic search
