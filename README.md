# LTL-BDI Pipeline: LLM AgentSpeak Generation

Research pipeline for generating BDI agent code from natural language using LLMs and Linear Temporal Logic (LTLf).

---

## Project Overview

This pipeline converts natural language instructions into executable AgentSpeak code through a three-stage process:

1. **Stage 1**: Natural Language → LTLf (Linear Temporal Logic on Finite Traces) Specification
2. **Stage 2**: LTLf → Recursive DFA Generation (DFS-based decomposition until physical actions)
3. **Stage 3**: DFAs → AgentSpeak Code Generation (LLM-guided by DFA transitions)

**Note**: This project previously supported a dual-branch comparison with FOND planning (Branch B). That functionality has been moved to `src/legacy/fond/` to focus development on LLM-based AgentSpeak generation.

---

## Quick Start

### Prerequisites

#### 1. Install MONA (Required for ltlf2dfa)

The `ltlf2dfa` library requires MONA (MONadic second-order logic Automata) to generate complete DFAs with transition labels. MONA is already included in this repository and needs to be compiled:

```bash
# Navigate to MONA directory
cd src/external/mona-1.4

# Configure MONA (with static libraries for compatibility)
./configure --prefix=$(pwd)/mona-install --disable-shared --enable-static

# Compile and install
make && make install-strip

# Verify installation
./mona-install/bin/mona --version
# Should output: MONA v1.4-18 for WS1S/WS2S
```

**Important**: The MONA binary at `src/external/mona-1.4/mona-install/bin/mona` must be accessible for ltlf2dfa to generate complete DFAs with transition labels. The project automatically adds MONA to PATH when you import from `src/` modules.

**Troubleshooting MONA**:
- If MONA not found: Check `./mona-install/bin/mona` exists after compilation
- If DFAs have no transitions: MONA wasn't in PATH when ltlf2dfa ran
- Verify: `python src/setup_mona_path.py` should show "✓ MONA is properly configured"

#### 2. Install Python Dependencies

```bash
# Install Python dependencies using uv (recommended)
uv sync

# Or using pip
pip install openai python-dotenv ltlf2dfa
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
|  STAGE 2: LTLf -> Recursive DFA Generation     |
|  recursive_dfa_builder.py (DFS-based)          |
|                                                |
|  Process:                                      |
|  1. Generate DFA for root goal F(on(c,b))      |
|  2. Analyze transitions for subgoals           |
|  3. Recursively generate DFAs (DFS) until:     |
|     - Physical actions reached (terminal)      |
|     - Existing DFA found (reuse)               |
|                                                |
|  Output: RecursiveDFAResult with:              |
|  - All DFAs (root + subgoals)                  |
|  - DFA transitions (key for plan generation)   |
|  - Physical actions identified                 |
|  - Decomposition tree                          |
+------------------------------------------------+
         |
         v
+------------------------------------------------+
|  STAGE 3: DFAs -> AgentSpeak Code              |
|  agentspeak_generator.py (LLM-based)           |
|                                                |
|  Input: All DFAs with transition information   |
|  Output: Complete AgentSpeak (.asl) program    |
|          - Plans guided by DFA transitions     |
|          - Context-sensitive plan alternatives |
|          - Belief updates and actions          |
+------------------------------------------------+
```

---

## Key Features

### LLM-Based AgentSpeak Generation
- Complete AgentSpeak (.asl) program generation from LTLf goals
- Automatic plan library creation with context-sensitive plans
- Declarative goal handling using LTL temporal operators
- Support for F (eventually), G (always), U (until), X (next), R (release), W (weak until), M (strong release) operators
- Domain-aware code generation with action preconditions
- Propositional constant handling (true/false) in LTL formulas

### Symbol Normalization and Grounding
- Centralized symbol normalization with `SymbolNormalizer` utility class
- Automatic hyphen handling: bidirectional encoding (`-` ↔ `hh`) for ltlf2dfa compatibility
- Predicate-to-proposition conversion (e.g., `on(block-1, block-2)` → `on_blockhh1_blockhh2`)
- Symbol mapping storage for restoration and debugging
- Consistent symbol handling across all pipeline stages

### Formal DFA Conversion
- LTLf to DFA conversion using ltlf2dfa library with MONA backend
- Recursive DFA generation with DFS-based goal decomposition
- Automatic predicate grounding and propositionalization
- DOT format output for visualization
- Formal verification and analysis capabilities
- Complete transition label extraction for plan generation

### Comprehensive Testing
- 28 test cases covering all LTL operators and syntax combinations
- Unit tests for symbol normalization with hyphen handling
- Integration tests for end-to-end pipeline validation
- Stage-specific test suites for isolated component testing

### Comprehensive Logging
- Timestamped execution logs in JSON and human-readable formats
- Full LLM prompts and responses recorded for reproducibility
- Separate log directories for each execution
- Complete trace of NL → LTLf → DFA → AgentSpeak transformation

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
[STAGE 2] Recursive DFA Generation
--------------------------------------------------------------------------------
✓ DFA Decomposition Complete
  Root formula: F(on(c, b))
  Total DFAs: 3
  Physical actions: 2
  Max depth: 2
  Saved to: logs/20251030_123456_dfa_agentspeak/dfa_decomposition.json

--------------------------------------------------------------------------------
[STAGE 3] LLM AgentSpeak Generation from DFAs
--------------------------------------------------------------------------------
✓ AgentSpeak Code Generated
  First few lines:
    // Main goal from LTLf F formula: F(on(c, b))
    +!achieve_on_c_b : true <-
        .print("Starting to achieve on(c,b)");
        !![on(c,b)].

  Saved to: logs/20251030_123456_llm_agentspeak/agentspeak_generated.asl

================================================================================
STAGES 1-3 COMPLETED SUCCESSFULLY
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
│   ├── ltl_bdi_pipeline.py              # Main pipeline orchestration (3 stages)
│   ├── pipeline_logger.py               # Logging utilities (3-stage logging)
│   ├── setup_mona_path.py               # Automatic MONA PATH configuration
│   ├── utils/
│   │   ├── __init__.py
│   │   └── symbol_normalizer.py         # Symbol normalization with hyphen handling
│   ├── stage1_interpretation/
│   │   ├── __init__.py
│   │   ├── ltlf_generator.py            # Stage 1: NL -> LTLf conversion (LLM)
│   │   ├── ltlf_formula.py              # LTLf formula data structures
│   │   ├── grounding_map.py             # Predicate grounding and propositionalization
│   │   ├── pddl_parser.py               # PDDL domain parser
│   │   └── prompts.py                   # LLM prompts for NL to LTLf
│   ├── stage2_dfa_generation/
│   │   ├── __init__.py
│   │   ├── recursive_dfa_builder.py     # Stage 2: Recursive DFA generation (DFS)
│   │   ├── ltlf_to_dfa.py               # LTLf -> DFA conversion (ltlf2dfa)
│   │   └── dfa_dot_cleaner.py           # DFA DOT file cleaning utilities
│   ├── stage3_code_generation/
│   │   ├── __init__.py
│   │   ├── agentspeak_generator.py      # Stage 3: DFAs -> AgentSpeak (LLM)
│   │   └── prompts.py                   # LLM prompts for AgentSpeak generation
│   ├── external/
│   │   ├── mona-1.4/                    # MONA automata tool (for ltlf2dfa)
│   │   └── pr2/                         # PR2 FOND planner
│   └── legacy/
│       └── fond/                        # Legacy FOND planning (Branch B)
│           ├── llm_planner.py
│           ├── llm_policy_generator.py
│           ├── stage2_planning/         # PDDL and PR2 planner
│           └── stage2_translation/      # LTLf to PDDL conversion
├── tests/                               # Test suites
│   ├── __init__.py
│   ├── test_symbol_normalizer.py        # Symbol normalizer unit tests
│   ├── test_integration_pipeline.py     # End-to-end integration tests
│   ├── stage1_interpretation/
│   │   └── test_nl_to_ltlf_generation.py    # Stage 1 NL -> LTLf tests (28 cases)
│   ├── stage2_dfa_generation/
│   │   ├── __init__.py
│   │   └── test_ltlf2dfa.py             # Stage 2 LTLf -> DFA tests
│   └── stage3_code_generation/
│       └── __init__.py
├── logs/                                # Execution logs (timestamped JSON + TXT)
├── run_with_mona.sh                     # Wrapper script to run with MONA in PATH
├── pyproject.toml                       # Project dependencies (uv managed)
└── uv.lock                              # Dependency lock file
```

---

## Development

### Running Tests

**Important**: Tests that use ltlf2dfa require MONA to be available. The project automatically adds MONA to PATH when you import from `src/` modules.

```bash
# Run Stage 1 tests: Natural Language -> LTLf (28 test cases)
python tests/stage1_interpretation/test_nl_to_ltlf_generation.py

# Run Stage 2 tests: LTLf -> DFA conversion
python tests/stage2_dfa_generation/test_ltlf2dfa.py

# Run symbol normalizer tests
python tests/test_symbol_normalizer.py

# Run integration tests (end-to-end pipeline)
python tests/test_integration_pipeline.py
```

**Alternative**: Use the wrapper script to run any Python script with MONA in PATH:

```bash
# Run with explicit MONA path setup
./run_with_mona.sh python tests/stage1_interpretation/test_nl_to_ltlf_generation.py
```

---

## Pipeline Stages

### Stage 1: Natural Language → LTLf
- **Input**: Natural language instruction (e.g., "Stack block C on block B")
- **Process**: LLM-based parser (`ltlf_generator.py`) extracts objects, predicates, and temporal goals
  - Uses structured prompts with JSON output format
  - Supports all LTL operators: F, G, U, X, R, W, M
  - Handles propositional constants (true/false)
  - Handles nested and complex temporal formulas
  - Creates grounding map for predicate propositionalization
- **Output**:
  - LTLf formulas in JSON format
  - Objects list
  - Initial state (optional)
  - Grounding map with predicate-to-proposition mappings
- **LLM Model**: Configured via `OPENAI_MODEL` in `.env` (e.g., `deepseek-chat`, `gpt-4o-mini`)
- **Testing**: 28 comprehensive test cases covering all operators and edge cases

### Stage 2: LTLf → Recursive DFA Generation
- **Input**: LTLf specification from Stage 1
- **Process**: Recursively decomposes LTLf goals into subgoals using DFS strategy
  1. Normalize LTLf formula using `SymbolNormalizer` (hyphen encoding)
  2. Generate DFA for root goal using ltlf2dfa + MONA
  3. Analyze DFA transitions to identify subgoals
  4. For each subgoal:
     - If physical action → mark as terminal (no further decomposition)
     - If DFA already exists → reuse cached DFA
     - Otherwise → recursively generate DFA (DFS deeper)
  5. Continue until all paths reach physical actions or cached DFAs
- **Output**: `RecursiveDFAResult` containing:
  - All generated DFAs (root + subgoals) in breadth-first order
  - DFA transitions for each goal (key information for plan generation)
  - Physical actions identified (terminal nodes)
  - Decomposition tree structure with depth information
  - DFA dependency graph
  - Symbol mappings (normalized ↔ original)
- **Tools**:
  - ltlf2dfa library (Python interface for DFA generation)
  - MONA v1.4 (underlying automata generator - must be compiled)
  - SymbolNormalizer (hyphen handling and propositionalization)
  - Custom recursive DFA builder with DFS decomposition
- **Requirements**: MONA must be compiled and available (see Prerequisites above)
- **Key Features**:
  - Automatic predicate-to-proposition encoding with hyphen handling
  - Example: `on(block-1, block-2)` → `on_blockhh1_blockhh2`
  - DFA caching to prevent regeneration
  - Cycle prevention through cache-before-recurse pattern
  - Complete transition labels extraction from MONA output
  - DOT file generation for DFA visualization

### Stage 3: DFAs → AgentSpeak Code
- **Input**: All DFAs with transition information from Stage 2
- **Process**: LLM generates complete AgentSpeak program guided by DFA decomposition:
  - Uses DFA transitions as plan context conditions
  - Generates plans for each subgoal in decomposition tree
  - Creates context-sensitive plan alternatives based on different DFA transitions
  - Incorporates initial beliefs (from Stage 1 initial state)
  - Adds belief updates and action sequences
  - Includes failure handling plans (-!goal)
- **Output**: Complete AgentSpeak code (.asl) ready for Jason/JAdex execution
- **LLM Model**: Same as Stage 1
- **Guidance Strategy**: DFA transitions provide explicit conditions for when plans should be triggered, enabling more accurate and context-aware plan generation

---

## Implementation Notes

### Symbol Normalization Architecture
The `SymbolNormalizer` utility class provides centralized symbol handling across all pipeline stages:

**Key Design Decisions**:
- **Hyphen Encoding**: Uses `hh` as replacement for `-` to avoid parsing issues in ltlf2dfa/MONA
  - Example: `block-1` → `blockhh1` → `block-1` (bidirectional)
  - Chosen `hh` because it's unlikely to appear naturally in domain symbols
- **Bidirectional Mapping**: Stores both normalized → original and original → normalized mappings
  - Enables symbol restoration for debugging and output
  - Maintains consistency across pipeline stages
- **Integration Points**:
  - Stage 1: `GroundingMap` uses normalizer for propositional symbol creation
  - Stage 2: `PredicateToProposition` uses normalizer for formula normalization
  - Stage 2: `RecursiveDFABuilder` passes normalizer through DFA generation

**Usage Example**:
```python
from utils.symbol_normalizer import SymbolNormalizer

normalizer = SymbolNormalizer()

# Encode hyphens for ltlf2dfa
normalized = normalizer.normalize_formula_string("F(on(block-1, block-2))")
# Result: "F(on_blockhh1_blockhh2)"

# Decode back to original
original = normalizer.denormalize_formula_string(normalized)
# Result: "F(on(block-1, block-2))"
```

### LTL-BDI Integration
The pipeline demonstrates LLM-based generation of BDI agent code from declarative temporal goals:
- **Declarative Goals**: LTLf formulas specify *what* should be achieved, not *how*
- **Plan Generation**: LLM translates temporal goals into procedural AgentSpeak plans
- **Domain Knowledge**: LLM incorporates domain actions and predicates into generated code
- **Formal Verification**: DFA representation enables formal analysis of goal achievability

### Blocksworld Domain
The blocksworld domain provides a testbed for:
- Goal dependency analysis (stacking order)
- State space exploration (block configurations)
- Action precondition/effect reasoning
- Symbol normalization with hyphenated identifiers

**Domain Actions**: pickup, putdown, stack, unstack
**Domain Predicates**: on(X, Y), ontable(X), clear(X), holding(X), handempty

---

## Current Status

### Implemented Features (Stages 1-3)
- ✓ **Stage 1**: Natural language → LTLf specification (LLM-based)
  - 28 comprehensive test cases covering all LTL operators
  - Propositional constant handling (true/false)
  - Support for F, G, U, X, R, W, M temporal operators
  - Complex nested formula support
  - JSON structured output format
  - Grounding map generation for propositionalization
- ✓ **Stage 2**: LTLf → Recursive DFA generation (DFS-based decomposition)
  - Recursive goal decomposition until physical actions
  - Symbol normalization with hyphen handling (`-` ↔ `hh`)
  - DFA caching and reuse to prevent regeneration
  - Predicate-to-proposition encoding with special character handling
  - Complete DFA transitions extraction from MONA
  - Decomposition tree with depth tracking
  - DOT format output for visualization
  - Bidirectional symbol mapping storage
- ✓ **Stage 3**: DFAs → LLM AgentSpeak code generation
  - DFA-guided plan generation using transition information
  - Context-sensitive plan alternatives
  - Failure handling plans
- ✓ **Utils**: Centralized symbol normalization utilities
  - `SymbolNormalizer` class for consistent symbol handling
  - Hyphen encoding/decoding for ltlf2dfa compatibility
  - Symbol mapping storage and retrieval
- ✓ **Testing Infrastructure**:
  - Stage 1: 28 test cases for NL → LTLf conversion
  - Symbol normalizer unit tests
  - Integration tests for end-to-end pipeline
  - Stage-specific test suites
- ✓ Blocksworld domain support with physical action identification
- ✓ Comprehensive JSON + text execution logs with full LLM prompts and DFA decomposition
- ✓ AgentSpeak code output (.asl files)

### Not Yet Implemented
- ⏳ **Stage 4**: Execution & Comparative Evaluation
  - Execution of AgentSpeak code in Jason/JAdex
  - Goal satisfaction verification
  - Performance metrics
  - (Code exists in `legacy/stage4_execution/` for future development)
- ⏳ **Extended Domain Support**: Mars Rover, Logistics, other IPC domains

### Known Limitations
1. **Single Domain**: Currently tested primarily with blocksworld domain
2. **No Execution**: Pipeline stops after code generation (Stage 3)
3. **Subgoal Extraction**: Current implementation extracts subgoals from transition labels, which may not capture all domain-specific decomposition strategies

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
