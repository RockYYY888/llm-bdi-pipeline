# LTL-BDI Pipeline: LLM AgentSpeak Generation

Research pipeline for generating BDI agent code from natural language using LLMs and Linear Temporal Logic (LTLf).

---

## Project Overview

This pipeline converts natural language instructions into executable AgentSpeak code through a three-stage process:

1. **Stage 1**: Natural Language → LTLf (Linear Temporal Logic on Finite Traces) Specification
2. **Stage 2**: LTLf → Recursive DFA Generation (DFS-based decomposition until physical actions)
3. **Stage 3**: DFA → AgentSpeak Code Generation (Backward Planning with state space exploration)

**Note**: This project originally used LLM-based code generation for Stage 3. The current implementation uses **Backward Planning** (forward state-space destruction) for deterministic, optimized code generation. The LLM-based Stage 1 (NL→LTLf) is retained. FOND planning functionality has been moved to `src/legacy/fond/`.

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
|  STAGE 3: DFA -> AgentSpeak Code               |
|  backward_planner_generator.py (Non-LLM)       |
|                                                |
|  Input: DFA with transition labels             |
|  Process: Backward Planning (State Space)      |
|  1. Parse DFA transition labels -> goals       |
|  2. For each goal: Run backward planning       |
|     - Complete state space exploration (BFS)   |
|     - Generate plans for all reachable states  |
|  3. Merge plans into AgentSpeak code           |
|                                                |
|  Output: Complete AgentSpeak (.asl) program    |
|          - Plans for achieving each goal       |
|          - Context-sensitive (state-based)     |
|          - Action preconditions verified       |
|                                                |
|  Optimizations:                                |
|  - Ground actions caching (99.9% reduction)    |
|  - Goal exploration caching (66.7% hit rate)   |
|  - Code deduplication (20-40% reduction)       |
+------------------------------------------------+
```

---

## Key Features

### Backward Planning-Based AgentSpeak Generation (Non-LLM)
- **Complete state-space exploration** from goal states using backward planning (forward destruction)
- **Deterministic code generation** - no LLM randomness in Stage 3
- **Guaranteed correctness** - all plans verified through state space exploration
- **Context-sensitive plans** - generated for every reachable state
- **Multi-goal support** - handles DFAs with multiple transitions
- **Boolean expression parsing** - supports `&`, `|`, `~`, `->`, `<->` in transition labels
- **Performance optimizations**:
  - Ground actions caching: 99.9% redundancy elimination
  - Goal exploration caching: 66.7% cache hit rate for duplicate goals
  - Code structure optimization: 20-40% code size reduction

### LLM-Based Natural Language Understanding (Stage 1 Only)
- Natural language → LTLf conversion using LLMs
- Support for F (eventually), G (always), U (until), X (next), R (release), W (weak until), M (strong release) operators
- Propositional constant handling (true/false) in LTL formulas
- Domain-aware predicate extraction

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
- **Stage 1**: 28 test cases covering all LTL operators and syntax combinations
- **Stage 3**: 14+ comprehensive tests including:
  - Stress tests (2-block scenarios with complex goals)
  - Multi-transition DFA handling
  - Scalability tests (state space explosion analysis)
  - AgentSpeak code validation
  - Goal caching verification
  - Performance measurement (redundancy, reuse ratio)
- Unit tests for symbol normalization with hyphen handling
- Integration tests for end-to-end pipeline validation
- All tests include detailed output analysis and verification

### Comprehensive Logging
- Timestamped execution logs in JSON and human-readable formats
- **Backward planning statistics**: states explored, transitions generated, cache metrics
- **Complete AgentSpeak code** saved to both JSON and separate `.asl` file
- Full LLM prompts and responses recorded for reproducibility (Stage 1)
- Performance metrics: reuse ratio, cache hit rate, redundancy eliminated
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
│   ├── ltl_bdi_pipeline.py              # Main pipeline orchestration (3 stages)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                    # Configuration management
│   │   ├── pddl_parser.py               # PDDL domain parser
│   │   ├── pipeline_logger.py           # Enhanced logging with backward planning stats
│   │   ├── setup_mona_path.py           # Automatic MONA PATH configuration
│   │   └── symbol_normalizer.py         # Symbol normalization with hyphen handling
│   ├── stage1_interpretation/
│   │   ├── __init__.py
│   │   ├── ltlf_generator.py            # Stage 1: NL -> LTLf conversion (LLM)
│   │   ├── ltlf_formula.py              # LTLf formula data structures
│   │   ├── grounding_map.py             # Predicate grounding and propositionalization
│   │   └── prompts.py                   # LLM prompts for NL to LTLf
│   ├── stage2_dfa_generation/
│   │   ├── __init__.py
│   │   ├── dfa_builder.py               # Stage 2: Recursive DFA generation (DFS)
│   │   ├── ltlf_to_dfa.py               # LTLf -> DFA conversion (ltlf2dfa)
│   │   └── dfa_dot_cleaner.py           # DFA DOT file cleaning utilities
│   ├── stage3_code_generation/
│   │   ├── __init__.py
│   │   ├── backward_planner_generator.py   # Stage 3: Main entry point (non-LLM)
│   │   ├── forward_planner.py              # Backward planning (forward destruction)
│   │   ├── agentspeak_codegen.py           # AgentSpeak code generation
│   │   ├── boolean_expression_parser.py    # Parse transition labels (DNF conversion)
│   │   ├── state_space.py                  # State representation and graph
│   │   └── pddl_condition_parser.py        # PDDL precondition/effect parsing
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
│   ├── test_logger_backward_planning.py # Logger integration tests
│   ├── stage1_interpretation/
│   │   └── test_nl_to_ltlf_generation.py    # Stage 1 NL -> LTLf tests (28 cases)
│   ├── stage2_dfa_generation/
│   │   ├── __init__.py
│   │   └── test_ltlf2dfa.py             # Stage 2 LTLf -> DFA tests
│   └── stage3_code_generation/          # Stage 3 comprehensive test suite (14 tests)
│       ├── README.md                             # Test documentation
│       ├── agentspeak_validator.py               # Code validation utility
│       ├── test_integration_backward_planner.py  # Main integration test
│       ├── test_stress_backward_planner.py       # Stress tests (4 scenarios)
│       ├── test_multi_transition_flow.py         # Full multi-transition flow
│       ├── test_multi_transition_simple.py       # Simple multi-transition demo
│       ├── test_goal_caching.py                  # Goal cache verification
│       ├── test_measure_redundancy.py            # Performance measurements
│       ├── test_scalability.py                   # Scalability analysis
│       ├── test_debug_state_explosion.py         # State explosion debugging
│       ├── test_explain_states.py                # State space explanation
│       ├── test_show_ground_actions.py           # Ground actions display
│       ├── test_show_real_code.py                # Real code examples
│       ├── test_visualize_3blocks.py             # 3-block visualization
│       ├── test_visualize_multi_transition.py    # Multi-transition visualization
│       ├── test_simple_2blocks.py                # Simple 2-block test
│       └── test_agentspeak_validator.py          # Validator unit tests
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

# Run Stage 3 tests: Backward Planning -> AgentSpeak

# Core integration and stress tests
python tests/stage3_code_generation/test_integration_backward_planner.py  # Main integration test
python tests/stage3_code_generation/test_stress_backward_planner.py       # Stress tests (4 scenarios)

# Multi-transition and flow tests
python tests/stage3_code_generation/test_multi_transition_flow.py         # Full multi-transition flow
python tests/stage3_code_generation/test_multi_transition_simple.py       # Simple multi-transition demo

# Performance and optimization tests
python tests/stage3_code_generation/test_goal_caching.py                  # Goal cache verification
python tests/stage3_code_generation/test_measure_redundancy.py            # Redundancy measurements
python tests/stage3_code_generation/test_scalability.py                   # Scalability analysis

# Debugging and analysis tests
python tests/stage3_code_generation/test_debug_state_explosion.py         # State explosion analysis
python tests/stage3_code_generation/test_explain_states.py                # State space explanation
python tests/stage3_code_generation/test_show_ground_actions.py           # Ground actions demo
python tests/stage3_code_generation/test_show_real_code.py                # Real code example

# Visualization tests
python tests/stage3_code_generation/test_visualize_3blocks.py             # 3-block visualization
python tests/stage3_code_generation/test_visualize_multi_transition.py    # Multi-transition viz

# Basic tests
python tests/stage3_code_generation/test_simple_2blocks.py                # Simple 2-block test

# Code validation
python tests/stage3_code_generation/test_agentspeak_validator.py          # Validator unit tests

# Run all Stage 3 tests at once (if script exists)
bash run_stage3_tests.sh 2>/dev/null || echo "Use individual test commands above"

# Run symbol normalizer tests
python tests/test_symbol_normalizer.py

# Run logger tests
python tests/test_logger_backward_planning.py

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

### Stage 3: DFA → AgentSpeak Code (Backward Planning)
- **Input**: DFA with transition labels from Stage 2
- **Process**: Backward planning (forward state-space destruction) for deterministic code generation:
  1. **Parse transition labels**: Extract goals from DFA transitions using boolean expression parser
     - Supports: `&` (AND), `|` (OR), `~` (NOT), `->` (IMPLIES), `<->` (IFF)
     - Converts to Disjunctive Normal Form (DNF)
     - Anti-grounds propositions back to predicates
  2. **Backward planning** for each goal:
     - Start from goal state
     - Apply actions in reverse (preconditions → effects swapped)
     - Complete BFS exploration of reachable state space
     - Generate plans for all states that can reach the goal
  3. **Code generation**:
     - Create AgentSpeak plans for each (state, action, next_state) transition
     - Context-sensitive: plans check current beliefs before executing actions
     - Shared components (initial beliefs, action plans) generated once
     - Goal-specific plans merged together
- **Output**: Complete AgentSpeak code (.asl) ready for Jason/JAdex execution
- **Performance**:
  - **States explored**: 1,000+ for simple 2-block scenarios
  - **Reuse ratio**: 57.1:1 (states reused vs created due to deduplication)
  - **Cache optimizations**: 99.9% ground actions caching, 66.7% goal cache hit rate
  - **Code size**: Typically 2,000-5,000 characters for 2-block problems
- **Key Features**:
  - ✅ Deterministic (no LLM randomness)
  - ✅ Guaranteed correctness (all plans verified)
  - ✅ Complete coverage (all reachable states)
  - ✅ Multi-transition DFA support
  - ✅ Conjunctive and disjunctive goal handling

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
- ✓ **Stage 3**: DFA → Backward Planning AgentSpeak code generation (Non-LLM)
  - Complete backward planning algorithm with state space exploration
  - Boolean expression parsing for transition labels (DNF conversion)
  - PDDL action parsing with precondition/effect handling
  - Context-sensitive plan generation for all reachable states
  - Multi-transition DFA support
  - **Performance Optimizations**:
    - Ground actions caching (99.9% redundancy elimination)
    - Goal exploration caching (66.7% cache hit rate)
    - Code structure optimization (20-40% size reduction)
  - Comprehensive testing suite (14+ tests)
  - AgentSpeak code validation
- ✓ **Utils**: Centralized utilities and enhanced logging
  - `SymbolNormalizer` class for consistent symbol handling
  - Hyphen encoding/decoding for ltlf2dfa compatibility
  - `PipelineLogger` with backward planning statistics
  - PDDL domain parser
  - Complete AgentSpeak code saved to `.asl` files
- ✓ **Testing Infrastructure**:
  - Stage 1: 28 test cases for NL → LTLf conversion
  - Stage 3: 14+ comprehensive tests covering:
    - Integration tests
    - Stress tests (4 scenarios with 2 blocks)
    - Multi-transition DFA handling
    - Scalability analysis
    - Goal caching verification
    - Performance measurements
    - AgentSpeak code validation
  - Symbol normalizer unit tests
  - Logger integration tests
  - Integration tests for end-to-end pipeline
- ✓ Blocksworld domain support with physical action identification
- ✓ Comprehensive JSON + text execution logs with:
  - Full LLM prompts and responses (Stage 1)
  - DFA decomposition structure
  - **Backward planning statistics** (states, transitions, cache metrics)
  - Performance metrics (reuse ratio, redundancy eliminated)
- ✓ AgentSpeak code output (.asl files) with complete plan library

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
3. **State Space Explosion**: Backward planning explores complete state space
   - 2 blocks: ~1,000 states
   - 3 blocks: ~30,000+ states (exponential growth)
   - See `docs/stage3_production_limitations.md` for detailed analysis
4. **Memory Requirements**: Complete state graphs stored in memory
5. **Code Size**: Generated AgentSpeak code can be large for complex goals (2,000-5,000 characters for 2-block problems)

---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **`stage3_integration_test_results.md`**: Complete test results for Stage 3 backward planning
  - Integration test outputs
  - Performance metrics
  - Code examples

- **`stage3_optimization_opportunities.md`**: Detailed analysis of implemented optimizations
  - Priority 1: Ground actions caching (99.9% reduction)
  - Priority 2: Goal exploration caching (66.7% hit rate)
  - Priority 3: Code structure optimization (20-40% reduction)
  - Performance impact analysis

- **`stage3_production_limitations.md`**: Production deployment considerations
  - State space explosion analysis
  - Scalability challenges
  - Memory requirements
  - Alternative approaches

- **`stage3_technical_debt.md`**: Technical implementation details and future improvements

- **`tests/stage3_code_generation/README.md`**: Stage 3 test suite documentation

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
4. **Backward Planning Scalability**
   - Partial state space exploration strategies
   - Heuristic-guided search (A*, greedy best-first)
   - Abstract planning for large domains
   - State abstraction techniques
   - Incremental plan refinement

5. **Enhanced LTL Support**
   - Complex nested temporal formulas
   - Quantified LTL (QLTL)
   - Probabilistic temporal logic

6. **Hybrid Approaches**
   - Combine backward planning with LLM-guided heuristics
   - Use LLMs for state abstraction
   - LLM-based plan verification

---

## Legacy Components

### FOND Planning (Branch B)
The project originally included a **Branch B** that used FOND (Fully Observable Non-Deterministic) planning with the PR2 planner. This has been **moved to `src/legacy/fond/`**.

See `src/legacy/fond/README.md` for detailed instructions on restoring the FOND planning branch if needed.

### LLM-Based Stage 3 (Previous Implementation)
The original Stage 3 used LLMs to generate AgentSpeak code from DFAs. This was **replaced with backward planning** for deterministic, verifiable code generation.

**Why Changed to Backward Planning?**
- ✅ **Deterministic**: No LLM randomness, reproducible results
- ✅ **Guaranteed Correctness**: All plans verified through state space exploration
- ✅ **Complete Coverage**: Plans generated for all reachable states
- ✅ **No API Costs**: No LLM calls for Stage 3
- ✅ **Testability**: Easier to validate and debug

**Trade-offs:**
- ⚠️ State space explosion for large domains (3+ blocks)
- ⚠️ Requires PDDL domain specification
- ⚠️ Limited to classical planning domains

**When LLM Might Be Better:**
- Complex domain-specific heuristics
- Natural language action descriptions
- Domains without formal PDDL specifications
- Need for creative problem-solving

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
