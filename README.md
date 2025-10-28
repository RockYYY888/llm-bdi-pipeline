# LTL-BDI Dual-Branch Pipeline

**Comparative Evaluation: Classical Planning vs. LLM-Generated AgentSpeak**

---

## 🎯 Project Overview

A research pipeline that compares two approaches to intelligent agent planning:
- **Branch A (Baseline)**: Classical PDDL planning
- **Branch B (Novel)**: LLM-generated AgentSpeak plan libraries

**Key Innovation**: Generate complete BDI agent plan libraries from LTLf specifications using LLMs, then compare against classical planning.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd llm-bdi-pipeline-dev
/opt/anaconda3/bin/pip install pyperplan openai python-dotenv
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
python src/main.py "Stack block C on block B"
```

**Expected Output**:
```
================================================================================
LTL-BDI PIPELINE - DUAL BRANCH DEMONSTRATION
================================================================================

[STAGE 1] Natural Language -> LTLf Specification
✓ LTLf Formula: ['F(on(c, b))']
  Objects: ['b', 'c']

[STAGE 2] LTLf -> PDDL Problem
✓ PDDL Problem Generated

[STAGE 3A] BRANCH A: Classical PDDL Planning
✓ Classical Plan Generated (2 actions)
  1. pickup(c)
  2. stack(c, b)

[STAGE 3B] BRANCH B: LLM AgentSpeak Generation
✓ AgentSpeak Plan Library Generated
  Plans: 17
  Saved to: output/generated_agent.asl

[STAGE 4] Execution & Comparative Evaluation
✓ Both branches succeeded

Efficiency:
  Classical Actions: 2
  AgentSpeak Actions: 2
  Efficiency Ratio: 1.00
```

---

## 📐 System Architecture

```
Natural Language Input
         │
         ▼
┌────────────────────────────────────┐
│  STAGE 1: NL → LTLf               │
│  ltl_parser.py                     │
│  Output: F(on(c,b))                │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  STAGE 2: LTLf → PDDL             │
│  ltl_to_pddl.py                    │
│  Output: problem.pddl              │
└────────────┬───────────────────────┘
             │
       ┌─────┴─────┐
       │           │
       ▼           ▼
┌──────────────┐  ┌──────────────────────┐
│  BRANCH A    │  │  BRANCH B            │
│  Classical   │  │  LLM AgentSpeak      │
└──────────────┘  └──────────────────────┘
       │                   │
       ▼                   ▼
┌──────────────────────────────────────────┐
│  STAGE 3A: Classical Planning            │
│  pddl_planner.py                         │
│  Output: [pickup(c), stack(c,b)]         │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  STAGE 3B: AgentSpeak Generation         │
│  agentspeak_generator.py                 │
│  Output: generated_agent.asl             │
│  (Complete BDI plan library)             │
└──────────────────────────────────────────┘
       │                   │
       └─────┬─────────────┘
             ▼
┌──────────────────────────────────────────┐
│  STAGE 4: Execution & Comparison         │
│  agentspeak_simulator.py                 │
│  comparative_evaluator.py                │
│                                          │
│  Metrics:                                │
│  - Goal satisfaction                     │
│  - Efficiency (action count)             │
│  - Success rate                          │
└──────────────────────────────────────────┘
```

---

## 🔬 Stage Details

### Stage 1: Natural Language → LTLf

**Input**: `"Stack block C on block B"`

**Output**:
```json
{
  "objects": ["c", "b"],
  "initial_state": [
    {"ontable": ["b"]},
    {"ontable": ["c"]},
    {"clear": ["b"]},
    {"clear": ["c"]},
    {"handempty": []}
  ],
  "formulas": ["F(on(c, b))"]
}
```

**Implementation**: `src/stage1_interpretation/ltl_parser.py`

---

### Stage 2: LTLf → PDDL Problem

**Input**: LTLf specification from Stage 1

**Output**: `problem.pddl`
```pddl
(define (problem stack-blocks)
  (:domain blocksworld)
  (:objects c b - block)
  (:init (ontable c) (ontable b) (clear c) (clear b) (handempty))
  (:goal (and (on c b)))
)
```

**Implementation**: `src/stage2_translation/ltl_to_pddl.py`

---

### Stage 3A: Classical PDDL Planning (Branch A - Baseline)

**Input**: `problem.pddl` + `domain.pddl`

**Output**: Action sequence
```python
[('pickup', ['c']), ('stack', ['c', 'b'])]
```

**Characteristics**:
- ✅ Optimal (minimal actions)
- ✅ Fast execution
- ❌ No failure recovery
- ❌ Brittle to unexpected states

**Implementation**: `src/stage3_codegen/pddl_planner.py` (using pyperplan)

---

### Stage 3B: LLM AgentSpeak Generation (Branch B - Novel)

**Input**: LTLf specification + domain context

**Output**: `generated_agent.asl` (AgentSpeak plan library)

**Example Generated Code**:
```agentspeak
// Main goal from LTLf F formula: F(on(c, b))
+!achieve_on_c_b : true <-
    .print("Starting to achieve on(c, b)");
    !![on(c, b)].

// Declarative goal for on(c, b)
+!![on(c, b)] : on(c, b) <-
    .print("Goal already achieved");
    !verify_success.

// Plan when blocks are clear and on table
+!![on(c, b)] : clear(c) & ontable(c) & clear(b) & handempty <-
    pickup(c);
    +holding(c);
    -handempty;
    stack(c, b);
    -holding(c);
    +handempty;
    +on(c, b).

// Failure handling
-!![on(c, b)] : true <-
    .print("Declarative goal failed, retrying");
    !recover_and_retry.
```

**Characteristics**:
- ✅ Context-adaptive plans
- ✅ Failure recovery strategies
- ✅ Multiple plan options
- ⚠️ May be sub-optimal

**Implementation**: `src/stage3_codegen/agentspeak_generator.py`

---

### Stage 4: Execution & Comparative Evaluation

**Branch A Execution**: Sequential action execution in blocksworld simulator

**Branch B Execution**: BDI reasoning cycle
1. Goal posting: `achieve_on_c_b`
2. Plan selection: Match trigger and context
3. Action execution: Execute plan body
4. Belief updates: Track state changes

**Comparison Metrics**:
```
✓ Both branches succeeded

Efficiency:
  Classical Actions: 2
  AgentSpeak Actions: 2
  Efficiency Ratio: 1.00

Robustness:
  Classical Failure Recovery: False
  AgentSpeak Failure Plans: False
```

**Implementation**:
- `src/stage4_execution/blocksworld_simulator.py` - Environment
- `src/stage4_execution/agentspeak_simulator.py` - BDI execution
- `src/stage4_execution/comparative_evaluator.py` - Metrics

---

## 🛠️ Implementation Status

### ✅ Completed Components

**Core Pipeline**:
- [x] Stage 1: NL → LTLf parser
- [x] Stage 2: LTLf → PDDL converter
- [x] Stage 3A: Classical PDDL planner
- [x] Stage 3B: AgentSpeak generator
- [x] Stage 4: Execution & comparison

**AgentSpeak Simulator**:
- [x] Multi-line plan parsing
- [x] Declarative goal support (`+!!`)
- [x] Variable unification
- [x] Belief format conversion (`ontable` ↔ `on(X,table)`)
- [x] Context checking with negation
- [x] Primitive action execution
- [x] BDI reasoning cycle

**Pipeline Infrastructure**:
- [x] Configuration management (.env)
- [x] Logging system
- [x] Dual-branch orchestration
- [x] Blocksworld environment simulator

### 🔧 Known Limitations (MVP Scope)

**LTLf Verification**:
- ✅ Supports: `F(φ)` (Eventually)
- ❌ Not yet: `G(φ)`, `X(φ)`, `φ U ψ` (requires temporal trace verification)

**AgentSpeak Features**:
- ✅ Supports: Achievement goals, declarative goals, context guards, belief updates
- ❌ Not yet: Annotations, complex internal actions, event handling

**Single Agent Only**:
- ✅ Single agent BDI execution
- ❌ Multi-agent coordination (out of scope)

---

## 📊 Test Results

### Simple Stack Scenario: "Stack block C on block B"

**Test Status**: ✅ **PASSING**

```
BRANCH A: Classical PDDL Planning
  Success: True
  Actions: 2 (pickup, stack)
  Final State: on(c,b) achieved

BRANCH B: LLM AgentSpeak
  Success: True
  Actions: 2 (pickup, stack)
  Final State: on(c,b) achieved

Comparison:
  Efficiency Ratio: 1.00 (equal)
  Both branches succeeded
```

### Additional Scenarios

| Scenario | Classical | AgentSpeak | Status |
|----------|-----------|------------|--------|
| Simple stack (C on B) | ✅ Pass | ✅ Pass | **VERIFIED** |
| Three-block tower | ❓ | ❓ | Not tested |
| Block rearrangement | ❓ | ❓ | Not tested |
| Parallel stacks | ❓ | ❓ | Not tested |

---

## 📁 Project Structure

```
llm-bdi-pipeline-dev/
├── src/
│   ├── stage1_interpretation/
│   │   └── ltl_parser.py              # NL → LTLf
│   ├── stage2_translation/
│   │   └── ltl_to_pddl.py            # LTLf → PDDL
│   ├── stage3_codegen/
│   │   ├── pddl_planner.py           # Classical planner
│   │   └── agentspeak_generator.py   # LLM AgentSpeak gen
│   ├── stage4_execution/
│   │   ├── blocksworld_simulator.py  # Environment
│   │   ├── agentspeak_simulator.py   # BDI execution
│   │   └── comparative_evaluator.py  # Metrics
│   ├── dual_branch_pipeline.py       # Pipeline orchestrator
│   ├── config.py                      # Configuration
│   ├── pipeline_logger.py             # Logging
│   └── main.py                        # Entry point
├── domains/
│   └── blocksworld/
│       └── domain.pddl                # PDDL domain (given)
├── docs/
│   └── IMPLEMENTATION_ISSUES_AND_FIXES.md  # Technical docs
├── output/
│   └── generated_agent.asl            # Generated plans
└── .env                               # API keys (gitignored)
```

---

## 🔑 Critical Implementation Details

### Fix #1: Initial State Format Conversion

**Problem**: LTL parser generates `ontable(X)` but simulator expects `on(X, table)`

**Solution**: Automatic conversion in `dual_branch_pipeline.py`:
```python
if pred_name == 'ontable' and args:
    for block in args:
        beliefs.append(f"on({block}, table)")
```

### Fix #2: Multi-Line Plan Parsing

**Problem**: Generated AgentSpeak code spans multiple lines but parser expected single-line plans

**Solution**: Accumulate lines until period found in `agentspeak_simulator.py`:
```python
def _parse_asl(self, asl_code: str):
    current_plan_text = ""
    for line in lines:
        if line.startswith(('+!', '-!')):
            if current_plan_text:
                self._parse_single_plan(current_plan_text)
            current_plan_text = line
        elif current_plan_text:
            current_plan_text += " " + line

        if current_plan_text and current_plan_text.endswith('.'):
            self._parse_single_plan(current_plan_text)
            current_plan_text = ""
```

### Fix #3: Belief Format Bidirectional Conversion

**Problem**: Generated code uses `ontable(c)` but beliefs contain `on(c,table)` (with/without space)

**Solution**: Handle both formats in belief checking:
```python
def _belief_exists(self, condition: str) -> bool:
    # Direct match
    if condition in self.beliefs:
        return True

    # Convert ontable(X) to on(X, table) - try both formats
    match = re.match(r'ontable\((\w+)\)', condition)
    if match:
        block = match.group(1)
        return (f"on({block}, table)" in self.beliefs or
                f"on({block},table)" in self.beliefs)

    # Convert on(X, table) to ontable(X)
    match = re.match(r'on\((\w+),\s*table\)', condition)
    if match:
        block = match.group(1)
        return f"ontable({block})" in self.beliefs

    return False
```

### Fix #4: Declarative Goal Execution

**Problem**: AgentSpeak uses `!!goal` for declarative goals but action executor didn't distinguish from `!goal`

**Solution**: Check satisfaction before plan selection:
```python
def _achieve_goal(self, goal: str) -> bool:
    # For declarative goals, check if already satisfied
    if goal.startswith('[') and goal.endswith(']'):
        goal_condition = goal.strip('[]')
        if self._belief_exists(goal_condition):
            return True  # Already satisfied!

    # Otherwise select and execute plan
    plan = self._select_plan(goal)
    ...
```

---

## 🎓 Research Context

**Project**: Final Year Project (FYP)
**Institution**: University of Nottingham Ningbo China
**Author**: Yiwei LI (20513831)
**Supervisor**: Yuan Yao

### Research Questions

**RQ1**: Can LLMs generate correct AgentSpeak plan libraries from LTLf specifications?
- ✅ Syntax: Generated code parses successfully
- ✅ Semantics: Plans execute correctly in blocksworld

**RQ2**: How do LLM-generated plans compare to classical planning?
- ✅ Efficiency: Equal action count for simple scenarios
- ⚠️ Optimality: Requires more test cases

**RQ3**: Are LLM-generated plans more robust to failures?
- ❓ Pending: Failure injection tests needed

**RQ4**: Can LLM plans handle novel situations?
- ❓ Pending: Unseen state tests needed

### Key Contributions

1. **Dual-Branch Comparative Framework**: First implementation comparing classical planning with LLM-generated BDI plans
2. **LTLf-to-AgentSpeak Pipeline**: Novel approach using temporal logic as intermediate representation
3. **Working BDI Simulator**: Pure Python implementation of AgentSpeak subset (no Jason dependency)
4. **Comprehensive Fix Documentation**: All implementation challenges and solutions documented

---

## 📝 License

Academic research project - University of Nottingham Ningbo China

---

## 📚 References

- **BDI Architecture**: Bordini et al. (2007) - Programming Multi-Agent Systems in AgentSpeak using Jason
- **LTLf**: De Giacomo & Vardi (2013) - Linear Temporal Logic on Finite Traces
- **PDDL Planning**: Geffner & Bonet (2013) - A Concise Introduction to Models and Methods for Automated Planning
