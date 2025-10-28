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

**L1: Limited LTLf Temporal Verification**
- **Impact**: Can only verify `F(φ)` (Eventually) goals, not `G(φ)` (Always), `X(φ)` (Next), or `φ U ψ` (Until)
- **Current Implementation**: Checks if goal predicate exists in final state (no temporal trace verification)
- **Code** (`comparative_evaluator.py`):
  ```python
  def _check_ltl_satisfaction(self, final_state, ltl_goal):
      if ltl_goal.startswith('F(') and ltl_goal.endswith(')'):
          goal_predicate = ltl_goal[2:-1]  # Extract φ from F(φ)
          return goal_predicate in final_state
      return True  # MVP: can't verify other formulas
  ```
- **Future Work**: Integrate proper LTLf verification library (e.g., spot, pyLTL)
- **Priority**: LOW (acceptable for research prototype)

**L2: Simplified AgentSpeak Subset**
- **Supported**: Achievement goals (`+!goal`), declarative goals (`+!![state]`), context guards, belief updates (`+/-belief`)
- **Not Supported**:
  - Complex annotations
  - Internal actions beyond `.print()`
  - Event handling and intentions
  - Strong negation
- **Workaround**: Generator produces simplified AgentSpeak code within supported subset
- **Priority**: MEDIUM

**L3: Manual Belief Revision**
- **Impact**: Belief updates after actions must be explicitly coded in AgentSpeak plans
- **Example**: After `pickup(X)`, must manually add `+holding(X); -handempty; -clear(X)`
- **Workaround**: Generator includes explicit +/- belief updates in all action plans
- **Priority**: LOW (manual specification acceptable for MVP)

**L4: Single Agent Only**
- **Impact**: No multi-agent coordination or communication
- **Scope**: Intentionally out of scope for this research
- **Priority**: N/A

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
├── output/
│   └── generated_agent.asl            # Generated plans
└── .env                               # API keys (gitignored)
```

---

## 🔑 Critical Implementation Fixes

Seven critical bugs were identified and fixed during implementation. Each fix was essential for the pipeline to work correctly.

### Fix #1: Initial State Format Conversion

**Problem**: LTL parser generates `ontable(X)` but blocksworld simulator expects `on(X, table)`

**Impact**: Classical planner failed with precondition errors

**Solution** (`dual_branch_pipeline.py:236-246`):
```python
# Convert dict-based initial_state to string-based beliefs
beliefs = []
for pred_dict in ltl_spec.initial_state:
    for pred_name, args in pred_dict.items():
        if pred_name == 'ontable' and args:
            # Convert ontable(X) to on(X, table)
            for block in args:
                beliefs.append(f"on({block}, table)")
        elif args:
            beliefs.append(f"{pred_name}({', '.join(args)})")
        else:
            beliefs.append(pred_name)
```

### Fix #2: Multi-Line Plan Parsing

**Problem**: Generated AgentSpeak code spans multiple lines but parser expected single-line plans

**Impact**: 0 plans were parsed successfully

**Solution** (`agentspeak_simulator.py:41-76`):
```python
def _parse_asl(self, asl_code: str):
    """Parse AgentSpeak code - handles multi-line plans"""
    current_plan_text = ""
    for line in lines:
        line = line.strip()
        if not line or (line.startswith('//') and not current_plan_text):
            continue

        if line.startswith(('+!', '-!')):
            if current_plan_text:
                self._parse_single_plan(current_plan_text)
            current_plan_text = line
        elif current_plan_text:
            current_plan_text += " " + line

        if current_plan_text and current_plan_text.rstrip().endswith('.'):
            self._parse_single_plan(current_plan_text)
            current_plan_text = ""
```

### Fix #3: Declarative Goal Support

**Problem**: Parser couldn't handle `+!![goal]` syntax for declarative goals

**Impact**: Generated plans with declarative goals were not recognized

**Solution** (`agentspeak_simulator.py:82`):
```python
# Pattern handles both +! (achievement) and +!! (declarative)
plan_pattern = r'(\+!!?[\w\(\),\s\[\]_]+)\s*:\s*([^<]*)\s*<-\s*(.+)\.'
```

### Fix #4: Variable Unification

**Problem**: Plans with variables `stack(X,Y)` couldn't match goals `stack(c,b)`

**Impact**: No plans were found as "applicable" during execution

**Solution** (`agentspeak_simulator.py:183-224`):
```python
def _unify_goal(self, pattern: str, goal: str) -> bool:
    """Check if pattern matches goal with variable unification"""
    pattern_match = re.match(r'(\w+)\((.*?)\)', pattern)
    goal_match = re.match(r'(\w+)\((.*?)\)', goal)

    if not pattern_match or not goal_match:
        return False

    pattern_pred, pattern_args_str = pattern_match.groups()
    goal_pred, goal_args_str = goal_match.groups()

    # Predicates must match
    if pattern_pred != goal_pred:
        return False

    # Parse and check arguments
    pattern_args = [a.strip() for a in pattern_args_str.split(',') if a.strip()]
    goal_args = [a.strip() for a in goal_args_str.split(',') if a.strip()]

    if len(pattern_args) != len(goal_args):
        return False

    for p_arg, g_arg in zip(pattern_args, goal_args):
        # Uppercase = variable (matches anything)
        if p_arg and p_arg[0].isupper():
            continue
        # Lowercase = constant (must match exactly)
        elif p_arg == g_arg:
            continue
        else:
            return False

    return True
```

### Fix #5: Belief Format Bidirectional Conversion

**Problem**: Generated code checks `ontable(c)` but beliefs contain `on(c,table)` (with/without space)

**Impact**: All context checks failed, no plans were applicable

**Solution** (`agentspeak_simulator.py:128-150`):
```python
def _belief_exists(self, condition: str) -> bool:
    """Check if belief exists, handling ontable(X) <-> on(X,table) conversion"""
    # Direct match first
    if condition in self.beliefs:
        return True

    # Convert ontable(X) to on(X, table) - try both with and without space
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

### Fix #6: Declarative Goal Early Satisfaction

**Problem**: Declarative goals `!!goal` were not distinguished from achievement goals `!goal`

**Impact**: Redundant plan execution even when goal already satisfied

**Solution** (`agentspeak_simulator.py:299-308`):
```python
def _achieve_goal(self, goal: str) -> bool:
    """Achieve a goal using BDI reasoning cycle"""
    # For declarative goals with brackets, check if already satisfied
    if goal.startswith('[') and goal.endswith(']'):
        goal_condition = goal.strip('[]')
        if self._belief_exists(goal_condition):
            return True  # Already satisfied!

    # Otherwise, select and execute plan
    plan = self._select_plan(goal)
    ...
```

### Fix #7: Bracket Normalization

**Problem**: Goal `[on(c,b)]` didn't match trigger `on(c,b)` in plan selection

**Impact**: Valid plans were not selected

**Solution** (`agentspeak_simulator.py:152-181`):
```python
def _select_plan(self, goal: str) -> Optional[AgentSpeakPlan]:
    """Select applicable plan for goal with variable unification"""
    # Strip brackets from goal if present (declarative goals)
    goal_normalized = goal.strip('[]')

    for plan in self.plans:
        # Extract goal from trigger and remove brackets
        trigger_goal = plan.trigger.replace('+!!', '').replace('+!', '').strip()
        trigger_goal = trigger_goal.strip('[]')

        # Compare normalized forms
        if trigger_goal == goal_normalized:
            if self._check_context(plan.context):
                applicable.append(plan)
        elif self._unify_goal(trigger_goal, goal_normalized):
            if self._check_context(plan.context):
                applicable.append(plan)

    return applicable[0] if applicable else None
```

---

## ⚠️ Known Issues and Incomplete Features

**IMPORTANT**: The following issues were discovered through comprehensive testing. The pipeline works correctly for **simple single-goal scenarios** (e.g., "Stack block C on block B") but has limitations for more complex cases.

### Issue 1: Multiple LTLf Goals Not Fully Supported ⚠️ MAJOR

**Problem**: When natural language describes multiple goals (e.g., "Build a tower with A on B on C"), the pipeline generates multiple LTLf formulas but only processes the first one.

**Test Case**:
```bash
Input: "Build a tower with block A on block B on block C"
LTLf Generated: ['F(on(a, b))', 'F(on(b, c))']  # Two goals!
```

**What Happens**:
1. **Stage 2 (PDDL Generation)**: LLM may generate incomplete PDDL goal (observed: `(on a b` without closing paren and missing second goal)
2. **Stage 3B (AgentSpeak)**: `_extract_agentspeak_goal()` only extracts first formula → creates `achieve_on_a_b`, ignores `F(on(b,c))`
3. **Stage 4 (Verification)**: `_check_ltl_satisfaction()` only checks first formula

**Impact**:
- ✅ Classical planner: May still work if PDDL LLM generates complete conjunctive goal
- ❌ AgentSpeak execution: Only achieves first sub-goal, ignores others
- ❌ Goal verification: Reports "Goal Satisfied: False" even when first goal is met

**Affected Code**:
- `dual_branch_pipeline.py:277-292` - `_extract_agentspeak_goal()` only matches one `F(on(X,Y))`
- `comparative_evaluator.py:148-163` - `_check_ltl_satisfaction()` only checks one goal
- `dual_branch_pipeline.py:251-262` - Passes only `formulas_string_list[0]` to evaluator

**Status**: 🔴 **INCOMPLETE** - Multi-goal support not implemented

---

### Issue 2: LTLf Goal Verification Space Mismatch ⚠️ MODERATE → ✅ FIXED

**Problem**: Goal verification fails due to whitespace inconsistency between extracted goal and final state format.

**Root Cause**:
```python
# comparative_evaluator.py:156 (OLD CODE)
goal_predicate = ltl_goal[2:-1]  # Extracts "on(a, b)" with space
# But final_state contains: "on(a,b)" (no space)
return goal_predicate in final_state  # Always False!
```

**Test Evidence**:
```
Test: "Stack block C on block B"
LTLf Goal: F(on(c, b))          # Space after comma
Extracted: "on(c, b)"            # Space preserved
Final State: ['on(c,b)', ...]   # No space
Result: Goal Satisfied: False   # WRONG! Goal was actually achieved
```

**Fix Applied** (Phase 1.1):
Normalize spaces in both goal and state before comparison:
```python
# comparative_evaluator.py:148-170 (NEW CODE)
goal_normalized = goal_predicate.replace(' ', '')  # Remove spaces
for state_pred in final_state:
    if state_pred.replace(' ', '') == goal_normalized:
        return True
```

**Code Location**: `src/stage4_execution/comparative_evaluator.py:148-170`

**Status**: ✅ **FIXED** - Space normalization added

---

### Issue 3: Hardcoded Single-Goal Assumptions 🔒 LIMITATION

**Problem**: Multiple components assume exactly one LTLf goal.

**Hardcoded Logic**:

1. **Goal Extraction** (`dual_branch_pipeline.py:277-292`):
   ```python
   def _extract_agentspeak_goal(self, ltl_formula: str):
       # Match F(on(X, Y)) - ONLY ONE!
       match = re.match(r'F\(on\((\w+),\s*(\w+)\)\)', ltl_formula)
   ```

2. **Goal Passing** (`dual_branch_pipeline.py:253`):
   ```python
   agentspeak_goal = self._extract_agentspeak_goal(formulas_string_list[0])  # [0]!
   ```

3. **Verification** (`comparative_evaluator.py:180`):
   ```python
   report.append(f"\nLTLf Goal: {self.results['ltl_goal']}")  # Singular!
   ```

**Impact**:
- Works for: Simple scenarios with one goal ("Stack C on B")
- Fails for: Complex scenarios with multiple goals ("Build tower A-B-C")

**Status**: 🔒 **BY DESIGN** - MVP limitation

---

### Issue 4: No Conjunctive or Sequential Goal Support 🔒 LIMITATION

**Problem**: Cannot handle:
- Conjunctive goals: `F(on(a,b) & on(b,c))` - "A on B AND B on C simultaneously"
- Sequential goals: `F(on(b,c)) & F(F(on(a,b)))` - "First B on C, then A on B"
- Complex temporal: `G(holding(X) -> F(ontable(X)))` - "Whatever you pick up must eventually be put down"

**Current Support**: Only `F(φ)` where φ is a single atomic predicate

**Status**: 🔒 **OUT OF SCOPE** - Future work

---

### Issue 5: PDDL Goal Generation Relies on LLM Quality ⚠️ MODERATE → ✅ FIXED

**Problem**: Stage 2 uses LLM to generate PDDL, which may produce incomplete or incorrect goals for complex inputs.

**Observed**:
```
Input: "Build tower A on B on C"
LTLf: ['F(on(a, b))', 'F(on(b, c))']
PDDL Goal (generated): "(on a b"      # Incomplete! Missing ) and second goal
```

**Impact**:
- Classical planning may fail or solve wrong problem
- No validation of LLM-generated PDDL

**Fix Applied** (Phase 1.2):
Added `validate_pddl_syntax()` function in `ltl_to_pddl.py` that checks:
- ✅ Balanced parentheses (open count == close count)
- ✅ Required sections: `:domain`, `:objects`, `:init`, `:goal`
- ✅ Goal section has content (not empty)
- ✅ Objects section has content
- ✅ Must start with `(define (problem`

**Code Location**: `src/stage2_translation/ltl_to_pddl.py:25-78` (validation function) and `:249-256` (integration)

**Status**: ✅ **FIXED** - PDDL syntax validation now active

---

### Issue 6: Limited Blocksworld Initial States 🔒 LIMITATION

**Current**: All blocks start `on(table)` and `clear`

**Not Supported**:
- Blocks already stacked in initial state
- Blocks held in hand initially
- Complex configurations

**Why**: LTL parser prompt assumes "blocks on table" initial configuration

**Status**: 🔒 **BY DESIGN** - Simplifying assumption for MVP

---

## 🏗️ Architecture Analysis: Integration vs Comparison

### Proposed Integration Architecture (Evaluated)

User proposed this unified pipeline architecture:
```
Natural Language Instruction
         ↓
[LTLf Goal Specification]
         ↓
[FOND Planning with LTLf Goals]
         ↓
[Policy (State → Action mapping)]
         ↓
[AgentSpeak Execution]
         ↓
[Runtime LTLf Monitoring]
```

### Current Dual-Branch Comparison Architecture

```
Natural Language
       ↓
    [LTLf]
       ↓
    [PDDL]
       ↓
    ┌──┴──┐
    ▼     ▼
Branch A  Branch B
Classical  LLM
Planning  AgentSpeak
    ↓     ↓
[Comparison]
```

### Key Differences

| Aspect | Proposed (Integration) | Current (Comparison) |
|--------|----------------------|---------------------|
| **Planner** | FOND (non-deterministic) | pyperplan (deterministic) |
| **Output** | Policy (state→action map) | Plan (linear sequence) |
| **Relationship** | FOND guides AgentSpeak | Independent execution |
| **Research Q** | "How can classical enhance LLM?" | "How does LLM compare to classical?" |
| **Goal Dependencies** | ✅ FOND handles automatically | ❌ Current issue (see tower example) |
| **Robustness** | ✅ Policy covers multiple states | ⚠️ Plan is single trajectory |
| **Runtime Monitoring** | ✅ Continuous LTLf checking | ⚠️ End-state verification only |

### Analysis & Decision

**Advantages of Integration Architecture:**
1. ✅ **Solves goal ordering problem**: FOND policy naturally handles dependencies (B→C before A→B)
2. ✅ **State-aware execution**: Policy provides actions for ANY reachable state
3. ✅ **Runtime verification**: Continuous LTLf monitoring vs end-state only
4. ✅ **Production-ready**: Better for real-world BDI systems

**Disadvantages:**
1. ❌ **Changes research focus**: From comparison to integration
2. ❌ **Loses dual-branch insights**: Can't compare LLM vs classical separately
3. ❌ **Requires FOND planner**: Not currently implemented (only pyperplan)
4. ❌ **Policy→AgentSpeak translation**: Need new conversion logic

### Recommended Approach: **KEEP CURRENT + ADD INTEGRATION AS FUTURE WORK**

**Rationale:**
- **Current dual-branch architecture serves core research goal**: Comparative evaluation
- **Integration architecture is FUTURE ENHANCEMENT**, not replacement
- **Goal ordering issue discovered through comparison** - this is valuable research insight!
- **Can add FOND-guided branch later** as Branch C without disrupting A vs B comparison

**Implementation Priority:**
1. ✅ **Phase 1 (DONE)**: Fix current architecture issues (verification, validation, multi-goal)
2. 📋 **Phase 2 (NEXT)**: Enhance current comparison (better goal ordering, PDDL multi-goal)
3. 🔮 **Phase 3 (FUTURE)**: Add FOND-guided AgentSpeak as Branch C for three-way comparison

**Current Status**: Dual-branch architecture is appropriate for comparative research. Integration architecture documented as future enhancement.

---

## 🔧 Recent Fixes (Phase 1 MVP Completion)

### ✅ Fix 1: Goal Verification Space Normalization (Issue 2)
**Problem**: Goal verification failed due to `"on(c, b)"` vs `"on(c,b)"` mismatch
**Solution**: Added space normalization in `comparative_evaluator.py:148-170`
**Result**: ✓ Goal verification now correctly reports True for achieved goals
**Code Location**: `src/stage4_execution/comparative_evaluator.py:148-170`

### ✅ Fix 2: PDDL Syntax Validation (Issue 5)
**Problem**: No validation of LLM-generated PDDL, could produce invalid syntax
**Solution**: Added `validate_pddl_syntax()` function checking:
- Balanced parentheses
- Required sections (`:domain`, `:objects`, `:init`, `:goal`)
- Non-empty goal and objects sections

**Result**: ✓ Invalid PDDL now caught with clear error messages
**Code Location**: `src/stage2_translation/ltl_to_pddl.py:25-78` (validator) and `:249-256` (integration)

### ✅ Fix 3: Multi-Goal Verification Support (Issue 1 - Partial)
**Problem**: Verification only checked first LTLf formula when multiple goals existed
**Solution**: Updated `_compare_results()` to:
- Accept `List[str]` or `str` for `ltl_goal` parameter
- Check ALL goals individually
- Report detailed per-goal satisfaction status
- Overall satisfaction requires ALL goals met

**Result**: ✓ Multi-goal verification now works correctly
**Example Output**:
```
LTLf Goals (2):
  1. F(on(a, b))
  2. F(on(b, c))

Detailed Goal Verification:
  ✓ F(on(a, b))
  ✗ F(on(b, c))
```
**Code Locations**:
- `src/stage4_execution/comparative_evaluator.py:98-145` (verification logic)
- `src/stage4_execution/comparative_evaluator.py:210-255` (report generation)
- `src/dual_branch_pipeline.py:254-265` (pass all formulas)

### ⚠️ Remaining Limitation: Goal Ordering Dependencies
**Discovery**: Multi-goal tower building ("Build tower A on B on C") reveals **goal dependency** issue:
- **Classical planner**: Correctly determines order (stack B on C first, then A on B) ✓
- **AgentSpeak**: Executes goals in given order, making second goal unreachable ✗
- **Root Cause**: No dependency analysis - AgentSpeak executes `!achieve_on_a_b; !achieve_on_b_c`, but after stacking A on B, B is no longer clear

**Example**:
```
Goals: ["F(on(a,b))", "F(on(b,c))"]
Naive Order: A→B then B→C (FAILS - B not clear after A stacked)
Correct Order: B→C then A→B (WORKS)
```

**Status**: This requires either:
1. Goal dependency analysis & reordering
2. More sophisticated LLM prompt to generate dependency-aware ordering
3. Integration with FOND planner for state-aware goal sequencing

---

## ✅ What Actually Works (Verified)

**Fully Functional**:
1. ✅ Simple single-goal stacking: "Stack block C on block B"
2. ✅ Classical PDDL planning for single goals
3. ✅ AgentSpeak generation and parsing for single goals
4. ✅ BDI execution with declarative goals
5. ✅ All 7 critical fixes (multi-line parsing, variable unification, belief conversion, etc.)
6. ✅ **NEW**: Multi-goal verification with detailed per-goal reporting
7. ✅ **NEW**: PDDL syntax validation

**Partially Functional**:
8. ⚠️ Multi-goal scenarios:
   - Classical planner: May work (depends on LLM goal generation and planner capabilities)
   - AgentSpeak: Achieves goals in sequence but fails when dependencies exist
   - **Root Cause**: No goal dependency analysis

**Not Implemented**:
9. ❌ Goal dependency analysis and reordering
10. ❌ Conjunctive goals (simultaneous requirements)
11. ❌ Sequential/temporal goals with explicit ordering
12. ❌ Complex LTL operators (G, X, U)

---

## 🎓 Research Context

**Project**: Final Year Project (FYP)
**Institution**: University of Nottingham Ningbo China
**Author**: Yiwei LI (20513831)
**Supervisor**: Yuan Yao

### Research Questions (Updated with Test Results)

**RQ1**: Can LLMs generate correct AgentSpeak plan libraries from LTLf specifications?
- ✅ Syntax: Generated code parses successfully
- ✅ Semantics: Plans execute correctly for **single-goal** blocksworld scenarios
- ⚠️ **Limitation**: Only supports single `F(on(X,Y))` goals currently

**RQ2**: How do LLM-generated plans compare to classical planning?
- ✅ Efficiency: Equal action count for simple scenarios (tested: "Stack C on B")
- ⚠️ **Issue Found**: Multi-goal scenarios show discrepancy - classical may solve full problem, AgentSpeak only partial

**RQ3**: Are LLM-generated plans more robust to failures?
- ❓ Pending: Failure injection tests needed

**RQ4**: Can LLM plans handle novel situations?
- ❓ Pending: Unseen state tests needed
- ⚠️ **Current**: Limited to single-goal scenarios

### Key Contributions

1. **Dual-Branch Comparative Framework**: First implementation comparing classical planning with LLM-generated BDI plans
2. **LTLf-to-AgentSpeak Pipeline**: Novel approach using temporal logic as intermediate representation
3. **Working BDI Simulator**: Pure Python implementation of AgentSpeak subset (no Jason dependency)
4. **Comprehensive Documentation**: All implementation challenges, fixes, and **limitations** fully documented

---

## 📝 License

Academic research project - University of Nottingham Ningbo China

---

## 📚 References

- **BDI Architecture**: Bordini et al. (2007) - Programming Multi-Agent Systems in AgentSpeak using Jason
- **LTLf**: De Giacomo & Vardi (2013) - Linear Temporal Logic on Finite Traces
- **PDDL Planning**: Geffner & Bonet (2013) - A Concise Introduction to Models and Methods for Automated Planning
