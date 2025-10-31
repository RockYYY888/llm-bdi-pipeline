# Pipeline Refactoring Plan

## Current Status
- ✅ Directories renamed:
  - `stage2_planning` → `stage3_code_generation`
  - `ltlf_dfa_conversion` → `stage2_dfa_generation`
- ✅ Created `recursive_dfa_builder.py` in stage2

## Remaining Tasks

### 1. Update Stage 3 AgentSpeak Generator
**File**: `src/stage3_code_generation/agentspeak_generator.py`

**Changes needed**:
- Add new parameter: `dfa_result: RecursiveDFAResult`
- Update prompt to include all DFAs with their transitions
- Emphasize transition labels as key information for plan generation
- Update system prompt to explain DFA-guided AgentSpeak generation

### 2. Update Pipeline Logger
**File**: `src/pipeline_logger.py`

**Changes needed**:
- Update `PipelineRecord` dataclass:
  - Rename `stage2_*` fields to `stage3_*` (AgentSpeak generation)
  - Add new `stage2_*` fields for DFA generation
  - Fields: `stage2_dfas`, `stage2_decomposition_tree`, etc.
- Update logging methods:
  - Keep `log_stage1` (NL → LTLf)
  - Create `log_stage2_dfas` (LTLf → DFAs)
  - Rename current `log_stage2` to `log_stage3` (DFAs → AgentSpeak)
- Update `_save_readable_format` to display 3 stages correctly

### 3. Update Main Pipeline
**File**: `src/ltl_bdi_pipeline.py`

**Changes needed**:
- Update imports (stage2_planning → stage3_code_generation)
- Add Stage 2: DFA generation between Stage 1 and Stage 3
- Pass DFA result to Stage 3 AgentSpeak generator
- Update comments and docstrings

### 4. Update README.md

**Sections to update**:
- System Architecture diagram (show 3 stages clearly)
- Pipeline Stages section
- Project Structure (directory names)
- Examples (show DFA generation step)

### 5. Create __init__.py files
- `src/stage2_dfa_generation/__init__.py`
- `src/stage3_code_generation/__init__.py`

## New Architecture

```
Stage 1: NL → LTLf
  Input: "Stack block A on block B"
  Output: F(on(a,b)), objects=[a,b], initial_state=[...]

Stage 2: LTLf → Recursive DFAs
  Input: LTLf specification
  Process:
    1. Generate DFA for root goal F(on(a,b))
    2. Analyze transitions to identify subgoals
    3. For each subgoal:
       - If physical action → terminal
       - If cached → reuse
       - Otherwise → recursive DFA generation
  Output: RecursiveDFAResult with all DFAs

Stage 3: DFAs → AgentSpeak Code
  Input: All DFAs with transition labels
  Process: LLM generates AgentSpeak plans guided by DFA transitions
  Output: Complete .asl file
```

## Testing Plan

1. Test recursive_dfa_builder.py standalone
2. Test updated pipeline with simple case
3. Test with complex nested goals
4. Verify logging captures all 3 stages
5. Check README accuracy

## Next Session Tasks

When continuing, start with:
1. Update Stage 3 AgentSpeak generator to accept DFAs
2. Update pipeline logger for 3-stage structure
3. Update main pipeline orchestrator
4. Test end-to-end
5. Update README
