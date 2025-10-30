# Branch B: FOND Planning (Legacy)

This directory contains the **Branch B (FOND Planning)** implementation that was previously part of the dual-branch pipeline. As of October 2025, the project has shifted focus to **Branch A (LLM AgentSpeak Generation)** only.

## Contents

### Stage 2 Planning Components
- **`stage2_planning/pddl_problem_generator.py`**: Converts LTLf specifications to PDDL problem formulations
- **`stage2_planning/pr2_planner.py`**: Wrapper for PR2/PRP FOND planner (Docker-based)

### Stage 2 Translation
- **`stage2_translation/ltl_to_pddl.py`**: LTLf to PDDL translation utilities

### Domains
- **`domains/blocksworld/domain.pddl`**: FOND-compatible PDDL domain definition for blocksworld

### External Tools
- **`external/pr2/`**: PR2 (Planner for Relevant Policies) Docker setup and binaries

## Why Moved to Legacy?

The project originally compared two approaches:
1. **Branch A**: LLM directly generates AgentSpeak code from LTLf (Baseline)
2. **Branch B**: FOND planner generates policies from PDDL (Research)

The decision was made to **focus exclusively on Branch A** for the following reasons:
- Direct LLM-to-BDI code generation is more aligned with modern LLM capabilities
- Simpler architecture without requiring classical planners
- Easier to extend to multiple domains without PDDL domain engineering
- More suitable for comparing pure LLM-based approaches

## How It Worked

### Branch B Pipeline
```
Natural Language
    ↓
LTLf Specification (Stage 1)
    ↓
PDDL Problem (pddl_problem_generator.py)
    ↓
PR2 FOND Planner (pr2_planner.py)
    ↓
Policy/Plan Output
```

### PR2 FOND Planner
- **PR2** (Planner for Relevant Policies) generates strong cyclic policies
- Handles **non-deterministic** action effects
- Runs via **Docker** for platform independence
- Takes PDDL domain + problem as input
- Outputs action sequences or policies

## Restoring Branch B

If you need to restore the FOND planning branch:

1. **Copy files back to original locations**:
   ```bash
   cp -r src/legacy/fond/stage2_planning/*.py src/stage2_planning/
   cp -r src/legacy/fond/stage2_translation src/
   cp -r src/legacy/fond/domains/blocksworld/*.pddl domains/blocksworld/
   ```

2. **Restore PR2 Docker setup**:
   ```bash
   mkdir -p external
   cp -r src/legacy/fond/external/pr2 external/
   ```

3. **Uncomment FOND mode in `dual_branch_pipeline.py`**

4. **Update README.md** to describe dual-branch architecture

5. **Rebuild PR2 Docker image**:
   ```bash
   cd external/pr2
   docker build -t pr2 .
   ```

## Original Implementation Details

### PDDL Problem Generator
- Converts LTLf temporal goals to PDDL goal conditions
- Handles F (eventually), G (always), U (until), X (next) operators
- Generates proper PDDL syntax with objects, init state, and goals

### PR2 Planner Integration
- Docker-based for reproducibility
- Mounts temporary PDDL files into container
- Parses PR2/PRP output to extract action sequences
- Full logging of planner execution

### Blocksworld FOND Domain
- Supports non-deterministic action effects
- Includes predicates: on, ontable, clear, holding, handempty
- Actions: pick-up-from-table, put-on-block, pick-up-from-block, put-down

## References

- **PR2/PRP**: Muise, C. et al. "Planning Over Multi-Agent Epistemic States" (AAAI 2015)
- **FOND Planning**: Geffner, H. & Bonet, B. "A Concise Introduction to Models and Methods for Automated Planning" (2013)
- **LTLf**: De Giacomo, G. & Vardi, M. "Linear Temporal Logic and Linear Dynamic Logic on Finite Traces" (IJCAI 2013)

---

**Note**: This code is preserved for reference and potential future research directions. The main pipeline now focuses on LLM AgentSpeak generation (Branch A) only.
