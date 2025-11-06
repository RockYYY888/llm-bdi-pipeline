"""
Stage 3: AgentSpeak Code Generation Prompts

Contains prompt templates for generating AgentSpeak plan libraries
from LTLf specifications and DFA decompositions.
"""

from typing import Optional


# System prompt for AgentSpeak expert
AGENTSPEAK_SYSTEM_PROMPT = """You are an expert AgentSpeak/Jason programmer with deep knowledge of BDI agent architectures.

**AgentSpeak Syntax Essentials**:

1. **Plans**: +triggering_event : context <- body.
   - triggering_event: +!goal (goal added), -!goal (goal failed), +belief (belief added), -belief (belief removed)
   - context: Boolean combination of beliefs (use & for AND, | for OR, not for negation)
   - body: Sequence of actions (use ; to separate), subgoals (!goal), belief updates (+belief, -belief)

2. **Goals**:
   - Achievement: !goal_name (execute plan to achieve)
   - Test: ?belief_name (query belief base)
   - Declarative: !![state] (achieve state, automatically satisfied when state holds)

3. **Actions**:
   - Lowercase: physical/primitive actions (pickup, move, etc.)
   - !goal: post achievement subgoal
   - ?belief: test/query belief
   - +belief: add belief
   - -belief: remove belief
   - .print(...): internal action for printing

4. **Context Operators**:
   - & (AND): multiple conditions must hold
   - | (OR): at least one condition must hold
   - not X or \\+ X: negation
   - Variables: Uppercase (X, Y, Block1)
   - Constants: lowercase (a, b, table)

5. **LTLf Integration**:
   - F(φ): "eventually φ" → generate plan to achieve φ
   - G(φ): "always φ" → monitor and maintain φ throughout execution
   - X(φ): "next φ" → ensure φ in next state
   - φ U ψ: "φ until ψ" → maintain φ until ψ

**Best Practices**:
- Always provide multiple plans for same goal (different contexts)
- Add failure handlers (-!goal) for critical goals
- Update beliefs after actions to maintain consistency
- Use meaningful variable names
- Add brief comments for complex logic
- For state-based goals from F formulas, consider using declarative goals !![state]
- For G formulas, add monitoring plans that check constraints

**Code Quality**:
- Generate syntactically correct AgentSpeak
- Use proper indentation (4 spaces)
- Follow Jason/AgentSpeak conventions
- Ensure plans are complete and executable

Generate ONLY AgentSpeak code. Do not include explanations, markdown, or prose.
Start directly with plan definitions.
"""


def get_agentspeak_user_prompt(
    domain_name: str,
    objects: list,
    actions_str: str,
    predicates_str: str,
    formulas_str: str,
    dfa_info: str = "",
    grounding_map_str: str = ""
) -> str:
    """
    Generate user prompt for AgentSpeak code generation

    Args:
        domain_name: Name of the planning domain (e.g., "blocksworld")
        objects: List of objects in the domain
        actions_str: Comma-separated string of available actions
        predicates_str: Comma-separated string of available predicates
        formulas_str: Multi-line string of LTLf formulas (one per line with "  - " prefix)
        dfa_info: Optional DFA decomposition information string
        grounding_map_str: Optional grounding map showing propositional to parameterized conversion

    Returns:
        User prompt string for AgentSpeak generation
    """
    objects_str = ', '.join(objects) if objects else 'none specified'

    return f"""Generate a complete AgentSpeak plan library for a BDI agent guided by DFA decomposition.

**CRITICAL REQUIREMENT**: The generated plans MUST be GENERAL and work from ANY initial configuration.
Do NOT assume any specific initial state. Plans must handle all possible scenarios through context-sensitive conditions.

**Domain**: {domain_name}

**Objects**: {objects_str}

**Available Actions**: {actions_str}

**Available Predicates**: {predicates_str}

**LTLf Temporal Goals**:
{formulas_str}

{grounding_map_str}

{dfa_info}

**CRITICAL: Use Parameterized Predicates, NOT Propositional Symbols**
- The DFAs use flattened propositional symbols (e.g., on_b1_b2) for internal processing
- Your AgentSpeak code MUST use the original parameterized predicates (e.g., on(b1, b2))
- Use the Grounding Map above to convert propositional symbols back to parameterized form
- Goal names should directly use PDDL action names or predicate names (NO "achieve_" prefix)
- Examples:
  - Propositional: on_b1_b2 → Parameterized: on(b1, b2)
  - Propositional: clear_b1 → Parameterized: clear(b1)
  - Goal: achieve_on_b1_b2 → CORRECT: on(b1, b2) or stack(b1, b2)

**Requirements**:
1. Use the DFA transitions as guidance for plan generation - transitions show which conditions/actions lead to goal achievement
2. Generate plans for each subgoal identified in the DFA decomposition
3. Follow the DFA structure: each transition label indicates when to trigger specific plans
4. Include multiple context-sensitive plans based on different DFA transitions
5. Add failure handling plans (-!goal) for robustness
6. Use PDDL action names or predicate names as goal names (NO "achieve_" prefix)
7. Add brief comments explaining the DFA-guided logic
8. Include belief updates after actions using parameterized predicates
9. Use declarative goals (!![state]) where appropriate for state-based goals
10. ALWAYS use parameterized form: on(X, Y), clear(X), NOT on_x_y, clear_x

**Example Plan Structure** (for reference):
```agentspeak
// Main goal from LTLf F(on(a, b)) - use parameterized form
+!on(A, B) : true <-
    !clear(A);
    !clear(B);
    !stack(A, B).

// Subgoal with context adaptation - parameterized predicates
+!stack(X, Y) : holding(X) & clear(Y) <-
    put_on_block(X, Y);
    +on(X, Y);
    -holding(X);
    +clear(X);
    -clear(Y).

+!stack(X, Y) : on(X, Z) & clear(X) & handempty <-
    pick_up(X, Z);
    -on(X, Z);
    +holding(X);
    !stack(X, Y).

// Failure handling
-!stack(X, Y) : true <-
    .print("Stack ", X, " on ", Y, " failed");
    !alternative_approach.

// G constraint monitoring (if any G formulas exist)
+on(X, Y) : violates_constraint(X, Y) <-
    .print("Constraint violated!");
    !recovery_action.
```

Now generate the COMPLETE AgentSpeak plan library for the {domain_name} domain.
Output ONLY the AgentSpeak code, no explanations before or after.
"""
