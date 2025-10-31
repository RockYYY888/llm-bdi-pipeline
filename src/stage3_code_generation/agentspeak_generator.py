"""
Stage 3B: AgentSpeak Plan Library Generator

Generates complete AgentSpeak (.asl) plan libraries from LTLf specifications
using LLM-based code generation.
"""

import os
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
import re


class AgentSpeakGenerator:
    """
    Generate AgentSpeak plan libraries from LTLf specifications using LLM
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 base_url: Optional[str] = None):
        """
        Initialize AgentSpeak generator

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            base_url: Optional custom base URL
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model = model

    def generate(self,
                 ltl_spec: Dict[str, Any],
                 domain_name: str,
                 domain_actions: list,
                 domain_predicates: list,
                 dfa_result: Optional[Any] = None) -> Tuple[str, Dict, str]:
        """
        Generate complete AgentSpeak plan library from DFAs

        Args:
            ltl_spec: LTL specification dictionary
            domain_name: Domain name (e.g., "blocksworld")
            domain_actions: List of available actions
            domain_predicates: List of available predicates
            dfa_result: RecursiveDFAResult with all generated DFAs and transitions (NEW)

        Returns:
            Tuple of (asl_code, prompt_dict, response_text)
        """
        if not self.client:
            raise RuntimeError(
                "No API key configured. Please set OPENAI_API_KEY in .env file.\n"
                "AgentSpeak generation requires LLM."
            )

        # Build comprehensive prompt with DFA information
        prompt = self._build_prompt(ltl_spec, domain_name, domain_actions, domain_predicates, dfa_result)

        # Call LLM
        messages = [
            {"role": "system", "content": AGENTSPEAK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2  # Lower for more consistent code generation
            )

            response_text = response.choices[0].message.content

            # Extract .asl code from response
            asl_code = self._extract_asl_code(response_text)

            # Prepare prompt dict for logging
            prompt_dict = {
                "system": AGENTSPEAK_SYSTEM_PROMPT,
                "user": prompt
            }

            return asl_code, prompt_dict, response_text

        except Exception as e:
            ltl_formulas = ltl_spec.get('formulas_string', [])
            raise RuntimeError(
                f"LLM AgentSpeak generation failed: {type(e).__name__}: {str(e)}\n"
                f"Domain: {domain_name}\n"
                f"LTL formulas: {ltl_formulas}\n"
                f"Model: {self.model}\n"
                f"Please check the LLM response format."
            ) from e

    def _build_prompt(self,
                      ltl_spec: Dict[str, Any],
                      domain_name: str,
                      actions: list,
                      predicates: list,
                      dfa_result: Optional[Any] = None) -> str:
        """Build comprehensive prompt for AgentSpeak generation with DFA guidance"""

        objects = ltl_spec.get('objects', [])
        formulas = ltl_spec.get('formulas_string', [])

        # Format formulas nicely
        formulas_str = "\n".join([f"  - {f}" for f in formulas])

        # Format actions
        actions_str = ", ".join(actions)

        # Format predicates
        predicates_str = ", ".join(predicates)

        # NEW: Format DFA information if available
        dfa_info = ""
        if dfa_result and hasattr(dfa_result, 'all_dfas'):
            dfa_info = self._format_dfa_information(dfa_result)

        prompt = f"""Generate a complete AgentSpeak plan library for a BDI agent guided by DFA decomposition.

**CRITICAL REQUIREMENT**: The generated plans MUST be GENERAL and work from ANY initial configuration.
Do NOT assume any specific initial state. Plans must handle all possible scenarios through context-sensitive conditions.

**Domain**: {domain_name}

**Objects**: {', '.join(objects) if objects else 'none specified'}

**Available Actions**: {actions_str}

**Available Predicates**: {predicates_str}

**LTLf Temporal Goals**:
{formulas_str}

{dfa_info}

**Requirements**:
1. Use the DFA transitions as guidance for plan generation - transitions show which conditions/actions lead to goal achievement
2. Generate plans for each subgoal identified in the DFA decomposition
3. Follow the DFA structure: each transition label indicates when to trigger specific plans
4. Include multiple context-sensitive plans based on different DFA transitions
5. Add failure handling plans (-!goal) for robustness
6. Use clear, meaningful plan names matching the DFA structure
7. Add brief comments explaining the DFA-guided logic
8. Include belief updates after actions
9. Use declarative goals (!![state]) where appropriate for state-based goals

**Example Plan Structure** (for reference):
```agentspeak
// Main goal from LTLf F formula
+!achieve_main_goal : true <-
    !subgoal1;
    !subgoal2;
    !verify_success.

// Subgoal with context adaptation
+!subgoal1 : favorable_condition <-
    action1;
    +new_belief;
    -old_belief.

+!subgoal1 : difficult_condition <-
    alternative_action;
    !recovery.

// Failure handling
-!subgoal1 : true <-
    .print("Subgoal failed, trying alternative");
    !alternative_approach.

// G constraint monitoring (if any G formulas exist)
+belief_change : violates_constraint <-
    .print("Constraint violated!");
    !recovery_action.
```

Now generate the COMPLETE AgentSpeak plan library for the {domain_name} domain.
Output ONLY the AgentSpeak code, no explanations before or after.
"""
        return prompt

    def _format_dfa_information(self, dfa_result: Any) -> str:
        """Format DFA decomposition information for the prompt"""

        if not dfa_result or not hasattr(dfa_result, 'all_dfas'):
            return ""

        info = "\n**DFA Decomposition (KEY GUIDANCE FOR PLAN GENERATION)**:\n"
        info += f"Total DFAs generated: {len(dfa_result.all_dfas)}\n"
        info += f"Decomposition depth: {dfa_result.max_depth}\n\n"

        info += "**DFA Structure and Transitions** (use these to guide plan creation):\n\n"

        for i, dfa_node in enumerate(dfa_result.all_dfas, 1):
            info += f"{i}. Goal: `{dfa_node.goal_formula}`\n"
            if dfa_node.is_physical_action:
                info += f"   Type: PHYSICAL ACTION (terminal - direct execution)\n"
            else:
                info += f"   Type: Subgoal (decompose further)\n"

            info += f"   Depth: {dfa_node.depth}\n"

            # Extract key transitions from DFA
            transitions = self._extract_key_transitions(dfa_node.dfa_dot)
            if transitions:
                info += f"   Key Transitions:\n"
                for trans in transitions[:5]:  # Limit to 5 most important
                    info += f"      - {trans}\n"

            if dfa_node.subgoals:
                info += f"   Subgoals to achieve: {', '.join(dfa_node.subgoals)}\n"

            info += "\n"

        info += "**IMPORTANT**: Use the transition labels above as conditions in your AgentSpeak plan contexts.\n"
        info += "For example, if a transition shows `on_a_b & clear_c`, create a plan:\n"
        info += "  +!goal : on(a,b) & clear(c) <- ...\n\n"

        return info

    def _extract_key_transitions(self, dfa_dot: str) -> list:
        """Extract important transitions from DFA DOT format"""
        import re

        transitions = []
        # Parse transitions: "1 -> 2 [label="condition"];"
        pattern = r'(\d+)\s*->\s*(\d+)\s*\[label="([^"]+)"\]'

        for match in re.finditer(pattern, dfa_dot):
            from_state, to_state, label = match.groups()
            if label != "true" and "init" not in from_state:
                transitions.append(f"State {from_state} → {to_state} when [{label}]")

        return transitions

    def _extract_asl_code(self, response: str) -> str:
        """Extract AgentSpeak code from LLM response"""

        # Try to find code block
        code_block_pattern = r'```(?:agentspeak|asl)?\n(.*?)```'
        match = re.search(code_block_pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()

        # If no code block, assume entire response is code
        # Remove common prefixes
        code = response.strip()
        if code.startswith("Here is") or code.startswith("Here's"):
            lines = code.split('\n')
            code = '\n'.join(lines[1:]).strip()

        return code


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


if __name__ == "__main__":
    # Quick test
    from config import get_config

    config = get_config()
    generator = AgentSpeakGenerator(
        api_key=config.openai_api_key,
        model=config.openai_model
    )

    # Test LTL spec
    test_spec = {
        'objects': ['c', 'b', 'a'],
        'initial_state': [
            'on(c, a)',
            'on(a, table)',
            'on(b, table)',
            'clear(c)',
            'clear(b)',
            'handempty'
        ],
        'formulas_string': ['F(on(c, b))']
    }

    test_actions = ['pickup', 'putdown', 'stack', 'unstack']
    test_predicates = ['on(X, Y)', 'clear(X)', 'holding(X)', 'handempty']

    print("Generating AgentSpeak plan library...")
    asl_code, prompt, response = generator.generate(
        test_spec,
        'blocksworld',
        test_actions,
        test_predicates
    )

    print("\n" + "="*80)
    print("GENERATED AGENTSPEAK CODE:")
    print("="*80)
    print(asl_code)
    print("="*80)
