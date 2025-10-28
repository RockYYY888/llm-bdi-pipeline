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
                 domain_predicates: list) -> Tuple[str, Dict, str]:
        """
        Generate complete AgentSpeak plan library

        Args:
            ltl_spec: LTL specification dictionary
            domain_name: Domain name (e.g., "blocksworld")
            domain_actions: List of available actions
            domain_predicates: List of available predicates

        Returns:
            Tuple of (asl_code, prompt_dict, response_text)
        """
        if not self.client:
            raise RuntimeError(
                "No API key configured. Please set OPENAI_API_KEY in .env file.\n"
                "AgentSpeak generation requires LLM."
            )

        # Build comprehensive prompt
        prompt = self._build_prompt(ltl_spec, domain_name, domain_actions, domain_predicates)

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
                      predicates: list) -> str:
        """Build comprehensive prompt for AgentSpeak generation"""

        objects = ltl_spec.get('objects', [])
        initial_state = ltl_spec.get('initial_state', [])
        formulas = ltl_spec.get('formulas_string', [])

        # Format formulas nicely
        formulas_str = "\n".join([f"  - {f}" for f in formulas])

        # Format initial state
        init_str = "\n".join([f"  - {s}" for s in initial_state])

        # Format actions
        actions_str = ", ".join(actions)

        # Format predicates
        predicates_str = ", ".join(predicates)

        prompt = f"""Generate a complete AgentSpeak plan library for a BDI agent.

**Domain**: {domain_name}

**Objects**: {', '.join(objects) if objects else 'none specified'}

**Available Actions**: {actions_str}

**Available Predicates**: {predicates_str}

**Initial State**:
{init_str}

**LTLf Temporal Goals**:
{formulas_str}

**Requirements**:
1. Generate plans to achieve ALL LTLf F (eventually) goals
2. Generate monitoring for ALL LTLf G (always) constraints
3. Include multiple context-sensitive plans per goal (handle different situations)
4. Add failure handling plans (-!goal) for robustness
5. Use clear, meaningful plan names
6. Add brief comments explaining complex logic
7. Include belief updates after actions
8. Use declarative goals (!![state]) where appropriate for state-based goals

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
