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
                 model: str = "deepseek-chat",
                 base_url: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize AgentSpeak generator

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            base_url: Optional custom base URL
            verbose: If True, print prompts and responses to terminal
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model = model
        self.verbose = verbose
        self.last_prompt_dict = None  # Store last prompt for error logging

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
        from .prompts import AGENTSPEAK_SYSTEM_PROMPT

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

        # Print prompts if verbose mode enabled
        if self.verbose:
            print("\n" + "="*80)
            print("STAGE 3: AgentSpeak Generation - LLM API Call")
            print("="*80)
            print(f"Model: {self.model}")
            print(f"Temperature: 0.2")
            print(f"Timeout: 60.0s")
            print("\n" + "-"*80)
            print("SYSTEM PROMPT:")
            print("-"*80)
            print(AGENTSPEAK_SYSTEM_PROMPT)
            print("\n" + "-"*80)
            print("USER PROMPT:")
            print("-"*80)
            print(prompt)
            print("-"*80)
            print("\nCalling LLM API...")
            print("="*80 + "\n")

        # Prepare prompt dict for logging (BEFORE try block so it's available even if error)
        prompt_dict = {
            "system": AGENTSPEAK_SYSTEM_PROMPT,
            "user": prompt
        }

        # Store in instance variable so it's accessible even if exception occurs
        self.last_prompt_dict = prompt_dict

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,  # Lower for more consistent code generation
                timeout=60.0  # 60 seconds timeout for API call (same as Stage 1)
            )

            response_text = response.choices[0].message.content

            # Print response if verbose mode enabled
            if self.verbose:
                print("\n" + "="*80)
                print("LLM RESPONSE RECEIVED")
                print("="*80)
                print(response_text)
                print("="*80 + "\n")

            # Extract .asl code from response
            asl_code = self._extract_asl_code(response_text)

            return asl_code, prompt_dict, response_text

        except Exception as e:
            ltl_formulas = ltl_spec.get('formulas_string', [])

            # Print error if verbose mode enabled
            if self.verbose:
                print("\n" + "="*80)
                print("LLM API CALL FAILED")
                print("="*80)
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Message: {str(e)}")
                print("="*80 + "\n")

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
        from .prompts import get_agentspeak_user_prompt

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

        return get_agentspeak_user_prompt(
            domain_name,
            objects,
            actions_str,
            predicates_str,
            formulas_str,
            dfa_info
        )

    def _format_dfa_information(self, dfa_result: Any) -> str:
        """Format DFA decomposition information for the prompt (using cleaned DFA)"""
        from stage2_dfa_generation.dfa_dot_cleaner import format_dfa_for_prompt

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

            # Use cleaned DFA format (no Graphviz formatting)
            cleaned_dfa = format_dfa_for_prompt(dfa_node.dfa_dot)
            # Indent each line
            indented_dfa = '\n'.join(f"   {line}" for line in cleaned_dfa.split('\n'))
            info += f"\n{indented_dfa}\n"

            if dfa_node.subgoals:
                info += f"\n   Subgoals to achieve: {', '.join(dfa_node.subgoals)}\n"

            info += "\n"

        info += "**IMPORTANT**: Use the transition conditions as plan contexts in AgentSpeak.\n"
        info += "For example, if a transition shows 'when [on_a_b & clear_c]', create a plan:\n"
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
