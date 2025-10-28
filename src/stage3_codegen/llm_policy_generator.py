"""
Stage 3A: LLM Policy Generator (Branch A)

Replaces classical PDDL planning with LLM-generated policy/plan.
Generates action sequences directly from LTLf goals using LLM reasoning.
"""

from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI


class LLMPolicyGenerator:
    """
    Generate execution policy/plan using LLM reasoning

    This replaces classical PDDL planning with LLM-based planning,
    allowing direct generation from LTLf goals without PDDL intermediate step.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 base_url: Optional[str] = None):
        """
        Initialize LLM Policy Generator

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            base_url: Optional custom base URL
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model = model

    def generate_plan(self,
                      ltl_spec: Dict[str, Any],
                      domain_name: str,
                      domain_actions: List[str],
                      domain_predicates: List[str]) -> Tuple[List[Tuple[str, List[str]]], Dict, str]:
        """
        Generate action plan from LTLf specification using LLM

        Args:
            ltl_spec: LTL specification dictionary with:
                - formulas_string: List of LTLf goals
                - initial_state: Initial state predicates
                - objects: Domain objects
            domain_name: Domain name (e.g., "blocksworld")
            domain_actions: Available actions with their effects
            domain_predicates: Available predicates for state description

        Returns:
            Tuple of:
            - plan: List of (action_name, [param1, param2, ...])
            - prompt_dict: {"system": "...", "user": "..."}
            - response_text: Raw LLM response

        Raises:
            RuntimeError: If no API key or LLM fails
        """
        if not self.client:
            raise RuntimeError(
                "No API key configured. Please set OPENAI_API_KEY in .env file.\n"
                "LLM policy generation requires API access."
            )

        # Build prompt
        system_prompt = self._build_system_prompt(domain_name, domain_actions, domain_predicates)
        user_prompt = self._build_user_prompt(ltl_spec)

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0  # Deterministic planning
            )

            response_text = response.choices[0].message.content

            # Parse plan from response
            plan = self._parse_plan(response_text)

            prompt_dict = {
                "system": system_prompt,
                "user": user_prompt
            }

            return plan, prompt_dict, response_text

        except Exception as e:
            formulas = ltl_spec.get('formulas_string', [])
            raise RuntimeError(
                f"LLM policy generation failed: {type(e).__name__}: {str(e)}\n"
                f"Domain: {domain_name}\n"
                f"LTLf goals: {formulas}\n"
                f"Model: {self.model}"
            ) from e

    def _build_system_prompt(self, domain: str, actions: List[str], predicates: List[str]) -> str:
        """Build system prompt for LLM planner"""

        # Domain-specific action descriptions
        action_descriptions = {
            "blocksworld": {
                "pickup": "pickup(X): Pick up block X from table. Preconditions: on(X,table), clear(X), handempty. Effects: holding(X), not on(X,table), not handempty, not clear(X)",
                "putdown": "putdown(X): Put block X on table. Preconditions: holding(X). Effects: on(X,table), clear(X), handempty, not holding(X)",
                "stack": "stack(X,Y): Stack block X on block Y. Preconditions: holding(X), clear(Y). Effects: on(X,Y), clear(X), handempty, not holding(X), not clear(Y)",
                "unstack": "unstack(X,Y): Unstack block X from block Y. Preconditions: on(X,Y), clear(X), handempty. Effects: holding(X), clear(Y), not on(X,Y), not handempty, not clear(X)"
            }
        }

        action_details = action_descriptions.get(domain, {})
        action_list = "\n".join([action_details.get(a, f"{a}: [description needed]") for a in actions])

        return f"""You are an expert AI planner for {domain} domain.

Your task: Generate an optimal action sequence to achieve LTLf temporal goals.

**Available Actions**:
{action_list}

**Available Predicates**: {', '.join(predicates)}

**Planning Rules**:
1. Analyze goal dependencies - determine correct ordering
2. For goals like ["F(on(a,b))", "F(on(b,c))"]:
   - Check if achieving one goal blocks another
   - If on(a,b) is achieved first, block b is no longer clear
   - Therefore: achieve on(b,c) FIRST, then on(a,b)
3. Generate shortest valid action sequence
4. Every action must have valid preconditions
5. Track state changes after each action

**Output Format**:
Return ONLY a numbered list of actions, one per line:
1. action_name(param1, param2, ...)
2. action_name(param1, param2, ...)

No explanations, no markdown, just the numbered action list."""

    def _build_user_prompt(self, ltl_spec: Dict[str, Any]) -> str:
        """Build user prompt with specific planning problem"""

        objects = ltl_spec.get('objects', [])
        initial_state = ltl_spec.get('initial_state', [])
        formulas = ltl_spec.get('formulas_string', [])

        # Format initial state
        init_beliefs = []
        for pred_dict in initial_state:
            for pred_name, args in pred_dict.items():
                if args:
                    for arg in args:
                        init_beliefs.append(f"{pred_name}({arg})")
                else:
                    init_beliefs.append(pred_name)

        init_str = ", ".join(init_beliefs) if init_beliefs else "empty"

        # Format goals
        goals_str = "\n".join([f"  - {f}" for f in formulas])

        return f"""Generate an action plan for this problem:

**Objects**: {', '.join(objects)}

**Initial State**: {init_str}

**LTLf Goals** (MUST achieve ALL):
{goals_str}

**Critical**: Analyze goal dependencies! If achieving goal G1 makes goal G2 unreachable, achieve G2 FIRST.

Generate the optimal action sequence:"""

    def _parse_plan(self, response_text: str) -> List[Tuple[str, List[str]]]:
        """
        Parse action plan from LLM response

        Expected format:
        1. pickup(a)
        2. stack(a, b)
        """
        import re

        plan = []
        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Match: "1. action(param1, param2)"
            match = re.match(r'\d+\.\s*(\w+)\((.*?)\)', line)
            if match:
                action_name = match.group(1)
                params_str = match.group(2)

                # Parse parameters
                params = [p.strip() for p in params_str.split(',')] if params_str else []

                plan.append((action_name, params))

        if not plan:
            raise ValueError(f"Failed to parse plan from LLM response:\n{response_text}")

        return plan


def test_llm_policy_generator():
    """Test LLM policy generator"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import get_config

    print("="*80)
    print("LLM Policy Generator Test")
    print("="*80)

    # Get config
    config = get_config()
    if not config.validate():
        print("ERROR: No API key configured")
        return

    # Create generator
    generator = LLMPolicyGenerator(
        api_key=config.openai_api_key,
        model=config.openai_model,
        base_url=config.openai_base_url
    )

    # Test case: Tower building
    ltl_spec = {
        "objects": ["a", "b", "c"],
        "initial_state": [
            {"ontable": ["a"]},
            {"ontable": ["b"]},
            {"ontable": ["c"]},
            {"clear": ["a"]},
            {"clear": ["b"]},
            {"clear": ["c"]},
            {"handempty": []}
        ],
        "formulas_string": ["F(on(a, b))", "F(on(b, c))"]
    }

    domain_actions = ["pickup", "putdown", "stack", "unstack"]
    domain_predicates = ["on(X,Y)", "clear(X)", "holding(X)", "handempty"]

    print("\nTest Problem:")
    print(f"  Objects: {ltl_spec['objects']}")
    print(f"  Goals: {ltl_spec['formulas_string']}")
    print(f"  Initial: All blocks on table, all clear")

    # Generate plan
    try:
        plan, prompt_dict, response = generator.generate_plan(
            ltl_spec,
            "blocksworld",
            domain_actions,
            domain_predicates
        )

        print("\n" + "="*80)
        print("Generated Plan:")
        print("="*80)
        for i, (action, params) in enumerate(plan, 1):
            print(f"  {i}. {action}({', '.join(params)})")

        print(f"\nTotal actions: {len(plan)}")

    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    test_llm_policy_generator()
