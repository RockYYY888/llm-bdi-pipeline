"""
LLM-Based Planner

Uses LLM to generate action plans that satisfy PDDL goals and LTL constraints.
This is particularly useful for handling G (Globally) constraints that classical
PDDL planners cannot handle.

Strategy:
- Takes PDDL domain, problem, and LTL specification as input
- Uses LLM to reason about action sequences
- Ensures G (Globally) constraints are maintained throughout execution
- Provides more flexible planning than classical planners
"""

from typing import Optional, List, Tuple, Dict, Any
import json
import re
from pathlib import Path
import sys

# Add src directory to path (only once)
_src_dir = str(Path(__file__).parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.ltlf_formula import LTLSpecification, TemporalOperator


class LLMPlanner:
    """
    LLM-based planner that can handle LTL constraints

    Unlike classical PDDL planners, this can reason about:
    - G (Globally) constraints: properties that must hold throughout execution
    - X (Next) constraints: properties that must hold in next state
    - U (Until) constraints: properties that must hold until another is true
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize LLM planner

        Args:
            api_key: OpenAI API key
            model: Model name (default: deepseek-chat)
            base_url: Optional base URL for API
        """
        self.api_key = api_key
        self.model = model or "deepseek-chat"
        self.base_url = base_url
        self.client = None

        if api_key:
            from openai import OpenAI
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)

    def solve(self,
              domain_file: str,
              problem_file: str,
              ltl_spec: Optional[LTLSpecification] = None):
        """
        Generate plan using LLM

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            ltl_spec: Optional LTL specification with constraints

        Returns:
            Tuple of (plan, prompt_dict, response_text)
            - plan: List of (action_name, parameters) tuples or None
            - prompt_dict: {"system": "...", "user": "..."} or None
            - response_text: raw LLM response or None
        """
        if not self.client:
            print("⚠️  No API key configured, LLM planner unavailable")
            return (None, None, None)

        # Read domain and problem files
        with open(domain_file, 'r') as f:
            domain_content = f.read()

        with open(problem_file, 'r') as f:
            problem_content = f.read()

        # Extract constraints from LTL spec
        constraints_info = ""
        if ltl_spec:
            g_constraints = self._extract_g_constraints(ltl_spec)
            x_constraints = self._extract_x_constraints(ltl_spec)
            u_constraints = self._extract_u_constraints(ltl_spec)
            nested_constraints = self._extract_nested_constraints(ltl_spec)

            if g_constraints:
                constraints_info += "\n**CRITICAL: Globally (G) Constraints - MUST hold at EVERY step:**\n"
                for c in g_constraints:
                    constraints_info += f"  - G({c}): This predicate must remain TRUE throughout the entire plan execution\n"
                constraints_info += "\nViolating any G constraint at any step makes the plan INVALID.\n"

            if x_constraints:
                constraints_info += "\n**Next (X) Constraints - MUST hold in next state:**\n"
                for c in x_constraints:
                    constraints_info += f"  - X({c}): This predicate must be TRUE in the immediate next state\n"

            if u_constraints:
                constraints_info += "\n**Until (U) Constraints:**\n"
                for c in u_constraints:
                    constraints_info += f"  - {c['left']} U {c['right']}: {c['left']} must hold until {c['right']} becomes true\n"

            if nested_constraints:
                constraints_info += "\n**Nested Temporal Constraints:**\n"
                for c in nested_constraints:
                    constraints_info += f"  - {c}: This nested temporal formula must be satisfied\n"
                constraints_info += "\nNested operators require careful reasoning about temporal dependencies.\n"

        system_prompt = f"""You are an expert PDDL planner with deep understanding of Linear Temporal Logic (LTL).

Your task: Generate a valid action sequence that satisfies the PDDL goal AND all LTL constraints.

**PDDL Domain:**
```
{domain_content}
```

**PDDL Problem:**
```
{problem_content}
```
{constraints_info}

**Planning Requirements:**
1. Start from the initial state defined in the problem
2. Apply valid actions from the domain
3. Reach the goal state specified in the problem
4. Maintain all G (Globally) constraints at EVERY step (if any)
5. Satisfy all X (Next) constraints in appropriate states (if any)
6. Satisfy all U (Until) constraints (if any)

**Output Format:**
Return ONLY a JSON array of actions in this exact format:
[
  {{"action": "action_name", "parameters": ["param1", "param2"]}},
  {{"action": "action_name", "parameters": ["param1"]}}
]

**Validation Process:**
1. For each action in your plan, verify it's applicable in current state
2. For each action, verify it doesn't violate G constraints
3. Ensure the final state satisfies all goal predicates

Return ONLY the JSON array (no explanation, no markdown, no code blocks)."""

        user_prompt = "Generate a valid action plan that satisfies the goal and all constraints."

        # Store prompt for logging
        prompt_dict = {
            "system": system_prompt,
            "user": user_prompt
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Deterministic for planning
                timeout=60.0
            )

            response_text = response.choices[0].message.content.strip()

            # Clean up markdown code blocks if present
            clean_text = response_text
            if "```json" in clean_text:
                clean_text = re.sub(r'```json\s*', '', clean_text)
                clean_text = re.sub(r'```\s*', '', clean_text)
            elif "```" in clean_text:
                clean_text = re.sub(r'```\s*', '', clean_text)

            # Parse JSON response
            plan_json = json.loads(clean_text)

            # Convert to internal format: [(action, [params]), ...]
            plan = []
            for step in plan_json:
                action = step["action"]
                params = step.get("parameters", [])
                plan.append((action, params))

            if not plan:
                print("⚠️  LLM returned empty plan")
                return (None, prompt_dict, response_text)

            return (plan, prompt_dict, response_text)

        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response as JSON: {e}")
            print(f"   Response: {response_text[:200]}...")
            return (None, prompt_dict, response_text)
        except Exception as e:
            print(f"⚠️  LLM planning failed: {e}")
            return (None, None, None)

    def _extract_g_constraints(self, ltl_spec: LTLSpecification) -> List[str]:
        """Extract G (Globally) constraints from LTL spec"""
        constraints = []
        for formula in ltl_spec.formulas:
            if formula.operator == TemporalOperator.GLOBALLY:
                if len(formula.sub_formulas) > 0:
                    atomic = formula.sub_formulas[0]
                    if atomic.predicate:
                        constraints.append(atomic.to_string())
        return constraints

    def _extract_x_constraints(self, ltl_spec: LTLSpecification) -> List[str]:
        """Extract X (Next) constraints from LTL spec"""
        constraints = []
        for formula in ltl_spec.formulas:
            if formula.operator == TemporalOperator.NEXT:
                if len(formula.sub_formulas) > 0:
                    atomic = formula.sub_formulas[0]
                    if atomic.predicate:
                        constraints.append(atomic.to_string())
        return constraints

    def _extract_u_constraints(self, ltl_spec: LTLSpecification) -> List[Dict[str, str]]:
        """Extract U (Until) constraints from LTL spec"""
        constraints = []
        for formula in ltl_spec.formulas:
            if formula.operator == TemporalOperator.UNTIL:
                if len(formula.sub_formulas) >= 2:
                    left = formula.sub_formulas[0].to_string()
                    right = formula.sub_formulas[1].to_string()
                    constraints.append({"left": left, "right": right})
        return constraints

    def _extract_nested_constraints(self, ltl_spec: LTLSpecification) -> List[str]:
        """
        Extract nested temporal constraints from LTL spec

        Detects patterns like F(G(φ)), G(F(φ)), etc.
        """
        constraints = []
        for formula in ltl_spec.formulas:
            # Check if this is a nested operator
            # A nested operator has a temporal operator at top level
            # and its sub-formula also has a temporal operator
            if formula.operator and len(formula.sub_formulas) > 0:
                sub_formula = formula.sub_formulas[0]
                # If sub-formula also has a temporal operator, it's nested
                if sub_formula.operator:
                    constraints.append(formula.to_string())
        return constraints


def test_llm_planner():
    """Test LLM planner"""
    from utils.config import get_config

    print("="*80)
    print("LLM Planner Test")
    print("="*80)

    config = get_config()
    if not config.validate():
        print("⚠️  No API key configured, cannot test LLM planner")
        return

    planner = LLMPlanner(
        api_key=config.openai_api_key,
        model=config.openai_model,
        base_url=config.openai_base_url
    )

    domain_file = "domains/blocksworld/domain.pddl"
    problem_file = "output/test_problem.pddl"

    print(f"Domain: {domain_file}")
    print(f"Problem: {problem_file}")
    print("\nGenerating plan with LLM...\n")

    plan, prompt, response = planner.solve(domain_file, problem_file)

    if plan:
        print(f"✓ Plan found: {len(plan)} actions")
        print("\nPlan:")
        for i, (action, params) in enumerate(plan, 1):
            print(f"  {i}. {action}({', '.join(params)})")
    else:
        print("✗ No plan generated")


if __name__ == "__main__":
    test_llm_planner()
