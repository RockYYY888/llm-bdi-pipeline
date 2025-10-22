"""
LTL to PDDL Converter

Uses LLM to intelligently convert LTL specifications to PDDL problem files.

Strategy:
- Use LLM to understand the LTL specification structure
- Generate appropriate PDDL problem representation
- Handle all LTL operators (F, G, X, U) and nested formulas
- Support flexible domain adaptation
- Requires API key - no fallback
"""

from typing import Optional, Dict, Any, List
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1_interpretation.ltl_parser import LTLSpecification


class LTLToPDDLConverter:
    """
    LLM-based LTL to PDDL converter

    Uses language model to convert LTL specifications to PDDL problem files.
    Provides more flexibility and intelligence than template-based conversion.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 domain_name: str = "blocksworld"):
        """
        Initialize LLM-based converter

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini)
            base_url: Optional base URL for API
            domain_name: PDDL domain name
        """
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"
        self.base_url = base_url
        self.domain_name = domain_name
        self.client = None

        if api_key:
            from openai import OpenAI
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)

    def convert(self,
                problem_name: str,
                ltl_spec: LTLSpecification,
                domain_file_path: Optional[str] = None):
        """
        Convert LTL specification to PDDL problem

        Args:
            problem_name: Name for the PDDL problem
            ltl_spec: LTL specification with temporal formulas
            domain_file_path: Optional path to domain file for context

        Returns:
            Tuple of (pddl_problem_string, prompt_dict, response_text)
            - prompt_dict: {"system": "...", "user": "..."}
            - response_text: raw LLM response

        Raises:
            RuntimeError: If no API key is configured
        """
        if not self.client:
            raise RuntimeError(
                "No API key configured. Please set OPENAI_API_KEY in .env file.\n"
                "LTL to PDDL conversion requires LLM for intelligent domain-aware translation."
            )

        return self._convert_with_llm(problem_name, ltl_spec, domain_file_path)

    def _convert_with_llm(self,
                          problem_name: str,
                          ltl_spec: LTLSpecification,
                          domain_file_path: Optional[str] = None):
        """
        Convert using LLM API

        Returns:
            Tuple of (pddl_problem_string, prompt_dict, response_text)
        """

        # Read domain file if provided
        domain_context = ""
        if domain_file_path:
            try:
                with open(domain_file_path, 'r') as f:
                    domain_context = f.read()
            except Exception as e:
                print(f"⚠️  Could not read domain file: {e}")

        system_prompt = f"""You are an expert in PDDL (Planning Domain Definition Language) and LTL (Linear Temporal Logic).

Your task: Convert LTL specifications to PDDL problem files.

PDDL Problem Structure:
```
(define (problem <name>)
  (:domain <domain-name>)
  (:objects <object-list>)
  (:init <initial-state-predicates>)
  (:goal (and <goal-predicates>))
)
```

LTL to PDDL Conversion Rules:
1. F(φ) (Finally/Eventually) → becomes a PDDL goal predicate
   - F(on(a, b)) → (:goal (on a b))

2. G(φ) (Globally/Always) → treated as verification constraint
   - Note: Standard PDDL doesn't support trajectory constraints
   - For G(φ), add φ to goals if it should be maintained
   - Comment constraints that need external verification

3. Atomic predicates → PDDL predicates
   - on(a, b) → (on a b)
   - clear(a) → (clear a)

4. Objects → (:objects ...) section
   - Extract all object names from formulas

5. Initial state → (:init ...) section
   - Convert initial_state predicates to PDDL format

Domain Context:
{domain_context if domain_context else "Domain: " + self.domain_name}

Return ONLY valid PDDL problem text (no markdown, no explanation, no code blocks).
The output should start with "(define (problem" and end with a closing parenthesis."""

        # Prepare LTL spec as JSON for the prompt
        ltl_dict = ltl_spec.to_dict()

        user_prompt = f"""Convert this LTL specification to a PDDL problem:

Problem Name: {problem_name}
Domain: {self.domain_name}

LTL Specification:
{json.dumps(ltl_dict, indent=2)}

Objects: {ltl_dict['objects']}
Initial State: {len(ltl_dict['initial_state'])} predicates
LTL Formulas: {ltl_dict['formulas_string']}

Generate a complete PDDL problem file."""

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
                temperature=0.0,  # Deterministic output
                timeout=30.0
            )

            pddl_text = response.choices[0].message.content.strip()
            response_text = pddl_text  # Store original response

            # Clean up markdown code blocks if present
            if pddl_text.startswith("```"):
                lines = pddl_text.split("\n")
                pddl_text = "\n".join(lines[1:-1])  # Remove first and last lines

            # Validate basic PDDL structure
            if not pddl_text.startswith("(define (problem"):
                raise ValueError("Generated text is not valid PDDL problem format")

            return (pddl_text, prompt_dict, response_text)

        except Exception as e:
            # Enhanced error information for debugging
            ltl_formulas = ltl_spec.to_dict().get('formulas_string', [])
            raise RuntimeError(
                f"LLM PDDL conversion failed: {type(e).__name__}: {str(e)}\n"
                f"Problem: {problem_name}\n"
                f"Domain: {self.domain_name}\n"
                f"LTL formulas: {ltl_formulas}\n"
                f"Model: {self.model}\n"
                f"Please check the LLM response format and PDDL syntax."
            ) from e

    def get_constraints(self, ltl_spec: LTLSpecification) -> List[Dict[str, Any]]:
        """
        Extract G (Globally) constraints for verification

        These constraints need to be verified during plan execution
        as standard PDDL doesn't support trajectory constraints.

        Args:
            ltl_spec: LTL specification

        Returns:
            List of constraints with type and predicate info
        """
        constraints = []
        from stage1_interpretation.ltl_parser import TemporalOperator

        for formula in ltl_spec.formulas:
            if formula.operator == TemporalOperator.GLOBALLY:
                if len(formula.sub_formulas) > 0:
                    atomic = formula.sub_formulas[0]
                    if atomic.predicate:
                        constraints.append({
                            "type": "globally",
                            "predicate": atomic.predicate,
                            "formula_string": formula.to_string()
                        })

        return constraints


def test_converter():
    """Test LTL to PDDL converter"""
    from stage1_interpretation.ltl_parser import NLToLTLParser
    from config import get_config

    print("="*80)
    print("LTL to PDDL Converter Test")
    print("="*80)

    # Get config
    config = get_config()
    api_key = config.openai_api_key if config.validate() else None
    model = config.openai_model
    base_url = config.openai_base_url

    # Parse natural language to LTL
    parser = NLToLTLParser(api_key=api_key, model=model, base_url=base_url)
    instruction = "Put block A on block B"

    print(f"Instruction: {instruction}\n")
    ltl_spec = parser.parse(instruction)

    print("LTL Formulas:")
    for i, formula in enumerate(ltl_spec.formulas, 1):
        print(f"  {i}. {formula.to_string()}")

    # Convert to PDDL (automatically uses LLM or template fallback)
    converter = LTLToPDDLConverter(
        api_key=api_key,
        model=model,
        base_url=base_url,
        domain_name="blocksworld"
    )

    domain_file = "domains/blocksworld/domain.pddl"
    pddl_problem = converter.convert("test_problem", ltl_spec, domain_file)

    print("\n" + "="*80)
    print("Generated PDDL Problem:")
    print("="*80)
    print(pddl_problem)

    # Check for constraints
    constraints = converter.get_constraints(ltl_spec)
    if constraints:
        print("\n" + "="*80)
        print("Constraints for Verification:")
        print("="*80)
        for c in constraints:
            print(f"  - {c['formula_string']}: {c['predicate']}")
    else:
        print("\nNo G (Globally) constraints in this example.")


if __name__ == "__main__":
    test_converter()
