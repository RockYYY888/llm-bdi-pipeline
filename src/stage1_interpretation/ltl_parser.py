"""
LTL (Linear Temporal Logic) Parser and Specification Module

This module converts natural language instructions into LTL formulas
using temporal operators: G (globally), F (finally), X (next), U (until)

Example LTL formulas for Blocksworld:
- F(on(a, b))           : Eventually a is on b
- G(clear(a))           : Always a is clear
- F(on(a, b) & clear(a)): Eventually a is on b AND a is clear
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class TemporalOperator(Enum):
    """LTL temporal operators"""
    FINALLY = "F"      # Eventually (◇)
    GLOBALLY = "G"     # Always (□)
    NEXT = "X"         # Next state
    UNTIL = "U"        # Until
    RELEASE = "R"      # Release


class LogicalOperator(Enum):
    """Logical operators"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"


@dataclass
class LTLFormula:
    """Represents an LTL formula"""
    operator: Optional[TemporalOperator]
    predicate: Optional[Dict[str, List[str]]]  # e.g., {"on": ["a", "b"]}
    sub_formulas: List['LTLFormula']
    logical_op: Optional[LogicalOperator]

    def to_string(self) -> str:
        """Convert LTL formula to string representation"""
        if self.predicate and not self.operator:
            # Atomic proposition: on(a, b)
            pred_name = list(self.predicate.keys())[0]
            args = self.predicate[pred_name]
            return f"{pred_name}({', '.join(args)})"

        if self.operator == TemporalOperator.UNTIL and len(self.sub_formulas) == 2:
            # Until operator: holding(a) U clear(b)
            left = self.sub_formulas[0].to_string()
            right = self.sub_formulas[1].to_string()
            return f"({left} U {right})"

        if self.operator and len(self.sub_formulas) == 1:
            # Temporal operator: F(on(a, b)), G(clear(c)), X(holding(a))
            inner = self.sub_formulas[0].to_string()
            return f"{self.operator.value}({inner})"

        if self.logical_op and len(self.sub_formulas) >= 2:
            # Logical combination: on(a, b) & clear(a)
            parts = [f.to_string() for f in self.sub_formulas]
            if self.logical_op == LogicalOperator.AND:
                return f"({' & '.join(parts)})"
            elif self.logical_op == LogicalOperator.OR:
                return f"({' | '.join(parts)})"
            elif self.logical_op == LogicalOperator.NOT:
                return f"¬({parts[0]})"

        return "true"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "operator": self.operator.value if self.operator else None,
            "predicate": self.predicate,
            "logical_op": self.logical_op.value if self.logical_op else None,
            "sub_formulas": [f.to_dict() for f in self.sub_formulas]
        }


class LTLSpecification:
    """
    LTL Specification for BDI goals

    This converts natural language to LTL formulas that express:
    - Safety properties (what should never happen)
    - Liveness properties (what should eventually happen)
    - Fairness properties (what should happen infinitely often)
    """

    def __init__(self):
        self.formulas: List[LTLFormula] = []
        self.objects: List[str] = []
        self.initial_state: List[Dict[str, List[str]]] = []

    def add_formula(self, formula: LTLFormula):
        """Add an LTL formula to the specification"""
        self.formulas.append(formula)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "formulas": [f.to_dict() for f in self.formulas],
            "formulas_string": [f.to_string() for f in self.formulas],
            "objects": self.objects,
            "initial_state": self.initial_state
        }


class NLToLTLParser:
    """
    Converts Natural Language to LTL formulas using LLM

    Uses OpenAI API to understand natural language and generate
    structured LTL specifications for temporal goals.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize parser with optional API key, model, and base URL"""
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"
        self.base_url = base_url
        self.client = None

        if api_key:
            from openai import OpenAI
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)

    def parse(self, nl_instruction: str):
        """
        Parse natural language instruction into LTL specification

        Args:
            nl_instruction: Natural language instruction

        Returns:
            Tuple of (LTLSpecification, prompt_dict, response_text)
            - prompt_dict: {"system": "...", "user": "..."}  or None if mock
            - response_text: raw LLM response or None if mock
        """
        if self.client:
            return self._parse_with_llm(nl_instruction)
        else:
            print("⚠️  No API key configured, using mock parser")
            spec = self._parse_mock(nl_instruction)
            return (spec, None, None)  # No LLM data for mock

    def _parse_with_llm(self, nl_instruction: str):
        """
        Parse using LLM API

        Returns:
            Tuple of (LTLSpecification, prompt_dict, response_text)
        """
        system_prompt = """You are an expert in Linear Temporal Logic (LTL) and BDI agent systems.

Domain: Blocksworld

Objects: blocks (e.g., a, b, c, d)

Predicates:
- on(X, Y): block X is on block Y
- ontable(X): block X is on the table
- clear(X): block X has nothing on top
- holding(X): robot arm is holding block X
- handempty: robot arm is empty

LTL Temporal Operators:
- F (Finally/Eventually): ◇ - the property will be true at some point in the future
- G (Globally/Always): □ - the property is always true throughout execution
- X (Next): ○ - the property is true in the immediate next state
- U (Until): - property P holds until property Q becomes true

Your task: Convert natural language to LTL formulas.

**Examples:**

1. "Put A on B"
   → F(on(a, b)): Eventually a is on b
   → F(clear(a)): Eventually a is clear

2. "Put A on B while keeping C clear"
   → F(on(a, b)): Eventually a is on b
   → G(clear(c)): C is always clear

3. "First pick up A, then immediately place it on B"
   → F(holding(a)): Eventually holding a
   → X(on(a, b)): In the next state, a is on b

4. "Keep holding A until B is clear"
   → holding(a) U clear(b): Hold a until b is clear

**Natural Language Patterns:**
- "always", "throughout", "keep", "maintain" → G (Globally)
- "eventually", "finally", "at some point" → F (Finally)
- "next", "immediately after", "then" → X (Next)
- "until", "while waiting for" → U (Until)

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{
  "objects": ["a", "b"],
  "initial_state": [
    {"ontable": ["a"]},
    {"ontable": ["b"]},
    {"clear": ["a"]},
    {"clear": ["b"]},
    {"handempty": []}
  ],
  "ltl_formulas": [
    {
      "type": "temporal",
      "operator": "F",
      "formula": {"on": ["a", "b"]}
    },
    {
      "type": "temporal",
      "operator": "G",
      "formula": {"clear": ["c"]}
    }
  ]
}

For U (Until) operator, use this format:
{
  "type": "until",
  "operator": "U",
  "left_formula": {"holding": ["a"]},
  "right_formula": {"clear": ["b"]}
}"""

        user_prompt = f"Natural language instruction: {nl_instruction}"

        # Store prompt for logging
        prompt_dict = {
            "system": system_prompt,
            "user": user_prompt
        }

        # Call LLM API with timeout handling
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                timeout=60.0  # 60 seconds timeout for API call
            )
        except Exception as e:
            print(f"⚠️  LLM API call failed: {e}")
            print("⚠️  Falling back to mock parser...")
            spec = self._parse_mock(nl_instruction)
            return (spec, None, None)  # No LLM data on fallback

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        result = json.loads(result_text)

        # Build LTL specification
        spec = LTLSpecification()
        spec.objects = result["objects"]
        spec.initial_state = result["initial_state"]

        # Convert formulas
        for ltl_def in result["ltl_formulas"]:
            if ltl_def["type"] == "temporal":
                operator = TemporalOperator(ltl_def["operator"])
                predicate = ltl_def["formula"]

                # Create atomic formula
                atomic = LTLFormula(
                    operator=None,
                    predicate=predicate,
                    sub_formulas=[],
                    logical_op=None
                )

                # Wrap in temporal operator
                formula = LTLFormula(
                    operator=operator,
                    predicate=None,
                    sub_formulas=[atomic],
                    logical_op=None
                )

                spec.add_formula(formula)

            elif ltl_def["type"] == "until":
                # U (Until) operator has two sub-formulas
                operator = TemporalOperator.UNTIL
                left_predicate = ltl_def["left_formula"]
                right_predicate = ltl_def["right_formula"]

                # Create left atomic formula
                left_atomic = LTLFormula(
                    operator=None,
                    predicate=left_predicate,
                    sub_formulas=[],
                    logical_op=None
                )

                # Create right atomic formula
                right_atomic = LTLFormula(
                    operator=None,
                    predicate=right_predicate,
                    sub_formulas=[],
                    logical_op=None
                )

                # Wrap in Until operator
                formula = LTLFormula(
                    operator=operator,
                    predicate=None,
                    sub_formulas=[left_atomic, right_atomic],
                    logical_op=None
                )

                spec.add_formula(formula)

        return (spec, prompt_dict, result_text)

    def _parse_mock(self, nl_instruction: str) -> LTLSpecification:
        """Mock parser for testing without LLM"""
        spec = LTLSpecification()

        # Example: "Put A on B"
        if "put" in nl_instruction.lower() and "on" in nl_instruction.lower():
            # Extract blocks (simple pattern matching)
            words = nl_instruction.lower().split()
            blocks = [w for w in words if len(w) == 1 and w.isalpha()]

            if len(blocks) >= 2:
                a, b = blocks[0], blocks[1]
                spec.objects = [a, b]

                # Initial state: both on table
                spec.initial_state = [
                    {"ontable": [a]},
                    {"ontable": [b]},
                    {"clear": [a]},
                    {"clear": [b]},
                    {"handempty": []}
                ]

                # LTL Formula 1: F(on(a, b))
                atomic1 = LTLFormula(
                    operator=None,
                    predicate={"on": [a, b]},
                    sub_formulas=[],
                    logical_op=None
                )
                formula1 = LTLFormula(
                    operator=TemporalOperator.FINALLY,
                    predicate=None,
                    sub_formulas=[atomic1],
                    logical_op=None
                )
                spec.add_formula(formula1)

                # LTL Formula 2: F(clear(a))
                atomic2 = LTLFormula(
                    operator=None,
                    predicate={"clear": [a]},
                    sub_formulas=[],
                    logical_op=None
                )
                formula2 = LTLFormula(
                    operator=TemporalOperator.FINALLY,
                    predicate=None,
                    sub_formulas=[atomic2],
                    logical_op=None
                )
                spec.add_formula(formula2)

        return spec


def test_ltl_parser():
    """Test LTL parser"""
    parser = NLToLTLParser()

    # Test case
    instruction = "Put block A on block B"
    spec = parser.parse(instruction)

    print("="*80)
    print("LTL Parser Test")
    print("="*80)
    print(f"Instruction: {instruction}")
    print(f"\nObjects: {spec.objects}")
    print(f"Initial State: {spec.initial_state}")
    print(f"\nLTL Formulas:")
    for i, formula in enumerate(spec.formulas, 1):
        print(f"  {i}. {formula.to_string()}")

    print(f"\nFull Specification:")
    print(json.dumps(spec.to_dict(), indent=2))


if __name__ == "__main__":
    test_ltl_parser()
