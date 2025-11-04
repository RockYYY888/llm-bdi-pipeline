"""
LTLf Generator Module

This module converts natural language instructions into LTLf formulas using LLM.

**LTLf Syntax Reference**: http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax
"""

from typing import Optional
import json

from .ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator, LogicalOperator
from .grounding_map import GroundingMap, create_propositional_symbol


class NLToLTLfGenerator:
    """
    Converts Natural Language to LTLf formulas using LLM

    Uses OpenAI API to understand natural language and generate
    structured LTLf specifications for temporal goals.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 domain_file: Optional[str] = None):
        """
        Initialize generator with optional API key, model, base URL, and domain file

        Args:
            api_key: OpenAI API key
            model: Model name
            base_url: Custom API base URL
            domain_file: Path to PDDL domain file (for dynamic prompt construction)
        """
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"
        self.base_url = base_url
        self.domain_file = domain_file
        self.client = None

        # Parse domain if provided
        self.domain = None
        if domain_file:
            from .pddl_parser import PDDLParser
            self.domain = PDDLParser.parse_domain(domain_file)

        if api_key:
            from openai import OpenAI
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)

    def generate(self, nl_instruction: str):
        """
        Generate LTLf specification from natural language instruction

        Args:
            nl_instruction: Natural language instruction

        Returns:
            Tuple of (LTLSpecification, prompt_dict, response_text)
            - prompt_dict: {"system": "...", "user": "..."}
            - response_text: raw LLM response

        Raises:
            RuntimeError: If no API key is configured
        """
        if not self.client:
            raise RuntimeError(
                "No API key configured. Please set OPENAI_API_KEY in .env file.\n"
                "Copy .env.example to .env and add your API key."
            )

        return self._generate_with_llm(nl_instruction)

    def _generate_with_llm(self, nl_instruction: str):
        """
        Generate using LLM API

        Returns:
            Tuple of (LTLSpecification, prompt_dict, response_text)
        """
        from .prompts import get_ltl_system_prompt, get_ltl_user_prompt

        # Build system prompt dynamically from domain
        if self.domain:
            domain_name = self.domain.name
            types_str = ', '.join(self.domain.types) if self.domain.types else 'objects'
            predicates_str = '\n'.join([f"- {pred.to_signature()}" for pred in self.domain.predicates])
            actions_str = '\n'.join([f"- {action.to_description()}" for action in self.domain.actions])
        else:
            # Fallback to hardcoded blocksworld
            domain_name = "Blocksworld"
            types_str = "blocks (e.g., a, b, c, d)"
            predicates_str = """- on(X, Y): block X is on block Y
- ontable(X): block X is on the table
- clear(X): block X has nothing on top
- holding(X): robot arm is holding block X
- handempty: robot arm is empty"""
            actions_str = """- pick-up(?b1, ?b2)
    Pre: hand empty, ?b1 is clear and on ?b2
    Eff: holding ?b1 (or fails and ?b1 becomes ontable)
- put-down(?b)
    Pre: holding ?b
    Eff: ?b is on table, hand empty"""

        system_prompt = get_ltl_system_prompt(domain_name, types_str, predicates_str, actions_str)
        user_prompt = get_ltl_user_prompt(nl_instruction)

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
            raise RuntimeError(
                f"LLM API call failed: {type(e).__name__}: {str(e)}\n"
                f"Model: {self.model}\n"
                f"Instruction: {nl_instruction[:100]}...\n"
                f"Please check your API key, network connection, and model availability."
            ) from e

        result_text = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if result_text.startswith("```"):
            first_newline = result_text.find('\n')
            if first_newline != -1:
                closing_fence = result_text.rfind('```')
                if closing_fence != -1 and closing_fence > first_newline:
                    result_text = result_text[first_newline+1:closing_fence].strip()

        # Parse JSON response
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse LLM response as JSON: {str(e)}\n"
                f"Response preview: {result_text[:200]}...\n"
                f"The LLM did not return valid JSON. Please try again or check the prompt."
            ) from e

        # Build LTL specification
        spec = LTLSpecification()
        spec.objects = result["objects"]
        spec.initial_state = []

        # Helper function to recursively build formula structures
        def build_formula_recursive(formula_def):
            """Recursively build LTLFormula from nested JSON structure"""
            if not isinstance(formula_def, dict):
                return None

            # Check if this is a nested formula type
            if "type" in formula_def:
                formula_type = formula_def["type"]

                if formula_type == "negation":
                    neg_formula = build_formula_recursive(formula_def["formula"])
                    if neg_formula is None:
                        # It's an atomic predicate
                        neg_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    return LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=[neg_formula],
                        logical_op=LogicalOperator.NOT
                    )

                elif formula_type == "conjunction":
                    conjuncts = []
                    for f_def in formula_def["formulas"]:
                        sub_formula = build_formula_recursive(f_def)
                        if sub_formula is None:
                            # It's an atomic predicate
                            sub_formula = LTLFormula(
                                operator=None,
                                predicate=f_def,
                                sub_formulas=[],
                                logical_op=None
                            )
                        conjuncts.append(sub_formula)
                    return LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=conjuncts,
                        logical_op=LogicalOperator.AND
                    )

                elif formula_type == "disjunction":
                    disjuncts = []
                    for f_def in formula_def["formulas"]:
                        sub_formula = build_formula_recursive(f_def)
                        if sub_formula is None:
                            # It's an atomic predicate
                            sub_formula = LTLFormula(
                                operator=None,
                                predicate=f_def,
                                sub_formulas=[],
                                logical_op=None
                            )
                        disjuncts.append(sub_formula)
                    return LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=disjuncts,
                        logical_op=LogicalOperator.OR
                    )

                elif formula_type == "implication":
                    left_formula = build_formula_recursive(formula_def["left_formula"])
                    right_formula = build_formula_recursive(formula_def["right_formula"])
                    if left_formula is None:
                        left_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["left_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    if right_formula is None:
                        right_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["right_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    return LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=[left_formula, right_formula],
                        logical_op=LogicalOperator.IMPLIES
                    )

                elif formula_type == "equivalence":
                    left_formula = build_formula_recursive(formula_def["left_formula"])
                    right_formula = build_formula_recursive(formula_def["right_formula"])
                    if left_formula is None:
                        left_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["left_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    if right_formula is None:
                        right_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["right_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    return LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=[left_formula, right_formula],
                        logical_op=LogicalOperator.EQUIVALENCE
                    )

                elif formula_type == "temporal":
                    # Nested temporal (e.g., F inside implication)
                    operator = TemporalOperator(formula_def["operator"])
                    inner = build_formula_recursive(formula_def["formula"])
                    if inner is None:
                        inner = LTLFormula(
                            operator=None,
                            predicate=formula_def["formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    return LTLFormula(
                        operator=operator,
                        predicate=None,
                        sub_formulas=[inner],
                        logical_op=None
                    )

            # Not a special type - return None to indicate it's an atomic predicate
            return None

        # Convert formulas
        for ltl_def in result["ltl_formulas"]:
            if ltl_def["type"] == "temporal":
                operator = TemporalOperator(ltl_def["operator"])
                inner_formula_def = ltl_def["formula"]

                # Use recursive builder
                atomic = build_formula_recursive(inner_formula_def)
                if atomic is None:
                    # It's an atomic predicate
                    atomic = LTLFormula(
                        operator=None,
                        predicate=inner_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                formula = LTLFormula(
                    operator=operator,
                    predicate=None,
                    sub_formulas=[atomic],
                    logical_op=None
                )

                spec.add_formula(formula)

            elif ltl_def["type"] == "until":
                operator = TemporalOperator.UNTIL
                left_formula_def = ltl_def["left_formula"]
                right_formula_def = ltl_def["right_formula"]

                # Use recursive builder
                left_formula = build_formula_recursive(left_formula_def)
                if left_formula is None:
                    left_formula = LTLFormula(
                        operator=None,
                        predicate=left_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                right_formula = build_formula_recursive(right_formula_def)
                if right_formula is None:
                    right_formula = LTLFormula(
                        operator=None,
                        predicate=right_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                formula = LTLFormula(
                    operator=operator,
                    predicate=None,
                    sub_formulas=[left_formula, right_formula],
                    logical_op=None
                )

                spec.add_formula(formula)

            elif ltl_def["type"] == "release":
                operator = TemporalOperator.RELEASE
                left_formula_def = ltl_def["left_formula"]
                right_formula_def = ltl_def["right_formula"]

                # Use recursive builder
                left_formula = build_formula_recursive(left_formula_def)
                if left_formula is None:
                    left_formula = LTLFormula(
                        operator=None,
                        predicate=left_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                right_formula = build_formula_recursive(right_formula_def)
                if right_formula is None:
                    right_formula = LTLFormula(
                        operator=None,
                        predicate=right_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                formula = LTLFormula(
                    operator=operator,
                    predicate=None,
                    sub_formulas=[left_formula, right_formula],
                    logical_op=None
                )

                spec.add_formula(formula)

            elif ltl_def["type"] == "nested":
                outer_op = TemporalOperator(ltl_def["outer_operator"])
                inner_op = TemporalOperator(ltl_def["inner_operator"])
                inner_formula_def = ltl_def["formula"]

                # Use recursive builder for the innermost formula
                atomic = build_formula_recursive(inner_formula_def)
                if atomic is None:
                    atomic = LTLFormula(
                        operator=None,
                        predicate=inner_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                inner_formula = LTLFormula(
                    operator=inner_op,
                    predicate=None,
                    sub_formulas=[atomic],
                    logical_op=None
                )

                outer_formula = LTLFormula(
                    operator=outer_op,
                    predicate=None,
                    sub_formulas=[inner_formula],
                    logical_op=None
                )

                spec.add_formula(outer_formula)

        # Create grounding map - use LLM-provided atoms if available, otherwise extract
        if "atoms" in result and result["atoms"]:
            spec.grounding_map = self._create_grounding_map_from_atoms(result["atoms"], spec)
        else:
            spec.grounding_map = self._create_grounding_map(spec)

        return (spec, prompt_dict, result_text)

    def _create_grounding_map(self, spec: LTLSpecification) -> GroundingMap:
        """
        Create grounding map from LTL specification

        Extracts all predicates from formulas and creates propositional symbols
        with their mappings back to original predicates and arguments.
        """
        gmap = GroundingMap()

        # Helper to recursively extract predicates from formula
        def extract_predicates(formula: LTLFormula):
            """Recursively extract all predicates from a formula"""
            if formula.predicate and isinstance(formula.predicate, dict):
                # Check if this is a real atomic predicate or a nested formula structure
                # Real predicates have predicate_name: [args] format
                # Nested structures have "type", "formulas", "left_formula", etc.
                special_keys = {"type", "formulas", "left_formula", "right_formula", "formula", "operator"}

                is_atomic_predicate = True
                for key in formula.predicate.keys():
                    if key in special_keys:
                        is_atomic_predicate = False
                        break

                if is_atomic_predicate:
                    # This is an atomic predicate like {"on": ["a", "b"]}
                    for pred_name, args in formula.predicate.items():
                        if isinstance(args, list):  # Ensure args is a list
                            # Create propositional symbol
                            symbol = create_propositional_symbol(pred_name, args)
                            # Add to grounding map
                            gmap.add_atom(symbol, pred_name, args)

            # Recursively process sub-formulas
            for sub_formula in formula.sub_formulas:
                extract_predicates(sub_formula)

        # Extract predicates from all formulas
        for formula in spec.formulas:
            extract_predicates(formula)

        return gmap

    def _create_grounding_map_from_atoms(self, atoms_list: list, spec: LTLSpecification) -> GroundingMap:
        """
        Create grounding map from LLM-provided atoms list

        Also validates that all atoms are consistent with formulas.

        Args:
            atoms_list: List of atom dicts from LLM response
            spec: LTL specification for validation

        Returns:
            GroundingMap with validated atoms
        """
        gmap = GroundingMap()

        # Add each atom from LLM response
        for atom_dict in atoms_list:
            symbol = atom_dict.get("symbol", "")
            predicate = atom_dict.get("predicate", "")
            args = atom_dict.get("args", [])

            # Validate symbol naming convention
            expected_symbol = create_propositional_symbol(predicate, args)
            if symbol != expected_symbol:
                print(f"⚠️  Warning: LLM provided symbol '{symbol}' doesn't match convention '{expected_symbol}'")
                # Use the expected symbol for consistency
                symbol = expected_symbol

            gmap.add_atom(symbol, predicate, args)

        # Also extract predicates from formulas to verify completeness
        extracted_gmap = self._create_grounding_map(spec)

        # Check if LLM missed any atoms
        for extracted_symbol, extracted_atom in extracted_gmap.atoms.items():
            if extracted_symbol not in gmap.atoms:
                print(f"⚠️  Warning: LLM missed atom '{extracted_symbol}', adding it")
                gmap.add_atom(
                    extracted_symbol,
                    extracted_atom.predicate,
                    extracted_atom.args
                )

        # Validate with domain if available
        if self.domain:
            domain_predicates = [pred.name for pred in self.domain.predicates]
            for symbol, atom in gmap.atoms.items():
                errors = gmap.validate_atom(symbol, domain_predicates, spec.objects)
                if errors:
                    print(f"⚠️  Validation errors for {symbol}: {errors}")

        return gmap


def test_ltlf_generator():
    """Test LTLf generator"""
    generator = NLToLTLfGenerator()

    instruction = "Put block A on block B"
    spec, _, _ = generator.generate(instruction)

    print("="*80)
    print("LTLf Generator Test")
    print("="*80)
    print(f"Instruction: {instruction}")
    print(f"\nObjects: {spec.objects}")
    print(f"Initial State: {spec.initial_state}")
    print(f"\nLTLf Formulas:")
    for i, formula in enumerate(spec.formulas, 1):
        print(f"  {i}. {formula.to_string()}")

    print(f"\nFull Specification:")
    print(json.dumps(spec.to_dict(), indent=2))


if __name__ == "__main__":
    test_ltlf_generator()
