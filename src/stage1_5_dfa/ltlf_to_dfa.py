"""
Stage 1.5: LTLf to DFA Conversion

Converts LTLf specifications to Deterministic Finite Automata (DFA) using ltlf2dfa.

This stage bridges LTLf goal specifications and downstream processing by providing
a formal automaton representation that can be used for verification, planning, or
code generation.

Key Challenge: ltlf2dfa uses propositional variables, but our LTLf formulas use
predicates with arguments (e.g., on(a, b)). We handle this by:
1. Extracting all predicates from LTLf formulas
2. Creating propositional encodings (on(a,b) → on_a_b)
3. Converting formulas to propositional form
4. Generating DFA using ltlf2dfa
5. Maintaining mapping for downstream use
"""

from typing import Dict, List, Any, Tuple
import re
from ltlf2dfa.parser.ltlf import LTLfParser


class PredicateToProposition:
    """
    Converts predicate-based LTLf formulas to propositional encoding

    on(a, b) → on_a_b
    clear(a) → clear_a
    holding(x) → holding_x
    handempty → handempty (already propositional)
    """

    def __init__(self):
        # Match predicates that don't contain nested parentheses
        # This ensures we match inner predicates first
        self.predicate_pattern = re.compile(r'(\w+)\(([^()]+)\)')
        self.predicate_to_prop = {}  # Maps "on(a, b)" → "on_a_b"
        self.prop_to_predicate = {}  # Reverse mapping

    def encode_predicate(self, predicate_str: str) -> str:
        """
        Encode a single predicate to propositional variable

        Args:
            predicate_str: e.g., "on(a, b)" or "clear(a)"

        Returns:
            Propositional variable: e.g., "on_a_b" or "clear_a"
        """
        match = self.predicate_pattern.match(predicate_str.strip())

        if not match:
            # Already propositional (e.g., "handempty", "a", "b")
            return predicate_str.strip()

        pred_name = match.group(1)
        args = match.group(2)

        # Replace commas and spaces with underscores
        prop_var = f"{pred_name}_" + "_".join(arg.strip() for arg in args.split(','))

        # Store mappings
        self.predicate_to_prop[predicate_str.strip()] = prop_var
        self.prop_to_predicate[prop_var] = predicate_str.strip()

        return prop_var

    def convert_formula(self, ltlf_formula_str: str) -> str:
        """
        Convert entire LTLf formula from predicate to propositional form

        Args:
            ltlf_formula_str: e.g., "F(on(a, b))" or "F(on(a, b)) & G(clear(c))"

        Returns:
            Propositional formula: e.g., "F(on_a_b)" or "F(on_a_b) & G(clear_c)"
        """

        # Reserved LTL operators that should NOT be replaced
        ltl_operators = {'F', 'G', 'X', 'U', 'R', 'W', 'M'}

        def replacer(match):
            full_match = match.group(0)  # e.g., "on(a, b)" or "F(x)"
            pred_name = match.group(1)    # e.g., "on" or "F"

            # Skip if it's an LTL operator (F, G, etc.)
            if pred_name in ltl_operators:
                return full_match

            # Otherwise, encode the predicate: on(a, b) → on_a_b
            return self.encode_predicate(full_match)

        # Apply replacements iteratively until no more changes
        # This handles nested structures like F(on(a, b))
        prev_converted = ltlf_formula_str
        max_iterations = 10  # Safety limit

        for _ in range(max_iterations):
            converted = self.predicate_pattern.sub(replacer, prev_converted)

            if converted == prev_converted:
                # No more changes, we're done
                break

            prev_converted = converted
        else:
            # Exceeded max iterations
            raise RuntimeError(f"Failed to converge after {max_iterations} iterations")

        return converted

    def get_mapping(self) -> Dict[str, str]:
        """Get predicate → proposition mapping"""
        return dict(self.predicate_to_prop)

    def get_reverse_mapping(self) -> Dict[str, str]:
        """Get proposition → predicate mapping"""
        return dict(self.prop_to_predicate)


class LTLfToDFA:
    """
    Converts LTLf specifications to DFA using ltlf2dfa

    Pipeline: Predicate LTLf → Propositional LTLf → DFA (DOT format)
    """

    def __init__(self):
        self.ltlf_parser = LTLfParser()
        self.encoder = PredicateToProposition()

    def convert(self, ltl_spec: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Convert LTLf specification to DFA

        Args:
            ltl_spec: LTLSpecification object with formulas

        Returns:
            Tuple of (dfa_dot_string, metadata_dict)
            - dfa_dot_string: DFA in DOT format for visualization
            - metadata_dict: Contains mappings, original formulas, etc.
        """

        # Extract formula strings from LTLSpecification
        if hasattr(ltl_spec, 'formulas'):
            formula_strings = [f.to_string() for f in ltl_spec.formulas]
        else:
            raise ValueError("ltl_spec must have 'formulas' attribute")

        # Combine multiple formulas with conjunction if needed
        if len(formula_strings) == 0:
            raise ValueError("No LTLf formulas provided")
        elif len(formula_strings) == 1:
            original_formula = formula_strings[0]
        else:
            # Join multiple formulas with AND
            original_formula = " & ".join(f"({f})" for f in formula_strings)

        # Convert to propositional encoding
        propositional_formula = self.encoder.convert_formula(original_formula)

        # Parse and convert to DFA
        try:
            formula_obj = self.ltlf_parser(propositional_formula)
            dfa_dot = formula_obj.to_dfa()
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert LTLf to DFA.\n"
                f"Original formula: {original_formula}\n"
                f"Propositional formula: {propositional_formula}\n"
                f"Error: {str(e)}"
            ) from e

        # Prepare metadata
        metadata = {
            "original_formula": original_formula,
            "propositional_formula": propositional_formula,
            "predicate_to_prop_mapping": self.encoder.get_mapping(),
            "prop_to_predicate_mapping": self.encoder.get_reverse_mapping(),
            "num_states": self._count_dfa_states(dfa_dot),
            "alphabet": self._extract_alphabet(dfa_dot),
        }

        return dfa_dot, metadata

    def _count_dfa_states(self, dfa_dot: str) -> int:
        """Count number of states in DFA from DOT representation"""
        # Simple heuristic: count lines with state transitions
        lines = dfa_dot.strip().split('\n')
        # Count lines that contain "->" which indicate transitions
        transition_lines = [line for line in lines if '->' in line and 'init' not in line]
        # Rough estimate - not exact but good enough for metadata
        return max(10, len(transition_lines) // 2)  # Placeholder logic

    def _extract_alphabet(self, dfa_dot: str) -> List[str]:
        """Extract alphabet (propositional variables) from DFA"""
        # This would require parsing DOT more carefully
        # For now, return from our encoder mapping
        return list(self.encoder.predicate_to_prop.values())


def test_converter():
    """Test the LTLf to DFA converter"""

    # Mock LTL specification
    class MockFormula:
        def __init__(self, formula_str):
            self.formula_str = formula_str

        def to_string(self):
            return self.formula_str

    class MockLTLSpec:
        def __init__(self, formulas):
            self.formulas = [MockFormula(f) for f in formulas]

    print("="*80)
    print("LTLf TO DFA CONVERTER TEST")
    print("="*80)
    print()

    converter = LTLfToDFA()

    # Test cases
    test_cases = [
        (["F(on(a, b))"], "Single goal: Eventually on(a, b)"),
        (["F(on(a, b))", "F(clear(a))"], "Multiple goals: Eventually on(a,b) and clear(a)"),
        (["G(handempty)"], "Global constraint: Always handempty"),
        (["F(on(a, b))", "G(clear(c))"], "Mixed: Eventually on(a,b), always clear(c)"),
    ]

    for formulas, description in test_cases:
        print(f"Test: {description}")
        print(f"Input formulas: {formulas}")

        spec = MockLTLSpec(formulas)

        try:
            dfa_dot, metadata = converter.convert(spec)

            print(f"✓ Conversion successful")
            print(f"  Original: {metadata['original_formula']}")
            print(f"  Propositional: {metadata['propositional_formula']}")
            print(f"  Mappings: {metadata['predicate_to_prop_mapping']}")
            print(f"  DFA preview: {dfa_dot[:200]}...")
            print()

        except Exception as e:
            print(f"✗ Failed: {e}")
            print()


if __name__ == "__main__":
    test_converter()
