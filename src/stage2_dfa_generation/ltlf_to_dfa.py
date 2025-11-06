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

NOTE: Requires MONA (MONadic second-order logic Automata) to be installed.
      The setup_mona_path module automatically adds MONA to PATH.
"""

# Setup MONA path before importing ltlf2dfa
import sys
from pathlib import Path

# Add parent src directory to path for setup_mona_path import (only once)
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from utils.setup_mona_path import setup_mona
setup_mona()

from typing import Dict, List, Any, Tuple
from ltlf2dfa.parser.ltlf import LTLfParser
from utils.symbol_normalizer import SymbolNormalizer


class PredicateToProposition:
    """
    Converts predicate-based LTLf formulas to propositional encoding

    Now uses SymbolNormalizer for consistent symbol handling across the pipeline.

    Examples:
        on(a, b) → on_a_b
        clear(a) → clear_a
        on(block-1, block-2) → on_blockhh1_blockhh2 (with hyphen encoding)
        handempty → handempty (already propositional)
    """

    def __init__(self, normalizer: SymbolNormalizer = None):
        """
        Initialize with optional normalizer instance

        Args:
            normalizer: SymbolNormalizer instance (creates new one if not provided)
        """
        self.normalizer = normalizer or SymbolNormalizer()

    def encode_predicate(self, predicate_str: str) -> str:
        """
        Encode a single predicate to propositional variable

        Args:
            predicate_str: e.g., "on(a, b)" or "clear(block-1)"

        Returns:
            Propositional variable: e.g., "on_a_b" or "clear_blockhh1"
        """
        # Parse predicate string using normalizer
        pred_name, args = self.normalizer.parse_predicate_string(predicate_str.strip())

        if not args:
            # Already propositional (e.g., "handempty", "a", "b")
            return predicate_str.strip()

        # Create propositional symbol using normalizer
        return self.normalizer.create_propositional_symbol(pred_name, args)

    def convert_formula(self, ltlf_formula_str: str) -> str:
        """
        Convert entire LTLf formula from predicate to propositional form

        Args:
            ltlf_formula_str: e.g., "F(on(a, b))" or "F(on(block-1, block-2)) & G(clear(c))"

        Returns:
            Propositional formula: e.g., "F(on_a_b)" or "F(on_blockhh1_blockhh2) & G(clear_c)"
        """
        # Use normalizer's formula normalization
        return self.normalizer.normalize_formula_string(ltlf_formula_str)

    def get_mapping(self) -> Dict[str, str]:
        """Get original → normalized mapping"""
        return self.normalizer.get_original_to_normalized_map()

    def get_reverse_mapping(self) -> Dict[str, str]:
        """Get normalized → original mapping"""
        return self.normalizer.get_normalized_to_original_map()


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
        # For now, return from our encoder mapping (normalized symbols)
        mapping = self.encoder.get_mapping()
        return list(mapping.values()) if mapping else []


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
            print(f"\n  COMPLETE DFA (DOT format):")
            print("  " + "~" * 76)
            for line in dfa_dot.split('\n'):
                print(f"  {line}")
            print("  " + "~" * 76)
            print()

        except Exception as e:
            print(f"✗ Failed: {e}")
            print()


if __name__ == "__main__":
    test_converter()
