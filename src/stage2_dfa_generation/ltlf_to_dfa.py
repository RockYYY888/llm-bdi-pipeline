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

from typing import Dict, List, Any, Tuple, Optional
from ltlf2dfa.parser.ltlf import LTLfParser
from utils.symbol_normalizer import SymbolNormalizer
from stage1_interpretation.ltlf_formula import LogicalOperator, TemporalOperator


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

    MAX_EXACT_INDEPENDENT_EVENTUALLY_FAST_PATH_STATES = 1 << 12

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

        if hasattr(ltl_spec, "combined_formula_string"):
            original_formula = ltl_spec.combined_formula_string()
        elif hasattr(ltl_spec, 'formulas'):
            formula_strings = [f.to_string() for f in ltl_spec.formulas]
            if len(formula_strings) == 0:
                raise ValueError("No LTLf formulas provided")
            if len(formula_strings) == 1:
                original_formula = formula_strings[0]
            else:
                original_formula = " & ".join(f"({f})" for f in formula_strings)
        else:
            raise ValueError("ltl_spec must have 'formulas' attribute")

        # Convert to propositional encoding
        propositional_formula = self.encoder.convert_formula(original_formula)

        ordered_atoms = self._extract_ordered_eventually_atoms(ltl_spec)
        if ordered_atoms:
            propositional_atoms = [
                self.encoder.encode_predicate(atom)
                for atom in ordered_atoms
            ]
            dfa_dot = self._build_ordered_eventually_atomic_dfa(propositional_atoms)
            unique_atoms: List[str] = []
            for atom in propositional_atoms:
                if atom not in unique_atoms:
                    unique_atoms.append(atom)
            metadata = {
                "original_formula": original_formula,
                "propositional_formula": propositional_formula,
                "predicate_to_prop_mapping": self.encoder.get_mapping(),
                "prop_to_predicate_mapping": self.encoder.get_reverse_mapping(),
                "num_states": len(propositional_atoms) + 1,
                "alphabet": list(unique_atoms),
                "construction": "ordered_eventually_atomic_fast_path",
            }
            return dfa_dot, metadata

        eventual_atoms = self._extract_independent_eventually_atoms(ltl_spec)
        if eventual_atoms:
            propositional_atoms = [
                self.encoder.encode_predicate(atom)
                for atom in eventual_atoms
            ]
            unique_atoms: List[str] = []
            for atom in propositional_atoms:
                if atom not in unique_atoms:
                    unique_atoms.append(atom)
            if self._can_materialize_exact_independent_eventually_dfa(unique_atoms):
                dfa_dot = self._build_independent_eventually_atomic_dfa(propositional_atoms)
                metadata = {
                    "original_formula": original_formula,
                    "propositional_formula": propositional_formula,
                    "predicate_to_prop_mapping": self.encoder.get_mapping(),
                    "prop_to_predicate_mapping": self.encoder.get_reverse_mapping(),
                    "num_states": 1 << len(unique_atoms),
                    "alphabet": list(unique_atoms),
                    "construction": "independent_eventually_atomic_fast_path",
                }
                return dfa_dot, metadata

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
            "num_transitions": self._count_dfa_transitions(dfa_dot),
            "alphabet": self._extract_alphabet(dfa_dot),
        }

        return dfa_dot, metadata

    def _can_materialize_exact_independent_eventually_dfa(self, unique_atoms: List[str]) -> bool:
        if not unique_atoms:
            return False
        return (1 << len(unique_atoms)) <= self.MAX_EXACT_INDEPENDENT_EVENTUALLY_FAST_PATH_STATES

    def _extract_independent_eventually_atoms(self, ltl_spec: Any) -> Optional[List[str]]:
        formulas = list(getattr(ltl_spec, "formulas", ()) or ())
        if not formulas:
            return None

        atoms: List[str] = []
        for formula in formulas:
            if not self._collect_independent_eventually_atoms(formula, atoms):
                return None
        return atoms or None

    def _collect_independent_eventually_atoms(self, formula: Any, atoms: List[str]) -> bool:
        logical_op = getattr(formula, "logical_op", None)
        operator = getattr(formula, "operator", None)
        predicate = getattr(formula, "predicate", None)
        sub_formulas = list(getattr(formula, "sub_formulas", ()) or ())

        if logical_op == LogicalOperator.AND and predicate is None and operator is None:
            return all(
                self._collect_independent_eventually_atoms(child, atoms)
                for child in sub_formulas
            )

        if operator != TemporalOperator.FINALLY or len(sub_formulas) != 1:
            return False

        atom = sub_formulas[0]
        if getattr(atom, "operator", None) is not None:
            return False
        if getattr(atom, "logical_op", None) is not None:
            return False
        if getattr(atom, "predicate", None) is None:
            return False

        atoms.append(atom.to_string())
        return True

    def _extract_ordered_eventually_atoms(self, ltl_spec: Any) -> Optional[List[str]]:
        explicit_ordering = bool(getattr(ltl_spec, "query_task_sequence_is_ordered", False))
        explicit_signatures = [
            str(signature).strip()
            for signature in (getattr(ltl_spec, "query_task_literal_signatures", ()) or ())
            if str(signature).strip()
        ]
        if explicit_ordering and explicit_signatures:
            return explicit_signatures

        formulas = list(getattr(ltl_spec, "formulas", ()) or ())
        if len(formulas) != 1:
            return None
        return self._collect_ordered_eventually_atoms(formulas[0])

    def _collect_ordered_eventually_atoms(self, formula: Any) -> Optional[List[str]]:
        operator = getattr(formula, "operator", None)
        sub_formulas = list(getattr(formula, "sub_formulas", ()) or ())
        if operator != TemporalOperator.FINALLY or len(sub_formulas) != 1:
            return None

        child = sub_formulas[0]
        if self._is_atomic_predicate_formula(child):
            return [child.to_string()]

        if getattr(child, "logical_op", None) != LogicalOperator.AND or len(child.sub_formulas) != 2:
            return None

        first, remainder = child.sub_formulas
        if not self._is_atomic_predicate_formula(first):
            return None

        suffix = self._collect_ordered_eventually_atoms(remainder)
        if not suffix:
            return None
        return [first.to_string(), *suffix]

    @staticmethod
    def _is_atomic_predicate_formula(formula: Any) -> bool:
        return (
            getattr(formula, "operator", None) is None
            and getattr(formula, "logical_op", None) is None
            and getattr(formula, "predicate", None) is not None
        )

    def _build_independent_eventually_atomic_dfa(self, propositional_atoms: List[str]) -> str:
        unique_atoms: List[str] = []
        for atom in propositional_atoms:
            if atom not in unique_atoms:
                unique_atoms.append(atom)

        transitions: List[Tuple[str, str, str]] = []
        full_mask = (1 << len(unique_atoms)) - 1

        def state_name(mask: int) -> str:
            return str(mask + 1)

        for mask in range(full_mask + 1):
            from_state = state_name(mask)
            if mask == full_mask:
                transitions.append((from_state, from_state, "true"))
                continue

            for atom_index, atom_name in enumerate(unique_atoms):
                if mask & (1 << atom_index):
                    continue
                next_mask = mask | (1 << atom_index)
                transitions.append(
                    (
                        from_state,
                        state_name(next_mask),
                        atom_name,
                    ),
                )

        accepting_states = {state_name(full_mask)}
        other_states = {
            state_name(mask)
            for mask in range(full_mask + 1)
            if mask != full_mask
        }

        lines = [
            "digraph MONA_DFA {",
            " rankdir = LR;",
            " center = true;",
            " size = \"7.5,10.5\";",
            " edge [fontname = Courier];",
            " node [height = .5, width = .5];",
            f" node [shape = doublecircle]; {' '.join(sorted(accepting_states))};",
            f" node [shape = circle]; {' '.join(sorted(other_states, key=int))};",
            " init [shape = plaintext, label = \"\"];",
            " init -> 1;",
        ]
        for from_state, to_state, label in transitions:
            lines.append(f" {from_state} -> {to_state} [label=\"{label}\"];")
        lines.append("}")
        return "\n".join(lines)

    def _build_ordered_eventually_atomic_dfa(self, propositional_atoms: List[str]) -> str:
        accepting_state = str(len(propositional_atoms) + 1)
        other_states = [str(index) for index in range(1, len(propositional_atoms) + 1)]

        lines = [
            "digraph MONA_DFA {",
            " rankdir = LR;",
            " center = true;",
            " size = \"7.5,10.5\";",
            " edge [fontname = Courier];",
            " node [height = .5, width = .5];",
            f" node [shape = doublecircle]; {accepting_state};",
        ]
        if other_states:
            lines.append(f" node [shape = circle]; {' '.join(other_states)};")
        lines.append(" init [shape = plaintext, label = \"\"];")
        lines.append(" init -> 1;")
        for index, atom_name in enumerate(propositional_atoms, start=1):
            lines.append(f" {index} -> {index + 1} [label=\"{atom_name}\"];")
        lines.append("}")
        return "\n".join(lines)

    def _count_dfa_states(self, dfa_dot: str) -> int:
        """Count number of states in DFA from DOT representation"""
        # Simple heuristic: count lines with state transitions
        lines = dfa_dot.strip().split('\n')
        # Count lines that contain "->" which indicate transitions
        transition_lines = [line for line in lines if '->' in line and 'init' not in line]
        # Rough estimate - not exact but good enough for metadata
        return max(10, len(transition_lines) // 2)  # Placeholder logic

    def _count_dfa_transitions(self, dfa_dot: str) -> int:
        """Count DFA transitions cheaply for large generic ltlf2dfa outputs."""
        total_edges = dfa_dot.count("->")
        init_edges = dfa_dot.count("init ->")
        return max(0, total_edges - init_edges)

    def _extract_alphabet(self, dfa_dot: str) -> List[str]:
        """Extract alphabet (propositional variables) from DFA"""
        # This would require parsing DOT more carefully
        # For now, return from our encoder mapping (normalized symbols)
        mapping = self.encoder.get_mapping()
        return list(mapping.values()) if mapping else []


def test_converter():
    """Test the LTLf to DFA converter"""

    # Example LTL specification
    class ExampleFormula:
        def __init__(self, formula_str):
            self.formula_str = formula_str

        def to_string(self):
            return self.formula_str

    class ExampleLTLSpec:
        def __init__(self, formulas):
            self.formulas = [ExampleFormula(f) for f in formulas]

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

        spec = ExampleLTLSpec(formulas)

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
