"""
Stage 2: DFA Generation from LTLf Specifications

Converts LTLf formulas to DFA (Deterministic Finite Automaton).
The DFA is used as context for Stage 3 AgentSpeak generation.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add parent src directory to path
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from utils.setup_mona_path import setup_mona
setup_mona()

from stage2_dfa_generation.ltlf_to_dfa import LTLfToDFA


class DFABuilder:
    """
    DFA builder that converts LTLf formula to DFA

    Converts the complete LTLf formula to a DFA for use as context in Stage 3.
    """

    def __init__(self):
        """Initialize DFA builder"""
        self.converter = LTLfToDFA()
    
    def build(self, ltl_spec) -> Dict[str, Any]:
        """
        Build DFA from LTLf specification

        Args:
            ltl_spec: LTLSpecification object with formulas

        Returns:
            Dict with:
                - formula: Original LTLf formula string
                - dfa_dot: DFA in DOT format
                - num_states: Number of states in DFA
                - num_transitions: Number of transitions
        """
        from stage1_interpretation.ltlf_formula import LogicalOperator
        
        # Combine all formulas with AND if multiple
        if len(ltl_spec.formulas) == 0:
            raise ValueError("No LTLf formulas in specification")
        
        if len(ltl_spec.formulas) == 1:
            combined_formula = ltl_spec.formulas[0]
        else:
            # Create AND of all formulas
            from stage1_interpretation.ltlf_formula import LTLFormula
            combined_formula = LTLFormula()
            combined_formula.logical_op = LogicalOperator.AND
            combined_formula.sub_formulas = ltl_spec.formulas
        
        # Convert to string
        formula_str = combined_formula.to_string()

        # Generate DFA using the convert method (takes ltl_spec object)
        dfa_dot, metadata = self.converter.convert(ltl_spec)
        
        # Parse DFA to get statistics
        num_states = self._count_states(dfa_dot)
        num_transitions = self._count_transitions(dfa_dot)
        
        result = {
            "formula": formula_str,
            "dfa_dot": dfa_dot,
            "num_states": num_states,
            "num_transitions": num_transitions
        }
        
        return result
    
    def _count_states(self, dfa_dot: str) -> int:
        """Count number of states in DFA"""
        import re
        # Match state declarations like: node [shape = doublecircle]; 2;
        state_pattern = r'\b(\d+)\b'
        states = set()
        
        for line in dfa_dot.split('\n'):
            # Skip init and label lines
            if 'init' in line or 'label' in line or '->' in line:
                continue
            matches = re.findall(state_pattern, line)
            states.update(matches)
        
        return len(states)
    
    def _count_transitions(self, dfa_dot: str) -> int:
        """Count number of transitions in DFA"""
        import re
        # Match transitions like: 1 -> 2 [label="condition"];
        transition_pattern = r'\d+\s*->\s*\d+'
        transitions = re.findall(transition_pattern, dfa_dot)
        return len(transitions)


def test_dfa_builder():
    """Test DFA builder"""
    from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator

    print("="*80)
    print("DFA BUILDER TEST")
    print("="*80)

    # Create test LTL spec: F(on_a_b)
    spec = LTLSpecification()
    spec.objects = ["a", "b"]

    # Create atomic formula on_a_b
    atom = LTLFormula(
        operator=None,
        predicate="on_a_b",
        sub_formulas=[],
        logical_op=None
    )

    # Create F(on_a_b) formula
    f_formula = LTLFormula(
        operator=TemporalOperator.FINALLY,
        predicate=None,
        sub_formulas=[atom],
        logical_op=None
    )

    spec.formulas = [f_formula]

    print(f"\nTest formula: {f_formula.to_string()}")

    # Build DFA
    builder = DFABuilder()
    result = builder.build(spec)

    print(f"\n✓ DFA Generated")
    print(f"  Formula: {result['formula']}")
    print(f"  States: {result['num_states']}")
    print(f"  Transitions: {result['num_transitions']}")
    print(f"\nDFA DOT (first 300 chars):")
    print(result['dfa_dot'][:300] + "...")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_dfa_builder()
