"""
Stage 2: DFA Generation from LTLf Specifications

Converts LTLf formulas to DFA (Deterministic Finite Automaton).
The DFA is used as context for Stage 3 AgentSpeak generation.
"""

import sys
import time
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
    DFA builder that converts LTLf formula to the raw DFA used downstream.
    """

    def __init__(self):
        """Initialize DFA builder."""
        self.converter = LTLfToDFA()
    
    def build(self, ltl_spec) -> Dict[str, Any]:
        """
        Build the raw DFA from the LTLf specification.

        Args:
            ltl_spec: LTLSpecification object with formulas and grounding_map

        Returns:
            Dict with:
                - formula: Original LTLf formula string
                - dfa_dot: Raw DFA in DOT format
                - dfa_path: Relative artifact path used by the logger
                - construction: DFA construction path reported by the converter
                - num_states: Number of states in the raw DFA
                - num_transitions: Number of transitions in the raw DFA
        """
        total_start = time.perf_counter()
        timing_profile: Dict[str, float] = {}

        formula_render_start = time.perf_counter()
        if hasattr(ltl_spec, "combined_formula_string"):
            formula_str = ltl_spec.combined_formula_string()
        else:
            if len(ltl_spec.formulas) == 0:
                raise ValueError("No LTLf formulas in specification")
            if len(ltl_spec.formulas) == 1:
                formula_str = ltl_spec.formulas[0].to_string()
            else:
                formula_str = " & ".join(
                    f"({formula.to_string()})"
                    for formula in ltl_spec.formulas
                )
        timing_profile["formula_render_seconds"] = time.perf_counter() - formula_render_start

        # Step 1: Generate DFA using the convert method (takes ltl_spec object)
        convert_start = time.perf_counter()
        dfa_dot, metadata = self.converter.convert(ltl_spec)
        timing_profile["convert_seconds"] = time.perf_counter() - convert_start

        stats_start = time.perf_counter()
        num_states = int(metadata.get("num_states") or self._count_states(dfa_dot))
        num_transitions = int(
            metadata.get("num_transitions") or self._count_transitions(dfa_dot)
        )
        timing_profile["stats_seconds"] = time.perf_counter() - stats_start
        timing_profile["total_seconds"] = time.perf_counter() - total_start

        result = {
            "formula": formula_str,
            "dfa_dot": dfa_dot,
            "dfa_path": "dfa.dot",
            "construction": metadata.get("construction") or "generic_ltlf2dfa",
            "num_states": num_states,
            "num_transitions": num_transitions,
            "num_predicates": len(metadata.get("alphabet", [])),
            "timing_profile": timing_profile,
        }

        return result
    
    def _count_states(self, dfa_dot: str) -> int:
        """Count number of states in DFA"""
        import re
        states = set()

        for line in dfa_dot.split('\n'):
            grouped_match = re.search(r'node\s+\[.*?\];\s*([^;]+);', line)
            if grouped_match:
                tokens = re.findall(r'[A-Za-z0-9_]+', grouped_match.group(1))
                states.update(token for token in tokens if token != "init")
                continue
            single_match = re.search(r'([A-Za-z0-9_]+)\s*\[\s*shape\s*=\s*', line)
            if single_match:
                token = single_match.group(1)
                if token != "init":
                    states.add(token)

        return len(states)
    
    def _count_transitions(self, dfa_dot: str) -> int:
        """Count number of transitions in DFA"""
        import re
        if "->" not in dfa_dot:
            return 0
        if "\n" not in dfa_dot:
            total_edges = dfa_dot.count("->")
            init_edges = len(re.findall(r'init\s*->', dfa_dot))
            return max(0, total_edges - init_edges)

        transition_count = 0
        for line in dfa_dot.splitlines():
            if "->" not in line:
                continue
            total_edges = line.count("->")
            init_edges = len(re.findall(r'init\s*->', line))
            transition_count += max(0, total_edges - init_edges)
        return transition_count


def test_dfa_builder():
    """Test DFA builder"""
    from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator
    from stage1_interpretation.grounding_map import GroundingMap

    print("="*80)
    print("DFA BUILDER TEST")
    print("="*80)

    # Create test LTL spec: F(on_a_b)
    spec = LTLSpecification()
    spec.objects = ["a", "b"]

    # Create grounding map
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    spec.grounding_map = gmap

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
    print(f"\n  DFA:")
    print(f"    States: {result['num_states']}")
    print(f"    Transitions: {result['num_transitions']}")
    print(f"  Construction: {result['construction']}")
    print(f"  Predicates: {result['num_predicates']}")

    print(f"\nDFA:")
    print(result['dfa_dot'])

    print("\n" + "="*80)


if __name__ == "__main__":
    test_dfa_builder()
