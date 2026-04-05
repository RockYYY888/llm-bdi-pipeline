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
from stage2_dfa_generation.dfa_simplifier import DFASimplifier


class DFABuilder:
    """
    DFA builder that converts LTLf formula to DFA and simplifies it

    Converts the complete LTLf formula to a DFA, then applies BDD Shannon Expansion
    to simplify transition labels to atomic literals (var or !var). This is a
    mandatory step that ensures each transition has a single atomic predicate.
    """

    def __init__(self):
        """Initialize DFA builder with simplifier"""
        self.converter = LTLfToDFA()
        self.simplifier = DFASimplifier()
    
    def build(self, ltl_spec) -> Dict[str, Any]:
        """
        Build and simplify DFA from LTLf specification

        This method performs two steps:
        1. Generate DFA from LTLf formula using ltlf2dfa
        2. Simplify DFA by applying BDD Shannon Expansion to transition labels

        Args:
            ltl_spec: LTLSpecification object with formulas and grounding_map

        Returns:
            Dict with:
                - formula: Original LTLf formula string
                - original_dfa_dot: Original DFA in DOT format (before simplification)
                - dfa_dot: Simplified DFA in DOT format (with atomic literals)
                - num_states: Number of states in simplified DFA
                - num_transitions: Number of transitions in simplified DFA (may increase after simplification)
                - original_num_states: Number of states in original DFA
                - original_num_transitions: Number of transitions in original DFA
                - simplification_stats: Statistics about the simplification process
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
        original_dfa_dot, metadata = self.converter.convert(ltl_spec)
        timing_profile["convert_seconds"] = time.perf_counter() - convert_start

        # Parse original DFA to get statistics (BEFORE simplification)
        original_stats_start = time.perf_counter()
        original_num_states = int(metadata.get("num_states") or self._count_states(original_dfa_dot))
        original_num_transitions = int(
            metadata.get("num_transitions") or self._count_transitions(original_dfa_dot)
        )
        timing_profile["original_stats_seconds"] = time.perf_counter() - original_stats_start

        if metadata.get("construction") in {
            "independent_eventually_atomic_fast_path",
            "independent_eventually_symbolic_surrogate",
            "ordered_eventually_atomic_fast_path",
        }:
            timing_profile["total_seconds"] = time.perf_counter() - total_start
            return {
                "formula": formula_str,
                "original_dfa_dot": original_dfa_dot,
                "dfa_dot": original_dfa_dot,
                "num_states": original_num_states,
                "num_transitions": original_num_transitions,
                "original_num_states": original_num_states,
                "original_num_transitions": original_num_transitions,
                "simplification_stats": {
                    "method": metadata.get("construction"),
                    "num_predicates": len(metadata.get("alphabet", [])),
                    "num_original_states": original_num_states,
                    "num_new_states": original_num_states,
                    "num_original_transitions": original_num_transitions,
                    "num_new_transitions": original_num_transitions,
                    "skipped_simplifier": True,
                },
                "timing_profile": timing_profile,
            }

        # Step 2: Simplify DFA (mandatory)
        # This replaces complex boolean expressions with atomic literals (var or !var)
        if not hasattr(ltl_spec, 'grounding_map') or ltl_spec.grounding_map is None:
            raise ValueError("LTLSpecification must have a grounding_map for DFA simplification")

        simplify_start = time.perf_counter()
        simplified_result = self.simplifier.simplify(original_dfa_dot, ltl_spec.grounding_map)
        timing_profile["simplify_seconds"] = time.perf_counter() - simplify_start

        # Parse simplified DFA to get statistics (AFTER simplification)
        simplified_stats_start = time.perf_counter()
        num_states = self._count_states(simplified_result.simplified_dot)
        num_transitions = self._count_transitions(simplified_result.simplified_dot)
        timing_profile["simplified_stats_seconds"] = (
            time.perf_counter() - simplified_stats_start
        )
        timing_profile["total_seconds"] = time.perf_counter() - total_start

        result = {
            "formula": formula_str,
            "original_dfa_dot": original_dfa_dot,  # NEW: Original DFA before simplification
            "dfa_dot": simplified_result.simplified_dot,  # Simplified DFA
            "num_states": num_states,  # Simplified DFA states
            "num_transitions": num_transitions,  # Simplified DFA transitions
            "original_num_states": original_num_states,  # NEW: Original DFA states
            "original_num_transitions": original_num_transitions,  # NEW: Original DFA transitions
            "simplification_stats": simplified_result.stats,
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
    print("DFA BUILDER TEST (with mandatory simplification)")
    print("="*80)

    # Create test LTL spec: F(on_a_b)
    spec = LTLSpecification()
    spec.objects = ["a", "b"]

    # Create grounding map (required for simplification)
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

    # Build DFA (now includes mandatory simplification)
    builder = DFABuilder()
    result = builder.build(spec)

    print(f"\n✓ DFA Generated and Simplified")
    print(f"  Formula: {result['formula']}")
    print(f"\n  Original DFA (before simplification):")
    print(f"    States: {result['original_num_states']}")
    print(f"    Transitions: {result['original_num_transitions']}")
    print(f"\n  Simplified DFA (after simplification):")
    print(f"    States: {result['num_states']}")
    print(f"    Transitions: {result['num_transitions']}")
    print(f"\nSimplification Stats:")
    stats = result['simplification_stats']
    print(f"  Method: {stats['method']}")
    print(f"  Predicates: {stats['num_predicates']}")

    # Only show detailed stats if simplification was actually performed
    if 'num_original_states' in stats:
        print(f"  Original States: {stats['num_original_states']}")
        print(f"  New States: {stats['num_new_states']}")
        print(f"  Original Transitions: {stats['num_original_transitions']}")
        print(f"  New Transitions: {stats['num_new_transitions']}")

    print(f"\nOriginal DFA:")
    print(result['original_dfa_dot'])
    print(f"\nSimplified DFA:")
    print(result['dfa_dot'])

    print("\n" + "="*80)


if __name__ == "__main__":
    test_dfa_builder()
