"""
Stage 2: DFA Generation from LTLf Specifications

Converts LTLf formulas to DFA (Deterministic Finite Automaton).
The DFA is used as context for Stage 3 AgentSpeak generation.
"""

import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

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

        symbolic_monitor_start = time.perf_counter()
        symbolic_monitor = self._build_symbolic_unordered_query_step_monitor(ltl_spec)
        timing_profile["symbolic_fragment_seconds"] = (
            time.perf_counter() - symbolic_monitor_start
        )
        if symbolic_monitor is not None:
            timing_profile["convert_seconds"] = 0.0
            timing_profile["stats_seconds"] = 0.0
            timing_profile["total_seconds"] = time.perf_counter() - total_start
            return {
                "formula": formula_str,
                "construction": "exact_symbolic_query_step_conjunction",
                "num_states": symbolic_monitor["num_states"],
                "num_transitions": symbolic_monitor["num_transitions"],
                "num_predicates": len(symbolic_monitor["query_step_indices"]),
                "symbolic_query_step_monitor": symbolic_monitor,
                "timing_profile": timing_profile,
            }

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

    def _build_symbolic_unordered_query_step_monitor(
        self,
        ltl_spec: Any,
    ) -> Optional[Dict[str, Any]]:
        if bool(getattr(ltl_spec, "query_task_sequence_is_ordered", False)):
            return None

        formulas = list(getattr(ltl_spec, "formulas", ()) or ())
        if not formulas:
            return None

        query_step_indices: List[int] = []
        seen_indices: set[int] = set()
        for formula in formulas:
            query_step_index = self._extract_unordered_query_step_eventually_index(formula)
            if query_step_index is None or query_step_index in seen_indices:
                return None
            seen_indices.add(query_step_index)
            query_step_indices.append(query_step_index)

        sorted_indices = sorted(query_step_indices)
        expected_indices = list(range(1, len(sorted_indices) + 1))
        if sorted_indices != expected_indices:
            return None

        query_count = len(sorted_indices)
        return {
            "mode": "unordered_eventually_conjunction",
            "query_step_indices": sorted_indices,
            "initial_state": "q0",
            "accepting_states": [],
            "num_states": 1 << query_count,
            "num_transitions": query_count * (1 << (query_count - 1)),
        }

    @staticmethod
    def _extract_unordered_query_step_eventually_index(formula: Any) -> Optional[int]:
        operator = getattr(formula, "operator", None)
        operator_name = getattr(operator, "value", None)
        if operator_name != "F":
            return None

        sub_formulas = list(getattr(formula, "sub_formulas", ()) or ())
        if len(sub_formulas) != 1:
            return None

        atom = sub_formulas[0]
        if getattr(atom, "operator", None) is not None:
            return None
        if getattr(atom, "logical_op", None) is not None:
            return None

        predicate = getattr(atom, "predicate", None)
        if isinstance(predicate, dict):
            if len(predicate) != 1:
                return None
            predicate_name = next(iter(predicate.keys()))
            args = predicate[predicate_name]
            if list(args or ()):
                return None
        elif isinstance(predicate, str):
            predicate_name = predicate
        else:
            return None

        match = re.fullmatch(r"query_step_(\d+)", str(predicate_name).strip())
        if match is None:
            return None
        return int(match.group(1))
    
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
