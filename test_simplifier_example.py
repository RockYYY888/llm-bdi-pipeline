#!/usr/bin/env python3
"""
Simple test to visualize DFA simplification effect
Shows before/after DOT strings side-by-side
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator, LogicalOperator
from stage2_dfa_generation.ltlf_to_dfa import LTLfToDFA
from stage2_dfa_generation.dfa_simplifier import DFASimplifier


def create_test_formula():
    """F(on(a, b) & clear(c) | on(d, e))"""
    spec = LTLSpecification()
    spec.objects = ["a", "b", "c", "d", "e"]

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    gmap.add_atom("on_d_e", "on", ["d", "e"])
    spec.grounding_map = gmap

    on_a_b = LTLFormula(operator=None, predicate={"on": ["a", "b"]}, sub_formulas=[], logical_op=None)
    clear_c = LTLFormula(operator=None, predicate={"clear": ["c"]}, sub_formulas=[], logical_op=None)
    on_d_e = LTLFormula(operator=None, predicate={"on": ["d", "e"]}, sub_formulas=[], logical_op=None)

    conj = LTLFormula(operator=None, predicate=None, sub_formulas=[on_a_b, clear_c], logical_op=LogicalOperator.AND)
    disj = LTLFormula(operator=None, predicate=None, sub_formulas=[conj, on_d_e], logical_op=LogicalOperator.OR)
    f_formula = LTLFormula(operator=TemporalOperator.FINALLY, predicate=None, sub_formulas=[disj], logical_op=None)

    spec.formulas = [f_formula]
    return spec


def main():
    spec = create_test_formula()

    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "DFA SIMPLIFICATION COMPARISON" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    print("Formula: F(on(a, b) & clear(c) | on(d, e))")
    print()

    # Generate original DFA
    converter = LTLfToDFA()
    original_dfa_dot, _ = converter.convert(spec)

    # Apply simplification
    simplifier = DFASimplifier()
    simplified_result = simplifier.simplify(original_dfa_dot, spec.grounding_map)

    # Side-by-side comparison
    print("┌" + "─" * 38 + "┬" + "─" * 39 + "┐")
    print("│" + " " * 10 + "ORIGINAL DFA" + " " * 16 + "│" + " " * 10 + "SIMPLIFIED DFA" + " " * 15 + "│")
    print("├" + "─" * 38 + "┼" + "─" * 39 + "┤")

    orig_lines = original_dfa_dot.split('\n')
    simp_lines = simplified_result.simplified_dot.split('\n')

    max_lines = max(len(orig_lines), len(simp_lines))

    for i in range(max_lines):
        orig_line = orig_lines[i] if i < len(orig_lines) else ""
        simp_line = simp_lines[i] if i < len(simp_lines) else ""

        # Truncate if too long
        orig_line = (orig_line[:36] + "..") if len(orig_line) > 38 else orig_line
        simp_line = (simp_line[:37] + "..") if len(simp_line) > 39 else simp_line

        print(f"│ {orig_line:<37}│ {simp_line:<38}│")

    print("└" + "─" * 38 + "┴" + "─" * 39 + "┘")
    print()

    # Extract and compare labels
    import re

    def extract_labels(dot_string):
        labels = set()
        for line in dot_string.split('\n'):
            match = re.search(r'\[label="([^"]+)"\]', line)
            if match and 'init' not in line:
                labels.add(match.group(1))
        return labels

    orig_labels = extract_labels(original_dfa_dot)
    simp_labels = extract_labels(simplified_result.simplified_dot)

    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 30 + "LABEL COMPARISON" + " " * 32 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    print(f"ORIGINAL ({len(orig_labels)} labels):")
    for label in sorted(orig_labels):
        print(f"  • {label}")
    print()

    print(f"SIMPLIFIED ({len(simp_labels)} labels):")
    grounded_atoms = ["on_a_b", "clear_c", "on_d_e"]
    for label in sorted(simp_labels):
        marker = "✅" if label in grounded_atoms else "  "
        print(f"  {marker} {label}")
    print()

    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 35 + "RESULT" + " " * 37 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    print(f"✅ Original: {len(orig_labels)} complex boolean expressions")
    print(f"✅ Simplified: {len(simp_labels)} atomic partitions")
    print(f"✅ Grounded atoms preserved: {[l for l in simp_labels if l in grounded_atoms]}")
    print()


if __name__ == "__main__":
    main()
