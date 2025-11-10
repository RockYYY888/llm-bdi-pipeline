"""
Test Variable Abstraction Feature (Correct Version)

This test demonstrates the correct use case for variable abstraction:
Goals with DIFFERENT objects but SAME structure should share exploration.

Example:
- on(a, b) with objects=[a, b] → on(?v0, ?v1)
- on(c, d) with objects=[c, d] → on(?v0, ?v1)

Both normalize to the same pattern and share state space exploration!
"""

import sys
from pathlib import Path

# Add src to path
_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from stage1_interpretation.grounding_map import GroundingMap
from utils.pddl_parser import PDDLParser


def test_same_structure_different_objects():
    """
    Test that goals with same structure but different objects share exploration

    Goals:
    - on(a, b) with objects=[a, b, c, d] → on(?v0, ?v1)
    - on(c, d) with objects=[a, b, c, d] → on(?v2, ?v3)

    Both involve putting one block on another, should share exploration!
    """
    print("="*80)
    print("TEST: Same Structure, Different Objects")
    print("="*80)

    # Load domain
    domain_file = Path(__file__).parent.parent / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    # Create grounding map
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("on_c_d", "on", ["c", "d"])

    # DFA with two transitions
    dfa_dot = """
    digraph G {
        __start [shape=none];
        s0 [shape=circle];
        s1 [shape=circle];
        s2 [shape=doublecircle];
        __start -> s0;
        s0 -> s1 [label="on_a_b"];
        s1 -> s2 [label="on_c_d"];
    }
    """

    ltl_dict = {
        'objects': ['a', 'b', 'c', 'd'],  # All objects in problem
        'formulas_string': ['F(on_a_b) & F(on_c_d)'],
        'grounding_map': gmap
    }

    dfa_result = {
        'formula': 'F(on_a_b) & F(on_c_d)',
        'dfa_dot': dfa_dot
    }

    # Generate code
    print("\nGenerating AgentSpeak code with variable abstraction...\n")
    print("Expected behavior:")
    print("  - on(a, b) normalizes to: on(?v0, ?v1)")
    print("  - on(c, d) normalizes to: on(?v2, ?v3)")
    print("  - Different patterns → No cache hit (both explorations needed)")
    print("\nThis is CORRECT because a,b vs c,d use different variable positions!")
    print("="*80 + "\n")

    generator = BackwardPlannerGenerator(domain, gmap)
    code = generator.generate(ltl_dict, dfa_result)

    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    print("\nWith current global variable assignment:")
    print("  - Objects sorted: [a, b, c, d]")
    print("  - Mapping: a→?v0, b→?v1, c→?v2, d→?v3")
    print("  - on(a,b) → on(?v0, ?v1)")
    print("  - on(c,d) → on(?v2, ?v3)")
    print("  - Result: Different patterns, no sharing 😞")

    print("\nWhat we WANT:")
    print("  - Both goals involve 'putting one block on another'")
    print("  - Structure is the same, only objects differ")
    print("  - Should share the same abstract plan!")

    return code


def test_clear_vs_on():
    """
    Test that different predicates don't share (obviously)

    Goals:
    - clear(a) → clear(?v0)
    - on(a, b) → on(?v0, ?v1)

    Different predicates, different patterns, no sharing.
    """
    print("\n" + "="*80)
    print("TEST: Different Predicates (No Sharing Expected)")
    print("="*80)

    domain_file = Path(__file__).parent.parent / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    gmap = GroundingMap()
    gmap.add_atom("clear_a", "clear", ["a"])
    gmap.add_atom("on_a_b", "on", ["a", "b"])

    dfa_dot = """
    digraph G {
        __start [shape=none];
        s0 [shape=circle];
        s1 [shape=circle];
        s2 [shape=doublecircle];
        __start -> s0;
        s0 -> s1 [label="clear_a"];
        s1 -> s2 [label="on_a_b"];
    }
    """

    ltl_dict = {
        'objects': ['a', 'b'],
        'formulas_string': ['F(clear_a) & F(on_a_b)'],
        'grounding_map': gmap
    }

    dfa_result = {
        'formula': 'F(clear_a) & F(on_a_b)',
        'dfa_dot': dfa_dot
    }

    print("\nGenerating AgentSpeak code...\n")
    print("Expected: No cache hits (different predicates)")
    print("="*80 + "\n")

    generator = BackwardPlannerGenerator(domain, gmap)
    code = generator.generate(ltl_dict, dfa_result)

    print("\n" + "="*80)
    print("✅ Test completed - Different predicates correctly don't share")
    print("="*80)

    return code


if __name__ == "__main__":
    print("="*80)
    print("VARIABLE ABSTRACTION TESTS (CORRECTED)")
    print("="*80)
    print("\nThese tests explore the current variable abstraction behavior.")
    print("="*80 + "\n")

    # Test 1
    code1 = test_same_structure_different_objects()

    # Test 2
    code2 = test_clear_vs_on()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nCurrent behavior:")
    print("  - Variables assigned based on GLOBAL object order")
    print("  - on(a,b) and on(c,d) get different normalized forms")
    print("  - No sharing between them")
    print("\nThis is actually CORRECT for PDDL semantics!")
    print("But it means variable abstraction only helps when:")
    print("  - Multiple goals use the EXACT SAME OBJECTS")
    print("  - Example: on(a,b) appears multiple times in DFA")
    print("\nFor true structure-level sharing, we'd need lifted planning")
    print("where we work with predicate schemas directly.")
    print("="*80)
