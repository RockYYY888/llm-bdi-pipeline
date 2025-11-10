"""
Test goal exploration caching with duplicate transitions
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from src.utils.pddl_parser import PDDLParser
from src.stage1_interpretation.grounding_map import GroundingMap


def test_goal_caching():
    print("=" * 80)
    print("TEST: Goal Exploration Caching with Duplicate Transitions")
    print("=" * 80)
    print()

    # Load domain
    domain_path = project_root / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Create grounding map with SAME goal repeated 3 times
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])

    # DFA with 3 transitions ALL using same label
    dfa_dot = """
digraph {
    rankdir=LR;
    __start [shape=point];
    __start -> state0;
    state0 [label="0"];
    state1 [label="1"];
    state2 [label="2"];
    state3 [label="3", shape=doublecircle];
    state0 -> state1 [label="on_a_b"];
    state1 -> state2 [label="on_a_b"];
    state2 -> state3 [label="on_a_b"];
}
"""

    dfa_result = {
        "formula": "F(on_a_b & F(on_a_b & F(on_a_b)))",
        "dfa_dot": dfa_dot,
        "num_states": 4,
        "num_transitions": 3
    }

    ltl_dict = {
        "objects": ["a", "b"],
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    print("DFA: 3 transitions ALL with label 'on_a_b'")
    print()
    print("Expected behavior:")
    print("  - Transition 1: Cache MISS - explore on(a,b)")
    print("  - Transition 2: Cache HIT - reuse exploration")
    print("  - Transition 3: Cache HIT - reuse exploration")
    print()
    print("Expected cache stats:")
    print("  - Cache hits: 2")
    print("  - Cache misses: 1")
    print("  - Hit rate: 66.7%")
    print()

    # Generate code
    generator = BackwardPlannerGenerator(domain, grounding_map)
    code = generator.generate(ltl_dict, dfa_result)

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print()
    print("✅ Test complete! Check cache statistics above.")
    print()


if __name__ == "__main__":
    test_goal_caching()
