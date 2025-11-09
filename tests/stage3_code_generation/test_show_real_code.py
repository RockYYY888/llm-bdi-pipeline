"""
Show real AgentSpeak code generated for a simple example
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from src.utils.pddl_parser import PDDLParser
from src.stage1_interpretation.grounding_map import GroundingMap


def show_simple_code_example():
    print("=" * 80)
    print("SIMPLE EXAMPLE: Generate AgentSpeak Code for on(a, b)")
    print("=" * 80)
    print()

    # Load domain
    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Create grounding map for single goal: on(a, b)
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])

    # Simple DFA: state0 --[on_a_b]-> state1
    dfa_dot = """
digraph {
    rankdir=LR;
    node [shape=circle];
    __start [shape=point];
    __start -> state0;
    state0 [label="0"];
    state1 [label="1", shape=doublecircle];
    state0 -> state1 [label="on_a_b"];
}
"""

    dfa_result = {
        "formula": "F(on(a, b))",
        "dfa_dot": dfa_dot,
        "num_states": 2,
        "num_transitions": 1
    }

    ltl_dict = {
        "objects": ["a", "b"],
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    print("Goal: on(a, b)")
    print("Objects: a, b")
    print()

    # Generate code
    generator = BackwardPlannerGenerator(domain, grounding_map)
    code = generator.generate(ltl_dict, dfa_result)

    print("=" * 80)
    print("GENERATED AGENTSPEAK CODE")
    print("=" * 80)
    print()
    print(code)
    print()

    # Show key sections
    print("=" * 80)
    print("KEY SECTIONS EXPLANATION")
    print("=" * 80)
    print()

    lines = code.split('\n')

    # Find a few example plans
    plan_lines = []
    for i, line in enumerate(lines):
        if line.strip().startswith('+!on(a, b)'):
            plan_lines.append((i+1, line))
            if len(plan_lines) >= 5:
                break

    if plan_lines:
        print("Example Plans (first 5):")
        print()
        for line_num, line in plan_lines:
            print(f"Line {line_num}: {line}")
        print()

        print("Plan Structure:")
        print("  +!on(a, b) : <context> <- <body>.")
        print()
        print("  - Trigger: +!on(a, b)  (achieve goal on(a, b))")
        print("  - Context: current world state predicates")
        print("  - Body: action to execute OR subgoal to achieve")
        print()


if __name__ == "__main__":
    show_simple_code_example()
