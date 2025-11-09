"""
Simple Multi-Transition Flow Demonstration

Shows how BackwardPlannerGenerator handles multiple DFA transitions
WITHOUT actually running the full state space exploration.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from src.utils.pddl_parser import PDDLParser
from src.stage1_interpretation.grounding_map import GroundingMap


def demonstrate_multi_transition_logic():
    print("=" * 80)
    print("MULTI-TRANSITION HANDLING - SIMPLIFIED DEMONSTRATION")
    print("=" * 80)
    print()

    # Load domain
    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Create grounding map for TWO transitions
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    grounding_map.add_atom("clear_a", "clear", ["a"])

    # DFA with TWO transitions:
    # state0 --[on_a_b]-> state1 --[clear_a]-> state2
    dfa_dot = """
digraph {
    rankdir=LR;
    node [shape=circle];
    __start [shape=point];
    __start -> state0;
    state0 [label="0"];
    state1 [label="1"];
    state2 [label="2", shape=doublecircle];
    state0 -> state1 [label="on_a_b"];
    state1 -> state2 [label="clear_a"];
}
"""

    dfa_result = {
        "formula": "F(on(a, b) & F(clear(a)))",
        "dfa_dot": dfa_dot,
        "num_states": 3,
        "num_transitions": 2
    }

    ltl_dict = {
        "objects": ["a", "b"],
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    print("DFA Structure:")
    print("  state0 --[on_a_b]-> state1 --[clear_a]-> state2")
    print()
    print("Expected Process:")
    print("  1. Parse DFA → find 2 transitions")
    print("  2. For transition 1 (on_a_b):")
    print("     - Run backward planning from goal 'on(a, b)'")
    print("     - Generate AgentSpeak code for all plans to achieve on(a, b)")
    print("  3. For transition 2 (clear_a):")
    print("     - Run backward planning from goal 'clear(a)'")
    print("     - Generate AgentSpeak code for all plans to achieve clear(a)")
    print("  4. Merge both code sections with separator")
    print()

    # Create generator
    generator = BackwardPlannerGenerator(domain, grounding_map)

    # Parse DFA to show transitions
    print("=" * 80)
    print("PARSING DFA")
    print("=" * 80)
    print()

    dfa_info = generator._parse_dfa(dfa_dot)

    print(f"Found {len(dfa_info.states)} states:")
    for state in dfa_info.states:
        print(f"  - {state}")
    print()

    print(f"Found {len(dfa_info.transitions)} transitions:")
    for i, (from_state, to_state, label) in enumerate(dfa_info.transitions, 1):
        print(f"  {i}. {from_state} --[{label}]-> {to_state}")
    print()

    # Parse each transition label
    print("=" * 80)
    print("PARSING TRANSITION LABELS")
    print("=" * 80)
    print()

    for i, (from_state, to_state, label) in enumerate(dfa_info.transitions, 1):
        print(f"Transition {i}: {from_state} --[{label}]-> {to_state}")

        # Parse label
        goal_disjuncts = generator._parse_transition_label(label)

        for j, goal_predicates in enumerate(goal_disjuncts, 1):
            print(f"  Disjunct {j}:")
            for pred in goal_predicates:
                print(f"    - {pred.to_agentspeak()}")

            goal_name = generator._format_goal_name(goal_predicates)
            print(f"  Goal name: {goal_name}")

        print()

    # Key insight
    print("=" * 80)
    print("KEY INSIGHT: How Code is Merged")
    print("=" * 80)
    print()

    print("The final AgentSpeak code structure will be:")
    print()
    print("/* Header with statistics */")
    print()
    print("/* ========== Goal: on(a, b) ========== */")
    print("// All plans for achieving on(a, b)")
    print("+!on(a, b) : <context1> <- <action1>.")
    print("+!on(a, b) : <context2> <- <action2>.")
    print("...")
    print()
    print("/* ========== Next Goal ========== */")
    print()
    print("/* ========== Goal: clear(a) ========== */")
    print("// All plans for achieving clear(a)")
    print("+!clear(a) : <context1> <- <action1>.")
    print("+!clear(a) : <context2> <- <action2>.")
    print("...")
    print()

    print("Each goal gets its own COMPLETE set of plans.")
    print("The agent can now handle BOTH sequential goals in the DFA.")
    print()

    # Show code section count
    print("=" * 80)
    print("CODE GENERATION STRUCTURE")
    print("=" * 80)
    print()

    print(f"Number of transitions: {len(dfa_info.transitions)}")
    print(f"Number of code sections: {len(dfa_info.transitions)}")
    print()

    print("Each code section contains:")
    print("  - Header comment identifying the goal")
    print("  - All plans (forward destruction results) for that goal")
    print("  - Statistics about state space exploration")
    print()

    print("Why separate sections?")
    print("  - Different DFA transitions = different subgoals")
    print("  - Agent needs plans for achieving EACH subgoal")
    print("  - Plans are context-specific (different world states)")
    print()

    # Explain why it's slow
    print("=" * 80)
    print("WHY FULL EXPLORATION TAKES TIME")
    print("=" * 80)
    print()

    print("For 2 transitions with 2 blocks:")
    print("  Transition 1 (on_a_b): ~1093 states to explore")
    print("  Transition 2 (clear_a): ~1000+ states to explore")
    print("  Total: ~2000+ states across BOTH explorations")
    print()

    print("Each exploration is independent:")
    print("  - Different goal state")
    print("  - Different reachable states")
    print("  - Different plans generated")
    print()

    print("This is CORRECT behavior:")
    print("  ✅ Each DFA transition needs its own plan set")
    print("  ✅ State spaces are different for different goals")
    print("  ✅ No shortcuts possible (must explore completely)")
    print()


if __name__ == "__main__":
    demonstrate_multi_transition_logic()
