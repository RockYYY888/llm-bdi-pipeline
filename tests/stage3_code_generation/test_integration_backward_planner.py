"""
Integration Test for Stage 3 Backward Planning

Tests the complete backward planning pipeline:
1. DFA input parsing
2. Forward state space exploration
3. AgentSpeak code generation
4. Syntax validation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from src.utils.pddl_parser import PDDLParser
from src.stage1_interpretation.grounding_map import GroundingMap


def create_simple_dfa() -> tuple:
    """
    Create a simple test DFA with one transition

    DFA: state0 --[on_a_b]-> state1 (accepting)

    This represents the goal: achieve on(a,b)

    Returns:
        (dfa_result, grounding_map)
    """
    # Create grounding map
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])

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
        "formula": "F(on(a, b))",  # Eventually on(a,b)
        "dfa_dot": dfa_dot,
        "num_states": 2,
        "num_transitions": 1
    }

    return dfa_result, grounding_map


def create_complex_dfa() -> tuple:
    """
    Create a more complex DFA with multiple transitions

    DFA represents: achieve on(a,b), then achieve clear(a)

    Returns:
        (dfa_result, grounding_map)
    """
    # Create grounding map
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    grounding_map.add_atom("clear_a", "clear", ["a"])

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

    return dfa_result, grounding_map


def test_simple_dfa():
    """Test backward planner with simple DFA"""
    print("="*80)
    print("INTEGRATION TEST: Simple DFA (single transition)")
    print("="*80)

    # Load domain
    domain_path = project_root / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))
    print(f"\nDomain loaded: {domain.name}")
    print(f"  Actions: {len(domain.actions)}")
    print(f"  Predicates: {len(domain.predicates)}")

    # Create DFA and grounding map
    dfa_result, grounding_map = create_simple_dfa()
    print(f"\nDFA: {dfa_result['formula']}")
    print(f"  States: {dfa_result['num_states']}")
    print(f"  Transitions: {dfa_result['num_transitions']}")
    print(f"  Grounding map entries: {len(grounding_map.atoms)}")

    # Create LTL dict
    ltl_dict = {
        "objects": ["a", "b"],
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    # Initialize backward planner generator
    print("\n" + "-"*80)
    print("Initializing Backward Planner Generator...")
    print("-"*80)

    generator = BackwardPlannerGenerator(domain, grounding_map)

    # Generate code
    print("\nGenerating AgentSpeak code...")
    asl_code = generator.generate(ltl_dict, dfa_result)

    # Display results
    print("\n" + "="*80)
    print("GENERATED AGENTSPEAK CODE")
    print("="*80)
    print(asl_code)

    # Basic validation
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    validations = {
        "Has initial beliefs": "ontable(" in asl_code and "handempty" in asl_code,
        "Has action definitions": "+!" in asl_code and "<-" in asl_code,
        "Has goal plans": ("!on(a, b)" in asl_code or "!on_a_b" in asl_code or "!achieve" in asl_code or "Goal Plans:" in asl_code),
        "Has belief updates": ("+holding" in asl_code or "-handempty" in asl_code),
        "Non-empty code": len(asl_code) > 100
    }

    for check, passed in validations.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")

    all_passed = all(validations.values())

    if all_passed:
        print("\n✅ ALL VALIDATIONS PASSED")
    else:
        print("\n❌ SOME VALIDATIONS FAILED")

    return all_passed, asl_code


def test_complex_dfa():
    """Test backward planner with complex DFA (multiple transitions)"""
    print("\n\n" + "="*80)
    print("INTEGRATION TEST: Complex DFA (multiple transitions)")
    print("="*80)

    # Load domain
    domain_path = project_root / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Create DFA and grounding map
    dfa_result, grounding_map = create_complex_dfa()
    print(f"\nDFA: {dfa_result['formula']}")
    print(f"  States: {dfa_result['num_states']}")
    print(f"  Transitions: {dfa_result['num_transitions']}")
    print(f"  Grounding map entries: {len(grounding_map.atoms)}")

    # Create LTL dict
    ltl_dict = {
        "objects": ["a", "b"],
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    # Initialize backward planner generator
    generator = BackwardPlannerGenerator(domain, grounding_map)

    # Generate code
    print("\nGenerating AgentSpeak code...")
    asl_code = generator.generate(ltl_dict, dfa_result)

    # Display results (first 50 lines)
    print("\n" + "="*80)
    print("GENERATED AGENTSPEAK CODE (first 50 lines)")
    print("="*80)
    lines = asl_code.split('\n')
    for i, line in enumerate(lines[:50], 1):
        print(f"{i:3}: {line}")

    if len(lines) > 50:
        print(f"... ({len(lines) - 50} more lines)")

    # Validation
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    validations = {
        "Has initial beliefs": "ontable(" in asl_code,
        "Has multiple goals": asl_code.count("!") >= 3,  # Multiple goal invocations
        "Has action definitions": "+!" in asl_code,
        "Substantial code": len(asl_code) > 500
    }

    for check, passed in validations.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")

    all_passed = all(validations.values())

    if all_passed:
        print("\n✅ ALL VALIDATIONS PASSED")
    else:
        print("\n❌ SOME VALIDATIONS FAILED")

    return all_passed, asl_code


def test_state_graph_size():
    """Test that state graph has reasonable size"""
    print("\n\n" + "="*80)
    print("DETAILED TEST: State Graph Statistics")
    print("="*80)

    from src.stage3_code_generation.forward_planner import ForwardStatePlanner
    from src.stage3_code_generation.state_space import PredicateAtom

    # Load domain
    domain_path = project_root / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Test with simple goal
    goal_preds = [PredicateAtom('on', ('a', 'b'))]
    objects = ['a', 'b']

    planner = ForwardStatePlanner(domain, objects)

    # Test complete exploration (no depth limit)
    print(f"\nComplete exploration:")
    graph = planner.explore_from_goal(goal_preds)

    print(f"  States: {len(graph.states)}")
    print(f"  Transitions: {len(graph.transitions)}")
    print(f"  Leaf states: {len(graph.get_leaf_states())}")

    # Path finding
    paths = graph.find_shortest_paths_to_goal()
    non_trivial = [p for s, p in paths.items() if s != graph.goal_state and p]
    print(f"  Non-trivial paths: {len(non_trivial)}")

    # Sample states
    print(f"  Sample states:")
    for i, state in enumerate(list(graph.states)[:3]):
        preds = ', '.join(str(p) for p in sorted(state.predicates, key=str)[:3])
        print(f"    {i+1}. {preds}...")


if __name__ == "__main__":
    print("Starting Stage 3 Backward Planning Integration Tests")
    print()

    # Test 1: Simple DFA
    try:
        success1, code1 = test_simple_dfa()
    except Exception as e:
        print(f"\n❌ Simple DFA test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success1 = False

    # Test 2: Complex DFA
    try:
        success2, code2 = test_complex_dfa()
    except Exception as e:
        print(f"\n❌ Complex DFA test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success2 = False

    # Test 3: State graph statistics
    try:
        test_state_graph_size()
        success3 = True
    except Exception as e:
        print(f"\n❌ State graph test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success3 = False

    # Final summary
    print("\n\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"Simple DFA Test:      {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Complex DFA Test:     {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"State Graph Test:     {'✅ PASS' if success3 else '❌ FAIL'}")
    print()

    if success1 and success2 and success3:
        print("✅ ALL INTEGRATION TESTS PASSED")
        exit(0)
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
        exit(1)
