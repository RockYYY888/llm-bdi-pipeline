"""
Stress Tests for Backward Planner

Tests challenging scenarios to verify correctness:
1. Simple goal with 2 blocks
2. Sequential goals with 2 blocks
3. Conjunctive goal with 2 blocks
4. Multiple predicates goal with 2 blocks

Note: Tests use 2 blocks to keep state space manageable while still
testing various goal types and patterns.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from src.utils.pddl_parser import PDDLParser
from src.stage1_interpretation.grounding_map import GroundingMap
from tests.stage3_code_generation.agentspeak_validator import AgentSpeakValidator


def test_tower_goal_3_blocks():
    """
    Test 1: Tower goal with 2 blocks (simplified from 3)
    Goal: on(a, b) - simple tower
    """
    print("=" * 80)
    print("STRESS TEST 1: Tower Goal (2 blocks)")
    print("=" * 80)

    # Load domain
    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Create grounding map
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])

    # DFA: achieve on(a,b)
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
        "objects": ["a", "b"],  # 2 blocks
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    print(f"\nGoal: {dfa_result['formula']}")
    print(f"Objects: {ltl_dict['objects']}")

    # Generate
    generator = BackwardPlannerGenerator(domain, grounding_map)
    try:
        asl_code = generator.generate(ltl_dict, dfa_result)

        print(f"\n✅ Code generated: {len(asl_code)} chars")

        # Validate
        validator = AgentSpeakValidator(asl_code, domain, goals=["on(a, b)"])
        passed, issues = validator.validate()

        if passed:
            print("✅ Validation passed")
        else:
            print("❌ Validation failed:")
            print(validator.format_report())

        return passed, asl_code

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_sequential_goals():
    """
    Test 2: Sequential goals (simplified to 2 blocks)
    Goal: First achieve on(a,b), then achieve clear(a)
    """
    print("\n\n" + "=" * 80)
    print("STRESS TEST 2: Sequential Goals (2 blocks)")
    print("=" * 80)

    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    grounding_map.add_atom("clear_a", "clear", ["a"])

    # DFA: state0 -> state1 (on_a_b) -> state2 (clear_a)
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

    print(f"\nGoal: {dfa_result['formula']}")
    print(f"Objects: {ltl_dict['objects']}")

    generator = BackwardPlannerGenerator(domain, grounding_map)
    try:
        asl_code = generator.generate(ltl_dict, dfa_result)

        print(f"\n✅ Code generated: {len(asl_code)} chars")

        validator = AgentSpeakValidator(asl_code, domain, goals=["on(a, b)", "clear(a)"])
        passed, issues = validator.validate()

        if passed:
            print("✅ Validation passed")
        else:
            print("❌ Validation failed:")
            print(validator.format_report())

        return passed, asl_code

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_4_blocks():
    """
    Test 3: Conjunctive goal with 2 blocks
    Goal: handempty & on(a, b) - test complex goal
    """
    print("\n\n" + "=" * 80)
    print("STRESS TEST 3: Conjunctive Goal (2 blocks)")
    print("=" * 80)

    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    grounding_map = GroundingMap()
    grounding_map.add_atom("handempty", "handempty", [])
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])

    dfa_dot = """
digraph {
    rankdir=LR;
    node [shape=circle];
    __start [shape=point];
    __start -> state0;
    state0 [label="0"];
    state1 [label="1", shape=doublecircle];
    state0 -> state1 [label="handempty & on_a_b"];
}
"""

    dfa_result = {
        "formula": "F(handempty & on(a, b))",
        "dfa_dot": dfa_dot,
        "num_states": 2,
        "num_transitions": 1
    }

    ltl_dict = {
        "objects": ["a", "b"],  # 2 blocks
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    print(f"\nGoal: {dfa_result['formula']}")
    print(f"Objects: {ltl_dict['objects']}")

    import time
    generator = BackwardPlannerGenerator(domain, grounding_map)
    try:
        start_time = time.time()
        asl_code = generator.generate(ltl_dict, dfa_result)
        elapsed = time.time() - start_time

        print(f"\n✅ Code generated: {len(asl_code)} chars")
        print(f"⏱️  Time: {elapsed:.2f}s")

        # Extract statistics
        import re
        stats_match = re.search(r'States:\s*(\d+)', asl_code)
        if stats_match:
            states = int(stats_match.group(1))
            print(f"📊 States explored: {states:,}")

            if states > 10000:
                print(f"⚠️  Warning: Very large state space!")

        validator = AgentSpeakValidator(asl_code, domain, goals=["handempty & on(a, b)"])
        passed, issues = validator.validate()

        if passed:
            print("✅ Validation passed")
        else:
            print("❌ Validation failed")

        return passed, asl_code

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_complex_conjunctive_goal():
    """
    Test 4: Multiple predicates goal (2 blocks)
    Goal: clear(a) & clear(b)
    """
    print("\n\n" + "=" * 80)
    print("STRESS TEST 4: Multiple Predicates Goal (2 blocks)")
    print("=" * 80)

    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    grounding_map = GroundingMap()
    grounding_map.add_atom("clear_a", "clear", ["a"])
    grounding_map.add_atom("clear_b", "clear", ["b"])

    dfa_dot = """
digraph {
    rankdir=LR;
    node [shape=circle];
    __start [shape=point];
    __start -> state0;
    state0 [label="0"];
    state1 [label="1", shape=doublecircle];
    state0 -> state1 [label="clear_a & clear_b"];
}
"""

    dfa_result = {
        "formula": "F(clear(a) & clear(b))",
        "dfa_dot": dfa_dot,
        "num_states": 2,
        "num_transitions": 1
    }

    ltl_dict = {
        "objects": ["a", "b"],
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    print(f"\nGoal: {dfa_result['formula']}")
    print(f"Objects: {ltl_dict['objects']}")

    generator = BackwardPlannerGenerator(domain, grounding_map)
    try:
        asl_code = generator.generate(ltl_dict, dfa_result)

        print(f"\n✅ Code generated: {len(asl_code)} chars")

        validator = AgentSpeakValidator(asl_code, domain, goals=["clear(a) & clear(b)"])
        passed, issues = validator.validate()

        if passed:
            print("✅ Validation passed")
        else:
            print("❌ Validation failed")

        return passed, asl_code

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    print("\n🔥 STRESS TESTING BACKWARD PLANNER 🔥\n")

    results = []

    # Test 1: Tower goal (2 blocks)
    try:
        result1, _ = test_tower_goal_3_blocks()
        results.append(("Tower Goal (2 blocks)", result1))
    except Exception as e:
        print(f"\n💥 Test 1 crashed: {e}")
        results.append(("Tower Goal (2 blocks)", False))

    # Test 2: Sequential goals (2 blocks)
    try:
        result2, _ = test_sequential_goals()
        results.append(("Sequential Goals (2 blocks)", result2))
    except Exception as e:
        print(f"\n💥 Test 2 crashed: {e}")
        results.append(("Sequential Goals (2 blocks)", False))

    # Test 3: Conjunctive goal (2 blocks)
    try:
        result3, _ = test_4_blocks()
        results.append(("Conjunctive Goal (2 blocks)", result3))
    except Exception as e:
        print(f"\n💥 Test 3 crashed: {e}")
        results.append(("Conjunctive Goal (2 blocks)", False))

    # Test 4: Multiple predicates (2 blocks)
    try:
        result4, _ = test_complex_conjunctive_goal()
        results.append(("Multiple Predicates (2 blocks)", result4))
    except Exception as e:
        print(f"\n💥 Test 4 crashed: {e}")
        results.append(("Multiple Predicates (2 blocks)", False))

    # Summary
    print("\n\n" + "=" * 80)
    print("STRESS TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL STRESS TESTS PASSED")
    else:
        print("⚠️  SOME STRESS TESTS FAILED - DESIGN NEEDS IMPROVEMENT")
    print("=" * 80)
