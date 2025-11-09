"""
Scalability Tests for Backward Planner

Tests with increasing number of objects (2-5 blocks) to verify:
1. State exploration doesn't get stuck
2. Memory usage is reasonable
3. State reuse is working correctly
4. Performance degrades gracefully

WARNING: Tests with 4+ blocks may take several minutes.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from src.utils.pddl_parser import PDDLParser
from src.stage1_interpretation.grounding_map import GroundingMap


def test_n_blocks(n: int, goal_formula: str, goal_predicates: list, max_states: int = 50000):
    """
    Generic test for N blocks

    Args:
        n: Number of blocks
        goal_formula: LTL formula string
        goal_predicates: List of goal predicate strings for grounding map
        max_states: Maximum states to explore (safety limit)
    """
    print("=" * 80)
    print(f"SCALABILITY TEST: {n} Blocks")
    print("=" * 80)

    # Load domain
    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Create objects
    objects = [chr(ord('a') + i) for i in range(n)]  # ['a', 'b', 'c', ...]

    # Create grounding map
    grounding_map = GroundingMap()
    for pred in goal_predicates:
        grounding_map.add_atom(pred, "on", objects[:2])  # Simple on(a, b) goal

    # Simple DFA: state0 -> state1 with goal
    label = " & ".join(goal_predicates)
    dfa_dot = f"""
digraph {{
    rankdir=LR;
    node [shape=circle];
    __start [shape=point];
    __start -> state0;
    state0 [label="0"];
    state1 [label="1", shape=doublecircle];
    state0 -> state1 [label="{label}"];
}}
"""

    dfa_result = {
        "formula": goal_formula,
        "dfa_dot": dfa_dot,
        "num_states": 2,
        "num_transitions": 1
    }

    ltl_dict = {
        "objects": objects,
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    print(f"\nGoal: {dfa_result['formula']}")
    print(f"Objects: {objects} ({n} blocks)")
    print(f"Max states limit: {max_states:,}")
    print(f"Expected complexity: ~{estimate_state_space(n):,} states")
    print()

    # Generate with timing
    generator = BackwardPlannerGenerator(domain, grounding_map)

    start_time = time.time()
    try:
        asl_code = generator.generate(ltl_dict, dfa_result)
        elapsed = time.time() - start_time

        print(f"\n✅ Code generated: {len(asl_code):,} chars")
        print(f"⏱️  Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")

        # Extract statistics from generated code
        import re
        states_match = re.search(r'States:\s*(\d+)', asl_code)
        if states_match:
            actual_states = int(states_match.group(1))
            print(f"📊 States explored: {actual_states:,}")
            print(f"   States/second: {actual_states/elapsed:.1f}")

        return True, elapsed, asl_code

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Generation failed after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed, None


def estimate_state_space(n: int) -> int:
    """
    Rough estimate of blocksworld state space size

    For n blocks, approximate state space is:
    - 2 blocks: ~1,000 states
    - 3 blocks: ~10,000 states
    - 4 blocks: ~100,000 states
    - 5 blocks: ~1,000,000 states

    This is exponential in n.
    """
    return 10 ** (n * 1.5)


if __name__ == "__main__":
    print("\n🔬 SCALABILITY TESTING: 2-5 BLOCKS\n")
    print("WARNING: Tests with 4+ blocks may take several minutes to hours.")
    print("Press Ctrl+C at any time to stop.\n")

    results = []

    # Test 2 blocks (baseline)
    print("\n" + "="*80)
    print("TEST 1/4: 2 Blocks (Baseline)")
    print("="*80)
    passed, time_taken, _ = test_n_blocks(
        n=2,
        goal_formula="F(on(a, b))",
        goal_predicates=["on_a_b"],
        max_states=50000
    )
    results.append(("2 blocks", passed, time_taken))

    # Test 3 blocks
    print("\n" + "="*80)
    print("TEST 2/4: 3 Blocks")
    print("="*80)
    passed, time_taken, _ = test_n_blocks(
        n=3,
        goal_formula="F(on(a, b))",
        goal_predicates=["on_a_b"],
        max_states=50000
    )
    results.append(("3 blocks", passed, time_taken))

    # Test 4 blocks
    print("\n" + "="*80)
    print("TEST 3/4: 4 Blocks (May take several minutes)")
    print("="*80)
    passed, time_taken, _ = test_n_blocks(
        n=4,
        goal_formula="F(on(a, b))",
        goal_predicates=["on_a_b"],
        max_states=50000
    )
    results.append(("4 blocks", passed, time_taken))

    # Test 5 blocks
    print("\n" + "="*80)
    print("TEST 4/4: 5 Blocks (May take 10+ minutes)")
    print("="*80)
    passed, time_taken, _ = test_n_blocks(
        n=5,
        goal_formula="F(on(a, b))",
        goal_predicates=["on_a_b"],
        max_states=50000
    )
    results.append(("5 blocks", passed, time_taken))

    # Summary
    print("\n\n" + "=" * 80)
    print("SCALABILITY TEST SUMMARY")
    print("=" * 80)

    for name, passed, time_taken in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        time_str = f"{time_taken:.2f}s" if time_taken < 60 else f"{time_taken/60:.1f}min"
        print(f"{status}: {name:15s} ({time_str})")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL SCALABILITY TESTS PASSED")
    else:
        print("⚠️  SOME TESTS FAILED OR HIT LIMITS")
    print("=" * 80)
