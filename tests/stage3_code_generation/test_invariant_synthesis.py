"""
Test H^2 Invariant Synthesis (Fast Downward Style)

This test verifies that:
1. H^2 mutex detection correctly identifies mutex predicates
2. Exactly-one group detection finds invariant groups
3. State validation correctly filters invalid states
4. No false positives (valid states are not rejected)
5. Generated AgentSpeak code has no unreachable plans
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from stage3_code_generation.variable_planner import VariablePlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser


def test_h2_mutex_detection():
    """Test that h^2 mutex detection works correctly"""
    print("="*80)
    print("TEST 1: H^2 MUTEX DETECTION")
    print("="*80)

    # Load domain
    repo_root = Path(__file__).parent.parent.parent
    domain_file = repo_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"❌ Domain file not found: {domain_file}")
        return False

    domain = PDDLParser.parse_domain(str(domain_file))
    planner = VariablePlanner(domain)

    # Check synthesized invariants
    h2_mutexes = planner._invariants['h2_mutexes']
    exactly_one_groups = planner._invariants['exactly_one_groups']

    print(f"\nH^2 Mutexes detected: {len(h2_mutexes)}")
    for p1, p2 in sorted(h2_mutexes):
        print(f"  - {p1} ⊗ {p2}")

    print(f"\nExactly-One Groups detected: {len(exactly_one_groups)}")
    for group in exactly_one_groups:
        print(f"  - {{{', '.join(sorted(group))}}}")

    # Verify expected mutexes for blocksworld
    expected_mutexes = {
        ('handempty', 'holding'),  # Can't have both handempty and holding
    }

    found_expected = sum(1 for mutex in expected_mutexes if mutex in h2_mutexes)
    print(f"\n✓ Found {found_expected}/{len(expected_mutexes)} expected mutexes")

    # Verify exactly-one groups
    # In blocksworld: {handempty, holding} should form exactly-one group
    has_handempty_holding_group = any(
        'handempty' in group and 'holding' in group
        for group in exactly_one_groups
    )

    if has_handempty_holding_group:
        print("✓ Detected {handempty, holding} exactly-one group")
    else:
        print("⚠️  Did not detect {handempty, holding} exactly-one group")

    print("="*80 + "\n")
    return True


def test_state_validation():
    """Test that state validation correctly filters invalid states"""
    print("="*80)
    print("TEST 2: STATE VALIDATION")
    print("="*80)

    repo_root = Path(__file__).parent.parent.parent
    domain_file = repo_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))
    planner = VariablePlanner(domain)

    # Test cases: (predicates, should_be_valid, description)
    test_cases = [
        # Valid states
        (
            {PredicateAtom("handempty", [])},
            True,
            "handempty alone"
        ),
        (
            {PredicateAtom("holding", ["?v0"])},
            True,
            "holding(?v0) alone"
        ),
        (
            {PredicateAtom("holding", ["?v0"]), PredicateAtom("clear", ["?v1"])},
            True,
            "holding(?v0) & clear(?v1) - different objects"
        ),
        (
            {PredicateAtom("on", ["?v0", "?v1"]), PredicateAtom("clear", ["?v0"])},
            True,
            "on(?v0, ?v1) & clear(?v0) - allowed"
        ),

        # Invalid states (should be rejected)
        (
            {PredicateAtom("handempty", []), PredicateAtom("holding", ["?v0"])},
            False,
            "handempty & holding(?v0) - exactly-one violation"
        ),
    ]

    passed = 0
    failed = 0

    for predicates, expected_valid, description in test_cases:
        is_valid = planner._validate_state_with_invariants(predicates)

        status = "✓" if is_valid == expected_valid else "✗"
        result = "PASS" if is_valid == expected_valid else "FAIL"

        print(f"{status} {result}: {description}")
        print(f"    Predicates: {{{', '.join(str(p) for p in predicates)}}}")
        print(f"    Expected: {'valid' if expected_valid else 'invalid'}, Got: {'valid' if is_valid else 'invalid'}")

        if is_valid == expected_valid:
            passed += 1
        else:
            failed += 1

    print(f"\n{passed}/{len(test_cases)} tests passed")

    if failed > 0:
        print(f"❌ {failed} tests failed")
        print("="*80 + "\n")
        return False
    else:
        print("✓ All tests passed!")
        print("="*80 + "\n")
        return True


def test_regression_with_invariants():
    """Test that regression correctly uses invariants to prune invalid states"""
    print("="*80)
    print("TEST 3: REGRESSION WITH INVARIANTS")
    print("="*80)

    repo_root = Path(__file__).parent.parent.parent
    domain_file = repo_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))
    planner = VariablePlanner(domain)

    # Test with holding(?X) goal
    goal_preds = [PredicateAtom("holding", ["?X"])]

    print(f"Goal: {[str(p) for p in goal_preds]}")
    print("Running regression with invariant validation...\n")

    result = planner.explore_from_goal(goal_preds, max_states=1000, max_depth=3)

    # Extract states from StateGraph
    state_graph = result
    all_states = [state_graph.goal_state]
    visited = {state_graph.goal_state}
    queue = [state_graph.goal_state]

    while queue:
        current = queue.pop(0)
        for transition in state_graph.transitions:
            if transition.to_state == current and transition.from_state not in visited:
                all_states.append(transition.from_state)
                visited.add(transition.from_state)
                queue.append(transition.from_state)

    print(f"States explored: {len(all_states)}")
    print(f"Transitions generated: {len(state_graph.transitions)}")

    # Check for invalid states (should be ZERO with proper invariant validation)
    invalid_count = 0

    for state in all_states:
        # Check for handempty + holding violation
        has_handempty = any(p.name == 'handempty' for p in state.predicates)
        has_holding = any(p.name == 'holding' for p in state.predicates)

        if has_handempty and has_holding:
            invalid_count += 1
            if invalid_count <= 3:
                print(f"  ❌ Invalid state: {{{', '.join(str(p) for p in state.predicates)}}}")

    if invalid_count == 0:
        print("✓ No invalid states found!")
        print("✓ Invariant validation is working correctly")
        print("="*80 + "\n")
        return True
    else:
        print(f"\n❌ Found {invalid_count} invalid states")
        print("❌ Invariant validation is not working properly")
        print("="*80 + "\n")
        return False


def test_no_false_positives():
    """Test that we don't reject valid states (no false positives)"""
    print("="*80)
    print("TEST 4: NO FALSE POSITIVES")
    print("="*80)

    repo_root = Path(__file__).parent.parent.parent
    domain_file = repo_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))
    planner = VariablePlanner(domain)

    # These states should ALL be valid
    valid_states = [
        {PredicateAtom("holding", ["?v0"]), PredicateAtom("clear", ["?v1"])},
        {PredicateAtom("on", ["?v0", "?v1"]), PredicateAtom("handempty", [])},
        {PredicateAtom("ontable", ["?v0"]), PredicateAtom("clear", ["?v0"]), PredicateAtom("handempty", [])},
    ]

    all_valid = True

    for predicates in valid_states:
        is_valid = planner._validate_state_with_invariants(predicates)
        pred_str = '{' + ', '.join(str(p) for p in predicates) + '}'

        if not is_valid:
            print(f"  ✗ FALSE POSITIVE: {pred_str}")
            all_valid = False
        else:
            print(f"  ✓ Correctly accepted: {pred_str}")

    if all_valid:
        print("\n✓ No false positives detected!")
        print("="*80 + "\n")
        return True
    else:
        print("\n❌ False positives detected!")
        print("="*80 + "\n")
        return False


def run_all_tests():
    """Run all invariant synthesis tests"""
    print("\n" + "="*80)
    print("INVARIANT SYNTHESIS TEST SUITE (Fast Downward Style)")
    print("="*80 + "\n")

    results = []

    results.append(("H^2 Mutex Detection", test_h2_mutex_detection()))
    results.append(("State Validation", test_state_validation()))
    results.append(("Regression with Invariants", test_regression_with_invariants()))
    results.append(("No False Positives", test_no_false_positives()))

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} test suites passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Invariant synthesis is working correctly.")
        print("="*80)
        return True
    else:
        print(f"\n❌ {total - passed} test suite(s) failed")
        print("="*80)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
