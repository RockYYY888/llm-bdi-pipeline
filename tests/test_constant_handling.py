"""
Test constant handling in variable normalization (Issue A)

This test verifies that the normalizer correctly distinguishes between:
- Objects (to be abstracted)
- Constants (to be preserved)
"""

import sys
from pathlib import Path

_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from stage3_code_generation.variable_normalizer import VariableNormalizer
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser


def test_constant_handling():
    """Test that constants are preserved during normalization"""
    print("=" * 80)
    print("CONSTANT HANDLING VERIFICATION (Issue A)")
    print("=" * 80)

    # Load domain
    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    # Create normalizer
    normalizer = VariableNormalizer(domain, ['robot1', 'robot2', 'a', 'b', 'c'])

    # Test cases
    test_cases = [
        {
            "name": "Pure objects",
            "input": [PredicateAtom("on", ["a", "b"])],
            "expected": "on(?arg0, ?arg1)",
            "description": "Both arguments are objects"
        },
        {
            "name": "Object + integer",
            "input": [PredicateAtom("move", ["robot1", "-2"])],
            "expected": "move(?arg0, -2)",
            "description": "Second argument is numeric constant"
        },
        {
            "name": "Object + float",
            "input": [PredicateAtom("move", ["robot1", "3.14"])],
            "expected": "move(?arg0, 3.14)",
            "description": "Second argument is float constant"
        },
        {
            "name": "Object + string literal (single quotes)",
            "input": [PredicateAtom("move", ["robot1", "'Left'"])],
            "expected": "move(?arg0, 'Left')",
            "description": "Second argument is string literal"
        },
        {
            "name": "Object + string literal (double quotes)",
            "input": [PredicateAtom("move", ["robot1", '"Right"'])],
            "expected": 'move(?arg0, "Right")',
            "description": "Second argument is string literal with double quotes"
        },
        {
            "name": "Object + uppercase constant",
            "input": [PredicateAtom("move", ["robot1", "LEFT"])],
            "expected": "move(?arg0, LEFT)",
            "description": "Second argument is uppercase constant"
        },
        {
            "name": "Mixed: Object + number + direction",
            "input": [PredicateAtom("move", ["robot1", "-2", "'Left'"])],
            "expected": "move(?arg0, -2, 'Left')",
            "description": "Mix of object and constants (user's example)"
        },
        {
            "name": "Multiple objects",
            "input": [PredicateAtom("on", ["a", "b"]), PredicateAtom("clear", ["a"])],
            "expected": "clear(?arg0)|on(?arg0, ?arg1)",
            "description": "Multiple predicates with shared object"
        },
        {
            "name": "Objects + constants mixed",
            "input": [
                PredicateAtom("move", ["robot1", "-2", "'Left'"]),
                PredicateAtom("at", ["robot1"])
            ],
            "expected": "at(?arg0)|move(?arg0, -2, 'Left')",
            "description": "Multiple predicates with objects and constants"
        },
    ]

    all_passed = True
    failed_tests = []

    print("\nRunning tests:\n")

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"  Description: {test['description']}")

        # Normalize
        normalized, mapping = normalizer.normalize_predicates(test["input"])
        result = normalizer.serialize_goal(normalized)

        # Show input and output
        input_str = ", ".join([p.to_agentspeak() for p in test["input"]])
        print(f"  Input:    {input_str}")
        print(f"  Output:   {result}")
        print(f"  Expected: {test['expected']}")

        # Check result
        if result == test["expected"]:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
            all_passed = False
            failed_tests.append(test['name'])

        # Show mapping
        if mapping.obj_to_var:
            print(f"  Mapping: {mapping.obj_to_var}")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {len(test_cases) - len(failed_tests)}")
    print(f"Failed: {len(failed_tests)}")

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("   Constants are correctly preserved during normalization.")
        print("   Issue A is FIXED!")
    else:
        print(f"\n❌ {len(failed_tests)} TEST(S) FAILED:")
        for name in failed_tests:
            print(f"   - {name}")
        print("\n   Issue A needs more work.")

    return all_passed


def test_cache_sharing_with_constants():
    """Test that goals with same constants share cache, different constants don't"""
    print("\n" + "=" * 80)
    print("CACHE SHARING WITH CONSTANTS")
    print("=" * 80)

    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))
    normalizer = VariableNormalizer(domain, ['robot1', 'robot2'])

    # Test scenarios
    scenarios = [
        {
            "name": "Same constants, different objects",
            "goals": [
                [PredicateAtom("move", ["robot1", "-2", "'Left'"])],
                [PredicateAtom("move", ["robot2", "-2", "'Left'"])],
            ],
            "should_share": True,
            "reason": "Both have same constants (-2, 'Left'), only objects differ"
        },
        {
            "name": "Different constants",
            "goals": [
                [PredicateAtom("move", ["robot1", "-2", "'Left'"])],
                [PredicateAtom("move", ["robot1", "5", "'Right'"])],
            ],
            "should_share": False,
            "reason": "Different constants (-2 vs 5, 'Left' vs 'Right')"
        },
        {
            "name": "Partial constant difference",
            "goals": [
                [PredicateAtom("move", ["robot1", "-2", "'Left'"])],
                [PredicateAtom("move", ["robot1", "-2", "'Right'"])],
            ],
            "should_share": False,
            "reason": "Same number (-2) but different direction"
        },
    ]

    print("\nTesting cache key generation:\n")

    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"  Reason: {scenario['reason']}")

        keys = []
        for i, goal in enumerate(scenario['goals'], 1):
            normalized, mapping = normalizer.normalize_predicates(goal)
            key = normalizer.serialize_goal(normalized)
            keys.append(key)
            goal_str = ", ".join([p.to_agentspeak() for p in goal])
            print(f"  Goal {i}: {goal_str}")
            print(f"    → {key}")

        # Check if keys match
        keys_match = len(set(keys)) == 1
        expected_match = scenario["should_share"]

        if keys_match == expected_match:
            print(f"  ✅ CORRECT: Keys {'match' if keys_match else 'differ'} (as expected)")
        else:
            print(f"  ❌ WRONG: Keys {'match' if keys_match else 'differ'} (should {'match' if expected_match else 'differ'})")

        print()

    print("=" * 80)


if __name__ == "__main__":
    passed = test_constant_handling()
    test_cache_sharing_with_constants()

    print("\n" + "=" * 80)
    if passed:
        print("✅ Issue A is FIXED - Constants are handled correctly!")
    else:
        print("❌ Issue A needs more investigation")
    print("=" * 80)
