"""
Test semantic-based constant detection (object_list-based approach)

This test verifies that the normalizer uses SEMANTIC information (object_list)
rather than syntactic patterns (uppercase/lowercase) to distinguish objects
from constants.

Addresses the critical bug where purely uppercase object names (e.g., "ROBOT")
were incorrectly treated as constants.
"""

import sys
from pathlib import Path

_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from stage3_code_generation.variable_normalizer import VariableNormalizer
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser


def test_uppercase_object_names():
    """Test that uppercase object names are correctly abstracted"""
    print("=" * 80)
    print("TEST: Uppercase Object Names (Critical Bug Fix)")
    print("=" * 80)

    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    # Test with purely uppercase object names
    objects = ['ROBOT', 'BLOCK', 'TABLE']
    normalizer = VariableNormalizer(domain, objects)

    test_cases = [
        {
            "input": PredicateAtom("holding", ["ROBOT"]),
            "expected": "holding(?arg0)",
            "desc": "Single uppercase object"
        },
        {
            "input": PredicateAtom("on", ["BLOCK", "TABLE"]),
            "expected": "on(?arg0, ?arg1)",
            "desc": "Multiple uppercase objects"
        },
    ]

    all_passed = True
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['desc']}")
        normalized, mapping = normalizer.normalize_predicates([test["input"]])
        actual = normalized[0].to_agentspeak()
        print(f"   Input:    {test['input'].to_agentspeak()}")
        print(f"   Output:   {actual}")
        print(f"   Expected: {test['expected']}")
        print(f"   Mapping:  {mapping.obj_to_var}")

        if actual == test["expected"]:
            print("   ✅ PASS")
        else:
            print("   ❌ FAIL")
            all_passed = False

    return all_passed


def test_mixed_case_objects():
    """Test various naming conventions for objects"""
    print("\n" + "=" * 80)
    print("TEST: Mixed Case and Special Characters in Object Names")
    print("=" * 80)

    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    test_cases = [
        {
            "objects": ["robot_1", "robot_2"],
            "pred": PredicateAtom("at", ["robot_1"]),
            "expected": "at(?arg0)",
            "desc": "Snake_case object names"
        },
        {
            "objects": ["Block1", "Block2"],
            "pred": PredicateAtom("on", ["Block1", "Block2"]),
            "expected": "on(?arg0, ?arg1)",
            "desc": "PascalCase object names"
        },
        {
            "objects": ["obj-A", "obj-B"],
            "pred": PredicateAtom("holding", ["obj-A"]),
            "expected": "holding(?arg0)",
            "desc": "Hyphenated object names"
        },
    ]

    all_passed = True
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['desc']}")
        normalizer = VariableNormalizer(domain, test["objects"])
        normalized, mapping = normalizer.normalize_predicates([test["pred"]])
        actual = normalized[0].to_agentspeak()

        print(f"   Objects:  {test['objects']}")
        print(f"   Input:    {test['pred'].to_agentspeak()}")
        print(f"   Output:   {actual}")
        print(f"   Expected: {test['expected']}")

        if actual == test["expected"]:
            print("   ✅ PASS")
        else:
            print("   ❌ FAIL")
            all_passed = False

    return all_passed


def test_domain_constants_vs_objects():
    """Test distinction between domain constants and problem objects"""
    print("\n" + "=" * 80)
    print("TEST: Domain Constants vs Problem Objects")
    print("=" * 80)

    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    print("\nScenario: Objects are lowercase, constants are uppercase")
    print("(This is just one convention - system should handle any convention)")

    objects = ["robot1", "a", "b"]
    normalizer = VariableNormalizer(domain, objects)

    # Simulate predicates with domain constants
    test_cases = [
        {
            "pred": PredicateAtom("move", ["robot1", "-2", "LEFT"]),
            "expected": "move(?arg0, -2, LEFT)",
            "desc": "Object + numeric + direction constant"
        },
        {
            "pred": PredicateAtom("at", ["robot1", "HOME"]),
            "expected": "at(?arg0, HOME)",
            "desc": "Object + location constant"
        },
        {
            "pred": PredicateAtom("on", ["a", "GROUND"]),
            "expected": "on(?arg0, GROUND)",
            "desc": "Object on named constant"
        },
    ]

    all_passed = True
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['desc']}")
        normalized, mapping = normalizer.normalize_predicates([test["pred"]])
        actual = normalized[0].to_agentspeak()

        print(f"   Input:    {test['pred'].to_agentspeak()}")
        print(f"   Output:   {actual}")
        print(f"   Expected: {test['expected']}")
        print(f"   Mapping:  {mapping.obj_to_var}")

        if actual == test["expected"]:
            print("   ✅ PASS")
        else:
            print("   ❌ FAIL")
            all_passed = False

    return all_passed


def test_object_list_is_ground_truth():
    """Test that object_list is the definitive source of truth"""
    print("\n" + "=" * 80)
    print("TEST: object_list as Ground Truth")
    print("=" * 80)

    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    print("\nPrinciple: Only what's in object_list gets abstracted")
    print("Everything else is preserved (constants, literals, unknowns)")

    # Scenario 1: 'c' not in object_list → preserved
    print("\n1. Object not in list → preserved")
    objects_1 = ["a", "b"]
    normalizer_1 = VariableNormalizer(domain, objects_1)
    pred_1 = PredicateAtom("on", ["a", "c"])
    normalized_1, mapping_1 = normalizer_1.normalize_predicates([pred_1])
    print(f"   Objects:  {objects_1}")
    print(f"   Input:    on(a, c)")
    print(f"   Output:   {normalized_1[0].to_agentspeak()}")
    print(f"   Expected: on(?arg0, c)")
    passed_1 = normalized_1[0].to_agentspeak() == "on(?arg0, c)"
    print(f"   {'✅ PASS' if passed_1 else '❌ FAIL'}")

    # Scenario 2: Now 'c' is in object_list → abstracted
    print("\n2. Same argument, now in object_list → abstracted")
    objects_2 = ["a", "b", "c"]
    normalizer_2 = VariableNormalizer(domain, objects_2)
    pred_2 = PredicateAtom("on", ["a", "c"])
    normalized_2, mapping_2 = normalizer_2.normalize_predicates([pred_2])
    print(f"   Objects:  {objects_2}")
    print(f"   Input:    on(a, c)")
    print(f"   Output:   {normalized_2[0].to_agentspeak()}")
    print(f"   Expected: on(?arg0, ?arg1)")
    passed_2 = normalized_2[0].to_agentspeak() == "on(?arg0, ?arg1)"
    print(f"   {'✅ PASS' if passed_2 else '❌ FAIL'}")

    # Scenario 3: Even "obvious constants" get abstracted if in object_list
    print("\n3. Even 'obvious constant' names get abstracted if in object_list")
    objects_3 = ["LEFT", "RIGHT", "UP"]  # Looks like constants, but in object_list!
    normalizer_3 = VariableNormalizer(domain, objects_3)
    pred_3 = PredicateAtom("next_to", ["LEFT", "RIGHT"])
    normalized_3, mapping_3 = normalizer_3.normalize_predicates([pred_3])
    print(f"   Objects:  {objects_3}")
    print(f"   Input:    next_to(LEFT, RIGHT)")
    print(f"   Output:   {normalized_3[0].to_agentspeak()}")
    print(f"   Expected: next_to(?arg0, ?arg1)")
    print(f"   (LEFT & RIGHT are in object_list → abstracted)")
    passed_3 = normalized_3[0].to_agentspeak() == "next_to(?arg0, ?arg1)"
    print(f"   {'✅ PASS' if passed_3 else '❌ FAIL'}")

    return passed_1 and passed_2 and passed_3


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + " SEMANTIC CONSTANT DETECTION TESTS (object_list-based)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = []
    results.append(("Uppercase Object Names", test_uppercase_object_names()))
    results.append(("Mixed Case Objects", test_mixed_case_objects()))
    results.append(("Domain Constants vs Objects", test_domain_constants_vs_objects()))
    results.append(("object_list as Ground Truth", test_object_list_is_ground_truth()))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_passed = all(passed for _, passed in results)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8s} : {name}")

    print("=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Semantic detection working correctly!")
        print()
        print("Key Principle Verified:")
        print("  object_list is the GROUND TRUTH for abstraction decisions")
        print("  Naming conventions (case, special chars) are irrelevant")
        print("  System works with ANY naming convention")
    else:
        print("❌ SOME TESTS FAILED - See details above")
    print("=" * 80)

    sys.exit(0 if all_passed else 1)
