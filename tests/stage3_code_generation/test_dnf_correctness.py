#!/usr/bin/env python3
"""
Deep test of DNF conversion correctness for complex nested expressions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.boolean_expression_parser import BooleanExpressionParser
from stage1_interpretation.grounding_map import GroundingMap


def create_test_grounding_map():
    """Create grounding map for testing"""
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("on_c_d", "on", ["c", "d"])
    gmap.add_atom("on_d_e", "on", ["d", "e"])
    gmap.add_atom("clear_a", "clear", ["a"])
    gmap.add_atom("clear_c", "clear", ["c"])
    gmap.add_atom("holding_d", "holding", ["d"])
    gmap.add_atom("handempty", "handempty", [])
    return gmap


def test_dnf_conversion(expr_str: str, gmap: GroundingMap):
    """Test DNF conversion for a single expression"""
    print(f"\n{'='*80}")
    print(f"Expression: {expr_str}")
    print(f"{'='*80}")

    parser = BooleanExpressionParser(gmap)

    # Parse to DNF
    try:
        dnf = parser.parse(expr_str)
        print(f"✓ DNF generated: {len(dnf)} disjunct(s)")

        for i, conjunction in enumerate(dnf, 1):
            pred_strs = [str(p) for p in conjunction]
            print(f"  Disjunct {i}: [{', '.join(pred_strs)}]")

        return dnf
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_dnf_correctness(expr_str: str, expected_disjuncts: int, gmap: GroundingMap):
    """Verify DNF conversion produces expected number of disjuncts"""
    dnf = test_dnf_conversion(expr_str, gmap)

    if dnf is None:
        print(f"  ✗ FAILED: Parse error")
        return False

    if len(dnf) != expected_disjuncts:
        print(f"  ✗ FAILED: Expected {expected_disjuncts} disjuncts, got {len(dnf)}")
        return False

    print(f"  ✓ PASSED: Correct number of disjuncts")
    return True


def main():
    print("="*80)
    print("DNF Conversion Correctness Test Suite")
    print("="*80)

    gmap = create_test_grounding_map()

    test_cases = [
        # (expression, expected_disjunct_count, description)

        # Level 1: Simple cases
        ("on_a_b", 1, "Single literal"),
        ("on_a_b | clear_c", 2, "Simple OR"),
        ("on_a_b & clear_c", 1, "Simple AND"),

        # Level 2: Negation
        ("~on_a_b", 1, "Negated literal"),
        ("~on_a_b | clear_c", 2, "OR with negation"),
        ("~on_a_b & ~clear_c", 1, "AND with negations"),

        # Level 3: Nested OR in AND (Distribution required)
        ("(on_a_b | clear_c) & holding_d", 2, "OR in AND - should distribute"),
        ("on_a_b & (clear_c | holding_d)", 2, "AND with OR - should distribute"),
        ("(on_a_b | clear_c) & (holding_d | handempty)", 4, "Two ORs in AND - 2×2=4"),

        # Level 4: Complex negations with De Morgan
        ("~(on_a_b & clear_c)", 2, "De Morgan: ~(A & B) = ~A | ~B"),
        ("~(on_a_b | clear_c)", 1, "De Morgan: ~(A | B) = ~A & ~B"),

        # Level 5: Real DFA transition labels (from TEST 3)
        ("~on_d_e & (~clear_c | ~on_a_b)", 2, "TEST 3 Transition 1"),
        ("on_d_e | (clear_c & on_a_b)", 2, "TEST 3 Transition 2"),

        # Level 6: Deeply nested expressions
        ("((on_a_b | clear_c) & holding_d) | handempty", 3, "Nested: (A|B)&C | D = AC|BC|D"),
        ("on_a_b & (clear_c | (holding_d & handempty))", 2, "Nested: A&(B|(C&D)) = AB|ACD"),

        # Level 7: Triple nesting
        ("(on_a_b | clear_c) & (holding_d | handempty) & on_c_d", 4, "Three-way AND with ORs"),

        # Level 8: Implication and equivalence
        ("on_a_b -> clear_c", 2, "Implication: A->B = ~A|B"),
        ("on_a_b <-> clear_c", 2, "Equivalence: A<->B = (A&B)|(~A&~B)"),

        # Level 9: Super complex
        ("~((on_a_b & clear_c) | (holding_d & ~handempty))", 4,
         "De Morgan on complex: ~((A&B)|(C&D)) = (~A|~B)&(~C|~D) = 4 disjuncts"),
    ]

    results = []
    for expr, expected, description in test_cases:
        print(f"\n{'='*80}")
        print(f"Test: {description}")
        result = verify_dnf_correctness(expr, expected, gmap)
        results.append((description, result))

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for desc, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {desc}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! DNF conversion is correct.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed!")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
