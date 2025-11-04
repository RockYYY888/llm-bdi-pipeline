#!/usr/bin/env python3
"""
Verify actual operator precedence used by ltlf2dfa
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ltlf2dfa.parser.ltlf import LTLfParser


def test_all_precedence_pairs():
    """Test precedence between all operator pairs"""

    parser = LTLfParser()

    tests = [
        # Format: (name, formula, expected_parse_description)

        # ! vs other operators
        ("! vs &", "!a & b", "(!a) & b - NOT binds tighter"),
        ("! vs |", "!a | b", "(!a) | b - NOT binds tighter"),
        ("! vs ->", "!a -> b", "(!a) -> b - NOT binds tighter"),
        ("! vs <->", "!a <-> b", "(!a) <-> b - NOT binds tighter"),

        # Unary temporal (F, G, X) vs binary operators
        ("F vs &", "F(a) & b", "(F(a)) & b"),
        ("G vs &", "G(a) & b", "(G(a)) & b"),
        ("X vs &", "X(a) & b", "(X(a)) & b"),

        # U vs other operators
        ("U vs &", "a U b & c", "Parse to determine if (a U b) & c or a U (b & c)"),
        ("U vs |", "a U b | c", "Parse to determine"),
        ("U vs ->", "a U b -> c", "Parse to determine"),

        # R vs other operators
        ("R vs &", "a R b & c", "Parse to determine"),
        ("R vs |", "a R b | c", "Parse to determine"),

        # & vs |
        ("& vs |", "a & b | c", "Parse to determine if (a & b) | c or a & (b | c)"),

        # & vs ->
        ("& vs ->", "a & b -> c", "Parse to determine"),

        # | vs ->
        ("| vs ->", "a | b -> c", "Parse to determine"),

        # -> vs <->
        ("-> vs <->", "a -> b <-> c", "Parse to determine"),
    ]

    print("="*80)
    print("LTLF2DFA OPERATOR PRECEDENCE VERIFICATION")
    print("="*80)

    results = {}

    for name, formula, description in tests:
        print(f"\n{'='*80}")
        print(f"Test: {name}")
        print(f"Formula: {formula}")
        print(f"Expected: {description}")
        print(f"{'='*80}")

        try:
            parsed = parser(formula)
            parsed_str = str(parsed)
            parsed_type = type(parsed).__name__

            print(f"Parsed Type: {parsed_type}")
            print(f"Parsed Result: {parsed_str}")

            # Store result
            results[name] = {
                "formula": formula,
                "parsed": parsed_str,
                "type": parsed_type
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            results[name] = {"error": str(e)}

    # Analyze results to determine precedence
    print("\n" + "="*80)
    print("PRECEDENCE ANALYSIS")
    print("="*80)

    # Determine precedence levels from test results
    precedence_rules = {}

    # Check U vs &
    if "U vs &" in results:
        u_and_result = results["U vs &"]["parsed"]
        if "U (b & c)" in u_and_result or "U(b & c)" in u_and_result:
            precedence_rules["U > &"] = "U binds looser than & (& inside U)"
            print("\n✓ U binds LOOSER than & (& has higher precedence)")
            print(f"  Evidence: '{results['U vs &']['formula']}' → '{u_and_result}'")
        elif "U b) &" in u_and_result or "(a U b) &" in u_and_result:
            precedence_rules["& < U"] = "& binds looser than U (U inside &)"
            print("\n✓ & binds LOOSER than U (U has higher precedence)")
            print(f"  Evidence: '{results['U vs &']['formula']}' → '{u_and_result}'")

    # Check & vs |
    if "& vs |" in results:
        and_or_result = results["& vs |"]["parsed"]
        if "& b) |" in and_or_result or "(a & b) |" in and_or_result:
            precedence_rules["| < &"] = "| binds looser than & (& inside |)"
            print("\n✓ | binds LOOSER than & (& has higher precedence)")
            print(f"  Evidence: '{results['& vs |']['formula']}' → '{and_or_result}'")
        elif "& (b | c)" in and_or_result:
            precedence_rules["& < |"] = "& binds looser than |"
            print("\n✓ & binds LOOSER than | (| has higher precedence)")
            print(f"  Evidence: '{results['& vs |']['formula']}' → '{and_or_result}'")

    # Generate final precedence ordering
    print("\n" + "="*80)
    print("FINAL PRECEDENCE ORDERING (Highest to Lowest)")
    print("="*80)

    # Based on results, construct the precedence hierarchy
    ordering = []

    # NOT always highest
    ordering.append("Level 1 (Highest): ! (NOT)")

    # Unary temporal
    ordering.append("Level 2: X, WX, F, G (Unary Temporal)")

    # Binary temporal vs boolean - check test results
    if "U > &" in precedence_rules:
        ordering.append("Level 3: & (AND)")
        ordering.append("Level 4: U, R (Binary Temporal)")
    else:
        ordering.append("Level 3: U, R (Binary Temporal)")
        ordering.append("Level 4: & (AND)")

    # OR
    ordering.append("Level 5: | (OR)")

    # IMPLIES
    ordering.append("Level 6: -> (IMPLIES)")

    # EQUIVALENCE
    ordering.append("Level 7 (Lowest): <-> (EQUIVALENCE)")

    for line in ordering:
        print(line)

    return results, precedence_rules


if __name__ == "__main__":
    test_all_precedence_pairs()
