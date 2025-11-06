#!/usr/bin/env python3
"""
Test script for ltlf2dfa integration

This script tests basic ltlf2dfa functionality to ensure it's working correctly
before integrating into the pipeline.

NOTE: ltlf2dfa uses PROPOSITIONAL variables, not predicates with arguments.
      So we use 'on_a_b' instead of 'on(a, b)'.
"""

# Add project root to path and setup MONA
import sys
from pathlib import Path

# Add src to path (only once)
_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.setup_mona_path import setup_mona
setup_mona(verbose=True)

from ltlf2dfa.parser.ltlf import LTLfParser


def test_simple_formulas():
    """Test parsing and DFA conversion for simple LTLf formulas"""

    parser = LTLfParser()

    # Test cases with different LTLf operators
    # NOTE: Using simple propositional variables (a, b, c) not predicates
    test_cases = [
        ("F(a)", "Eventually a (Finally)"),
        ("G(a)", "Always a (Globally)"),
        ("X(a)", "Next a"),
        ("a U b", "a Until b"),
        ("F(a) & F(b)", "Multiple goals with conjunction"),
        ("F(a) | G(b)", "Disjunction of temporal formulas"),
        ("!(a)", "Negation"),
        ("a -> b", "Implication"),
    ]

    print("=" * 80)
    print("LTLf2DFA TEST SUITE")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for formula_str, description in test_cases:
        print("-" * 80)
        print(f"Test: {description}")
        print(f"Formula: {formula_str}")
        print("-" * 80)

        try:
            # Parse formula
            formula = parser(formula_str)
            print(f"✓ Parsed: {formula}")

            # Convert to DFA (returns DOT string representation)
            dfa_dot = formula.to_dfa()
            print(f"✓ DFA generated successfully (DOT format)")

            # Show COMPLETE DOT output (no truncation)
            print(f"\n DOT representation:")
            print("  " + "~" * 76)
            for line in dfa_dot.split('\n'):
                print(f"  {line}")
            print("  " + "~" * 76)

            print("\n✓ TEST PASSED")
            passed += 1

        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

        print()

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)


def test_blocksworld_formulas():
    """Test blocksworld-specific LTLf formulas with propositional encoding"""

    parser = LTLfParser()

    print("\n" + "=" * 80)
    print("BLOCKSWORLD-SPECIFIC TESTS (Propositional Encoding)")
    print("=" * 80)
    print()
    print("NOTE: We encode predicates as propositional variables:")
    print("  on(a,b) → on_a_b")
    print("  clear(a) → clear_a")
    print("  holding(a) → holding_a")
    print("  handempty → handempty")
    print()

    # Blocksworld temporal goals using propositional encoding
    blocksworld_formulas = [
        ("F(on_a_b)", "Eventually: on(a, b)"),
        ("F(on_a_b) & F(clear_a)", "Eventually: on(a,b) and clear(a)"),
        ("F(on_a_b & clear_a)", "Eventually: on(a,b) AND clear(a) hold together"),
        ("G(handempty)", "Always: hand is empty"),
        ("F(on_a_b) & G(clear_c)", "Eventually on(a,b), always clear(c)"),
        ("F(on_a_b & on_b_c)", "Eventually: tower a-b-c"),
        ("clear_a U holding_a", "clear(a) Until holding(a)"),
        ("G(F(clear_a))", "Always eventually clear(a) - infinite repetition"),
    ]

    passed = 0
    failed = 0

    for formula_str, description in blocksworld_formulas:
        print("-" * 80)
        print(f"Formula: {formula_str}")
        print(f"Meaning: {description}")
        print("-" * 80)

        try:
            formula = parser(formula_str)
            dfa_dot = formula.to_dfa()

            # Count lines in DOT to get sense of DFA size
            dfa_lines = len(dfa_dot.strip().split('\n'))

            print(f"✓ Success - DFA with {dfa_lines} lines in DOT format")

            # Show COMPLETE DOT output (no truncation)
            print(f"\n  COMPLETE DOT representation:")
            print("  " + "~" * 76)
            for line in dfa_dot.split('\n'):
                print(f"  {line}")
            print("  " + "~" * 76)

            passed += 1
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed += 1

        print()

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)


def test_complex_formulas():
    """Test more complex nested formulas"""

    parser = LTLfParser()

    print("\n" + "=" * 80)
    print("COMPLEX NESTED FORMULAS")
    print("=" * 80)
    print()

    complex_formulas = [
        "F(G(a))",  # Eventually always a
        "G(F(a))",  # Always eventually a
        "F(a & X(b))",  # Eventually a and next b
        "(a U b) & G(c)",  # a until b, and always c
    ]

    for formula_str in complex_formulas:
        print("-" * 80)
        print(f"Formula: {formula_str}")
        print("-" * 80)
        try:
            formula = parser(formula_str)
            dfa_dot = formula.to_dfa()

            # Show COMPLETE DOT output (no truncation)
            print(f"\n  COMPLETE DOT representation:")
            print("  " + "~" * 76)
            for line in dfa_dot.split('\n'):
                print(f"  {line}")
            print("  " + "~" * 76)

        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
        print()


if __name__ == "__main__":
    test_simple_formulas()
    test_blocksworld_formulas()
    test_complex_formulas()
