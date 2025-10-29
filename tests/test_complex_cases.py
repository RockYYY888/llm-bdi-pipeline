#!/usr/bin/env python3
"""
Complex Test Cases for LTL-BDI Pipeline

Tests various complex scenarios including:
- Temporal sequences (G, F, U, X operators)
- Nested goals
- Multiple objects
- Complex initial states
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dual_branch_pipeline import DualBranchPipeline
import traceback


class TestCase:
    def __init__(self, name, instruction, mode="fond"):
        self.name = name
        self.instruction = instruction
        self.mode = mode
        self.result = None
        self.error = None


def run_test_case(test_case: TestCase, pipeline: DualBranchPipeline):
    """Run a single test case"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_case.name}")
    print(f"{'='*80}")
    print(f"Instruction: {test_case.instruction}")
    print(f"Mode: {test_case.mode}")
    print(f"{'-'*80}\n")

    try:
        result = pipeline.execute(test_case.instruction, mode=test_case.mode)
        test_case.result = result

        if result.get("success"):
            print(f"\n✓ TEST PASSED: {test_case.name}")
        else:
            print(f"\n✗ TEST FAILED: {test_case.name}")
            print(f"  Error: {result.get('error', 'Unknown error')}")
            test_case.error = result.get('error')

    except Exception as e:
        print(f"\n✗ TEST CRASHED: {test_case.name}")
        print(f"  Exception: {str(e)}")
        print(f"  Traceback:")
        traceback.print_exc()
        test_case.error = str(e)

    print(f"\n{'='*80}\n")


def main():
    """Run all complex test cases"""

    print(f"\n{'#'*80}")
    print(f"# COMPLEX TEST SUITE")
    print(f"# Testing LTL-BDI Pipeline with complex scenarios")
    print(f"{'#'*80}\n")

    # Create test cases
    test_cases = [
        # Basic cases (sanity check)
        TestCase(
            "Basic-1: Simple stacking",
            "Stack block C on block B",
            mode="fond"
        ),

        # Temporal sequence cases
        TestCase(
            "Temporal-1: Sequential stacking",
            "First stack A on B, then stack C on A",
            mode="fond"
        ),

        TestCase(
            "Temporal-2: Build tower ABC",
            "Build a tower with A on top, B in the middle, and C at the bottom",
            mode="fond"
        ),

        TestCase(
            "Temporal-3: Sequence of three actions",
            "Put A on the table, then put B on A, then put C on B",
            mode="fond"
        ),

        # Nested goal cases
        TestCase(
            "Nested-1: Multiple stacks",
            "Stack A on B and stack C on D",
            mode="fond"
        ),

        TestCase(
            "Nested-2: Complex arrangement",
            "Arrange the blocks so that A is on B, and C is clear",
            mode="fond"
        ),

        # Multiple objects cases
        TestCase(
            "Multi-1: Four blocks tower",
            "Build a tower using blocks A, B, C, and D with D at the bottom",
            mode="fond"
        ),

        TestCase(
            "Multi-2: Multiple towers",
            "Create two towers: one with A on B, another with C on D",
            mode="fond"
        ),

        # Complex initial state cases
        TestCase(
            "Complex-1: Rearrangement",
            "Move block C from on top of A to on top of B",
            mode="fond"
        ),

        TestCase(
            "Complex-2: Clear and stack",
            "Clear block B and then put C on it",
            mode="fond"
        ),

        # Edge cases
        TestCase(
            "Edge-1: Single block",
            "Make sure block A is on the table",
            mode="fond"
        ),

        TestCase(
            "Edge-2: Complex temporal",
            "Eventually put A on B, and always keep C clear until then",
            mode="fond"
        ),

        # LLM AgentSpeak tests (if time permits)
        TestCase(
            "LLM-1: Simple with LLM",
            "Stack block C on block B",
            mode="llm_agentspeak"
        ),

        # Both branches test
        TestCase(
            "Both-1: Compare both approaches",
            "Stack block C on block B",
            mode="both"
        ),
    ]

    # Run all tests
    pipeline = DualBranchPipeline()
    passed = 0
    failed = 0
    crashed = 0

    for test_case in test_cases:
        run_test_case(test_case, pipeline)

        if test_case.error:
            if "crashed" in str(test_case.error).lower():
                crashed += 1
            else:
                failed += 1
        elif test_case.result and test_case.result.get("success"):
            passed += 1
        else:
            failed += 1

    # Print summary
    print(f"\n{'#'*80}")
    print(f"# TEST SUMMARY")
    print(f"{'#'*80}")
    print(f"Total tests: {len(test_cases)}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"💥 Crashed: {crashed}")
    print(f"Success rate: {(passed / len(test_cases) * 100):.1f}%")
    print(f"{'#'*80}\n")

    # Print detailed failure report
    if failed > 0 or crashed > 0:
        print(f"\n{'#'*80}")
        print(f"# FAILURE DETAILS")
        print(f"{'#'*80}\n")

        for test_case in test_cases:
            if test_case.error:
                print(f"Test: {test_case.name}")
                print(f"  Instruction: {test_case.instruction}")
                print(f"  Error: {test_case.error}")
                print()

    return 0 if crashed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
