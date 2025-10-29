#!/usr/bin/env python3
"""
Complex Test Cases for LTL-BDI Pipeline

Tests 3 most complex scenarios covering:
- Temporal sequences with multiple operators
- Multiple objects (4 blocks)
- Complex rearrangement from non-trivial initial states
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


# Test cases - Only 3 most complex scenarios
test_cases = [
    # Complex temporal sequence with 3 goals
    TestCase(
        "Complex-1: Sequential tower building",
        "Build a tower with A on top, B in middle, C at bottom",
        mode="fond"
    ),

    # Multiple objects (4 blocks) with multiple goals
    TestCase(
        "Complex-2: Four-block tower",
        "Build tower using blocks A, B, C, D with D at bottom and A on top",
        mode="fond"
    ),

    # Complex rearrangement from non-trivial initial state
    TestCase(
        "Complex-3: Rearrangement task",
        "Move block C from on top of A to on top of B",
        mode="fond"
    ),
]


def run_tests():
    """Run all test cases"""
    print("#" * 80)
    print("# COMPLEX TEST SUITE")
    print("# Testing LTL-BDI Pipeline with 3 most complex scenarios")
    print("#" * 80)
    print()

    pipeline = DualBranchPipeline()
    passed = 0
    failed = 0
    crashed = 0

    for test in test_cases:
        print("=" * 80)
        print(f"TEST: {test.name}")
        print("=" * 80)
        print(f"Instruction: {test.instruction}")
        print(f"Mode: {test.mode}")
        print("-" * 80)
        print()

        try:
            result = pipeline.execute(test.instruction, mode=test.mode)
            test.result = result

            if result.get("success"):
                print(f"\n✓ TEST PASSED: {test.name}")
                passed += 1
            else:
                print(f"\n✗ TEST FAILED: {test.name}")
                print(f"  Error: {result.get('error', 'Unknown error')}")
                test.error = result.get("error", "Unknown error")
                failed += 1

        except Exception as e:
            print(f"\n✗ TEST CRASHED: {test.name}")
            print(f"  Exception: {str(e)}")
            print(f"  Traceback:")
            traceback.print_exc()
            test.error = str(e)
            crashed += 1

        print()
        print("=" * 80)
        print()

    # Summary
    total = len(test_cases)
    print()
    print("#" * 80)
    print("# TEST SUMMARY")
    print("#" * 80)
    print(f"Total tests: {total}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"💥 Crashed: {crashed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print("#" * 80)
    print()

    # Failure details
    if failed > 0 or crashed > 0:
        print()
        print("#" * 80)
        print("# FAILURE DETAILS")
        print("#" * 80)
        print()

        for test in test_cases:
            if test.error:
                print(f"Test: {test.name}")
                print(f"  Instruction: {test.instruction}")
                print(f"  Error: {test.error}")
                print()


if __name__ == "__main__":
    run_tests()
