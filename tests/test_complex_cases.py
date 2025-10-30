#!/usr/bin/env python3
"""
Complex Test Cases for LTL-BDI Pipeline

Tests based on FOND benchmark problems (bw_5_1, bw_5_3, bw_5_5):
- Complex rearrangement tasks with 5 blocks
- Non-trivial initial states (pre-built towers)
- Multiple goals requiring sophisticated planning

Note: Only tests LLM AgentSpeak generation (Branch A).
FOND planning (Branch B) has been moved to legacy/fond/
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltl_bdi_pipeline import LTL_BDI_Pipeline
import traceback


class TestCase:
    def __init__(self, name, instruction):
        self.name = name
        self.instruction = instruction
        self.result = None
        self.error = None


# Test cases - Based on FOND benchmark problems (bw_5_1, bw_5_3, bw_5_5)
test_cases = [
    # bw_5_1: Complex rearrangement from two towers to one tower
    TestCase(
        "Complex-1: Two towers to single tower (bw_5_1)",
        "Given blocks b1, b2, b3, b4, b5 where initially b2 is on b1, b1 is on b3 (b3 is on table), and b5 is on b4 (b4 is on table). Rearrange them so that b1 is on b2, b2 is on b5 (b5 is on table), and b3, b4 are separately on table."
    ),

    # bw_5_3: Complex rearrangement with tower reconstruction
    TestCase(
        "Complex-2: Tower reconstruction (bw_5_3)",
        "Given blocks b1, b2, b3, b4, b5 where initially b3 is on b2, b2 is on b1 (b1 is on table), and b4 is on b5 (b5 is on table). Rearrange them so that b5 is on b3, b3 is on b1 (b1 is on table), and b2, b4 are separately on table."
    ),

    # bw_5_5: Complex rearrangement to 4-block tower
    TestCase(
        "Complex-3: Four-block tower assembly (bw_5_5)",
        "Given blocks b1, b2, b3, b4, b5 where initially b3 is on b5, b5 is on b1 (b1 is on table), and b4 is on b2 (b2 is on table). Rearrange them to form a tower with b2 on b4, b4 on b1, b1 on b3 (b3 is on table), and b5 separately on table."
    ),
]


def run_tests():
    """Run all test cases with LLM AgentSpeak generation"""
    print("#" * 80)
    print("# LTL-BDI PIPELINE TEST SUITE")
    print("# Testing LLM AgentSpeak Generation with FOND benchmark problems")
    print("# Based on: bw_5_1, bw_5_3, bw_5_5 from PR2 benchmarks")
    print("#" * 80)
    print()

    pipeline = LTL_BDI_Pipeline()

    # Statistics
    total_runs = 0
    passed = 0
    failed = 0
    crashed = 0

    for test in test_cases:
        print("=" * 80)
        print(f"TEST: {test.name}")
        print("=" * 80)
        print(f"Instruction: {test.instruction}")
        print("=" * 80)
        print()

        total_runs += 1

        try:
            result = pipeline.execute(test.instruction)
            test.result = result

            if result.get("success"):
                print(f"✓ PASSED: {test.name}")
                passed += 1
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"✗ FAILED: {test.name}")
                print(f"  Error: {error_msg}")
                test.error = error_msg
                failed += 1

        except Exception as e:
            print(f"✗ CRASHED: {test.name}")
            print(f"  Exception: {str(e)}")
            print(f"  Traceback:")
            traceback.print_exc()
            test.error = str(e)
            crashed += 1

        print()
        print("=" * 80)
        print()

    # Summary
    print()
    print("#" * 80)
    print("# TEST SUMMARY")
    print("#" * 80)
    print(f"Total test scenarios: {len(test_cases)}")
    print(f"Total test runs: {total_runs}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"💥 Crashed: {crashed}")
    print(f"Success rate: {passed/total_runs*100:.1f}%")
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
                print(f"  Instruction: {test.instruction[:100]}...")
                print(f"  Error: {test.error}")
                print()


if __name__ == "__main__":
    run_tests()
