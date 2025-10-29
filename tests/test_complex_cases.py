#!/usr/bin/env python3
"""
Complex Test Cases for LTL-BDI Pipeline

Tests based on FOND benchmark problems (bw_5_1, bw_5_3, bw_5_5):
- Complex rearrangement tasks with 5 blocks
- Non-trivial initial states (pre-built towers)
- Multiple goals requiring sophisticated planning
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dual_branch_pipeline import DualBranchPipeline
import traceback


class TestCase:
    def __init__(self, name, instruction):
        self.name = name
        self.instruction = instruction
        self.results = {}  # Store results for each mode
        self.errors = {}   # Store errors for each mode


# Test cases - Based on FOND benchmark problems (bw_5_1, bw_5_3, bw_5_5)
# Each test case will be run with BOTH modes: fond and llm_agentspeak
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
    """Run all test cases with both FOND and LLM AgentSpeak modes"""
    print("#" * 80)
    print("# FOND BENCHMARK TEST SUITE - DUAL MODE COMPARISON")
    print("# Testing LTL-BDI Pipeline with 3 FOND benchmark problems")
    print("# Each test runs TWICE: (1) FOND Planning, (2) LLM AgentSpeak")
    print("# Based on: bw_5_1, bw_5_3, bw_5_5 from PR2 benchmarks")
    print("#" * 80)
    print()

    pipeline = DualBranchPipeline()
    modes = ["fond", "llm_agentspeak"]

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

        # Run with both modes
        for mode in modes:
            mode_name = "FOND Planning (PR2)" if mode == "fond" else "LLM AgentSpeak Generation"
            print(f"\n[MODE: {mode.upper()}] {mode_name}")
            print("-" * 80)

            total_runs += 1

            try:
                result = pipeline.execute(test.instruction, mode=mode)
                test.results[mode] = result

                if result.get("success"):
                    print(f"✓ PASSED: {test.name} [{mode.upper()}]")
                    passed += 1
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"✗ FAILED: {test.name} [{mode.upper()}]")
                    print(f"  Error: {error_msg}")
                    test.errors[mode] = error_msg
                    failed += 1

            except Exception as e:
                print(f"✗ CRASHED: {test.name} [{mode.upper()}]")
                print(f"  Exception: {str(e)}")
                print(f"  Traceback:")
                traceback.print_exc()
                test.errors[mode] = str(e)
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
    print(f"Total test runs: {total_runs} ({len(test_cases)} × {len(modes)} modes)")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"💥 Crashed: {crashed}")
    print(f"Success rate: {passed/total_runs*100:.1f}%")
    print()

    # Per-mode statistics
    print("Per-mode breakdown:")
    for mode in modes:
        mode_passed = sum(1 for t in test_cases if mode in t.results and t.results[mode].get("success"))
        mode_failed = sum(1 for t in test_cases if mode in t.errors)
        print(f"  {mode.upper()}: {mode_passed}/{len(test_cases)} passed")

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
            if test.errors:
                print(f"Test: {test.name}")
                print(f"  Instruction: {test.instruction[:100]}...")
                for mode, error in test.errors.items():
                    print(f"  [{mode.upper()}] Error: {error}")
                print()


if __name__ == "__main__":
    run_tests()
