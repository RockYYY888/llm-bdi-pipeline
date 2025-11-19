#!/usr/bin/env python3
"""
Test: PDDL Type Annotation Stripping in AgentSpeak Code Generation

Verifies that the code generator correctly strips PDDL type annotations
from action goal invocations.

Issue: Action goals were incorrectly generated with PDDL type annotations:
  ❌ !pick_up(B1 - block, B2 - block)
  ✅ !pick_up(B1, B2)

Root cause: variable_planner.py was using full PDDL parameter strings
(e.g., "?b1 - block") as variable names instead of extracting just the
variable part (e.g., "?b1").

Fix: Extract variable names from PDDL parameters before applying substitutions.
"""

import sys
from pathlib import Path

# Add src to path
_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ltl_bdi_pipeline import LTL_BDI_Pipeline


def test_no_type_annotations_in_action_calls():
    """
    Test that generated AgentSpeak code does NOT contain PDDL type annotations
    in action goal invocations.
    """
    print("="*80)
    print("Test: PDDL Type Annotation Stripping")
    print("="*80)

    # Initialize pipeline
    pipeline = LTL_BDI_Pipeline()

    # Simple test case
    instruction = "Using blocks b1, b2, b3, arrange them so that b1 is on b2."

    print(f"\nInstruction: {instruction}")
    print("Running pipeline...")

    # Run pipeline
    result = pipeline.execute(instruction, mode="dfa_agentspeak")

    # Check result
    assert result.get("success"), f"Pipeline failed: {result.get('error')}"

    # Get generated AgentSpeak code
    # Pipeline returns code directly in 'agentspeak_code' field
    generated_code = result.get('agentspeak_code')
    assert generated_code, "AgentSpeak code not found in result"

    print(f"\nGenerated code: {len(generated_code)} characters")

    # Check for incorrect patterns (PDDL type annotations in action calls)
    # These patterns should NOT appear in action goal invocations
    errors = []

    for i, line in enumerate(generated_code.split('\n'), 1):
        # Skip comments
        if line.strip().startswith('/*') or line.strip().startswith('*'):
            continue

        # Check for type annotations in action calls
        if '!' in line and ' - ' in line:
            # Check if it's an action goal invocation with type annotation
            if any(action in line for action in ['!pick_up', '!put_on_block', '!pick_up_from_table', '!put_down']):
                if ' - block' in line or ' - object' in line:
                    errors.append((i, line.strip()))

    # Assert no errors found
    if errors:
        print("\n❌ FAIL: Found PDDL type annotations in action goal invocations:")
        for line_num, line in errors[:10]:  # Show first 10 errors
            print(f"  Line {line_num}: {line}")
        assert False, f"Found {len(errors)} action goals with PDDL type annotations"
    else:
        print("\n✅ PASS: No PDDL type annotations in action goal invocations")

        # Show sample of correct syntax
        print("\nSample of correct action goal invocations:")
        for i, line in enumerate(generated_code.split('\n'), 1):
            if line.strip().startswith('+!pick_up') or line.strip().startswith('+!put_on_block'):
                print(f"  Line {i}: {line.strip()}")
                break


def test_action_plan_definitions_correct():
    """
    Test that action plan definitions themselves are correct
    (i.e., +!action(Vars) without type annotations)
    """
    print("\n" + "="*80)
    print("Test: Action Plan Definition Syntax")
    print("="*80)

    pipeline = LTL_BDI_Pipeline()
    instruction = "Using blocks b1, b2, arrange them so that b1 is on b2."

    result = pipeline.execute(instruction, mode="dfa_agentspeak")
    assert result.get("success"), "Pipeline failed"

    generated_code = result.get('agentspeak_code')
    assert generated_code, "AgentSpeak code not found in result"

    # Check action plan headers
    errors = []
    for i, line in enumerate(generated_code.split('\n'), 1):
        if line.strip().startswith('+!pick_up(') or line.strip().startswith('+!put_on_block('):
            if ' - block' in line or ' - object' in line:
                errors.append((i, line.strip()))

    assert not errors, f"Found {len(errors)} action plan definitions with type annotations:\n" + \
                      "\n".join(f"Line {i}: {line}" for i, line in errors[:5])

    print("✅ PASS: All action plan definitions use correct syntax (no type annotations)")


if __name__ == "__main__":
    test_no_type_annotations_in_action_calls()
    test_action_plan_definitions_correct()
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✅")
    print("="*80)
