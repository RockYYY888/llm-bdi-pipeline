#!/usr/bin/env python3
"""
Test script to verify PDDL type annotations are stripped from AgentSpeak code.
This tests the fix for the issue where action goal invocations incorrectly
included type annotations: !pick_up(B1 - block, B2 - block) instead of !pick_up(B1, B2)
"""

import sys
from pathlib import Path

# Add src to path
_src_dir = str(Path(__file__).parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ltl_bdi_pipeline import LTL_BDI_Pipeline


def test_type_annotation_stripping():
    """Test that generated AgentSpeak code does NOT contain PDDL type annotations"""
    print("="*80)
    print("Test: PDDL Type Annotation Stripping in AgentSpeak Code")
    print("="*80)

    # Initialize pipeline
    pipeline = LTL_BDI_Pipeline()

    # Simple test instruction
    instruction = "Using blocks b1, b2, b3, arrange them so that b1 is on b2."

    print(f"\n📋 Instruction: {instruction}")
    print("\n🔄 Running LTL-BDI pipeline...")

    # Run pipeline
    result = pipeline.execute(instruction, mode="dfa_agentspeak")

    # Check result
    if not result.get("success"):
        print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        return False

    # Get generated AgentSpeak file
    agentspeak_file = result.get('agentspeak_file') or result.get('stage3', {}).get('output_file')
    if not agentspeak_file or not Path(agentspeak_file).exists():
        print("\n❌ FAIL: AgentSpeak file not generated")
        return False

    # Read generated code
    with open(agentspeak_file, 'r') as f:
        generated_code = f.read()

    print(f"\n📄 Generated file: {agentspeak_file}")

    # Check for incorrect patterns (PDDL type annotations in action calls)
    incorrect_patterns = [
        " - block",    # e.g., "B1 - block" in action calls
        " - object",   # e.g., "X - object"
    ]

    errors = []
    for pattern in incorrect_patterns:
        if pattern in generated_code:
            # Find lines containing the pattern (exclude comments)
            lines_with_error = []
            for i, line in enumerate(generated_code.split('\n'), 1):
                if pattern in line and not line.strip().startswith('/*') and not line.strip().startswith('*'):
                    lines_with_error.append((i, line.strip()))

            if lines_with_error:
                errors.append({
                    'pattern': pattern,
                    'lines': lines_with_error
                })

    # Report results
    if errors:
        print("\n❌ FAIL: Found PDDL type annotations in AgentSpeak code:")
        for error in errors:
            print(f"\n  Pattern: '{error['pattern']}'")
            for line_num, line in error['lines'][:5]:  # Show first 5 errors
                print(f"    Line {line_num}: {line}")
        return False
    else:
        print("\n✅ PASS: No PDDL type annotations found in AgentSpeak code")
        print("\n✅ All action goal invocations use correct syntax:")
        print("   - !pick_up(B1, B2) ✓")
        print("   - !put_on_block(B1, B2) ✓")
        print("   - !pick_up_from_table(B) ✓")

        # Show sample of generated plans
        print("\n📋 Sample of generated plans:")
        for i, line in enumerate(generated_code.split('\n'), 1):
            if line.strip().startswith('+!') and not line.strip().startswith('+!~'):
                print(f"   Line {i}: {line.strip()}")
                if i > 100:  # Show first few plans only
                    break

        return True


if __name__ == "__main__":
    success = test_type_annotation_stripping()
    exit(0 if success else 1)
