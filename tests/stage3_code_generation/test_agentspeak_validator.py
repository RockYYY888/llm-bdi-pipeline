"""
Test the AgentSpeak validator on generated code
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.stage3_code_generation.agentspeak_validator import AgentSpeakValidator, validate_agentspeak_code
from tests.stage3_code_generation.test_integration_backward_planner import test_simple_dfa, test_complex_dfa


def test_validator_on_simple_dfa():
    """Test validator on simple DFA generated code"""
    print("=" * 80)
    print("TEST: Validator on Simple DFA")
    print("=" * 80)

    # Generate code using existing test
    passed, asl_code = test_simple_dfa()

    if not passed:
        print("\n❌ Code generation failed, skipping validation test")
        return False

    print("\n" + "=" * 80)
    print("RUNNING AGENTSPEAK VALIDATOR")
    print("=" * 80)

    # Run validator
    validator = AgentSpeakValidator(asl_code, goals=["on(a, b)"])
    validation_passed, issues = validator.validate()

    # Print report
    print(validator.format_report())

    # Print detailed issues if any
    if not validation_passed:
        print("\n❌ VALIDATION FAILED")
        print(f"\nTotal errors: {len(validator.errors)}")
        print(f"Total warnings: {len(validator.warnings)}")
    else:
        print("\n✅ VALIDATION PASSED")

    return validation_passed


def test_validator_on_complex_dfa():
    """Test validator on complex DFA generated code"""
    print("\n\n" + "=" * 80)
    print("TEST: Validator on Complex DFA")
    print("=" * 80)

    # Generate code using existing test
    passed, asl_code = test_complex_dfa()

    if not passed:
        print("\n❌ Code generation failed, skipping validation test")
        return False

    print("\n" + "=" * 80)
    print("RUNNING AGENTSPEAK VALIDATOR")
    print("=" * 80)

    # Run validator
    validator = AgentSpeakValidator(asl_code, goals=["on(a, b)", "clear(a)"])
    validation_passed, issues = validator.validate()

    # Print report
    print(validator.format_report())

    if not validation_passed:
        print("\n❌ VALIDATION FAILED")
        print(f"\nTotal errors: {len(validator.errors)}")
        print(f"Total warnings: {len(validator.warnings)}")
    else:
        print("\n✅ VALIDATION PASSED")

    return validation_passed


def test_validator_on_bad_code():
    """Test validator on intentionally bad code"""
    print("\n\n" + "=" * 80)
    print("TEST: Validator on Bad Code (should fail)")
    print("=" * 80)

    bad_code = """
/* Initial Beliefs */
ontable(a).
handempty.

/* PDDL Action Plans (as AgentSpeak goals) */

+!pick_up_from_table : <-
    -handempty
    +holding

/* Goal Achievement Plans for: on(a, b) */

+!on(a, b) : true <-
    .print("doing nothing").
"""

    print("\nBad code:")
    print(bad_code)

    print("\n" + "=" * 80)
    print("RUNNING AGENTSPEAK VALIDATOR (expecting failures)")
    print("=" * 80)

    validator = AgentSpeakValidator(bad_code, goals=["on(a, b)"])
    validation_passed, issues = validator.validate()

    print(validator.format_report())

    # This should FAIL
    if validation_passed:
        print("\n❌ TEST FAILED: Validator should have caught errors in bad code!")
        return False
    else:
        print("\n✅ TEST PASSED: Validator correctly detected bad code")
        return True


if __name__ == "__main__":
    print("Testing AgentSpeak Validator\n")

    results = []

    # Test 1: Simple DFA
    try:
        result1 = test_validator_on_simple_dfa()
        results.append(("Simple DFA", result1))
    except Exception as e:
        print(f"\n❌ Exception in test 1: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Simple DFA", False))

    # Test 2: Complex DFA
    try:
        result2 = test_validator_on_complex_dfa()
        results.append(("Complex DFA", result2))
    except Exception as e:
        print(f"\n❌ Exception in test 2: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Complex DFA", False))

    # Test 3: Bad code
    try:
        result3 = test_validator_on_bad_code()
        results.append(("Bad Code Detection", result3))
    except Exception as e:
        print(f"\n❌ Exception in test 3: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Bad Code Detection", False))

    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL VALIDATOR TESTS PASSED")
    else:
        print("❌ SOME VALIDATOR TESTS FAILED")
    print("=" * 80)
