#!/usr/bin/env python
"""
Test script to verify strong negation (~) is used instead of weak negation (not)
in AgentSpeak code generation.
"""

from src.stage3_code_generation.state_space import PredicateAtom, WorldState


def test_predicate_strong_negation():
    """Test that PredicateAtom uses strong negation (~) in AgentSpeak"""
    print("="*80)
    print("Test 1: PredicateAtom strong negation")
    print("="*80)

    # Test positive predicate
    p1 = PredicateAtom('ontable', ['c'], negated=False)
    result1 = p1.to_agentspeak()
    expected1 = 'ontable(c)'
    print(f"Positive predicate: {result1}")
    print(f"Expected: {expected1}")
    assert result1 == expected1, f"Expected {expected1}, got {result1}"
    print("✅ PASS\n")

    # Test negated predicate - should use strong negation (~)
    p2 = PredicateAtom('ontable', ['c'], negated=True)
    result2 = p2.to_agentspeak()
    expected2 = '~ontable(c)'
    print(f"Negated predicate: {result2}")
    print(f"Expected: {expected2}")
    assert result2 == expected2, f"Expected {expected2}, got {result2}"

    # Ensure NOT using weak negation
    wrong_result = 'not ontable(c)'
    assert result2 != wrong_result, f"Should NOT use weak negation: {wrong_result}"
    print("✅ PASS - Uses strong negation (~), not weak negation (not)\n")


def test_worldstate_strong_negation():
    """Test that WorldState uses strong negation in context"""
    print("="*80)
    print("Test 2: WorldState context with strong negation")
    print("="*80)

    state = WorldState({
        PredicateAtom('holding', ['a'], negated=False),
        PredicateAtom('ontable', ['c'], negated=True),  # Negated predicate
        PredicateAtom('clear', ['b'], negated=False)
    })

    context = state.to_agentspeak_context()
    print(f"Context: {context}")

    # Should contain ~ontable(c), not "not ontable(c)"
    assert '~ontable(c)' in context, f"Expected ~ontable(c) in context, got: {context}"
    assert 'not ontable(c)' not in context, f"Should NOT use weak negation in context: {context}"
    print("✅ PASS - Context uses strong negation (~)\n")


def test_variable_conversion_with_negation():
    """Test strong negation with variable conversion"""
    print("="*80)
    print("Test 3: Strong negation with AgentSpeak variables")
    print("="*80)

    # Test with PDDL variables
    p = PredicateAtom('ontable', ['?v0'], negated=True)
    result = p.to_agentspeak(convert_vars=True)
    expected = '~ontable(V0)'
    print(f"With variable conversion: {result}")
    print(f"Expected: {expected}")
    assert result == expected, f"Expected {expected}, got {result}"
    print("✅ PASS\n")

    # Test with object-to-variable mapping
    obj_to_var = {'c': '?v0'}
    p2 = PredicateAtom('ontable', ['c'], negated=True)
    result2 = p2.to_agentspeak(obj_to_var=obj_to_var)
    expected2 = '~ontable(V0)'
    print(f"With object mapping: {result2}")
    print(f"Expected: {expected2}")
    assert result2 == expected2, f"Expected {expected2}, got {result2}"
    print("✅ PASS\n")


def test_semantics_explanation():
    """Verify the semantics are correct"""
    print("="*80)
    print("Test 4: Semantic verification")
    print("="*80)

    print("Strong negation (~ontable(c)):")
    print("  Semantics: 'c is definitely NOT on the table' (achievable state)")
    print("  Corresponds to PDDL effect: (not (ontable c))")
    print("  Can be achieved by action: pickup(c)")
    print()

    print("Weak negation (not ontable(c)) - INCORRECT:")
    print("  Semantics: 'lack knowledge about ontable(c)' (epistemic)")
    print("  NOT an achievable goal in AgentSpeak")
    print("  Confusing and semantically problematic")
    print()

    p = PredicateAtom('ontable', ['c'], negated=True)
    result = p.to_agentspeak()

    # Verify we're using the correct one
    is_strong_negation = result.startswith('~')
    is_weak_negation = result.startswith('not ')

    assert is_strong_negation, "Should use strong negation"
    assert not is_weak_negation, "Should NOT use weak negation"
    print("✅ PASS - Using semantically correct strong negation\n")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("STRONG NEGATION FIX VERIFICATION")
    print("="*80 + "\n")

    test_predicate_strong_negation()
    test_worldstate_strong_negation()
    test_variable_conversion_with_negation()
    test_semantics_explanation()

    print("="*80)
    print("ALL TESTS PASSED ✅")
    print("="*80)
    print("\nSummary:")
    print("  - PredicateAtom.to_agentspeak() uses strong negation (~)")
    print("  - WorldState.to_agentspeak_context() preserves strong negation")
    print("  - Variable conversion maintains strong negation")
    print("  - Semantics are correct: ~p = 'p is false' (achievable state)")
    print()
