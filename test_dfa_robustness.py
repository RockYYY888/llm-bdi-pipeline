#!/usr/bin/env python3
"""
DFA Simplifier Robustness Test

Tests that ALL types of DFAs are correctly simplified to atomic transitions.
Verifies that every transition label is either a single partition symbol or
a simple atomic predicate.
"""

import sys
import re
from pathlib import Path

# Add src to path
_src_dir = str(Path(__file__).parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator, LogicalOperator
from stage2_dfa_generation.dfa_builder import DFABuilder


def check_dfa_is_atomic(dfa_dot: str, partition_map: dict) -> dict:
    """
    Verify that all transition labels in DFA are atomic

    Returns:
        dict with:
            - is_atomic: bool
            - transitions: list of (from, to, label)
            - non_atomic_labels: list of problematic labels
            - analysis: detailed analysis
    """
    transitions = []
    non_atomic_labels = []

    # Parse all transitions
    for line in dfa_dot.split('\n'):
        match = re.match(r'\s*(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]', line)
        if match and match.group(1) not in ['init', '__start']:
            from_state, to_state, label = match.groups()
            transitions.append((from_state, to_state, label))

            # Check if label is atomic
            # Atomic means: partition symbol (pN, αN) OR simple predicate OR true/false
            is_partition = label in partition_map if partition_map else False
            is_simple = not any(op in label for op in ['&', '|', '!', '~', '(', ')']) or label in ['true', 'false']

            if not (is_partition or is_simple):
                non_atomic_labels.append((from_state, to_state, label))

    analysis = {
        'total_transitions': len(transitions),
        'atomic_transitions': len(transitions) - len(non_atomic_labels),
        'non_atomic_transitions': len(non_atomic_labels),
    }

    return {
        'is_atomic': len(non_atomic_labels) == 0,
        'transitions': transitions,
        'non_atomic_labels': non_atomic_labels,
        'analysis': analysis
    }


def test_simple_finally():
    """Test: F(on(a, b))"""
    print("\n" + "="*80)
    print("TEST 1: Simple Finally - F(on(a, b))")
    print("="*80)

    spec = LTLSpecification()
    spec.objects = ["a", "b"]

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    spec.grounding_map = gmap

    atom = LTLFormula(predicate="on_a_b", operator=None, sub_formulas=[], logical_op=None)
    f_formula = LTLFormula(predicate=None, operator=TemporalOperator.FINALLY, sub_formulas=[atom], logical_op=None)
    spec.formulas = [f_formula]

    builder = DFABuilder()
    dfa_result = builder.build(spec)

    print(f"Formula: {f_formula.to_string()}")
    print(f"DFA: {dfa_result['num_states']} states, {dfa_result['num_transitions']} transitions")
    print(f"Partitions: {dfa_result['simplification_stats']['num_partitions']}")

    check = check_dfa_is_atomic(dfa_result['dfa_dot'], dfa_result.get('partition_map'))
    print(f"\nAtomicity Check:")
    print(f"  Total transitions: {check['analysis']['total_transitions']}")
    print(f"  Atomic: {check['analysis']['atomic_transitions']}")
    print(f"  Non-atomic: {check['analysis']['non_atomic_transitions']}")

    if not check['is_atomic']:
        print(f"\n❌ FAILED - Non-atomic labels found:")
        for from_s, to_s, label in check['non_atomic_labels']:
            print(f"  {from_s} -> {to_s} [label=\"{label}\"]")
        return False

    print(f"\n✅ PASSED - All transitions are atomic")
    return True


def test_conjunction():
    """Test: F(on(a, b) & clear(c))"""
    print("\n" + "="*80)
    print("TEST 2: Conjunction - F(on(a, b) & clear(c))")
    print("="*80)

    spec = LTLSpecification()
    spec.objects = ["a", "b", "c"]

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    spec.grounding_map = gmap

    on_a_b = LTLFormula(predicate="on_a_b", operator=None, sub_formulas=[], logical_op=None)
    clear_c = LTLFormula(predicate="clear_c", operator=None, sub_formulas=[], logical_op=None)
    conjunction = LTLFormula(predicate=None, operator=None, sub_formulas=[on_a_b, clear_c], logical_op=LogicalOperator.AND)
    f_formula = LTLFormula(predicate=None, operator=TemporalOperator.FINALLY, sub_formulas=[conjunction], logical_op=None)
    spec.formulas = [f_formula]

    builder = DFABuilder()
    dfa_result = builder.build(spec)

    print(f"Formula: {f_formula.to_string()}")
    print(f"DFA: {dfa_result['num_states']} states, {dfa_result['num_transitions']} transitions")
    print(f"Partitions: {dfa_result['simplification_stats']['num_partitions']}")

    check = check_dfa_is_atomic(dfa_result['dfa_dot'], dfa_result.get('partition_map'))
    print(f"\nAtomicity Check:")
    print(f"  Total transitions: {check['analysis']['total_transitions']}")
    print(f"  Atomic: {check['analysis']['atomic_transitions']}")
    print(f"  Non-atomic: {check['analysis']['non_atomic_transitions']}")

    if not check['is_atomic']:
        print(f"\n❌ FAILED - Non-atomic labels found:")
        for from_s, to_s, label in check['non_atomic_labels']:
            print(f"  {from_s} -> {to_s} [label=\"{label}\"]")
        return False

    print(f"\n✅ PASSED - All transitions are atomic")
    return True


def test_globally_negation():
    """Test: G(!(on(a, b)))"""
    print("\n" + "="*80)
    print("TEST 3: Globally with Negation - G(!(on(a, b)))")
    print("="*80)

    spec = LTLSpecification()
    spec.objects = ["a", "b"]

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    spec.grounding_map = gmap

    on_a_b = LTLFormula(predicate="on_a_b", operator=None, sub_formulas=[], logical_op=None)
    not_on = LTLFormula(predicate=None, operator=None, sub_formulas=[on_a_b], logical_op=LogicalOperator.NOT)
    g_formula = LTLFormula(predicate=None, operator=TemporalOperator.GLOBALLY, sub_formulas=[not_on], logical_op=None)
    spec.formulas = [g_formula]

    builder = DFABuilder()
    dfa_result = builder.build(spec)

    print(f"Formula: {g_formula.to_string()}")
    print(f"DFA: {dfa_result['num_states']} states, {dfa_result['num_transitions']} transitions")
    print(f"Partitions: {dfa_result['simplification_stats']['num_partitions']}")

    check = check_dfa_is_atomic(dfa_result['dfa_dot'], dfa_result.get('partition_map'))
    print(f"\nAtomicity Check:")
    print(f"  Total transitions: {check['analysis']['total_transitions']}")
    print(f"  Atomic: {check['analysis']['atomic_transitions']}")
    print(f"  Non-atomic: {check['analysis']['non_atomic_transitions']}")

    if not check['is_atomic']:
        print(f"\n❌ FAILED - Non-atomic labels found:")
        for from_s, to_s, label in check['non_atomic_labels']:
            print(f"  {from_s} -> {to_s} [label=\"{label}\"]")
        return False

    print(f"\n✅ PASSED - All transitions are atomic")
    return True


def test_disjunction():
    """Test: F(on(a, b) | clear(c))"""
    print("\n" + "="*80)
    print("TEST 4: Disjunction - F(on(a, b) | clear(c))")
    print("="*80)

    spec = LTLSpecification()
    spec.objects = ["a", "b", "c"]

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    spec.grounding_map = gmap

    on_a_b = LTLFormula(predicate="on_a_b", operator=None, sub_formulas=[], logical_op=None)
    clear_c = LTLFormula(predicate="clear_c", operator=None, sub_formulas=[], logical_op=None)
    disjunction = LTLFormula(predicate=None, operator=None, sub_formulas=[on_a_b, clear_c], logical_op=LogicalOperator.OR)
    f_formula = LTLFormula(predicate=None, operator=TemporalOperator.FINALLY, sub_formulas=[disjunction], logical_op=None)
    spec.formulas = [f_formula]

    builder = DFABuilder()
    dfa_result = builder.build(spec)

    print(f"Formula: {f_formula.to_string()}")
    print(f"DFA: {dfa_result['num_states']} states, {dfa_result['num_transitions']} transitions")
    print(f"Partitions: {dfa_result['simplification_stats']['num_partitions']}")

    check = check_dfa_is_atomic(dfa_result['dfa_dot'], dfa_result.get('partition_map'))
    print(f"\nAtomicity Check:")
    print(f"  Total transitions: {check['analysis']['total_transitions']}")
    print(f"  Atomic: {check['analysis']['atomic_transitions']}")
    print(f"  Non-atomic: {check['analysis']['non_atomic_transitions']}")

    if not check['is_atomic']:
        print(f"\n❌ FAILED - Non-atomic labels found:")
        for from_s, to_s, label in check['non_atomic_labels']:
            print(f"  {from_s} -> {to_s} [label=\"{label}\"]")
        return False

    print(f"\n✅ PASSED - All transitions are atomic")
    return True


def test_complex_nested():
    """Test: F((on(a, b) & clear(c)) | holding(d))"""
    print("\n" + "="*80)
    print("TEST 5: Complex Nested - F((on(a, b) & clear(c)) | holding(d))")
    print("="*80)

    spec = LTLSpecification()
    spec.objects = ["a", "b", "c", "d"]

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    gmap.add_atom("holding_d", "holding", ["d"])
    spec.grounding_map = gmap

    on_a_b = LTLFormula(predicate="on_a_b", operator=None, sub_formulas=[], logical_op=None)
    clear_c = LTLFormula(predicate="clear_c", operator=None, sub_formulas=[], logical_op=None)
    holding_d = LTLFormula(predicate="holding_d", operator=None, sub_formulas=[], logical_op=None)

    conj = LTLFormula(predicate=None, operator=None, sub_formulas=[on_a_b, clear_c], logical_op=LogicalOperator.AND)
    disj = LTLFormula(predicate=None, operator=None, sub_formulas=[conj, holding_d], logical_op=LogicalOperator.OR)
    f_formula = LTLFormula(predicate=None, operator=TemporalOperator.FINALLY, sub_formulas=[disj], logical_op=None)
    spec.formulas = [f_formula]

    builder = DFABuilder()
    dfa_result = builder.build(spec)

    print(f"Formula: {f_formula.to_string()}")
    print(f"DFA: {dfa_result['num_states']} states, {dfa_result['num_transitions']} transitions")
    print(f"Partitions: {dfa_result['simplification_stats']['num_partitions']}")

    check = check_dfa_is_atomic(dfa_result['dfa_dot'], dfa_result.get('partition_map'))
    print(f"\nAtomicity Check:")
    print(f"  Total transitions: {check['analysis']['total_transitions']}")
    print(f"  Atomic: {check['analysis']['atomic_transitions']}")
    print(f"  Non-atomic: {check['analysis']['non_atomic_transitions']}")

    if not check['is_atomic']:
        print(f"\n❌ FAILED - Non-atomic labels found:")
        for from_s, to_s, label in check['non_atomic_labels']:
            print(f"  {from_s} -> {to_s} [label=\"{label}\"]")
        return False

    print(f"\n✅ PASSED - All transitions are atomic")
    return True


def test_until_operator():
    """Test: on(a, b) U clear(c)"""
    print("\n" + "="*80)
    print("TEST 6: Until Operator - on(a, b) U clear(c)")
    print("="*80)

    spec = LTLSpecification()
    spec.objects = ["a", "b", "c"]

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    spec.grounding_map = gmap

    on_a_b = LTLFormula(predicate="on_a_b", operator=None, sub_formulas=[], logical_op=None)
    clear_c = LTLFormula(predicate="clear_c", operator=None, sub_formulas=[], logical_op=None)
    u_formula = LTLFormula(predicate=None, operator=TemporalOperator.UNTIL, sub_formulas=[on_a_b, clear_c], logical_op=None)
    spec.formulas = [u_formula]

    builder = DFABuilder()
    dfa_result = builder.build(spec)

    print(f"Formula: {u_formula.to_string()}")
    print(f"DFA: {dfa_result['num_states']} states, {dfa_result['num_transitions']} transitions")
    print(f"Partitions: {dfa_result['simplification_stats']['num_partitions']}")

    check = check_dfa_is_atomic(dfa_result['dfa_dot'], dfa_result.get('partition_map'))
    print(f"\nAtomicity Check:")
    print(f"  Total transitions: {check['analysis']['total_transitions']}")
    print(f"  Atomic: {check['analysis']['atomic_transitions']}")
    print(f"  Non-atomic: {check['analysis']['non_atomic_transitions']}")

    if not check['is_atomic']:
        print(f"\n❌ FAILED - Non-atomic labels found:")
        for from_s, to_s, label in check['non_atomic_labels']:
            print(f"  {from_s} -> {to_s} [label=\"{label}\"]")
        return False

    print(f"\n✅ PASSED - All transitions are atomic")
    return True


if __name__ == "__main__":
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*25 + "DFA ROBUSTNESS TEST" + " "*34 + "║")
    print("║" + " "*20 + "Atomic Transition Verification" + " "*27 + "║")
    print("╚" + "═"*78 + "╝")

    tests = [
        ("Simple Finally", test_simple_finally),
        ("Conjunction", test_conjunction),
        ("Globally with Negation", test_globally_negation),
        ("Disjunction", test_disjunction),
        ("Complex Nested", test_complex_nested),
        ("Until Operator", test_until_operator),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\n{passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - DFA SIMPLIFIER IS ROBUST")
        print("="*80)
        print("\nConclusion:")
        print("  Every DFA type tested produces ONLY atomic transition labels.")
        print("  No complex boolean expressions found in any simplified DFA.")
        print("  The simplification is working correctly across all LTL operators.")
    else:
        print("\n⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
        sys.exit(1)
