#!/usr/bin/env python3
"""
Test constraint-aware pruning to ensure the bug fix works correctly.

This tests that the new pruning logic correctly handles:
1. Variables without constraints (can share objects) - should NOT prune
2. Variables with full mutual constraints (must be different) - should prune
3. Ground objects creating implicit constraints
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.backward_search_refactored import (
    InequalityConstraint,
    _compute_minimum_objects_needed,
    _should_prune_state_constraint_aware,
    _extract_implicit_constraints_from_predicates,
)
from stage3_code_generation.state_space import PredicateAtom


def test_constraint_aware_pruning():
    """Test the constraint-aware pruning logic"""

    print("=" * 80)
    print("Test 1: Old logic would prune, new logic should NOT prune")
    print("=" * 80)

    # State with 4 variables but only 1 constraint
    # ?v3 and ?v4 have NO constraint → can bind to same object
    variables = {'?v1', '?v2', '?v3', '?v4'}
    constraints = {
        InequalityConstraint('?v1', '?v2'),  # ?v1 != ?v2
    }
    max_objects = 2

    print(f"Variables: {variables}")
    print(f"Constraints: {[str(c) for c in constraints]}")
    print(f"Max objects: {max_objects}")

    min_needed = _compute_minimum_objects_needed(variables, constraints, predicates=None)
    should_prune = _should_prune_state_constraint_aware(variables, constraints, max_objects, predicates=None)

    print(f"Minimum objects needed (lower bound): {min_needed}")
    print(f"Should prune: {should_prune}")
    print(f"OLD logic would say: {len(variables)} > {max_objects} = {len(variables) > max_objects}")
    print()

    # This should NOT be pruned because:
    # ?v1 != ?v2 requires 2 objects
    # ?v3 and ?v4 can both be same as ?v1 or ?v2 (or a new object, but not needed)
    if should_prune:
        print("❌ FAILED: State was incorrectly pruned!")
        print("   ?v3 and ?v4 have no constraints, so they can share objects with ?v1 or ?v2")
        return False
    else:
        print("✅ PASSED: State correctly NOT pruned")
    print()

    print("=" * 80)
    print("Test 2: Should prune - clique exceeds max_objects")
    print("=" * 80)

    # State where all 3 variables must be different
    variables = {'?v1', '?v2', '?v3'}
    constraints = {
        InequalityConstraint('?v1', '?v2'),
        InequalityConstraint('?v1', '?v3'),
        InequalityConstraint('?v2', '?v3'),
    }
    max_objects = 2

    print(f"Variables: {variables}")
    print(f"Constraints: {[str(c) for c in constraints]}")
    print(f"Max objects: {max_objects}")

    min_needed = _compute_minimum_objects_needed(variables, constraints, predicates=None)
    should_prune = _should_prune_state_constraint_aware(variables, constraints, max_objects, predicates=None)

    print(f"Minimum objects needed (lower bound): {min_needed}")
    print(f"Should prune: {should_prune}")

    if not should_prune:
        print("❌ FAILED: State should have been pruned!")
        print("   All 3 variables are mutually constrained, needing 3 objects")
        return False
    else:
        print("✅ PASSED: State correctly pruned")
    print()

    print("=" * 80)
    print("Test 3: Ground objects create implicit constraints")
    print("=" * 80)

    # State with ground objects a, b and variable ?v1
    variables = {'a', 'b', '?v1'}
    constraints = set()  # No explicit constraints
    max_objects = 2
    ground_objects = {'a', 'b'}  # These are ground → implicitly a != b

    print(f"Variables: {variables}")
    print(f"Ground objects: {ground_objects}")
    print(f"Constraints: {constraints}")
    print(f"Max objects: {max_objects}")

    min_needed = _compute_minimum_objects_needed(variables, constraints, predicates=None, ground_objects=ground_objects)
    should_prune = _should_prune_state_constraint_aware(variables, constraints, max_objects, predicates=None, ground_objects=ground_objects)

    print(f"Minimum objects needed (lower bound): {min_needed}")
    print(f"Should prune: {should_prune}")

    # a != b implicitly (ground objects)
    # ?v1 can be same as a or b
    # So we need 2 objects: a and b
    if should_prune:
        print("❌ FAILED: State was incorrectly pruned!")
        print("   ?v1 can be same as a or b, so only 2 objects needed")
        return False
    else:
        print("✅ PASSED: State correctly NOT pruned")
    print()

    print("=" * 80)
    print("Test 4: Ground objects with constraint to variable")
    print("=" * 80)

    # a and b are ground, ?v1 must differ from both
    variables = {'a', 'b', '?v1'}
    constraints = {
        InequalityConstraint('a', '?v1'),
        InequalityConstraint('b', '?v1'),
    }
    max_objects = 2
    ground_objects = {'a', 'b'}

    print(f"Variables: {variables}")
    print(f"Ground objects: {ground_objects}")
    print(f"Constraints: {[str(c) for c in constraints]}")
    print(f"Max objects: {max_objects}")

    min_needed = _compute_minimum_objects_needed(variables, constraints, predicates=None, ground_objects=ground_objects)
    should_prune = _should_prune_state_constraint_aware(variables, constraints, max_objects, predicates=None, ground_objects=ground_objects)

    print(f"Minimum objects needed (lower bound): {min_needed}")
    print(f"Should prune: {should_prune}")

    # a != b (implicit), a != ?v1, b != ?v1
    # This is a 3-clique! Needs 3 objects.
    if not should_prune:
        print("❌ FAILED: State should have been pruned!")
        print("   3-clique (a, b, ?v1) needs 3 objects but only 2 available")
        return False
    else:
        print("✅ PASSED: State correctly pruned")
    print()

    print("=" * 80)
    print("Test 5: Blocksworld example - on(?v0, ?v1) with 2 objects")
    print("=" * 80)

    # Typical blocksworld predicate: on(?v0, ?v1)
    # The predicate implicitly requires ?v0 != ?v1 (can't stack on itself)
    variables = {'?v0', '?v1'}
    constraints = {
        InequalityConstraint('?v0', '?v1'),  # From on predicate
    }
    max_objects = 2

    print(f"Variables: {variables}")
    print(f"Constraints: {[str(c) for c in constraints]}")
    print(f"Max objects: {max_objects}")

    min_needed = _compute_minimum_objects_needed(variables, constraints, predicates=None)
    should_prune = _should_prune_state_constraint_aware(variables, constraints, max_objects, predicates=None)

    print(f"Minimum objects needed (lower bound): {min_needed}")
    print(f"Should prune: {should_prune}")

    if should_prune:
        print("❌ FAILED: State was incorrectly pruned!")
        print("   Only need 2 objects for on(?v0, ?v1)")
        return False
    else:
        print("✅ PASSED: State correctly NOT pruned")
    print()

    print("=" * 80)
    print("Test 6: Implicit constraints from predicates - on(?v0, ?v1)")
    print("=" * 80)

    # State with on(?v0, ?v1) predicate
    # This should extract implicit constraint ?v0 != ?v1
    predicates = {PredicateAtom("on", ["?v0", "?v1"])}
    variables = {'?v0', '?v1'}
    constraints = set()  # No explicit constraints
    max_objects = 1  # Only 1 object available

    print(f"Predicates: {[str(p) for p in predicates]}")
    print(f"Variables: {variables}")
    print(f"Explicit constraints: {constraints}")
    print(f"Max objects: {max_objects}")

    # Extract implicit constraints
    implicit = _extract_implicit_constraints_from_predicates(predicates)
    print(f"Implicit constraints from predicates: {[str(c) for c in implicit]}")

    min_needed = _compute_minimum_objects_needed(variables, constraints, predicates)
    should_prune = _should_prune_state_constraint_aware(variables, constraints, max_objects, predicates)

    print(f"Minimum objects needed (with implicit): {min_needed}")
    print(f"Should prune: {should_prune}")

    # Should be pruned because on(?v0, ?v1) implicitly requires ?v0 != ?v1
    # So we need 2 objects, but only 1 available
    if not should_prune:
        print("❌ FAILED: State should have been pruned!")
        print("   on(?v0, ?v1) implicitly requires ?v0 != ?v1, needing 2 objects")
        return False
    else:
        print("✅ PASSED: State correctly pruned using implicit constraints")
    print()

    print("=" * 80)
    print("Test 7: Multiple implicit constraints - on(?v0, ?v1) ∧ on(?v1, ?v2)")
    print("=" * 80)

    # State with multiple on predicates creating a chain
    predicates = {
        PredicateAtom("on", ["?v0", "?v1"]),
        PredicateAtom("on", ["?v1", "?v2"]),
    }
    variables = {'?v0', '?v1', '?v2'}
    constraints = set()  # No explicit constraints
    max_objects = 2

    print(f"Predicates: {[str(p) for p in predicates]}")
    print(f"Variables: {variables}")
    print(f"Explicit constraints: {constraints}")
    print(f"Max objects: {max_objects}")

    # Extract implicit constraints
    implicit = _extract_implicit_constraints_from_predicates(predicates)
    print(f"Implicit constraints from predicates: {[str(c) for c in implicit]}")

    min_needed = _compute_minimum_objects_needed(variables, constraints, predicates)
    should_prune = _should_prune_state_constraint_aware(variables, constraints, max_objects, predicates)

    print(f"Minimum objects needed (with implicit): {min_needed}")
    print(f"Should prune: {should_prune}")

    # Should NOT be pruned:
    # Implicit constraints: ?v0 != ?v1, ?v1 != ?v2
    # But NO constraint between ?v0 and ?v2, so they can be the same
    # So we need only 2 objects: {?v0=a, ?v1=b, ?v2=a}
    if should_prune:
        print("❌ FAILED: State was incorrectly pruned!")
        print("   ?v0 and ?v2 have no constraint, so they can be the same object")
        return False
    else:
        print("✅ PASSED: State correctly NOT pruned")
    print()

    print("=" * 80)
    print("All tests passed! ✅")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_constraint_aware_pruning()
    sys.exit(0 if success else 1)
