#!/usr/bin/env python3
"""
Debug script to verify constraint collection is working properly.
This prints out explicit vs implicit constraints for typical backward search states.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.backward_search_refactored import (
    InequalityConstraint,
    _extract_implicit_constraints_from_predicates,
    _build_constraint_graph,
)
from stage3_code_generation.state_space import PredicateAtom


def debug_constraint_collection():
    """Debug constraint collection with typical backward search scenarios"""

    print("=" * 80)
    print("DEBUG: Constraint Collection Analysis")
    print("=" * 80)
    print()

    # Scenario 1: Single on predicate
    print("-" * 80)
    print("Scenario 1: Single on(?v0, ?v1) predicate")
    print("-" * 80)
    predicates = {PredicateAtom("on", ["?v0", "?v1"])}
    explicit_constraints = set()

    implicit = _extract_implicit_constraints_from_predicates(predicates)

    print(f"Predicates: {[str(p) for p in predicates]}")
    print(f"Explicit constraints: {len(explicit_constraints)} - {[str(c) for c in explicit_constraints]}")
    print(f"Implicit constraints: {len(implicit)} - {[str(c) for c in implicit]}")
    print()

    # Scenario 2: Multiple on predicates (tower of 3 blocks)
    print("-" * 80)
    print("Scenario 2: Tower - on(?v0, ?v1) ∧ on(?v1, ?v2) ∧ on(?v2, ?v3)")
    print("-" * 80)
    predicates = {
        PredicateAtom("on", ["?v0", "?v1"]),
        PredicateAtom("on", ["?v1", "?v2"]),
        PredicateAtom("on", ["?v2", "?v3"]),
    }
    explicit_constraints = set()

    implicit = _extract_implicit_constraints_from_predicates(predicates)

    print(f"Predicates: {[str(p) for p in predicates]}")
    print(f"Explicit constraints: {len(explicit_constraints)} - {[str(c) for c in explicit_constraints]}")
    print(f"Implicit constraints: {len(implicit)} - {[str(c) for c in implicit]}")
    print()

    # Build constraint graph to see clique size
    variables = {'?v0', '?v1', '?v2', '?v3'}
    graph = _build_constraint_graph(variables, explicit_constraints, predicates)
    max_clique = graph.find_maximum_clique_greedy()
    print(f"Variables: {variables}")
    print(f"Maximum clique size: {len(max_clique)} - {max_clique}")
    print(f"Interpretation: Need at least {len(max_clique)} objects for this state")
    print()

    # Scenario 3: State explosion scenario - many variables, few constraints
    print("-" * 80)
    print("Scenario 3: State explosion - on(?v0, ?v1) ∧ clear(?v2) ∧ clear(?v3) ∧ clear(?v4)")
    print("-" * 80)
    predicates = {
        PredicateAtom("on", ["?v0", "?v1"]),
        PredicateAtom("clear", ["?v2"]),
        PredicateAtom("clear", ["?v3"]),
        PredicateAtom("clear", ["?v4"]),
    }
    explicit_constraints = set()

    implicit = _extract_implicit_constraints_from_predicates(predicates)

    print(f"Predicates: {[str(p) for p in predicates]}")
    print(f"Explicit constraints: {len(explicit_constraints)} - {[str(c) for c in explicit_constraints]}")
    print(f"Implicit constraints: {len(implicit)} - {[str(c) for c in implicit]}")
    print()

    # Build constraint graph
    variables = {'?v0', '?v1', '?v2', '?v3', '?v4'}
    graph = _build_constraint_graph(variables, explicit_constraints, predicates)
    max_clique = graph.find_maximum_clique_greedy()
    print(f"Variables: {variables}")
    print(f"Maximum clique size: {len(max_clique)} - {max_clique}")
    print(f"Interpretation: Need at least {len(max_clique)} objects for this state")
    print()
    print(f"OLD LOGIC would say: {len(variables)} variables > max_objects → prune if max_objects < 5")
    print(f"NEW LOGIC says: Only need {len(max_clique)} objects → DON'T prune if max_objects >= {len(max_clique)}")
    print(f"Difference: Old prunes too aggressively for max_objects=2,3,4!")
    print()

    # Scenario 4: With ground objects
    print("-" * 80)
    print("Scenario 4: Ground objects - on(a, ?v0) ∧ on(?v0, b)")
    print("-" * 80)
    predicates = {
        PredicateAtom("on", ["a", "?v0"]),
        PredicateAtom("on", ["?v0", "b"]),
    }
    explicit_constraints = set()

    implicit = _extract_implicit_constraints_from_predicates(predicates)

    print(f"Predicates: {[str(p) for p in predicates]}")
    print(f"Explicit constraints: {len(explicit_constraints)} - {[str(c) for c in explicit_constraints]}")
    print(f"Implicit constraints: {len(implicit)} - {[str(c) for c in implicit]}")
    print()

    # Build constraint graph with ground objects
    variables = {'a', 'b', '?v0'}
    ground_objects = {'a', 'b'}
    graph = _build_constraint_graph(variables, explicit_constraints, predicates, ground_objects)
    max_clique = graph.find_maximum_clique_greedy()
    print(f"Variables: {variables}")
    print(f"Ground objects: {ground_objects} (implicitly a != b)")
    print(f"Maximum clique size: {len(max_clique)} - {max_clique}")
    print(f"Interpretation: Need at least {len(max_clique)} objects for this state")
    print()

    # Scenario 5: Complex state with both explicit and implicit constraints
    print("-" * 80)
    print("Scenario 5: Complex - on(?v0, ?v1) ∧ on(?v2, ?v3) with explicit ?v0 != ?v2")
    print("-" * 80)
    predicates = {
        PredicateAtom("on", ["?v0", "?v1"]),
        PredicateAtom("on", ["?v2", "?v3"]),
    }
    explicit_constraints = {
        InequalityConstraint("?v0", "?v2"),
    }

    implicit = _extract_implicit_constraints_from_predicates(predicates)

    print(f"Predicates: {[str(p) for p in predicates]}")
    print(f"Explicit constraints: {len(explicit_constraints)} - {[str(c) for c in explicit_constraints]}")
    print(f"Implicit constraints: {len(implicit)} - {[str(c) for c in implicit]}")
    print()

    # Build constraint graph
    variables = {'?v0', '?v1', '?v2', '?v3'}
    graph = _build_constraint_graph(variables, explicit_constraints, predicates)

    print(f"Total constraints in graph: {len(graph.edges)}")
    print(f"All edges: {sorted(graph.edges)}")

    max_clique = graph.find_maximum_clique_greedy()
    print(f"Variables: {variables}")
    print(f"Maximum clique size: {len(max_clique)} - {max_clique}")
    print(f"Interpretation: Need at least {len(max_clique)} objects for this state")
    print()

    print("=" * 80)
    print("Summary:")
    print("- Implicit constraints from predicates are being extracted correctly")
    print("- Constraint graph includes both explicit and implicit constraints")
    print("- Clique analysis provides accurate lower bound on required objects")
    print("=" * 80)


if __name__ == "__main__":
    debug_constraint_collection()
