"""
Simplified tests for critical fixes - Problems 1 and 2.

Tests backtracking unification and context preservation using internal methods.
"""

import pytest
from src.stage3_code_generation.abstract_state import AbstractState, ConstraintSet, Constraint
from src.stage3_code_generation.state_space import PredicateAtom
from src.stage3_code_generation.variable_planner import VariablePlanner
from src.stage3_code_generation.unification import Substitution
from utils.pddl_parser import PDDLDomain


class TestBacktrackingUnification:
    """Tests for Problem 1 fix: Backtracking unification"""

    def setup_method(self):
        """Create minimal planner for testing"""
        # Create minimal domain with no actions (we only test unification)
        self.planner = VariablePlanner.__new__(VariablePlanner)
        self.planner._abstract_actions = []
        self.planner._mutexes = {}

    def test_sequential_dependency_unification_success(self):
        """
        Test backtracking finds valid unification for sequentially dependent preconditions.

        State: {on(a, b), on(b, c)}
        Preconditions: [on(?X, ?Y), on(?Y, ?Z)]  # Shared variable ?Y

        Correct unification: ?X=a, ?Y=b, ?Z=c
        """
        # Create state predicates
        on_a_b = PredicateAtom("on", ["a", "b"])
        on_b_c = PredicateAtom("on", ["b", "c"])
        state_preds = frozenset([on_a_b, on_b_c])

        # Create preconditions with shared variable
        precond1 = PredicateAtom("on", ["?X", "?Y"])
        precond2 = PredicateAtom("on", ["?Y", "?Z"])
        preconditions = [precond1, precond2]

        # Create constraints
        constraints = ConstraintSet()

        # Test unification
        result_subst, unsatisfied = self.planner._find_consistent_unification(
            preconditions, state_preds, constraints
        )

        # Should succeed
        assert result_subst is not None, "Should find valid unification"
        assert len(unsatisfied) == 0, "All preconditions should be satisfied"

        # Verify bindings
        assert result_subst.apply("?X") == "a"
        assert result_subst.apply("?Y") == "b"
        assert result_subst.apply("?Z") == "c"

    def test_backtracking_with_constraints(self):
        """
        Test backtracking respects inequality constraints.

        State: {on(a, a), on(a, b), on(b, c)}
        Preconditions: [on(?X, ?Y), on(?Y, ?Z)]
        Constraints: ?X != ?Y, ?Y != ?Z

        Should skip on(a, a) and find valid chain
        """
        state_preds = frozenset([
            PredicateAtom("on", ["a", "a"]),  # Invalid (violates ?X != ?Y)
            PredicateAtom("on", ["a", "b"]),
            PredicateAtom("on", ["b", "c"])
        ])

        preconditions = [
            PredicateAtom("on", ["?X", "?Y"]),
            PredicateAtom("on", ["?Y", "?Z"])
        ]

        # Add inequality constraints
        constraints = ConstraintSet({
            Constraint("?X", "?Y", Constraint.INEQUALITY),
            Constraint("?Y", "?Z", Constraint.INEQUALITY)
        })

        result_subst, unsatisfied = self.planner._find_consistent_unification(
            preconditions, state_preds, constraints
        )

        # Should find valid solution
        assert result_subst is not None
        assert len(unsatisfied) == 0

        # Should bind to valid chain (not the on(a,a))
        x_val = result_subst.apply("?X")
        y_val = result_subst.apply("?Y")
        z_val = result_subst.apply("?Z")

        # Verify constraints satisfied
        assert x_val != y_val, "Should satisfy ?X != ?Y"
        assert y_val != z_val, "Should satisfy ?Y != ?Z"

    def test_unsatisfiable_preconditions(self):
        """
        Test that backtracking correctly identifies unsatisfiable cases.

        State: {on(a, b), on(c, d)}  # No shared variables
        Preconditions: [on(?X, ?Y), on(?Y, ?Z)]  # Require shared variable

        Should return None, indicating no solution
        """
        state_preds = frozenset([
            PredicateAtom("on", ["a", "b"]),
            PredicateAtom("on", ["c", "d"])
        ])

        preconditions = [
            PredicateAtom("on", ["?X", "?Y"]),
            PredicateAtom("on", ["?Y", "?Z"])
        ]

        constraints = ConstraintSet()

        result_subst, unsatisfied = self.planner._find_consistent_unification(
            preconditions, state_preds, constraints
        )

        # Should fail - no valid unification
        assert result_subst is None
        assert len(unsatisfied) == 2

    def test_multiple_valid_solutions(self):
        """
        Test that backtracking finds A valid solution when multiple exist.

        State: {on(a, b), on(b, c), on(c, d)}
        Preconditions: [on(?X, ?Y), on(?Y, ?Z)]

        Valid solutions: (a,b,c) or (b,c,d)
        Should find at least one
        """
        state_preds = frozenset([
            PredicateAtom("on", ["a", "b"]),
            PredicateAtom("on", ["b", "c"]),
            PredicateAtom("on", ["c", "d"])
        ])

        preconditions = [
            PredicateAtom("on", ["?X", "?Y"]),
            PredicateAtom("on", ["?Y", "?Z"])
        ]

        constraints = ConstraintSet()

        result_subst, unsatisfied = self.planner._find_consistent_unification(
            preconditions, state_preds, constraints
        )

        assert result_subst is not None
        assert len(unsatisfied) == 0

        # Verify it's one of the valid solutions
        chain = (
            result_subst.apply("?X"),
            result_subst.apply("?Y"),
            result_subst.apply("?Z")
        )
        assert chain in [("a", "b", "c"), ("b", "c", "d")]

    def test_no_predicate_reuse(self):
        """
        Test that each state predicate can only be used once.

        State: {on(a, b)}
        Preconditions: [on(?X, ?Y), on(?Z, ?W)]

        Should fail - only one on() predicate available
        """
        state_preds = frozenset([
            PredicateAtom("on", ["a", "b"])
        ])

        preconditions = [
            PredicateAtom("on", ["?X", "?Y"]),
            PredicateAtom("on", ["?Z", "?W"])
        ]

        constraints = ConstraintSet()

        result_subst, unsatisfied = self.planner._find_consistent_unification(
            preconditions, state_preds, constraints
        )

        # Should fail - can't match both preconditions with one state predicate
        assert result_subst is None
        assert len(unsatisfied) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
