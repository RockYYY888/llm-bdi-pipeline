"""
Unification Algorithm for Lifted Planning

Implements Robinson's unification algorithm for first-order logic terms.
Used to match action preconditions against abstract state predicates.

Example:
    State: {on(?X, ?Y), clear(?Z)}
    Action precondition: on(?A, ?B)
    Unification: σ = {?A/?X, ?B/?Y}
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List, Set, Tuple
from dataclasses import dataclass

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import PredicateAtom


@dataclass(frozen=True)
class Substitution:
    """
    Represents a variable substitution (mapping from variables to terms)

    Example: {?X: ?Y, ?A: a, ?B: ?C}

    Attributes:
        mapping: Dictionary from variable names to their substitutions
    """
    mapping: Dict[str, str]

    def __init__(self, mapping: Dict[str, str] = None):
        # Use object.__setattr__ to bypass frozen dataclass restriction
        object.__setattr__(self, 'mapping', dict(mapping) if mapping else {})

    def apply(self, term: str) -> str:
        """
        Apply substitution to a term recursively

        Args:
            term: Variable or constant

        Returns:
            Substituted term
        """
        # Follow substitution chain: ?X -> ?Y -> ?Z -> a
        seen = set()
        current = term
        while current in self.mapping:
            if current in seen:
                # Cycle detected - should not happen in valid unification
                break
            seen.add(current)
            current = self.mapping[current]
        return current

    def apply_to_predicate(self, pred: PredicateAtom) -> PredicateAtom:
        """
        Apply substitution to all arguments in a predicate

        Args:
            pred: Predicate atom

        Returns:
            New predicate with substitutions applied
        """
        new_args = [self.apply(arg) for arg in pred.args]
        return PredicateAtom(pred.name, new_args, pred.negated)

    def compose(self, other: 'Substitution') -> 'Substitution':
        """
        Compose two substitutions: self ∘ other

        First apply 'other', then apply 'self'

        Args:
            other: Another substitution

        Returns:
            Composed substitution
        """
        # Apply self to all values in other
        new_mapping = {var: self.apply(term) for var, term in other.mapping.items()}

        # Add mappings from self that are not in other
        for var, term in self.mapping.items():
            if var not in new_mapping:
                new_mapping[var] = term

        return Substitution(new_mapping)

    def __str__(self):
        if not self.mapping:
            return "{}"
        items = [f"{var}/{term}" for var, term in sorted(self.mapping.items())]
        return "{" + ", ".join(items) + "}"

    def __bool__(self):
        """A substitution is truthy if it has any mappings"""
        return bool(self.mapping)


class Unifier:
    """
    Unification engine for lifted planning

    Implements Robinson's unification algorithm with occurs check.
    """

    @staticmethod
    def unify_terms(term1: str, term2: str, subst: Substitution = None) -> Optional[Substitution]:
        """
        Unify two terms (variables or constants)

        Args:
            term1: First term (e.g., "?X", "a", "?Y")
            term2: Second term
            subst: Current substitution (for recursive calls)

        Returns:
            Unified substitution, or None if unification fails
        """
        if subst is None:
            subst = Substitution()

        # Apply current substitution
        term1 = subst.apply(term1)
        term2 = subst.apply(term2)

        # Same terms unify trivially
        if term1 == term2:
            return subst

        # Check if both are variables
        is_var1 = term1.startswith('?')
        is_var2 = term2.startswith('?')

        if is_var1 and is_var2:
            # Both variables: bind term1 to term2
            new_mapping = dict(subst.mapping)
            new_mapping[term1] = term2
            return Substitution(new_mapping)
        elif is_var1:
            # term1 is variable, term2 is constant or variable
            # Occurs check: ensure term1 doesn't appear in term2's substitution chain
            if Unifier._occurs_check(term1, term2, subst):
                return None
            new_mapping = dict(subst.mapping)
            new_mapping[term1] = term2
            return Substitution(new_mapping)
        elif is_var2:
            # term2 is variable, term1 is constant or variable
            if Unifier._occurs_check(term2, term1, subst):
                return None
            new_mapping = dict(subst.mapping)
            new_mapping[term2] = term1
            return Substitution(new_mapping)
        else:
            # Both are constants - unification fails if they're different
            return None

    @staticmethod
    def unify_predicates(pred1: PredicateAtom, pred2: PredicateAtom,
                        subst: Substitution = None) -> Optional[Substitution]:
        """
        Unify two predicates

        Predicates unify if:
        1. Same predicate name
        2. Same negation status
        3. All arguments unify pairwise

        Args:
            pred1: First predicate
            pred2: Second predicate
            subst: Current substitution

        Returns:
            Unified substitution, or None if unification fails
        """
        if subst is None:
            subst = Substitution()

        # Check predicate name and negation
        if pred1.name != pred2.name or pred1.negated != pred2.negated:
            return None

        # Check arity
        if len(pred1.args) != len(pred2.args):
            return None

        # Unify arguments pairwise
        current_subst = subst
        for arg1, arg2 in zip(pred1.args, pred2.args):
            current_subst = Unifier.unify_terms(arg1, arg2, current_subst)
            if current_subst is None:
                return None

        return current_subst

    @staticmethod
    def unify_predicate_list(state_preds: Set[PredicateAtom],
                           query_preds: List[PredicateAtom],
                           subst: Substitution = None) -> Optional[Substitution]:
        """
        Find a substitution that unifies all query predicates with state predicates

        This is used to match action preconditions against a state.

        Args:
            state_preds: Set of predicates in the current state
            query_preds: List of predicates to match (e.g., action preconditions)
            subst: Starting substitution

        Returns:
            Unified substitution, or None if any predicate cannot be matched
        """
        if subst is None:
            subst = Substitution()

        current_subst = subst

        for query_pred in query_preds:
            # Try to find a matching predicate in state
            found = False
            for state_pred in state_preds:
                # Try to unify
                unified = Unifier.unify_predicates(query_pred, state_pred, current_subst)
                if unified is not None:
                    current_subst = unified
                    found = True
                    break

            if not found:
                # Could not unify this predicate - matching fails
                return None

        return current_subst

    @staticmethod
    def _occurs_check(var: str, term: str, subst: Substitution) -> bool:
        """
        Check if variable occurs in term (prevents infinite substitutions)

        Args:
            var: Variable to check
            term: Term to check in
            subst: Current substitution

        Returns:
            True if var occurs in term's substitution chain
        """
        current = term
        seen = set()
        while current in subst.mapping:
            if current == var:
                return True
            if current in seen:
                break
            seen.add(current)
            current = subst.mapping[current]
        return current == var


# Test functions
def test_unification_basic():
    """Test basic unification"""
    print("="*80)
    print("Testing Unification - Basic")
    print("="*80)

    # Test 1: Unify two variables
    subst = Unifier.unify_terms("?X", "?Y")
    print(f"unify(?X, ?Y) = {subst}")
    assert subst is not None
    assert subst.apply("?X") == "?Y"

    # Test 2: Unify variable with constant
    subst = Unifier.unify_terms("?X", "a")
    print(f"unify(?X, a) = {subst}")
    assert subst is not None
    assert subst.apply("?X") == "a"

    # Test 3: Unify two constants (same)
    subst = Unifier.unify_terms("a", "a")
    print(f"unify(a, a) = {subst}")
    assert subst is not None

    # Test 4: Unify two constants (different) - should fail
    subst = Unifier.unify_terms("a", "b")
    print(f"unify(a, b) = {subst}")
    assert subst is None

    print("✓ All basic tests passed\n")


def test_unification_predicates():
    """Test predicate unification"""
    print("="*80)
    print("Testing Unification - Predicates")
    print("="*80)

    # Test 1: Unify on(?X, ?Y) with on(a, b)
    pred1 = PredicateAtom("on", ["?X", "?Y"])
    pred2 = PredicateAtom("on", ["a", "b"])
    subst = Unifier.unify_predicates(pred1, pred2)
    print(f"unify(on(?X, ?Y), on(a, b)) = {subst}")
    assert subst is not None
    assert subst.apply("?X") == "a"
    assert subst.apply("?Y") == "b"

    # Test 2: Unify on(?X, ?X) with on(a, b) - should fail
    pred1 = PredicateAtom("on", ["?X", "?X"])
    pred2 = PredicateAtom("on", ["a", "b"])
    subst = Unifier.unify_predicates(pred1, pred2)
    print(f"unify(on(?X, ?X), on(a, b)) = {subst}")
    assert subst is None

    # Test 3: Unify on(?X, ?X) with on(a, a) - should succeed
    pred1 = PredicateAtom("on", ["?X", "?X"])
    pred2 = PredicateAtom("on", ["a", "a"])
    subst = Unifier.unify_predicates(pred1, pred2)
    print(f"unify(on(?X, ?X), on(a, a)) = {subst}")
    assert subst is not None
    assert subst.apply("?X") == "a"

    print("✓ All predicate tests passed\n")


def test_unification_composition():
    """Test substitution composition"""
    print("="*80)
    print("Testing Unification - Composition")
    print("="*80)

    # Create two substitutions
    subst1 = Substitution({"?X": "?Y"})
    subst2 = Substitution({"?Y": "a"})

    # Compose them
    composed = subst1.compose(subst2)
    print(f"subst1 = {subst1}")
    print(f"subst2 = {subst2}")
    print(f"composed = {composed}")

    # Test application
    print(f"composed.apply('?X') = {composed.apply('?X')}")
    assert composed.apply("?X") == "a"

    print("✓ Composition tests passed\n")


if __name__ == "__main__":
    test_unification_basic()
    test_unification_predicates()
    test_unification_composition()
    print("All unification tests passed!")
