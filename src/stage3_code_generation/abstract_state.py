"""
Abstract State Representation for Lifted Planning

An abstract state contains:
1. Abstract predicates with variables (e.g., on(?X, ?Y), clear(?Z))
2. Variable constraints (e.g., ?X != ?Y, ?X = ?Z)

This enables true lifted planning by maintaining constraints without grounding.
"""

import sys
from pathlib import Path
from typing import Set, FrozenSet, Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import PredicateAtom
from stage3_code_generation.unification import Substitution


@dataclass(frozen=True)
class Constraint:
    """
    Represents a constraint between two variables

    Types:
    - INEQUALITY: ?X != ?Y (variables must be different)
    - EQUALITY: ?X = ?Y (variables must be same)
    """
    INEQUALITY = "!="
    EQUALITY = "="

    var1: str
    var2: str
    constraint_type: str  # "!=" or "="

    def __post_init__(self):
        # Normalize constraint: always put lexicographically smaller var first
        if self.var1 > self.var2:
            object.__setattr__(self, 'var1', self.var2)
            object.__setattr__(self, 'var2', self.var1)

    def is_satisfied(self, subst: Substitution) -> bool:
        """
        Check if constraint is satisfied under substitution

        Args:
            subst: Variable substitution

        Returns:
            True if constraint is satisfied
        """
        val1 = subst.apply(self.var1)
        val2 = subst.apply(self.var2)

        if self.constraint_type == Constraint.INEQUALITY:
            return val1 != val2
        elif self.constraint_type == Constraint.EQUALITY:
            return val1 == val2
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")

    def __str__(self):
        return f"{self.var1} {self.constraint_type} {self.var2}"


@dataclass(frozen=True)
class ConstraintSet:
    """
    Set of constraints on variables

    Immutable for use as dict key in state graph.
    """
    constraints: FrozenSet[Constraint]

    def __init__(self, constraints: Set[Constraint] = None):
        if constraints is None:
            constraints = frozenset()
        elif isinstance(constraints, set):
            constraints = frozenset(constraints)
        object.__setattr__(self, 'constraints', constraints)

    def add(self, constraint: Constraint) -> 'ConstraintSet':
        """
        Add a constraint, returning new ConstraintSet

        Args:
            constraint: Constraint to add

        Returns:
            New ConstraintSet with added constraint
        """
        new_constraints = set(self.constraints)
        new_constraints.add(constraint)
        return ConstraintSet(new_constraints)

    def is_consistent(self, subst: Substitution = None) -> bool:
        """
        Check if constraint set is consistent

        Args:
            subst: Optional substitution to apply before checking

        Returns:
            True if all constraints are consistent
        """
        if not subst:
            subst = Substitution()

        # Build equivalence classes from equality constraints
        equiv_classes: Dict[str, Set[str]] = {}

        for constraint in self.constraints:
            if constraint.constraint_type == Constraint.EQUALITY:
                val1 = subst.apply(constraint.var1)
                val2 = subst.apply(constraint.var2)

                # Find or create equivalence classes
                class1 = equiv_classes.get(val1, {val1})
                class2 = equiv_classes.get(val2, {val2})

                # Merge classes
                merged = class1 | class2
                for var in merged:
                    equiv_classes[var] = merged

        # Check inequality constraints
        for constraint in self.constraints:
            if constraint.constraint_type == Constraint.INEQUALITY:
                val1 = subst.apply(constraint.var1)
                val2 = subst.apply(constraint.var2)

                # If both values are in same equivalence class, inconsistent
                if val1 in equiv_classes.get(val2, set()):
                    return False

                # If both are constants and equal, inconsistent
                if not val1.startswith('?') and not val2.startswith('?') and val1 == val2:
                    return False

        return True

    def apply_substitution(self, subst: Substitution) -> 'ConstraintSet':
        """
        Apply substitution to all constraints

        Args:
            subst: Substitution to apply

        Returns:
            New ConstraintSet with substitution applied
        """
        new_constraints = set()
        for constraint in self.constraints:
            new_var1 = subst.apply(constraint.var1)
            new_var2 = subst.apply(constraint.var2)

            # Skip trivial constraints (var = var)
            if new_var1 == new_var2:
                if constraint.constraint_type == Constraint.INEQUALITY:
                    # Inconsistent: x != x
                    return None
                else:
                    # Trivial: x = x
                    continue

            new_constraints.add(Constraint(new_var1, new_var2, constraint.constraint_type))

        return ConstraintSet(new_constraints)

    def merge(self, other: 'ConstraintSet') -> Optional['ConstraintSet']:
        """
        Merge two constraint sets

        Args:
            other: Another constraint set

        Returns:
            Merged constraint set, or None if inconsistent
        """
        merged = ConstraintSet(self.constraints | other.constraints)
        if not merged.is_consistent():
            return None
        return merged

    def __str__(self):
        if not self.constraints:
            return "{}"
        return "{" + ", ".join(str(c) for c in sorted(self.constraints, key=str)) + "}"

    def __bool__(self):
        return bool(self.constraints)


@dataclass(frozen=True)
class AbstractState:
    """
    Abstract state for lifted planning

    Contains:
    - predicates: Set of abstract predicates with variables (e.g., on(?X, ?Y))
    - constraints: Constraints on variables
    - depth: Depth in state graph (for search)

    Example:
        predicates = {on(?X, ?Y), clear(?X), handempty}
        constraints = {?X != ?Y}
    """
    predicates: FrozenSet[PredicateAtom]
    constraints: ConstraintSet
    depth: int = 0

    def __init__(self, predicates: Set[PredicateAtom], constraints: ConstraintSet = None,
                 depth: int = 0):
        if isinstance(predicates, set):
            predicates = frozenset(predicates)
        if constraints is None:
            constraints = ConstraintSet()

        object.__setattr__(self, 'predicates', predicates)
        object.__setattr__(self, 'constraints', constraints)
        object.__setattr__(self, 'depth', depth)

    def apply_substitution(self, subst: Substitution) -> Optional['AbstractState']:
        """
        Apply substitution to state

        Args:
            subst: Substitution to apply

        Returns:
            New state with substitution applied, or None if inconsistent
        """
        # Apply to predicates
        new_predicates = {subst.apply_to_predicate(p) for p in self.predicates}

        # Apply to constraints
        new_constraints = self.constraints.apply_substitution(subst)
        if new_constraints is None:
            return None

        # Check consistency
        if not new_constraints.is_consistent(subst):
            return None

        return AbstractState(new_predicates, new_constraints, self.depth)

    def get_variables(self) -> Set[str]:
        """
        Get all variables used in this state

        Returns:
            Set of variable names
        """
        variables = set()
        for pred in self.predicates:
            for arg in pred.args:
                if arg.startswith('?'):
                    variables.add(arg)
        return variables

    def extract_implicit_constraints(self) -> ConstraintSet:
        """
        Extract implicit constraints from predicates

        DOMAIN-INDEPENDENT: For any binary predicate P(?X, ?Y) where ?X and ?Y
        are different variables, we infer ?X != ?Y (reflexivity constraint).

        This is a reasonable general assumption: binary relations typically
        relate different objects (e.g., on(?X, ?Y), at(?X, ?Y), connected(?X, ?Y)).

        Returns:
            ConstraintSet with implicit constraints
        """
        constraints = set(self.constraints.constraints)

        # CRITICAL FIX #3: Domain-independent constraint extraction
        # Apply to ALL binary predicates, not just "on"
        for pred in self.predicates:
            # For any binary predicate P(?X, ?Y)
            # If both arguments are different variables, infer ?X != ?Y
            if len(pred.args) == 2:
                arg0, arg1 = pred.args
                # Both are variables and they're different variable names
                if (arg0.startswith('?') and arg1.startswith('?') and arg0 != arg1):
                    constraints.add(Constraint(arg0, arg1, Constraint.INEQUALITY))

        return ConstraintSet(constraints)

    def __str__(self):
        # Predicates
        if self.predicates:
            pred_str = ", ".join(str(p) for p in sorted(self.predicates, key=str))
            result = "{" + pred_str + "}"
        else:
            result = "{}"

        if self.constraints:
            result += f" where {self.constraints}"

        return result

    def __hash__(self):
        # Hash based on predicates and constraints
        return hash((self.predicates, self.constraints))

    def __eq__(self, other):
        if not isinstance(other, AbstractState):
            return False
        return (self.predicates == other.predicates and
                self.constraints == other.constraints)


def test_abstract_state():
    """Test abstract state operations"""
    print("="*80)
    print("Testing Abstract State")
    print("="*80)

    # Create abstract state
    predicates = {
        PredicateAtom("on", ["?X", "?Y"]),
        PredicateAtom("clear", ["?Z"])
    }
    constraints = ConstraintSet({
        Constraint("?X", "?Y", Constraint.INEQUALITY),
        Constraint("?Y", "?Z", Constraint.INEQUALITY)
    })
    state = AbstractState(predicates, constraints)

    print(f"State: {state}")
    print(f"Variables: {state.get_variables()}")

    # Test substitution
    from stage3_code_generation.unification import Substitution
    subst = Substitution({"?X": "a", "?Y": "b", "?Z": "c"})
    new_state = state.apply_substitution(subst)
    print(f"After substitution {subst}:")
    print(f"  New state: {new_state}")

    # Test inconsistent substitution
    bad_subst = Substitution({"?X": "a", "?Y": "a"})  # Violates ?X != ?Y
    inconsistent_state = state.apply_substitution(bad_subst)
    print(f"After bad substitution {bad_subst}:")
    print(f"  Result: {inconsistent_state} (should be None)")

    print("✓ Abstract state tests passed\n")


if __name__ == "__main__":
    test_abstract_state()
