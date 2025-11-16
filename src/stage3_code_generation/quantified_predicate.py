"""
Quantified Predicates for Lifted Planning

This module implements First-Order Logic quantifiers (∃, ∀) for true lifted planning.

Key concepts:
- Existential quantification (∃): "There exists at least one object..."
- Universal quantification (∀): "For all objects..."
- Quantified predicates allow representing sets of concrete predicates abstractly
- Reduces state space from O(n^k) to O(1) where n = number of objects

Example:
    Concrete predicates: {on(a, b), on(c, b), on(d, b)}
    Quantified form: ∃?Z. on(?Z, b)

    This represents "there exists some block on b" without enumerating all instances.
"""

import sys
from pathlib import Path
from enum import Enum
from typing import List, Set, FrozenSet, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import PredicateAtom
from stage3_code_generation.abstract_state import Constraint, ConstraintSet


class Quantifier(Enum):
    """
    First-Order Logic Quantifiers

    EXISTS (∃): Existential quantification - "there exists at least one"
    FORALL (∀): Universal quantification - "for all"
    """
    EXISTS = "∃"
    FORALL = "∀"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class QuantifiedPredicate:
    """
    A quantified predicate representing a pattern over multiple concrete predicates

    Examples:
        ∃?X. on(?X, b)        - "There exists some block on b"
        ∀?X. clear(?X)        - "All blocks are clear"
        ∃?X,?Y. on(?X, ?Y)    - "There exists some stacking relationship"

    Attributes:
        quantifier: The quantifier type (EXISTS or FORALL)
        variables: List of quantified variables (e.g., ["?X", "?Y"])
        formula: The predicate pattern (e.g., on(?X, b))
        constraints: Additional constraints on quantified variables
        count_bound: Optional bound on quantity (e.g., "at least 2", "exactly 1")
    """
    quantifier: Quantifier
    variables: Tuple[str, ...]  # Quantified variables
    formula: PredicateAtom       # The predicate template
    constraints: ConstraintSet    # Constraints on quantified variables
    count_bound: Optional[Tuple[str, int]] = None  # ("at_least", n), ("exactly", n), etc.

    def __str__(self):
        vars_str = ",".join(self.variables)
        result = f"{self.quantifier}{vars_str}. {self.formula}"

        if self.constraints and self.constraints.constraints:
            constraint_strs = [str(c) for c in self.constraints.constraints]
            result += f" where {{{', '.join(constraint_strs)}}}"

        if self.count_bound:
            bound_type, n = self.count_bound
            result += f" ({bound_type} {n})"

        return result

    def __hash__(self):
        return hash((
            self.quantifier,
            self.variables,
            self.formula,
            self.constraints,
            self.count_bound
        ))

    def get_free_variables(self) -> Set[str]:
        """
        Get free variables (not quantified) in the formula

        Returns:
            Set of variable names that are not quantified
        """
        formula_vars = set(arg for arg in self.formula.args if arg.startswith('?'))
        quantified_vars = set(self.variables)
        return formula_vars - quantified_vars

    def matches_concrete_predicate(self, concrete_pred: PredicateAtom,
                                    bindings: Optional[dict] = None) -> bool:
        """
        Check if a concrete predicate matches this quantified pattern

        Args:
            concrete_pred: Concrete predicate to check
            bindings: Optional existing variable bindings

        Returns:
            True if the concrete predicate matches the pattern
        """
        # Must have same predicate name
        if self.formula.name != concrete_pred.name:
            return False

        # Must have same arity
        if len(self.formula.args) != len(concrete_pred.args):
            return False

        # Check if arguments match under quantified variable substitution
        bindings = bindings or {}
        temp_bindings = dict(bindings)

        for pattern_arg, concrete_arg in zip(self.formula.args, concrete_pred.args):
            if pattern_arg.startswith('?'):
                # Variable in pattern
                if pattern_arg in self.variables:
                    # Quantified variable - can bind to concrete_arg
                    if pattern_arg in temp_bindings:
                        # Already bound - must match
                        if temp_bindings[pattern_arg] != concrete_arg:
                            return False
                    else:
                        # Bind it
                        temp_bindings[pattern_arg] = concrete_arg
                else:
                    # Free variable - must already be bound and match
                    if pattern_arg not in bindings:
                        return False
                    if bindings[pattern_arg] != concrete_arg:
                        return False
            else:
                # Constant in pattern - must match exactly
                if pattern_arg != concrete_arg:
                    return False

        # Check constraints
        if self.constraints:
            for constraint in self.constraints.constraints:
                var1_val = temp_bindings.get(constraint.var1, constraint.var1)
                var2_val = temp_bindings.get(constraint.var2, constraint.var2)

                if constraint.constraint_type == Constraint.INEQUALITY:
                    if var1_val == var2_val:
                        return False
                elif constraint.constraint_type == Constraint.EQUALITY:
                    if var1_val != var2_val:
                        return False

        return True


def detect_quantifiable_pattern(predicates: Set[PredicateAtom],
                                 min_instances: int = 2) -> List[QuantifiedPredicate]:
    """
    Detect patterns in concrete predicates that can be quantified

    This is a key algorithm for lifted planning: automatically identifying
    when multiple concrete predicates can be abstracted into a quantified form.

    Algorithm:
    1. Group predicates by name and argument structure
    2. For each group with >= min_instances predicates:
       - Identify which arguments vary (candidates for quantification)
       - Identify which arguments are constant (part of the pattern)
       - Generate quantified predicate if beneficial

    Args:
        predicates: Set of concrete predicates
        min_instances: Minimum number of instances to consider quantification

    Returns:
        List of detected quantified predicates
    """
    quantified = []

    # Group predicates by name
    by_name = {}
    for pred in predicates:
        if pred.name not in by_name:
            by_name[pred.name] = []
        by_name[pred.name].append(pred)

    for pred_name, pred_group in by_name.items():
        if len(pred_group) < min_instances:
            continue

        # Check if all have same arity
        if len(set(len(p.args) for p in pred_group)) > 1:
            continue

        arity = len(pred_group[0].args)
        if arity == 0:
            continue

        # For each argument position, check if it's constant or varying
        arg_patterns = []
        for arg_pos in range(arity):
            values_at_pos = set(p.args[arg_pos] for p in pred_group)

            if len(values_at_pos) == 1:
                # Constant argument
                arg_patterns.append(('constant', list(values_at_pos)[0]))
            elif all(v.startswith('?') for v in values_at_pos):
                # All variables - can quantify
                arg_patterns.append(('quantified', f'?Q{arg_pos}'))
            else:
                # Mix or all constants but different - can quantify over objects
                arg_patterns.append(('quantified', f'?Q{arg_pos}'))

        # Check if there's at least one quantified argument
        if not any(pattern == 'quantified' for pattern, _ in arg_patterns):
            continue

        # Build quantified predicate
        quantified_vars = []
        formula_args = []

        for arg_pos, (pattern_type, value) in enumerate(arg_patterns):
            if pattern_type == 'quantified':
                var_name = f'?Q{arg_pos}'
                quantified_vars.append(var_name)
                formula_args.append(var_name)
            else:
                formula_args.append(value)

        # Create formula
        formula = PredicateAtom(pred_name, tuple(formula_args), negated=False)

        # Create quantified predicate WITHOUT specific count bound
        # This is critical for state space reduction:
        # - With count_bound: each count creates unique state (at_least 2, at_least 3, etc.)
        # - Without count_bound: all counts merge to single quantified state
        #
        # Example:
        #   {on(a,b), on(c,b)} → ∃?Z. on(?Z, b)
        #   {on(a,b), on(c,b), on(d,b)} → ∃?Z. on(?Z, b) (SAME quantified predicate!)
        #
        # This dramatically reduces state space.
        qp = QuantifiedPredicate(
            quantifier=Quantifier.EXISTS,
            variables=tuple(quantified_vars),
            formula=formula,
            constraints=ConstraintSet(set()),
            count_bound=None  # No specific bound - just "exists at least one"
        )

        quantified.append(qp)

    return quantified


def instantiate_quantified_predicate(qpred: QuantifiedPredicate,
                                      objects: List[str],
                                      max_instances: int = 100) -> Set[PredicateAtom]:
    """
    Instantiate a quantified predicate to concrete predicates

    This is used for plan instantiation: converting abstract quantified plan
    to concrete grounded plan.

    Args:
        qpred: Quantified predicate to instantiate
        objects: Available objects for binding
        max_instances: Maximum number of instances to generate

    Returns:
        Set of concrete predicates
    """
    from itertools import product

    instances = set()

    # Generate all possible bindings for quantified variables
    num_vars = len(qpred.variables)

    for binding_tuple in product(objects, repeat=num_vars):
        if len(instances) >= max_instances:
            break

        # Create binding dictionary
        bindings = dict(zip(qpred.variables, binding_tuple))

        # Check constraints
        valid = True
        for constraint in qpred.constraints.constraints:
            var1_val = bindings.get(constraint.var1, constraint.var1)
            var2_val = bindings.get(constraint.var2, constraint.var2)

            if constraint.constraint_type == Constraint.INEQUALITY:
                if var1_val == var2_val:
                    valid = False
                    break
            elif constraint.constraint_type == Constraint.EQUALITY:
                if var1_val != var2_val:
                    valid = False
                    break

        if not valid:
            continue

        # Instantiate formula
        concrete_args = []
        for arg in qpred.formula.args:
            if arg in bindings:
                concrete_args.append(bindings[arg])
            else:
                concrete_args.append(arg)

        concrete_pred = PredicateAtom(
            qpred.formula.name,
            tuple(concrete_args),
            negated=qpred.formula.negated
        )

        instances.add(concrete_pred)

    return instances
