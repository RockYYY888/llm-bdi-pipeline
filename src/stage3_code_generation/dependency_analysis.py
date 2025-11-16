"""
Dependency Analysis for Integrated Quantification

This module provides tools to analyze dependencies and determine when
to use quantified representation vs concrete enumeration.

Key concept:
- Parallel dependencies: Multiple alternatives that are mutually exclusive
  → Should be quantified: ∃?X. P(?X)
- Sequential dependencies: Must be satisfied in order
  → Should be enumerated: P(a), P(b), P(c)
"""

import sys
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import PredicateAtom
from stage3_code_generation.abstract_state import AbstractState
from stage3_code_generation.unification import Unifier


@dataclass
class DependencyPattern:
    """
    Represents a pattern of dependencies for a precondition

    Attributes:
        precondition: The precondition being analyzed
        achieving_actions: List of actions that can achieve it
        is_parallel: True if dependencies are parallel (quantifiable)
        is_sequential: True if dependencies are sequential (must enumerate)
        common_structure: Common predicate structure across actions
    """
    precondition: PredicateAtom
    achieving_actions: List  # List of AbstractAction
    is_parallel: bool
    is_sequential: bool
    common_structure: Optional[PredicateAtom] = None
    varying_arguments: Set[int] = None  # Argument positions that vary


def analyze_dependency_pattern(precondition: PredicateAtom,
                               achieving_actions: List,
                               current_state: AbstractState) -> DependencyPattern:
    """
    Analyze the dependency pattern for a precondition

    Determines whether dependencies are parallel (can be quantified)
    or sequential (must be enumerated).

    Parallel pattern example:
        Precondition: clear(?X)
        Actions all produce: clear(?Y) where ?Y varies
        → Can quantify: ∃?Y. (clear(?Y) is achievable)

    Sequential pattern example:
        Precondition: tower(?X, ?Y, ?Z)
        Must build tower bottom-up in sequence
        → Cannot quantify, must enumerate steps

    Args:
        precondition: The precondition to analyze
        achieving_actions: Actions that can achieve this precondition
        current_state: Current state context

    Returns:
        DependencyPattern describing the pattern
    """
    if not achieving_actions:
        return DependencyPattern(
            precondition=precondition,
            achieving_actions=[],
            is_parallel=False,
            is_sequential=False
        )

    # Key insight for parallel dependencies:
    # If multiple actions produce the SAME effect pattern (e.g., clear(?X))
    # with only the variable binding changing, then these are parallel alternatives.
    #
    # We should quantify: instead of enumerating all possible bindings,
    # generate one quantified subgoal: ∃?X. (preconditions to achieve clear(?X))

    # Identify varying arguments in the effects
    varying_args = _identify_varying_arguments(achieving_actions, precondition)

    # If we have multiple actions producing same effect with varying args → parallel
    if len(achieving_actions) >= 2 and varying_args:
        is_parallel = True
        is_sequential = False
        common_structure = precondition
    else:
        # Single action or no varying args → enumerate
        is_parallel = False
        is_sequential = True
        common_structure = None

    return DependencyPattern(
        precondition=precondition,
        achieving_actions=achieving_actions,
        is_parallel=is_parallel,
        is_sequential=is_sequential,
        common_structure=common_structure,
        varying_arguments=varying_args
    )


def _generalize_predicate(pred: PredicateAtom) -> Tuple[str, int]:
    """
    Generalize a predicate to its structure (name + arity)

    Args:
        pred: Predicate to generalize

    Returns:
        Tuple of (name, arity)
    """
    return (pred.name, len(pred.args))


def _identify_varying_arguments(actions: List, target_precondition: PredicateAtom) -> Set[int]:
    """
    Identify which argument positions vary across actions' effects

    Args:
        actions: List of actions
        target_precondition: Target precondition to analyze

    Returns:
        Set of argument positions (indices) that vary
    """
    if not actions:
        return set()

    # Find the effect in each action that produces the target precondition
    effect_predicates = []

    for action in actions:
        for effect_branch in action.effects:
            for effect_atom in effect_branch:
                if effect_atom.is_add:
                    # Check if this effect matches target precondition structure
                    if (effect_atom.predicate.name == target_precondition.name and
                        len(effect_atom.predicate.args) == len(target_precondition.args)):
                        effect_predicates.append(effect_atom.predicate)
                        break

    if not effect_predicates:
        return set()

    # Identify varying positions
    varying_positions = set()
    num_args = len(effect_predicates[0].args)

    for pos in range(num_args):
        # Get all values at this position
        values_at_pos = set(pred.args[pos] for pred in effect_predicates if pos < len(pred.args))

        # If multiple different values (all variables), this position varies
        if len(values_at_pos) > 1 and all(v.startswith('?') for v in values_at_pos):
            varying_positions.add(pos)

    return varying_positions


def should_quantify_subgoal(pattern: DependencyPattern,
                            num_achieving_actions: int,
                            threshold: int = 2) -> bool:
    """
    Determine if a subgoal should be quantified or enumerated

    Args:
        pattern: Dependency pattern analysis
        num_achieving_actions: Number of actions that achieve the precondition
        threshold: Minimum number of actions to trigger quantification

    Returns:
        True if should generate quantified subgoal, False if should enumerate
    """
    # Quantify if:
    # 1. Pattern is parallel (not sequential)
    # 2. Multiple actions achieve it (>= threshold)
    # 3. Has varying arguments

    if not pattern.is_parallel:
        return False

    if num_achieving_actions < threshold:
        return False

    if not pattern.varying_arguments:
        return False

    return True


def test_dependency_analysis():
    """Test dependency analysis"""
    print("="*80)
    print("Test: Dependency Pattern Analysis")
    print("="*80)

    # Example: clear(?X) precondition
    # Multiple actions can achieve it: pick-up(?Y, ?X), put-down, etc.
    # This should be identified as parallel dependency

    from stage3_code_generation.lifted_planner import LiftedPlanner
    from utils.pddl_parser import PDDLParser

    # Load blocksworld domain
    from pathlib import Path
    domain_file = Path(__file__).parent.parent / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))

    planner = LiftedPlanner(domain)

    # Analyze a precondition
    precond = PredicateAtom("clear", ("?X",))

    # Find achieving actions
    achieving_actions = []
    for action in planner._abstract_actions:
        if planner._action_produces_predicate(action, precond):
            achieving_actions.append(action)

    print(f"\nPrecondition: {precond}")
    print(f"Achieving actions: {len(achieving_actions)}")
    for action in achieving_actions:
        print(f"  - {action.action.name}")

    # Analyze pattern
    from stage3_code_generation.abstract_state import AbstractState, ConstraintSet
    dummy_state = AbstractState(set(), ConstraintSet())

    pattern = analyze_dependency_pattern(precond, achieving_actions, dummy_state)

    print(f"\nPattern analysis:")
    print(f"  Is parallel: {pattern.is_parallel}")
    print(f"  Is sequential: {pattern.is_sequential}")
    print(f"  Varying arguments: {pattern.varying_arguments}")

    should_quantify = should_quantify_subgoal(pattern, len(achieving_actions))
    print(f"  Should quantify: {should_quantify}")

    if should_quantify:
        print(f"\n✓ This precondition should generate QUANTIFIED subgoal")
        print(f"  Instead of {len(achieving_actions)} concrete subgoals")
        print(f"  Generate 1 quantified subgoal: ∃?Z. (actions achieving clear(?Z))")
    else:
        print(f"\n✗ This precondition should ENUMERATE subgoals")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_dependency_analysis()
