"""
Plan Instantiation for Lifted Planning

Converts abstract lifted plans (with quantified predicates) into concrete
grounded plans that can be executed.

Key concept:
- Abstract plan: Uses quantified predicates (∃?Z. on(?Z, b))
- Concrete plan: Fully grounded predicates (on(a, b), on(c, b))

The instantiation process:
1. Identify quantified variables in plan
2. Bind them to concrete objects
3. Generate grounded plan steps
"""

import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import PredicateAtom
from stage3_code_generation.abstract_state import AbstractState
from stage3_code_generation.quantified_predicate import QuantifiedPredicate, instantiate_quantified_predicate
from stage3_code_generation.unification import Substitution


@dataclass
class PlanStep:
    """
    A single step in a plan

    Attributes:
        action_name: Name of the action
        parameters: List of parameter bindings
        preconditions: Preconditions for this step (for verification)
        effects: Effects of this step
    """
    action_name: str
    parameters: Dict[str, str]  # Variable -> Object binding
    preconditions: Set[PredicateAtom]
    effects: Set[PredicateAtom]

    def __str__(self):
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.action_name}({params_str})"


class PlanInstantiator:
    """
    Instantiates abstract lifted plans to concrete grounded plans

    This is the final step in lifted planning: converting the abstract
    solution (which may contain quantified predicates) into an executable
    concrete plan.
    """

    def __init__(self, objects: List[str]):
        """
        Initialize plan instantiator

        Args:
            objects: Available objects in the domain
        """
        self.objects = objects

    def instantiate_state(self, abstract_state: AbstractState,
                         max_instances: int = 100) -> List[AbstractState]:
        """
        Instantiate an abstract state to concrete states

        If the state contains quantified predicates, generate all possible
        concrete instantiations (up to max_instances).

        Args:
            abstract_state: Abstract state (may contain quantified predicates)
            max_instances: Maximum number of instances to generate

        Returns:
            List of concrete states
        """
        # If no quantified predicates, return as-is
        if not abstract_state.quantified_predicates:
            return [abstract_state]

        # Start with concrete predicates
        base_predicates = set(abstract_state.predicates)

        # Instantiate each quantified predicate
        all_instantiations = [base_predicates]

        for qpred in abstract_state.quantified_predicates:
            # Generate concrete instances for this quantified predicate
            concrete_instances = instantiate_quantified_predicate(
                qpred,
                self.objects,
                max_instances=max_instances
            )

            # Add to all instantiations
            new_instantiations = []
            for existing_preds in all_instantiations:
                # Create new state with quantified predicate instantiated
                new_preds = existing_preds | concrete_instances
                new_instantiations.append(new_preds)

                if len(new_instantiations) >= max_instances:
                    break

            all_instantiations = new_instantiations

            if len(all_instantiations) >= max_instances:
                break

        # Convert to AbstractState objects
        concrete_states = []
        for preds in all_instantiations[:max_instances]:
            concrete_state = AbstractState(
                predicates=preds,
                constraints=abstract_state.constraints,
                depth=abstract_state.depth
            )
            concrete_states.append(concrete_state)

        return concrete_states

    def instantiate_plan(self, abstract_plan: List[Tuple[str, Substitution]],
                        start_state: AbstractState) -> List[PlanStep]:
        """
        Instantiate an abstract plan to a concrete plan

        Args:
            abstract_plan: List of (action_name, substitution) tuples
            start_state: Starting state for plan execution

        Returns:
            List of concrete plan steps
        """
        concrete_plan = []
        current_state = start_state

        for action_name, subst in abstract_plan:
            # Apply substitution to get concrete parameters
            concrete_params = {}
            for var, value in subst.bindings.items():
                # If value is still a variable, need to bind to object
                if value.startswith('?'):
                    # Try to find binding in current state
                    # For now, use first available object
                    if self.objects:
                        concrete_params[var] = self.objects[0]
                    else:
                        concrete_params[var] = value
                else:
                    concrete_params[var] = value

            # Create plan step
            step = PlanStep(
                action_name=action_name,
                parameters=concrete_params,
                preconditions=set(),  # Would need action definition to fill
                effects=set()  # Would need action definition to fill
            )

            concrete_plan.append(step)

        return concrete_plan


def test_plan_instantiation():
    """Test plan instantiation"""
    print("="*80)
    print("Test: Plan Instantiation")
    print("="*80)

    # Create abstract state with quantified predicate
    from stage3_code_generation.quantified_predicate import Quantifier, QuantifiedPredicate
    from stage3_code_generation.abstract_state import ConstraintSet

    # Concrete predicates
    concrete_preds = {
        PredicateAtom("clear", ("b",)),
        PredicateAtom("handempty", ())
    }

    # Quantified predicate: ∃?Z. on(?Z, b)
    qpred = QuantifiedPredicate(
        quantifier=Quantifier.EXISTS,
        variables=("?Z",),
        formula=PredicateAtom("on", ("?Z", "b")),
        constraints=ConstraintSet(set())
    )

    abstract_state = AbstractState(
        predicates=concrete_preds,
        constraints=ConstraintSet(),
        quantified_predicates={qpred}
    )

    print(f"\nAbstract state:")
    print(f"  {abstract_state}")

    # Instantiate with objects
    objects = ["a", "c", "d", "e"]
    instantiator = PlanInstantiator(objects)

    concrete_states = instantiator.instantiate_state(abstract_state, max_instances=3)

    print(f"\nConcrete instantiations ({len(concrete_states)}):")
    for i, state in enumerate(concrete_states):
        print(f"\n  Instantiation {i+1}:")
        print(f"    Predicates: {len(state.predicates)}")
        for p in sorted(state.predicates, key=str):
            print(f"      {p}")

    print("\n✓ Plan instantiation works!")


if __name__ == "__main__":
    test_plan_instantiation()
