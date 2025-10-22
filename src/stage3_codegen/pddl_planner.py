"""
Stage 3: Code Generation Framework - PDDL Planning
Uses pyperplan to generate plans from PDDL domain and problem files.
"""

import os
import tempfile
from typing import List, Tuple, Optional
import pyperplan
from pyperplan import planner


class PDDLPlanner:
    """
    Wrapper for PDDL planning using pyperplan.
    """

    def __init__(self):
        """Initialize planner."""
        pass

    def solve(self, domain_file: str, problem_file: str) -> Optional[List[Tuple[str, List[str]]]]:
        """
        Solve a PDDL planning problem.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file

        Returns:
            List of (action_name, [parameters]) tuples, or None if no plan found
        """
        try:
            # Use pyperplan's search_plan function
            # Signature: (domain_file, problem_file, search, heuristic_class)
            plan = planner.search_plan(
                domain_file,
                problem_file,
                planner.SEARCHES['gbf'],     # Greedy best-first search
                planner.HEURISTICS['hff']    # FF heuristic function
            )

            if plan is None:
                return None

            # Convert plan to readable format
            action_sequence = []
            for action in plan:
                # Extract action name and parameters from action.name
                # pyperplan action names are like "(pickup a)" or "(stack a b)"
                # Remove parentheses and split
                name_cleaned = action.name.strip('()')
                parts = name_cleaned.split()
                action_name = parts[0] if parts else name_cleaned
                params = parts[1:] if len(parts) > 1 else []
                action_sequence.append((action_name, params))

            return action_sequence
        except Exception as e:
            print(f"Planning failed: {e}")
            return None

    def solve_from_strings(self, domain_str: str, problem_str: str) -> Optional[List[Tuple[str, List[str]]]]:
        """
        Solve a PDDL planning problem from string representations.

        Args:
            domain_str: PDDL domain as string
            problem_str: PDDL problem as string

        Returns:
            List of (action_name, [parameters]) tuples, or None if no plan found
        """
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as domain_f:
            domain_f.write(domain_str)
            domain_file = domain_f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as problem_f:
            problem_f.write(problem_str)
            problem_file = problem_f.name

        try:
            plan = self.solve(domain_file, problem_file)
        finally:
            # Clean up temporary files
            os.unlink(domain_file)
            os.unlink(problem_file)

        return plan


# Example usage
if __name__ == "__main__":
    # Test with example Blocksworld problem
    domain_str = """(define (domain blocksworld)
  (:requirements :strips)
  (:predicates
    (on ?x ?y)
    (ontable ?x)
    (clear ?x)
    (holding ?x)
    (handempty)
  )
  (:action pickup
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (holding ?x) (not (ontable ?x)) (not (clear ?x)) (not (handempty)))
  )
  (:action putdown
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (ontable ?x) (clear ?x) (handempty) (not (holding ?x)))
  )
  (:action stack
    :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y) (clear ?x) (handempty) (not (holding ?x)) (not (clear ?y)))
  )
  (:action unstack
    :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (clear ?x)) (not (handempty)))
  )
)"""

    problem_str = """(define (problem stack-a-on-b)
  (:domain blocksworld)
  (:objects a b)
  (:init (ontable a) (ontable b) (clear a) (clear b) (handempty))
  (:goal (and (on a b) (clear a)))
)"""

    planner_instance = PDDLPlanner()
    plan = planner_instance.solve_from_strings(domain_str, problem_str)

    if plan:
        print("Plan found:")
        for i, (action, params) in enumerate(plan, 1):
            print(f"  {i}. {action}({', '.join(params)})")
    else:
        print("No plan found!")
