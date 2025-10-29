"""
PDDL Problem Generator from LTLf Specifications

Converts LTLf specifications to PDDL problem files for classical planning.
"""

from typing import List, Dict, Any


class PDDLProblemGenerator:
    """
    Generates PDDL problem files from LTLf specifications
    """

    def __init__(self, domain_name: str = "blocksworld"):
        """
        Initialize problem generator

        Args:
            domain_name: Name of the PDDL domain
        """
        self.domain_name = domain_name

    def generate_problem(
        self,
        ltl_spec: Dict[str, Any],
        problem_name: str = "ltl_problem"
    ) -> str:
        """
        Generate PDDL problem from LTLf specification

        Args:
            ltl_spec: LTLf specification dict with:
                - objects: list of objects
                - initial_state: list of predicates
                - formulas_string: list of LTL goal formulas
            problem_name: Name for the problem

        Returns:
            PDDL problem as string
        """
        # Extract components
        objects = ltl_spec.get("objects", [])
        initial_state = ltl_spec.get("initial_state", [])
        formulas = ltl_spec.get("formulas_string", [])

        # Build PDDL problem
        problem = []
        problem.append(f"(define (problem {problem_name})")
        problem.append(f"  (:domain {self.domain_name})")

        # Objects (with type declarations for FOND domain compatibility)
        if objects:
            # Add type declarations for blocksworld domain
            typed_objects = ' '.join(f"{obj} - block" for obj in objects)
            problem.append(f"  (:objects {typed_objects})")

        # Initial state
        problem.append("  (:init")
        for pred_dict in initial_state:
            for pred_name, args in pred_dict.items():
                if pred_name == "ontable" and args:
                    # Special handling for ontable
                    for obj in args:
                        problem.append(f"    (ontable {obj})")
                elif pred_name == "clear" and args:
                    for obj in args:
                        problem.append(f"    (clear {obj})")
                elif pred_name == "handempty":
                    problem.append("    (handempty)")
                elif args:
                    problem.append(f"    ({pred_name} {' '.join(args)})")
                else:
                    problem.append(f"    ({pred_name})")
        problem.append("  )")

        # Goal from LTLf formulas
        goal_predicates = []
        for formula in formulas:
            # Extract goal from F(predicate) format
            goal_pred = self._extract_goal_from_ltl(formula)
            if goal_pred:
                goal_predicates.append(goal_pred)

        if goal_predicates:
            if len(goal_predicates) == 1:
                problem.append(f"  (:goal {goal_predicates[0]})")
            else:
                problem.append("  (:goal (and")
                for goal in goal_predicates:
                    problem.append(f"    {goal}")
                problem.append("  ))")
        else:
            # Default goal if none specified
            problem.append("  (:goal (handempty))")

        problem.append(")")

        return "\n".join(problem)

    def _extract_goal_from_ltl(self, ltl_formula: str) -> str:
        """
        Extract PDDL goal predicate from LTLf formula

        Args:
            ltl_formula: LTL formula string (e.g., "F(on(c, b))")

        Returns:
            PDDL predicate string (e.g., "(on c b)")
        """
        # Handle F(predicate) format (eventually)
        if ltl_formula.startswith("F(") and ltl_formula.endswith(")"):
            predicate = ltl_formula[2:-1]  # Remove "F(" and ")"
            # Recursively handle nested operators
            return self._convert_predicate_to_pddl(predicate)

        # Handle G(predicate) format (always)
        if ltl_formula.startswith("G(") and ltl_formula.endswith(")"):
            predicate = ltl_formula[2:-1]
            # For G (always), we include it as a goal to maintain
            return self._convert_predicate_to_pddl(predicate)

        # Handle X(predicate) format (next)
        if ltl_formula.startswith("X(") and ltl_formula.endswith(")"):
            predicate = ltl_formula[2:-1]  # Remove "X(" and ")"
            # For X (next), extract the inner predicate as goal
            return self._convert_predicate_to_pddl(predicate)

        # Handle U (until) format: (phi U psi)
        # For PDDL, we take the right side (psi) as the goal
        if " U " in ltl_formula:
            # Find the U operator (not inside parentheses)
            parts = ltl_formula.split(" U ", 1)
            if len(parts) == 2:
                # Extract the right side (psi) - what we want to achieve
                right_side = parts[1].strip()
                # Remove outer parentheses if present
                if right_side.startswith("(") and right_side.endswith(")"):
                    right_side = right_side[1:-1]
                return self._convert_predicate_to_pddl(right_side)

        # Fallback: try to convert as predicate
        return self._convert_predicate_to_pddl(ltl_formula)

    def _convert_predicate_to_pddl(self, predicate: str) -> str:
        """
        Convert predicate from LTL format to PDDL format

        Args:
            predicate: Predicate string (e.g., "on(c, b)" or "on(c,b)")

        Returns:
            PDDL predicate string (e.g., "(on c b)")
        """
        # Remove outer parentheses if wrapping entire predicate
        predicate = predicate.strip()
        if predicate.startswith("(") and predicate.endswith(")"):
            # Check if these are outer wrapping parens, not function call parens
            paren_count = 0
            for i, char in enumerate(predicate):
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                if paren_count == 0 and i < len(predicate) - 1:
                    # Parentheses closed before end - not outer wrapping
                    break
            if paren_count == 0 and i == len(predicate) - 1:
                # Outer wrapping parentheses - remove them
                predicate = predicate[1:-1].strip()

        # Convert from on(c,b) to (on c b)
        if "(" in predicate and ")" in predicate:
            pred_name = predicate[:predicate.index("(")]
            args_str = predicate[predicate.index("(") + 1:predicate.rindex(")")]
            args = [arg.strip() for arg in args_str.split(",")]
            return f"({pred_name} {' '.join(args)})"
        else:
            # Simple predicate with no arguments
            return f"({predicate})"


# Example usage
if __name__ == "__main__":
    generator = PDDLProblemGenerator("blocksworld")

    # Example LTLf spec
    ltl_spec = {
        "objects": ["b", "c"],
        "initial_state": [
            {"ontable": ["b"]},
            {"ontable": ["c"]},
            {"clear": ["b"]},
            {"clear": ["c"]},
            {"handempty": []}
        ],
        "formulas_string": ["F(on(c, b))"]
    }

    problem = generator.generate_problem(ltl_spec, "stack_c_on_b")
    print(problem)
