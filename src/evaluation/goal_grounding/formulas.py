"""
LTLf Formula Representation Module

This module defines data structures for representing LTLf (Linear Temporal Logic on Finite Traces) formulas.

Supported syntax (in order):

1. Propositional Symbols:
   - true, false (constants)
   - [a-z][a-z0-9_]* (atomic propositions)

2. Boolean Operators:
   - & or && (And)
   - | or || (Or)
   - ! or ~ (Not)
   - -> or => (Implication)
   - <-> or <=> (Equivalence)

3. Future Temporal Operators:
   - X (Next)
   - WX (WeakNext)
   - U (Until)
   - R (Release)
   - F (Eventually)
   - G (Always)

Example formulas:
- F(on(a, b))           : Eventually a is on b
- G(clear(a))           : Always a is clear
- F(on(a, b) & clear(a)): Eventually a is on b AND a is clear
- (holding(a) U clear(b)): Hold a until b becomes clear
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .grounding_map import GroundingMap


class TemporalOperator(Enum):
    """Future Temporal Operators (ordered as per LTLf syntax specification)"""
    NEXT = "X"         # Next state (strong)
    WEAK_NEXT = "WX"   # Weak next state
    UNTIL = "U"        # Until (φ U ψ)
    RELEASE = "R"      # Release (φ R ψ)
    FINALLY = "F"      # Eventually (◇)
    GLOBALLY = "G"     # Always (□)


class LogicalOperator(Enum):
    """Boolean Operators (ordered as per LTLf syntax specification)"""
    AND = "and"         # & or &&
    OR = "or"           # | or ||
    NOT = "not"         # ! or ~
    IMPLIES = "implies" # -> or =>
    EQUIVALENCE = "equivalence"  # <-> or <=>


@dataclass
class LTLFormula:
    """Represents an LTLf formula with full syntax support"""
    operator: Optional[TemporalOperator]
    predicate: Optional[Union[Dict[str, List[str]], str]]  # Dict for predicates, str for constants
    sub_formulas: List['LTLFormula']
    logical_op: Optional[LogicalOperator]

    def _is_atomic(self) -> bool:
        """Check if this formula is atomic (no operators, just a predicate/constant)"""
        return (self.predicate is not None and
                self.operator is None and
                self.logical_op is None)

    def _is_unary(self) -> bool:
        """Check if this formula uses a unary operator"""
        if self.logical_op == LogicalOperator.NOT:
            return True
        if self.operator in [TemporalOperator.NEXT, TemporalOperator.WEAK_NEXT,
                             TemporalOperator.FINALLY, TemporalOperator.GLOBALLY]:
            return True
        return False

    def _is_binary(self) -> bool:
        """Check if this formula uses a binary operator"""
        if self.operator in [TemporalOperator.UNTIL, TemporalOperator.RELEASE]:
            return True
        if self.logical_op in [LogicalOperator.AND, LogicalOperator.OR,
                               LogicalOperator.IMPLIES, LogicalOperator.EQUIVALENCE]:
            return True
        return False

    def _needs_parens(self) -> bool:
        """
        Determine if this formula needs parentheses when used as a sub-formula.

        Rules:
        - Atomic formulas: NO parentheses
        - Unary operators (!, X, WX, F, G): NO parentheses (they have operator(...) format)
        - Binary operators (&, |, ->, <->, U, R): YES parentheses when nested
        """
        return self._is_binary()

    def to_string(self) -> str:
        """
        Convert LTL formula to string representation with structural parenthesization.

        Strategy: Add parentheses around binary operators when they appear as sub-formulas.
        This ensures unambiguous parsing while keeping atomic/unary formulas clean.
        """
        rendered: Dict[int, str] = {}
        stack: List[tuple['LTLFormula', bool]] = [(self, False)]

        while stack:
            formula, expanded = stack.pop()
            formula_id = id(formula)
            if not expanded:
                stack.append((formula, True))
                for child in reversed(formula.sub_formulas):
                    stack.append((child, False))
                continue

            # Propositional constants: true, false
            if isinstance(formula.predicate, str) and formula.predicate in ["true", "false"]:
                rendered[formula_id] = formula.predicate
                continue

            # Atomic proposition: on(a, b), clear(a), or handempty
            if formula._is_atomic():
                if isinstance(formula.predicate, dict):
                    pred_name = list(formula.predicate.keys())[0]
                    args = formula.predicate[pred_name]
                    if len(args) == 0:
                        rendered[formula_id] = pred_name
                    else:
                        rendered[formula_id] = f"{pred_name}({', '.join(args)})"
                    continue
                rendered[formula_id] = "true"
                continue

            child_strings = [rendered[id(child)] for child in formula.sub_formulas]

            # Unary temporal operators: X, WX, F, G
            if formula.operator in [
                TemporalOperator.NEXT,
                TemporalOperator.WEAK_NEXT,
                TemporalOperator.FINALLY,
                TemporalOperator.GLOBALLY,
            ]:
                rendered[formula_id] = f"{formula.operator.value}({child_strings[0]})"
                continue

            # Binary temporal operators: U (Until), R (Release)
            if formula.operator in [TemporalOperator.UNTIL, TemporalOperator.RELEASE]:
                left = child_strings[0]
                right = child_strings[1]
                if formula.sub_formulas[0]._is_binary():
                    left = f"({left})"
                if formula.sub_formulas[1]._is_binary():
                    right = f"({right})"
                rendered[formula_id] = f"({left} {formula.operator.value} {right})"
                continue

            # Boolean operators
            if formula.logical_op == LogicalOperator.NOT:
                rendered[formula_id] = f"!({child_strings[0]})"
                continue

            if formula.logical_op == LogicalOperator.AND:
                parts = []
                for child, child_string in zip(formula.sub_formulas, child_strings):
                    part = child_string
                    if child._is_binary() and child.logical_op != LogicalOperator.AND:
                        part = f"({part})"
                    parts.append(part)
                rendered[formula_id] = f"({' & '.join(parts)})"
                continue

            if formula.logical_op == LogicalOperator.OR:
                parts = []
                for child, child_string in zip(formula.sub_formulas, child_strings):
                    part = child_string
                    if child._is_binary() and child.logical_op != LogicalOperator.OR:
                        part = f"({part})"
                    parts.append(part)
                rendered[formula_id] = f"({' | '.join(parts)})"
                continue

            if formula.logical_op == LogicalOperator.IMPLIES:
                left = child_strings[0]
                right = child_strings[1]
                if formula.sub_formulas[0]._is_binary():
                    left = f"({left})"
                if formula.sub_formulas[1]._is_binary():
                    right = f"({right})"
                rendered[formula_id] = f"{left} -> {right}"
                continue

            if formula.logical_op == LogicalOperator.EQUIVALENCE:
                left = child_strings[0]
                right = child_strings[1]
                if formula.sub_formulas[0]._is_binary():
                    left = f"({left})"
                if formula.sub_formulas[1]._is_binary():
                    right = f"({right})"
                rendered[formula_id] = f"{left} <-> {right}"
                continue

            rendered[formula_id] = "true"

        return rendered[id(self)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        serialised: Dict[int, Dict[str, Any]] = {}
        stack: List[tuple['LTLFormula', bool]] = [(self, False)]

        while stack:
            formula, expanded = stack.pop()
            formula_id = id(formula)
            if not expanded:
                stack.append((formula, True))
                for child in reversed(formula.sub_formulas):
                    stack.append((child, False))
                continue

            serialised[formula_id] = {
                "operator": formula.operator.value if formula.operator else None,
                "predicate": formula.predicate,
                "logical_op": formula.logical_op.value if formula.logical_op else None,
                "sub_formulas": [serialised[id(child)] for child in formula.sub_formulas],
            }

        return serialised[id(self)]


class LTLSpecification:
    """
    LTL specification for grounded query goals.

    This converts natural language to LTL formulas that express:
    - Safety properties (what should never happen)
    - Liveness properties (what should eventually happen)
    - Fairness properties (what should happen infinitely often)

    Uses propositional symbols (e.g., on_a_b) with a grounding map
    to maintain the connection to domain predicates and objects.
    """

    def __init__(self):
        self.formulas: List[LTLFormula] = []
        self.objects: List[str] = []
        self.initial_state: List[Dict[str, List[str]]] = []
        self.grounding_map: Optional['GroundingMap'] = None
        self.source_instruction: str = ""
        self.negation_hints: Dict[str, Any] = {}
        self.query_object_inventory: List[Dict[str, Any]] = []
        self.query_task_literal_signatures: List[str] = []

    def add_formula(self, formula: LTLFormula):
        """Add an LTL formula to the specification"""
        self.formulas.append(formula)

    def combined_formula(self) -> LTLFormula:
        """
        Return the specification as a single formula.

        Goal grounding may emit multiple top-level formulas when the instruction
        expresses multiple independent obligations. The pipeline treats the
        specification semantics as the conjunction of those top-level formulas.
        """
        if not self.formulas:
            raise ValueError("No LTLf formulas in specification")
        if len(self.formulas) == 1:
            return self.formulas[0]
        current_level = list(self.formulas)
        while len(current_level) > 1:
            next_level: List[LTLFormula] = []
            index = 0
            while index < len(current_level):
                left = current_level[index]
                if index + 1 >= len(current_level):
                    next_level.append(left)
                    break
                right = current_level[index + 1]
                next_level.append(
                    LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=[left, right],
                        logical_op=LogicalOperator.AND,
                    ),
                )
                index += 2
            current_level = next_level
        return current_level[0]

    def combined_formula_string(self) -> str:
        """Render the full specification semantics as one LTLf string."""
        return self.combined_formula().to_string()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "formulas": [f.to_dict() for f in self.formulas],
            "formulas_string": [f.to_string() for f in self.formulas],
            "objects": self.objects,
            "semantic_objects": self.objects,
            "initial_state": self.initial_state,
        }

        # Include grounding map if available
        if self.grounding_map:
            result["grounding_map"] = self.grounding_map.to_dict()
        if self.source_instruction:
            result["source_instruction"] = self.source_instruction
        if self.negation_hints:
            result["negation_hints"] = dict(self.negation_hints)
        if self.query_object_inventory:
            result["query_object_inventory"] = list(self.query_object_inventory)
        if self.query_task_literal_signatures:
            result["query_task_literal_signatures"] = list(self.query_task_literal_signatures)

        return result
