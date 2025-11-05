"""
LTLf Formula Representation Module

This module defines data structures for representing LTLf (Linear Temporal Logic on Finite Traces) formulas.

**LTLf Syntax Reference**: http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax

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
        # Propositional constants: true, false
        if isinstance(self.predicate, str) and self.predicate in ["true", "false"]:
            return self.predicate

        # Atomic proposition: on(a, b), clear(a), or handempty
        if self._is_atomic():
            if isinstance(self.predicate, dict):
                pred_name = list(self.predicate.keys())[0]
                args = self.predicate[pred_name]
                if len(args) == 0:
                    return pred_name
                return f"{pred_name}({', '.join(args)})"

        # Unary temporal operators: X, WX, F, G
        if self.operator in [TemporalOperator.NEXT, TemporalOperator.WEAK_NEXT,
                             TemporalOperator.FINALLY, TemporalOperator.GLOBALLY]:
            inner = self.sub_formulas[0].to_string()
            return f"{self.operator.value}({inner})"

        # Binary temporal operators: U (Until), R (Release)
        if self.operator in [TemporalOperator.UNTIL, TemporalOperator.RELEASE]:
            left = self.sub_formulas[0].to_string()
            right = self.sub_formulas[1].to_string()
            op_symbol = self.operator.value

            # ALWAYS add parens to binary sub-formulas for clarity
            if self.sub_formulas[0]._is_binary():
                left = f"({left})"
            if self.sub_formulas[1]._is_binary():
                right = f"({right})"

            # Wrap entire Until/Release in parens (they're binary operators)
            return f"({left} {op_symbol} {right})"

        # Boolean operators
        if self.logical_op:
            if self.logical_op == LogicalOperator.NOT:
                # NOT is unary
                inner = self.sub_formulas[0].to_string()
                return f"!({inner})"

            elif self.logical_op == LogicalOperator.AND:
                parts = []
                for f in self.sub_formulas:
                    s = f.to_string()
                    # ALWAYS add parens to binary sub-formulas (except AND)
                    if f._is_binary() and not (f.logical_op == LogicalOperator.AND):
                        s = f"({s})"
                    parts.append(s)
                # Wrap entire AND in parens
                return f"({' & '.join(parts)})"

            elif self.logical_op == LogicalOperator.OR:
                parts = []
                for f in self.sub_formulas:
                    s = f.to_string()
                    # ALWAYS add parens to binary sub-formulas (except OR)
                    if f._is_binary() and not (f.logical_op == LogicalOperator.OR):
                        s = f"({s})"
                    parts.append(s)
                # Wrap entire OR in parens
                return f"({' | '.join(parts)})"

            elif self.logical_op == LogicalOperator.IMPLIES:
                left = self.sub_formulas[0].to_string()
                right = self.sub_formulas[1].to_string()

                # ALWAYS add parens to binary sub-formulas
                if self.sub_formulas[0]._is_binary():
                    left = f"({left})"
                if self.sub_formulas[1]._is_binary():
                    right = f"({right})"

                return f"{left} -> {right}"

            elif self.logical_op == LogicalOperator.EQUIVALENCE:
                left = self.sub_formulas[0].to_string()
                right = self.sub_formulas[1].to_string()

                # ALWAYS add parens to binary sub-formulas
                if self.sub_formulas[0]._is_binary():
                    left = f"({left})"
                if self.sub_formulas[1]._is_binary():
                    right = f"({right})"

                return f"{left} <-> {right}"

        return "true"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "operator": self.operator.value if self.operator else None,
            "predicate": self.predicate,
            "logical_op": self.logical_op.value if self.logical_op else None,
            "sub_formulas": [f.to_dict() for f in self.sub_formulas]
        }


class LTLSpecification:
    """
    LTL Specification for BDI goals

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

    def add_formula(self, formula: LTLFormula):
        """Add an LTL formula to the specification"""
        self.formulas.append(formula)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "formulas": [f.to_dict() for f in self.formulas],
            "formulas_string": [f.to_string() for f in self.formulas],
            "objects": self.objects,
            "initial_state": self.initial_state
        }

        # Include grounding map if available
        if self.grounding_map:
            result["grounding_map"] = self.grounding_map.to_dict()

        return result
