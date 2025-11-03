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
from typing import List, Dict, Any, Optional, Union
from enum import Enum


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

    def to_string(self) -> str:
        """Convert LTL formula to string representation"""
        # Propositional constants: true, false
        if isinstance(self.predicate, str) and self.predicate in ["true", "false"]:
            return self.predicate

        if self.predicate and not self.operator:
            # Atomic proposition: on(a, b), clear(a), or handempty
            if isinstance(self.predicate, dict):
                pred_name = list(self.predicate.keys())[0]
                args = self.predicate[pred_name]
                if len(args) == 0:
                    return pred_name
                return f"{pred_name}({', '.join(args)})"

        # Binary temporal operators: U (Until), R (Release)
        if self.operator in [TemporalOperator.UNTIL, TemporalOperator.RELEASE] and len(self.sub_formulas) == 2:
            left = self.sub_formulas[0].to_string()
            right = self.sub_formulas[1].to_string()
            op_symbol = self.operator.value
            return f"({left} {op_symbol} {right})"

        # Unary temporal operators: X, WX, F, G
        if self.operator and len(self.sub_formulas) == 1:
            inner = self.sub_formulas[0].to_string()
            return f"{self.operator.value}({inner})"

        # Boolean operators
        if self.logical_op and len(self.sub_formulas) >= 1:
            parts = [f.to_string() for f in self.sub_formulas]

            if self.logical_op == LogicalOperator.AND:
                return f"{' & '.join(parts)}"
            elif self.logical_op == LogicalOperator.OR:
                return f"{' | '.join(parts)}"
            elif self.logical_op == LogicalOperator.NOT:
                return f"!({parts[0]})"
            elif self.logical_op == LogicalOperator.IMPLIES:
                return f"{parts[0]} -> {parts[1]}"
            elif self.logical_op == LogicalOperator.EQUIVALENCE:
                return f"{parts[0]} <-> {parts[1]}"

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
    """

    def __init__(self):
        self.formulas: List[LTLFormula] = []
        self.objects: List[str] = []
        self.initial_state: List[Dict[str, List[str]]] = []

    def add_formula(self, formula: LTLFormula):
        """Add an LTL formula to the specification"""
        self.formulas.append(formula)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "formulas": [f.to_dict() for f in self.formulas],
            "formulas_string": [f.to_string() for f in self.formulas],
            "objects": self.objects,
            "initial_state": self.initial_state
        }
