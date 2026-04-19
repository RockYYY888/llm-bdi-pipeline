"""
Goal-grounding exports for natural-language query interpretation.
"""

from .formulas import LTLFormula, LTLSpecification, TemporalOperator, LogicalOperator
from .grounder import NLToLTLfGenerator
from .grounding_map import GroundingMap, GroundedAtom, create_propositional_symbol

# Re-export HDDL parser from utils for domain access across the pipeline
from utils.hddl_parser import HDDLParser, HDDLDomain, HDDLPredicate, HDDLAction, HDDLTask, HDDLMethod

__all__ = [
    'LTLFormula',
    'LTLSpecification',
    'TemporalOperator',
    'LogicalOperator',
    'NLToLTLfGenerator',
    'HDDLParser',
    'HDDLDomain',
    'HDDLPredicate',
    'HDDLAction',
    'HDDLTask',
    'HDDLMethod',
    'GroundingMap',
    'GroundedAtom',
    'create_propositional_symbol',
]
