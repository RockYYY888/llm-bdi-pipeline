"""
Stage 1: Natural Language to LTLf Interpretation

This module handles the conversion of natural language instructions
to Linear Temporal Logic on Finite Traces (LTLf) specifications.
"""

from .ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator, LogicalOperator
from .ltlf_generator import NLToLTLfGenerator
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
