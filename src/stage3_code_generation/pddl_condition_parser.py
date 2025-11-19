"""
PDDL Condition and Effect Parser

Parses PDDL preconditions and effects from their string representations.
Handles:
- Boolean connectives: and, or, not
- Equality: = (ignored in grounding)
- Grounding: Apply variable bindings to convert ?b1 -> a

Example preconditions:
    "and (handempty) (clear ?b1) (not (= ?b1 ?b2))"
    → [handempty, clear(a)] (with bindings {?b1: a, ?b2: b})

Example effects:
    "and (holding ?b1) (not (handempty))"
    → [+holding(a), -handempty]
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import PredicateAtom


@dataclass
class EffectAtom:
    """
    Represents an effect: +predicate or -predicate

    Attributes:
        predicate: The predicate being added/deleted
        is_add: True for add (+), False for delete (-)
    """
    predicate: PredicateAtom
    is_add: bool

    def to_agentspeak(self) -> str:
        """Convert to AgentSpeak belief update format"""
        prefix = "+" if self.is_add else "-"
        return f"{prefix}{self.predicate.to_agentspeak()}"

    def __str__(self) -> str:
        return self.to_agentspeak()

    def __repr__(self) -> str:
        return f"EffectAtom({self})"


class PDDLSExpressionParser:
    """
    Parser for PDDL S-expressions

    Handles tokenization and parsing of PDDL's LISP-like syntax.
    """

    @staticmethod
    def tokenize(s: str) -> List[str]:
        """
        Tokenize PDDL S-expression string

        Args:
            s: PDDL formula string

        Returns:
            List of tokens

        Example:
            "(and (on ?b1 ?b2))" → ['(', 'and', '(', 'on', '?b1', '?b2', ')', ')']
        """
        # Add spaces around parentheses
        s = s.replace('(', ' ( ').replace(')', ' ) ')

        # Split and filter empty strings
        tokens = [t.strip() for t in s.split() if t.strip()]

        return tokens

    @staticmethod
    def parse(tokens: List[str], start: int = 0) -> Tuple[Any, int]:
        """
        Parse tokens into nested structure

        Args:
            tokens: List of tokens
            start: Starting index

        Returns:
            (parsed_structure, next_index)

        Example:
            ['(', 'and', '(', 'on', 'a', 'b', ')', ')']
            → (['and', ['on', 'a', 'b']], 8)
        """
        if start >= len(tokens):
            raise ValueError("Unexpected end of tokens")

        token = tokens[start]

        if token == '(':
            # Parse list
            result = []
            i = start + 1

            while i < len(tokens) and tokens[i] != ')':
                element, i = PDDLSExpressionParser.parse(tokens, i)
                result.append(element)

            if i >= len(tokens):
                raise ValueError("Unclosed parenthesis")

            return result, i + 1

        elif token == ')':
            raise ValueError("Unexpected closing parenthesis")

        else:
            # Atom
            return token, start + 1


class PDDLConditionParser:
    """
    Parser for PDDL preconditions

    Handles:
    - and, or, not
    - Equality predicates (=) - ignored during grounding
    - Variable bindings

    Example:
        precond_str = "and (handempty) (clear ?b1) (not (= ?b1 ?b2))"
        bindings = {'?b1': 'a', '?b2': 'b'}
        result = [handempty, clear(a)]
    """

    def __init__(self):
        self.sexp_parser = PDDLSExpressionParser()

    def parse(self, condition_str: str, bindings: Dict[str, str]) -> List[PredicateAtom]:
        """
        Parse PDDL precondition string and apply bindings

        Args:
            condition_str: PDDL precondition (e.g., "and (handempty) (clear ?b1)")
            bindings: Variable bindings (e.g., {'?b1': 'a', '?b2': 'b'})

        Returns:
            List of ground PredicateAtoms
        """
        if not condition_str or condition_str.strip() == "none":
            return []

        # Tokenize and parse
        tokens = self.sexp_parser.tokenize(condition_str)
        if not tokens:
            return []

        # Parse s-expression
        if tokens[0] != '(':
            # Single predicate without parentheses
            tokens = ['('] + tokens + [')']

        sexp, _ = self.sexp_parser.parse(tokens)

        # Extract predicates
        predicates = self._extract_predicates(sexp, bindings)

        return predicates

    def _extract_predicates(self, sexp: Any, bindings: Dict[str, str],
                           negated: bool = False) -> List[PredicateAtom]:
        """
        Recursively extract predicates from s-expression tree

        Args:
            sexp: Parsed s-expression
            bindings: Variable bindings
            negated: Whether current context is negated

        Returns:
            List of PredicateAtoms
        """
        if not isinstance(sexp, list):
            # Atom (shouldn't reach here for well-formed formulas)
            return []

        if len(sexp) == 0:
            return []

        operator = sexp[0]

        if operator == 'and':
            # Conjunction: extract from all children
            predicates = []
            for child in sexp[1:]:
                predicates.extend(self._extract_predicates(child, bindings, negated))
            return predicates

        elif operator == 'or':
            # Disjunction: for preconditions, we conservatively treat as requiring all
            # (This is a simplification - in reality, OR in preconditions means "any one is sufficient")
            # For backward planning, we skip OR in preconditions
            predicates = []
            for child in sexp[1:]:
                predicates.extend(self._extract_predicates(child, bindings, negated))
            return predicates

        elif operator == 'not':
            # Negation: flip negated flag
            if len(sexp) < 2:
                return []
            return self._extract_predicates(sexp[1], bindings, negated=not negated)

        elif operator == '=':
            # Equality: skip during grounding (handled by action grounding logic)
            return []

        else:
            # Predicate: (predicate_name ?arg1 ?arg2 ...)
            return [self._create_predicate(sexp, bindings, negated)]

    def _create_predicate(self, sexp: List, bindings: Dict[str, str],
                         negated: bool) -> PredicateAtom:
        """
        Create a PredicateAtom from s-expression

        Args:
            sexp: ['on', '?b1', '?b2'] or ['handempty']
            bindings: Variable bindings
            negated: Whether predicate is negated

        Returns:
            Ground PredicateAtom
        """
        if not isinstance(sexp, list) or len(sexp) == 0:
            raise ValueError(f"Invalid predicate s-expression: {sexp}")

        pred_name = sexp[0]
        args = []

        # Ground arguments
        for arg in sexp[1:]:
            if arg.startswith('?'):
                # Variable: look up in bindings
                if arg in bindings:
                    args.append(bindings[arg])
                else:
                    # Unbound variable: keep as-is (shouldn't happen in well-formed grounding)
                    args.append(arg)
            else:
                # Constant
                args.append(arg)

        return PredicateAtom(pred_name, args, negated=negated)


class PDDLEffectParser:
    """
    Parser for PDDL effects

    Handles:
    - and: Conjunction of effects
    - not: Deletion effects

    Example:
        effect_str = "and (holding ?b1) (not (handempty))"
        bindings = {'?b1': 'a'}
        result = [[+holding(a), -handempty]]
    """

    def __init__(self):
        self.sexp_parser = PDDLSExpressionParser()

    def parse(self, effect_str: str, bindings: Dict[str, str]) -> List[List[EffectAtom]]:
        """
        Parse PDDL effect string and apply bindings

        Args:
            effect_str: PDDL effect (e.g., "and (holding ?b1) (not (handempty))")
            bindings: Variable bindings

        Returns:
            List containing single branch (list of EffectAtoms)
        """
        if not effect_str or effect_str.strip() == "none":
            return [[]]  # Empty effect

        # Tokenize and parse
        tokens = self.sexp_parser.tokenize(effect_str)
        if not tokens:
            return [[]]

        # Parse s-expression
        if tokens[0] != '(':
            # Single predicate
            tokens = ['('] + tokens + [')']

        sexp, _ = self.sexp_parser.parse(tokens)

        # Extract effects
        effects = self._extract_effects(sexp, bindings)

        return [effects]  # Return single branch

    def _extract_effects(self, sexp: Any, bindings: Dict[str, str]) -> List[EffectAtom]:
        """
        Recursively extract effects from s-expression tree

        Args:
            sexp: Parsed s-expression
            bindings: Variable bindings

        Returns:
            List of EffectAtoms
        """
        if not isinstance(sexp, list) or len(sexp) == 0:
            return []

        operator = sexp[0]

        if operator == 'and':
            # Conjunction: combine all effects
            effects = []
            for child in sexp[1:]:
                child_effects = self._extract_single_effect(child, bindings, is_add=True)
                if child_effects:
                    effects.extend(child_effects)
            return effects

        elif operator == 'not':
            # Deletion effect
            if len(sexp) < 2:
                return []
            effects = self._extract_single_effect(sexp[1], bindings, is_add=False)
            return effects

        else:
            # Single predicate (add effect)
            effects = self._extract_single_effect(sexp, bindings, is_add=True)
            return effects

    def _extract_single_effect(self, sexp: Any, bindings: Dict[str, str],
                               is_add: bool) -> List[EffectAtom]:
        """
        Extract a single effect (not a branch)

        Args:
            sexp: S-expression
            bindings: Variable bindings
            is_add: True for add effect, False for delete

        Returns:
            List of EffectAtoms (usually length 1, unless nested)
        """
        if not isinstance(sexp, list) or len(sexp) == 0:
            return []

        operator = sexp[0]

        if operator == 'not':
            # Deletion: flip is_add
            if len(sexp) < 2:
                return []
            return self._extract_single_effect(sexp[1], bindings, is_add=False)

        elif operator == 'and':
            # Nested and: flatten
            effects = []
            for child in sexp[1:]:
                effects.extend(self._extract_single_effect(child, bindings, is_add))
            return effects

        else:
            # Predicate effect
            pred = self._create_predicate(sexp, bindings)
            return [EffectAtom(pred, is_add)]

    def _create_predicate(self, sexp: List, bindings: Dict[str, str]) -> PredicateAtom:
        """Create PredicateAtom from s-expression"""
        if not isinstance(sexp, list) or len(sexp) == 0:
            raise ValueError(f"Invalid predicate s-expression: {sexp}")

        pred_name = sexp[0]
        args = []

        # Ground arguments
        for arg in sexp[1:]:
            if arg.startswith('?'):
                # Variable
                if arg in bindings:
                    args.append(bindings[arg])
                else:
                    args.append(arg)  # Keep unbound
            else:
                # Constant
                args.append(arg)

        # Effects are never negated in their PredicateAtom form
        # (negation is represented by is_add=False in EffectAtom)
        return PredicateAtom(pred_name, args, negated=False)


# Test functions
def test_sexp_parser():
    """Test S-expression parser"""
    print("="*80)
    print("Testing S-Expression Parser")
    print("="*80)

    parser = PDDLSExpressionParser()

    # Test 1: Simple predicate
    tokens1 = parser.tokenize("(handempty)")
    print(f"Tokens: {tokens1}")
    sexp1, _ = parser.parse(tokens1)
    print(f"Parsed: {sexp1}\n")

    # Test 2: Nested structure
    tokens2 = parser.tokenize("(and (handempty) (clear ?b1))")
    print(f"Tokens: {tokens2}")
    sexp2, _ = parser.parse(tokens2)
    print(f"Parsed: {sexp2}\n")

    # Test 3: Complex nesting
    tokens3 = parser.tokenize("(and (not (= ?b1 ?b2)) (clear ?b1) (on ?b1 ?b2))")
    print(f"Tokens: {tokens3}")
    sexp3, _ = parser.parse(tokens3)
    print(f"Parsed: {sexp3}\n")

    print()


def test_condition_parser():
    """Test precondition parser"""
    print("="*80)
    print("Testing Condition Parser")
    print("="*80)

    parser = PDDLConditionParser()

    # Test 1: Simple conjunction
    precond1 = "and (handempty) (clear ?b1) (on ?b1 ?b2)"
    bindings1 = {'?b1': 'a', '?b2': 'b'}
    result1 = parser.parse(precond1, bindings1)
    print(f"Precondition: {precond1}")
    print(f"Bindings: {bindings1}")
    print(f"Result: {result1}\n")

    # Test 2: With negation
    precond2 = "and (not (= ?b1 ?b2)) (handempty) (clear ?b1)"
    bindings2 = {'?b1': 'a', '?b2': 'b'}
    result2 = parser.parse(precond2, bindings2)
    print(f"Precondition: {precond2}")
    print(f"Bindings: {bindings2}")
    print(f"Result: {result2}\n")

    # Test 3: Single predicate
    precond3 = "holding ?b"
    bindings3 = {'?b': 'c'}
    result3 = parser.parse(precond3, bindings3)
    print(f"Precondition: {precond3}")
    print(f"Bindings: {bindings3}")
    print(f"Result: {result3}\n")

    print()


def test_effect_parser():
    """Test effect parser"""
    print("="*80)
    print("Testing Effect Parser")
    print("="*80)

    parser = PDDLEffectParser()

    # Test 1: Simple conjunction
    effect1 = "and (ontable ?b) (handempty) (clear ?b) (not (holding ?b))"
    bindings1 = {'?b': 'a'}
    result1 = parser.parse(effect1, bindings1)
    print(f"Effect: {effect1}")
    print(f"Bindings: {bindings1}")
    print(f"Result: {result1}")
    for i, branch in enumerate(result1):
        print(f"  Branch {i}: {[str(e) for e in branch]}\n")

    # Test 2: Single deletion effect
    effect2 = "not (handempty)"
    bindings2 = {}
    result2 = parser.parse(effect2, bindings2)
    print(f"Effect: {effect2}")
    print(f"Bindings: {bindings2}")
    print(f"Result: {result2}")
    for i, branch in enumerate(result2):
        print(f"  Branch {i}: {[str(e) for e in branch]}")
    print()

    print()


if __name__ == "__main__":
    test_sexp_parser()
    test_condition_parser()
    test_effect_parser()
