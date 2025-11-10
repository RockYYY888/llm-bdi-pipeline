"""
Boolean Expression Parser for DFA Transition Labels

Parses DFA transition labels containing boolean expressions and converts them
to Disjunctive Normal Form (DNF) for independent goal exploration.

Supported operators:
- &, &&: AND
- |, ||: OR
- !, ~: NOT
- ->, =>: Implication
- <->, <=>: Equivalence

Example:
    Input: "on_a_b | (clear_c & holding_d)"
    Output: [[on(a,b)], [clear(c), holding(d)]]

    Input: "on_a_b -> clear_c"
    Output: [[~on(a,b)], [clear(c)]]  (from ~on_a_b | clear_c)
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import PredicateAtom
from stage1_interpretation.grounding_map import GroundingMap


class BoolOp(Enum):
    """Boolean operators"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # if and only if (equivalence)


@dataclass
class BoolExpr:
    """
    Boolean expression AST node

    Attributes:
        op: Operator (None for literals)
        children: Child expressions (for operators)
        literal: Literal value (for leaf nodes)
        negated: Whether literal is negated
    """
    op: Optional[BoolOp] = None
    children: List['BoolExpr'] = None
    literal: Optional[str] = None
    negated: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def is_literal(self) -> bool:
        """Check if this is a literal (not an operator)"""
        return self.op is None and self.literal is not None

    def __repr__(self) -> str:
        if self.is_literal():
            neg = "~" if self.negated else ""
            return f"{neg}{self.literal}"
        else:
            return f"({self.op.value} {' '.join(repr(c) for c in self.children)})"


class BooleanExpressionParser:
    """
    Parser for boolean expressions in DFA transition labels

    Converts expressions to DNF (Disjunctive Normal Form) for independent
    exploration of each disjunct.
    """

    def __init__(self, grounding_map: GroundingMap):
        """
        Initialize parser

        Args:
            grounding_map: Grounding map for anti-grounding symbols
        """
        self.grounding_map = grounding_map

    def parse(self, expr_str: str) -> List[List[PredicateAtom]]:
        """
        Parse boolean expression and convert to DNF

        Args:
            expr_str: Boolean expression (e.g., "on_a_b | clear_c")

        Returns:
            List of conjunctions (DNF form)
            Each inner list is a conjunction (AND)
            Outer list is disjunction (OR)

        Example:
            "on_a_b | clear_c" → [[on(a,b)], [clear(c)]]
            "on_a_b & clear_c" → [[on(a,b), clear(c)]]
        """
        # Tokenize
        tokens = self._tokenize(expr_str)

        # Parse to AST
        ast, _ = self._parse_expression(tokens, 0)

        # Apply logical transformations
        ast = self._eliminate_implications(ast)
        ast = self._push_negations(ast)
        ast = self._distribute_or_over_and(ast)

        # Extract DNF
        dnf = self._extract_dnf(ast)

        # Anti-ground literals to predicates
        result = []
        for conjunction in dnf:
            predicates = []
            for literal, negated in conjunction:
                predicate = self._antiground_literal(literal, negated)
                if predicate:
                    predicates.append(predicate)
            result.append(predicates)

        return result

    def _tokenize(self, expr_str: str) -> List[str]:
        """
        Tokenize boolean expression

        Args:
            expr_str: Expression string

        Returns:
            List of tokens
        """
        # Normalize operators
        expr_str = expr_str.replace("&&", "&")
        expr_str = expr_str.replace("||", "|")
        expr_str = expr_str.replace("=>", "->")
        expr_str = expr_str.replace("<=>", "<->")

        tokens = []
        current_token = []
        i = 0

        while i < len(expr_str):
            char = expr_str[i]

            if char in '()&|!~':
                # Operator or parenthesis
                if current_token:
                    tokens.append(''.join(current_token).strip())
                    current_token = []
                tokens.append(char)
                i += 1
            elif char == '-' and i + 1 < len(expr_str) and expr_str[i + 1] == '>':
                # Implication ->
                if current_token:
                    tokens.append(''.join(current_token).strip())
                    current_token = []
                tokens.append('->')
                i += 2
            elif char == '<' and i + 2 < len(expr_str) and expr_str[i:i+3] == '<->':
                # Equivalence <->
                if current_token:
                    tokens.append(''.join(current_token).strip())
                    current_token = []
                tokens.append('<->')
                i += 3
            elif char.isspace():
                # Whitespace
                if current_token:
                    tokens.append(''.join(current_token).strip())
                    current_token = []
                i += 1
            else:
                # Literal character
                current_token.append(char)
                i += 1

        # Add remaining token
        if current_token:
            tokens.append(''.join(current_token).strip())

        # Filter empty tokens
        tokens = [t for t in tokens if t]

        return tokens

    def _parse_expression(self, tokens: List[str], pos: int) -> Tuple[BoolExpr, int]:
        """
        Parse expression with operator precedence

        Precedence (lowest to highest):
        1. <-> (iff)
        2. -> (implies)
        3. | (or)
        4. & (and)
        5. !, ~ (not)
        6. literals, parentheses
        """
        return self._parse_iff(tokens, pos)

    def _parse_iff(self, tokens: List[str], pos: int) -> Tuple[BoolExpr, int]:
        """Parse equivalence (iff) - lowest precedence"""
        left, pos = self._parse_implies(tokens, pos)

        while pos < len(tokens) and tokens[pos] == '<->':
            pos += 1  # consume <->
            right, pos = self._parse_implies(tokens, pos)
            left = BoolExpr(op=BoolOp.IFF, children=[left, right])

        return left, pos

    def _parse_implies(self, tokens: List[str], pos: int) -> Tuple[BoolExpr, int]:
        """Parse implication"""
        left, pos = self._parse_or(tokens, pos)

        while pos < len(tokens) and tokens[pos] == '->':
            pos += 1  # consume ->
            right, pos = self._parse_or(tokens, pos)
            left = BoolExpr(op=BoolOp.IMPLIES, children=[left, right])

        return left, pos

    def _parse_or(self, tokens: List[str], pos: int) -> Tuple[BoolExpr, int]:
        """Parse OR"""
        left, pos = self._parse_and(tokens, pos)

        while pos < len(tokens) and tokens[pos] == '|':
            pos += 1  # consume |
            right, pos = self._parse_and(tokens, pos)
            left = BoolExpr(op=BoolOp.OR, children=[left, right])

        return left, pos

    def _parse_and(self, tokens: List[str], pos: int) -> Tuple[BoolExpr, int]:
        """Parse AND"""
        left, pos = self._parse_not(tokens, pos)

        while pos < len(tokens) and tokens[pos] == '&':
            pos += 1  # consume &
            right, pos = self._parse_not(tokens, pos)
            left = BoolExpr(op=BoolOp.AND, children=[left, right])

        return left, pos

    def _parse_not(self, tokens: List[str], pos: int) -> Tuple[BoolExpr, int]:
        """Parse NOT"""
        if pos < len(tokens) and tokens[pos] in ['!', '~']:
            pos += 1  # consume ! or ~
            expr, pos = self._parse_not(tokens, pos)  # Right associative
            return BoolExpr(op=BoolOp.NOT, children=[expr]), pos

        return self._parse_primary(tokens, pos)

    def _parse_primary(self, tokens: List[str], pos: int) -> Tuple[BoolExpr, int]:
        """Parse primary expression (literal or parenthesized expression)"""
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token = tokens[pos]

        if token == '(':
            # Parenthesized expression
            pos += 1  # consume (
            expr, pos = self._parse_expression(tokens, pos)

            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError(f"Expected ')' at position {pos}")

            pos += 1  # consume )
            return expr, pos

        else:
            # Literal
            return BoolExpr(literal=token), pos + 1

    def _eliminate_implications(self, expr: BoolExpr) -> BoolExpr:
        """
        Eliminate implications and equivalences

        A -> B becomes ~A | B
        A <-> B becomes (A & B) | (~A & ~B)
        """
        if expr.is_literal():
            return expr

        # Recursively process children
        new_children = [self._eliminate_implications(child) for child in expr.children]

        if expr.op == BoolOp.IMPLIES:
            # A -> B becomes ~A | B
            left, right = new_children
            return BoolExpr(op=BoolOp.OR, children=[
                BoolExpr(op=BoolOp.NOT, children=[left]),
                right
            ])

        elif expr.op == BoolOp.IFF:
            # A <-> B becomes (A & B) | (~A & ~B)
            left, right = new_children
            return BoolExpr(op=BoolOp.OR, children=[
                BoolExpr(op=BoolOp.AND, children=[left, right]),
                BoolExpr(op=BoolOp.AND, children=[
                    BoolExpr(op=BoolOp.NOT, children=[left]),
                    BoolExpr(op=BoolOp.NOT, children=[right])
                ])
            ])

        else:
            return BoolExpr(op=expr.op, children=new_children)

    def _push_negations(self, expr: BoolExpr) -> BoolExpr:
        """
        Push negations inward using De Morgan's laws

        ~(A & B) becomes ~A | ~B
        ~(A | B) becomes ~A & ~B
        ~~A becomes A
        """
        if expr.is_literal():
            return expr

        if expr.op == BoolOp.NOT:
            child = expr.children[0]

            if child.is_literal():
                # ~literal: mark as negated
                return BoolExpr(literal=child.literal, negated=not child.negated)

            elif child.op == BoolOp.NOT:
                # ~~A becomes A
                return self._push_negations(child.children[0])

            elif child.op == BoolOp.AND:
                # ~(A & B) becomes ~A | ~B
                negated_children = [
                    self._push_negations(BoolExpr(op=BoolOp.NOT, children=[c]))
                    for c in child.children
                ]
                return BoolExpr(op=BoolOp.OR, children=negated_children)

            elif child.op == BoolOp.OR:
                # ~(A | B) becomes ~A & ~B
                negated_children = [
                    self._push_negations(BoolExpr(op=BoolOp.NOT, children=[c]))
                    for c in child.children
                ]
                return BoolExpr(op=BoolOp.AND, children=negated_children)

        else:
            # Recursively process children
            new_children = [self._push_negations(child) for child in expr.children]
            return BoolExpr(op=expr.op, children=new_children)

    def _distribute_or_over_and(self, expr: BoolExpr) -> BoolExpr:
        """
        Distribute OR over AND to get DNF

        Note: For DNF, we actually want to distribute AND over OR, not OR over AND.
        This converts to CNF. For DNF extraction, we'll handle it differently in _extract_dnf.
        """
        if expr.is_literal():
            return expr

        # Recursively process children
        new_children = [self._distribute_or_over_and(child) for child in expr.children]
        return BoolExpr(op=expr.op, children=new_children)

    def _extract_dnf(self, expr: BoolExpr) -> List[List[Tuple[str, bool]]]:
        """
        Extract DNF from expression

        DNF form: (A & B) | (C & D) | E
        = OR of conjunctions (AND clauses)

        Returns:
            List of conjunctions
            Each conjunction is a list of (literal, negated) tuples
        """
        if expr.is_literal():
            # Single literal
            return [[(expr.literal, expr.negated)]]

        if expr.op == BoolOp.OR:
            # Disjunction: concatenate all disjuncts from children
            result = []
            for child in expr.children:
                child_dnf = self._extract_dnf(child)
                result.extend(child_dnf)
            return result

        elif expr.op == BoolOp.AND:
            # Conjunction: Cartesian product of all child DNFs
            # (A | B) & (C | D) = (A & C) | (A & D) | (B & C) | (B & D)

            child_dnfs = [self._extract_dnf(child) for child in expr.children]

            # Start with first child's DNF
            result = child_dnfs[0] if child_dnfs else [[]]

            # Combine with each subsequent child's DNF
            for child_dnf in child_dnfs[1:]:
                new_result = []
                for conjunction1 in result:
                    for conjunction2 in child_dnf:
                        # Combine two conjunctions
                        combined = conjunction1 + conjunction2
                        new_result.append(combined)
                result = new_result

            return result

        else:
            # Should not reach here after transformations
            return [[]]

    def _antiground_literal(self, symbol: str, negated: bool) -> Optional[PredicateAtom]:
        """
        Anti-ground symbol using grounding map

        Args:
            symbol: Propositional symbol (e.g., "on_a_b")
            negated: Whether symbol is negated

        Returns:
            PredicateAtom or None if not found in grounding map
        """
        atom = self.grounding_map.get_atom(symbol)

        if atom is None:
            print(f"Warning: Symbol '{symbol}' not found in grounding map")
            return None

        return PredicateAtom(atom.predicate, atom.args, negated=negated)


# Test functions
def test_tokenizer():
    """Test tokenizer"""
    print("="*80)
    print("Testing Tokenizer")
    print("="*80)

    from stage1_interpretation.grounding_map import GroundingMap
    gmap = GroundingMap()
    parser = BooleanExpressionParser(gmap)

    test_cases = [
        "on_a_b",
        "on_a_b & clear_c",
        "on_a_b | clear_c",
        "~on_a_b",
        "on_a_b -> clear_c",
        "on_a_b <-> clear_c",
        "(on_a_b | clear_c) & holding_d",
    ]

    for expr in test_cases:
        tokens = parser._tokenize(expr)
        print(f"{expr}")
        print(f"  Tokens: {tokens}\n")

    print()


def test_parser():
    """Test parser and DNF conversion"""
    print("="*80)
    print("Testing Parser and DNF Conversion")
    print("="*80)

    # Create grounding map
    from stage1_interpretation.grounding_map import GroundingMap
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    gmap.add_atom("holding_d", "holding", ["d"])
    gmap.add_atom("handempty", "handempty", [])

    parser = BooleanExpressionParser(gmap)

    test_cases = [
        "on_a_b",
        "on_a_b & clear_c",
        "on_a_b | clear_c",
        "~on_a_b",
        "on_a_b -> clear_c",
        "on_a_b <-> clear_c",
        "(on_a_b | clear_c) & holding_d",
        "~(on_a_b & clear_c)",
    ]

    for expr in test_cases:
        print(f"Expression: {expr}")
        try:
            dnf = parser.parse(expr)
            print(f"  DNF ({len(dnf)} disjuncts):")
            for i, conjunction in enumerate(dnf):
                pred_strs = [str(p) for p in conjunction]
                print(f"    Disjunct {i}: [{', '.join(pred_strs)}]")
        except Exception as e:
            print(f"  Error: {e}")
        print()

    print()


if __name__ == "__main__":
    test_tokenizer()
    test_parser()
