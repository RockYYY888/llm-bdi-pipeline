"""
Minimal PDDL condition parser used by the HTN Stage 3 pipeline.

The parser keeps only symbolic literals and ignores equality constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class PDDLLiteralPattern:
    """A symbolic literal pattern extracted from PDDL."""

    predicate: str
    args: Tuple[str, ...]
    is_positive: bool = True

    def bind(self, bindings: Dict[str, str]) -> "PDDLLiteralPattern":
        return PDDLLiteralPattern(
            predicate=self.predicate,
            args=tuple(bindings.get(arg, arg) for arg in self.args),
            is_positive=self.is_positive,
        )


@dataclass(frozen=True)
class ParsedActionSemantics:
    """The symbolic semantics of a PDDL action schema."""

    name: str
    parameters: Tuple[str, ...]
    preconditions: Tuple[PDDLLiteralPattern, ...]
    effects: Tuple[PDDLLiteralPattern, ...]

    @property
    def positive_preconditions(self) -> Tuple[PDDLLiteralPattern, ...]:
        return tuple(item for item in self.preconditions if item.is_positive)

    @property
    def negative_preconditions(self) -> Tuple[PDDLLiteralPattern, ...]:
        return tuple(item for item in self.preconditions if not item.is_positive)

    @property
    def positive_effects(self) -> Tuple[PDDLLiteralPattern, ...]:
        return tuple(item for item in self.effects if item.is_positive)

    @property
    def negative_effects(self) -> Tuple[PDDLLiteralPattern, ...]:
        return tuple(item for item in self.effects if not item.is_positive)


class PDDLSExpressionParser:
    """Tiny S-expression parser for the subset of PDDL used here."""

    @staticmethod
    def tokenize(expression: str) -> List[str]:
        spaced = expression.replace("(", " ( ").replace(")", " ) ")
        return [token for token in spaced.split() if token]

    @classmethod
    def parse(cls, tokens: List[str], index: int = 0) -> Tuple[Any, int]:
        if index >= len(tokens):
            raise ValueError("Unexpected end of PDDL expression")

        token = tokens[index]
        if token == "(":
            result: List[Any] = []
            cursor = index + 1
            while cursor < len(tokens) and tokens[cursor] != ")":
                item, cursor = cls.parse(tokens, cursor)
                result.append(item)
            if cursor >= len(tokens):
                raise ValueError("Unclosed PDDL expression")
            return result, cursor + 1

        if token == ")":
            raise ValueError("Unexpected ')' in PDDL expression")

        return token, index + 1

    @classmethod
    def parse_expression(cls, expression: str) -> Any:
        tokens = cls.tokenize(expression)
        if not tokens:
            return []
        if tokens[0] != "(":
            tokens = ["("] + tokens + [")"]
        tree, _ = cls.parse(tokens, 0)
        return tree


class PDDLConditionParser:
    """Extract literal patterns from preconditions or effects."""

    def parse_literals(self, expression: str) -> Tuple[PDDLLiteralPattern, ...]:
        if not expression or expression.strip() == "none":
            return ()
        tree = PDDLSExpressionParser.parse_expression(expression)
        return tuple(self._walk(tree))

    def parse_action(self, action: Any) -> ParsedActionSemantics:
        parameters = tuple(self._extract_parameter_names(action.parameters))
        return ParsedActionSemantics(
            name=action.name,
            parameters=parameters,
            preconditions=self.parse_literals(action.preconditions),
            effects=self.parse_literals(action.effects),
        )

    def _walk(self, node: Any, negated: bool = False) -> Iterable[PDDLLiteralPattern]:
        if not isinstance(node, list) or not node:
            return ()

        head = node[0]
        if head == "and":
            items: List[PDDLLiteralPattern] = []
            for child in node[1:]:
                items.extend(self._walk(child, negated))
            return tuple(items)

        if head == "not":
            if len(node) < 2:
                return ()
            return self._walk(node[1], not negated)

        if head in {"or", "when"}:
            items: List[PDDLLiteralPattern] = []
            for child in node[1:]:
                items.extend(self._walk(child, negated))
            return tuple(items)

        if head == "=":
            return ()

        args = tuple(str(value) for value in node[1:])
        return (
            PDDLLiteralPattern(
                predicate=str(head),
                args=args,
                is_positive=not negated,
            ),
        )

    @staticmethod
    def _extract_parameter_names(parameters: Iterable[str]) -> List[str]:
        names: List[str] = []
        for item in parameters:
            parts = item.split("-")[0].strip().split()
            for part in parts:
                if part:
                    names.append(part)
        return names
