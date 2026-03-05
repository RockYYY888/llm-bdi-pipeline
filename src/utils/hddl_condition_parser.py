"""
Minimal HDDL condition parser shared by Stage 3 synthesis and Stage 5 rendering.

The parser keeps symbolic literals, including equality and disequality constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


class UnsupportedHDDLConstructError(ValueError):
    """Raised when an action uses HDDL constructs outside the supported conjunction subset."""

    def __init__(
        self,
        construct: str,
        *,
        action_name: str | None = None,
        expression: str | None = None,
    ) -> None:
        self.construct = construct
        self.action_name = action_name
        self.expression = expression
        message = (
            f"Unsupported HDDL construct '{construct}'. "
            "Only conjunctions, negation, and (in)equality literals are supported."
        )
        if action_name:
            message += f" Action: {action_name}."
        if expression:
            message += f" Expression: {expression}"
        super().__init__(message)


@dataclass(frozen=True)
class HDDLLiteralPattern:
    """A symbolic literal pattern extracted from an HDDL expression."""

    predicate: str
    args: Tuple[str, ...]
    is_positive: bool = True
    negation_mode: str = "naf"

    def bind(self, bindings: Dict[str, str]) -> "HDDLLiteralPattern":
        return HDDLLiteralPattern(
            predicate=self.predicate,
            args=tuple(bindings.get(arg, arg) for arg in self.args),
            is_positive=self.is_positive,
            negation_mode=self.negation_mode,
        )


@dataclass(frozen=True)
class ParsedActionSchema:
    """The symbolic semantics of an HDDL action schema."""

    name: str
    parameters: Tuple[str, ...]
    preconditions: Tuple[HDDLLiteralPattern, ...]
    effects: Tuple[HDDLLiteralPattern, ...]

    @property
    def positive_preconditions(self) -> Tuple[HDDLLiteralPattern, ...]:
        return tuple(item for item in self.preconditions if item.is_positive)

    @property
    def negative_preconditions(self) -> Tuple[HDDLLiteralPattern, ...]:
        return tuple(item for item in self.preconditions if not item.is_positive)

    @property
    def positive_effects(self) -> Tuple[HDDLLiteralPattern, ...]:
        return tuple(item for item in self.effects if item.is_positive)

    @property
    def negative_effects(self) -> Tuple[HDDLLiteralPattern, ...]:
        return tuple(item for item in self.effects if not item.is_positive)


class HDDLSExpressionParser:
    """Tiny S-expression parser for the subset of HDDL used here."""

    @staticmethod
    def tokenize(expression: str) -> List[str]:
        spaced = expression.replace("(", " ( ").replace(")", " ) ")
        return [token for token in spaced.split() if token]

    @classmethod
    def parse(cls, tokens: List[str], index: int = 0) -> Tuple[Any, int]:
        if index >= len(tokens):
            raise ValueError("Unexpected end of HDDL expression")

        token = tokens[index]
        if token == "(":
            result: List[Any] = []
            cursor = index + 1
            while cursor < len(tokens) and tokens[cursor] != ")":
                item, cursor = cls.parse(tokens, cursor)
                result.append(item)
            if cursor >= len(tokens):
                raise ValueError("Unclosed HDDL expression")
            return result, cursor + 1

        if token == ")":
            raise ValueError("Unexpected ')' in HDDL expression")

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


class HDDLConditionParser:
    """Extract literal patterns from HDDL preconditions or effects."""

    def parse_literals(
        self,
        expression: str,
        *,
        action_name: str | None = None,
        scope: str = "condition",
    ) -> Tuple[HDDLLiteralPattern, ...]:
        if not expression or expression.strip() == "none":
            return ()
        tree = HDDLSExpressionParser.parse_expression(expression)
        return tuple(
            self._walk(
                tree,
                action_name=action_name,
                source_scope=scope,
            )
        )

    def parse_action(self, action: Any) -> ParsedActionSchema:
        parameters = tuple(self._extract_parameter_names(action.parameters))
        return ParsedActionSchema(
            name=action.name,
            parameters=parameters,
            preconditions=self.parse_literals(
                action.preconditions,
                action_name=action.name,
                scope="precondition",
            ),
            effects=self.parse_literals(
                action.effects,
                action_name=action.name,
                scope="effect",
            ),
        )

    def _walk(
        self,
        node: Any,
        negated: bool = False,
        *,
        action_name: str | None = None,
        source_scope: str = "condition",
    ) -> Iterable[HDDLLiteralPattern]:
        if not isinstance(node, list) or not node:
            return ()

        head = str(node[0])
        if head == "and":
            items: List[HDDLLiteralPattern] = []
            for child in node[1:]:
                items.extend(
                    self._walk(
                        child,
                        negated,
                        action_name=action_name,
                        source_scope=source_scope,
                    )
                )
            return tuple(items)

        if head == "not":
            if len(node) != 2:
                return ()
            return self._walk(
                node[1],
                not negated,
                action_name=action_name,
                source_scope=source_scope,
            )

        if head in {"or", "when", "forall", "exists", "imply"}:
            raise UnsupportedHDDLConstructError(
                head,
                action_name=action_name,
                expression=f"{source_scope}: {node}",
            )

        if head == "=":
            args = tuple(str(value) for value in node[1:])
            return (
                HDDLLiteralPattern(
                    predicate="=",
                    args=args,
                    is_positive=not negated,
                    negation_mode="naf",
                ),
            )

        args = tuple(str(value) for value in node[1:])
        return (
            HDDLLiteralPattern(
                predicate=str(head),
                args=args,
                is_positive=not negated,
                negation_mode="naf",
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
