"""
Minimal HDDL condition parser shared by method synthesis and plan rendering.

The parser keeps symbolic literals, including equality and disequality constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


class UnsupportedHDDLConstructError(ValueError):
    """Raised when an action uses HDDL constructs outside the supported boolean subset."""

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
            "Only and/or/not/imply plus (in)equality literals are supported."
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
    precondition_clauses: Tuple[Tuple[HDDLLiteralPattern, ...], ...]
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
        clauses = self.parse_dnf(
            expression,
            action_name=action_name,
            scope=scope,
        )
        return self._required_literals_from_clauses(clauses)

    def parse_dnf(
        self,
        expression: str,
        *,
        action_name: str | None = None,
        scope: str = "condition",
    ) -> Tuple[Tuple[HDDLLiteralPattern, ...], ...]:
        normalised_expression = str(expression or "").strip()
        if not normalised_expression or normalised_expression in {"none", "()"}:
            return ((),)
        tree = HDDLSExpressionParser.parse_expression(expression)
        return self._walk_dnf(
            tree,
            action_name=action_name,
            source_scope=scope,
        )

    def parse_action(self, action: Any) -> ParsedActionSchema:
        parameters = tuple(self._extract_parameter_names(action.parameters))
        precondition_clauses = self.parse_dnf(
            action.preconditions,
            action_name=action.name,
            scope="precondition",
        )
        effect_clauses = self.parse_dnf(
            action.effects,
            action_name=action.name,
            scope="effect",
        )
        if len(effect_clauses) > 1:
            raise UnsupportedHDDLConstructError(
                "disjunctive_effect",
                action_name=action.name,
                expression=f"effect: {action.effects}",
            )
        return ParsedActionSchema(
            name=action.name,
            parameters=parameters,
            preconditions=self._required_literals_from_clauses(precondition_clauses),
            precondition_clauses=precondition_clauses,
            effects=effect_clauses[0] if effect_clauses else (),
        )

    def _walk_dnf(
        self,
        node: Any,
        negated: bool = False,
        *,
        action_name: str | None = None,
        source_scope: str = "condition",
    ) -> Tuple[Tuple[HDDLLiteralPattern, ...], ...]:
        if not isinstance(node, list) or not node:
            return ()

        head = str(node[0])
        if head == "and":
            if negated:
                return self._disjoin_dnfs(
                    [
                        self._walk_dnf(
                            child,
                            True,
                            action_name=action_name,
                            source_scope=source_scope,
                        )
                        for child in node[1:]
                    ]
                )
            return self._conjoin_dnfs(
                [
                    self._walk_dnf(
                        child,
                        False,
                        action_name=action_name,
                        source_scope=source_scope,
                    )
                    for child in node[1:]
                ]
            )

        if head == "or":
            if negated:
                return self._conjoin_dnfs(
                    [
                        self._walk_dnf(
                            child,
                            True,
                            action_name=action_name,
                            source_scope=source_scope,
                        )
                        for child in node[1:]
                    ]
                )
            return self._disjoin_dnfs(
                [
                    self._walk_dnf(
                        child,
                        False,
                        action_name=action_name,
                        source_scope=source_scope,
                    )
                    for child in node[1:]
                ]
            )

        if head == "not":
            if len(node) != 2:
                return ()
            return self._walk_dnf(
                node[1],
                not negated,
                action_name=action_name,
                source_scope=source_scope,
            )

        if head == "imply":
            if len(node) != 3:
                return ()
            antecedent = node[1]
            consequent = node[2]
            if negated:
                return self._conjoin_dnfs(
                    [
                        self._walk_dnf(
                            antecedent,
                            False,
                            action_name=action_name,
                            source_scope=source_scope,
                        ),
                        self._walk_dnf(
                            consequent,
                            True,
                            action_name=action_name,
                            source_scope=source_scope,
                        ),
                    ]
                )
            return self._disjoin_dnfs(
                [
                    self._walk_dnf(
                        antecedent,
                        True,
                        action_name=action_name,
                        source_scope=source_scope,
                    ),
                    self._walk_dnf(
                        consequent,
                        False,
                        action_name=action_name,
                        source_scope=source_scope,
                    ),
                ]
            )

        if head in {"when", "forall", "exists"}:
            raise UnsupportedHDDLConstructError(
                head,
                action_name=action_name,
                expression=f"{source_scope}: {node}",
            )

        if head == "=":
            args = tuple(str(value) for value in node[1:])
            return (
                (
                    HDDLLiteralPattern(
                        predicate="=",
                        args=args,
                        is_positive=not negated,
                        negation_mode="naf",
                    ),
                ),
            )

        args = tuple(str(value) for value in node[1:])
        return (
            (
                HDDLLiteralPattern(
                    predicate=str(head),
                    args=args,
                    is_positive=not negated,
                    negation_mode="naf",
                ),
            ),
        )

    def _disjoin_dnfs(
        self,
        dnf_items: List[Tuple[Tuple[HDDLLiteralPattern, ...], ...]],
    ) -> Tuple[Tuple[HDDLLiteralPattern, ...], ...]:
        merged: List[Tuple[HDDLLiteralPattern, ...]] = []
        for dnf in dnf_items:
            merged.extend(dnf)
        return self._normalise_dnf(tuple(merged))

    def _conjoin_dnfs(
        self,
        dnf_items: List[Tuple[Tuple[HDDLLiteralPattern, ...], ...]],
    ) -> Tuple[Tuple[HDDLLiteralPattern, ...], ...]:
        if not dnf_items:
            return ((),)

        result = dnf_items[0]
        for dnf in dnf_items[1:]:
            if not result or not dnf:
                return ()
            combined: List[Tuple[HDDLLiteralPattern, ...]] = []
            for left in result:
                for right in dnf:
                    merged = self._merge_clause_literals(left, right)
                    if merged is not None:
                        combined.append(merged)
            result = self._normalise_dnf(tuple(combined))
        return result

    def _normalise_dnf(
        self,
        clauses: Tuple[Tuple[HDDLLiteralPattern, ...], ...],
    ) -> Tuple[Tuple[HDDLLiteralPattern, ...], ...]:
        unique: List[Tuple[HDDLLiteralPattern, ...]] = []
        seen: set[Tuple[str, ...]] = set()
        for clause in clauses:
            signature = tuple(self._literal_signature(item) for item in clause)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(clause)
        return tuple(unique)

    def _merge_clause_literals(
        self,
        left: Tuple[HDDLLiteralPattern, ...],
        right: Tuple[HDDLLiteralPattern, ...],
    ) -> Tuple[HDDLLiteralPattern, ...] | None:
        merged: List[HDDLLiteralPattern] = list(left)
        literal_by_key = {
            self._literal_key(item): item
            for item in left
        }
        for item in right:
            key = self._literal_key(item)
            existing = literal_by_key.get(key)
            if existing is None:
                merged.append(item)
                literal_by_key[key] = item
                continue
            if existing.is_positive != item.is_positive:
                return None
        return tuple(merged)

    def _required_literals_from_clauses(
        self,
        clauses: Tuple[Tuple[HDDLLiteralPattern, ...], ...],
    ) -> Tuple[HDDLLiteralPattern, ...]:
        if not clauses:
            return ()

        signature_sets = [
            {self._literal_signature(item) for item in clause}
            for clause in clauses
        ]
        shared_signatures = set.intersection(*signature_sets) if signature_sets else set()
        if not shared_signatures:
            return ()

        ordered_shared: List[HDDLLiteralPattern] = []
        for item in clauses[0]:
            signature = self._literal_signature(item)
            if signature in shared_signatures:
                ordered_shared.append(item)
        return tuple(ordered_shared)

    @staticmethod
    def _literal_key(item: HDDLLiteralPattern) -> Tuple[str, Tuple[str, ...]]:
        return (item.predicate, item.args)

    @staticmethod
    def _literal_signature(item: HDDLLiteralPattern) -> str:
        atom = item.predicate if not item.args else f"{item.predicate}({', '.join(item.args)})"
        return atom if item.is_positive else f"not {atom}"

    @staticmethod
    def _extract_parameter_names(parameters: Iterable[str]) -> List[str]:
        names: List[str] = []
        for item in parameters:
            parts = item.split("-")[0].strip().split()
            for part in parts:
                if part:
                    names.append(part)
        return names
