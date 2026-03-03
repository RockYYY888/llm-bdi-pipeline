"""
Stage 3 HTN method-synthesis data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple


TaskKind = Literal["compound", "primitive", "guard"]


def _serialise_literal_list(values: Iterable["HTNLiteral"]) -> List[Dict[str, Any]]:
    return [value.to_dict() for value in values]


@dataclass(frozen=True)
class HTNLiteral:
    """A symbolic literal used by HTN methods and downstream planning artifacts."""

    predicate: str
    args: Tuple[str, ...] = ()
    is_positive: bool = True
    source_symbol: Optional[str] = None

    def to_signature(self) -> str:
        base = self.predicate
        if self.args:
            base = f"{base}({', '.join(self.args)})"
        if self.is_positive:
            return base
        return f"!{base}"

    def to_agentspeak(self) -> str:
        base = self.predicate
        if self.args:
            base = f"{base}({', '.join(self.args)})"
        if self.is_positive:
            return base
        return f"not {base}"

    def with_args(self, args: Iterable[str]) -> "HTNLiteral":
        return HTNLiteral(
            predicate=self.predicate,
            args=tuple(args),
            is_positive=self.is_positive,
            source_symbol=self.source_symbol,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "args": list(self.args),
            "is_positive": self.is_positive,
            "source_symbol": self.source_symbol,
        }


@dataclass(frozen=True)
class HTNTask:
    """A named task in the HTN library."""

    name: str
    parameters: Tuple[str, ...]
    is_primitive: bool
    source_predicates: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": list(self.parameters),
            "is_primitive": self.is_primitive,
            "source_predicates": list(self.source_predicates),
        }


@dataclass(frozen=True)
class HTNMethodStep:
    """A subtask in a method body."""

    step_id: str
    task_name: str
    args: Tuple[str, ...]
    kind: TaskKind
    action_name: Optional[str] = None
    literal: Optional[HTNLiteral] = None
    preconditions: Tuple[HTNLiteral, ...] = ()
    effects: Tuple[HTNLiteral, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "task_name": self.task_name,
            "args": list(self.args),
            "kind": self.kind,
            "action_name": self.action_name,
            "literal": self.literal.to_dict() if self.literal else None,
            "preconditions": _serialise_literal_list(self.preconditions),
            "effects": _serialise_literal_list(self.effects),
        }


@dataclass(frozen=True)
class HTNMethod:
    """A method that decomposes one compound task into subtasks."""

    method_name: str
    task_name: str
    parameters: Tuple[str, ...]
    context: Tuple[HTNLiteral, ...] = ()
    subtasks: Tuple[HTNMethodStep, ...] = ()
    ordering: Tuple[Tuple[str, str], ...] = ()
    origin: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "task_name": self.task_name,
            "parameters": list(self.parameters),
            "context": _serialise_literal_list(self.context),
            "subtasks": [step.to_dict() for step in self.subtasks],
            "ordering": [list(edge) for edge in self.ordering],
            "origin": self.origin,
        }


@dataclass
class HTNMethodLibrary:
    """The Stage 3 output: a reusable HTN method library."""

    compound_tasks: List[HTNTask] = field(default_factory=list)
    primitive_tasks: List[HTNTask] = field(default_factory=list)
    methods: List[HTNMethod] = field(default_factory=list)
    target_literals: List[HTNLiteral] = field(default_factory=list)

    def methods_for_task(self, task_name: str) -> List[HTNMethod]:
        return [method for method in self.methods if method.task_name == task_name]

    def task_for_name(self, task_name: str) -> Optional[HTNTask]:
        for task in self.compound_tasks + self.primitive_tasks:
            if task.name == task_name:
                return task
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compound_tasks": [task.to_dict() for task in self.compound_tasks],
            "primitive_tasks": [task.to_dict() for task in self.primitive_tasks],
            "methods": [method.to_dict() for method in self.methods],
            "target_literals": _serialise_literal_list(self.target_literals),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HTNMethodLibrary":
        def load_task(item: Dict[str, Any]) -> HTNTask:
            return HTNTask(
                name=item["name"],
                parameters=tuple(item.get("parameters", [])),
                is_primitive=bool(item.get("is_primitive", False)),
                source_predicates=tuple(item.get("source_predicates", [])),
            )

        def load_literal(item: Optional[Dict[str, Any]]) -> Optional[HTNLiteral]:
            if item is None:
                return None
            return HTNLiteral(
                predicate=item["predicate"],
                args=tuple(item.get("args", [])),
                is_positive=bool(item.get("is_positive", True)),
                source_symbol=item.get("source_symbol"),
            )

        def load_method_step(item: Dict[str, Any]) -> HTNMethodStep:
            return HTNMethodStep(
                step_id=item["step_id"],
                task_name=item["task_name"],
                args=tuple(item.get("args", [])),
                kind=item["kind"],
                action_name=item.get("action_name"),
                literal=load_literal(item.get("literal")),
                preconditions=tuple(
                    literal
                    for literal in (
                        load_literal(value) for value in item.get("preconditions", [])
                    )
                    if literal is not None
                ),
                effects=tuple(
                    literal
                    for literal in (
                        load_literal(value) for value in item.get("effects", [])
                    )
                    if literal is not None
                ),
            )

        def load_method(item: Dict[str, Any]) -> HTNMethod:
            return HTNMethod(
                method_name=item["method_name"],
                task_name=item["task_name"],
                parameters=tuple(item.get("parameters", [])),
                context=tuple(
                    literal
                    for literal in (
                        load_literal(value) for value in item.get("context", [])
                    )
                    if literal is not None
                ),
                subtasks=tuple(load_method_step(value) for value in item.get("subtasks", [])),
                ordering=tuple(
                    (edge[0], edge[1])
                    for edge in item.get("ordering", [])
                ),
                origin=item.get("origin", "heuristic"),
            )

        return HTNMethodLibrary(
            compound_tasks=[load_task(item) for item in payload.get("compound_tasks", [])],
            primitive_tasks=[load_task(item) for item in payload.get("primitive_tasks", [])],
            methods=[load_method(item) for item in payload.get("methods", [])],
            target_literals=[
                literal
                for literal in (
                    load_literal(item) for item in payload.get("target_literals", [])
                )
                if literal is not None
            ],
        )
