"""
Stage 3 HTN method-synthesis data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple


TaskKind = Literal["compound", "primitive", "guard"]


def _serialise_literal_list(values: Iterable["HTNLiteral"]) -> List[Dict[str, Any]]:
    return [value.to_dict() for value in values]


def _normalise_negation_mode(raw_value: Any) -> str:
    _ = raw_value
    return "naf"


def _load_literal(item: Optional[Dict[str, Any]]) -> Optional["HTNLiteral"]:
    if item is None:
        return None
    return HTNLiteral(
        predicate=item["predicate"],
        args=tuple(item.get("args", [])),
        is_positive=bool(item.get("is_positive", True)),
        negation_mode=_normalise_negation_mode(item.get("negation_mode", "naf")),
        source_symbol=item.get("source_symbol"),
    )


@dataclass(frozen=True)
class HTNLiteral:
    """A symbolic literal used by HTN methods and downstream planning artifacts."""

    predicate: str
    args: Tuple[str, ...] = ()
    is_positive: bool = True
    source_symbol: Optional[str] = None
    negation_mode: str = "naf"

    @property
    def is_equality(self) -> bool:
        return self.predicate == "="

    def to_signature(self) -> str:
        if self.is_equality and len(self.args) == 2:
            operator = "==" if self.is_positive else "!="
            return f"{self.args[0]} {operator} {self.args[1]}"
        base = self.predicate
        if self.args:
            base = f"{base}({', '.join(self.args)})"
        if self.is_positive:
            return base
        return f"!{base}"

    def to_agentspeak(self) -> str:
        if self.is_equality and len(self.args) == 2:
            operator = "==" if self.is_positive else "\\=="
            return f"{self.args[0]} {operator} {self.args[1]}"
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
            negation_mode=self.negation_mode,
            source_symbol=self.source_symbol,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "args": list(self.args),
            "is_positive": self.is_positive,
            "negation_mode": "naf",
            "source_symbol": self.source_symbol,
        }


@dataclass(frozen=True)
class HTNTask:
    """A named task in the HTN library."""

    name: str
    parameters: Tuple[str, ...]
    is_primitive: bool
    source_predicates: Tuple[str, ...] = ()
    source_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": list(self.parameters),
            "is_primitive": self.is_primitive,
            "source_predicates": list(self.source_predicates),
            "source_name": self.source_name,
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
    task_args: Tuple[str, ...] = ()
    context: Tuple[HTNLiteral, ...] = ()
    subtasks: Tuple[HTNMethodStep, ...] = ()
    ordering: Tuple[Tuple[str, str], ...] = ()
    origin: str = "heuristic"
    source_method_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "task_name": self.task_name,
            "parameters": list(self.parameters),
            "task_args": list(self.task_args),
            "context": _serialise_literal_list(self.context),
            "subtasks": [step.to_dict() for step in self.subtasks],
            "ordering": [list(edge) for edge in self.ordering],
            "origin": self.origin,
            "source_method_name": self.source_method_name,
        }


@dataclass(frozen=True)
class HTNTargetTaskBinding:
    """Maps one target literal signature to the top-level task chosen by the LLM."""

    target_literal: str
    task_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_literal": self.target_literal,
            "task_name": self.task_name,
        }


@dataclass
class HTNMethodLibrary:
    """The Stage 3 output: a reusable HTN method library."""

    compound_tasks: List[HTNTask] = field(default_factory=list)
    primitive_tasks: List[HTNTask] = field(default_factory=list)
    methods: List[HTNMethod] = field(default_factory=list)
    target_literals: List[HTNLiteral] = field(default_factory=list)
    target_task_bindings: List[HTNTargetTaskBinding] = field(default_factory=list)

    def methods_for_task(self, task_name: str) -> List[HTNMethod]:
        return [method for method in self.methods if method.task_name == task_name]

    def task_for_name(self, task_name: str) -> Optional[HTNTask]:
        for task in self.compound_tasks + self.primitive_tasks:
            if task.name == task_name:
                return task
        return None

    def task_name_for_literal(self, literal: HTNLiteral) -> Optional[str]:
        signature = literal.to_signature()
        for binding in self.target_task_bindings:
            if binding.target_literal == signature:
                return binding.task_name
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compound_tasks": [task.to_dict() for task in self.compound_tasks],
            "primitive_tasks": [task.to_dict() for task in self.primitive_tasks],
            "methods": [method.to_dict() for method in self.methods],
            "target_literals": _serialise_literal_list(self.target_literals),
            "target_task_bindings": [binding.to_dict() for binding in self.target_task_bindings],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HTNMethodLibrary":
        if "tasks" in payload and "compound_tasks" not in payload and "methods" not in payload:
            payload = cls._compile_ast_payload(payload)

        def load_task(item: Dict[str, Any]) -> HTNTask:
            return HTNTask(
                name=item["name"],
                parameters=tuple(item.get("parameters", [])),
                is_primitive=bool(item.get("is_primitive", False)),
                source_predicates=tuple(item.get("source_predicates", [])),
                source_name=item.get("source_name"),
            )

        def load_method_step(item: Dict[str, Any]) -> HTNMethodStep:
            return HTNMethodStep(
                step_id=item["step_id"],
                task_name=item["task_name"],
                args=tuple(item.get("args", [])),
                kind=item["kind"],
                action_name=item.get("action_name"),
                literal=_load_literal(item.get("literal")),
                preconditions=tuple(
                    literal
                    for literal in (
                        _load_literal(value) for value in item.get("preconditions", [])
                    )
                    if literal is not None
                ),
                effects=tuple(
                    literal
                    for literal in (
                        _load_literal(value) for value in item.get("effects", [])
                    )
                    if literal is not None
                ),
            )

        def load_method(item: Dict[str, Any]) -> HTNMethod:
            raw_ordering = item.get("ordering")
            if raw_ordering in (None, []):
                for alias in ("orderings", "ordering_edges"):
                    alias_value = item.get(alias)
                    if alias_value not in (None, []):
                        raw_ordering = alias_value
                        break
                else:
                    raw_ordering = []
            ordering: List[Tuple[str, str]] = []
            for edge in raw_ordering:
                if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                    raise ValueError(
                        "ordering edges must be length-2 arrays like "
                        '["s1", "s2"]',
                    )
                ordering.append((str(edge[0]), str(edge[1])))
            return HTNMethod(
                method_name=item["method_name"],
                task_name=item["task_name"],
                parameters=tuple(item.get("parameters", [])),
                task_args=tuple(item.get("task_args", [])),
                context=tuple(
                    literal
                    for literal in (
                        _load_literal(value) for value in item.get("context", [])
                    )
                    if literal is not None
                ),
                subtasks=tuple(load_method_step(value) for value in item.get("subtasks", [])),
                ordering=tuple(ordering),
                origin=item.get("origin", "heuristic"),
                source_method_name=item.get("source_method_name"),
            )

        def load_binding(item: Dict[str, Any]) -> HTNTargetTaskBinding:
            return HTNTargetTaskBinding(
                target_literal=item["target_literal"],
                task_name=item["task_name"],
            )

        return HTNMethodLibrary(
            compound_tasks=[load_task(item) for item in payload.get("compound_tasks", [])],
            primitive_tasks=[load_task(item) for item in payload.get("primitive_tasks", [])],
            methods=[load_method(item) for item in payload.get("methods", [])],
            target_literals=[
                literal
                for literal in (
                    _load_literal(item) for item in payload.get("target_literals", [])
                )
                if literal is not None
            ],
            target_task_bindings=[
                load_binding(item) for item in payload.get("target_task_bindings", [])
            ],
        )

    @staticmethod
    def _compile_ast_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        tasks_payload = payload.get("tasks", [])
        if not isinstance(tasks_payload, list):
            raise ValueError("Stage 3 tasks payload must be a list.")

        compound_tasks: List[Dict[str, Any]] = []
        methods: List[Dict[str, Any]] = []

        for task_entry in tasks_payload:
            if not isinstance(task_entry, dict):
                raise ValueError("Each Stage 3 task entry must be a JSON object.")

            task_name = task_entry["name"]
            task_parameters = list(task_entry.get("parameters", []))
            source_predicates = list(task_entry.get("source_predicates", []))
            source_name = task_entry.get("source_name")
            compound_tasks.append(
                {
                    "name": task_name,
                    "parameters": task_parameters,
                    "is_primitive": False,
                    "source_predicates": source_predicates,
                    "source_name": source_name,
                },
            )

            noop_branch = task_entry.get("noop")
            if noop_branch is not None:
                methods.append(
                    HTNMethodLibrary._compile_ast_branch(
                        task_name=task_name,
                        task_parameters=task_parameters,
                        branch_key="noop",
                        branch_payload=noop_branch,
                        branch_index=1,
                    ),
                )

            constructive_payload = task_entry.get("constructive", [])
            if isinstance(constructive_payload, dict):
                constructive_entries = [constructive_payload]
            else:
                constructive_entries = list(constructive_payload or [])
            for branch_index, branch_payload in enumerate(constructive_entries, start=1):
                methods.append(
                    HTNMethodLibrary._compile_ast_branch(
                        task_name=task_name,
                        task_parameters=task_parameters,
                        branch_key="constructive",
                        branch_payload=branch_payload,
                        branch_index=branch_index,
                    ),
                )

        return {
            "target_task_bindings": list(payload.get("target_task_bindings", [])),
            "compound_tasks": compound_tasks,
            "methods": methods,
        }

    @staticmethod
    def _compile_ast_branch(
        *,
        task_name: str,
        task_parameters: List[str],
        branch_key: str,
        branch_payload: Dict[str, Any],
        branch_index: int,
    ) -> Dict[str, Any]:
        if not isinstance(branch_payload, dict):
            raise ValueError(f"Stage 3 branch '{branch_key}' for task '{task_name}' must be an object.")

        label = str(branch_payload.get("label", "")).strip()
        if branch_key == "noop":
            method_suffix = "noop"
        elif label:
            method_suffix = label
        elif branch_index == 1:
            method_suffix = "constructive"
        else:
            method_suffix = f"constructive_{branch_index}"

        raw_ordering = branch_payload.get("ordering")
        if raw_ordering in (None, []):
            for alias in ("orderings", "ordering_edges"):
                alias_value = branch_payload.get(alias)
                if alias_value not in (None, []):
                    raw_ordering = alias_value
                    break
            else:
                raw_ordering = []

        compiled_steps: List[Dict[str, Any]] = []
        for step_payload in branch_payload.get("steps", []):
            if not isinstance(step_payload, dict):
                raise ValueError(f"Stage 3 steps for task '{task_name}' must be objects.")
            step_kind = str(step_payload.get("kind", "")).strip() or "compound"
            call_name = (
                step_payload.get("call")
                or step_payload.get("task_name")
                or step_payload.get("task")
                or step_payload.get("action_name")
                or step_payload.get("action")
            )
            if not call_name:
                raise ValueError(f"Stage 3 step for task '{task_name}' is missing call/task_name.")
            action_name = None
            if step_kind == "primitive":
                action_name = (
                    step_payload.get("action_name")
                    or step_payload.get("action")
                    or call_name
                )
            compiled_steps.append(
                {
                    "step_id": step_payload.get("id") or step_payload.get("step_id"),
                    "task_name": call_name,
                    "args": list(step_payload.get("args", [])),
                    "kind": step_kind,
                    "action_name": action_name,
                    "literal": step_payload.get("literal"),
                    "preconditions": list(step_payload.get("preconditions", [])),
                    "effects": list(step_payload.get("effects", [])),
                },
            )

        return {
            "method_name": f"m_{task_name}_{method_suffix}",
            "task_name": task_name,
            "parameters": list(branch_payload.get("parameters", task_parameters)),
            "task_args": list(branch_payload.get("task_args", [])),
            "context": list(branch_payload.get("context", [])),
            "subtasks": compiled_steps,
            "ordering": list(raw_ordering or []),
        }
