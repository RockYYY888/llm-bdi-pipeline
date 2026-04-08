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


def _parse_signature_literal(raw_value: Any) -> Optional["HTNLiteral"]:
    text = str(raw_value or "").strip()
    if not text:
        return None

    positive = True
    if text.startswith("!"):
        positive = False
        text = text[1:].strip()
    elif text.lower().startswith("not "):
        positive = False
        text = text[4:].strip()

    if "==" in text:
        left, right = text.split("==", 1)
        return HTNLiteral(
            predicate="=",
            args=(left.strip(), right.strip()),
            is_positive=positive,
            negation_mode="naf",
        )
    if "!=" in text:
        left, right = text.split("!=", 1)
        return HTNLiteral(
            predicate="=",
            args=(left.strip(), right.strip()),
            is_positive=False,
            negation_mode="naf",
        )

    if "(" not in text:
        return HTNLiteral(
            predicate=text,
            args=(),
            is_positive=positive,
            negation_mode="naf",
        )
    if not text.endswith(")"):
        raise ValueError(f"Invalid literal signature: {raw_value!r}")
    predicate, raw_args = text.split("(", 1)
    args = tuple(
        value.strip()
        for value in raw_args[:-1].split(",")
        if value.strip()
    )
    return HTNLiteral(
        predicate=predicate.strip(),
        args=args,
        is_positive=positive,
        negation_mode="naf",
    )


def _parse_invocation_signature(raw_value: Any) -> Optional[Dict[str, Any]]:
    text = str(raw_value or "").strip()
    if not text:
        return None
    if "(" not in text:
        return {"call": text, "args": []}
    if not text.endswith(")"):
        raise ValueError(f"Invalid task invocation signature: {raw_value!r}")
    call_name, raw_args = text.split("(", 1)
    args = [
        value.strip()
        for value in raw_args[:-1].split(",")
        if value.strip()
    ]
    return {
        "call": call_name.strip(),
        "args": args,
    }


def _canonical_parameter_symbol(raw_value: Any) -> str:
    token = str(raw_value or "").strip()
    if token.startswith("?"):
        token = token[1:].strip()
    return token.upper()


def _parameter_aliases(symbols: Iterable[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for raw_symbol in symbols:
        raw_text = str(raw_symbol or "").strip()
        if not raw_text:
            continue
        canonical = _canonical_parameter_symbol(raw_text)
        base = raw_text[1:].strip() if raw_text.startswith("?") else raw_text
        for alias in {
            raw_text,
            base,
            f"?{base}" if base else "",
            raw_text.lower(),
            raw_text.upper(),
            base.lower(),
            base.upper(),
            f"?{base.lower()}" if base else "",
            f"?{base.upper()}" if base else "",
        }:
            alias_text = str(alias or "").strip()
            if alias_text:
                aliases[alias_text] = canonical
    return aliases


def _normalise_symbol_with_aliases(raw_value: Any, aliases: Dict[str, str]) -> str:
    token = str(raw_value or "").strip()
    if not token:
        return token
    if token in aliases:
        return aliases[token]
    if token.startswith("?"):
        stripped = token[1:].strip()
        if stripped in aliases:
            return aliases[stripped]
        return _canonical_parameter_symbol(token)
    return token


def _load_literal(item: Optional[Dict[str, Any]]) -> Optional["HTNLiteral"]:
    if item is None:
        return None
    if isinstance(item, str):
        return _parse_signature_literal(item)
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
    headline_literal: Optional[HTNLiteral] = None
    source_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": list(self.parameters),
            "is_primitive": self.is_primitive,
            "source_predicates": list(self.source_predicates),
            "headline": self.headline_literal.to_dict() if self.headline_literal else None,
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
                headline_literal=_load_literal(item.get("headline")),
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
            raw_task_name = item.get("task_name")
            parsed_task_invocation = (
                _parse_invocation_signature(raw_task_name)
                if isinstance(raw_task_name, str)
                else None
            )
            return HTNTargetTaskBinding(
                target_literal=item["target_literal"],
                task_name=(
                    parsed_task_invocation["call"]
                    if parsed_task_invocation is not None
                    else item["task_name"]
                ),
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
        grouped_task_entries: Dict[str, Dict[str, Any]] = {}

        for task_index, task_entry in enumerate(tasks_payload):
            if not isinstance(task_entry, dict):
                raise ValueError("Each Stage 3 task entry must be a JSON object.")

            task_name = task_entry["name"]
            raw_task_parameters = list(task_entry.get("parameters", []))
            task_parameter_score = (
                len(raw_task_parameters),
                sum(
                    1
                    for value in raw_task_parameters
                    if not _canonical_parameter_symbol(value).startswith("ARG")
                ),
                task_index,
            )
            grouped_entry = grouped_task_entries.setdefault(
                task_name,
                {
                    "parameter_score": task_parameter_score,
                    "task_parameters": list(raw_task_parameters),
                    "source_predicates": [],
                    "headline": task_entry.get("headline"),
                    "source_name": task_entry.get("source_name"),
                    "noop": None,
                    "constructive": [],
                },
            )
            if task_parameter_score < grouped_entry["parameter_score"]:
                grouped_entry["parameter_score"] = task_parameter_score
                grouped_entry["task_parameters"] = list(raw_task_parameters)
            for predicate_name in list(task_entry.get("source_predicates", [])):
                if predicate_name not in grouped_entry["source_predicates"]:
                    grouped_entry["source_predicates"].append(predicate_name)
            if grouped_entry.get("source_name") in (None, "") and task_entry.get("source_name") not in (None, ""):
                grouped_entry["source_name"] = task_entry.get("source_name")
            task_headline = task_entry.get("headline")
            if task_headline is not None:
                existing_headline = grouped_entry.get("headline")
                if existing_headline is None:
                    grouped_entry["headline"] = task_headline
                elif existing_headline != task_headline:
                    raise ValueError(
                        f"Stage 3 task '{task_name}' contains conflicting headline literals "
                        "across duplicate task entries."
                    )

            noop_branch = task_entry.get("noop")
            if noop_branch is not None:
                existing_noop = grouped_entry.get("noop")
                if existing_noop is None:
                    grouped_entry["noop"] = noop_branch
                elif existing_noop != noop_branch:
                    raise ValueError(
                        f"Stage 3 task '{task_name}' contains conflicting noop branches "
                        "across duplicate task entries."
                    )

            constructive_payload = task_entry.get("constructive", [])
            if isinstance(constructive_payload, dict):
                grouped_entry["constructive"].append(constructive_payload)
            else:
                grouped_entry["constructive"].extend(list(constructive_payload or []))

            task_level_constructive = HTNMethodLibrary._extract_task_level_constructive_branch(
                task_entry,
            )
            if task_level_constructive is not None:
                grouped_entry["constructive"].append(task_level_constructive)

        for task_name, grouped_entry in grouped_task_entries.items():
            task_parameters = [
                _canonical_parameter_symbol(value)
                for value in grouped_entry.get("task_parameters", [])
            ]
            task_symbol_aliases = _parameter_aliases(task_parameters)
            compound_tasks.append(
                {
                    "name": task_name,
                    "parameters": task_parameters,
                    "is_primitive": False,
                    "source_predicates": list(grouped_entry.get("source_predicates", [])),
                    "headline": HTNMethodLibrary._normalise_ast_literal(
                        grouped_entry.get("headline"),
                        task_symbol_aliases,
                    ),
                    "source_name": grouped_entry.get("source_name"),
                },
            )
            compiled_task = compound_tasks[-1]
            if (
                not compiled_task["source_predicates"]
                and isinstance(compiled_task.get("headline"), dict)
                and str(compiled_task["headline"].get("predicate", "")).strip()
            ):
                compiled_task["source_predicates"] = [
                    str(compiled_task["headline"]["predicate"]).strip()
                ]

            noop_branch = grouped_entry.get("noop")
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

            for branch_index, branch_payload in enumerate(
                grouped_entry.get("constructive", []),
                start=1,
            ):
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
    def _extract_task_level_constructive_branch(
        task_entry: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        branch_keys = (
            "label",
            "parameters",
            "task_args",
            "precondition",
            "context",
            "ordered_subtasks",
            "ordering",
            "orderings",
            "ordering_edges",
            "subtasks",
            "steps",
            "support_before",
            "producer",
            "produce",
            "followup",
            "followups",
        )
        if task_entry.get("constructive") not in (None, [], {}):
            return None
        if not any(key in task_entry for key in branch_keys):
            return None

        branch_payload: Dict[str, Any] = {}
        for key in branch_keys:
            if key in task_entry:
                branch_payload[key] = task_entry[key]
        return branch_payload

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

        raw_branch_parameters = list(branch_payload.get("parameters", task_parameters))
        branch_parameters = [
            _canonical_parameter_symbol(value)
            for value in raw_branch_parameters
        ]
        symbol_aliases = _parameter_aliases(task_parameters)
        symbol_aliases.update(_parameter_aliases(raw_branch_parameters))
        default_task_args = [
            _normalise_symbol_with_aliases(value, symbol_aliases)
            for value in (
                branch_payload.get("task_args")
                or task_parameters
            )
        ]

        label = str(branch_payload.get("label", "")).strip()
        if branch_key == "noop":
            method_suffix = "noop"
        elif label:
            method_suffix = label
        elif branch_index == 1:
            method_suffix = "constructive"
        else:
            method_suffix = f"constructive_{branch_index}"

        raw_ordered_subtasks = branch_payload.get("ordered_subtasks")
        raw_ordering = branch_payload.get("ordering")
        used_compact_linear_slots = False
        if raw_ordering in (None, []):
            for alias in ("orderings", "ordering_edges"):
                alias_value = branch_payload.get(alias)
                if alias_value not in (None, []):
                    raw_ordering = alias_value
                    break
            else:
                raw_ordering = []

        compiled_steps: List[Dict[str, Any]] = []
        raw_steps_payload = branch_payload.get("steps")
        if raw_steps_payload in (None, []):
            raw_steps_payload = branch_payload.get("subtasks")
        if raw_steps_payload in (None, []):
            raw_steps_payload = raw_ordered_subtasks
        if raw_steps_payload in (None, []):
            raw_support_before = branch_payload.get("support_before")
            if isinstance(raw_support_before, (str, dict)):
                support_before_payload = [raw_support_before]
            else:
                support_before_payload = list(raw_support_before or [])

            raw_producer = branch_payload.get("producer")
            if raw_producer in (None, ""):
                raw_producer = branch_payload.get("produce")

            raw_followup = branch_payload.get("followup")
            if raw_followup in (None, []):
                raw_followup = branch_payload.get("followups")
            if isinstance(raw_followup, (str, dict)):
                followup_payload = [raw_followup]
            else:
                followup_payload = list(raw_followup or [])

            if support_before_payload or raw_producer not in (None, "") or followup_payload:
                used_compact_linear_slots = True
                if isinstance(raw_producer, str):
                    parsed_producer = _parse_invocation_signature(raw_producer)
                    if parsed_producer is not None and not parsed_producer.get("args"):
                        raw_producer = {
                            "call": parsed_producer["call"],
                            "args": list(default_task_args),
                        }
                raw_steps_payload = [
                    *support_before_payload,
                    *([raw_producer] if raw_producer not in (None, "") else []),
                    *followup_payload,
                ]

        for step_payload in raw_steps_payload or []:
            compiled_steps.append(
                HTNMethodLibrary._compile_ast_step(
                    step_payload,
                    task_name=task_name,
                    symbol_aliases=symbol_aliases,
                    step_index=len(compiled_steps) + 1,
                ),
            )

        if raw_ordering in (None, []):
            raw_ordering = []
        if not raw_ordering and (
            raw_ordered_subtasks not in (None, []) or used_compact_linear_slots
        ):
            ordered_step_ids = [
                str(step.get("step_id") or "").strip()
                for step in compiled_steps
            ]
            if len(compiled_steps) > 1 and all(ordered_step_ids):
                raw_ordering = [
                    [ordered_step_ids[index], ordered_step_ids[index + 1]]
                    for index in range(len(ordered_step_ids) - 1)
                ]

        return {
            "method_name": f"m_{task_name}_{method_suffix}",
            "task_name": task_name,
            "parameters": branch_parameters,
            "task_args": [
                _normalise_symbol_with_aliases(value, symbol_aliases)
                for value in branch_payload.get("task_args", [])
            ],
            "context": [
                HTNMethodLibrary._normalise_ast_literal(value, symbol_aliases)
                for value in (
                    branch_payload.get("precondition")
                    if branch_payload.get("precondition") is not None
                    else branch_payload.get("context", [])
                )
            ],
            "subtasks": compiled_steps,
            "ordering": list(raw_ordering or []),
        }

    @staticmethod
    def _compile_ast_step(
        step_payload: Any,
        *,
        task_name: str,
        symbol_aliases: Dict[str, str],
        step_index: int,
    ) -> Dict[str, Any]:
        if isinstance(step_payload, str):
            parsed_step_payload = _parse_invocation_signature(step_payload)
            if parsed_step_payload is None:
                raise ValueError(f"Stage 3 step for task '{task_name}' cannot be empty.")
            step_payload = parsed_step_payload
        if not isinstance(step_payload, dict):
            raise ValueError(
                f"Stage 3 steps for task '{task_name}' must be objects or invocation strings."
            )

        raw_call = (
            step_payload.get("call")
            or step_payload.get("task_name")
            or step_payload.get("task")
            or step_payload.get("action_name")
            or step_payload.get("action")
            or step_payload.get("producer")
        )
        parsed_invocation = (
            _parse_invocation_signature(raw_call)
            if isinstance(raw_call, str)
            else None
        )
        call_name = (
            parsed_invocation["call"]
            if parsed_invocation is not None
            else raw_call
        )
        if not call_name:
            raise ValueError(f"Stage 3 step for task '{task_name}' is missing call/task_name.")

        raw_args = step_payload.get("args")
        if raw_args in (None, []):
            if parsed_invocation is not None and parsed_invocation.get("args"):
                raw_args = parsed_invocation["args"]
            else:
                raw_args = step_payload.get("parameters", [])

        step_kind = str(step_payload.get("kind", "")).strip() or "compound"
        action_name = None
        if step_kind == "primitive":
            action_name = (
                step_payload.get("action_name")
                or step_payload.get("action")
                or call_name
            )

        return {
            "step_id": (
                step_payload.get("id")
                or step_payload.get("step_id")
                or f"s{step_index}"
            ),
            "task_name": str(call_name).strip(),
            "args": [
                _normalise_symbol_with_aliases(value, symbol_aliases)
                for value in (raw_args or [])
            ],
            "kind": step_kind,
            "action_name": action_name,
            "literal": HTNMethodLibrary._normalise_ast_literal(
                step_payload.get("literal"),
                symbol_aliases,
            ),
            "preconditions": [
                HTNMethodLibrary._normalise_ast_literal(value, symbol_aliases)
                for value in step_payload.get("preconditions", [])
            ],
            "effects": [
                HTNMethodLibrary._normalise_ast_literal(value, symbol_aliases)
                for value in step_payload.get("effects", [])
            ],
        }

    @staticmethod
    def _normalise_ast_literal(
        literal_payload: Any,
        symbol_aliases: Dict[str, str],
    ) -> Any:
        if isinstance(literal_payload, str):
            parsed_literal = _parse_signature_literal(literal_payload)
            if parsed_literal is None:
                return literal_payload
            literal_payload = parsed_literal.to_dict()
        if not isinstance(literal_payload, dict):
            return literal_payload
        normalised = dict(literal_payload)
        normalised["args"] = [
            _normalise_symbol_with_aliases(value, symbol_aliases)
            for value in literal_payload.get("args", [])
        ]
        return normalised
