"""
method synthesis HTN method-synthesis data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple


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

	prefix_equality_match = re.fullmatch(
		r"([!=]=)\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)",
		text,
	)
	if prefix_equality_match is not None:
		operator = prefix_equality_match.group(1)
		return HTNLiteral(
			predicate="=",
			args=(
				prefix_equality_match.group(2).strip(),
				prefix_equality_match.group(3).strip(),
			),
			is_positive=(operator == "=="),
			negation_mode="naf",
		)

	if text.startswith("!="):
		remainder = text[2:].strip()
		if "," in remainder:
			left, right = remainder.split(",", 1)
			return HTNLiteral(
				predicate="=",
				args=(left.strip(), right.strip()),
				is_positive=False,
				negation_mode="naf",
			)

	positive = True
	if text.startswith("!"):
		positive = False
		text = text[1:].strip()
	elif text.lower().startswith("not "):
		positive = False
		text = text[4:].strip()

	def _boolean_comparison_literal(
		left: str,
		right: str,
		*,
		comparison_is_positive: bool,
	) -> Optional["HTNLiteral"]:
		left_text = str(left or "").strip()
		right_text = str(right or "").strip()
		literal_text: Optional[str] = None
		boolean_text: Optional[str] = None
		if left_text.lower() in {"true", "false"}:
			literal_text = right_text
			boolean_text = left_text.lower()
		elif right_text.lower() in {"true", "false"}:
			literal_text = left_text
			boolean_text = right_text.lower()
		if literal_text is None or boolean_text is None:
			return None
		literal = _parse_signature_literal(literal_text)
		if literal is None or literal.is_equality:
			return None
		literal_positive = comparison_is_positive
		if boolean_text == "false":
			literal_positive = not literal_positive
		if not positive:
			literal_positive = not literal_positive
		return HTNLiteral(
			predicate=literal.predicate,
			args=literal.args,
			is_positive=literal_positive,
			negation_mode="naf",
		)

	if "==" in text:
		left, right = text.split("==", 1)
		boolean_literal = _boolean_comparison_literal(
			left,
			right,
			comparison_is_positive=True,
		)
		if boolean_literal is not None:
			return boolean_literal
		return HTNLiteral(
			predicate="=",
			args=(left.strip(), right.strip()),
			is_positive=positive,
			negation_mode="naf",
		)
	if "!=" in text:
		left, right = text.split("!=", 1)
		boolean_literal = _boolean_comparison_literal(
			left,
			right,
			comparison_is_positive=False,
		)
		if boolean_literal is not None:
			return boolean_literal
		return HTNLiteral(
			predicate="=",
			args=(left.strip(), right.strip()),
			is_positive=False,
			negation_mode="naf",
		)
	equal_match = re.fullmatch(
		r"equal\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)",
		text,
		flags=re.IGNORECASE,
	)
	if equal_match is not None:
		return HTNLiteral(
			predicate="=",
			args=(equal_match.group(1).strip(), equal_match.group(2).strip()),
			is_positive=positive,
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


def _split_top_level_invocation_sequence(raw_value: Any) -> List[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    depth = 0
    current: List[str] = []
    parts: List[str] = []
    for character in text:
        if character == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if character == "(":
            depth += 1
        elif character == ")" and depth > 0:
            depth -= 1
        current.append(character)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _contains_top_level_literal_sequence(text: str) -> bool:
    depth = 0
    closed_parenthesized_literal = False
    for character in str(text or ""):
        if character == "(":
            depth += 1
            continue
        if character == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                closed_parenthesized_literal = True
            continue
        if character == "," and depth == 0 and closed_parenthesized_literal:
            return True
    return False


def _canonical_parameter_symbol(raw_value: Any) -> str:
    token = str(raw_value or "").strip()
    if token.startswith("?"):
        token = token[1:].strip()
    if ":" in token:
        token = token.split(":", 1)[0].strip()
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


def _expand_literal_payload(item: Any) -> List[Any]:
    if item in (None, [], ""):
        return []
    if not isinstance(item, str):
        return [item]
    stripped = item.strip()
    if not stripped:
        return []
    parts: List[str] = []
    for conjunction_part in re.split(r"\s*&\s*", stripped):
        conjunction_text = conjunction_part.strip()
        if not conjunction_text:
            continue
        if conjunction_text.startswith("!="):
            parts.append(conjunction_text)
            continue
        invocation_parts = _split_top_level_invocation_sequence(conjunction_text)
        if (
            len(invocation_parts) > 1
            and all(_parse_signature_literal(part) is not None for part in invocation_parts)
        ):
            parts.extend(part.strip() for part in invocation_parts if part.strip())
            continue
        parts.append(conjunction_text)
    return parts or [stripped]


def _load_literal_values(item: Any) -> List["HTNLiteral"]:
    literals: List[HTNLiteral] = []
    for value in _expand_literal_payload(item):
        literal = _load_literal(value)
        if literal is not None:
            literals.append(literal)
    return literals


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
    source_instruction_ids: Tuple[str, ...] = ()

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
            "source_instruction_ids": list(self.source_instruction_ids),
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
    """The method synthesis output: a reusable HTN method library."""

    compound_tasks: List[HTNTask] = field(default_factory=list)
    primitive_tasks: List[HTNTask] = field(default_factory=list)
    methods: List[HTNMethod] = field(default_factory=list)
    target_literals: List[HTNLiteral] = field(default_factory=list)
    target_task_bindings: List[HTNTargetTaskBinding] = field(default_factory=list)
    _task_index: Optional[Dict[str, HTNTask]] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _task_index_size: int = field(
        default=-1,
        init=False,
        repr=False,
        compare=False,
    )
    _methods_index: Optional[Dict[str, List[HTNMethod]]] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _methods_index_size: int = field(
        default=-1,
        init=False,
        repr=False,
        compare=False,
    )

    def _ensure_task_index(self) -> None:
        task_count = len(self.compound_tasks) + len(self.primitive_tasks)
        if self._task_index is not None and self._task_index_size == task_count:
            return
        self._task_index = {
            task.name: task
            for task in [*self.compound_tasks, *self.primitive_tasks]
        }
        self._task_index_size = task_count

    def _ensure_methods_index(self) -> None:
        method_count = len(self.methods)
        if self._methods_index is not None and self._methods_index_size == method_count:
            return
        index: Dict[str, List[HTNMethod]] = {}
        for method in self.methods:
            index.setdefault(method.task_name, []).append(method)
        self._methods_index = index
        self._methods_index_size = method_count

    def methods_for_task(self, task_name: str) -> List[HTNMethod]:
        self._ensure_methods_index()
        return list((self._methods_index or {}).get(task_name, ()))

    def task_for_name(self, task_name: str) -> Optional[HTNTask]:
        self._ensure_task_index()
        return (self._task_index or {}).get(task_name)

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
                    for value in item.get("preconditions", [])
                    for literal in _load_literal_values(value)
                ),
                effects=tuple(
                    literal
                    for value in item.get("effects", [])
                    for literal in _load_literal_values(value)
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
                if isinstance(edge, dict):
                    before_value = edge.get(
                        "pre",
                        edge.get(
                            "before",
                            edge.get(
                                "parent",
                                edge.get(
                                    "from",
                                    edge.get(
                                        "first",
                                        edge.get(
                                            "precedent",
                                            edge.get("sup", edge.get("先行")),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )
                    after_value = edge.get(
                        "post",
                        edge.get(
                            "after",
                            edge.get(
                                "child",
                                edge.get(
                                    "to",
                                    edge.get(
                                        "second",
                                        edge.get(
                                            "subsequent",
                                            edge.get("sub", edge.get("后继")),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )
                    if before_value is None or after_value is None:
                        raise ValueError(
                            "ordering edge objects must include pre/post step ids",
                        )
                    normalised_edge = [str(before_value), str(after_value)]
                else:
                    if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                        raise ValueError(
                            "ordering edges must be arrays with at least two step ids like "
                            '["s1", "s2"]',
                        )
                    normalised_edge = [str(value) for value in edge]
                if len(normalised_edge) == 2:
                    ordering.append((normalised_edge[0], normalised_edge[1]))
                    continue
                for before_step, after_step in zip(normalised_edge, normalised_edge[1:]):
                    ordering.append((before_step, after_step))
            return HTNMethod(
                method_name=item["method_name"],
                task_name=item["task_name"],
                parameters=tuple(item.get("parameters", [])),
                task_args=tuple(item.get("task_args", [])),
                context=tuple(
                    literal
                    for value in item.get("context", [])
                    for literal in _load_literal_values(value)
                ),
                subtasks=tuple(load_method_step(value) for value in item.get("subtasks", [])),
                ordering=tuple(ordering),
                origin=item.get("origin", "heuristic"),
                source_method_name=item.get("source_method_name"),
                source_instruction_ids=tuple(
                    str(value).strip()
                    for value in (item.get("source_instruction_ids") or ())
                    if str(value).strip()
                ),
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
                for item in payload.get("target_literals", [])
                for literal in _load_literal_values(item)
            ],
            target_task_bindings=[
                load_binding(item) for item in payload.get("target_task_bindings", [])
            ],
        )

    @staticmethod
    def _compile_ast_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        tasks_payload = payload.get("tasks", [])
        if not isinstance(tasks_payload, list):
            raise ValueError("method synthesis tasks payload must be a list.")
        primitive_aliases = {
            str(value).strip()
            for value in (payload.get("primitive_aliases") or [])
            if str(value).strip()
        }
        call_arities = {
            str(key).strip(): int(value)
            for key, value in dict(payload.get("call_arities") or {}).items()
            if str(key).strip()
        }

        compound_tasks: List[Dict[str, Any]] = []
        methods: List[Dict[str, Any]] = []
        grouped_task_entries: Dict[str, Dict[str, Any]] = {}

        for task_index, task_entry in enumerate(tasks_payload):
            if not isinstance(task_entry, dict):
                raise ValueError("Each method synthesis task entry must be a JSON object.")

            task_name = task_entry["name"]
            raw_task_parameters = list(task_entry.get("parameters", []))
            if not raw_task_parameters:
                raw_task_parameters = HTNMethodLibrary._infer_ast_task_parameters(
                    task_entry,
                )
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
	                    "grounded_args": list(task_entry.get("grounded_args") or []),
	                    "source_predicates": [],
	                    "headline": task_entry.get("headline"),
	                    "source_name": task_entry.get("source_name"),
	                    "constructive": [],
	                },
	            )
            if task_parameter_score < grouped_entry["parameter_score"]:
                grouped_entry["parameter_score"] = task_parameter_score
                grouped_entry["task_parameters"] = list(raw_task_parameters)
            if not grouped_entry.get("grounded_args") and task_entry.get("grounded_args"):
                grouped_entry["grounded_args"] = list(task_entry.get("grounded_args") or [])
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
                        f"method synthesis task '{task_name}' contains conflicting headline literals "
                        "across duplicate task entries."
                    )

            constructive_payload = task_entry.get("constructive", [])
            if isinstance(constructive_payload, dict):
                if "branch" in constructive_payload or "branches" in constructive_payload:
                    raw_branches = (
                        constructive_payload.get("branch")
                        if constructive_payload.get("branch") is not None
                        else constructive_payload.get("branches")
                    )
                    if isinstance(raw_branches, dict):
                        grouped_entry["constructive"].append(raw_branches)
                    elif isinstance(raw_branches, list):
                        grouped_entry["constructive"].extend(list(raw_branches))
                    else:
                        grouped_entry["constructive"].append(constructive_payload)
                else:
                    grouped_entry["constructive"].append(constructive_payload)
            elif isinstance(constructive_payload, list):
                if (
                    constructive_payload
                    and len(constructive_payload) == 1
                    and isinstance(constructive_payload[0], dict)
                    and (
                        "branch" in constructive_payload[0]
                        or "branches" in constructive_payload[0]
                    )
                ):
                    wrapper_payload = constructive_payload[0]
                    raw_branches = (
                        wrapper_payload.get("branch")
                        if wrapper_payload.get("branch") is not None
                        else wrapper_payload.get("branches")
                    )
                    if isinstance(raw_branches, dict):
                        grouped_entry["constructive"].append(raw_branches)
                    elif isinstance(raw_branches, list):
                        grouped_entry["constructive"].extend(list(raw_branches))
                    else:
                        grouped_entry["constructive"].append(wrapper_payload)
                elif constructive_payload and all(
                    isinstance(item, dict)
                    for item in constructive_payload
                ):
                    grouped_entry["constructive"].extend(list(constructive_payload or []))
                elif constructive_payload and all(
                    isinstance(item, (list, tuple))
                    for item in constructive_payload
                ):
                    grouped_entry["constructive"].extend(
                        {"ordered_subtasks": list(item)}
                        for item in constructive_payload
                    )
                elif constructive_payload:
                    grouped_entry["constructive"].append(
                        {"ordered_subtasks": list(constructive_payload)},
                    )
            elif constructive_payload not in (None, "", []):
                grouped_entry["constructive"].append(
                    {"ordered_subtasks": [constructive_payload]},
                )

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
            grounded_args = [
                str(value).strip()
                for value in grouped_entry.get("grounded_args", [])
                if str(value).strip()
            ]
            if grounded_args:
                task_symbol_aliases.update(
                    {
                        grounded_arg: parameter
                        for grounded_arg, parameter in zip(grounded_args, task_parameters)
                    },
                )
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

            for branch_index, branch_payload in enumerate(
                grouped_entry.get("constructive", []),
                start=1,
            ):
                methods.append(
                    HTNMethodLibrary._compile_ast_branch(
                        task_name=task_name,
                        task_parameters=task_parameters,
                        base_symbol_aliases=task_symbol_aliases,
                        primitive_aliases=primitive_aliases,
                        call_arities=call_arities,
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
    def _infer_ast_task_parameters(task_entry: Dict[str, Any]) -> List[str]:
        headline_literal = _load_literal(task_entry.get("headline"))
        if headline_literal is not None and headline_literal.args:
            return [
                _canonical_parameter_symbol(value)
                for value in headline_literal.args
                if str(value).strip()
            ]
        return []

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
        branch_indicator_keys = (
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
        if not any(key in task_entry for key in branch_indicator_keys):
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
        base_symbol_aliases: Optional[Dict[str, str]] = None,
        primitive_aliases: Optional[set[str]] = None,
        call_arities: Optional[Dict[str, int]] = None,
        branch_key: str,
        branch_payload: Dict[str, Any],
        branch_index: int,
    ) -> Dict[str, Any]:
        if not isinstance(branch_payload, dict):
            if isinstance(branch_payload, str):
                branch_payload = {"ordered_subtasks": [branch_payload]}
            elif isinstance(branch_payload, (list, tuple)):
                branch_payload = {"ordered_subtasks": list(branch_payload)}
            else:
                raise ValueError(
                    f"method synthesis branch '{branch_key}' for task '{task_name}' must be an object, string, or list."
                )

        raw_branch_parameters = list(branch_payload.get("parameters", []))
        if raw_branch_parameters:
            branch_parameters = []
            for value in (*task_parameters, *raw_branch_parameters):
                canonical = _canonical_parameter_symbol(value)
                if canonical and canonical not in branch_parameters:
                    branch_parameters.append(canonical)
        else:
            raw_branch_parameters = list(task_parameters)
            branch_parameters = [
                _canonical_parameter_symbol(value)
                for value in raw_branch_parameters
            ]
        symbol_aliases = dict(base_symbol_aliases or {})
        symbol_aliases.update(_parameter_aliases(task_parameters))
        symbol_aliases.update(_parameter_aliases(raw_branch_parameters))
        symbol_aliases.update(_parameter_aliases(branch_parameters))
        default_task_args = [
            _normalise_symbol_with_aliases(value, symbol_aliases)
            for value in (
                branch_payload.get("task_args")
                or task_parameters
            )
        ]

        label = str(branch_payload.get("label", "")).strip()
        if label:
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

        raw_steps_payload = branch_payload.get("steps")
        if raw_steps_payload in (None, []):
            raw_steps_payload = branch_payload.get("subtasks")
        if raw_steps_payload in (None, []):
            raw_steps_payload = raw_ordered_subtasks
        if raw_steps_payload not in (None, []):
            explicit_steps_payload = list(raw_steps_payload or [])
            if support_before_payload:
                raw_steps_payload = [
                    *support_before_payload,
                    *explicit_steps_payload,
                ]
            else:
                raw_steps_payload = explicit_steps_payload
        elif support_before_payload or raw_producer not in (None, "") or followup_payload:
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

        expanded_steps_payload: List[Any] = []
        for step_payload in raw_steps_payload or []:
            if isinstance(step_payload, str):
                split_payload = _split_top_level_invocation_sequence(step_payload)
                if len(split_payload) > 1:
                    expanded_steps_payload.extend(split_payload)
                    continue
            expanded_steps_payload.append(step_payload)

        for step_payload in expanded_steps_payload:
            compiled_steps.append(
                HTNMethodLibrary._compile_ast_step(
                    step_payload,
                    task_name=task_name,
                    symbol_aliases=symbol_aliases,
                    default_step_args=default_task_args,
                    primitive_aliases=primitive_aliases,
                    call_arities=call_arities,
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
        if raw_ordering:
            def _step_signature(step_payload: Dict[str, Any]) -> str:
                task_label = str(step_payload.get("task_name") or "").strip()
                args = tuple(str(arg) for arg in (step_payload.get("args") or ()))
                return (
                    f"{task_label}({', '.join(args)})"
                    if args
                    else f"{task_label}()"
                )

            step_id_by_signature = {
                _step_signature(step): str(step.get("step_id") or "").strip()
                for step in compiled_steps
                if str(step.get("step_id") or "").strip()
            }
            known_step_ids = {
                str(step.get("step_id") or "").strip()
                for step in compiled_steps
                if str(step.get("step_id") or "").strip()
            }
            normalised_ordering: List[List[str]] = []
            for edge in raw_ordering:
                if not isinstance(edge, (list, tuple)):
                    raise ValueError(
                        f"method synthesis ordering for task '{task_name}' must use arrays of step references."
                    )
                normalised_edge: List[str] = []
                for raw_reference in edge:
                    reference_text = str(raw_reference or "").strip()
                    if not reference_text:
                        raise ValueError(
                            f"method synthesis ordering for task '{task_name}' contains an empty step reference."
                        )
                    if reference_text in known_step_ids:
                        normalised_edge.append(reference_text)
                        continue
                    parsed_reference = _parse_invocation_signature(reference_text)
                    if parsed_reference is not None:
                        canonical_call_name = str(parsed_reference["call"]).strip()
                        canonical_args = tuple(
                            _normalise_symbol_with_aliases(value, symbol_aliases)
                            for value in (parsed_reference.get("args") or [])
                        )
                        canonical_signature = (
                            f"{canonical_call_name}({', '.join(canonical_args)})"
                            if canonical_args
                            else f"{canonical_call_name}()"
                        )
                        resolved_step_id = step_id_by_signature.get(canonical_signature)
                        if resolved_step_id:
                            normalised_edge.append(resolved_step_id)
                            continue
                    raise ValueError(
                        f"method synthesis ordering for task '{task_name}' references unknown step "
                        f"'{reference_text}'."
                    )
                normalised_ordering.append(normalised_edge)
            raw_ordering = normalised_ordering

        raw_context_payload = (
            branch_payload.get("precondition")
            if branch_payload.get("precondition") is not None
            else branch_payload.get("context", [])
        )

        return {
            "method_name": f"m_{task_name}_{method_suffix}",
            "task_name": task_name,
            "parameters": branch_parameters,
            "task_args": [
                _normalise_symbol_with_aliases(value, symbol_aliases)
                for value in branch_payload.get("task_args", [])
            ],
            "context": HTNMethodLibrary._normalise_ast_literal_list(
                raw_context_payload,
                symbol_aliases,
            ),
            "subtasks": compiled_steps,
            "ordering": list(raw_ordering or []),
        }

    @staticmethod
    def _compile_ast_step(
        step_payload: Any,
        *,
        task_name: str,
        symbol_aliases: Dict[str, str],
        default_step_args: Sequence[str],
        primitive_aliases: Optional[set[str]] = None,
        call_arities: Optional[Dict[str, int]] = None,
        step_index: int,
    ) -> Dict[str, Any]:
        if isinstance(step_payload, str):
            parsed_step_payload = _parse_invocation_signature(step_payload)
            if parsed_step_payload is None:
                raise ValueError(f"method synthesis step for task '{task_name}' cannot be empty.")
            step_payload = parsed_step_payload
        if not isinstance(step_payload, dict):
            raise ValueError(
                f"method synthesis steps for task '{task_name}' must be objects or invocation strings."
            )

        raw_call = (
            step_payload.get("call")
            or step_payload.get("task_name")
            or step_payload.get("task")
            or step_payload.get("name")
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
            raise ValueError(f"method synthesis step for task '{task_name}' is missing call/task_name.")

        canonical_call_name = str(call_name).strip()
        expected_arity: Optional[int] = None
        if canonical_call_name in (call_arities or {}):
            expected_arity = max(
                int((call_arities or {}).get(canonical_call_name, 0)),
                0,
            )

        raw_args = step_payload.get("args")
        if raw_args in (None, []):
            if parsed_invocation is not None and parsed_invocation.get("args"):
                raw_args = parsed_invocation["args"]
            elif expected_arity is not None:
                raw_args = list(default_step_args[:expected_arity]) if expected_arity else []
            else:
                raw_args = step_payload.get("parameters") or list(default_step_args)
        elif expected_arity == 0:
            raw_args = []

        explicit_step_kind = str(step_payload.get("kind", "")).strip()
        if explicit_step_kind:
            step_kind = explicit_step_kind
        elif canonical_call_name in (primitive_aliases or set()):
            step_kind = "primitive"
        else:
            step_kind = "compound"
        if (
            step_kind == "compound"
            and expected_arity is not None
            and isinstance(raw_args, (list, tuple))
            and len(raw_args) > expected_arity
        ):
            # method synthesis contracts own compound-task headers. If the provider spills
            # branch-local AUX witnesses into a child call, keep only the callee's
            # declared header arity instead of rejecting a mechanically recoverable
            # invocation shape.
            raw_args = list(raw_args[:expected_arity])
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
            "preconditions": HTNMethodLibrary._normalise_ast_literal_list(
                step_payload.get("preconditions", []),
                symbol_aliases,
            ),
            "effects": HTNMethodLibrary._normalise_ast_literal_list(
                step_payload.get("effects", []),
                symbol_aliases,
            ),
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

    @staticmethod
    def _normalise_ast_literal_list(
        literal_payload: Any,
        symbol_aliases: Dict[str, str],
    ) -> List[Any]:
        if literal_payload in (None, [], ""):
            return []
        if isinstance(literal_payload, (str, dict)):
            raw_values = HTNMethodLibrary._expand_ast_literal_payload(literal_payload)
        else:
            raw_values = []
            for item in literal_payload or []:
                raw_values.extend(HTNMethodLibrary._expand_ast_literal_payload(item))
        return [
            HTNMethodLibrary._normalise_ast_literal(value, symbol_aliases)
            for value in raw_values
        ]

    @staticmethod
    def _expand_ast_literal_payload(literal_payload: Any) -> List[Any]:
        if isinstance(literal_payload, str):
            stripped = literal_payload.strip()
            if not stripped:
                return []
            parts: List[str] = []
            for conjunction_part in re.split(r"\s*&\s*", stripped):
                conjunction_text = conjunction_part.strip()
                if not conjunction_text:
                    continue
                if conjunction_text.startswith("!="):
                    parts.append(conjunction_text)
                    continue
                invocation_parts = _split_top_level_invocation_sequence(conjunction_text)
                if (
                    len(invocation_parts) > 1
                    and all(
                        _parse_signature_literal(part) is not None
                        for part in invocation_parts
                    )
                ):
                    parts.extend(
                        part.strip()
                        for part in invocation_parts
                        if part.strip()
                    )
                    continue
                if _parse_signature_literal(conjunction_text) is not None:
                    parts.append(conjunction_text)
                    continue
                parts.append(conjunction_text)
            return parts or [stripped]
        if literal_payload in (None, []):
            return []
        return [literal_payload]
