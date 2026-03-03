"""
HTN method synthesis for Stage 3.

The synthesizer uses LLM output as the only source of compound tasks and methods.
Primitive action tasks are still injected from the domain so Stage 3 can render
and validate executable AgentSpeak wrappers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from stage1_interpretation.grounding_map import GroundingMap
from stage3_method_synthesis.htn_prompts import (
    build_htn_system_prompt,
    build_htn_user_prompt,
)
from stage3_method_synthesis.htn_schema import (
    HTNLiteral,
    HTNMethod,
    HTNMethodLibrary,
    HTNMethodStep,
    HTNTask,
)
from utils.hddl_condition_parser import HDDLConditionParser


class HTNSynthesisError(RuntimeError):
    """Raised when Stage 3 cannot produce a valid HTN method library."""

    def __init__(
        self,
        message: str,
        *,
        model: Optional[str],
        llm_prompt: Optional[Dict[str, str]],
        llm_response: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.llm_prompt = llm_prompt
        self.llm_response = llm_response
        self.metadata = dict(metadata or {})


class HTNMethodSynthesizer:
    """Build an HTN method library for the current DFA targets."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model or "deepseek-chat"
        self.base_url = base_url
        self.timeout = timeout
        self.parser = HDDLConditionParser()
        self.client = None

        if api_key:
            from openai import OpenAI

            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)

    def synthesize(
        self,
        domain: Any,
        grounding_map: GroundingMap | Dict[str, Any] | None,
        dfa_result: Dict[str, Any],
    ) -> Tuple[HTNMethodLibrary, Dict[str, Any]]:
        """Create a method library and metadata for logging."""

        normalised_grounding_map = self._normalise_grounding_map(grounding_map)
        target_literals = self.extract_target_literals(normalised_grounding_map, dfa_result)
        primitive_tasks = self._build_primitive_tasks(domain)

        metadata: Dict[str, Any] = {
            "used_llm": False,
            "model": self.model if self.client else None,
            "target_literals": [literal.to_signature() for literal in target_literals],
            "compound_tasks": 0,
            "primitive_tasks": len(primitive_tasks),
            "methods": 0,
            "failure_stage": None,
            "failure_reason": None,
            "llm_prompt": None,
            "llm_response": None,
        }

        if not self.client:
            raise self._build_synthesis_error(
                metadata,
                "preflight",
                (
                    "Stage 3 requires a configured OPENAI_API_KEY. "
                    "HTN method synthesis only accepts live LLM output."
                ),
            )

        prompt = {
            "system": build_htn_system_prompt(),
            "user": build_htn_user_prompt(
                domain,
                metadata["target_literals"],
                self._schema_hint(),
            ),
        }
        metadata["llm_prompt"] = prompt

        try:
            response_text = self._call_llm(prompt)
        except Exception as exc:
            raise self._build_synthesis_error(
                metadata,
                "llm_call",
                f"LLM request failed: {exc}",
            ) from exc

        metadata["llm_response"] = response_text

        try:
            llm_library = self._normalise_llm_library(
                self._parse_llm_library(response_text),
                domain,
            )
        except Exception as exc:
            raise self._build_synthesis_error(
                metadata,
                "response_parse",
                f"LLM response could not be parsed as a valid HTN library: {exc}",
            ) from exc

        llm_only_library = HTNMethodLibrary(
            compound_tasks=list(llm_library.compound_tasks),
            primitive_tasks=primitive_tasks,
            methods=list(llm_library.methods),
            target_literals=list(target_literals),
        )
        try:
            self._validate_library(llm_only_library, domain)
        except Exception as exc:
            raise self._build_synthesis_error(
                metadata,
                "library_validation",
                f"LLM HTN library failed validation: {exc}",
            ) from exc

        metadata["used_llm"] = True
        metadata["compound_tasks"] = len(llm_only_library.compound_tasks)
        metadata["primitive_tasks"] = len(llm_only_library.primitive_tasks)
        metadata["methods"] = len(llm_only_library.methods)
        return llm_only_library, metadata

    def extract_target_literals(
        self,
        grounding_map: GroundingMap | None,
        dfa_result: Dict[str, Any],
    ) -> List[HTNLiteral]:
        """Read atomic transition labels from the simplified DFA."""

        if grounding_map is None:
            return []

        dfa_dot = dfa_result.get("dfa_dot", "")
        labels = re.findall(r'label="([^"]+)"', dfa_dot)
        seen: set[str] = set()
        literals: List[HTNLiteral] = []

        for label in labels:
            if label in {"true", "false"}:
                continue

            is_positive = not label.startswith("!")
            symbol = label[1:] if not is_positive else label
            if label in seen:
                continue
            seen.add(label)

            atom = grounding_map.get_atom(symbol)
            if atom is None:
                literal = HTNLiteral(
                    predicate=symbol,
                    args=(),
                    is_positive=is_positive,
                    source_symbol=symbol,
                )
            else:
                literal = HTNLiteral(
                    predicate=atom.predicate,
                    args=tuple(atom.args),
                    is_positive=is_positive,
                    source_symbol=symbol,
                )
            literals.append(literal)

        return literals

    def _build_primitive_tasks(self, domain: Any) -> List[HTNTask]:
        actions = [self.parser.parse_action(action) for action in domain.actions]
        return [
            HTNTask(
                name=self._sanitize_name(action.name),
                parameters=tuple(f"X{index + 1}" for index, _ in enumerate(action.parameters)),
                is_primitive=True,
                source_predicates=tuple(
                    sorted({literal.predicate for literal in action.positive_effects})
                ),
            )
            for action in actions
        ]

    def _call_llm(self, prompt: Dict[str, str]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            temperature=0.0,
            timeout=self.timeout,
        )
        return response.choices[0].message.content.strip()

    def _parse_llm_library(self, response_text: str) -> HTNMethodLibrary:
        clean_text = self._strip_code_fences(response_text)
        payload = json.loads(clean_text)
        if not isinstance(payload, dict):
            raise ValueError("HTN synthesis response must be a JSON object")
        return HTNMethodLibrary.from_dict(payload)

    def _normalise_llm_library(
        self,
        library: HTNMethodLibrary,
        domain: Any,
    ) -> HTNMethodLibrary:
        action_schemas = self._action_schema_map(domain)
        compound_tasks = [
            HTNTask(
                name=task.name,
                parameters=task.parameters,
                is_primitive=task.is_primitive,
                source_predicates=task.source_predicates,
            )
            for task in library.compound_tasks
        ]

        methods = []
        for method in library.methods:
            normalised_steps = []
            for step in method.subtasks:
                preconditions = step.preconditions
                effects = step.effects

                if step.kind == "primitive":
                    action_schema = self._resolve_action_schema(step, action_schemas)
                    if action_schema is not None:
                        preconditions = self._materialise_action_literals(
                            action_schema.preconditions,
                            action_schema.parameters,
                            step.args,
                        )
                        effects = self._materialise_action_literals(
                            action_schema.effects,
                            action_schema.parameters,
                            step.args,
                        )

                normalised_steps.append(
                    HTNMethodStep(
                        step_id=step.step_id,
                        task_name=step.task_name,
                        args=step.args,
                        kind=step.kind,
                        action_name=step.action_name,
                        literal=step.literal,
                        preconditions=preconditions,
                        effects=effects,
                    )
                )

            methods.append(
                HTNMethod(
                    method_name=method.method_name,
                    task_name=method.task_name,
                    parameters=method.parameters,
                    context=method.context,
                    subtasks=tuple(normalised_steps),
                    ordering=method.ordering,
                    origin=method.origin,
                )
            )

        return HTNMethodLibrary(
            compound_tasks=compound_tasks,
            primitive_tasks=list(library.primitive_tasks),
            methods=methods,
            target_literals=list(library.target_literals),
        )

    def _validate_library(self, library: HTNMethodLibrary, domain: Any) -> None:
        primitive_names = {self._sanitize_name(action.name) for action in domain.actions}
        compound_names = {task.name for task in library.compound_tasks}
        all_tasks = compound_names | {task.name for task in library.primitive_tasks}

        method_counts: Dict[str, int] = {}
        for method in library.methods:
            method_counts[method.task_name] = method_counts.get(method.task_name, 0) + 1

        if primitive_names - {task.name for task in library.primitive_tasks}:
            raise ValueError("HTN library is missing primitive action tasks")

        for literal in library.target_literals:
            preferred_task_name = self._preferred_task_name(literal)
            if preferred_task_name not in compound_names:
                raise ValueError(
                    "HTN library is missing the top-level compound task "
                    f"'{preferred_task_name}' required for target literal "
                    f"'{literal.to_signature()}'."
                )
            if method_counts.get(preferred_task_name, 0) == 0:
                raise ValueError(
                    "HTN library is missing a method for the top-level task "
                    f"'{preferred_task_name}' required for target literal "
                    f"'{literal.to_signature()}'."
                )

        for task in library.compound_tasks + library.primitive_tasks:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", task.name):
                raise ValueError(
                    f"Invalid task identifier '{task.name}'. "
                    "Task names must match [a-z][a-z0-9_]* for AgentSpeak compatibility."
                )

        for method in library.methods:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", method.method_name):
                raise ValueError(
                    f"Invalid method identifier '{method.method_name}'. "
                    "Method names must match [a-z][a-z0-9_]* for AgentSpeak compatibility."
                )
            if method.task_name not in compound_names:
                raise ValueError(
                    f"Unknown compound task in method '{method.method_name}': {method.task_name}. "
                    f"Known compound tasks: {sorted(compound_names)}"
                )
            if method.method_name.endswith("__guard"):
                expected_guard_name = f"{method.task_name}__guard"
                if method.method_name != expected_guard_name:
                    raise ValueError(
                        f"Guard method '{method.method_name}' must be named exactly "
                        f"'{expected_guard_name}'."
                    )
                if method.subtasks:
                    raise ValueError(
                        f"Guard method '{method.method_name}' must have an empty subtasks list."
                    )
                if method.ordering:
                    raise ValueError(
                        f"Guard method '{method.method_name}' must have an empty ordering list."
                    )
                if not method.context:
                    raise ValueError(
                        f"Guard method '{method.method_name}' must have a non-empty context."
                    )
            else:
                required_prefix = f"{method.task_name}__via_"
                if not method.method_name.startswith(required_prefix):
                    raise ValueError(
                        f"Non-guard method '{method.method_name}' must start with "
                        f"'{required_prefix}'."
                    )
                if method.method_name == required_prefix:
                    raise ValueError(
                        f"Non-guard method '{method.method_name}' must include a strategy suffix "
                        "after '__via_'."
                    )

            step_ids = {step.step_id for step in method.subtasks}
            if len(step_ids) != len(method.subtasks):
                raise ValueError(
                    f"Duplicate step_id values in method '{method.method_name}'. "
                    "Each subtask step_id must be unique."
                )
            if method.subtasks and not method.ordering:
                raise ValueError(
                    f"Method '{method.method_name}' must include explicit ordering edges for "
                    "its subtasks."
                )

            for before, after in method.ordering:
                if before not in step_ids or after not in step_ids:
                    raise ValueError(
                        f"Invalid ordering edge ({before}, {after}) in method '{method.method_name}'. "
                        f"Known step_ids: {sorted(step_ids)}"
                    )

            for step in method.subtasks:
                if not re.fullmatch(r"[a-z][a-z0-9_]*", step.task_name):
                    raise ValueError(
                        f"Invalid subtask identifier '{step.task_name}' in method '{method.method_name}'. "
                        "Subtask names must match [a-z][a-z0-9_]*."
                    )

                if step.kind == "primitive":
                    if step.task_name not in primitive_names:
                        hint = ""
                        if step.action_name is not None:
                            alias = self._sanitize_name(step.action_name)
                            if alias in primitive_names:
                                hint = (
                                    f" Use runtime alias '{alias}' for task_name and keep "
                                    f"action_name='{step.action_name}'."
                                )
                        raise ValueError(
                            f"Primitive step '{step.step_id}' in method '{method.method_name}' "
                            f"references unknown action task '{step.task_name}'. "
                            f"Allowed primitive task names: {sorted(primitive_names)}.{hint}"
                        )
                    continue

                if step.kind == "compound" and step.task_name not in compound_names:
                    raise ValueError(
                        f"Compound step '{step.step_id}' in method '{method.method_name}' "
                        f"references unknown compound task '{step.task_name}'. "
                        f"Known compound tasks: {sorted(compound_names)}"
                    )

                if step.task_name not in all_tasks:
                    raise ValueError(
                        f"Unknown subtask reference '{step.task_name}' in method '{method.method_name}'. "
                        f"Known tasks: {sorted(all_tasks)}"
                    )

    def _build_synthesis_error(
        self,
        metadata: Dict[str, Any],
        failure_stage: str,
        failure_reason: str,
    ) -> HTNSynthesisError:
        error_metadata = dict(metadata)
        error_metadata["failure_stage"] = failure_stage
        error_metadata["failure_reason"] = failure_reason
        return HTNSynthesisError(
            f"HTN synthesis failed during {failure_stage}: {failure_reason}",
            model=error_metadata.get("model"),
            llm_prompt=error_metadata.get("llm_prompt"),
            llm_response=error_metadata.get("llm_response"),
            metadata=error_metadata,
        )

    def _action_schema_map(self, domain: Any) -> Dict[str, Any]:
        action_schemas: Dict[str, Any] = {}
        for action in domain.actions:
            parsed_action = self.parser.parse_action(action)
            action_schemas[action.name] = parsed_action
            action_schemas[self._sanitize_name(action.name)] = parsed_action
        return action_schemas

    def _resolve_action_schema(self, step: HTNMethodStep, action_schemas: Dict[str, Any]) -> Any:
        if step.action_name and step.action_name in action_schemas:
            return action_schemas[step.action_name]
        return action_schemas.get(self._sanitize_name(step.task_name))

    def _materialise_action_literals(
        self,
        patterns: Tuple[Any, ...],
        schema_parameters: Tuple[str, ...],
        step_args: Tuple[str, ...],
    ) -> Tuple[HTNLiteral, ...]:
        bindings = {
            parameter: arg
            for parameter, arg in zip(schema_parameters, step_args)
        }
        return tuple(
            HTNLiteral(
                predicate=pattern.predicate,
                args=tuple(bindings.get(arg, arg) for arg in pattern.args),
                is_positive=pattern.is_positive,
                source_symbol=None,
            )
            for pattern in patterns
        )

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("-", "_")

    @classmethod
    def _preferred_task_name(cls, literal: HTNLiteral) -> str:
        predicate = cls._sanitize_name(literal.predicate)
        if literal.is_positive:
            return f"achieve_{predicate}"
        return f"maintain_not_{predicate}"

    @staticmethod
    def _schema_hint() -> str:
        return json.dumps(
            {
                "compound_tasks": [
                    {
                        "name": "achieve_delivered",
                        "parameters": ["X1"],
                        "is_primitive": False,
                        "source_predicates": ["delivered"],
                    },
                    {
                        "name": "achieve_loaded",
                        "parameters": ["X1"],
                        "is_primitive": False,
                        "source_predicates": ["loaded"],
                    }
                ],
                "methods": [
                    {
                        "method_name": "achieve_delivered__via_drop_parcel",
                        "task_name": "achieve_delivered",
                        "parameters": ["X1"],
                        "context": [],
                        "subtasks": [
                            {
                                "step_id": "s1",
                                "task_name": "achieve_loaded",
                                "args": ["X1"],
                                "kind": "compound",
                                "action_name": None,
                                "literal": {
                                    "predicate": "loaded",
                                    "args": ["X1"],
                                    "is_positive": True,
                                    "source_symbol": None,
                                },
                                "preconditions": [],
                                "effects": [],
                            },
                            {
                                "step_id": "s2",
                                "task_name": "drop_parcel",
                                "args": ["X1"],
                                "kind": "primitive",
                                "action_name": "drop-parcel",
                                "literal": {
                                    "predicate": "delivered",
                                    "args": ["X1"],
                                    "is_positive": True,
                                    "source_symbol": None,
                                },
                                "preconditions": [],
                                "effects": [],
                            }
                        ],
                        "ordering": [["s1", "s2"]],
                        "origin": "llm",
                    }
                ]
            },
            indent=2,
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if not text.startswith("```"):
            return text
        first_newline = text.find("\n")
        closing_fence = text.rfind("```")
        if first_newline == -1 or closing_fence <= first_newline:
            return text
        return text[first_newline + 1:closing_fence].strip()

    @staticmethod
    def _normalise_grounding_map(
        grounding_map: GroundingMap | Dict[str, Any] | None,
    ) -> GroundingMap | None:
        if grounding_map is None:
            return None
        if isinstance(grounding_map, GroundingMap):
            return grounding_map
        if isinstance(grounding_map, dict):
            return GroundingMap.from_dict(grounding_map)
        return None
