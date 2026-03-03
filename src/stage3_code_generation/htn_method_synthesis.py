"""
HTN method synthesis for Stage 3A.

The synthesizer builds a deterministic baseline HTN method library from PDDL,
and can optionally replace/augment it with an LLM-produced library when an API
client is configured.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from stage1_interpretation.grounding_map import GroundingMap
from stage3_code_generation.htn_prompts import (
    build_htn_system_prompt,
    build_htn_user_prompt,
)
from stage3_code_generation.htn_schema import (
    HTNLiteral,
    HTNMethod,
    HTNMethodLibrary,
    HTNMethodStep,
    HTNTask,
)
from stage3_code_generation.pddl_condition_parser import (
    PDDLConditionParser,
    PDDLLiteralPattern,
    ParsedActionSemantics,
)


class HTNMethodSynthesizer:
    """Build an HTN method library for the current DFA targets."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        client: Any = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or "deepseek-chat"
        self.base_url = base_url
        self.timeout = timeout
        self.parser = PDDLConditionParser()
        self.client = client

        if self.client is None and api_key:
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
        baseline = self._build_baseline_library(domain, target_literals)

        metadata: Dict[str, Any] = {
            "used_llm": False,
            "model": self.model if self.client else None,
            "target_literals": [literal.to_signature() for literal in target_literals],
            "compound_tasks": len(baseline.compound_tasks),
            "primitive_tasks": len(baseline.primitive_tasks),
            "methods": len(baseline.methods),
            "fallback_reason": None,
            "llm_prompt": None,
            "llm_response": None,
        }

        if not self.client:
            return baseline, metadata

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
            llm_library = self._parse_llm_library(response_text)
            self._validate_library(llm_library, domain)
            merged = self._merge_libraries(baseline, llm_library, target_literals)

            metadata["used_llm"] = True
            metadata["llm_response"] = response_text
            metadata["compound_tasks"] = len(merged.compound_tasks)
            metadata["primitive_tasks"] = len(merged.primitive_tasks)
            metadata["methods"] = len(merged.methods)
            return merged, metadata
        except Exception as exc:  # noqa: BLE001 - we need a reliable fallback.
            metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return baseline, metadata

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

    def _build_baseline_library(
        self,
        domain: Any,
        target_literals: Sequence[HTNLiteral],
    ) -> HTNMethodLibrary:
        actions = [self.parser.parse_action(action) for action in domain.actions]
        primitive_tasks = [
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

        compound_tasks: Dict[str, HTNTask] = {}
        methods: Dict[str, HTNMethod] = {}

        for literal in target_literals:
            self._ensure_task_for_literal(
                literal=literal,
                actions=actions,
                compound_tasks=compound_tasks,
                methods=methods,
                ancestry=(),
            )

        return HTNMethodLibrary(
            compound_tasks=list(compound_tasks.values()),
            primitive_tasks=primitive_tasks,
            methods=list(methods.values()),
            target_literals=list(target_literals),
        )

    def _ensure_task_for_literal(
        self,
        literal: HTNLiteral,
        actions: Sequence[ParsedActionSemantics],
        compound_tasks: Dict[str, HTNTask],
        methods: Dict[str, HTNMethod],
        ancestry: Tuple[str, ...],
    ) -> None:
        task_name = self._task_name_for_literal(literal)
        parameters = self._task_parameters(len(literal.args))

        if task_name not in compound_tasks:
            compound_tasks[task_name] = HTNTask(
                name=task_name,
                parameters=parameters,
                is_primitive=False,
                source_predicates=(literal.predicate,),
            )

        guard_method = self._build_guard_method(task_name, parameters, literal)
        methods.setdefault(guard_method.method_name, guard_method)

        if not literal.is_positive or task_name in ancestry:
            return

        candidate = self._select_action(literal, actions)
        if candidate is None:
            return

        action, effect = candidate
        action_method, dependencies = self._build_action_method(
            literal=literal,
            task_name=task_name,
            task_parameters=parameters,
            action=action,
            matched_effect=effect,
        )
        methods.setdefault(action_method.method_name, action_method)

        next_ancestry = ancestry + (task_name,)
        for dependency in dependencies:
            dependency_task = self._task_name_for_literal(dependency)
            if dependency_task in next_ancestry:
                continue
            self._ensure_task_for_literal(
                literal=dependency,
                actions=actions,
                compound_tasks=compound_tasks,
                methods=methods,
                ancestry=next_ancestry,
            )

    def _build_guard_method(
        self,
        task_name: str,
        task_parameters: Tuple[str, ...],
        literal: HTNLiteral,
    ) -> HTNMethod:
        guard_literal = literal.with_args(task_parameters)
        return HTNMethod(
            method_name=f"{task_name}__guard",
            task_name=task_name,
            parameters=task_parameters,
            context=(guard_literal,),
            subtasks=(),
            ordering=(),
            origin="guard",
        )

    def _build_action_method(
        self,
        literal: HTNLiteral,
        task_name: str,
        task_parameters: Tuple[str, ...],
        action: ParsedActionSemantics,
        matched_effect: PDDLLiteralPattern,
    ) -> Tuple[HTNMethod, List[HTNLiteral]]:
        bindings: Dict[str, str] = {}
        for index, symbol in enumerate(matched_effect.args):
            bindings[symbol] = task_parameters[index]

        support_index = 1
        for parameter in action.parameters:
            if parameter not in bindings:
                bindings[parameter] = f"S{support_index}"
                support_index += 1

        dependencies: List[HTNLiteral] = []
        steps: List[HTNMethodStep] = []
        context = tuple(
            self._pattern_to_literal(pattern, bindings)
            for pattern in action.negative_preconditions
        )

        current_literal = literal.with_args(task_parameters)
        for index, pattern in enumerate(action.positive_preconditions, start=1):
            bound_literal = self._pattern_to_literal(pattern, bindings)
            if (
                bound_literal.predicate == current_literal.predicate
                and bound_literal.args == current_literal.args
                and bound_literal.is_positive == current_literal.is_positive
            ):
                continue

            dependencies.append(bound_literal)
            steps.append(
                HTNMethodStep(
                    step_id=f"s{len(steps) + 1}",
                    task_name=self._task_name_for_literal(bound_literal),
                    args=bound_literal.args,
                    kind="compound",
                    literal=bound_literal,
                )
            )

        steps.append(
            HTNMethodStep(
                step_id=f"s{len(steps) + 1}",
                task_name=self._sanitize_name(action.name),
                args=tuple(bindings[parameter] for parameter in action.parameters),
                kind="primitive",
                action_name=action.name,
                preconditions=tuple(
                    self._pattern_to_literal(pattern, bindings)
                    for pattern in action.preconditions
                ),
                effects=tuple(
                    self._pattern_to_literal(pattern, bindings)
                    for pattern in action.effects
                ),
                literal=current_literal,
            )
        )

        ordering = tuple(
            (steps[index].step_id, steps[index + 1].step_id)
            for index in range(len(steps) - 1)
        )

        return (
            HTNMethod(
                method_name=f"{task_name}__via_{self._sanitize_name(action.name)}",
                task_name=task_name,
                parameters=task_parameters,
                context=context,
                subtasks=tuple(steps),
                ordering=ordering,
                origin="heuristic",
            ),
            dependencies,
        )

    def _select_action(
        self,
        literal: HTNLiteral,
        actions: Sequence[ParsedActionSemantics],
    ) -> Optional[Tuple[ParsedActionSemantics, PDDLLiteralPattern]]:
        candidates: List[Tuple[int, int, int, int, ParsedActionSemantics, PDDLLiteralPattern]] = []

        for action_index, action in enumerate(actions):
            for effect in action.positive_effects:
                if effect.predicate != literal.predicate:
                    continue
                if len(effect.args) != len(literal.args):
                    continue

                self_references = sum(
                    1 for item in action.positive_preconditions if item.predicate == literal.predicate
                )
                unbound_params = len([item for item in action.parameters if item not in effect.args])
                candidates.append(
                    (
                        self_references,
                        len(action.positive_preconditions),
                        unbound_params,
                        action_index,
                        action,
                        effect,
                    )
                )

        if not candidates:
            return None

        _, _, _, _, action, effect = min(candidates, key=lambda item: item[:4])
        return action, effect

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

    def _validate_library(self, library: HTNMethodLibrary, domain: Any) -> None:
        primitive_names = {self._sanitize_name(action.name) for action in domain.actions}
        all_tasks = {task.name for task in library.compound_tasks + library.primitive_tasks}

        if primitive_names - {task.name for task in library.primitive_tasks}:
            raise ValueError("HTN library is missing primitive action tasks")

        for method in library.methods:
            if method.task_name not in all_tasks:
                raise ValueError(f"Unknown task in method: {method.task_name}")
            for step in method.subtasks:
                if step.task_name not in all_tasks:
                    raise ValueError(f"Unknown subtask reference: {step.task_name}")
                if step.kind == "primitive" and step.task_name not in primitive_names:
                    raise ValueError(f"Primitive step references unknown action: {step.task_name}")

    def _merge_libraries(
        self,
        baseline: HTNMethodLibrary,
        llm_library: HTNMethodLibrary,
        target_literals: Sequence[HTNLiteral],
    ) -> HTNMethodLibrary:
        compound_tasks: Dict[str, HTNTask] = {task.name: task for task in baseline.compound_tasks}
        for task in llm_library.compound_tasks:
            compound_tasks[task.name] = task

        primitive_tasks: Dict[str, HTNTask] = {task.name: task for task in baseline.primitive_tasks}
        for task in llm_library.primitive_tasks:
            primitive_tasks[task.name] = task

        methods: Dict[str, HTNMethod] = {method.method_name: method for method in baseline.methods}
        for method in llm_library.methods:
            methods[method.method_name] = method

        merged = HTNMethodLibrary(
            compound_tasks=list(compound_tasks.values()),
            primitive_tasks=list(primitive_tasks.values()),
            methods=list(methods.values()),
            target_literals=list(target_literals),
        )
        return merged

    def _pattern_to_literal(
        self,
        pattern: PDDLLiteralPattern,
        bindings: Dict[str, str],
    ) -> HTNLiteral:
        return HTNLiteral(
            predicate=pattern.predicate,
            args=tuple(bindings.get(arg, arg) for arg in pattern.args),
            is_positive=pattern.is_positive,
        )

    @staticmethod
    def _task_parameters(arity: int) -> Tuple[str, ...]:
        return tuple(f"X{index + 1}" for index in range(arity))

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("-", "_")

    def _task_name_for_literal(self, literal: HTNLiteral) -> str:
        base = self._sanitize_name(literal.predicate)
        if literal.is_positive:
            return f"achieve_{base}"
        return f"maintain_not_{base}"

    @staticmethod
    def _schema_hint() -> str:
        return json.dumps(
            {
                "compound_tasks": [
                    {
                        "name": "achieve_on",
                        "parameters": ["X1", "X2"],
                        "is_primitive": False,
                        "source_predicates": ["on"],
                    }
                ],
                "primitive_tasks": [
                    {
                        "name": "put_on_block",
                        "parameters": ["X1", "X2"],
                        "is_primitive": True,
                        "source_predicates": ["on"],
                    }
                ],
                "methods": [
                    {
                        "method_name": "achieve_on__via_put_on_block",
                        "task_name": "achieve_on",
                        "parameters": ["X1", "X2"],
                        "context": [],
                        "subtasks": [
                            {
                                "step_id": "s1",
                                "task_name": "achieve_holding",
                                "args": ["X1"],
                                "kind": "compound",
                                "action_name": None,
                                "literal": {
                                    "predicate": "holding",
                                    "args": ["X1"],
                                    "is_positive": True,
                                    "source_symbol": None,
                                },
                                "preconditions": [],
                                "effects": [],
                            }
                        ],
                        "ordering": [],
                        "origin": "llm",
                    }
                ],
                "target_literals": [],
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
