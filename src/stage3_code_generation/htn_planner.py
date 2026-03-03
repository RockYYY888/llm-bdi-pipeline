"""
Lightweight HTN planner used by the refactored Stage 3 pipeline.

This is a deterministic decomposition planner. It selects one method per task,
expands it into a decomposition tree, and records the resulting trace.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from stage3_code_generation.htn_schema import (
    DecompositionNode,
    DecompositionTrace,
    HTNLiteral,
    HTNMethod,
    HTNMethodLibrary,
)


class HTNPlanner:
    """Build one decomposition trace for a target task."""

    def __init__(self, max_depth: int = 6) -> None:
        self.max_depth = max_depth
        self._node_counter = 0

    def build_trace(
        self,
        library: HTNMethodLibrary,
        task_name: str,
        args: Sequence[str],
        literal: HTNLiteral,
    ) -> DecompositionTrace:
        self._node_counter = 0
        root = self._expand_task(
            library=library,
            task_name=task_name,
            args=tuple(args),
            literal=literal,
            depth=0,
            stack=(),
        )
        return DecompositionTrace(root=root)

    def _expand_task(
        self,
        library: HTNMethodLibrary,
        task_name: str,
        args: tuple[str, ...],
        literal: HTNLiteral,
        depth: int,
        stack: tuple[str, ...],
    ) -> DecompositionNode:
        node_id = self._next_node_id()
        methods = library.methods_for_task(task_name)
        if not methods:
            return DecompositionNode(
                node_id=node_id,
                task_name=task_name,
                args=args,
                kind="guard",
                context=(literal.with_args(args),),
                method_name=None,
                literal=literal.with_args(args),
            )

        method = self._select_method(methods, task_name, args, depth, stack)
        bindings = {parameter: value for parameter, value in zip(method.parameters, args)}

        children: List[DecompositionNode] = []
        for step in method.subtasks:
            step_args = tuple(
                self._resolve_token(
                    token=token,
                    bindings=bindings,
                    node_id=node_id,
                    child_index=len(children) + 1,
                )
                for token in step.args
            )

            bound_literal = step.literal.with_args(step_args) if step.literal else None

            if step.kind == "primitive":
                child = DecompositionNode(
                    node_id=self._next_node_id(),
                    task_name=step.task_name,
                    args=step_args,
                    kind="primitive",
                    context=(),
                    method_name=None,
                    action_name=step.action_name,
                    literal=bound_literal,
                    preconditions=tuple(
                        self._bind_literal(item, bindings, node_id, len(children) + 1)
                        for item in step.preconditions
                    ),
                    effects=tuple(
                        self._bind_literal(item, bindings, node_id, len(children) + 1)
                        for item in step.effects
                    ),
                    children=[],
                )
            else:
                child_literal = bound_literal or literal.with_args(step_args)
                child = self._expand_task(
                    library=library,
                    task_name=step.task_name,
                    args=step_args,
                    literal=child_literal,
                    depth=depth + 1,
                    stack=stack + (self._task_signature(task_name, args),),
                )

            children.append(child)

        return DecompositionNode(
            node_id=node_id,
            task_name=task_name,
            args=args,
            kind="compound" if children else "guard",
            context=tuple(
                self._bind_literal(item, bindings, node_id, 0)
                for item in method.context
            ),
            method_name=method.method_name,
            action_name=None,
            literal=literal.with_args(args),
            preconditions=(),
            effects=(),
            children=children,
        )

    def _select_method(
        self,
        methods: List[HTNMethod],
        task_name: str,
        args: tuple[str, ...],
        depth: int,
        stack: tuple[str, ...],
    ) -> HTNMethod:
        signature = self._task_signature(task_name, args)
        guard_method = next((item for item in methods if item.origin == "guard"), None)
        active_methods = [item for item in methods if item.origin != "guard"]

        if depth >= self.max_depth or signature in stack:
            return guard_method or methods[0]

        if active_methods:
            return active_methods[0]

        return guard_method or methods[0]

    def _bind_literal(
        self,
        literal: HTNLiteral,
        bindings: Dict[str, str],
        node_id: str,
        child_index: int,
    ) -> HTNLiteral:
        bound_args = tuple(
            self._resolve_token(
                token=token,
                bindings=bindings,
                node_id=node_id,
                child_index=child_index,
            )
            for token in literal.args
        )
        return literal.with_args(bound_args)

    @staticmethod
    def _task_signature(task_name: str, args: tuple[str, ...]) -> str:
        if not args:
            return task_name
        return f"{task_name}({', '.join(args)})"

    @staticmethod
    def _resolve_token(
        token: str,
        bindings: Dict[str, str],
        node_id: str,
        child_index: int,
    ) -> str:
        if token in bindings:
            return bindings[token]

        if token and token[0].isupper():
            value = f"{token.lower()}_{node_id}_{child_index}"
            bindings[token] = value
            return value

        return token

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter}"
