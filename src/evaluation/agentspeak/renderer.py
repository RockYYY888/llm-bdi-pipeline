"""
AgentSpeak rendering for the generated plan library S.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from method_library.synthesis.schema import HTNMethod, HTNMethodLibrary
from planning.plan_models import PANDAPlanResult
from plan_library.models import AgentSpeakBodyStep, AgentSpeakPlan, PlanLibrary
from utils.hddl_condition_parser import HDDLConditionParser


class AgentSpeakRenderer:
    """Render primitive action wrappers and AgentSpeak(L) plan-library entries."""

    _VARIABLE_MAP_INFERENCE_STEP_LIMIT = 64
    _VARIABLE_MAP_INFERENCE_CONTEXT_LIMIT = 256

    def __init__(self) -> None:
        self.parser = HDDLConditionParser()
        self._object_symbol_set: set[str] = set()
        self._object_type_map: Dict[str, str] = {}
        self._type_parent_map: Dict[str, Optional[str]] = {}
        self._asl_term_cache: Dict[str, str] = {}

    def generate(
        self,
        domain: Any,
        objects: Sequence[str],
        method_library: HTNMethodLibrary,
        plan_records: Sequence[Dict[str, Any]],
        plan_library: Optional[PlanLibrary] = None,
        typed_objects: Optional[Sequence[Tuple[str, str]]] = None,
        subgoals: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        self._asl_term_cache = {}
        self._object_symbol_set = {str(obj) for obj in objects}
        self._object_type_map = {
            str(name): str(type_name)
            for name, type_name in (typed_objects or ())
        }
        self._type_parent_map = self._build_type_parent_map(domain)
        subgoals = tuple(subgoals or ())
        _ = subgoals
        lines: List[str] = []
        lines.extend(
            self._render_header(
                domain,
                objects,
                plan_records,
            ),
        )
        lines.extend(self._render_primitive_wrappers(domain))
        if plan_library is not None:
            lines.extend(self._render_structured_plan_library(plan_library))
        else:
            lines.extend(self._render_method_plans(domain, method_library, plan_records))
        return "\n".join(lines).strip() + "\n"

    def _render_header(
        self,
        domain: Any,
        objects: Sequence[str],
        plan_records: Sequence[Dict[str, Any]],
    ) -> List[str]:
        lines = [
            "/* Initial Beliefs */",
            f"domain({domain.name}).",
        ]

        for obj in objects:
            lines.append(f"{self._call('object', (obj,))}.")
            type_name = self._object_type_map.get(str(obj))
            for closure_type in self._type_closure(type_name):
                lines.append(
                    f"{self._call('object_type', (obj, self._type_atom(closure_type)))}."
                )

        lines.append("")
        return lines

    def _render_structured_plan_library(
        self,
        plan_library: PlanLibrary,
    ) -> List[str]:
        lines = ["/* HTN Method Plans */"]
        for plan in plan_library.plans:
            if not str(getattr(plan, "plan_name", "") or "").strip():
                continue
            lines.extend(self._render_structured_plan(plan))
            lines.append("")
        return lines

    def _render_structured_plan(self, plan: AgentSpeakPlan) -> List[str]:
        trigger = self._task_call(
            plan.trigger.symbol,
            tuple(self._strip_type_annotation(argument) for argument in plan.trigger.arguments),
        )
        context_literals = [
            self._render_structured_context_literal(literal)
            for literal in plan.context
            if str(literal or "").strip()
        ]
        context = " & ".join(context_literals) if context_literals else "true"
        body = [self._render_structured_body_step(step) for step in plan.body]
        if not body:
            body = ["true"]
        lines = [f"+!{trigger} : {context} <-"]
        lines.extend(self._indent_body(body))
        return lines

    def _render_structured_body_step(self, step: AgentSpeakBodyStep) -> str:
        call = self._task_call(step.symbol, step.arguments)
        if step.kind == "subgoal":
            return f"!{call}"
        return call

    def _render_structured_context_literal(self, literal_text: str) -> str:
        text = str(literal_text or "").strip()
        if not text:
            return "true"
        if text.lower().startswith("not "):
            rendered = self._render_structured_atom(text[4:].strip())
            return f"not {rendered}"
        if text.startswith("!"):
            rendered = self._render_structured_atom(text[1:].strip())
            return f"not {rendered}"
        if "!=" in text:
            left, right = text.split("!=", 1)
            return f"{self._asl_term(left.strip())} \\== {self._asl_term(right.strip())}"
        if "==" in text:
            left, right = text.split("==", 1)
            return f"{self._asl_term(left.strip())} == {self._asl_term(right.strip())}"
        return self._render_structured_atom(text)

    def _render_structured_atom(self, atom_text: str) -> str:
        text = str(atom_text or "").strip()
        if not text:
            return "true"
        if "(" not in text or not text.endswith(")"):
            return self._sanitize_name(text)
        functor, raw_args = text.split("(", 1)
        args = [
            value.strip()
            for value in raw_args[:-1].split(",")
            if value.strip()
        ]
        if self._sanitize_name(functor.strip()) == "object_type" and len(args) == 2:
            return (
                f"object_type({self._asl_term(args[0])}, "
                f"{self._type_atom(args[1])})"
            )
        return self._call(functor.strip(), tuple(args))

    @staticmethod
    def _strip_type_annotation(argument: str) -> str:
        text = str(argument or "").strip()
        if ":" in text:
            return text.split(":", 1)[0].strip()
        return text

    def _render_primitive_wrappers(
        self,
        domain: Any,
    ) -> List[str]:
        lines = ["/* Primitive Action Plans */"]

        for action in domain.actions:
            semantics = self.parser.parse_action(action)
            task_name = self._sanitize_name(action.name)
            parameter_types = tuple(
                self._parameter_type(parameter)
                for parameter in action.parameters
            )
            type_counts: Dict[str, int] = defaultdict(int)
            for type_name in parameter_types:
                type_counts[type_name] += 1
            args = self._canonical_param_names(parameter_types, dict(type_counts))
            trigger = self._call(task_name, args)
            context = self._context_from_precondition_clauses(
                tuple(semantics.precondition_clauses),
                semantics.parameters,
                args,
                prefix_literals=self._type_guard_literals(args, parameter_types),
            )
            body_lines = [self._call(task_name, args)]

            bindings = {
                parameter: value
                for parameter, value in zip(semantics.parameters, args)
            }
            for effect in semantics.effects:
                bound_args = tuple(bindings.get(item, item) for item in effect.args)
                effect_literal = effect
                atom = self._call(effect_literal.predicate, bound_args)
                if effect_literal.is_positive:
                    body_lines.append(f"+{atom}")
                else:
                    body_lines.append(f"-{atom}")

            lines.append(f"+!{trigger} : {context} <-")
            lines.extend(self._indent_body(body_lines))
            lines.append("")

        return lines

    def _render_method_plans(
        self,
        domain: Any,
        method_library: HTNMethodLibrary,
        plan_records: Sequence[Dict[str, Any]],
    ) -> List[str]:
        lines = ["/* HTN Method Plans */"]
        task_lookup = {
            task.name: task
            for task in method_library.compound_tasks
        }
        methods_by_task: Dict[str, List[HTNMethod]] = defaultdict(list)
        for method in method_library.methods:
            methods_by_task[method.task_name].append(method)
        render_specs = self._build_task_render_specs(domain, method_library)
        effect_cache: Dict[str, Tuple[Any, ...]] = {}
        effect_predicate_cache: Dict[str, Tuple[str, ...]] = {}
        self._populate_effect_predicate_cache(
            methods_by_task,
            render_specs,
            effect_predicate_cache,
        )

        for method in self._ordered_methods_for_rendering(
            method_library.methods,
            task_lookup,
            methods_by_task,
            render_specs,
            effect_cache,
            effect_predicate_cache,
        ):
            lines.extend(
                self._render_method_plan(
                    method,
                    task_lookup,
                    methods_by_task,
                    render_specs,
                    effect_cache,
                    effect_predicate_cache,
                ),
            )
            lines.append("")

        return lines

    def _render_exact_grounded_method_plans(
        self,
        *,
        method_library: HTNMethodLibrary,
        plan_records: Sequence[Dict[str, Any]],
        task_lookup: Dict[str, Any],
        render_specs: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        if not plan_records:
            return []

        method_lookup = {
            str(method.method_name).strip(): method
            for method in method_library.methods
            if str(method.method_name).strip()
        }
        methods_by_task: Dict[str, List[HTNMethod]] = defaultdict(list)
        for method in method_library.methods:
            methods_by_task[method.task_name].append(method)
        rendered_chunks: List[str] = []
        seen_chunks: set[str] = set()
        for record in plan_records:
            plan = record.get("plan")
            if not isinstance(plan, PANDAPlanResult):
                continue
            actual_plan_text = str(plan.actual_plan or plan.raw_plan or "").strip()
            if not actual_plan_text:
                continue
            for node in self._ordered_actual_plan_method_nodes(actual_plan_text):
                method_name = str(node.get("method_name") or "").strip()
                method = method_lookup.get(method_name)
                if method is None:
                    continue
                chunk = self._render_exact_grounded_method_chunk(
                    method=method,
                    method_node=node,
                    actual_plan_nodes=self._parse_actual_plan_nodes(actual_plan_text),
                    methods_by_task=methods_by_task,
                    task_lookup=task_lookup,
                    render_specs=render_specs,
                )
                if not chunk or chunk in seen_chunks:
                    continue
                seen_chunks.add(chunk)
                rendered_chunks.append(chunk)
        return rendered_chunks

    def _render_exact_grounded_method_chunk(
        self,
        *,
        method: HTNMethod,
        method_node: Dict[str, Any],
        actual_plan_nodes: Dict[int, Dict[str, Any]],
        methods_by_task: Dict[str, List[HTNMethod]],
        task_lookup: Dict[str, Any],
        render_specs: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        task_name = str(method_node.get("task_name") or "").strip()
        task_args = tuple(str(arg).strip() for arg in (method_node.get("task_args") or ()))
        ordered_children = self._ordered_actual_plan_child_nodes(
            method_node,
            actual_plan_nodes,
        )
        binding = self._exact_method_binding(
            method=method,
            task_args=task_args,
            ordered_children=ordered_children,
        )
        if binding is None or self._binding_has_synthetic_witness_tokens(binding):
            return None

        trigger = self._task_call(task_name, task_args)
        render_spec = render_specs.get(method.task_name, {})
        grounded_context_literals = self._ground_method_context_literals(
            method,
            binding,
        )
        grounded_context_literals = self._merge_context_literals(
            grounded_context_literals,
            self._type_guard_literals(
                task_args,
                tuple(render_spec.get("task_param_types", ())),
            ),
        )
        ordered_context_literals = self._order_context_literals_for_jason(
            grounded_context_literals,
            initially_bound=(),
        )
        context = self._literal_clause(ordered_context_literals, {})
        body_lines = [
            self._render_method_trace_statement(method, task_args),
        ]
        for child_node in ordered_children:
            child_task_name = str(child_node.get("task_name") or "").strip()
            if (
                str(child_node.get("kind") or "") == "compound"
                and self._compound_task_can_reach_task(
                    child_task_name,
                    method.task_name,
                    methods_by_task,
                )
            ):
                body_lines.append(
                    self._call(
                        "pipeline.no_ancestor_goal",
                        (
                            self._sanitize_name(child_task_name),
                            *tuple(child_node.get("task_args") or ()),
                        ),
                    ),
                )
            body_lines.append(
                f"!{self._task_call(str(child_node.get('task_name') or '').strip(), tuple(child_node.get('task_args') or ()))}"
            )
        if len(body_lines) == 1:
            body_lines.append("true")
        return "\n".join(
            [
                f"+!{trigger} : {context} <-",
                *self._indent_body(body_lines),
            ],
        )

    def _parse_actual_plan_nodes(
        self,
        plan_text: str,
    ) -> Dict[int, Dict[str, Any]]:
        nodes: Dict[int, Dict[str, Any]] = {}
        for raw_line in str(plan_text or "").splitlines():
            line = raw_line.strip()
            if not line or line == "==>" or line.startswith("root "):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                node_id = int(parts[0])
            except ValueError:
                continue
            if "->" in parts:
                arrow_index = parts.index("->")
                if arrow_index < 2 or arrow_index + 1 >= len(parts):
                    continue
                nodes[node_id] = {
                    "node_id": node_id,
                    "kind": "compound",
                    "task_name": parts[1],
                    "task_args": tuple(parts[2:arrow_index]),
                    "method_name": parts[arrow_index + 1],
                    "children": tuple(
                        child_id
                        for child_id in self._parse_plan_node_ids(parts[arrow_index + 2:])
                    ),
                }
                continue
            nodes[node_id] = {
                "node_id": node_id,
                "kind": "primitive",
                "task_name": parts[1],
                "task_args": tuple(parts[2:]),
                "method_name": None,
                "children": (),
            }
        return nodes

    def _ordered_actual_plan_method_nodes(
        self,
        plan_text: str,
    ) -> Tuple[Dict[str, Any], ...]:
        nodes = self._parse_actual_plan_nodes(plan_text)
        if not nodes:
            return ()

        root_ids: List[int] = []
        for raw_line in str(plan_text or "").splitlines():
            line = raw_line.strip()
            if not line.startswith("root "):
                continue
            root_ids.extend(self._parse_plan_node_ids(line.split()[1:]))

        first_primitive_cache: Dict[int, int] = {}

        def first_primitive_id(node_id: int) -> int:
            cached = first_primitive_cache.get(node_id)
            if cached is not None:
                return cached
            node = nodes.get(node_id)
            if node is None:
                first_primitive_cache[node_id] = node_id
                return node_id
            if str(node.get("kind") or "") == "primitive":
                first_primitive_cache[node_id] = node_id
                return node_id
            child_ids = tuple(node.get("children") or ())
            if not child_ids:
                first_primitive_cache[node_id] = node_id
                return node_id
            first_primitive_cache[node_id] = min(
                first_primitive_id(child_id)
                for child_id in child_ids
            )
            return first_primitive_cache[node_id]

        ordered_nodes: List[Dict[str, Any]] = []
        visited: set[int] = set()

        def visit(node_id: int) -> None:
            if node_id in visited:
                return
            node = nodes.get(node_id)
            if node is None or str(node.get("kind") or "") != "compound":
                return
            visited.add(node_id)
            ordered_nodes.append(node)
            for child_id in sorted(
                tuple(node.get("children") or ()),
                key=lambda child_id: (first_primitive_id(child_id), child_id),
            ):
                visit(child_id)

        for node_id in sorted(root_ids, key=lambda node_id: (first_primitive_id(node_id), node_id)):
            visit(node_id)
        for node_id in sorted(nodes, key=lambda node_id: (first_primitive_id(node_id), node_id)):
            visit(node_id)
        return tuple(ordered_nodes)

    def _ordered_actual_plan_child_nodes(
        self,
        method_node: Dict[str, Any],
        actual_plan_nodes: Dict[int, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], ...]:
        first_primitive_cache: Dict[int, int] = {}

        def first_primitive_id(node_id: int) -> int:
            cached = first_primitive_cache.get(node_id)
            if cached is not None:
                return cached
            node = actual_plan_nodes.get(node_id)
            if node is None:
                first_primitive_cache[node_id] = node_id
                return node_id
            if str(node.get("kind") or "") == "primitive":
                first_primitive_cache[node_id] = node_id
                return node_id
            child_ids = tuple(node.get("children") or ())
            if not child_ids:
                first_primitive_cache[node_id] = node_id
                return node_id
            first_primitive_cache[node_id] = min(
                first_primitive_id(child_id)
                for child_id in child_ids
            )
            return first_primitive_cache[node_id]

        child_ids = sorted(
            tuple(method_node.get("children") or ()),
            key=lambda child_id: (first_primitive_id(child_id), child_id),
        )
        return tuple(
            node
            for child_id in child_ids
            if (node := actual_plan_nodes.get(child_id)) is not None
        )

    def _exact_method_binding(
        self,
        *,
        method: HTNMethod,
        task_args: Sequence[str],
        ordered_children: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, str]]:
        binding: Dict[str, str] = {}
        if not self._extend_exact_binding(
            binding,
            self._task_binding_parameters(method, len(task_args)),
            task_args,
        ):
            return None

        ordered_steps = [
            step
            for step in self._ordered_method_steps(method)
            if getattr(step, "kind", None) in {"primitive", "compound"}
        ]
        if len(ordered_steps) != len(ordered_children):
            return None

        for step, child_node in zip(ordered_steps, ordered_children):
            if not self._actual_child_matches_method_step(step, child_node):
                return None
            if not self._extend_exact_binding(
                binding,
                tuple(getattr(step, "args", ()) or ()),
                tuple(child_node.get("task_args") or ()),
            ):
                return None
        return binding

    def _actual_child_matches_method_step(
        self,
        step: Any,
        child_node: Dict[str, Any],
    ) -> bool:
        child_name = self._sanitize_name(str(child_node.get("task_name") or "").strip())
        if getattr(step, "kind", None) == "primitive":
            candidate_names = {
                self._sanitize_name(str(getattr(step, "task_name", "") or "").strip()),
                self._sanitize_name(str(getattr(step, "action_name", "") or "").strip()),
            }
            return child_name in candidate_names
        if getattr(step, "kind", None) == "compound":
            return child_name == self._sanitize_name(str(getattr(step, "task_name", "") or "").strip())
        return False

    def _extend_exact_binding(
        self,
        binding: Dict[str, str],
        symbols: Sequence[str],
        grounded_args: Sequence[str],
    ) -> bool:
        if len(symbols) != len(grounded_args):
            return False
        for symbol, grounded_arg in zip(symbols, grounded_args):
            symbol_text = str(symbol).strip()
            grounded_text = str(grounded_arg).strip()
            if not symbol_text or not grounded_text:
                return False
            if self._looks_like_variable(symbol_text):
                existing = binding.get(symbol_text)
                if existing is None:
                    binding[symbol_text] = grounded_text
                    continue
                if existing != grounded_text:
                    return False
                continue
            if self._sanitize_name(symbol_text) != self._sanitize_name(grounded_text):
                return False
        return True

    def _ground_method_context_literals(
        self,
        method: HTNMethod,
        binding: Dict[str, str],
    ) -> Tuple[Any, ...]:
        grounded_literals: List[Any] = []
        seen_signatures: set[Tuple[str, bool, Tuple[str, ...]]] = set()
        for literal in tuple(getattr(method, "context", ()) or ()):
            grounded_literal = self._ground_literal_with_exact_binding(
                literal,
                binding,
            )
            if grounded_literal is None:
                continue
            signature = self._literal_structural_key(grounded_literal)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            grounded_literals.append(grounded_literal)
        return tuple(grounded_literals)

    def _ground_literal_with_exact_binding(
        self,
        literal: Any,
        binding: Dict[str, str],
    ) -> Optional[Any]:
        grounded_args: List[str] = []
        for arg in tuple(getattr(literal, "args", ()) or ()):
            arg_text = str(arg).strip()
            if self._looks_like_variable(arg_text):
                grounded_value = binding.get(arg_text)
                if grounded_value is None:
                    return None
                grounded_args.append(grounded_value)
                continue
            grounded_args.append(arg_text)
        return type(
            "LiteralProxy",
            (),
            {
                "predicate": str(getattr(literal, "predicate", "") or "").strip(),
                "args": tuple(grounded_args),
                "is_positive": bool(getattr(literal, "is_positive", True)),
            },
        )()

    @staticmethod
    def _binding_has_synthetic_witness_tokens(binding: Dict[str, str]) -> bool:
        return any(
            AgentSpeakRenderer._is_synthetic_witness_token(value)
            for value in binding.values()
        )

    @staticmethod
    def _is_synthetic_witness_token(token: str) -> bool:
        value = str(token or "").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value.startswith("witness_") or value.startswith("witness-")

    @staticmethod
    def _parse_plan_node_ids(tokens: Sequence[str]) -> Tuple[int, ...]:
        node_ids: List[int] = []
        for token in tokens:
            try:
                node_ids.append(int(str(token).strip()))
            except ValueError:
                continue
        return tuple(node_ids)

    def _ordered_methods_for_rendering(
        self,
        methods: Sequence[HTNMethod],
        task_lookup: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        render_specs: Dict[str, Dict[str, Any]],
        effect_cache: Dict[str, Tuple[Any, ...]],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
    ) -> Tuple[HTNMethod, ...]:
        ordered: List[HTNMethod] = []
        seen_tasks: set[str] = set()

        for method in methods:
            task_name = method.task_name
            if task_name in seen_tasks:
                continue
            seen_tasks.add(task_name)
            ordered.extend(
                self._ordered_task_methods_for_rendering(
                    task_name,
                    methods_by_task.get(task_name, ()),
                    task_lookup,
                    methods_by_task,
                    render_specs,
                    effect_cache,
                    effect_predicate_cache,
                ),
            )

        return tuple(ordered)

    def _ordered_task_methods_for_rendering(
        self,
        task_name: str,
        methods: Sequence[HTNMethod],
        task_lookup: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        render_specs: Dict[str, Dict[str, Any]],
        effect_cache: Dict[str, Tuple[Any, ...]],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
    ) -> Tuple[HTNMethod, ...]:
        task_effect_signatures = self._task_render_priority_effect_signatures(
            task_name,
            task_lookup,
        )
        enumerated_methods = list(enumerate(methods))
        enumerated_methods.sort(
            key=lambda entry: self._method_render_priority(
                entry[1],
                entry[0],
                task_lookup,
                methods_by_task,
                render_specs,
                effect_cache,
                effect_predicate_cache,
                task_effect_signatures,
            ),
        )
        return tuple(method for _, method in enumerated_methods)

    def _task_render_priority_effect_signatures(
        self,
        task_name: str,
        task_lookup: Dict[str, Any],
    ) -> set[str]:
        task_schema = task_lookup.get(task_name)
        if task_schema is None:
            return set()

        headline_literal = getattr(task_schema, "headline_literal", None)
        if headline_literal is not None:
            return {self._literal_signature(headline_literal)}

        parameters = tuple(getattr(task_schema, "parameters", ()) or ())
        signatures: set[str] = set()
        for predicate_name in tuple(getattr(task_schema, "source_predicates", ()) or ()):
            if not str(predicate_name or "").strip():
                continue
            literal = type(
                "LiteralProxy",
                (),
                {
                    "predicate": str(predicate_name).strip(),
                    "args": parameters,
                    "is_positive": True,
                },
            )()
            signatures.add(self._literal_signature(literal))
        return signatures

    def _method_render_priority(
        self,
        method: HTNMethod,
        original_index: int,
        task_lookup: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        render_specs: Dict[str, Dict[str, Any]],
        effect_cache: Dict[str, Tuple[Any, ...]],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
        task_effect_signatures: set[str],
    ) -> Tuple[int, int, int, int]:
        ordered_steps = self._ordered_method_steps(method)
        directly_supports_effect = self._context_literals_support_task_effect(
            method,
            tuple(getattr(method, "context", ()) or ()),
            task_lookup,
            task_effect_signatures,
        )
        has_compound_subtasks = any(
            getattr(step, "kind", None) == "compound"
            for step in ordered_steps
        )
        is_recursive = any(
            getattr(step, "kind", None) == "compound"
            and getattr(step, "task_name", None) == method.task_name
            for step in ordered_steps
        )
        return (
            0 if directly_supports_effect else 1,
            0 if not has_compound_subtasks else 1,
            0 if not is_recursive else 1,
            original_index,
        )

    def _render_method_plan(
        self,
        method: HTNMethod,
        task_lookup: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        render_specs: Dict[str, Dict[str, Any]],
        effect_cache: Dict[str, Tuple[Any, ...]],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
    ) -> List[str]:
        task_schema = task_lookup.get(method.task_name)
        ordered_steps = self._ordered_method_steps(method)
        render_spec = render_specs.get(method.task_name, {})
        trigger_args = tuple(
            render_spec.get("trigger_args")
            or (task_schema.parameters if task_schema else method.parameters)
        )
        use_simple_variable_map = self._should_use_simple_variable_map(method, ordered_steps)
        if use_simple_variable_map:
            variable_map = self._simple_method_variable_map(method, trigger_args)
        else:
            variable_map = self._method_variable_map(
                method,
                ordered_steps,
                render_spec,
            )
        expand_render_context = self._should_expand_render_context(
            ordered_steps,
            methods_by_task,
        )
        if expand_render_context:
            context_literals = self._method_context_literals(
                method,
                ordered_steps,
                task_lookup,
                methods_by_task,
                render_spec,
                variable_map,
                effect_cache,
                effect_predicate_cache,
            )
        else:
            context_literals = tuple(getattr(method, "context", ()) or ())
        context_literals = self._merge_context_literals(
            context_literals,
            (
                ()
                if use_simple_variable_map
                else self._method_type_guard_literals(
                    method,
                    render_spec,
                    variable_map,
                )
            ),
        )
        context_literals = self._merge_context_literals(
            context_literals,
            self._self_recursive_progress_literals(
                method,
                ordered_steps,
            ),
        )
        specialisations = (
            self._first_compound_child_specialisations(
                method,
                ordered_steps,
                task_lookup,
                methods_by_task,
                render_specs,
                variable_map,
                effect_cache,
                effect_predicate_cache,
            )
            if expand_render_context
            else ((),)
        )
        trigger = self._task_call(method.task_name, trigger_args)
        method_trace_line = self._render_method_trace_statement(
            method,
            self._map_args(trigger_args, variable_map),
        )
        body = [method_trace_line]
        for step in ordered_steps:
            if self._step_needs_recursive_ancestor_guard(method, step, methods_by_task):
                body.append(self._recursive_ancestor_guard_call(step, variable_map))
            body.append(
                f"!{self._task_call(step.task_name, self._map_args(step.args, variable_map))}"
            )
        if len(body) == 1:
            body.append("true")

        lines: List[str] = []
        for index, extra_literals in enumerate(specialisations):
            merged_context_literals = self._merge_context_literals(
                context_literals,
                extra_literals,
            )
            ordered_context_literals = self._order_context_literals_for_jason(
                merged_context_literals,
                initially_bound=self._context_binding_parameters(
                    method,
                    trigger_args,
                ),
            )
            context = self._literal_clause(ordered_context_literals, variable_map)
            if index > 0:
                lines.append("")
            lines.append(f"+!{trigger} : {context} <-")
            lines.extend(self._indent_body(body))
        return lines

    def _step_needs_recursive_ancestor_guard(
        self,
        method: HTNMethod,
        step: Any,
        methods_by_task: Dict[str, List[HTNMethod]],
    ) -> bool:
        if getattr(step, "kind", None) != "compound":
            return False
        return self._compound_task_can_reach_task(
            str(getattr(step, "task_name", "") or "").strip(),
            method.task_name,
            methods_by_task,
        )

    def _compound_task_can_reach_task(
        self,
        start_task: str,
        target_task: str,
        methods_by_task: Dict[str, List[HTNMethod]],
        seen_tasks: Optional[set[str]] = None,
    ) -> bool:
        start_task = str(start_task or "").strip()
        target_task = str(target_task or "").strip()
        if not start_task or not target_task:
            return False
        if start_task == target_task:
            return True
        seen = seen_tasks if seen_tasks is not None else set()
        if start_task in seen:
            return False
        seen.add(start_task)
        for candidate_method in methods_by_task.get(start_task, ()):
            for candidate_step in candidate_method.subtasks:
                if getattr(candidate_step, "kind", None) != "compound":
                    continue
                child_task = str(getattr(candidate_step, "task_name", "") or "").strip()
                if self._compound_task_can_reach_task(
                    child_task,
                    target_task,
                    methods_by_task,
                    seen,
                ):
                    return True
        return False

    def _recursive_ancestor_guard_call(
        self,
        step: Any,
        variable_map: Dict[str, str],
    ) -> str:
        guard_args = (
            self._sanitize_name(str(getattr(step, "task_name", "") or "").strip()),
            *self._map_args(getattr(step, "args", ()) or (), variable_map),
        )
        return self._call("pipeline.no_ancestor_goal", guard_args)

    def _should_expand_render_context(
        self,
        ordered_steps: Sequence[Any],
        methods_by_task: Dict[str, List[HTNMethod]],
    ) -> bool:
        if any(getattr(step, "kind", None) == "primitive" for step in ordered_steps):
            return True
        for step in ordered_steps[1:]:
            if getattr(step, "kind", None) != "compound":
                continue
            if methods_by_task.get(str(getattr(step, "task_name", "") or "").strip()):
                return True
        return False

    def _should_use_simple_variable_map(
        self,
        method: HTNMethod,
        ordered_steps: Sequence[Any],
    ) -> bool:
        if len(ordered_steps) > self._VARIABLE_MAP_INFERENCE_STEP_LIMIT:
            return True
        return (
            len(tuple(getattr(method, "context", ()) or ()))
            > self._VARIABLE_MAP_INFERENCE_CONTEXT_LIMIT
        )

    def _simple_method_variable_map(
        self,
        method: HTNMethod,
        trigger_args: Sequence[str],
    ) -> Dict[str, str]:
        variable_map = {
            original: canonical
            for original, canonical in zip(
                self._context_binding_parameters(method, trigger_args),
                trigger_args,
            )
        }
        used_names = set(variable_map.values())
        for parameter in tuple(getattr(method, "parameters", ()) or ()):
            parameter_name = str(parameter).strip()
            if not parameter_name or parameter_name in variable_map:
                continue
            candidate = self._sanitize_name(parameter_name).upper() or "VAR"
            if not candidate[0].isalpha():
                candidate = f"VAR_{candidate}"
            suffix = 2
            while candidate in used_names:
                candidate = f"{self._sanitize_name(parameter_name).upper() or 'VAR'}_{suffix}"
                suffix += 1
            variable_map[parameter_name] = candidate
            used_names.add(candidate)
        return variable_map

    def _render_method_trace_statement(
        self,
        method: HTNMethod,
        trigger_args: Sequence[str],
    ) -> str:
        trace_args = [self._asl_string("runtime trace method flat "), self._asl_string(method.method_name)]
        for arg in trigger_args:
            trace_args.extend((self._asl_string("|"), self._asl_term(arg)))
        return f".print({', '.join(trace_args)})"

    def _ordered_method_steps(self, method: HTNMethod) -> List[Any]:
        if len(method.subtasks) <= 1 or not method.ordering:
            return list(method.subtasks)

        step_lookup = {
            step.step_id: step
            for step in method.subtasks
        }
        dependents: Dict[str, List[str]] = {
            step.step_id: []
            for step in method.subtasks
        }
        in_degree: Dict[str, int] = {
            step.step_id: 0
            for step in method.subtasks
        }

        for before, after in method.ordering:
            if before not in step_lookup or after not in step_lookup:
                return list(method.subtasks)
            dependents[before].append(after)
            in_degree[after] += 1

        ordered_steps: List[Any] = []
        ready = [
            step.step_id
            for step in method.subtasks
            if in_degree[step.step_id] == 0
        ]

        while ready:
            current_id = ready.pop(0)
            ordered_steps.append(step_lookup[current_id])
            for next_id in dependents[current_id]:
                in_degree[next_id] -= 1
                if in_degree[next_id] == 0:
                    ready.append(next_id)

        if len(ordered_steps) != len(method.subtasks):
            return list(method.subtasks)

        return ordered_steps

    def _method_context_literals(
        self,
        method: HTNMethod,
        ordered_steps: Sequence[Any],
        task_lookup: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        render_spec: Dict[str, Any],
        variable_map: Dict[str, str],
        effect_cache: Dict[str, Tuple[Any, ...]],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
        *,
        respect_prior_effects: bool = True,
    ) -> Tuple[Any, ...]:
        context_literals: List[Any] = list(method.context)
        seen_signatures = {
            self._literal_signature(literal)
            for literal in context_literals
        }
        task_param_types = tuple(render_spec.get("task_param_types", ()))
        predicate_types = render_spec.get("predicate_types", {})
        action_types = render_spec.get("action_types", {})
        default_type = render_spec.get("default_type", "OBJECT")
        method_types, _ = self._infer_method_variable_types(
            method,
            task_param_types,
            predicate_types,
            action_types,
            default_type,
            render_spec.get("task_render_specs", {}),
        )
        mutable_predicates = self._mutable_predicates(
            render_spec,
            methods_by_task,
            effect_predicate_cache,
        )
        bound_variables = set(method.parameters)
        bound_variables.update(self._bound_variables_from_literals(tuple(context_literals)))
        prior_effect_signatures: set[str] = set()
        prior_effect_literals: List[Any] = []

        first_step_id = ordered_steps[0].step_id if ordered_steps else None
        prior_compound_step_seen = False
        for step in ordered_steps:
            step_preconditions: List[Any] = []
            if not prior_compound_step_seen:
                step_preconditions.extend(self._step_precondition_literals(step, render_spec))
            else:
                step_preconditions.extend(
                    self._static_step_precondition_literals(
                        step,
                        render_spec,
                        mutable_predicates,
                    )
                )
            if (
                getattr(step, "kind", None) == "compound"
                and step.step_id != first_step_id
            ):
                step_preconditions.extend(
                    self._later_compound_step_binding_literals(
                        step,
                        task_lookup,
                        methods_by_task,
                        render_spec,
                        effect_cache,
                        effect_predicate_cache,
                    )
                )
            for literal in step_preconditions:
                if not literal.is_positive:
                    continue
                if (
                    respect_prior_effects
                    and self._literal_signature(literal) in prior_effect_signatures
                ):
                    continue
                if respect_prior_effects and any(
                    self._literals_may_conflict(
                        literal,
                        effect,
                        method_types,
                        predicate_types,
                    )
                    for effect in prior_effect_literals
                ):
                    continue

                literal_variables = [
                    arg
                    for arg in self._literal_variables(literal)
                    if self._looks_like_variable(arg)
                ]
                if not literal_variables:
                    continue

                references_task_scope = any(
                    arg in method.parameters or arg in bound_variables
                    for arg in literal.args
                )
                if not references_task_scope:
                    continue

                literal_signature = self._literal_signature(literal)
                if literal_signature not in seen_signatures:
                    context_literals.append(literal)
                    seen_signatures.add(literal_signature)
                bound_variables.update(
                    arg
                    for arg in literal_variables
                    if arg in variable_map
                )

            for effect in self._step_effect_literals(
                step,
                task_lookup,
                methods_by_task,
                render_spec,
                effect_cache,
                effect_predicate_cache,
                stack=(method.task_name,),
            ):
                prior_effect_signatures.add(self._literal_signature(effect))
                prior_effect_literals.append(effect)
            if getattr(step, "kind", None) == "compound":
                prior_compound_step_seen = True

        return tuple(context_literals)

    def _mutable_predicates(
        self,
        render_spec: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
    ) -> set[str]:
        mutable: set[str] = set()
        for action_schema in render_spec.get("action_schemas", {}).values():
            for effect in tuple(getattr(action_schema, "effects", ()) or ()):
                predicate_name = str(getattr(effect, "predicate", "") or "").strip()
                if predicate_name and predicate_name != "=":
                    mutable.add(predicate_name)

        self._populate_effect_predicate_cache(
            methods_by_task,
            render_spec.get("task_render_specs", {}),
            effect_predicate_cache,
        )
        for predicate_names in effect_predicate_cache.values():
            for predicate_name in tuple(predicate_names or ()):
                rendered = str(predicate_name or "").strip()
                if rendered and rendered != "=":
                    mutable.add(rendered)
        return mutable

    def _static_step_precondition_literals(
        self,
        step: Any,
        render_spec: Dict[str, Any],
        mutable_predicates: set[str],
    ) -> Tuple[Any, ...]:
        static_literals: List[Any] = []
        for literal in self._step_precondition_literals(step, render_spec):
            predicate_name = str(getattr(literal, "predicate", "") or "").strip()
            if not predicate_name or predicate_name in mutable_predicates:
                continue
            static_literals.append(literal)
        return tuple(static_literals)

    def _literals_may_conflict(
        self,
        literal: Any,
        effect: Any,
        variable_types: Dict[str, str],
        predicate_types: Dict[str, Tuple[str, ...]],
    ) -> bool:
        if getattr(literal, "predicate", None) != getattr(effect, "predicate", None):
            return False
        literal_args = tuple(getattr(literal, "args", ()) or ())
        effect_args = tuple(getattr(effect, "args", ()) or ())
        if len(literal_args) != len(effect_args):
            return False

        signature = predicate_types.get(getattr(literal, "predicate", None), ())
        for index, (left_arg, right_arg) in enumerate(zip(literal_args, effect_args)):
            if left_arg == right_arg:
                continue
            left_type = self._symbol_type_hint(
                left_arg,
                variable_types,
                signature,
                index,
            )
            right_type = self._symbol_type_hint(
                right_arg,
                variable_types,
                signature,
                index,
            )
            if not self._types_may_overlap(left_type, right_type):
                return False
            if (
                not self._looks_like_variable(left_arg)
                and not self._looks_like_variable(right_arg)
                and left_arg != right_arg
            ):
                return False
        return True

    def _symbol_type_hint(
        self,
        symbol: str,
        variable_types: Dict[str, str],
        signature: Sequence[str],
        index: int,
    ) -> Optional[str]:
        if self._looks_like_variable(symbol):
            return variable_types.get(symbol)
        if symbol in self._object_type_map:
            return str(self._object_type_map[symbol]).upper()
        if index < len(signature):
            return signature[index]
        return None

    @staticmethod
    def _types_may_overlap(left_type: Optional[str], right_type: Optional[str]) -> bool:
        if not left_type or not right_type:
            return True
        left = str(left_type).upper()
        right = str(right_type).upper()
        if left == right:
            return True
        return left == "OBJECT" or right == "OBJECT"

    def _merge_context_literals(
        self,
        base_literals: Sequence[Any],
        extra_literals: Sequence[Any],
    ) -> Tuple[Any, ...]:
        merged: List[Any] = list(base_literals)
        seen_signatures = {
            self._literal_structural_key(literal)
            for literal in merged
        }
        for literal in extra_literals:
            signature = self._literal_structural_key(literal)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            merged.append(literal)
        return tuple(merged)

    @staticmethod
    def _literal_structural_key(literal: Any) -> Tuple[str, bool, Tuple[str, ...]]:
        return (
            str(getattr(literal, "predicate", "") or ""),
            bool(getattr(literal, "is_positive", True)),
            tuple(str(arg) for arg in (getattr(literal, "args", ()) or ())),
        )

    def _later_compound_step_binding_literals(
        self,
        step: Any,
        task_lookup: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        render_spec: Dict[str, Any],
        effect_cache: Dict[str, Tuple[Any, ...]],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
    ) -> Tuple[Any, ...]:
        child_methods = tuple(methods_by_task.get(step.task_name, ()))
        if len(child_methods) != 1:
            return ()
        child_method = child_methods[0]
        if any(getattr(child_step, "kind", None) == "compound" for child_step in child_method.subtasks):
            return ()

        child_render_spec = render_spec.get("task_render_specs", {}).get(step.task_name, {})
        child_ordered_steps = self._ordered_method_steps(child_method)
        child_variable_map = self._method_variable_map(
            child_method,
            child_ordered_steps,
            child_render_spec,
        )
        child_literals = self._method_context_literals(
            child_method,
            child_ordered_steps,
            task_lookup,
            methods_by_task,
            child_render_spec,
            child_variable_map,
            effect_cache,
            effect_predicate_cache,
            respect_prior_effects=True,
        )

        translated = self._translate_child_context_literals(
            child_method,
            step,
            child_literals,
            child_render_spec,
            {},
        )
        return tuple(
            literal
            for literal in translated
            if getattr(literal, "is_positive", True)
        )

    def _first_compound_child_specialisations(
        self,
        method: HTNMethod,
        ordered_steps: Sequence[Any],
        task_lookup: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        render_specs: Dict[str, Dict[str, Any]],
        variable_map: Dict[str, str],
        effect_cache: Dict[str, Tuple[Any, ...]],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
        *,
        specialisation_stack: Tuple[str, ...] = (),
    ) -> Tuple[Tuple[Any, ...], ...]:
        if not any(getattr(step, "kind", None) == "primitive" for step in ordered_steps):
            return ((),)

        method_name = str(getattr(method, "method_name", "") or "")
        if method_name and method_name in specialisation_stack:
            return ((),)

        first_compound_index = self._first_compound_step_index(ordered_steps)

        if first_compound_index is None:
            return ((),)

        prior_effect_signatures: set[str] = set()
        variant_records: List[Tuple[int, Tuple[Any, ...]]] = []
        seen_variants: set[Tuple[str, ...]] = set()

        preserve_generic_variant = False
        for step_index, step in enumerate(ordered_steps):
            if step_index == first_compound_index and getattr(step, "kind", None) == "compound":
                child_methods = tuple(methods_by_task.get(step.task_name, ()))
                has_recursive_child_method = any(
                    self._method_is_recursive(child_method)
                    for child_method in child_methods
                )
                if has_recursive_child_method:
                    preserve_generic_variant = True
                if child_methods:
                    needs_specialisation = True
                    if needs_specialisation:
                        for child_method in child_methods:
                            child_method_is_recursive = self._method_is_recursive(child_method)
                            if child_method_is_recursive and child_method.task_name == method.task_name:
                                continue
                            child_render_spec = render_specs.get(step.task_name, {})
                            child_ordered_steps = self._ordered_method_steps(child_method)
                            child_variable_map = self._method_variable_map(
                                child_method,
                                child_ordered_steps,
                                child_render_spec,
                            )
                            child_context_literals = self._method_context_literals(
                                child_method,
                                child_ordered_steps,
                                task_lookup,
                                methods_by_task,
                                child_render_spec,
                                child_variable_map,
                                effect_cache,
                                effect_predicate_cache,
                                respect_prior_effects=True,
                            )
                            child_specialisations = ((),)
                            if not child_method_is_recursive:
                                child_first_compound_index = self._first_compound_step_index(
                                    child_ordered_steps,
                                )
                                if child_first_compound_index != 0:
                                    child_first_compound_index = None
                            else:
                                child_first_compound_index = None
                            if child_first_compound_index == 0:
                                child_specialisations = self._first_compound_child_specialisations(
                                    child_method,
                                    child_ordered_steps,
                                    task_lookup,
                                    methods_by_task,
                                    render_specs,
                                    child_variable_map,
                                    effect_cache,
                                    effect_predicate_cache,
                                    specialisation_stack=specialisation_stack + ((method_name,) if method_name else ()),
                                )
                            for child_extra_literals in child_specialisations:
                                child_variant_literals = self._merge_context_literals(
                                    child_context_literals,
                                    child_extra_literals,
                                )
                                translated = self._translate_child_context_literals(
                                    child_method,
                                    step,
                                    child_variant_literals,
                                    child_render_spec,
                                    variable_map,
                                )
                                translated = tuple(
                                    literal
                                    for literal in translated
                                    if self._literal_signature(literal) not in prior_effect_signatures
                                )
                                signature = tuple(
                                    self._literal_signature(literal)
                                    for literal in translated
                                )
                                if signature in seen_variants:
                                    continue
                                seen_variants.add(signature)
                                variant_records.append((step_index, translated))

            for effect in self._step_effect_literals(
                step,
                task_lookup,
                methods_by_task,
                render_specs.get(method.task_name, {}),
                effect_cache,
                effect_predicate_cache,
                stack=(method.task_name,),
            ):
                prior_effect_signatures.add(self._literal_signature(effect))

        if preserve_generic_variant:
            base_signature: Tuple[str, ...] = ()
            if base_signature not in seen_variants:
                seen_variants.add(base_signature)
                variant_records.append((len(ordered_steps), ()))

        if not variant_records:
            return ((),)

        ordered_variants = [
            variant
            for _, variant in sorted(
                variant_records,
                key=lambda item: (
                    1 if len(item[1]) == 0 else 0,
                    -item[0],
                    -len(item[1]),
                ),
            )
        ]
        return tuple(ordered_variants)

    @staticmethod
    def _first_compound_step_index(ordered_steps: Sequence[Any]) -> Optional[int]:
        for step_index, step in enumerate(ordered_steps):
            if getattr(step, "kind", None) == "compound":
                return step_index
        return None

    def _method_is_recursive(self, method: HTNMethod) -> bool:
        return any(
            getattr(step, "kind", None) == "compound"
            and getattr(step, "task_name", None) == method.task_name
            for step in method.subtasks
        )

    def _self_recursive_progress_literals(
        self,
        method: HTNMethod,
        ordered_steps: Sequence[Any],
    ) -> Tuple[Any, ...]:
        if not self._method_is_recursive(method):
            return ()

        task_binding_parameters = self._task_binding_parameters(
            method,
            len(method.task_args),
        )
        for step in ordered_steps:
            if getattr(step, "kind", None) != "compound":
                continue
            if getattr(step, "task_name", None) != method.task_name:
                continue
            for child_arg, parent_arg in zip(step.args, task_binding_parameters):
                if child_arg == parent_arg:
                    continue
                return (
                    type(
                        "LiteralProxy",
                        (),
                        {
                            "predicate": "=",
                            "args": (child_arg, parent_arg),
                            "is_positive": False,
                        },
                    )(),
                )
        return ()

    def _context_literals_support_task_effect(
        self,
        method: HTNMethod,
        context_literals: Sequence[Any],
        task_lookup: Dict[str, Any],
        task_effect_signatures: set[str],
    ) -> bool:
        if not task_effect_signatures:
            return False

        task_schema = task_lookup.get(method.task_name)
        if task_schema is None:
            return False

        task_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
        task_binding_parameters = self._task_binding_parameters(
            method,
            len(task_parameters),
        )
        task_bindings = {
            parameter: task_parameter
            for parameter, task_parameter in zip(task_binding_parameters, task_parameters)
        }

        for literal in context_literals:
            if not getattr(literal, "is_positive", True):
                continue
            lifted_literal = self._lift_literal_to_task_scope(
                literal,
                task_bindings,
            )
            if lifted_literal is None:
                continue
            if self._literal_signature(lifted_literal) in task_effect_signatures:
                return True

        return False

    def _literal_supports_task_effect(
        self,
        literal: Any,
        method: HTNMethod,
        task_lookup: Dict[str, Any],
        task_effect_signatures: set[str],
    ) -> bool:
        if not task_effect_signatures or not getattr(literal, "is_positive", True):
            return False

        task_schema = task_lookup.get(method.task_name)
        if task_schema is None:
            return False

        task_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
        task_binding_parameters = self._task_binding_parameters(
            method,
            len(task_parameters),
        )
        task_bindings = {
            parameter: task_parameter
            for parameter, task_parameter in zip(task_binding_parameters, task_parameters)
        }
        lifted_literal = self._lift_literal_to_task_scope(
            literal,
            task_bindings,
        )
        if lifted_literal is None:
            return False
        if self._literal_signature(lifted_literal) in task_effect_signatures:
            return True

        source_predicates = {
            str(predicate_name).strip()
            for predicate_name in tuple(getattr(task_schema, "source_predicates", ()) or ())
            if str(predicate_name).strip()
        }
        if str(getattr(lifted_literal, "predicate", "") or "").strip() not in source_predicates:
            return False

        task_parameter_sequence = tuple(str(parameter).strip() for parameter in task_parameters)
        cursor = 0
        for argument in tuple(getattr(lifted_literal, "args", ()) or ()):
            while cursor < len(task_parameter_sequence) and task_parameter_sequence[cursor] != argument:
                cursor += 1
            if cursor >= len(task_parameter_sequence):
                return False
            cursor += 1
        return True

    def _lift_literal_to_task_scope(
        self,
        literal: Any,
        task_bindings: Dict[str, str],
    ) -> Optional[Any]:
        lifted_args: List[str] = []
        for arg in getattr(literal, "args", ()) or ():
            if arg in task_bindings:
                lifted_args.append(task_bindings[arg])
                continue
            if self._looks_like_variable(arg):
                return None
            lifted_args.append(arg)

        return type(
            "LiteralProxy",
            (),
            {
                "predicate": literal.predicate,
                "args": tuple(lifted_args),
                "is_positive": literal.is_positive,
            },
        )()

    def _compound_method_internal_parameters(
        self,
        method: HTNMethod,
        task_arity: int,
    ) -> Tuple[str, ...]:
        task_binding_parameters = self._task_binding_parameters(method, task_arity)
        return tuple(
            parameter
            for parameter in method.parameters
            if parameter not in task_binding_parameters
        )

    def _translate_child_context_literals(
        self,
        child_method: HTNMethod,
        child_step: Any,
        literals: Sequence[Any],
        child_render_spec: Dict[str, Any],
        parent_variable_map: Dict[str, str],
    ) -> Tuple[Any, ...]:
        task_binding_parameters = self._task_binding_parameters(
            child_method,
            len(child_step.args),
        )
        bindings = {
            parameter: arg
            for parameter, arg in zip(task_binding_parameters, child_step.args)
        }
        used_names = set(parent_variable_map.values())
        default_type = child_render_spec.get("default_type", "OBJECT")
        predicate_types = child_render_spec.get("predicate_types", {})
        action_types = child_render_spec.get("action_types", {})
        task_param_types = tuple(child_render_spec.get("task_param_types", ()))
        child_variable_types, _ = self._infer_method_variable_types(
            child_method,
            task_param_types,
            predicate_types,
            action_types,
            default_type,
            child_render_spec.get("task_render_specs", {}),
        )
        fresh_bindings: Dict[str, str] = {}
        translated_literals: List[Any] = []

        for literal in literals:
            translated_args: List[str] = []
            for arg in literal.args:
                if arg in bindings:
                    translated_args.append(bindings[arg])
                    continue
                if self._looks_like_variable(arg):
                    if arg not in fresh_bindings:
                        type_name = child_variable_types.get(arg, default_type)
                        fresh_bindings[arg] = self._fresh_canonical_variable_name(
                            type_name,
                            used_names,
                        )
                    translated_args.append(fresh_bindings[arg])
                    continue
                translated_args.append(arg)
            translated_literals.append(
                type(
                    "LiteralProxy",
                    (),
                    {
                        "predicate": literal.predicate,
                        "args": tuple(translated_args),
                        "is_positive": literal.is_positive,
                    },
                )(),
            )

        return tuple(translated_literals)

    def _fresh_canonical_variable_name(
        self,
        type_name: str,
        used_names: set[str],
    ) -> str:
        base_name = self._variable_stem(type_name)
        if base_name not in used_names:
            used_names.add(base_name)
            return base_name

        next_index = 2
        while f"{base_name}{next_index}" in used_names:
            next_index += 1
        candidate = f"{base_name}{next_index}"
        used_names.add(candidate)
        return candidate

    def _context_binding_parameters(
        self,
        method: HTNMethod,
        trigger_args: Sequence[str],
    ) -> Tuple[str, ...]:
        return self._task_binding_parameters(method, len(trigger_args))

    def _order_context_literals_for_jason(
        self,
        literals: Sequence[Any],
        *,
        initially_bound: Sequence[str],
    ) -> Tuple[Any, ...]:
        remaining = list(literals)
        ordered: List[Any] = []
        known_variables = {
            symbol
            for symbol in initially_bound
            if self._looks_like_variable(symbol)
        }

        while remaining:
            progress = False
            for index, literal in enumerate(remaining):
                if not self._literal_ready_for_context(literal, known_variables):
                    continue
                ordered.append(literal)
                known_variables.update(
                    arg
                    for arg in self._literal_variables(literal)
                    if self._looks_like_variable(arg)
                )
                remaining.pop(index)
                progress = True
                break
            if not progress:
                ordered.extend(remaining)
                break

        return tuple(ordered)

    def _literal_ready_for_context(
        self,
        literal: Any,
        known_variables: set[str],
    ) -> bool:
        variable_args = [
            arg
            for arg in self._literal_variables(literal)
            if self._looks_like_variable(arg)
        ]
        if not variable_args:
            return True

        if getattr(literal, "predicate", None) == "=" and len(getattr(literal, "args", ())) == 2:
            return any(
                (not self._looks_like_variable(arg)) or arg in known_variables
                for arg in literal.args
            )

        if getattr(literal, "is_positive", True):
            return any(
                (not self._looks_like_variable(arg)) or arg in known_variables
                for arg in literal.args
            )

        return all(arg in known_variables for arg in variable_args)

    def _literal_clause(self, literals: Iterable[Any], variable_map: Dict[str, str]) -> str:
        parts = [
            self._render_literal(literal, variable_map)
            for literal in literals
        ]
        return " & ".join(parts) if parts else "true"

    def _bound_variables_from_literals(self, literals: Tuple[Any, ...]) -> set[str]:
        known_variables: set[str] = set()
        changed = True

        while changed:
            changed = False
            for literal in literals:
                if getattr(literal, "predicate", None) == "=":
                    continue
                literal_variables = [
                    arg
                    for arg in self._literal_variables(literal)
                    if self._looks_like_variable(arg)
                ]
                if not literal_variables:
                    continue
                has_anchor = any(
                    (not self._looks_like_variable(arg)) or arg in known_variables
                    for arg in literal.args
                )
                if not has_anchor:
                    continue
                new_variables = [
                    arg
                    for arg in literal_variables
                    if arg not in known_variables
                ]
                if not new_variables:
                    continue
                known_variables.update(new_variables)
                changed = True

        return known_variables

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("-", "_")

    @staticmethod
    def _asl_string(value: str) -> str:
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _asl_atom_or_string(self, value: str) -> str:
        text = str(value or "").strip()
        if re.fullmatch(r"[a-z][a-zA-Z0-9_]*", text):
            return text
        return self._asl_string(text)

    def _asl_term(self, value: str) -> str:
        text = str(value or "").strip()
        cached = self._asl_term_cache.get(text)
        if cached is not None:
            return cached
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            rendered = text
        elif text in self._object_symbol_set:
            rendered = self._asl_atom_or_string(text)
        elif self._looks_like_variable(text):
            rendered = text
        else:
            rendered = self._asl_atom_or_string(text)
        self._asl_term_cache[text] = rendered
        return rendered

    def _build_task_render_specs(
        self,
        domain: Any,
        method_library: HTNMethodLibrary,
    ) -> Dict[str, Dict[str, Any]]:
        predicate_types = self._predicate_type_map(domain)
        action_types = self._action_type_map(domain)
        declared_task_types = self._declared_task_type_map(domain)
        action_schemas = self._action_schema_map(domain)
        default_type = (domain.types[0] if getattr(domain, "types", None) else "object").upper()
        methods_by_task: Dict[str, List[HTNMethod]] = defaultdict(list)
        for method in method_library.methods:
            methods_by_task[method.task_name].append(method)

        specs: Dict[str, Dict[str, Any]] = {}
        for task in method_library.compound_tasks:
            task_methods = methods_by_task.get(task.name, [])
            task_param_types = self._infer_task_parameter_types(
                task,
                task_methods,
                predicate_types,
                action_types,
                declared_task_types,
                default_type,
            )
            max_counts = self._max_method_type_counts(
                task_methods,
                task_param_types,
                predicate_types,
                action_types,
                default_type,
            )
            trigger_args = self._canonical_param_names(task_param_types, max_counts)
            task_param_type_counts: Dict[str, int] = defaultdict(int)
            for type_name in task_param_types:
                task_param_type_counts[type_name] += 1
            specs[task.name] = {
                "task_param_types": task_param_types,
                "trigger_args": trigger_args,
                "max_counts": dict(max_counts),
                "task_param_type_counts": dict(task_param_type_counts),
                "predicate_types": predicate_types,
                "action_types": action_types,
                "action_schemas": action_schemas,
                "default_type": default_type,
            }
        for task_name, spec in specs.items():
            spec["task_render_specs"] = specs
        return specs

    def _infer_task_parameter_types(
        self,
        task: Any,
        task_methods: Sequence[HTNMethod],
        predicate_types: Dict[str, Tuple[str, ...]],
        action_types: Dict[str, Tuple[str, ...]],
        declared_task_types: Dict[str, Tuple[str, ...]],
        default_type: str,
    ) -> Tuple[str, ...]:
        inferred: List[str | None] = [None] * len(task.parameters)

        declared_signature = declared_task_types.get(task.name)
        if declared_signature and len(declared_signature) == len(task.parameters):
            inferred = list(declared_signature)

        if len(getattr(task, "source_predicates", ())) == 1:
            predicate_name = task.source_predicates[0]
            predicate_signature = predicate_types.get(predicate_name)
            if (
                predicate_signature
                and len(predicate_signature) == len(task.parameters)
                and not declared_signature
            ):
                inferred = list(predicate_signature)

        for method in task_methods:
            candidate_map, _ = self._method_variable_type_candidate_map(
                method,
                predicate_types,
                action_types,
            )
            for index, parameter in enumerate(
                self._task_binding_parameters(method, len(task.parameters)),
            ):
                if inferred[index] is not None:
                    continue
                candidates = candidate_map.get(parameter, ())
                if candidates:
                    inferred[index] = candidates[0]

        return tuple(type_name or default_type for type_name in inferred)

    def _declared_task_type_map(self, domain: Any) -> Dict[str, Tuple[str, ...]]:
        mapping: Dict[str, Tuple[str, ...]] = {}
        for task in getattr(domain, "tasks", []) or ():
            mapping[str(task.name)] = tuple(
                self._parameter_type(parameter)
                for parameter in getattr(task, "parameters", ()) or ()
            )
        return mapping

    def _max_method_type_counts(
        self,
        task_methods: Sequence[HTNMethod],
        task_param_types: Tuple[str, ...],
        predicate_types: Dict[str, Tuple[str, ...]],
        action_types: Dict[str, Tuple[str, ...]],
        default_type: str,
    ) -> Dict[str, int]:
        max_counts: Dict[str, int] = defaultdict(int)
        for type_name in task_param_types:
            max_counts[type_name] += 1

        for method in task_methods:
            method_types, _ = self._infer_method_variable_types(
                method,
                task_param_types,
                predicate_types,
                action_types,
                default_type,
            )
            counts: Dict[str, int] = defaultdict(int)
            for type_name in method_types.values():
                counts[type_name] += 1
            for type_name, count in counts.items():
                max_counts[type_name] = max(max_counts[type_name], count)

        return max_counts

    def _method_variable_map(
        self,
        method: HTNMethod,
        ordered_steps: Sequence[Any],
        render_spec: Dict[str, Any],
    ) -> Dict[str, str]:
        task_param_types = tuple(render_spec.get("task_param_types", ()))
        trigger_args = tuple(render_spec.get("trigger_args", ()))
        default_type = render_spec.get("default_type", "OBJECT")
        predicate_types = render_spec.get("predicate_types", {})
        action_types = render_spec.get("action_types", {})
        method_types, appearance_order = self._infer_method_variable_types(
            method,
            task_param_types,
            predicate_types,
            action_types,
            default_type,
            render_spec.get("task_render_specs", {}),
        )

        variable_map: Dict[str, str] = {}
        task_binding_parameters = self._context_binding_parameters(method, trigger_args)
        for original, canonical in zip(task_binding_parameters, trigger_args):
            variable_map[original] = canonical

        used_counts: Dict[str, int] = defaultdict(int)
        task_param_type_counts = render_spec.get("task_param_type_counts", {})
        max_counts = render_spec.get("max_counts", {})
        for type_name, count in task_param_type_counts.items():
            used_counts[type_name] = count

        method_type_counts: Dict[str, int] = defaultdict(int)
        for type_name in method_types.values():
            method_type_counts[type_name] += 1

        for variable in appearance_order:
            if variable in variable_map:
                continue
            type_name = method_types.get(variable, default_type)
            variable_stem = self._variable_stem(type_name)
            next_index = used_counts[type_name] + 1
            overall_count = max(max_counts.get(type_name, 0), method_type_counts.get(type_name, 0))
            if overall_count > 1 or used_counts[type_name] > 0:
                variable_map[variable] = f"{variable_stem}{next_index}"
            else:
                variable_map[variable] = variable_stem
            used_counts[type_name] += 1

        return variable_map

    def _task_binding_parameters(
        self,
        method: HTNMethod,
        task_arity: int,
    ) -> Tuple[str, ...]:
        task_binding_parameters = tuple(method.task_args or ())
        if len(task_binding_parameters) != task_arity:
            task_binding_parameters = tuple(method.parameters[:task_arity])
        return task_binding_parameters

    def _infer_method_variable_types(
        self,
        method: HTNMethod,
        task_param_types: Tuple[str, ...],
        predicate_types: Dict[str, Tuple[str, ...]],
        action_types: Dict[str, Tuple[str, ...]],
        default_type: str,
        task_render_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, str], List[str]]:
        variable_types: Dict[str, str] = {}
        appearance_order: List[str] = []

        task_binding_parameters = self._task_binding_parameters(method, len(task_param_types))
        for parameter in method.parameters:
            if parameter not in appearance_order:
                appearance_order.append(parameter)
        for index, parameter in enumerate(task_binding_parameters):
            if index >= len(task_param_types):
                continue
            variable_types[parameter] = task_param_types[index]

        candidate_map, discovered_order = self._method_variable_type_candidate_map(
            method,
            predicate_types,
            action_types,
            task_render_specs,
        )
        for variable in discovered_order:
            if variable not in appearance_order:
                appearance_order.append(variable)
            candidates = candidate_map.get(variable, ())
            if variable in variable_types:
                variable_types[variable] = self._narrow_variable_type(
                    variable_types[variable],
                    candidates,
                )
                continue
            variable_types[variable] = candidates[0] if candidates else default_type

        for parameter in method.parameters:
            variable_types.setdefault(parameter, default_type)

        return variable_types, appearance_order

    def _narrow_variable_type(
        self,
        current_type: str,
        candidate_types: Sequence[str],
    ) -> str:
        normalized_current = str(current_type or "").strip().upper()
        normalized_candidates = [
            str(candidate or "").strip().upper()
            for candidate in candidate_types
            if str(candidate or "").strip()
        ]
        if not normalized_candidates:
            return normalized_current or current_type

        supported_types = []
        if normalized_current:
            supported_types.append(normalized_current)
        supported_types.extend(normalized_candidates)
        supported_types = list(dict.fromkeys(supported_types))

        best_type: Optional[str] = None
        best_depth = -1
        for candidate in supported_types:
            if all(self._is_type_compatible(candidate, required) for required in supported_types):
                candidate_depth = len(self._type_closure(candidate))
                if candidate_depth > best_depth:
                    best_type = candidate
                    best_depth = candidate_depth
        if best_type is not None:
            return best_type
        return normalized_current or normalized_candidates[0]

    def _is_type_compatible(
        self,
        candidate_type: str,
        required_type: str,
    ) -> bool:
        normalized_candidate = str(candidate_type or "").strip().lower()
        normalized_required = str(required_type or "").strip().lower()
        if not normalized_candidate or not normalized_required:
            return True
        if normalized_candidate == normalized_required:
            return True

        closure = {
            item.lower()
            for item in self._type_closure(normalized_candidate)
        }
        closure.add(normalized_candidate)
        closure.add("object")
        return normalized_required in closure

    def _variables_in_method(self, method: HTNMethod) -> List[str]:
        ordered_steps = self._ordered_method_steps(method)
        variables: List[str] = []

        def add(symbol: str) -> None:
            if self._looks_like_variable(symbol) and symbol not in variables:
                variables.append(symbol)

        for literal in method.context:
            for arg in literal.args:
                add(arg)
        for step in ordered_steps:
            for arg in step.args:
                add(arg)
            for literal in (step.literal, *step.preconditions, *step.effects):
                if literal is None:
                    continue
                for arg in literal.args:
                    add(arg)

        return variables

    def _variable_type_candidates(
        self,
        method: HTNMethod,
        variable: str,
        predicate_types: Dict[str, Tuple[str, ...]],
        action_types: Dict[str, Tuple[str, ...]],
        task_render_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        candidate_map, _ = self._method_variable_type_candidate_map(
            method,
            predicate_types,
            action_types,
            task_render_specs,
        )
        return list(candidate_map.get(variable, ()))

    def _method_variable_type_candidate_map(
        self,
        method: HTNMethod,
        predicate_types: Dict[str, Tuple[str, ...]],
        action_types: Dict[str, Tuple[str, ...]],
        task_render_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        candidates: Dict[str, List[str]] = defaultdict(list)
        appearance_order: List[str] = []

        def remember(symbol: str) -> None:
            if self._looks_like_variable(symbol) and symbol not in appearance_order:
                appearance_order.append(symbol)

        def collect_from_literal(literal: Any) -> None:
            signature = predicate_types.get(literal.predicate)
            for index, arg in enumerate(literal.args):
                remember(arg)
                if signature and index < len(signature):
                    candidates[arg].append(signature[index])

        for literal in method.context:
            collect_from_literal(literal)
        for step in self._ordered_method_steps(method):
            for arg in step.args:
                remember(arg)
            action_signature = action_types.get(step.action_name or "")
            if step.kind == "primitive" and not action_signature:
                action_signature = action_types.get(step.task_name)
            if action_signature:
                for index, arg in enumerate(step.args):
                    if index < len(action_signature):
                        candidates[arg].append(action_signature[index])
            elif step.kind == "compound":
                child_render_spec = (task_render_specs or {}).get(step.task_name, {})
                child_signature = tuple(child_render_spec.get("task_param_types", ()))
                for index, arg in enumerate(step.args):
                    if index < len(child_signature):
                        candidates[arg].append(child_signature[index])
            for literal in (step.literal, *step.preconditions, *step.effects):
                if literal is not None:
                    collect_from_literal(literal)

        return candidates, appearance_order

    def _canonical_param_names(
        self,
        task_param_types: Tuple[str, ...],
        max_counts: Dict[str, int],
    ) -> Tuple[str, ...]:
        per_type_seen: Dict[str, int] = defaultdict(int)
        names: List[str] = []
        for type_name in task_param_types:
            variable_stem = self._variable_stem(type_name)
            per_type_seen[type_name] += 1
            if max_counts.get(type_name, 0) > 1:
                names.append(f"{variable_stem}{per_type_seen[type_name]}")
            else:
                names.append(variable_stem)
        return tuple(names)

    def _variable_stem(self, type_name: str) -> str:
        stem = self._sanitize_name(str(type_name or "OBJECT")).upper()
        return stem or "OBJECT"

    def _type_atom(self, type_name: str) -> str:
        atom = self._sanitize_name(str(type_name or "object")).lower()
        return atom or "object"

    def _build_type_parent_map(self, domain: Any) -> Dict[str, Optional[str]]:
        tokens = [
            token.strip()
            for token in (getattr(domain, "types", []) or ())
            if token and token.strip()
        ]
        if not tokens:
            return {"object": None}

        parent_map: Dict[str, Optional[str]] = {}
        pending_children: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "-":
                if not pending_children or index + 1 >= len(tokens):
                    return {"object": None}
                parent_type = tokens[index + 1].strip()
                for child_type in pending_children:
                    parent_map[child_type] = parent_type or "object"
                pending_children = []
                index += 2
                continue

            pending_children.append(token)
            index += 1

        for child_type in pending_children:
            parent_map.setdefault(child_type, "object")

        parent_map["object"] = None
        changed = True
        while changed:
            changed = False
            for parent_type in list(parent_map.values()):
                if parent_type is None or parent_type in parent_map:
                    continue
                parent_map[parent_type] = "object" if parent_type != "object" else None
                changed = True

        return parent_map

    def _type_closure(self, type_name: Optional[str]) -> Tuple[str, ...]:
        if not type_name:
            return ()

        closure: List[str] = []
        visited: set[str] = set()
        cursor: Optional[str] = str(type_name).strip()
        while cursor and cursor not in visited:
            visited.add(cursor)
            if cursor != "object":
                closure.append(cursor)
            cursor = self._type_parent_map.get(cursor)
        return tuple(closure)

    def _predicate_type_map(self, domain: Any) -> Dict[str, Tuple[str, ...]]:
        mapping: Dict[str, Tuple[str, ...]] = {}
        for predicate in getattr(domain, "predicates", []):
            mapping[predicate.name] = tuple(
                self._parameter_type(parameter)
                for parameter in predicate.parameters
            )
        return mapping

    def _action_type_map(self, domain: Any) -> Dict[str, Tuple[str, ...]]:
        mapping: Dict[str, Tuple[str, ...]] = {}
        for action in getattr(domain, "actions", []):
            type_signature = tuple(
                self._parameter_type(parameter)
                for parameter in action.parameters
            )
            mapping[action.name] = type_signature
            mapping[self._sanitize_name(action.name)] = type_signature
        return mapping

    def _action_schema_map(self, domain: Any) -> Dict[str, Any]:
        mapping: Dict[str, Any] = {}
        for action in getattr(domain, "actions", []):
            semantics = self.parser.parse_action(action)
            mapping[action.name] = semantics
            mapping[self._sanitize_name(action.name)] = semantics
        return mapping

    def _parameter_type(self, parameter: str) -> str:
        if "-" not in parameter:
            return "OBJECT"
        return parameter.split("-", 1)[1].strip().upper()

    def _map_args(self, args: Sequence[str], variable_map: Dict[str, str]) -> Tuple[str, ...]:
        return tuple(self._asl_term(variable_map.get(arg, arg)) for arg in args)

    def _render_literal(self, literal: Any, variable_map: Dict[str, str]) -> str:
        mapped_args = self._map_args(literal.args, variable_map)
        if getattr(literal, "predicate", None) == "=" and len(mapped_args) == 2:
            operator = "==" if getattr(literal, "is_positive", True) else "\\=="
            return f"{mapped_args[0]} {operator} {mapped_args[1]}"
        predicate = self._call(literal.predicate, mapped_args)
        if literal.is_positive:
            return predicate
        return f"not {predicate}"

    def _step_precondition_literals(
        self,
        step: Any,
        render_spec: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        if getattr(step, "kind", None) != "primitive":
            return tuple(step.preconditions)
        action_schema = self._resolve_action_schema(step, render_spec)
        literals = list(step.preconditions)
        seen_signatures = {
            self._literal_signature(literal)
            for literal in literals
        }
        if action_schema is None:
            return tuple(literals)
        for literal in self._materialise_action_literals(
            action_schema.preconditions,
            action_schema.parameters,
            step.args,
        ):
            signature = self._literal_signature(literal)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            literals.append(literal)
        return tuple(literals)

    def _step_effect_literals(
        self,
        step: Any,
        task_lookup: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        render_spec: Dict[str, Any],
        effect_cache: Dict[str, Tuple[Any, ...]],
        effect_predicate_cache: Optional[Dict[str, Tuple[str, ...]]] = None,
        stack: Tuple[str, ...] = (),
    ) -> Tuple[Any, ...]:
        if getattr(step, "kind", None) == "primitive":
            literals = list(step.effects)
            seen_signatures = {
                self._literal_signature(literal)
                for literal in literals
            }
            action_schema = self._resolve_action_schema(step, render_spec)
            if action_schema is None:
                return tuple(literals)
            for literal in self._materialise_action_literals(
                action_schema.effects,
                action_schema.parameters,
                step.args,
            ):
                signature = self._literal_signature(literal)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                literals.append(literal)
            return tuple(literals)
        if step.effects:
            return tuple(step.effects)
        if getattr(step, "kind", None) != "compound":
            return ()
        task_schema = task_lookup.get(step.task_name)
        if not task_schema:
            return ()
        effect_templates = self._compound_task_effect_templates(
            step.task_name,
            task_lookup,
            methods_by_task,
            render_spec.get("task_render_specs", {}),
            effect_cache,
            stack,
        )
        if not effect_templates:
            return ()

        task_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
        bindings = {
            parameter: arg
            for parameter, arg in zip(task_parameters, step.args)
        }
        projected_literals: List[Any] = []
        seen_signatures: set[str] = set()
        for template in effect_templates:
            projected_args = tuple(bindings.get(arg, arg) for arg in template.args)
            literal = type(
                "LiteralProxy",
                (),
                {
                    "predicate": template.predicate,
                    "args": projected_args,
                    "is_positive": template.is_positive,
                },
            )()
            signature = self._literal_signature(literal)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            projected_literals.append(literal)
        concrete_effect_predicates = {
            str(getattr(literal, "predicate", "") or "").strip()
            for literal in projected_literals
            if str(getattr(literal, "predicate", "") or "").strip()
        }
        for literal in self._abstract_compound_effect_literals(
            step,
            render_spec,
            methods_by_task,
            effect_predicate_cache or {},
            stack,
        ):
            predicate_name = str(getattr(literal, "predicate", "") or "").strip()
            if predicate_name in concrete_effect_predicates:
                continue
            signature = self._literal_signature(literal)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            projected_literals.append(literal)
        return tuple(projected_literals)

    def _abstract_compound_effect_literals(
        self,
        step: Any,
        render_spec: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
        stack: Tuple[str, ...],
    ) -> Tuple[Any, ...]:
        task_name = str(getattr(step, "task_name", "") or "").strip()
        task_render_specs = render_spec.get("task_render_specs", {})
        task_render_spec = task_render_specs.get(task_name, {})
        task_types = tuple(task_render_spec.get("task_param_types", ()))
        predicate_types = render_spec.get("predicate_types", {})
        if not task_name or not task_types or not predicate_types:
            return ()

        effect_predicates = self._step_effect_predicates(
            step,
            methods_by_task,
            render_spec,
            effect_predicate_cache,
            stack,
        )
        if not effect_predicates:
            return ()

        used_names = {
            str(arg)
            for arg in getattr(step, "args", ()) or ()
            if self._looks_like_variable(arg)
        }
        literals: List[Any] = []
        for predicate_name in effect_predicates:
            signature = tuple(predicate_types.get(predicate_name, ()))
            if not signature:
                continue
            projected_args = self._project_args_with_type_witnesses(
                getattr(step, "args", ()) or (),
                task_types,
                signature,
                used_names,
            )
            if projected_args is None:
                continue
            literals.append(
                type(
                    "LiteralProxy",
                    (),
                    {
                        "predicate": predicate_name,
                        "args": projected_args,
                        "is_positive": True,
                    },
                )()
            )
        return tuple(literals)

    def _step_effect_predicates(
        self,
        step: Any,
        methods_by_task: Dict[str, List[HTNMethod]],
        render_spec: Dict[str, Any],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
        stack: Tuple[str, ...] = (),
    ) -> Tuple[str, ...]:
        predicates: List[str] = []
        seen: set[str] = set()

        def add(predicate_name: Any) -> None:
            name = str(predicate_name or "").strip()
            if not name or name == "=" or name in seen:
                return
            seen.add(name)
            predicates.append(name)

        for literal in getattr(step, "effects", ()) or ():
            add(getattr(literal, "predicate", None))

        if getattr(step, "kind", None) == "primitive":
            action_schema = self._resolve_action_schema(step, render_spec)
            if action_schema is None:
                return tuple(predicates)
            for effect in getattr(action_schema, "effects", ()) or ():
                add(getattr(effect, "predicate", None))
            return tuple(predicates)

        if getattr(step, "kind", None) != "compound":
            return tuple(predicates)

        task_name = str(getattr(step, "task_name", "") or "").strip()
        if not task_name or task_name in stack:
            return tuple(predicates)

        if task_name not in effect_predicate_cache:
            self._populate_effect_predicate_cache(
                methods_by_task,
                render_spec.get("task_render_specs", {}),
                effect_predicate_cache,
            )
        if task_name in effect_predicate_cache:
            for predicate_name in effect_predicate_cache[task_name]:
                add(predicate_name)
            return tuple(predicates)
        return tuple(predicates)

    def _method_effect_predicates(
        self,
        method: HTNMethod,
        methods_by_task: Dict[str, List[HTNMethod]],
        render_spec: Dict[str, Any],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
        stack: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        predicates: List[str] = []
        seen: set[str] = set()
        for step in self._ordered_method_steps(method):
            for predicate_name in self._step_effect_predicates(
                step,
                methods_by_task,
                render_spec,
                effect_predicate_cache,
                stack,
            ):
                if predicate_name in seen:
                    continue
                seen.add(predicate_name)
                predicates.append(predicate_name)
        return tuple(predicates)

    def _populate_effect_predicate_cache(
        self,
        methods_by_task: Dict[str, List[HTNMethod]],
        render_specs: Dict[str, Dict[str, Any]],
        effect_predicate_cache: Dict[str, Tuple[str, ...]],
    ) -> None:
        task_names = list(
            dict.fromkeys(
                [
                    *tuple(render_specs.keys()),
                    *tuple(methods_by_task.keys()),
                ],
            ),
        )
        if not task_names or all(task_name in effect_predicate_cache for task_name in task_names):
            return

        direct_predicates_by_task: Dict[str, Tuple[str, ...]] = {}
        child_dependencies_by_task: Dict[str, Tuple[str, ...]] = {}

        for task_name in task_names:
            task_render_spec = render_specs.get(task_name, {})
            direct_predicates: List[str] = []
            direct_seen: set[str] = set()
            child_dependencies: List[str] = []
            child_seen: set[str] = set()

            def add_direct(predicate_name: Any) -> None:
                name = str(predicate_name or "").strip()
                if not name or name == "=" or name in direct_seen:
                    return
                direct_seen.add(name)
                direct_predicates.append(name)

            for method in methods_by_task.get(task_name, ()):
                for step in self._ordered_method_steps(method):
                    for literal in getattr(step, "effects", ()) or ():
                        add_direct(getattr(literal, "predicate", None))
                    if getattr(step, "kind", None) == "primitive":
                        action_schema = self._resolve_action_schema(step, task_render_spec)
                        if action_schema is None:
                            continue
                        for effect in getattr(action_schema, "effects", ()) or ():
                            add_direct(getattr(effect, "predicate", None))
                        continue
                    if getattr(step, "kind", None) != "compound":
                        continue
                    child_task_name = str(getattr(step, "task_name", "") or "").strip()
                    if not child_task_name or child_task_name in child_seen:
                        continue
                    child_seen.add(child_task_name)
                    child_dependencies.append(child_task_name)

            direct_predicates_by_task[task_name] = tuple(direct_predicates)
            child_dependencies_by_task[task_name] = tuple(child_dependencies)
            effect_predicate_cache.setdefault(task_name, tuple(direct_predicates))

        changed = True
        while changed:
            changed = False
            for task_name in task_names:
                merged: List[str] = list(direct_predicates_by_task.get(task_name, ()))
                merged_seen: set[str] = set(merged)
                for child_task_name in child_dependencies_by_task.get(task_name, ()):
                    for predicate_name in effect_predicate_cache.get(child_task_name, ()):
                        if predicate_name in merged_seen:
                            continue
                        merged_seen.add(predicate_name)
                        merged.append(predicate_name)
                merged_tuple = tuple(merged)
                if effect_predicate_cache.get(task_name) == merged_tuple:
                    continue
                effect_predicate_cache[task_name] = merged_tuple
                changed = True

    def _compound_task_effect_templates(
        self,
        task_name: str,
        task_lookup: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        task_render_specs: Dict[str, Dict[str, Any]],
        effect_cache: Dict[str, Tuple[Any, ...]],
        stack: Tuple[str, ...] = (),
    ) -> Tuple[Any, ...]:
        if task_name in effect_cache:
            return effect_cache[task_name]

        task_schema = task_lookup.get(task_name)
        task_spec = task_render_specs.get(task_name, {})
        predicate_types = task_spec.get("predicate_types", {})
        task_types = tuple(task_spec.get("task_param_types", ()))
        task_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
        collected: List[Any] = []
        seen_signatures: set[str] = set()

        if task_schema and len(getattr(task_schema, "source_predicates", ())) == 1:
            predicate_name = task_schema.source_predicates[0]
            projected_args = self._project_compound_effect_args(
                task_parameters,
                task_types,
                predicate_types.get(predicate_name, ()),
            )
            if projected_args is not None:
                literal = type(
                    "LiteralProxy",
                    (),
                    {
                        "predicate": predicate_name,
                        "args": projected_args,
                        "is_positive": True,
                    },
                )()
                seen_signatures.add(self._literal_signature(literal))
                collected.append(literal)

        # Seed the cache before descending into child methods so mutual-recursion
        # cycles can still reuse already-known headline effects without recursing
        # indefinitely through A -> B -> A support summaries.
        effect_cache[task_name] = tuple(collected)
        candidate_methods = tuple(methods_by_task.get(task_name, ()))
        if task_name in stack:
            candidate_methods = tuple(
                method
                for method in candidate_methods
                if not self._method_is_recursive(method)
            )

        for method in candidate_methods:
            for literal in self._method_effect_templates(
                method,
                task_lookup,
                methods_by_task,
                task_render_specs,
                effect_cache,
                stack + (task_name,),
            ):
                signature = self._literal_signature(literal)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                collected.append(literal)

        effect_cache[task_name] = tuple(collected)
        return effect_cache[task_name]

    def _method_effect_templates(
        self,
        method: HTNMethod,
        task_lookup: Dict[str, Any],
        methods_by_task: Dict[str, List[HTNMethod]],
        task_render_specs: Dict[str, Dict[str, Any]],
        effect_cache: Dict[str, Tuple[Any, ...]],
        stack: Tuple[str, ...],
    ) -> Tuple[Any, ...]:
        task_schema = task_lookup.get(method.task_name)
        if task_schema is None:
            return ()

        task_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
        task_binding_parameters = self._task_binding_parameters(
            method,
            len(task_parameters),
        )
        task_bindings = {
            parameter: task_parameter
            for parameter, task_parameter in zip(task_binding_parameters, task_parameters)
        }
        method_render_spec = task_render_specs.get(method.task_name, {})
        ordered_steps = self._ordered_method_steps(method)
        collected: List[Any] = []
        seen_signatures: set[str] = set()

        for step in ordered_steps:
            step_effects = self._step_effect_literals(
                step,
                task_lookup,
                methods_by_task,
                method_render_spec,
                effect_cache,
                {},
                stack,
            )
            for literal in step_effects:
                if not getattr(literal, "is_positive", True):
                    continue
                lifted_args: List[str] = []
                for arg in literal.args:
                    if arg in task_bindings:
                        lifted_args.append(task_bindings[arg])
                        continue
                    if self._looks_like_variable(arg):
                        lifted_args = []
                        break
                    lifted_args.append(arg)
                if not lifted_args and literal.args:
                    continue
                lifted_literal = type(
                    "LiteralProxy",
                    (),
                    {
                        "predicate": literal.predicate,
                        "args": tuple(lifted_args),
                        "is_positive": True,
                    },
                )()
                signature = self._literal_signature(lifted_literal)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                collected.append(lifted_literal)

        return tuple(collected)

    def _project_compound_effect_args(
        self,
        step_args: Sequence[str],
        task_types: Sequence[str],
        predicate_types: Sequence[str],
    ) -> Optional[Tuple[str, ...]]:
        if not predicate_types:
            return tuple(step_args)
        if len(task_types) != len(step_args):
            if len(step_args) == len(predicate_types):
                return tuple(step_args)
            return None

        if len(step_args) == len(predicate_types):
            projected = self._project_args_by_type(
                step_args,
                task_types,
                predicate_types,
            )
            if projected is not None:
                return projected
            return tuple(step_args)

        projected = self._project_args_by_type(
            step_args,
            task_types,
            predicate_types,
        )
        return projected

    def _project_args_by_type(
        self,
        step_args: Sequence[str],
        task_types: Sequence[str],
        predicate_types: Sequence[str],
    ) -> Optional[Tuple[str, ...]]:
        if len(task_types) != len(step_args):
            return None

        used_indexes: set[int] = set()
        projected: List[str] = []
        for required_type in predicate_types:
            candidates = [
                index
                for index, task_type in enumerate(task_types)
                if index not in used_indexes and task_type == required_type
            ]
            if len(candidates) != 1:
                return None
            chosen_index = candidates[0]
            used_indexes.add(chosen_index)
            projected.append(step_args[chosen_index])
        return tuple(projected)

    def _project_args_with_type_witnesses(
        self,
        step_args: Sequence[str],
        task_types: Sequence[str],
        predicate_types: Sequence[str],
        used_names: set[str],
    ) -> Optional[Tuple[str, ...]]:
        if len(task_types) != len(step_args):
            return None

        projected: List[str] = []
        mapped_any = False
        for required_type in predicate_types:
            candidate_indexes = [
                index
                for index, task_type in enumerate(task_types)
                if task_type == required_type
            ]
            if len(candidate_indexes) == 1:
                mapped_any = True
                projected.append(str(step_args[candidate_indexes[0]]))
                continue
            projected.append(
                self._fresh_canonical_variable_name(
                    required_type,
                    used_names,
                ),
            )
        if not mapped_any:
            return None
        return tuple(projected)

    def _resolve_action_schema(self, step: Any, render_spec: Dict[str, Any]) -> Any:
        action_schemas = render_spec.get("action_schemas", {})
        action_name = getattr(step, "action_name", None)
        if action_name and action_name in action_schemas:
            return action_schemas[action_name]
        return action_schemas.get(step.task_name)

    def _materialise_action_literals(
        self,
        patterns: Tuple[Any, ...],
        schema_parameters: Tuple[str, ...],
        step_args: Tuple[str, ...],
    ) -> Tuple[Any, ...]:
        bindings = {
            parameter: arg
            for parameter, arg in zip(schema_parameters, step_args)
        }
        return tuple(
            type(
                "LiteralProxy",
                (),
                {
                    "predicate": pattern.predicate,
                    "args": tuple(bindings.get(arg, arg) for arg in pattern.args),
                    "is_positive": pattern.is_positive,
                },
            )()
            for pattern in patterns
        )

    def _literal_signature(self, literal: Any) -> str:
        if getattr(literal, "predicate", None) == "=" and len(literal.args) == 2:
            operator = "==" if getattr(literal, "is_positive", True) else "!="
            return f"{literal.args[0]} {operator} {literal.args[1]}"
        inner = self._call(literal.predicate, tuple(literal.args))
        if literal.is_positive:
            return inner
        return f"not {inner}"

    @staticmethod
    def _literal_variables(literal: Any) -> Tuple[str, ...]:
        return tuple(literal.args)

    @staticmethod
    def _looks_like_variable(symbol: str) -> bool:
        text = str(symbol or "").strip()
        if not text:
            return False
        if text.startswith("?"):
            return len(text) > 1 and text[1].isalpha()
        return text[0].isupper()

    def _context_from_precondition_clauses(
        self,
        clauses: Sequence[Sequence[Any]],
        parameters: Sequence[str],
        args: Sequence[str],
        prefix_literals: Sequence[Any] = (),
    ) -> str:
        bindings = {
            parameter: value
            for parameter, value in zip(parameters, args)
        }
        if not clauses:
            return "__hddl_unsat_condition__"

        rendered_clauses: List[str] = []
        for clause in clauses:
            parts = [
                *(
                    self._render_literal(literal, {})
                    for literal in prefix_literals
                ),
                *(
                    self._render_literal(literal, bindings)
                    for literal in clause
                ),
            ]
            rendered_clauses.append(" & ".join(parts) if parts else "true")

        if len(rendered_clauses) == 1:
            return rendered_clauses[0]

        return " | ".join(
            f"({item})" if " & " in item else item
            for item in rendered_clauses
        )

    def _call(self, name: str, args: Sequence[str]) -> str:
        functor = self._sanitize_name(name)
        if not args:
            return functor
        rendered_args = tuple(self._asl_term(arg) for arg in args)
        return f"{functor}({', '.join(rendered_args)})"

    def _type_guard_literals(
        self,
        args: Sequence[str],
        type_signature: Sequence[str],
    ) -> Tuple[Any, ...]:
        if not self._object_type_map:
            return ()

        guards: List[Any] = []
        seen: set[Tuple[str, str]] = set()
        for arg, type_name in zip(args, type_signature):
            type_atom = self._type_atom(type_name)
            if type_atom == "object":
                continue
            key = (str(arg), type_atom)
            if key in seen:
                continue
            seen.add(key)
            guards.append(
                type(
                    "LiteralProxy",
                    (),
                    {
                        "predicate": "object_type",
                        "args": (arg, type_atom),
                        "is_positive": True,
                    },
                )()
            )
        return tuple(guards)

    def _method_type_guard_literals(
        self,
        method: HTNMethod,
        render_spec: Dict[str, Any],
        variable_map: Dict[str, str],
    ) -> Tuple[Any, ...]:
        if not self._object_type_map:
            return ()

        task_param_types = tuple(render_spec.get("task_param_types", ()))
        default_type = render_spec.get("default_type", "OBJECT")
        predicate_types = render_spec.get("predicate_types", {})
        action_types = render_spec.get("action_types", {})
        method_types, appearance_order = self._infer_method_variable_types(
            method,
            task_param_types,
            predicate_types,
            action_types,
            default_type,
            render_spec.get("task_render_specs", {}),
        )
        ordered_variables = list(variable_map.keys())
        for variable in appearance_order:
            if variable not in ordered_variables:
                ordered_variables.append(variable)
        guarded_variables = {
            variable
            for variable in ordered_variables
            if self._looks_like_variable(variable)
        }

        guards: List[Any] = []
        seen: set[Tuple[str, str]] = set()
        for variable in ordered_variables:
            if variable not in guarded_variables:
                continue
            canonical = variable_map.get(variable, variable)
            type_name = method_types.get(variable)
            if not type_name:
                continue
            type_atom = self._type_atom(type_name)
            if type_atom == "object":
                continue
            key = (canonical, type_atom)
            if key in seen:
                continue
            seen.add(key)
            guards.append(
                type(
                    "LiteralProxy",
                    (),
                    {
                        "predicate": "object_type",
                        "args": (canonical, type_atom),
                        "is_positive": True,
                    },
                )()
            )
        return tuple(guards)

    def _task_call(self, name: str, args: Sequence[str]) -> str:
        return self._call(self._sanitize_name(name), args)

    @staticmethod
    def _indent_body(body_lines: Sequence[str]) -> List[str]:
        if not body_lines:
            return ["\ttrue."]

        lines: List[str] = []
        last_index = len(body_lines) - 1
        for index, line in enumerate(body_lines):
            suffix = "." if index == last_index else ";"
            lines.append(f"\t{line}{suffix}")
        return lines
