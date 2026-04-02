"""
Stage 5 AgentSpeak rendering for the HTN library + DFA wrapper pipeline.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from stage3_method_synthesis.htn_schema import HTNMethod, HTNMethodLibrary
from stage4_panda_planning.panda_schema import PANDAPlanResult
from utils.hddl_condition_parser import HDDLConditionParser


class AgentSpeakRenderer:
    """Render the HTN method library plus DFA wrappers as AgentSpeak code."""

    def __init__(self) -> None:
        self.parser = HDDLConditionParser()
        self._object_symbol_set: set[str] = set()
        self._object_type_map: Dict[str, str] = {}
        self._type_parent_map: Dict[str, Optional[str]] = {}

    def generate(
        self,
        domain: Any,
        objects: Sequence[str],
        method_library: HTNMethodLibrary,
        plan_records: Sequence[Dict[str, Any]],
        typed_objects: Optional[Sequence[Tuple[str, str]]] = None,
        ordered_query_sequence: bool = True,
    ) -> str:
        self._object_symbol_set = {str(obj) for obj in objects}
        self._object_type_map = {
            str(name): str(type_name)
            for name, type_name in (typed_objects or ())
        }
        self._type_parent_map = self._build_type_parent_map(domain)
        _ = ordered_query_sequence
        lines: List[str] = []
        lines.extend(self._render_header(domain, objects, plan_records))
        lines.extend(self._render_primitive_wrappers(domain))
        lines.extend(self._render_method_plans(domain, method_library))
        lines.extend(self._render_transition_plans(plan_records))
        lines.extend(self._render_control_plans(plan_records))
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

        if plan_records:
            initial_state = plan_records[0].get("initial_state")
            if initial_state:
                lines.append(f"dfa_state({initial_state}).")
            accepting_states = sorted(
                {
                    state
                    for record in plan_records
                    for state in record.get("accepting_states", [])
                },
            )
            for state in accepting_states:
                lines.append(f"accepting_state({state}).")

        for record in plan_records:
            lines.append(
                f"dfa_edge_label({record['transition_name']}, \"{record['label']}\")."
            )

        lines.append("")
        return lines

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

    def _render_method_plans(self, domain: Any, method_library: HTNMethodLibrary) -> List[str]:
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

    @staticmethod
    def _method_is_pure_noop(method: HTNMethod) -> bool:
        """Return True when a method has no subtasks and performs no decomposition."""

        return not tuple(getattr(method, "subtasks", ()) or ())

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
        task_effect_signatures = {
            self._literal_signature(literal)
            for literal in self._compound_task_effect_templates(
                task_name,
                task_lookup,
                methods_by_task,
                render_specs,
                effect_cache,
            )
            if getattr(literal, "is_positive", True)
        }
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
        render_spec = render_specs.get(method.task_name, {})
        variable_map = self._method_variable_map(
            method,
            ordered_steps,
            render_spec,
        )
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
        directly_supports_effect = self._context_literals_support_task_effect(
            method,
            context_literals,
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
        variable_map = self._method_variable_map(
            method,
            ordered_steps,
            render_spec,
        )
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
        context_literals = self._merge_context_literals(
            context_literals,
            self._method_type_guard_literals(
                method,
                render_spec,
                variable_map,
            ),
        )
        context_literals = self._merge_context_literals(
            context_literals,
            self._self_recursive_progress_literals(
                method,
                ordered_steps,
            ),
        )
        specialisations = self._first_compound_child_specialisations(
            method,
            ordered_steps,
            task_lookup,
            methods_by_task,
            render_specs,
            variable_map,
            effect_cache,
            effect_predicate_cache,
        )
        trigger = self._task_call(method.task_name, trigger_args)
        method_trace_line = self._render_method_trace_statement(
            method,
            self._map_args(trigger_args, variable_map),
        )
        body = [method_trace_line]
        body.extend(
            f"!{self._task_call(step.task_name, self._map_args(step.args, variable_map))}"
            for step in ordered_steps
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

    def _render_method_trace_statement(
        self,
        method: HTNMethod,
        trigger_args: Sequence[str],
    ) -> str:
        trace_term = self._call(
            "trace_method",
            (self._asl_atom_or_string(method.method_name), *trigger_args),
        )
        return f'.print("runtime trace method ", {trace_term})'

    def _render_transition_plans(self, plan_records: Sequence[Dict[str, Any]]) -> List[str]:
        lines = ["/* DFA Transition Wrappers */"]

        for record in plan_records:
            plan: PANDAPlanResult = record["plan"]
            source_state = record["source_state"]
            target_state = record["target_state"]
            body = [f"!{self._task_call(plan.task_name, plan.task_args)}"]
            if source_state != target_state:
                body.extend(
                    [
                        f"-{self._call('dfa_state', (source_state,))}",
                        f"+{self._call('dfa_state', (target_state,))}",
                    ],
                )
            lines.append(
                f"+!{record['transition_name']} : {self._call('dfa_state', (source_state,))} <-"
            )
            lines.extend(self._indent_body(body))
            lines.append("")

        return lines

    def _render_control_plans(self, plan_records: Sequence[Dict[str, Any]]) -> List[str]:
        lines = ["/* DFA Control Plans */"]
        transitions_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        accepting_states = {
            state
            for record in plan_records
            for state in record.get("accepting_states", [])
        }

        for record in plan_records:
            transitions_by_source[record["source_state"]].append(record)

        for state in sorted(accepting_states):
            lines.append(
                f"+!run_dfa : {self._call('dfa_state', (state,))} & "
                f"{self._call('accepting_state', (state,))} <-"
            )
            lines.extend(self._indent_body(["true"]))
            lines.append("")

        for source_state in sorted(transitions_by_source):
            if source_state in accepting_states:
                continue
            state_records = transitions_by_source[source_state]
            for record in state_records:
                lines.append(
                    f"+!run_dfa : {self._call('dfa_state', (source_state,))} <-"
                )
                lines.extend(
                    self._indent_body(
                        [
                            f"!{record['transition_name']}",
                            "!run_dfa",
                        ],
                    ),
                )
                lines.append("")

        return lines

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
        bound_variables = set(method.parameters)
        bound_variables.update(self._bound_variables_from_literals(tuple(context_literals)))
        prior_effect_signatures: set[str] = set()
        prior_effect_literals: List[Any] = []

        first_step_id = ordered_steps[0].step_id if ordered_steps else None
        for step in ordered_steps:
            step_preconditions = list(self._step_precondition_literals(step, render_spec))
            if getattr(step, "kind", None) == "compound" and step.step_id != first_step_id:
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

        return tuple(context_literals)

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
            self._literal_signature(literal)
            for literal in merged
        }
        for literal in extra_literals:
            signature = self._literal_signature(literal)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            merged.append(literal)
        return tuple(merged)

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
    ) -> Tuple[Tuple[Any, ...], ...]:
        first_compound_step = next(
            (step for step in ordered_steps if getattr(step, "kind", None) == "compound"),
            None,
        )
        if first_compound_step is None:
            return ((),)

        child_methods = tuple(methods_by_task.get(first_compound_step.task_name, ()))
        if not child_methods:
            return ((),)
        if any(self._method_is_recursive(child_method) for child_method in child_methods):
            return ((),)

        prior_effect_signatures: set[str] = set()
        for step in ordered_steps:
            if step.step_id == first_compound_step.step_id:
                break
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

        needs_specialisation = len(child_methods) > 1 or any(
            len(
                self._compound_method_internal_parameters(
                    child_method,
                    len(first_compound_step.args),
                ),
            ) > 0
            for child_method in child_methods
        )
        if not needs_specialisation:
            return ((),)

        variants: List[Tuple[Any, ...]] = []
        seen_variants: set[Tuple[str, ...]] = set()
        for child_method in child_methods:
            child_render_spec = render_specs.get(first_compound_step.task_name, {})
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
            translated = self._translate_child_context_literals(
                child_method,
                first_compound_step,
                child_context_literals,
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
            variants.append(translated)

        return tuple(variants) if variants else ((),)

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
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            return text
        if text in self._object_symbol_set:
            return self._asl_atom_or_string(text)
        if self._looks_like_variable(text):
            return text
        return self._asl_atom_or_string(text)

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
            for index, parameter in enumerate(
                self._task_binding_parameters(method, len(task.parameters)),
            ):
                if inferred[index] is not None:
                    continue
                candidates = self._variable_type_candidates(
                    method,
                    parameter,
                    predicate_types,
                    action_types,
                )
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

        for variable in self._variables_in_method(method):
            if variable not in appearance_order:
                appearance_order.append(variable)
            if variable in variable_types:
                continue
            candidates = self._variable_type_candidates(
                method,
                variable,
                predicate_types,
                action_types,
                task_render_specs,
            )
            variable_types[variable] = candidates[0] if candidates else default_type

        for parameter in method.parameters:
            variable_types.setdefault(parameter, default_type)

        return variable_types, appearance_order

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
        candidates: List[str] = []

        def collect_from_literal(literal: Any) -> None:
            signature = predicate_types.get(literal.predicate)
            if not signature:
                return
            for index, arg in enumerate(literal.args):
                if arg == variable and index < len(signature):
                    candidates.append(signature[index])

        for literal in method.context:
            collect_from_literal(literal)
        for step in method.subtasks:
            action_signature = action_types.get(step.action_name or "")
            if step.kind == "primitive" and not action_signature:
                action_signature = action_types.get(step.task_name)
            if action_signature:
                for index, arg in enumerate(step.args):
                    if arg == variable and index < len(action_signature):
                        candidates.append(action_signature[index])
            elif step.kind == "compound":
                child_render_spec = (task_render_specs or {}).get(step.task_name, {})
                child_signature = tuple(child_render_spec.get("task_param_types", ()))
                for index, arg in enumerate(step.args):
                    if arg == variable and index < len(child_signature):
                        candidates.append(child_signature[index])
            for literal in (step.literal, *step.preconditions, *step.effects):
                if literal is not None:
                    collect_from_literal(literal)

        return candidates

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

        if task_name in effect_predicate_cache:
            for predicate_name in effect_predicate_cache[task_name]:
                add(predicate_name)
            return tuple(predicates)

        child_render_specs = render_spec.get("task_render_specs", {})
        child_render_spec = child_render_specs.get(task_name, {})
        task_predicates: List[str] = []
        task_seen: set[str] = set()
        for method in methods_by_task.get(task_name, ()):
            for predicate_name in self._method_effect_predicates(
                method,
                methods_by_task,
                child_render_spec,
                effect_predicate_cache,
                stack + (task_name,),
            ):
                if predicate_name in task_seen:
                    continue
                task_seen.add(predicate_name)
                task_predicates.append(predicate_name)

        effect_predicate_cache[task_name] = tuple(task_predicates)
        for predicate_name in task_predicates:
            add(predicate_name)
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

        if task_name in stack:
            for method in methods_by_task.get(task_name, ()):
                if self._method_is_recursive(method):
                    continue
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
            return tuple(collected)

        for method in methods_by_task.get(task_name, ()):
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
        context_variables = {
            arg
            for literal in method.context
            for arg in literal.args
            if self._looks_like_variable(arg)
        }
        task_binding_parameters = set(
            self._task_binding_parameters(method, len(task_param_types)),
        )
        guarded_variables = task_binding_parameters | context_variables
        ordered_variables = list(variable_map.keys())
        for variable in appearance_order:
            if variable not in ordered_variables:
                ordered_variables.append(variable)

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
