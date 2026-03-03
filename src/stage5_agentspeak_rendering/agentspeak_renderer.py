"""
Stage 5 AgentSpeak rendering for the HTN library + DFA wrapper pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from stage3_method_synthesis.htn_schema import HTNMethod, HTNMethodLibrary
from stage4_panda_planning.panda_schema import PANDAPlanResult
from utils.hddl_condition_parser import HDDLConditionParser


class AgentSpeakRenderer:
    """Render the HTN method library plus DFA wrappers as AgentSpeak code."""

    def __init__(self) -> None:
        self.parser = HDDLConditionParser()

    def generate(
        self,
        domain: Any,
        objects: Sequence[str],
        method_library: HTNMethodLibrary,
        plan_records: Sequence[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        lines.extend(self._render_header(domain, objects, plan_records))
        lines.extend(self._render_primitive_wrappers(domain))
        lines.extend(self._render_method_plans(domain, method_library))
        lines.extend(self._render_transition_plans(plan_records))
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
            lines.append(f"object({obj}).")

        if plan_records:
            initial_state = plan_records[0].get("initial_state")
            if initial_state:
                lines.append(f"dfa_state({initial_state}).")

        for record in plan_records:
            lines.append(
                f"dfa_edge_label({record['transition_name']}, \"{record['label']}\")."
            )

        lines.append("")
        return lines

    def _render_primitive_wrappers(self, domain: Any) -> List[str]:
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
            context = self._context_clause(semantics.preconditions, semantics.parameters, args)
            body_lines = [f"{task_name}({', '.join(args)})"]

            bindings = {
                parameter: value
                for parameter, value in zip(semantics.parameters, args)
            }
            for effect in semantics.effects:
                bound_args = tuple(bindings.get(item, item) for item in effect.args)
                prefix = "+" if effect.is_positive else "-"
                body_lines.append(f"{prefix}{self._call(effect.predicate, bound_args)}")

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
        render_specs = self._build_task_render_specs(domain, method_library)

        for method in method_library.methods:
            lines.extend(self._render_method_plan(method, task_lookup, render_specs))
            lines.append("")

        return lines

    def _render_method_plan(
        self,
        method: HTNMethod,
        task_lookup: Dict[str, Any],
        render_specs: Dict[str, Dict[str, Any]],
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
            render_spec,
            variable_map,
        )
        trigger = self._call(method.task_name, trigger_args)
        context = self._literal_clause(context_literals, variable_map)
        body = [
            f"!{self._call(step.task_name, self._map_args(step.args, variable_map))}"
            for step in ordered_steps
        ] or ["true"]

        lines = [f"+!{trigger} : {context} <-"]
        lines.extend(self._indent_body(body))
        return lines

    def _render_transition_plans(self, plan_records: Sequence[Dict[str, Any]]) -> List[str]:
        lines = ["/* DFA Transition Wrappers */"]

        for record in plan_records:
            plan: PANDAPlanResult = record["plan"]
            source_state = record["source_state"]
            target_state = record["target_state"]
            body = [
                f"!{self._call(plan.task_name, plan.task_args)}",
                f"-{self._call('dfa_state', (source_state,))}",
                f"+{self._call('dfa_state', (target_state,))}",
            ]
            lines.append(
                f"+!{record['transition_name']} : {self._call('dfa_state', (source_state,))} <-"
            )
            lines.extend(self._indent_body(body))
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
        render_spec: Dict[str, Any],
        variable_map: Dict[str, str],
    ) -> Tuple[Any, ...]:
        context_literals: List[Any] = list(method.context)
        seen_signatures = {
            self._literal_signature(literal)
            for literal in context_literals
        }
        bound_variables = set(method.parameters)
        bound_variables.update(self._bound_variables_from_literals(tuple(context_literals)))
        prior_effect_signatures: set[str] = set()

        for step in ordered_steps:
            step_preconditions = self._step_precondition_literals(step, render_spec)
            for literal in step_preconditions:
                if not literal.is_positive:
                    continue

                if self._literal_signature(literal) in prior_effect_signatures:
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

            for effect in self._step_effect_literals(step, task_lookup, render_spec):
                prior_effect_signatures.add(self._literal_signature(effect))

        return tuple(context_literals)

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

    def _build_task_render_specs(
        self,
        domain: Any,
        method_library: HTNMethodLibrary,
    ) -> Dict[str, Dict[str, Any]]:
        predicate_types = self._predicate_type_map(domain)
        action_types = self._action_type_map(domain)
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
        return specs

    def _infer_task_parameter_types(
        self,
        task: Any,
        task_methods: Sequence[HTNMethod],
        predicate_types: Dict[str, Tuple[str, ...]],
        action_types: Dict[str, Tuple[str, ...]],
        default_type: str,
    ) -> Tuple[str, ...]:
        inferred: List[str | None] = [None] * len(task.parameters)

        if len(getattr(task, "source_predicates", ())) == 1:
            predicate_name = task.source_predicates[0]
            predicate_signature = predicate_types.get(predicate_name)
            if predicate_signature and len(predicate_signature) == len(task.parameters):
                inferred = list(predicate_signature)

        for method in task_methods:
            for index, parameter in enumerate(method.parameters[: len(task.parameters)]):
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
        )

        variable_map: Dict[str, str] = {}
        for original, canonical in zip(method.parameters, trigger_args):
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
            next_index = used_counts[type_name] + 1
            overall_count = max(max_counts.get(type_name, 0), method_type_counts.get(type_name, 0))
            if overall_count > 1 or used_counts[type_name] > 0:
                variable_map[variable] = f"{type_name}{next_index}"
            else:
                variable_map[variable] = type_name
            used_counts[type_name] += 1

        return variable_map

    def _infer_method_variable_types(
        self,
        method: HTNMethod,
        task_param_types: Tuple[str, ...],
        predicate_types: Dict[str, Tuple[str, ...]],
        action_types: Dict[str, Tuple[str, ...]],
        default_type: str,
    ) -> Tuple[Dict[str, str], List[str]]:
        variable_types: Dict[str, str] = {}
        appearance_order: List[str] = []

        for index, parameter in enumerate(method.parameters):
            type_name = task_param_types[index] if index < len(task_param_types) else default_type
            variable_types[parameter] = type_name
            if parameter not in appearance_order:
                appearance_order.append(parameter)

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
            )
            variable_types[variable] = candidates[0] if candidates else default_type

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
            per_type_seen[type_name] += 1
            if max_counts.get(type_name, 0) > 1:
                names.append(f"{type_name}{per_type_seen[type_name]}")
            else:
                names.append(type_name)
        return tuple(names)

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
        return tuple(variable_map.get(arg, arg) for arg in args)

    def _render_literal(self, literal: Any, variable_map: Dict[str, str]) -> str:
        mapped_args = self._map_args(literal.args, variable_map)
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
        render_spec: Dict[str, Any],
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
        if len(getattr(task_schema, "source_predicates", ())) != 1:
            return ()
        predicate_name = task_schema.source_predicates[0]
        if len(step.args) == 0:
            return (type("LiteralProxy", (), {"predicate": predicate_name, "args": (), "is_positive": True})(),)
        return (
            type(
                "LiteralProxy",
                (),
                {
                    "predicate": predicate_name,
                    "args": tuple(step.args),
                    "is_positive": True,
                },
            )(),
        )

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
        inner = self._call(literal.predicate, tuple(literal.args))
        return inner if literal.is_positive else f"not {inner}"

    @staticmethod
    def _literal_variables(literal: Any) -> Tuple[str, ...]:
        return tuple(literal.args)

    @staticmethod
    def _looks_like_variable(symbol: str) -> bool:
        return bool(symbol) and symbol[0].isupper()

    def _context_clause(
        self,
        literals: Iterable[Any],
        parameters: Sequence[str],
        args: Sequence[str],
    ) -> str:
        bindings = {
            parameter: value
            for parameter, value in zip(parameters, args)
        }
        parts = []
        for literal in literals:
            bound_args = tuple(bindings.get(item, item) for item in literal.args)
            predicate = self._call(literal.predicate, bound_args)
            if literal.is_positive:
                parts.append(predicate)
            else:
                parts.append(f"not {predicate}")
        return " & ".join(parts) if parts else "true"

    @staticmethod
    def _call(name: str, args: Sequence[str]) -> str:
        if not args:
            return name
        return f"{name}({', '.join(args)})"

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
