"""
Stage 5 AgentSpeak rendering for the HTN library + DFA wrapper pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from stage3_method_synthesis.htn_schema import HTNMethod, HTNMethodLibrary
from stage4_panda_planning.panda_schema import PANDAPlanResult
from utils.hddl_condition_parser import HDDLConditionParser


class AgentSpeakRenderer:
    """Render PANDA-generated goal plans as AgentSpeak code."""

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
        lines.extend(self._render_method_plans(method_library))
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
            args = tuple(f"X{index + 1}" for index, _ in enumerate(semantics.parameters))
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

    def _render_method_plans(self, method_library: HTNMethodLibrary) -> List[str]:
        lines = ["/* HTN Method Plans */"]
        task_lookup = {
            task.name: task
            for task in method_library.compound_tasks
        }

        for method in method_library.methods:
            lines.extend(self._render_method_plan(method, task_lookup))
            lines.append("")

        return lines

    def _render_method_plan(
        self,
        method: HTNMethod,
        task_lookup: Dict[str, Any],
    ) -> List[str]:
        task_schema = task_lookup.get(method.task_name)
        trigger_args = task_schema.parameters if task_schema else method.parameters
        trigger = self._call(method.task_name, trigger_args)
        context = self._literal_clause(method.context)
        ordered_steps = self._ordered_method_steps(method)
        body = [
            f"!{self._call(step.task_name, step.args)}"
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

    def _literal_clause(self, literals: Iterable[Any]) -> str:
        parts = [
            literal.to_agentspeak()
            for literal in literals
        ]
        return " & ".join(parts) if parts else "true"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("-", "_")

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
