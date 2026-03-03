"""
AgentSpeak code generation for the PANDA-backed Stage 3 pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from stage3_code_generation.htn_schema import PANDAPlanResult
from stage3_code_generation.hddl_condition_parser import HDDLConditionParser


class AgentSpeakCodeGenerator:
    """Render PANDA-backed HTN goal plans as AgentSpeak code."""

    def __init__(self) -> None:
        self.parser = HDDLConditionParser()

    def generate(
        self,
        domain: Any,
        objects: Sequence[str],
        plan_records: Sequence[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        lines.extend(self._render_header(domain, objects, plan_records))
        lines.extend(self._render_primitive_wrappers(domain))
        lines.extend(self._render_compound_plans(plan_records))
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

        for record in plan_records:
            lines.append(
                f"target_label({record['transition_name']}, \"{record['label']}\")."
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

    def _render_compound_plans(self, plan_records: Sequence[Dict[str, Any]]) -> List[str]:
        lines = ["/* PANDA Goal Plans */"]
        emitted: set[str] = set()

        for record in plan_records:
            plan: PANDAPlanResult = record["plan"]
            signature = self._call(plan.task_name, plan.task_args)
            if signature in emitted:
                continue
            emitted.add(signature)

            body = [
                f"!{self._call(step.task_name, step.args)}"
                for step in plan.steps
            ] or ["true"]
            lines.append(f"+!{signature} : true <-")
            lines.extend(self._indent_body(body))
            lines.append("")

        return lines

    def _render_transition_plans(self, plan_records: Sequence[Dict[str, Any]]) -> List[str]:
        lines = ["/* Transition Dispatch Plans */"]

        for record in plan_records:
            plan: PANDAPlanResult = record["plan"]
            body = [f"!{self._call(plan.task_name, plan.task_args)}"]
            lines.append(f"+!{record['transition_name']} : true <-")
            lines.extend(self._indent_body(body))
            lines.append("")

        return lines

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
