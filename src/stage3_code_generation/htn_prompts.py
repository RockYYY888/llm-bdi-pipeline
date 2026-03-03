"""
Prompt builders for HTN method synthesis.
"""

from __future__ import annotations

from typing import Any, Iterable


def build_htn_system_prompt() -> str:
    return (
        "You generate a compact HTN method library for a symbolic planning domain.\n"
        "Return valid JSON only.\n"
        "Preserve the provided primitive actions exactly as named.\n"
        "Only create reusable compound tasks and methods that decompose into those "
        "primitive actions or other generated compound tasks.\n"
        "Prefer shallow, goal-oriented decompositions and avoid redundant subtasks."
    )


def build_htn_user_prompt(domain: Any, target_literals: Iterable[str], baseline_schema: str) -> str:
    action_lines = []
    for action in domain.actions:
        params = ", ".join(action.parameters) if action.parameters else "none"
        action_lines.append(
            f"- {action.name}({params}) | pre: {action.preconditions} | eff: {action.effects}"
        )

    predicate_lines = [f"- {predicate.to_signature()}" for predicate in domain.predicates]
    targets = "\n".join(f"- {item}" for item in target_literals)

    return (
        f"Domain: {domain.name}\n"
        f"Predicates:\n{chr(10).join(predicate_lines)}\n\n"
        f"Primitive actions:\n{chr(10).join(action_lines)}\n\n"
        f"Target literals that must be supported:\n{targets}\n\n"
        "Return JSON with this exact top-level schema:\n"
        f"{baseline_schema}\n\n"
        "Requirements:\n"
        "- Each method must reference an existing task.\n"
        "- Each primitive subtask must reference an existing primitive action task.\n"
        "- Use parameter names already introduced by the task or method.\n"
        "- Keep methods deterministic and concise.\n"
        "- Do not include explanations."
    )
