"""
Prompt builders for HTN method synthesis.
"""

from __future__ import annotations

from typing import Any, Iterable


def _sanitize_name(name: str) -> str:
    return name.replace("-", "_")


def _task_name_for_literal(literal: str) -> str:
    is_negative = literal.startswith("!")
    base = literal[1:] if is_negative else literal
    predicate = base.split("(", 1)[0]
    predicate = _sanitize_name(predicate)
    if is_negative:
        return f"maintain_not_{predicate}"
    return f"achieve_{predicate}"


def build_htn_system_prompt() -> str:
    return (
        "You generate a compact HTN method library for a symbolic planning domain.\n"
        "Your JSON is compiled directly into Jason AgentSpeak plans with syntax like:\n"
        "+!goal(X) : context <- !subgoal(X); action(X).\n"
        "If your identifiers or task structure are invalid, the generated .asl will be invalid.\n"
        "\n"
        "NON-NEGOTIABLE OUTPUT RULES:\n"
        "1. Return valid JSON only. No markdown, no code fences, no prose, no comments.\n"
        "2. Return exactly one top-level JSON object.\n"
        "3. The top-level object must contain only the keys requested in the prompt.\n"
        "4. Every identifier that becomes AgentSpeak code must already be AgentSpeak-safe.\n"
        "5. Always include both top-level keys: compound_tasks and methods.\n"
        "6. Every method object must contain exactly: method_name, task_name, parameters, "
        "context, subtasks, ordering, origin.\n"
        "7. Every subtask object must contain exactly: step_id, task_name, args, kind, "
        "action_name, literal, preconditions, effects.\n"
        "8. Use JSON null for missing optional values. Never omit required keys.\n"
        "\n"
        "AGENTSPEAK IDENTIFIER CONTRACT:\n"
        "- task_name and method_name must match [a-z][a-z0-9_]*.\n"
        "- Never use hyphens, spaces, CamelCase, quotes, or punctuation in identifiers.\n"
        "- Primitive subtasks must use the runtime aliases provided in the prompt, not raw source action names.\n"
        "- action_name may keep the source action name, but task_name must use the runtime alias.\n"
        "- Predicate names inside literal objects must also be simple lower-case atoms.\n"
        "- For target literals, the required compound task names are fixed by the prompt. Do not rename them.\n"
        "- Non-guard method names must follow exactly: {task_name}__via_{strategy}.\n"
        "- Guard method names must follow exactly: {task_name}__guard.\n"
        "- Double underscore __ is the only separator between task_name and strategy.\n"
        "- The {strategy} suffix must be short, stable, and describe HOW the task is done.\n"
        "\n"
        "HTN STRUCTURE CONTRACT:\n"
        "- Only create reusable compound tasks and methods.\n"
        "- Primitive action tasks are injected automatically by the runtime.\n"
        "- The library must be closed under compound references.\n"
        "- If a compound subtask mentions helper task X, X must appear in compound_tasks.\n"
        "- If a compound subtask mentions helper task X, at least one method must have task_name X.\n"
        "- Never reference an undefined helper task. If you cannot define it, inline primitive steps instead.\n"
        "- For positive literals, use compound task names shaped like achieve_<predicate>.\n"
        "- For negative literals, use compound task names shaped like maintain_not_<predicate>.\n"
        "- For helper tasks that correspond to a positive predicate P, use achieve_<P>.\n"
        "- Only introduce alternate helper verbs like clear_<noun> or remove_<noun> when there is no direct predicate-backed task.\n"
        "- When a negative target can be satisfied by context alone, emit a guard method with empty subtasks.\n"
        "- Keep methods shallow, deterministic, and non-redundant.\n"
        "- context is for method-level preconditions checked before decomposition.\n"
        "- subtask.preconditions is for step-level checks that hold when that step executes.\n"
        "- Guard methods must put the already-satisfied condition in context and leave subtasks empty.\n"
        "- Non-guard methods should usually keep context empty unless the whole decomposition requires a shared method-level precondition.\n"
        "- Every ordering edge must reference existing step_id values.\n"
        "- ordering must be a list of edges [from_step_id, to_step_id].\n"
        "- For a total order [s1, s2, s3], write ordering as [[\"s1\", \"s2\"], [\"s2\", \"s3\"]].\n"
        "- For guard methods with empty subtasks, ordering must be [].\n"
        "- Every step args list must use only parameters already introduced by the method.\n"
    )


def build_htn_user_prompt(domain: Any, target_literals: Iterable[str], schema_hint: str) -> str:
    action_lines = []
    for action in domain.actions:
        params = ", ".join(action.parameters) if action.parameters else "none"
        action_lines.append(
            f"- runtime task: {_sanitize_name(action.name)} | source action: {action.name}"
            f"({params}) | pre: {action.preconditions} | eff: {action.effects}"
        )

    predicate_lines = [f"- {predicate.to_signature()}" for predicate in domain.predicates]
    targets = "\n".join(f"- {item}" for item in target_literals)
    task_hints = "\n".join(
        f"- REQUIRED compound task name for {literal}: {_task_name_for_literal(literal)}"
        for literal in target_literals
    )
    few_shot_examples = """
Few-shot guidance (illustrative only, not your domain):

Example 1: Positive target with runtime primitive aliases
Target literal: delivered(pkg1)
Required compound task: achieve_delivered
Because the first compound step references achieve_loaded, the final top-level
compound_tasks array must include BOTH achieve_delivered and achieve_loaded, and
the final methods array must include at least one method whose task_name is
achieve_loaded.
Valid method fragment:
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
      "action_name": null,
      "literal": {"predicate": "loaded", "args": ["X1"], "is_positive": true, "source_symbol": null},
      "preconditions": [],
      "effects": []
    },
    {
      "step_id": "s2",
      "task_name": "drop_parcel",
      "args": ["X1"],
      "kind": "primitive",
      "action_name": "drop-parcel",
      "literal": {"predicate": "delivered", "args": ["X1"], "is_positive": true, "source_symbol": null},
      "preconditions": [],
      "effects": []
    }
  ],
  "ordering": [["s1", "s2"]],
  "origin": "llm"
}

Example 2: Negative target as a guard method
Target literal: !door_open(room1)
Required compound task: maintain_not_door_open
Valid method fragment:
{
  "method_name": "maintain_not_door_open__guard",
  "task_name": "maintain_not_door_open",
  "parameters": ["X1"],
  "context": [
    {"predicate": "door_open", "args": ["X1"], "is_positive": false, "source_symbol": null}
  ],
  "subtasks": [],
  "ordering": [],
  "origin": "llm"
}
""".strip()

    return (
        "TASK:\n"
        "Generate a JSON HTN method library that will be compiled into valid AgentSpeak.\n"
        "The JSON must satisfy the syntax and naming contract below.\n\n"
        f"DOMAIN:\n{domain.name}\n\n"
        f"PREDICATES:\n{chr(10).join(predicate_lines)}\n\n"
        f"RUNTIME PRIMITIVE ACTION ALIASES:\n{chr(10).join(action_lines)}\n\n"
        f"TARGET LITERALS:\n{targets}\n\n"
        f"REQUIRED COMPOUND TASK NAMES:\n{task_hints}\n\n"
        "TOP-LEVEL JSON SHAPE:\n"
        "Primitive tasks are injected automatically by the runtime, so only define compound_tasks and methods.\n"
        f"{schema_hint}\n\n"
        f"{few_shot_examples}\n\n"
        "REQUIRED CONTENT RULES:\n"
        "- Each method.task_name must reference an existing compound task.\n"
        "- Each primitive subtask.task_name must match one of the provided runtime primitive action aliases exactly.\n"
        "- If the source action is written with hyphens, convert task_name to the provided underscore alias.\n"
        "- Each compound subtask.task_name must reference a compound task that exists in compound_tasks.\n"
        "- For every compound subtask.task_name X, include a compound_tasks entry named X.\n"
        "- For every compound subtask.task_name X, include at least one method whose task_name is X.\n"
        "- Do not reference helper tasks unless you fully define them in the same JSON object.\n"
        "- If you only need one extra primitive step, inline it instead of inventing an undefined helper.\n"
        "- If a helper subtask corresponds to a positive predicate P, name it achieve_P.\n"
        "- If no predicate-backed helper fits, use a short underscore_case helper verb such as clear_block_top or remove_supporting_block.\n"
        "- Every task_name and method_name must already be underscore_case.\n"
        "- For every non-guard method, method_name must be exactly task_name + '__via_' + a short strategy suffix.\n"
        "- For every guard method, method_name must end exactly in '__guard'.\n"
        "- Do not use any other method naming pattern.\n"
        "- Use parameter names already introduced by the task or method.\n"
        "- For a guard method, use a non-empty context, an empty subtasks list, and an empty ordering list.\n"
        "- For a non-guard method, put method-wide decomposition checks in context; put step-local checks in subtask.preconditions.\n"
        "- ordering is a list of [from_step_id, to_step_id] edges.\n"
        "- For a total order s1 -> s2 -> s3, write ordering as [[\"s1\", \"s2\"], [\"s2\", \"s3\"]].\n"
        "- Do not invent unsupported top-level keys.\n"
        "- Do not omit required method keys or required subtask keys.\n"
        "- Use JSON null, not the string \"null\", for missing optional values.\n"
        "- Do not include explanations.\n"
        "\n"
        "FINAL SELF-CHECK BEFORE RETURNING JSON:\n"
        "1. Do all target literals use the exact required compound task names from the prompt?\n"
        "2. Are all task_name values underscore_case?\n"
        "3. Do primitive steps use runtime aliases rather than raw source action names?\n"
        "4. Do all ordering edges reference real step_id values?\n"
        "5. Is every compound helper task declared in compound_tasks and implemented by at least one method?\n"
        "6. Do all non-guard methods use the exact task_name__via_strategy naming pattern?\n"
        "7. Does every guard method end in __guard and keep subtasks/orderings empty?\n"
        "8. Does the JSON contain only the requested keys?\n"
        "9. Would each task_name compile cleanly inside +!task_name(...) AgentSpeak syntax?\n"
    )
