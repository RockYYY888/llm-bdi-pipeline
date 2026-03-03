"""
Prompt builders for HTN method synthesis.
"""

from __future__ import annotations

from typing import Any, Iterable


def _sanitize_name(name: str) -> str:
    return name.replace("-", "_")


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
        "5. Always include exactly these top-level keys: target_task_bindings, "
        "compound_tasks, methods.\n"
        "6. Every target_task_bindings item must contain exactly: target_literal, task_name.\n"
        "7. Every method object must contain exactly: method_name, task_name, parameters, "
        "context, subtasks, ordering, origin.\n"
        "8. Every subtask object must contain exactly: step_id, task_name, args, kind, "
        "action_name, literal, preconditions, effects.\n"
        "9. Use JSON null for missing optional values. Never omit required keys.\n"
        "\n"
        "AGENTSPEAK IDENTIFIER CONTRACT:\n"
        "- task_name and method_name must match [a-z][a-z0-9_]*.\n"
        "- Never use hyphens, spaces, CamelCase, quotes, or punctuation in identifiers.\n"
        "- Primitive subtasks must use the runtime aliases provided in the prompt, not raw source action names.\n"
        "- action_name may keep the source action name, but task_name must use the runtime alias.\n"
        "- Predicate names inside literal objects must also be simple lower-case atoms.\n"
        "- Compound task names must be semantic intention names such as place_on, "
        "hold_block, clear_top, deliver_parcel.\n"
        "- Do not mechanically encode polarity in task names. Never use prefixes "
        "achieve_, maintain_not_, ensure_, or goal_.\n"
        "- method_name must follow exactly: m_{task_name}_{strategy}.\n"
        "- The {strategy} suffix must be short, stable, and describe HOW the task "
        "is done, such as stack, from_table, recursive, noop.\n"
        "- For zero-subtask methods that mean the task is already satisfied, prefer "
        "strategy noop or already_<state>.\n"
        "\n"
        "HTN STRUCTURE CONTRACT:\n"
        "- Only create reusable compound tasks and methods.\n"
        "- Primitive action tasks are injected automatically by the runtime.\n"
        "- target_task_bindings must contain one entry for every target literal in the prompt.\n"
        "- Each target_task_bindings.task_name must match a compound task declared in compound_tasks.\n"
        "- The library must be closed under compound references.\n"
        "- If a compound subtask mentions helper task X, X must appear in compound_tasks.\n"
        "- If a compound subtask mentions helper task X, at least one method must have task_name X.\n"
        "- Never reference an undefined helper task. If you cannot define it, inline primitive steps instead.\n"
        "- For helper tasks, keep the same semantic naming style: clear_top, move_support, "
        "free_hand, load_parcel.\n"
        "- Do not dump raw logical conditions into task names.\n"
        "- When a target is already satisfied, emit a zero-subtask method with a semantic "
        "noop/already_* strategy.\n"
        "- Keep methods shallow, deterministic, and non-redundant.\n"
        "- context is for method-level preconditions checked before decomposition.\n"
        "- subtask.preconditions is for step-level checks that hold when that step executes.\n"
        "- For zero-subtask methods, put the already-satisfied condition in context and leave subtasks empty.\n"
        "- For non-zero-subtask methods, context should usually stay empty unless the whole decomposition "
        "requires a shared method-level precondition.\n"
        "- Every ordering edge must reference existing step_id values.\n"
        "- ordering must be a list of edges [from_step_id, to_step_id].\n"
        "- For a total order [s1, s2, s3], write ordering as [[\"s1\", \"s2\"], [\"s2\", \"s3\"]].\n"
        "- For a single-step method, ordering may be [].\n"
        "- For zero-subtask methods, ordering must be [].\n"
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
    binding_hints = "\n".join(
        f'- {{"target_literal": "{literal}", "task_name": "<semantic_task_name>"}}'
        for literal in target_literals
    )
    few_shot_examples = """
Few-shot guidance (illustrative only, not your domain):

Example 1: Positive target with runtime primitive aliases
Target literal: delivered(pkg1)
Required target_task_bindings entry:
{"target_literal": "delivered(pkg1)", "task_name": "deliver_parcel"}
Because the first compound step references load_parcel, the final top-level
compound_tasks array must include BOTH deliver_parcel and load_parcel, and
the final methods array must include at least one method whose task_name is
load_parcel.
Valid method fragment:
{
  "method_name": "m_deliver_parcel_drop",
  "task_name": "deliver_parcel",
  "parameters": ["X1"],
  "context": [],
  "subtasks": [
    {
      "step_id": "s1",
      "task_name": "load_parcel",
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
Required target_task_bindings entry:
{"target_literal": "!door_open(room1)", "task_name": "keep_door_closed"}
Valid method fragment:
{
  "method_name": "m_keep_door_closed_noop",
  "task_name": "keep_door_closed",
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
        f"REQUIRED target_task_bindings ENTRIES:\n{binding_hints}\n\n"
        "TOP-LEVEL JSON SHAPE:\n"
        "Primitive tasks are injected automatically by the runtime, so only define "
        "target_task_bindings, compound_tasks, and methods.\n"
        f"{schema_hint}\n\n"
        f"{few_shot_examples}\n\n"
        "REQUIRED CONTENT RULES:\n"
        "- Include one target_task_bindings entry for every target literal shown above.\n"
        "- Each target_task_bindings.task_name must be a semantic task name, not a mechanical "
        "logic-name such as achieve_on or maintain_not_clear.\n"
        "- Each method.task_name must reference an existing compound task.\n"
        "- Each primitive subtask.task_name must match one of the provided runtime primitive action aliases exactly.\n"
        "- If the source action is written with hyphens, convert task_name to the provided underscore alias.\n"
        "- Each compound subtask.task_name must reference a compound task that exists in compound_tasks.\n"
        "- For every compound subtask.task_name X, include a compound_tasks entry named X.\n"
        "- For every compound subtask.task_name X, include at least one method whose task_name is X.\n"
        "- Do not reference helper tasks unless you fully define them in the same JSON object.\n"
        "- If you only need one extra primitive step, inline it instead of inventing an undefined helper.\n"
        "- Use semantic helper task names such as clear_top, remove_support, free_hand, load_parcel.\n"
        "- Do not use achieve_, maintain_not_, ensure_, or goal_ prefixes anywhere in compound task names.\n"
        "- Every task_name and method_name must already be underscore_case.\n"
        "- For every method, method_name must be exactly 'm_' + task_name + '_' + a short strategy suffix.\n"
        "- For zero-subtask methods, prefer a strategy suffix of noop or already_<state>.\n"
        "- Do not use any other method naming pattern.\n"
        "- Use parameter names already introduced by the task or method.\n"
        "- For a zero-subtask method, use a non-empty context, an empty subtasks list, and an empty ordering list.\n"
        "- For a non-zero-subtask method, put method-wide decomposition checks in context; "
        "put step-local checks in subtask.preconditions.\n"
        "- ordering is a list of [from_step_id, to_step_id] edges.\n"
        "- For a total order s1 -> s2 -> s3, write ordering as [[\"s1\", \"s2\"], [\"s2\", \"s3\"]].\n"
        "- For a single-step method, ordering may be [].\n"
        "- Do not invent unsupported top-level keys.\n"
        "- Do not omit required method keys or required subtask keys.\n"
        "- Use JSON null, not the string \"null\", for missing optional values.\n"
        "- Do not include explanations.\n"
        "\n"
        "FINAL SELF-CHECK BEFORE RETURNING JSON:\n"
        "1. Is there exactly one target_task_bindings entry for every target literal shown above?\n"
        "2. Are all task_name values semantic underscore_case identifiers?\n"
        "3. Did you avoid achieve_, maintain_not_, ensure_, and goal_ prefixes in compound task names?\n"
        "4. Do primitive steps use runtime aliases rather than raw source action names?\n"
        "5. Do all ordering edges reference real step_id values?\n"
        "6. Is every compound helper task declared in compound_tasks and implemented by at least one method?\n"
        "7. Do all methods use the exact m_task_strategy naming pattern?\n"
        "8. Do all zero-subtask methods keep subtasks/orderings empty and use a clear already-satisfied strategy name?\n"
        "9. Does the JSON contain only the requested keys?\n"
        "10. Would each task_name compile cleanly inside +!task_name(...) AgentSpeak syntax?\n"
    )
