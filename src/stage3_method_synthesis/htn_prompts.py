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
        "Do an internal two-pass review before returning: first check JSON/schema validity, "
        "then check task-coverage completeness. Do not output that internal reasoning.\n"
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
        "- subtask.kind must be exactly 'primitive' or 'compound'. Never use 'guard' in subtasks.\n"
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
        "- Each target_task_binding must be semantically aligned with the bound literal. "
        "A positive literal must bind to a task whose methods make that predicate true. "
        "A negative literal must bind to a task whose methods remove, prevent, or keep the "
        "predicate false. Never bind !p(...) to a task that constructively establishes p(...).\n"
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
        "- Internally reason by explicit case partitioning before you emit JSON. For each target "
        "task and helper task, think through at least these families of cases: "
        "(a) already satisfied, "
        "(b) directly achievable with current support conditions, "
        "(c) blocked because a required support condition is missing and must be established first, "
        "(d) recursively blocked because the support-establishing helper itself has multiple cases. "
        "Do not output that reasoning; only output the final JSON.\n"
        "- Do not overfit to one witness plan or one canonical initial state.\n"
        "- Methods must stay reusable across multiple runtime situations, not just one example state.\n"
        "- For every target-bound task, include the important runtime branches the agent may face.\n"
        "- For every helper task that can already be satisfied at runtime, include an already-satisfied "
        "zero-subtask method with a non-empty context.\n"
        "- For every helper task that can be achieved in multiple common ways, include separate methods "
        "for those distinct ways when they are relevant (for example from_table vs from_block).\n"
        "- Do not invent an extra sibling method whose only difference is that it performs extra "
        "prerequisite helper steps under the same applicability conditions. If two constructive "
        "siblings would apply in the same state, merge the prerequisite helper work into the "
        "correctly-scoped strategy method instead of emitting a redundant clear_first/prepare_first branch.\n"
        "- Do not rely on later stages to invent missing helper-task branches.\n"
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
        "- Respect type discipline. If the domain says a position expects a block, vehicle, "
        "room, or package, do not reuse a variable of the wrong role there.\n"
        "- If two roles must stay distinct for the method to make sense, do not silently alias "
        "them to the same variable. Use separate variables and explicit binding conditions.\n"
        "- Every step args list must use only parameters already introduced by the method.\n"
        "- If a method needs an extra local helper variable beyond the task arguments, you may "
        "introduce it in method.parameters, but you must still connect it to the task arguments "
        "with a positive binding condition before first use.\n"
        "- Use stable upper-case variable names derived from the domain types whenever possible.\n"
        "- If a type appears once in a scope, use TYPE (for example BLOCK, ROOM, PACKAGE).\n"
        "- If the same type appears multiple times in one task or method scope, use TYPE1, TYPE2, "
        "TYPE3 in left-to-right semantic order.\n"
        "- Keep the task-level parameter names stable across all methods of that task.\n"
        "- Never introduce free variables in subtasks. Every upper-case placeholder used in "
        "subtask args, subtask literals, preconditions, or effects must already appear in the "
        "method parameters or be explicitly constrained in method.context or in a binding "
        "subtask.preconditions literal.\n"
        "- Constructive sibling methods for the same task must be semantically distinguishable by "
        "their promoted method-level context. Do not return multiple constructive siblings that all "
        "use the same empty or generic context.\n"
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
    type_lines = [f"- {type_name}" for type_name in domain.types]
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

Example 2: Negative target needs BOTH a constructive method and an already-satisfied method
Target literal: !clear(block2)
Required target_task_bindings entry:
{"target_literal": "!clear(block2)", "task_name": "keep_not_clear"}
Valid method fragments:
{
  "method_name": "m_keep_not_clear_stack",
  "task_name": "keep_not_clear",
  "parameters": ["BLOCK", "BLOCK1"],
  "context": [
    {"predicate": "holding", "args": ["BLOCK1"], "is_positive": true, "source_symbol": null}
  ],
  "subtasks": [
    {
      "step_id": "s1",
      "task_name": "put_on_block",
      "args": ["BLOCK1", "BLOCK"],
      "kind": "primitive",
      "action_name": "put-on-block",
      "literal": {"predicate": "clear", "args": ["BLOCK"], "is_positive": false, "source_symbol": null},
      "preconditions": [],
      "effects": []
    }
  ],
  "ordering": [],
  "origin": "llm"
}
{
  "method_name": "m_keep_not_clear_noop",
  "task_name": "keep_not_clear",
  "parameters": ["X2"],
  "context": [
    {"predicate": "clear", "args": ["X2"], "is_positive": false, "source_symbol": null}
  ],
  "subtasks": [],
  "ordering": [],
  "origin": "llm"
}

Example 3: Never bind a negative target to a task that establishes the opposite relation
Target literal: !linked(node1, node2)
INVALID target_task_bindings entry:
{"target_literal": "!linked(node1, node2)", "task_name": "link_nodes"}
Why invalid: link_nodes constructively makes linked(...) true, so it cannot represent !linked(...).
VALID target_task_bindings entry:
{"target_literal": "!linked(node1, node2)", "task_name": "unlink_nodes"}
Rule: if the literal is negated, the task name and its constructive methods must semantically
remove, undo, block, or keep false that same relation.
""".strip()

    return (
        "TASK:\n"
        "Generate a JSON HTN method library that will be compiled into valid AgentSpeak.\n"
        "The JSON must satisfy the syntax and naming contract below.\n\n"
        f"DOMAIN:\n{domain.name}\n\n"
        f"DOMAIN TYPES:\n{chr(10).join(type_lines) if type_lines else '- object'}\n\n"
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
        "- Think twice before returning: first verify the JSON shape, then verify task coverage. "
        "Do not print your reasoning.\n"
        "- For every negative target literal, include at least one constructive (non-zero-subtask) "
        "method for its bound task. Do not return only a noop/already-satisfied method.\n"
        "- For every negative target literal, bind it to a task that semantically makes the negated "
        "relation true (for example remove_on, detach_block, keep_not_clear). Never bind a negative "
        "literal to a task whose constructive branch makes the positive relation true.\n"
        "- Invalid pattern: binding !p(...) to a task like add_p, place_p, connect_p, or any other "
        "task whose constructive branch establishes p(...).\n"
        "- For every target-bound task, do not stop at one witness path. Include the key reusable "
        "branches needed by the agent across different runtime situations.\n"
        "- Think in explicit case splits before returning. For each target-bound task, first "
        "mentally enumerate: already-satisfied, directly-achievable, blocked-because-support-is-missing, "
        "and recursively-blocked helper cases. Then emit methods that cover those reusable cases. "
        "Do not print the reasoning.\n"
        "- Each method.task_name must reference an existing compound task.\n"
        "- Each primitive subtask.task_name must match one of the provided runtime primitive action aliases exactly.\n"
        "- If subtask.task_name is a runtime primitive action alias, then kind must be 'primitive'.\n"
        "- If subtask.task_name is a compound helper task, then kind must be 'compound'.\n"
        "- If the source action is written with hyphens, convert task_name to the provided underscore alias.\n"
        "- Each compound subtask.task_name must reference a compound task that exists in compound_tasks.\n"
        "- For every compound subtask.task_name X, include a compound_tasks entry named X.\n"
        "- For every compound subtask.task_name X, include at least one method whose task_name is X.\n"
        "- Do not reference helper tasks unless you fully define them in the same JSON object.\n"
        "- If you only need one extra primitive step, inline it instead of inventing an undefined helper.\n"
        "- Use semantic helper task names such as clear_top, remove_support, free_hand, load_parcel.\n"
        "- Do not overfit methods to the default all-objects-on-table example state. The methods must "
        "remain reusable when an object is already held, already clear, already on another object, or "
        "already at the target relation.\n"
        "- Use the declared HDDL types and typed action signatures exactly. Method parameters, helper "
        "variables, and subtask arguments must respect the domain's type roles.\n"
        "- Do not collapse two distinct semantic roles onto one variable unless the domain semantics "
        "really allow that aliasing.\n"
        "- If a decomposition needs two different objects, keep two distinct variables and bind them "
        "explicitly; do not fake distinctness by reusing one variable name.\n"
        "- If a task changes the state of object X because another object is blocking, supporting, "
        "or occupying X, introduce and bind the actual blocker/support object as a separate variable. "
        "Do not incorrectly reuse X where the domain action really operates on a distinct related object.\n"
        "- If your method logic depends on two objects being different, encode that with safe symbolic "
        "binding conditions that the downstream pipeline can represent. Do not rely on unsupported "
        "equality or inequality syntax.\n"
        "- For every helper task that denotes a reusable stateful intention (for example hold_block, "
        "clear_top, remove_on, place_on, make_clear), include both:\n"
        "  (a) at least one constructive method, and\n"
        "  (b) an already-satisfied zero-subtask method with a non-empty context whenever that task can "
        "already hold at runtime.\n"
        "- If a helper task has multiple common acquisition modes that matter at runtime (for example "
        "from_table and from_block), include separate methods for each relevant mode.\n"
        "- Those sibling strategy methods must be distinguishable by reusable method-level context. "
        "For example, a from_table branch should expose a context such as ontable(BLOCK), while a "
        "from_block branch should expose a context such as on(BLOCK1, BLOCK2).\n"
        "- Do not create a generic clear_first or prepare_first sibling unless it has its own real "
        "method-level applicability condition that is different from every other sibling. Extra helper "
        "steps alone are not a valid reason to add another sibling.\n"
        "- Invalid pattern: one sibling says acquire, another says stack, but acquire has no unique "
        "context and merely performs prerequisite work for stack. Merge them instead of emitting both.\n"
        "- Do not assume PANDA, the renderer, or any later stage will synthesize missing branches for you.\n"
        "- Do not use achieve_, maintain_not_, ensure_, or goal_ prefixes anywhere in compound task names.\n"
        "- Every task_name and method_name must already be underscore_case.\n"
        "- For every method, method_name must be exactly 'm_' + task_name + '_' + a short strategy suffix.\n"
        "- For zero-subtask methods, prefer a strategy suffix of noop or already_<state>.\n"
        "- Do not use any other method naming pattern.\n"
        "- Use parameter names already introduced by the task or method.\n"
        "- Do not invent free variables such as TOP, SUPPORT, X, or Y unless they are already "
        "task parameters or are explicitly constrained in method.context or by a binding "
        "subtask.preconditions literal.\n"
        "- If you introduce an extra local variable in method.parameters (for example a blocker or "
        "support object), do not leave it floating. Add a positive binding relation in method.context "
        "before any subtask uses it.\n"
        "- Do not rename an existing task parameter to a fresh variable. If the task parameter "
        "is B, B1, B2, PKG, ROOM, or similar, keep using that same variable name instead of "
        "switching to X, Y, OBJ, or another alias.\n"
        "- Prefer type-based upper-case variable names in the final JSON. For example, use BLOCK "
        "or BLOCK1/BLOCK2 instead of random X/Y placeholders.\n"
        "- If a helper method needs a local helper variable, bind it in method.context first "
        "or by an explicit binding precondition (for example a predicate relating that variable "
        "to an existing task parameter) before reusing it in subtasks.\n"
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
        "8. Does every negative target task have at least one constructive non-zero-subtask method?\n"
        "9. Do all zero-subtask methods keep subtasks/orderings empty and use a clear already-satisfied strategy name?\n"
        "10. Does the JSON contain only the requested keys?\n"
        "11. Would each task_name compile cleanly inside +!task_name(...) AgentSpeak syntax?\n"
        "12. For every helper task that can already be true at runtime, did you include an already-satisfied "
        "zero-subtask method with a non-empty context?\n"
        "13. For every helper task with multiple common runtime modes, did you include the distinct relevant methods?\n"
        "14. Did you avoid overfitting to one canonical initial state or one witness plan?\n"
        "15. Did you avoid every unbound free variable in subtasks by binding it in the task "
        "parameters or method.context first?\n"
        "16. Did you explicitly cover the already-satisfied, direct, blocked, and recursive-helper "
        "case families for each top-level target task?\n"
        "17. Did every variable and subtask argument respect the domain's declared types?\n"
        "18. Did you avoid unsupported equality/inequality constructs and instead use representable "
        "binding conditions only?\n"
        "19. Did you keep existing task parameters stable instead of renaming them to fresh "
        "single-letter aliases such as X or Y?\n"
        "20. Did every constructive sibling method for the same task expose a distinguishable "
        "method-level context rather than sharing the same generic context?\n"
        "21. Did you use stable upper-case type-based variable names such as BLOCK or BLOCK1/BLOCK2 "
        "instead of random placeholders?\n"
        "22. Did every target_task_binding use a task whose semantics match the target literal's "
        "polarity, especially for negative literals?\n"
        "23. Did you avoid adding redundant clear_first/prepare_first siblings that do not introduce "
        "a new applicability condition?\n"
    )
