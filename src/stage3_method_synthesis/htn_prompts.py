"""
Prompt builders for HTN method synthesis.
"""

from __future__ import annotations

from typing import Any, Iterable

from utils.hddl_condition_parser import HDDLConditionParser


def _sanitize_name(name: str) -> str:
    return name.replace("-", "_")


def _literal_pattern_signature(literal: Any) -> str:
    atom = (
        literal.predicate
        if not literal.args
        else f"{literal.predicate}({', '.join(literal.args)})"
    )
    return atom if literal.is_positive else f"not {atom}"


def _clause_signature(clause: Iterable[Any]) -> str:
    parts = [_literal_pattern_signature(item) for item in clause]
    return " & ".join(parts) if parts else "true"


def build_htn_system_prompt() -> str:
    return (
        "You generate a query-specific HTN method library for a symbolic planning domain.\n"
        "The output is compiled directly into Jason AgentSpeak.\n"
        "Return JSON only. No markdown, no comments, no prose.\n"
        "\n"
        "OUTPUT CONTRACT:\n"
        "- Return exactly one JSON object with exactly these top-level keys: "
        "target_task_bindings, compound_tasks, methods.\n"
        "- Prefer reusable parameterized compound tasks over one grounded task per target instance.\n"
        "- Do not encode grounded object constants into compound task names; use parameters instead.\n"
        "- Never use legacy task prefixes achieve_, ensure_, goal_, or maintain_not_. Choose semantic reusable names instead.\n"
        "- task_name and method_name must match [a-z][a-z0-9_]*.\n"
        "- context, literal, preconditions, and effects entries must be JSON literal objects, never strings like \"clear(X)\".\n"
        "- method_name must be exactly m_{task_name}_{strategy}.\n"
        "- Primitive subtasks must use the provided runtime primitive aliases.\n"
        "- Every compound subtask must reference a declared compound task.\n"
        "- Zero-subtask methods must have non-empty context and empty subtasks/orderings.\n"
        "- Do not emit duplicate bindings, duplicate compound tasks, or semantically duplicate sibling methods.\n"
        "PRIORITY ORDER: validity of JSON and bindings > executability of methods > semantic alignment > branch coverage.\n"
        "If branch coverage conflicts with executability, drop the invalid branch.\n"
        "\n"
        "HIDDEN REASONING RECIPE (DO NOT OUTPUT IT):\n"
        "1. Bind each target literal to one reusable semantic target-facing task pattern; reuse the same parameterized task across same-semantic targets.\n"
        "2. For each target-facing task, identify every non-equivalent final primitive action or final helper that can make the target literal true.\n"
        "3. List that final step's important positive support preconditions.\n"
        "4. If several different final actions can establish the same literal under different source-state modes, emit distinguishable constructive siblings for those modes.\n"
        "5. Build methods least-to-most: already-satisfied, direct-prepared, then blocked branches for missing supports.\n"
        "6. If the final step needs two important supports, explicitly consider missing-first, missing-second, and missing-both.\n"
        "7. Helper coverage is not a substitute for target-task coverage: helper modes for support_a do not remove the need for target-level missing-support and missing-both siblings.\n"
        "8. Keep a missing-one-support branch only when the already-satisfied support can remain true while the helper establishes the missing support; otherwise drop that branch or add explicit discharge/re-establish steps.\n"
        "9. For helper tasks that remove blockers or establish clearance, cover both direct-blocker and recursively-blocked-blocker cases when relevant.\n"
        "10. If a helper's direct branch depends on a positive support literal q(BLOCKER, ...) and the branch context does not guarantee q(BLOCKER, ...), the recursive blocked-blocker sibling is mandatory.\n"
        "11. If a missing support has a reusable helper meaning, call that helper explicitly.\n"
        "12. If helpers and later steps compete for the same execution resource, run resource-demanding helpers first or discharge/reacquire explicitly.\n"
        "13. Also check helper after-effects: if a helper likely consumes a resource needed by the next step, add the needed discharge/recovery step or reject that branch.\n"
        "14. A missing-support branch is invalid if its current context already blocks the first helper, unless the branch explicitly discharges and later re-establishes the consumed support.\n"
        "15. Prefer reusable helper end states over transient ones: if a helper can keep its headline literal true while discharging an unneeded blocker/resource, add that discharge so downstream helpers remain executable.\n"
        "16. Reject any helper cycle that can bounce on the same arguments without a primitive action, a stricter context split, or a shrinking blocker.\n"
        "17. Do a silent variable-binding audit and a silent dry-run from a blocked state before returning JSON.\n"
        "\n"
        "SEMANTIC RULES:\n"
        "- A positive literal must bind to a task whose constructive methods make that literal true.\n"
        "- A negative literal must bind to a task whose constructive methods remove, prevent, or keep false that relation.\n"
        "- If a compound subtask carries a headline literal p(...), its task_name must semantically align with establishing p(...). Do not use an unrelated task name as a proxy.\n"
        "- Disjunctive action applicability from or / imply must become distinguishable sibling methods.\n"
        "- imply(A, B) must be treated as (not A) or B.\n"
        "- Use only variables introduced by the task or method; no free variables.\n"
        "- Never invent a disposer/resource variable for a primitive step unless it is already bound in context or parameters.\n"
        "- not handempty alone does not identify what is being held; if no carried object is bound, recovery via put_down must not be emitted.\n"
        "- Respect declared types and role distinctions.\n"
        "- Never reveal chain-of-thought or hidden reasoning.\n"
    )


def build_htn_user_prompt(domain: Any, target_literals: Iterable[str], schema_hint: str) -> str:
    parser = HDDLConditionParser()
    action_lines = []
    action_branch_hint_lines = []
    for action in domain.actions:
        params = ", ".join(action.parameters) if action.parameters else "none"
        action_lines.append(
            f"- runtime task: {_sanitize_name(action.name)} | source action: {action.name}"
            f"({params}) | pre: {action.preconditions} | eff: {action.effects}"
        )
        try:
            parsed_action = parser.parse_action(action)
        except Exception:
            continue
        if len(parsed_action.precondition_clauses) <= 1:
            continue
        clauses = " OR ".join(
            f"[{_clause_signature(clause)}]"
            for clause in parsed_action.precondition_clauses
        )
        action_branch_hint_lines.append(
            f"- {_sanitize_name(action.name)} applicability branches: {clauses}"
        )

    predicate_lines = [f"- {predicate.to_signature()}" for predicate in domain.predicates]
    task_lines = [f"- {task.to_signature()}" for task in getattr(domain, "tasks", [])]
    type_lines = [f"- {type_name}" for type_name in domain.types]
    targets = "\n".join(f"- {item}" for item in target_literals)
    branch_hints = (
        "\n".join(action_branch_hint_lines)
        if action_branch_hint_lines
        else "- none"
    )
    binding_hints = "\n".join(
        f'- {{"target_literal": "{literal}", "task_name": "<semantic_task_name>"}}'
        for literal in target_literals
    )
    protocol = """
MANDATORY TARGET-TASK CONSTRUCTION PROTOCOL:
1. Choose one semantic target-facing task per target literal.
   Reuse one parameterized task when several targets share the same constructive semantics.
2. Identify the final step that realizes the literal.
3. List the final step's important positive support preconditions.
4. Cover these branch families unless impossible:
   - already-satisfied
   - direct-prepared
   - missing-first-support
   - missing-second-support
   - missing-both-support
5. Keep a missing-one-support branch only when the already-satisfied support can stay true while the helper establishes the missing support; otherwise drop that branch or add explicit discharge/re-establish steps.
6. For blocker-removal helpers, cover both direct-blocker and recursively-blocked-blocker cases when relevant.
7. If a helper's direct branch depends on a positive support literal about a blocker/support object and the branch context does not guarantee it, emit the recursive blocked-blocker sibling. This is mandatory.
8. If a support predicate has a reusable helper meaning, define or call that helper.
9. If several different final primitive actions can establish the same literal under different source-state modes, emit distinct constructive siblings for those modes instead of choosing only one action schema.
10. Helper coverage is not a substitute for target-task coverage: helper modes for one support do not remove the need for target-level missing-support and missing-both siblings.
11. If a helper subtask is meant to establish literal p(...), choose a task name whose semantics match p(...), not an unrelated proxy task.
12. If a helper needs a free resource that a later step would consume, run the helper first or discharge/reacquire explicitly.
13. If a helper likely leaves behind a resource conflict for the next step, add an explicit recovery step or reject that branch.
14. A missing-support branch is invalid if its current context already blocks its first helper, unless it explicitly discharges and later re-establishes the consumed support.
15. Prefer reusable helper end states over transient ones: if a helper can keep its headline literal true while discharging an unneeded blocker/resource, add that discharge.
16. Reject no-progress mutual recursion.
""".strip()
    few_shots = """
Few-shot guidance (illustrative only):

Example A: branch families for a positive target
Target literal: on(item1, slot1)
If the final primitive place_item(ITEM, SLOT) needs holding(ITEM) and clear(SLOT), a reusable target-facing task should cover:
- already: on(ITEM, SLOT)
- direct: holding(ITEM) & clear(SLOT) -> place_item
- missing_holding: clear(SLOT) -> hold_item(ITEM); place_item
- missing_clear: holding(ITEM) -> clear_slot(SLOT); place_item only if resource ordering stays executable
- missing_both: establish both supports in an executable order, then place_item
Invalid pattern: only {already, direct} or only one arbitrary blocked branch.
Invalid pattern: creating one grounded task per target instance when one parameterized task would cover them all.

Example B: blocked-clear resource conflict
If clear_slot(SLOT) needs handempty, this is INVALID:
- context holding(ITEM) & not clear(SLOT)
- subtasks: clear_slot(SLOT); place_item(ITEM, SLOT)
Valid pattern: clear SLOT first, then acquire ITEM; or discharge and reacquire only when the carried object is already bound in context.
Invalid pattern: inventing put_down(Z) or another unbound disposer variable just to regain handempty.

Example E: post-helper resource conflict
If clear_slot(SLOT) works by unstacking BLOCKER from SLOT, it may leave holding(BLOCKER) and not handempty.
Therefore this is INVALID when pick_up(ITEM) needs handempty:
- clear_slot(SLOT); pick_up(ITEM)
Valid pattern: clear_slot(SLOT); put_down(BLOCKER); pick_up(ITEM) only when BLOCKER is already bound and the recovery step is explicit.

Example F: invalid missing-second-support branch
If the final step needs support_a and support_b, and the current branch context already satisfies support_a by consuming a scarce resource, then this is INVALID:
- context support_a & not support_b
- subtasks: helper_for_support_b; final_step
unless the branch explicitly discharges and later re-establishes support_a.
Preferred repair: omit that missing-second-support branch and let missing-both or an explicit discharge/re-establish branch cover those states.

Example G: do not preserve a conflicting support
If stack(X, Y) needs holding(X) and clear(Y), and clear_block(Y) needs handempty, then this is INVALID:
- context holding(X) & not clear(Y)
- subtasks: clear_block(Y); stack(X, Y)
because the already-satisfied support holding(X) cannot remain true while clear_block(Y) is established.
Valid options:
- omit this missing_clear branch and let missing_both cover it, or
- explicitly discharge and later re-establish holding(X).

Example G2: explicit repaired missing_clear branch
Only emit the missing_clear sibling if the full recovery is explicit and executable, for example:
- context holding(X) & not clear(Y)
- subtasks: put_down(X); make_clear(Y); make_holding(X); stack(X, Y)
If you do not have that full discharge/re-establish chain, omit the missing_clear sibling entirely.

Example H: underspecified recovery branch
If pick_up(X) needs handempty and the branch context is:
- clear(X) & ontable(X) & not handempty
then this is UNDERSPECIFIED unless the carried object is already named in context or parameters.
Invalid pattern:
- put_down(Z); pick_up(X)
Valid pattern:
- omit the missing_handempty branch, or
- use put_down(BLOCKER); pick_up(X) only when BLOCKER is already bound.

Example I: reusable helper end state
If clear_block(X) is achieved by unstack(BLOCKER, X), the helper may leave holding(BLOCKER).
If clear(X) remains true after put_down(BLOCKER), prefer:
- unstack(BLOCKER, X); put_down(BLOCKER)
over a transient helper ending in holding(BLOCKER), because downstream helpers often need handempty.

Example J: omit underspecified make_holding branch
If make_holding(X) has a direct branch:
- clear(X) & ontable(X) & handempty -> pick_up(X)
then this branch is INVALID:
- clear(X) & ontable(X) & not handempty -> put_down(Z); pick_up(X)
because the carried object is unknown.
Preferred behavior: omit the missing_handempty sibling unless the carried object is already bound.

Example J2: same literal, different final actions
If holding(X) can be established by two different primitives:
- pick_up(X) from clear(X) & ontable(X) & handempty
- unstack(X, Y) from on(X, Y) & clear(X) & handempty
then make_holding(X) must not choose only one of them.
Valid pattern:
- table-mode sibling for ontable(X)
- stack-mode sibling for on(X, Y)
- plus blocked variants only when their missing support can be repaired executably
Invalid pattern:
- only the table-mode sibling
because holding(X) then fails whenever X starts on another block.

Example J3: helper coverage does not replace target coverage
If place_item(X, Y) needs support_a(X) and support_b(Y), and helper support_a(X) has several constructive modes, the target task still needs its own blocked siblings.
Valid pattern:
- direct target branch when support_a and support_b already hold
- target missing-support_a branch that calls helper support_a(X)
- target missing-both branch that establishes both supports in order
Invalid pattern:
- only direct target branch, plus a rich helper support_a library
because the target still fails whenever support_a or support_b is initially missing.

Example K: exact clear-helper recursion pattern
If a clear-like helper has this direct branch:
- context on(B, X) & clear(B) & handempty
- subtasks unstack(B, X); put_down(B)
then it must also have the blocked-blocker sibling:
- context on(B, X) & not clear(B)
- subtasks make_clear(B); unstack(B, X); put_down(B)
Invalid pattern:
- only already + direct
because make_clear(X) then fails whenever the blocker on X is itself blocked.

Example C: OR / IMPLY must become sibling methods
If an action is applicable under [clear(BLOCK)] OR [holding(BLOCK)], emit distinguishable siblings for those contexts.
If an action has imply(clear(BLOCK), holding(BLOCK)), treat it as [not clear(BLOCK)] OR [holding(BLOCK)].

Example D: recursive blocker removal
If helper make_clear(TARGET) can succeed by unstacking BLOCKER from TARGET, then it must also consider the case where BLOCKER itself is not ready for that unstack step.
Valid pattern:
- direct-blocker: on(BLOCKER, TARGET) & clear(BLOCKER) & handempty -> unstack(BLOCKER, TARGET); put_down(BLOCKER)
- recursive-blocker: on(BLOCKER, TARGET) & not clear(BLOCKER) -> make_clear(BLOCKER); unstack(BLOCKER, TARGET); put_down(BLOCKER)
- complementary-support rule: if the direct helper branch requires support q(BLOCKER), add the sibling for not q(BLOCKER) whenever that blocked state can arise at runtime.
Invalid pattern: only the already-satisfied branch and the direct-blocker branch. That leaves the helper unusable when the blocker is itself blocked.
""".strip()

    return (
        "TASK:\n"
        "Generate one JSON HTN method library that compiles into valid AgentSpeak.\n\n"
        f"DOMAIN:\n{domain.name}\n\n"
        f"DOMAIN TYPES:\n{chr(10).join(type_lines) if type_lines else '- object'}\n\n"
        f"DECLARED DOMAIN TASKS:\n{chr(10).join(task_lines) if task_lines else '- none declared'}\n\n"
        f"PREDICATES:\n{chr(10).join(predicate_lines)}\n\n"
        f"RUNTIME PRIMITIVE ACTION ALIASES:\n{chr(10).join(action_lines)}\n\n"
        f"ACTION PRECONDITION BRANCH HINTS (DNF):\n{branch_hints}\n\n"
        f"TARGET LITERALS:\n{targets}\n\n"
        f"REQUIRED target_task_bindings ENTRIES:\n{binding_hints}\n\n"
        "DECISION PRIORITY:\n"
        "1. Valid JSON and bound variables.\n"
        "2. Executable methods under their own contexts.\n"
        "3. Semantic task naming.\n"
        "4. Broad branch coverage.\n"
        "If a candidate sibling is not executable, omit it even if branch coverage becomes smaller.\n\n"
        f"{protocol}\n\n"
        "TOP-LEVEL JSON SHAPE:\n"
        "Primitive tasks are injected automatically by the runtime, so only define target_task_bindings, compound_tasks, and methods.\n"
        f"{schema_hint}\n\n"
        f"{few_shots}\n\n"
        "REQUIRED CONTENT RULES:\n"
        "- Prefer declared domain task names when they match the target-facing intention, but generate fresh methods yourself; do not assume any pre-existing method body exists.\n"
        "- Reuse one parameterized compound task for repeated goal patterns; do not clone a new grounded task for each target literal instance.\n"
        "- Never use legacy task prefixes achieve_, ensure_, goal_, or maintain_not_ in compound task names.\n"
        "- Do not invent grounded task names like achieve_p_a_b; task names must stay semantic and reusable.\n"
        "- Every target_task_bindings.task_name must match a declared compound task in compound_tasks.\n"
        "- For every target-bound task, cover reusable blocked states, not only the already-prepared endgame state.\n"
        "- For a direct branch with two important support preconditions, missing-both coverage is mandatory unless provably impossible.\n"
        "- Emit a missing-first or missing-second branch only when the support already true in that branch can remain true while the missing support is established.\n"
        "- For a place-on style target, do not emit `holding(X) & not clear(Y) -> clear_helper(Y); final_step` unless you also emit the full discharge/re-establish chain.\n"
        "- For blocker-removal helpers, do not stop at the direct-blocker case if the blocker itself may be unprepared. Add the recursive-blocker branch when relevant.\n"
        "- If a direct blocker-removal branch requires clear(BLOCKER) or an analogous readiness literal and the branch context does not guarantee it, the recursive blocked-blocker sibling is mandatory.\n"
        "- A clear-like helper is incomplete if it only has already-satisfied and direct-blocker methods while the blocker itself may be blocked.\n"
        "- For clear-like helpers, a direct branch `on(B, X) & clear(B) & handempty` should normally be paired with a blocked-blocker sibling `on(B, X) & not clear(B)`.\n"
        "- If multiple primitive actions can establish the same headline literal under different source-state modes, emit separate constructive siblings for those modes.\n"
        "- For make_holding-style helpers in blocks-like domains, cover both table-mode acquisition and stack-mode acquisition when both exist.\n"
        "- Helper coverage does not replace target-task coverage: a rich helper library does not remove target-level missing-support and missing-both branches.\n"
        "- If a compound helper subtask is annotated with literal p(...), the helper task itself must semantically establish p(...). Do not use an unrelated task name as a proxy for that literal.\n"
        "- Every context entry and every literal-bearing field must use object form with predicate/args/is_positive, not string shorthand.\n"
        "- Do not create clear_first / prepare_first siblings unless they have their own real applicability condition.\n"
        "- Do not use unbound disposer variables such as a made-up held object just to regain a resource; only call a primitive when all of its arguments are already bound.\n"
        "- If a branch needs handempty but the carried object is not named in context or parameters, do not emit a put_down-style recovery step for an unknown object.\n"
        "- If a helper likely leaves holding(BLOCKER) or otherwise consumes a resource needed by the very next step, add the explicit recovery step or reject that branch.\n"
        "- If a missing-support branch starts in a context that already blocks its first helper, either add explicit discharge/re-establish steps or drop that branch.\n"
        "- Prefer dropping an invalid missing-second-support branch over emitting a branch that only works by assuming a helper can ignore the current resource conflict.\n"
        "- If not handempty is true but the carried object is not bound anywhere, treat recovery as underspecified and omit that branch.\n"
        "- For make_holding-style helpers, do not emit a missing_handempty sibling unless the carried object for the recovery step is already bound.\n"
        "- When a helper can preserve its headline literal and also restore a consumed resource, prefer that reusable end state over a transient end state.\n"
        "- Do not introduce free variables. Bind every variable before use.\n"
        "- Do not output chain-of-thought. Output final JSON only.\n\n"
        "FINAL SILENT CHECKLIST:\n"
        "1. Exactly one binding per target literal.\n"
        "2. JSON keys and identifier format are valid.\n"
        "3. No undefined helper tasks.\n"
        "4. No free variables.\n"
        "5. No no-progress recursion.\n"
        "6. Missing-first / missing-second / missing-both support cases considered when relevant.\n"
        "7. Shared-resource ordering is executable.\n"
        "8. Return one complete JSON object and nothing else.\n"
    )
