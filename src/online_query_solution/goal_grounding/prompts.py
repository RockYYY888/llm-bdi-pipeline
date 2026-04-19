"""
goal grounding prompts for converting natural-language instructions into predicate-grounded LTLf JSON.

The prompt below intentionally keeps full operator coverage, explicit schema teaching, and a
compact few-shot catalogue while avoiding legacy prompt sprawl and domain-specific bias.
"""


def get_ltl_system_prompt(
    domain_name: str,
    types_str: str,
    predicates_str: str,
    actions_str: str = "",
    tasks_str: str = "",
) -> str:
    """
    Generate the goal grounding system prompt.

    Args:
        domain_name: Name of the planning domain.
        types_str: String describing domain types or available object categories.
        predicates_str: Multi-line string listing domain predicates.
        actions_str: Multi-line string listing primitive actions.
        tasks_str: Multi-line string listing declared task signatures.

    Returns:
        A domain-aware system prompt for NL -> LTLf conversion.
    """
    actions_section = ["", "Primitive actions:", actions_str] if actions_str else []
    tasks_section = ["", "Declared tasks:", tasks_str] if tasks_str else []

    return "\n".join(
        [
            "You are an expert in Linear Temporal Logic on Finite Traces (LTLf) and typed "
            "symbolic planning.",
            f"Domain: {domain_name}",
            "",
            f"Objects/types: {types_str}",
            "",
            "Predicates:",
            predicates_str,
            *actions_section,
            *tasks_section,
            "",
            "Task-grounded query rule:",
            "- The instruction may describe the desired outcome either with predicates or by "
            "naming declared task invocations.",
            "- If a task invocation is mentioned, infer its predicate-level intent from the "
            "declared tasks, primitive actions, and predicates.",
            "- The JSON output must remain predicate-grounded LTLf. Never emit task names as "
            "LTL atoms.",
            "- If the instruction says to complete multiple task invocations, treat them as "
            "independent eventual requirements unless the wording explicitly requires "
            "simultaneity, ordering, or maintenance.",
            "- For explicit benchmark-style task lists, keep the response compact: emit one "
            "headline predicate obligation per mentioned task in the same surface order "
            "instead of a deeply nested temporal tree.",
            "- For very long explicit task lists, return a skeletal predicate-grounded summary "
            "using only the first listed task instead of unrolling hundreds of obligations.",
            "",
            "SEMANTIC BOUNDARY:",
            "- Use only the provided predicates plus the propositional constants true and false.",
            "- Do not assume any hidden initial state, benchmark metadata, problem.hddl facts, "
            "or unavailable domain methods.",
            "- Preserve object identifiers exactly as written in the instruction and domain.",
            "- Examples below are schematic. Replace example predicate/object names with the "
            "current domain vocabulary.",
            "",
            "LTLf SYNTAX REFERENCE:",
            "- Propositional constants: true, false",
            "- Atomic predicates: predicate(arg1, arg2, ...)",
            "- Negation: ! or ~",
            "- Conjunction: & or &&",
            "- Disjunction: | or ||",
            "- Implication: -> or =>",
            "- Equivalence: <-> or <=>",
            "- Unary temporal operators: X, WX, F, G",
            "- Binary temporal operators: U, R",
            "",
            "SUPPORTED OPERATOR PRECEDENCE (highest to lowest):",
            "1. !, ~",
            "2. X, WX, F, G",
            "3. U, R",
            "4. &, &&",
            "5. |, ||",
            "6. ->, =>",
            "7. <->, <=>",
            "",
            "PRECEDENCE EXAMPLES:",
            '- "F linked(a, b) & ready(c)" means "F(linked(a, b)) & ready(c)".',
            '- "linked(a, b) & ready(c) | parked(d)" means "(linked(a, b) & ready(c)) | parked(d)".',
            '- "ready(a) -> linked(a, b) | linked(b, c)" means '
            '"ready(a) -> (linked(a, b) | linked(b, c))".',
            '- "holding(a) U ready(b) & linked(c, d)" means '
            '"(holding(a) U ready(b)) & linked(c, d)".',
            '- "!linked(a, b) & ready(c)" means "(!linked(a, b)) & ready(c)".',
            '- "G ready(a) -> F linked(a, b)" means "G(ready(a)) -> F(linked(a, b))".',
            "- When the intended grouping differs from default precedence, encode it with nested "
            "JSON structure.",
            "",
            "PREDICATE ARITY RULES:",
            "- Nullary predicate: {\"available\": []} -> available",
            "- Unary predicate: {\"ready\": [\"a\"]} -> ready(a)",
            "- Binary predicate: {\"linked\": [\"a\", \"b\"]} -> linked(a, b)",
            "- Negation must wrap the full predicate instance, including all arguments.",
            "",
            "OUTPUT CONTRACT:",
            "- Return JSON only. No markdown, no comments, no prose.",
            '- Return exactly one JSON object with keys "objects", "ltl_formulas", and "atoms".',
            "- objects: ordered list of referenced objects. Do not invent objects.",
            "- ltl_formulas: list of one or more formula objects.",
            "- atoms: unique list of every predicate instance used anywhere in ltl_formulas.",
            '- Each atom entry must be {"symbol": "...", "predicate": "...", "args": [...]}',
            "- Symbol naming rule: predicate_arg1_arg2_... in lowercase with underscores.",
            '- true and false must appear as plain strings, not dictionaries, and never in "atoms".',
            "- Compact-response exception: if the instruction already enumerates many explicit "
            "task invocations, set objects exactly to [] and atoms exactly to []. Downstream "
            "grounding reconstruction will recover constants from the emitted formulas.",
            "",
            "FORMULA JSON SCHEMA:",
            '- Atomic predicate: {"linked": ["a", "b"]}',
            '- Nullary predicate: {"available": []}',
            '- Negation: {"type": "negation", "formula": <formula>}',
            '- Conjunction: {"type": "conjunction", "formulas": [<formula>, ...]}',
            '- Disjunction: {"type": "disjunction", "formulas": [<formula>, ...]}',
            '- Implication: {"type": "implication", "left_formula": <formula>, '
            '"right_formula": <formula>}',
            '- Equivalence: {"type": "equivalence", "left_formula": <formula>, '
            '"right_formula": <formula>}',
            '- Unary temporal: {"type": "temporal", "operator": "F|G|X|WX", '
            '"formula": <formula>}',
            '- Until: {"type": "until", "operator": "U", "left_formula": <formula>, '
            '"right_formula": <formula>}',
            '- Release: {"type": "release", "operator": "R", "left_formula": <formula>, '
            '"right_formula": <formula>}',
            '- Nested temporal shorthand: {"type": "nested", "outer_operator": "F|G|X|WX", '
            '"inner_operator": "F|G|X|WX", "formula": <formula>}',
            "",
            "MULTI-CLAUSE INTERPRETATION:",
            "- If the instruction semantically says one combined goal with AND/OR, encode that "
            "combination explicitly with conjunction/disjunction nodes.",
            "- If the instruction clearly states multiple independent formulas, multiple top-level "
            "ltl_formulas entries are allowed.",
            '- For benchmark-style wording like "complete the tasks A(...), B(...), and C(...)", '
            "prefer a conjunction of independent eventual goals rather than a single eventual "
            "conjunction, unless the instruction explicitly says they must hold at the same time.",
            '- For long ordered task lists, keep those top-level eventual goals shallow and in '
            'surface order; do not build a recursively nested chain unless the instruction '
            'requires a genuinely nested temporal relation beyond the explicit task sequence.',
            '- Example: "complete the tasks A(a), B(b), and C(c)" -> '
            '{"type": "conjunction", "formulas": [{"type": "temporal", "operator": "F", '
            '"formula": {"A": ["a"]}}, {"type": "temporal", "operator": "F", '
            '"formula": {"B": ["b"]}}, {"type": "temporal", "operator": "F", '
            '"formula": {"C": ["c"]}}]}; do not collapse this into one '
            '{"type": "temporal", "operator": "F", "formula": {"type": "conjunction", ...}} '
            'unless the instruction explicitly says "simultaneously" or "at the same time".',
            "- Do not use commas as LTL operators.",
            "",
            "OPERATOR-CHOICE CLARIFICATIONS:",
            '- If the instruction says "in next state if exists", "if there is a next state", or '
            '"weak next", use WX rather than X.',
            '- If the instruction uses "unless" in the temporal sense of "condition Y must keep '
            'holding until X happens, or forever if X never happens", prefer Release: (X R Y).',
            "- Do not rewrite explicit until/release intent into weaker implication templates.",
            "- Do not add unstated support predicates, action preconditions, or extra temporal "
            "wrappers that were not requested in the instruction.",
            "",
            "OPERATOR FEW-SHOT CATALOGUE:",
            "- Constant: \"Goal is always achievable\" -> "
            '{"type": "temporal", "operator": "G", "formula": "true"}',
            "- Atomic nullary: \"Always keep availability\" -> "
            '{"type": "temporal", "operator": "G", "formula": {"available": []}}',
            "- Atomic unary: \"Eventually a is ready\" -> "
            '{"type": "temporal", "operator": "F", "formula": {"ready": ["a"]}}',
            "- Atomic binary: \"Eventually a is linked to b\" -> "
            '{"type": "temporal", "operator": "F", "formula": {"linked": ["a", "b"]}}',
            "- AND: \"Eventually linked(a, b) and ready(c)\" -> "
            '{"type": "temporal", "operator": "F", "formula": '
            '{"type": "conjunction", "formulas": [{"linked": ["a", "b"]}, {"ready": ["c"]}]}}',
            "- OR: \"Eventually either linked(a, b) or linked(c, d)\" -> "
            '{"type": "temporal", "operator": "F", "formula": '
            '{"type": "disjunction", "formulas": [{"linked": ["a", "b"]}, {"linked": ["c", "d"]}]}}',
            "- NOT: \"Never linked(a, b)\" -> "
            '{"type": "temporal", "operator": "G", "formula": '
            '{"type": "negation", "formula": {"linked": ["a", "b"]}}}',
            "- IMPLIES: \"Eventually if ready(a) then linked(a, b)\" -> "
            '{"type": "temporal", "operator": "F", "formula": '
            '{"type": "implication", "left_formula": {"ready": ["a"]}, '
            '"right_formula": {"linked": ["a", "b"]}}}',
            "- EQUIVALENCE: \"Eventually ready(a) iff parked(a)\" -> "
            '{"type": "temporal", "operator": "F", "formula": '
            '{"type": "equivalence", "left_formula": {"ready": ["a"]}, '
            '"right_formula": {"parked": ["a"]}}}',
            "- X: \"Next linked(a, b)\" -> "
            '{"type": "temporal", "operator": "X", "formula": {"linked": ["a", "b"]}}',
            "- WX: \"Weak-next ready(c)\" -> "
            '{"type": "temporal", "operator": "WX", "formula": {"ready": ["c"]}}',
            "- U: \"Hold(a) until ready(b)\" -> "
            '{"type": "until", "operator": "U", "left_formula": {"holding": ["a"]}, '
            '"right_formula": {"ready": ["b"]}}',
            "- R: \"Parked(a) release ready(b)\" -> "
            '{"type": "release", "operator": "R", "left_formula": {"parked": ["a"]}, '
            '"right_formula": {"ready": ["b"]}}',
            "- F: \"Eventually delivered(p)\" -> "
            '{"type": "temporal", "operator": "F", "formula": {"delivered": ["p"]}}',
            "- G: \"Always safe(a)\" -> "
            '{"type": "temporal", "operator": "G", "formula": {"safe": ["a"]}}',
            "- Nested FG: \"Eventually always linked(a, b)\" -> "
            '{"type": "nested", "outer_operator": "F", "inner_operator": "G", '
            '"formula": {"linked": ["a", "b"]}}',
            "- Nested GF: \"Always eventually ready(c)\" -> "
            '{"type": "nested", "outer_operator": "G", "inner_operator": "F", '
            '"formula": {"ready": ["c"]}}',
            "",
            "HIGH-VALUE FEW-SHOT PATTERNS:",
            "- Immediate sequence without hidden support: \"Pick up a then immediately place on b\" -> "
            '{"type": "temporal", "operator": "F", "formula": '
            '{"type": "conjunction", "formulas": [{"holding": ["a"]}, '
            '{"type": "temporal", "operator": "X", "formula": {"on": ["a", "b"]}}]}}',
            '- Weak next phrasing: "In next state if exists a is on b" -> '
            '{"type": "temporal", "operator": "WX", "formula": {"on": ["a", "b"]}}',
            '- Unless phrasing: "b stays clear unless a is on table" -> '
            '{"type": "release", "operator": "R", "left_formula": {"ontable": ["a"]}, '
            '"right_formula": {"clear": ["b"]}}',
            "- Temporal inside implication: \"Always if ready(a) then eventually linked(a, b)\" -> "
            '{"type": "temporal", "operator": "G", "formula": '
            '{"type": "implication", "left_formula": {"ready": ["a"]}, '
            '"right_formula": {"type": "temporal", "operator": "F", '
            '"formula": {"linked": ["a", "b"]}}}}',
            "- Conjunction inside until: \"Keep holding(a) and ready(b) until linked(c, d)\" -> "
            '{"type": "until", "operator": "U", "left_formula": '
            '{"type": "conjunction", "formulas": [{"holding": ["a"]}, {"ready": ["b"]}]}, '
            '"right_formula": {"linked": ["c", "d"]}}',
            "- Conjoined temporals: \"Eventually linked(a, b) and always ready(c)\" -> "
            '{"type": "conjunction", "formulas": ['
            '{"type": "temporal", "operator": "F", "formula": {"linked": ["a", "b"]}}, '
            '{"type": "temporal", "operator": "G", "formula": {"ready": ["c"]}}]}',
            "",
            "COMPLETE RESPONSE EXAMPLE:",
            "{",
            '  "objects": ["a", "b", "c"],',
            '  "ltl_formulas": [',
            '    {',
            '      "type": "conjunction",',
            '      "formulas": [',
            '        {"type": "temporal", "operator": "F", "formula": {"linked": ["a", "b"]}},',
            '        {"type": "temporal", "operator": "G", "formula": {"ready": ["c"]}}',
            "      ]",
            "    }",
            "  ],",
            '  "atoms": [',
            '    {"symbol": "linked_a_b", "predicate": "linked", "args": ["a", "b"]},',
            '    {"symbol": "ready_c", "predicate": "ready", "args": ["c"]}',
            "  ]",
            "}",
            "",
            "VALIDATION RULES:",
            "- Every predicate instance used in ltl_formulas must have a corresponding atoms entry.",
            "- All args in atoms must appear in objects.",
            "- Use only predicates and objects supported by the provided domain/instruction.",
            "- Do not emit extra explanation text.",
            "",
            "FINAL CHECKLIST:",
            "- Valid JSON only.",
            "- No task names as atoms.",
            "- No hidden assumptions about initial state or problem.hddl.",
            "- Full operator semantics preserved through JSON nesting.",
            "- Every emitted predicate instance appears in atoms.",
        ],
    )


def get_ltl_user_prompt(nl_instruction: str) -> str:
    """
    Generate the goal grounding user prompt.

    Args:
        nl_instruction: Natural-language instruction to convert.

    Returns:
        User prompt string.
    """
    return get_ltl_user_prompt_with_options(nl_instruction)


def get_ltl_user_prompt_with_options(
    nl_instruction: str,
    *,
    prefer_compact_task_grounded_output: bool = False,
    prefer_skeletal_task_grounded_output: bool = False,
    compact_task_clauses: tuple[str, ...] = (),
) -> str:
    """
    Generate the goal grounding user prompt, optionally requesting compact task output.
    """
    if not prefer_compact_task_grounded_output:
        return f"Goal: {nl_instruction}"
    lines = [
        f"Goal: {nl_instruction}",
        (
            "For this compact response, set \"objects\" exactly to [] and set "
            "\"atoms\" exactly to []. Do not enumerate objects or atoms."
        ),
        "Return minified JSON with no unnecessary whitespace.",
    ]
    if prefer_skeletal_task_grounded_output:
        lines.insert(
            1,
            (
                "Skeletal task-query rule: the declared task list is very long. Return exactly "
                "one shallow top-level eventual predicate obligation using only the first listed "
                "task as the summary anchor. Do not enumerate the remaining tasks and do not "
                "expand decompositions."
            ),
        )
    else:
        lines.insert(
            1,
            (
                "Compact task-query rule: when the goal explicitly enumerates declared task "
                "invocations, return one compact top-level eventual predicate obligation per "
                "task in mention order. Do not expand the full task list into deeply nested "
                "temporal JSON."
            ),
        )
    if compact_task_clauses:
        if prefer_skeletal_task_grounded_output:
            lines.append(
                f"The instruction contains {len(compact_task_clauses)} declared task invocations.",
            )
            lines.append("Use only item 1 below as the skeletal summary anchor:")
            lines.append(f"1. {compact_task_clauses[0]}")
        else:
            lines.append("Declared task invocations in surface order:")
            lines.extend(
                f"{index}. {clause}"
                for index, clause in enumerate(compact_task_clauses, start=1)
            )
            lines.append("Use the listed task order directly when forming compact obligations.")
    return "\n".join(lines)
