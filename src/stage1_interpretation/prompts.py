"""
Stage 1: Natural Language to LTLf Prompts

Contains prompt templates for converting natural language instructions
to Linear Temporal Logic on Finite Traces (LTLf) specifications.

**LTLf Syntax Reference**: http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax

Supported Syntax (in order):

1. Propositional Symbols:
   - true, false (constants)
   - [a-z][a-z0-9_]* (atomic propositions)

2. Boolean Operators:
   - & or && (And)
   - | or || (Or)
   - ! or ~ (Not)
   - -> or => (Implication)
   - <-> or <=> (Equivalence)

3. Future Temporal Operators:
   - X (Next)
   - WX (WeakNext)
   - U (Until)
   - R (Release)
   - F (Eventually)
   - G (Always)
"""


def get_ltl_system_prompt(domain_name: str, types_str: str, predicates_str: str, actions_str: str = "") -> str:
    """
    Generate system prompt for NL -> LTLf conversion

    Args:
        domain_name: Name of the planning domain (e.g., "Blocksworld")
        types_str: String describing domain types (e.g., "blocks (e.g., a, b, c)")
        predicates_str: Multi-line string listing available predicates with signatures
        actions_str: Multi-line string listing available actions with preconditions and effects

    Returns:
        System prompt string with domain-specific information
    """
    actions_section = f"\n\nActions:\n{actions_str}" if actions_str else ""

    return f"""You are an expert in Linear Temporal Logic on Finite Traces (LTLf) and BDI agent systems.

Domain: {domain_name}

Objects: {types_str}

Predicates:
{predicates_str}{actions_section}

**Complete LTLf Syntax (use in this order):**

**1. Propositional Symbols:**
- true: Propositional constant (always true)
- false: Propositional constant (always false)
- Atomic propositions: predicates from domain (e.g., on(a,b), clear(c), handempty)

**2. Boolean Operators:**
- & or &&: AND (conjunction)
- | or ||: OR (disjunction)
- ! or ~: NOT (negation)
- -> or =>: IMPLIES (implication)
- <-> or <=>: EQUIVALENCE (if and only if)

**3. Future Temporal Operators:**
- X: Next (strong next state)
- WX: WeakNext (next state or no next state exists)
- U: Until (φ U ψ - φ holds until ψ)
- R: Release (φ R ψ - ψ holds until φ)
- F: Eventually/Finally (will be true at some future point)
- G: Globally/Always (true at all future points)

Your task: Convert natural language to LTLf formulas following the official syntax.

**Examples:**

**Propositional Symbols:**
1. Atomic: "Block a is on b" → on(a, b)
2. Constant: "Goal is always achievable" → G(true)

**Boolean Operators:**
3. AND: "A is on B and C is clear" → F(on(a, b) & clear(c))
4. OR: "Either A on B or C on D" → F(on(a, b) | on(c, d))
5. NOT: "Never put A on B" → G(!(on(a, b)))
6. IMPLIES: "If A clear then put A on B" → F(clear(a) -> on(a, b))
7. EQUIVALENCE: "A clear iff A on table" → F(clear(a) <-> ontable(a))

**Future Temporal Operators:**
8. X (Next): "Pick A then immediately place on B" → F(holding(a)), X(on(a, b))
9. WX (WeakNext): "In next state if exists, A on B" → WX(on(a, b))
10. U (Until): "Hold A until B is clear" → (holding(a) U clear(b))
11. R (Release): "B stays clear unless A on table" → (ontable(a) R clear(b))
12. F (Eventually): "Put A on B" → F(on(a, b))
13. G (Always): "Always keep C clear" → G(clear(c))

**Nested:**
14. "Eventually ensure A always on B" → F(G(on(a, b)))
15. "Keep trying to clear C" → G(F(clear(c)))

**Complex:**
16. "Eventually (A on B and C clear) or D on table" → F((on(a,b) & clear(c)) | ontable(d))
17. "Always: if A clear then eventually A on B" → G(clear(a) -> F(on(a,b)))

**Natural Language Patterns:**
- Propositional: "true", "false", predicates
- Boolean: "and" (& ), "or" (|), "not" (!), "if...then" (->), "iff" (<->)
- Temporal: "next" (X), "weak next" (WX), "until" (U), "unless" (R), "eventually" (F), "always" (G)

**CRITICAL: Predicate Argument Rules**

Predicates have different arities (number of arguments). You MUST include the correct arguments:

1. **Nullary predicates (0 arguments)**: NO arguments, NO parentheses in formula field
   - Example: handempty
   - JSON: {{{{"handempty": []}}}}
   - Output formula: G(handempty) or F(handempty)

2. **Unary predicates (1 argument)**: ONE argument in list
   - Examples: clear(a), holding(b)
   - JSON: {{{{"clear": ["a"]}}}}, {{{{"holding": ["b"]}}}}
   - Output formula: F(clear(a)), G(holding(b))

3. **Binary predicates (2 arguments)**: TWO arguments in list
   - Example: on(a, b)
   - JSON: {{{{"on": ["a", "b"]}}}}
   - Output formula: F(on(a, b))

4. **Negation**: ALWAYS include FULL predicate with ALL arguments
   - CORRECT: {{{{"type": "negation", "formula": {{{{"on": ["a", "b"]}}}}}}}}
   - WRONG: {{{{"not": ["on"]}}}} ❌ Missing arguments!
   - Output formula: G(!(on(a, b)))

**IMPORTANT**: Do NOT assume or specify any initial state. The generated plans must work from ANY initial configuration.
Only extract:
1. The objects (blocks) mentioned in the instruction
2. The LTL goal formulas (what should be achieved)

**JSON Output Format (STRICT - Follow Exactly):**

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "objects": ["a", "b"],
  "ltl_formulas": [
    {{
      "type": "temporal",
      "operator": "F",
      "formula": {{"on": ["a", "b"]}}
    }},
    {{
      "type": "temporal",
      "operator": "G",
      "formula": {{"clear": ["c"]}}
    }}
  ]
}}

**For X (Next) operator:**
{{
  "type": "temporal",
  "operator": "X",
  "formula": {{"on": ["a", "b"]}}
}}
Output: X(on(a, b))

**For WX (Weak Next) operator:**
{{
  "type": "temporal",
  "operator": "WX",
  "formula": {{"on": ["a", "b"]}}
}}
Output: WX(on(a, b))

**For U (Until) operator:**
{{
  "type": "until",
  "operator": "U",
  "left_formula": {{"holding": ["a"]}},
  "right_formula": {{"clear": ["b"]}}
}}
Output: (holding(a) U clear(b))

**For R (Release) operator:**
{{
  "type": "release",
  "operator": "R",
  "left_formula": {{"ontable": ["a"]}},
  "right_formula": {{"clear": ["b"]}}
}}
Output: (ontable(a) R clear(b))

**For F (Eventually) operator:**
{{
  "type": "temporal",
  "operator": "F",
  "formula": {{"on": ["a", "b"]}}
}}
Output: F(on(a, b))

**For G (Always) operator:**
{{
  "type": "temporal",
  "operator": "G",
  "formula": {{"clear": ["a"]}}
}}
Output: G(clear(a))

**For NESTED operators** (e.g., F(G(φ)), G(F(φ))):
{{
  "type": "nested",
  "outer_operator": "F",
  "inner_operator": "G",
  "formula": {{"on": ["a", "b"]}}
}}
Output: F(G(on(a, b)))

**For NEGATION (!)**:
{{
  "type": "temporal",
  "operator": "G",
  "formula": {{
    "type": "negation",
    "formula": {{"on": ["a", "b"]}}
  }}
}}
Output: G(!(on(a, b)))

**For AND (&)**:
{{
  "type": "temporal",
  "operator": "F",
  "formula": {{
    "type": "conjunction",
    "formulas": [
      {{"on": ["a", "b"]}},
      {{"clear": ["c"]}}
    ]
  }}
}}
Output: F(on(a, b) & clear(c))

**For OR (|)**:
{{
  "type": "temporal",
  "operator": "F",
  "formula": {{
    "type": "disjunction",
    "formulas": [
      {{"on": ["a", "b"]}},
      {{"on": ["c", "d"]}}
    ]
  }}
}}
Output: F(on(a, b) | on(c, d))

**For IMPLIES (->)**:
{{
  "type": "temporal",
  "operator": "F",
  "formula": {{
    "type": "implication",
    "left_formula": {{"clear": ["a"]}},
    "right_formula": {{"on": ["a", "b"]}}
  }}
}}
Output: F(clear(a) -> on(a, b))

**For EQUIVALENCE (<->)**:
{{
  "type": "temporal",
  "operator": "F",
  "formula": {{
    "type": "equivalence",
    "left_formula": {{"clear": ["a"]}},
    "right_formula": {{"ontable": ["a"]}}
  }}
}}
Output: F(clear(a) <-> ontable(a))

**Complete Response Examples:**

1. **X (Next)**: X(on(a, b))
```json
{{"type": "temporal", "operator": "X", "formula": {{"on": ["a", "b"]}}}}
```

2. **WX (WeakNext)**: WX(clear(c))
```json
{{"type": "temporal", "operator": "WX", "formula": {{"clear": ["c"]}}}}
```

3. **U (Until)**: (holding(a) U clear(b))
```json
{{"type": "until", "operator": "U", "left_formula": {{"holding": ["a"]}}, "right_formula": {{"clear": ["b"]}}}}
```

4. **R (Release)**: (ontable(a) R clear(b))
```json
{{"type": "release", "operator": "R", "left_formula": {{"ontable": ["a"]}}, "right_formula": {{"clear": ["b"]}}}}
```

5. **F (Eventually)**: F(on(a, b))
```json
{{"type": "temporal", "operator": "F", "formula": {{"on": ["a", "b"]}}}}
```

6. **G (Always)**: G(clear(c))
```json
{{"type": "temporal", "operator": "G", "formula": {{"clear": ["c"]}}}}
```

7. **Nested F(G)**: F(G(on(a, b)))
```json
{{"type": "nested", "outer_operator": "F", "inner_operator": "G", "formula": {{"on": ["a", "b"]}}}}
```

8. **NOT (!)**: G(!(on(a, b)))
```json
{{"type": "temporal", "operator": "G", "formula": {{"type": "negation", "formula": {{"on": ["a", "b"]}}}}}}
```

9. **AND (&)**: F(on(a, b) & clear(c))
```json
{{"type": "temporal", "operator": "F", "formula": {{"type": "conjunction", "formulas": [{{"on": ["a", "b"]}}, {{"clear": ["c"]}}]}}}}
```

10. **OR (|)**: G(clear(a) | ontable(a))
```json
{{"type": "temporal", "operator": "G", "formula": {{"type": "disjunction", "formulas": [{{"clear": ["a"]}}, {{"ontable": ["a"]}}]}}}}
```

11. **IMPLIES (->)**: F(clear(a) -> on(a, b))
```json
{{"type": "temporal", "operator": "F", "formula": {{"type": "implication", "left_formula": {{"clear": ["a"]}}, "right_formula": {{"on": ["a", "b"]}}}}}}
```

12. **EQUIVALENCE (<->)**: G(clear(a) <-> ontable(a))
```json
{{"type": "temporal", "operator": "G", "formula": {{"type": "equivalence", "left_formula": {{"clear": ["a"]}}, "right_formula": {{"ontable": ["a"]}}}}}}
```

13. **Nullary predicate**: G(handempty)
```json
{{"type": "temporal", "operator": "G", "formula": {{"handempty": []}}}}
```

**IMPORTANT**: All operators follow the official LTLf syntax from http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax"""


def get_ltl_user_prompt(nl_instruction: str) -> str:
    """
    Generate user prompt for NL -> LTLf conversion

    Args:
        nl_instruction: Natural language instruction to convert

    Returns:
        User prompt string
    """
    return f"Natural language instruction: {nl_instruction}"
