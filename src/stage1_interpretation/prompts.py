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

**LTLf Syntax Reference**

┌─────────────────────────────────────────────────────────────────┐
│ PROPOSITIONAL SYMBOLS                                            │
├──────────────────┬──────────────────────────────────────────────┤
│ Symbol           │ Syntax                                        │
├──────────────────┼──────────────────────────────────────────────┤
│ true             │ True                                          │
│ false            │ False                                         │
│ [a-z][a-z0-9_]*  │ Atomic propositions                          │
└──────────────────┴──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ BOOLEAN OPERATORS                                                │
├──────────────────┬──────────────────────────────────────────────┤
│ Symbol           │ Syntax                                        │
├──────────────────┼──────────────────────────────────────────────┤
│ &, &&            │ And (conjunction)                            │
│ |, ||            │ Or (disjunction)                             │
│ !, ~             │ Not (negation)                               │
│ ->, =>           │ Implication                                  │
│ <->, <=>         │ Equivalence                                  │
└──────────────────┴──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ FUTURE TEMPORAL OPERATORS                                        │
├──────────────────┬──────────────────────────────────────────────┤
│ Symbol           │ Syntax                                        │
├──────────────────┼──────────────────────────────────────────────┤
│ X                │ Next (strong next state)                     │
│ WX               │ WeakNext (next state or no next)             │
│ U                │ Until (φ U ψ - φ holds until ψ)              │
│ R                │ Release (φ R ψ - ψ holds until φ)            │
│ F                │ Eventually (will be true at some point)      │
│ G                │ Always (true at all future points)           │
└──────────────────┴──────────────────────────────────────────────┘

**CRITICAL: These are the ONLY valid symbols. DO NOT use:**
- Commas (,)
- Semicolons (;)
- Any other punctuation (symbol) not in the tables above

**CRITICAL - Operator Precedence:**

Understanding precedence is CRITICAL for correct LTLf formula construction. The parser applies operators
based on the following precedence levels (higher number = higher precedence = binds tighter):

**Operator Precedence Table (JSON format for clarity):**

```json
{{
  "!": 6,
  "~": 6,
  "X": 5,
  "WX": 5,
  "F": 5,
  "G": 5,
  "U": 4,
  "R": 4,
  "&": 3,
  "&&": 3,
  "|": 2,
  "||": 2,
  "->": 1,
  "=>": 1,
  "<->": 0,
  "<=>": 0
}}
```

**Precedence Levels Explained:**

- **Level 6 (Highest - Binds Tightest)**: ! ~ (Negation)
- **Level 5**: X WX F G (Unary Temporal Operators)
- **Level 4**: U R (Binary Temporal Operators - Until, Release)
- **Level 3**: & && (Conjunction/AND)
- **Level 2**: | || (Disjunction/OR)
- **Level 1**: -> => (Implication)
- **Level 0 (Lowest - Binds Loosest)**: <-> <=> (Equivalence)

**Precedence Examples:**

1. **"F on(a,b) & clear(c)"** parses as **"F(on(a,b)) & clear(c)"** (F has higher precedence than &)
   - Unary F binds tighter than binary &
   - Equivalent to: "Eventually on(a,b) AND clear(c)"

2. **"on(a,b) & clear(c) | on(d,e)"** parses as **"(on(a,b) & clear(c)) | on(d,e)"** (& has higher precedence than |)
   - Conjunction groups before disjunction
   - Equivalent to: "(on(a,b) AND clear(c)) OR on(d,e)"

3. **"clear(a) -> on(a,b) | on(b,c)"** parses as **"clear(a) -> (on(a,b) | on(b,c))"** (| has higher precedence than ->)
   - Disjunction groups before implication
   - Equivalent to: "clear(a) IMPLIES (on(a,b) OR on(b,c))"

4. **"holding(a) U clear(b) & on(c,d)"** parses as **"(holding(a) U clear(b)) & on(c,d)"** (U has higher precedence than &)
   - Until binds tighter than conjunction
   - Equivalent to: "(holding(a) UNTIL clear(b)) AND on(c,d)"
   - NOTE: If you want conjunction in right operand, use explicit parentheses: "holding(a) U (clear(b) & on(c,d))"

5. **"!on(a,b) & clear(c)"** parses as **"(!on(a,b)) & clear(c)"** (! has higher precedence than &)
   - Negation binds tighter than conjunction
   - Equivalent to: "NOT on(a,b) AND clear(c)"

6. **"G clear(a) -> F on(a,b)"** parses as **"G(clear(a)) -> F(on(a,b))"** (G, F have higher precedence than ->)
   - Both temporal operators bind before implication
   - Equivalent to: "Always clear(a) IMPLIES Eventually on(a,b)"

**How to Reflect Precedence in JSON Output:**

Your JSON structure MUST reflect the correct operator precedence through proper nesting:

**Example 0 (Propositional Constants):** "Goal is always achievable" → G(true)
```json
{{
  "type": "temporal",
  "operator": "G",
  "formula": "true"
}}
```
**CRITICAL**: Propositional constants "true" and "false" are represented as plain strings, NOT as dictionaries.
- ✅ CORRECT: "formula": "true"
- ❌ WRONG: "formula": {{"true": []}}, "formula": {{"true": "constant"}}, "formula": {{"constant": []}}

**Example 1:** "F on(a,b) & clear(c)" → F(on(a,b)) & clear(c)
```json
{{
  "type": "conjunction",
  "formulas": [
    {{
      "type": "temporal",
      "operator": "F",
      "formula": {{"on": ["a", "b"]}}
    }},
    {{"clear": ["c"]}}
  ]
}}
```

**Example 2:** "on(a,b) & clear(c) | on(d,e)" → (on(a,b) & clear(c)) | on(d,e)
```json
{{
  "type": "disjunction",
  "formulas": [
    {{
      "type": "conjunction",
      "formulas": [
        {{"on": ["a", "b"]}},
        {{"clear": ["c"]}}
      ]
    }},
    {{"on": ["d", "e"]}}
  ]
}}
```

**Example 3:** "holding(a) U clear(b) & on(c,d)" → (holding(a) U clear(b)) & on(c,d)
```json
{{
  "type": "conjunction",
  "formulas": [
    {{
      "type": "until",
      "operator": "U",
      "left_formula": {{"holding": ["a"]}},
      "right_formula": {{"clear": ["b"]}}
    }},
    {{"on": ["c", "d"]}}
  ]
}}
```

**Example 3b:** If you want conjunction in right operand: "holding(a) U (clear(b) & on(c,d))"
```json
{{
  "type": "until",
  "operator": "U",
  "left_formula": {{"holding": ["a"]}},
  "right_formula": {{
    "type": "conjunction",
    "formulas": [
      {{"clear": ["b"]}},
      {{"on": ["c", "d"]}}
    ]
  }}
}}
```

**When to Use Explicit Parentheses:**

In your JSON output, the nesting structure implicitly defines precedence. However, when generating
the final LTLf string formula, parentheses should be used to:

1. Override default precedence (rare, since JSON nesting handles this)
2. Improve readability for complex nested expressions
3. Group operands for Until (U) and Release (R) operators (always use outer parentheses)

**Key Rules for Precedence:**

1. **Unary operators (Level 6 & 5) bind tightest:**
   - !, X, WX, F, G always apply to the immediate next atom/formula
   - Example: `F a & b` means `(F(a)) & b`, NOT `F(a & b)`

2. **Binary temporal (U, R) at Level 4 bind tighter than boolean operators:**
   - `a U b & c` means `(a U b) & c`, NOT `a U (b & c)`
   - `a U b | c` means `(a U b) | c`, NOT `a U (b | c)`
   - **CRITICAL**: If you want conjunction/disjunction in right operand, use explicit parentheses!

3. **AND (&) at Level 3 binds tighter than OR (|) at Level 2:**
   - Like multiplication before addition in arithmetic
   - `a & b | c` means `(a & b) | c`

4. **Implication (->) and Equivalence (<->) have lowest precedence:**
   - They group last after all other operators
   - `a & b -> c` means `(a & b) -> c`
   - `a | b <-> c` means `(a | b) <-> c`

5. **When in doubt, use explicit parentheses in your JSON structure:**
   - Nesting in JSON defines precedence
   - Clear nesting avoids ambiguity

**Multiple Goals Handling:**

When the natural language describes multiple goals connected with "and", "conjoined with", or similar:
→ **ALWAYS use & operator to combine into a SINGLE formula**

Example: "Eventually on a b and always clear c"
```json
{{
  "ltl_formulas": [
    {{
      "type": "conjunction",
      "formulas": [
        {{"type": "temporal", "operator": "F", "formula": {{"on": ["a", "b"]}}}},
        {{"type": "temporal", "operator": "G", "formula": {{"clear": ["c"]}}}}
      ]
    }}
  ]
}}
```
Output: **F(on(a, b)) & G(clear(c))**

**COMMON MISTAKES TO AVOID:**

❌ **WRONG - Using commas to separate formulas:**
```json
{{"ltl_formulas": [{{"type": "temporal", "operator": "F", ...}}, {{"type": "temporal", "operator": "G", ...}}]}}
```
This outputs: `F(on(a, b)), G(clear(c))` - **INVALID! Commas are NOT LTL operators!**

✓ **CORRECT - Using & operator in a conjunction:**
```json
{{
  "ltl_formulas": [
    {{
      "type": "conjunction",
      "formulas": [
        {{"type": "temporal", "operator": "F", "formula": {{"on": ["a", "b"]}}}},
        {{"type": "temporal", "operator": "G", "formula": {{"clear": ["c"]}}}}
      ]
    }}
  ]
}}
```
This outputs: `F(on(a, b)) & G(clear(c))` - **CORRECT!**

❌ **WRONG - Treating "and" as a separator:**
Natural language: "Eventually on a b and always clear c"
Wrong interpretation: Two separate goals → outputs `F(on(a, b)), G(clear(c))`

✓ **CORRECT - Treating "and" as Boolean AND operator:**
Natural language: "Eventually on a b and always clear c"
Correct interpretation: Single conjunction → outputs `F(on(a, b)) & G(clear(c))`

**Parentheses for Clarity:**
Even when operator precedence allows omitting parentheses, include them for readability:
- `G(clear(a) | on(a, b) -> F(on(c, d)))` works but is hard to read
- `G((clear(a) | on(a, b)) -> F(on(c, d)))` is clearer - **PREFER THIS**

Your task: Convert natural language to LTLf formulas following the syntax.

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
8. X (Next): "Pick A then immediately place on B" → F(holding(a) & X(on(a, b)))
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
3. The propositional atoms used (for grounding map)

**CRITICAL - Propositionalization Concept:**

LTLf only supports propositional atoms (not parameterized predicates). However, for human readability
and domain mapping, we maintain a grounding map between propositional symbols and original predicates.

Naming Convention for Propositional Symbols:
- Nullary predicates: handempty → handempty
- Unary predicates: clear(a) → clear_a
- Binary predicates: on(a, b) → on_a_b
- General rule: predicate_arg1_arg2_... (lowercase, underscore-separated)

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
  ],
  "atoms": [
    {{"symbol": "on_a_b", "predicate": "on", "args": ["a", "b"]}},
    {{"symbol": "clear_c", "predicate": "clear", "args": ["c"]}}
  ]
}}

**Special Case - Propositional Constants (true/false):**
{{
  "objects": [],
  "ltl_formulas": [
    {{
      "type": "temporal",
      "operator": "G",
      "formula": "true"
    }}
  ],
  "atoms": []
}}
**CRITICAL**:
- Use plain string "true" or "false", NOT a dictionary
- Do NOT include "true" or "false" in the atoms list
- Objects list should be empty if no domain objects are mentioned

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

**COMPLETE RESPONSE EXAMPLES:**

**Example 1**: Simple temporals
For "Put block a on block b, and keep c clear":
```json
{{
  "objects": ["a", "b", "c"],
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
  ],
  "atoms": [
    {{"symbol": "on_a_b", "predicate": "on", "args": ["a", "b"]}},
    {{"symbol": "clear_c", "predicate": "clear", "args": ["c"]}}
  ]
}}
```

**Example 2**: Nested temporal in implication (CRITICAL PATTERN!)
For "Always if a is clear then eventually a is on b":
```json
{{
  "objects": ["a", "b"],
  "ltl_formulas": [
    {{
      "type": "temporal",
      "operator": "G",
      "formula": {{
        "type": "implication",
        "left_formula": {{"clear": ["a"]}},
        "right_formula": {{
          "type": "temporal",
          "operator": "F",
          "formula": {{"on": ["a", "b"]}}
        }}
      }}
    }}
  ],
  "atoms": [
    {{"symbol": "clear_a", "predicate": "clear", "args": ["a"]}},
    {{"symbol": "on_a_b", "predicate": "on", "args": ["a", "b"]}}
  ]
}}
```
Output: G(clear(a) -> F(on(a, b)))

**Example 3**: Conjunction in Until
For "Keep holding a and b clear until c is on d":
```json
{{
  "objects": ["a", "b", "c", "d"],
  "ltl_formulas": [
    {{
      "type": "until",
      "operator": "U",
      "left_formula": {{
        "type": "conjunction",
        "formulas": [
          {{"holding": ["a"]}},
          {{"clear": ["b"]}}
        ]
      }},
      "right_formula": {{"on": ["c", "d"]}}
    }}
  ],
  "atoms": [
    {{"symbol": "holding_a", "predicate": "holding", "args": ["a"]}},
    {{"symbol": "clear_b", "predicate": "clear", "args": ["b"]}},
    {{"symbol": "on_c_d", "predicate": "on", "args": ["c", "d"]}}
  ]
}}
```
Output: ((holding(a) & clear(b)) U on(c, d))

**VALIDATION RULES:**
1. Every predicate used in ltl_formulas MUST have a corresponding entry in atoms
2. Symbol naming MUST follow: predicate_arg1_arg2_... (lowercase, underscore)
3. All args in atoms must be from the objects list
4. Predicates and objects MUST exist in the provided domain"""


def get_ltl_user_prompt(nl_instruction: str) -> str:
    """
    Generate user prompt for NL -> LTLf conversion

    Args:
        nl_instruction: Natural language instruction to convert

    Returns:
        User prompt string
    """
    return f"Goal: {nl_instruction}"
