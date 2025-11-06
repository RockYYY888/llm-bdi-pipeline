# Natural Language Instruction Template for LTLf Generation

This document provides a template and guidelines for writing natural language (NL) instructions that will be correctly translated into LTLf (Linear Temporal Logic on Finite Traces) formulas by the LLM-BDI pipeline.

## Purpose

This template serves as a **reference guide** for:
1. Understanding how to express LTLf operators in natural language
2. Creating test cases for new domains
3. Ensuring consistent NL → LTLf translation quality across different domains
4. Debugging failed LTLf generation attempts

## Domain Requirements

Before creating NL instructions for a new domain, you need:

### 1. Domain Definition (PDDL File)
A PDDL domain file (`domain.pddl`) that specifies:
- **Predicates**: The states/properties that can be true or false
  - Example (Blocksworld): `on(?x - block, ?y - block)`, `clear(?x - block)`, `handempty`, `holding(?x - block)`, `ontable(?x - block)`
- **Actions**: The operations that can be performed
  - Example (Blocksworld): `pick-up`, `put-down`, `stack`, `unstack`
- **Types**: Object types in the domain
  - Example (Blocksworld): `block`

### 2. Domain Objects
Specific objects that exist in the problem instance:
- Example (Blocksworld): `a`, `b`, `c`, `block-1`, `block-2`

### 3. Common Naming Conventions
- Use lowercase for object names
- Hyphens in object names are supported (e.g., `block-1`)
- Predicates use underscores or hyphens consistently

---

## LTLf Syntax Reference

LTLf formulas use the following operators (in order of precedence):

### Propositional Symbols
- `true` / `false` - Boolean constants
- `predicate(obj1, obj2, ...)` - Atomic propositions
  - Nullary: `handempty`
  - Unary: `clear(a)`
  - Binary: `on(a, b)`
  - N-ary: `at(robot, location, floor)`

### Boolean Operators (High to Low Precedence)
1. `!` or `~` - Negation (NOT)
2. `&` or `&&` - Conjunction (AND)
3. `|` or `||` - Disjunction (OR)
4. `->` or `=>` - Implication (IMPLIES)
5. `<->` or `<=>` - Equivalence (IFF)

### Temporal Operators (Unary - Highest Precedence)
- `X(φ)` - Next: φ holds in the next state
- `WX(φ)` - Weak Next: φ holds in next state if it exists
- `F(φ)` - Eventually/Finally: φ will hold at some future point
- `G(φ)` - Globally/Always: φ holds at all future points

### Temporal Operators (Binary - Medium Precedence)
- `(φ U ψ)` - Until: φ holds until ψ becomes true
- `(φ R ψ)` - Release: ψ holds until φ becomes true (or forever)

### Operator Precedence Summary (Highest to Lowest)
1. **Unary operators**: `!`, `X`, `WX`, `F`, `G` (tightest binding)
2. **Binary temporal**: `U`, `R`
3. **Conjunction**: `&`
4. **Disjunction**: `|`
5. **Implication**: `->`
6. **Equivalence**: `<->`
7. **Top-level conjunction** (for multiple formulas)

---

## Natural Language Patterns

Below are tested patterns for expressing each LTLf operator in natural language.

### Pattern Categories

Each pattern includes:
- **Category**: Type of LTLf construct
- **NL Pattern**: How to express it in natural language
- **Expected LTLf**: The LTLf formula that should be generated
- **Example**: Concrete example from Blocksworld domain
- **Objects**: Required objects for the example
- **Notes**: Important considerations

---

## 📋 Template Sections

### Section 1: Propositional Constructs

#### 1.1 Propositional Constants

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| `true` constant | "Goal is always achievable" | `G(true)` | Goal is always achievable | `[]` | Trivially satisfied goal |
| `false` constant | "Goal is impossible" | `G(false)` | Goal is impossible | `[]` | Unsatisfiable goal |

#### 1.2 Atomic Propositions (Nullary)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Nullary predicate | "Always keep {predicate}" | `G({predicate})` | Keep the hand empty | `[]` | Predicates with no arguments |
| | "Eventually {predicate}" | `F({predicate})` | Eventually hand is empty | `[]` | |

**Domain-Specific Examples:**
- Blocksworld: `handempty`
- Logistics: `robot_idle`
- Gripper: `room_empty`

#### 1.3 Atomic Propositions (Unary)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Unary predicate | "Eventually {object} is {predicate}" | `F({predicate}({object}))` | Eventually block a is clear | `['a']` | Single-argument predicates |
| | "Always keep {object} {predicate}" | `G({predicate}({object}))` | Always keep block a clear | `['a']` | |

**Domain-Specific Examples:**
- Blocksworld: `clear(a)`, `ontable(b)`
- Logistics: `at_location(package1)`, `loaded(truck)`
- Gripper: `carrying(ball1)`

#### 1.4 Atomic Propositions (Binary)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Binary predicate | "Eventually {obj1} is {relation} {obj2}" | `F({relation}({obj1}, {obj2}))` | Eventually a is on b | `['a', 'b']` | Two-argument predicates |
| | "Put {obj1} {relation} {obj2}" | `F({relation}({obj1}, {obj2}))` | Put block a on block b | `['a', 'b']` | Action-oriented phrasing |

**Domain-Specific Examples:**
- Blocksworld: `on(a, b)`
- Logistics: `in(package, truck)`, `connected(city1, city2)`
- Gripper: `at(ball, room)`

---

### Section 2: Boolean Operators

#### 2.1 Conjunction (AND)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| AND within temporal | "Eventually, {pred1} and {pred2}" | `F({pred1} & {pred2})` | Eventually, a is on b and c is clear | `['a', 'b', 'c']` | Use comma before "and" for clarity |
| Multiple conditions | "Eventually both {pred1} and {pred2}" | `F({pred1} & {pred2})` | Eventually both a is clear and b is on table | `['a', 'b']` | Alternative phrasing |
| Explicit conjunction | "Eventually the conjunction of {pred1} and {pred2}" | `F({pred1} & {pred2})` | Eventually the conjunction of a is on b and c is clear | `['a', 'b', 'c']` | Emphasizes structure |

**Critical Notes:**
- Use commas to separate clauses: "Eventually, X and Y" → `F(X & Y)`
- Without comma: "Eventually X and eventually Y" → `F(X) & F(Y)` (different!)
- Operator precedence: `&` binds tighter than `|`
- For clarity in complex formulas: "the conjunction of X and Y" explicitly marks `&` grouping

#### 2.2 Disjunction (OR)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| OR within temporal | "Eventually either {pred1} or {pred2}" | `F({pred1} \| {pred2})` | Eventually either a is on b or c is on d | `['a', 'b', 'c', 'd']` | Use "either...or" for clarity |
| Alternative phrasing | "Eventually {pred1} or {pred2}" | `F({pred1} \| {pred2})` | Eventually a is clear or b is on table | `['a', 'b']` | Without "either" |
| Explicit disjunction | "Eventually the disjunction of {pred1} or {pred2}" | `F({pred1} \| {pred2})` | Eventually the disjunction of a is clear or b is on table | `['a', 'b']` | Emphasizes structure |

**Critical Notes:**
- "Either...or" emphasizes mutual exclusivity
- Precedence: `&` > `|`, so `F(A & B | C)` = `F((A & B) | C)`
- For clarity: "the disjunction of X or Y" explicitly marks `|` grouping

#### 2.3 Negation (NOT)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Negative goal | "Never {action/state}" | `G(!({pred}))` | Never put a on b | `['a', 'b']` | Always NOT |
| Negative condition | "Eventually not {state}" | `F(!({pred}))` | Eventually not holding a | `['a']` | |

**Critical Notes:**
- "Never X" = `G(!(X))` (always not X)
- "Not eventually X" = `!(F(X))` (different meaning!)
- Precedence: `!` binds tightest (before `&`)

#### 2.4 Implication (IMPLIES)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Conditional goal | "Eventually if {condition} then {consequence}" | `F({cond} -> {conseq})` | Eventually if a is clear then a is on b | `['a', 'b']` | "If...then" structure |
| Alternative phrasing | "Eventually when {condition}, {consequence}" | `F({cond} -> {conseq})` | Eventually when a is clear, put a on b | `['a', 'b']` | |

**Critical Notes:**
- `A -> B` is equivalent to `!A | B`
- Precedence: `|` > `->`, so `A | B -> C` = `(A | B) -> C`

#### 2.5 Equivalence (IFF)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Bidirectional condition | "Eventually {pred1} if and only if {pred2}" | `F({pred1} <-> {pred2})` | Eventually a is clear if and only if a is on table | `['a']` | "IFF" relationship |
| Alternative phrasing | "Eventually {pred1} exactly when {pred2}" | `F({pred1} <-> {pred2})` | Eventually a is clear exactly when b is clear | `['a', 'b']` | |

---

### Section 3: Temporal Operators (Unary)

#### 3.1 Next (X)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Immediate successor | "{action1} then immediately {action2}" | `F({pred1} & X({pred2}))` | Pick up a then immediately place on b | `['a', 'b']` | Next state requirement |
| | "In the next state {predicate}" | `X({pred})` | In next state a is on b | `['a', 'b']` | Strong next |

**Critical Notes:**
- `X(φ)` requires exactly one more state (fails at trace end)
- Often combined with `F`: `F(A & X(B))` - "eventually A, then immediately B"

#### 3.2 Weak Next (WX)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Weak successor | "In next state if exists {predicate}" | `WX({pred})` | In next state if exists a is on b | `['a', 'b']` | Succeeds at trace end |

**Critical Notes:**
- `WX(φ)` allows termination (true if no next state)
- Rarely used in goal specifications

#### 3.3 Eventually/Finally (F)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Achievement goal | "Put {object1} {relation} {object2}" | `F({relation}({obj1}, {obj2}))` | Put block a on block b | `['a', 'b']` | Most common goal form |
| Explicit temporal | "Eventually {predicate}" | `F({pred})` | Eventually a is on b | `['a', 'b']` | |
| Future achievement | "At some point {predicate}" | `F({pred})` | At some point a is clear | `['a']` | Alternative phrasing |

**Critical Notes:**
- `F(φ)` = "φ holds now or at some future point"
- Most domain goals are `F(...)` formulas
- Can be nested: `F(G(...))` - "eventually always"

#### 3.4 Globally/Always (G)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Maintenance goal | "Always keep {predicate}" | `G({pred})` | Always keep block a clear | `['a']` | Safety property |
| Negative constraint | "Never {predicate}" | `G(!({pred}))` | Never put a on b | `['a', 'b']` | Prohibition |

**Critical Notes:**
- `G(φ)` = "φ holds at all future points (including now)"
- Used for safety constraints and invariants
- Can be nested: `G(F(...))` - "always eventually"

---

### Section 4: Temporal Operators (Binary)

#### 4.1 Until (U)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Hold until | "Keep {pred1} until {pred2}" | `({pred1} U {pred2})` | Keep holding a until b is clear | `['a', 'b']` | Strong until (ψ must occur) |
| Complex until | "Keep both {pred1} and {pred2} until {pred3}" | `(({pred1} & {pred2}) U {pred3})` | Keep both holding a and b clear until c is on d | `['a', 'b', 'c', 'd']` | Conjunction in left operand |
| Explicit structure | "The state that {pred1} until {pred2}" | `({pred1} U {pred2})` | The state that holding a until b is clear | `['a', 'b']` | Emphasizes Until scope |

**Critical Notes:**
- `(φ U ψ)` requires ψ to eventually hold
- φ must hold at all points before ψ
- Precedence: `&` > `U`, so `A & B U C` = `(A & B) U C`
- Use explicit parens for clarity: `((A & B) U C)`
- "The state that X until Y" helps clarify Until formula boundaries

#### 4.2 Release (R)

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Release condition | "{pred1} stays {state} unless {pred2}" | `({pred2} R {pred1})` | B stays clear unless a is placed on table | `['a', 'b']` | Weak until (dual) |
| Maintain until | "{pred1} holds release condition {pred2}" | `({pred2} R {pred1})` | Clear b holds release condition ontable a | `['a', 'b']` | Alternative phrasing |

**Critical Notes:**
- `(φ R ψ)` is the dual of Until
- ψ must hold until (and including) the point where φ first holds
- φ may never hold (unlike Until)
- `(φ R ψ)` ≡ `!((!φ) U (!ψ))`

---

### Section 5: Nested Operators

#### 5.1 F(G(...)) - Eventually Always

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Reach stable state | "Eventually ensure {pred} always holds" | `F(G({pred}))` | Eventually ensure a is always on b | `['a', 'b']` | Stability goal |
| Alternative | "Reach a state where {pred} remains true" | `F(G({pred}))` | Reach a state where a stays clear | `['a']` | |

**Semantic Meaning:** "Eventually reach a point after which φ always holds"

#### 5.2 G(F(...)) - Always Eventually

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Repeated achievement | "Keep trying to {action}" | `G(F({pred}))` | Keep trying to clear c | `['c']` | Liveness property |
| Recurrence | "Always eventually {predicate}" | `G(F({pred}))` | Always eventually a is clear | `['a']` | |

**Semantic Meaning:** "At every point, φ will eventually hold again"

#### 5.3 Complex Nesting

| Category | NL Pattern | Expected LTLf | Example | Objects | Notes |
|----------|------------|---------------|---------|---------|-------|
| Conditional temporal | "Always if {cond} then eventually {conseq}" | `G({cond} -> F({conseq}))` | Always if a is clear then eventually a is on b | `['a', 'b']` | Nested `F` in implication |
| Structured nesting | "Always the following holds: if {cond} then eventually {conseq}" | `G({cond} -> F({conseq}))` | Always the following holds: if a is clear then eventually a is on b | `['a', 'b']` | Explicit structure marking |
| Deep nesting | "{temporal1} {complex formula with temporal2}" | Complex nesting | See COMPLEX01 test case | Multiple | Requires careful phrasing |

**Critical Notes:**
- Break down complex formulas with explicit structure
- Use punctuation (commas, colons) to clarify nesting
- Explicitly state nesting levels: "the following disjunction holds: first disjunct is..."
- Phrases like "the state that", "the conjunction of", "the following holds" help mark formula boundaries

---

### Section 6: Operator Precedence

Understanding precedence is critical for correct LTLf generation.

#### Precedence Table (Highest to Lowest)

| Precedence Level | Operators | Example Formula | Interpretation |
|------------------|-----------|-----------------|----------------|
| 1 (Highest) | `!`, `X`, `WX`, `F`, `G` | `F A & B` | `(F(A)) & B` NOT `F(A & B)` |
| 2 | `U`, `R` | `A & B U C` | `(A & B) U C` |
| 3 | `&` | `A & B \| C` | `(A & B) \| C` |
| 4 | `\|` | `A \| B -> C` | `(A \| B) -> C` |
| 5 | `->` | `A -> B <-> C` | `(A -> B) <-> C` |
| 6 (Lowest) | `<->` | Multiple formulas | Top-level conjunction |

#### Precedence Examples with NL Patterns

| Test ID | Category | NL Pattern | Expected LTLf | Explanation |
|---------|----------|------------|---------------|-------------|
| PRE01 | Unary vs AND | "Eventually on a b **conjoined with** always clear c" | `F(on(a, b)) & G(clear(c))` | `F` and `G` bind before `&` |
| PRE02 | AND vs OR | "Eventually, it happens **either** a is on b **and** c is clear, **or** d is on e" | `F(on(a, b) & clear(c) \| on(d, e))` | `&` groups before `\|` |
| PRE03 | OR vs IMPLIES | "Always clear a **or** on a b **implies** eventually on c d" | `G(clear(a) \| on(a, b) -> F(on(c, d)))` | `\|` groups before `->` |
| PRE04 | Until vs AND | "A conjunction of holding a **until** clear b **with** c is on d" | `(holding(a) U clear(b) & on(c, d))` | `&` in right operand of `U` |
| PRE05 | Negation vs AND | "Eventually **not** on a b **and** clear c" | `F(!(on(a, b)) & clear(c))` | `!` binds tighter than `&` |

#### NL Patterns for Controlling Precedence

1. **Use explicit grouping phrases:**
   - "the conjunction of X and Y" → `(X & Y)`
   - "the disjunction: either X or Y" → `(X | Y)`
   - "the state that X" → clarifies temporal formula scope
   - "the following holds: X" → clarifies scope

2. **Use punctuation:**
   - Commas separate clauses: "Eventually, X and Y"
   - Colons introduce nested structure: "the following: X"
   - Semicolons separate major parts

3. **Use explicit parenthetical descriptions:**
   - "eventually (the conjunction of A and B)" → `F(A & B)`
   - "the until formula: A until B" → `(A U B)`

4. **When in doubt, over-specify:**
   - Bad: "Eventually a and b or c"
   - Good: "Eventually, either the conjunction of a and b, or c"
   - Better: "Eventually the following disjunction holds: first disjunct is the conjunction of a and b, second disjunct is c"

---

### Section 7: Complex Formulas

#### 7.1 Deeply Nested Structures

| Test ID | NL Pattern (Structured) | Expected LTLf | Objects |
|---------|-------------------------|---------------|---------|
| COMPLEX01 | "Always if block a is clear then eventually **the following disjunction holds**: **first disjunct is** hold a and in the next state block b is clear, **second disjunct is** block c is on d until block e is on table" | `G(clear(a) -> F((holding(a) & X(clear(b))) \| (on(c, d) U ontable(e))))` | `['a', 'b', 'c', 'd', 'e']` |

**Explanation:**
- Uses hierarchical structure: "the following X holds: first Y, second Z"
- Explicitly numbers disjuncts/conjuncts
- Uses punctuation (commas, colons) to clarify nesting

#### 7.2 Multiple Implications

| Test ID | NL Pattern (Structured) | Expected LTLf | Objects |
|---------|-------------------------|---------------|---------|
| COMPLEX02 | "Eventually **the following implication holds**: if block a is on b then **the following nested implication holds**: if block c is clear then always **the following disjunction holds**: **first disjunct is** block d is on e, **second disjunct is** a conjunction where block f is eventually on g and block h is not clear" | `F(on(a, b) -> (clear(c) -> G(on(d, e) \| (F(on(f, g)) & !(clear(h))))))` | `['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']` |

**Explanation:**
- Each nesting level is explicitly introduced
- "The following X holds" signals a new nested level
- Implications are explicitly named: "the following implication", "nested implication"

#### 7.3 Temporal Formula Equivalence

| Test ID | NL Pattern (Structured) | Expected LTLf | Objects |
|---------|-------------------------|---------------|---------|
| COMPLEX03 | "**The following equivalence holds**: **on the left side**, both holding a, and the state that block b is clear until block c is on d hold; **right side is** block e stays on table release condition which is a conjunction of eventually block f is on g and always block h is clear" | `(holding(a) & clear(b) U on(c, d)) <-> (ontable(e) R (F(on(f, g)) & G(clear(h))))` | `['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']` |

**Explanation:**
- "Left side" and "right side" clearly delineate equivalence operands
- Complex temporal formulas on both sides
- Uses "release condition" to describe `R` operator

---

### Section 8: Domain-Specific Guidelines

When creating NL patterns for a **new domain**, follow these steps:

#### Step 1: Identify Domain Predicates

Create a table of all predicates in the domain:

| Predicate | Arity | Example | Description |
|-----------|-------|---------|-------------|
| `on(X, Y)` | 2 | `on(a, b)` | X is on top of Y |
| `clear(X)` | 1 | `clear(a)` | X has nothing on top |
| `handempty` | 0 | `handempty` | Robot hand is empty |

#### Step 2: Identify Domain Actions

List all actions (for context, though not directly used in LTLf goals):

| Action | Parameters | Preconditions | Effects |
|--------|------------|---------------|---------|
| `pick-up(?x)` | `?x - block` | `clear(?x), ontable(?x), handempty` | `holding(?x), not ontable(?x), not clear(?x), not handempty` |

#### Step 3: Create Common NL Patterns

For each predicate, create natural phrasings:

**Example (Blocksworld `on` predicate):**
- "Put X on Y" → `F(on(X, Y))`
- "X is on Y" → `on(X, Y)`
- "Stack X on top of Y" → `F(on(X, Y))`
- "Place X on Y" → `F(on(X, Y))`

**Example (Blocksworld `clear` predicate):**
- "X is clear" → `clear(X)`
- "Nothing on top of X" → `clear(X)`
- "X has nothing on it" → `clear(X)`

#### Step 4: Test Basic Patterns

Start with simple test cases (like P01-P04):
1. Propositional constants (`true`, `false`)
2. Nullary predicates (e.g., `handempty`)
3. Unary predicates (e.g., `clear(a)`)
4. Binary predicates (e.g., `on(a, b)`)

#### Step 5: Test Boolean Operators

Add test cases for:
1. AND: "A and B"
2. OR: "A or B"
3. NOT: "not A", "never A"
4. IMPLIES: "if A then B"
5. EQUIVALENCE: "A if and only if B"

#### Step 6: Test Temporal Operators

Add test cases for:
1. F (Eventually): "eventually A", "at some point A"
2. G (Always): "always A", "keep A"
3. X (Next): "in next state A", "immediately A"
4. U (Until): "A until B"
5. R (Release): "B unless A"

#### Step 7: Test Nesting and Precedence

Create complex test cases with:
1. Nested temporal operators: `F(G(...))`, `G(F(...))`
2. Mixed precedence: AND before OR, unary before binary
3. Explicit parentheses for clarity

---

### Section 9: Common Pitfalls and Solutions

#### Pitfall 1: Ambiguous Scope

**Problem:**
"Eventually a is on b and c is clear"

**Ambiguity:**
- `F(on(a, b)) & clear(c)` - "eventually a on b" AND "c is clear now"
- `F(on(a, b) & clear(c))` - "eventually both conditions"

**Solution:**
Add a comma after temporal operator:
"Eventually, a is on b and c is clear" → `F(on(a, b) & clear(c))`

#### Pitfall 2: Negation Scope

**Problem:**
"Not eventually a is on b"

**Ambiguity:**
- `!(F(on(a, b)))` - "it's not the case that eventually a is on b" (a will never be on b)
- `F(!(on(a, b)))` - "eventually a is not on b"

**Solution:**
Use clear phrasing:
- "Never put a on b" → `G(!(on(a, b)))`
- "Eventually not on b" → `F(!(on(a, b)))`

#### Pitfall 3: Multiple Temporal Operators

**Problem:**
"Eventually a is on b and eventually c is clear"

**Ambiguity:**
- `F(on(a, b) & clear(c))` - both conditions in same eventual state
- `F(on(a, b)) & F(clear(c))` - two separate eventual states

**Solution:**
- Same eventual state: "Eventually, both a is on b and c is clear"
- Separate states: "Eventually a is on b, and separately, eventually c is clear"
- Or use explicit conjunction: "Eventually on a b conjoined with eventually clear c"

#### Pitfall 4: Until vs Implies

**Problem:**
"Keep holding a until b is clear" vs "If holding a then eventually b is clear"

**Difference:**
- Until: `(holding(a) U clear(b))` - a must hold continuously until b
- Implies: `holding(a) -> F(clear(b))` - if a holds (at one point), then eventually b

**Solution:**
- Use "until" explicitly for Until operator
- Use "if...then" for implication

#### Pitfall 5: Object Naming

**Problem:**
Inconsistent object references: "block a", "block-1", "blocka"

**Solution:**
- Choose consistent naming: "block a" → object `a`
- With hyphens: "block-1" → object `block-1`
- Always use the exact object name from the PDDL problem file

---

### Section 10: Template Usage Workflow

#### For Creating Test Cases:

1. **Choose target LTLf formula** (e.g., `F(on(a, b) & clear(c))`)
2. **Select appropriate NL pattern** from this template (e.g., Boolean AND within temporal)
3. **Substitute domain-specific terms**:
   - Replace generic `{pred1}`, `{pred2}` with actual predicates
   - Replace `{obj1}`, `{obj2}` with actual objects
4. **Write NL instruction**: "Eventually, a is on b and c is clear"
5. **Specify objects list**: `['a', 'b', 'c']`
6. **Add to test CSV** with expected LTLf formula

#### For Debugging Failed Generation:

1. **Check test results CSV** for failed test case
2. **Compare actual vs expected LTLf**
3. **Identify mismatch type**:
   - Extra/missing parentheses → Check precedence
   - Wrong operator → Check NL phrasing (e.g., "until" vs "if")
   - Wrong scope → Check comma placement
4. **Consult this template** for correct NL pattern
5. **Refine NL instruction** and re-test

#### For New Domain Setup:

1. **Parse PDDL domain file** to extract predicates and actions
2. **Create domain-specific predicate table** (Step 1 in Section 8)
3. **Generate common NL phrasings** for each predicate (Step 3 in Section 8)
4. **Start with simple test cases** (propositional, unary, binary)
5. **Gradually add complexity** (boolean, temporal, nested)
6. **Run test suite** and iterate based on results

---

## Example: Complete Test Case Creation

### Domain: Blocksworld

**Target LTLf:** `F(on(a, b) & clear(c))`

**Step 1: Choose Pattern**
Category: Boolean AND within temporal
Pattern: "Eventually, {pred1} and {pred2}"

**Step 2: Substitute Terms**
- `{pred1}` → `on(a, b)` → "a is on b"
- `{pred2}` → `clear(c)` → "c is clear"

**Step 3: Construct NL**
"Eventually, a is on b and c is clear"

**Step 4: Specify Objects**
Objects: `['a', 'b', 'c']`

**Step 5: Create Test Entry**
```csv
B01,boolean_and,"Eventually, a is on b and c is clear","F(on(a, b) & clear(c))","['a', 'b', 'c']",Boolean AND operator,Boolean Operators (And)
```

**Step 6: Run Test**
Expected: `F(on(a, b) & clear(c))`
Actual: (generated by LLM)
Match: DFA equivalence check

---

## Reference: Blocksworld Domain Specifics

### Predicates
- `on(?x - block, ?y - block)` - X is on top of Y
- `ontable(?x - block)` - X is on the table
- `clear(?x - block)` - Nothing is on top of X
- `holding(?x - block)` - Robot hand is holding X
- `handempty` - Robot hand is empty

### Actions
- `pick-up(?x - block)` - Pick up block X from table
- `put-down(?x - block)` - Put down block X on table
- `stack(?x - block, ?y - block)` - Stack X on top of Y
- `unstack(?x - block, ?y - block)` - Remove X from top of Y

### Common NL Patterns (Blocksworld)
- "Put X on Y" → `F(on(X, Y))`
- "Stack X on Y" → `F(on(X, Y))`
- "Clear X" → `F(clear(X))`
- "X is on table" → `ontable(X)`
- "Pick up X" → Goal implies `F(holding(X))` or context-dependent
- "Keep hand empty" → `G(handempty)`

