# Stage 3: Backward Planning for AgentSpeak Code Generation

## Document Information
- **Created**: 2025-11-06
- **Status**: Design Phase
- **Purpose**: Complete design specification for Stage 3 refactoring from LLM-based to backward planning-based AgentSpeak code generation

---

## Table of Contents
1. [Background and Motivation](#background-and-motivation)
2. [Core Design Decisions](#core-design-decisions)
3. [Complete Q&A Record](#complete-qa-record)
4. [Technical Architecture](#technical-architecture)
5. [Implementation Plan](#implementation-plan)
6. [Data Structures](#data-structures)
7. [Algorithms](#algorithms)
8. [Example Outputs](#example-outputs)
9. [Testing Strategy](#testing-strategy)
10. [Potential Challenges](#potential-challenges)

---

## Background and Motivation

### Current State (Stage 2 → Stage 3)
Currently, Stage 3 uses an LLM to generate AgentSpeak code from DFA transitions. This approach has limitations:
- Non-deterministic outputs
- Requires API calls and tokens
- Difficult to debug and validate
- May generate syntactically or semantically incorrect code

### Proposed Solution
Replace LLM-based generation with **programmatic backward planning**:
1. Parse DFA transitions to extract goal predicates
2. Perform forward state-space exploration ("destruction") from goal states
3. Build a state graph with all reachable states and actions
4. Extract shortest paths from leaf states to goal states
5. Generate AgentSpeak plans with proper contexts and belief updates

### Key Innovation
**Forward "Destruction" Planning**: Starting from a goal state (e.g., `on(a, b)`), apply all possible PDDL actions to "destroy" the current state, exploring what states can lead to this goal. This builds a complete state graph that can be used to generate context-sensitive AgentSpeak plans.

---

## Core Design Decisions

### 1. DFA Semantics
- **Decision**: Transition label is both the goal state AND the precondition for the transition
- **Implication**: When DFA has transition `1 --[on(a,b)]-> 2`, we need to achieve `on(a,b)` to make this transition happen

### 2. Search Direction
- **Decision**: Forward "destruction" from goal state
- **Method**:
  - Start with goal predicates (e.g., `{on(a, b), clear(c)}`)
  - Apply all possible PDDL actions to this state
  - Generate new states by modifying predicates according to action effects
  - Continue until no new states can be generated
- **Rationale**: Easier to track what states can lead to goal, builds complete graph

### 3. State Representation
- **Decision**: "Minimal predicates" - dynamically expanded from goal
- **Method**:
  - Start with predicates in DFA transition label
  - As actions are applied, track all affected predicates
  - NOT just predicates in original goal - include all relevant world state
- **Example**: Goal `on(a, b)` may expand to include `holding(a)`, `clear(b)`, `ontable(a)`, etc.

### 4. Non-Deterministic Effects
- **Decision**: Generate separate plans for each `oneof` branch
- **Method**: When PDDL action has `oneof` effects, create multiple state transitions (one per branch)
- **AgentSpeak Handling**: Multiple plans with different contexts - runtime selects matching one

### 5. Search Termination
- **Decision**: Dynamic depth limit based on goal complexity
- **Heuristics**:
  - 1 predicate: depth = 5
  - 2-3 predicates: depth = 10
  - 4+ predicates: depth = 20
- **Additional Condition**: Stop when all frontier states produce no new states (all next states already in graph)

### 6. Graph Structure
- **Decision**: Allow cycles in state graph
- **Plan Extraction**: Use only acyclic paths (BFS to find shortest paths)
- **Rationale**: Cycles may exist in state space, but plans should not loop infinitely

### 7. Plan Generation Strategy
- **Decision**: Generate one plan per non-goal state
- **Method**: BFS from each state to goal, extract shortest path
- **Plan Structure**:
  ```asl
  +!goal : context <-
      !precond1;        // Subgoals for action preconditions
      !precond2;
      action(args);     // Physical action (with embedded belief updates)
      !goal.            // Recursive call to check if goal achieved
  ```

### 8. Context Definition
- **Decision**: Context = all minimal predicates in current state
- **Format**: AgentSpeak conjunction `holding(a) & clear(b) & handempty`
- **Empty State**: Context is `true`

### 9. Plan Body Structure
- **Decision**: Subgoals for preconditions + action + recursive call
- **Rationale**: Action preconditions may not be satisfied, need to establish them first
- **Ordering Matters**: Some preconditions may conflict (achieving one breaks another)

### 10. Precondition Handling
- **Decision**:
  - If precondition violated (known to be false): Skip action
  - If precondition unknown (not in state): Generate subgoal
  - If precondition satisfied: Proceed
- **Recursive Subgoals**: Yes - preconditions may require multiple steps to achieve

### 11. DFA Processing
- **Decision**: Process each transition independently
- **Method**: For each edge in DFA, run separate backward planning
- **Output**: Multiple goal names (one per transition)

### 12. Belief Updates
- **Decision**: Physical actions must include explicit belief updates
- **Method**:
  - Convert each PDDL action to AgentSpeak action
  - Include belief updates derived from PDDL effects
  - Action definitions separate from plan definitions
- **Format**:
  ```asl
  +!pickup(X) : handempty & ontable(X) & clear(X) <-
      pickup_physical(X);  // External action
      +holding(X);         // Belief updates from PDDL effects
      -ontable(X);
      -handempty.
  ```

### 13. Boolean Operators in Transition Labels
- **Decision**: Convert complex boolean expressions to DNF (Disjunctive Normal Form)
- **Method**: For label like `on(a,b) | clear(c)`, generate two independent explorations
- **Operators Supported**: `&, &&, |, ||, !, ~, ->, =>, <->, <=>`

### 14. Initial Beliefs
- **Decision**: Fixed initial state based on domain
- **Blocksworld Default**: All blocks on table, hand empty, all blocks clear
- **Format**:
  ```asl
  // Initial beliefs
  ontable(a).
  ontable(b).
  ontable(c).
  clear(a).
  clear(b).
  clear(c).
  handempty.
  ```

### 15. Jason Compatibility
- **Decision**: Ensure full Jason syntax compliance
- **Requirements**:
  - Proper plan syntax
  - Valid belief literals
  - Correct action calls
  - Initial belief declarations
  - Entry point plans (e.g., `+!start`)

### 16. Visualization
- **Decision**: Generate DOT format for state graphs
- **Purpose**: Debugging and understanding state space
- **Output**: `.dot` files alongside `.asl` files

---

## Complete Q&A Record

### Round 1: Initial Clarification

**Q1**: DFA transition label是goal state还是前提条件？
**A1**: 既是goal state，也是transition的前提条件

**Q2**: "Destroy state"是regression还是其他？
**A2**: 是Destruction - 从当前state尝试所有actions，探索向外一步会产生哪些states

**Q3**: "Minimal predicates"的定义？
**A3**: 从goal state开始向外探索，记录所有受影响的predicates。不是只包含DFA label中的predicates

**Q4**: 如何处理PDDL non-deterministic effects (oneof)？
**A4**: 为每个分支生成不同的plan

---

### Round 2: Technical Details

**Q5**: 搜索空间很大时需要depth limit吗？
**A5**: 动态决定，根据goal复杂度设置。简单goal深度限制小，复杂goal限制大

**Q6**: 多条路径如何生成plans - 多个plans还是选最短？
**A6**: BFS找最短路径，每一步选择最近的next state

**Q7**: State graph是DAG还是允许环？
**A7**: 允许环，但plan提取时只用acyclic paths

**Q8**: Context condition包含什么信息？
**A8**: 当前state的所有minimal predicates

---

### Round 3: Plan Body Details

**Q9**: Plan body具体形式？
**A9**: 单个action + (action内部处理belief updates) + 递归subgoal `!goal`。例如：
```asl
+!on(a,b) : holding(a) <- putdown(a); !on(a,b)
```

**Q10**: Action preconditions不满足怎么办？
**A10**:
- 已知违反 → 跳过action
- 未知（state为空） → 可生成subgoal需求

**Q11**: DFA多个transitions如何处理？
**A11**: 为每个transition单独做backward planning

**Q12**: 是否需要显式belief updates？
**A12**: 需要，根据PDDL action effects生成

---

### Round 4: Final Clarifications

**Q13**: Empty state是什么？Goal state是empty吗？
**A13**: Goal state是包含transition label predicates的state（例如`{on(a,b)}`），不是empty

**Q14**: Leaf states的类型？
**A14**: 所有类型都是valid执行起点：
- 类型1: Predicates全部被destroy（空state）
- 类型2: 有predicates但无valid actions
- 类型3: 所有next states都已在图中

**Q15**: 复杂Boolean表达式（|, ->, <=>）如何处理？
**A15**: 转换为DNF，为每个disjunct独立探索

**Q16**: Precondition subgoals需要递归处理吗？
**A16**: 是的，需要递归，并且顺序可能影响结果（某些precondition达成会破坏之前的）

**Q17**: PDDL initial state从哪来？
**A17**: 现阶段假设固定数量blocks都在桌子上

**Q18**: Physical action如何更新beliefs？
**A18**: PDDL actions转换到AgentSpeak时，需要生成包含belief updates的action定义

---

## Technical Architecture

### High-Level Flow

```
DFA Transitions
    ↓
[Parse Transition Labels]
    ↓
Goal Predicates (DNF if complex boolean)
    ↓
[Forward State Planner]
    ↓
State Graph (states + transitions)
    ↓
[Path Extraction] (BFS)
    ↓
Paths from leaf states to goal
    ↓
[AgentSpeak Code Generator]
    ↓
.asl File (actions + plans + initial beliefs)
```

### Module Breakdown

```
src/stage3_code_generation/
├── backward_planner_generator.py    # Main entry point
├── state_space.py                   # Data structures
├── forward_planner.py               # State space exploration
├── pddl_parser.py                   # Parse PDDL conditions/effects
├── agentspeak_codegen.py            # Generate .asl code
├── boolean_expression_parser.py     # Parse transition labels
└── visualization.py                 # DOT file generation
```

---

## Implementation Plan

### Phase 1: Core Data Structures (Week 1)
**Files**: `state_space.py`, `pddl_parser.py`

**Tasks**:
1. Implement `PredicateAtom`, `WorldState`, `StateGraph`
2. Implement `PDDLConditionParser` for preconditions
3. Implement `PDDLEffectParser` for effects (with oneof support)
4. Unit tests for parsing and data structures

**Deliverables**:
- ✅ Can parse PDDL preconditions/effects
- ✅ Can represent world states and state graphs
- ✅ All data structures tested

---

### Phase 2: Forward Planner (Week 2)
**Files**: `forward_planner.py`, `boolean_expression_parser.py`

**Tasks**:
1. Implement `ForwardStatePlanner.explore_from_goal()`
2. Implement action grounding logic
3. Implement precondition checking
4. Implement effect application (with oneof branching)
5. Implement DNF conversion for boolean expressions
6. Dynamic depth calculation
7. Unit tests for state exploration

**Deliverables**:
- ✅ Can generate complete state graph from goal
- ✅ Handles non-deterministic effects
- ✅ Handles complex boolean expressions
- ✅ Terminates correctly with depth limits

---

### Phase 3: AgentSpeak Code Generation (Week 3)
**Files**: `agentspeak_codegen.py`

**Tasks**:
1. Implement `AgentSpeakCodeGenerator.generate_plans()`
2. Implement BFS path extraction
3. Implement context formatting
4. Implement plan body generation (with precondition subgoals)
5. Implement action definition generation (from PDDL actions)
6. Implement initial beliefs generation
7. Unit tests for code generation

**Deliverables**:
- ✅ Can generate syntactically correct AgentSpeak code
- ✅ Plans include proper contexts and subgoals
- ✅ Action definitions include belief updates
- ✅ Jason-compatible output

---

### Phase 4: DFA Integration (Week 4)
**Files**: `backward_planner_generator.py`, modifications to `ltl_bdi_pipeline.py`

**Tasks**:
1. Implement `BackwardPlannerGenerator`
2. Implement DFA parser to extract transitions from raw MONA DOT format
3. Handle multiple transitions
4. Integrate into main pipeline
5. End-to-end testing

**Deliverables**:
- ✅ Complete pipeline from DFA to .asl
- ✅ Works with existing Stage 1 and Stage 2
- ✅ No LLM dependencies

---

### Phase 5: Visualization & Testing (Week 5)
**Files**: `visualization.py`, test files

**Tasks**:
1. Implement state graph visualization (DOT format)
2. Comprehensive integration tests
3. Test with blocksworld domain
4. Test with complex boolean expressions
5. Performance optimization
6. Bug fixes and edge cases

**Deliverables**:
- ✅ Full test coverage
- ✅ Visualization for debugging
- ✅ Performance benchmarks
- ✅ Production-ready code

---

## Data Structures

### PredicateAtom
```python
@dataclass
class PredicateAtom:
    """Represents a ground predicate: on(a, b)"""
    name: str              # "on"
    args: List[str]        # ["a", "b"]
    negated: bool = False  # True for ~on(a, b)

    def to_agentspeak(self) -> str:
        """Convert to AgentSpeak format: not on(a, b)"""
        prefix = "not " if self.negated else ""
        if self.args:
            return f"{prefix}{self.name}({', '.join(self.args)})"
        return f"{prefix}{self.name}"

    def __hash__(self):
        return hash((self.name, tuple(self.args), self.negated))

    def __eq__(self, other):
        return (self.name == other.name and
                self.args == other.args and
                self.negated == other.negated)
```

### WorldState
```python
@dataclass
class WorldState:
    """Represents a world state (set of predicates)"""
    predicates: FrozenSet[PredicateAtom]
    depth: int = 0  # Distance from goal state

    def __hash__(self):
        return hash(self.predicates)

    def __eq__(self, other):
        return self.predicates == other.predicates

    def __repr__(self):
        preds = ", ".join(p.to_agentspeak() for p in self.predicates)
        return f"State(depth={self.depth}, predicates=[{preds}])"
```

### StateTransition
```python
@dataclass
class StateTransition:
    """Represents an edge in the state graph"""
    from_state: WorldState
    to_state: WorldState
    action: PDDLAction
    action_args: List[str]          # Ground arguments
    belief_updates: List[str]       # ["+holding(a)", "-ontable(a)"]
    preconditions: List[PredicateAtom]  # For generating subgoals

    def __repr__(self):
        action_call = f"{self.action.name}({', '.join(self.action_args)})"
        return f"Transition({action_call}: {self.from_state} -> {self.to_state})"
```

### StateGraph
```python
class StateGraph:
    """Forward exploration state space graph"""

    def __init__(self, goal_state: WorldState):
        self.goal_state = goal_state
        self.states: Set[WorldState] = {goal_state}
        self.transitions: List[StateTransition] = []
        self.state_to_outgoing: Dict[WorldState, List[StateTransition]] = {}
        self.state_to_incoming: Dict[WorldState, List[StateTransition]] = {}

    def add_transition(self, transition: StateTransition):
        """Add edge to graph"""
        self.states.add(transition.from_state)
        self.states.add(transition.to_state)
        self.transitions.append(transition)

        # Build adjacency lists
        if transition.from_state not in self.state_to_outgoing:
            self.state_to_outgoing[transition.from_state] = []
        self.state_to_outgoing[transition.from_state].append(transition)

        if transition.to_state not in self.state_to_incoming:
            self.state_to_incoming[transition.to_state] = []
        self.state_to_incoming[transition.to_state].append(transition)

    def find_shortest_paths_to_goal(self) -> Dict[WorldState, List[StateTransition]]:
        """BFS to find shortest path from each state to goal"""
        paths = {}

        # BFS from goal (backward direction)
        queue = deque([(self.goal_state, [])])
        visited = {self.goal_state}

        while queue:
            current, path = queue.popleft()
            paths[current] = path

            # Explore incoming edges (reverse direction)
            for transition in self.state_to_incoming.get(current, []):
                if transition.from_state not in visited:
                    visited.add(transition.from_state)
                    queue.append((transition.from_state, [transition] + path))

        return paths

    def get_leaf_states(self) -> Set[WorldState]:
        """Get all leaf states (no outgoing transitions or all outgoing lead to visited)"""
        return {s for s in self.states if s not in self.state_to_outgoing}

    def to_dot(self) -> str:
        """Generate DOT format for visualization"""
        lines = ["digraph StateGraph {"]
        lines.append("  rankdir=LR;")

        # Add goal state styling
        lines.append(f'  "{id(self.goal_state)}" [label="GOAL\\n{self.goal_state}", shape=doublecircle, color=green];')

        # Add other states
        for state in self.states:
            if state != self.goal_state:
                label = "\\n".join(p.to_agentspeak() for p in state.predicates) or "empty"
                lines.append(f'  "{id(state)}" [label="{label}", shape=circle];')

        # Add transitions
        for trans in self.transitions:
            action_label = f"{trans.action.name}({', '.join(trans.action_args)})"
            lines.append(f'  "{id(trans.from_state)}" -> "{id(trans.to_state)}" [label="{action_label}"];')

        lines.append("}")
        return "\n".join(lines)
```

---

## Algorithms

### Algorithm 1: Forward State Space Exploration

```python
def explore_from_goal(goal_predicates: List[PredicateAtom],
                      max_depth: int) -> StateGraph:
    """
    Forward exploration from goal state

    Algorithm:
    1. Initialize goal state with goal predicates
    2. BFS/DFS exploration:
       a. For each state, try all possible ground actions
       b. Check if action preconditions are satisfied
       c. Apply action effects to generate new states
       d. Add transitions to graph
       e. Continue until max_depth or no new states
    3. Return complete state graph
    """

    # Step 1: Initialize
    goal_state = WorldState(predicates=frozenset(goal_predicates), depth=0)
    graph = StateGraph(goal_state)

    queue = deque([goal_state])
    visited = {goal_state}

    # Step 2: BFS Exploration
    while queue:
        current_state = queue.popleft()

        # Check depth limit
        if current_state.depth >= max_depth:
            continue

        # Try all ground actions
        for action in domain.actions:
            for grounded_action in ground_action(action, objects):

                # Step 2a: Check preconditions
                if not check_preconditions(grounded_action, current_state):
                    continue  # Skip invalid action

                # Step 2b: Apply action (handle oneof)
                new_states_with_updates = apply_action(grounded_action, current_state)

                for new_state, belief_updates, preconditions in new_states_with_updates:
                    # Step 2c: Create transition
                    transition = StateTransition(
                        from_state=current_state,
                        to_state=new_state,
                        action=action,
                        action_args=grounded_action['args'],
                        belief_updates=belief_updates,
                        preconditions=preconditions
                    )
                    graph.add_transition(transition)

                    # Step 2d: Add to queue if new
                    if new_state not in visited:
                        new_state.depth = current_state.depth + 1
                        visited.add(new_state)
                        queue.append(new_state)

    return graph
```

### Algorithm 2: Precondition Checking

```python
def check_preconditions(grounded_action: Dict, state: WorldState) -> bool:
    """
    Check if action's preconditions are satisfied in current state

    Three cases:
    1. Precondition violated (known false) → return False
    2. Precondition unknown (not in state) → can proceed (will generate subgoal later)
    3. Precondition satisfied → return True
    """

    precond_predicates = parse_preconditions(
        grounded_action['action'].preconditions,
        grounded_action['bindings']
    )

    for precond in precond_predicates:
        if precond.negated:
            # Requires predicate to NOT exist
            positive_pred = PredicateAtom(precond.name, precond.args, negated=False)
            if positive_pred in state.predicates:
                return False  # VIOLATION: predicate exists but shouldn't
        else:
            # Requires predicate to exist
            if len(state.predicates) > 0 and precond not in state.predicates:
                # State is non-empty but missing required predicate
                # Still allow (will generate subgoal in plan)
                pass

    return True
```

### Algorithm 3: Action Effect Application

```python
def apply_action(grounded_action: Dict, state: WorldState) -> List[Tuple[WorldState, List[str], List[PredicateAtom]]]:
    """
    Apply action to state, return possible new states

    Handles:
    - Add effects: +predicate
    - Delete effects: -predicate
    - Non-deterministic effects: oneof branches

    Returns: List of (new_state, belief_updates, preconditions)
    """

    effects = parse_effects(
        grounded_action['action'].effects,
        grounded_action['bindings']
    )

    preconditions = parse_preconditions(
        grounded_action['action'].preconditions,
        grounded_action['bindings']
    )

    results = []

    # Handle oneof branches
    for effect_branch in effects:
        new_predicates = set(state.predicates)
        belief_updates = []

        for effect in effect_branch:
            if effect.is_add:
                # Add effect: +on(a, b)
                new_predicates.add(effect.predicate)
                belief_updates.append(f"+{effect.predicate.to_agentspeak()}")
            else:
                # Delete effect: -ontable(a)
                new_predicates.discard(effect.predicate)
                belief_updates.append(f"-{effect.predicate.to_agentspeak()}")

        new_state = WorldState(predicates=frozenset(new_predicates))
        results.append((new_state, belief_updates, preconditions))

    return results
```

### Algorithm 4: AgentSpeak Plan Generation

```python
def generate_plan_for_state(state: WorldState,
                            path: List[StateTransition],
                            goal_name: str) -> str:
    """
    Generate AgentSpeak plan for a given state

    Plan structure:
    +!goal : context <-
        !precond1;
        !precond2;
        action(args);
        !goal.
    """

    if not path:
        # Already at goal - generate success plan
        return generate_success_plan(state, goal_name)

    # Next step in path
    next_transition = path[0]

    # Format context
    context = format_context(state.predicates)

    # Generate precondition subgoals
    subgoals = []
    for precond in next_transition.preconditions:
        if precond not in state.predicates:
            # Need to establish this precondition
            subgoal_name = predicate_to_goal_name(precond)
            subgoals.append(f"!{subgoal_name}")

    # Format action call
    action_call = format_action_call(
        next_transition.action,
        next_transition.action_args
    )

    # Build plan body
    body_lines = []
    body_lines.extend(subgoals)
    body_lines.append(action_call)
    body_lines.append(f"!{goal_name}")  # Recursive call

    body = ";\n    ".join(body_lines)

    plan = f"+!{goal_name} : {context} <-\n    {body}."

    return plan
```

### Algorithm 5: DNF Conversion for Boolean Expressions

```python
def convert_to_dnf(boolean_expr: str) -> List[List[PredicateAtom]]:
    """
    Convert boolean expression to Disjunctive Normal Form

    Example:
        Input: "on(a,b) | (clear(c) & holding(d))"
        Output: [[on(a,b)], [clear(c), holding(d)]]

    Each inner list is a conjunction (AND), outer list is disjunction (OR)
    """

    # Step 1: Parse expression into AST
    tokens = tokenize(boolean_expr)
    ast = parse_boolean_expression(tokens)

    # Step 2: Apply logical transformations
    # - Eliminate implications: A -> B becomes ~A | B
    # - Eliminate equivalences: A <=> B becomes (A & B) | (~A & ~B)
    ast = eliminate_implications(ast)

    # Step 3: Push negations inward (De Morgan's laws)
    ast = push_negations(ast)

    # Step 4: Distribute OR over AND
    ast = distribute_or_over_and(ast)

    # Step 5: Extract disjuncts (top-level OR)
    disjuncts = extract_disjuncts(ast)

    # Step 6: For each disjunct, extract conjuncts (predicates)
    result = []
    for disjunct in disjuncts:
        conjuncts = extract_conjuncts(disjunct)
        result.append(conjuncts)

    return result
```

---

## Example Outputs

### Example 1: Simple Blocksworld Goal

**Input**:
- DFA transition: `1 --[on_a_b]-> 2`
- Goal predicate: `on(a, b)`
- Objects: `[a, b, c]`

**State Graph** (simplified):
```
State 0 (GOAL): {on(a, b)}
  ├─ putdown(a, b) ─> State 1: {holding(a), clear(b)}
  │   ├─ pickup(a) ─> State 2: {ontable(a), clear(a), handempty}
  │   └─ unstack(a, c) ─> State 3: {on(a, c), clear(a), handempty}
  └─ stack(a, b) ─> State 4: {holding(a), clear(b)}
      └─ ... (same as State 1)
```

**Generated AgentSpeak Code**:
```asl
/* AgentSpeak Plan Library
 * Generated from LTLf specification
 * Goal: on(a, b)
 */

// Initial beliefs
ontable(a).
ontable(b).
ontable(c).
clear(a).
clear(b).
clear(c).
handempty.

// Action definitions (with belief updates)
+!pickup_physical(X) : handempty & ontable(X) & clear(X) <-
    .external_action(pickup, X);
    +holding(X);
    -ontable(X);
    -clear(X);
    -handempty.

+!putdown_physical(X, Y) : holding(X) & clear(Y) <-
    .external_action(putdown, X, Y);
    +on(X, Y);
    -holding(X);
    +handempty;
    -clear(Y).

// Plans for goal: on(a, b)

// Plan 1: Already achieved
+!on(a, b) : on(a, b) <-
    .print("Goal on(a, b) achieved!").

// Plan 2: From state {holding(a), clear(b)}
+!on(a, b) : holding(a) & clear(b) <-
    !putdown_physical(a, b);
    !on(a, b).

// Plan 3: From state {ontable(a), clear(a), clear(b), handempty}
+!on(a, b) : ontable(a) & clear(a) & clear(b) & handempty <-
    !pickup_physical(a);
    !on(a, b).

// Plan 4: From state {on(a, c), clear(a), clear(b), handempty}
+!on(a, b) : on(a, c) & clear(a) & clear(b) & handempty <-
    !unstack_physical(a, c);
    !on(a, b).

// Failure handler
-!on(a, b) : true <-
    .print("Failed to achieve on(a, b)");
    .fail.
```

### Example 2: Complex Boolean Expression

**Input**:
- DFA transition: `1 --[on_a_b | clear_c]-> 2`
- Boolean expression: `on(a, b) | clear(c)`

**DNF Conversion**:
```
Disjunct 1: [on(a, b)]
Disjunct 2: [clear(c)]
```

**Processing**:
1. Generate state graph for `on(a, b)` → Goal name: `on_a_b`
2. Generate state graph for `clear(c)` → Goal name: `clear_c`
3. Combine plans into single .asl file

**Generated Code** (partial):
```asl
// Plans for on(a, b)
+!on_a_b : on(a, b) <- .print("on(a, b) achieved").
+!on_a_b : holding(a) & clear(b) <- !putdown_physical(a, b); !on_a_b.
...

// Plans for clear(c)
+!clear_c : clear(c) <- .print("clear(c) achieved").
+!clear_c : on(X, c) <- !unstack_physical(X, c); !clear_c.
...

// Main goal (either on_a_b OR clear_c)
+!transition_1_to_2 : true <-
    !on_a_b | !clear_c.  // AgentSpeak tries both
```

### Example 3: Non-Deterministic Action (oneof)

**PDDL Action**:
```lisp
(:action pickup
 :parameters (?b1 - block)
 :precondition (and (clear ?b1) (ontable ?b1) (handempty))
 :effect (oneof
   (and (holding ?b1) (not (ontable ?b1)) (not (handempty)))  ; Success
   (and (ontable ?b1))  ; Failure (no change)
 ))
```

**State Graph**:
```
State: {ontable(a), clear(a), handempty}
  ├─ pickup(a) [branch 1] ─> State: {holding(a)}
  └─ pickup(a) [branch 2] ─> State: {ontable(a), clear(a), handempty}  (same state)
```

**Generated Plans**:
```asl
// Plan for successful pickup
+!on(a, b) : ontable(a) & clear(a) & handempty <-
    !pickup_physical(a);
    !on(a, b).

// Plan for failed pickup (same preconditions, but different outcome)
// AgentSpeak runtime will retry or fail gracefully
```

---

## Testing Strategy

### Unit Tests

#### Test 1: PDDL Parsing
```python
def test_parse_preconditions():
    """Test parsing PDDL preconditions"""
    precond_str = "and (handempty) (clear ?b1) (not (= ?b1 ?b2))"
    bindings = {'?b1': 'a', '?b2': 'b'}

    parser = PDDLConditionParser()
    predicates = parser.parse(precond_str, bindings)

    assert len(predicates) == 2
    assert PredicateAtom('handempty', []) in predicates
    assert PredicateAtom('clear', ['a']) in predicates

def test_parse_effects_with_oneof():
    """Test parsing non-deterministic effects"""
    effect_str = "oneof (and (holding ?b1) (not (handempty))) (and (ontable ?b1))"
    bindings = {'?b1': 'a'}

    parser = PDDLEffectParser()
    branches = parser.parse(effect_str, bindings)

    assert len(branches) == 2
    assert len(branches[0]) == 2  # holding, not handempty
    assert len(branches[1]) == 1  # ontable
```

#### Test 2: State Space Exploration
```python
def test_forward_exploration():
    """Test forward state space exploration"""
    goal_predicates = [PredicateAtom('on', ['a', 'b'])]

    planner = ForwardStatePlanner(domain, grounding_map, objects=['a', 'b', 'c'])
    graph = planner.explore_from_goal(goal_predicates, max_depth=5)

    assert graph.goal_state in graph.states
    assert len(graph.states) > 1  # Should generate multiple states
    assert len(graph.transitions) > 0

    # Check goal state is reachable
    paths = graph.find_shortest_paths_to_goal()
    assert graph.goal_state in paths

def test_precondition_violation():
    """Test that actions with violated preconditions are skipped"""
    state = WorldState(predicates=frozenset([
        PredicateAtom('holding', ['a'])  # Hand NOT empty
    ]))

    action = get_action_by_name(domain, 'pickup')
    grounded = {'action': action, 'args': ['b'], 'bindings': {'?b1': 'b'}}

    planner = ForwardStatePlanner(domain, grounding_map, objects)
    result = planner._check_preconditions(grounded, state)

    assert result == False  # Should fail because handempty required
```

#### Test 3: AgentSpeak Code Generation
```python
def test_generate_plan():
    """Test AgentSpeak plan generation"""
    state = WorldState(predicates=frozenset([
        PredicateAtom('holding', ['a']),
        PredicateAtom('clear', ['b'])
    ]))

    # Create mock transition
    transition = StateTransition(
        from_state=state,
        to_state=goal_state,
        action=putdown_action,
        action_args=['a', 'b'],
        belief_updates=['+on(a, b)', '-holding(a)', '+handempty'],
        preconditions=[
            PredicateAtom('holding', ['a']),
            PredicateAtom('clear', ['b'])
        ]
    )

    codegen = AgentSpeakCodeGenerator(state_graph, goal_name='on(a, b)')
    plan = codegen._generate_plan_for_state(state, [transition])

    assert '+!on(a, b)' in plan
    assert 'holding(a) & clear(b)' in plan
    assert 'putdown' in plan
    assert '!on(a, b)' in plan  # Recursive call
```

### Integration Tests

#### Test 4: End-to-End Pipeline
```python
def test_full_pipeline_blocksworld():
    """Test complete pipeline with blocksworld domain"""

    # Setup
    domain = load_pddl_domain('domains/blocksworld.pddl')
    ltl_spec = create_ltl_spec(
        formula='F(on_a_b & on_b_c)',
        objects=['a', 'b', 'c']
    )
    dfa_result = generate_dfa(ltl_spec)

    # Run Stage 3
    generator = BackwardPlannerGenerator(domain, ltl_spec.grounding_map)
    asl_code = generator.generate(ltl_spec, dfa_result)

    # Validate output
    assert 'on(a, b)' in asl_code
    assert '+!' in asl_code
    assert '<-' in asl_code

    # Check Jason syntax validity
    validate_jason_syntax(asl_code)

    # Save and visualize
    save_file('output/test_blocksworld.asl', asl_code)

    # Check visualization
    assert os.path.exists('output/test_blocksworld_state_graph.dot')
```

#### Test 5: Complex Boolean Expression
```python
def test_complex_boolean_expression():
    """Test handling of complex boolean expressions in DFA"""

    # DFA with OR expression
    dfa_with_or = create_mock_dfa(
        transitions=[
            (1, 2, 'on_a_b | clear_c')
        ]
    )

    generator = BackwardPlannerGenerator(domain, grounding_map)
    asl_code = generator.generate(ltl_spec, dfa_with_or)

    # Should generate plans for both disjuncts
    assert 'on_a_b' in asl_code
    assert 'clear_c' in asl_code
```

### Performance Tests

#### Test 6: Scalability
```python
def test_scalability():
    """Test performance with increasing complexity"""

    test_cases = [
        {'objects': ['a', 'b'], 'max_depth': 5},
        {'objects': ['a', 'b', 'c'], 'max_depth': 8},
        {'objects': ['a', 'b', 'c', 'd'], 'max_depth': 10},
    ]

    for case in test_cases:
        start_time = time.time()

        planner = ForwardStatePlanner(domain, grounding_map, case['objects'])
        graph = planner.explore_from_goal(goal_predicates, case['max_depth'])

        elapsed = time.time() - start_time

        print(f"Objects: {len(case['objects'])}, "
              f"States: {len(graph.states)}, "
              f"Transitions: {len(graph.transitions)}, "
              f"Time: {elapsed:.2f}s")

        assert elapsed < 60  # Should complete within 1 minute
```

---

## Potential Challenges

### Challenge 1: State Space Explosion

**Problem**: For complex goals with many objects, state space may grow exponentially.

**Example**:
- 5 blocks → ~100 possible states
- 10 blocks → ~10,000 states

**Solutions**:
1. **Dynamic Depth Limiting**: Adjust max_depth based on goal complexity
   ```python
   def calculate_max_depth(goal_predicates):
       n = len(goal_predicates)
       if n == 1: return 5
       elif n <= 3: return 10
       else: return 15
   ```

2. **State Pruning**: Remove predicates clearly irrelevant to goal
   ```python
   def is_relevant_predicate(pred, goal_predicates):
       # Only keep predicates involving objects in goal
       goal_objects = set(obj for g in goal_predicates for obj in g.args)
       return any(arg in goal_objects for arg in pred.args)
   ```

3. **Heuristic Guidance**: Prioritize states closer to "empty" or initial state
   ```python
   def heuristic(state):
       # Prefer states with fewer predicates (closer to empty)
       return len(state.predicates)

   # Use priority queue instead of regular queue
   queue = PriorityQueue()
   queue.put((heuristic(goal_state), goal_state))
   ```

4. **Early Termination**: Stop when sufficient plans generated
   ```python
   if len(graph.states) > MAX_STATES or len(paths) > MAX_PLANS:
       break
   ```

---

### Challenge 2: PDDL Parsing Complexity

**Problem**: PDDL supports complex nested expressions (and, or, not, forall, exists, when).

**Example**:
```lisp
(:action move
 :precondition (and
   (not (= ?from ?to))
   (forall (?b - block)
     (when (on ?b ?from) (clear ?b))))
 :effect ...)
```

**Solutions**:
1. **Recursive Descent Parser**: Handle nested s-expressions
   ```python
   def parse_sexp(tokens):
       if tokens[0] == '(':
           expr_type = tokens[1]
           if expr_type == 'and':
               return {'type': 'and', 'children': [parse_sexp(t) for t in ...]}
           elif expr_type == 'not':
               return {'type': 'not', 'child': parse_sexp(...)}
           ...
   ```

2. **Incremental Support**: Start with blocksworld subset, expand later
   - Phase 1: Support `and`, `not`, simple predicates
   - Phase 2: Add `or`, `forall`, `exists`
   - Phase 3: Add `when`, conditional effects

3. **Leverage Existing Libraries**: Consider `pddlpy` or `pyperplan`
   ```python
   from pddl.parser import Parser

   parser = Parser()
   domain = parser.parse_domain('blocksworld.pddl')

   # Extract preconditions/effects in structured format
   ```

---

### Challenge 3: Precondition Ordering Conflicts

**Problem**: Achieving one precondition may break another.

**Example**:
```
Goal: stack(a, b)
Preconditions: holding(a) & clear(b)

Current state: {ontable(a), on(c, b), handempty}

Issue:
- To achieve holding(a): pickup(a) ✓
- To achieve clear(b): unstack(c, b) → but this makes holding(c), breaking handempty needed for pickup(a)!
```

**Solutions**:
1. **Dependency Analysis**: Detect conflicting preconditions
   ```python
   def check_precondition_conflicts(precond1, precond2):
       # Check if achieving precond1 negates precond2
       if precond1.name == precond2.name and precond1.negated != precond2.negated:
           return True
       return False
   ```

2. **Ordering Constraints**: Determine safe ordering
   ```python
   def order_preconditions(preconditions):
       # Use dependency graph to find safe ordering
       # Rule: If achieving A breaks B, do B first
       ordered = topological_sort(build_dependency_graph(preconditions))
       return ordered
   ```

3. **Multiple Plans**: Generate different plans for different orderings
   ```python
   for ordering in all_permutations(preconditions):
       plan = generate_plan_with_ordering(ordering)
       plans.append(plan)
   ```

4. **Runtime Handling**: Let AgentSpeak runtime select appropriate plan
   ```asl
   // Plan 1: If already holding(a), just need clear(b)
   +!stack(a, b) : holding(a) & not clear(b) <-
       !clear(b);
       stack(a, b).

   // Plan 2: If clear(b), just need holding(a)
   +!stack(a, b) : not holding(a) & clear(b) <-
       !holding(a);
       stack(a, b).

   // Plan 3: Neither - need both in safe order
   +!stack(a, b) : not holding(a) & not clear(b) <-
       !clear(b);
       !holding(a);
       stack(a, b).
   ```

---

### Challenge 4: Non-Deterministic Effects Multiplication

**Problem**: Actions with `oneof` create multiple branches, exponentially increasing plans.

**Example**:
```
Action1: oneof (outcome_a1, outcome_a2)
Action2: oneof (outcome_b1, outcome_b2)
Action3: oneof (outcome_c1, outcome_c2)

Total paths: 2 × 2 × 2 = 8 plans
```

**Solutions**:
1. **Branch Pruning**: Merge branches that lead to same state
   ```python
   def add_transition(self, transition):
       # Check if to_state already exists
       if transition.to_state in self.states:
           # Don't create new state, just add edge
           pass
   ```

2. **Plan Deduplication**: Merge plans with identical contexts
   ```python
   def deduplicate_plans(plans):
       seen_contexts = {}
       unique_plans = []

       for plan in plans:
           context_key = frozenset(plan.context)
           if context_key not in seen_contexts:
               seen_contexts[context_key] = plan
               unique_plans.append(plan)

       return unique_plans
   ```

3. **Subsumption Checking**: Remove plans subsumed by others
   ```python
   def is_subsumed(plan1, plan2):
       # plan1 is subsumed by plan2 if:
       # - plan2's context is a subset of plan1's context
       # - plan2's body is equivalent to plan1's body
       return (plan2.context.issubset(plan1.context) and
               plan2.body == plan1.body)
   ```

4. **Limit Branch Depth**: Only explore N levels of non-determinism
   ```python
   MAX_NONDETERMINISTIC_DEPTH = 3

   if branch_depth > MAX_NONDETERMINISTIC_DEPTH:
       # Only keep most likely branch (e.g., success branch)
       branches = [branches[0]]
   ```

---

### Challenge 5: Visualization Scalability

**Problem**: Large state graphs (100+ states) difficult to visualize.

**Solutions**:
1. **Hierarchical Visualization**: Group states by depth
2. **Interactive Viewer**: Use web-based tool (e.g., Cytoscape.js)
3. **Selective Rendering**: Only show states on shortest paths
4. **Abstraction**: Merge similar states

---

## Open Questions & Future Work

### Open Questions
1. **Optimal Depth Limits**: Is there a theoretical bound (like pumping lemma) for state space exploration depth?
2. **Completeness**: Does this approach guarantee finding all possible plans?
3. **Optimality**: Are generated plans optimal (shortest execution)?

### Future Enhancements
1. **Learning-Based Depth Prediction**: Use ML to predict optimal max_depth
2. **Parallel Exploration**: Multi-threaded state space exploration
3. **Incremental Planning**: Cache and reuse state graphs across runs
4. **Domain-Specific Optimizations**: Specialized handling for common domains (blocksworld, logistics, rovers)
5. **Plan Quality Metrics**: Rank plans by expected success rate, execution time, etc.
6. **Interactive Debugging**: Web UI for exploring state graphs and generated plans

---

## References

### PDDL Resources
- [PDDL Specification](https://planning.wiki/)
- [Blocksworld Domain](https://github.com/potassco/pddl-instances/blob/master/ipc-2000/domains/blocks-strips-typed/domain.pddl)

### AgentSpeak Resources
- [Jason Manual](http://jason.sourceforge.net/wp/documents/)
- [AgentSpeak(L) Specification](https://www.sciencedirect.com/science/article/pii/S0004370296000674)

### Planning Algorithms
- [Fast Forward (FF) Planner](https://fai.cs.uni-saarland.de/hoffmann/ff.html)
- [GraphPlan](https://www.cs.cmu.edu/~avrim/Papers/graphplan.pdf)

---

## Maintenance Log

| Date | Author | Change |
|------|--------|--------|
| 2025-11-06 | Claude | Initial design document created |
|  |  |  |

---

## Appendix A: AgentSpeak Syntax Reference

### Plan Structure
```asl
+!goal_name : context_condition <-
    subgoal1;
    subgoal2;
    action(args);
    +belief;
    -old_belief;
    !recursive_goal.
```

### Triggering Events
- `+!goal`: Achievement goal added
- `-!goal`: Achievement goal removed (failure)
- `+belief`: Belief added
- `-belief`: Belief removed

### Context Conditions
- Conjunction: `belief1 & belief2`
- Disjunction: `belief1 | belief2`
- Negation: `not belief`
- Always true: `true`

### Plan Body Actions
- Subgoal: `!goal_name`
- Action: `action_name(arg1, arg2)`
- Belief update: `+belief(X)` or `-belief(X)`
- Internal action: `.print("message")`
- Test goal: `?belief(X)`

---

## Appendix B: PDDL Syntax Reference

### Action Definition
```lisp
(:action action-name
  :parameters (?v1 - type1 ?v2 - type2)
  :precondition (and
    (predicate1 ?v1)
    (not (predicate2 ?v2)))
  :effect (and
    (predicate3 ?v1 ?v2)
    (not (predicate1 ?v1))))
```

### Non-Deterministic Effects
```lisp
:effect (oneof
  (and (outcome1) (not (state1)))
  (and (outcome2)))
```

### Conditional Effects
```lisp
:effect (when
  (condition)
  (and (effect1) (effect2)))
```

---

**END OF DESIGN DOCUMENT**
