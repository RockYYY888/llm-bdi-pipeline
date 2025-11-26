"""
Refactored Backward Search (Regression-based Planning)

This implements proper backward search following classical AI planning:
1. Start from goal state (conjunction of predicates)
2. For each predicate in goal, find actions that achieve it (in additive effects)
3. Apply regression formula: goal ∧ prec ∧ deleted_effects ∧ ¬additive_effects
4. Process conjunctions by creating separate branches for each predicate
5. Use BFS to explore states level by level
6. Support variable-level planning with proper variable generation

Key improvements over previous implementation:
- Proper conjunction destruction (separate branches per predicate)
- Correct variable numbering (inherit from parent state)
- Inequality constraints as predicates
- Parameter binding with partial bindings (e.g., pick-up(a, ?1))
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from collections import deque, defaultdict
from dataclasses import dataclass

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import PredicateAtom, WorldState, StateGraph, StateTransition
from stage3_code_generation.pddl_condition_parser import PDDLConditionParser, PDDLEffectParser
from utils.pddl_parser import PDDLDomain, PDDLAction


@dataclass(frozen=True)
class InequalityConstraint:
    """
    Represents an inequality constraint: ?x ≠ ?y

    Attributes:
        var1: First variable (e.g., "?1")
        var2: Second variable (e.g., "?2")
    """
    var1: str
    var2: str

    def __str__(self) -> str:
        return f"{self.var1} ≠ {self.var2}"

    def __hash__(self) -> int:
        # Normalize: smaller variable first
        v1, v2 = (self.var1, self.var2) if self.var1 <= self.var2 else (self.var2, self.var1)
        return hash((v1, v2))

    def __eq__(self, other) -> bool:
        if not isinstance(other, InequalityConstraint):
            return False
        # Order-independent equality
        return {self.var1, self.var2} == {other.var1, other.var2}


class ConstraintGraph:
    """
    Constraint graph for analyzing variable inequality relationships.

    Graph structure:
    - Nodes = variables (including ground objects)
    - Edges = inequality constraints (must be different)

    Ground objects like 'a', 'b' implicitly have edges between them
    since they are definitionally different.

    Used to determine minimum number of objects needed to satisfy all constraints
    by finding maximum clique (variables that are ALL mutually constrained).
    """

    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Set[Tuple[str, str]] = set()  # (var1, var2) where var1 < var2
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, var: str):
        """Add a variable node to the graph"""
        self.nodes.add(var)

    def add_edge(self, var1: str, var2: str):
        """Add inequality constraint: var1 != var2"""
        if var1 == var2:
            return  # No self-loops

        # Normalize edge
        if var1 > var2:
            var1, var2 = var2, var1

        self.nodes.add(var1)
        self.nodes.add(var2)
        self.edges.add((var1, var2))
        self.adjacency[var1].add(var2)
        self.adjacency[var2].add(var1)

    def find_maximum_clique_greedy(self) -> Set[str]:
        """
        Find a maximal clique using greedy algorithm.

        A clique = set of variables that are ALL mutually constrained (all != each other).
        These variables MUST all bind to different objects.

        Returns a maximal clique (not necessarily maximum, but good enough for pruning).
        """
        if not self.nodes:
            return set()

        # Sort nodes by degree (most constrained first) for better greedy choices
        nodes_by_degree = sorted(
            self.nodes,
            key=lambda n: len(self.adjacency.get(n, set())),
            reverse=True
        )

        best_clique = set()

        # Try starting from each node (limit iterations for efficiency)
        for start_node in nodes_by_degree[:10]:
            clique = {start_node}
            candidates = self.adjacency.get(start_node, set()).copy()

            while candidates:
                # Find node in candidates connected to all clique members
                best_candidate = None
                best_future_candidates = -1

                for candidate in candidates:
                    # Check if candidate is connected to ALL current clique members
                    neighbors = self.adjacency.get(candidate, set())
                    if clique.issubset(neighbors | {candidate}):
                        # Valid candidate - count future candidates
                        future = len(candidates & neighbors)
                        if future > best_future_candidates:
                            best_future_candidates = future
                            best_candidate = candidate

                if best_candidate is None:
                    break

                clique.add(best_candidate)
                # Update candidates to only include nodes connected to all clique members
                candidates = candidates & self.adjacency.get(best_candidate, set())

            if len(clique) > len(best_clique):
                best_clique = clique

        return best_clique


def _build_constraint_graph(
    variables: Set[str],
    inequality_constraints: Set[InequalityConstraint],
    ground_objects: Set[str] = None
) -> ConstraintGraph:
    """
    Build constraint graph from variables and constraints.

    Uses ONLY explicit constraints from PDDL (not (= ?x ?y)) - no implicit inference.

    Args:
        variables: All variables in the state (both ?v-style and ground like 'a')
        inequality_constraints: Explicit != constraints from PDDL
        ground_objects: Set of known ground objects (if provided, adds implicit constraints)

    Returns:
        ConstraintGraph with all constraints
    """
    graph = ConstraintGraph()

    # Add all variables as nodes
    for var in variables:
        graph.add_node(var)

    # Add explicit inequality constraints from PDDL
    for constraint in inequality_constraints:
        graph.add_edge(constraint.var1, constraint.var2)

    # Add implicit constraints between ground objects
    # Ground objects are definitionally different: a != b, a != c, b != c, etc.
    if ground_objects:
        ground_list = list(ground_objects)
        for i in range(len(ground_list)):
            for j in range(i + 1, len(ground_list)):
                graph.add_edge(ground_list[i], ground_list[j])
    else:
        # Infer ground objects from variables (those not starting with ?)
        ground_vars = [v for v in variables if not v.startswith('?')]
        for i in range(len(ground_vars)):
            for j in range(i + 1, len(ground_vars)):
                graph.add_edge(ground_vars[i], ground_vars[j])

    return graph


def _compute_minimum_objects_needed(
    variables: Set[str],
    inequality_constraints: Set[InequalityConstraint],
    ground_objects: Set[str] = None
) -> int:
    """
    Compute lower bound on minimum objects needed to satisfy all constraints.

    The exact answer is the chromatic number (NP-hard).
    We return the size of the maximum clique as a lower bound.

    Uses ONLY explicit PDDL constraints (not (= ?x ?y)) - no implicit inference.

    Args:
        variables: All variables in state
        inequality_constraints: Explicit != constraints from PDDL
        ground_objects: Known ground objects

    Returns:
        Lower bound on minimum objects needed
    """
    graph = _build_constraint_graph(variables, inequality_constraints, ground_objects)

    if not graph.nodes:
        return 0

    # The size of the maximum clique is a lower bound on chromatic number
    # (A clique of size k needs k colors/objects)
    max_clique = graph.find_maximum_clique_greedy()

    return len(max_clique)


def _should_prune_state_constraint_aware(
    variables: Set[str],
    inequality_constraints: Set[InequalityConstraint],
    max_objects: int,
    ground_objects: Set[str] = None
) -> bool:
    """
    Determine if a state should be pruned based on constraint analysis.

    A state should be pruned if we can PROVE it's unreachable:
    - If there's a clique larger than max_objects, we need more objects than available

    This is SOUND (never prunes reachable states) but not COMPLETE
    (may not prune all unreachable states). That's appropriate for search.

    Uses ONLY explicit PDDL constraints (not (= ?x ?y)) - no implicit inference.

    Args:
        variables: All variables in state
        inequality_constraints: Explicit != constraints from PDDL
        max_objects: Maximum available objects
        ground_objects: Known ground objects (optional)

    Returns:
        True if state should be pruned, False otherwise
    """
    min_objects_needed = _compute_minimum_objects_needed(
        variables, inequality_constraints, ground_objects
    )

    return min_objects_needed > max_objects


@dataclass
class BackwardState:
    """
    Represents a state in backward search

    Attributes:
        predicates: Set of predicates that must hold
        constraints: Set of inequality constraints
        depth: Distance from goal state
        max_var_number: Maximum variable number used in this state (for variable generation)
    """
    predicates: FrozenSet[PredicateAtom]
    constraints: FrozenSet[InequalityConstraint]
    depth: int
    max_var_number: int

    def __init__(self, predicates: Set[PredicateAtom],
                 constraints: Set[InequalityConstraint] = None,
                 depth: int = 0,
                 max_var_number: int = 0):
        """Initialize BackwardState"""
        object.__setattr__(self, 'predicates', frozenset(predicates))
        object.__setattr__(self, 'constraints', frozenset(constraints or set()))
        object.__setattr__(self, 'depth', depth)
        object.__setattr__(self, 'max_var_number', max_var_number)

    def is_goal_achieved(self) -> bool:
        """
        Check if goal is achieved (empty predicate set or equivalent to 'true')
        """
        return len(self.predicates) == 0

    def get_all_variables(self) -> Set[str]:
        """Get all variables used in this state"""
        variables = set()
        for pred in self.predicates:
            for arg in pred.args:
                if arg.startswith('?'):
                    variables.add(arg)
        for constraint in self.constraints:
            variables.add(constraint.var1)
            variables.add(constraint.var2)
        return variables

    def __str__(self) -> str:
        pred_str = ", ".join(str(p) for p in sorted(self.predicates, key=str))
        const_str = ", ".join(str(c) for c in sorted(self.constraints, key=str))
        return f"State(depth={self.depth}, preds=[{pred_str}], constraints=[{const_str}])"

    def __hash__(self) -> int:
        return hash((self.predicates, self.constraints))

    def __eq__(self, other) -> bool:
        if not isinstance(other, BackwardState):
            return False
        return self.predicates == other.predicates and self.constraints == other.constraints


@dataclass
class ParsedAction:
    """
    Parsed PDDL action with separated components

    Attributes:
        action: Original PDDL action
        parameters: List of parameter variables (e.g., ["?b1", "?b2"])
        preconditions: List of precondition predicates (positive and negative)
        additive_effects: List of predicates added by this action
        deletion_effects: List of predicates deleted by this action
        inequality_constraints: Set of inequality constraints from preconditions
    """
    action: PDDLAction
    parameters: List[str]
    preconditions: List[PredicateAtom]
    additive_effects: List[PredicateAtom]
    deletion_effects: List[PredicateAtom]
    inequality_constraints: Set[InequalityConstraint]


class BackwardSearchPlanner:
    """
    Backward search planner using regression

    Implements proper backward planning:
    1. Start from goal (conjunction of predicates)
    2. For each predicate, find achieving actions
    3. Apply regression: goal ∧ prec ∧ del_effects ∧ ¬add_effects
    4. Use BFS for level-by-level exploration
    5. Support variable-level planning with proper variable generation
    """

    def __init__(self, domain: PDDLDomain, domain_path: str = None,
                 precomputed_mutex_groups: Dict[str, Set[str]] = None,
                 precomputed_singleton_predicates: Set[str] = None):
        """
        Initialize backward search planner

        Args:
            domain: PDDL domain definition
            domain_path: Path to PDDL domain file (optional, needed for FD invariant extraction)
            precomputed_mutex_groups: Optional pre-computed static mutex groups (avoids recomputation)
            precomputed_singleton_predicates: Optional pre-computed singleton predicates
        """
        self.domain = domain
        self.domain_path = domain_path
        self.condition_parser = PDDLConditionParser()
        self.effect_parser = PDDLEffectParser()
        self.max_objects = None  # Will be set in search() method

        # Parse all actions to structured form
        self.parsed_actions: List[ParsedAction] = self._parse_all_actions()

        # Use precomputed mutex groups if provided, otherwise compute
        if precomputed_mutex_groups is not None:
            self.mutex_groups = precomputed_mutex_groups
            self.singleton_predicates = precomputed_singleton_predicates or set()
            # Skip printing - already printed when precomputed
        else:
            self.mutex_groups = self._compute_mutex_groups()
            print(f"[Mutex Analysis] Computed {len(self.mutex_groups)} mutex groups from domain")

    def search(self, goal_predicates: List[PredicateAtom],
               max_states: int = 200000,
               max_objects: Optional[int] = None) -> StateGraph:
        """
        Perform backward search from goal

        Args:
            goal_predicates: Goal predicates to achieve
            max_states: Maximum states to explore (防止无限搜索)
            max_objects: Maximum number of objects (caps variable generation)

        Returns:
            StateGraph containing all explored states and transitions
        """
        # Store max_objects for use in _complete_binding
        self.max_objects = max_objects

        print(f"\n[Backward Search] Starting from goal: {[str(p) for p in goal_predicates]}")
        print(f"[Backward Search] Max states: {max_states:,}")
        if max_objects is not None:
            print(f"[Backward Search] Max objects: {max_objects} (variable cap)")

        # Create initial goal state
        goal_state = self._create_initial_goal_state(goal_predicates)
        print(f"[Backward Search] Initial goal state: {goal_state}")

        # Check if goal is already achieved
        if goal_state.is_goal_achieved():
            print(f"[Backward Search] Goal already achieved (empty)")
            return self._create_empty_state_graph(goal_state)

        # BFS exploration
        queue = deque([goal_state])
        visited: Dict[Tuple, BackwardState] = {}
        visited[self._state_key(goal_state)] = goal_state

        transitions = []
        states_explored = 0

        while queue and states_explored < max_states:
            current_state = queue.popleft()
            states_explored += 1

            if states_explored % 10000 == 0:
                print(f"  Explored {states_explored:,} states, queue: {len(queue):,}")

            # Check if goal achieved (reached initial state with empty predicates)
            if current_state.is_goal_achieved():
                print(f"  Found achieved goal at depth {current_state.depth}")
                continue

            # CONJUNCTION DESTRUCTION (following instruction precisely):
            # "destroy one predicate at one time"
            # For each predicate in the conjunction, find all achieving actions
            # This creates separate branches for each predicate in the goal
            num_predicates = len(current_state.predicates)
            for pred_idx, target_predicate in enumerate(current_state.predicates):
                # Find all actions that can achieve this specific predicate
                achieving_actions = self._find_achieving_actions(target_predicate)

                # if states_explored <= 50:  # Detailed logging for first few states
                #     print(f"    [Depth {current_state.depth}] Predicate {pred_idx+1}/{num_predicates}: {target_predicate}")
                #     print(f"      Found {len(achieving_actions)} achieving action(s)")

                # For each achieving action, apply regression
                for action_idx, (parsed_action, binding) in enumerate(achieving_actions):
                    # if states_explored <= 50:
                    #     print(f"        Action {action_idx+1}: {parsed_action.action.name} with binding {binding}")

                    # Apply regression formula: goal ∧ prec ∧ del_effects ∧ ¬add_effects
                    predecessor_states = self._apply_regression(
                        current_state,
                        target_predicate,
                        parsed_action,
                        binding
                    )

                    for pred_state in predecessor_states:
                        state_key = self._state_key(pred_state)

                        if state_key in visited:
                            final_state = visited[state_key]
                        else:
                            visited[state_key] = pred_state
                            queue.append(pred_state)
                            final_state = pred_state

                            # if states_explored <= 50:
                            #     print(f"          → New state: {final_state}")

                        # Record transition
                        transitions.append((
                            final_state,
                            current_state,
                            parsed_action.action,
                            binding
                        ))

        print(f"[Backward Search] Exploration complete:")
        print(f"  States explored: {states_explored:,}")
        print(f"  Unique states: {len(visited):,}")
        print(f"  Transitions: {len(transitions):,}")

        # Convert to StateGraph
        return self._build_state_graph(goal_state, visited, transitions, states_explored >= max_states)

    def _create_initial_goal_state(self, goal_predicates: List[PredicateAtom]) -> BackwardState:
        """
        Create initial goal state from goal predicates

        Args:
            goal_predicates: List of goal predicates

        Returns:
            BackwardState representing the goal
        """
        # Extract ALL variables (if any) and determine max_var_number
        # In Grounded Search, goal_predicates should be grounded (e.g., on(a, b))
        # Variables only appear when we can't fully bind parameters
        # Standard format: ?v1, ?v2, ?v3, ... (for AgentSpeak compatibility: ?v1 → V1)
        max_var = 0  # Start at 0 so first generated variable is ?v1
        for pred in goal_predicates:
            for arg in pred.args:
                if arg.startswith('?v') and len(arg) > 2 and arg[2:].isdigit():
                    # Extract number from ?v1, ?v2, ?v3, ...
                    var_num = int(arg[2:])
                    max_var = max(max_var, var_num)

        return BackwardState(
            predicates=set(goal_predicates),
            constraints=set(),
            depth=0,
            max_var_number=max_var
        )

    def _parse_all_actions(self) -> List[ParsedAction]:
        """
        Parse all PDDL actions to structured form

        Returns:
            List of ParsedAction objects
        """
        parsed_actions = []

        for action in self.domain.actions:
            # Extract parameters
            parameters = self._extract_parameters(action)

            # Create identity bindings (variables stay as variables)
            identity_bindings = {p: p for p in parameters}

            # Parse preconditions
            preconditions = self.condition_parser.parse(action.preconditions, identity_bindings)

            # Parse effects
            effect_branches = self.effect_parser.parse(action.effects, identity_bindings)
            effects = effect_branches[0] if effect_branches else []

            # Separate additive and deletion effects
            additive_effects = [eff.predicate for eff in effects if eff.is_add]

            # CRITICAL FIX: Remove negation from deletion effects
            # In PDDL: (not (holding ?b1)) means "delete holding(?b1)"
            # In regression: We need "holding(?b1)" as a POSITIVE precondition
            # Because: if action deletes P, then P must be TRUE before the action
            deletion_effects = []
            for eff in effects:
                if not eff.is_add:
                    # Create non-negated version of the predicate
                    pred = PredicateAtom(eff.predicate.name, eff.predicate.args, negated=False)
                    deletion_effects.append(pred)

            # Extract inequality constraints
            inequality_constraints = self._extract_inequality_constraints(action.preconditions, parameters)

            parsed_actions.append(ParsedAction(
                action=action,
                parameters=parameters,
                preconditions=preconditions,
                additive_effects=additive_effects,
                deletion_effects=deletion_effects,
                inequality_constraints=inequality_constraints
            ))

        return parsed_actions

    def _extract_parameters(self, action: PDDLAction) -> List[str]:
        """
        Extract parameter variable names from action

        Args:
            action: PDDL action

        Returns:
            List of parameter variable names (e.g., ["?b1", "?b2"])
        """
        parameters = []
        for param in action.parameters:
            # Split by '-' and take variable name
            var_name = param.split('-')[0].strip() if '-' in param else param.strip()
            parameters.append(var_name)
        return parameters

    def _extract_inequality_constraints(self, preconditions: str, parameters: List[str]) -> Set[InequalityConstraint]:
        """
        Extract inequality constraints from PDDL preconditions

        Parses patterns like: (not (= ?b1 ?b2))

        Args:
            preconditions: PDDL precondition string
            parameters: List of parameter variables

        Returns:
            Set of InequalityConstraint objects
        """
        constraints = set()
        # Match pattern: (not (= ?var1 ?var2))
        pattern = r'\(not\s+\(=\s+(\?\w+)\s+(\?\w+)\)\)'

        for match in re.finditer(pattern, preconditions):
            var1, var2 = match.group(1), match.group(2)
            constraints.add(InequalityConstraint(var1, var2))

        return constraints

    def _find_achieving_actions(self, target_predicate: PredicateAtom) -> List[Tuple[ParsedAction, Dict[str, str]]]:
        """
        Find all actions that can achieve the target predicate

        CRITICAL: Handle both positive and negative goal predicates:
        - Positive predicate P: Find actions with P in additive effects
        - Negative predicate ~P: Find actions with P in deletion effects

        Args:
            target_predicate: Predicate to achieve (can be positive or negative)

        Returns:
            List of (ParsedAction, binding) tuples where binding maps action parameters to goal objects/variables
        """
        achieving_actions = []

        if target_predicate.negated:
            # NEGATIVE goal: ~P
            # Find actions that DELETE P (have P in deletion effects)
            # Create positive version of target for matching
            positive_target = PredicateAtom(
                target_predicate.name,
                target_predicate.args,
                negated=False
            )

            for parsed_action in self.parsed_actions:
                # Check each deletion effect
                for del_effect in parsed_action.deletion_effects:
                    # Try to unify del_effect with positive version of target
                    binding = self._unify_predicates(del_effect, positive_target)

                    if binding is not None:
                        achieving_actions.append((parsed_action, binding))

        else:
            # POSITIVE goal: P
            # Find actions that ADD P (have P in additive effects)
            for parsed_action in self.parsed_actions:
                # Check each additive effect
                for add_effect in parsed_action.additive_effects:
                    # Try to unify add_effect with target_predicate
                    binding = self._unify_predicates(add_effect, target_predicate)

                    if binding is not None:
                        achieving_actions.append((parsed_action, binding))

        return achieving_actions

    def _unify_predicates(self, pattern: PredicateAtom, target: PredicateAtom) -> Optional[Dict[str, str]]:
        """
        Unify two predicates to find variable bindings

        Examples:
            pattern=on(?b1, ?b2), target=on(a, b) → {?b1: a, ?b2: b}
            pattern=on(?b1, ?b2), target=on(a, ?1) → {?b1: a, ?b2: ?1}
            pattern=clear(?b), target=on(a, b) → None (names don't match)

        Args:
            pattern: Pattern predicate (from action effect)
            target: Target predicate (from goal)

        Returns:
            Dictionary mapping pattern variables to target arguments, or None if unification fails
        """
        # Check predicate names match
        if pattern.name != target.name:
            return None

        # Check arity matches
        if len(pattern.args) != len(target.args):
            return None

        # Build binding
        binding = {}
        for pattern_arg, target_arg in zip(pattern.args, target.args):
            if pattern_arg.startswith('?'):
                # Pattern variable: bind to target argument
                if pattern_arg in binding:
                    # Variable already bound: check consistency
                    if binding[pattern_arg] != target_arg:
                        return None  # Inconsistent binding
                else:
                    binding[pattern_arg] = target_arg
            else:
                # Pattern constant: must match target
                if pattern_arg != target_arg:
                    return None

        return binding

    def _apply_regression(self, current_state: BackwardState,
                          target_predicate: PredicateAtom,
                          parsed_action: ParsedAction,
                          binding: Dict[str, str]) -> List[BackwardState]:
        """
        Apply regression formula to compute predecessor states

        Regression formula: goal ∧ prec ∧ deleted_effects ∧ ¬additive_effects

        CRITICAL: ¬additive_effects means:
        - For each additive effect in the action:
          - If it matches a predicate in current goal, REMOVE that predicate from goal
          - Because the action will add it, so we don't need it in the predecessor state

        Steps (following instruction precisely):
        1. Start with current goal predicates
        2. Remove additive effects that are IN current goal (¬additive_effects)
        3. Add preconditions (with complete binding)
        4. Add deletion effects (with complete binding)
        5. Add inequality constraints

        Args:
            current_state: Current state in backward search
            target_predicate: The predicate being achieved by this action
            parsed_action: The action achieving the predicate
            binding: Partial binding from unification

        Returns:
            List of predecessor BackwardStates
        """
        # Complete the binding by generating new variables for unbound parameters
        # CRITICAL: Always refer to parent's max_var_number and increment by 1
        complete_binding, next_var_number = self._complete_binding(
            parsed_action.parameters,
            binding,
            current_state.max_var_number
        )

        # Step 1: Start with current goal predicates
        original_goal = set(current_state.predicates)
        new_predicates = set(current_state.predicates)

        # Step 2: Check for conflicts with ADDITIVE effects (before processing)
        # CRITICAL: Check against ORIGINAL GOAL, not new_predicates
        for add_effect in parsed_action.additive_effects:
            instantiated_effect = self._instantiate_predicate(add_effect, complete_binding)

            # Check for conflict: action adds P but original goal requires ~P
            negated_version = PredicateAtom(
                instantiated_effect.name,
                instantiated_effect.args,
                negated=not instantiated_effect.negated
            )
            if negated_version in original_goal:
                # CONFLICT: Cannot achieve goal with this action
                # Example: action adds on(a,b) but goal requires ~on(a,b)
                return []  # Skip this action

        # Step 3: Check for conflicts with DELETION effects (before processing)
        # CRITICAL: Check against ORIGINAL GOAL, not new_predicates
        for del_effect in parsed_action.deletion_effects:
            instantiated_del = self._instantiate_predicate(del_effect, complete_binding)

            # Check for conflict: action deletes P but original goal requires P (positive)
            if instantiated_del in original_goal:
                # CONFLICT: Cannot achieve goal with this action
                # Example: action deletes on(a,b) but goal requires on(a,b)
                return []  # Skip this action

        # Step 4: Apply ¬additive_effects
        # Remove additive effects from goal (will be achieved by action)
        for add_effect in parsed_action.additive_effects:
            instantiated_effect = self._instantiate_predicate(add_effect, complete_binding)
            # Remove if in goal (positive match)
            if instantiated_effect in new_predicates:
                new_predicates.discard(instantiated_effect)

        # Step 5: Process deletion effects
        # - If deletion satisfies negative goal (~P), remove ~P from goal
        # - Otherwise, add P as precondition (must exist before deletion)
        for del_effect in parsed_action.deletion_effects:
            instantiated_del = self._instantiate_predicate(del_effect, complete_binding)

            # Check if deletion satisfies a negative goal: ~P in goal and action deletes P
            negated_del = PredicateAtom(
                instantiated_del.name,
                instantiated_del.args,
                negated=not instantiated_del.negated
            )
            if negated_del in new_predicates:
                # SATISFIES NEGATIVE GOAL: Remove ~P from goal (action will delete P)
                # Example: goal has ~on(a,b), action deletes on(a,b) → satisfies goal
                new_predicates.discard(negated_del)
                # NOTE: We do NOT add P as precondition in this case!
            else:
                # Normal case: Add P as positive precondition (must exist before deletion)
                new_predicates.add(instantiated_del)

        # Step 6: Add preconditions (all of them)
        # NOTE: We do NOT check for conflicts with original goal here!
        # Why? Because it's valid for a precondition to require P even if goal requires ~P
        # Example: To achieve ~on(a,b), we use pick-up(a,b) which requires on(a,b) as precondition
        #          The action will DELETE on(a,b), thus achieving ~on(a,b)
        for precond in parsed_action.preconditions:
            instantiated_precond = self._instantiate_predicate(precond, complete_binding)
            new_predicates.add(instantiated_precond)

        # Step 5: Add inequality constraints
        # CRITICAL: Must instantiate ALL inequality constraints from the action
        # Even if the original binding was empty, complete_binding now has variables
        new_constraints = set(current_state.constraints)
        for constraint in parsed_action.inequality_constraints:
            # Use complete_binding (which includes generated variables)
            var1_inst = complete_binding.get(constraint.var1, constraint.var1)
            var2_inst = complete_binding.get(constraint.var2, constraint.var2)

            instantiated_constraint = InequalityConstraint(var1_inst, var2_inst)
            new_constraints.add(instantiated_constraint)

        # CRITICAL: Validate constraints before creating state
        # Check if inequality constraints are violated by the predicates
        if not self._validate_constraints(new_predicates, new_constraints):
            # Constraint violation detected - skip this state
            return []

        # CRITICAL: Check for contradictions (P and ~P in same state)
        if not self._check_no_contradictions(new_predicates):
            # Contradiction detected - skip this state
            return []

        # CRITICAL: Check for mutex violations using h^2 analysis
        if hasattr(self, 'mutex_groups') and self.mutex_groups:
            if not self._check_no_mutex_violations(new_predicates):
                # Mutex violation detected - skip this state
                return []

        # CRITICAL: Constraint-aware pruning based on variable feasibility
        # OLD LOGIC (TOO AGGRESSIVE): if len(unique_vars) > max_objects → PRUNE
        # This was wrong because variables WITHOUT constraints can share objects
        # NEW LOGIC: Use constraint graph analysis to find maximum clique
        # Only prune if the clique size > max_objects (variables that MUST be different)
        if self.max_objects is not None:
            # Extract all unique variables from predicates
            unique_vars = set()
            for pred in new_predicates:
                for arg in pred.args:
                    unique_vars.add(arg)

            # Quick check: if unique_vars <= max_objects, definitely feasible
            if len(unique_vars) > self.max_objects:
                # Need constraint analysis - check if state is actually infeasible
                # Extract ground objects (those not starting with ?)
                ground_objects = {v for v in unique_vars if not v.startswith('?')}

                # Use constraint-aware pruning with explicit PDDL constraints only
                if _should_prune_state_constraint_aware(
                    variables=unique_vars,
                    inequality_constraints=new_constraints,
                    max_objects=self.max_objects,
                    ground_objects=ground_objects if ground_objects else None
                ):
                    # Proven unreachable - maximum clique exceeds available objects
                    return []

        # Create predecessor state
        predecessor = BackwardState(
            predicates=new_predicates,
            constraints=new_constraints,
            depth=current_state.depth + 1,
            max_var_number=next_var_number
        )

        return [predecessor]

    def _check_no_contradictions(self, predicates: Set[PredicateAtom]) -> bool:
        """
        Check that predicates don't contain contradictions (P and ~P)

        Args:
            predicates: Set of predicates to check

        Returns:
            True if no contradictions, False if contradiction detected
        """
        # Build a set of (name, args) for both positive and negative predicates
        positive_preds = set()
        negative_preds = set()

        for pred in predicates:
            key = (pred.name, tuple(pred.args))
            if pred.negated:
                negative_preds.add(key)
            else:
                positive_preds.add(key)

        # Check for overlap: if any predicate exists in both sets, it's a contradiction
        contradictions = positive_preds & negative_preds
        if contradictions:
            # Found contradiction: P and ~P in same state
            return False

        return True

    def _compute_mutex_groups(self) -> Dict[str, Set[str]]:
        """
        Compute STATIC mutex relationships using Fast Downward invariant synthesis

        This is an instance method that calls the static version and stores singleton predicates.

        Returns:
            Dictionary mapping predicate names to sets of STATIC mutex predicates
        """
        mutex_groups, singleton_preds = BackwardSearchPlanner.compute_static_mutex(
            self.domain_path,
            objects=['b1', 'b2', 'b3', 'b4', 'b5']  # Default objects for grounding
        )
        self.singleton_predicates = singleton_preds
        return mutex_groups

    @staticmethod
    def compute_static_mutex(domain_path: str, objects: List[str]) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        Compute STATIC mutex relationships using Fast Downward invariant synthesis

        This is a STATIC method that can be called before creating planner instances.
        Use this for precomputing mutex groups to share across multiple planners.

        This extracts true domain invariants (not transient effect-based mutex).
        Key distinction:
        - Static mutex: holding(x) ↔ handempty (impossible to co-exist)
        - Transient mutex: holding(x) ↔ on(x,y) (can be satisfied sequentially)

        In backward planning, we only check static mutex because regression
        predicates represent goals to achieve at different time points, not
        simultaneous world state facts.

        Args:
            domain_path: Path to PDDL domain file
            objects: List of objects for grounding (e.g., ['b1', 'b2', 'b3'])

        Returns:
            Tuple of (static_mutex_map, singleton_predicates)
            - static_mutex_map: Dictionary mapping predicate names to sets of mutex predicates
            - singleton_predicates: Set of predicate names that can only have one instance

        Raises:
            SystemExit: If Fast Downward invariant extraction fails
        """
        try:
            from stage3_code_generation.fd_invariant_extractor import FDInvariantExtractor
        except ImportError as e:
            print("\n" + "="*80)
            print("[FATAL ERROR] Fast Downward Invariant Extractor not available")
            print("="*80)
            print(f"Import error: {e}")
            print("\nFast Downward is required for static mutex analysis.")
            print("Please ensure:")
            print("  1. Fast Downward is installed in the project directory")
            print("  2. fd_invariant_extractor.py exists in stage3_code_generation/")
            print("="*80)
            sys.exit(1)

        try:
            # Use Fast Downward to extract static invariants
            print("[Mutex Analysis] Using Fast Downward invariant synthesis...")
            extractor = FDInvariantExtractor(domain_path, objects)
            static_mutex_map, singleton_preds = extractor.extract_invariants()

            print(f"[Mutex Analysis] Extracted {len(static_mutex_map)} static mutex groups")
            print(f"[Mutex Analysis] Singleton predicates: {singleton_preds}")

            return static_mutex_map, singleton_preds

        except Exception as e:
            print("\n" + "="*80)
            print("[FATAL ERROR] Fast Downward invariant extraction failed")
            print("="*80)
            print(f"Error: {e}")
            print(f"\nDomain path: {domain_path}")
            print(f"Objects: {objects}")
            print("\nPlease check:")
            print("  1. Domain file exists and is valid PDDL")
            print("  2. Fast Downward is properly installed")
            print("  3. Fast Downward can process the domain")
            print("="*80)
            import traceback
            traceback.print_exc()
            sys.exit(1)


    def _check_no_singleton_violations(self, predicates: Set[PredicateAtom]) -> bool:
        """
        Check if predicates violate singleton constraints (e.g., multiple holding predicates)

        This is used in backward planning where mutex pairs can coexist (representing dependencies),
        but singleton violations are always invalid (e.g., can't hold two blocks at once).

        Args:
            predicates: Set of predicates to check

        Returns:
            True if no singleton violations, False if violation detected
        """
        # Collect positive predicates by name
        preds_by_name = {}
        for pred in predicates:
            if not pred.negated:
                if pred.name not in preds_by_name:
                    preds_by_name[pred.name] = []
                preds_by_name[pred.name].append(pred)

        # Check singleton predicates (e.g., handempty, holding)
        for pred_name in self.singleton_predicates:
            if pred_name in preds_by_name:
                if len(preds_by_name[pred_name]) > 1:
                    # Multiple instances of singleton predicate!
                    return False

        return True

    def _check_no_mutex_violations(self, predicates: Set[PredicateAtom]) -> bool:
        """
        Check if predicates violate any mutex constraints derived from domain

        Args:
            predicates: Set of predicates to check

        Returns:
            True if no mutex violations, False if violation detected
        """
        # Collect positive predicates by name
        preds_by_name = {}
        for pred in predicates:
            if not pred.negated:
                if pred.name not in preds_by_name:
                    preds_by_name[pred.name] = []
                preds_by_name[pred.name].append(pred)

        # Check Pattern 1: Mutex pairs (e.g., handempty vs holding)
        for pred_name, mutex_names in self.mutex_groups.items():
            if pred_name in preds_by_name:
                # This predicate appears in the state
                # Check if any of its mutex predicates also appear
                for mutex_name in mutex_names:
                    if mutex_name in preds_by_name:
                        # Mutex violation!
                        return False

        # Check Pattern 2: Singleton predicates (e.g., multiple holding)
        for pred_name in self.singleton_predicates:
            if pred_name in preds_by_name:
                if len(preds_by_name[pred_name]) > 1:
                    # Multiple instances of singleton predicate!
                    return False

        return True

    def _validate_constraints(self, predicates: Set[PredicateAtom],
                            constraints: Set[InequalityConstraint]) -> bool:
        """
        Validate that predicates don't violate inequality constraints

        This is the CORRECT way to detect invalid states:
        - DON'T hardcode "on(X,X) is invalid"
        - DO check if inequality constraints from PDDL are violated

        Example:
        - Constraint: ?v0 ≠ ?v1
        - Predicate: on(?v0, ?v0)  ← INVALID (violates constraint)
        - Predicate: on(?v0, ?v1)  ← VALID (satisfies constraint)

        Args:
            predicates: Set of predicates in the state
            constraints: Set of inequality constraints

        Returns:
            True if all constraints are satisfied, False if any are violated
        """
        # Check each inequality constraint
        for constraint in constraints:
            var1 = constraint.var1
            var2 = constraint.var2

            # Constraint says var1 ≠ var2
            # If var1 == var2, this is a contradiction
            if var1 == var2:
                # Constraint violation: ?v0 ≠ ?v0 is always false
                return False

        # All constraints are satisfiable
        return True

    def _complete_binding(self, parameters: List[str],
                          partial_binding: Dict[str, str],
                          parent_max_var: int) -> Tuple[Dict[str, str], int]:
        """
        Complete partial binding by generating new variables for unbound parameters

        Variable numbering: ?v1, ?v2, ?v3, ... (for AgentSpeak compatibility: ?v1 → V1)
        Starts from parent_max_var + 1

        CRITICAL: Ensure generated variables don't clash with existing variables in binding

        If max_objects is set, caps variable generation at that number

        Args:
            parameters: List of action parameter variables
            partial_binding: Partial binding from unification
            parent_max_var: Maximum variable number from parent state

        Returns:
            (complete_binding, next_max_var_number)
        """
        complete_binding = dict(partial_binding)

        # Collect all variables already used in the binding
        used_vars = set(partial_binding.values())

        next_var_num = parent_max_var + 1

        for param in parameters:
            if param not in complete_binding:
                # Generate new variable that hasn't been used yet
                # Format: ?v1, ?v2, ?v3, ... (AgentSpeak compatible)
                new_var = f"?v{next_var_num}"

                # CRITICAL: Skip variables that are already used in the binding
                while new_var in used_vars:
                    next_var_num += 1
                    new_var = f"?v{next_var_num}"

                # NOTE: Variable generation IS capped at max_objects via pruning check (line 644)
                # States with more variables than available objects are physically unreachable
                # and are pruned as infeasible during regression
                # max_objects is also used for final_max_var calculation below

                complete_binding[param] = new_var
                used_vars.add(new_var)
                next_var_num += 1

        # Return the actual max variable number used
        # NOTE: We do NOT cap this at max_objects
        # In backward planning, variables represent abstract entities, not concrete objects
        final_max_var = next_var_num - 1

        return complete_binding, final_max_var

    def _instantiate_predicate(self, predicate: PredicateAtom,
                                binding: Dict[str, str]) -> PredicateAtom:
        """
        Instantiate predicate by applying variable binding

        Args:
            predicate: Predicate with variables
            binding: Variable binding

        Returns:
            Instantiated predicate
        """
        new_args = [binding.get(arg, arg) for arg in predicate.args]
        return PredicateAtom(predicate.name, new_args, predicate.negated)

    def _build_variable_mapping(self, predicates: Set[PredicateAtom],
                                 constraints: Set[InequalityConstraint]) -> Dict[str, str]:
        """
        Build consistent variable mapping for both predicates and constraints

        Algorithm:
        1. Sort predicates by (name, args, negated) for canonical order
        2. Scan predicate arguments in sorted order
        3. First occurrence of ?vN gets mapped to ?v1, second to ?v2, etc.
        4. Also collect variables from constraints (for completeness)

        Args:
            predicates: Set of PredicateAtom
            constraints: Set of InequalityConstraint

        Returns:
            Dictionary mapping old variable names to normalized names
        """
        var_map = {}
        next_var = 1

        # Sort predicates for canonical order
        sorted_preds = sorted(predicates, key=lambda p: (p.name, p.args, p.negated))

        # Scan predicate arguments
        for pred in sorted_preds:
            for arg in pred.args:
                if arg.startswith('?v') and arg not in var_map:
                    var_map[arg] = f'?v{next_var}'
                    next_var += 1

        # Also scan constraints (in case there are variables only in constraints)
        for constraint in sorted(constraints, key=lambda c: (c.var1, c.var2)):
            for var in [constraint.var1, constraint.var2]:
                if var.startswith('?v') and var not in var_map:
                    var_map[var] = f'?v{next_var}'
                    next_var += 1

        return var_map

    def _normalize_predicates_with_mapping(self, predicates: Set[PredicateAtom],
                                          var_map: Dict[str, str]) -> Tuple[Tuple, ...]:
        """
        Normalize predicates using provided variable mapping

        Args:
            predicates: Set of PredicateAtom
            var_map: Variable name mapping

        Returns:
            Tuple of normalized predicate representations
        """
        # Sort predicates for canonical order
        sorted_preds = sorted(predicates, key=lambda p: (p.name, p.args, p.negated))

        # Normalize predicates
        normalized = []
        for pred in sorted_preds:
            new_args = tuple(var_map.get(arg, arg) for arg in pred.args)
            normalized.append((pred.name, new_args, pred.negated))

        return tuple(normalized)

    def _normalize_constraints_with_mapping(self, constraints: Set[InequalityConstraint],
                                           var_map: Dict[str, str]) -> Tuple[Tuple, ...]:
        """
        Normalize constraints using provided variable mapping

        Args:
            constraints: Set of InequalityConstraint
            var_map: Variable name mapping

        Returns:
            Tuple of normalized constraint representations
        """
        if not constraints:
            return ()

        # Normalize constraints
        normalized = []
        for constraint in constraints:
            var1_norm = var_map.get(constraint.var1, constraint.var1)
            var2_norm = var_map.get(constraint.var2, constraint.var2)
            # Sort for canonical form: always (smaller, larger)
            normalized.append(tuple(sorted([var1_norm, var2_norm])))

        return tuple(sorted(normalized))

    def _state_key(self, state: BackwardState) -> Tuple:
        """
        Generate hashable key for state with VARIABLE NORMALIZATION

        CRITICAL: Normalize variable names to detect equivalent states
        Example:
            State 1: on(a, ?v2) ∧ clear(?v5) → Normalized: on(a, ?v1) ∧ clear(?v2)
            State 2: on(a, ?v7) ∧ clear(?v3) → Normalized: on(a, ?v1) ∧ clear(?v2)
            → SAME KEY (deduplicated!)

        Without normalization:
            State 1 key: ((on(a, ?v2), clear(?v5)), ())
            State 2 key: ((on(a, ?v7), clear(?v3)), ())
            → DIFFERENT KEYS (state explosion!)

        Args:
            state: BackwardState

        Returns:
            Tuple representing normalized state (for visited set)
        """
        # Build consistent variable mapping for both predicates and constraints
        var_map = self._build_variable_mapping(state.predicates, state.constraints)

        # Normalize both using the same mapping
        normalized_preds = self._normalize_predicates_with_mapping(state.predicates, var_map)
        normalized_constraints = self._normalize_constraints_with_mapping(state.constraints, var_map)

        return (normalized_preds, normalized_constraints)

    def _create_empty_state_graph(self, goal_state: BackwardState) -> StateGraph:
        """Create empty state graph for already-achieved goals"""
        world_state = WorldState(
            set(goal_state.predicates),
            depth=0,
            constraints=set(goal_state.constraints)  # CRITICAL: Pass constraints!
        )
        state_graph = StateGraph(world_state)
        state_graph.truncated = False
        return state_graph

    def _build_state_graph(self, goal_state: BackwardState,
                           visited: Dict[Tuple, BackwardState],
                           transitions: List[Tuple],
                           truncated: bool) -> StateGraph:
        """
        Build StateGraph from explored states and transitions

        Args:
            goal_state: Initial goal state
            visited: Dictionary of visited states
            transitions: List of (from_state, to_state, action, binding) tuples
            truncated: Whether search was truncated

        Returns:
            StateGraph object
        """
        # Convert BackwardStates to WorldStates
        # CRITICAL: Must pass constraints from BackwardState to WorldState!
        backward_to_world = {}
        for backward_state in visited.values():
            world_state = WorldState(
                set(backward_state.predicates),
                depth=backward_state.depth,
                constraints=set(backward_state.constraints)  # ← CRITICAL FIX!
            )
            backward_to_world[backward_state] = world_state

        # Create StateGraph
        goal_world_state = backward_to_world[goal_state]
        state_graph = StateGraph(goal_world_state)
        state_graph.truncated = truncated

        # Add transitions
        for from_state, to_state, action, binding in transitions:
            from_world = backward_to_world[from_state]
            to_world = backward_to_world[to_state]

            # Extract action arguments from binding
            param_vars = self._extract_parameters(action)
            action_args = tuple(binding.get(var, var) for var in param_vars)

            # Parse belief updates
            try:
                bindings_for_effects = {var: binding.get(var, var) for var in param_vars}
                effect_branches = self.effect_parser.parse(action.effects, bindings_for_effects)
                belief_updates = []
                if effect_branches:
                    for effect_atom in effect_branches[0]:
                        belief_updates.append(effect_atom.to_agentspeak())
            except:
                belief_updates = []

            # Parse preconditions
            try:
                bindings_for_preconds = {var: binding.get(var, var) for var in param_vars}
                preconditions = self.condition_parser.parse(action.preconditions, bindings_for_preconds)
            except:
                preconditions = []

            # Create transition
            transition = StateTransition(
                from_state=from_world,
                to_state=to_world,
                action=action,
                action_args=action_args,
                belief_updates=tuple(belief_updates),
                preconditions=tuple(preconditions)
            )
            state_graph.add_transition(transition)

        return state_graph


# Test functions
def test_backward_search_detailed():
    """Test backward search with detailed trace"""
    print("="*80)
    print("Testing Backward Search - Detailed Trace")
    print("="*80)

    from utils.pddl_parser import PDDLParser

    # Load blocksworld domain
    domain_file = Path(__file__).parent.parent / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))

    # Create planner
    planner = BackwardSearchPlanner(domain)

    # Test 1: Simple goal - on(a, b)
    print("\n" + "="*60)
    print("Test 1: Goal = on(a, b)")
    print("="*60)

    goal1 = [PredicateAtom("on", ["a", "b"])]
    graph1 = planner.search(goal1, max_states=100, max_depth=3)

    print(f"\nResult: {graph1.get_statistics()}")

    # Print some sample states
    print("\nSample states (first 5):")
    for i, state in enumerate(list(graph1.states)[:5]):
        print(f"  {i+1}. {state}")

    # Test 2: Conjunction goal - holding(a) ∧ clear(b)
    print("\n" + "="*60)
    print("Test 2: Goal = holding(a) ∧ clear(b)")
    print("="*60)

    goal2 = [
        PredicateAtom("holding", ["a"]),
        PredicateAtom("clear", ["b"])
    ]
    graph2 = planner.search(goal2, max_states=100, max_depth=3)

    print(f"\nResult: {graph2.get_statistics()}")

    # Test 3: Goal with variables
    print("\n" + "="*60)
    print("Test 3: Goal = on(?v1, ?v2)")
    print("="*60)

    goal3 = [PredicateAtom("on", ["?v1", "?v2"])]
    graph3 = planner.search(goal3, max_states=100, max_depth=3)

    print(f"\nResult: {graph3.get_statistics()}")

    print("\n" + "="*80)


def test_backward_search():
    """Test backward search planner"""
    print("="*80)
    print("Testing Backward Search Planner")
    print("="*80)

    from utils.pddl_parser import PDDLParser

    # Load blocksworld domain
    domain_file = Path(__file__).parent.parent / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))

    # Create planner
    planner = BackwardSearchPlanner(domain)

    # Test goal: on(a, b)
    goal_predicates = [PredicateAtom("on", ["a", "b"])]

    print(f"\nGoal: {[str(p) for p in goal_predicates]}")

    # Search
    state_graph = planner.search(goal_predicates, max_states=10000, max_depth=5)

    print(f"\nState Graph: {state_graph}")
    print(f"Statistics: {state_graph.get_statistics()}")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_backward_search()
