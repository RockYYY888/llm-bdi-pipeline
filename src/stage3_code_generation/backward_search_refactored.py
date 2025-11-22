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
from collections import deque
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

    def __init__(self, domain: PDDLDomain):
        """
        Initialize backward search planner

        Args:
            domain: PDDL domain definition
        """
        self.domain = domain
        self.condition_parser = PDDLConditionParser()
        self.effect_parser = PDDLEffectParser()

        # Parse all actions to structured form
        self.parsed_actions: List[ParsedAction] = self._parse_all_actions()

    def search(self, goal_predicates: List[PredicateAtom],
               max_states: int = 200000,
               max_depth: int = 5) -> StateGraph:
        """
        Perform backward search from goal

        Args:
            goal_predicates: Goal predicates to achieve
            max_states: Maximum states to explore
            max_depth: Maximum search depth

        Returns:
            StateGraph containing all explored states and transitions
        """
        print(f"\n[Backward Search] Starting from goal: {[str(p) for p in goal_predicates]}")
        print(f"[Backward Search] Max states: {max_states:,}, Max depth: {max_depth}")

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

            # Skip if max depth reached
            if current_state.depth >= max_depth:
                continue

            # Check if goal achieved
            if current_state.is_goal_achieved():
                print(f"  Found achieved goal at depth {current_state.depth}")
                continue

            # Process each predicate in the conjunction separately (conjunction destruction)
            for target_predicate in current_state.predicates:
                # Find all actions that can achieve this predicate
                achieving_actions = self._find_achieving_actions(target_predicate)

                for parsed_action, binding in achieving_actions:
                    # Apply regression to compute predecessor state
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
        # Extract variables and determine max_var_number
        max_var = 0
        for pred in goal_predicates:
            for arg in pred.args:
                if arg.startswith('?') and arg[1:].isdigit():
                    var_num = int(arg[1:])
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
            deletion_effects = [eff.predicate for eff in effects if not eff.is_add]

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

        An action can achieve a predicate if it appears in the action's additive effects.

        Args:
            target_predicate: Predicate to achieve

        Returns:
            List of (ParsedAction, binding) tuples where binding maps action parameters to goal objects/variables
        """
        achieving_actions = []

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

        Steps:
        1. Start with current goal predicates
        2. Add preconditions (with partial binding)
        3. Add deletion effects (with partial binding)
        4. Remove additive effects that match current goal
        5. Generate new variables for unbound parameters
        6. Add inequality constraints

        Args:
            current_state: Current state in backward search
            target_predicate: The predicate being achieved by this action
            parsed_action: The action achieving the predicate
            binding: Partial binding from unification

        Returns:
            List of predecessor BackwardStates
        """
        # Complete the binding by generating new variables for unbound parameters
        complete_binding, next_var_number = self._complete_binding(
            parsed_action.parameters,
            binding,
            current_state.max_var_number
        )

        # Start with current predicates
        new_predicates = set(current_state.predicates)

        # Remove target predicate (it's being achieved by this action)
        new_predicates.discard(target_predicate)

        # Remove other additive effects if they're in the goal
        for add_effect in parsed_action.additive_effects:
            instantiated_effect = self._instantiate_predicate(add_effect, complete_binding)
            new_predicates.discard(instantiated_effect)

        # Add preconditions
        for precond in parsed_action.preconditions:
            instantiated_precond = self._instantiate_predicate(precond, complete_binding)
            new_predicates.add(instantiated_precond)

        # Add deletion effects (must exist before action deletes them)
        for del_effect in parsed_action.deletion_effects:
            instantiated_del = self._instantiate_predicate(del_effect, complete_binding)
            new_predicates.add(instantiated_del)

        # Add inequality constraints
        new_constraints = set(current_state.constraints)
        for constraint in parsed_action.inequality_constraints:
            instantiated_constraint = InequalityConstraint(
                complete_binding.get(constraint.var1, constraint.var1),
                complete_binding.get(constraint.var2, constraint.var2)
            )
            new_constraints.add(instantiated_constraint)

        # Create predecessor state
        predecessor = BackwardState(
            predicates=new_predicates,
            constraints=new_constraints,
            depth=current_state.depth + 1,
            max_var_number=next_var_number
        )

        return [predecessor]

    def _complete_binding(self, parameters: List[str],
                          partial_binding: Dict[str, str],
                          parent_max_var: int) -> Tuple[Dict[str, str], int]:
        """
        Complete partial binding by generating new variables for unbound parameters

        Variable numbering starts from parent_max_var + 1

        Args:
            parameters: List of action parameter variables
            partial_binding: Partial binding from unification
            parent_max_var: Maximum variable number from parent state

        Returns:
            (complete_binding, next_max_var_number)
        """
        complete_binding = dict(partial_binding)
        next_var_num = parent_max_var + 1

        for param in parameters:
            if param not in complete_binding:
                # Generate new variable
                new_var = f"?{next_var_num}"
                complete_binding[param] = new_var
                next_var_num += 1

        return complete_binding, next_var_num - 1

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

    def _state_key(self, state: BackwardState) -> Tuple:
        """
        Generate hashable key for state

        Args:
            state: BackwardState

        Returns:
            Tuple representing state (for visited set)
        """
        pred_tuple = tuple(sorted(state.predicates, key=str))
        constraint_tuple = tuple(sorted(state.constraints, key=str))
        return (pred_tuple, constraint_tuple)

    def _create_empty_state_graph(self, goal_state: BackwardState) -> StateGraph:
        """Create empty state graph for already-achieved goals"""
        world_state = WorldState(set(goal_state.predicates), depth=0)
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
        backward_to_world = {}
        for backward_state in visited.values():
            world_state = WorldState(
                set(backward_state.predicates),
                depth=backward_state.depth
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
    print("Test 3: Goal = on(?1, ?2)")
    print("="*60)

    goal3 = [PredicateAtom("on", ["?1", "?2"])]
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
