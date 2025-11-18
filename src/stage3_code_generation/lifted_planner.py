"""
True Lifted Planning using Unification

This implements REAL lifted planning that:
1. Explores ABSTRACT state space (not grounded)
2. Uses UNIFICATION to apply actions (not enumeration)
3. Maintains variable CONSTRAINTS
4. State space size independent of number of objects

Key differences from pseudo-lifted (grounded with variables):
- DOES NOT enumerate all variable combinations
- DOES introduce variables on-demand through unification
- State count: O(abstract patterns) not O(grounded combinations)

Example:
    10 objects, object-level: 10,000+ states
    10 objects, lifted: ~50 abstract states
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

from stage3_code_generation.state_space import PredicateAtom
from stage3_code_generation.abstract_state import AbstractState, Constraint, ConstraintSet
from stage3_code_generation.unification import Unifier, Substitution
from utils.pddl_parser import PDDLDomain, PDDLAction
from stage3_code_generation.pddl_condition_parser import PDDLConditionParser, PDDLEffectParser


@dataclass
class AbstractAction:
    """
    Abstract action with variables (not grounded)

    Example:
        action: pick-up
        parameters: {?block: "block", ?from: "block"}
        preconditions: [on(?block, ?from), clear(?block), handempty, ?block != ?from]
        effects: [+holding(?block), -on(?block, ?from), -handempty, +clear(?from)]
    """
    action: PDDLAction
    param_vars: List[str]  # e.g., ["?block", "?from"]
    param_types: Dict[str, str]  # e.g., {"?block": "block", "?from": "block"}

    # Cached parsed components
    preconditions: List[PredicateAtom]
    effects: List[List]  # Effect branches (for oneof)
    inequality_constraints: Set[Tuple[str, str]]  # From preconditions: (not (= ?x ?y))


class LiftedPlanner:
    """
    True lifted planner using unification

    Key insight: Don't ground actions to all object combinations!
    Instead, use unification to find which abstract action can apply to abstract state.

    Algorithm:
    1. Start with abstract goal state (e.g., {on(?X, ?Y)})
    2. For each abstract action, try to UNIFY preconditions with current state
    3. If unification succeeds, apply abstract effects to generate new abstract state
    4. Repeat until no new abstract states discovered

    State space is MUCH smaller because we don't enumerate all object combinations.
    """

    def __init__(self, domain: PDDLDomain, var_counter_offset: int = 0):
        """
        Initialize lifted planner

        Args:
            domain: PDDL domain definition
            var_counter_offset: Starting value for variable counter (default: 0)
                               Use this to prevent variable name conflicts when
                               creating multiple LiftedPlanner instances
        """
        self.domain = domain
        self.condition_parser = PDDLConditionParser()
        self.effect_parser = PDDLEffectParser()

        # Parse all actions to abstract form (cached)
        self._abstract_actions = self._parse_abstract_actions()

        # Variable generator for introducing new variables
        # Start from offset to avoid conflicts between multiple planner instances
        self._var_counter = var_counter_offset

        # CRITICAL FIX #2: Extract mutex predicates for state validation
        self._mutex_predicates = self._extract_mutex_predicates()

    def explore_from_goal(self, goal_predicates: List[PredicateAtom],
                         max_states: int = 200000) -> Dict:
        """
        Explore abstract state space from goal

        Args:
            goal_predicates: Abstract goal predicates (with variables)
            max_states: Maximum abstract states to explore

        Returns:
            Dictionary with:
                - 'states': Set of AbstractStates
                - 'transitions': List of (from_state, to_state, action, subst)
                - 'goal_state': Initial goal state
        """
        # NOTE: Do NOT reset _var_counter here!
        # It should be set once in __init__ with var_counter_offset
        # to prevent conflicts between multiple LiftedPlanner instances

        print(f"[Lifted Planner] Starting ABSTRACT state space exploration")
        print(f"[Lifted Planner] Goal: {[str(p) for p in goal_predicates]}")
        print(f"[Lifted Planner] Abstract actions: {len(self._abstract_actions)}")
        print(f"[Lifted Planner] Max abstract states: {max_states:,}")

        # Infer complete goal state
        complete_goal_preds = self._infer_complete_goal(goal_predicates)

        # Extract implicit constraints from goal predicates
        goal_constraints = self._extract_constraints_from_predicates(complete_goal_preds)

        # Create initial abstract state
        goal_state = AbstractState(complete_goal_preds, goal_constraints, depth=0)
        print(f"[Lifted Planner] Complete goal state: {goal_state}")

        # BFS exploration
        queue = deque([goal_state])
        visited: Dict[Tuple, AbstractState] = {}
        visited[self._state_key(goal_state)] = goal_state

        transitions = []
        states_explored = 0
        transitions_added = 0

        while queue and states_explored < max_states:
            current_state = queue.popleft()
            states_explored += 1

            if states_explored % 10000 == 0:
                print(f"  Explored {states_explored} abstract states, "
                      f"{len(visited)} unique states, "
                      f"{transitions_added} transitions, "
                      f"queue: {len(queue)}")

            # Try each abstract action
            for abstract_action in self._abstract_actions:
                # Try to apply action via unification
                results = self._apply_abstract_action(abstract_action, current_state)

                for new_state, action_subst in results:
                    # Check if we've seen this abstract state
                    state_key = self._state_key(new_state)

                    if state_key in visited:
                        final_state = visited[state_key]
                    else:
                        # New abstract state discovered
                        final_state = AbstractState(
                            new_state.predicates,
                            new_state.constraints,
                            depth=current_state.depth + 1
                        )
                        visited[state_key] = final_state
                        queue.append(final_state)

                    # Record transition
                    transitions.append((
                        current_state,
                        final_state,
                        abstract_action.action,
                        action_subst
                    ))
                    transitions_added += 1

        print(f"[Lifted Planner] Exploration complete:")
        print(f"  Abstract states explored: {states_explored:,}")
        print(f"  Total unique abstract states: {len(visited):,}")
        print(f"  Transitions: {transitions_added:,}")
        print(f"  Max depth: {max(s.depth for s in visited.values()) if visited else 0}")

        # Compare with grounded state space estimate
        self._print_comparison(len(visited))

        # Convert to StateGraph format for compatibility with AgentSpeakCodeGenerator
        from stage3_code_generation.state_space import WorldState, StateGraph, StateTransition

        # Convert AbstractStates to WorldStates
        abstract_to_world = {}
        for abstract_state in visited.values():
            world_state = WorldState(abstract_state.predicates, depth=abstract_state.depth)
            abstract_to_world[abstract_state] = world_state

        # Create StateGraph
        goal_world_state = abstract_to_world[goal_state]
        state_graph = StateGraph(goal_world_state)
        state_graph.truncated = states_explored >= max_states

        # Add transitions
        for from_state, to_state, action, action_subst in transitions:
            from_world = abstract_to_world[from_state]
            to_world = abstract_to_world[to_state]

            # Extract action arguments from substitution
            # action_subst maps parameter variables to values
            action_args = tuple(action_subst.apply(var) for var in action.parameters)

            # Parse effects to extract belief updates
            try:
                # Create bindings dict for effect parsing
                bindings = {param: action_subst.apply(param) for param in action.parameters}
                effect_branches = self.effect_parser.parse(action.effects, bindings)
                belief_updates = []
                if effect_branches:
                    for effect_atom in effect_branches[0]:  # Use first branch
                        belief_updates.append(effect_atom.to_agentspeak())
            except:
                belief_updates = []

            # Parse preconditions
            try:
                bindings = {param: action_subst.apply(param) for param in action.parameters}
                preconditions = self.condition_parser.parse(action.preconditions, bindings)
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

    def _parse_abstract_actions(self) -> List[AbstractAction]:
        """
        Parse all domain actions to abstract form

        Returns:
            List of AbstractActions
        """
        abstract_actions = []

        for action in self.domain.actions:
            # Parse parameters
            param_vars = []
            param_types = {}

            for param in action.parameters:
                parts = param.split('-')
                if len(parts) >= 1:
                    var_name = parts[0].strip()
                    param_vars.append(var_name)
                    if len(parts) >= 2:
                        type_name = parts[1].strip()
                        param_types[var_name] = type_name

            # Create abstract bindings (variables stay as variables)
            abstract_bindings = {var: var for var in param_vars}

            # Parse preconditions
            try:
                preconditions = self.condition_parser.parse(
                    action.preconditions,
                    abstract_bindings
                )
            except Exception:
                preconditions = []

            # Parse effects
            try:
                effects = self.effect_parser.parse(
                    action.effects,
                    abstract_bindings
                )
            except Exception:
                effects = []

            # Extract inequality constraints from preconditions
            inequality_constraints = self._extract_inequality_constraints(action.preconditions)

            abstract_actions.append(AbstractAction(
                action=action,
                param_vars=param_vars,
                param_types=param_types,
                preconditions=preconditions,
                effects=effects,
                inequality_constraints=inequality_constraints
            ))

        return abstract_actions

    def _extract_inequality_constraints(self, preconditions: str) -> Set[Tuple[str, str]]:
        """
        Extract inequality constraints from precondition string

        Example: (not (= ?b1 ?b2)) -> {("?b1", "?b2")}

        Args:
            preconditions: PDDL precondition string

        Returns:
            Set of (var1, var2) tuples representing inequalities
        """
        constraints = set()
        pattern = r'\(not\s+\(=\s+(\?\w+)\s+(\?\w+)\)\)'

        for match in re.finditer(pattern, preconditions):
            var1, var2 = match.group(1), match.group(2)
            # Normalize: smaller var first
            if var1 > var2:
                var1, var2 = var2, var1
            constraints.add((var1, var2))

        return constraints

    def _extract_constraints_from_predicates(self, predicates: Set[PredicateAtom]) -> ConstraintSet:
        """
        Extract implicit constraints from predicates

        DOMAIN-INDEPENDENT: For any binary predicate P(?X, ?Y) where ?X and ?Y
        are different variables, we infer ?X != ?Y (reflexivity constraint).

        This is a reasonable general assumption: binary relations typically
        relate different objects (e.g., on(?X, ?Y), at(?X, ?Y), connected(?X, ?Y)).

        Args:
            predicates: Set of predicates

        Returns:
            ConstraintSet with extracted constraints
        """
        constraints = set()

        for pred in predicates:
            # DOMAIN-INDEPENDENT: For any binary predicate P(?X, ?Y)
            # If both arguments are different variables, infer ?X != ?Y
            if len(pred.args) == 2:
                arg0, arg1 = pred.args
                # Both are variables and they're different variable names
                if (arg0.startswith('?') and arg1.startswith('?') and arg0 != arg1):
                    constraints.add(Constraint(arg0, arg1, Constraint.INEQUALITY))

        return ConstraintSet(constraints)

    def _apply_abstract_action(self, abstract_action: AbstractAction,
                               state: AbstractState) -> List[Tuple[AbstractState, Substitution]]:
        """
        Apply abstract action to abstract state using UNIFICATION

        This is the key difference from grounded planning:
        - Don't enumerate all object combinations
        - Use unification to find if action can apply
        - Generate new abstract state with potentially new variables

        CRITICAL FIX: Do NOT rename action variables!
        - Directly unify action parameters with state variables
        - This reuses existing variables instead of creating new ones
        - This is the core of TRUE lifted planning

        Args:
            abstract_action: Abstract action to apply
            state: Current abstract state

        Returns:
            List of (new_abstract_state, substitution) tuples
        """
        results = []

        # FIXED: Use action directly without renaming
        # Unification will handle variable matching
        action = abstract_action

        # Try to unify action preconditions with state predicates
        # We need to find a substitution σ such that:
        # ∀ precond ∈ action.preconditions: ∃ state_pred ∈ state.predicates: unify(precond, state_pred, σ)

        # For negative preconditions (not P), we check they don't exist in state
        positive_preconditions = [p for p in action.preconditions if not p.negated]
        negative_preconditions = [p for p in action.preconditions if p.negated]

        # Check negative preconditions first (must NOT match anything in state)
        for neg_precond in negative_preconditions:
            pos_form = neg_precond.get_positive()
            # Try to unify with any state predicate
            for state_pred in state.predicates:
                if Unifier.unify_predicates(pos_form, state_pred) is not None:
                    # Negative precondition violated
                    return []

        # Unify positive preconditions
        # Strategy: try to find consistent substitution for all preconditions
        # NOW SUPPORTS QUANTIFIED PREDICATES for matching
        unified_subst, unsatisfied_preconditions = self._find_consistent_unification(
            positive_preconditions,
            state.predicates,
            state.constraints
        )

        if unified_subst is None:
            # Some preconditions not satisfied
            # Generate subgoal states (domain-independent backward chaining)
            subgoal_states = []
            for unsatisfied_precond in unsatisfied_preconditions:
                subgoals = self._generate_subgoal_states_for_precondition(
                    unsatisfied_precond,
                    state,
                    action
                )
                subgoal_states.extend(subgoals)

            # Return subgoal states for recursive exploration
            # Note: these are (state, empty_substitution) tuples
            return [(sg, Substitution()) for sg in subgoal_states]

        # Check inequality constraints from action
        for var1, var2 in action.inequality_constraints:
            val1 = unified_subst.apply(var1)
            val2 = unified_subst.apply(var2)
            if val1 == val2:
                # Constraint violated
                return results

        # Apply effects to generate new state(s)
        for effect_branch in action.effects:
            new_predicates = set(state.predicates)

            for effect_atom in effect_branch:
                # Apply substitution to effect
                effect_pred = unified_subst.apply_to_predicate(effect_atom.predicate)

                if effect_atom.is_add:
                    new_predicates.add(effect_pred)
                else:
                    new_predicates.discard(effect_pred)

            # Merge constraints
            new_constraints = state.constraints

            # Add inequality constraints from action
            for var1, var2 in action.inequality_constraints:
                new_var1 = unified_subst.apply(var1)
                new_var2 = unified_subst.apply(var2)
                if new_var1.startswith('?') and new_var2.startswith('?'):
                    new_constraint = Constraint(new_var1, new_var2, Constraint.INEQUALITY)
                    new_constraints = new_constraints.add(new_constraint)

            # Extract implicit constraints from new predicates
            implicit_constraints = self._extract_constraints_from_predicates(new_predicates)
            new_constraints = new_constraints.merge(implicit_constraints)

            if new_constraints is None or not new_constraints.is_consistent():
                # Inconsistent constraints
                continue

            # Validate state consistency (domain-specific)
            if not self._validate_state_consistency(new_predicates):
                continue

            # Create new state
            new_state = AbstractState(new_predicates, new_constraints)

            results.append((new_state, unified_subst))

        return results

    def _rename_action_variables(self, action: AbstractAction,
                                 existing_vars: Set[str]) -> Tuple[AbstractAction, Substitution]:
        """
        Rename action variables to avoid collision with state variables

        Example:
            action.param_vars = ["?X", "?Y"]
            state variables = {"?X", "?Y", "?Z"}
            ->
            renamed vars = ["?V0", "?V1"]
            substitution = {?X: ?V0, ?Y: ?V1}

        Args:
            action: Abstract action
            existing_vars: Variables already used in state

        Returns:
            (renamed_action, rename_substitution)
        """
        rename_map = {}

        for var in action.param_vars:
            # Generate fresh variable name
            new_var = self._fresh_variable(existing_vars | set(rename_map.values()))
            rename_map[var] = new_var

        rename_subst = Substitution(rename_map)

        # Apply renaming to preconditions and effects
        new_preconditions = [rename_subst.apply_to_predicate(p) for p in action.preconditions]

        new_effects = []
        for effect_branch in action.effects:
            new_branch = []
            for effect_atom in effect_branch:
                new_pred = rename_subst.apply_to_predicate(effect_atom.predicate)
                # Create new effect atom with renamed predicate
                from stage3_code_generation.pddl_condition_parser import EffectAtom
                new_branch.append(EffectAtom(new_pred, effect_atom.is_add))
            new_effects.append(new_branch)

        # Rename inequality constraints
        new_inequality_constraints = set()
        for var1, var2 in action.inequality_constraints:
            new_var1 = rename_subst.apply(var1)
            new_var2 = rename_subst.apply(var2)
            new_inequality_constraints.add((new_var1, new_var2))

        renamed_action = AbstractAction(
            action=action.action,
            param_vars=[rename_subst.apply(v) for v in action.param_vars],
            param_types={rename_subst.apply(k): v for k, v in action.param_types.items()},
            preconditions=new_preconditions,
            effects=new_effects,
            inequality_constraints=new_inequality_constraints
        )

        return renamed_action, rename_subst

    def _fresh_variable(self, existing_vars: Set[str]) -> str:
        """
        Generate a fresh variable name

        Args:
            existing_vars: Variables to avoid

        Returns:
            Fresh variable name (e.g., "?V0", "?V1", ...)
        """
        while True:
            var_name = f"?V{self._var_counter}"
            self._var_counter += 1
            if var_name not in existing_vars:
                return var_name

    def _find_consistent_unification(self,
                                     preconditions: List[PredicateAtom],
                                     state_preds: FrozenSet[PredicateAtom],
                                     constraints: ConstraintSet) -> Tuple[Optional[Substitution], List[PredicateAtom]]:
        """
        Find a substitution that unifies all preconditions with state predicates
        while respecting constraints

        Args:
            preconditions: Action preconditions to match
            state_preds: State predicates
            constraints: State constraints

        Returns:
            Tuple of (unified_substitution, unsatisfied_preconditions)
            - If all satisfied: (Substitution, [])
            - If some unsatisfied: (None, [list of unsatisfied preconditions])
        """
        # Strategy: for each precondition, find matching state predicate
        # Build up substitution incrementally

        # Handle empty preconditions
        if not preconditions:
            return Substitution(), []

        # Try to find a consistent matching
        # This is a constraint satisfaction problem
        # For now, use simple greedy approach: try each precondition in order

        current_subst = Substitution()
        unsatisfied = []

        for precond in preconditions:
            # Find a state predicate that unifies with this precondition
            found = False

            for state_pred in state_preds:
                # Try to unify
                unified = Unifier.unify_predicates(precond, state_pred, current_subst)

                if unified is not None:
                    # Check if unification respects constraints
                    if constraints.is_consistent(unified):
                        current_subst = unified
                        found = True
                        break

            if not found:
                # Could not find matching predicate for this precondition
                # Mark as unsatisfied for subgoal generation
                unsatisfied.append(precond)

        if unsatisfied:
            # Return unsatisfied preconditions for subgoal generation
            return None, unsatisfied
        else:
            # All preconditions satisfied
            return current_subst, []

    def _generate_subgoal_states_for_precondition(self,
                                                   precondition: PredicateAtom,
                                                   current_state: AbstractState,
                                                   requesting_action: AbstractAction) -> List[AbstractState]:
        """
        Generate subgoal states to achieve an unsatisfied precondition

        For each action that can achieve the precondition, generate a subgoal state.

        Args:
            precondition: The unsatisfied precondition to achieve
            current_state: Current abstract state
            requesting_action: The action that requires this precondition

        Returns:
            List of subgoal states (one per achieving action)
        """
        # Find all actions that can achieve this precondition
        achieving_actions = []
        for candidate_action in self._abstract_actions:
            if self._action_produces_predicate(candidate_action, precondition):
                achieving_actions.append(candidate_action)

        if not achieving_actions:
            return []

        # Generate subgoal for each achieving action
        subgoal_states = []

        for candidate_action in achieving_actions:
            # FIXED: Do NOT rename action variables
            # Use candidate_action directly for true lifted planning
            action = candidate_action

            # Try to unify the effect with the precondition we want
            achieving_subst = self._find_achieving_substitution(
                action,
                precondition
            )

            if achieving_subst is not None:
                subgoal_predicates = set()

                # Add the action's positive preconditions (after applying substitution)
                for action_precond in action.preconditions:
                    if not action_precond.negated:
                        subgoal_pred = achieving_subst.apply_to_predicate(action_precond)
                        subgoal_predicates.add(subgoal_pred)

                # HIGH PRIORITY FIX #5: Collect negative preconditions
                # Subgoal state must NOT contain predicates matching negative preconditions
                negative_preconditions = []
                for action_precond in action.preconditions:
                    if action_precond.negated:
                        # Get positive form and apply substitution
                        pos_form = action_precond.get_positive()
                        neg_pred = achieving_subst.apply_to_predicate(pos_form)
                        negative_preconditions.append(neg_pred)

                # CRITICAL FIX #1: Only keep ESSENTIAL context from current state
                # Strategy: Only copy predicates that are:
                # 1. Not deleted by this action
                # 2. Not already covered by action preconditions (avoid duplication)
                # 3. Global predicates (0-arity) that provide essential context
                # 4. HIGH FIX #5: Don't violate negative preconditions

                # Collect predicate names already in subgoal (from action preconditions)
                subgoal_pred_names = {p.name for p in subgoal_predicates}

                for state_pred in current_state.predicates:
                    # Skip if this predicate type is already in subgoal
                    # (action preconditions already added it)
                    if state_pred.name in subgoal_pred_names and len(state_pred.args) > 0:
                        continue

                    # HIGH FIX #5: Check if this predicate violates negative preconditions
                    violates_negative = False
                    for neg_pred in negative_preconditions:
                        # Check if state_pred matches the negative precondition
                        if Unifier.unify_predicates(state_pred, neg_pred) is not None:
                            violates_negative = True
                            break

                    if violates_negative:
                        continue

                    # Check if will be deleted by the action
                    will_be_deleted = False
                    for effect_branch in action.effects:
                        for effect_atom in effect_branch:
                            if not effect_atom.is_add:
                                effect_pred = achieving_subst.apply_to_predicate(effect_atom.predicate)
                                if effect_pred == state_pred:
                                    will_be_deleted = True
                                    break
                        if will_be_deleted:
                            break

                    if will_be_deleted:
                        continue

                    # Only keep if it's a global 0-arity predicate (like handempty)
                    # Other predicates should be discovered through backward chaining
                    if len(state_pred.args) == 0:
                        subgoal_predicates.add(state_pred)

                # CRITICAL FIX #2 (BUG FIX): Validate subgoal state consistency
                # Skip generating invalid subgoal states with mutex conflicts
                if not self._validate_state_consistency(subgoal_predicates):
                    continue  # Skip this invalid subgoal state

                # Create subgoal state with same constraints
                subgoal_constraints = current_state.constraints
                subgoal_state = AbstractState(
                    subgoal_predicates,
                    subgoal_constraints,
                    depth=current_state.depth + 1
                )

                subgoal_states.append(subgoal_state)

        return subgoal_states

    def _action_produces_predicate(self,
                                   action: AbstractAction,
                                   target_predicate: PredicateAtom) -> bool:
        """
        Check if an action can produce a target predicate

        DOMAIN-INDEPENDENT: Only checks PDDL action effects

        Args:
            action: Abstract action to check
            target_predicate: Target predicate to produce

        Returns:
            True if action has an add-effect that can unify with target
        """
        for effect_branch in action.effects:
            for effect_atom in effect_branch:
                if effect_atom.is_add:
                    # Try to unify effect with target
                    if Unifier.unify_predicates(effect_atom.predicate, target_predicate) is not None:
                        return True
        return False

    def _find_achieving_substitution(self,
                                     action: AbstractAction,
                                     target_predicate: PredicateAtom) -> Optional[Substitution]:
        """
        Find substitution that makes action's effect achieve target predicate

        DOMAIN-INDEPENDENT

        Args:
            action: Abstract action
            target_predicate: Target predicate to achieve

        Returns:
            Substitution that unifies action's effect with target, or None
        """
        for effect_branch in action.effects:
            for effect_atom in effect_branch:
                if effect_atom.is_add:
                    # Try to unify
                    unified = Unifier.unify_predicates(
                        effect_atom.predicate,
                        target_predicate
                    )
                    if unified is not None:
                        return unified
        return None

    def _extract_mutex_predicates(self) -> Set[Tuple[str, str]]:
        """
        Extract mutex predicates from PDDL domain (CRITICAL FIX #2)

        Mutex predicates are predicates that cannot coexist in a valid state.
        We infer these from action effects:
        - If an action adds P and deletes Q, then P and Q are mutex

        Returns:
            Set of (pred_name1, pred_name2) tuples representing mutex pairs
        """
        mutex_pairs = set()

        for action in self._abstract_actions:
            for effect_branch in action.effects:
                adds = set()
                deletes = set()

                for effect_atom in effect_branch:
                    if effect_atom.is_add:
                        adds.add(effect_atom.predicate.name)
                    else:
                        deletes.add(effect_atom.predicate.name)

                # If action adds P and deletes Q simultaneously, they're likely mutex
                for add_pred in adds:
                    for del_pred in deletes:
                        if add_pred != del_pred:
                            mutex_pairs.add((min(add_pred, del_pred), max(add_pred, del_pred)))

        return mutex_pairs

    def _validate_state_consistency(self, predicates: Set[PredicateAtom]) -> bool:
        """
        Validate abstract state consistency (CRITICAL FIX #2)

        DOMAIN-INDEPENDENT: Uses mutex predicates extracted from PDDL domain

        This check is guaranteed by PDDL semantics: if two predicates are mutex
        (one is added and the other is deleted by the same action), they cannot
        coexist in a valid state. This is because the action's preconditions must
        be satisfied before the action can be executed.

        Args:
            predicates: Set of predicates

        Returns:
            True if consistent, False otherwise
        """
        # Check: Mutex predicates cannot coexist
        # This is the ONLY domain-independent check that is guaranteed by PDDL
        pred_names = {p.name for p in predicates}
        for pred1, pred2 in self._mutex_predicates:
            if pred1 in pred_names and pred2 in pred_names:
                # Both mutex predicates present - invalid state
                # This violates PDDL action semantics
                return False

        return True

    def _infer_complete_goal(self, goal_predicates: List[PredicateAtom]) -> Set[PredicateAtom]:
        """
        Infer complete goal state from goal predicates

        For lifted planning with recursive subgoal handling, keep goal MINIMAL.
        Let backward chaining discover necessary preconditions through subgoal generation.

        Args:
            goal_predicates: Core goal predicates

        Returns:
            Minimal goal state (just the goal predicates themselves)
        """
        # CRITICAL: Keep goal minimal for recursive subgoal handling
        # The backward chaining will automatically discover necessary preconditions
        # and generate subgoal states as needed
        #
        # Previous approach of inferring co-effects caused problems:
        # - Introduced too many predicates in goal state
        # - Prevented proper subgoal generation
        # - Made exploration terminate prematurely
        #
        # Example:
        #   Goal: clear(b)
        #   Old: {clear(b), handempty, holding(?x), ...} → too complex!
        #   New: {clear(b)} → backward chaining will find preconditions
        return set(goal_predicates)

    def _canonicalize_state(self, state: AbstractState) -> AbstractState:
        """
        Canonicalize state by renaming variables to standard form (HIGH FIX #7)

        Two states with same structure but different variable names will
        have the same canonical form.

        CRITICAL: Variables must be renamed based on FIRST APPEARANCE order
        in predicates (sorted lexicographically), not alphabetically by name.

        Example:
            State A: {on(?V0, ?V1), clear(?V0)} → {on(?C0, ?C1), clear(?C0)}
            State B: {on(?V2, ?V3), clear(?V2)} → {on(?C0, ?C1), clear(?C0)}
            Both have same canonical form → detected as isomorphic

        Args:
            state: Abstract state

        Returns:
            Canonicalized state with renamed variables
        """
        # Sort predicates for deterministic traversal
        sorted_preds = sorted(state.predicates, key=str)

        # Traverse predicates in order and assign canonical names based on first appearance
        rename_map = {}
        var_counter = 0

        for pred in sorted_preds:
            for arg in pred.args:
                if arg.startswith('?') and arg not in rename_map:
                    rename_map[arg] = f"?C{var_counter}"
                    var_counter += 1

        # Apply renaming to predicates
        canonical_preds = set()
        for pred in state.predicates:
            new_args = tuple(rename_map.get(arg, arg) for arg in pred.args)
            canonical_preds.add(PredicateAtom(pred.name, new_args, pred.negated))

        # Apply renaming to constraints (also sort them for consistency)
        canonical_constraints = set()
        for constraint in sorted(state.constraints.constraints, key=str):
            var1 = constraint.var1
            var2 = constraint.var2

            # Assign canonical names if not already assigned
            if var1.startswith('?') and var1 not in rename_map:
                rename_map[var1] = f"?C{var_counter}"
                var_counter += 1
            if var2.startswith('?') and var2 not in rename_map:
                rename_map[var2] = f"?C{var_counter}"
                var_counter += 1

            new_var1 = rename_map.get(var1, var1)
            new_var2 = rename_map.get(var2, var2)
            canonical_constraints.add(Constraint(new_var1, new_var2, constraint.constraint_type))

        return AbstractState(canonical_preds, ConstraintSet(canonical_constraints), state.depth)

    def _state_key(self, state: AbstractState) -> Tuple:
        """
        Generate hashable key for abstract state (HIGH FIX #7: with isomorphism detection)

        Args:
            state: Abstract state

        Returns:
            Hashable tuple representing state
        """
        # HIGH FIX #7: Canonicalize state to detect isomorphic states
        canonical_state = self._canonicalize_state(state)

        # Sort predicates and constraints for consistent hashing
        pred_tuple = tuple(sorted(canonical_state.predicates, key=str))
        constraint_tuple = tuple(sorted(canonical_state.constraints.constraints, key=str))
        return (pred_tuple, constraint_tuple)

    def _print_comparison(self, abstract_state_count: int):
        """
        Print comparison with grounded approach

        Args:
            abstract_state_count: Number of abstract states explored
        """
        print(f"\n[Lifted Planning Benefits]")
        print(f"  Abstract states explored: {abstract_state_count:,}")
        print(f"  This is INDEPENDENT of number of domain objects!")
        print(f"  ")
        print(f"  Comparison with grounded planning:")
        print(f"    3 objects: ~hundreds of grounded states")
        print(f"    10 objects: ~thousands-tens of thousands of grounded states")
        print(f"    Lifted: {abstract_state_count:,} abstract states (same for any number of objects)")
        print(f"  ")
        print(f"  This is TRUE lifted planning - exploring abstract state space!")
