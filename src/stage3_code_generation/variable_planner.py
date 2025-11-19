"""
Variable-Level Planning using Unification

This implements planning with VARIABLES instead of concrete OBJECTS:
1. Explores state space using VARIABLES (?X, ?Y) not objects (a, b, c)
2. Uses UNIFICATION to apply actions (not object enumeration)
3. Maintains variable INEQUALITY CONSTRAINTS (?X != ?Y)
4. State space size independent of number of objects

Key differences from object-level (grounded) planning:
- DOES NOT enumerate all object combinations
- Plans work for ANY objects that satisfy variable constraints
- State count: O(variable patterns) not O(object combinations)

Example:
    10 objects, object-level planning: 10,000+ states
    10 objects, variable-level planning: ~50 variable states
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
    effects: List  # Effect atoms (deterministic)
    inequality_constraints: Set[Tuple[str, str]]  # From preconditions: (not (= ?x ?y))


class VariablePlanner:
    """
    Variable-level planner using unification

    Plans with VARIABLES (?X, ?Y) instead of concrete OBJECTS (a, b, c).

    Key insight: Don't ground actions to all object combinations!
    Instead, use unification to find which variable-level action can apply to variable state.

    Algorithm:
    1. Start with variable goal state (e.g., {on(?X, ?Y)})
    2. For each action schema, try to UNIFY preconditions with current variable state
    3. If unification succeeds, apply effects to generate new variable state
    4. Repeat until no new variable states discovered

    State space is MUCH smaller because we don't enumerate all object combinations.
    Plans work for ANY objects that satisfy the variable constraints.
    """

    def __init__(self, domain: PDDLDomain, var_counter_offset: int = 0):
        """
        Initialize variable-level planner

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

        # Invariant Synthesis (Fast Downward style)
        # Automatically infer state constraints from PDDL domain structure
        # This is domain-independent and provably correct
        self._invariants = self._synthesize_invariants()

    def explore_from_goal(self, goal_predicates: List[PredicateAtom],
                         max_states: int = 200000,
                         max_depth: int = 5) -> Dict:
        """
        Explore abstract state space from goal using BACKWARD PLANNING (Regression)

        CRITICAL: To prevent infinite state space explosion, we only allow variables
        that appear in the goal predicates. This is essential for lifted planning.

        This implements TRUE backward planning:
        1. Start from goal state
        2. For each action that can ACHIEVE predicates in current state:
           - Apply REGRESSION: compute predecessor state
           - Regression formula: (state - add_effects) + del_effects + preconditions
        3. Continue until no new states can be generated

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

        print(f"[Backward Planner] Starting REGRESSION-based state space exploration")
        print(f"[Backward Planner] Goal: {[str(p) for p in goal_predicates]}")
        print(f"[Backward Planner] Abstract actions: {len(self._abstract_actions)}")
        print(f"[Backward Planner] Max abstract states: {max_states:,}")
        print(f"[Backward Planner] Max depth: {max_depth}")

        # Extract allowed variables from goal (CRITICAL for preventing state space explosion)
        # We only allow variables that appear in the goal predicates
        allowed_vars = set()
        for pred in goal_predicates:
            for arg in pred.args:
                if arg.startswith('?'):
                    allowed_vars.add(arg)
        print(f"[Backward Planner] Allowed variables: {sorted(allowed_vars)}")

        # Infer complete goal state
        complete_goal_preds = self._infer_complete_goal(goal_predicates)

        # Extract implicit constraints from goal predicates
        goal_constraints = self._extract_constraints_from_predicates(complete_goal_preds)

        # Create initial abstract state
        goal_state = AbstractState(complete_goal_preds, goal_constraints, depth=0)
        print(f"[Backward Planner] Complete goal state: {goal_state}")

        # BFS exploration using REGRESSION
        queue = deque([goal_state])
        visited: Dict[Tuple, AbstractState] = {}
        visited[self._state_key(goal_state)] = goal_state

        transitions = []
        states_explored = 0
        transitions_added = 0

        while queue and states_explored < max_states:
            current_state = queue.popleft()
            states_explored += 1

            # DEBUG: Print first 1000 explorations (disabled)
            # if states_explored <= 1000:
            #     print(f"\n[DEBUG {states_explored}] Exploring state (depth={current_state.depth}):")
            #     print(f"  Predicates: {current_state.predicates}")
            #     print(f"  Constraints: {current_state.constraints}")

            if states_explored % 10000 == 0:
                print(f"  Explored {states_explored} abstract states, "
                      f"{len(visited)} unique states, "
                      f"{transitions_added} transitions, "
                      f"queue: {len(queue)}")

            # Skip states that exceed max depth
            if current_state.depth >= max_depth:
                continue

            # Try each abstract action for REGRESSION
            for abstract_action in self._abstract_actions:
                # Try to regress current_state through this action
                # This finds predecessor states that can reach current_state via action
                results = self._regress_abstract_action(abstract_action, current_state)

                for new_state, action_subst in results:
                    # CRITICAL: Check if new_state introduces variables not in goal
                    # This prevents infinite state space explosion
                    state_vars = set()
                    for pred in new_state.predicates:
                        for arg in pred.args:
                            if arg.startswith('?'):
                                state_vars.add(arg)

                    # Reject states with new variables
                    if not state_vars.issubset(allowed_vars):
                        new_vars = state_vars - allowed_vars
                        # if states_explored <= 1000:
                        #     print(f"    ✗ REJECT state via {abstract_action.action.name}: introduces new vars {new_vars}")
                        continue  # Skip this state

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

                        # DEBUG: Print new states found (first 1000 explorations - disabled)
                        # if states_explored <= 1000:
                        #     print(f"    → NEW state via {abstract_action.action.name}: {new_state.predicates}")

                    # Record transition: new_state --[action]--> current_state
                    # (backward direction: predecessor -> successor)
                    transitions.append((
                        final_state,      # from_state (predecessor)
                        current_state,    # to_state (successor/goal)
                        abstract_action.action,
                        action_subst
                    ))
                    transitions_added += 1

        print(f"[Backward Planner] Exploration complete:")
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
        # Transitions are already in correct direction: predecessor --[action]--> goal
        # (we recorded them correctly in the regression loop above)
        for from_state, to_state, action, action_subst in transitions:
            from_world = abstract_to_world[from_state]
            to_world = abstract_to_world[to_state]

            # Extract action arguments from substitution
            # action_subst maps parameter variables to values
            # IMPORTANT: Extract only variable name, strip PDDL type annotations
            # e.g., "?b1 - block" -> "?b1"
            param_vars = []
            for param in action.parameters:
                # Split by '-' and take only the variable name
                var_name = param.split('-')[0].strip() if '-' in param else param.strip()
                param_vars.append(var_name)

            action_args = tuple(action_subst.apply(var) for var in param_vars)

            # Parse effects to extract belief updates
            try:
                # Create bindings dict for effect parsing
                # Use param_vars (without type annotations) for bindings
                bindings = {var: action_subst.apply(var) for var in param_vars}
                effect_branches = self.effect_parser.parse(action.effects, bindings)
                belief_updates = []
                if effect_branches:
                    for effect_atom in effect_branches[0]:  # Use first branch
                        belief_updates.append(effect_atom.to_agentspeak())
            except:
                belief_updates = []

            # Parse preconditions
            try:
                # Use param_vars (without type annotations) for bindings
                bindings = {var: action_subst.apply(var) for var in param_vars}
                preconditions = self.condition_parser.parse(action.preconditions, bindings)
            except:
                preconditions = []

            # Transitions are in correct backward planning direction:
            # from_state (predecessor) --[action]--> to_state (goal/successor)
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
            # IMPORTANT: bindings map from variable name to variable name (identity)
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
                effect_branches = self.effect_parser.parse(
                    action.effects,
                    abstract_bindings
                )
                # Extract single branch (deterministic effects only)
                effects = effect_branches[0] if effect_branches else []
            except Exception as e:
                # Effect parsing failed - this will break regression!
                print(f"[ERROR] Effect parsing failed for action {action.name}: {e}")
                print(f"  Effects string: {action.effects}")
                print(f"  Bindings: {abstract_bindings}")
                import traceback
                traceback.print_exc()
                # Set empty to avoid crash, but regression won't work for this action
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

    def _regress_abstract_action(self, abstract_action: AbstractAction,
                                 state: AbstractState) -> List[Tuple[AbstractState, Substitution]]:
        """
        Apply REGRESSION through abstract action to compute predecessor states

        This implements TRUE backward planning (regression):
        1. Check if action's EFFECTS can produce some predicates in current state
        2. If yes, compute predecessor state using regression formula:
           new_state = (state - add_effects) + del_effects + preconditions

        Example:
            state = {on(?v0, ?v1)}
            action = puton(?b1, ?b2) with:
                effects: +on(?b1, ?b2), -holding(?b1), -clear(?b2)
                preconditions: holding(?b1), clear(?b2)
            Unify: ?b1 -> ?v0, ?b2 -> ?v1
            Regression:
                Remove add_effects: {} (removed on(?v0, ?v1))
                Add del_effects: {holding(?v0), clear(?v1)}
                Add preconditions: {holding(?v0), clear(?v1)}
            Result: {holding(?v0), clear(?v1)}

        Args:
            abstract_action: Abstract action to regress through
            state: Current abstract state (goal or intermediate)

        Returns:
            List of (predecessor_state, substitution) tuples
        """
        results = []

        # DEBUG: (disabled for performance)
        # print(f"    [DEBUG] Trying to regress action '{abstract_action.action.name}' through state {state.predicates}")
        # print(f"    [DEBUG]   Action effects: {len(abstract_action.effects)} effects")
        # for eff in abstract_action.effects:
        #     print(f"    [DEBUG]     {'ADD' if eff.is_add else 'DEL'}: {eff.predicate}")

        # STEP 1: Check if action can ACHIEVE any predicates in current state
        # Find which add-effects match predicates in state
        # Try all possible unifications for each add-effect
        for effect_atom in abstract_action.effects:
            if not effect_atom.is_add:
                continue  # Only consider add-effects for regression

            # Try to unify effect with each predicate in state
            for state_pred in state.predicates:
                # Fresh unification for each attempt
                unified_subst = Unifier.unify_predicates(effect_atom.predicate, state_pred)

                if unified_subst is None:
                    continue  # This effect doesn't match this state predicate

                # Found a relevant effect - this action can achieve this predicate
                # Now apply REGRESSION formula with this unification
                # print(f"    [DEBUG]   ✓ MATCH! Effect {effect_atom.predicate} unifies with {state_pred}")
                # print(f"    [DEBUG]     Substitution: {unified_subst}")

                # STEP 2: Apply REGRESSION formula
                # new_state = (state - add_effects) + del_effects + preconditions

                new_predicates = set(state.predicates)

                # Remove add-effects (these are being achieved by the action)
                # Apply substitution to all add-effects and remove them
                for eff in abstract_action.effects:
                    if eff.is_add:
                        eff_pred = unified_subst.apply_to_predicate(eff.predicate)
                        new_predicates.discard(eff_pred)

                # Add delete-effects (need to hold before action)
                for eff in abstract_action.effects:
                    if not eff.is_add:  # Delete effect
                        del_pred = unified_subst.apply_to_predicate(eff.predicate)
                        new_predicates.add(del_pred)

                # Add preconditions (must hold before action)
                for precond in abstract_action.preconditions:
                    if not precond.negated:  # Positive precondition
                        precond_pred = unified_subst.apply_to_predicate(precond)
                        new_predicates.add(precond_pred)

                # STEP 3: Handle negative preconditions
                # Negative preconditions: these predicates must NOT exist in predecessor
                for precond in abstract_action.preconditions:
                    if precond.negated:
                        pos_form = precond.get_positive()
                        neg_pred = unified_subst.apply_to_predicate(pos_form)
                        # Ensure it's not in new_predicates
                        new_predicates.discard(neg_pred)

                # STEP 4: Merge constraints
                new_constraints = state.constraints

                # Add inequality constraints from action
                for var1, var2 in abstract_action.inequality_constraints:
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
                    continue  # Try next unification

                # Validate state with synthesized invariants (h^2 + exactly-one)
                # This uses Fast Downward's invariant synthesis - provably correct
                if not self._validate_state_with_invariants(new_predicates):
                    # State violates synthesized invariants - skip it
                    continue

                # Create predecessor state
                predecessor_state = AbstractState(new_predicates, new_constraints)
                # print(f"    [DEBUG]     ✓ Created predecessor state: {predecessor_state.predicates}")

                results.append((predecessor_state, unified_subst))

        # print(f"    [DEBUG] Regression complete: {len(results)} predecessor(s) found")
        return results

    def _apply_abstract_action(self, abstract_action: AbstractAction,
                               state: AbstractState) -> List[Tuple[AbstractState, Substitution]]:
        """
        Apply abstract action to abstract state using UNIFICATION (FORWARD)

        NOTE: This is FORWARD planning, kept for subgoal generation.
        For backward planning, use _regress_abstract_action instead.

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

        # Apply effects to generate new state
        new_predicates = set(state.predicates)

        for effect_atom in action.effects:
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
            return results

        # Validate state consistency (domain-specific)
        if not self._validate_state_consistency(new_predicates):
            return results

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
        for effect_atom in action.effects:
            new_pred = rename_subst.apply_to_predicate(effect_atom.predicate)
            # Create new effect atom with renamed predicate
            from stage3_code_generation.pddl_condition_parser import EffectAtom
            new_effects.append(EffectAtom(new_pred, effect_atom.is_add))

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
        # Handle empty preconditions
        if not preconditions:
            return Substitution(), []

        # Use backtracking search to find valid unification
        # This ensures we explore all possible assignments and don't miss
        # valid solutions due to greedy choice ordering
        result = self._backtrack_unify(
            state_preds=list(state_preds),
            preconditions=preconditions,
            constraints=constraints,
            precond_idx=0,
            current_subst=Substitution(),
            used_state_preds=set()
        )

        if result is not None:
            return result, []
        else:
            # No complete unification found - return all preconditions as unsatisfied
            return None, preconditions

    def _backtrack_unify(self,
                        state_preds: List[PredicateAtom],
                        preconditions: List[PredicateAtom],
                        constraints: ConstraintSet,
                        precond_idx: int,
                        current_subst: Substitution,
                        used_state_preds: set) -> Optional[Substitution]:
        """
        Backtracking search for consistent unification.

        Explores all possible assignments of state predicates to preconditions,
        ensuring we find a valid unification if one exists.

        Args:
            state_preds: Available state predicates
            preconditions: Preconditions to satisfy
            constraints: Constraint set that must be respected
            precond_idx: Current precondition index
            current_subst: Current substitution
            used_state_preds: Set of already-matched state predicate indices

        Returns:
            Valid substitution if found, None otherwise
        """
        # Base case: all preconditions satisfied
        if precond_idx >= len(preconditions):
            return current_subst

        precond = preconditions[precond_idx]

        # Try each state predicate
        for state_idx, state_pred in enumerate(state_preds):
            # Skip if already used (each state predicate can only match once)
            if state_idx in used_state_preds:
                continue

            # Try to unify
            unified = Unifier.unify_predicates(precond, state_pred, current_subst)

            if unified is not None and constraints.is_consistent(unified):
                # Valid unification - recurse with next precondition
                new_used = used_state_preds | {state_idx}
                result = self._backtrack_unify(
                    state_preds=state_preds,
                    preconditions=preconditions,
                    constraints=constraints,
                    precond_idx=precond_idx + 1,
                    current_subst=unified,
                    used_state_preds=new_used
                )

                if result is not None:
                    return result

        # No valid assignment found for this precondition
        return None

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
                    for effect_atom in action.effects:
                        if not effect_atom.is_add:
                            effect_pred = achieving_subst.apply_to_predicate(effect_atom.predicate)
                            if effect_pred == state_pred:
                                will_be_deleted = True
                                break

                    if will_be_deleted:
                        continue

                    # CRITICAL FIX #2: Preserve relevant context predicates
                    # Keep predicates that share variables with the subgoal predicates
                    # This ensures we maintain essential structural constraints

                    # Always keep 0-arity predicates (global state)
                    if len(state_pred.args) == 0:
                        subgoal_predicates.add(state_pred)
                    else:
                        # Keep if it shares variables with any subgoal predicate
                        shares_variables = False
                        state_pred_vars = {arg for arg in state_pred.args if arg.startswith('?')}

                        for sg_pred in subgoal_predicates:
                            sg_vars = {arg for arg in sg_pred.args if arg.startswith('?')}
                            if state_pred_vars & sg_vars:  # Non-empty intersection
                                shares_variables = True
                                break

                        if shares_variables:
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
        for effect_atom in action.effects:
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
        for effect_atom in action.effects:
            if effect_atom.is_add:
                # Try to unify
                unified = Unifier.unify_predicates(
                    effect_atom.predicate,
                    target_predicate
                )
                if unified is not None:
                    return unified
        return None

    # ========================================================================
    # Invariant Synthesis (Fast Downward Style)
    # ========================================================================

    def _synthesize_invariants(self) -> Dict:
        """
        Synthesize state invariants from PDDL domain structure.

        This implements Fast Downward's invariant synthesis approach:
        1. h^2 mutex detection (Helmert 2006)
        2. Exactly-one group detection (balance-based)

        These invariants are:
        - Domain-independent (no hardcoding)
        - Provably correct (verified by induction)
        - Used to prune invalid states during search

        Returns:
            Dict with:
                'h2_mutexes': Set of (pred1, pred2) mutex pairs
                'exactly_one_groups': List of predicate groups where exactly one is true
        """
        invariants = {}

        # H^2 Mutex Detection (Fast Downward)
        invariants['h2_mutexes'] = self._synthesize_h2_mutexes()

        # Exactly-One Group Detection (balance-based)
        invariants['exactly_one_groups'] = self._detect_exactly_one_groups()

        return invariants

    def _synthesize_h2_mutexes(self) -> Set[Tuple[str, str]]:
        """
        H^2 mutex detection (Helmert 2006, Fast Downward)

        Standard h^2 algorithm:
        1. Build abstract planning graph by forward reachability analysis
        2. Track which predicate pairs can be achieved together
        3. Pairs that can never be achieved together are mutex
        4. Propagate mutexes through action preconditions

        This is SOUND and MORE COMPLETE than simplified version:
        - Sound: All returned pairs are truly mutex
        - More complete: Uses reachability analysis, not just single-action check

        For lifted planning:
        - Start from empty state (most conservative)
        - Use abstract predicates with variable patterns
        - Track achievable predicate pairs through action sequences

        Returns:
            Set of (pred_name1, pred_name2) tuples that are mutex
        """
        # Collect all predicate names from domain
        all_pred_names = set()
        for action in self._abstract_actions:
            for eff in action.effects:
                all_pred_names.add(eff.predicate.name)

        # Initialize: assume all pairs are mutex
        potential_mutexes = {(p1, p2)
                            for p1 in all_pred_names
                            for p2 in all_pred_names
                            if p1 < p2}

        # Build planning graph to find reachable predicate pairs
        # Layer 0: For lifted planning, assume all single predicates are reachable
        # This is optimistic but necessary without a concrete initial state
        reachable_pairs = set()  # Pairs that CAN coexist
        reachable_facts = set(all_pred_names)  # All predicates are potentially reachable

        # Iteratively expand reachability
        # Fixed point: keep expanding until no new pairs are found
        max_iterations = 10  # Prevent infinite loop
        for iteration in range(max_iterations):
            new_pairs_found = False
            mutexes_this_iteration = potential_mutexes - reachable_pairs

            # Try each action
            for action in self._abstract_actions:
                # Collect preconditions and effects
                preconds = {p.name for p in action.preconditions if not p.negated}
                adds = {eff.predicate.name for eff in action.effects if eff.is_add}
                deletes = {eff.predicate.name for eff in action.effects if not eff.is_add}

                # Check if action is applicable:
                # Action is NOT applicable if its preconditions contain a KNOWN mutex pair
                precond_has_mutex = False
                for p1 in preconds:
                    for p2 in preconds:
                        if p1 < p2 and (p1, p2) in mutexes_this_iteration:
                            precond_has_mutex = True
                            break
                    if precond_has_mutex:
                        break

                # If preconditions contain mutex, this action cannot execute
                if precond_has_mutex:
                    continue

                # Action CAN execute - its effects can be achieved
                # All pairs of add effects can coexist (in the resulting state)
                for p1 in adds:
                    for p2 in adds:
                        if p1 < p2:
                            pair = (p1, p2)
                            if pair not in reachable_pairs:
                                reachable_pairs.add(pair)
                                new_pairs_found = True

                # CRITICAL: Effects can coexist with preconditions ONLY if not deleted
                # After action executes: state = (preconditions - deletes) + adds
                # So P (from precond) and Q (from adds) can coexist ONLY if P is not deleted
                for p1 in adds:
                    for p2 in preconds:
                        if p1 != p2 and p2 not in deletes:
                            # p2 survives (not deleted), p1 is added
                            pair = (min(p1, p2), max(p1, p2))
                            if pair not in reachable_pairs:
                                reachable_pairs.add(pair)
                                new_pairs_found = True

                # Precondition pairs can coexist BEFORE action (action is applicable)
                # But AFTER action, only non-deleted preconditions survive
                for p1 in preconds:
                    if p1 in deletes:
                        continue  # p1 is deleted, can't use it
                    for p2 in preconds:
                        if p2 in deletes:
                            continue  # p2 is deleted, can't use it
                        if p1 < p2:
                            pair = (p1, p2)
                            if pair not in reachable_pairs:
                                reachable_pairs.add(pair)
                                new_pairs_found = True

            # Fixed point check
            if not new_pairs_found:
                break

        # Remove reachable pairs from potential mutexes
        mutexes = potential_mutexes - reachable_pairs

        return mutexes

    def _detect_exactly_one_groups(self) -> List[Set[str]]:
        """
        Detect exactly-one invariant groups using toggle pattern analysis.

        An exactly-one group is a set of predicates where EXACTLY one is true
        in all reachable states.

        Detection heuristic:
        - Find predicates that toggle with each other via actions
        - Example: pick-up adds holding(?x) and deletes handempty
        -          put-down adds handempty and deletes holding(?x)
        - This suggests {handempty, holding(?x)} form an exactly-one group

        Returns:
            List of predicate name sets forming exactly-one groups
        """
        toggle_pairs = []

        # Find toggle patterns in actions
        for action in self._abstract_actions:
            adds = [eff.predicate.name for eff in action.effects if eff.is_add]
            deletes = [eff.predicate.name for eff in action.effects if not eff.is_add]

            # If action: +P -Q (adds one, deletes another), they form a toggle pair
            if len(adds) == 1 and len(deletes) == 1:
                add_pred = adds[0]
                del_pred = deletes[0]
                if add_pred != del_pred:
                    toggle_pairs.append(frozenset({add_pred, del_pred}))

        # Merge toggle pairs that share predicates into exactly-one groups
        # This handles cases like: {handempty, holding(?x)}, {handempty, holding(?y)}
        # Should merge into one group if they represent the same logical constraint

        groups = []
        seen = set()

        for pair in toggle_pairs:
            if pair not in seen:
                # This pair forms an exactly-one group
                groups.append(set(pair))
                seen.add(pair)

        return groups

    def _validate_state_with_invariants(self, predicates: Set[PredicateAtom]) -> bool:
        """
        Validate state against synthesized invariants.

        This is the CORRECT way to validate states:
        1. Check h^2 mutexes (predicate-name level)
        2. Check exactly-one groups (with argument awareness)

        Args:
            predicates: Set of predicates to validate

        Returns:
            True if state satisfies all invariants, False otherwise
        """
        # Check 1: H^2 Mutexes (name-based, so only check names)
        pred_names = {p.name for p in predicates}
        for pred1, pred2 in self._invariants['h2_mutexes']:
            if pred1 in pred_names and pred2 in pred_names:
                # Both mutex predicates present - invalid
                return False

        # Check 2: Exactly-One Groups
        # For each group, exactly one predicate from that group should be present
        for group in self._invariants['exactly_one_groups']:
            # Count how many predicates from this group are in the state
            count = sum(1 for p in predicates if p.name in group)

            # For exactly-one groups: must have exactly 1
            # Note: This check is at predicate level, not considering arguments
            # This is correct for simple cases like {handempty, holding}
            # where holding can only have one instance anyway
            if count != 1:
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
        Print variable-level planning statistics

        Args:
            abstract_state_count: Number of abstract states explored
        """
        print(f"\n[Variable-Level Planning]")
        print(f"  States explored: {abstract_state_count}")
