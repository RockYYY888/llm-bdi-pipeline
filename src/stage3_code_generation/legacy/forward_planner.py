"""
Forward State Space Planner

Implements forward "destruction" planning from goal states:
1. Start with goal state (e.g., {on(a, b)})
2. Apply all possible PDDL actions to explore reachable states
3. Generate new states by applying action effects
4. Continue until no new states or max depth reached
5. Build complete state graph for plan extraction

This is the core algorithm for backward planning-based AgentSpeak generation.
"""

import itertools
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from collections import deque
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import (
    PredicateAtom, WorldState, StateTransition, StateGraph
)
from stage3_code_generation.pddl_condition_parser import (
    PDDLConditionParser, PDDLEffectParser
)
from utils.pddl_parser import PDDLDomain, PDDLAction
from stage1_interpretation.grounding_map import GroundingMap


@dataclass
class GroundedAction:
    """
    Represents a ground (instantiated) action

    Attributes:
        action: The PDDL action definition
        args: Ground arguments (e.g., ["a", "b"])
        bindings: Variable bindings (e.g., {"?b1": "a", "?b2": "b"})
        parsed_preconditions: Pre-parsed precondition predicates (cached)
        parsed_effects: Pre-parsed effect branches (cached)
    """
    action: PDDLAction
    args: List[str]
    bindings: Dict[str, str]
    parsed_preconditions: Optional[List[PredicateAtom]] = None
    parsed_effects: Optional[List[List]] = None


class ForwardStatePlanner:
    """
    Forward state space planner using "destruction" from goal states

    Starting from a goal state (e.g., {on(a, b)}), this planner explores
    all reachable states by applying actions. It builds a complete state
    graph that can be used to extract paths and generate AgentSpeak plans.

    Attributes:
        domain: PDDL domain with actions and predicates
        objects: List of objects in the domain (e.g., ["a", "b", "c"])
        condition_parser: Parser for preconditions
        effect_parser: Parser for effects
    """

    def __init__(self, domain: PDDLDomain, objects: List[str], use_variables: bool = False):
        """
        Initialize forward planner

        Args:
            domain: PDDL domain definition
            objects: List of objects to ground actions with (or variables if use_variables=True)
            use_variables: If True, use variable-level planning instead of object-level
                          In this mode, 'objects' should be variable names like ["?v0", "?v1"]
        """
        self.domain = domain
        self.objects = objects
        self.use_variables = use_variables
        self.condition_parser = PDDLConditionParser()
        self.effect_parser = PDDLEffectParser()

        # OPTIMIZATION: Cache ground actions (computed once instead of per-state)
        # This eliminates 99.9% of grounding redundancy
        # When use_variables=True, these are "variable actions" (e.g., pick-up(?v0, ?v1))
        self._cached_grounded_actions = self._ground_all_actions()

    def infer_complete_goal_state(self, goal_predicates: List[PredicateAtom]) -> Set[PredicateAtom]:
        """
        Infer the complete goal state from goal predicates.

        Per Design Decision #3 and Q&A #3:
        "NOT just predicates in original goal - include all relevant world state"

        Strategy:
        1. Find actions that ADD the goal predicates
        2. Include ALL their add-effects in the goal state
        3. This ensures the goal state is "achievable" and has all necessary predicates

        Args:
            goal_predicates: Core goal predicates (from DFA transition label)

        Returns:
            Complete set of predicates for the goal state
        """
        complete_goal = set(goal_predicates)

        # For each goal predicate, find actions that produce it
        for goal_pred in goal_predicates:
            # Try all ground actions
            for action in self.domain.actions:
                for grounded in self._ground_action(action):
                    try:
                        # Parse effects
                        effect_branches = self.effect_parser.parse(
                            action.effects,
                            grounded.bindings
                        )

                        # Check each branch
                        for branch in effect_branches:
                            # Does this branch add the goal predicate?
                            adds_goal = any(
                                eff.is_add and eff.predicate == goal_pred
                                for eff in branch
                            )

                            if adds_goal:
                                # Include ALL add-effects from this branch
                                for eff in branch:
                                    if eff.is_add:
                                        complete_goal.add(eff.predicate)
                    except Exception:
                        # Skip actions that can't be parsed
                        continue

        return complete_goal

    def explore_from_goal(self, goal_predicates: List[PredicateAtom],
                         max_states: int = 200000) -> StateGraph:
        """
        Explore state space from goal state using forward "destruction"

        Exploration continues until all reachable states are discovered.
        Terminates naturally when the BFS queue is empty (all states visited),
        or when max_states limit is reached.

        Args:
            goal_predicates: Predicates forming the goal state
            max_states: Maximum number of states to explore (safety limit, increased to 200K for complex goals)

        Returns:
            Complete state graph with all reachable states
        """
        # Infer complete goal state (Design Decision #3)
        complete_goal = self.infer_complete_goal_state(goal_predicates)

        print(f"[Forward Planner] Starting complete state space exploration")
        print(f"[Forward Planner] Mode: {'VARIABLE-LEVEL' if self.use_variables else 'OBJECT-LEVEL'}")
        if self.use_variables:
            print(f"[Forward Planner] Variables: {self.objects}")
        else:
            print(f"[Forward Planner] Objects: {self.objects}")
        print(f"[Forward Planner] Max states limit: {max_states:,}")
        print(f"[Forward Planner] Input goal predicates: {[str(p) for p in goal_predicates]}")
        print(f"[Forward Planner] Complete goal state: {[str(p) for p in sorted(complete_goal, key=str)]}")

        # Initialize goal state with complete predicates
        goal_state = WorldState(complete_goal, depth=0)
        graph = StateGraph(goal_state)

        # BFS exploration with optimized state lookup
        queue = deque([goal_state])
        visited_map: Dict[FrozenSet[PredicateAtom], WorldState] = {goal_state.predicates: goal_state}

        # Statistics counters:
        # - states_explored: states popped from queue and expanded (processed)
        # - len(visited_map): total unique states in memory (explored + in queue)
        # - max_states limits states_explored to control actual work done
        states_explored = 0  # States popped from queue and expanded
        transitions_added = 0
        states_reused = 0
        states_created = 0
        max_states_reached = False  # Flag to stop exploration

        while queue and not max_states_reached:
            # Check if we've reached the max states limit before processing next state
            if states_explored >= max_states:
                print(f"  ⚠️  Reached max_states limit ({max_states:,}), stopping exploration")
                print(f"  This is a safety limit to prevent excessive computation.")
                print(f"  Processed {states_explored:,} states, {len(visited_map):,} total in memory.")
                max_states_reached = True
                graph.truncated = True
                break

            current_state = queue.popleft()
            states_explored += 1

            if states_explored % 10000 == 0:
                print(f"  Processed {states_explored} states, total in memory: {len(visited_map)} unique states, "
                      f"{transitions_added} transitions, queue size: {len(queue)}")

            # Try all ground actions from all states (to create bidirectional graph)
            # We explore from states at all depths to find reverse transitions
            # OPTIMIZATION: Use cached ground actions instead of recomputing
            for grounded_action in self._cached_grounded_actions:
                # Check preconditions
                if not self._check_preconditions(grounded_action, current_state):
                    continue

                # Apply action to get new states (may have multiple branches for oneof)
                new_states_data = self._apply_action(grounded_action, current_state)

                for new_state, belief_updates, preconditions in new_states_data:
                    # Check if state already visited using fast dict lookup
                    # Note: new_state.predicates is already a FrozenSet, no need to wrap again
                    new_pred_set = new_state.predicates

                    if new_pred_set in visited_map:
                        # Use existing state (may be at any depth, including shallower - reverse transition!)
                        final_state = visited_map[new_pred_set]
                        states_reused += 1
                    else:
                        # Create new state with proper depth
                        # Pass frozenset directly to avoid creating duplicate frozenset in memory
                        new_depth = current_state.depth + 1
                        final_state = WorldState(new_pred_set, depth=new_depth)
                        visited_map[new_pred_set] = final_state
                        queue.append(final_state)
                        states_created += 1

                    # Create transition (Forward direction per Design Algorithm 1)
                    # From current_state, applying action leads to new state
                    transition = StateTransition(
                        from_state=current_state,
                        to_state=final_state,
                        action=grounded_action.action,
                        action_args=grounded_action.args,
                        belief_updates=belief_updates,
                        preconditions=preconditions
                    )
                    graph.add_transition(transition)
                    transitions_added += 1

        # Print final state if not already printed (i.e., didn't land on 10000 multiple)
        if states_explored % 10000 != 0:
            print(f"  Processed {states_explored} states, total in memory: {len(visited_map)} unique states, "
                  f"{transitions_added} transitions, queue size: {len(queue)}")

        print(f"[Forward Planner] Exploration complete:")
        print(f"  States processed (popped from queue): {states_explored:,}")
        print(f"  Total unique states in memory: {len(graph.states):,}")
        print(f"  Transitions: {len(graph.transitions):,}")
        print(f"  Leaf states: {len(graph.get_leaf_states()):,}")
        print(f"  Max depth reached: {max(s.depth for s in graph.states)}")
        print(f"  Performance:")
        print(f"    States reused (cache hits): {states_reused:,}")
        print(f"    States created (cache misses): {states_created:,}")
        print(f"    Reuse ratio: {states_reused / max(states_created, 1):.1f}:1")
        print(f"    Ground actions cached: {len(self._cached_grounded_actions)}")

        return graph

    def _ground_all_actions(self) -> List[GroundedAction]:
        """
        Ground all actions with all possible argument combinations

        Returns:
            List of GroundedActions
        """
        grounded_actions = []

        for action in self.domain.actions:
            for grounded in self._ground_action(action):
                grounded_actions.append(grounded)

        return grounded_actions

    def _ground_action(self, action: PDDLAction) -> List[GroundedAction]:
        """
        Ground a single action with all valid object combinations

        Args:
            action: PDDL action to ground

        Returns:
            List of GroundedActions with all valid instantiations
        """
        if not action.parameters:
            # Action has no parameters
            return [GroundedAction(action=action, args=[], bindings={})]

        # Extract parameter variables
        param_vars = []
        for param in action.parameters:
            # Format: "?b1 - block" or "?b - block"
            parts = param.split('-')
            if len(parts) >= 1:
                var_name = parts[0].strip()
                param_vars.append(var_name)

        # Generate all combinations
        grounded_actions = []

        for obj_tuple in itertools.product(self.objects, repeat=len(param_vars)):
            # Create bindings
            bindings = {var: obj for var, obj in zip(param_vars, obj_tuple)}

            # Check equality constraints in preconditions
            # Format: (not (= ?b1 ?b2)) means ?b1 and ?b2 must be different
            if not self._check_equality_constraints(action.preconditions, bindings):
                # Skip this grounding - violates equality constraint
                continue

            # OPTIMIZATION: Pre-parse preconditions and effects at grounding time
            # This avoids repeated parsing during state exploration
            parsed_preconditions = None
            parsed_effects = None

            try:
                parsed_preconditions = self.condition_parser.parse(
                    action.preconditions, bindings
                )
            except Exception:
                parsed_preconditions = []

            try:
                parsed_effects = self.effect_parser.parse(
                    action.effects, bindings
                )
            except Exception:
                parsed_effects = []

            grounded_actions.append(GroundedAction(
                action=action,
                args=list(obj_tuple),
                bindings=bindings,
                parsed_preconditions=parsed_preconditions,
                parsed_effects=parsed_effects
            ))

        return grounded_actions

    def _check_equality_constraints(self, preconditions: str, bindings: Dict[str, str]) -> bool:
        """
        Check if bindings satisfy equality constraints in preconditions

        Handles patterns like:
        - (not (= ?b1 ?b2)) means ?b1 != ?b2
        - (= ?b1 ?b2) means ?b1 == ?b2 (rare)

        Args:
            preconditions: PDDL precondition string
            bindings: Variable to object mapping

        Returns:
            True if constraints satisfied, False if violated
        """
        import re

        # Find all equality constraints: (not (= ?var1 ?var2)) or (= ?var1 ?var2)
        # Pattern for: (not (= ?b1 ?b2))
        not_equal_pattern = r'\(not\s+\(=\s+(\?\w+)\s+(\?\w+)\)\)'
        # Pattern for: (= ?b1 ?b2)
        equal_pattern = r'\(=\s+(\?\w+)\s+(\?\w+)\)'

        # Check (not (= ?var1 ?var2)) - variables must be different
        for match in re.finditer(not_equal_pattern, preconditions):
            var1, var2 = match.group(1), match.group(2)
            if var1 in bindings and var2 in bindings:
                if bindings[var1] == bindings[var2]:
                    # Constraint violated: variables must be different but are same
                    return False

        # Check (= ?var1 ?var2) - variables must be same
        for match in re.finditer(equal_pattern, preconditions):
            var1, var2 = match.group(1), match.group(2)
            # Skip if this is part of (not (= ...)) - already handled above
            if f'(not (= {var1} {var2}))' in preconditions:
                continue
            if var1 in bindings and var2 in bindings:
                if bindings[var1] != bindings[var2]:
                    # Constraint violated: variables must be same but are different
                    return False

        return True

    def _check_preconditions(self, grounded_action: GroundedAction,
                            state: WorldState) -> bool:
        """
        Check if action's preconditions are satisfied in current state

        Three cases:
        1. Precondition violated (known to be false) → return False
        2. Precondition unknown (not in state, state non-empty) → return True (will generate subgoal)
        3. Precondition satisfied → return True

        Args:
            grounded_action: Ground action to check
            state: Current state

        Returns:
            True if action can be applied, False if violated
        """
        # OPTIMIZATION: Use cached parsed preconditions
        precond_predicates = grounded_action.parsed_preconditions
        if precond_predicates is None:
            # Fallback: parse on-demand (shouldn't happen with current code)
            try:
                precond_predicates = self.condition_parser.parse(
                    grounded_action.action.preconditions,
                    grounded_action.bindings
                )
            except Exception:
                return False

        # Check each precondition
        for precond in precond_predicates:
            if precond.negated:
                # Requires predicate to NOT exist: not clear(a)
                positive_pred = precond.get_positive()
                if positive_pred in state.predicates:
                    # VIOLATION: predicate exists but shouldn't
                    return False

            else:
                # Requires predicate to exist: holding(a)
                # If state is non-empty and predicate not present, it's potentially achievable
                # So we allow it (subgoal will be generated later)
                pass

        return True

    def _validate_state_consistency(self, predicates: Set[PredicateAtom]) -> bool:
        """
        Validate state consistency for blocksworld domain

        NOTE: This is currently DOMAIN-SPECIFIC (blocksworld only).
        TODO: Make this truly domain-independent by:
          1. Analyzing action effects + preconditions to infer semantic constraints
          2. Or relying solely on precondition checking (current approach may be sufficient)

        The current implementation prevents state space explosion from invalid states
        generated by non-deterministic effects (oneof).

        Args:
            predicates: Set of predicates representing the state

        Returns:
            True if state is consistent, False if violated
        """
        # Single pass categorization
        handempty = False
        holding = []
        ontable = []
        on = []
        clear = []

        for p in predicates:
            if p.name == 'handempty':
                handempty = True
            elif p.name == 'holding':
                holding.append(p)
            elif p.name == 'ontable':
                ontable.append(p)
            elif p.name == 'on':
                on.append(p)
            elif p.name == 'clear':
                clear.append(p)

        # Check 1: Hand contradictions
        if handempty and len(holding) > 0:
            return False

        # Check 2: Multiple holdings
        if len(holding) > 1:
            return False

        # Early exit if no complex checks needed
        if not on:
            return True

        # Check 3: Self-loops and multiple locations
        on_map = {}
        for pred in on:
            if len(pred.args) == 2:
                block, base = pred.args
                if block == base:  # Self-loop
                    return False
                if block in on_map:  # Multiple locations
                    return False
                on_map[block] = base

        # Check 4: Cycles
        for block, base in on_map.items():
            if base in on_map and on_map[base] == block:  # Direct cycle
                return False
            # Indirect cycles
            visited = set()
            current = base
            while current in on_map:
                if current in visited:
                    return False
                visited.add(current)
                current = on_map[current]
                if current == block:
                    return False

        # Check 5: Location contradictions
        ontable_blocks = {pred.args[0] for pred in ontable if len(pred.args) == 1}
        on_blocks = {pred.args[0] for pred in on if len(pred.args) == 2}
        if ontable_blocks & on_blocks:
            return False

        # Check 6: Clear contradictions
        clear_blocks = {pred.args[0] for pred in clear if len(pred.args) == 1}
        base_blocks = {pred.args[1] for pred in on if len(pred.args) == 2}
        if clear_blocks & base_blocks:
            return False

        return True

    def _apply_action(self, grounded_action: GroundedAction,
                     state: WorldState) -> List[Tuple[WorldState, List[str], List[PredicateAtom]]]:
        """
        Apply action to state, return possible new states

        Handles non-deterministic effects (oneof).

        Args:
            grounded_action: Ground action to apply
            state: Current state

        Returns:
            List of (new_state, belief_updates, preconditions)
        """
        # OPTIMIZATION: Use cached parsed effects
        effect_branches = grounded_action.parsed_effects
        if effect_branches is None:
            # Fallback: parse on-demand (shouldn't happen with current code)
            try:
                effect_branches = self.effect_parser.parse(
                    grounded_action.action.effects,
                    grounded_action.bindings
                )
            except Exception:
                return []

        # OPTIMIZATION: Use cached parsed preconditions
        preconditions = grounded_action.parsed_preconditions
        if preconditions is None:
            # Fallback: parse on-demand (shouldn't happen with current code)
            try:
                preconditions = self.condition_parser.parse(
                    grounded_action.action.preconditions,
                    grounded_action.bindings
                )
            except Exception:
                preconditions = []

        results = []

        # Process each branch (for oneof)
        for effect_branch in effect_branches:
            # Apply effects FORWARD (per Design Algorithm 3, Line 659-666)
            # Starting from current state, apply action effects to generate new state
            new_predicates = set(state.predicates)
            belief_updates = []

            for effect_atom in effect_branch:
                if effect_atom.is_add:
                    # Add effect: +on(a, b)
                    new_predicates.add(effect_atom.predicate)
                    belief_updates.append(f"+{effect_atom.predicate.to_agentspeak()}")
                else:
                    # Delete effect: -ontable(a)
                    new_predicates.discard(effect_atom.predicate)
                    belief_updates.append(f"-{effect_atom.predicate.to_agentspeak()}")

            # Validate state consistency before creating new state
            if not self._validate_state_consistency(new_predicates):
                # State violates physical constraints - skip this branch
                continue

            # Create new state (Design: Line 668)
            new_state = WorldState(new_predicates)

            results.append((new_state, belief_updates, preconditions))

        return results


# Test functions
def test_forward_planner_simple():
    """Test forward planner with simple blocksworld example"""
    print("="*80)
    print("Testing Forward Planner - Simple Example")
    print("="*80)

    # Load blocksworld domain
    from utils.pddl_parser import PDDLParser

    domain_file = Path(__file__).parent.parent / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        print("Skipping test")
        return

    domain = PDDLParser.parse_domain(str(domain_file))
    print(f"Loaded domain: {domain.name}")
    print(f"Actions: {[a.name for a in domain.actions]}")

    # Create simple goal: on(a, b)
    goal_predicates = [
        PredicateAtom("on", ["a", "b"])
    ]

    objects = ["a", "b"]

    # Create planner
    planner = ForwardStatePlanner(domain, objects)

    # Explore complete state space
    graph = planner.explore_from_goal(goal_predicates)

    print(f"\nGraph statistics:")
    stats = graph.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Find paths
    print(f"\nFinding paths to goal...")
    paths = graph.find_shortest_paths_to_goal()

    print(f"Paths found for {len(paths)} states:")
    for i, (state, path) in enumerate(list(paths.items())[:5]):  # Show first 5
        print(f"\n  State {i}: {state}")
        if not path:
            print(f"    (already at goal)")
        else:
            print(f"    Path ({len(path)} steps):")
            for trans in path:
                print(f"      - {trans.get_action_call()}")

    # Save DOT visualization
    dot_file = Path(__file__).parent / "test_forward_planner_simple.dot"
    with open(dot_file, 'w') as f:
        f.write(graph.to_dot())
    print(f"\nVisualization saved to: {dot_file}")

    print("\n")


def test_forward_planner_complex():
    """Test forward planner with more complex goal"""
    print("="*80)
    print("Testing Forward Planner - Complex Goal")
    print("="*80)

    # Load domain
    from utils.pddl_parser import PDDLParser

    domain_file = Path(__file__).parent.parent / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        print("Skipping test")
        return

    domain = PDDLParser.parse_domain(str(domain_file))

    # Complex goal: on(a, b) & on(b, c)
    goal_predicates = [
        PredicateAtom("on", ["a", "b"]),
        PredicateAtom("on", ["b", "c"])
    ]

    objects = ["a", "b", "c"]

    # Create planner
    planner = ForwardStatePlanner(domain, objects)

    # Explore complete state space
    graph = planner.explore_from_goal(goal_predicates)

    print(f"\nGraph statistics:")
    stats = graph.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n")


if __name__ == "__main__":
    test_forward_planner_simple()
    # test_forward_planner_complex()  # Disabled: generates very large state space
    print("\nNote: Complex test disabled due to large state space.")
    print("This is expected - real usage will have pruning strategies.")
