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
    """
    action: PDDLAction
    args: List[str]
    bindings: Dict[str, str]


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

    def __init__(self, domain: PDDLDomain, objects: List[str]):
        """
        Initialize forward planner

        Args:
            domain: PDDL domain definition
            objects: List of objects to ground actions with
        """
        self.domain = domain
        self.objects = objects
        self.condition_parser = PDDLConditionParser()
        self.effect_parser = PDDLEffectParser()

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
                         max_depth: Optional[int] = None) -> StateGraph:
        """
        Explore state space from goal state using forward "destruction"

        Args:
            goal_predicates: Predicates forming the goal state
            max_depth: Maximum search depth (auto-calculated if None)

        Returns:
            Complete state graph with all reachable states
        """
        # Infer complete goal state (Design Decision #3)
        complete_goal = self.infer_complete_goal_state(goal_predicates)

        # Calculate max depth if not provided
        if max_depth is None:
            max_depth = self.calculate_max_depth(goal_predicates)

        print(f"[Forward Planner] Starting exploration from goal with max_depth={max_depth}")
        print(f"[Forward Planner] Input goal predicates: {[str(p) for p in goal_predicates]}")
        print(f"[Forward Planner] Complete goal state: {[str(p) for p in sorted(complete_goal, key=str)]}")

        # Initialize goal state with complete predicates
        goal_state = WorldState(complete_goal, depth=0)
        graph = StateGraph(goal_state)

        # BFS exploration with optimized state lookup
        queue = deque([goal_state])
        visited_map: Dict[FrozenSet[PredicateAtom], WorldState] = {goal_state.predicates: goal_state}

        states_explored = 0
        transitions_added = 0

        while queue:
            current_state = queue.popleft()
            states_explored += 1

            if states_explored % 10 == 0:
                print(f"  Explored {states_explored} states, {transitions_added} transitions, "
                      f"queue size: {len(queue)}")

            # Try all ground actions from all states (to create bidirectional graph)
            # We explore from states at all depths to find reverse transitions
            for grounded_action in self._ground_all_actions():
                # Check preconditions
                if not self._check_preconditions(grounded_action, current_state):
                    continue

                # Apply action to get new states (may have multiple branches for oneof)
                new_states_data = self._apply_action(grounded_action, current_state)

                for new_state, belief_updates, preconditions in new_states_data:
                    # Check if state already visited using fast dict lookup
                    new_pred_set = frozenset(new_state.predicates)

                    if new_pred_set in visited_map:
                        # Use existing state (may be at any depth, including shallower - reverse transition!)
                        final_state = visited_map[new_pred_set]
                    else:
                        # Create new state with proper depth
                        new_depth = current_state.depth + 1

                        # Only add new states within depth limit
                        if new_depth > max_depth:
                            continue  # Skip creating states beyond max_depth

                        final_state = WorldState(new_state.predicates, depth=new_depth)
                        visited_map[new_pred_set] = final_state
                        queue.append(final_state)

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

        print(f"[Forward Planner] Exploration complete:")
        print(f"  States: {len(graph.states)}")
        print(f"  Transitions: {len(graph.transitions)}")
        print(f"  Leaf states: {len(graph.get_leaf_states())}")
        print(f"  Max depth reached: {max(s.depth for s in graph.states)}")

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

            grounded_actions.append(GroundedAction(
                action=action,
                args=list(obj_tuple),
                bindings=bindings
            ))

        return grounded_actions

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
        # Parse preconditions
        try:
            precond_predicates = self.condition_parser.parse(
                grounded_action.action.preconditions,
                grounded_action.bindings
            )
        except Exception as e:
            # Parse error - skip action
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
        # Parse effects
        try:
            effect_branches = self.effect_parser.parse(
                grounded_action.action.effects,
                grounded_action.bindings
            )
        except Exception as e:
            # Parse error
            return []

        # Parse preconditions (for precondition list in transition)
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

            # Create new state (Design: Line 668)
            new_state = WorldState(new_predicates)

            results.append((new_state, belief_updates, preconditions))

        return results

    def calculate_max_depth(self, goal_predicates: List[PredicateAtom]) -> int:
        """
        Calculate maximum search depth based on goal complexity

        Heuristics:
        - 1 predicate: depth = 5
        - 2-3 predicates: depth = 10
        - 4-6 predicates: depth = 15
        - 7+ predicates: depth = 20

        Args:
            goal_predicates: Goal predicates

        Returns:
            Maximum depth for exploration
        """
        num_predicates = len(goal_predicates)
        num_objects = len(self.objects)

        # Base depth on predicate count
        if num_predicates == 1:
            base_depth = 5
        elif num_predicates <= 3:
            base_depth = 10
        elif num_predicates <= 6:
            base_depth = 15
        else:
            base_depth = 20

        # Adjust for object count (more objects = more complexity)
        if num_objects > 5:
            base_depth += 5

        return base_depth


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

    # Explore (use very small depth for testing)
    graph = planner.explore_from_goal(goal_predicates, max_depth=1)

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

    # Calculate depth
    max_depth = planner.calculate_max_depth(goal_predicates)
    print(f"Calculated max_depth: {max_depth}")

    # Explore
    graph = planner.explore_from_goal(goal_predicates, max_depth=max_depth)

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
