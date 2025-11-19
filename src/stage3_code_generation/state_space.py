"""
State Space Data Structures for Backward Planning

This module defines the core data structures for forward state-space exploration:
- PredicateAtom: A ground predicate (e.g., on(a, b))
- WorldState: A set of predicates representing a world state
- StateTransition: An edge in the state graph (action application)
- StateGraph: Complete state space graph with BFS path finding

These structures are used by the ForwardStatePlanner to explore from goal
states and generate AgentSpeak plans.
"""

from dataclasses import dataclass, field
from typing import Set, List, Dict, FrozenSet, Optional, Tuple
from collections import deque
import sys
from pathlib import Path

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from utils.pddl_parser import PDDLAction


@dataclass(frozen=True)
class PredicateAtom:
    """
    Represents a ground predicate atom

    Examples:
        on(a, b) → PredicateAtom(name="on", args=("a", "b"), negated=False)
        ~clear(c) → PredicateAtom(name="clear", args=("c",), negated=True)
        handempty → PredicateAtom(name="handempty", args=(), negated=False)

    Attributes:
        name: Predicate name (e.g., "on", "clear")
        args: Tuple of ground arguments (e.g., ("a", "b"))
        negated: Whether this is a negated predicate
    """
    name: str
    args: Tuple[str, ...]
    negated: bool = False

    def __init__(self, name: str, args: List[str], negated: bool = False):
        """
        Initialize PredicateAtom

        Note: Using object.__setattr__ because dataclass is frozen
        """
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'args', tuple(args) if args else ())
        object.__setattr__(self, 'negated', negated)

    def to_agentspeak(self, convert_vars: bool = False, obj_to_var: dict = None) -> str:
        """
        Convert to AgentSpeak format

        Args:
            convert_vars: If True, convert PDDL variables (?v0) to AgentSpeak variables (V0)
            obj_to_var: Optional mapping from objects to variables for parameterization
                       Example: {"a": "?v0", "b": "?v1"} will convert on(a,b) → on(V0, V1)

        Returns:
            String like "on(a, b)" or "~clear(c)" or "on(V0, V1)" if parameterized

        Note:
            Uses strong negation (~) instead of weak negation (not) because:
            - ~ontable(c): "c is definitely not on the table" (achievable state)
            - not ontable(c): "lack knowledge about ontable(c)" (not an achievable goal)
            Strong negation corresponds to PDDL effects like (not (ontable c))
        """
        prefix = "~" if self.negated else ""
        if self.args:
            if obj_to_var:
                # Object-level parameterization: a → ?v0 → V0
                args_str = ", ".join(self._obj_to_agentspeak_var(arg, obj_to_var) for arg in self.args)
            elif convert_vars:
                # Variable-level conversion: ?v0 → V0
                args_str = ", ".join(self._pddl_var_to_agentspeak(arg) for arg in self.args)
            else:
                # Keep as-is
                args_str = ", ".join(self.args)
            return f"{prefix}{self.name}({args_str})"
        return f"{prefix}{self.name}"

    def _obj_to_agentspeak_var(self, arg: str, obj_to_var: dict) -> str:
        """
        Convert object to AgentSpeak variable using mapping

        Examples:
            a with {a: ?v0} → V0
            b with {b: ?v1} → V1
            c (not in mapping) → c (unchanged)

        Args:
            arg: Object or constant string
            obj_to_var: Mapping from objects to PDDL variables

        Returns:
            AgentSpeak format variable or constant
        """
        if arg in obj_to_var:
            # Map object to PDDL variable, then convert to AgentSpeak
            pddl_var = obj_to_var[arg]
            return self._pddl_var_to_agentspeak(pddl_var)
        return arg  # Not in mapping, keep as-is (constant/literal)

    def _pddl_var_to_agentspeak(self, arg: str) -> str:
        """
        Convert PDDL variable to AgentSpeak variable

        Examples:
            ?v0 → V0
            ?v1 → V1
            ?x → X
            a → a (constants/objects unchanged)

        Args:
            arg: PDDL argument string

        Returns:
            AgentSpeak format argument
        """
        if arg.startswith('?'):
            # Remove ? and capitalize
            var_name = arg[1:]
            # Capitalize first letter, keep rest as-is
            if var_name:
                return var_name[0].upper() + var_name[1:]
            return var_name
        return arg  # Not a variable, return as-is

    def to_pddl(self) -> str:
        """
        Convert to PDDL format

        Returns:
            String like "(on a b)" or "(not (clear c))"
        """
        if self.args:
            args_str = " ".join(self.args)
            pred_str = f"({self.name} {args_str})"
        else:
            pred_str = f"({self.name})"

        if self.negated:
            return f"(not {pred_str})"
        return pred_str

    def get_positive(self) -> 'PredicateAtom':
        """Get the positive version of this predicate"""
        if not self.negated:
            return self
        return PredicateAtom(self.name, list(self.args), negated=False)

    def get_negated(self) -> 'PredicateAtom':
        """Get the negated version of this predicate"""
        return PredicateAtom(self.name, list(self.args), negated=not self.negated)

    def __str__(self) -> str:
        return self.to_agentspeak()

    def __repr__(self) -> str:
        neg_str = "~" if self.negated else ""
        if self.args:
            return f"{neg_str}{self.name}({', '.join(self.args)})"
        return f"{neg_str}{self.name}"

    def __hash__(self) -> int:
        """Hash based on name, args, and negation"""
        return hash((self.name, self.args, self.negated))

    def __eq__(self, other) -> bool:
        """Equality based on name, args, and negation"""
        if not isinstance(other, PredicateAtom):
            return False
        return (self.name == other.name and
                self.args == other.args and
                self.negated == other.negated)

    def is_variable_arg(self, arg: str) -> bool:
        """
        Check if an argument is a variable (starts with ?)

        Args:
            arg: Argument string

        Returns:
            True if argument is a variable
        """
        return arg.startswith('?')

    def is_grounded(self) -> bool:
        """
        Check if predicate is fully grounded (no variables)

        Returns:
            True if all arguments are grounded (not variables)
        """
        return all(not self.is_variable_arg(arg) for arg in self.args)

    def is_variable_predicate(self) -> bool:
        """
        Check if predicate uses variables

        Returns:
            True if any argument is a variable
        """
        return any(self.is_variable_arg(arg) for arg in self.args)

    def get_variables(self) -> List[str]:
        """
        Get all variable arguments in this predicate

        Returns:
            List of variable names (e.g., ["?v0", "?v1"])
        """
        return [arg for arg in self.args if self.is_variable_arg(arg)]

    def instantiate(self, var_mapping: Dict[str, str]) -> 'PredicateAtom':
        """
        Instantiate variables in this predicate with concrete objects

        Example:
            on(?v0, ?v1) with {?v0: a, ?v1: b} → on(a, b)

        Args:
            var_mapping: Dictionary mapping variables to objects

        Returns:
            New PredicateAtom with variables replaced
        """
        new_args = [var_mapping.get(arg, arg) for arg in self.args]
        return PredicateAtom(self.name, new_args, self.negated)


@dataclass(frozen=True)
class WorldState:
    """
    Represents a world state as a set of predicates

    Example:
        State({on(a, b), clear(c), handempty})

    Attributes:
        predicates: Frozen set of PredicateAtoms
        depth: Distance from goal state (used during exploration)
    """
    predicates: FrozenSet[PredicateAtom]
    depth: int = 0

    def __init__(self, predicates, depth: int = 0):
        """
        Initialize WorldState

        Args:
            predicates: Either a Set[PredicateAtom] or FrozenSet[PredicateAtom]
                       If already a frozenset, reuse it to save memory
            depth: Distance from goal state
        """
        if isinstance(predicates, frozenset):
            # Reuse existing frozenset to avoid duplication
            object.__setattr__(self, 'predicates', predicates)
        else:
            # Convert set to frozenset
            object.__setattr__(self, 'predicates', frozenset(predicates))
        object.__setattr__(self, 'depth', depth)

    def contains(self, predicate: PredicateAtom) -> bool:
        """Check if predicate exists in this state"""
        return predicate in self.predicates

    def contains_any(self, predicates: List[PredicateAtom]) -> bool:
        """Check if any of the predicates exist in this state"""
        return any(p in self.predicates for p in predicates)

    def is_empty(self) -> bool:
        """Check if state has no predicates"""
        return len(self.predicates) == 0

    def to_agentspeak_context(self, convert_vars: bool = False, obj_to_var: dict = None) -> str:
        """
        Convert to AgentSpeak context condition

        Args:
            convert_vars: If True, convert PDDL variables to AgentSpeak variables
            obj_to_var: Optional mapping from objects to variables for parameterization

        Returns:
            String like "on(a, b) & clear(c) & handempty" or "true" if empty
            With convert_vars=True or obj_to_var: "on(V0, V1) & clear(V0)" etc.
        """
        if self.is_empty():
            return "true"

        # Sort for deterministic output
        sorted_preds = sorted(self.predicates, key=lambda p: (p.name, p.args))
        return " & ".join(p.to_agentspeak(convert_vars=convert_vars, obj_to_var=obj_to_var) for p in sorted_preds)

    def __str__(self) -> str:
        if self.is_empty():
            return "State(empty)"
        preds_str = ", ".join(str(p) for p in sorted(self.predicates, key=lambda p: (p.name, p.args)))
        return f"State({preds_str})"

    def __repr__(self) -> str:
        return f"WorldState(depth={self.depth}, n_predicates={len(self.predicates)})"

    def __hash__(self) -> int:
        """Hash based on predicates only (not depth)"""
        return hash(self.predicates)

    def __eq__(self, other) -> bool:
        """Equality based on predicates only (not depth)"""
        if not isinstance(other, WorldState):
            return False
        return self.predicates == other.predicates


@dataclass
class StateTransition:
    """
    Represents a transition (edge) in the state graph

    Example:
        State({holding(a), clear(b)}) --[putdown(a, b)]-> State({on(a, b), handempty})

    Attributes:
        from_state: Source state
        to_state: Destination state
        action: PDDL action being applied
        action_args: Ground arguments for the action
        belief_updates: List of belief updates (AgentSpeak format)
        preconditions: Preconditions of the action (for subgoal generation)
    """
    from_state: WorldState
    to_state: WorldState
    action: PDDLAction
    action_args: Tuple[str, ...]
    belief_updates: Tuple[str, ...]
    preconditions: Tuple[PredicateAtom, ...]

    def __init__(self, from_state: WorldState, to_state: WorldState,
                 action: PDDLAction, action_args: List[str],
                 belief_updates: List[str], preconditions: List[PredicateAtom]):
        """Initialize StateTransition"""
        self.from_state = from_state
        self.to_state = to_state
        self.action = action
        self.action_args = tuple(action_args)
        self.belief_updates = tuple(belief_updates)
        self.preconditions = tuple(preconditions)

    def get_action_call(self) -> str:
        """
        Get AgentSpeak action call

        Returns:
            String like "pick_up(a, b)" or "put_down(c)"
        """
        # Convert PDDL action name to AgentSpeak (replace - with _)
        action_name = self.action.name.replace('-', '_')

        if self.action_args:
            args_str = ", ".join(self.action_args)
            return f"{action_name}({args_str})"
        return f"{action_name}"

    def __str__(self) -> str:
        return f"{self.from_state} --[{self.get_action_call()}]-> {self.to_state}"

    def __repr__(self) -> str:
        return f"StateTransition({self.get_action_call()})"


class StateGraph:
    """
    State space graph generated by forward exploration

    This graph represents all reachable states from a goal state, along with
    the actions that connect them. It's used to extract paths and generate
    AgentSpeak plans.

    Attributes:
        goal_state: The goal state (root of exploration)
        states: All states in the graph
        transitions: All transitions (edges)
        state_to_outgoing: Adjacency list (state -> outgoing transitions)
        state_to_incoming: Reverse adjacency list (state -> incoming transitions)
        truncated: Whether exploration was stopped due to max_states limit
    """

    def __init__(self, goal_state: WorldState):
        """
        Initialize state graph with goal state

        Args:
            goal_state: The goal state (starting point of exploration)
        """
        self.goal_state = goal_state
        self.states: Set[WorldState] = {goal_state}
        self.transitions: List[StateTransition] = []
        self.state_to_outgoing: Dict[WorldState, List[StateTransition]] = {goal_state: []}
        self.state_to_incoming: Dict[WorldState, List[StateTransition]] = {goal_state: []}
        self.truncated: bool = False  # Set to True if max_states limit reached

    def add_transition(self, transition: StateTransition) -> None:
        """
        Add a transition to the graph

        Args:
            transition: StateTransition to add
        """
        # Add states
        self.states.add(transition.from_state)
        self.states.add(transition.to_state)

        # Add transition
        self.transitions.append(transition)

        # Update adjacency lists
        if transition.from_state not in self.state_to_outgoing:
            self.state_to_outgoing[transition.from_state] = []
        self.state_to_outgoing[transition.from_state].append(transition)

        if transition.to_state not in self.state_to_incoming:
            self.state_to_incoming[transition.to_state] = []
        self.state_to_incoming[transition.to_state].append(transition)

    def get_outgoing_transitions(self, state: WorldState) -> List[StateTransition]:
        """Get all transitions leaving from a state"""
        return self.state_to_outgoing.get(state, [])

    def get_incoming_transitions(self, state: WorldState) -> List[StateTransition]:
        """Get all transitions entering a state"""
        return self.state_to_incoming.get(state, [])

    def get_leaf_states(self) -> Set[WorldState]:
        """
        Get all leaf states (no outgoing transitions)

        Returns:
            Set of states with no outgoing transitions
        """
        return {s for s in self.states if s not in self.state_to_outgoing or
                len(self.state_to_outgoing[s]) == 0}

    def find_shortest_paths_to_goal(self) -> Dict[WorldState, List[StateTransition]]:
        """
        Find shortest path from each state to goal using BFS

        Returns:
            Dict mapping each state to list of transitions forming shortest path to goal
        """
        paths: Dict[WorldState, List[StateTransition]] = {}

        # BFS from goal state (backward direction)
        queue: deque[Tuple[WorldState, List[StateTransition]]] = deque([(self.goal_state, [])])
        visited: Set[WorldState] = {self.goal_state}

        paths[self.goal_state] = []  # Goal has empty path to itself

        while queue:
            current_state, path_to_current = queue.popleft()

            # Explore incoming transitions (states that can reach current_state)
            for transition in self.get_incoming_transitions(current_state):
                from_state = transition.from_state

                if from_state not in visited:
                    visited.add(from_state)
                    # Path from from_state to goal = [transition] + path_to_current
                    path_to_goal = [transition] + path_to_current
                    paths[from_state] = path_to_goal
                    queue.append((from_state, path_to_goal))

        return paths

    def to_dot(self, highlight_states: Optional[Set[WorldState]] = None) -> str:
        """
        Generate DOT format for visualization

        Args:
            highlight_states: Optional set of states to highlight

        Returns:
            DOT format string
        """
        lines = ["digraph StateGraph {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=circle];")
        lines.append("")

        # Track state IDs
        state_ids: Dict[WorldState, int] = {}
        for i, state in enumerate(sorted(self.states, key=lambda s: s.depth)):
            state_ids[state] = i

        # Add goal state with special styling
        goal_id = state_ids[self.goal_state]
        goal_label = self._format_state_label(self.goal_state)
        lines.append(f'  s{goal_id} [label="{goal_label}", shape=doublecircle, color=green, style=bold];')

        # Add other states
        for state, state_id in state_ids.items():
            if state == self.goal_state:
                continue

            label = self._format_state_label(state)

            # Highlight if requested
            if highlight_states and state in highlight_states:
                lines.append(f'  s{state_id} [label="{label}", color=blue, style=bold];')
            else:
                lines.append(f'  s{state_id} [label="{label}"];')

        lines.append("")

        # Add transitions
        for trans in self.transitions:
            from_id = state_ids[trans.from_state]
            to_id = state_ids[trans.to_state]
            action_label = trans.get_action_call().replace('"', '\\"')
            lines.append(f'  s{from_id} -> s{to_id} [label="{action_label}"];')

        lines.append("}")
        return "\n".join(lines)

    def _format_state_label(self, state: WorldState) -> str:
        """Format state for DOT label"""
        if state.is_empty():
            return "empty\\n(depth={})".format(state.depth)

        # Sort predicates for readability
        sorted_preds = sorted(state.predicates, key=lambda p: (p.name, p.args))

        # Limit to first 5 predicates
        if len(sorted_preds) > 5:
            pred_strs = [str(p) for p in sorted_preds[:5]]
            pred_strs.append(f"... ({len(sorted_preds) - 5} more)")
        else:
            pred_strs = [str(p) for p in sorted_preds]

        label = "\\n".join(pred_strs)
        label += f"\\n(depth={state.depth})"
        return label

    def get_statistics(self) -> Dict[str, int]:
        """Get graph statistics"""
        return {
            "num_states": len(self.states),
            "num_transitions": len(self.transitions),
            "num_leaf_states": len(self.get_leaf_states()),
            "max_depth": max((s.depth for s in self.states), default=0),
            "goal_depth": self.goal_state.depth
        }

    def __str__(self) -> str:
        stats = self.get_statistics()
        return (f"StateGraph(states={stats['num_states']}, "
                f"transitions={stats['num_transitions']}, "
                f"max_depth={stats['max_depth']})")

    def __repr__(self) -> str:
        return self.__str__()


# Test functions
def test_predicate_atom():
    """Test PredicateAtom"""
    print("="*80)
    print("Testing PredicateAtom")
    print("="*80)

    # Positive predicate
    p1 = PredicateAtom("on", ["a", "b"])
    print(f"p1: {p1}")
    print(f"  AgentSpeak: {p1.to_agentspeak()}")
    print(f"  PDDL: {p1.to_pddl()}")

    # Negated predicate
    p2 = PredicateAtom("clear", ["c"], negated=True)
    print(f"\np2: {p2}")
    print(f"  AgentSpeak: {p2.to_agentspeak()}")
    print(f"  PDDL: {p2.to_pddl()}")

    # Zero-arity predicate
    p3 = PredicateAtom("handempty", [])
    print(f"\np3: {p3}")
    print(f"  AgentSpeak: {p3.to_agentspeak()}")
    print(f"  PDDL: {p3.to_pddl()}")

    # Test equality
    p4 = PredicateAtom("on", ["a", "b"])
    print(f"\np1 == p4: {p1 == p4}")
    print(f"p1 == p2: {p1 == p2}")

    # Test hash (for set membership)
    pred_set = {p1, p2, p3, p4}
    print(f"\nSet size (should be 3): {len(pred_set)}")

    print("\n")


def test_world_state():
    """Test WorldState"""
    print("="*80)
    print("Testing WorldState")
    print("="*80)

    # Create states
    s1 = WorldState({
        PredicateAtom("on", ["a", "b"]),
        PredicateAtom("clear", ["c"]),
        PredicateAtom("handempty", [])
    }, depth=0)

    print(f"s1: {s1}")
    print(f"  Context: {s1.to_agentspeak_context()}")
    print(f"  Depth: {s1.depth}")
    print(f"  Is empty: {s1.is_empty()}")

    # Empty state
    s2 = WorldState(set(), depth=5)
    print(f"\ns2: {s2}")
    print(f"  Context: {s2.to_agentspeak_context()}")
    print(f"  Is empty: {s2.is_empty()}")

    # Test equality
    s3 = WorldState({
        PredicateAtom("on", ["a", "b"]),
        PredicateAtom("clear", ["c"]),
        PredicateAtom("handempty", [])
    }, depth=10)  # Different depth

    print(f"\ns1 == s3 (different depth): {s1 == s3}")

    # Test contains
    p_on_ab = PredicateAtom("on", ["a", "b"])
    print(f"\ns1 contains on(a,b): {s1.contains(p_on_ab)}")

    print("\n")


def test_state_graph():
    """Test StateGraph"""
    print("="*80)
    print("Testing StateGraph")
    print("="*80)

    # Create mock action
    from utils.pddl_parser import PDDLAction

    action_pickup = PDDLAction(
        name="pick-up",
        parameters=["?b1 - block", "?b2 - block"],
        preconditions="and (handempty) (clear ?b1) (on ?b1 ?b2)",
        effects="and (holding ?b1) (not (handempty))"
    )

    # Create states
    goal_state = WorldState({PredicateAtom("on", ["a", "b"])}, depth=0)
    state1 = WorldState({
        PredicateAtom("holding", ["a"]),
        PredicateAtom("clear", ["b"])
    }, depth=1)
    state2 = WorldState({
        PredicateAtom("ontable", ["a"]),
        PredicateAtom("clear", ["a"]),
        PredicateAtom("handempty", [])
    }, depth=2)

    # Create graph
    graph = StateGraph(goal_state)

    # Add transitions
    trans1 = StateTransition(
        from_state=state1,
        to_state=goal_state,
        action=action_pickup,
        action_args=["a", "b"],
        belief_updates=["+on(a, b)", "-holding(a)"],
        preconditions=[PredicateAtom("holding", ["a"])]
    )
    graph.add_transition(trans1)

    trans2 = StateTransition(
        from_state=state2,
        to_state=state1,
        action=action_pickup,
        action_args=["a"],
        belief_updates=["+holding(a)", "-ontable(a)"],
        preconditions=[PredicateAtom("handempty", [])]
    )
    graph.add_transition(trans2)

    print(f"Graph: {graph}")
    print(f"Statistics: {graph.get_statistics()}")

    # Find paths
    paths = graph.find_shortest_paths_to_goal()
    print(f"\nPaths to goal:")
    for state, path in paths.items():
        print(f"  From {state}:")
        if not path:
            print(f"    (already at goal)")
        else:
            for trans in path:
                print(f"    -> {trans.get_action_call()}")

    # Get leaf states
    leaves = graph.get_leaf_states()
    print(f"\nLeaf states: {len(leaves)}")
    for leaf in leaves:
        print(f"  {leaf}")

    # Generate DOT
    print(f"\nDOT format:")
    print(graph.to_dot())

    print("\n")


if __name__ == "__main__":
    test_predicate_atom()
    test_world_state()
    test_state_graph()
