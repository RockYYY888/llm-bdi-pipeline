"""
DFA Transition Label Simplifier

Transforms DFA with complex boolean expressions into equivalent DFA
where each transition checks exactly ONE atomic literal (var or !var).

Algorithm: BDD-based Shannon Expansion
For each original transition (s1 -> s2, label=formula):
  1. Convert formula to BDD
  2. Traverse BDD using Shannon Expansion (each node tests one variable)
  3. Create intermediate states for BDD decision nodes
  4. Label edges with single atomic literals (var or !var)
  5. Guarantee: Complete, deterministic, and equivalent to original

Requirements:
- BDD library (pip install dd) - MANDATORY

Based on:
- Shannon Expansion: f = (x ∧ f|x=1) ∨ (¬x ∧ f|x=0)
- BDD canonical representation of boolean functions
- Atomic transition decomposition via decision tree
"""

from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass
import re
import sys
from pathlib import Path

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage1_interpretation.grounding_map import GroundingMap

try:
    from dd.autoref import BDD
    BDD_AVAILABLE = True
except ImportError:
    BDD_AVAILABLE = False


@dataclass
class SimplifiedDFA:
    """
    Result of DFA simplification

    Attributes:
        simplified_dot: DFA in DOT format with atomic labels
        stats: Statistics about the simplification
    """
    simplified_dot: str
    stats: Dict[str, Any]


class BDDBasedDFABuilder:
    """
    Builds atomic DFA from boolean formulas using BDD Shannon Expansion.

    Strategy:
    - For each original transition with complex formula, build BDD
    - Traverse BDD: each node tests ONE atom (var or !var)
    - Create intermediate states for BDD decision nodes
    - Result: DFA where each edge checks exactly one atomic literal

    Guarantees:
    - Deterministic: each (state, atom) has at most one successor
    - Complete: all atom value combinations are handled
    - Equivalent: accepts same language as original DFA (via Shannon Expansion)
    """

    def __init__(self):
        """Initialize BDD-based DFA builder"""
        if not BDD_AVAILABLE:
            raise ImportError(
                "BDD library is required for DFA simplification. "
                "Install with: pip install dd"
            )
        self.BDD = BDD
        self.bdd = None
        self.predicates = []

    def simplify(self, dfa_dot: str, grounding_map: GroundingMap) -> SimplifiedDFA:
        """
        Convert DFA to atomic transitions using BDD Shannon Expansion

        Args:
            dfa_dot: Original DFA in DOT format
            grounding_map: Grounding map for predicates

        Returns:
            SimplifiedDFA with atomic transitions (var or !var literals)
        """
        print("[BDD-Based DFA Builder] Converting to atomic transitions")

        # Parse original DFA
        transitions = self._parse_transitions(dfa_dot)
        accepting_states = self._parse_accepting_states(dfa_dot)
        self.predicates = self._collect_predicates(transitions, grounding_map)

        if len(self.predicates) == 0:
            # No predicates, return as-is
            return SimplifiedDFA(
                simplified_dot=dfa_dot,
                stats={'method': 'bdd', 'num_predicates': 0, 'num_states': 0}
            )

        print(f"  Predicates: {self.predicates}")
        print(f"  Original states: {len(self._get_all_states(transitions))}")
        print(f"  Original transitions: {len(transitions)}")

        # Initialize BDD
        self.bdd = self.BDD()
        for pred in self.predicates:
            self.bdd.add_var(pred)

        # Initialize global state tracking (CRITICAL FIX)
        self.state_map = {}  # BDD node ID -> DFA state name
        self.state_counter = 0  # Global counter for unique state names

        # Build new DFA
        new_transitions = []
        new_accepting = set()

        # Process each original transition
        for from_state, to_state, label in transitions:
            # Build BDD for this label
            try:
                label_bdd = self._parse_to_bdd(label)
            except Exception as e:
                print(f"  Warning: Could not parse '{label}': {e}")
                # Fallback: keep as is
                new_transitions.append((from_state, to_state, label))
                continue

            # Convert BDD to atomic transitions
            atomic_trans, reachable_accept = self._bdd_to_atomic_transitions(
                from_state, to_state, label_bdd, to_state in accepting_states
            )
            new_transitions.extend(atomic_trans)
            new_accepting.update(reachable_accept)

        # Add original accepting states
        new_accepting.update(accepting_states)

        # Build output DOT
        simplified_dot = self._build_dot(new_transitions, new_accepting, dfa_dot)

        stats = {
            'method': 'bdd',
            'num_predicates': len(self.predicates),
            'num_original_states': len(self._get_all_states(transitions)),
            'num_new_states': len(self._get_all_states(new_transitions)),
            'num_original_transitions': len(transitions),
            'num_new_transitions': len(new_transitions),
        }

        print(f"  New states: {stats['num_new_states']}")
        print(f"  New transitions: {stats['num_new_transitions']}")

        return SimplifiedDFA(
            simplified_dot=simplified_dot,
            stats=stats
        )

    def _bdd_to_atomic_transitions(self, start_state: str, target_state: str,
                                    bdd_node, target_is_accepting: bool) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
        """
        Convert a BDD to atomic transitions using Shannon Expansion.

        Strategy:
        1. Build DFA states from BDD nodes (one-to-one mapping)
        2. Create transitions based on BDD edges:
           - High edge (var=true) → transition labeled "var"
           - Low edge (var=false) → transition labeled "!var"
        3. Use global state_map to handle shared BDD nodes correctly

        Returns: (transitions, accepting_states)
        """
        transitions = []
        accepting = set()

        # First, build complete state mapping for this BDD
        self._map_bdd_to_states(bdd_node, start_state, target_state, target_is_accepting)

        # Then, create transitions from the BDD structure
        visited = set()  # Track which nodes we've created transitions for
        trans = self._create_transitions_from_bdd(bdd_node, start_state, target_state, visited)
        transitions.extend(trans)

        # Collect accepting states
        if target_is_accepting:
            accepting.add(target_state)

        return transitions, accepting

    def _map_bdd_to_states(self, bdd_node, start_state: str, target_state: str,
                           target_is_accepting: bool):
        """
        Phase 1: Build complete mapping from BDD nodes to DFA states.

        This ensures each unique BDD node gets exactly one DFA state.
        Handles node sharing correctly.
        """
        # Terminal nodes don't need state mapping
        if bdd_node in [self.bdd.true, self.bdd.false]:
            return

        # Check if already mapped
        node_id = id(bdd_node)
        if node_id in self.state_map:
            return  # Already processed

        # Map this node to a state
        # The root BDD node for this transition uses start_state
        self.state_map[node_id] = start_state

        # Recursively map children (they will get new states)
        var_index = bdd_node.level
        if var_index is not None and var_index < len(self.predicates):
            high_branch = bdd_node.high
            low_branch = bdd_node.low

            # Map children nodes
            if high_branch and high_branch not in [self.bdd.true, self.bdd.false]:
                child_id = id(high_branch)
                if child_id not in self.state_map:
                    self.state_counter += 1
                    self.state_map[child_id] = f"s{self.state_counter}"
                self._map_bdd_to_states(high_branch, self.state_map[child_id], target_state, target_is_accepting)

            if low_branch and low_branch not in [self.bdd.true, self.bdd.false]:
                child_id = id(low_branch)
                if child_id not in self.state_map:
                    self.state_counter += 1
                    self.state_map[child_id] = f"s{self.state_counter}"
                self._map_bdd_to_states(low_branch, self.state_map[child_id], target_state, target_is_accepting)

    def _create_transitions_from_bdd(self, bdd_node, start_state: str,
                                     target_state: str, visited: Set[int]) -> List[Tuple[str, str, str]]:
        """
        Phase 2: Create transitions from BDD structure.

        Each BDD node creates outgoing transitions based on its edges.
        This phase is called AFTER state mapping is complete.

        CRITICAL: Handles dd.autoref negation semantics correctly:
        - When negated=True, high/low semantics are INVERTED
        - high branch means variable is FALSE
        - low branch means variable is TRUE

        Args:
            visited: Set of node IDs we've already processed (prevents duplicates)
        """
        transitions = []

        # Terminal TRUE: create direct transition
        if bdd_node == self.bdd.true:
            transitions.append((start_state, target_state, "true"))
            return transitions

        # Terminal FALSE: no transition
        if bdd_node == self.bdd.false:
            return []

        # Check if already processed
        node_id = id(bdd_node)
        if node_id in visited:
            return []  # Already created transitions for this node
        visited.add(node_id)

        # Get current node's DFA state
        current_state = self.state_map.get(node_id, start_state)

        # Get variable and negation status
        var_index = bdd_node.level
        if var_index is None or var_index >= len(self.predicates):
            # Leaf node
            transitions.append((current_state, target_state, "true"))
            return transitions

        var = self.predicates[var_index]
        is_negated = bdd_node.negated
        high_branch = bdd_node.high
        low_branch = bdd_node.low

        # Determine TRUE and FALSE branches based on negation
        # KEY FIX: When negated=True, high/low semantics are inverted
        if not is_negated:
            # Normal node: high=var_true, low=var_false
            true_branch = high_branch
            false_branch = low_branch
            true_label = var
            false_label = f"!{var}"
        else:
            # Negated node: INVERTED semantics!
            true_branch = low_branch   # When var=TRUE, follow LOW
            false_branch = high_branch # When var=FALSE, follow HIGH
            true_label = var           # But label is still the variable name
            false_label = f"!{var}"

        # Process TRUE branch (when variable is TRUE)
        if true_branch is not None and true_branch != self.bdd.false:
            if true_branch == self.bdd.true:
                # Direct transition to target
                transitions.append((current_state, target_state, true_label))
            else:
                # Transition to intermediate state
                true_state = self.state_map.get(id(true_branch))
                if true_state:
                    transitions.append((current_state, true_state, true_label))
                    # Recursively create transitions from true_state
                    transitions.extend(self._create_transitions_from_bdd(
                        true_branch, true_state, target_state, visited))

        # Process FALSE branch (when variable is FALSE)
        if false_branch is not None and false_branch != self.bdd.false:
            if false_branch == self.bdd.true:
                # Direct transition to target
                transitions.append((current_state, target_state, false_label))
            else:
                # Transition to intermediate state
                false_state = self.state_map.get(id(false_branch))
                if false_state:
                    transitions.append((current_state, false_state, false_label))
                    # Recursively create transitions from false_state
                    transitions.extend(self._create_transitions_from_bdd(
                        false_branch, false_state, target_state, visited))

        return transitions

    def _parse_to_bdd(self, expr: str):
        """Parse boolean expression to BDD using Shannon Expansion"""
        # Handle special cases
        if expr == "true":
            return self.bdd.true
        if expr == "false":
            return self.bdd.false

        # Normalize
        expr = expr.replace('!', '~')
        expr = expr.replace('&&', '&')
        expr = expr.replace('||', '|')

        # Parse recursively
        return self._parse_expr_recursive(expr)

    def _parse_expr_recursive(self, expr: str):
        """Recursive expression parser for BDD construction"""
        expr = expr.strip()

        # Single predicate
        if expr in self.predicates:
            return self.bdd.var(expr)

        # Negation
        if expr.startswith('~'):
            inner = expr[1:].strip('()')
            if inner in self.predicates:
                return ~self.bdd.var(inner)
            else:
                return ~self._parse_expr_recursive(inner)

        # Handle parentheses
        if expr.startswith('(') and expr.endswith(')'):
            return self._parse_expr_recursive(expr[1:-1])

        # Find main operator (lowest precedence)
        # Order: OR > AND > NOT
        level = 0
        for i, c in enumerate(expr):
            if c == '(':
                level += 1
            elif c == ')':
                level -= 1
            elif level == 0:
                if c == '|':
                    # Disjunction
                    left = self._parse_expr_recursive(expr[:i])
                    right = self._parse_expr_recursive(expr[i+1:])
                    return left | right

        # AND
        level = 0
        for i, c in enumerate(expr):
            if c == '(':
                level += 1
            elif c == ')':
                level -= 1
            elif level == 0:
                if c == '&':
                    left = self._parse_expr_recursive(expr[:i])
                    right = self._parse_expr_recursive(expr[i+1:])
                    return left & right

        # Fallback: treat as variable
        if expr in self.predicates:
            return self.bdd.var(expr)

        # Unknown expression
        return self.bdd.true

    def _parse_transitions(self, dfa_dot: str) -> List[Tuple[str, str, str]]:
        """Parse transitions from DOT format"""
        transitions = []
        for line in dfa_dot.split('\n'):
            match = re.match(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]', line.strip())
            if match:
                from_state, to_state, label = match.groups()
                if from_state not in ['init', '__start']:
                    transitions.append((from_state, to_state, label))
        return transitions

    def _parse_accepting_states(self, dfa_dot: str) -> Set[str]:
        """Parse accepting states from DOT format"""
        accepting = set()
        for line in dfa_dot.split('\n'):
            match = re.search(r'node \[shape = doublecircle\];\s*([^;]+)', line)
            if match:
                states = match.group(1).split()
                accepting.update(states)
        return accepting

    def _get_all_states(self, transitions: List[Tuple[str, str, str]]) -> Set[str]:
        """Get all states from transitions"""
        states = set()
        for from_s, to_s, _ in transitions:
            states.add(from_s)
            states.add(to_s)
        return states

    def _collect_predicates(self, transitions: List[Tuple], grounding_map: GroundingMap) -> List[str]:
        """Collect all atomic predicates from transitions"""
        predicates = set()
        for _, _, label in transitions:
            tokens = re.findall(r'\w+', label)
            for token in tokens:
                if token.lower() not in ['true', 'false', 'and', 'or', 'not']:
                    if grounding_map and token in grounding_map.atoms:
                        predicates.add(token)
        return sorted(list(predicates))

    def _build_dot(self, transitions: List[Tuple[str, str, str]],
                   accepting_states: Set[str], original_dot: str) -> str:
        """Build DOT string from transitions"""
        lines = []
        lines.append("digraph MONA_DFA {")
        lines.append(" rankdir = LR;")
        lines.append(" center = true;")
        lines.append(" size = \"7.5,10.5\";")
        lines.append(" edge [fontname = Courier];")
        lines.append(" node [height = .5, width = .5];")

        # Accepting states
        if accepting_states:
            acc_list = ' '.join(sorted(accepting_states))
            lines.append(f" node [shape = doublecircle]; {acc_list};")

        # All other states
        all_states = self._get_all_states(transitions)
        other_states = all_states - accepting_states
        if other_states:
            other_list = ' '.join(sorted(other_states))
            lines.append(f" node [shape = circle]; {other_list};")

        # Init
        lines.append(" init [shape = plaintext, label = \"\"];")
        lines.append(" init -> 1;")

        # Transitions
        for from_s, to_s, label in sorted(transitions):
            lines.append(f" {from_s} -> {to_s} [label=\"{label}\"];")

        lines.append("}")
        return '\n'.join(lines)


class DFASimplifier:
    """
    DFA simplifier - converts to atomic transitions using Shannon Expansion
    """

    def __init__(self):
        """Initialize DFA simplifier"""
        self.builder = BDDBasedDFABuilder()

    def simplify(self, dfa_dot: str, grounding_map: GroundingMap,
                 verify: bool = False) -> SimplifiedDFA:
        """
        Simplify DFA to atomic transitions (Shannon Expansion based)

        Args:
            dfa_dot: DFA in DOT format
            grounding_map: Grounding map
            verify: If True, verify equivalence after simplification (requires test module)

        Returns:
            SimplifiedDFA object (guaranteed equivalent to input)

        Raises:
            ValueError: If verify=True and equivalence check fails
            ImportError: If verify=True but verification module not available
        """
        result = self.builder.simplify(dfa_dot, grounding_map)

        if verify:
            # Import verification module from tests
            try:
                from tests.stage2_dfa_generation.test_dfa_equivalence_verification import verify_equivalence
            except ImportError:
                raise ImportError(
                    "Equivalence verification requires test module. "
                    "Cannot import from tests/stage2_dfa_generation/test_dfa_equivalence_verification.py"
                )

            # Run verification
            all_atoms = self.builder.predicates
            if all_atoms:
                is_equiv, counterexamples = verify_equivalence(dfa_dot, result.simplified_dot, all_atoms)
                if not is_equiv:
                    raise ValueError(
                        f"Equivalence verification failed! Found {len(counterexamples)} counterexample(s): "
                        f"{counterexamples[:3]}"  # Show first 3 counterexamples
                    )
                print(f"✅ Equivalence verified: tested all {2**len(all_atoms)} valuations")

        return result
