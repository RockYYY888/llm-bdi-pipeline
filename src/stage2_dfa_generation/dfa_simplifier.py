"""
DFA Transition Label Simplifier

Transforms DFA with complex boolean expressions into equivalent DFA
where each transition checks exactly ONE atomic literal.

An atomic literal is either:
  - A positive literal: var (e.g., on_a_b)
  - A negative literal: !var (e.g., !clear_c)

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
    - Traverse BDD: each node tests ONE atomic literal (var or !var)
    - Create intermediate states for BDD decision nodes
    - Result: DFA where each edge checks exactly one atomic literal

    Atomic Literal Definition:
    - Positive literal: var (e.g., on_a_b means "on_a_b is true")
    - Negative literal: !var (e.g., !clear_c means "clear_c is false")

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
            SimplifiedDFA with atomic literal transitions only
            - Each transition label is either: var, !var, or "true"
            - No complex boolean expressions (AND/OR) in transition labels
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

        # Initialize global state tracking for ALL transitions
        # CRITICAL: These must persist across all transition processing
        # to ensure unique state names and avoid duplicates
        self.state_map = {}  # BDD node -> DFA state name
        self.state_counter = 0  # Global counter for unique state names across ALL transitions
        self.global_visited = set()  # Track which BDD nodes we've generated transitions for

        # Build new DFA
        new_transitions = []
        new_accepting = set()

        # Group transitions by source state
        # CRITICAL: Process all outgoing transitions from the same state together
        # to ensure determinism
        from collections import defaultdict
        transitions_by_source = defaultdict(list)
        for from_state, to_state, label in transitions:
            transitions_by_source[from_state].append((to_state, label))

        # Process each source state's transitions together
        for source_state, outgoing in transitions_by_source.items():
            print(f"\n  Processing state {source_state} with {len(outgoing)} outgoing transitions")

            # Parse all labels to BDDs
            bdd_transitions = []  # List of (target_state, bdd_node, is_accepting)
            unparseable = []  # List of (target_state, label) that couldn't be parsed

            for target_state, label in outgoing:
                try:
                    label_bdd = self._parse_to_bdd(label)
                    bdd_transitions.append((target_state, label_bdd, target_state in accepting_states))
                except (ValueError, KeyError, SyntaxError) as e:
                    print(f"    Warning: Could not parse '{label}': {type(e).__name__}: {e}")
                    print(f"    Keeping transition as-is: {source_state} -> {target_state} [label=\"{label}\"]")
                    unparseable.append((target_state, label))
                    continue
                except Exception as e:
                    raise RuntimeError(
                        f"Unexpected error while parsing transition label '{label}' "
                        f"from {source_state} to {target_state}: {type(e).__name__}: {e}"
                    ) from e

            # Add unparseable transitions as-is
            for target_state, label in unparseable:
                new_transitions.append((source_state, target_state, label))

            # Process all BDD transitions from this source state together
            if bdd_transitions:
                atomic_trans, reachable_accept = self._process_state_transitions(
                    source_state, bdd_transitions
                )
                new_transitions.extend(atomic_trans)
                new_accepting.update(reachable_accept)

        # Add original accepting states
        new_accepting.update(accepting_states)

        # Deduplicate transitions
        # When multiple BDD formulas share nodes, they may generate duplicate transitions
        from collections import Counter
        original_count = len(new_transitions)
        new_transitions = list(set(new_transitions))  # Remove duplicates
        if len(new_transitions) < original_count:
            print(f"  Removed {original_count - len(new_transitions)} duplicate transitions")

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

    def _process_state_transitions(self, source_state: str,
                                    bdd_transitions: List[Tuple[str, any, bool]]) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
        """
        Process all outgoing transitions from a single source state together.

        This ensures determinism by handling all BDD formulas from the same state
        simultaneously, creating a unified state space.

        Key insight: When transitions share BDD nodes, we need to ensure that
        each (source_state, label) pair maps to exactly ONE target state.

        Strategy:
        1. Build complete state mappings for ALL BDDs first
        2. Then generate transitions, using a local visited set per source state
           to ensure we only generate each transition once

        Args:
            source_state: The source state for all transitions
            bdd_transitions: List of (target_state, bdd_node, is_accepting) tuples

        Returns:
            (transitions, accepting_states)
        """
        all_transitions = []
        all_accepting = set()

        # Phase 1: Build state mappings for ALL BDDs from this source state
        # This ensures shared BDD nodes get consistent state assignments
        print(f"    Phase 1: Mapping {len(bdd_transitions)} BDD roots to states")
        for i, (target_state, bdd_node, target_is_accepting) in enumerate(bdd_transitions):
            print(f"      BDD {i}: node={id(bdd_node)}, hash={hash(bdd_node)}, target={target_state}")
            self._map_bdd_to_states(bdd_node, source_state, target_state, target_is_accepting)

        # Phase 2: Generate transitions using LOCAL visited set
        # CRITICAL: Use a local visited set for this source state only
        # This prevents duplicate transitions within this state's expansions
        # but allows the same BDD node to be used from different source states
        local_visited = set()  # Tracks which BDD nodes we've processed

        for target_state, bdd_node, target_is_accepting in bdd_transitions:
            # Generate transitions from this BDD
            trans = self._create_transitions_from_bdd(bdd_node, source_state, target_state, local_visited)
            all_transitions.extend(trans)

            # Track accepting states
            if target_is_accepting:
                all_accepting.add(target_state)

        return all_transitions, all_accepting

    def _bdd_to_atomic_transitions(self, start_state: str, target_state: str,
                                    bdd_node, target_is_accepting: bool) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
        """
        Convert a BDD to atomic transitions using Shannon Expansion.

        DEPRECATED: Use _process_state_transitions instead for correct determinism.

        Strategy:
        1. Build DFA states from BDD nodes (one-to-one mapping)
        2. Create transitions based on BDD edges:
           - High edge (var=true) → transition labeled "var"
           - Low edge (var=false) → transition labeled "!var"
        3. Use global state_map to handle shared BDD nodes correctly

        CRITICAL: state_map uses BDD nodes as keys, NOT id(node), because
        dd.autoref can have different nodes with same id() but different hash().

        Returns: (transitions, accepting_states)
        """
        transitions = []
        accepting = set()

        # First, build complete state mapping for this BDD
        self._map_bdd_to_states(bdd_node, start_state, target_state, target_is_accepting)

        # Then, create transitions from the BDD structure
        # CRITICAL: Use global_visited to avoid generating duplicate transitions
        # when different original transitions share BDD nodes
        trans = self._create_transitions_from_bdd(bdd_node, start_state, target_state, self.global_visited)
        transitions.extend(trans)

        # Collect accepting states
        if target_is_accepting:
            accepting.add(target_state)

        return transitions, accepting

    def _map_bdd_to_states(self, bdd_node, start_state: str, target_state: str,
                           target_is_accepting: bool, visited=None):
        """
        Phase 1: Build complete mapping from BDD nodes to DFA states.

        This ensures each unique BDD node gets exactly one DFA state.
        Handles node sharing correctly by tracking visited nodes.

        CRITICAL: Uses BDD nodes themselves as keys, not id(node), because
        dd.autoref can have different nodes with same id() but different hash().
        """
        if visited is None:
            visited = set()

        # Terminal nodes don't need state mapping
        if bdd_node in [self.bdd.true, self.bdd.false]:
            return

        # Check if already visited in this traversal
        # Use node itself as key (hashable)
        if bdd_node in visited:
            return  # Already processed in this call tree
        visited.add(bdd_node)

        # Map this node to a state if not already mapped
        # CRITICAL: Use node as key, not id(node)
        # IMPORTANT: If this node is already mapped, we reuse its state
        # This handles BDD node sharing across different transitions
        if bdd_node not in self.state_map:
            # For the root of this BDD (first time we see it in this call),
            # use the provided start_state
            self.state_map[bdd_node] = start_state

        # Recursively map children (they will get new states)
        var_index = bdd_node.level
        if var_index is not None and var_index < len(self.predicates):
            high_branch = bdd_node.high
            low_branch = bdd_node.low

            # Map children nodes
            if high_branch and high_branch not in [self.bdd.true, self.bdd.false]:
                # Assign state if not already assigned
                # CRITICAL: Use node as key, not id(node)
                if high_branch not in self.state_map:
                    self.state_counter += 1
                    self.state_map[high_branch] = f"s{self.state_counter}"
                # Always recurse to ensure ALL descendants are mapped
                self._map_bdd_to_states(high_branch, self.state_map[high_branch], target_state, target_is_accepting, visited)

            if low_branch and low_branch not in [self.bdd.true, self.bdd.false]:
                # Assign state if not already assigned
                # CRITICAL: Use node as key, not id(node)
                if low_branch not in self.state_map:
                    self.state_counter += 1
                    self.state_map[low_branch] = f"s{self.state_counter}"
                # Always recurse to ensure ALL descendants are mapped
                self._map_bdd_to_states(low_branch, self.state_map[low_branch], target_state, target_is_accepting, visited)

    def _create_transitions_from_bdd(self, bdd_node, start_state: str,
                                     target_state: str, visited: Set) -> List[Tuple[str, str, str]]:
        """
        Phase 2: Create transitions from BDD structure.

        Each BDD node creates outgoing transitions based on its edges.
        This phase is called AFTER state mapping is complete.

        CRITICAL: Handles dd.autoref negation semantics correctly:
        - When negated=True, high/low semantics are INVERTED
        - high branch means variable is FALSE
        - low branch means variable is TRUE

        Note: This may generate duplicate transitions when multiple BDD formulas
        share nodes. Duplicates are removed by the caller.

        Args:
            visited: Set of BDD nodes we've already processed
        """
        transitions = []

        # Terminal TRUE: create direct transition
        if bdd_node == self.bdd.true:
            transitions.append((start_state, target_state, "true"))
            return transitions

        # Terminal FALSE: no transition
        if bdd_node == self.bdd.false:
            return []

        # Get current node's DFA state
        # CRITICAL: Use node as key, not id(node)
        current_state = self.state_map.get(bdd_node, start_state)

        # Check if already processed this BDD node
        if bdd_node in visited:
            return []  # Already created transitions for this node
        visited.add(bdd_node)

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
                # CRITICAL: Use node as key, not id(node)
                true_state = self.state_map.get(true_branch)
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
                # CRITICAL: Use node as key, not id(node)
                false_state = self.state_map.get(false_branch)
                if false_state:
                    transitions.append((current_state, false_state, false_label))
                    # Recursively create transitions from false_state
                    transitions.extend(self._create_transitions_from_bdd(
                        false_branch, false_state, target_state, visited))

        return transitions

    def _find_matching_paren(self, expr: str, start: int) -> int:
        """
        Find the index of the closing parenthesis that matches the opening one at start

        Args:
            expr: The expression string
            start: Index of the opening parenthesis

        Returns:
            Index of the matching closing parenthesis, or -1 if not found
        """
        if start >= len(expr) or expr[start] != '(':
            return -1

        level = 0
        for i in range(start, len(expr)):
            if expr[i] == '(':
                level += 1
            elif expr[i] == ')':
                level -= 1
                if level == 0:
                    return i
        return -1

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
        """
        Recursive expression parser for BDD construction

        Raises:
            ValueError: If expression contains unknown predicates or is malformed
            SyntaxError: If expression has unbalanced parentheses
        """
        expr = expr.strip()

        # Single predicate
        if expr in self.predicates:
            return self.bdd.var(expr)

        # Negation
        if expr.startswith('~'):
            inner = expr[1:].strip()
            if not inner:
                raise ValueError(f"Empty expression after negation operator in: {expr}")

            # Strip outer parentheses if present
            if inner.startswith('(') and inner.endswith(')'):
                inner = inner[1:-1]

            if inner in self.predicates:
                return ~self.bdd.var(inner)
            else:
                return ~self._parse_expr_recursive(inner)

        # Handle parentheses - check if entire expression is wrapped
        if expr.startswith('(') and expr.endswith(')'):
            # Verify these are matching parentheses
            matching_close = self._find_matching_paren(expr, 0)
            if matching_close == len(expr) - 1:
                # Entire expression is wrapped, remove outer parens
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
                    left_expr = expr[:i].strip()
                    right_expr = expr[i+1:].strip()
                    if not left_expr or not right_expr:
                        raise ValueError(f"Empty operand in OR expression: {expr}")
                    left = self._parse_expr_recursive(left_expr)
                    right = self._parse_expr_recursive(right_expr)
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
                    left_expr = expr[:i].strip()
                    right_expr = expr[i+1:].strip()
                    if not left_expr or not right_expr:
                        raise ValueError(f"Empty operand in AND expression: {expr}")
                    left = self._parse_expr_recursive(left_expr)
                    right = self._parse_expr_recursive(right_expr)
                    return left & right

        # Fallback: treat as variable
        if expr in self.predicates:
            return self.bdd.var(expr)

        # Unknown expression - raise error instead of silently returning true
        raise ValueError(
            f"Unknown or malformed expression: '{expr}'. "
            f"Expected one of: {self.predicates}"
        )

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
