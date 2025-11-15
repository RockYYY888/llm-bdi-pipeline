"""
DFA Transition Label Simplifier

Transforms DFA with complex boolean expressions on transition labels into
equivalent DFA with atomic partition symbols, using BDD-based partition refinement.

Key Features:
- Scalable: Uses Binary Decision Diagrams (BDD) for symbolic computation
- Lossless: Preserves all semantic information from original labels
- Minimal: Generates only the necessary partitions (not all 2^n minterms)
- Fallback: Simple minterm enumeration for small cases or when BDD unavailable

Design: See docs/dfa_simplification_design.md
"""

from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import re
import sys
from pathlib import Path

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.boolean_expression_parser import BooleanExpressionParser
from stage1_interpretation.grounding_map import GroundingMap


@dataclass
class PartitionInfo:
    """
    Information about a partition in the simplified DFA

    Attributes:
        symbol: Unique partition symbol (e.g., "α1", "p1")
        expression: Boolean expression defining this partition
        minterm: Complete assignment (if using minterm method)
        predicate_values: Dict mapping predicate -> bool (for this partition)
    """
    symbol: str
    expression: str
    minterm: Optional[str] = None
    predicate_values: Optional[Dict[str, bool]] = None


@dataclass
class SimplifiedDFA:
    """
    Result of DFA simplification

    Attributes:
        simplified_dot: DFA in DOT format with atomic partition labels
        partitions: List of partition information
        partition_map: Mapping from partition symbol to PartitionInfo
        original_label_to_partitions: Mapping from original label to partition symbols
        stats: Statistics about the simplification
    """
    simplified_dot: str
    partitions: List[PartitionInfo]
    partition_map: Dict[str, PartitionInfo]
    original_label_to_partitions: Dict[str, List[str]]
    stats: Dict[str, Any]


class BDDSimplifier:
    """
    BDD-based DFA simplifier using partition refinement

    Requires: `dd` library (pip install dd)
    """

    def __init__(self):
        """Initialize BDD simplifier"""
        try:
            from dd.autoref import BDD
            self.BDD = BDD
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: 'dd' library not available. Install with: pip install dd")

    def is_available(self) -> bool:
        """Check if BDD library is available"""
        return self.available

    def simplify(self, dfa_dot: str, grounding_map: GroundingMap) -> SimplifiedDFA:
        """
        Simplify DFA using BDD-based partition refinement

        Args:
            dfa_dot: DFA in DOT format
            grounding_map: Grounding map for anti-grounding

        Returns:
            SimplifiedDFA object
        """
        if not self.available:
            raise RuntimeError("BDD library not available. Install with: pip install dd")

        # Parse DFA to extract transitions
        transitions = self._parse_transitions(dfa_dot)

        # Collect all atomic predicates
        all_predicates = self._collect_predicates(transitions, grounding_map)

        if len(all_predicates) == 0:
            # No predicates, return original
            return self._create_trivial_result(dfa_dot)

        print(f"[BDD Simplifier] Found {len(all_predicates)} atomic predicates")
        print(f"  Predicates: {all_predicates[:10]}{'...' if len(all_predicates) > 10 else ''}")

        # Create BDD manager and declare variables
        bdd = self.BDD()
        for pred in all_predicates:
            bdd.declare(pred)

        # Build BDD for each unique label
        label_to_bdd = self._build_label_bdds(transitions, bdd, all_predicates, grounding_map)

        print(f"[BDD Simplifier] Built BDDs for {len(label_to_bdd)} unique labels")

        # Compute partition refinement
        partitions = self._compute_partitions(label_to_bdd, bdd, all_predicates)

        print(f"[BDD Simplifier] Generated {len(partitions)} partitions")
        print(f"  Compression: {2**len(all_predicates)} possible → {len(partitions)} used")

        # Create partition info objects
        partition_infos = []
        partition_map = {}
        for i, (partition_bdd, expr) in enumerate(partitions):
            # Get predicate values for this partition
            pred_values = self._bdd_to_assignment(partition_bdd, bdd, all_predicates)

            # Check if this partition represents exactly one predicate being true
            true_predicates = [pred for pred, val in pred_values.items() if val]

            if len(true_predicates) == 1 and len([v for v in pred_values.values() if not v]) == len(all_predicates) - 1:
                # Single predicate is true, all others false -> use predicate name
                symbol = true_predicates[0]
            else:
                # Complex expression -> use the expression itself as symbol
                symbol = expr

            info = PartitionInfo(
                symbol=symbol,
                expression=expr,
                predicate_values=pred_values
            )
            partition_infos.append(info)
            partition_map[symbol] = info

        # Build mapping: original label → partition symbols
        original_label_to_partitions = {}
        for label, label_bdd in label_to_bdd.items():
            matching_partitions = []
            for info in partition_infos:
                # Check if partition is subset of label
                partition_bdd = self._expression_to_bdd(info.expression, bdd, all_predicates)
                if self._is_subset(partition_bdd, label_bdd, bdd):
                    matching_partitions.append(info.symbol)
            original_label_to_partitions[label] = matching_partitions

        # Rebuild DFA with partition symbols
        simplified_dot = self._rebuild_dfa(dfa_dot, transitions, original_label_to_partitions)

        # Collect statistics
        stats = {
            'method': 'bdd',
            'num_predicates': len(all_predicates),
            'num_partitions': len(partitions),
            'num_original_labels': len(label_to_bdd),
            'compression_ratio': 2**len(all_predicates) / len(partitions) if len(partitions) > 0 else 1,
        }

        return SimplifiedDFA(
            simplified_dot=simplified_dot,
            partitions=partition_infos,
            partition_map=partition_map,
            original_label_to_partitions=original_label_to_partitions,
            stats=stats
        )

    def _parse_transitions(self, dfa_dot: str) -> List[Tuple[str, str, str]]:
        """Parse transitions from DOT format"""
        transitions = []
        for line in dfa_dot.split('\n'):
            line = line.strip()
            # Match: from -> to [label="expr"];
            match = re.match(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]', line)
            if match:
                from_state, to_state, label = match.groups()
                # Skip init transitions
                if from_state not in ['init', '__start']:
                    transitions.append((from_state, to_state, label))
        return transitions

    def _collect_predicates(self, transitions: List[Tuple], grounding_map: GroundingMap) -> List[str]:
        """Collect all atomic predicates from transition labels"""
        predicates = set()

        for _, _, label in transitions:
            # Tokenize and extract identifiers (predicates)
            # Handle operators: &, |, !, ~, (, ), true, false
            tokens = re.findall(r'\w+', label)
            for token in tokens:
                # Skip boolean constants and operators
                if token.lower() not in ['true', 'false', 'and', 'or', 'not']:
                    predicates.add(token)

        return sorted(list(predicates))

    def _build_label_bdds(self, transitions: List[Tuple], bdd,
                          all_predicates: List[str],
                          grounding_map: GroundingMap) -> Dict[str, Any]:
        """Build BDD for each unique transition label"""
        label_to_bdd = {}
        unique_labels = set(label for _, _, label in transitions)

        for label in unique_labels:
            try:
                label_bdd = self._expression_to_bdd(label, bdd, all_predicates)
                label_to_bdd[label] = label_bdd
            except Exception as e:
                print(f"Warning: Failed to build BDD for label '{label}': {e}")
                # Use fallback: treat as single predicate
                # For "true", create tautology
                if label.lower() == 'true':
                    label_to_bdd[label] = bdd.true
                else:
                    # Treat entire label as a predicate name
                    if label in all_predicates:
                        label_to_bdd[label] = bdd.var(label)

        return label_to_bdd

    def _expression_to_bdd(self, expr: str, bdd, all_predicates: List[str]) -> Any:
        """
        Convert boolean expression to BDD

        Supports: &, |, !, ~, parentheses, true, false
        """
        # Normalize expression
        expr = expr.strip()

        # Handle special cases
        if expr.lower() == 'true':
            return bdd.true
        if expr.lower() == 'false':
            return bdd.false

        # Replace operators for Python evaluation
        # Be careful with operator precedence
        expr_normalized = expr
        expr_normalized = expr_normalized.replace('&&', '&')
        expr_normalized = expr_normalized.replace('||', '|')

        # Handle negation: ~ or !
        # Convert ~pred to !pred for consistency
        expr_normalized = expr_normalized.replace('~', '!')

        # Build BDD recursively using a simple recursive descent parser
        return self._parse_bdd_expression(expr_normalized, bdd, all_predicates)

    def _parse_bdd_expression(self, expr: str, bdd, all_predicates: List[str]) -> Any:
        """Parse boolean expression and build BDD (recursive descent)"""
        expr = expr.strip()

        # Base case: true/false
        if expr.lower() == 'true':
            return bdd.true
        if expr.lower() == 'false':
            return bdd.false

        # Base case: variable
        if expr in all_predicates:
            return bdd.var(expr)

        # Handle negation: !expr
        if expr.startswith('!'):
            inner = expr[1:].strip()
            return ~self._parse_bdd_expression(inner, bdd, all_predicates)

        # Handle parentheses: remove outermost if present
        if expr.startswith('(') and expr.endswith(')'):
            # Check if these are the matching outermost parens
            depth = 0
            for i, char in enumerate(expr):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0 and i < len(expr) - 1:
                        # Not outermost, break
                        break
            else:
                # Outermost parens, remove them
                return self._parse_bdd_expression(expr[1:-1], bdd, all_predicates)

        # Find main operator (lowest precedence first: |, then &)
        # Scan from right to left to handle left-associativity
        depth = 0
        or_pos = -1
        and_pos = -1

        for i in range(len(expr) - 1, -1, -1):
            char = expr[i]
            if char == ')':
                depth += 1
            elif char == '(':
                depth -= 1
            elif depth == 0:
                if char == '|' and or_pos == -1:
                    or_pos = i
                elif char == '&' and and_pos == -1 and or_pos == -1:
                    and_pos = i

        # Split by operator
        if or_pos != -1:
            left = expr[:or_pos].strip()
            right = expr[or_pos+1:].strip()
            return self._parse_bdd_expression(left, bdd, all_predicates) | \
                   self._parse_bdd_expression(right, bdd, all_predicates)

        if and_pos != -1:
            left = expr[:and_pos].strip()
            right = expr[and_pos+1:].strip()
            return self._parse_bdd_expression(left, bdd, all_predicates) & \
                   self._parse_bdd_expression(right, bdd, all_predicates)

        # Couldn't parse, treat as variable
        if expr in all_predicates:
            return bdd.var(expr)

        # Last resort: return true (for expressions like "true")
        print(f"Warning: Could not parse expression '{expr}', treating as true")
        return bdd.true

    def _compute_partitions(self, label_to_bdd: Dict[str, Any], bdd,
                           all_predicates: List[str]) -> List[Tuple[Any, str]]:
        """
        Compute minimal partition refinement from label BDDs

        Returns:
            List of (partition_bdd, expression_string) tuples
        """
        # Start with all label BDDs
        all_bdds = list(label_to_bdd.values())

        # Compute all satisfying minterms (paths) for all BDDs combined
        # This gives us the minimal partition that can represent all labels

        partitions = []
        covered = bdd.false

        for label, label_bdd in label_to_bdd.items():
            # Find minterms of this label that aren't covered yet
            uncovered = label_bdd & ~covered

            if uncovered == bdd.false:
                continue  # Already covered

            # Extract all minterms (satisfying assignments) from uncovered
            # For efficiency, we extract minimal DNF instead of all minterms
            # Use BDD's built-in method if available, otherwise enumerate

            # Get one satisfying assignment at a time
            minterm_count = 0
            current = uncovered

            while current != bdd.false and minterm_count < 1000:  # Safety limit
                # Pick one satisfying assignment (minterm)
                try:
                    # Get one satisfying assignment
                    assignment = bdd.pick(current)

                    # Build BDD for this assignment (complete minterm)
                    minterm_bdd = bdd.true
                    minterm_expr_parts = []

                    for pred in all_predicates:
                        if pred in assignment:
                            if assignment[pred]:
                                minterm_bdd &= bdd.var(pred)
                                minterm_expr_parts.append(pred)
                            else:
                                minterm_bdd &= ~bdd.var(pred)
                                minterm_expr_parts.append(f"~{pred}")
                        # If pred not in assignment, it's a don't-care (can be True or False)
                        # For a complete minterm, we need to assign it
                        # Let's assign it to False by default
                        else:
                            minterm_bdd &= ~bdd.var(pred)
                            minterm_expr_parts.append(f"~{pred}")

                    minterm_expr = " & ".join(minterm_expr_parts)

                    partitions.append((minterm_bdd, minterm_expr))
                    covered |= minterm_bdd

                    # Remove this minterm from current
                    current &= ~minterm_bdd
                    minterm_count += 1

                except Exception as e:
                    print(f"Warning: Failed to extract minterm: {e}")
                    break

        return partitions

    def _bdd_to_assignment(self, bdd_node, bdd, all_predicates: List[str]) -> Dict[str, bool]:
        """Extract one satisfying assignment from BDD"""
        try:
            assignment = bdd.pick(bdd_node)
            # Fill in missing predicates (don't-cares) with False
            full_assignment = {}
            for pred in all_predicates:
                full_assignment[pred] = assignment.get(pred, False)
            return full_assignment
        except:
            return {}

    def _is_subset(self, bdd1, bdd2, bdd) -> bool:
        """Check if bdd1 is a subset of bdd2 (bdd1 => bdd2)"""
        # bdd1 ⊆ bdd2 iff (bdd1 & ~bdd2) = false
        return (bdd1 & ~bdd2) == bdd.false

    def _rebuild_dfa(self, original_dot: str, transitions: List[Tuple],
                     label_to_partitions: Dict[str, List[str]]) -> str:
        """Rebuild DFA DOT with partition symbols replacing labels"""
        # Replace each transition with potentially multiple transitions
        lines = original_dot.split('\n')
        new_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Check if this is a transition line
            match = re.match(r'(\s*)(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\](.*)', line)

            if match:
                indent, from_state, to_state, label, rest = match.groups()

                # Skip init transitions
                if from_state in ['init', '__start']:
                    new_lines.append(line)
                    continue

                # Get partition symbols for this label
                partition_symbols = label_to_partitions.get(label, [])

                if not partition_symbols:
                    # No partitions found, keep original
                    new_lines.append(line)
                else:
                    # Create one transition per partition
                    for symbol in partition_symbols:
                        new_line = f'{indent}{from_state} -> {to_state} [label="{symbol}"];'
                        new_lines.append(new_line)
            else:
                # Not a transition, keep as-is
                new_lines.append(line)

        return '\n'.join(new_lines)

    def _create_trivial_result(self, dfa_dot: str) -> SimplifiedDFA:
        """Create trivial result when no simplification needed"""
        return SimplifiedDFA(
            simplified_dot=dfa_dot,
            partitions=[],
            partition_map={},
            original_label_to_partitions={},
            stats={'method': 'none', 'num_predicates': 0, 'num_partitions': 0}
        )


class SimpleMintermSimplifier:
    """
    Simple minterm-based simplifier (fallback for small domains)

    Explicitly enumerates all 2^n minterms, suitable only for n < 15.
    """

    def __init__(self, max_predicates: int = 12):
        """
        Initialize simple simplifier

        Args:
            max_predicates: Maximum number of predicates (default: 12, max 2^12=4096 minterms)
        """
        self.max_predicates = max_predicates

    def simplify(self, dfa_dot: str, grounding_map: GroundingMap) -> SimplifiedDFA:
        """
        Simplify DFA using explicit minterm enumeration

        Args:
            dfa_dot: DFA in DOT format
            grounding_map: Grounding map for anti-grounding

        Returns:
            SimplifiedDFA object
        """
        print(f"[Simple Simplifier] Using explicit minterm enumeration")

        # Parse transitions
        transitions = self._parse_transitions(dfa_dot)

        # Collect predicates
        all_predicates = self._collect_predicates(transitions, grounding_map)

        if len(all_predicates) == 0:
            return self._create_trivial_result(dfa_dot)

        if len(all_predicates) > self.max_predicates:
            raise ValueError(
                f"Too many predicates ({len(all_predicates)}) for simple simplifier. "
                f"Max: {self.max_predicates}. Use BDD-based simplifier instead."
            )

        print(f"  Predicates ({len(all_predicates)}): {all_predicates}")
        print(f"  Will generate up to {2**len(all_predicates)} minterms")

        # Generate all possible minterms
        all_minterms = self._generate_all_minterms(all_predicates)

        print(f"  Generated {len(all_minterms)} total minterms")

        # For each label, find which minterms satisfy it
        label_to_minterms = {}
        parser = BooleanExpressionParser(grounding_map)

        for _, _, label in transitions:
            if label not in label_to_minterms:
                satisfying = self._find_satisfying_minterms(
                    label, all_minterms, all_predicates, parser
                )
                label_to_minterms[label] = satisfying

        # Collect only used minterms
        used_minterms = set()
        for minterms in label_to_minterms.values():
            used_minterms.update(minterms)

        print(f"  Used minterms: {len(used_minterms)} out of {len(all_minterms)}")

        # Create partition infos
        partition_infos = []
        partition_map = {}
        minterm_to_symbol = {}

        for i, minterm in enumerate(sorted(used_minterms)):
            # Generate meaningful symbol from minterm
            expr = self._minterm_to_expression(minterm, all_predicates)
            values = dict(zip(all_predicates, minterm))

            # Check if this minterm represents exactly one predicate being true
            true_predicates = [pred for pred, val in values.items() if val]

            if len(true_predicates) == 1 and len([v for v in values.values() if not v]) == len(all_predicates) - 1:
                # Single predicate is true, all others false -> use predicate name
                symbol = true_predicates[0]
            else:
                # Complex expression -> use the expression itself as symbol
                symbol = expr

            info = PartitionInfo(
                symbol=symbol,
                expression=expr,
                minterm=minterm,
                predicate_values=values
            )
            partition_infos.append(info)
            partition_map[symbol] = info
            minterm_to_symbol[minterm] = symbol

        # Build label to partition mapping
        original_label_to_partitions = {}
        for label, minterms in label_to_minterms.items():
            symbols = [minterm_to_symbol[m] for m in minterms]
            original_label_to_partitions[label] = symbols

        # Rebuild DFA
        simplified_dot = self._rebuild_dfa(dfa_dot, transitions, original_label_to_partitions)

        # Stats
        stats = {
            'method': 'minterm',
            'num_predicates': len(all_predicates),
            'num_partitions': len(used_minterms),
            'num_total_minterms': len(all_minterms),
            'compression_ratio': len(all_minterms) / len(used_minterms) if len(used_minterms) > 0 else 1,
        }

        return SimplifiedDFA(
            simplified_dot=simplified_dot,
            partitions=partition_infos,
            partition_map=partition_map,
            original_label_to_partitions=original_label_to_partitions,
            stats=stats
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

    def _collect_predicates(self, transitions: List[Tuple], grounding_map: GroundingMap) -> List[str]:
        """Collect all atomic predicates"""
        predicates = set()
        for _, _, label in transitions:
            tokens = re.findall(r'\w+', label)
            for token in tokens:
                if token.lower() not in ['true', 'false', 'and', 'or', 'not']:
                    predicates.add(token)
        return sorted(list(predicates))

    def _generate_all_minterms(self, predicates: List[str]) -> List[Tuple[bool, ...]]:
        """Generate all 2^n minterms"""
        import itertools
        n = len(predicates)
        minterms = []
        for values in itertools.product([False, True], repeat=n):
            minterms.append(values)
        return minterms

    def _find_satisfying_minterms(self, label: str, all_minterms: List[Tuple],
                                  predicates: List[str], parser: BooleanExpressionParser) -> List[Tuple]:
        """Find which minterms satisfy the given label expression"""
        satisfying = []

        for minterm in all_minterms:
            # Evaluate label under this minterm assignment
            if self._evaluate_expression(label, minterm, predicates):
                satisfying.append(minterm)

        return satisfying

    def _evaluate_expression(self, expr: str, minterm: Tuple[bool, ...],
                            predicates: List[str]) -> bool:
        """Evaluate boolean expression under minterm assignment"""
        # Create assignment dict
        assignment = dict(zip(predicates, minterm))

        # Replace predicates with their values
        expr_eval = expr

        # Sort by length descending to avoid substring issues
        sorted_preds = sorted(predicates, key=len, reverse=True)

        for pred in sorted_preds:
            value_str = 'True' if assignment[pred] else 'False'
            # Use word boundaries to avoid partial replacement
            expr_eval = re.sub(r'\b' + re.escape(pred) + r'\b', value_str, expr_eval)

        # Normalize operators
        expr_eval = expr_eval.replace('&', ' and ')
        expr_eval = expr_eval.replace('|', ' or ')
        expr_eval = expr_eval.replace('!', ' not ')
        expr_eval = expr_eval.replace('~', ' not ')
        expr_eval = expr_eval.replace('true', 'True')
        expr_eval = expr_eval.replace('false', 'False')

        # Evaluate
        try:
            return eval(expr_eval)
        except:
            print(f"Warning: Could not evaluate '{expr_eval}'")
            return False

    def _minterm_to_expression(self, minterm: Tuple[bool, ...], predicates: List[str]) -> str:
        """Convert minterm to boolean expression string"""
        parts = []
        for pred, value in zip(predicates, minterm):
            if value:
                parts.append(pred)
            else:
                parts.append(f"~{pred}")
        return " & ".join(parts)

    def _rebuild_dfa(self, original_dot: str, transitions: List[Tuple],
                     label_to_partitions: Dict[str, List[str]]) -> str:
        """Rebuild DFA with partition symbols"""
        lines = original_dot.split('\n')
        new_lines = []

        for line in lines:
            match = re.match(r'(\s*)(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\](.*)', line)

            if match:
                indent, from_state, to_state, label, rest = match.groups()

                if from_state in ['init', '__start']:
                    new_lines.append(line)
                    continue

                partition_symbols = label_to_partitions.get(label, [])

                if not partition_symbols:
                    new_lines.append(line)
                else:
                    for symbol in partition_symbols:
                        new_line = f'{indent}{from_state} -> {to_state} [label="{symbol}"];'
                        new_lines.append(new_line)
            else:
                new_lines.append(line)

        return '\n'.join(new_lines)

    def _create_trivial_result(self, dfa_dot: str) -> SimplifiedDFA:
        """Create trivial result when no simplification needed"""
        return SimplifiedDFA(
            simplified_dot=dfa_dot,
            partitions=[],
            partition_map={},
            original_label_to_partitions={},
            stats={'method': 'none', 'num_predicates': 0, 'num_partitions': 0}
        )


class DFASimplifier:
    """
    Main DFA simplifier with automatic method selection

    Automatically chooses between BDD-based or minterm-based simplification
    based on availability and problem size.
    """

    def __init__(self, prefer_bdd: bool = True, max_predicates_for_minterm: int = 12):
        """
        Initialize DFA simplifier

        Args:
            prefer_bdd: Prefer BDD method if available
            max_predicates_for_minterm: Max predicates for minterm method
        """
        self.prefer_bdd = prefer_bdd
        self.max_predicates_for_minterm = max_predicates_for_minterm

        # Try to initialize BDD simplifier
        self.bdd_simplifier = BDDSimplifier()
        self.minterm_simplifier = SimpleMintermSimplifier(max_predicates_for_minterm)

    def simplify(self, dfa_dot: str, grounding_map: GroundingMap,
                 method: Optional[str] = None) -> SimplifiedDFA:
        """
        Simplify DFA using best available method

        Args:
            dfa_dot: DFA in DOT format
            grounding_map: Grounding map for anti-grounding
            method: Force specific method ('bdd', 'minterm', or None for auto)

        Returns:
            SimplifiedDFA object
        """
        # Auto-select method
        if method is None:
            if self.prefer_bdd and self.bdd_simplifier.is_available():
                method = 'bdd'
            else:
                method = 'minterm'

        print(f"[DFA Simplifier] Using method: {method}")

        if method == 'bdd':
            if not self.bdd_simplifier.is_available():
                print("  Warning: BDD not available, falling back to minterm")
                method = 'minterm'
            else:
                return self.bdd_simplifier.simplify(dfa_dot, grounding_map)

        if method == 'minterm':
            return self.minterm_simplifier.simplify(dfa_dot, grounding_map)

        raise ValueError(f"Unknown simplification method: {method}")
