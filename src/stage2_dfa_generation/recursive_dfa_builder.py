"""
Stage 2: Recursive DFA Generation from LTLf Specifications

This module implements recursive DFA generation that breaks down high-level LTLf goals
into sub-goals until reaching physical actions (domain actions).

Architecture:
    Input: LTLf specification with formulas like F(on(a,b))
    Process: Recursively decompose goals → subgoals → physical actions
    Output: Collection of DFAs, each modeling a subgoal or action sequence

Key Concepts:
    - High-level goals: F(on(a,b)), G(clear(c)) - need decomposition
    - Physical actions: pickup, putdown, stack, unstack - terminal nodes
    - Recursive decomposition: Continue until all paths reach physical actions
    - DFA reuse: If a DFA already models a subgoal, reuse it
"""

# Setup MONA path before importing ltlf2dfa
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from setup_mona_path import setup_mona
setup_mona()

from typing import Dict, List, Any, Tuple, Set
import re
from ltlf2dfa.parser.ltlf import LTLfParser
from dataclasses import dataclass, asdict
import json


@dataclass
class DFANode:
    """
    Represents a node in the recursive DFA decomposition tree

    Each node represents either:
    - A high-level goal (needs further decomposition)
    - A physical action (terminal node)
    """
    goal_formula: str  # Original LTLf formula (e.g., "F(on(a,b))")
    propositional_formula: str  # Encoded formula (e.g., "F(on_a_b)")
    dfa_dot: str  # DOT representation of DFA
    predicate_mappings: Dict[str, str]  # Predicate → proposition mapping
    is_physical_action: bool  # True if this represents a direct physical action
    subgoals: List[str]  # List of subgoal formulas identified from transitions
    depth: int  # Depth in decomposition tree


@dataclass
class RecursiveDFAResult:
    """Complete result of recursive DFA generation"""
    root_formula: str  # Top-level goal
    all_dfas: List[DFANode]  # All generated DFAs (breadth-first order)
    physical_actions: List[str]  # Set of physical actions identified
    dfa_graph: Dict[str, List[str]]  # Dependency graph: formula → [subgoal formulas]
    max_depth: int  # Maximum decomposition depth reached

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "root_formula": self.root_formula,
            "all_dfas": [asdict(dfa) for dfa in self.all_dfas],
            "physical_actions": self.physical_actions,
            "dfa_graph": self.dfa_graph,
            "max_depth": self.max_depth
        }


class RecursiveDFABuilder:
    """
    Builds DFAs recursively from LTLf specifications

    Process:
    1. Convert top-level LTLf goal to DFA
    2. Analyze DFA transitions to identify subgoals
    3. For each subgoal:
       - If it's a physical action → terminal (no further decomposition)
       - If it's already modeled → reuse existing DFA
       - Otherwise → recursively generate DFA
    4. Continue until all paths reach physical actions
    """

    def __init__(self, domain_actions: List[str]):
        """
        Initialize builder

        Args:
            domain_actions: List of physical actions in domain (e.g., ['pickup', 'putdown'])
        """
        self.ltlf_parser = LTLfParser()
        self.domain_actions = set(domain_actions)

        # Cache of generated DFAs to avoid regeneration
        self.dfa_cache: Dict[str, DFANode] = {}  # formula → DFANode

        # Track decomposition
        self.all_dfas: List[DFANode] = []
        self.dfa_graph: Dict[str, List[str]] = {}  # formula → subgoals

    def build(self, ltl_spec: Any) -> RecursiveDFAResult:
        """
        Build recursive DFA decomposition using DFS search strategy

        DFS Algorithm:
        1. Generate DFA for root formula
        2. Make it the current searching DFA
        3. For each transition in current DFA:
           - Check if transition target is physical action → terminal
           - Check if DFA already exists for subgoal → reuse
           - Otherwise → recursively generate DFA (DFS deeper)
        4. Backtrack when all transitions of current DFA are resolved
        5. Continue until all paths reach physical actions or existing DFAs

        Args:
            ltl_spec: LTLSpecification object with formulas

        Returns:
            RecursiveDFAResult containing all generated DFAs and metadata
        """
        # Clear state
        self.dfa_cache.clear()
        self.all_dfas.clear()
        self.dfa_graph.clear()

        # Extract formula strings
        if hasattr(ltl_spec, 'formulas'):
            formula_strings = [f.to_string() for f in ltl_spec.formulas]
        else:
            raise ValueError("ltl_spec must have 'formulas' attribute")

        # Combine multiple formulas with conjunction
        if len(formula_strings) == 0:
            raise ValueError("No LTLf formulas provided")
        elif len(formula_strings) == 1:
            root_formula = formula_strings[0]
        else:
            root_formula = " & ".join(f"({f})" for f in formula_strings)

        print(f"\n[DFS] Starting DFA generation from root: {root_formula}")
        print("="*80)

        # Start DFS decomposition from root
        self._dfs_decompose(root_formula, depth=0)

        # Collect physical actions
        physical_actions = [
            dfa.goal_formula for dfa in self.all_dfas
            if dfa.is_physical_action
        ]

        # Calculate max depth
        max_depth = max((dfa.depth for dfa in self.all_dfas), default=0)

        print(f"\n[DFS] Decomposition complete:")
        print(f"  Total DFAs generated: {len(self.all_dfas)}")
        print(f"  Physical actions found: {len(physical_actions)}")
        print(f"  Max depth: {max_depth}")
        print("="*80)

        return RecursiveDFAResult(
            root_formula=root_formula,
            all_dfas=self.all_dfas,
            physical_actions=physical_actions,
            dfa_graph=self.dfa_graph,
            max_depth=max_depth
        )

    def _dfs_decompose(self, formula: str, depth: int) -> DFANode:
        """
        DFS-based recursive DFA decomposition

        DFS Process:
        1. Check if DFA already exists (cached) → reuse
        2. Generate DFA for current formula (make it current searching DFA)
        3. Analyze all transitions in the DFA
        4. For each transition trigger (subgoal):
           a. If physical action → mark terminal, continue
           b. If already exists in cache → reuse, continue
           c. Otherwise → DFS deeper (recursive call)
        5. Backtrack when all transitions resolved

        Args:
            formula: LTLf formula to decompose
            depth: Current depth in DFS tree

        Returns:
            DFANode representing this formula
        """
        indent = "  " * depth

        # Check cache first - DFA already exists, reuse it
        if formula in self.dfa_cache:
            print(f"{indent}[DFS] REUSE cached DFA: {formula}")
            return self.dfa_cache[formula]

        # Check if this is a physical action (terminal node)
        is_physical = self._is_physical_action(formula)
        if is_physical:
            print(f"{indent}[DFS] TERMINAL physical action: {formula}")

        # Generate DFA for this formula (current searching DFA)
        print(f"{indent}[DFS] Generating DFA for: {formula}")
        dfa_dot, predicate_mappings, propositional_formula = self._generate_dfa(formula)

        # Extract subgoals from DFA transitions
        subgoals = []
        if not is_physical:
            subgoals = self._extract_subgoals(dfa_dot, predicate_mappings)
            if subgoals:
                print(f"{indent}[DFS] Found {len(subgoals)} subgoals in transitions:")
                for sg in subgoals:
                    print(f"{indent}      - {sg}")
            else:
                print(f"{indent}[DFS] No subgoals found (may be terminal state DFA)")

        # Create DFA node
        dfa_node = DFANode(
            goal_formula=formula,
            propositional_formula=propositional_formula,
            dfa_dot=dfa_dot,
            predicate_mappings=predicate_mappings,
            is_physical_action=is_physical,
            subgoals=subgoals,
            depth=depth
        )

        # Cache and store BEFORE processing subgoals (prevents cycles)
        self.dfa_cache[formula] = dfa_node
        self.all_dfas.append(dfa_node)
        self.dfa_graph[formula] = subgoals

        # DFS: Process each subgoal depth-first
        for i, subgoal in enumerate(subgoals, 1):
            print(f"{indent}[DFS] Processing subgoal {i}/{len(subgoals)}: {subgoal}")
            self._dfs_decompose(subgoal, depth=depth + 1)
            print(f"{indent}[DFS] Backtrack from: {subgoal}")

        if subgoals:
            print(f"{indent}[DFS] All subgoals of '{formula}' resolved")

        return dfa_node

    def _is_physical_action(self, formula: str) -> bool:
        """
        Check if formula represents a physical action

        Args:
            formula: LTLf formula string

        Returns:
            True if formula is a direct physical action
        """
        # Extract predicate name from formula (e.g., "pickup(a)" → "pickup")
        predicate_pattern = re.compile(r'(\w+)\(')
        match = predicate_pattern.search(formula)

        if match:
            predicate_name = match.group(1)
            return predicate_name in self.domain_actions

        return False

    def _generate_dfa(self, formula: str) -> Tuple[str, Dict[str, str], str]:
        """
        Generate DFA for a single formula

        Args:
            formula: LTLf formula

        Returns:
            Tuple of (dfa_dot, predicate_mappings, propositional_formula)
        """
        # Use ltlf_to_dfa module's encoding logic
        from stage2_dfa_generation.ltlf_to_dfa import PredicateToProposition

        encoder = PredicateToProposition()
        propositional_formula = encoder.convert_formula(formula)

        try:
            formula_obj = self.ltlf_parser(propositional_formula)
            dfa_dot = formula_obj.to_dfa()
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert LTLf to DFA.\n"
                f"Original formula: {formula}\n"
                f"Propositional formula: {propositional_formula}\n"
                f"Error: {str(e)}"
            ) from e

        return dfa_dot, encoder.get_mapping(), propositional_formula

    def _extract_subgoals(self, dfa_dot: str, predicate_mappings: Dict[str, str]) -> List[str]:
        """
        Extract subgoals from DFA transition labels

        Analyzes transition labels to identify predicates that need to be achieved.
        These become subgoals for further decomposition.

        Args:
            dfa_dot: DFA in DOT format
            predicate_mappings: Map of predicate → proposition

        Returns:
            List of subgoal formulas
        """
        subgoals = set()

        # Parse transition labels from DOT
        # Format: "1 -> 2 [label="on_a_b & clear_c"];"
        label_pattern = re.compile(r'\[label="([^"]+)"\]')

        for line in dfa_dot.split('\n'):
            if '->' in line and '[label=' in line:
                match = label_pattern.search(line)
                if match:
                    label = match.group(1)
                    # Extract propositions from label (handle &, |, ~, true)
                    props = self._parse_label_propositions(label)

                    # Convert propositions back to predicates
                    for prop in props:
                        if prop in predicate_mappings.values():
                            # Find original predicate
                            reverse_map = {v: k for k, v in predicate_mappings.items()}
                            predicate = reverse_map.get(prop)
                            if predicate:
                                # Wrap in F() to make it a goal
                                subgoals.add(f"F({predicate})")

        return list(subgoals)

    def _parse_label_propositions(self, label: str) -> Set[str]:
        """
        Parse propositions from a transition label

        Args:
            label: Transition label (e.g., "on_a_b & clear_c" or "~on_a_b | clear_c")

        Returns:
            Set of positive propositions (without negations)
        """
        # Remove negations, split by operators
        cleaned = label.replace('~', ' ').replace('&', ' ').replace('|', ' ')
        tokens = cleaned.split()

        # Filter out 'true', 'false', and keep only valid propositions
        propositions = set()
        for token in tokens:
            token = token.strip()
            if token and token not in ['true', 'false', '(', ')']:
                # Valid proposition (starts with letter)
                if token[0].isalpha():
                    propositions.add(token)

        return propositions


def test_recursive_builder():
    """Test the recursive DFA builder"""

    # Mock LTL specification
    class MockFormula:
        def __init__(self, formula_str):
            self.formula_str = formula_str

        def to_string(self):
            return self.formula_str

    class MockLTLSpec:
        def __init__(self, formulas):
            self.formulas = [MockFormula(f) for f in formulas]

    print("="*80)
    print("RECURSIVE DFA BUILDER TEST")
    print("="*80)
    print()

    # Test with blocksworld domain
    domain_actions = ['pickup', 'putdown', 'stack', 'unstack']
    builder = RecursiveDFABuilder(domain_actions)

    # Test case: Stack block A on block B
    spec = MockLTLSpec(["F(on(a, b))"])

    print("Test: Stack block A on block B")
    print(f"Input formula: {spec.formulas[0].to_string()}")
    print()

    try:
        result = builder.build(spec)

        print(f"✓ Recursive decomposition complete")
        print(f"  Root formula: {result.root_formula}")
        print(f"  Total DFAs generated: {len(result.all_dfas)}")
        print(f"  Max decomposition depth: {result.max_depth}")
        print(f"  Physical actions identified: {result.physical_actions}")
        print()

        print("DFA Decomposition Tree:")
        print("-" * 80)
        for i, dfa in enumerate(result.all_dfas, 1):
            indent = "  " * dfa.depth
            action_marker = " [PHYSICAL ACTION]" if dfa.is_physical_action else ""
            print(f"{indent}{i}. {dfa.goal_formula}{action_marker}")
            if dfa.subgoals:
                print(f"{indent}   Subgoals: {dfa.subgoals}")
        print()

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_recursive_builder()
