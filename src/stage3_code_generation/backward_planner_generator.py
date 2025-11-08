"""
Backward Planner Generator

Main entry point for Stage 3: Non-LLM AgentSpeak code generation using
backward planning.

This module integrates all components:
1. Parse DFA transitions
2. Extract goal predicates from transition labels
3. Run forward planning for each goal
4. Generate AgentSpeak code from state graphs

Replaces the LLM-based AgentSpeakGenerator.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import sys

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import PredicateAtom, StateGraph
from stage3_code_generation.forward_planner import ForwardStatePlanner
from stage3_code_generation.boolean_expression_parser import BooleanExpressionParser
from stage3_code_generation.agentspeak_codegen import AgentSpeakCodeGenerator
from utils.pddl_parser import PDDLDomain
from stage1_interpretation.grounding_map import GroundingMap


class DFAInfo:
    """
    Parsed DFA information

    Attributes:
        states: List of state IDs
        transitions: List of (from_state, to_state, label) tuples
        initial_state: Initial state ID
        accepting_states: List of accepting state IDs
    """

    def __init__(self):
        self.states: List[str] = []
        self.transitions: List[tuple] = []
        self.initial_state: Optional[str] = None
        self.accepting_states: List[str] = []


class BackwardPlannerGenerator:
    """
    Stage 3: Generate AgentSpeak code using backward planning

    Replaces LLM-based generation with programmatic approach:
    - Parse DFA structure
    - Extract goals from transition labels
    - Forward planning from each goal
    - AgentSpeak code generation
    """

    def __init__(self, domain: PDDLDomain, grounding_map: GroundingMap):
        """
        Initialize backward planner generator

        Args:
            domain: PDDL domain
            grounding_map: Grounding map for anti-grounding symbols
        """
        self.domain = domain
        self.grounding_map = grounding_map

    def generate(self, ltl_dict: Dict[str, Any], dfa_result: Dict[str, Any]) -> str:
        """
        Generate AgentSpeak code from DFA

        Args:
            ltl_dict: LTL specification dict with:
                - 'objects': List of objects
                - 'formulas_string': List of formula strings
                - 'grounding_map': GroundingMap
            dfa_result: DFA result dict with:
                - 'formula': Formula string
                - 'dfa_dot': DOT format string
                - other metadata

        Returns:
            Complete AgentSpeak .asl code
        """
        print("\n[Backward Planner Generator] Starting code generation")
        print("="*80)

        # Extract objects
        objects = ltl_dict['objects']
        print(f"Objects: {objects}")

        # Parse DFA
        dfa_info = self._parse_dfa(dfa_result['dfa_dot'])
        print(f"DFA: {len(dfa_info.states)} states, {len(dfa_info.transitions)} transitions")

        # Generate code for each transition
        all_code_sections = []

        for i, (from_state, to_state, label) in enumerate(dfa_info.transitions):
            print(f"\n[Transition {i+1}/{len(dfa_info.transitions)}] {from_state} --[{label}]-> {to_state}")

            # Parse transition label to get goal predicates
            try:
                goal_disjuncts = self._parse_transition_label(label)
            except Exception as e:
                print(f"  Warning: Failed to parse label '{label}': {e}")
                print(f"  Skipping this transition")
                continue

            print(f"  DNF: {len(goal_disjuncts)} disjunct(s)")

            # Generate code for each disjunct
            for j, goal_predicates in enumerate(goal_disjuncts):
                print(f"  Disjunct {j+1}: {[str(p) for p in goal_predicates]}")

                if not goal_predicates:
                    print(f"    Skipping empty disjunct")
                    continue

                # Create goal name
                goal_name = self._format_goal_name(goal_predicates)
                print(f"    Goal name: {goal_name}")

                # Run forward planning
                try:
                    planner = ForwardStatePlanner(self.domain, objects)
                    state_graph = planner.explore_from_goal(goal_predicates)
                    print(f"    State graph: {state_graph}")

                    # Generate AgentSpeak code
                    codegen = AgentSpeakCodeGenerator(
                        state_graph=state_graph,
                        goal_name=goal_name,
                        domain=self.domain,
                        objects=objects
                    )

                    code = codegen.generate()
                    all_code_sections.append(code)
                    print(f"    Generated {len(code)} characters of code")

                except Exception as e:
                    print(f"    Error during planning/codegen: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Combine all code sections
        if not all_code_sections:
            print("\nWarning: No code generated!")
            return self._generate_empty_code()

        # Add main header
        header = self._generate_main_header(ltl_dict, dfa_info)

        final_code = header + "\n\n" + "\n\n/* ========== Next Goal ========== */\n\n".join(all_code_sections)

        print(f"\n[Backward Planner Generator] Code generation complete")
        print(f"Total code length: {len(final_code)} characters")
        print("="*80)

        return final_code

    def _parse_dfa(self, dfa_dot: str) -> DFAInfo:
        """
        Parse DFA from DOT format

        Args:
            dfa_dot: DFA in DOT format

        Returns:
            DFAInfo object
        """
        dfa_info = DFAInfo()

        # Extract states by parsing each line
        # Node definition: state_id [attributes];
        # Transition: from -> to [label="..."];
        # We only want node definitions, not transitions
        for line in dfa_dot.split('\n'):
            line = line.strip()

            # Skip transitions (contain ->)
            if '->' in line:
                continue

            # Match node definitions: state_id [attributes]
            state_match = re.match(r'(\w+)\s*\[([^\]]*)\]', line)
            if not state_match:
                continue

            state_id = state_match.group(1)
            attributes = state_match.group(2)

            # Skip node/graph/edge declarations and __start
            if state_id in ['digraph', 'graph', 'node', 'edge', '__start']:
                continue

            dfa_info.states.append(state_id)

            # Check for initial state (usually has different shape or label)
            if 'init' in attributes.lower():
                dfa_info.initial_state = state_id

            # Check for accepting state (usually doublecircle)
            if 'doublecircle' in attributes:
                dfa_info.accepting_states.append(state_id)

        # Extract transitions
        transition_pattern = r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]'
        for match in re.finditer(transition_pattern, dfa_dot):
            from_state = match.group(1)
            to_state = match.group(2)
            label = match.group(3)

            # Filter out __start transitions
            if from_state != '__start' and to_state != '__start':
                dfa_info.transitions.append((from_state, to_state, label))

        return dfa_info

    def _parse_transition_label(self, label: str) -> List[List[PredicateAtom]]:
        """
        Parse transition label to extract goal predicates

        Uses BooleanExpressionParser to convert to DNF.

        Args:
            label: Transition label (e.g., "on_a_b & clear_c")

        Returns:
            List of goal predicate lists (DNF form)
        """
        parser = BooleanExpressionParser(self.grounding_map)
        dnf = parser.parse(label)
        return dnf

    def _format_goal_name(self, predicates: List[PredicateAtom]) -> str:
        """
        Format goal name from predicates

        Args:
            predicates: List of predicates

        Returns:
            Goal name string (e.g., "on(a, b)")
        """
        if len(predicates) == 1:
            # Single predicate: use as-is
            return predicates[0].to_agentspeak()
        else:
            # Multiple predicates: combine
            names = [p.to_agentspeak().replace(" ", "_").replace("(", "_").replace(")", "").replace(",", "") for p in predicates]
            return "_and_".join(names)

    def _generate_main_header(self, ltl_dict: Dict[str, Any], dfa_info: DFAInfo) -> str:
        """Generate main file header"""
        formulas_str = ", ".join(str(f) for f in ltl_dict.get('formulas_string', []))

        return f"""/* AgentSpeak Plan Library
 * Generated by Backward Planning (non-LLM)
 *
 * LTLf Specification: {formulas_str}
 * Objects: {', '.join(ltl_dict['objects'])}
 *
 * DFA Information:
 *   States: {len(dfa_info.states)}
 *   Transitions: {len(dfa_info.transitions)}
 *   Initial state: {dfa_info.initial_state}
 *   Accepting states: {', '.join(dfa_info.accepting_states)}
 */"""

    def _generate_empty_code(self) -> str:
        """Generate empty placeholder code"""
        return """/* AgentSpeak Plan Library
 * WARNING: No code generated - check DFA transitions
 */

+!placeholder : true <-
    .print("No plans generated").
"""


# Test function
def test_backward_planner_generator():
    """Test backward planner generator with mock data"""
    print("="*80)
    print("Testing Backward Planner Generator")
    print("="*80)

    # Load domain
    from utils.pddl_parser import PDDLParser

    domain_file = Path(__file__).parent.parent / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        print("Skipping test")
        return

    domain = PDDLParser.parse_domain(str(domain_file))

    # Create mock grounding map
    from stage1_interpretation.grounding_map import GroundingMap
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])

    # Create mock DFA (simple example)
    mock_dfa_dot = """
    digraph G {
        __start [shape=none];
        s0 [shape=circle];
        s1 [shape=doublecircle];
        __start -> s0;
        s0 -> s1 [label="on_a_b"];
    }
    """

    # Create mock LTL dict
    ltl_dict = {
        'objects': ['a', 'b', 'c'],
        'formulas_string': ['F(on_a_b)'],
        'grounding_map': gmap
    }

    dfa_result = {
        'formula': 'F(on_a_b)',
        'dfa_dot': mock_dfa_dot
    }

    # Generate code
    generator = BackwardPlannerGenerator(domain, gmap)
    code = generator.generate(ltl_dict, dfa_result)

    print("\n" + "="*80)
    print("Generated AgentSpeak Code:")
    print("="*80)
    print(code)
    print("="*80)

    # Save to file
    output_file = Path(__file__).parent / "test_backward_planner_output.asl"
    with open(output_file, 'w') as f:
        f.write(code)

    print(f"\nSaved to: {output_file}")
    print()


if __name__ == "__main__":
    test_backward_planner_generator()
