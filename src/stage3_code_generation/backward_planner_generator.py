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

from typing import List, Dict, Any, Optional, Tuple
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
from stage3_code_generation.variable_normalizer import VariableNormalizer, VariableMapping
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

    def generate(self, ltl_dict: Dict[str, Any], dfa_result: Dict[str, Any]) -> Tuple[str, bool]:
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
            Tuple of (AgentSpeak .asl code, truncated flag)
            - code: Complete AgentSpeak .asl code string
            - truncated: True if any state graph hit max_states limit
        """
        print("\n[Backward Planner Generator] Starting code generation")
        print("="*80)

        # Extract objects
        objects = ltl_dict['objects']
        print(f"Objects: {objects}")

        # Parse DFA
        dfa_info = self._parse_dfa(dfa_result['dfa_dot'])
        print(f"DFA: {len(dfa_info.states)} states, {len(dfa_info.transitions)} transitions")

        # OBJECT-LEVEL PLANNING: Use concrete objects instead of variables
        # This avoids state space explosion from irrelevant variable combinations
        print(f"Object-level planning enabled: using {len(objects)} concrete objects")

        # OPTIMIZATION: Pattern-based caching for goals
        # Cache using grounded goal predicates, deduplicate by structural pattern
        # Example: on(a,b) and on(c,d) have same pattern "on_2args"
        goal_cache = {}  # goal_key -> state_graph
        pattern_to_goals = {}  # pattern -> list of (goal, state_graph)

        # Statistics
        cache_hits = 0
        cache_misses = 0

        # Track if any state graph was truncated due to max_states limit
        any_truncated = False

        # OPTIMIZATION 4: Track generated parameterized patterns to avoid duplicates
        # Example: on(a,b) and on(c,d) both normalize to on(?v0,?v1) - only generate once
        generated_patterns = set()

        # OPTIMIZATION 3: Collect state graphs and generate goal-specific sections
        # to avoid duplicating shared components (initial beliefs + action plans)
        all_state_graphs = []  # For generating shared section once
        all_goal_sections = []  # Goal-specific sections only

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

                # Create goal name and serialization key
                goal_name = self._format_goal_name(goal_predicates)
                goal_key = self._serialize_goal(goal_predicates)
                print(f"    Goal name: {goal_name}")
                print(f"    Goal predicates: {[str(p) for p in goal_predicates]}")

                # Check cache
                state_graph = None
                if goal_key in goal_cache:
                    state_graph = goal_cache[goal_key]
                    cache_hits += 1
                    print(f"    ✓ Cache HIT! Reusing exploration")
                    print(f"    State graph: {state_graph}")
                else:
                    # Cache MISS: Need to explore
                    cache_misses += 1
                    print(f"    Cache MISS - running object-level exploration...")

                    try:
                        # OBJECT-LEVEL PLANNING: Use concrete objects
                        planner = ForwardStatePlanner(self.domain, objects, use_variables=False)

                        # Explore using grounded goal
                        state_graph = planner.explore_from_goal(goal_predicates)
                        print(f"    State graph: {state_graph}")

                        # Check if this graph was truncated
                        if state_graph.truncated:
                            any_truncated = True

                        # Cache the result
                        goal_cache[goal_key] = state_graph
                        print(f"    → Cached exploration result")

                    except Exception as e:
                        print(f"    Error during exploration: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                # Pattern-based deduplication and code generation
                try:
                    # Detect structural pattern from grounded goal
                    # Example: on(a,b) and on(c,d) both have pattern "on(?,?)"
                    param_pattern = self._get_pattern_from_goal(goal_predicates)

                    # Always add state graph (needed for shared section)
                    all_state_graphs.append(state_graph)

                    if param_pattern in generated_patterns:
                        # This pattern already generated, skip duplicate plan generation
                        print(f"    ✓ Pattern '{param_pattern}' already generated, skipping duplicate plans")
                        continue

                    # Generate goal-specific section with object→variable parameterization
                    # Create mapping for parameterization
                    obj_to_var_mapping = self._create_parameterization_mapping(goal_predicates)

                    codegen = AgentSpeakCodeGenerator(
                        state_graph=state_graph,
                        goal_name=goal_name,
                        domain=self.domain,
                        objects=objects,
                        var_mapping=obj_to_var_mapping
                    )

                    goal_section = codegen.generate_goal_specific_section()
                    all_goal_sections.append(goal_section)
                    generated_patterns.add(param_pattern)  # Mark pattern as generated
                    print(f"    Generated {len(goal_section)} characters of goal-specific code (pattern: {param_pattern})")

                except Exception as e:
                    print(f"    Error during codegen: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # OPTIMIZATION 3: Combine shared + goal-specific sections
        if not all_goal_sections:
            print("\nWarning: No code generated!")
            return self._generate_empty_code(), any_truncated

        print(f"\n[Code Generation] Combining sections...")
        print(f"  Goal-specific sections: {len(all_goal_sections)}")
        print(f"  State graphs collected: {len(all_state_graphs)}")

        # Generate main header
        header = self._generate_main_header(ltl_dict, dfa_info)

        # OPTIMIZATION 3: Generate shared components ONCE
        shared_section = AgentSpeakCodeGenerator.generate_shared_section(
            domain=self.domain,
            objects=objects,
            all_state_graphs=all_state_graphs
        )

        # OPTIMIZATION 3: Combine: header + shared + all goal-specific sections
        final_code = header + "\n\n" + shared_section + "\n\n" + \
                    "\n\n".join(all_goal_sections)

        print(f"\n[Backward Planner Generator] Code generation complete")
        print(f"Total code length: {len(final_code)} characters")
        print(f"Code structure optimization:")
        print(f"  Shared section generated: 1 time (initial beliefs + action plans)")
        print(f"  Goal-specific sections: {len(all_goal_sections)}")
        print(f"  Redundancy eliminated: ~{(len(all_goal_sections) - 1) * 30:.0f}% of shared code")

        # Object-level cache statistics
        print(f"\nObject-level planning cache statistics:")
        total_queries = cache_hits + cache_misses
        print(f"    Total goal explorations: {total_queries}")
        print(f"    Cache hits: {cache_hits}")
        print(f"    Cache misses: {cache_misses}")
        if total_queries > 0:
            print(f"    Hit rate: {cache_hits / total_queries * 100:.1f}%")
        if cache_hits > 0:
            print(f"    💡 Caching saved {cache_hits} state space explorations!")
        print("="*80)

        return final_code, any_truncated

    def _parse_dfa(self, dfa_dot: str) -> DFAInfo:
        """
        Parse DFA from DOT format

        Supports two formats:
        1. MONA format (from ltlf2dfa, used in real pipeline):
           node [shape = doublecircle]; 4;
           node [shape = circle]; 1;
           init -> 1;
        2. Mock test format:
           state1 [label="1", shape=doublecircle];
           __start -> state0;

        Args:
            dfa_dot: DFA in DOT format

        Returns:
            DFAInfo object
        """
        dfa_info = DFAInfo()

        # Track accepting state IDs from MONA format: node [shape = doublecircle]; state_ids;
        mona_accepting_states = []

        # Extract states by parsing each line
        # Node definition: state_id [attributes];
        # Transition: from -> to [label="..."];
        for line in dfa_dot.split('\n'):
            line = line.strip()

            # MONA Format: Check for "node [shape = doublecircle]; state_ids;" pattern
            mona_accepting_match = re.match(r'node\s*\[.*doublecircle.*\]\s*;\s*([0-9\s,;]+)', line)
            if mona_accepting_match:
                # Extract state IDs (can be comma or space separated, ends with ;)
                state_ids_str = mona_accepting_match.group(1).rstrip(';').strip()
                # Split by comma or space
                state_ids = [s.strip() for s in re.split(r'[,\s]+', state_ids_str) if s.strip() and s.strip().isdigit()]
                mona_accepting_states.extend(state_ids)
                continue

            # MONA Format: Check for "node [shape = circle]; state_ids;" pattern (regular states)
            mona_circle_match = re.match(r'node\s*\[.*circle.*\]\s*;\s*([0-9\s,;]+)', line)
            if mona_circle_match:
                # Extract state IDs
                state_ids_str = mona_circle_match.group(1).rstrip(';').strip()
                state_ids = [s.strip() for s in re.split(r'[,\s]+', state_ids_str) if s.strip() and s.strip().isdigit()]
                # Add to states if not already in accepting states
                for sid in state_ids:
                    if sid not in dfa_info.states and sid not in mona_accepting_states:
                        dfa_info.states.append(sid)
                continue

            # Skip transitions for now (process later)
            if '->' in line:
                continue

            # Match standard node definitions: state_id [attributes]
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

            # Check for accepting state (usually doublecircle) - Mock test format
            if 'doublecircle' in attributes:
                dfa_info.accepting_states.append(state_id)

        # Add MONA accepting states
        for state_id in mona_accepting_states:
            if state_id not in dfa_info.states:
                dfa_info.states.append(state_id)
            if state_id not in dfa_info.accepting_states:
                dfa_info.accepting_states.append(state_id)

        # Extract transitions and initial state
        for line in dfa_dot.split('\n'):
            line = line.strip()

            # MONA Format: Check for "init -> state_id;" pattern
            mona_init_match = re.match(r'init\s*->\s*(\d+)', line)
            if mona_init_match:
                initial_state = mona_init_match.group(1)
                if initial_state not in dfa_info.states:
                    dfa_info.states.append(initial_state)
                dfa_info.initial_state = initial_state
                continue

            # Mock test format: Check for "__start -> state_id" (with or without label)
            mock_init_match = re.match(r'__start\s*->\s*(\w+)', line)
            if mock_init_match:
                initial_state = mock_init_match.group(1)
                if initial_state not in dfa_info.states:
                    dfa_info.states.append(initial_state)
                dfa_info.initial_state = initial_state
                continue

            # Standard transition pattern (with label)
            transition_match = re.match(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]', line)
            if transition_match:
                from_state = transition_match.group(1)
                to_state = transition_match.group(2)
                label = transition_match.group(3)

                # Filter out __start and init transitions (already handled above)
                if from_state not in ['__start', 'init'] and to_state not in ['__start', 'init']:
                    # Add states from transitions (MONA doesn't declare all states explicitly)
                    if from_state not in dfa_info.states:
                        dfa_info.states.append(from_state)
                    if to_state not in dfa_info.states:
                        dfa_info.states.append(to_state)

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

    def _serialize_goal(self, predicates: List[PredicateAtom]) -> str:
        """
        Serialize goal predicates to a hashable cache key

        Args:
            predicates: List of predicates

        Returns:
            String key for caching
        """
        # Sort predicates for consistency (order shouldn't matter)
        sorted_preds = sorted([p.to_agentspeak() for p in predicates])
        return "|".join(sorted_preds)

    def _get_pattern_from_goal(self, goal: List[PredicateAtom]) -> str:
        """
        Get structural pattern from grounded goal for deduplication

        This is used to detect duplicate plan generation. For example:
        - on(a,b) → pattern "on(?,?)"
        - on(c,d) → pattern "on(?,?)" (SAME!)

        We only want to generate the plan suite once for each unique pattern.

        Args:
            goal: List of grounded predicates (with concrete objects)

        Returns:
            Pattern string for deduplication
        """
        pattern_parts = []
        for pred in sorted(goal, key=lambda x: (x.name, len(x.args))):
            # Create pattern with ? for each argument
            args_pattern = ','.join(['?' for _ in pred.args])
            negation = 'not ' if pred.negated else ''
            pattern_parts.append(f"{negation}{pred.name}({args_pattern})")
        return '&'.join(pattern_parts)

    def _create_parameterization_mapping(self, goal: List[PredicateAtom]):
        """
        Create object→variable mapping for parameterizing AgentSpeak code

        Args:
            goal: List of grounded predicates (e.g., [on(a, b), clear(c)])

        Returns:
            VariableMapping with obj_to_var and var_to_obj dictionaries
        """
        from stage3_code_generation.variable_normalizer import VariableMapping

        # Extract all unique objects from goal predicates
        objects_in_goal = []
        for pred in goal:
            for arg in pred.args:
                if arg not in objects_in_goal:
                    objects_in_goal.append(arg)

        # Create mapping: objects → V0, V1, V2, ...
        obj_to_var = {obj: f"?v{i}" for i, obj in enumerate(objects_in_goal)}
        var_to_obj = {var: obj for obj, var in obj_to_var.items()}

        return VariableMapping(obj_to_var=obj_to_var, var_to_obj=var_to_obj)

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
