"""
AgentSpeak Code Generator

Generates AgentSpeak (.asl) code from state graphs produced by forward planning.

Following the design specification (Decision #12), generates:
1. PDDL action → AgentSpeak action goal plans (e.g., +!pickup(X))
   - Each includes external action calls
   - Explicit belief updates from PDDL effects
2. Goal achievement plans with:
   - Context-sensitive conditions
   - Precondition subgoals (recursive)
   - Action goal invocations
   - Recursive goal checking
3. Initial beliefs
4. Success and failure handlers

Design ref: docs/stage3_backward_planning_design.md Decision #12, Q&A #16
"""

from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import sys

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import (
    PredicateAtom, WorldState, StateTransition, StateGraph
)
from stage3_code_generation.pddl_condition_parser import PDDLConditionParser, PDDLEffectParser
from utils.pddl_parser import PDDLDomain, PDDLAction


class AgentSpeakCodeGenerator:
    """
    Generate AgentSpeak code from state graph

    Implements the design specification for AgentSpeak generation:
    - PDDL actions → AgentSpeak action goal plans
    - Goal plans with precondition subgoals
    - Belief updates
    - Jason-compatible syntax
    """

    def __init__(self, state_graph: StateGraph, goal_name: str,
                 domain: PDDLDomain, objects: List[str], var_mapping=None):
        """
        Initialize code generator

        Args:
            state_graph: State graph from forward planning (may use variables)
            goal_name: Name of the goal (e.g., "on(a, b)")
            domain: PDDL domain (for action definitions)
            objects: List of objects in domain
            var_mapping: VariableMapping for instantiating variables (optional)
                        If provided, state graph uses variables and needs instantiation
        """
        self.graph = state_graph
        self.goal_name = goal_name
        self.domain = domain
        self.objects = objects
        self.var_mapping = var_mapping  # NEW: Variable mapping for instantiation
        self.condition_parser = PDDLConditionParser()
        self.effect_parser = PDDLEffectParser()
        self.goal_plan_count = 0  # Track number of goal plans generated

    def _instantiate_predicate(self, pred: 'PredicateAtom') -> 'PredicateAtom':
        """
        Instantiate a predicate by replacing variables with objects

        If var_mapping is None, returns predicate as-is (already grounded).
        Otherwise, uses var_mapping to replace variables with concrete objects.

        Args:
            pred: PredicateAtom (may contain variables)

        Returns:
            PredicateAtom with variables instantiated
        """
        if self.var_mapping is None:
            return pred  # Already grounded

        # Use var_to_obj mapping to replace variables
        return pred.instantiate(self.var_mapping.var_to_obj)

    def _instantiate_action_args(self, args: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Instantiate action arguments by replacing variables with objects

        Args:
            args: Tuple of argument strings (may contain variables)

        Returns:
            Tuple with variables instantiated
        """
        if self.var_mapping is None:
            return args  # Already grounded

        instantiated = [self.var_mapping.var_to_obj.get(arg, arg) for arg in args]
        return tuple(instantiated)

    def _get_parameterized_goal_pattern(self) -> str:
        """
        Get parameterized goal pattern from goal state predicates

        CRITICAL FIX: For variable-level planning, goal pattern must use the SAME
        variable names as the state graph, not a separate obj_to_var mapping.

        Why? Because:
        - State predicates use variables from lifted planning (e.g., ?b, ?b1, ?b2)
        - Goal pattern must match these variable names
        - Otherwise we get mismatches like +!on(V0,V1) : holding(B)
          where V0/V1 and B come from different namespaces!

        Example (variable-level):
            Goal state: [on(?b1, ?b2)]  # From lifted planning
            Returns: "on(B1, B2)"       # Matches state variables

        Example (object-level with mapping):
            Goal state: [on(a, b)] with var_mapping {a: ?v0, b: ?v1}
            Returns: "on(V0, V1)"

        Returns:
            Parameterized goal pattern string
        """
        # Get goal predicates (may contain objects or variables)
        goal_preds = list(self.graph.goal_state.predicates)

        # NEVER use obj_to_var for variable-level planning
        # The goal state already contains the correct variables from lifted planning
        # We just need to convert them to AgentSpeak format (?b1 → B1)
        if len(goal_preds) == 1:
            # Single predicate goal
            return goal_preds[0].to_agentspeak(convert_vars=True)
        else:
            # Multiple predicates - create compound goal name
            pred_strs = [p.to_agentspeak(convert_vars=True)
                       for p in sorted(goal_preds, key=lambda x: (x.name, x.args))]
            return "_and_".join(pred_strs).replace("(", "_").replace(")", "").replace(", ", "_")

    def generate(self) -> str:
        """
        Generate complete AgentSpeak code (legacy method for single-goal case)

        Returns:
            AgentSpeak .asl file content
        """
        sections = []

        # Generate sections first (to collect statistics)
        initial_beliefs = self._generate_initial_beliefs()
        action_plans = self._generate_action_plans()
        goal_plans = self._generate_goal_plans()  # Updates self.goal_plan_count
        success_plan = self._generate_success_plan()
        failure_plan = self._generate_failure_plan()

        # Header with accurate statistics
        sections.append(self._generate_header())
        sections.append(initial_beliefs)
        sections.append(action_plans)
        sections.append(goal_plans)
        sections.append(success_plan)
        sections.append(failure_plan)

        return "\n\n".join(sections)

    def generate_goal_specific_section(self) -> str:
        """
        Generate ONLY goal-specific plans (for multi-goal optimization)

        This method generates only the parts that differ between goals:
        - Goal achievement plans
        - Success/failure handlers for this specific goal

        Returns:
            Goal-specific AgentSpeak code section
        """
        sections = []

        # Generate goal-specific parts
        goal_plans = self._generate_goal_plans()  # Updates self.goal_plan_count
        success_plan = self._generate_success_plan()
        failure_plan = self._generate_failure_plan()

        # Add comment header for this goal
        sections.append(f"/* ========== Goal: {self.goal_name} ========== */")
        sections.append("")
        sections.append(goal_plans)
        sections.append(success_plan)
        sections.append(failure_plan)

        return "\n\n".join(sections)

    @staticmethod
    def generate_shared_section(domain: PDDLDomain, objects: List[str],
                               all_state_graphs: List['StateGraph']) -> str:
        """
        Generate shared components (initial beliefs + action plans)

        These components are identical across all goals and only need to be
        generated once for the entire multi-goal AgentSpeak file.

        Args:
            domain: PDDL domain
            objects: List of objects
            all_state_graphs: All state graphs (to collect used actions)

        Returns:
            Shared AgentSpeak code section
        """
        sections = []

        # Initial beliefs
        sections.append("/* ========== Shared Components ========== */")
        sections.append("")
        sections.append("/* Initial Beliefs */")
        for obj in objects:
            sections.append(f"ontable({obj}).")
            sections.append(f"clear({obj}).")
        sections.append("handempty.")

        # Collect all used actions from all state graphs
        used_actions = set()
        for graph in all_state_graphs:
            for transition in graph.transitions:
                used_actions.add(transition.action.name)

        # Generate action plans
        sections.append("")
        sections.append("/* PDDL Action Plans (as AgentSpeak goals) */")
        sections.append("/* Each PDDL action is converted to an AgentSpeak goal plan */")
        sections.append("/* with explicit belief updates from PDDL effects */")

        # Create temporary instance to use helper methods
        temp_codegen = AgentSpeakCodeGenerator(
            state_graph=all_state_graphs[0],  # Just need any graph for helper methods
            goal_name="temp",
            domain=domain,
            objects=objects
        )

        for action in domain.actions:
            if action.name not in used_actions:
                continue

            action_plans = temp_codegen._generate_action_plan_variants(action)
            sections.extend(action_plans)

        return "\n\n".join(sections)

    def _generate_header(self) -> str:
        """Generate file header"""
        stats = self.graph.get_statistics()
        return f"""/* AgentSpeak Plan Library
 * Generated by Backward Planning (non-LLM)
 *
 * Goal: {self.goal_name}
 * Objects: {', '.join(self.objects)}
 *
 * Statistics:
 *   States: {stats['num_states']}
 *   Transitions: {stats['num_transitions']}
 *   Goal Plans: {self.goal_plan_count}
 *   Action Plans: {len(self.domain.actions)}
 */"""

    def _generate_initial_beliefs(self) -> str:
        """
        Generate initial beliefs

        For blocksworld: all blocks on table, hand empty, all clear
        Design ref: Decision #14
        """
        lines = ["/* Initial Beliefs */"]

        # Blocksworld-specific initial beliefs
        for obj in self.objects:
            lines.append(f"ontable({obj}).")
            lines.append(f"clear({obj}).")

        lines.append("handempty.")

        return "\n".join(lines)

    def _generate_action_plans(self) -> str:
        """
        Generate PDDL action → AgentSpeak action goal plans

        Design ref: Decision #12

        For each PDDL action, generates:
        +!action_name(Args) : preconditions <-
            external_action(Args);
            +effect1;
            -effect2;
            ...

        Returns:
            Action plans section
        """
        lines = ["/* PDDL Action Plans (as AgentSpeak goals) */"]
        lines.append("/* Each PDDL action is converted to an AgentSpeak goal plan */")
        lines.append("/* with explicit belief updates from PDDL effects */")

        # Collect all unique actions used in the state graph
        used_actions = set()
        for transition in self.graph.transitions:
            used_actions.add(transition.action.name)

        # Generate action plans only for used actions
        for action in self.domain.actions:
            if action.name not in used_actions:
                continue  # Skip unused actions

            # Generate all groundings for this action
            action_plans = self._generate_action_plan_variants(action)
            lines.extend(action_plans)

        return "\n\n".join(lines)

    def _generate_action_plan_variants(self, action: PDDLAction) -> List[str]:
        """
        Generate AgentSpeak plans for all groundings of a PDDL action

        Args:
            action: PDDL action

        Returns:
            List of AgentSpeak action plans
        """
        plans = []

        # Get action parameter variables
        param_vars = []
        for param in action.parameters:
            parts = param.split('-')
            if parts:
                var_name = parts[0].strip()
                param_vars.append(var_name)

        # If no parameters, generate single plan
        if not param_vars:
            plan = self._generate_single_action_plan(action, [], {})
            if plan:
                plans.append(plan)
            return plans

        # For parametric actions, generate generic plan with variables
        # (AgentSpeak will handle grounding at runtime)
        plan = self._generate_parametric_action_plan(action, param_vars)
        if plan:
            plans.append(plan)

        return plans

    def _generate_parametric_action_plan(self, action: PDDLAction, param_vars: List[str]) -> Optional[str]:
        """
        Generate parametric action plan (with variables)

        Args:
            action: PDDL action
            param_vars: Parameter variables (e.g., ["?b1", "?b2"])

        Returns:
            AgentSpeak plan string
        """
        # Convert action name to valid AgentSpeak identifier
        action_name = action.name.replace('-', '_')

        # Parse preconditions (keep variables)
        try:
            # For parametric plan, we don't bind variables yet
            # Just format the precondition string
            precond_str = action.preconditions
            if precond_str and precond_str != "none":
                # Convert PDDL preconditions to AgentSpeak format
                # This is simplified - just extract predicates
                context = self._format_pddl_condition_as_agentspeak(precond_str, param_vars)
            else:
                context = "true"
        except Exception as e:
            print(f"Warning: Failed to parse preconditions for {action.name}: {e}")
            context = "true"

        # Parse effects (keep variables)
        try:
            # Extract belief updates from effects
            belief_updates = self._extract_belief_updates_parametric(action.effects, param_vars)
        except Exception as e:
            print(f"Warning: Failed to parse effects for {action.name}: {e}")
            belief_updates = []

        # Format parameters for plan header
        if param_vars:
            # Convert ?b1 to B1, ?b2 to B2 (AgentSpeak style)
            agentspeak_vars = [v.lstrip('?').upper() if v.startswith('?') else v.upper()
                              for v in param_vars]
            params_str = ", ".join(agentspeak_vars)
            external_call = f"{action_name}_physical({params_str})"
        else:
            params_str = ""
            external_call = f"{action_name}_physical"

        # Build plan
        body_lines = [external_call]
        body_lines.extend(belief_updates)

        body = ";\n    ".join(body_lines)

        if params_str:
            plan = f"+!{action_name}({params_str}) : {context} <-\n    {body}."
        else:
            plan = f"+!{action_name} : {context} <-\n    {body}."

        return plan

    def _generate_single_action_plan(self, action: PDDLAction, args: List[str],
                                     bindings: Dict[str, str]) -> Optional[str]:
        """
        Generate single grounded action plan

        Args:
            action: PDDL action
            args: Ground arguments
            bindings: Variable bindings

        Returns:
            AgentSpeak plan string
        """
        # Convert action name
        action_name = action.name.replace('-', '_')

        # Parse preconditions with bindings
        try:
            precond_predicates = self.condition_parser.parse(action.preconditions, bindings)
            if precond_predicates:
                context = " & ".join(p.to_agentspeak() for p in precond_predicates)
            else:
                context = "true"
        except:
            context = "true"

        # Parse effects with bindings
        try:
            effect_branches = self.effect_parser.parse(action.effects, bindings)
            # Extract single branch (deterministic effects only)
            belief_updates = []
            if effect_branches and effect_branches[0]:
                for effect_atom in effect_branches[0]:
                    belief_updates.append(effect_atom.to_agentspeak())
        except:
            belief_updates = []

        # Build plan
        if args:
            args_str = ", ".join(args)
            external_call = f"{action_name}_physical({args_str})"
            plan_header = f"+!{action_name}({args_str})"
        else:
            external_call = f"{action_name}_physical"
            plan_header = f"+!{action_name}"

        body_lines = [external_call]
        body_lines.extend(belief_updates)

        body = ";\n    ".join(body_lines)

        plan = f"{plan_header} : {context} <-\n    {body}."

        return plan

    def _format_pddl_condition_as_agentspeak(self, condition: str, param_vars: List[str]) -> str:
        """
        Convert PDDL condition to AgentSpeak format

        Args:
            condition: PDDL condition string (e.g., "and (handempty) (clear ?b1)")
            param_vars: Parameter variables (e.g., ["?b1", "?b2"])

        Returns:
            AgentSpeak context string (e.g., "handempty & clear(B1)")
        """
        if not condition or condition.strip() == "none":
            return "true"

        try:
            # Create variable bindings mapping ?var -> VAR (AgentSpeak style)
            bindings = {}
            for var in param_vars:
                if var.startswith('?'):
                    agentspeak_var = var.lstrip('?').upper()
                    bindings[var] = agentspeak_var
                else:
                    bindings[var] = var.upper()

            # Parse PDDL condition to get predicates
            predicates = self.condition_parser.parse(condition, bindings)

            if not predicates:
                return "true"

            # Convert predicates to AgentSpeak format
            # PredicateAtom.to_agentspeak() gives us "name(arg1, arg2)"
            agentspeak_atoms = [pred.to_agentspeak() for pred in predicates]

            # Join with & connector
            return " & ".join(agentspeak_atoms)

        except Exception as e:
            # Fallback to simplified parsing if proper parsing fails
            print(f"Warning: Failed to parse condition '{condition}': {e}")
            print(f"Using simplified fallback")

            # Simplified fallback - replace variables only
            result = condition
            for var in param_vars:
                if var.startswith('?'):
                    result = result.replace(var, var.lstrip('?').upper())

            # Remove outer and/parentheses
            result = result.strip()
            if result.startswith('and'):
                result = result[3:].strip()
            if result.startswith('(') and result.endswith(')'):
                result = result[1:-1].strip()

            return result if result else "true"

    def _extract_belief_updates_parametric(self, effects: str, param_vars: List[str]) -> List[str]:
        """
        Extract belief updates from PDDL effects (parametric)

        Args:
            effects: PDDL effects string
            param_vars: Parameter variables

        Returns:
            List of belief update strings
        """
        # Simplified extraction
        updates = []

        # Replace variables
        effect_str = effects
        for var in param_vars:
            if var.startswith('?'):
                agentspeak_var = var.lstrip('?').upper()
                effect_str = effect_str.replace(var, agentspeak_var)

        # Very simplified parsing - just look for (not and predicates
        # Full implementation would use effect parser
        import re

        # Find all (not (predicate ...))
        not_pattern = r'\(not\s+\(([a-zA-Z][a-zA-Z0-9_-]*)\s*([^)]*)\)\)'
        for match in re.finditer(not_pattern, effect_str):
            pred_name = match.group(1)
            pred_args = match.group(2).strip()
            if pred_args:
                # Convert PDDL format (spaces) to AgentSpeak format (commas)
                pred_args = ', '.join(pred_args.split())
                updates.append(f"-{pred_name}({pred_args})")
            else:
                updates.append(f"-{pred_name}")

        # Find all positive predicates
        pos_pattern = r'\(([a-zA-Z][a-zA-Z0-9_-]*)\s+([^)]+)\)'
        for match in re.finditer(pos_pattern, effect_str):
            pred_name = match.group(1)
            if pred_name in ['and', 'or', 'not']:
                continue
            pred_args = match.group(2).strip()
            # Check if already captured as negative
            if not any(f"-{pred_name}" in u for u in updates):
                if pred_args:
                    # Convert PDDL format (spaces) to AgentSpeak format (commas)
                    pred_args = ', '.join(pred_args.split())
                    updates.append(f"+{pred_name}({pred_args})")
                else:
                    updates.append(f"+{pred_name}")

        return updates

    def _generate_goal_plans(self) -> str:
        """
        Generate plans for achieving the goal

        NOW GENERATES PARAMETERIZED PLANS with AgentSpeak variables.

        Design ref: Decision #9, Q&A #16

        For each non-goal state, generate a plan showing how to progress
        toward the goal with precondition subgoals.

        Plans use AgentSpeak variables (e.g., +!on(X, Y)) so they work
        for ANY objects of the correct type, not just specific instances.
        """
        lines = []

        # Find shortest paths to goal for all states
        paths = self.graph.find_shortest_paths_to_goal()

        # Generate plans for states that have paths to goal
        # Per Design Decision #7: Generate one plan per non-goal state
        plan_count = 0
        for state, path in paths.items():
            if state == self.graph.goal_state:
                # Skip goal state (handled by success plan)
                continue

            if not path:
                # No path to goal
                continue

            plan = self._generate_plan_for_state(state, path)
            if plan:
                lines.append(plan)
                plan_count += 1

        # Store count for statistics
        self.goal_plan_count = plan_count

        return "\n\n".join(lines) if lines else ""

    def _generate_plan_for_state(self, state: WorldState,
                                  path: List[StateTransition]) -> Optional[str]:
        """
        Generate AgentSpeak plan for a specific state

        This now generates PARAMETERIZED plans with AgentSpeak variables.

        Plan structure:
        +!goal_pattern(Vars) : context <-
            !precond1;      // Precondition subgoals
            !precond2;
            !action_goal(args);  // Action goal invocation
            !goal_pattern(Vars).     // Recursive check

        Example:
            +!on(X, Y) : holding(X) & clear(Y) <-
                !put_on_block(X, Y);
                !on(X, Y).

        Args:
            state: Current state (with variables)
            path: Path to goal (list of transitions)

        Returns:
            AgentSpeak plan string with parameterized goals
        """
        if not path:
            return None

        # Get next transition in path
        next_transition = path[0]

        # Get parameterized goal pattern
        param_goal_pattern = self._get_parameterized_goal_pattern()

        # CRITICAL: For variable-level planning, always use convert_vars=True
        # Never use obj_to_var mapping - state graph already has correct variable names
        # Format context (with AgentSpeak variables)
        context = state.to_agentspeak_context(convert_vars=True)

        # Generate precondition subgoals (per Design Algorithm 4, Line 701-708)
        subgoals = []
        for precond in next_transition.preconditions:
            # Convert precondition to AgentSpeak format with variables
            if precond not in state.predicates:
                # Need to establish this precondition
                subgoal_name = precond.to_agentspeak(convert_vars=True)
                subgoals.append(f"!{subgoal_name}")

        # Format action goal invocation
        # Action args may contain PDDL variables - convert them to AgentSpeak format
        action_args_as = []
        for arg in next_transition.action_args:
            if arg.startswith('?'):
                # Variable-level: Convert PDDL variable to AgentSpeak
                var_name = arg[1:]
                action_args_as.append(var_name[0].upper() + var_name[1:] if var_name else var_name)
            else:
                # Constant/literal: keep as-is
                action_args_as.append(arg)

        # Format action goal invocation
        action_goal = self._format_action_goal_invocation(
            next_transition.action,
            action_args_as
        )

        # Build plan body (per Design Algorithm 4, Line 716-719)
        # Structure: !precond1; !precond2; action(args); !goal_pattern.
        body_lines = []
        body_lines.extend(subgoals)                    # Precondition subgoals first
        body_lines.append(action_goal)                 # Then action goal
        body_lines.append(f"!{param_goal_pattern}")    # Recursive goal check (parameterized)

        # Format body
        body = ";\n    ".join(body_lines)

        # Generate plan (PARAMETERIZED)
        plan = f"+!{param_goal_pattern} : {context} <-\n    {body}."

        return plan

    def _format_action_goal_invocation(self, action: PDDLAction, args: List[str]) -> str:
        """
        Format action goal invocation

        Args:
            action: PDDL action
            args: Ground arguments

        Returns:
            Action goal invocation string (e.g., "!pick_up(a, b)")
        """
        action_name = action.name.replace('-', '_')

        if args:
            args_str = ", ".join(args)
            return f"!{action_name}({args_str})"
        return f"!{action_name}"

    def _predicate_to_goal_name(self, predicate: PredicateAtom) -> str:
        """
        Convert predicate to goal name (without !)

        Args:
            predicate: PredicateAtom

        Returns:
            Goal name string (e.g., "clear(b)")
        """
        return predicate.to_agentspeak()

    def _predicate_to_goal_invocation(self, predicate: PredicateAtom) -> str:
        """
        Convert predicate to goal invocation

        Args:
            predicate: PredicateAtom

        Returns:
            Goal invocation string (e.g., "!clear(b)")
        """
        return f"!{self._predicate_to_goal_name(predicate)}"

    def _generate_success_plan(self) -> str:
        """
        Generate plan for when goal is already achieved

        This plan uses PARAMETERIZED goal pattern with AgentSpeak variables.

        Example:
            +!on(X, Y) : on(X, Y) <- .print("Goal on(", X, ", ", Y, ") achieved!").
        """
        # Get parameterized goal pattern (with AgentSpeak variables)
        param_goal_pattern = self._get_parameterized_goal_pattern()

        # Get context condition (also with AgentSpeak variables)
        context = self.graph.goal_state.to_agentspeak_context(convert_vars=True)

        return f"""+!{param_goal_pattern} : {context} <-
    .print("Goal {param_goal_pattern} already achieved!")."""

    def _generate_failure_plan(self) -> str:
        """
        Generate failure handler for parameterized goal

        Example:
            -!on(X, Y) : true <- .print("Failed to achieve on(", X, ", ", Y, ")"); .fail.
        """
        # Get parameterized goal pattern
        param_goal_pattern = self._get_parameterized_goal_pattern()

        return f"""-!{param_goal_pattern} : true <-
    .print("Failed to achieve goal {param_goal_pattern}");
    .fail."""


# Test function
def test_agentspeak_codegen():
    """Test AgentSpeak code generator"""
    print("="*80)
    print("Testing AgentSpeak Code Generator (Improved)")
    print("="*80)

    # Create mock state graph
    from utils.pddl_parser import PDDLParser

    # Load domain
    domain_file = Path(__file__).parent.parent / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        print("Skipping test")
        return

    domain = PDDLParser.parse_domain(str(domain_file))

    # Create simple state graph
    goal_state = WorldState({PredicateAtom("on", ["a", "b"])}, depth=0)
    state1 = WorldState({
        PredicateAtom("holding", ["a"]),
        PredicateAtom("clear", ["b"])
    }, depth=1)
    state2 = WorldState({
        PredicateAtom("ontable", ["a"]),
        PredicateAtom("clear", ["a"]),
        PredicateAtom("clear", ["b"]),
        PredicateAtom("handempty", [])
    }, depth=2)

    graph = StateGraph(goal_state)

    # Find actions
    action_putonblock = None
    action_pickupfromtable = None
    for action in domain.actions:
        if action.name == "put-on-block":
            action_putonblock = action
        elif action.name == "pick-up-from-table":
            action_pickupfromtable = action

    if not action_putonblock or not action_pickupfromtable:
        print("Required actions not found in domain")
        return

    # Add transitions
    trans1 = StateTransition(
        from_state=state1,
        to_state=goal_state,
        action=action_putonblock,
        action_args=["a", "b"],
        belief_updates=["+on(a, b)", "+handempty", "+clear(a)", "-holding(a)", "-clear(b)"],
        preconditions=[PredicateAtom("holding", ["a"]), PredicateAtom("clear", ["b"])]
    )
    graph.add_transition(trans1)

    trans2 = StateTransition(
        from_state=state2,
        to_state=state1,
        action=action_pickupfromtable,
        action_args=["a"],
        belief_updates=["+holding(a)", "-handempty", "-ontable(a)"],
        preconditions=[PredicateAtom("handempty", []), PredicateAtom("clear", ["a"]), PredicateAtom("ontable", ["a"])]
    )
    graph.add_transition(trans2)

    # Generate code
    codegen = AgentSpeakCodeGenerator(
        state_graph=graph,
        goal_name="on(a, b)",
        domain=domain,
        objects=["a", "b"]
    )

    asl_code = codegen.generate()

    print("\nGenerated AgentSpeak Code:")
    print("="*80)
    print(asl_code)
    print("="*80)

    # Save to file
    output_file = Path(__file__).parent / "test_agentspeak_improved.asl"
    with open(output_file, 'w') as f:
        f.write(asl_code)

    print(f"\nSaved to: {output_file}")
    print()


if __name__ == "__main__":
    test_agentspeak_codegen()
