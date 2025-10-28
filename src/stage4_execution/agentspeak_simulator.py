"""
AgentSpeak BDI Simulator

Simulates AgentSpeak execution with BDI reasoning cycle.
For MVP: Simple parser and interpreter for generated .asl code.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
import re
from .blocksworld_simulator import BlocksworldEnvironment, BlocksworldState


class AgentSpeakPlan:
    """Represents a single AgentSpeak plan"""

    def __init__(self, trigger: str, context: str, body: List[str]):
        self.trigger = trigger  # e.g., "+!stack(c,b)"
        self.context = context  # e.g., "clear(c) & clear(b)"
        self.body = body  # List of actions/subgoals

    def __repr__(self):
        return f"Plan({self.trigger} : {self.context} <- {'; '.join(self.body)})"


class AgentSpeakAgent:
    """Simple AgentSpeak interpreter for MVP"""

    def __init__(self, asl_code: str, environment: BlocksworldEnvironment):
        self.env = environment
        self.beliefs: Set[str] = set()
        self.goals: List[str] = []
        self.plans: List[AgentSpeakPlan] = []
        self.execution_trace: List[str] = []

        # Parse ASL code
        self._parse_asl(asl_code)

        # Initialize beliefs from environment
        self._sync_beliefs()

    def _parse_asl(self, asl_code: str):
        """Parse AgentSpeak code into plan structures - handles multi-line plans"""
        lines = asl_code.split('\n')

        current_plan_text = ""

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip comments (but don't skip if we're building a plan)
            if line.startswith('//') and not current_plan_text:
                continue

            # If this line starts with +! or -!, it's a new plan
            if line.startswith(('+!', '-!')):
                # If we have a previous plan, try to parse it first
                if current_plan_text:
                    self._parse_single_plan(current_plan_text)
                # Start new plan
                current_plan_text = line
            elif current_plan_text:
                # Continue building current plan
                current_plan_text += " " + line

            # If plan ends with period, parse it
            if current_plan_text and current_plan_text.rstrip().endswith('.'):
                self._parse_single_plan(current_plan_text)
                current_plan_text = ""

        # Parse any remaining plan
        if current_plan_text:
            self._parse_single_plan(current_plan_text)

    def _parse_single_plan(self, plan_text: str):
        """Parse a single plan from accumulated text"""
        # Match plan structure: +!goal or +!!goal (declarative) : context <- body.
        # Updated pattern to handle identifiers with underscores
        plan_pattern = r'(\+!!?[\w\(\),\s\[\]_]+)\s*:\s*([^<]*)\s*<-\s*(.+)\.'
        match = re.match(plan_pattern, plan_text)

        if match:
            trigger = match.group(1).strip()
            context = match.group(2).strip()
            body_str = match.group(3).strip()

            # Split body by semicolons
            body = [action.strip() for action in body_str.split(';')]

            plan = AgentSpeakPlan(trigger, context, body)
            self.plans.append(plan)
            # Debug output

    def _sync_beliefs(self):
        """Sync beliefs from environment state"""
        self.beliefs = set(self.env.state.to_beliefs())

    def _check_context(self, context: str) -> bool:
        """Check if context condition holds"""
        if context == 'true' or not context:
            return True

        # Handle simple conjunction (AND)
        if '&' in context:
            conditions = [c.strip() for c in context.split('&')]
            return all(self._check_single_condition(c) for c in conditions)

        # Handle single condition
        return self._check_single_condition(context)

    def _check_single_condition(self, condition: str) -> bool:
        """Check single belief condition"""
        condition = condition.strip()

        # Handle negation
        if condition.startswith('not ') or condition.startswith('\\+ '):
            inner = condition.split(' ', 1)[1].strip()
            result = not self._belief_exists(inner)
            return result

        # Check if belief exists
        result = self._belief_exists(condition)
        return result

    def _belief_exists(self, condition: str) -> bool:
        """Check if a belief exists, handling ontable(X) <-> on(X,table) conversion"""
        # Direct match
        if condition in self.beliefs:
            return True

        # Convert ontable(X) to on(X, table) and check BOTH with and without space
        match = re.match(r'ontable\((\w+)\)', condition)
        if match:
            block = match.group(1)
            # Try both formats: with and without space
            converted_with_space = f"on({block}, table)"
            converted_no_space = f"on({block},table)"
            return converted_with_space in self.beliefs or converted_no_space in self.beliefs

        # Convert on(X, table) to ontable(X) and check
        match = re.match(r'on\((\w+),\s*table\)', condition)
        if match:
            block = match.group(1)
            converted = f"ontable({block})"
            return converted in self.beliefs

        return False

    def _select_plan(self, goal: str) -> Optional[AgentSpeakPlan]:
        """Select applicable plan for goal with variable unification"""
        # Find plans that match the goal trigger
        applicable = []

        # Strip brackets from goal if present (declarative goals)
        goal_normalized = goal.strip('[]')


        for plan in self.plans:
            # Extract goal name from trigger
            # Handle both +!goal and +!![goal] (declarative)
            trigger_goal = plan.trigger.replace('+!!', '').replace('+!', '').strip()
            # Remove brackets for declarative goals: [on(c,b)] -> on(c,b)
            trigger_goal = trigger_goal.strip('[]')


            # Try exact match first
            if trigger_goal == goal_normalized:
                if self._check_context(plan.context):
                    applicable.append(plan)
                    continue

            # Try variable unification (e.g., stack(X,Y) matches stack(c,b))
            if self._unify_goal(trigger_goal, goal_normalized):
                if self._check_context(plan.context):
                    applicable.append(plan)

        # Return first applicable plan (simple selection)
        return applicable[0] if applicable else None

    def _unify_goal(self, pattern: str, goal: str) -> bool:
        """
        Simple unification: check if pattern matches goal
        Examples:
            pattern="stack(X, Y)", goal="stack(c, b)" -> True
            pattern="stack(c, b)", goal="stack(c, b)" -> True
            pattern="stack(a, b)", goal="stack(c, b)" -> False
        """
        # Extract predicate name and arguments
        pattern_match = re.match(r'(\w+)\((.*?)\)', pattern)
        goal_match = re.match(r'(\w+)\((.*?)\)', goal)

        if not pattern_match or not goal_match:
            return False

        pattern_pred, pattern_args_str = pattern_match.groups()
        goal_pred, goal_args_str = goal_match.groups()

        # Predicates must match
        if pattern_pred != goal_pred:
            return False

        # Parse arguments
        pattern_args = [a.strip() for a in pattern_args_str.split(',') if a.strip()]
        goal_args = [a.strip() for a in goal_args_str.split(',') if a.strip()]

        # Arity must match
        if len(pattern_args) != len(goal_args):
            return False

        # Check each argument
        for p_arg, g_arg in zip(pattern_args, goal_args):
            # If pattern arg is variable (uppercase), it matches anything
            if p_arg and p_arg[0].isupper():
                continue
            # If pattern arg is constant, must match exactly
            elif p_arg == g_arg:
                continue
            else:
                return False

        return True

    def _execute_action(self, action_str: str) -> bool:
        """Execute a single action from plan body"""
        action_str = action_str.strip()


        # Declarative goal (!!goal)
        if action_str.startswith('!!'):
            subgoal = action_str[2:].strip()  # Remove both !!
            self.execution_trace.append(f"Posting declarative goal: {subgoal}")
            return self._achieve_goal(subgoal)

        # Subgoal (achievement goal !goal)
        if action_str.startswith('!'):
            subgoal = action_str[1:].strip()
            self.execution_trace.append(f"Posting subgoal: {subgoal}")
            return self._achieve_goal(subgoal)

        # Belief addition
        if action_str.startswith('+'):
            belief = action_str[1:].strip()
            self.beliefs.add(belief)
            self.execution_trace.append(f"Added belief: {belief}")
            return True

        # Belief removal
        if action_str.startswith('-'):
            belief = action_str[1:].strip()
            self.beliefs.discard(belief)
            self.execution_trace.append(f"Removed belief: {belief}")
            return True

        # Query/test goal
        if action_str.startswith('?'):
            belief = action_str[1:].strip()
            result = belief in self.beliefs
            self.execution_trace.append(f"Query {belief}: {result}")
            return result

        # Print action
        if action_str.startswith('.print('):
            message = action_str[7:-1]  # Extract message
            self.execution_trace.append(f"Print: {message}")
            return True

        # Physical action (primitive)
        return self._execute_primitive_action(action_str)

    def _execute_primitive_action(self, action_str: str) -> bool:
        """Execute primitive action in environment"""
        # Parse action: action_name(param1, param2, ...)
        match = re.match(r'(\w+)\((.*?)\)', action_str)

        if not match:
            self.execution_trace.append(f"Failed to parse action: {action_str}")
            return False

        action_name = match.group(1)
        params_str = match.group(2)
        params = [p.strip() for p in params_str.split(',')] if params_str else []

        self.execution_trace.append(f"Executing: {action_name}({', '.join(params)})")

        # Execute in environment
        success = self.env.execute_action(action_name, params)

        if success:
            # Sync beliefs after successful action
            self._sync_beliefs()
        else:
            self.execution_trace.append(f"Action failed: {action_str}")

        return success

    def _achieve_goal(self, goal: str) -> bool:
        """Achieve a goal using BDI reasoning cycle"""
        self.execution_trace.append(f"Attempting goal: {goal}")

        # For declarative goals (with brackets), check if already satisfied
        if goal.startswith('[') and goal.endswith(']'):
            goal_condition = goal.strip('[]')
            if self._belief_exists(goal_condition):
                self.execution_trace.append(f"Declarative goal already satisfied: {goal_condition}")
                return True

        # Select applicable plan
        plan = self._select_plan(goal)

        if not plan:
            self.execution_trace.append(f"No applicable plan for: {goal}")
            return False

        self.execution_trace.append(f"Selected plan: {plan.trigger}")

        # Execute plan body
        for action in plan.body:
            success = self._execute_action(action)

            if not success:
                self.execution_trace.append(f"Plan failed at action: {action}")
                return False

        self.execution_trace.append(f"Goal achieved: {goal}")
        return True

    def run(self, initial_goal: str) -> Dict[str, Any]:
        """
        Run AgentSpeak agent with initial goal

        Returns:
            Execution result with success status and metrics
        """
        result = {
            'success': False,
            'goal': initial_goal,
            'final_state': None,
            'trace': [],
            'beliefs': None,
            'plans_used': 0
        }

        # Attempt to achieve initial goal
        try:
            success = self._achieve_goal(initial_goal)

            result['success'] = success
            result['final_state'] = list(self.beliefs)
            result['trace'] = self.execution_trace.copy()
            result['beliefs'] = list(self.beliefs)
            result['plans_used'] = len([t for t in self.execution_trace if 'Selected plan' in t])

        except Exception as e:
            result['trace'] = self.execution_trace.copy()
            result['trace'].append(f"Exception: {type(e).__name__}: {str(e)}")
            result['final_state'] = list(self.beliefs)

        return result


class AgentSpeakExecutor:
    """Executes AgentSpeak plan library in blocksworld"""

    def __init__(self, environment: BlocksworldEnvironment):
        self.env = environment

    def execute(self, asl_code: str, initial_goal: str) -> Dict[str, Any]:
        """
        Execute AgentSpeak program

        Args:
            asl_code: AgentSpeak plan library code
            initial_goal: Initial goal to achieve (e.g., "stack(c,b)")

        Returns:
            Execution result with success status and metrics
        """
        # Create agent
        agent = AgentSpeakAgent(asl_code, self.env)

        # Run agent
        result = agent.run(initial_goal)

        # Add final environment state
        result['env_final_state'] = self.env.state.to_beliefs()

        return result


if __name__ == "__main__":
    # Quick test
    print("Testing AgentSpeak Simulator...")

    # Create initial state
    init_state = BlocksworldState()
    init_state.from_beliefs([
        'on(c, a)',
        'on(a, table)',
        'on(b, table)',
        'clear(c)',
        'clear(b)',
        'handempty'
    ])

    print("Initial state:", init_state.to_beliefs())

    # Create environment
    env = BlocksworldEnvironment(init_state)

    # Sample AgentSpeak code
    sample_asl = """
    // Main goal: stack c on b
    +!stack(c, b) : clear(c) & clear(b) & handempty <-
        !pickup(c);
        !putdown(c, b).

    // Pickup from table
    +!pickup(c) : clear(c) & on(c, a) & handempty <-
        unstack(c, a);
        +holding(c);
        -handempty;
        -on(c, a);
        +clear(a).

    // Putdown on block
    +!putdown(c, b) : holding(c) & clear(b) <-
        stack(c, b);
        -holding(c);
        +handempty;
        +on(c, b);
        -clear(b).
    """

    # Test AgentSpeak execution
    executor = AgentSpeakExecutor(env)
    result = executor.execute(sample_asl, "stack(c, b)")

    print("\nResult:", result['success'])
    print("Final state:", result['env_final_state'])
    print("\nExecution trace:")
    for line in result['trace']:
        print(f"  {line}")
