"""
Blocksworld Environment Simulator

Simple Python simulation of blocksworld for MVP demo.
Supports both classical plan execution and AgentSpeak simulation.
"""

from typing import Dict, List, Set, Tuple, Optional
import re


class BlocksworldState:
    """Represents a blocksworld state"""

    def __init__(self):
        self.on: Dict[str, str] = {}  # on[X] = Y means X is on Y
        self.clear: Set[str] = set()
        self.holding: Optional[str] = None
        self.handempty: bool = True

    def copy(self):
        """Deep copy of state"""
        new_state = BlocksworldState()
        new_state.on = self.on.copy()
        new_state.clear = self.clear.copy()
        new_state.holding = self.holding
        new_state.handempty = self.handempty
        return new_state

    def to_beliefs(self) -> List[str]:
        """Convert to AgentSpeak belief list"""
        beliefs = []
        for block, support in self.on.items():
            beliefs.append(f"on({block},{support})")
        for block in self.clear:
            beliefs.append(f"clear({block})")
        if self.holding:
            beliefs.append(f"holding({self.holding})")
        if self.handempty:
            beliefs.append("handempty")
        return beliefs

    def from_beliefs(self, beliefs: List[str]):
        """Initialize from belief list"""
        self.on = {}
        self.clear = set()
        self.holding = None
        self.handempty = False

        for belief in beliefs:
            if belief.startswith('on('):
                match = re.match(r'on\((\w+),\s*(\w+)\)', belief)
                if match:
                    self.on[match.group(1)] = match.group(2)
            elif belief.startswith('clear('):
                match = re.match(r'clear\((\w+)\)', belief)
                if match:
                    self.clear.add(match.group(1))
            elif belief.startswith('holding('):
                match = re.match(r'holding\((\w+)\)', belief)
                if match:
                    self.holding = match.group(1)
            elif belief == 'handempty':
                self.handempty = True

        return self


class BlocksworldEnvironment:
    """Simulates blocksworld environment with actions"""

    def __init__(self, initial_state: BlocksworldState):
        self.state = initial_state.copy()
        self.action_count = 0

    def reset(self, initial_state: BlocksworldState):
        """Reset environment to initial state"""
        self.state = initial_state.copy()
        self.action_count = 0

    def execute_action(self, action: str, params: List[str]) -> bool:
        """
        Execute an action in the environment

        Args:
            action: Action name (pickup, putdown, stack, unstack)
            params: Action parameters

        Returns:
            True if action succeeded, False otherwise
        """
        self.action_count += 1

        if action == 'pickup':
            return self._pickup(params[0])
        elif action == 'putdown':
            return self._putdown(params[0], params[1])
        elif action == 'stack':
            return self._stack(params[0], params[1])
        elif action == 'unstack':
            return self._unstack(params[0], params[1])
        else:
            print(f"Unknown action: {action}")
            return False

    def _pickup(self, block: str) -> bool:
        """Pickup block from table"""
        # Check preconditions
        if not self.state.handempty:
            return False
        if block not in self.state.clear:
            return False
        if block not in self.state.on:
            return False
        if self.state.on[block] != 'table':
            return False

        # Execute
        del self.state.on[block]
        self.state.clear.discard(block)
        self.state.holding = block
        self.state.handempty = False
        return True

    def _putdown(self, block: str, target: str) -> bool:
        """Put down block on target"""
        # Check preconditions
        if self.state.holding != block:
            return False
        if target != 'table' and target not in self.state.clear:
            return False

        # Execute
        self.state.on[block] = target
        self.state.clear.add(block)
        if target != 'table':
            self.state.clear.discard(target)
        self.state.holding = None
        self.state.handempty = True
        return True

    def _stack(self, block: str, target: str) -> bool:
        """Stack block on target (combines pickup + putdown)"""
        # This is a convenience action
        # In pure PDDL it might be separate, but here we combine

        # Check if we're already holding the block
        if self.state.holding == block:
            return self._putdown(block, target)

        # Otherwise pickup first
        if self._pickup(block):
            return self._putdown(block, target)

        return False

    def _unstack(self, block: str, from_block: str) -> bool:
        """Unstack block from another block"""
        # Check preconditions
        if not self.state.handempty:
            return False
        if block not in self.state.clear:
            return False
        if block not in self.state.on:
            return False
        if self.state.on[block] != from_block:
            return False

        # Execute
        del self.state.on[block]
        self.state.clear.discard(block)
        self.state.clear.add(from_block)
        self.state.holding = block
        self.state.handempty = False
        return True

    def check_condition(self, condition: str) -> bool:
        """Check if a condition holds in current state"""
        condition = condition.strip()

        if condition == 'true':
            return True

        if condition.startswith('on('):
            match = re.match(r'on\((\w+),\s*(\w+)\)', condition)
            if match:
                block, target = match.groups()
                return self.state.on.get(block) == target

        if condition.startswith('clear('):
            match = re.match(r'clear\((\w+)\)', condition)
            if match:
                block = match.group(1)
                return block in self.state.clear

        if condition.startswith('holding('):
            match = re.match(r'holding\((\w+)\)', condition)
            if match:
                block = match.group(1)
                return self.state.holding == block

        if condition == 'handempty':
            return self.state.handempty

        # Negation
        if condition.startswith('not ') or condition.startswith('\\+ '):
            inner = condition.split(' ', 1)[1]
            return not self.check_condition(inner)

        return False


class ClassicalPlanExecutor:
    """Executes a classical action sequence in blocksworld"""

    def __init__(self, environment: BlocksworldEnvironment):
        self.env = environment

    def execute(self, plan: List[Tuple[str, List[str]]]) -> Dict:
        """
        Execute classical plan

        Returns:
            Execution result with success status and metrics
        """
        result = {
            'success': False,
            'actions_executed': 0,
            'final_state': None,
            'failure_action': None,
            'trace': []
        }

        for action, params in plan:
            result['trace'].append(f"{action}({', '.join(params)})")

            success = self.env.execute_action(action, params)
            result['actions_executed'] += 1

            if not success:
                result['failure_action'] = (action, params)
                result['final_state'] = self.env.state.to_beliefs()
                return result

        # All actions succeeded
        result['success'] = True
        result['final_state'] = self.env.state.to_beliefs()
        return result


if __name__ == "__main__":
    # Quick test
    print("Testing Blocksworld Simulator...")

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

    # Test classical execution
    executor = ClassicalPlanExecutor(env)
    plan = [
        ('pickup', ['c']),
        ('putdown', ['c', 'b'])
    ]

    print("\nExecuting plan:", plan)
    result = executor.execute(plan)

    print("Result:", result['success'])
    print("Actions executed:", result['actions_executed'])
    print("Final state:", result['final_state'])
    print("Trace:", result['trace'])
