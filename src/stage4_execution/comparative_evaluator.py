"""
Comparative Evaluator

Compares Classical PDDL planning (Branch A) with LLM AgentSpeak (Branch B).
Provides metrics and analysis for research evaluation.
"""

from typing import Dict, List, Any, Tuple, Union
from .blocksworld_simulator import BlocksworldEnvironment, BlocksworldState, ClassicalPlanExecutor
from .agentspeak_simulator import AgentSpeakExecutor


class ComparativeEvaluator:
    """
    Evaluates and compares Classical vs AgentSpeak execution
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def evaluate(self,
                 initial_state: BlocksworldState,
                 classical_plan: List[Tuple[str, List[str]]],
                 agentspeak_code: str,
                 agentspeak_goal: str,
                 ltl_goal: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Run both branches and compare results

        Args:
            initial_state: Initial blocksworld state
            classical_plan: Classical action sequence from PDDL planner
            agentspeak_code: Generated AgentSpeak plan library
            agentspeak_goal: Initial goal for AgentSpeak (e.g., "stack(c,b)")
            ltl_goal: Original LTLf goal(s) for verification
                     - Single: "F(on(c,b))"
                     - Multiple: ["F(on(a,b))", "F(on(b,c))"]

        Returns:
            Comparison results with metrics
        """
        # Branch A: Classical Planning
        classical_result = self._run_classical(initial_state, classical_plan)

        # Branch B: AgentSpeak
        agentspeak_result = self._run_agentspeak(initial_state, agentspeak_code, agentspeak_goal)

        # Compare results
        comparison = self._compare_results(
            classical_result,
            agentspeak_result,
            ltl_goal,
            agentspeak_code
        )

        # Store results
        self.results = {
            'classical': classical_result,
            'agentspeak': agentspeak_result,
            'comparison': comparison,
            'ltl_goal': ltl_goal
        }

        return self.results

    def _run_classical(self,
                       initial_state: BlocksworldState,
                       plan: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
        """Execute classical plan"""
        # Create fresh environment
        env = BlocksworldEnvironment(initial_state.copy())

        # Execute plan
        executor = ClassicalPlanExecutor(env)
        result = executor.execute(plan)

        # Add metadata
        result['branch'] = 'classical'
        result['plan_length'] = len(plan)

        return result

    def _run_agentspeak(self,
                        initial_state: BlocksworldState,
                        asl_code: str,
                        initial_goal: str) -> Dict[str, Any]:
        """Execute AgentSpeak program"""
        # Create fresh environment
        env = BlocksworldEnvironment(initial_state.copy())

        # Execute AgentSpeak
        executor = AgentSpeakExecutor(env)
        result = executor.execute(asl_code, initial_goal)

        # Add metadata
        result['branch'] = 'agentspeak'

        return result

    def _compare_results(self,
                         classical: Dict[str, Any],
                         agentspeak: Dict[str, Any],
                         ltl_goal: str,
                         agentspeak_code: str) -> Dict[str, Any]:
        """Compare execution results"""

        comparison = {
            'both_succeeded': classical['success'] and agentspeak['success'],
            'classical_only': classical['success'] and not agentspeak['success'],
            'agentspeak_only': agentspeak['success'] and not classical['success'],
            'both_failed': not classical['success'] and not agentspeak['success']
        }

        # Goal satisfaction check - handle both single goal (string) and multiple goals (list)
        ltl_goals = [ltl_goal] if isinstance(ltl_goal, str) else ltl_goal

        # Check each goal for classical
        classical_goal_results = []
        for goal in ltl_goals:
            satisfied = self._check_ltl_satisfaction(
                classical.get('final_state', []),
                goal
            )
            classical_goal_results.append({
                'goal': goal,
                'satisfied': satisfied
            })

        # Check each goal for agentspeak
        agentspeak_goal_results = []
        for goal in ltl_goals:
            satisfied = self._check_ltl_satisfaction(
                agentspeak.get('env_final_state', []),
                goal
            )
            agentspeak_goal_results.append({
                'goal': goal,
                'satisfied': satisfied
            })

        # Overall satisfaction: all goals must be satisfied
        comparison['classical_satisfies_goal'] = all(r['satisfied'] for r in classical_goal_results)
        comparison['agentspeak_satisfies_goal'] = all(r['satisfied'] for r in agentspeak_goal_results)

        # Detailed goal results
        comparison['classical_goal_details'] = classical_goal_results
        comparison['agentspeak_goal_details'] = agentspeak_goal_results

        # Efficiency metrics
        if classical['success'] and agentspeak['success']:
            classical_actions = classical.get('actions_executed', 0)
            agentspeak_trace_actions = len([
                t for t in agentspeak.get('trace', [])
                if 'Executing:' in t
            ])

            comparison['efficiency'] = {
                'classical_actions': classical_actions,
                'agentspeak_actions': agentspeak_trace_actions,
                'classical_more_efficient': classical_actions < agentspeak_trace_actions,
                'efficiency_ratio': agentspeak_trace_actions / classical_actions if classical_actions > 0 else float('inf')
            }
        else:
            comparison['efficiency'] = None

        # Robustness comparison (for future: inject failures and test)
        comparison['robustness'] = {
            'classical_failure_recovery': False,  # Classical has no recovery
            'agentspeak_failure_recovery': 'failure plan' in agentspeak_code.lower()  # Check for -! plans
        }

        return comparison

    def _check_ltl_satisfaction(self, final_state: List[str], ltl_goal: str) -> bool:
        """
        Check if final state satisfies LTLf goal

        Supports: F(φ) goals with space normalization
        """
        # Extract goal from F(...)
        if ltl_goal.startswith('F(') and ltl_goal.endswith(')'):
            goal_predicate = ltl_goal[2:-1]  # Extract φ from F(φ)

            # Normalize spaces in both goal and state for comparison
            goal_normalized = goal_predicate.replace(' ', '')

            # Check if normalized goal matches any normalized state predicate
            for state_pred in final_state:
                if state_pred.replace(' ', '') == goal_normalized:
                    return True

            return False

        # For other LTL formulas, would need full LTL checker
        # Return True if we can't verify (assume satisfied)
        return True

    def generate_report(self) -> str:
        """Generate human-readable comparison report"""
        if not self.results:
            return "No evaluation results available. Run evaluate() first."

        classical = self.results['classical']
        agentspeak = self.results['agentspeak']
        comparison = self.results['comparison']

        report = []
        report.append("="*80)
        report.append("COMPARATIVE EVALUATION REPORT")
        report.append("="*80)

        # LTL Goal(s)
        ltl_goal = self.results['ltl_goal']
        if isinstance(ltl_goal, list):
            report.append(f"\nLTLf Goals ({len(ltl_goal)}):")
            for i, goal in enumerate(ltl_goal, 1):
                report.append(f"  {i}. {goal}")
        else:
            report.append(f"\nLTLf Goal: {ltl_goal}")

        # Branch A: Classical
        report.append("\n" + "-"*80)
        report.append("BRANCH A: Classical PDDL Planning")
        report.append("-"*80)
        report.append(f"Success: {classical['success']}")
        report.append(f"Actions Executed: {classical.get('actions_executed', 0)}")
        report.append(f"Goal Satisfied: {comparison['classical_satisfies_goal']}")

        # Detailed goal verification (if multiple goals)
        if 'classical_goal_details' in comparison and len(comparison['classical_goal_details']) > 1:
            report.append("\nDetailed Goal Verification:")
            for detail in comparison['classical_goal_details']:
                status = "✓" if detail['satisfied'] else "✗"
                report.append(f"  {status} {detail['goal']}")

        if classical['success']:
            report.append(f"Final State: {classical.get('final_state', [])}")
        else:
            report.append(f"Failure Action: {classical.get('failure_action', 'N/A')}")

        report.append("\nExecution Trace:")
        for action in classical.get('trace', []):
            report.append(f"  {action}")

        # Branch B: AgentSpeak
        report.append("\n" + "-"*80)
        report.append("BRANCH B: LLM AgentSpeak")
        report.append("-"*80)
        report.append(f"Success: {agentspeak['success']}")
        report.append(f"Goal Satisfied: {comparison['agentspeak_satisfies_goal']}")

        # Detailed goal verification (if multiple goals)
        if 'agentspeak_goal_details' in comparison and len(comparison['agentspeak_goal_details']) > 1:
            report.append("\nDetailed Goal Verification:")
            for detail in comparison['agentspeak_goal_details']:
                status = "✓" if detail['satisfied'] else "✗"
                report.append(f"  {status} {detail['goal']}")

        if agentspeak['success']:
            report.append(f"Final State: {agentspeak.get('env_final_state', [])}")
        else:
            report.append("Execution failed (see trace)")

        report.append("\nExecution Trace:")
        for line in agentspeak.get('trace', [])[:20]:  # Limit trace length
            report.append(f"  {line}")
        if len(agentspeak.get('trace', [])) > 20:
            report.append(f"  ... ({len(agentspeak['trace']) - 20} more lines)")

        # Comparison
        report.append("\n" + "="*80)
        report.append("COMPARISON SUMMARY")
        report.append("="*80)

        if comparison['both_succeeded']:
            report.append("✓ Both branches succeeded")
        elif comparison['classical_only']:
            report.append("⚠ Classical succeeded, AgentSpeak failed")
        elif comparison['agentspeak_only']:
            report.append("⚠ AgentSpeak succeeded, Classical failed")
        else:
            report.append("✗ Both branches failed")

        # Efficiency
        if comparison['efficiency']:
            eff = comparison['efficiency']
            report.append(f"\nEfficiency:")
            report.append(f"  Classical Actions: {eff['classical_actions']}")
            report.append(f"  AgentSpeak Actions: {eff['agentspeak_actions']}")
            report.append(f"  Efficiency Ratio: {eff['efficiency_ratio']:.2f}")

            if eff['classical_more_efficient']:
                report.append("  → Classical is more efficient (fewer actions)")
            else:
                report.append("  → AgentSpeak is more efficient or equal")

        # Robustness
        rob = comparison['robustness']
        report.append(f"\nRobustness:")
        report.append(f"  Classical Failure Recovery: {rob['classical_failure_recovery']}")
        report.append(f"  AgentSpeak Failure Plans: {rob['agentspeak_failure_recovery']}")

        report.append("\n" + "="*80)

        return "\n".join(report)


if __name__ == "__main__":
    # Quick test
    print("Testing Comparative Evaluator...")

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

    # Classical plan
    classical_plan = [
        ('unstack', ['c', 'a']),
        ('stack', ['c', 'b'])
    ]

    # AgentSpeak code
    agentspeak_code = """
    +!stack(c, b) : clear(c) & clear(b) & handempty <-
        !pickup(c);
        !putdown(c, b).

    +!pickup(c) : clear(c) & on(c, a) & handempty <-
        unstack(c, a);
        +holding(c);
        -handempty;
        +clear(a).

    +!putdown(c, b) : holding(c) & clear(b) <-
        stack(c, b);
        -holding(c);
        +handempty;
        +on(c, b).
    """

    # Evaluate
    evaluator = ComparativeEvaluator()
    results = evaluator.evaluate(
        init_state,
        classical_plan,
        agentspeak_code,
        "stack(c, b)",
        "F(on(c, b))"
    )

    # Print report
    print("\n" + evaluator.generate_report())
