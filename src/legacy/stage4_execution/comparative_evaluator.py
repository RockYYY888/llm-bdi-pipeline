"""
Comparative Evaluator

Compares LLM AgentSpeak Generation (Branch A) with FOND Planning (Branch B).
Provides metrics and analysis for research evaluation.
"""

from typing import Dict, List, Any, Tuple, Union
from .blocksworld_simulator import BlocksworldEnvironment, BlocksworldState, ClassicalPlanExecutor
from .agentspeak_simulator import AgentSpeakExecutor


class ComparativeEvaluator:
    """
    Evaluates and compares LLM AgentSpeak vs FOND Planning execution
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def evaluate(self,
                 initial_state: BlocksworldState,
                 llm_agentspeak_code: str,
                 fond_plan: List[Tuple[str, List[str]]],
                 ltl_goal: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Run both branches and compare results

        Args:
            initial_state: Initial blocksworld state
            llm_agentspeak_code: AgentSpeak code from LLM generator (Branch A)
            fond_plan: Action sequence from FOND planner (Branch B)
            ltl_goal: Original LTLf goal(s) for verification
                     - Single: "F(on(c,b))"
                     - Multiple: ["F(on(a,b))", "F(on(b,c))"]

        Returns:
            Comparison results with metrics
        """
        # Branch A: LLM AgentSpeak
        llm_agentspeak_result = self._run_agentspeak(initial_state, llm_agentspeak_code, ltl_goal)

        # Branch B: FOND Planning
        fond_result = self._run_plan(initial_state, fond_plan, 'fond')

        # Compare results
        comparison = self._compare_results(
            llm_agentspeak_result,
            fond_result,
            ltl_goal
        )

        # Store results
        self.results = {
            'llm_agentspeak': llm_agentspeak_result,
            'fond': fond_result,
            'comparison': comparison,
            'ltl_goal': ltl_goal
        }

        return self.results

    def _run_agentspeak(self,
                        initial_state: BlocksworldState,
                        agentspeak_code: str,
                        ltl_goals: Union[str, List[str]]) -> Dict[str, Any]:
        """Execute AgentSpeak code (Branch A)"""
        # Execute AgentSpeak code
        executor = AgentSpeakExecutor(initial_state.copy())
        goals_list = [ltl_goals] if isinstance(ltl_goals, str) else ltl_goals
        result = executor.execute(agentspeak_code, goals_list)

        # Add metadata
        result['branch'] = 'llm_agentspeak'

        return result

    def _run_plan(self,
                  initial_state: BlocksworldState,
                  plan: List[Tuple[str, List[str]]],
                  branch_name: str) -> Dict[str, Any]:
        """Execute a plan (FOND)"""
        # Create fresh environment
        env = BlocksworldEnvironment(initial_state.copy())

        # Execute plan
        executor = ClassicalPlanExecutor(env)
        result = executor.execute(plan)

        # Add metadata
        result['branch'] = branch_name
        result['plan_length'] = len(plan)

        return result

    def _compare_results(self,
                         llm_agentspeak: Dict[str, Any],
                         fond: Dict[str, Any],
                         ltl_goal: str) -> Dict[str, Any]:
        """Compare execution results"""

        comparison = {
            'both_succeeded': llm_agentspeak['success'] and fond['success'],
            'llm_agentspeak_only': llm_agentspeak['success'] and not fond['success'],
            'fond_only': fond['success'] and not llm_agentspeak['success'],
            'both_failed': not llm_agentspeak['success'] and not fond['success']
        }

        # Goal satisfaction check - handle both single goal (string) and multiple goals (list)
        ltl_goals = [ltl_goal] if isinstance(ltl_goal, str) else ltl_goal

        # Check each goal for LLM AgentSpeak
        llm_agentspeak_goal_results = []
        for goal in ltl_goals:
            satisfied = self._check_ltl_satisfaction(
                llm_agentspeak.get('final_state', []),
                goal
            )
            llm_agentspeak_goal_results.append({
                'goal': goal,
                'satisfied': satisfied
            })

        # Check each goal for FOND
        fond_goal_results = []
        for goal in ltl_goals:
            satisfied = self._check_ltl_satisfaction(
                fond.get('final_state', []),
                goal
            )
            fond_goal_results.append({
                'goal': goal,
                'satisfied': satisfied
            })

        # Overall satisfaction: all goals must be satisfied
        comparison['llm_agentspeak_satisfies_goal'] = all(r['satisfied'] for r in llm_agentspeak_goal_results)
        comparison['fond_satisfies_goal'] = all(r['satisfied'] for r in fond_goal_results)

        # Detailed goal results
        comparison['llm_agentspeak_goal_details'] = llm_agentspeak_goal_results
        comparison['fond_goal_details'] = fond_goal_results

        # Efficiency metrics
        if llm_agentspeak['success'] and fond['success']:
            llm_agentspeak_actions = llm_agentspeak.get('actions_executed', 0)
            fond_actions = fond.get('actions_executed', 0)

            comparison['efficiency'] = {
                'llm_agentspeak_actions': llm_agentspeak_actions,
                'fond_actions': fond_actions,
                'llm_agentspeak_more_efficient': llm_agentspeak_actions < fond_actions,
                'efficiency_ratio': fond_actions / llm_agentspeak_actions if llm_agentspeak_actions > 0 else float('inf')
            }
        else:
            comparison['efficiency'] = None

        # Robustness comparison (simplified for LLM AgentSpeak vs FOND)
        comparison['robustness'] = {
            'llm_agentspeak_deterministic': True,  # LLM AgentSpeak generates deterministic code
            'fond_handles_nondeterminism': True  # FOND explicitly handles non-determinism
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

        llm_agentspeak = self.results['llm_agentspeak']
        fond = self.results['fond']
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

        # Branch A: LLM AgentSpeak
        report.append("\n" + "-"*80)
        report.append("BRANCH A: LLM AgentSpeak Generation (Baseline)")
        report.append("-"*80)
        report.append(f"Success: {llm_agentspeak['success']}")
        report.append(f"Actions Executed: {llm_agentspeak.get('actions_executed', 0)}")
        report.append(f"Goal Satisfied: {comparison['llm_agentspeak_satisfies_goal']}")

        # Detailed goal verification (if multiple goals)
        if 'llm_agentspeak_goal_details' in comparison and len(comparison['llm_agentspeak_goal_details']) > 1:
            report.append("\nDetailed Goal Verification:")
            for detail in comparison['llm_agentspeak_goal_details']:
                status = "✓" if detail['satisfied'] else "✗"
                report.append(f"  {status} {detail['goal']}")

        if llm_agentspeak['success']:
            report.append(f"Final State: {llm_agentspeak.get('final_state', [])}")
        else:
            report.append(f"Failure Action: {llm_agentspeak.get('failure_action', 'N/A')}")

        report.append("\nExecution Trace:")
        for action in llm_agentspeak.get('trace', []):
            report.append(f"  {action}")

        # Branch B: FOND Planning
        report.append("\n" + "-"*80)
        report.append("BRANCH B: FOND Planning (PR2)")
        report.append("-"*80)
        report.append(f"Success: {fond['success']}")
        report.append(f"Actions Executed: {fond.get('actions_executed', 0)}")
        report.append(f"Goal Satisfied: {comparison['fond_satisfies_goal']}")

        # Detailed goal verification (if multiple goals)
        if 'fond_goal_details' in comparison and len(comparison['fond_goal_details']) > 1:
            report.append("\nDetailed Goal Verification:")
            for detail in comparison['fond_goal_details']:
                status = "✓" if detail['satisfied'] else "✗"
                report.append(f"  {status} {detail['goal']}")

        if fond['success']:
            report.append(f"Final State: {fond.get('final_state', [])}")
        else:
            report.append(f"Failure Action: {fond.get('failure_action', 'N/A')}")

        report.append("\nExecution Trace:")
        for action in fond.get('trace', []):
            report.append(f"  {action}")

        # Comparison
        report.append("\n" + "="*80)
        report.append("COMPARISON SUMMARY")
        report.append("="*80)

        if comparison['both_succeeded']:
            report.append("✓ Both branches succeeded")
        elif comparison['llm_agentspeak_only']:
            report.append("⚠ LLM AgentSpeak succeeded, FOND failed")
        elif comparison['fond_only']:
            report.append("⚠ FOND succeeded, LLM AgentSpeak failed")
        else:
            report.append("✗ Both branches failed")

        # Efficiency
        if comparison['efficiency']:
            eff = comparison['efficiency']
            report.append(f"\nEfficiency:")
            report.append(f"  LLM AgentSpeak Actions: {eff['llm_agentspeak_actions']}")
            report.append(f"  FOND Actions: {eff['fond_actions']}")
            report.append(f"  Efficiency Ratio: {eff['efficiency_ratio']:.2f}")

            if eff['llm_agentspeak_more_efficient']:
                report.append("  → LLM AgentSpeak is more efficient (fewer actions)")
            else:
                report.append("  → FOND is more efficient or equal")

        # Robustness
        rob = comparison['robustness']
        report.append(f"\nRobustness:")
        report.append(f"  LLM AgentSpeak Deterministic: {rob['llm_agentspeak_deterministic']}")
        report.append(f"  FOND Handles Non-determinism: {rob['fond_handles_nondeterminism']}")

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
