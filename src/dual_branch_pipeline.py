"""
Dual-Branch Pipeline: LLM AgentSpeak vs FOND Planning Comparison

Orchestrates the complete dual-branch workflow:
    Stage 1: NL -> LTLf Goals
    Stage 2A: Branch A - LLM AgentSpeak Generation (Baseline: LTLf -> LLM -> AgentSpeak)
    Stage 2B: Branch B - FOND Planning (Research: LTLf -> PDDL -> PR2 FOND planner -> Policy)
    Stage 3: Execution & Comparison (Both execute in Jason)

Compares:
- Branch A (llm_agentspeak): End-to-end LLM directly generates AgentSpeak code
- Branch B (fond): Formal FOND planning generates policy from PDDL
"""

import re
from pathlib import Path
from typing import Dict, Any

from config import get_config
from stage1_interpretation.ltl_parser import NLToLTLParser
from stage2_pddl.pddl_problem_generator import PDDLProblemGenerator
from stage3_codegen.agentspeak_generator import AgentSpeakGenerator
from stage3_codegen.pr2_planner import PR2Planner
from stage4_execution.blocksworld_simulator import BlocksworldState, BlocksworldEnvironment, ClassicalPlanExecutor
from stage4_execution.agentspeak_simulator import AgentSpeakExecutor
from stage4_execution.comparative_evaluator import ComparativeEvaluator
from pipeline_logger import PipelineLogger


class DualBranchPipeline:
    """
    Complete dual-branch pipeline for LTL-BDI comparison

    Executes both classical planning and LLM-generated AgentSpeak,
    then compares their performance.
    """

    def __init__(self):
        self.config = get_config()
        self.logger = PipelineLogger()

        # Domain configuration
        self.domain_path = Path(__file__).parent.parent / "domains" / "blocksworld" / "domain.pddl"
        self.domain_actions = ['pickup', 'putdown', 'stack', 'unstack']
        self.domain_predicates = ['on(X, Y)', 'clear(X)', 'holding(X)', 'handempty']

        # Output directory
        self.output_dir = Path(__file__).parent.parent / "output"
        self.output_dir.mkdir(exist_ok=True)

    def execute(self, nl_instruction: str, mode: str = "both") -> Dict[str, Any]:
        """
        Execute complete dual-branch pipeline

        Args:
            nl_instruction: Natural language instruction
            mode: Execution mode - "both", "llm_agentspeak", or "fond"
                - "both": Run both branches and compare
                - "llm_agentspeak": Run only LLM AgentSpeak baseline (Branch A)
                - "fond": Run only FOND planning (Branch B)

        Returns:
            Complete results including branch results and comparison (if mode="both")
        """
        # Start logger
        self.logger.start_pipeline(nl_instruction, domain_file=str(self.domain_path), output_dir=str(self.output_dir))

        print("="*80)
        mode_names = {"both": "DUAL BRANCH COMPARISON", "llm_agentspeak": "LLM AGENTSPEAK BASELINE", "fond": "FOND PLANNING BRANCH"}
        print(f"LTL-BDI PIPELINE - {mode_names.get(mode, 'DUAL BRANCH COMPARISON')}")
        print("="*80)
        print(f"\nNatural Language Instruction: \"{nl_instruction}\"")
        if mode != "both":
            print(f"Mode: {mode.upper()} only")
        print("\n" + "-"*80)

        # Stage 1: NL -> LTLf
        ltl_spec = self._stage1_parse_nl(nl_instruction)
        if not ltl_spec:
            return {"success": False, "stage": "Stage 1"}

        # Execute based on mode
        llm_agentspeak_code = None
        fond_plan = None
        comparison_results = None

        if mode in ["both", "llm_agentspeak"]:
            # Stage 2A: Branch A - LLM AgentSpeak Generation (Baseline)
            llm_agentspeak_code = self._stage2a_llm_agentspeak_generation(ltl_spec)
            if not llm_agentspeak_code and mode == "llm_agentspeak":
                return {"success": False, "stage": "Stage 2A"}

        if mode in ["both", "fond"]:
            # Stage 2B: Branch B - FOND Planning (Research)
            fond_plan = self._stage2b_fond_planning(ltl_spec)
            if not fond_plan and mode == "fond":
                return {"success": False, "stage": "Stage 2B"}

        # Stage 3: Execution & Comparison
        if mode == "both":
            comparison_results = self._stage3_execute_and_compare(
                ltl_spec,
                llm_agentspeak_code,
                fond_plan
            )
        elif mode == "llm_agentspeak":
            comparison_results = self._stage3_execute_single(ltl_spec, llm_agentspeak_code, "llm_agentspeak")
        elif mode == "fond":
            comparison_results = self._stage3_execute_single(ltl_spec, fond_plan, "fond")

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)

        # End logger and save results
        log_filepath = self.logger.end_pipeline(success=True)
        print(f"\nExecution log saved to: {log_filepath}")

        return {
            "success": True,
            "mode": mode,
            "ltl_spec": ltl_spec,
            "llm_agentspeak_code": llm_agentspeak_code,
            "fond_plan": fond_plan,
            "results": comparison_results
        }

    def _stage1_parse_nl(self, nl_instruction: str):
        """Stage 1: Natural Language -> LTLf Specification"""
        print("\n[STAGE 1] Natural Language -> LTLf Specification")
        print("-"*80)

        parser = NLToLTLParser(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model
        )

        try:
            ltl_spec, prompt_dict, response_text = parser.parse(nl_instruction)
            self.logger.log_stage1(nl_instruction, ltl_spec, "Success")

            formulas_string = [f.to_string() for f in ltl_spec.formulas]
            print(f"✓ LTLf Formula: {formulas_string}")
            print(f"  Objects: {ltl_spec.objects}")
            print(f"  Initial State: {ltl_spec.initial_state}")

            return ltl_spec

        except Exception as e:
            self.logger.log_stage1(nl_instruction, None, "Failed", str(e))
            print(f"✗ Stage 1 Failed: {e}")
            return None

    def _stage2a_llm_agentspeak_generation(self, ltl_spec):
        """Stage 2A: LTLf -> LLM AgentSpeak Generation (Branch A - Baseline)"""
        print("\n[STAGE 2A] BRANCH A: LLM AgentSpeak Generation (Baseline)")
        print("-"*80)

        generator = AgentSpeakGenerator(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model
        )

        try:
            # Convert ltl_spec to dict format expected by AgentSpeakGenerator
            ltl_dict = {
                "objects": ltl_spec.objects,
                "initial_state": ltl_spec.initial_state,
                "formulas_string": [f.to_string() for f in ltl_spec.formulas]
            }

            asl_code, prompt_dict, response_text = generator.generate_agentspeak(
                ltl_dict,
                'blocksworld',
                self.domain_actions,
                self.domain_predicates
            )

            self.logger.log_stage3a(ltl_spec, asl_code, "Success")

            print(f"✓ AgentSpeak Code Generated")
            print("  First few lines:")
            for line in asl_code.split('\n')[:5]:
                if line.strip():
                    print(f"    {line}")

            return asl_code

        except Exception as e:
            self.logger.log_stage3a(ltl_spec, None, "Failed", str(e))
            print(f"✗ Stage 2A Failed: {e}")
            return None

    def _stage2b_fond_planning(self, ltl_spec):
        """Stage 2B: LTLf -> PDDL Problem -> FOND Plan (Branch B)"""
        print("\n[STAGE 2B] BRANCH B: FOND Planning (PR2)")
        print("-"*80)

        try:
            # Step 1: Generate PDDL problem from LTLf
            pddl_generator = PDDLProblemGenerator("blocksworld")
            pddl_problem = pddl_generator.generate_problem(
                ltl_spec.to_dict(),
                "ltl_problem"
            )

            print("✓ PDDL Problem Generated")
            print(f"  Objects: {ltl_spec.objects}")
            print(f"  Goal Formulas: {[f.to_string() for f in ltl_spec.formulas]}")

            # Log Stage 2 success (PDDL problem generation)
            self.logger.log_stage2(ltl_spec, pddl_problem, "Success")

            # Step 2: Plan with PR2/PRP FOND planner
            planner = PR2Planner()
            plan = planner.solve_from_strings(
                str(self.domain_path.read_text()),
                pddl_problem
            )

            if plan:
                print(f"✓ FOND Plan Generated ({len(plan)} actions)")
                for i, (action, params) in enumerate(plan, 1):
                    print(f"  {i}. {action}({', '.join(params)})")

                # Log Stage 3 success (PR2 planning)
                self.logger.log_stage3a(pddl_problem, plan, "Success")
                return plan
            else:
                print("✗ No plan found by FOND planner")
                # Log Stage 3 failure
                self.logger.log_stage3a(pddl_problem, None, "Failed", "No plan found by PR2 planner")
                return None

        except Exception as e:
            print(f"✗ Stage 2B Failed: {e}")
            import traceback
            traceback.print_exc()
            # Log Stage 2 or 3 failure depending on where it failed
            self.logger.log_stage2(ltl_spec, None, "Failed", str(e))
            return None


    def _stage3_execute_single(self, ltl_spec, plan_or_code, branch_type):
        """Stage 3: Execute single branch only"""
        print("\n[STAGE 3] Execution")
        print("-"*80)

        # Create initial state
        beliefs = []
        for pred_dict in ltl_spec.initial_state:
            for pred_name, args in pred_dict.items():
                if pred_name == 'ontable' and args:
                    for block in args:
                        beliefs.append(f"on({block}, table)")
                elif args:
                    beliefs.append(f"{pred_name}({', '.join(args)})")
                else:
                    beliefs.append(pred_name)

        init_state = BlocksworldState()
        init_state.from_beliefs(beliefs)

        formulas_string_list = [f.to_string() for f in ltl_spec.formulas]

        try:
            if branch_type == "llm_agentspeak":
                # Execute LLM-generated AgentSpeak code
                executor = AgentSpeakExecutor(init_state.copy())
                result = executor.execute(plan_or_code, formulas_string_list)
                result['branch'] = branch_type
            elif branch_type == "fond":
                # Execute FOND plan
                env = BlocksworldEnvironment(init_state.copy())
                executor = ClassicalPlanExecutor(env)
                result = executor.execute(plan_or_code)
                result['branch'] = branch_type
                result['plan_length'] = len(plan_or_code)

            # Check goal satisfaction
            final_state = result.get('final_state', [])
            goals_satisfied = all(
                self._check_ltl_satisfaction(final_state, goal)
                for goal in formulas_string_list
            )

            print(f"\nExecution: {'Success' if result['success'] else 'Failed'}")
            print(f"Goal Satisfaction: {'True' if goals_satisfied else 'False'}")
            print(f"Final State: {final_state}")

            return result

        except Exception as e:
            print(f"Execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _stage3_execute_and_compare(self, ltl_spec, llm_agentspeak_code, fond_plan):
        """Stage 3: Execution & Comparative Evaluation"""
        print("\n[STAGE 3] Execution & Comparative Evaluation")
        print("-"*80)

        # Create initial state
        beliefs = []
        for pred_dict in ltl_spec.initial_state:
            for pred_name, args in pred_dict.items():
                if pred_name == 'ontable' and args:
                    for block in args:
                        beliefs.append(f"on({block}, table)")
                elif args:
                    beliefs.append(f"{pred_name}({', '.join(args)})")
                else:
                    beliefs.append(pred_name)

        init_state = BlocksworldState()
        init_state.from_beliefs(beliefs)

        formulas_string_list = [f.to_string() for f in ltl_spec.formulas]

        try:
            evaluator = ComparativeEvaluator()
            results = evaluator.evaluate(
                init_state,
                llm_agentspeak_code,
                fond_plan,
                formulas_string_list
            )

            self.logger.log_stage4(results, "Success")

            # Print report
            print("\n" + evaluator.generate_report())

            return results

        except Exception as e:
            self.logger.log_stage4(None, "Failed", str(e))
            print(f"✗ Stage 3 Failed: {e}")
            return None

    def _check_ltl_satisfaction(self, final_state, ltl_goal: str) -> bool:
        """Check if final state satisfies LTLf goal"""
        if ltl_goal.startswith('F(') and ltl_goal.endswith(')'):
            goal_predicate = ltl_goal[2:-1]
            goal_normalized = goal_predicate.replace(' ', '')
            for state_pred in final_state:
                if state_pred.replace(' ', '') == goal_normalized:
                    return True
            return False
        return True
