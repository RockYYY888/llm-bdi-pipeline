"""
Dual-Branch Pipeline (LLM-Based)

Orchestrates the complete dual-branch workflow:
    Stage 1: NL -> LTLf Goals
    Stage 2A: Branch A - LLM Policy Generation (direct from LTLf)
    Stage 2B: Branch B - LLM AgentSpeak Generation (direct from LTLf)
    Stage 3: Execution & Comparison

Both branches use LLM directly from LTLf goals without PDDL intermediate step.
"""

import re
from pathlib import Path
from typing import Dict, Any

from config import get_config
from stage1_interpretation.ltl_parser import NLToLTLParser
from stage3_codegen.llm_policy_generator import LLMPolicyGenerator
from stage3_codegen.agentspeak_generator import AgentSpeakGenerator
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
            mode: Execution mode - "both", "llm", or "asl"

        Returns:
            Complete results including branch results and comparison (if mode="both")
        """
        print("="*80)
        mode_names = {"both": "DUAL BRANCH DEMONSTRATION", "llm": "LLM POLICY BRANCH", "asl": "AGENTSPEAK BRANCH"}
        print(f"LTL-BDI PIPELINE - {mode_names.get(mode, 'DUAL BRANCH DEMONSTRATION')}")
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
        llm_plan = None
        asl_code = None
        comparison_results = None

        if mode in ["both", "llm"]:
            # Stage 2A: Branch A - LLM Policy Generation
            llm_plan = self._stage2a_llm_policy(ltl_spec)
            if not llm_plan and mode == "llm":
                return {"success": False, "stage": "Stage 2A"}

        if mode in ["both", "asl"]:
            # Stage 2B: Branch B - AgentSpeak Generation
            asl_code = self._stage2b_generate_agentspeak(ltl_spec)
            if not asl_code and mode == "asl":
                return {"success": False, "stage": "Stage 2B"}

        # Stage 3: Execution & Comparison
        if mode == "both":
            comparison_results = self._stage3_execute_and_compare(
                ltl_spec,
                llm_plan,
                asl_code
            )
        elif mode == "llm":
            comparison_results = self._stage3_execute_single(ltl_spec, llm_plan, "llm")
        elif mode == "asl":
            comparison_results = self._stage3_execute_single(ltl_spec, asl_code, "asl")

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)

        return {
            "success": True,
            "mode": mode,
            "ltl_spec": ltl_spec,
            "llm_plan": llm_plan,
            "agentspeak_code": asl_code,
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

    def _stage2a_llm_policy(self, ltl_spec):
        """Stage 2A: LTLf -> LLM-Generated Policy (Branch A)"""
        print("\n[STAGE 2A] BRANCH A: LLM Policy Generation")
        print("-"*80)

        generator = LLMPolicyGenerator(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model
        )

        try:
            # Convert ltl_spec to dict format expected by LLMPolicyGenerator
            ltl_dict = {
                "objects": ltl_spec.objects,
                "initial_state": ltl_spec.initial_state,
                "formulas_string": [f.to_string() for f in ltl_spec.formulas]
            }

            plan, prompt_dict, response_text = generator.generate_plan(
                ltl_dict,
                'blocksworld',
                self.domain_actions,
                self.domain_predicates
            )

            self.logger.log_stage3a(ltl_spec, plan, "Success")

            print(f"✓ LLM Policy Generated ({len(plan)} actions)")
            for i, (action, params) in enumerate(plan, 1):
                print(f"  {i}. {action}({', '.join(params)})")

            return plan

        except Exception as e:
            self.logger.log_stage3a(ltl_spec, None, "Failed", str(e))
            print(f"✗ Stage 2A Failed: {e}")
            return None

    def _stage2b_generate_agentspeak(self, ltl_spec):
        """Stage 2B: LTLf -> AgentSpeak Plan Library (Branch B)"""
        print("\n[STAGE 2B] BRANCH B: LLM AgentSpeak Generation")
        print("-"*80)

        generator = AgentSpeakGenerator(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model
        )

        try:
            asl_code, prompt, response = generator.generate(
                ltl_spec.to_dict(),
                'blocksworld',
                self.domain_actions,
                self.domain_predicates
            )

            self.logger.log_stage3b(ltl_spec, asl_code, "Success")

            print(f"✓ AgentSpeak Plan Library Generated")
            print(f"  Code Length: {len(asl_code)} characters")
            print(f"  Plans: {asl_code.count('+!')}")

            # Save ASL code
            asl_path = self.output_dir / "generated_agent.asl"
            asl_path.write_text(asl_code)
            print(f"  Saved to: {asl_path}")

            return asl_code

        except Exception as e:
            self.logger.log_stage3b(ltl_spec, None, "Failed", str(e))
            print(f"✗ Stage 3B Failed: {e}")
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
            if branch_type == "llm":
                # Execute LLM plan
                env = BlocksworldEnvironment(init_state.copy())
                executor = ClassicalPlanExecutor(env)
                result = executor.execute(plan_or_code)
                result['branch'] = 'llm'
                result['plan_length'] = len(plan_or_code)
            else:  # asl
                # Execute AgentSpeak
                env = BlocksworldEnvironment(init_state.copy())
                agentspeak_goal = self._create_multi_goal_agentspeak_goal(formulas_string_list)
                executor = AgentSpeakExecutor(env)
                result = executor.execute(plan_or_code, agentspeak_goal)
                result['branch'] = 'agentspeak'

            # Check goal satisfaction
            final_state = result.get('final_state' if branch_type == 'llm' else 'env_final_state', [])
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

    def _stage3_execute_and_compare(self, ltl_spec, llm_plan, asl_code):
        """Stage 3: Execution & Comparative Evaluation"""
        print("\n[STAGE 3] Execution & Comparative Evaluation")
        print("-"*80)

        # Create initial state
        # Convert dict-based initial_state to string-based beliefs
        # FIX: Convert 'ontable(X)' to 'on(X, table)' for blocksworld domain
        beliefs = []
        for pred_dict in ltl_spec.initial_state:
            for pred_name, args in pred_dict.items():
                if pred_name == 'ontable' and args:
                    # Convert ontable(X) to on(X, table)
                    for block in args:
                        beliefs.append(f"on({block}, table)")
                elif args:
                    beliefs.append(f"{pred_name}({', '.join(args)})")
                else:
                    beliefs.append(pred_name)

        init_state = BlocksworldState()
        init_state.from_beliefs(beliefs)

        # Extract AgentSpeak goal from LTLf formula(s)
        formulas_string_list = [f.to_string() for f in ltl_spec.formulas]

        # Multi-goal support: Use composite goal for multiple formulas
        agentspeak_goal = self._create_multi_goal_agentspeak_goal(formulas_string_list)

        try:
            evaluator = ComparativeEvaluator()
            results = evaluator.evaluate(
                init_state,
                llm_plan,
                asl_code,
                agentspeak_goal,
                formulas_string_list  # Pass ALL formulas for verification
            )

            self.logger.log_stage4(results, "Success")

            # Print report
            print("\n" + evaluator.generate_report())

            return results

        except Exception as e:
            self.logger.log_stage4(None, "Failed", str(e))
            print(f"✗ Stage 3 Failed: {e}")
            return None

    def _extract_agentspeak_goal(self, ltl_formula: str) -> str:
        """
        Extract AgentSpeak goal from single LTLf formula

        For MVP: Simple extraction for F(on(X,Y)) -> achieve_on_X_Y or on(X,Y)
        """
        # Match F(on(X, Y))
        match = re.match(r'F\(on\((\w+),\s*(\w+)\)\)', ltl_formula)
        if match:
            obj1, obj2 = match.groups()
            # Use the main goal name that LLM generates
            # Check generated code - it uses achieve_on_X_Y or declarative on(X,Y)
            return f"achieve_on_{obj1}_{obj2}"

        # Default: use formula as-is
        return ltl_formula

    def _create_multi_goal_agentspeak_goal(self, formulas: list) -> str:
        """
        Create a composite AgentSpeak goal for multiple LTLf formulas

        Strategy: Create a main goal that achieves all sub-goals sequentially
        Returns: "achieve_ltlf_goals" (expects LLM to generate this main goal)
        """
        if len(formulas) == 1:
            return self._extract_agentspeak_goal(formulas[0])
        else:
            # Multi-goal: Use the main composite goal that LLM should generate
            # The AgentSpeak generator is instructed to create achieve_ltlf_goals
            return "achieve_ltlf_goals"

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
