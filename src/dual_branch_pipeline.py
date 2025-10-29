"""
Dual-Branch Pipeline: LLM AgentSpeak vs FOND Planning

Implements the 3-stage architecture:
    Stage 1: NL -> LTLf Goals
    Stage 2A: Branch A - LLM AgentSpeak Generation (Baseline)
    Stage 2B: Branch B - FOND Planning (Research: LTLf -> PDDL -> PR2 FOND planner)

Current Implementation: Stages 1 and 2 only
Stage 3 (Execution & Evaluation) is planned for future development
"""

from pathlib import Path
from typing import Dict, Any

from config import get_config
from stage1_interpretation.ltl_parser import NLToLTLParser
from stage2_planning.pddl_problem_generator import PDDLProblemGenerator
from stage2_planning.agentspeak_generator import AgentSpeakGenerator
from stage2_planning.pr2_planner import PR2Planner
from pipeline_logger import PipelineLogger


class DualBranchPipeline:
    """
    Dual-branch pipeline implementing Stages 1 and 2

    Stage 1: Natural Language -> LTLf Specification
    Stage 2A: LTLf -> LLM AgentSpeak Code (Branch A - Baseline)
    Stage 2B: LTLf -> PDDL -> FOND Plan (Branch B - Research)
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
        Execute dual-branch pipeline (Stages 1 and 2 only)

        Args:
            nl_instruction: Natural language instruction
            mode: Execution mode - "both", "llm_agentspeak", or "fond"
                - "both": Run both branches (Stage 2A and 2B)
                - "llm_agentspeak": Run only LLM AgentSpeak generation (Stage 2A)
                - "fond": Run only FOND planning (Stage 2B)

        Returns:
            Results from Stage 1 and Stage 2 (no execution/evaluation)
        """
        # Start logger
        self.logger.start_pipeline(nl_instruction, domain_file=str(self.domain_path), output_dir=str(self.output_dir))

        print("="*80)
        mode_names = {
            "both": "DUAL BRANCH (STAGES 1-2)",
            "llm_agentspeak": "LLM AGENTSPEAK BRANCH (STAGES 1-2A)",
            "fond": "FOND PLANNING BRANCH (STAGES 1-2B)"
        }
        print(f"LTL-BDI PIPELINE - {mode_names.get(mode, 'DUAL BRANCH')}")
        print("="*80)
        print(f"\nNatural Language Instruction: \"{nl_instruction}\"")
        if mode != "both":
            print(f"Mode: {mode.upper()} only")
        print("\n" + "-"*80)

        # Stage 1: NL -> LTLf
        ltl_spec = self._stage1_parse_nl(nl_instruction)
        if not ltl_spec:
            return {"success": False, "stage": "Stage 1", "error": "LTLf parsing failed"}

        # Stage 2: Dual branches
        llm_agentspeak_code = None
        fond_plan = None

        if mode in ["both", "llm_agentspeak"]:
            # Stage 2A: LLM AgentSpeak Generation
            llm_agentspeak_code = self._stage2a_llm_agentspeak_generation(ltl_spec)
            if not llm_agentspeak_code and mode == "llm_agentspeak":
                return {"success": False, "stage": "Stage 2A", "error": "AgentSpeak generation failed"}

        if mode in ["both", "fond"]:
            # Stage 2B: FOND Planning
            fond_plan = self._stage2b_fond_planning(ltl_spec)
            if not fond_plan and mode == "fond":
                return {"success": False, "stage": "Stage 2B", "error": "FOND planning failed"}

        print("\n" + "="*80)
        print("STAGES 1-2 COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNote: Stage 3 (Execution & Evaluation) not yet implemented")

        # End logger and save results
        log_filepath = self.logger.end_pipeline(success=True)
        print(f"\nExecution log saved to: {log_filepath}")

        return {
            "success": True,
            "mode": mode,
            "ltl_spec": ltl_spec,
            "llm_agentspeak_code": llm_agentspeak_code,
            "fond_plan": fond_plan
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

            # Log to stage3a for now (will rename logger methods later)
            self.logger.log_stage3a(ltl_spec, asl_code, "Success")

            print(f"✓ AgentSpeak Code Generated")
            print("  First few lines:")
            for line in asl_code.split('\n')[:5]:
                if line.strip():
                    print(f"    {line}")

            # Save to output
            output_file = self.output_dir / "agentspeak_generated.asl"
            output_file.write_text(asl_code)
            print(f"  Saved to: {output_file}")

            return asl_code

        except Exception as e:
            self.logger.log_stage3a(ltl_spec, None, "Failed", str(e))
            print(f"✗ Stage 2A Failed: {e}")
            return None

    def _stage2b_fond_planning(self, ltl_spec):
        """Stage 2B: LTLf -> PDDL Problem -> FOND Plan (Branch B - Research)"""
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

            # Save PDDL problem
            pddl_output_file = self.output_dir / "problem_generated.pddl"
            pddl_output_file.write_text(pddl_problem)
            print(f"  Saved to: {pddl_output_file}")

            # Step 2: Plan with PR2/PRP FOND planner
            planner = PR2Planner()
            plan, pr2_info = planner.solve_from_strings(
                str(self.domain_path.read_text()),
                pddl_problem,
                verbose=False  # Set to True for detailed PR2 output
            )

            # Log PR2 info
            if pr2_info["success"]:
                print(f"✓ FOND Plan Generated ({pr2_info['plan_length']} actions)")
                for i, (action, params) in enumerate(plan, 1):
                    print(f"  {i}. {action}({', '.join(params)})")

                # Log Stage 3 success (PR2 planning) with PR2 output
                self.logger.log_stage3a(pddl_problem, plan, "Success")

                # Save plan
                plan_output_file = self.output_dir / "fond_plan.txt"
                with open(plan_output_file, 'w') as f:
                    f.write(f"# FOND Plan generated by PR2/PRP planner\n")
                    f.write(f"# Plan length: {pr2_info['plan_length']}\n\n")
                    for action, params in plan:
                        f.write(f"{action}({', '.join(params)})\n")
                print(f"  Saved to: {plan_output_file}")

                # Save PR2 output log
                pr2_log_file = self.output_dir / "pr2_output.log"
                pr2_log_file.write_text(pr2_info["pr2_output"])
                print(f"  PR2 log saved to: {pr2_log_file}")

                return plan
            else:
                error_msg = pr2_info.get("error", "No plan found by PR2 planner")
                print(f"✗ FOND Planning Failed: {error_msg}")
                self.logger.log_stage3a(pddl_problem, None, "Failed", error_msg)

                # Still save PR2 output for debugging
                if pr2_info.get("pr2_output"):
                    pr2_log_file = self.output_dir / "pr2_output_failed.log"
                    pr2_log_file.write_text(pr2_info["pr2_output"])
                    print(f"  PR2 error log saved to: {pr2_log_file}")

                return None

        except Exception as e:
            print(f"✗ Stage 2B Failed: {e}")
            import traceback
            traceback.print_exc()
            self.logger.log_stage2(ltl_spec, None, "Failed", str(e))
            return None
