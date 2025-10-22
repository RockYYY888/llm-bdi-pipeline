"""
Pipeline Orchestrator

Coordinates the execution of all three stages:
- Stage 1: Natural Language -> LTL Specification
- Stage 2: LTL Specification -> PDDL Problem
- Stage 3: PDDL Problem -> Action Plan
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from config import get_config
from stage1_interpretation.ltl_parser import NLToLTLParser
from stage2_translation.ltl_to_pddl import LTLToPDDLConverter
from stage3_codegen.pddl_planner import PDDLPlanner
from stage3_codegen.llm_planner import LLMPlanner
from pipeline_logger import PipelineLogger


class PipelineOrchestrator:
    """
    Pipeline Orchestrator - Coordinates all stages

    Orchestrates the complete workflow:
    1. Natural Language -> LTL Specification (Stage 1)
    2. LTL -> PDDL Problem (Stage 2)
    3. PDDL -> Plan (Stage 3 - STOPS HERE)

    This is NOT a pipeline itself, but the coordinator that calls each stage.
    """

    def __init__(self, domain_file: str = "domains/blocksworld/domain.pddl"):
        """
        Initialize orchestrator

        Args:
            domain_file: Path to PDDL domain file
        """
        self.config = get_config()
        self.domain_file = domain_file

        # Initialize components
        api_key = self.config.openai_api_key if self.config.validate() else None
        model = self.config.openai_model
        base_url = self.config.openai_base_url

        # Stage 1: NL to LTL parser
        self.ltl_parser = NLToLTLParser(
            api_key=api_key,
            model=model,
            base_url=base_url
        )

        # Stage 2: LTL to PDDL converter (LLM-based with fallback)
        self.pddl_converter = LTLToPDDLConverter(
            api_key=api_key,
            model=model,
            base_url=base_url
        )

        # Stage 3: Planner (classical or LLM-based)
        self.use_llm_planner = self.config.use_llm_planner
        self.classical_planner = PDDLPlanner()
        self.llm_planner = LLMPlanner(
            api_key=api_key,
            model=model,
            base_url=base_url
        ) if self.use_llm_planner else None

    def execute(self,
                nl_instruction: str,
                output_dir: str = "output",
                enable_logging: bool = True) -> Dict[str, Any]:
        """
        Orchestrate the complete workflow through all stages

        Args:
            nl_instruction: Natural language instruction
            output_dir: Base directory for output files (timestamp subdirectory will be created)
            enable_logging: Whether to save execution logs

        Returns:
            Dictionary with results from each stage
        """
        # Create timestamp for this execution
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create timestamp-specific output directory
        timestamped_output_dir = Path(output_dir) / timestamp
        timestamped_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger with the same timestamp
        logger = PipelineLogger() if enable_logging else None
        if logger:
            logger.start_pipeline(nl_instruction, self.domain_file, str(timestamped_output_dir), timestamp)

        results = {
            "nl_instruction": nl_instruction,
            "stage1_ltl": None,
            "stage2_pddl": None,
            "stage3_plan": None
        }

        pipeline_success = False

        print("="*80)
        print("LTL-Based LLM-BDI Pipeline")
        print("="*80)
        print(f"Instruction: {nl_instruction}\n")

        # ===== Stage 1: Natural Language -> LTL =====
        print("Stage 1: Natural Language -> LTL Specification")
        print("-"*80)

        try:
            # Parse returns (spec, prompt_dict, response_text)
            ltl_spec, stage1_prompt, stage1_response = self.ltl_parser.parse(nl_instruction)

            print(f"Objects: {ltl_spec.objects}")
            print(f"Initial State: {len(ltl_spec.initial_state)} predicates")
            print(f"LTL Formulas:")
            for i, formula in enumerate(ltl_spec.formulas, 1):
                print(f"  {i}. {formula.to_string()}")

            results["stage1_ltl"] = ltl_spec.to_dict()

            # Log Stage 1 success with LLM data
            if logger:
                used_llm = self.ltl_parser.client is not None
                model = self.ltl_parser.model if used_llm else None
                logger.log_stage1_success(
                    ltl_spec.to_dict(), used_llm, model,
                    stage1_prompt, stage1_response
                )

            # Save LTL specification
            ltl_file = timestamped_output_dir / "ltl_specification.json"
            with open(ltl_file, 'w') as f:
                json.dump(ltl_spec.to_dict(), indent=2, fp=f)
            print(f"\n✓ LTL specification saved: {ltl_file}")

        except Exception as e:
            print(f"\n✗ Stage 1 failed: {e}")
            if logger:
                logger.log_stage1_error(str(e))
                logger.end_pipeline(success=False)
            raise

        # ===== Stage 2: LTL -> PDDL Problem =====
        print("\n" + "="*80)
        print("Stage 2: LTL -> PDDL Problem")
        print("-"*80)

        try:
            problem_name = "ltl_generated_problem"

            # Convert returns (pddl_problem, prompt_dict, response_text)
            pddl_problem, stage2_prompt, stage2_response = self.pddl_converter.convert(
                problem_name, ltl_spec, self.domain_file
            )

            # Count predicates
            goal_count = pddl_problem.count("(:goal")
            init_count = len(ltl_spec.initial_state)

            print(f"Problem name: {problem_name}")
            print(f"Objects: {len(ltl_spec.objects)}")
            print(f"Initial predicates: {init_count}")
            print(f"Goal predicates: {len(ltl_spec.formulas)}")

            results["stage2_pddl"] = pddl_problem

            # Log Stage 2 success with LLM data
            if logger:
                used_llm = self.pddl_converter.client is not None
                model = self.pddl_converter.model if used_llm else None
                logger.log_stage2_success(
                    pddl_problem, used_llm, model,
                    stage2_prompt, stage2_response
                )

            # Save PDDL problem
            pddl_file = timestamped_output_dir / "problem.pddl"
            with open(pddl_file, 'w') as f:
                f.write(pddl_problem)
            print(f"\n✓ PDDL problem saved: {pddl_file}")

            # Check for constraints
            constraints = self.pddl_converter.get_constraints(ltl_spec)
            if constraints:
                print(f"\nGlobally constraints (for verification): {len(constraints)}")
                for c in constraints:
                    print(f"  - {c['formula_string']}")

        except Exception as e:
            print(f"\n✗ Stage 2 failed: {e}")
            if logger:
                logger.log_stage2_error(str(e))
                logger.end_pipeline(success=False)
            raise

        # ===== Stage 3: PDDL -> Plan =====
        print("\n" + "="*80)
        print("Stage 3: PDDL Planning")
        print("-"*80)

        try:
            print(f"Domain: {self.domain_file}")
            print(f"Problem: {pddl_file}")

            # Choose planner based on configuration
            if self.use_llm_planner and self.llm_planner:
                print("Using LLM-based planner (can handle G, X, U constraints)...")
                plan_result = self.llm_planner.solve(self.domain_file, pddl_file, ltl_spec)
                plan, stage3_prompt, stage3_response = plan_result
            else:
                print("Using classical PDDL planner...")
                plan = self.classical_planner.solve(self.domain_file, pddl_file)
                stage3_prompt = None
                stage3_response = None

            if plan:
                print(f"\n✓ Plan found: {len(plan)} actions")
                print("\nPlan:")
                for i, (action, params) in enumerate(plan, 1):
                    print(f"  {i}. {action}({', '.join(params)})")

                results["stage3_plan"] = plan
                pipeline_success = True

                # Log Stage 3 success with LLM data if available
                if logger:
                    if stage3_prompt and stage3_response:
                        # LLM planner was used
                        logger.log_stage3_success(
                            plan,
                            used_llm=True,
                            model=self.llm_planner.model,
                            prompt=stage3_prompt,
                            response=stage3_response
                        )
                    else:
                        # Classical planner was used
                        logger.log_stage3_success(plan)

                # Save plan
                plan_file = timestamped_output_dir / "plan.txt"
                with open(plan_file, 'w') as f:
                    f.write(f"Plan for: {nl_instruction}\n")
                    f.write(f"LTL Formulas:\n")
                    for i, formula in enumerate(ltl_spec.formulas, 1):
                        f.write(f"  {i}. {formula.to_string()}\n")
                    f.write(f"\nPlan ({len(plan)} actions):\n")
                    for i, (action, params) in enumerate(plan, 1):
                        f.write(f"  {i}. {action}({', '.join(params)})\n")
                print(f"\n✓ Plan saved: {plan_file}")

            else:
                print("\n✗ No plan found!")
                results["stage3_plan"] = None
                error_msg = "No plan found by planner"
                if logger:
                    logger.log_stage3_error(error_msg)

        except Exception as e:
            print(f"\n✗ Stage 3 failed: {e}")
            if logger:
                logger.log_stage3_error(str(e))
                logger.end_pipeline(success=False)
            raise

        # ===== Pipeline Complete =====
        print("\n" + "="*80)
        print("Pipeline Execution Complete")
        print("="*80)
        print(f"\nStage 1: LTL Specification   ✓")
        print(f"Stage 2: PDDL Problem        ✓")
        print(f"Stage 3: Plan Generation     {'✓' if plan else '✗'}")
        print(f"\nNote: Pipeline stops at plan generation (no code generation)")

        # Print output summary
        print(f"\n📁 Pipeline output saved: {timestamped_output_dir}/")
        print(f"   - LTL specification: {timestamped_output_dir}/ltl_specification.json")
        print(f"   - PDDL problem: {timestamped_output_dir}/problem.pddl")
        if pipeline_success:
            print(f"   - Plan: {timestamped_output_dir}/plan.txt")

        # Save execution log
        if logger:
            log_file = logger.end_pipeline(success=pipeline_success)
            log_dir = log_file.parent
            print(f"\n📝 Execution log saved: {log_dir}/")
            print(f"   - JSON format: {log_dir}/execution.json")
            print(f"   - Readable format: {log_dir}/execution.txt")

        return results
