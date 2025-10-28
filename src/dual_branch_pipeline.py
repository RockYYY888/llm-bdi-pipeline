"""
Dual-Branch Pipeline

Orchestrates the complete dual-branch workflow:
    Stage 1: NL -> LTLf
    Stage 2: LTLf -> PDDL
    Stage 3A: PDDL -> Classical Plan (Branch A)
    Stage 3B: LTLf -> AgentSpeak (Branch B)
    Stage 4: Execution & Comparison
"""

import re
from pathlib import Path
from typing import Dict, Any

from config import get_config
from stage1_interpretation.ltl_parser import NLToLTLParser
from stage2_translation.ltl_to_pddl import LTLToPDDLConverter
from stage3_codegen.pddl_planner import PDDLPlanner
from stage3_codegen.agentspeak_generator import AgentSpeakGenerator
from stage4_execution.blocksworld_simulator import BlocksworldState
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

    def execute(self, nl_instruction: str) -> Dict[str, Any]:
        """
        Execute complete dual-branch pipeline

        Args:
            nl_instruction: Natural language instruction

        Returns:
            Complete results including both branches and comparison
        """
        print("="*80)
        print("LTL-BDI PIPELINE - DUAL BRANCH DEMONSTRATION")
        print("="*80)
        print(f"\nNatural Language Instruction: \"{nl_instruction}\"")
        print("\n" + "-"*80)

        # Stage 1: NL -> LTLf
        ltl_spec = self._stage1_parse_nl(nl_instruction)
        if not ltl_spec:
            return {"success": False, "stage": "Stage 1"}

        # Stage 2: LTLf -> PDDL
        pddl_problem = self._stage2_generate_pddl(ltl_spec)
        if not pddl_problem:
            return {"success": False, "stage": "Stage 2"}

        # Stage 3A: Classical Planning
        classical_plan = self._stage3a_classical_plan(pddl_problem)
        if not classical_plan:
            return {"success": False, "stage": "Stage 3A"}

        # Stage 3B: AgentSpeak Generation
        asl_code = self._stage3b_generate_agentspeak(ltl_spec)
        if not asl_code:
            return {"success": False, "stage": "Stage 3B"}

        # Stage 4: Execution & Comparison
        comparison_results = self._stage4_execute_and_compare(
            ltl_spec,
            classical_plan,
            asl_code
        )

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)

        return {
            "success": True,
            "ltl_spec": ltl_spec,
            "pddl_problem": pddl_problem,
            "classical_plan": classical_plan,
            "agentspeak_code": asl_code,
            "comparison": comparison_results
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

    def _stage2_generate_pddl(self, ltl_spec):
        """Stage 2: LTLf -> PDDL Problem"""
        print("\n[STAGE 2] LTLf -> PDDL Problem")
        print("-"*80)

        converter = LTLToPDDLConverter(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model
        )

        try:
            pddl_problem, prompt_dict, response_text = converter.convert(
                "stack-blocks",
                ltl_spec,
                str(self.domain_path)
            )
            self.logger.log_stage2(ltl_spec, pddl_problem, "Success")

            print(f"✓ PDDL Problem Generated")

            # Extract and display key information
            objects_match = re.search(r':objects\s+([^\)]+)', pddl_problem)
            goal_match = re.search(r':goal\s+\(and\s+([^\)]+)\)', pddl_problem)
            print(f"  Objects: {objects_match.group(1).strip() if objects_match else 'N/A'}")
            print(f"  Goal: {goal_match.group(1).strip() if goal_match else 'N/A'}")

            return pddl_problem

        except Exception as e:
            self.logger.log_stage2(ltl_spec, None, "Failed", str(e))
            print(f"✗ Stage 2 Failed: {e}")
            return None

    def _stage3a_classical_plan(self, pddl_problem):
        """Stage 3A: PDDL -> Classical Plan (Branch A)"""
        print("\n[STAGE 3A] BRANCH A: Classical PDDL Planning")
        print("-"*80)

        planner = PDDLPlanner()

        try:
            # Read domain file
            domain_str = self.domain_path.read_text()

            # Solve using classical planner
            plan = planner.solve_from_strings(domain_str, pddl_problem)

            if not plan:
                raise RuntimeError("Classical planner failed to find a solution")

            self.logger.log_stage3a(pddl_problem, plan, "Success")

            print(f"✓ Classical Plan Generated ({len(plan)} actions)")
            for i, (action, params) in enumerate(plan, 1):
                print(f"  {i}. {action}({', '.join(params)})")

            return plan

        except Exception as e:
            self.logger.log_stage3a(pddl_problem, None, "Failed", str(e))
            print(f"✗ Stage 3A Failed: {e}")
            return None

    def _stage3b_generate_agentspeak(self, ltl_spec):
        """Stage 3B: LTLf -> AgentSpeak Plan Library (Branch B)"""
        print("\n[STAGE 3B] BRANCH B: LLM AgentSpeak Generation")
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

    def _stage4_execute_and_compare(self, ltl_spec, classical_plan, asl_code):
        """Stage 4: Execution & Comparative Evaluation"""
        print("\n[STAGE 4] Execution & Comparative Evaluation")
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
                classical_plan,
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
            print(f"✗ Stage 4 Failed: {e}")
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
