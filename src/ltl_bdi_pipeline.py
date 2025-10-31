"""
LTL-BDI Pipeline: DFA-AgentSpeak Generation

Implements the 3-stage architecture:
    Stage 1: NL -> LTLf Goals
    Stage 2: LTLf -> DFA Conversion
    Stage 3: LTLf -> AgentSpeak Code Generation

Note: FOND Planning (Branch B) has been moved to legacy/fond/
"""

from pathlib import Path
from typing import Dict, Any

from config import get_config
from stage1_interpretation.ltl_parser import NLToLTLParser
from stage2_dfa_generation.recursive_dfa_builder import RecursiveDFABuilder
from stage3_code_generation.agentspeak_generator import AgentSpeakGenerator
from pipeline_logger import PipelineLogger


class LTL_BDI_Pipeline:
    """
    LTL-BDI pipeline implementing Stages 1-3 (dfa_agentspeak mode)

    Stage 1: Natural Language -> LTLf Specification
    Stage 2: LTLf -> DFA Conversion (ltlf2dfa)
    Stage 3: LTLf -> AgentSpeak Code Generation (LLM)

    Legacy Note: FOND Planning (Branch B) has been deprecated and moved to src/legacy/fond/
    The pipeline now focuses exclusively on DFA-AgentSpeak generation.
    """

    def __init__(self, domain_file: str = None):
        """
        Initialize pipeline

        Args:
            domain_file: Path to PDDL domain file. If None, uses default blocksworld domain.
        """
        self.config = get_config()
        self.logger = PipelineLogger()

        # Domain file path
        if domain_file is None:
            # Default to blocksworld domain
            from pathlib import Path
            domain_file = str(Path(__file__).parent / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl")

        self.domain_file = domain_file

        # Parse domain to extract actions and predicates
        from stage1_interpretation.pddl_parser import PDDLParser
        self.domain = PDDLParser.parse_domain(domain_file)
        self.domain_actions = self.domain.get_action_names()
        self.domain_predicates = self.domain.get_predicate_signatures()

        # Output directory (set during execution - will use logger's directory)
        self.output_dir = None

    def execute(self, nl_instruction: str, mode: str = "dfa_agentspeak") -> Dict[str, Any]:
        """
        Execute LTL-BDI pipeline (Stages 1-3: NL -> LTLf -> DFA -> AgentSpeak)

        Args:
            nl_instruction: Natural language instruction
            mode: Execution mode (only "dfa_agentspeak" is supported)

        Returns:
            Results from Stage 1-3 (no execution/evaluation yet)
        """
        if mode != "dfa_agentspeak":
            raise ValueError(
                f"Unknown mode '{mode}'. Only 'dfa_agentspeak' is supported. "
                "FOND planning has been moved to src/legacy/fond/"
            )

        # Start logger (creates timestamped directory in logs/)
        self.logger.start_pipeline(nl_instruction, mode=mode, domain_file="N/A", output_dir="logs")

        # Use logger's directory for all output files
        self.output_dir = self.logger.current_log_dir

        print("="*80)
        print(f"LTL-BDI PIPELINE - {mode.upper()} MODE")
        print("="*80)
        print(f"\nNatural Language Instruction: \"{nl_instruction}\"")
        print(f"Mode: {mode}")
        print(f"Output directory: {self.output_dir}")
        print("\n" + "-"*80)

        # Stage 1: NL -> LTLf
        ltl_spec = self._stage1_parse_nl(nl_instruction)
        if not ltl_spec:
            return {"success": False, "stage": "Stage 1", "error": "LTLf parsing failed"}

        # Stage 2: LTLf -> Recursive DFA Generation
        dfa_result = self._stage2_recursive_dfa_generation(ltl_spec)
        if not dfa_result:
            return {"success": False, "stage": "Stage 2", "error": "DFA generation failed"}

        # Stage 3: DFAs -> AgentSpeak Code
        asl_code = self._stage3_llm_agentspeak_generation(ltl_spec, dfa_result)
        if not asl_code:
            return {"success": False, "stage": "Stage 3", "error": "AgentSpeak generation failed"}

        print("\n" + "="*80)
        print("STAGES 1-3 COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNote: Stage 4 (Execution & Evaluation) not yet implemented")

        # End logger and save results
        log_filepath = self.logger.end_pipeline(success=True)
        print(f"\nExecution log saved to: {log_filepath}")

        return {
            "success": True,
            "ltl_spec": ltl_spec,
            "agentspeak_code": asl_code
        }

    def _stage1_parse_nl(self, nl_instruction: str):
        """Stage 1: Natural Language -> LTLf Specification"""
        print("\n[STAGE 1] Natural Language -> LTLf Specification")
        print("-"*80)

        parser = NLToLTLParser(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model,
            domain_file=self.domain_file  # Pass domain file for dynamic prompt
        )

        try:
            ltl_spec, prompt_dict, response_text = parser.parse(nl_instruction)
            self.logger.log_stage1(
                nl_instruction,
                ltl_spec,
                "Success",
                model=self.config.openai_model,
                llm_prompt=prompt_dict,
                llm_response=response_text
            )

            formulas_string = [f.to_string() for f in ltl_spec.formulas]
            print(f"✓ LTLf Formula: {formulas_string}")
            print(f"  Objects: {ltl_spec.objects}")
            print(f"  (No initial state - plans work from any configuration)")

            return ltl_spec

        except Exception as e:
            self.logger.log_stage1(nl_instruction, None, "Failed", str(e))
            print(f"✗ Stage 1 Failed: {e}")
            return None

    def _stage2_recursive_dfa_generation(self, ltl_spec):
        """Stage 2: LTLf -> Recursive DFA Generation"""
        print("\n[STAGE 2] Recursive DFA Generation")
        print("-"*80)

        builder = RecursiveDFABuilder(domain_actions=self.domain_actions)

        try:
            dfa_result = builder.build(ltl_spec)

            # Log Stage 2 success
            self.logger.log_stage2_dfas(
                ltl_spec,
                dfa_result,
                "Success"
            )

            print(f"✓ DFA Decomposition Complete")
            print(f"  Root formula: {dfa_result.root_formula}")
            print(f"  Total DFAs: {len(dfa_result.all_dfas)}")
            print(f"  Physical actions: {len(dfa_result.physical_actions)}")
            print(f"  Max depth: {dfa_result.max_depth}")

            # Save DFA result to output
            output_file = self.output_dir / "dfa_decomposition.json"
            import json
            output_file.write_text(json.dumps(dfa_result.to_dict(), indent=2))
            print(f"  Saved to: {output_file}")

            return dfa_result

        except Exception as e:
            self.logger.log_stage2_dfas(ltl_spec, None, "Failed", str(e))
            print(f"✗ Stage 2 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _stage3_llm_agentspeak_generation(self, ltl_spec, dfa_result):
        """Stage 3: DFAs -> LLM AgentSpeak Generation"""
        print("\n[STAGE 3] LLM AgentSpeak Generation from DFAs")
        print("-"*80)

        generator = AgentSpeakGenerator(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model
        )

        try:
            # Convert ltl_spec to dict format
            # Note: No initial_state - plans must work from ANY configuration
            ltl_dict = {
                "objects": ltl_spec.objects,
                "formulas_string": [f.to_string() for f in ltl_spec.formulas]
            }

            # Pass DFA result to generator
            asl_code, prompt_dict, response_text = generator.generate(
                ltl_dict,
                'blocksworld',
                self.domain_actions,
                self.domain_predicates,
                dfa_result=dfa_result  # NEW: Pass all DFAs with transitions
            )

            # Log Stage 3 success
            self.logger.log_stage3(
                ltl_spec,
                dfa_result,
                asl_code,
                "Success",
                model=self.config.openai_model,
                llm_prompt=prompt_dict,
                llm_response=response_text
            )

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
            self.logger.log_stage3(ltl_spec, dfa_result, None, "Failed", str(e))
            print(f"✗ Stage 3 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None
