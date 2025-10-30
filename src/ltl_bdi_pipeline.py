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
from stage2_planning.agentspeak_generator import AgentSpeakGenerator
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

    def __init__(self):
        self.config = get_config()
        self.logger = PipelineLogger()

        # Domain configuration (for AgentSpeak generation)
        self.domain_actions = ['pickup', 'putdown', 'stack', 'unstack']
        self.domain_predicates = ['on(X, Y)', 'clear(X)', 'holding(X)', 'handempty']

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

        # Stage 2: LTLf -> DFA (optional visualization/verification)
        # Note: DFA conversion is available but not yet integrated into execution flow
        # Can be called manually via: from ltlf_dfa_conversion.ltlf_to_dfa import LTLfToDFA

        # Stage 3: LTLf -> AgentSpeak
        asl_code = self._stage3_llm_agentspeak_generation(ltl_spec)
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
            model=self.config.openai_model
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
            print(f"  Initial State: {ltl_spec.initial_state}")

            return ltl_spec

        except Exception as e:
            self.logger.log_stage1(nl_instruction, None, "Failed", str(e))
            print(f"✗ Stage 1 Failed: {e}")
            return None

    def _stage3_llm_agentspeak_generation(self, ltl_spec):
        """Stage 3: LTLf -> LLM AgentSpeak Generation"""
        print("\n[STAGE 3] LLM AgentSpeak Generation")
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

            asl_code, prompt_dict, response_text = generator.generate(
                ltl_dict,
                'blocksworld',
                self.domain_actions,
                self.domain_predicates
            )

            # Log Stage 3 success (AgentSpeak generation)
            # Note: Still using log_stage2 for backward compatibility with logger
            self.logger.log_stage2(
                ltl_spec,
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
            self.logger.log_stage2(ltl_spec, None, "Failed", str(e))
            print(f"✗ Stage 3 Failed: {e}")
            return None
