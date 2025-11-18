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

from utils.config import get_config
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
from stage2_dfa_generation.dfa_builder import DFABuilder
from stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from utils.pipeline_logger import PipelineLogger


class LTL_BDI_Pipeline:
    """
    LTL-BDI pipeline implementing Stages 1-3 (dfa_agentspeak mode)

    Stage 1: Natural Language -> LTLf Specification
    Stage 2: LTLf -> DFA Conversion (ltlf2dfa)
    Stage 3: DFA -> AgentSpeak Code Generation (Backward Planning)

    Legacy Note: FOND Planning (Branch B) has been deprecated and moved to src/legacy/fond/
    The pipeline now focuses exclusively on DFA-AgentSpeak generation using backward planning.
    """

    def __init__(self, domain_file: str = None):
        """
        Initialize pipeline

        Args:
            domain_file: Path to PDDL domain file. If None, uses default blocksworld domain.
        """
        self.config = get_config()

        # Use absolute path for logs directory (project root/logs)
        # This ensures logs go to same location regardless of where tests are run from
        from pathlib import Path
        project_root = Path(__file__).parent.parent  # src/ -> project root
        self.logger = PipelineLogger(logs_dir=str(project_root / "logs"))

        # Domain file path
        if domain_file is None:
            # Default to blocksworld domain
            from pathlib import Path
            domain_file = str(Path(__file__).parent / "domains" / "blocksworld" / "domain.pddl")

        self.domain_file = domain_file

        # Parse domain to extract actions and predicates
        from utils.pddl_parser import PDDLParser
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
        print(f"\n\"{nl_instruction}\"")
        print(f"Mode: {mode}")
        print(f"Output directory: {self.output_dir}")
        print("\n" + "-"*80)

        # Stage 1: NL -> LTLf
        ltl_spec = self._stage1_parse_nl(nl_instruction)
        if not ltl_spec:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 1", "error": "LTLf parsing failed"}

        # Stage 2: LTLf -> DFA Generation
        dfa_result = self._stage2_dfa_generation(ltl_spec)
        if not dfa_result:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 2", "error": "DFA generation failed"}

        # Stage 3: DFA -> AgentSpeak Code (Backward Planning)
        asl_code, stage3_stats = self._stage3_backward_planning_generation(ltl_spec, dfa_result)
        if not asl_code:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
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

        generator = NLToLTLfGenerator(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model,
            domain_file=self.domain_file  # Pass domain file for dynamic prompt
        )

        try:
            ltl_spec, prompt_dict, response_text = generator.generate(nl_instruction)
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

    def _stage2_dfa_generation(self, ltl_spec):
        """Stage 2: LTLf -> DFA Generation"""
        print("\n[STAGE 2] DFA Generation")
        print("-"*80)

        builder = DFABuilder()

        try:
            dfa_result = builder.build(ltl_spec)

            # Log Stage 2 success
            self.logger.log_stage2_dfas(
                ltl_spec,
                dfa_result,
                "Success"
            )

            print(f"✓ DFA Generation Complete")
            print(f"  Formula: {dfa_result['formula']}")
            print(f"\n  Original DFA (before simplification):")
            print(f"    States: {dfa_result['original_num_states']}")
            print(f"    Transitions: {dfa_result['original_num_transitions']}")
            print(f"    Saved to: {self.output_dir / 'dfa_original.dot'}")
            print(f"\n  Simplified DFA (after simplification):")
            print(f"    States: {dfa_result['num_states']}")
            print(f"    Transitions: {dfa_result['num_transitions']}")
            print(f"    Saved to: {self.output_dir / 'dfa_simplified.dot'}")

            # Save complete DFA result to JSON
            output_file = self.output_dir / "dfa.json"
            import json
            # Remove the actual DOT strings from JSON to keep it readable
            # (DOT files are saved separately)
            json_data = {k: v for k, v in dfa_result.items()
                        if k not in ['dfa_dot', 'original_dfa_dot']}
            output_file.write_text(json.dumps(json_data, indent=2))
            print(f"\n  Metadata saved to: {output_file}")

            return dfa_result

        except Exception as e:
            self.logger.log_stage2_dfas(ltl_spec, None, "Failed", str(e))
            print(f"✗ Stage 2 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _stage3_backward_planning_generation(self, ltl_spec, dfa_result):
        """Stage 3: DFA -> Backward Planning AgentSpeak Generation (Non-LLM)"""
        print("\n[STAGE 3] Backward Planning AgentSpeak Generation")
        print("-"*80)

        try:
            # Create grounding map from ltl_spec
            grounding_map = ltl_spec.grounding_map

            # Initialize backward planner generator
            generator = BackwardPlannerGenerator(self.domain, grounding_map)

            # Convert ltl_spec to dict format for generator
            ltl_dict = {
                "objects": ltl_spec.objects,
                "formulas_string": [f.to_string() for f in ltl_spec.formulas],
                "grounding_map": grounding_map
            }

            # Generate AgentSpeak code
            import time
            start_time = time.time()
            asl_code, truncated = generator.generate(ltl_dict, dfa_result)
            elapsed = time.time() - start_time

            # Extract statistics from generator output (printed during generation)
            # Note: BackwardPlannerGenerator prints statistics but doesn't return them yet
            # We'll capture what we can from the generation process
            stage3_stats = {
                "method": "backward_planning",
                "code_size_chars": len(asl_code),
                "truncated": truncated,
                "generation_time_seconds": elapsed
            }

            # Log Stage 3 success with backward planning statistics
            self.logger.log_stage3(
                ltl_spec,
                dfa_result,
                asl_code,
                "Success",
                method="backward_planning",
                states_explored=0,  # TODO: Extract from generator
                transitions_generated=0,  # TODO: Extract from generator
                goal_plans_count=0,  # TODO: Extract from generator
                action_plans_count=0,  # TODO: Extract from generator
                cache_hits=0,  # TODO: Extract from generator
                cache_misses=0,  # TODO: Extract from generator
                ground_actions_cached=0,  # TODO: Extract from generator
                redundancy_eliminated_pct=0.0  # TODO: Extract from generator
            )

            print(f"\n✓ AgentSpeak Code Generated ({len(asl_code)} characters in {elapsed:.2f}s)")
            print(f"  Method: Backward Planning (non-LLM)")
            print(f"  Truncated: {truncated}")
            print("\n  First 10 lines of generated code:")
            for i, line in enumerate(asl_code.split('\n')[:10], 1):
                if line.strip():
                    print(f"    {i:2d}. {line}")

            # Save complete AgentSpeak code to output
            output_file = self.output_dir / "agentspeak_generated.asl"
            output_file.write_text(asl_code)
            print(f"\n  ✓ Complete AgentSpeak code saved to: {output_file}")

            return asl_code, stage3_stats

        except Exception as e:
            # Log failure
            self.logger.log_stage3(
                ltl_spec,
                dfa_result,
                None,
                "Failed",
                error=str(e),
                method="backward_planning"
            )
            print(f"✗ Stage 3 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
