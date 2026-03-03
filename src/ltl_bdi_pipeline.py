"""
LTL-BDI pipeline: NL -> LTLf -> DFA -> HTN-specialised AgentSpeak.
"""

from pathlib import Path
from typing import Dict, Any

from utils.config import get_config
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
from stage2_dfa_generation.dfa_builder import DFABuilder
from stage3_code_generation.htn_planner_generator import HTNPlannerGenerator
from utils.pipeline_logger import PipelineLogger


class LTL_BDI_Pipeline:
    """
    LTL-BDI pipeline implementing Stages 1-3 (dfa_agentspeak mode)

    Stage 1: Natural Language -> LTLf Specification
    Stage 2: LTLf -> DFA Conversion (ltlf2dfa)
    Stage 3: DFA -> AgentSpeak Code Generation (HTN planning + specialisation)
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
                "The pipeline currently supports the HTN-specialised AgentSpeak path only."
            )

        # Start logger (creates timestamped directory in logs/)
        self.logger.start_pipeline(
            nl_instruction,
            mode=mode,
            domain_file=self.domain_file,
            output_dir="logs",
        )

        # Use logger's directory for all output files
        self.output_dir = self.logger.current_log_dir
        if self.logger.current_record is not None and self.output_dir is not None:
            self.logger.current_record.output_dir = str(self.output_dir)
            self.logger._save_current_state()

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

        # Stage 3: DFA -> AgentSpeak Code (HTN)
        asl_code, stage3_stats = self._stage3_htn_generation(ltl_spec, dfa_result)
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
            base_url=self.config.openai_base_url,
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

    def _stage3_htn_generation(self, ltl_spec, dfa_result):
        """Stage 3: DFA -> HTN-specialised AgentSpeak generation."""
        print("\n[STAGE 3] HTN Method Synthesis + Planning + Specialisation")
        print("-"*80)

        try:
            # Create grounding map from ltl_spec
            grounding_map = ltl_spec.grounding_map

            # Initialize HTN planner generator
            generator = HTNPlannerGenerator(
                self.domain,
                grounding_map,
                api_key=self.config.openai_api_key,
                model=self.config.openai_model,
                base_url=self.config.openai_base_url,
                timeout=float(self.config.openai_timeout),
            )

            # Convert ltl_spec to dict format for generator
            ltl_dict = {
                "objects": ltl_spec.objects,
                "formulas_string": [f.to_string() for f in ltl_spec.formulas],
                "grounding_map": grounding_map
            }

            # Generate AgentSpeak code
            import time
            start_time = time.time()
            asl_code, artifacts = generator.generate(ltl_dict, dfa_result)
            elapsed = time.time() - start_time

            summary = artifacts["summary"]
            stage3_stats = {
                "method": "htn",
                "code_size_chars": len(asl_code),
                "used_llm": artifacts["llm"]["used"],
                "compound_tasks": summary["compound_tasks"],
                "methods": summary["methods"],
                "transition_count": summary["transition_count"],
                "generation_time_seconds": elapsed
            }

            # Log Stage 3 success with HTN metadata
            llm_attempted = artifacts["llm"]["prompt"] is not None
            self.logger.log_stage3(
                ltl_spec,
                dfa_result,
                asl_code,
                "Success",
                method="htn",
                model=artifacts["llm"]["model"] if llm_attempted else None,
                llm_prompt=artifacts["llm"]["prompt"],
                llm_response=artifacts["llm"]["response"],
                metadata=summary,
                artifacts={
                    "method_library": artifacts["method_library"],
                    "transitions": artifacts["transitions"],
                },
            )

            print(f"\n✓ AgentSpeak Code Generated ({len(asl_code)} characters in {elapsed:.2f}s)")
            print("  Method: HTN method synthesis + specialisation")
            print(f"  Attempted LLM synthesis in Stage 3A: {llm_attempted}")
            print(f"  Accepted LLM output in Stage 3A: {artifacts['llm']['used']}")
            print(f"  Compound tasks: {summary['compound_tasks']}")
            print(f"  Methods: {summary['methods']}")
            print(f"  Transition plans: {summary['transition_count']}")
            print("\n  First 10 lines of generated code:")
            for i, line in enumerate(asl_code.split('\n')[:10], 1):
                if line.strip():
                    print(f"    {i:2d}. {line}")

            # Save complete AgentSpeak code to output
            output_file = self.output_dir / "agentspeak_generated.asl"
            output_file.write_text(asl_code)
            print(f"\n  ✓ Complete AgentSpeak code saved to: {output_file}")

            import json
            method_library_file = self.output_dir / "htn_method_library.json"
            method_library_file.write_text(json.dumps(artifacts["method_library"], indent=2))
            print(f"  ✓ HTN method library saved to: {method_library_file}")

            transitions_file = self.output_dir / "htn_transitions.json"
            transitions_file.write_text(json.dumps(artifacts["transitions"], indent=2))
            print(f"  ✓ HTN decomposition traces saved to: {transitions_file}")

            return asl_code, stage3_stats

        except Exception as e:
            # Log failure
            self.logger.log_stage3(
                ltl_spec,
                dfa_result,
                None,
                "Failed",
                error=str(e),
                method="htn"
            )
            print(f"✗ Stage 3 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
