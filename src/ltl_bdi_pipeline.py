"""
LTL-BDI pipeline: NL -> LTLf -> DFA -> HTN synthesis -> PANDA -> AgentSpeak.
"""

from pathlib import Path
from typing import Dict, Any

from utils.config import get_config
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
from stage2_dfa_generation.dfa_builder import DFABuilder
from stage3_method_synthesis.htn_method_synthesis import HTNMethodSynthesizer
from stage4_panda_planning.panda_planner import PANDAPlanner
from stage5_agentspeak_rendering.agentspeak_renderer import AgentSpeakRenderer
from utils.pipeline_logger import PipelineLogger


class LTL_BDI_Pipeline:
    """
    LTL-BDI pipeline implementing Stages 1-5 (dfa_agentspeak mode)

    Stage 1: Natural Language -> LTLf Specification
    Stage 2: LTLf -> DFA Conversion (ltlf2dfa)
    Stage 3: DFA -> HTN Method Synthesis
    Stage 4: HTN Method Library -> PANDA Planning
    Stage 5: PANDA Plans -> AgentSpeak Rendering
    """

    def __init__(self, domain_file: str = None):
        """
        Initialize pipeline

        Args:
            domain_file: Path to HDDL domain file. If None, uses default blocksworld domain.
        """
        self.config = get_config()

        # Use absolute path for logs directory (project root/logs)
        # This ensures logs go to same location regardless of where tests are run from
        project_root = Path(__file__).parent.parent  # src/ -> project root
        self.logger = PipelineLogger(logs_dir=str(project_root / "logs"))

        # Domain file path
        if domain_file is None:
            # Default to blocksworld domain
            domain_file = str(Path(__file__).parent / "domains" / "blocksworld" / "domain.hddl")

        self.domain_file = domain_file

        # Parse domain to extract actions, predicates, tasks, and methods
        from utils.hddl_parser import HDDLParser
        self.domain = HDDLParser.parse_domain(domain_file)
        self.domain_actions = self.domain.get_action_names()
        self.domain_predicates = self.domain.get_predicate_signatures()

        # Output directory (set during execution - will use logger's directory)
        self.output_dir = None

    def execute(self, nl_instruction: str, mode: str = "dfa_agentspeak") -> Dict[str, Any]:
        """
        Execute LTL-BDI pipeline (Stages 1-5: NL -> LTLf -> DFA -> AgentSpeak)

        Args:
            nl_instruction: Natural language instruction
            mode: Execution mode (only "dfa_agentspeak" is supported)

        Returns:
            Results from Stage 1-5 (no execution/evaluation yet)
        """
        if mode != "dfa_agentspeak":
            raise ValueError(
                f"Unknown mode '{mode}'. Only 'dfa_agentspeak' is supported. "
                "The pipeline currently supports the PANDA-backed AgentSpeak path only."
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

        # Stage 3: DFA -> HTN method synthesis
        method_library, stage3_data = self._stage3_method_synthesis(ltl_spec, dfa_result)
        if not method_library:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 3", "error": "HTN method synthesis failed"}

        # Stage 4: HTN method library -> PANDA planning
        plan_records, stage4_data = self._stage4_panda_planning(ltl_spec, method_library)
        if plan_records is None:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 4", "error": "PANDA planning failed"}

        # Stage 5: PANDA plans -> AgentSpeak rendering
        asl_code, _ = self._stage5_agentspeak_rendering(ltl_spec, plan_records)
        if not asl_code:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 5", "error": "AgentSpeak rendering failed"}

        print("\n" + "="*80)
        print("STAGES 1-5 COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNote: Stage 6 (Jason validation / execution assets) is not part of the default run path")

        # End logger and save results
        log_filepath = self.logger.end_pipeline(success=True)
        print(f"\nExecution log saved to: {log_filepath}")

        return {
            "success": True,
            "ltl_spec": ltl_spec,
            "method_library": stage3_data["method_library"],
            "plans": stage4_data["transitions"],
            "agentspeak_code": asl_code,
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
            print("  (Stage 1 only captures goal semantics; Stage 4 instantiates a concrete HDDL problem)")

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

    def _stage3_method_synthesis(self, ltl_spec, dfa_result):
        """Stage 3: DFA -> HTN method synthesis."""
        print("\n[STAGE 3] HTN Method Synthesis")
        print("-"*80)

        try:
            grounding_map = ltl_spec.grounding_map
            synthesizer = HTNMethodSynthesizer(
                api_key=self.config.openai_api_key,
                model=self.config.openai_model,
                base_url=self.config.openai_base_url,
                timeout=float(self.config.openai_timeout),
            )

            method_library, synthesis_meta = synthesizer.synthesize(
                domain=self.domain,
                grounding_map=grounding_map,
                dfa_result=dfa_result,
            )
            summary = {
                "used_llm": synthesis_meta["used_llm"],
                "llm_attempted": synthesis_meta["llm_prompt"] is not None,
                "target_literals": synthesis_meta["target_literals"],
                "compound_tasks": synthesis_meta["compound_tasks"],
                "primitive_tasks": synthesis_meta["primitive_tasks"],
                "methods": synthesis_meta["methods"],
            }

            self.logger.log_stage3_method_synthesis(
                method_library.to_dict(),
                "Success",
                model=synthesis_meta["model"] if synthesis_meta["llm_prompt"] is not None else None,
                llm_prompt=synthesis_meta["llm_prompt"],
                llm_response=synthesis_meta["llm_response"],
                metadata=summary,
            )

            print("✓ HTN method synthesis complete")
            print(f"  Attempted LLM synthesis: {summary['llm_attempted']}")
            print(f"  Accepted LLM output: {summary['used_llm']}")
            print(f"  Compound tasks: {summary['compound_tasks']}")
            print(f"  Primitive tasks: {summary['primitive_tasks']}")
            print(f"  Methods: {summary['methods']}")
            method_library_file = self.output_dir / "htn_method_library.json"
            print(f"  ✓ HTN method library saved to: {method_library_file}")

            return method_library, {
                "method_library": method_library.to_dict(),
                "summary": summary,
                "llm": {
                    "used": synthesis_meta["used_llm"],
                    "model": synthesis_meta["model"],
                    "prompt": synthesis_meta["llm_prompt"],
                    "response": synthesis_meta["llm_response"],
                },
            }

        except Exception as e:
            self.logger.log_stage3_method_synthesis(
                None,
                "Failed",
                error=str(e),
                model=getattr(e, "model", None),
                llm_prompt=getattr(e, "llm_prompt", None),
                llm_response=getattr(e, "llm_response", None),
                metadata=getattr(e, "metadata", None),
            )
            print(f"✗ Stage 3 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _stage4_panda_planning(self, ltl_spec, method_library):
        """Stage 4: HTN method library -> PANDA planning."""
        print("\n[STAGE 4] PANDA Planning")
        print("-"*80)

        planner = PANDAPlanner(workspace=str(self.output_dir))

        try:
            plan_records = []
            transition_artifacts = []
            for index, literal in enumerate(method_library.target_literals, start=1):
                transition_name = f"transition_{index}"
                task_name = method_library.task_name_for_literal(literal)
                if not task_name:
                    raise ValueError(
                        "Stage 3 output is missing a target_task_binding for "
                        f"'{literal.to_signature()}'."
                    )
                plan = planner.plan(
                    domain=self.domain,
                    method_library=method_library,
                    objects=ltl_spec.objects,
                    target_literal=literal,
                    task_name=task_name,
                    transition_name=transition_name,
                )
                label = literal.to_signature()
                plan_records.append(
                    {
                        "transition_name": transition_name,
                        "label": label,
                        "target_literal": literal,
                        "plan": plan,
                    }
                )
                transition_artifacts.append(
                    {
                        "transition_name": transition_name,
                        "label": label,
                        "target_literal": literal.to_dict(),
                        "plan": plan.to_dict(),
                    }
                )

            summary = {
                "backend": "pandaPI",
                "transition_count": len(transition_artifacts),
                "planned_tasks": [record["plan"].task_name for record in plan_records],
            }

            self.logger.log_stage4_panda_planning(
                {
                    "transitions": transition_artifacts,
                },
                "Success",
                metadata=summary,
            )

            print("✓ PANDA planning complete")
            print(f"  Backend: {summary['backend']}")
            print(f"  Transition plans: {summary['transition_count']}")
            transitions_file = self.output_dir / "panda_transitions.json"
            print(f"  ✓ PANDA planning artifacts saved to: {transitions_file}")

            return plan_records, {
                "summary": summary,
                "transitions": transition_artifacts,
            }

        except Exception as e:
            self.logger.log_stage4_panda_planning(
                None,
                "Failed",
                error=str(e),
                metadata=getattr(e, "metadata", None),
            )
            print(f"✗ Stage 4 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _stage5_agentspeak_rendering(self, ltl_spec, plan_records):
        """Stage 5: PANDA plans -> AgentSpeak rendering."""
        print("\n[STAGE 5] AgentSpeak Rendering")
        print("-"*80)

        try:
            renderer = AgentSpeakRenderer()
            asl_code = renderer.generate(
                domain=self.domain,
                objects=ltl_spec.objects,
                plan_records=plan_records,
            )
            metadata = {
                "transition_count": len(plan_records),
                "code_size_chars": len(asl_code),
            }
            self.logger.log_stage5_agentspeak_rendering(
                asl_code,
                "Success",
                metadata=metadata,
            )

            print(f"✓ AgentSpeak rendering complete ({len(asl_code)} characters)")
            print(f"  Transition plans rendered: {len(plan_records)}")
            print("\n  First 10 lines of generated code:")
            for index, line in enumerate(asl_code.split("\n")[:10], start=1):
                if line.strip():
                    print(f"    {index:2d}. {line}")

            output_file = self.output_dir / "agentspeak_generated.asl"
            output_file.write_text(asl_code)
            print(f"\n  ✓ Complete AgentSpeak code saved to: {output_file}")

            return asl_code, metadata

        except Exception as e:
            self.logger.log_stage5_agentspeak_rendering(
                None,
                "Failed",
                error=str(e),
                metadata={"stage": "render"},
            )
            print(f"✗ Stage 5 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
