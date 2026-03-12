"""
LTL-BDI pipeline: NL -> LTLf -> DFA -> HTN synthesis -> PANDA -> AgentSpeak.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Set, Tuple

from utils.config import get_config
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
from stage2_dfa_generation.dfa_builder import DFABuilder
from stage3_method_synthesis.htn_method_synthesis import HTNMethodSynthesizer
from stage3_method_synthesis.htn_schema import HTNLiteral
from stage4_panda_planning.panda_planner import PANDAPlanner
from stage5_agentspeak_rendering.agentspeak_renderer import AgentSpeakRenderer
from stage6_jason_validation.jason_runner import JasonRunner, JasonValidationError
from utils.hddl_condition_parser import HDDLConditionParser
from utils.ipc_plan_verifier import IPCPlanVerifier
from utils.pipeline_logger import PipelineLogger


class TypeResolutionError(RuntimeError):
    """Raised when object/variable type inference is ambiguous or inconsistent."""


class LTL_BDI_Pipeline:
    """
    LTL-BDI pipeline implementing compile, execute, and official verification.

    Stage 1: Natural Language -> LTLf Specification
    Stage 2: LTLf -> DFA Conversion (ltlf2dfa)
    Stage 3: DFA -> HTN Method Synthesis
    Stage 4: HTN Method Library -> PANDA Planning
    Stage 5: HTN Methods + DFA Wrappers -> AgentSpeak Rendering
    Stage 6: AgentSpeak -> Jason Runtime Validation
    Stage 7: Official IPC HTN Plan Verification
    """

    def __init__(self, domain_file: str, problem_file: str | None = None):
        """
        Initialize pipeline

        Args:
            domain_file: Path to HDDL domain file.
            problem_file: Optional path to HDDL problem file used for runtime initialisation.
        """
        self.config = get_config()

        # Use absolute path for logs directory (project root/logs)
        # This ensures logs go to same location regardless of where tests are run from
        project_root = Path(__file__).parent.parent  # src/ -> project root
        self.logger = PipelineLogger(logs_dir=str(project_root / "logs"))

        if not domain_file:
            raise ValueError(
                "domain_file is required. Pass an explicit HDDL domain path to LTL_BDI_Pipeline.",
            )

        self.domain_file = domain_file
        self.problem_file = problem_file

        # Parse domain to extract actions, predicates, tasks, and methods
        from utils.hddl_parser import HDDLParser
        self.domain = HDDLParser.parse_domain(domain_file)
        self.problem = HDDLParser.parse_problem(problem_file) if problem_file else None
        if self.problem is not None and self.problem.domain_name.lower() != self.domain.name.lower():
            raise ValueError(
                "problem_file domain does not match domain_file: "
                f"{self.problem.domain_name} != {self.domain.name}",
            )
        self.domain_actions = self.domain.get_action_names()
        self.domain_predicates = self.domain.get_predicate_signatures()
        self.type_parent_map = self._build_type_parent_map()
        self.domain_type_names = set(self.type_parent_map.keys())
        self.predicate_type_map = self._predicate_type_map()
        self.action_type_map = self._action_type_map()
        self.task_type_map = self._task_type_map()

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
            problem_file=self.problem_file,
            domain_name=self.domain.name,
            problem_name=self.problem.name if self.problem is not None else None,
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
        plan_records, stage4_data = self._stage4_panda_planning(
            ltl_spec,
            method_library,
            stage3_data["transition_specs"],
        )
        if plan_records is None:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 4", "error": "PANDA planning failed"}

        # Stage 5: HTN method library + validated DFA wrappers -> AgentSpeak rendering
        asl_code, _ = self._stage5_agentspeak_rendering(ltl_spec, method_library, plan_records)
        if not asl_code:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 5", "error": "AgentSpeak rendering failed"}

        # Stage 6: AgentSpeak -> Jason runtime validation
        stage6_data = self._stage6_jason_validation(
            ltl_spec,
            method_library,
            plan_records,
            asl_code,
        )
        if stage6_data is None:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 6", "error": "Jason runtime validation failed"}

        print("\n" + "="*80)
        # Stage 7: Official IPC verifier on generated hierarchical plan
        stage7_data = self._stage7_official_verification(method_library, stage6_data)
        if stage7_data is None:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 7", "error": "Official IPC verification failed"}

        stage7_summary = stage7_data.get("summary") or {}
        if stage7_summary.get("status") == "skipped":
            print("STAGES 1-6 COMPLETED SUCCESSFULLY (STAGE 7 SKIPPED)")
        else:
            print("STAGES 1-7 COMPLETED SUCCESSFULLY")
        print("="*80)

        # End logger and save results
        log_filepath = self.logger.end_pipeline(success=True)
        print(f"\nExecution log saved to: {log_filepath}")

        return {
            "success": True,
            "ltl_spec": ltl_spec,
            "method_library": stage3_data["method_library"],
            "plans": stage4_data["transitions"],
            "agentspeak_code": asl_code,
            "stage6": stage6_data,
            "stage7": stage7_data,
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
            ordered_literal_signatures = self._ordered_literal_signatures_from_spec(ltl_spec)
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
                query_text=getattr(ltl_spec, "source_instruction", ""),
                negation_hints=getattr(ltl_spec, "negation_hints", {}),
                ordered_literal_signatures=ordered_literal_signatures,
            )
            self._validate_method_library_typing(method_library)
            summary = {
                "used_llm": synthesis_meta["used_llm"],
                "llm_attempted": synthesis_meta["llm_prompt"] is not None,
                "llm_finish_reason": synthesis_meta.get("llm_finish_reason"),
                "llm_attempts": synthesis_meta.get("llm_attempts"),
                "llm_response_time_seconds": synthesis_meta.get("llm_response_time_seconds"),
                "llm_attempt_durations_seconds": synthesis_meta.get(
                    "llm_attempt_durations_seconds",
                ),
                "target_literals": synthesis_meta["target_literals"],
                "negation_resolution": synthesis_meta.get("negation_resolution", {}),
                "domain_projection_used": synthesis_meta.get("domain_projection_used", False),
                "compound_tasks": synthesis_meta["compound_tasks"],
                "primitive_tasks": synthesis_meta["primitive_tasks"],
                "methods": synthesis_meta["methods"],
            }
            transition_specs = synthesizer.extract_progressing_transitions(
                grounding_map,
                dfa_result,
                ordered_literal_signatures=ordered_literal_signatures,
            )
            summary["dfa_progress_transitions"] = len(transition_specs)

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
                "transition_specs": transition_specs,
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

    @classmethod
    def _ordered_literal_signatures_from_spec(cls, ltl_spec) -> Tuple[str, ...]:
        ordered: List[str] = []
        seen: Set[str] = set()
        for formula in getattr(ltl_spec, "formulas", []) or []:
            formula_literals = cls._ordered_literal_signatures_from_formula(formula)
            if not formula_literals:
                return ()
            for signature in formula_literals:
                if signature in seen:
                    continue
                seen.add(signature)
                ordered.append(signature)
        return tuple(ordered)

    @classmethod
    def _ordered_literal_signatures_from_formula(cls, formula) -> Tuple[str, ...]:
        predicate = getattr(formula, "predicate", None)
        operator = getattr(formula, "operator", None)
        logical_op = getattr(formula, "logical_op", None)
        sub_formulas = tuple(getattr(formula, "sub_formulas", ()) or ())

        if predicate is not None and operator is None and logical_op is None:
            if not isinstance(predicate, dict):
                return ()
            pred_name = next(iter(predicate.keys()), None)
            if pred_name is None:
                return ()
            args = predicate[pred_name]
            return (cls._literal_signature(pred_name, args, True),)

        if operator is not None:
            operator_name = getattr(operator, "value", None)
            if operator_name in {"F", "G"} and len(sub_formulas) == 1:
                return cls._ordered_literal_signatures_from_formula(sub_formulas[0])
            return ()

        logical_name = getattr(logical_op, "value", None)
        if logical_name == "not":
            if len(sub_formulas) != 1:
                return ()
            child = sub_formulas[0]
            child_predicate = getattr(child, "predicate", None)
            if (
                getattr(child, "operator", None) is not None
                or getattr(child, "logical_op", None) is not None
                or not isinstance(child_predicate, dict)
            ):
                return ()
            pred_name = next(iter(child_predicate.keys()), None)
            if pred_name is None:
                return ()
            args = child_predicate[pred_name]
            return (cls._literal_signature(pred_name, args, False),)

        if logical_name == "and":
            ordered: List[str] = []
            for child in sub_formulas:
                child_literals = cls._ordered_literal_signatures_from_formula(child)
                if not child_literals:
                    return ()
                ordered.extend(child_literals)
            return tuple(ordered)

        return ()

    @staticmethod
    def _literal_signature(predicate: str, args: Sequence[str], is_positive: bool) -> str:
        atom = predicate if not args else f"{predicate}({', '.join(args)})"
        return atom if is_positive else f"!{atom}"

    def _stage4_panda_planning(self, ltl_spec, method_library, transition_specs):
        """Stage 4: HTN method library -> PANDA planning."""
        print("\n[STAGE 4] PANDA Planning")
        print("-"*80)

        planner = PANDAPlanner(workspace=str(self.output_dir))

        try:
            plan_records = []
            transition_artifacts = []
            for transition_spec in transition_specs:
                literal = transition_spec["literal"]
                transition_name = transition_spec["transition_name"]
                task_name = method_library.task_name_for_literal(literal)
                if not task_name:
                    raise ValueError(
                        "Stage 3 output is missing a target_task_binding for "
                        f"'{literal.to_signature()}'."
                    )
                witness_objects, witness_object_types = self._seed_validation_scope(
                    task_name,
                    method_library,
                    tuple(literal.args),
                    ltl_spec.objects,
                )
                witness_initial_facts = self._task_witness_initial_facts(
                    planner,
                    task_name,
                    method_library,
                    tuple(literal.args),
                    ltl_spec.objects,
                    object_pool=witness_objects,
                    object_types=witness_object_types,
                )
                plan = planner.plan(
                    domain=self.domain,
                    method_library=method_library,
                    objects=tuple(witness_objects),
                    target_literal=literal,
                    task_name=task_name,
                    transition_name=transition_name,
                    typed_objects=self._typed_object_entries(
                        witness_objects,
                        witness_object_types,
                    ),
                    allow_empty_plan=True,
                    initial_facts=witness_initial_facts,
                )
                label = literal.to_signature()
                plan_records.append(
                    {
                        "transition_name": transition_name,
                        "source_state": transition_spec["source_state"],
                        "target_state": transition_spec["target_state"],
                        "raw_source_state": transition_spec["raw_source_state"],
                        "raw_target_state": transition_spec["raw_target_state"],
                        "initial_state": transition_spec["initial_state"],
                        "accepting_states": list(transition_spec.get("accepting_states", [])),
                        "label": label,
                        "target_literal": literal,
                        "objects": list(witness_objects),
                        "initial_facts": witness_initial_facts,
                        "plan": plan,
                    }
                )
                transition_artifacts.append(
                    {
                        "transition_name": transition_name,
                        "source_state": transition_spec["source_state"],
                        "target_state": transition_spec["target_state"],
                        "raw_source_state": transition_spec["raw_source_state"],
                        "raw_target_state": transition_spec["raw_target_state"],
                        "initial_state": transition_spec["initial_state"],
                        "accepting_states": list(transition_spec.get("accepting_states", [])),
                        "label": label,
                        "target_literal": literal.to_dict(),
                        "objects": list(witness_objects),
                        "initial_facts": list(witness_initial_facts),
                        "plan": plan.to_dict(),
                    }
                )

            method_validation_artifacts = []
            relevant_task_names = self._query_relevant_task_names(method_library, plan_records)
            for task_name in relevant_task_names:
                representative_task_args = self._representative_task_args(
                    task_name,
                    method_library,
                    ltl_spec.objects,
                    plan_records,
                )
                for method in method_library.methods_for_task(task_name):
                    validation_task_args = self._method_validation_task_args(
                        method,
                        representative_task_args,
                        method_library,
                    )
                    validation_name = f"method_{method.method_name}"
                    validation_objects, validation_object_types = self._seed_validation_scope(
                        task_name,
                        method_library,
                        validation_task_args,
                        ltl_spec.objects,
                    )
                    validation_initial_facts = self._method_validation_initial_facts(
                        planner,
                        method,
                        method_library,
                        validation_task_args,
                        ltl_spec.objects,
                        object_pool=validation_objects,
                        object_types=validation_object_types,
                    )
                    try:
                        validation_plan = planner.plan(
                            domain=self.domain,
                            method_library=method_library,
                            objects=tuple(validation_objects),
                            target_literal=None,
                            task_name=task_name,
                            transition_name=validation_name,
                            typed_objects=self._typed_object_entries(
                                validation_objects,
                                validation_object_types,
                            ),
                            task_args=validation_task_args,
                            root_method=method,
                            allow_empty_plan=not method.subtasks,
                            initial_facts=validation_initial_facts,
                        )
                        method_validation_artifacts.append(
                            {
                                "validation_name": validation_name,
                                "task_name": task_name,
                                "task_args": list(validation_task_args),
                                "method_name": method.method_name,
                                "allow_empty_plan": not method.subtasks,
                                "objects": list(validation_objects),
                                "initial_facts": list(validation_initial_facts),
                                "status": "success",
                                "plan": validation_plan.to_dict(),
                            }
                        )
                    except Exception as exc:
                        method_validation_artifacts.append(
                            {
                                "validation_name": validation_name,
                                "task_name": task_name,
                                "task_args": list(validation_task_args),
                                "method_name": method.method_name,
                                "allow_empty_plan": not method.subtasks,
                                "objects": list(validation_objects),
                                "initial_facts": list(validation_initial_facts),
                                "status": "failed",
                                "error": str(exc),
                                "metadata": getattr(exc, "metadata", None),
                            }
                        )

            summary = {
                "backend": "pandaPI",
                "transition_count": len(transition_artifacts),
                "validated_method_count": len(method_validation_artifacts),
                "successful_method_validations": sum(
                    1
                    for item in method_validation_artifacts
                    if item["status"] == "success"
                ),
                "failed_method_validations": sum(
                    1
                    for item in method_validation_artifacts
                    if item["status"] == "failed"
                ),
                "dfa_states": sorted(
                    {
                        transition["source_state"]
                        for transition in transition_artifacts
                    }
                    |
                    {
                        transition["target_state"]
                        for transition in transition_artifacts
                    }
                ),
                "planned_tasks": [record["plan"].task_name for record in plan_records],
            }

            self.logger.log_stage4_panda_planning(
                {
                    "transitions": transition_artifacts,
                    "method_validations": method_validation_artifacts,
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
                "method_validations": method_validation_artifacts,
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

    @staticmethod
    def _query_relevant_task_names(method_library, plan_records):
        task_names = {
            binding.task_name
            for binding in method_library.target_task_bindings
        }
        task_names.update({
            record["plan"].task_name
            for record in plan_records
        })
        queue = list(task_names)

        while queue:
            task_name = queue.pop(0)
            for method in method_library.methods_for_task(task_name):
                for step in method.subtasks:
                    if step.kind != "compound" or step.task_name in task_names:
                        continue
                    task_names.add(step.task_name)
                    queue.append(step.task_name)

        return sorted(task_names)

    def _representative_task_args(self, task_name, method_library, objects, plan_records):
        for record in plan_records:
            plan = record["plan"]
            if plan.task_name == task_name and plan.task_args:
                return tuple(plan.task_args)

        for binding in method_library.target_task_bindings:
            if binding.task_name != task_name:
                continue
            for literal in method_library.target_literals:
                if literal.to_signature() == binding.target_literal:
                    return tuple(literal.args)

        task_schema = method_library.task_for_name(task_name)
        arity = len(task_schema.parameters) if task_schema else 0
        if arity == 0:
            return ()

        signature = self._task_type_signature(task_name, method_library)
        if signature and len(signature) == arity:
            return tuple(
                f"witness_{type_name}_{index + 1}"
                for index, type_name in enumerate(signature)
            )

        source_objects = list(objects)
        if not source_objects:
            return tuple(f"obj{index + 1}" for index in range(arity))
        if len(source_objects) >= arity:
            return tuple(source_objects[:arity])

        values = list(source_objects)
        while len(values) < arity:
            values.append(source_objects[(len(values)) % len(source_objects)])
        return tuple(values[:arity])

    def _method_validation_task_args(self, method, fallback_task_args, method_library):
        task_schema = method_library.task_for_name(method.task_name)
        task_pattern = tuple(
            getattr(method, "task_args", ())
            or getattr(task_schema, "parameters", ())
            or tuple(method.parameters[: len(fallback_task_args)])
        )
        if not task_pattern:
            return tuple(fallback_task_args)

        constant_bindings = self._context_constant_bindings(method.context)
        resolved_args = []
        for index, pattern_symbol in enumerate(task_pattern):
            fallback = fallback_task_args[index] if index < len(fallback_task_args) else pattern_symbol
            resolved_args.append(constant_bindings.get(pattern_symbol, fallback))
        return tuple(resolved_args)

    def _context_constant_bindings(self, literals):
        direct_constant_bindings = {}
        pending_equalities = []

        for literal in literals:
            if not getattr(literal, "is_equality", False) or not literal.is_positive:
                continue
            left, right = literal.args
            left_is_var = self._is_variable_symbol(left)
            right_is_var = self._is_variable_symbol(right)
            if left_is_var and not right_is_var:
                direct_constant_bindings[left] = right
                continue
            if right_is_var and not left_is_var:
                direct_constant_bindings[right] = left
                continue
            if left_is_var and right_is_var:
                pending_equalities.append((left, right))

        changed = True
        while changed and pending_equalities:
            changed = False
            remaining = []
            for left, right in pending_equalities:
                if left in direct_constant_bindings and right not in direct_constant_bindings:
                    direct_constant_bindings[right] = direct_constant_bindings[left]
                    changed = True
                    continue
                if right in direct_constant_bindings and left not in direct_constant_bindings:
                    direct_constant_bindings[left] = direct_constant_bindings[right]
                    changed = True
                    continue
                remaining.append((left, right))
            pending_equalities = remaining

        return direct_constant_bindings

    def _seed_validation_scope(
        self,
        task_name,
        method_library,
        task_args,
        objects,
    ):
        object_pool = list(dict.fromkeys(task_args or objects))
        type_candidates: Dict[str, Set[str]] = defaultdict(set)
        self._merge_type_candidates(
            type_candidates,
            self._target_literal_type_candidates(method_library.target_literals),
        )
        self._merge_type_candidates(
            type_candidates,
            self._task_argument_type_candidates(
                task_name,
                task_args,
                method_library,
            ),
        )
        object_types = {}
        for obj in object_pool:
            object_types[obj] = self._resolve_symbol_type(
                symbol=obj,
                candidate_types=type_candidates.get(obj, set()),
                scope=f"Stage 4 task '{task_name}' object typing",
            )

        return object_pool, object_types

    def _typed_object_entries(self, object_pool, object_types):
        missing = [
            obj
            for obj in object_pool
            if obj not in object_types
        ]
        if missing:
            raise TypeResolutionError(
                "Missing resolved object types for Stage 4 problem export: "
                + ", ".join(sorted(missing)),
            )
        return tuple((obj, object_types[obj]) for obj in object_pool)

    def _build_type_parent_map(self) -> Dict[str, Optional[str]]:
        tokens = [
            token.strip()
            for token in (getattr(self.domain, "types", []) or [])
            if token and token.strip()
        ]
        if not tokens:
            return {"object": None}

        parent_map: Dict[str, Optional[str]] = {}
        pending_children: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "-":
                if not pending_children or index + 1 >= len(tokens):
                    raise ValueError(
                        "Malformed HDDL :types declaration (dangling '-').",
                    )
                parent_type = tokens[index + 1]
                for child_type in pending_children:
                    previous = parent_map.get(child_type)
                    if previous is not None and previous != parent_type:
                        raise ValueError(
                            f"Type '{child_type}' has conflicting parents "
                            f"('{previous}' vs '{parent_type}').",
                        )
                    parent_map[child_type] = parent_type
                pending_children = []
                index += 2
                continue

            pending_children.append(token)
            index += 1

        for child_type in pending_children:
            parent_map.setdefault(child_type, "object")

        parent_map["object"] = None
        changed = True
        while changed:
            changed = False
            for parent_type in list(parent_map.values()):
                if parent_type is None or parent_type in parent_map:
                    continue
                parent_map[parent_type] = "object" if parent_type != "object" else None
                changed = True

        for type_name in list(parent_map.keys()):
            if type_name == "object":
                parent_map[type_name] = None
                continue
            if parent_map[type_name] == type_name:
                raise ValueError(f"Type '{type_name}' cannot inherit from itself.")

            seen = {type_name}
            cursor = parent_map[type_name]
            while cursor is not None:
                if cursor in seen:
                    raise ValueError(f"Cyclic type hierarchy detected at '{type_name}'.")
                seen.add(cursor)
                cursor = parent_map.get(cursor)

        return parent_map

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("-", "_")

    @staticmethod
    def _is_variable_symbol(symbol: str) -> bool:
        return bool(symbol) and symbol[0].isupper()

    @staticmethod
    def _parameter_type(parameter: str) -> str:
        if "-" not in parameter:
            return "object"
        type_name = parameter.split("-", 1)[1].strip()
        return type_name or "object"

    def _require_known_type(self, type_name: str, source: str) -> str:
        if type_name in self.domain_type_names:
            return type_name
        raise TypeResolutionError(
            f"{source} references unknown type '{type_name}'. "
            f"Known types: {sorted(self.domain_type_names)}",
        )

    def _predicate_type_map(self) -> Dict[str, Tuple[str, ...]]:
        mapping: Dict[str, Tuple[str, ...]] = {}
        for predicate in getattr(self.domain, "predicates", []):
            mapping[predicate.name] = tuple(
                self._require_known_type(
                    self._parameter_type(parameter),
                    f"Predicate '{predicate.name}'",
                )
                for parameter in predicate.parameters
            )
        return mapping

    def _action_type_map(self) -> Dict[str, Tuple[str, ...]]:
        mapping: Dict[str, Tuple[str, ...]] = {}
        for action in getattr(self.domain, "actions", []):
            type_signature = tuple(
                self._require_known_type(
                    self._parameter_type(parameter),
                    f"Action '{action.name}'",
                )
                for parameter in action.parameters
            )
            mapping[action.name] = type_signature
            mapping[self._sanitize_name(action.name)] = type_signature
        return mapping

    def _task_type_map(self) -> Dict[str, Tuple[str, ...]]:
        mapping: Dict[str, Tuple[str, ...]] = {}
        for task in getattr(self.domain, "tasks", []):
            mapping[task.name] = tuple(
                self._require_known_type(
                    self._parameter_type(parameter),
                    f"Task '{task.name}'",
                )
                for parameter in task.parameters
            )
        return mapping

    @staticmethod
    def _merge_type_candidates(
        target: Dict[str, Set[str]],
        incoming: Dict[str, Set[str]],
    ) -> None:
        for symbol, type_names in incoming.items():
            if not symbol:
                continue
            if symbol not in target:
                target[symbol] = set()
            target[symbol].update(item for item in type_names if item)

    @staticmethod
    def _add_type_candidate(
        candidates: Dict[str, Set[str]],
        symbol: str,
        type_name: Optional[str],
    ) -> None:
        if not symbol or not type_name:
            return
        candidates.setdefault(symbol, set()).add(type_name)

    def _is_subtype(self, candidate_type: str, expected_type: str) -> bool:
        if candidate_type == expected_type:
            return True
        if candidate_type not in self.type_parent_map or expected_type not in self.type_parent_map:
            return False
        cursor = self.type_parent_map.get(candidate_type)
        visited = {candidate_type}
        while cursor is not None and cursor not in visited:
            if cursor == expected_type:
                return True
            visited.add(cursor)
            cursor = self.type_parent_map.get(cursor)
        return False

    def _resolve_symbol_type(
        self,
        *,
        symbol: str,
        candidate_types: Set[str],
        scope: str,
    ) -> str:
        if not candidate_types:
            raise TypeResolutionError(
                f"{scope}: symbol '{symbol}' has no type evidence.",
            )

        unknown_types = sorted(
            type_name
            for type_name in candidate_types
            if type_name not in self.domain_type_names
        )
        if unknown_types:
            raise TypeResolutionError(
                f"{scope}: symbol '{symbol}' references unknown types {unknown_types}.",
            )

        feasible = sorted(
            type_name
            for type_name in self.domain_type_names
            if all(self._is_subtype(type_name, required) for required in candidate_types)
        )
        if not feasible:
            raise TypeResolutionError(
                f"{scope}: symbol '{symbol}' has conflicting type constraints "
                f"{sorted(candidate_types)}.",
            )

        most_specific = sorted(
            type_name
            for type_name in feasible
            if not any(
                other != type_name and self._is_subtype(other, type_name)
                for other in feasible
            )
        )
        if len(most_specific) != 1:
            raise TypeResolutionError(
                f"{scope}: symbol '{symbol}' is ambiguous under constraints "
                f"{sorted(candidate_types)}; candidate leaf types={most_specific}.",
            )
        return most_specific[0]

    def _task_type_signature(self, task_name: str, method_library=None) -> Tuple[str, ...]:
        signature = self.task_type_map.get(task_name)
        if signature is not None:
            return signature
        if method_library is None:
            return ()

        task_schema = method_library.task_for_name(task_name)
        if task_schema is None or len(task_schema.source_predicates) != 1:
            return ()

        predicate_name = task_schema.source_predicates[0]
        predicate_signature = self.predicate_type_map.get(predicate_name, ())
        if not predicate_signature:
            return ()
        if len(predicate_signature) != len(task_schema.parameters):
            raise TypeResolutionError(
                f"Task '{task_name}' source predicate '{predicate_name}' arity mismatch: "
                f"task has {len(task_schema.parameters)} args, predicate has "
                f"{len(predicate_signature)}.",
            )
        return predicate_signature

    def _collect_argument_signature_constraints(
        self,
        *,
        candidates: Dict[str, Set[str]],
        args: Sequence[str],
        signature: Sequence[str],
        scope: str,
    ) -> None:
        if not signature:
            return
        if len(args) != len(signature):
            raise TypeResolutionError(
                f"{scope}: arity mismatch (args={len(args)}, signature={len(signature)}).",
            )
        for index, arg in enumerate(args):
            self._add_type_candidate(candidates, arg, signature[index])

    def _literal_type_candidates(
        self,
        literal: HTNLiteral,
    ) -> Dict[str, Set[str]]:
        if literal.is_equality:
            return {}
        candidates: Dict[str, Set[str]] = defaultdict(set)
        predicate_types = self.predicate_type_map.get(literal.predicate)
        if predicate_types is None:
            raise TypeResolutionError(
                f"Unknown predicate '{literal.predicate}' in literal '{literal.to_signature()}'.",
            )
        self._collect_argument_signature_constraints(
            candidates=candidates,
            args=literal.args,
            signature=predicate_types,
            scope=f"Literal '{literal.to_signature()}' typing",
        )
        return candidates

    def _target_literal_type_candidates(
        self,
        target_literals: Sequence[HTNLiteral],
    ) -> Dict[str, Set[str]]:
        candidates: Dict[str, Set[str]] = defaultdict(set)
        for literal in target_literals:
            self._merge_type_candidates(candidates, self._literal_type_candidates(literal))
        return candidates

    def _method_variable_type_hints(
        self,
        method,
        method_library,
    ) -> Dict[str, str]:
        candidates: Dict[str, Set[str]] = defaultdict(set)
        task_signature = self._task_type_signature(method.task_name, method_library)
        task_schema = method_library.task_for_name(method.task_name)
        task_binding_args: List[str] = []
        if task_schema is not None and task_schema.parameters:
            for index, task_parameter in enumerate(task_schema.parameters):
                if task_parameter in method.parameters:
                    task_binding_args.append(task_parameter)
                elif index < len(method.parameters):
                    task_binding_args.append(method.parameters[index])
                else:
                    raise TypeResolutionError(
                        f"Method '{method.method_name}' is missing parameter mapping for "
                        f"task argument '{task_parameter}'.",
                    )
        else:
            task_binding_args = list(method.parameters[:len(task_signature)])
        self._collect_argument_signature_constraints(
            candidates=candidates,
            args=tuple(task_binding_args),
            signature=task_signature,
            scope=f"Method '{method.method_name}' task parameter typing",
        )

        def collect_literal(literal: Optional[HTNLiteral]) -> None:
            if literal is None or literal.is_equality:
                return
            literal_candidates = self._literal_type_candidates(literal)
            self._merge_type_candidates(candidates, literal_candidates)

        for literal in method.context:
            collect_literal(literal)

        for step in method.subtasks:
            collect_literal(step.literal)
            for literal in step.preconditions:
                collect_literal(literal)
            for literal in step.effects:
                collect_literal(literal)
            if step.kind == "compound":
                step_signature = self._task_type_signature(step.task_name, method_library)
                if not step_signature:
                    continue
                self._collect_argument_signature_constraints(
                    candidates=candidates,
                    args=step.args,
                    signature=step_signature,
                    scope=(
                        f"Method '{method.method_name}' compound step "
                        f"'{step.step_id}:{step.task_name}' typing"
                    ),
                )
                continue

            if step.kind != "primitive":
                continue

            action_types = self.action_type_map.get(step.action_name or "")
            if action_types is None:
                action_types = self.action_type_map.get(step.task_name)
            if action_types is None and step.action_name:
                action_types = self.action_type_map.get(self._sanitize_name(step.action_name))
            if action_types is None:
                raise TypeResolutionError(
                    f"Method '{method.method_name}' references primitive step "
                    f"'{step.step_id}:{step.task_name}' without known action signature.",
                )
            self._collect_argument_signature_constraints(
                candidates=candidates,
                args=step.args,
                signature=action_types,
                scope=(
                    f"Method '{method.method_name}' primitive step "
                    f"'{step.step_id}:{step.task_name}' typing"
                ),
            )

        variable_symbols: Set[str] = set()
        for token in method.parameters:
            if self._is_variable_symbol(token):
                variable_symbols.add(token)
        for literal in method.context:
            variable_symbols.update(
                arg
                for arg in literal.args
                if self._is_variable_symbol(arg)
            )
        for step in method.subtasks:
            variable_symbols.update(
                arg
                for arg in step.args
                if self._is_variable_symbol(arg)
            )
            if step.literal:
                variable_symbols.update(
                    arg
                    for arg in step.literal.args
                    if self._is_variable_symbol(arg)
                )
            for literal in (*step.preconditions, *step.effects):
                variable_symbols.update(
                    arg
                    for arg in literal.args
                    if self._is_variable_symbol(arg)
                )

        return {
            symbol: self._resolve_symbol_type(
                symbol=symbol,
                candidate_types=candidates.get(symbol, set()),
                scope=f"Stage 3 method '{method.method_name}' variable typing",
            )
            for symbol in sorted(variable_symbols)
        }

    def _task_argument_type_candidates(
        self,
        task_name: str,
        task_args: Sequence[str],
        method_library,
    ) -> Dict[str, Set[str]]:
        candidates: Dict[str, Set[str]] = defaultdict(set)

        task_signature = self._task_type_signature(task_name, method_library)
        self._collect_argument_signature_constraints(
            candidates=candidates,
            args=task_args,
            signature=task_signature,
            scope=f"Task '{task_name}' argument typing",
        )

        for binding in method_library.target_task_bindings:
            if binding.task_name != task_name:
                continue
            target_literal = next(
                (
                    literal
                    for literal in method_library.target_literals
                    if literal.to_signature() == binding.target_literal
                ),
                None,
            )
            if target_literal is None:
                continue
            self._merge_type_candidates(
                candidates,
                self._literal_type_candidates(target_literal),
            )

        for method in method_library.methods_for_task(task_name):
            variable_types = self._method_variable_type_hints(method, method_library)
            for parameter, arg in zip(method.parameters, task_args):
                self._add_type_candidate(candidates, arg, variable_types.get(parameter))
            for literal in method.context:
                literal_candidates = self._literal_type_candidates(literal)
                for symbol, type_names in literal_candidates.items():
                    if self._is_variable_symbol(symbol):
                        continue
                    self._merge_type_candidates(candidates, {symbol: type_names})

        return candidates

    def _validate_method_library_typing(self, method_library) -> None:
        for literal in method_library.target_literals:
            self._literal_type_candidates(literal)

        for binding in method_library.target_task_bindings:
            task_name = binding.task_name
            target_literal = next(
                (
                    literal
                    for literal in method_library.target_literals
                    if literal.to_signature() == binding.target_literal
                ),
                None,
            )
            if target_literal is None:
                raise TypeResolutionError(
                    f"Stage 3 binding references missing target literal "
                    f"'{binding.target_literal}'.",
                )
            candidates = self._task_argument_type_candidates(
                task_name,
                target_literal.args,
                method_library,
            )
            for arg in target_literal.args:
                self._resolve_symbol_type(
                    symbol=arg,
                    candidate_types=candidates.get(arg, set()),
                    scope=(
                        "Stage 3 target-task binding typing "
                        f"('{binding.target_literal}' -> '{task_name}')"
                    ),
                )

        for method in method_library.methods:
            self._method_variable_type_hints(method, method_library)

    def _method_validation_initial_facts(
        self,
        planner,
        method,
        method_library,
        task_args,
        objects,
        object_pool=None,
        object_types=None,
    ):
        predicate_arity = {
            predicate.name: len(predicate.parameters)
            for predicate in getattr(self.domain, "predicates", [])
        }
        bindings = {
            parameter: arg
            for parameter, arg in zip(method.parameters, task_args)
        }
        object_pool = object_pool if object_pool is not None else list(dict.fromkeys(task_args or objects))
        if not object_pool:
            object_pool = list(task_args)
        method_variable_types = self._method_variable_type_hints(method, method_library)
        inferred_candidates = self._task_argument_type_candidates(
            method.task_name,
            task_args,
            method_library,
        )
        if object_types is None:
            object_types = {}
        required_bound_objects = set(bindings.values())
        for obj in required_bound_objects:
            if obj not in object_pool:
                object_pool.append(obj)
            if obj in object_types:
                continue
            expected_candidates = {
                method_variable_types[parameter]
                for parameter, bound_object in bindings.items()
                if bound_object == obj and parameter in method_variable_types
            }
            if len(expected_candidates) == 1:
                object_types[obj] = next(iter(expected_candidates))
                continue
            object_types[obj] = self._resolve_symbol_type(
                symbol=obj,
                candidate_types=inferred_candidates.get(obj, set()),
                scope=f"Stage 4 method '{method.method_name}' object typing",
            )

        for parameter, bound_object in bindings.items():
            expected_type = method_variable_types.get(parameter)
            if expected_type is None:
                continue
            actual_type = object_types.get(bound_object)
            if actual_type is None:
                actual_type = self._resolve_symbol_type(
                    symbol=bound_object,
                    candidate_types=inferred_candidates.get(bound_object, set()),
                    scope=f"Stage 4 method '{method.method_name}' object typing",
                )
                object_types[bound_object] = actual_type
            if not self._is_subtype(actual_type, expected_type):
                raise TypeResolutionError(
                    f"Stage 4 method '{method.method_name}' binds parameter '{parameter}' "
                    f"(expected {expected_type}) to object '{bound_object}' of type "
                    f"{actual_type}.",
                )

        def bind_symbol(symbol):
            if not symbol:
                return symbol
            if symbol[0].islower():
                return symbol
            if symbol not in bindings:
                expected_type = method_variable_types.get(symbol)
                if expected_type is None:
                    raise TypeResolutionError(
                        f"Stage 4 method '{method.method_name}' cannot type variable "
                        f"'{symbol}' while constructing witness initial facts.",
                    )
                for candidate in object_pool:
                    if candidate in bindings.values():
                        continue
                    candidate_type = object_types.get(candidate)
                    if candidate_type is None:
                        continue
                    if not self._is_subtype(candidate_type, expected_type):
                        continue
                    bindings[symbol] = candidate
                    break
                else:
                    index = 1
                    candidate = f"witness_{expected_type}_{index}"
                    while candidate in object_pool:
                        index += 1
                        candidate = f"witness_{expected_type}_{index}"
                    object_pool.append(candidate)
                    object_types[candidate] = expected_type
                    bindings[symbol] = candidate
            return bindings[symbol]

        blocked_signatures = set()
        for literal in method.context:
            if literal.is_equality:
                continue
            if literal.is_positive:
                continue
            grounded_args = tuple(bind_symbol(arg) for arg in literal.args)
            signature = (
                f"{literal.predicate}({', '.join(grounded_args)})"
                if grounded_args
                else literal.predicate
            )
            blocked_signatures.add(signature)

        literal_pool = []
        seen_literal_signatures = set()

        def add_grounded_literal(literal):
            if literal.is_equality:
                return
            if not literal.is_positive:
                return
            grounded_literal = HTNLiteral(
                predicate=literal.predicate,
                args=tuple(bind_symbol(arg) for arg in literal.args),
                is_positive=True,
                source_symbol=None,
            )
            signature = grounded_literal.to_signature()
            if signature in blocked_signatures or signature in seen_literal_signatures:
                return
            seen_literal_signatures.add(signature)
            literal_pool.append(grounded_literal)

        for literal in method.context:
            add_grounded_literal(literal)

        parser = HDDLConditionParser()
        action_semantics = {
            action.name: parser.parse_action(action)
            for action in self.domain.actions
        }
        for action in self.domain.actions:
            action_semantics[action.name.replace("-", "_")] = action_semantics[action.name]
        witness_steps = tuple(method.subtasks)
        step_positive_literals = {}
        planning_hint_signatures = set(seen_literal_signatures)

        for step in witness_steps:
            step_preconditions = list(step.preconditions)
            explicit_signatures = {
                literal.to_signature()
                for literal in step_preconditions
            }
            if step.kind == "primitive":
                action_schema = action_semantics.get(step.action_name or step.task_name)
                if action_schema is not None:
                    step_bindings = {
                        parameter: arg
                        for parameter, arg in zip(action_schema.parameters, step.args)
                    }
                    for pattern in action_schema.preconditions:
                        literal = HTNLiteral(
                            predicate=pattern.predicate,
                            args=tuple(step_bindings.get(arg, arg) for arg in pattern.args),
                            is_positive=pattern.is_positive,
                            source_symbol=None,
                        )
                        signature = literal.to_signature()
                        if signature in explicit_signatures:
                            continue
                        explicit_signatures.add(signature)
                        step_preconditions.append(literal)

            grounded_positive_literals = []
            grounded_seen = set()
            for literal in step_preconditions:
                if literal.is_equality:
                    continue
                if not literal.is_positive:
                    continue
                grounded_literal = HTNLiteral(
                    predicate=literal.predicate,
                    args=tuple(bind_symbol(arg) for arg in literal.args),
                    is_positive=True,
                    source_symbol=None,
                )
                signature = grounded_literal.to_signature()
                if signature in blocked_signatures or signature in grounded_seen:
                    continue
                grounded_seen.add(signature)
                grounded_positive_literals.append(grounded_literal)
                planning_hint_signatures.add(signature)

            step_positive_literals[step.step_id] = tuple(grounded_positive_literals)

        for step in witness_steps:
            for literal in step_positive_literals.get(step.step_id, ()):
                if literal.to_signature() in seen_literal_signatures:
                    continue
                seen_literal_signatures.add(literal.to_signature())
                literal_pool.append(literal)
            if step.kind == "compound":
                grounded_args = tuple(bind_symbol(arg) for arg in step.args)
                child_literals = self._select_child_witness_context(
                    method_library,
                    step.task_name,
                    grounded_args,
                    blocked_signatures,
                    planning_hint_signatures | set(seen_literal_signatures),
                    action_semantics,
                    object_pool,
                    object_types,
                )
                for literal in child_literals:
                    if literal.to_signature() in seen_literal_signatures:
                        continue
                    seen_literal_signatures.add(literal.to_signature())
                    literal_pool.append(literal)

        facts = []
        seen = set()

        for literal in literal_pool:
            if literal.is_equality:
                continue
            grounded_args = literal.args
            signature = (
                f"{literal.predicate}({', '.join(grounded_args)})"
                if grounded_args
                else literal.predicate
            )
            if signature in blocked_signatures:
                continue
            fact = (
                f"({literal.predicate} {' '.join(grounded_args)})"
                if grounded_args
                else f"({literal.predicate})"
            )
            if fact in seen:
                continue
            seen.add(fact)
            facts.append(fact)

        return tuple(facts)

    @staticmethod
    def _initial_frontier_steps(method):
        if not method.subtasks:
            return ()
        if not method.ordering:
            return tuple(method.subtasks)

        in_degree = {
            step.step_id: 0
            for step in method.subtasks
        }
        step_lookup = {
            step.step_id: step
            for step in method.subtasks
        }
        for _, after in method.ordering:
            if after in in_degree:
                in_degree[after] += 1

        return tuple(
            step_lookup[step.step_id]
            for step in method.subtasks
            if in_degree.get(step.step_id, 0) == 0
        )

    def _select_child_witness_context(
        self,
        method_library,
        task_name,
        grounded_args,
        blocked_signatures,
        known_signatures,
        action_semantics,
        object_pool,
        object_types,
    ):
        best_literals = ()
        best_score = (-1, -1)
        parsed_known_signatures = [
            parsed
            for parsed in (
                self._parse_signature_text(signature)
                for signature in known_signatures
            )
            if parsed is not None
        ]

        for child_method in method_library.methods_for_task(task_name):
            local_bindings = {
                parameter: arg
                for parameter, arg in zip(child_method.parameters, grounded_args)
            }
            child_variable_types = self._method_variable_type_hints(
                child_method,
                method_library,
            )
            grounded_literals = []
            grounded_seen = set()
            promoted_literals = self._promoted_child_context_literals(
                child_method,
                action_semantics,
            )

            for literal in promoted_literals:
                grounded_literal = self._ground_child_witness_literal(
                    literal,
                    local_bindings,
                    parsed_known_signatures,
                    object_pool,
                    object_types,
                    child_variable_types,
                )
                signature = grounded_literal.to_signature()
                if signature in blocked_signatures or signature in grounded_seen:
                    continue
                grounded_seen.add(signature)
                grounded_literals.append(grounded_literal)

            overlap = sum(
                1
                for literal in grounded_literals
                if literal.to_signature() in known_signatures
            )
            score = (overlap, len(grounded_literals))
            if score > best_score:
                best_score = score
                best_literals = tuple(grounded_literals)

        return best_literals

    def _promoted_child_context_literals(self, method, action_semantics):
        literals = []
        seen = set()

        def add(literal):
            if literal.is_equality:
                return
            if not literal.is_positive:
                return
            signature = literal.to_signature()
            if signature in seen:
                return
            seen.add(signature)
            literals.append(literal)

        for literal in method.context:
            add(literal)

        for step in method.subtasks:
            for literal in step.preconditions:
                add(literal)
            if step.kind != "primitive":
                continue

            action_schema = action_semantics.get(step.action_name or step.task_name)
            if action_schema is None:
                continue

            step_bindings = {
                parameter: arg
                for parameter, arg in zip(action_schema.parameters, step.args)
            }
            for pattern in action_schema.preconditions:
                add(
                    HTNLiteral(
                        predicate=pattern.predicate,
                        args=tuple(step_bindings.get(arg, arg) for arg in pattern.args),
                        is_positive=pattern.is_positive,
                        source_symbol=None,
                    )
                )

        return tuple(literals)

    def _ground_child_witness_literal(
        self,
        literal,
        local_bindings,
        parsed_known_signatures,
        object_pool,
        object_types,
        variable_type_hints,
    ):
        for candidate in parsed_known_signatures:
            if candidate["predicate"] != literal.predicate:
                continue
            if len(candidate["args"]) != len(literal.args):
                continue

            trial_bindings = dict(local_bindings)
            grounded_args = []
            matches = True
            for token, actual in zip(literal.args, candidate["args"]):
                if not token:
                    grounded_args.append(actual)
                    continue
                if token[0].islower():
                    if token != actual:
                        matches = False
                        break
                    grounded_args.append(actual)
                    continue

                bound_value = trial_bindings.get(token)
                if bound_value is not None and bound_value != actual:
                    matches = False
                    break
                trial_bindings[token] = actual
                grounded_args.append(actual)

            if not matches:
                continue

            local_bindings.clear()
            local_bindings.update(trial_bindings)
            return HTNLiteral(
                predicate=literal.predicate,
                args=tuple(grounded_args),
                is_positive=literal.is_positive,
                source_symbol=None,
            )

        grounded_args = []
        used_values = set(local_bindings.values())
        for token in literal.args:
            if not token:
                grounded_args.append(token)
                continue
            if token[0].islower():
                grounded_args.append(token)
                continue
            if token not in local_bindings:
                expected_type = variable_type_hints.get(token)
                if expected_type is None:
                    raise TypeResolutionError(
                        f"Stage 4 child witness typing cannot resolve variable '{token}' "
                        f"for literal '{literal.to_signature()}'.",
                    )
                candidate = next(
                    (
                        obj
                        for obj in object_pool
                        if obj not in used_values
                        and obj in object_types
                        and self._is_subtype(object_types[obj], expected_type)
                    ),
                    None,
                )
                if candidate is None:
                    index = 1
                    candidate = f"witness_{expected_type}_{index}"
                    while candidate in object_pool:
                        index += 1
                        candidate = f"witness_{expected_type}_{index}"
                    object_pool.append(candidate)
                    object_types[candidate] = expected_type
                local_bindings[token] = candidate
                used_values.add(candidate)
            grounded_args.append(local_bindings[token])

        return HTNLiteral(
            predicate=literal.predicate,
            args=tuple(grounded_args),
            is_positive=literal.is_positive,
            source_symbol=None,
        )

    @staticmethod
    def _parse_signature_text(signature):
        if not signature:
            return None
        if " == " in signature:
            left, right = signature.split(" == ", 1)
            return {
                "predicate": "=",
                "args": (left.strip(), right.strip()),
                "is_positive": True,
            }
        if " != " in signature:
            left, right = signature.split(" != ", 1)
            return {
                "predicate": "=",
                "args": (left.strip(), right.strip()),
                "is_positive": False,
            }
        is_positive = not signature.startswith("!")
        text = signature[1:] if not is_positive else signature
        if "(" not in text:
            return {
                "predicate": text,
                "args": (),
                "is_positive": is_positive,
            }

        predicate, remainder = text.split("(", 1)
        args_blob = remainder.rsplit(")", 1)[0]
        args = tuple(
            part.strip()
            for part in args_blob.split(",")
            if part.strip()
        )
        return {
            "predicate": predicate,
            "args": args,
            "is_positive": is_positive,
        }

    def _task_witness_initial_facts(
        self,
        planner,
        task_name,
        method_library,
        task_args,
        objects,
        object_pool=None,
        object_types=None,
    ):
        facts = []
        seen = set()
        task_methods = method_library.methods_for_task(task_name)
        if not task_methods:
            return ()

        for method in task_methods:
            for fact in self._method_validation_initial_facts(
                planner,
                method,
                method_library,
                task_args,
                objects,
                object_pool=object_pool,
                object_types=object_types,
            ):
                if fact in seen:
                    continue
                seen.add(fact)
                facts.append(fact)

        return tuple(facts)

    def _stage5_agentspeak_rendering(self, ltl_spec, method_library, plan_records):
        """Stage 5: HTN methods + validated DFA wrappers -> AgentSpeak rendering."""
        print("\n[STAGE 5] AgentSpeak Rendering")
        print("-"*80)

        try:
            renderer = AgentSpeakRenderer()
            asl_code = renderer.generate(
                domain=self.domain,
                objects=ltl_spec.objects,
                method_library=method_library,
                plan_records=plan_records,
            )
            metadata = {
                "transition_count": len(plan_records),
                "rendered_methods": len(method_library.methods),
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

    def _stage6_jason_validation(self, ltl_spec, method_library, plan_records, asl_code):
        """Stage 6: run generated AgentSpeak with Jason (RunLocalMAS)."""
        print("\n[STAGE 6] Jason Runtime Validation")
        print("-"*80)

        try:
            stage6_dir = Path(__file__).parent / "stage6_jason_validation"
            runner = JasonRunner(stage6_dir=stage6_dir)
            seed_facts, seed_transition = self._stage6_runtime_seed_facts(
                plan_records,
                method_library.target_literals,
            )
            stage6_objects = self._stage6_runtime_objects(ltl_spec.objects, seed_facts)
            stage6_object_types = self._stage6_object_types(
                stage6_objects,
                method_library,
                seed_facts,
                problem_object_types=self.problem.object_types if self.problem is not None else None,
            )
            action_schemas = self._stage6_action_schemas()
            result = runner.validate(
                agentspeak_code=asl_code,
                target_literals=method_library.target_literals,
                method_library=method_library,
                action_schemas=action_schemas,
                seed_facts=seed_facts,
                domain_name=self.domain.name,
                output_dir=self.output_dir,
            )
            summary = {
                "backend": result.backend,
                "status": result.status,
                "java_path": result.java_path,
                "java_version": result.java_version,
                "javac_path": result.javac_path,
                "jason_jar": result.jason_jar,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "transition_count": len(plan_records),
                "target_literal_count": len(method_library.target_literals),
                "seed_fact_count": len(seed_facts),
                "seed_transition": seed_transition,
                "executed_action_count": len(result.action_path),
                "action_path_artifact": result.artifacts.get("action_path"),
                "runtime_object_count": len(stage6_objects),
                "resolved_object_types": stage6_object_types,
                "action_schema_count": len(action_schemas),
                "environment_adapter": result.environment_adapter,
            }
            artifacts = result.to_dict()
            self.logger.log_stage6_jason_validation(
                artifacts,
                "Success",
                metadata=summary,
            )

            print("✓ Jason runtime validation complete")
            print(f"  Backend: {result.backend}")
            print(f"  Java: {result.java_path} (major={result.java_version})")
            print(f"  Jason jar: {result.jason_jar}")
            print(f"  Exit code: {result.exit_code}")
            print(f"  Seed facts: {len(seed_facts)} (from {seed_transition})")
            print(f"  Stage 6 artifacts saved to: {self.output_dir}")

            return {
                "summary": summary,
                "artifacts": artifacts,
            }

        except JasonValidationError as e:
            metadata = dict(getattr(e, "metadata", {}) or {})
            summary = {
                "backend": "RunLocalMAS",
                "status": "failed",
                "error": str(e),
            }
            if metadata:
                summary.update({
                    "java_path": metadata.get("java_path"),
                    "java_version": metadata.get("java_version"),
                    "jason_jar": metadata.get("jason_jar"),
                    "exit_code": metadata.get("exit_code"),
                    "timed_out": metadata.get("timed_out"),
                })
            self.logger.log_stage6_jason_validation(
                metadata if metadata else None,
                "Failed",
                error=str(e),
                metadata=summary,
            )
            print(f"✗ Stage 6 Failed: {e}")
            return None
        except Exception as e:
            self.logger.log_stage6_jason_validation(
                None,
                "Failed",
                error=str(e),
                metadata={
                    "backend": "RunLocalMAS",
                    "status": "failed",
                },
            )
            print(f"✗ Stage 6 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _stage7_official_verification(self, method_library, stage6_data):
        """Stage 7: verify the generated hierarchical plan with the official IPC verifier."""
        print("\n[STAGE 7] Official IPC HTN Plan Verification")
        print("-"*80)

        if self.problem is None or not self.problem_file:
            summary = {
                "backend": "pandaPIparser",
                "status": "skipped",
                "reason": "No problem_file was provided",
            }
            artifacts = {
                "tool_available": None,
                "plan_kind": None,
                "verification_result": None,
                "primitive_plan_executable": None,
                "reached_goal_state": None,
            }
            self.logger.log_stage7_official_verification(
                artifacts,
                "Skipped",
                metadata=summary,
            )
            print("• Skipped: no problem file was provided")
            return {
                "summary": summary,
                "artifacts": artifacts,
            }

        verifier = IPCPlanVerifier()
        if not verifier.tool_available():
            error = "pandaPIparser is not available on PATH for official IPC verification"
            self.logger.log_stage7_official_verification(
                None,
                "Failed",
                error=error,
                metadata={
                    "backend": "pandaPIparser",
                    "status": "failed",
                },
            )
            print(f"✗ Stage 7 Failed: {error}")
            return None

        stage6_artifacts = stage6_data.get("artifacts") or {}
        verifier_result = verifier.verify_plan(
            domain_file=self.domain_file,
            problem_file=self.problem_file,
            action_path=stage6_artifacts.get("action_path") or [],
            method_library=method_library,
            method_trace=stage6_artifacts.get("method_trace") or [],
            output_dir=self.output_dir,
        )
        artifacts = verifier_result.to_dict()
        summary = {
            "backend": "pandaPIparser",
            "status": "success" if verifier_result.verification_result is True else "failed",
            "tool_available": verifier_result.tool_available,
            "plan_kind": verifier_result.plan_kind,
            "verification_result": verifier_result.verification_result,
            "primitive_plan_executable": verifier_result.primitive_plan_executable,
            "reached_goal_state": verifier_result.reached_goal_state,
            "build_warning": verifier_result.build_warning,
        }

        if (
            not verifier_result.tool_available
            or verifier_result.plan_kind != "hierarchical"
            or verifier_result.verification_result is not True
        ):
            error = (
                "Official IPC verifier rejected the generated hierarchical plan: "
                f"plan_kind={verifier_result.plan_kind}, "
                f"verification_result={verifier_result.verification_result}"
            )
            self.logger.log_stage7_official_verification(
                artifacts,
                "Failed",
                error=error,
                metadata=summary,
            )
            print(f"✗ Stage 7 Failed: {error}")
            return None

        self.logger.log_stage7_official_verification(
            artifacts,
            "Success",
            metadata=summary,
        )
        print("✓ Official IPC verification complete")
        print(f"  Plan kind: {verifier_result.plan_kind}")
        print(f"  Verification result: {verifier_result.verification_result}")
        print(f"  Verifier output: {artifacts.get('output_file')}")

        return {
            "summary": summary,
            "artifacts": artifacts,
        }

    def _stage6_runtime_seed_facts(self, plan_records, target_literals):
        if self.problem is not None:
            return self._stage6_problem_seed_facts()
        return self._stage6_seed_facts(plan_records, target_literals)

    def _stage6_problem_seed_facts(self):
        if self.problem is None or not self.problem_file:
            return (), None
        return (
            tuple(self._render_problem_fact(fact) for fact in self.problem.init_facts),
            f"problem_init:{Path(self.problem_file).name}",
        )

    def _stage6_runtime_objects(self, objects, seed_facts):
        runtime_objects = list(self.problem.objects) if self.problem is not None else list(objects or [])
        for fact in seed_facts:
            parsed = self._parse_positive_hddl_fact(fact)
            if parsed is None:
                continue
            _, args = parsed
            for arg in args:
                if arg not in runtime_objects:
                    runtime_objects.append(arg)
        return tuple(runtime_objects)

    @staticmethod
    def _stage6_seed_facts(plan_records, target_literals):
        if not plan_records:
            return (), None
        negative_targets = {
            (literal.predicate, tuple(literal.args))
            for literal in target_literals
            if not literal.is_positive and not literal.is_equality
        }
        facts: List[str] = []
        seen_facts: Set[str] = set()
        source_steps: List[str] = []

        for record in plan_records:
            source_steps.append(record.get("transition_name", "unknown"))
            for fact in tuple(record.get("initial_facts", ()) or ()):
                parsed = LTL_BDI_Pipeline._parse_positive_hddl_fact(fact)
                if parsed is not None and parsed in negative_targets:
                    continue
                if fact in seen_facts:
                    continue
                seen_facts.add(fact)
                facts.append(fact)

        return tuple(facts), ",".join(source_steps)

    def _stage6_action_schemas(self):
        parser = HDDLConditionParser()
        schemas = []
        for action in self.domain.actions:
            parsed = parser.parse_action(action)
            schemas.append(
                {
                    "functor": self._sanitize_name(action.name),
                    "source_name": action.name,
                    "parameters": list(parsed.parameters),
                    "preconditions": [
                        {
                            "predicate": literal.predicate,
                            "args": list(literal.args),
                            "is_positive": literal.is_positive,
                        }
                        for literal in parsed.preconditions
                    ],
                    "precondition_clauses": [
                        [
                            {
                                "predicate": literal.predicate,
                                "args": list(literal.args),
                                "is_positive": literal.is_positive,
                            }
                            for literal in clause
                        ]
                        for clause in parsed.precondition_clauses
                    ],
                    "effects": [
                        {
                            "predicate": literal.predicate,
                            "args": list(literal.args),
                            "is_positive": literal.is_positive,
                        }
                        for literal in parsed.effects
                    ],
                }
            )
        return schemas

    def _stage6_object_types(
        self,
        objects,
        method_library,
        seed_facts,
        *,
        problem_object_types: Optional[Dict[str, str]] = None,
    ):
        candidates: Dict[str, Set[str]] = defaultdict(set)
        required_objects: Set[str] = set()
        self._merge_type_candidates(
            candidates,
            self._target_literal_type_candidates(method_library.target_literals),
        )
        for literal in method_library.target_literals:
            required_objects.update(literal.args)
        for binding in method_library.target_task_bindings:
            target_literal = next(
                (
                    literal
                    for literal in method_library.target_literals
                    if literal.to_signature() == binding.target_literal
                ),
                None,
            )
            if target_literal is None:
                continue
            self._merge_type_candidates(
                candidates,
                self._task_argument_type_candidates(
                    binding.task_name,
                    target_literal.args,
                    method_library,
                ),
            )

        for fact in seed_facts:
            parsed = self._parse_positive_hddl_fact(fact)
            if parsed is None:
                continue
            predicate, args = parsed
            required_objects.update(args)
            signature = self.predicate_type_map.get(predicate)
            if signature is None:
                continue
            temp_candidates: Dict[str, Set[str]] = defaultdict(set)
            self._collect_argument_signature_constraints(
                candidates=temp_candidates,
                args=args,
                signature=signature,
                scope=f"Stage 6 seed fact '{fact}' typing",
            )
            self._merge_type_candidates(candidates, temp_candidates)

        resolved = {}
        available_objects = list(dict.fromkeys(objects or ()))
        for obj in sorted(required_objects):
            if obj not in available_objects:
                available_objects.append(obj)
        for obj in available_objects:
            if obj not in required_objects:
                continue
            if problem_object_types and obj in problem_object_types:
                type_name = problem_object_types[obj]
                if type_name not in self.domain_type_names:
                    raise TypeResolutionError(
                        f"Stage 6 object typing: problem object '{obj}' has unknown type '{type_name}'.",
                    )
                resolved[obj] = type_name
                continue
            resolved[obj] = self._resolve_symbol_type(
                symbol=obj,
                candidate_types=candidates.get(obj, set()),
                scope="Stage 6 object typing",
            )
        return resolved

    @staticmethod
    def _render_problem_fact(fact) -> str:
        inner = fact.predicate
        if fact.args:
            inner = f"{inner} {' '.join(fact.args)}"
        return f"({inner})" if fact.is_positive else f"(not ({inner}))"

    @staticmethod
    def _parse_positive_hddl_fact(fact: str):
        text = (fact or "").strip()
        if not text.startswith("(") or not text.endswith(")"):
            return None
        inner = text[1:-1].strip()
        if not inner or inner.startswith("not "):
            return None
        tokens = inner.split()
        if not tokens or tokens[0] == "=":
            return None
        return tokens[0], tuple(tokens[1:])
