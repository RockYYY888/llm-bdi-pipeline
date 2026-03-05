"""
LTL-BDI pipeline: NL -> LTLf -> DFA -> HTN synthesis -> PANDA -> AgentSpeak.
"""

from pathlib import Path
from typing import Dict, Any

from utils.config import get_config
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
from stage2_dfa_generation.dfa_builder import DFABuilder
from stage3_method_synthesis.htn_method_synthesis import HTNMethodSynthesizer
from stage3_method_synthesis.htn_schema import HTNLiteral
from stage4_panda_planning.panda_planner import PANDAPlanner
from stage5_agentspeak_rendering.agentspeak_renderer import AgentSpeakRenderer
from stage6_jason_validation.jason_runner import JasonRunner, JasonValidationError
from utils.hddl_condition_parser import HDDLConditionParser
from utils.pipeline_logger import PipelineLogger


class LTL_BDI_Pipeline:
    """
    LTL-BDI pipeline implementing Stages 1-5 (dfa_agentspeak mode)

    Stage 1: Natural Language -> LTLf Specification
    Stage 2: LTLf -> DFA Conversion (ltlf2dfa)
    Stage 3: DFA -> HTN Method Synthesis
    Stage 4: HTN Method Library -> PANDA Planning
    Stage 5: HTN Methods + DFA Wrappers -> AgentSpeak Rendering
    Stage 6: AgentSpeak -> Jason Runtime Validation
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
        print("STAGES 1-6 COMPLETED SUCCESSFULLY")
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
                "llm_finish_reason": synthesis_meta.get("llm_finish_reason"),
                "llm_attempts": synthesis_meta.get("llm_attempts"),
                "llm_response_time_seconds": synthesis_meta.get("llm_response_time_seconds"),
                "llm_attempt_durations_seconds": synthesis_meta.get(
                    "llm_attempt_durations_seconds",
                ),
                "target_literals": synthesis_meta["target_literals"],
                "compound_tasks": synthesis_meta["compound_tasks"],
                "primitive_tasks": synthesis_meta["primitive_tasks"],
                "methods": synthesis_meta["methods"],
            }
            transition_specs = synthesizer.extract_progressing_transitions(
                grounding_map,
                dfa_result,
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
                representative_args = self._representative_task_args(
                    task_name,
                    method_library,
                    ltl_spec.objects,
                    plan_records,
                )
                for method in method_library.methods_for_task(task_name):
                    validation_name = f"method_{method.method_name}"
                    validation_objects, validation_object_types = self._seed_validation_scope(
                        task_name,
                        method_library,
                        representative_args,
                        ltl_spec.objects,
                    )
                    validation_initial_facts = self._method_validation_initial_facts(
                        planner,
                        method,
                        method_library,
                        representative_args,
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
                            task_args=representative_args,
                            root_method=method,
                            allow_empty_plan=not method.subtasks,
                            initial_facts=validation_initial_facts,
                        )
                        method_validation_artifacts.append(
                            {
                                "validation_name": validation_name,
                                "task_name": task_name,
                                "task_args": list(representative_args),
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
                                "task_args": list(representative_args),
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

    @staticmethod
    def _representative_task_args(task_name, method_library, objects, plan_records):
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

        source_objects = list(objects)
        if not source_objects:
            return tuple(f"obj{index + 1}" for index in range(arity))
        if len(source_objects) >= arity:
            return tuple(source_objects[:arity])

        values = list(source_objects)
        while len(values) < arity:
            values.append(source_objects[(len(values)) % len(source_objects)])
        return tuple(values[:arity])

    def _seed_validation_scope(
        self,
        task_name,
        method_library,
        task_args,
        objects,
    ):
        default_type = self._default_object_type()
        object_pool = list(dict.fromkeys(objects or task_args))
        object_types = {
            obj: default_type
            for obj in object_pool
        }

        task_schema = method_library.task_for_name(task_name)
        if task_schema is not None:
            for parameter, arg in zip(task_schema.parameters, task_args):
                object_types[arg] = self._infer_type_from_symbol(parameter)

        return object_pool, object_types

    def _typed_object_entries(self, object_pool, object_types):
        default_type = self._default_object_type()
        return tuple(
            (obj, object_types.get(obj, default_type))
            for obj in object_pool
        )

    def _default_object_type(self):
        if getattr(self.domain, "types", None):
            return self.domain.types[0]
        return "object"

    def _infer_type_from_symbol(self, symbol):
        if not symbol:
            return self._default_object_type()
        base = symbol.rstrip("0123456789").lower()
        domain_types = set(getattr(self.domain, "types", []) or [])
        if base in domain_types:
            return base
        return self._default_object_type()

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
        object_pool = object_pool if object_pool is not None else list(dict.fromkeys(objects or task_args))
        if not object_pool:
            object_pool = list(task_args)
        object_types = object_types if object_types is not None else {
            obj: self._default_object_type()
            for obj in object_pool
        }

        def bind_symbol(symbol):
            if not symbol:
                return symbol
            if symbol[0].islower():
                return symbol
            if symbol not in bindings:
                for candidate in object_pool:
                    if candidate not in bindings.values():
                        bindings[symbol] = candidate
                        break
                else:
                    candidate_type = self._infer_type_from_symbol(symbol)
                    index = 1
                    candidate = f"witness_{candidate_type}_{index}"
                    while candidate in object_pool:
                        index += 1
                        candidate = f"witness_{candidate_type}_{index}"
                    object_pool.append(candidate)
                    object_types[candidate] = candidate_type
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
                candidate = next(
                    (obj for obj in object_pool if obj not in used_values),
                    None,
                )
                if candidate is None:
                    candidate_type = self._infer_type_from_symbol(token)
                    index = 1
                    candidate = f"witness_{candidate_type}_{index}"
                    while candidate in object_pool:
                        index += 1
                        candidate = f"witness_{candidate_type}_{index}"
                    object_pool.append(candidate)
                    object_types[candidate] = candidate_type
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
            result = runner.validate(
                agentspeak_code=asl_code,
                target_literals=method_library.target_literals,
                domain_name=self.domain.name,
                output_dir=self.output_dir,
            )
            summary = {
                "backend": result.backend,
                "status": result.status,
                "java_path": result.java_path,
                "java_version": result.java_version,
                "jason_jar": result.jason_jar,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "transition_count": len(plan_records),
                "target_literal_count": len(method_library.target_literals),
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
