"""
Domain-complete Hierarchical Task Network pipeline with Temporally Extended Goal support.
"""

import copy
import json
import re
import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Any, FrozenSet, List, Optional, Sequence, Set, Tuple

_src_dir = str(Path(__file__).resolve().parent)
if _src_dir in sys.path:
    sys.path.remove(_src_dir)
sys.path.insert(0, _src_dir)

from utils.config import get_config
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
from stage1_interpretation.ltlf_formula import LTLFormula, LogicalOperator, TemporalOperator
from stage2_dfa_generation.dfa_builder import DFABuilder
from stage3_method_synthesis.htn_method_synthesis import HTNMethodSynthesizer
from stage3_method_synthesis.htn_schema import (
    HTNLiteral,
    HTNMethod,
    HTNMethodLibrary,
    HTNTask,
    HTNTargetTaskBinding,
    _parse_signature_literal,
)
from stage3_method_synthesis.task_naming import query_root_alias_task_name, sanitize_identifier
from stage4_panda_planning.panda_planner import PANDAPlanner
from stage5_agentspeak_rendering.agentspeak_renderer import AgentSpeakRenderer
from stage5_agentspeak_rendering.asl_method_lowering import ASLMethodLowering
from stage5_agentspeak_rendering.dfa_runtime import build_agentspeak_transition_specs
from stage6_jason_validation.jason_runner import JasonRunner, JasonValidationError
from pipeline_artifacts import (
    DomainBuildArtifact,
    PlanningRequestContext,
    QueryExecutionContext,
    TemporallyExtendedGoal,
    TemporallyExtendedGoalNode,
    load_domain_build_artifact,
    persist_domain_build_artifact,
    query_bound_method_library,
)
from utils.hddl_condition_parser import HDDLConditionParser
from utils.ipc_plan_verifier import IPCPlanVerifier
from utils.pipeline_logger import PipelineLogger

class TypeResolutionError(RuntimeError):
    """Raised when object/variable type inference is ambiguous or inconsistent."""

class LTL_BDI_Pipeline:
    """
    Domain-complete Hierarchical Task Network pipeline with offline library build.

    Offline:
    Stage 3: Domain-Complete HTN Method Synthesis
    Stage 4: Domain Gate

    Online:
    Stage 1: Natural Language Goal Grounding
    Stage 5: Hierarchical Task Network Solve
    Stage 7: Official IPC HTN Plan Verification
    """

    STAGE4_COMPACT_TASK_ARG_THRESHOLD = 32
    STAGE4_UNORDERED_RUNTIME_TARGET_RECORD_LIMIT = 24
    STAGE4_MAX_VALIDATION_COMPOUND_STEPS = 8
    STAGE6_GUIDED_REPLAY_BEFORE_RUNTIME_TRANSITION_THRESHOLD = 128

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
        self.project_root = project_root
        self.logger = PipelineLogger(logs_dir=str(project_root / "artifacts" / "runs"))

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
        self.domain_actions = self.domain.get_action_names()
        self.domain_predicates = self.domain.get_predicate_signatures()
        self.type_parent_map = self._build_type_parent_map()
        self.domain_type_names = set(self.type_parent_map.keys())
        self.predicate_type_map = self._predicate_type_map()
        self.action_type_map = self._action_type_map()
        self.task_type_map = self._task_type_map()
        self._validate_problem_domain_compatibility()
        self._query_task_action_analysis_cache: Optional[Dict[str, Any]] = None
        self._latest_transition_specs: Tuple[Dict[str, Any], ...] = ()
        self._latest_transition_prompt_analysis: Dict[str, Any] = {}

        # Output directory (set during execution - will use logger's directory)
        self.output_dir = None

    def _record_stage_timing(
        self,
        stage_name: str,
        stage_start: float,
        *,
        breakdown: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger.record_stage_timing(
            stage_name,
            time.perf_counter() - stage_start,
            breakdown=breakdown,
            metadata=metadata,
        )

    @staticmethod
    def _timing_breakdown_without_total(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not profile:
            return {}
        return {
            str(key): value
            for key, value in profile.items()
            if key != "total_seconds" and value is not None
        }

    def execute(self, nl_instruction: str, mode: str = "htn_planner") -> Dict[str, Any]:
        """
        Execute the compatibility wrapper over the refactored two-pipeline flow.

        Args:
            nl_instruction: Natural language instruction
            mode: Execution mode (default: "htn_planner")

        Returns:
            Stage-by-stage execution results and saved artifact metadata
        """
        if mode not in {"htn_planner", "dfa_agentspeak"}:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported modes: 'htn_planner'."
            )

        # Start logger (creates timestamped directory in logs/)
        self.logger.start_pipeline(
            nl_instruction,
            mode=mode,
            domain_file=self.domain_file,
            problem_file=self.problem_file,
            domain_name=self.domain.name,
            problem_name=self.problem.name if self.problem is not None else None,
            output_dir="artifacts/runs",
        )

        # Use logger's directory for all output files
        self.output_dir = self.logger.current_log_dir
        if self.logger.current_record is not None and self.output_dir is not None:
            self.logger.current_record.output_dir = str(self.output_dir)
            self.logger._save_current_state()

        print("="*80)
        print("DOMAIN-COMPLETE HTN PIPELINE")
        print("="*80)
        print(f"\n\"{nl_instruction}\"")
        print("Mode: htn_planner")
        print(f"Output directory: {self.output_dir}")
        print("\n" + "-"*80)

        method_library, stage3_data = self._stage3_domain_method_synthesis()
        if not method_library:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 3", "error": "Domain HTN synthesis failed"}

        stage4_data = self._stage4_domain_gate(method_library)
        if stage4_data is None:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {"success": False, "stage": "Stage 4", "error": "Domain gate failed"}

        artifact = DomainBuildArtifact(
            domain_name=self.domain.name,
            method_library=method_library,
            stage3_metadata=stage3_data,
            stage4_domain_gate=stage4_data,
        )
        artifact_paths = self._persist_domain_build_artifact(artifact)

        query_result = self._execute_query_with_loaded_library(
            nl_instruction,
            artifact,
        )
        if not query_result.get("success", False):
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return query_result

        print("\n" + "="*80)
        stage7_summary = (query_result.get("stage7") or {}).get("summary") or {}
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
            "domain_build": {
                "method_library": method_library.to_dict(),
                "stage3_metadata": stage3_data,
                "stage4_domain_gate": stage4_data,
                "artifact_paths": artifact_paths,
            },
            **query_result,
        }

    def build_domain_library(
        self,
        *,
        output_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build and persist one domain-complete Stage 3/Stage 4 artifact."""

        self.logger.start_pipeline(
            f"Build domain-complete HTN library for {self.domain.name}",
            mode="domain_build",
            domain_file=self.domain_file,
            problem_file=None,
            domain_name=self.domain.name,
            problem_name=None,
            output_dir="artifacts/runs",
        )
        self.output_dir = self.logger.current_log_dir
        if self.logger.current_record is not None and self.output_dir is not None:
            self.logger.current_record.output_dir = str(self.output_dir)
            self.logger._save_current_state()

        method_library, stage3_data = self._stage3_domain_method_synthesis()
        if not method_library:
            log_filepath = self.logger.end_pipeline(success=False)
            return {
                "success": False,
                "stage": "Stage 3",
                "error": "Domain HTN synthesis failed",
                "log_path": str(log_filepath),
            }

        stage4_data = self._stage4_domain_gate(method_library)
        if stage4_data is None:
            log_filepath = self.logger.end_pipeline(success=False)
            return {
                "success": False,
                "stage": "Stage 4",
                "error": "Domain gate failed",
                "log_path": str(log_filepath),
            }

        artifact = DomainBuildArtifact(
            domain_name=self.domain.name,
            method_library=method_library,
            stage3_metadata=stage3_data,
            stage4_domain_gate=stage4_data,
        )
        artifact_paths = self._persist_domain_build_artifact(
            artifact,
            output_root=output_root,
        )
        log_filepath = self.logger.end_pipeline(success=True)
        return {
            "success": True,
            "domain_name": self.domain.name,
            "artifact": artifact.to_dict(),
            "artifact_paths": artifact_paths,
            "log_path": str(log_filepath),
        }

    def execute_query_with_library(
        self,
        nl_query: str,
        *,
        library_artifact,
        mode: str = "htn_planner",
    ) -> Dict[str, Any]:
        """Execute one query against a cached domain-complete library artifact."""

        if mode not in {"htn_planner", "dfa_agentspeak"}:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported modes: 'htn_planner'.",
            )

        self.logger.start_pipeline(
            nl_query,
            mode="query_execution",
            domain_file=self.domain_file,
            problem_file=self.problem_file,
            domain_name=self.domain.name,
            problem_name=self.problem.name if self.problem is not None else None,
            output_dir="artifacts/runs",
        )
        self.output_dir = self.logger.current_log_dir
        if self.logger.current_record is not None and self.output_dir is not None:
            self.logger.current_record.output_dir = str(self.output_dir)
            self.logger._save_current_state()

        result = self._execute_query_with_loaded_library(
            nl_query,
            load_domain_build_artifact(library_artifact),
        )
        log_filepath = self.logger.end_pipeline(success=result.get("success", False))
        result["log_path"] = str(log_filepath)
        return result

    def _stable_domain_artifact_root(self, output_root: Optional[str] = None) -> Path:
        if output_root:
            return Path(output_root).expanduser().resolve()
        if getattr(self.logger, "run_origin", "") == "tests" and self.output_dir is not None:
            return (
                Path(self.output_dir)
                / "domain_build_artifacts"
                / sanitize_identifier(self.domain.name)
            )
        return (
            self.project_root
            / "artifacts"
            / "domain_builds"
            / sanitize_identifier(self.domain.name)
        )

    def _persist_domain_build_artifact(
        self,
        artifact: DomainBuildArtifact,
        *,
        output_root: Optional[str] = None,
    ) -> Dict[str, str]:
        return persist_domain_build_artifact(
            artifact_root=self._stable_domain_artifact_root(output_root),
            artifact=artifact,
        )

    def _execute_query_with_loaded_library(
        self,
        nl_instruction: str,
        artifact: DomainBuildArtifact,
    ) -> Dict[str, Any]:
        """Run the new query-time path with a prebuilt domain-complete method library."""

        domain_library = artifact.method_library
        planning_request = self._stage1_goal_grounding(
            nl_instruction,
            domain_library,
        )
        if planning_request is None:
            return {
                "success": False,
                "stage": "Stage 1",
                "error": "Goal grounding failed",
                "method_library": domain_library.to_dict(),
            }

        stage5_data = self._stage5_hierarchical_planning(
            domain_library,
            planning_request,
        )
        if stage5_data is None:
            return {
                "success": False,
                "stage": "Stage 5",
                "error": "Hierarchical Task Network planning failed",
                "method_library": domain_library.to_dict(),
                "planning_request_context": planning_request.to_dict(),
            }

        stage7_data = self._stage7_official_verification(
            None,
            domain_library,
            stage5_data,
        )
        if stage7_data is None:
            return {
                "success": False,
                "stage": "Stage 7",
                "error": "Official IPC verification failed",
                "method_library": domain_library.to_dict(),
                "planning_request_context": planning_request.to_dict(),
                "stage5": stage5_data,
            }

        return {
            "success": True,
            "method_library": domain_library.to_dict(),
            "temporally_extended_goal": planning_request.temporally_extended_goal.to_dict(),
            "planning_request_context": planning_request.to_dict(),
            "plans": [],
            "stage5": stage5_data,
            "stage7": stage7_data,
        }

    def _stage1_goal_grounding(
        self,
        nl_instruction: str,
        method_library,
    ) -> Optional[PlanningRequestContext]:
        """Stage 1: natural-language grounding into a Temporally Extended Goal."""
        print("\n[STAGE 1] Natural Language -> Goal Grounding")
        print("-"*80)
        stage_start = time.perf_counter()

        try:
            query_text = str(nl_instruction or "").strip()
            if not query_text:
                raise ValueError("Natural-language query is empty.")

            query_object_inventory = self._extract_query_object_inventory(query_text)
            query_task_anchors = self._extract_query_task_anchors(query_text)
            if not query_task_anchors:
                raise ValueError(
                    "No declared domain task invocations were found in the natural-language query.",
                )

            query_task_name_map = self._method_library_source_task_name_map(method_library)
            merged_object_types = self._goal_grounding_object_types(query_object_inventory)
            grounding_inventory = (
                query_object_inventory
                if query_object_inventory
                else self._problem_object_inventory()
            )
            variable_assignments: Dict[str, str] = {}
            diagnostics: List[str] = []
            teg_nodes: List[TemporallyExtendedGoalNode] = []

            for index, anchor in enumerate(query_task_anchors, start=1):
                source_task_name = str(anchor.get("task_name") or "").strip()
                resolved_task_name = query_task_name_map.get(source_task_name, source_task_name)
                raw_args = tuple(
                    str(arg).strip()
                    for arg in (anchor.get("args") or ())
                    if str(arg).strip()
                )
                grounded_args = self._ground_query_task_arguments(
                    task_name=resolved_task_name,
                    task_args=raw_args,
                    query_object_inventory=grounding_inventory,
                    variable_assignments=variable_assignments,
                )
                unresolved_args = [
                    arg
                    for arg in grounded_args
                    if self._is_query_variable_symbol(arg)
                ]
                if unresolved_args:
                    raise ValueError(
                        "Goal grounding could not resolve query variables for "
                        f"{resolved_task_name}{grounded_args}: {unresolved_args}",
                    )
                argument_types = self._task_type_signature(resolved_task_name, method_library)
                self._validate_grounded_task_arguments(
                    task_name=resolved_task_name,
                    grounded_args=grounded_args,
                    argument_types=argument_types,
                    object_types=merged_object_types,
                )
                teg_nodes.append(
                    TemporallyExtendedGoalNode(
                        node_id=f"t{index}",
                        task_name=resolved_task_name,
                        args=grounded_args,
                        argument_types=tuple(argument_types),
                    ),
                )

            precedence_edges = self._goal_grounding_precedence_edges(
                query_text=query_text,
                teg_nodes=tuple(teg_nodes),
            )
            temporally_extended_goal = TemporallyExtendedGoal(
                query_text=query_text,
                nodes=tuple(teg_nodes),
                precedence_edges=precedence_edges,
                query_object_inventory=tuple(query_object_inventory),
                typed_objects=dict(merged_object_types),
                diagnostics=tuple(diagnostics),
            )
            planning_request = self._build_planning_request_context(temporally_extended_goal)

            self.logger.log_stage1_success(
                {
                    "query_text": planning_request.query_text,
                    "temporally_extended_goal": temporally_extended_goal.to_dict(),
                    "task_network": [
                        {
                            "task_name": task_name,
                            "args": list(task_args),
                        }
                        for task_name, task_args in planning_request.task_network
                    ],
                    "task_network_ordered": planning_request.task_network_ordered,
                    "ordering_edges": [
                        {"before": before, "after": after}
                        for before, after in planning_request.ordering_edges
                    ],
                    "problem_objects": list(planning_request.problem_objects),
                    "typed_objects": dict(planning_request.typed_objects),
                    "diagnostics": list(planning_request.diagnostics),
                },
                used_llm=False,
                model=None,
                llm_prompt=None,
                llm_response=None,
            )
            self._record_stage_timing(
                "stage1",
                stage_start,
                breakdown={
                    "goal_grounding_seconds": time.perf_counter() - stage_start,
                },
                metadata={
                    "grounded_task_count": len(teg_nodes),
                    "precedence_edge_count": len(precedence_edges),
                },
            )
            print(f"✓ Grounded tasks: {len(teg_nodes)}")
            print(f"  Ordered edges: {len(precedence_edges)}")
            return planning_request
        except Exception as exc:
            self.logger.log_stage1_error(str(exc))
            self._record_stage_timing("stage1", stage_start)
            print(f"✗ Stage 1 Failed: {exc}")
            return None

    def _problem_object_inventory(self) -> Tuple[Dict[str, Any], ...]:
        if self.problem is None:
            return ()
        grouped: Dict[str, List[str]] = defaultdict(list)
        for object_name, type_name in dict(getattr(self.problem, "object_types", {}) or {}).items():
            grouped[str(type_name).strip() or "object"].append(str(object_name).strip())
        return tuple(
            {
                "type": type_name,
                "label": self._plural_query_type_label(type_name),
                "objects": sorted(
                    object_name
                    for object_name in objects
                    if object_name
                ),
            }
            for type_name, objects in sorted(grouped.items())
        )

    def _goal_grounding_object_types(
        self,
        query_object_inventory: Sequence[Dict[str, Any]],
    ) -> Dict[str, str]:
        query_object_types = self._query_inventory_object_type_map(query_object_inventory)
        if self.problem is None:
            return dict(query_object_types)

        problem_object_types = {
            str(name).strip(): str(type_name).strip()
            for name, type_name in dict(getattr(self.problem, "object_types", {}) or {}).items()
            if str(name).strip() and str(type_name).strip()
        }
        for object_name, query_type in query_object_types.items():
            problem_type = problem_object_types.get(object_name)
            if problem_type is None:
                raise TypeResolutionError(
                    f"Goal grounding references object '{object_name}' that is not declared in "
                    f"problem '{self.problem.name}'.",
                )
            if not (
                query_type == problem_type
                or self._is_subtype(problem_type, query_type)
                or self._is_subtype(query_type, problem_type)
            ):
                raise TypeResolutionError(
                    f"Goal grounding assigned incompatible types '{query_type}' and "
                    f"'{problem_type}' to object '{object_name}'.",
                )
        return problem_object_types

    def _validate_grounded_task_arguments(
        self,
        *,
        task_name: str,
        grounded_args: Sequence[str],
        argument_types: Sequence[str],
        object_types: Dict[str, str],
    ) -> None:
        if argument_types and len(grounded_args) != len(argument_types):
            raise TypeResolutionError(
                f"Grounded task '{task_name}' arity mismatch: "
                f"{len(grounded_args)} args vs {len(argument_types)} signature entries.",
            )
        for index, arg in enumerate(grounded_args):
            object_name = str(arg).strip()
            if not object_name:
                raise TypeResolutionError(
                    f"Grounded task '{task_name}' contains an empty argument.",
                )
            if self.problem is not None and object_name not in set(self.problem.objects or ()):
                raise TypeResolutionError(
                    f"Grounded task '{task_name}' references object '{object_name}' that is "
                    f"absent from problem '{self.problem.name}'.",
                )
            if index >= len(argument_types):
                continue
            actual_type = object_types.get(object_name)
            expected_type = str(argument_types[index] or "").strip()
            if not actual_type or not expected_type:
                continue
            if actual_type == expected_type or self._is_subtype(actual_type, expected_type):
                continue
            raise TypeResolutionError(
                f"Grounded task '{task_name}' argument '{object_name}' has type '{actual_type}', "
                f"expected '{expected_type}'.",
            )

    def _goal_grounding_precedence_edges(
        self,
        *,
        query_text: str,
        teg_nodes: Sequence[TemporallyExtendedGoalNode],
    ) -> Tuple[Tuple[str, str], ...]:
        if len(teg_nodes) <= 1:
            return ()
        if not self._query_requests_ordered_task_sequence(query_text):
            return ()
        return tuple(
            (teg_nodes[index].node_id, teg_nodes[index + 1].node_id)
            for index in range(len(teg_nodes) - 1)
        )

    def _build_planning_request_context(
        self,
        temporally_extended_goal: TemporallyExtendedGoal,
    ) -> PlanningRequestContext:
        task_network = tuple(
            (node.task_name, tuple(node.args))
            for node in temporally_extended_goal.nodes
        )
        problem_objects = tuple(
            str(obj).strip()
            for obj in (self.problem.objects if self.problem is not None else ())
            if str(obj).strip()
        )
        if not problem_objects:
            problem_objects = tuple(
                object_name
                for entry in temporally_extended_goal.query_object_inventory
                for object_name in entry.get("objects", ())
                if str(object_name).strip()
            )
        task_network_ordered = (
            len(task_network) <= 1
            and not temporally_extended_goal.precedence_edges
        )
        return PlanningRequestContext(
            query_text=temporally_extended_goal.query_text,
            temporally_extended_goal=temporally_extended_goal,
            problem_objects=problem_objects,
            typed_objects=dict(temporally_extended_goal.typed_objects),
            task_network=task_network,
            task_network_ordered=task_network_ordered,
            ordering_edges=tuple(temporally_extended_goal.precedence_edges),
            diagnostics=tuple(temporally_extended_goal.diagnostics),
        )

    def _stage5_hierarchical_planning(
        self,
        method_library,
        planning_request: PlanningRequestContext,
    ) -> Optional[Dict[str, Any]]:
        """Stage 5: solve the grounded Temporally Extended Goal with PANDA."""
        print("\n[STAGE 5] Hierarchical Task Network Solve")
        print("-"*80)
        stage_start = time.perf_counter()
        planner = PANDAPlanner(workspace=str(self.output_dir))

        try:
            if not planning_request.task_network:
                raise ValueError("Planning request contains no grounded task network.")

            if not planner.toolchain_available():
                raise ValueError(
                    "PANDA planning toolchain is unavailable on PATH.",
                )

            problem_objects = tuple(planning_request.problem_objects)
            if not problem_objects:
                raise ValueError("Planning request contains no problem objects.")
            typed_objects = self._typed_object_entries(
                problem_objects,
                planning_request.typed_objects,
            )
            initial_facts = (
                tuple(self._render_problem_fact(fact) for fact in (self.problem.init_facts or ()))
                if self.problem is not None
                else ()
            )
            planning_mode = "single_request"
            combined_plan_text: Optional[str] = None
            if self._stage5_should_sequence_temporally_extended_goal(planning_request):
                sequential_result = self._stage5_sequential_ordered_planning(
                    planner=planner,
                    method_library=method_library,
                    planning_request=planning_request,
                    problem_objects=problem_objects,
                    typed_objects=typed_objects,
                    initial_facts=initial_facts,
                )
                action_path = list(sequential_result["action_path"])
                method_trace = list(sequential_result["method_trace"])
                timing_profile = dict(sequential_result["timing_profile"])
                combined_plan_text = sequential_result.get("hierarchical_plan_text")
                planning_mode = str(sequential_result.get("planning_mode") or planning_mode)
                work_dir = None
            else:
                primary_task_name, primary_task_args = planning_request.task_network[0]
                plan = planner.plan(
                    domain=self.domain,
                    method_library=method_library,
                    objects=problem_objects,
                    target_literal=None,
                    task_name=str(primary_task_name),
                    transition_name="teg_request",
                    typed_objects=typed_objects,
                    task_args=tuple(primary_task_args),
                    task_network=planning_request.task_network,
                    task_network_ordered=planning_request.task_network_ordered,
                    ordering_edges=planning_request.ordering_edges,
                    allow_empty_plan=False,
                    initial_facts=initial_facts,
                    timeout_seconds=float(self.config.stage5_planning_timeout),
                )

                action_path = [
                    (
                        f"{step.action_name}({', '.join(step.args)})"
                        if step.args
                        else str(step.action_name)
                    )
                    for step in plan.steps
                ]
                method_trace = planner.extract_method_trace(plan.actual_plan)
                timing_profile = dict(plan.timing_profile or {})
                combined_plan_text = plan.actual_plan or ""
                work_dir = Path(str(plan.work_dir)).resolve() if plan.work_dir else None

            action_path_file = self.output_dir / "stage5_action_path.txt"
            action_path_file.write_text(
                "".join(f"{step}\n" for step in action_path),
            )
            method_trace_file = self.output_dir / "stage5_method_trace.json"
            method_trace_file.write_text(json.dumps(method_trace, indent=2))
            combined_plan_file = None
            if combined_plan_text:
                combined_plan_file = self.output_dir / "stage5_hierarchical_plan.txt"
                combined_plan_file.write_text(str(combined_plan_text))

            artifacts = {
                "backend": "pandaPI",
                "status": "success",
                "planning_mode": planning_mode,
                "task_network": [
                    {
                        "task_name": task_name,
                        "args": list(task_args),
                    }
                    for task_name, task_args in planning_request.task_network
                ],
                "task_network_ordered": planning_request.task_network_ordered,
                "ordering_edges": [
                    {"before": before, "after": after}
                    for before, after in planning_request.ordering_edges
                ],
                "step_count": len(action_path),
                "action_path": action_path,
                "method_trace": method_trace,
                "guided_hierarchical_plan_text": combined_plan_text,
                "guided_hierarchical_plan_source": (
                    "planner_reconstructed_hierarchical_plan"
                    if planning_mode != "single_request"
                    else "panda_plan"
                ),
                "timing_profile": timing_profile,
                "artifacts": {
                    "domain_hddl": str(work_dir / "domain.hddl") if work_dir else None,
                    "problem_hddl": str(work_dir / "problem.hddl") if work_dir else None,
                    "parsed_problem": str(work_dir / "problem.psas") if work_dir else None,
                    "grounded_problem": str(work_dir / "problem.psas.grounded") if work_dir else None,
                    "raw_plan": str(work_dir / "plan.original") if work_dir else None,
                    "actual_plan": str(combined_plan_file) if combined_plan_file else None,
                    "action_path": str(action_path_file),
                    "method_trace": str(method_trace_file),
                },
            }
            summary = {
                "backend": "pandaPI",
                "status": "success",
                "planning_mode": planning_mode,
                "task_count": len(planning_request.task_network),
                "precedence_edge_count": len(planning_request.ordering_edges),
                "step_count": len(action_path),
            }
            self.logger.log_stage5_hierarchical_planning(
                artifacts,
                "Success",
                metadata=summary,
            )
            self._record_stage_timing(
                "stage5",
                stage_start,
                breakdown={
                    key: value
                    for key, value in timing_profile.items()
                    if key != "total_seconds"
                },
                metadata={
                    "task_count": len(planning_request.task_network),
                    "step_count": len(action_path),
                    "planning_mode": planning_mode,
                },
            )
            print(f"✓ Planner returned {len(action_path)} primitive steps")
            print(f"  Task network size: {len(planning_request.task_network)}")
            return {
                "summary": summary,
                "artifacts": artifacts,
            }
        except Exception as exc:
            failure_artifacts = {
                "backend": "pandaPI",
                "status": "failed",
                "task_network": [
                    {
                        "task_name": task_name,
                        "args": list(task_args),
                    }
                    for task_name, task_args in planning_request.task_network
                ],
                "task_network_ordered": planning_request.task_network_ordered,
                "ordering_edges": [
                    {"before": before, "after": after}
                    for before, after in planning_request.ordering_edges
                ],
                "step_count": 0,
            }
            self.logger.log_stage5_hierarchical_planning(
                failure_artifacts,
                "Failed",
                error=str(exc),
                metadata={
                    "backend": "pandaPI",
                    "status": "failed",
                },
            )
            self._record_stage_timing("stage5", stage_start)
            print(f"✗ Stage 5 Failed: {exc}")
            return None

    def _problem_root_task_network_ordering_edges(self) -> Tuple[Tuple[str, str], ...]:
        if self.problem is None:
            return ()
        ordering_edges = tuple(getattr(self.problem, "htn_ordering", ()) or ())
        if not ordering_edges:
            return ()

        label_to_runtime_id: Dict[str, str] = {}
        for index, task in enumerate(self.problem.htn_tasks or (), start=1):
            label = str(getattr(task, "label", None) or f"t{index}").strip()
            if label:
                label_to_runtime_id[label] = f"t{index}"

        resolved_edges: List[Tuple[str, str]] = []
        for before, after in ordering_edges:
            before_id = label_to_runtime_id.get(str(before).strip())
            after_id = label_to_runtime_id.get(str(after).strip())
            if before_id and after_id:
                resolved_edges.append((before_id, after_id))
        return tuple(resolved_edges)

    def _stage5_problem_root_planning(
        self,
        method_library: HTNMethodLibrary,
    ) -> Optional[Dict[str, Any]]:
        """Stage 5 baseline: solve the official problem root task network directly."""
        print("\n[STAGE 5] Hierarchical Task Network Solve")
        print("-"*80)
        stage_start = time.perf_counter()
        planner = PANDAPlanner(workspace=str(self.output_dir))

        try:
            if self.problem is None:
                raise ValueError("Problem root planning requires a loaded problem file.")
            if not self.problem.htn_tasks:
                raise ValueError("Problem file contains no root HTN tasks.")
            if not planner.toolchain_available():
                raise ValueError(
                    "PANDA planning toolchain is unavailable on PATH.",
                )

            problem_objects = tuple(self.problem.objects or ())
            if not problem_objects:
                raise ValueError("Problem file contains no declared objects.")
            task_network = tuple(
                (str(task.task_name), tuple(str(arg) for arg in (task.args or ())))
                for task in (self.problem.htn_tasks or ())
            )
            ordering_edges = self._problem_root_task_network_ordering_edges()
            task_network_ordered = bool(self.problem.htn_ordered) and not ordering_edges
            primary_task_name, primary_task_args = task_network[0]
            plan = planner.plan_hddl_files(
                domain=self.domain,
                domain_file=self.domain_file,
                problem_file=self.problem_file,
                task_name=str(primary_task_name),
                transition_name="official_problem_root",
                task_args=tuple(primary_task_args),
                target_literal=None,
                allow_empty_plan=False,
                timeout_seconds=float(self.config.stage5_planning_timeout),
            )
            action_path = [
                (
                    f"{step.action_name}({', '.join(step.args)})"
                    if step.args
                    else str(step.action_name)
                )
                for step in plan.steps
            ]
            method_trace = planner.extract_method_trace(plan.actual_plan)
            timing_profile = dict(plan.timing_profile or {})
            work_dir = Path(str(plan.work_dir)).resolve() if plan.work_dir else None

            action_path_file = self.output_dir / "stage5_action_path.txt"
            action_path_file.write_text("".join(f"{step}\n" for step in action_path))
            method_trace_file = self.output_dir / "stage5_method_trace.json"
            method_trace_file.write_text(json.dumps(method_trace, indent=2))
            combined_plan_file = self.output_dir / "stage5_hierarchical_plan.txt"
            combined_plan_file.write_text(str(plan.actual_plan or ""))

            artifacts = {
                "backend": "pandaPI",
                "status": "success",
                "planning_mode": "official_problem_root",
                "engine_mode": plan.engine_mode,
                "task_network": [
                    {
                        "task_name": task_name,
                        "args": list(task_args),
                    }
                    for task_name, task_args in task_network
                ],
                "task_network_ordered": task_network_ordered,
                "ordering_edges": [
                    {"before": before, "after": after}
                    for before, after in ordering_edges
                ],
                "step_count": len(action_path),
                "action_path": action_path,
                "method_trace": method_trace,
                "guided_hierarchical_plan_text": plan.actual_plan or "",
                "guided_hierarchical_plan_source": "panda_plan",
                "timing_profile": timing_profile,
                "artifacts": {
                    "domain_hddl": str(work_dir / "domain.hddl") if work_dir else None,
                    "problem_hddl": str(work_dir / "problem.hddl") if work_dir else None,
                    "parsed_problem": str(work_dir / "problem.psas") if work_dir else None,
                    "grounded_problem": str(work_dir / "problem.psas.grounded") if work_dir else None,
                    "raw_plan": str(work_dir / "plan.original") if work_dir else None,
                    "actual_plan": str(combined_plan_file),
                    "action_path": str(action_path_file),
                    "method_trace": str(method_trace_file),
                },
            }
            summary = {
                "backend": "pandaPI",
                "status": "success",
                "planning_mode": "official_problem_root",
                "engine_mode": plan.engine_mode,
                "task_count": len(task_network),
                "precedence_edge_count": len(ordering_edges),
                "step_count": len(action_path),
            }
            self.logger.log_stage5_hierarchical_planning(
                artifacts,
                "Success",
                metadata=summary,
            )
            self._record_stage_timing(
                "stage5",
                stage_start,
                breakdown={
                    key: value
                    for key, value in timing_profile.items()
                    if key != "total_seconds"
                },
                metadata={
                    "task_count": len(task_network),
                    "step_count": len(action_path),
                    "planning_mode": "official_problem_root",
                },
            )
            print(f"✓ Planner returned {len(action_path)} primitive steps")
            print(f"  Problem root task count: {len(task_network)}")
            return {
                "summary": summary,
                "artifacts": artifacts,
            }
        except Exception as exc:
            failure_metadata = dict(getattr(exc, "metadata", {}) or {})
            failure_artifacts = {
                "backend": "pandaPI",
                "status": "failed",
                "planning_mode": "official_problem_root",
                "task_network": [
                    {
                        "task_name": str(task.task_name),
                        "args": list(task.args or ()),
                    }
                    for task in (self.problem.htn_tasks or ())
                ] if self.problem is not None else [],
                "task_network_ordered": bool(getattr(self.problem, "htn_ordered", False)),
                "ordering_edges": [
                    {"before": before, "after": after}
                    for before, after in self._problem_root_task_network_ordering_edges()
                ] if self.problem is not None else [],
                "step_count": 0,
                "failure_metadata": failure_metadata,
            }
            self.logger.log_stage5_hierarchical_planning(
                failure_artifacts,
                "Failed",
                error=str(exc),
                metadata={
                    "backend": "pandaPI",
                    "status": "failed",
                    "planning_mode": "official_problem_root",
                    "engine_mode": failure_metadata.get("engine_mode"),
                },
            )
            self._record_stage_timing("stage5", stage_start)
            print(f"✗ Stage 5 Failed: {exc}")
            return None

    def _stage5_should_sequence_temporally_extended_goal(
        self,
        planning_request: PlanningRequestContext,
    ) -> bool:
        nodes = tuple(planning_request.temporally_extended_goal.nodes or ())
        if len(nodes) <= 1 or len(planning_request.task_network) <= 1:
            return False
        if len(nodes) != len(planning_request.task_network):
            return False
        node_ids = tuple(str(node.node_id or "").strip() for node in nodes)
        if any(not node_id for node_id in node_ids):
            return False
        expected_edges = tuple(
            (node_ids[index], node_ids[index + 1])
            for index in range(len(node_ids) - 1)
        )
        return tuple(planning_request.ordering_edges or ()) == expected_edges

    def _stage5_sequential_ordered_planning(
        self,
        *,
        planner: PANDAPlanner,
        method_library: HTNMethodLibrary,
        planning_request: PlanningRequestContext,
        problem_objects: Sequence[str],
        typed_objects: Sequence[Tuple[str, str]],
        initial_facts: Sequence[str],
    ) -> Dict[str, Any]:
        action_schemas = self._planner_action_schemas()
        current_facts = tuple(str(fact).strip() for fact in (initial_facts or ()) if str(fact).strip())
        aggregated_action_path: List[str] = []
        aggregated_method_trace: List[Dict[str, Any]] = []
        timing_profile: Dict[str, float] = {}

        for index, (task_name, task_args) in enumerate(planning_request.task_network, start=1):
            transition_name = (
                f"teg_request_step_{index:03d}_{sanitize_identifier(str(task_name))}"
            )
            plan = planner.plan(
                domain=self.domain,
                method_library=method_library,
                objects=problem_objects,
                target_literal=None,
                task_name=str(task_name),
                transition_name=transition_name,
                typed_objects=typed_objects,
                task_args=tuple(task_args),
                task_network=((str(task_name), tuple(task_args)),),
                task_network_ordered=True,
                ordering_edges=(),
                allow_empty_plan=False,
                initial_facts=current_facts,
                timeout_seconds=float(self.config.stage5_planning_timeout),
            )
            step_action_path = [
                (
                    f"{step.action_name}({', '.join(step.args)})"
                    if step.args
                    else str(step.action_name)
                )
                for step in plan.steps
            ]
            replay = self._planner_replay_action_path(
                action_path=step_action_path,
                action_schemas=action_schemas,
                seed_facts=current_facts,
            )
            if replay.get("passed") is not True:
                raise ValueError(
                    "Sequential Stage 5 world replay failed after "
                    f"{task_name}{tuple(task_args)}: {replay.get('message')}",
                )
            current_facts = tuple(replay.get("world_facts_hddl") or ())
            aggregated_action_path.extend(step_action_path)
            aggregated_method_trace.extend(planner.extract_method_trace(plan.actual_plan))
            for key, value in dict(plan.timing_profile or {}).items():
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                timing_profile[key] = timing_profile.get(key, 0.0) + numeric_value

        hierarchical_plan_text = self._stage5_render_supported_hierarchical_plan(
            action_path=aggregated_action_path,
            method_library=method_library,
            method_trace=aggregated_method_trace,
        )
        return {
            "planning_mode": "ordered_sequential_node_planning",
            "action_path": aggregated_action_path,
            "method_trace": aggregated_method_trace,
            "hierarchical_plan_text": hierarchical_plan_text,
            "timing_profile": timing_profile,
        }

    def _stage5_render_supported_hierarchical_plan(
        self,
        *,
        action_path: Sequence[str],
        method_library: HTNMethodLibrary,
        method_trace: Sequence[Dict[str, Any]],
    ) -> Optional[str]:
        if self.problem_file is None:
            return None
        verifier = IPCPlanVerifier()
        try:
            return verifier._render_supported_hierarchical_plan(
                domain_file=self.domain_file,
                problem_file=self.problem_file,
                action_path=action_path,
                method_library=method_library,
                method_trace=method_trace,
            )
        except Exception:
            return None

    def _planner_action_schemas(self) -> Sequence[Dict[str, Any]]:
        return tuple(self._stage6_action_schemas())

    def _planner_action_schema_lookup(
        self,
        action_schemas: Sequence[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        for schema in action_schemas or ():
            source_name = str(schema.get("source_name") or "").strip()
            functor = str(schema.get("functor") or "").strip()
            for key in (
                functor,
                source_name,
                self._sanitize_name(source_name),
            ):
                if key:
                    lookup.setdefault(key, schema)
        return lookup

    @staticmethod
    def _planner_parse_action_step(step: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
        text = str(step or "").strip()
        match = re.fullmatch(r"([A-Za-z0-9_-]+)(?:\((.*)\))?", text)
        if match is None:
            return None
        action_name = str(match.group(1) or "").strip()
        args_text = str(match.group(2) or "").strip()
        if not args_text:
            return action_name, ()
        return action_name, tuple(
            part.strip()
            for part in args_text.split(",")
            if part.strip()
        )

    @staticmethod
    def _planner_runtime_token(token: str) -> str:
        value = str(token or "").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value

    def _planner_resolve_runtime_token(
        self,
        token: str,
        bindings: Dict[str, str],
    ) -> str:
        canonical = self._planner_runtime_token(token)
        if canonical in bindings:
            return bindings[canonical]
        if canonical.startswith("?") and canonical[1:] in bindings:
            return bindings[canonical[1:]]
        return canonical

    def _planner_ground_pattern(
        self,
        predicate: str,
        args: Sequence[str],
        bindings: Dict[str, str],
    ) -> str:
        functor = str(predicate or "").strip()
        if not args:
            return functor
        grounded_args = [
            self._planner_runtime_token(self._planner_resolve_runtime_token(arg, bindings))
            for arg in args
        ]
        return f"{functor}({','.join(grounded_args)})"

    def _planner_replay_precondition_clause_holds(
        self,
        clause: Sequence[Dict[str, Any]],
        bindings: Dict[str, str],
        world: Set[str],
    ) -> bool:
        for pattern in clause:
            predicate = str(pattern.get("predicate") or "").strip()
            args = [str(item) for item in (pattern.get("args") or ())]
            is_positive = bool(pattern.get("is_positive", True))
            if predicate == "=" and len(args) == 2:
                left = self._planner_resolve_runtime_token(args[0], bindings)
                right = self._planner_resolve_runtime_token(args[1], bindings)
                if (left == right) != is_positive:
                    return False
                continue
            grounded = self._planner_ground_pattern(predicate, args, bindings)
            holds = grounded in world if is_positive else grounded not in world
            if not holds:
                return False
        return True

    def _planner_hddl_fact_to_atom(self, fact: str) -> Optional[str]:
        parsed = self._parse_positive_hddl_fact(str(fact or ""))
        if parsed is None:
            return None
        predicate, args = parsed
        if not args:
            return predicate
        return f"{predicate}({','.join(args)})"

    def _planner_atom_to_hddl_fact(self, atom: str) -> str:
        text = str(atom or "").strip()
        if not text:
            return "()"
        if "(" not in text:
            return f"({text})"
        functor, remainder = text.split("(", 1)
        functor = functor.strip()
        args_text = remainder[:-1].strip()
        if not args_text:
            return f"({functor})"
        args = [
            self._planner_runtime_token(part.strip())
            for part in args_text.split(",")
            if part.strip()
        ]
        return f"({functor} {' '.join(args)})"

    def _planner_runtime_world_to_hddl_facts(
        self,
        world: Sequence[str],
    ) -> Tuple[str, ...]:
        return tuple(
            self._planner_atom_to_hddl_fact(atom)
            for atom in sorted(str(atom).strip() for atom in (world or ()) if str(atom).strip())
        )

    def _planner_replay_action_path(
        self,
        *,
        action_path: Sequence[str],
        action_schemas: Sequence[Dict[str, Any]],
        seed_facts: Sequence[str],
    ) -> Dict[str, Any]:
        world = {
            atom
            for atom in (self._planner_hddl_fact_to_atom(fact) for fact in seed_facts)
            if atom is not None
        }
        schema_lookup = self._planner_action_schema_lookup(action_schemas)

        for index, step in enumerate(action_path):
            parsed_step = self._planner_parse_action_step(step)
            if parsed_step is None:
                return {
                    "passed": False,
                    "message": f"planner action step #{index + 1} is malformed: {step}",
                }
            action_name, action_args = parsed_step
            schema = (
                schema_lookup.get(action_name)
                or schema_lookup.get(self._sanitize_name(action_name))
            )
            if schema is None:
                return {
                    "passed": False,
                    "message": (
                        f"planner action step #{index + 1} references unknown action "
                        f"'{action_name}'"
                    ),
                }
            parameters = [str(item) for item in (schema.get("parameters") or ())]
            if len(parameters) != len(action_args):
                return {
                    "passed": False,
                    "message": (
                        f"planner action step #{index + 1} has arity {len(action_args)} for "
                        f"'{action_name}', expected {len(parameters)}"
                    ),
                }

            bindings: Dict[str, str] = {}
            for parameter, value in zip(parameters, action_args):
                token = self._planner_runtime_token(parameter)
                bindings[token] = value
                if token.startswith("?"):
                    bindings[token[1:]] = value

            precondition_clauses = list(schema.get("precondition_clauses") or ())
            if not precondition_clauses:
                precondition_clauses = [list(schema.get("preconditions") or ())]
            if not any(
                self._planner_replay_precondition_clause_holds(clause, bindings, world)
                for clause in precondition_clauses
            ):
                return {
                    "passed": False,
                    "message": (
                        f"planner action step #{index + 1} violates schema preconditions "
                        f"for '{action_name}'"
                    ),
                }

            for effect in schema.get("effects") or ():
                predicate = str(effect.get("predicate") or "").strip()
                if not predicate or predicate == "=":
                    continue
                grounded = self._planner_ground_pattern(
                    predicate,
                    effect.get("args") or (),
                    bindings,
                )
                if effect.get("is_positive", True):
                    world.add(grounded)
                else:
                    world.discard(grounded)

        return {
            "passed": True,
            "message": None,
            "world_facts_hddl": self._planner_runtime_world_to_hddl_facts(world),
        }

    def _stage1_parse_nl(self, nl_instruction: str):
        """Stage 1: Natural Language -> LTLf Specification"""
        print("\n[STAGE 1] Natural Language -> LTLf Specification")
        print("-"*80)
        stage_start = time.perf_counter()

        generator = NLToLTLfGenerator(
            api_key=self.config.openai_api_key,
            model=self.config.openai_stage1_model,
            base_url=self.config.openai_base_url,
            domain_file=self.domain_file,  # Pass domain file for dynamic prompt
            request_timeout=self.config.openai_timeout,
            response_max_tokens=self.config.openai_stage1_max_tokens,
        )

        try:
            ltl_spec, prompt_dict, response_text = generator.generate(nl_instruction)
            postprocess_start = time.perf_counter()
            query_object_inventory = self._extract_query_object_inventory(nl_instruction)
            ltl_spec.query_object_inventory = list(query_object_inventory)
            query_task_anchors = self._extract_query_task_anchors(nl_instruction)
            self._normalise_task_grounded_stage1_spec(
                ltl_spec,
                query_task_anchors=query_task_anchors,
                query_text=nl_instruction,
            )
            generator_metadata = dict(getattr(generator, "last_generation_metadata", {}) or {})
            stage1_breakdown = self._timing_breakdown_without_total(
                generator_metadata.get("timing_profile"),
            )
            stage1_breakdown["postprocess_seconds"] = time.perf_counter() - postprocess_start
            self._record_stage_timing(
                "stage1",
                stage_start,
                breakdown=stage1_breakdown,
                metadata={
                    "task_clause_count": generator_metadata.get("task_clause_count"),
                    "prefer_compact_task_grounded_output": generator_metadata.get(
                        "prefer_compact_task_grounded_output",
                    ),
                    "prefer_skeletal_task_grounded_output": generator_metadata.get(
                        "prefer_skeletal_task_grounded_output",
                    ),
                },
            )
            self.logger.log_stage1(
                nl_instruction,
                ltl_spec,
                "Success",
                model=self.config.openai_stage1_model,
                llm_prompt=prompt_dict,
                llm_response=response_text
            )

            formulas_string = [f.to_string() for f in ltl_spec.formulas]
            print(f"✓ LTLf Formula: {formulas_string}")
            print(f"  Objects: {ltl_spec.objects}")
            print("  (Stage 1 only captures goal semantics; Stage 4 instantiates a concrete HDDL problem)")

            return ltl_spec

        except Exception as e:
            generator_metadata = dict(getattr(generator, "last_generation_metadata", {}) or {})
            self._record_stage_timing(
                "stage1",
                stage_start,
                breakdown=self._timing_breakdown_without_total(
                    generator_metadata.get("timing_profile"),
                ),
                metadata={
                    "failure_stage": generator_metadata.get("failure_stage"),
                },
            )
            self.logger.log_stage1(
                nl_instruction,
                None,
                "Failed",
                str(e),
                model=self.config.openai_stage1_model,
            )
            print(f"✗ Stage 1 Failed: {e}")
            return None

    def _stage2_dfa_generation(self, ltl_spec):
        """Stage 2: LTLf -> DFA Generation"""
        print("\n[STAGE 2] DFA Generation")
        print("-"*80)
        stage_start = time.perf_counter()

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
            print(f"\n  Raw DFA:")
            print(f"    States: {dfa_result['num_states']}")
            print(f"    Transitions: {dfa_result['num_transitions']}")
            if dfa_result.get("dfa_dot"):
                print(f"    Saved to: {self.output_dir / 'dfa.dot'}")
            elif dfa_result.get("symbolic_query_step_monitor"):
                print("    Representation: exact symbolic unordered query-step conjunction monitor")

            # Save complete DFA result to JSON
            output_file = self.output_dir / "dfa.json"
            import json
            # Remove the actual DOT strings from JSON to keep it readable
            # (DOT files are saved separately)
            persist_start = time.perf_counter()
            json_data = {k: v for k, v in dfa_result.items() if k != 'dfa_dot'}
            output_file.write_text(json.dumps(json_data, indent=2))
            stage2_breakdown = self._timing_breakdown_without_total(
                dfa_result.get("timing_profile"),
            )
            stage2_breakdown["persist_metadata_seconds"] = (
                time.perf_counter() - persist_start
            )
            self._record_stage_timing(
                "stage2",
                stage_start,
                breakdown=stage2_breakdown,
                metadata={
                    "construction": dfa_result.get("construction"),
                    "num_states": dfa_result.get("num_states"),
                    "num_transitions": dfa_result.get("num_transitions"),
                },
            )
            print(f"\n  Metadata saved to: {output_file}")

            return dfa_result

        except Exception as e:
            self._record_stage_timing("stage2", stage_start)
            self.logger.log_stage2_dfas(ltl_spec, None, "Failed", str(e))
            print(f"✗ Stage 2 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _stage3_method_synthesis(self, ltl_spec):
        """Stage 3: query-task literals -> domain HTN method synthesis."""
        print("\n[STAGE 3] HTN Method Synthesis")
        print("-"*80)
        stage_start = time.perf_counter()

        try:
            ordered_literal_signatures = self._ordered_literal_signatures_from_spec(ltl_spec)
            query_text = getattr(ltl_spec, "source_instruction", "")
            query_task_anchors = self._extract_query_task_anchors(query_text)
            semantic_objects = tuple(getattr(ltl_spec, "objects", ()) or ())
            query_object_inventory = tuple(
                getattr(ltl_spec, "query_object_inventory", ()) or (),
            )
            if not query_object_inventory:
                query_object_inventory = self._extract_query_object_inventory(query_text)
            query_objects = tuple(
                object_name
                for entry in query_object_inventory
                for object_name in entry.get("objects", ())
            )
            synthesizer = HTNMethodSynthesizer(
                api_key=self.config.openai_api_key,
                model=self.config.openai_stage3_model,
                base_url=self.config.openai_base_url,
                timeout=float(self.config.openai_stage3_timeout),
                max_tokens=int(self.config.openai_stage3_max_tokens),
            )

            synthesis_start = time.perf_counter()
            method_library, synthesis_meta = synthesizer.synthesize(
                domain=self.domain,
                query_text=query_text,
                query_task_anchors=query_task_anchors,
                semantic_objects=semantic_objects,
                query_object_inventory=query_object_inventory,
                query_objects=query_objects,
                negation_hints=getattr(ltl_spec, "negation_hints", {}),
                ordered_literal_signatures=ordered_literal_signatures,
            )
            synthesis_seconds = time.perf_counter() - synthesis_start
            validation_start = time.perf_counter()
            self._validate_method_library_typing(method_library)
            typing_validation_seconds = time.perf_counter() - validation_start
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
                "query_task_anchors": synthesis_meta.get("query_task_anchors", []),
                "semantic_objects": synthesis_meta.get("semantic_objects", []),
                "query_object_inventory": synthesis_meta.get("query_object_inventory", []),
                "query_objects": synthesis_meta.get("query_objects", []),
                "negation_resolution": synthesis_meta.get("negation_resolution", {}),
                "action_analysis": synthesis_meta.get("action_analysis", {}),
                "derived_analysis": synthesis_meta.get("derived_analysis", {}),
                "failure_class": synthesis_meta.get("failure_class"),
                "compound_tasks": synthesis_meta["compound_tasks"],
                "primitive_tasks": synthesis_meta["primitive_tasks"],
                "methods": synthesis_meta["methods"],
            }
            self._latest_transition_prompt_analysis = dict(
                synthesis_meta.get("derived_analysis", {}) or {},
            )
            self._record_stage_timing(
                "stage3",
                stage_start,
                breakdown={
                    "synthesis_seconds": synthesis_seconds,
                    "typing_validation_seconds": typing_validation_seconds,
                    "llm_response_seconds": synthesis_meta.get("llm_response_time_seconds"),
                },
                metadata={
                    "used_llm": synthesis_meta.get("used_llm"),
                    "llm_attempted": synthesis_meta.get("llm_prompt") is not None,
                },
            )

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
            self._latest_transition_specs = ()
            self._latest_transition_prompt_analysis = {}
            self._record_stage_timing("stage3", stage_start)
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
        explicit_signatures = tuple(
            str(signature).strip()
            for signature in getattr(ltl_spec, "query_task_literal_signatures", ()) or ()
            if str(signature).strip()
        )
        if explicit_signatures:
            return explicit_signatures
        ordered: List[str] = []
        for formula in getattr(ltl_spec, "formulas", []) or []:
            formula_literals = cls._ordered_literal_signatures_from_formula(formula)
            if not formula_literals:
                return ()
            ordered.extend(formula_literals)
        return tuple(ordered)

    @staticmethod
    def _query_task_sequence_is_ordered(ltl_spec) -> bool:
        explicit = getattr(ltl_spec, "query_task_sequence_is_ordered", None)
        if explicit is None:
            return False
        return bool(explicit)

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

    def _extract_query_task_anchors(self, query_text: str) -> Tuple[Dict[str, Any], ...]:
        if not query_text.strip():
            return ()

        declared_task_names = [
            task.name
            for task in getattr(self.domain, "tasks", [])
            if getattr(task, "name", None)
        ]
        if not declared_task_names:
            return ()

        pattern = re.compile(
            r"(?P<task_name>"
            + "|".join(
                re.escape(task_name)
                for task_name in sorted(declared_task_names, key=len, reverse=True)
            )
            + r")\((?P<args>[^()]*)\)",
        )
        anchors: List[Dict[str, Any]] = []
        for match in pattern.finditer(query_text):
            args_text = match.group("args").strip()
            args = [
                part.strip()
                for part in args_text.split(",")
                if part.strip()
            ]
            anchors.append({
                "task_name": match.group("task_name"),
                "args": args,
            })
        return tuple(anchors)

    def _normalise_task_grounded_stage1_spec(
        self,
        ltl_spec,
        *,
        query_task_anchors: Sequence[Dict[str, Any]],
        query_text: str = "",
    ) -> None:
        """
        Collapse over-eager support bundles in task-grounded eventual clauses.

        Stage 1 is allowed to infer predicate-level intent from declared tasks and
        primitive actions, but benchmark queries explicitly mention declared task
        invocations. When an eventual clause for one such task becomes a conjunction
        of multiple positive atoms, only the same-arity headline literal should stay
        as the top-level obligation; auxiliary support literals belong inside the
        task's decomposition rather than as separate target bindings.
        """
        anchors = tuple(query_task_anchors or ())
        if not anchors:
            return
        query_object_inventory = tuple(
            getattr(ltl_spec, "query_object_inventory", ()) or (),
        )

        eventual_clauses: List[Any] = []
        for formula in getattr(ltl_spec, "formulas", ()) or ():
            self._collect_eventual_task_clauses(formula, eventual_clauses)

        changed = False
        if len(eventual_clauses) == len(anchors):
            for anchor, eventual_clause in zip(anchors, eventual_clauses):
                replacement = self._select_query_task_headline_formula(
                    anchor=anchor,
                    eventual_clause=eventual_clause,
                    query_object_inventory=query_object_inventory,
                )
                if replacement is None:
                    continue
                current_children = list(getattr(eventual_clause, "sub_formulas", ()) or ())
                if len(current_children) != 1 or current_children[0] is replacement:
                    continue
                eventual_clause.sub_formulas = [replacement]
                changed = True

        if changed:
            ltl_spec.grounding_map = NLToLTLfGenerator()._create_grounding_map(ltl_spec)

        canonical_signatures, canonical_formulas = self._canonical_task_grounded_formulas(
            anchors,
            query_object_inventory=query_object_inventory,
            ordered_sequence=self._query_requests_ordered_task_sequence(query_text),
        )
        if canonical_signatures:
            ltl_spec.query_task_sequence_is_ordered = self._query_requests_ordered_task_sequence(
                query_text,
            )
            ltl_spec.query_task_literal_signatures = list(canonical_signatures)
            current_signatures = self._ordered_literal_signatures_from_spec(ltl_spec)
            current_formula_strings = tuple(
                formula.to_string()
                for formula in getattr(ltl_spec, "formulas", ()) or ()
            )
            canonical_formula_strings = tuple(
                formula.to_string()
                for formula in canonical_formulas
            )
            if (
                current_signatures != canonical_signatures
                or current_formula_strings != canonical_formula_strings
            ):
                ltl_spec.formulas = canonical_formulas
                ltl_spec.grounding_map = NLToLTLfGenerator()._create_grounding_map(ltl_spec)

        canonical_objects = self._canonical_task_grounded_semantic_objects(
            ltl_spec,
            query_object_inventory=query_object_inventory,
        )
        if canonical_objects:
            ltl_spec.objects = list(canonical_objects)

    def _collect_eventual_task_clauses(
        self,
        formula,
        clauses: List[Any],
    ) -> None:
        operator = getattr(getattr(formula, "operator", None), "value", None)
        if operator == "F" and len(getattr(formula, "sub_formulas", ()) or ()) == 1:
            clauses.append(formula)
            return

        for child in getattr(formula, "sub_formulas", ()) or ():
            self._collect_eventual_task_clauses(child, clauses)

    def _select_query_task_headline_formula(
        self,
        *,
        anchor: Dict[str, Any],
        eventual_clause,
        query_object_inventory: Sequence[Dict[str, Any]],
    ):
        children = tuple(getattr(eventual_clause, "sub_formulas", ()) or ())
        if len(children) != 1:
            return None

        task_name = str(anchor.get("task_name", "")).strip()
        task_args = tuple(str(arg).strip() for arg in (anchor.get("args") or ()))
        if not task_name:
            return None
        if any(self._is_query_variable_symbol(arg) for arg in task_args):
            return None

        preferred_predicate = self._preferred_query_task_headline_predicate(
            task_name=task_name,
            task_args=task_args,
        )
        if preferred_predicate:
            canonical_formula = self._build_atomic_formula_like(
                eventual_clause,
                predicate_name=preferred_predicate,
                args=task_args,
            )
        else:
            canonical_formula = None

        atomic_child = self._positive_atomic_formula(children[0])
        if atomic_child is not None:
            if canonical_formula is None:
                return None
            child_predicate = next(iter(atomic_child.predicate.keys()), "")
            child_args = tuple(atomic_child.predicate.get(child_predicate) or ())
            if child_predicate == preferred_predicate and child_args == task_args:
                return None
            return canonical_formula

        atomic_conjuncts = self._positive_atomic_conjuncts(children[0])
        if len(atomic_conjuncts) < 2:
            return None

        task_schema = next(
            (
                task
                for task in getattr(self.domain, "tasks", [])
                if getattr(task, "name", None) == task_name
            ),
            None,
        )
        explicit_predicates = {
            str(predicate_name).strip()
            for predicate_name in (getattr(task_schema, "source_predicates", ()) or ())
            if str(predicate_name).strip()
        }
        task_arity = len(task_args)
        task_signature = self.task_type_map.get(task_name, ())
        task_tokens = self._name_tokens(task_name)
        task_compact = "".join(task_tokens)

        scored_candidates: List[Tuple[int, str, Any]] = []
        for atom_formula in atomic_conjuncts:
            predicate = getattr(atom_formula, "predicate", None)
            if not isinstance(predicate, dict):
                continue
            predicate_name = next(iter(predicate.keys()), "")
            predicate_args = tuple(predicate.get(predicate_name) or ())
            if len(predicate_args) != task_arity:
                continue

            score = 0
            if predicate_name in explicit_predicates:
                score += 100
            score += self._query_task_predicate_signature_score(
                predicate_name=predicate_name,
                task_signature=task_signature,
            )

            predicate_tokens = self._name_tokens(predicate_name)
            predicate_compact = "".join(predicate_tokens)
            score += 10 * len(set(task_tokens) & set(predicate_tokens))
            if task_compact and predicate_compact:
                if task_compact == predicate_compact:
                    score += 50
                elif (
                    task_compact.endswith(predicate_compact)
                    or predicate_compact.endswith(task_compact)
                ):
                    score += 30
                elif predicate_compact in task_compact or task_compact in predicate_compact:
                    score += 10

            scored_candidates.append((score, predicate_name, atom_formula))

        if not scored_candidates:
            return canonical_formula

        scored_candidates.sort(key=lambda item: (-item[0], item[1]))
        if len(scored_candidates) == 1:
            return scored_candidates[0][2]

        best_score = scored_candidates[0][0]
        best_predicates = {
            predicate_name
            for score, predicate_name, _ in scored_candidates
            if score == best_score
        }
        if len(best_predicates) != 1 and best_score <= 0:
            return canonical_formula
        if len(best_predicates) != 1:
            return canonical_formula
        return scored_candidates[0][2]

    @classmethod
    def _positive_atomic_conjuncts(cls, formula) -> Tuple[Any, ...]:
        predicate = getattr(formula, "predicate", None)
        operator = getattr(formula, "operator", None)
        logical_op = getattr(formula, "logical_op", None)
        sub_formulas = tuple(getattr(formula, "sub_formulas", ()) or ())

        if predicate is not None and operator is None and logical_op is None:
            return (formula,) if isinstance(predicate, dict) else ()

        logical_name = getattr(logical_op, "value", None)
        if logical_name != "and":
            return ()

        flattened: List[Any] = []
        for child in sub_formulas:
            child_atoms = cls._positive_atomic_conjuncts(child)
            if not child_atoms:
                return ()
            flattened.extend(child_atoms)
        return tuple(flattened)

    @staticmethod
    def _positive_atomic_formula(formula):
        predicate = getattr(formula, "predicate", None)
        operator = getattr(formula, "operator", None)
        logical_op = getattr(formula, "logical_op", None)
        if predicate is None or operator is not None or logical_op is not None:
            return None
        return formula if isinstance(predicate, dict) else None

    def _canonical_task_grounded_semantic_objects(
        self,
        ltl_spec,
        *,
        query_object_inventory: Sequence[Dict[str, Any]],
    ) -> Tuple[str, ...]:
        """
        Prefer grounded query objects over free-form Stage 1 placeholder tokens.

        Task-grounded benchmark queries already enumerate the relevant object
        inventory in natural language. When the Stage 1 response echoes schema-like
        placeholders such as `ROVER` inside `objects`, those placeholders should
        not leak into Stage 5 or Stage 6. Rebuild the semantic object list from the
        query inventory first, then add any extra grounded constants still present
        in the canonicalised formulas.
        """
        ordered_objects: List[str] = []
        seen: set[str] = set()

        def add_object(value: object) -> None:
            if not isinstance(value, str):
                return
            token = value.strip()
            if not token or token.startswith("?"):
                return
            if token in seen:
                return
            seen.add(token)
            ordered_objects.append(token)

        for entry in query_object_inventory or ():
            for object_name in entry.get("objects", ()) or ():
                add_object(object_name)

        formula_stack = list(reversed(list(getattr(ltl_spec, "formulas", ()) or ())))
        while formula_stack:
            formula = formula_stack.pop()
            predicate = getattr(formula, "predicate", None)
            if isinstance(predicate, dict):
                special_keys = {
                    "type",
                    "formulas",
                    "left_formula",
                    "right_formula",
                    "formula",
                    "operator",
                }
                if all(key not in special_keys for key in predicate.keys()):
                    for args in predicate.values():
                        if isinstance(args, list):
                            for arg in args:
                                add_object(arg)
            formula_stack.extend(
                reversed(list(getattr(formula, "sub_formulas", ()) or ())),
            )

        return tuple(ordered_objects)

    @staticmethod
    def _build_atomic_formula_like(
        template_formula,
        *,
        predicate_name: str,
        args: Sequence[str],
    ):
        return template_formula.__class__(
            operator=None,
            predicate={predicate_name: list(args)},
            sub_formulas=[],
            logical_op=None,
        )

    @staticmethod
    def _name_tokens(name: str) -> Tuple[str, ...]:
        tokens = re.findall(r"[a-z0-9]+", str(name or "").replace("-", "_").lower())
        return tuple(token for token in tokens if token)

    def _preferred_query_task_headline_predicate(
        self,
        *,
        task_name: str,
        task_args: Sequence[str],
    ) -> Optional[str]:
        task_schema = next(
            (
                task
                for task in getattr(self.domain, "tasks", [])
                if getattr(task, "name", None) == task_name
            ),
            None,
        )
        task_arity = len(tuple(task_args))
        explicit_candidates = [
            str(predicate_name).strip()
            for predicate_name in (getattr(task_schema, "source_predicates", ()) or ())
            if str(predicate_name).strip()
            and len(self.predicate_type_map.get(str(predicate_name).strip(), ())) == task_arity
        ]
        if explicit_candidates:
            return explicit_candidates[0]

        action_analysis = self._query_task_action_analysis()
        task_tokens = self._name_tokens(task_name)
        task_compact = "".join(task_tokens)
        task_signature = self.task_type_map.get(task_name, ())
        scored: List[Tuple[int, str]] = []
        for predicate_name, patterns in action_analysis.get("producer_patterns_by_predicate", {}).items():
            if not any(len(pattern.get("effect_args") or ()) == task_arity for pattern in patterns):
                continue
            predicate_tokens = self._name_tokens(predicate_name)
            predicate_compact = "".join(predicate_tokens)
            score = self._query_task_predicate_signature_score(
                predicate_name=predicate_name,
                task_signature=task_signature,
            )
            score += 10 * len(set(task_tokens) & set(predicate_tokens))
            if task_compact and predicate_compact:
                if task_compact == predicate_compact:
                    score += 50
                elif (
                    task_compact.endswith(predicate_compact)
                    or predicate_compact.endswith(task_compact)
                ):
                    score += 30
                elif predicate_compact in task_compact or task_compact in predicate_compact:
                    score += 10
            for pattern in patterns:
                score += 5 * len(
                    set(task_tokens) & set(self._name_tokens(str(pattern.get("action_name") or ""))),
                )
            if score <= 0:
                continue
            scored.append((score, predicate_name))

        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score = scored[0][0]
        best_predicates = [predicate for score, predicate in scored if score == best_score]
        if len(best_predicates) != 1:
            return None
        return best_predicates[0]

    def _query_task_predicate_signature_score(
        self,
        *,
        predicate_name: str,
        task_signature: Sequence[str],
    ) -> int:
        predicate_signature = self.predicate_type_map.get(predicate_name, ())
        if not task_signature or len(predicate_signature) != len(task_signature):
            return 0

        score = 0
        for predicate_type, task_type in zip(predicate_signature, task_signature):
            if predicate_type == task_type:
                score += 100
                continue
            if self._is_subtype(predicate_type, task_type) or self._is_subtype(task_type, predicate_type):
                score += 25
                continue
            return -1000
        return score

    def _canonical_task_grounded_formulas(
        self,
        query_task_anchors: Sequence[Dict[str, Any]],
        *,
        query_object_inventory: Sequence[Dict[str, Any]],
        ordered_sequence: bool = False,
    ) -> Tuple[Tuple[str, ...], List[LTLFormula]]:
        canonical_signatures: List[str] = []
        query_step_formulas: List[LTLFormula] = []
        variable_assignments: Dict[str, str] = {}
        for index, anchor in enumerate(query_task_anchors, start=1):
            task_name = str(anchor.get("task_name", "")).strip()
            raw_task_args = tuple(str(arg).strip() for arg in (anchor.get("args") or ()))
            if not task_name:
                return (), []
            task_args = self._ground_query_task_arguments(
                task_name=task_name,
                task_args=raw_task_args,
                query_object_inventory=query_object_inventory,
                variable_assignments=variable_assignments,
            )
            if any(self._is_query_variable_symbol(arg) for arg in task_args):
                return (), []
            predicate_name = self._preferred_query_task_headline_predicate(
                task_name=task_name,
                task_args=task_args,
            )
            if not predicate_name:
                return (), []
            signature = self._literal_signature(predicate_name, task_args, True)
            canonical_signatures.append(signature)
            query_step_formulas.append(
                self._query_step_formula(index),
            )
        if not query_step_formulas:
            return tuple(canonical_signatures), []
        if ordered_sequence:
            return (
                tuple(canonical_signatures),
                self._ordered_query_step_formulas(query_step_formulas),
            )
        return (
            tuple(canonical_signatures),
            [
                LTLFormula(
                    operator=TemporalOperator.FINALLY,
                    predicate=None,
                    sub_formulas=[formula],
                    logical_op=None,
                )
                for formula in query_step_formulas
            ],
        )

    @staticmethod
    def _query_requests_ordered_task_sequence(query_text: str) -> bool:
        lowered = str(query_text or "").lower()
        return " then " in lowered or ", then " in lowered

    def _ordered_eventual_formula(
        self,
        atomic_formulas: Sequence[LTLFormula],
    ) -> LTLFormula:
        current = LTLFormula(
            operator=TemporalOperator.FINALLY,
            predicate=None,
            sub_formulas=[atomic_formulas[-1]],
            logical_op=None,
        )
        for atomic_formula in reversed(tuple(atomic_formulas[:-1])):
            current = LTLFormula(
                operator=TemporalOperator.FINALLY,
                predicate=None,
                sub_formulas=[
                    LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=[atomic_formula, current],
                        logical_op=LogicalOperator.AND,
                    ),
                ],
                logical_op=None,
            )
        return current

    @staticmethod
    def _query_step_formula(index: int) -> LTLFormula:
        return LTLFormula(
            operator=None,
            predicate={f"query_step_{index}": []},
            sub_formulas=[],
            logical_op=None,
        )

    def _ordered_query_step_formulas(
        self,
        query_step_formulas: Sequence[LTLFormula],
    ) -> List[LTLFormula]:
        if not query_step_formulas:
            return []
        if len(query_step_formulas) == 1:
            return [
                LTLFormula(
                    operator=TemporalOperator.FINALLY,
                    predicate=None,
                    sub_formulas=[query_step_formulas[0]],
                    logical_op=None,
                ),
            ]

        formulas: List[LTLFormula] = []
        for current_step, next_step in zip(
            query_step_formulas,
            query_step_formulas[1:],
        ):
            formulas.append(
                LTLFormula(
                    operator=TemporalOperator.UNTIL,
                    predicate=None,
                    sub_formulas=[
                        LTLFormula(
                            operator=None,
                            predicate=None,
                            sub_formulas=[copy.deepcopy(next_step)],
                            logical_op=LogicalOperator.NOT,
                        ),
                        copy.deepcopy(current_step),
                    ],
                    logical_op=None,
                ),
            )
        formulas.append(
            LTLFormula(
                operator=TemporalOperator.FINALLY,
                predicate=None,
                sub_formulas=[copy.deepcopy(query_step_formulas[-1])],
                logical_op=None,
            ),
        )
        return formulas

    @staticmethod
    def _conjoin_formulas(formulas: Sequence[LTLFormula]) -> LTLFormula:
        if len(formulas) == 1:
            return formulas[0]
        return LTLFormula(
            operator=None,
            predicate=None,
            sub_formulas=list(formulas),
            logical_op=LogicalOperator.AND,
        )

    @staticmethod
    def _is_query_variable_symbol(symbol: str) -> bool:
        return str(symbol or "").strip().startswith("?")

    def _ground_query_task_arguments(
        self,
        *,
        task_name: str,
        task_args: Sequence[str],
        query_object_inventory: Sequence[Dict[str, Any]],
        variable_assignments: Dict[str, str],
    ) -> Tuple[str, ...]:
        task_signature = tuple(self.task_type_map.get(task_name, ()))
        grounded_args: List[str] = []
        for index, arg in enumerate(task_args):
            token = str(arg).strip()
            if not self._is_query_variable_symbol(token):
                grounded_args.append(token)
                continue
            if token in variable_assignments:
                grounded_args.append(variable_assignments[token])
                continue
            expected_type = task_signature[index] if index < len(task_signature) else "object"
            candidates = self._query_inventory_candidates_for_type(
                expected_type=expected_type,
                query_object_inventory=query_object_inventory,
            )
            if not candidates:
                grounded_args.append(token)
                continue
            variable_assignments[token] = candidates[0]
            grounded_args.append(candidates[0])
        return tuple(grounded_args)

    def _query_inventory_candidates_for_type(
        self,
        *,
        expected_type: str,
        query_object_inventory: Sequence[Dict[str, Any]],
    ) -> Tuple[str, ...]:
        required_type = str(expected_type or "object").strip() or "object"
        candidates: List[str] = []
        seen: Set[str] = set()
        for entry in query_object_inventory:
            entry_type = str(entry.get("type") or "").strip() or "object"
            if required_type != "object" and not self._is_subtype(entry_type, required_type):
                continue
            for obj in entry.get("objects") or ():
                object_name = str(obj).strip()
                if not object_name or object_name in seen:
                    continue
                seen.add(object_name)
                candidates.append(object_name)
        return tuple(candidates)

    def _query_inventory_object_type_map(
        self,
        query_object_inventory: Sequence[Dict[str, Any]],
    ) -> Dict[str, str]:
        object_types: Dict[str, str] = {}
        for entry in query_object_inventory:
            entry_type = str(entry.get("type") or "").strip() or "object"
            if entry_type not in self.domain_type_names:
                raise TypeResolutionError(
                    f"Query object inventory references unknown type '{entry_type}'.",
                )
            for obj in entry.get("objects") or ():
                object_name = str(obj).strip()
                if not object_name:
                    continue
                existing_type = object_types.get(object_name)
                if existing_type is None:
                    object_types[object_name] = entry_type
                    continue
                if entry_type == existing_type:
                    continue
                if self._is_subtype(entry_type, existing_type):
                    object_types[object_name] = entry_type
                    continue
                if self._is_subtype(existing_type, entry_type):
                    continue
                raise TypeResolutionError(
                    "Query object inventory assigns incompatible types "
                    f"'{existing_type}' and '{entry_type}' to object '{object_name}'.",
                )
        return object_types

    def _query_task_action_analysis(self) -> Dict[str, Any]:
        if self._query_task_action_analysis_cache is None:
            parser = HDDLConditionParser()
            producer_patterns_by_predicate: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for action in getattr(self.domain, "actions", []) or ():
                semantics = parser.parse_action(action)
                for effect in getattr(semantics, "effects", ()) or ():
                    if not getattr(effect, "is_positive", True):
                        continue
                    producer_patterns_by_predicate[effect.predicate].append(
                        {
                            "action_name": str(getattr(action, "name", "")).strip(),
                            "effect_args": list(getattr(effect, "args", ()) or ()),
                        },
                    )
            self._query_task_action_analysis_cache = {
                "producer_patterns_by_predicate": dict(producer_patterns_by_predicate),
            }
        return self._query_task_action_analysis_cache

    @staticmethod
    def _plural_query_type_label(type_name: str) -> str:
        text = str(type_name).strip()
        if not text:
            return "objects"
        return text if text.endswith("s") else f"{text}s"

    def _extract_query_object_inventory(
        self,
        query_text: str,
    ) -> Tuple[Dict[str, Any], ...]:
        text = (query_text or "").strip()
        if not text.lower().startswith("using "):
            return ()

        complete_marker = re.search(r",?\s+complete the tasks\s+", text, re.IGNORECASE)
        if complete_marker is None:
            return ()

        inventory_text = text[len("Using "):complete_marker.start()].strip().rstrip(",")
        if not inventory_text or inventory_text == "the task arguments":
            return ()

        label_to_type: Dict[str, str] = {}
        for type_name in sorted(self.domain_type_names, key=len, reverse=True):
            singular = str(type_name).strip()
            if not singular:
                continue
            plural = self._plural_query_type_label(singular)
            label_to_type.setdefault(plural, singular)
            label_to_type.setdefault(singular, singular)
        if not label_to_type:
            return ()

        type_labels = sorted(label_to_type, key=len, reverse=True)
        label_pattern = "|".join(re.escape(label) for label in type_labels)
        group_pattern = re.compile(
            rf"(?P<prefix>^|,\s+|,\s+and\s+| and\s+)"
            rf"(?P<label>{label_pattern})\s+"
            rf"(?P<objects>.*?)(?=(?:,\s+|,\s+and\s+| and\s+)(?:{label_pattern})\s+|$)",
        )
        inventory: List[Dict[str, Any]] = []
        for match in group_pattern.finditer(inventory_text):
            label = match.group("label").strip()
            object_text = match.group("objects").strip().rstrip(",")
            if not object_text:
                continue
            normalised_object_text = re.sub(r",\s+and\s+", ", ", object_text)
            normalised_object_text = re.sub(r"\s+and\s+", ", ", normalised_object_text)
            object_names = [
                item.strip()
                for item in normalised_object_text.split(",")
                if item.strip()
            ]
            if not object_names:
                continue
            inventory.append(
                {
                    "type": label_to_type[label],
                    "label": label,
                    "objects": object_names,
                },
            )
        return tuple(inventory)

    def _stage3_domain_method_synthesis(self):
        """Stage 3: domain-only HTN method synthesis over declared compound tasks."""
        print("\n[STAGE 3] Domain-Complete HTN Method Synthesis")
        print("-"*80)
        stage_start = time.perf_counter()

        try:
            synthesizer = HTNMethodSynthesizer(
                api_key=self.config.openai_api_key,
                model=self.config.openai_stage3_model,
                base_url=self.config.openai_base_url,
                timeout=float(self.config.openai_stage3_timeout),
                max_tokens=int(self.config.openai_stage3_max_tokens),
            )
            synthesis_start = time.perf_counter()
            method_library, synthesis_meta = synthesizer.synthesize_domain_complete(
                domain=self.domain,
            )
            synthesis_seconds = time.perf_counter() - synthesis_start
            validation_start = time.perf_counter()
            self._validate_method_library_typing(method_library)
            typing_validation_seconds = time.perf_counter() - validation_start
            summary = {
                "used_llm": synthesis_meta["used_llm"],
                "llm_attempted": synthesis_meta["llm_prompt"] is not None,
                "llm_finish_reason": synthesis_meta.get("llm_finish_reason"),
                "llm_attempts": synthesis_meta.get("llm_attempts"),
                "llm_response_time_seconds": synthesis_meta.get("llm_response_time_seconds"),
                "llm_attempt_durations_seconds": synthesis_meta.get(
                    "llm_attempt_durations_seconds",
                ),
                "domain_task_contracts": synthesis_meta.get("domain_task_contracts", []),
                "action_analysis": synthesis_meta.get("action_analysis", {}),
                "derived_analysis": synthesis_meta.get("derived_analysis", {}),
                "failure_class": synthesis_meta.get("failure_class"),
                "declared_compound_tasks": synthesis_meta.get("declared_compound_tasks", []),
                "compound_tasks": synthesis_meta["compound_tasks"],
                "primitive_tasks": synthesis_meta["primitive_tasks"],
                "methods": synthesis_meta["methods"],
            }
            self._latest_transition_prompt_analysis = {}
            self._record_stage_timing(
                "stage3",
                stage_start,
                breakdown={
                    "synthesis_seconds": synthesis_seconds,
                    "typing_validation_seconds": typing_validation_seconds,
                    "llm_response_seconds": synthesis_meta.get("llm_response_time_seconds"),
                },
                metadata={
                    "used_llm": synthesis_meta.get("used_llm"),
                    "llm_attempted": synthesis_meta.get("llm_prompt") is not None,
                    "domain_complete": True,
                },
            )
            self.logger.log_stage3_method_synthesis(
                method_library.to_dict(),
                "Success",
                model=synthesis_meta["model"] if synthesis_meta["llm_prompt"] is not None else None,
                llm_prompt=synthesis_meta["llm_prompt"],
                llm_response=synthesis_meta["llm_response"],
                metadata=summary,
            )

            print("✓ Domain-complete HTN method synthesis complete")
            print(f"  Attempted LLM synthesis: {summary['llm_attempted']}")
            print(f"  Accepted LLM output: {summary['used_llm']}")
            print(f"  Declared compound tasks: {len(summary['declared_compound_tasks'])}")
            print(f"  Synthesised compound tasks: {summary['compound_tasks']}")
            print(f"  Primitive tasks: {summary['primitive_tasks']}")
            print(f"  Methods: {summary['methods']}")
            return method_library, summary

        except Exception as e:
            self._latest_transition_specs = ()
            self._latest_transition_prompt_analysis = {}
            self._record_stage_timing("stage3", stage_start)
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

    def _stage4_domain_gate(self, method_library):
        """Stage 4: run the Panda hard-fail gate once per declared compound task."""
        print("\n[STAGE 4] Domain Gate")
        print("-"*80)
        stage_start = time.perf_counter()
        planner = PANDAPlanner(workspace=str(self.output_dir))

        try:
            declared_compound_names = {
                str(getattr(task, "name", "")).strip()
                for task in getattr(self.domain, "tasks", ())
                if str(getattr(task, "name", "")).strip()
            }
            library_compound_names = {
                str(getattr(task, "name", "")).strip()
                for task in getattr(method_library, "compound_tasks", ())
                if str(getattr(task, "name", "")).strip()
            }
            missing_tasks = sorted(declared_compound_names - library_compound_names)
            if missing_tasks:
                raise ValueError(
                    "Stage 4 domain gate missing declared compound tasks: "
                    f"{missing_tasks}",
                )

            referenced_child_names = sorted(
                {
                    str(getattr(step, "task_name", "")).strip()
                    for method in getattr(method_library, "methods", ()) or ()
                    for step in (getattr(method, "subtasks", ()) or ())
                    if getattr(step, "kind", "") == "compound"
                    and str(getattr(step, "task_name", "")).strip()
                }
            )
            undefined_child_names = sorted(
                child_name
                for child_name in referenced_child_names
                if child_name not in library_compound_names
            )
            if undefined_child_names:
                raise ValueError(
                    "Stage 4 domain gate found undefined compound children: "
                    f"{undefined_child_names}",
                )

            gate_cases = self._stage4_domain_gate_cases(method_library)
            gate_results: List[Dict[str, Any]] = []
            planner_seconds = 0.0
            witness_fact_seconds = 0.0
            object_scope_seconds = 0.0

            for case in gate_cases:
                task_name = case["task_name"]
                task_args = case["task_args"]
                object_types = case["object_types"]
                object_pool = case["object_pool"]
                object_scope_start = time.perf_counter()
                case_object_pool = list(object_pool)
                case_object_types = dict(object_types)
                object_scope_seconds += time.perf_counter() - object_scope_start

                witness_start = time.perf_counter()
                initial_facts = self._task_witness_initial_facts(
                    planner,
                    task_name,
                    method_library,
                    task_args,
                    case_object_pool,
                    object_pool=case_object_pool,
                    object_types=case_object_types,
                )
                witness_fact_seconds += time.perf_counter() - witness_start

                validation_library, validation_args, projection = (
                    self._stage4_compact_validation_library_for_task(
                        method_library,
                        task_name,
                        task_args,
                        target_literal=None,
                        compact_arg_threshold=self.STAGE4_COMPACT_TASK_ARG_THRESHOLD,
                        max_compound_steps=self.STAGE4_MAX_VALIDATION_COMPOUND_STEPS,
                    )
                )
                plan_start = time.perf_counter()
                plan = planner.plan(
                    domain=self.domain,
                    method_library=validation_library,
                    objects=case_object_pool,
                    target_literal=None,
                    task_name=task_name,
                    transition_name=f"domain_gate_{sanitize_identifier(task_name)}",
                    typed_objects=tuple(case_object_types.items()),
                    task_args=validation_args,
                    initial_facts=initial_facts,
                    allow_empty_plan=True,
                )
                planner_seconds += time.perf_counter() - plan_start
                gate_results.append(
                    {
                        "task_name": task_name,
                        "task_args": list(validation_args),
                        "object_types": dict(case_object_types),
                        "initial_fact_count": len(initial_facts),
                        "plan": self._stage4_plan_artifact_summary(plan),
                        "projection": projection,
                    }
                )

            summary = {
                "gate_type": "domain_complete",
                "declared_compound_task_count": len(declared_compound_names),
                "validated_task_count": len(gate_results),
                "validated_tasks": [record["task_name"] for record in gate_results],
                "undefined_child_task_count": 0,
                "missing_declared_task_count": 0,
                "query_specific_runtime_records": 0,
                "task_validations": gate_results,
            }
            self._record_stage_timing(
                "stage4",
                stage_start,
                breakdown={
                    "object_scope_seconds": object_scope_seconds,
                    "witness_initial_facts_seconds": witness_fact_seconds,
                    "planner_seconds": planner_seconds,
                },
                metadata={
                    "gate_type": "domain_complete",
                    "validated_task_count": len(gate_results),
                },
            )
            self.logger.log_stage4_panda_planning(
                summary,
                "Success",
                metadata={
                    "backend": "pandaPI",
                    "gate_type": "domain_complete",
                    "validated_task_count": len(gate_results),
                },
            )

            print("✓ Domain gate complete")
            print(f"  Declared compound tasks: {len(declared_compound_names)}")
            print(f"  Validated tasks: {len(gate_results)}")
            return summary

        except Exception as e:
            self._record_stage_timing("stage4", stage_start)
            self.logger.log_stage4_panda_planning(
                None,
                "Failed",
                error=str(e),
                metadata={"gate_type": "domain_complete"},
            )
            print(f"✗ Stage 4 Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _stage4_domain_gate_cases(self, method_library) -> Tuple[Dict[str, Any], ...]:
        """Build synthetic task validation cases for the domain gate."""

        candidate_root_types = sorted(
            type_name
            for type_name in self.domain_type_names
            if self.type_parent_map.get(type_name) is None
        )
        default_type_name = (
            "object"
            if "object" in self.domain_type_names
            else (candidate_root_types[0] if candidate_root_types else "")
        )
        cases: List[Dict[str, Any]] = []
        for task in getattr(method_library, "compound_tasks", ()) or ():
            task_name = str(getattr(task, "name", "") or "").strip()
            if not task_name:
                continue
            type_signature = tuple(
                type_name or default_type_name
                for type_name in self._task_type_signature(task_name, method_library)
            )
            task_args: List[str] = []
            object_pool: List[str] = []
            object_types: Dict[str, str] = {}
            parameters = tuple(getattr(task, "parameters", ()) or ())
            for index, parameter in enumerate(parameters, start=1):
                object_name = f"gate_{sanitize_identifier(task_name)}_{index}"
                type_name = (
                    type_signature[index - 1]
                    if index - 1 < len(type_signature)
                    else default_type_name
                )
                task_args.append(object_name)
                object_pool.append(object_name)
                if type_name:
                    object_types[object_name] = type_name
            cases.append(
                {
                    "task_name": task_name,
                    "task_args": tuple(task_args),
                    "type_signature": type_signature,
                    "object_pool": tuple(object_pool),
                    "object_types": object_types,
                }
            )
        return tuple(cases)

    def _stage4_panda_planning(self, ltl_spec, method_library):
        """Stage 4: hard-fail reachability gate for the same Stage 3 method library."""
        print("\n[STAGE 4] PANDA Planning")
        print("-"*80)
        stage_start = time.perf_counter()

        planner = PANDAPlanner(workspace=str(self.output_dir))

        try:
            task_network = self._stage4_query_task_network(ltl_spec, method_library)
            if not task_network:
                raise ValueError(
                    "Stage 4 requires at least one query task anchor to validate the "
                    "Stage 3 method library.",
                )
            ordered_query_sequence = self._query_task_sequence_is_ordered(ltl_spec)

            query_inventory_object_types = self._query_inventory_object_type_map(
                tuple(getattr(ltl_spec, "query_object_inventory", ()) or ()),
            )
            stage4_breakdown: Dict[str, float] = {
                "seed_scope_seconds": 0.0,
                "witness_initial_facts_seconds": 0.0,
                "planner_seconds": 0.0,
            }
            seed_scope_start = time.perf_counter()
            witness_objects, witness_object_types = self._stage4_query_network_validation_scope(
                task_network,
                method_library,
                ltl_spec.objects,
                explicit_object_types=query_inventory_object_types,
            )
            stage4_breakdown["seed_scope_seconds"] += time.perf_counter() - seed_scope_start

            witness_facts_start = time.perf_counter()
            global_witness_initial_facts = self._stage4_query_network_initial_facts(
                planner,
                task_network,
                method_library,
                ltl_spec.objects,
                object_pool=witness_objects,
                object_types=witness_object_types,
            )
            stage4_breakdown["witness_initial_facts_seconds"] += (
                time.perf_counter() - witness_facts_start
            )
            target_literals = tuple(getattr(method_library, "target_literals", ()) or ())
            witness_initial_facts = self._stage4_strip_target_literals_from_initial_facts(
                global_witness_initial_facts,
                target_literals,
            )
            validation_cases = self._stage4_query_task_validation_cases(
                task_network,
                witness_object_types,
            )
            plan_records = []
            runtime_ordering_plan_records = []
            validation_artifacts = []
            for validation_index, query_index, task_name, task_args, type_signature in validation_cases:
                target_literal = (
                    target_literals[query_index - 1]
                    if query_index - 1 < len(target_literals)
                    else None
                )
                case_witness_objects = list(witness_objects)
                case_witness_object_types = dict(witness_object_types)
                local_witness_facts = self._task_witness_initial_facts(
                    planner,
                    task_name,
                    method_library,
                    task_args,
                    ltl_spec.objects,
                    object_pool=case_witness_objects,
                    object_types=case_witness_object_types,
                )
                case_initial_facts = tuple(
                    dict.fromkeys(
                        [
                            *(
                                str(fact).strip()
                                for fact in (local_witness_facts or ())
                                if str(fact).strip()
                            ),
                            *witness_initial_facts,
                        ],
                    ),
                )
                typed_objects = self._typed_object_entries(
                    case_witness_objects,
                    case_witness_object_types,
                )
                validation_name = (
                    f"query_task_schema_{validation_index}_"
                    f"{self._sanitize_name(str(task_name))}"
                )
                planner_start = time.perf_counter()
                plan = planner.plan(
                    domain=self.domain,
                    method_library=method_library,
                    objects=tuple(case_witness_objects),
                    target_literal=target_literal,
                    task_name=str(task_name),
                    transition_name=validation_name,
                    task_args=tuple(task_args),
                    typed_objects=typed_objects,
                    allow_empty_plan=False,
                    initial_facts=case_initial_facts,
                )
                stage4_breakdown["planner_seconds"] += time.perf_counter() - planner_start
                plan_records.append(
                    {
                        "validation_name": validation_name,
                        "transition_name": validation_name,
                        "task_name": str(task_name),
                        "task_args": tuple(task_args),
                        "task_network": tuple(
                            (str(network_task_name), tuple(network_task_args))
                            for network_task_name, network_task_args in task_network
                        ),
                        "target_literal": target_literal,
                        "objects": list(case_witness_objects),
                        "object_types": dict(case_witness_object_types),
                        "initial_facts": tuple(case_initial_facts),
                        "plan": plan,
                    }
                )
                validation_artifacts.append(
                    {
                        "validation_name": validation_name,
                        "query_index": query_index,
                        "task_name": str(task_name),
                        "task_args": list(task_args),
                        "type_signature": list(type_signature),
                        "target_literal": (
                            target_literal.to_dict()
                            if target_literal is not None
                            else None
                        ),
                        "object_count": len(case_witness_objects),
                        "objects_path": self._stage4_write_sequence_artifact(
                            plan,
                            "validation_objects.json",
                            list(case_witness_objects),
                            as_json=True,
                        ),
                        "initial_fact_count": len(case_initial_facts),
                        "initial_facts_path": self._stage4_write_sequence_artifact(
                            plan,
                            "validation_initial_facts.txt",
                            list(case_initial_facts),
                        ),
                        "plan": self._stage4_plan_artifact_summary(plan),
                    }
                )
            if (
                not ordered_query_sequence
                and len(task_network) <= self.STAGE4_UNORDERED_RUNTIME_TARGET_RECORD_LIMIT
            ):
                runtime_target_cases = self._stage4_unordered_runtime_target_cases(
                    task_network,
                    witness_object_types,
                )
                for query_index, task_name, task_args, type_signature in runtime_target_cases:
                    target_literal = (
                        target_literals[query_index - 1]
                        if query_index - 1 < len(target_literals)
                        else None
                    )
                    use_problem_runtime_seed = self.problem is not None
                    runtime_seed_kind = "problem_exact" if use_problem_runtime_seed else "witness_fallback"
                    if use_problem_runtime_seed:
                        case_witness_objects = list(self.problem.objects or ())
                        case_witness_object_types = dict(self.problem.object_types or {})
                        case_initial_facts = tuple(
                            self._render_problem_fact(fact)
                            for fact in (self.problem.init_facts or ())
                        )
                    else:
                        case_witness_objects = list(witness_objects)
                        case_witness_object_types = dict(witness_object_types)
                        local_witness_facts = self._task_witness_initial_facts(
                            planner,
                            task_name,
                            method_library,
                            task_args,
                            ltl_spec.objects,
                            object_pool=case_witness_objects,
                            object_types=case_witness_object_types,
                        )
                        case_initial_facts = tuple(
                            dict.fromkeys(
                                [
                                    *(
                                        str(fact).strip()
                                        for fact in (local_witness_facts or ())
                                        if str(fact).strip()
                                    ),
                                    *witness_initial_facts,
                                ],
                            ),
                        )
                    typed_objects = self._typed_object_entries(
                        case_witness_objects,
                        case_witness_object_types,
                    )
                    runtime_validation_name = (
                        f"unordered_runtime_target_{query_index}_"
                        f"{self._sanitize_name(str(task_name))}"
                    )
                    planner_start = time.perf_counter()
                    try:
                        plan = planner.plan(
                            domain=self.domain,
                            method_library=method_library,
                            objects=tuple(case_witness_objects),
                            target_literal=target_literal,
                            task_name=str(task_name),
                            transition_name=runtime_validation_name,
                            task_args=tuple(task_args),
                            typed_objects=typed_objects,
                            allow_empty_plan=False,
                            initial_facts=case_initial_facts,
                        )
                    except Exception:
                        if not use_problem_runtime_seed:
                            raise
                        runtime_seed_kind = "witness_fallback"
                        case_witness_objects = list(witness_objects)
                        case_witness_object_types = dict(witness_object_types)
                        local_witness_facts = self._task_witness_initial_facts(
                            planner,
                            task_name,
                            method_library,
                            task_args,
                            ltl_spec.objects,
                            object_pool=case_witness_objects,
                            object_types=case_witness_object_types,
                        )
                        case_initial_facts = tuple(
                            dict.fromkeys(
                                [
                                    *(
                                        str(fact).strip()
                                        for fact in (local_witness_facts or ())
                                        if str(fact).strip()
                                    ),
                                    *witness_initial_facts,
                                ],
                            ),
                        )
                        typed_objects = self._typed_object_entries(
                            case_witness_objects,
                            case_witness_object_types,
                        )
                        plan = planner.plan(
                            domain=self.domain,
                            method_library=method_library,
                            objects=tuple(case_witness_objects),
                            target_literal=target_literal,
                            task_name=str(task_name),
                            transition_name=runtime_validation_name,
                            task_args=tuple(task_args),
                            typed_objects=typed_objects,
                            allow_empty_plan=False,
                            initial_facts=case_initial_facts,
                        )
                    finally:
                        stage4_breakdown["planner_seconds"] += time.perf_counter() - planner_start
                    runtime_ordering_plan_records.append(
                        {
                            "record_kind": "unordered_runtime_target_exact",
                            "runtime_seed_kind": runtime_seed_kind,
                            "query_index": query_index,
                            "target_id": f"t{query_index}",
                            "task_name": str(task_name),
                            "task_args": tuple(task_args),
                            "type_signature": tuple(type_signature),
                            "target_literal": target_literal,
                            "objects": list(case_witness_objects),
                            "object_types": dict(case_witness_object_types),
                            "initial_facts": tuple(case_initial_facts),
                            "plan": plan,
                        }
                    )
            summary = {
                "backend": "pandaPI",
                "validation_mode": "same_library_query_task_schema_cases",
                "query_task_count": len(task_network),
                "validation_task_count": len(validation_cases),
                "runtime_ordering_record_count": len(runtime_ordering_plan_records),
                "planned_tasks": [str(task_name) for task_name, _ in task_network],
            }
            self._record_stage_timing(
                "stage4",
                stage_start,
                breakdown=stage4_breakdown,
                metadata={
                    "query_task_count": len(task_network),
                    "validation_task_count": len(validation_cases),
                },
            )

            self.logger.log_stage4_panda_planning(
                {
                    "query_validations": validation_artifacts,
                    "method_validations": [],
                },
                "Success",
                metadata=summary,
            )

            print("✓ PANDA planning complete")
            print(f"  Backend: {summary['backend']}")
            print(f"  Query tasks validated: {summary['query_task_count']}")
            print(f"  Schema validation cases: {summary['validation_task_count']}")
            if runtime_ordering_plan_records:
                print(
                    "  Runtime ordering exact target plans: "
                    f"{len(runtime_ordering_plan_records)}"
                )
            transitions_file = self.output_dir / "panda_transitions.json"
            print(f"  ✓ PANDA planning artifacts saved to: {transitions_file}")

            return plan_records, {
                "summary": summary,
                "query_validations": validation_artifacts,
                "method_validations": [],
                "unordered_runtime_plan_records": runtime_ordering_plan_records,
            }

        except Exception as e:
            self._record_stage_timing("stage4", stage_start)
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

    def _stage4_query_task_network(
        self,
        ltl_spec,
        method_library,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        query_text = str(getattr(ltl_spec, "source_instruction", "") or "")
        query_task_anchors = self._extract_query_task_anchors(query_text)
        literal_signatures = tuple(
            str(signature).strip()
            for signature in (getattr(ltl_spec, "query_task_literal_signatures", ()) or ())
            if str(signature).strip()
        )
        query_object_inventory = tuple(getattr(ltl_spec, "query_object_inventory", ()) or ())
        if not query_object_inventory:
            query_object_inventory = self._extract_query_object_inventory(query_text)
        task_name_map = self._method_library_source_task_name_map(method_library)
        variable_assignments: Dict[str, str] = {}

        task_network: List[Tuple[str, Tuple[str, ...]]] = []
        for index, anchor in enumerate(query_task_anchors):
            anchor_task_name = str(anchor.get("task_name") or "").strip()
            resolved_task_name = task_name_map.get(anchor_task_name, anchor_task_name)
            if not resolved_task_name:
                continue
            if method_library.task_for_name(resolved_task_name) is None:
                raise ValueError(
                    f"Stage 4 query task '{anchor_task_name}' is not defined in the "
                    "Stage 3 method library.",
                )
            raw_task_args = tuple(
                str(arg).strip()
                for arg in (anchor.get("args") or ())
                if str(arg).strip()
            )
            task_args = self._stage6_ground_query_task_arguments(
                task_name=resolved_task_name,
                task_args=raw_task_args,
                literal_signature=(
                    literal_signatures[index]
                    if index < len(literal_signatures)
                    else None
                ),
                method_library=method_library,
                query_object_inventory=query_object_inventory,
                variable_assignments=variable_assignments,
            )
            task_network.append(
                (
                    resolved_task_name,
                    task_args,
                ),
            )

        if task_network:
            return tuple(task_network)

        fallback_network: List[Tuple[str, Tuple[str, ...]]] = []
        for binding in tuple(getattr(method_library, "target_task_bindings", ()) or ()):
            task_name = str(getattr(binding, "task_name", "") or "").strip()
            literal = _parse_signature_literal(str(getattr(binding, "target_literal", "") or "").strip())
            if not task_name or literal is None:
                continue
            fallback_network.append((task_name, tuple(str(arg) for arg in literal.args)))
        return tuple(fallback_network)

    @staticmethod
    def _method_library_source_task_name_map(method_library) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for task in tuple(getattr(method_library, "compound_tasks", ()) or ()):
            internal_name = str(getattr(task, "name", "") or "").strip()
            source_name = str(getattr(task, "source_name", "") or "").strip()
            if internal_name:
                mapping.setdefault(internal_name, internal_name)
            if source_name:
                mapping.setdefault(source_name, internal_name or source_name)
        return mapping

    def _stage4_query_network_validation_scope(
        self,
        task_network: Sequence[Tuple[str, Sequence[str]]],
        method_library,
        objects,
        *,
        explicit_object_types: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        object_pool: List[str] = list(dict.fromkeys(objects or ()))
        explicit_object_types = explicit_object_types or {}
        hard_type_candidates: Dict[str, Set[str]] = defaultdict(set)
        for task_name, task_args in task_network:
            for obj in task_args:
                rendered = str(obj).strip()
                if rendered and rendered not in object_pool:
                    object_pool.append(rendered)
            self._collect_argument_signature_constraints(
                candidates=hard_type_candidates,
                args=tuple(task_args),
                signature=self._task_type_signature(task_name, method_library),
                scope=f"Stage 4 query task '{task_name}' argument typing",
            )

        soft_type_candidates = self._target_literal_type_candidates(
            tuple(getattr(method_library, "target_literals", ()) or ()),
        )
        object_types: Dict[str, str] = {}
        for obj in object_pool:
            symbol = str(obj).strip()
            if not symbol:
                continue
            explicit_type = explicit_object_types.get(symbol)
            if explicit_type is not None:
                object_types[symbol] = self._resolve_explicit_symbol_type(
                    symbol=symbol,
                    explicit_type=explicit_type,
                    hard_candidate_types=hard_type_candidates.get(symbol, set()),
                    soft_candidate_types=soft_type_candidates.get(symbol, set()),
                    scope="Stage 4 query task network object typing",
                )
                continue
            object_types[symbol] = self._resolve_symbol_type(
                symbol=symbol,
                candidate_types=(
                    hard_type_candidates.get(symbol, set())
                    or soft_type_candidates.get(symbol, set())
                ),
                scope="Stage 4 query task network object typing",
            )
        for obj, explicit_type in explicit_object_types.items():
            symbol = str(obj).strip()
            type_name = str(explicit_type).strip()
            if symbol and type_name and symbol not in object_types:
                object_types[symbol] = type_name

        return object_pool, object_types

    @staticmethod
    def _stage4_query_task_validation_cases(
        task_network: Sequence[Tuple[str, Sequence[str]]],
        object_types: Dict[str, str],
    ) -> Tuple[Tuple[int, int, str, Tuple[str, ...], Tuple[str, ...]], ...]:
        cases: List[Tuple[int, int, str, Tuple[str, ...], Tuple[str, ...]]] = []
        seen_keys: Set[Tuple[str, Tuple[str, ...]]] = set()
        for query_index, (task_name, task_args) in enumerate(task_network, start=1):
            rendered_task_name = str(task_name).strip()
            rendered_args = tuple(str(arg).strip() for arg in task_args)
            type_signature = tuple(
                str(object_types.get(arg) or "").strip()
                for arg in rendered_args
            )
            key = (rendered_task_name, type_signature)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cases.append(
                (
                    len(cases) + 1,
                    query_index,
                    rendered_task_name,
                    rendered_args,
                    type_signature,
                ),
            )
        return tuple(cases)

    @staticmethod
    def _stage4_unordered_runtime_target_cases(
        task_network: Sequence[Tuple[str, Sequence[str]]],
        object_types: Dict[str, str],
    ) -> Tuple[Tuple[int, str, Tuple[str, ...], Tuple[str, ...]], ...]:
        cases: List[Tuple[int, str, Tuple[str, ...], Tuple[str, ...]]] = []
        for query_index, (task_name, task_args) in enumerate(task_network, start=1):
            rendered_task_name = str(task_name).strip()
            rendered_args = tuple(str(arg).strip() for arg in task_args)
            type_signature = tuple(
                str(object_types.get(arg) or "").strip()
                for arg in rendered_args
            )
            cases.append(
                (
                    query_index,
                    rendered_task_name,
                    rendered_args,
                    type_signature,
                ),
            )
        return tuple(cases)

    def _stage4_query_network_initial_facts(
        self,
        planner,
        task_network: Sequence[Tuple[str, Sequence[str]]],
        method_library,
        objects,
        *,
        object_pool: Sequence[str],
        object_types: Dict[str, str],
    ) -> Tuple[str, ...]:
        facts: List[str] = []
        seen: Set[str] = set()
        for task_name, task_args in task_network:
            for fact in self._task_witness_initial_facts(
                planner,
                task_name,
                method_library,
                task_args,
                objects,
                object_pool=object_pool,
                object_types=object_types,
            ):
                rendered = str(fact).strip()
                if not rendered or rendered in seen:
                    continue
                seen.add(rendered)
                facts.append(rendered)
        return self._stage4_strip_target_literals_from_initial_facts(
            tuple(facts),
            tuple(getattr(method_library, "target_literals", ()) or ()),
        )

    @staticmethod
    def _stage4_strip_target_literals_from_initial_facts(
        initial_facts: Sequence[str],
        target_literals: Sequence[HTNLiteral],
    ) -> Tuple[str, ...]:
        blocked_positive_facts = set()
        blocked_negative_atoms = set()
        for literal in target_literals:
            if getattr(literal, "is_equality", False):
                continue
            rendered_fact = (
                f"({literal.predicate} {' '.join(literal.args)})"
                if literal.args
                else f"({literal.predicate})"
            )
            if getattr(literal, "is_positive", True):
                blocked_positive_facts.add(rendered_fact)
            else:
                blocked_negative_atoms.add((literal.predicate, tuple(literal.args)))

        filtered: List[str] = []
        for fact in initial_facts:
            rendered = str(fact).strip()
            if rendered in blocked_positive_facts:
                continue
            parsed = LTL_BDI_Pipeline._parse_positive_hddl_fact(rendered)
            if parsed is not None and parsed in blocked_negative_atoms:
                continue
            filtered.append(rendered)
        return tuple(filtered)

    @staticmethod
    def _stage4_compact_validation_library_for_task(
        method_library,
        root_task_name: str,
        task_args: Sequence[str],
        *,
        target_literal: Optional[HTNLiteral],
        compact_arg_threshold: int,
        max_compound_steps: int,
    ):
        if method_library is None:
            return method_library, tuple(task_args or ()), None
        root_task_name = str(root_task_name or "").strip()
        task_args = tuple(str(arg) for arg in (task_args or ()))
        if not root_task_name or len(task_args) <= compact_arg_threshold:
            return method_library, task_args, None

        root_task = method_library.task_for_name(root_task_name)
        if root_task is None:
            return method_library, task_args, None
        root_methods = list(method_library.methods_for_task(root_task_name))
        if not root_methods:
            return method_library, task_args, None

        constructive_methods = [
            method
            for method in root_methods
            if tuple(getattr(method, "subtasks", ()) or ())
        ]
        validation_root_methods = constructive_methods or root_methods
        root_parameters = tuple(getattr(root_task, "parameters", ()) or ())
        if not root_parameters:
            root_parameters = tuple(getattr(validation_root_methods[0], "parameters", ()) or ())
        if not root_parameters or len(root_parameters) != len(task_args):
            return method_library, task_args, None

        headline_args = tuple(
            str(arg)
            for arg in (
                getattr(getattr(root_task, "headline_literal", None), "args", ()) or ()
            )
            if str(arg) in root_parameters
        )
        if not headline_args and target_literal is not None:
            headline_args = root_parameters[: len(getattr(target_literal, "args", ()) or ())]
        protected_parameters = set(headline_args)
        parameter_index = {parameter: index for index, parameter in enumerate(root_parameters)}
        max_compound_steps = max(1, int(max_compound_steps))

        compacted_methods: List[HTNMethod] = []
        used_parameters: Set[str] = set(protected_parameters)
        original_subtask_count = 0
        compacted_subtask_count = 0

        def _literal_parameters(literal) -> Set[str]:
            return {
                str(arg)
                for arg in (getattr(literal, "args", ()) or ())
                if str(arg) in parameter_index
            }

        for method in validation_root_methods:
            selected_steps: List[HTNMethodStep] = []
            seen_compound_shapes: Set[Tuple[str, int]] = set()
            method_steps = tuple(getattr(method, "subtasks", ()) or ())
            original_subtask_count += len(method_steps)
            for step in method_steps:
                step_args = tuple(str(arg) for arg in (getattr(step, "args", ()) or ()))
                step_parameters = {
                    arg
                    for arg in step_args
                    if arg in parameter_index
                }
                keep_step = getattr(step, "kind", "") != "compound"
                shape = (
                    str(getattr(step, "task_name", "") or ""),
                    len(step_args),
                )
                if not keep_step:
                    keep_step = (
                        bool(step_parameters)
                        and step_parameters.issubset(protected_parameters)
                    )
                if not keep_step and shape not in seen_compound_shapes:
                    keep_step = len(seen_compound_shapes) < max_compound_steps
                if not keep_step:
                    continue
                if getattr(step, "kind", "") == "compound":
                    seen_compound_shapes.add(shape)
                selected_steps.append(step)
                used_parameters.update(step_parameters)

            if method_steps and not selected_steps:
                first_step = method_steps[0]
                selected_steps.append(first_step)
                used_parameters.update(
                    str(arg)
                    for arg in (getattr(first_step, "args", ()) or ())
                    if str(arg) in parameter_index
                )

            selected_step_ids = {
                str(getattr(step, "step_id", ""))
                for step in selected_steps
            }
            selected_context: List[HTNLiteral] = []
            changed = True
            while changed:
                changed = False
                for literal in tuple(getattr(method, "context", ()) or ()):
                    literal_parameters = _literal_parameters(literal)
                    if not literal_parameters:
                        continue
                    if literal in selected_context:
                        continue
                    if not (literal_parameters & used_parameters):
                        continue
                    selected_context.append(literal)
                    before = len(used_parameters)
                    used_parameters.update(literal_parameters)
                    changed = changed or len(used_parameters) != before

            compacted_ordering = tuple(
                (before, after)
                for before, after in (getattr(method, "ordering", ()) or ())
                if before in selected_step_ids and after in selected_step_ids
            )
            compacted_methods.append(
                HTNMethod(
                    method_name=method.method_name,
                    task_name=method.task_name,
                    parameters=(),
                    task_args=(),
                    context=tuple(selected_context),
                    subtasks=tuple(selected_steps),
                    ordering=compacted_ordering,
                    origin=getattr(method, "origin", "heuristic"),
                    source_method_name=getattr(method, "source_method_name", None),
                )
            )
            compacted_subtask_count += len(selected_steps)

        used_parameters = {
            parameter
            for parameter in used_parameters
            if parameter in parameter_index
        }
        if not used_parameters:
            return method_library, task_args, None
        validation_parameters = tuple(
            parameter
            for parameter in root_parameters
            if parameter in used_parameters
        )
        validation_args = tuple(task_args[parameter_index[parameter]] for parameter in validation_parameters)

        final_methods = [
            HTNMethod(
                method_name=method.method_name,
                task_name=method.task_name,
                parameters=validation_parameters,
                task_args=validation_parameters,
                context=method.context,
                subtasks=method.subtasks,
                ordering=method.ordering,
                origin=method.origin,
                source_method_name=method.source_method_name,
            )
            for method in compacted_methods
        ]
        compacted_task = HTNTask(
            name=root_task.name,
            parameters=validation_parameters,
            is_primitive=root_task.is_primitive,
            source_predicates=root_task.source_predicates,
            headline_literal=root_task.headline_literal,
            source_name=root_task.source_name,
        )
        compacted_library = HTNMethodLibrary(
            compound_tasks=[
                compacted_task if task.name == root_task_name else task
                for task in getattr(method_library, "compound_tasks", ()) or ()
            ],
            primitive_tasks=list(getattr(method_library, "primitive_tasks", ()) or ()),
            methods=[
                *[
                    method
                    for method in getattr(method_library, "methods", ()) or ()
                    if method.task_name != root_task_name
                ],
                *final_methods,
            ],
            target_literals=list(getattr(method_library, "target_literals", ()) or ()),
            target_task_bindings=list(
                getattr(method_library, "target_task_bindings", ()) or (),
            ),
        )
        compacted_library = LTL_BDI_Pipeline._stage4_pruned_method_library_for_task(
            compacted_library,
            root_task_name,
        )
        projection = {
            "mode": "validation_method_body_compaction",
            "original_task_arg_count": len(task_args),
            "validation_task_arg_count": len(validation_args),
            "original_root_method_count": len(root_methods),
            "validation_root_method_count": len(final_methods),
            "original_subtask_count": original_subtask_count,
            "validation_subtask_count": compacted_subtask_count,
        }
        return compacted_library, validation_args, projection

    @staticmethod
    def _stage4_pruned_method_library_for_task(method_library, root_task_name: str):
        if method_library is None:
            return method_library
        root_task_name = str(root_task_name or "").strip()
        if not root_task_name:
            return method_library

        methods_by_task: Dict[str, List[HTNMethod]] = {}
        for method in getattr(method_library, "methods", ()) or ():
            task_name = str(getattr(method, "task_name", "") or "").strip()
            if task_name:
                methods_by_task.setdefault(task_name, []).append(method)

        compound_task_names = {
            str(getattr(task, "name", "") or "").strip()
            for task in getattr(method_library, "compound_tasks", ()) or ()
            if str(getattr(task, "name", "") or "").strip()
        }

        def is_transition_task(task_name: str) -> bool:
            return str(task_name or "").strip().startswith("dfa_step_")

        root_is_transition_task = is_transition_task(root_task_name)
        validation_methods_cache: Dict[str, Tuple[HTNMethod, ...]] = {}
        validation_method_id_cache: Dict[str, FrozenSet[int]] = {}

        def validation_methods_for_task(task_name: str) -> Tuple[HTNMethod, ...]:
            cached = validation_methods_cache.get(task_name)
            if cached is not None:
                return cached

            methods = tuple(methods_by_task.get(task_name, ()))
            if (
                root_is_transition_task
                and task_name != root_task_name
                and is_transition_task(task_name)
            ):
                noop_methods = tuple(
                    method
                    for method in methods
                    if not tuple(getattr(method, "subtasks", ()) or ())
                )
                if noop_methods:
                    methods = noop_methods

            validation_methods_cache[task_name] = methods
            validation_method_id_cache[task_name] = frozenset(
                id(method)
                for method in methods
            )
            return methods

        def validation_method_ids_for_task(task_name: str) -> FrozenSet[int]:
            if task_name not in validation_method_id_cache:
                validation_methods_for_task(task_name)
            return validation_method_id_cache.get(task_name, frozenset())

        reachable_compound_names: Set[str] = set()
        pending = [root_task_name]
        while pending:
            task_name = pending.pop()
            if task_name in reachable_compound_names:
                continue
            reachable_compound_names.add(task_name)
            for method in validation_methods_for_task(task_name):
                for step in getattr(method, "subtasks", ()) or ():
                    if getattr(step, "kind", "") != "compound":
                        continue
                    child_task_name = str(getattr(step, "task_name", "") or "").strip()
                    if child_task_name and child_task_name not in reachable_compound_names:
                        pending.append(child_task_name)

        if not reachable_compound_names or reachable_compound_names == compound_task_names:
            return method_library

        compound_tasks = [
            task
            for task in getattr(method_library, "compound_tasks", ()) or ()
            if str(getattr(task, "name", "") or "").strip() in reachable_compound_names
        ]
        methods: List[HTNMethod] = []
        for method in getattr(method_library, "methods", ()) or ():
            task_name = str(getattr(method, "task_name", "") or "").strip()
            if task_name not in reachable_compound_names:
                continue
            if id(method) not in validation_method_ids_for_task(task_name):
                continue
            methods.append(method)
        return HTNMethodLibrary(
            compound_tasks=list(compound_tasks),
            primitive_tasks=list(getattr(method_library, "primitive_tasks", ()) or ()),
            methods=list(methods),
            target_literals=list(getattr(method_library, "target_literals", ()) or ()),
            target_task_bindings=list(
                getattr(method_library, "target_task_bindings", ()) or (),
            ),
        )

    def _stage4_plan_artifact_summary(self, plan) -> Dict[str, Any]:
        """Return a compact, path-based Stage 4 plan summary for persisted logs."""
        actual_plan_text = str(getattr(plan, "actual_plan", "") or "")
        raw_plan_text = str(getattr(plan, "raw_plan", "") or "")
        work_dir = Path(plan.work_dir) if getattr(plan, "work_dir", None) else None
        artifacts: Dict[str, str] = {}
        if work_dir is not None:
            for artifact_name, filename in (
                ("domain_hddl", "domain.hddl"),
                ("problem_hddl", "problem.hddl"),
                ("problem_psas", "problem.psas"),
                ("problem_psas_grounded", "problem.psas.grounded"),
                ("raw_plan", "plan.original"),
                ("actual_plan", "plan.actual"),
            ):
                artifact_path = work_dir / filename
                if artifact_path.exists():
                    artifacts[artifact_name] = self._stage4_relative_artifact_path(artifact_path)

        return {
            "task_name": str(getattr(plan, "task_name", "")),
            "task_args": list(getattr(plan, "task_args", ()) or ()),
            "target_literal": (
                plan.target_literal.to_dict()
                if getattr(plan, "target_literal", None) is not None
                else None
            ),
            "step_count": len(getattr(plan, "steps", ()) or ()),
            "raw_plan_line_count": len(raw_plan_text.splitlines()),
            "actual_plan_line_count": len(actual_plan_text.splitlines()),
            "work_dir": (
                self._stage4_relative_artifact_path(work_dir)
                if work_dir is not None
                else None
            ),
            "artifacts": artifacts,
            "timing_profile": dict(getattr(plan, "timing_profile", {}) or {}),
        }

    def _stage4_write_sequence_artifact(
        self,
        plan,
        filename: str,
        items: Sequence[Any],
        *,
        as_json: bool = False,
    ) -> Optional[str]:
        work_dir = Path(plan.work_dir) if getattr(plan, "work_dir", None) else None
        if work_dir is None:
            return None
        artifact_path = work_dir / filename
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if as_json:
            artifact_path.write_text(json.dumps(list(items), indent=2))
        else:
            artifact_path.write_text("\n".join(str(item) for item in items) + "\n")
        return self._stage4_relative_artifact_path(artifact_path)

    def _stage4_relative_artifact_path(self, artifact_path: str | Path) -> str:
        path = Path(artifact_path)
        if self.output_dir is not None:
            try:
                return str(path.resolve().relative_to(Path(self.output_dir).resolve()))
            except ValueError:
                pass
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path)

    def _seed_validation_scope(
        self,
        task_name,
        method_library,
        task_args,
        objects,
        *,
        explicit_object_types: Optional[Dict[str, str]] = None,
    ):
        object_pool = list(dict.fromkeys(task_args or objects))
        type_candidates: Dict[str, Set[str]] = defaultdict(set)
        hard_type_candidates: Dict[str, Set[str]] = defaultdict(set)
        self._merge_type_candidates(
            type_candidates,
            self._target_literal_type_candidates(method_library.target_literals),
        )
        self._merge_type_candidates(
            hard_type_candidates,
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
        self._merge_type_candidates(
            hard_type_candidates,
            self._task_argument_type_candidates(
                task_name,
                task_args,
                method_library,
                include_method_alternatives=False,
            ),
        )
        explicit_object_types = explicit_object_types or {}
        object_types = {}
        for obj in object_pool:
            explicit_type = explicit_object_types.get(obj)
            if explicit_type is not None:
                object_types[obj] = self._resolve_explicit_symbol_type(
                    symbol=obj,
                    explicit_type=explicit_type,
                    hard_candidate_types=hard_type_candidates.get(obj, set()),
                    soft_candidate_types=type_candidates.get(obj, set()),
                    scope=f"Stage 4 task '{task_name}' object typing",
                )
                continue
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

    def _validate_problem_domain_compatibility(self) -> None:
        if self.problem is None:
            return

        domain_task_names = set(self.task_type_map.keys())
        domain_predicate_names = set(self.predicate_type_map.keys())

        unknown_problem_types = sorted(
            {
                type_name
                for type_name in self.problem.object_types.values()
                if type_name not in self.domain_type_names
            },
        )
        if unknown_problem_types:
            raise ValueError(
                "problem_file references object types missing from domain_file: "
                f"{unknown_problem_types}",
            )

        for task in self.problem.htn_tasks:
            if task.task_name not in domain_task_names:
                raise ValueError(
                    "problem_file HTN task is not declared in domain_file: "
                    f"{task.task_name}",
                )
            self._validate_problem_arguments_against_signature(
                args=task.args,
                signature=self.task_type_map[task.task_name],
                object_types=self.problem.object_types,
                scope=f"problem HTN task '{task.to_signature()}'",
            )

        for fact in (*self.problem.init_facts, *self.problem.goal_facts):
            if fact.predicate not in domain_predicate_names:
                raise ValueError(
                    "problem_file predicate is not declared in domain_file: "
                    f"{fact.predicate}",
                )
            self._validate_problem_arguments_against_signature(
                args=fact.args,
                signature=self.predicate_type_map[fact.predicate],
                object_types=self.problem.object_types,
                scope=f"problem fact '{fact.to_signature()}'",
            )

        if self.problem.domain_name.lower() == self.domain.name.lower():
            return

    def _validate_problem_arguments_against_signature(
        self,
        *,
        args: Sequence[str],
        signature: Sequence[str],
        object_types: Dict[str, str],
        scope: str,
    ) -> None:
        if len(args) != len(signature):
            raise ValueError(
                f"{scope}: arity mismatch (args={len(args)}, signature={len(signature)}).",
            )
        for arg, expected_type in zip(args, signature):
            actual_type = object_types.get(arg)
            if actual_type is None:
                continue
            if actual_type not in self.domain_type_names:
                raise ValueError(
                    f"{scope}: object '{arg}' uses unknown type '{actual_type}'.",
                )
            if not self._is_subtype(actual_type, expected_type):
                raise ValueError(
                    f"{scope}: object '{arg}' has type '{actual_type}', expected "
                    f"'{expected_type}'.",
                )

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
        cache = getattr(self, "_subtype_check_cache", None)
        if cache is None:
            cache = {}
            self._subtype_check_cache = cache
        cache_key = (candidate_type, expected_type)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        if candidate_type == expected_type:
            cache[cache_key] = True
            return True
        if candidate_type not in self.type_parent_map or expected_type not in self.type_parent_map:
            cache[cache_key] = False
            return False
        cursor = self.type_parent_map.get(candidate_type)
        visited = {candidate_type}
        while cursor is not None and cursor not in visited:
            if cursor == expected_type:
                cache[cache_key] = True
                return True
            visited.add(cursor)
            cursor = self.type_parent_map.get(cursor)
        cache[cache_key] = False
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

        candidate_key = frozenset(candidate_types)
        cache = getattr(self, "_resolved_symbol_type_cache", None)
        if cache is None:
            cache = {}
            self._resolved_symbol_type_cache = cache
        cached = cache.get(candidate_key)
        if cached is not None:
            return cached

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

        most_general = sorted(
            type_name
            for type_name in feasible
            if not any(
                other != type_name and self._is_subtype(type_name, other)
                for other in feasible
            )
        )
        if len(most_general) != 1:
            raise TypeResolutionError(
                f"{scope}: symbol '{symbol}' is ambiguous under constraints "
                f"{sorted(candidate_types)}; candidate schema types={most_general}.",
            )
        cache[candidate_key] = most_general[0]
        return most_general[0]

    def _resolve_explicit_symbol_type(
        self,
        *,
        symbol: str,
        explicit_type: str,
        hard_candidate_types: Set[str],
        soft_candidate_types: Set[str],
        scope: str,
    ) -> str:
        type_name = str(explicit_type or "").strip()
        if not type_name:
            raise TypeResolutionError(
                f"{scope}: symbol '{symbol}' has an empty explicit query type.",
            )
        unknown_types = sorted(
            candidate
            for candidate in {type_name, *hard_candidate_types, *soft_candidate_types}
            if candidate not in self.domain_type_names
        )
        if unknown_types:
            raise TypeResolutionError(
                f"{scope}: symbol '{symbol}' references unknown types {unknown_types}.",
            )

        incompatible_hard = sorted(
            required
            for required in hard_candidate_types
            if not self._is_subtype(type_name, required)
        )
        if incompatible_hard:
            raise TypeResolutionError(
                f"{scope}: explicit query type '{type_name}' for symbol '{symbol}' "
                f"does not satisfy hard constraints {incompatible_hard}.",
            )

        if not hard_candidate_types and soft_candidate_types:
            if not any(self._is_subtype(type_name, candidate) for candidate in soft_candidate_types):
                raise TypeResolutionError(
                    f"{scope}: explicit query type '{type_name}' for symbol '{symbol}' "
                    f"is incompatible with inferred branch constraints "
                    f"{sorted(soft_candidate_types)}.",
                )
        return type_name

    def _task_type_signature(self, task_name: str, method_library=None) -> Tuple[str, ...]:
        signature = self.task_type_map.get(task_name)
        if signature is not None:
            return signature
        if method_library is None:
            return ()

        cache = getattr(self, "_dynamic_task_signature_cache", None)
        if cache is None:
            cache = {}
            self._dynamic_task_signature_cache = cache
        cache_key = (id(method_library), task_name)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        task_schema = method_library.task_for_name(task_name)
        if task_schema is None:
            cache[cache_key] = ()
            return ()
        predicate_name = ""
        if len(getattr(task_schema, "source_predicates", ()) or ()) == 1:
            predicate_name = str(task_schema.source_predicates[0]).strip()
        elif getattr(task_schema, "headline_literal", None) is not None:
            predicate_name = str(task_schema.headline_literal.predicate).strip()
        if not predicate_name:
            cache[cache_key] = ()
            return ()
        predicate_signature = self.predicate_type_map.get(predicate_name, ())
        if not predicate_signature:
            cache[cache_key] = ()
            return ()
        if len(predicate_signature) != len(task_schema.parameters):
            raise TypeResolutionError(
                f"Task '{task_name}' source predicate '{predicate_name}' arity mismatch: "
                f"task has {len(task_schema.parameters)} args, predicate has "
                f"{len(predicate_signature)}.",
            )
        cache[cache_key] = predicate_signature
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
        cache = getattr(self, "_method_variable_type_hint_cache", None)
        if cache is None:
            cache = {}
            self._method_variable_type_hint_cache = cache
        cache_key = (id(method_library), id(method))
        cached = cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        candidates: Dict[str, Set[str]] = defaultdict(set)
        task_signature = self._task_type_signature(method.task_name, method_library)
        task_binding_args = list(
            self._method_task_binding_args(
                method,
                method_library,
                signature=task_signature,
            ),
        )
        self._collect_argument_signature_constraints(
            candidates=candidates,
            args=tuple(task_binding_args),
            signature=task_signature,
            scope=f"Method '{method.method_name}' task parameter typing",
        )
        schematic_symbols = set(method.parameters) | set(task_binding_args)

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

        variable_symbols: Set[str] = set(schematic_symbols)
        for literal in method.context:
            variable_symbols.update(
                arg
                for arg in literal.args
                if arg in schematic_symbols or self._is_variable_symbol(arg)
            )
        for step in method.subtasks:
            variable_symbols.update(
                arg
                for arg in step.args
                if arg in schematic_symbols or self._is_variable_symbol(arg)
            )
            if step.literal:
                variable_symbols.update(
                    arg
                    for arg in step.literal.args
                    if arg in schematic_symbols or self._is_variable_symbol(arg)
                )
            for literal in (*step.preconditions, *step.effects):
                variable_symbols.update(
                    arg
                    for arg in literal.args
                    if arg in schematic_symbols or self._is_variable_symbol(arg)
                )

        resolved = {
            symbol: self._resolve_symbol_type(
                symbol=symbol,
                candidate_types=candidates.get(symbol, set()),
                scope=f"Stage 3 method '{method.method_name}' variable typing",
            )
            for symbol in sorted(variable_symbols)
        }
        cache[cache_key] = dict(resolved)
        return resolved

    def _task_argument_type_candidates(
        self,
        task_name: str,
        task_args: Sequence[str],
        method_library,
        *,
        include_method_alternatives: bool = True,
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

        if not include_method_alternatives:
            return candidates

        for method in method_library.methods_for_task(task_name):
            variable_types = self._method_variable_type_hints(method, method_library)
            schematic_symbols = set(method.parameters) | set(method.task_args)
            for parameter, arg in zip(
                self._method_task_binding_args(
                    method,
                    method_library,
                    signature=task_signature,
                ),
                task_args,
            ):
                self._add_type_candidate(candidates, arg, variable_types.get(parameter))
            for literal in method.context:
                literal_candidates = self._literal_type_candidates(literal)
                for symbol, type_names in literal_candidates.items():
                    if symbol in schematic_symbols or self._is_variable_symbol(symbol):
                        continue
                    self._merge_type_candidates(candidates, {symbol: type_names})

        return candidates

    def _method_task_binding_args(
        self,
        method,
        method_library,
        *,
        signature: Sequence[str] = (),
    ) -> Tuple[str, ...]:
        explicit_task_args = tuple(getattr(method, "task_args", ()) or ())
        if explicit_task_args:
            if signature and len(explicit_task_args) != len(signature):
                raise TypeResolutionError(
                    f"Method '{method.method_name}' task-argument arity mismatch: "
                    f"task_args={len(explicit_task_args)}, signature={len(signature)}.",
                )
            return explicit_task_args

        task_schema = method_library.task_for_name(method.task_name)
        if task_schema is not None and task_schema.parameters:
            task_binding_args: List[str] = []
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
            return tuple(task_binding_args)

        if signature:
            return tuple(method.parameters[:len(signature)])
        return tuple(method.parameters)

    def _validate_method_library_typing(self, method_library) -> None:
        target_literal_by_signature: Dict[str, HTNLiteral] = {}
        target_literal_candidates_by_signature: Dict[str, Dict[str, Set[str]]] = {}
        for literal in method_library.target_literals:
            signature = literal.to_signature()
            if signature in target_literal_by_signature:
                continue
            target_literal_by_signature[signature] = literal
            target_literal_candidates_by_signature[signature] = (
                self._literal_type_candidates(literal)
            )

        target_binding_candidates_by_task: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set),
        )
        for binding in method_library.target_task_bindings:
            target_literal = target_literal_by_signature.get(binding.target_literal)
            if target_literal is None:
                raise TypeResolutionError(
                    f"Stage 3 binding references missing target literal "
                    f"'{binding.target_literal}'.",
                )
            self._merge_type_candidates(
                target_binding_candidates_by_task[binding.task_name],
                target_literal_candidates_by_signature.get(binding.target_literal, {}),
            )

        def binding_type_candidates(
            task_name: str,
            task_args: Sequence[str],
        ) -> Dict[str, Set[str]]:
            candidates: Dict[str, Set[str]] = defaultdict(set)
            self._merge_type_candidates(
                candidates,
                target_binding_candidates_by_task.get(task_name, {}),
            )
            task_signature = self._task_type_signature(task_name, method_library)
            self._collect_argument_signature_constraints(
                candidates=candidates,
                args=task_args,
                signature=task_signature,
                scope=f"Task '{task_name}' argument typing",
            )

            for method in method_library.methods_for_task(task_name):
                variable_types = self._method_variable_type_hints(method, method_library)
                schematic_symbols = set(method.parameters) | set(method.task_args)
                for parameter, arg in zip(
                    self._method_task_binding_args(
                        method,
                        method_library,
                        signature=task_signature,
                    ),
                    task_args,
                ):
                    self._add_type_candidate(candidates, arg, variable_types.get(parameter))
                for literal in method.context:
                    literal_candidates = self._literal_type_candidates(literal)
                    for symbol, type_names in literal_candidates.items():
                        if symbol in schematic_symbols or self._is_variable_symbol(symbol):
                            continue
                        self._merge_type_candidates(candidates, {symbol: type_names})
            return candidates

        for binding in method_library.target_task_bindings:
            task_name = binding.task_name
            target_literal = target_literal_by_signature.get(binding.target_literal)
            if target_literal is None:
                raise TypeResolutionError(
                    f"Stage 3 binding references missing target literal "
                    f"'{binding.target_literal}'.",
                )
            candidates = binding_type_candidates(
                task_name,
                target_literal.args,
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

    def _method_task_argument_type_mismatches(
        self,
        method,
        method_library,
        task_args,
        object_types: Optional[Dict[str, str]],
    ) -> Tuple[str, ...]:
        if not object_types:
            return ()
        method_variable_types = self._method_variable_type_hints(method, method_library)
        mismatches: List[str] = []
        for parameter, bound_object in zip(
            self._method_task_binding_args(
                method,
                method_library,
                signature=self._task_type_signature(method.task_name, method_library),
            ),
            task_args,
        ):
            expected_type = method_variable_types.get(parameter)
            actual_type = object_types.get(bound_object)
            if expected_type is None or actual_type is None:
                continue
            if self._is_subtype(actual_type, expected_type):
                continue
            mismatches.append(
                f"{parameter}:{bound_object} expected {expected_type}, got {actual_type}",
            )
        return tuple(mismatches)

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
            for parameter, arg in zip(
                self._method_task_binding_args(
                    method,
                    method_library,
                    signature=self._task_type_signature(method.task_name, method_library),
                ),
                task_args,
            )
        }
        object_pool = object_pool if object_pool is not None else list(dict.fromkeys(task_args or objects))
        if not object_pool:
            object_pool = list(task_args)
        if object_types is None:
            object_types = {}
        method_variable_types = self._method_variable_type_hints(method, method_library)
        required_bound_objects = set(bindings.values())
        inferred_candidates: Dict[str, Set[str]] = {}
        if any(obj not in object_types for obj in required_bound_objects):
            inferred_candidates = self._task_argument_type_candidates(
                method.task_name,
                task_args,
                method_library,
            )
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
            if symbol in bindings:
                return bindings[symbol]
            if symbol not in method_variable_types:
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
        def is_schematic(token):
            return bool(token) and (token in local_bindings or token in variable_type_hints or token[0].isupper())

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
                if not is_schematic(token):
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
            if not is_schematic(token):
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

        skipped_mismatches: Dict[str, Tuple[str, ...]] = {}
        for method in task_methods:
            mismatches = self._method_task_argument_type_mismatches(
                method,
                method_library,
                task_args,
                object_types,
            )
            if mismatches:
                skipped_mismatches[method.method_name] = mismatches
                continue
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

        if len(skipped_mismatches) == len(task_methods):
            details = "; ".join(
                f"{method_name}: {', '.join(mismatches)}"
                for method_name, mismatches in sorted(skipped_mismatches.items())
            )
            raise TypeResolutionError(
                f"Stage 4 task '{task_name}' has no type-compatible method branch "
                f"for arguments {tuple(task_args)}. {details}",
            )

        return tuple(facts)

    @staticmethod
    def _transition_witness_initial_facts(initial_facts, target_literal):
        if target_literal is None or not target_literal.is_positive or target_literal.is_equality:
            return tuple(initial_facts)
        target_fact = (
            f"({target_literal.predicate} {' '.join(target_literal.args)})"
            if target_literal.args
            else f"({target_literal.predicate})"
        )
        return tuple(
            fact
            for fact in initial_facts
            if fact != target_fact
        )

    def _build_query_execution_context(
        self,
        ltl_spec,
        method_library,
    ) -> QueryExecutionContext:
        """Build all query-time bindings outside the Stage 3 domain artifact."""

        query_text = str(getattr(ltl_spec, "source_instruction", "") or "")
        literal_signatures = tuple(
            str(signature).strip()
            for signature in (getattr(ltl_spec, "query_task_literal_signatures", ()) or ())
            if str(signature).strip()
        )
        query_object_inventory = tuple(getattr(ltl_spec, "query_object_inventory", ()) or ())
        if not query_object_inventory:
            query_object_inventory = self._extract_query_object_inventory(query_text)

        query_task_name_map = self._method_library_source_task_name_map(method_library)
        variable_assignments: Dict[str, str] = {}
        query_task_anchors: List[Dict[str, Any]] = []
        query_task_network: List[Tuple[str, Tuple[str, ...]]] = []
        target_literals: List[HTNLiteral] = []
        target_task_bindings: List[Any] = []

        for index, anchor in enumerate(self._extract_query_task_anchors(query_text), start=1):
            source_task_name = str(anchor.get("task_name") or "").strip()
            resolved_task_name = query_task_name_map.get(source_task_name, source_task_name)
            literal_signature = (
                literal_signatures[index - 1]
                if index - 1 < len(literal_signatures)
                else None
            )
            grounded_args = self._stage6_ground_query_task_arguments(
                task_name=resolved_task_name,
                task_args=tuple(
                    str(arg).strip()
                    for arg in (anchor.get("args") or ())
                    if str(arg).strip()
                ),
                literal_signature=literal_signature,
                method_library=method_library,
                query_object_inventory=query_object_inventory,
                variable_assignments=variable_assignments,
            )
            grounded_anchor = {
                **dict(anchor),
                "task_name": source_task_name,
                "resolved_task_name": resolved_task_name,
                "args": list(grounded_args),
                "literal_signature": literal_signature,
            }
            query_task_anchors.append(grounded_anchor)
            query_task_network.append(
                (
                    query_root_alias_task_name(index, source_task_name),
                    tuple(grounded_args),
                )
            )
            if literal_signature:
                literal = _parse_signature_literal(literal_signature)
                if literal is not None:
                    target_literals.append(literal)
                    target_task_bindings.append(
                        HTNTargetTaskBinding(
                            target_literal=literal_signature,
                            task_name=resolved_task_name,
                        )
                    )

        query_objects = tuple(
            object_name
            for entry in query_object_inventory
            for object_name in entry.get("objects", ())
        )
        return QueryExecutionContext(
            query_text=query_text,
            ordered_query_sequence=self._query_task_sequence_is_ordered(ltl_spec),
            query_task_anchors=tuple(query_task_anchors),
            query_task_network=tuple(query_task_network),
            query_task_name_map=query_task_name_map,
            target_literals=tuple(target_literals),
            target_task_bindings=tuple(target_task_bindings),
            literal_signatures=literal_signatures,
            query_object_inventory=query_object_inventory,
            query_objects=query_objects,
            typed_objects=dict(self._stage5_query_typed_objects(ltl_spec) or ()),
        )

    def _stage5_agentspeak_rendering(
        self,
        ltl_spec,
        dfa_result,
        method_library,
        plan_records,
        *,
        query_context: Optional[QueryExecutionContext] = None,
    ):
        """Stage 5: Stage 3 methods + raw DFA -> runnable AgentSpeak code."""
        print("\n[STAGE 5] AgentSpeak Rendering")
        print("-"*80)
        stage_start = time.perf_counter()

        try:
            renderer = AgentSpeakRenderer()
            if query_context is None:
                query_context = self._build_query_execution_context(ltl_spec, method_library)
            typed_objects = (
                tuple(query_context.typed_objects.items())
                if query_context.typed_objects
                else self._stage5_query_typed_objects(ltl_spec)
            )
            transition_specs_start = time.perf_counter()
            transition_specs = build_agentspeak_transition_specs(
                dfa_result=dfa_result,
                grounding_map=getattr(ltl_spec, "grounding_map", None),
                ordered_query_sequence=self._query_task_sequence_is_ordered(ltl_spec),
            )
            transition_specs_seconds = time.perf_counter() - transition_specs_start
            self._latest_transition_specs = tuple(transition_specs)
            query_anchor_setup_start = time.perf_counter()
            query_task_name_map = dict(query_context.query_task_name_map)
            query_task_anchors = [
                {
                    key: value
                    for key, value in dict(anchor).items()
                    if key != "resolved_task_name"
                }
                for anchor in query_context.query_task_anchors
            ]
            query_anchor_setup_seconds = time.perf_counter() - query_anchor_setup_start
            render_start = time.perf_counter()
            asl_code = renderer.generate(
                domain=self.domain,
                objects=ltl_spec.objects,
                typed_objects=typed_objects,
                method_library=method_library,
                plan_records=plan_records,
                ordered_query_sequence=self._query_task_sequence_is_ordered(ltl_spec),
                prompt_analysis=self._latest_transition_prompt_analysis,
                transition_specs=transition_specs,
                query_task_anchors=query_task_anchors,
                query_task_name_map=query_task_name_map,
            )
            render_seconds = time.perf_counter() - render_start
            metadata = {
                "transition_count": len(transition_specs),
                "rendered_methods": len(method_library.methods),
                "code_size_chars": len(asl_code),
                "query_task_count": len(query_task_anchors),
            }
            self.logger.log_stage5_agentspeak_rendering(
                asl_code,
                "Success",
                metadata=metadata,
            )

            print(f"✓ AgentSpeak rendering complete ({len(asl_code)} characters)")
            print(f"  Raw DFA transitions rendered: {len(transition_specs)}")
            print("\n  First 10 lines of generated code:")
            for index, line in enumerate(asl_code.split("\n")[:10], start=1):
                if line.strip():
                    print(f"    {index:2d}. {line}")

            output_file = self.output_dir / "stage5_agentspeak.asl"
            write_start = time.perf_counter()
            output_file.write_text(asl_code)
            self._record_stage_timing(
                "stage5",
                stage_start,
                breakdown={
                    "transition_spec_build_seconds": transition_specs_seconds,
                    "query_anchor_setup_seconds": query_anchor_setup_seconds,
                    "render_seconds": render_seconds,
                    "write_output_seconds": time.perf_counter() - write_start,
                },
                metadata={
                    "transition_count": len(transition_specs),
                    "rendered_methods": len(method_library.methods),
                },
            )
            print(f"\n  ✓ Complete AgentSpeak code saved to: {output_file}")

            return asl_code, metadata

        except Exception as e:
            self._record_stage_timing("stage5", stage_start)
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

    def _stage5_lowering_runtime_objects(self, objects, plan_records, seed_facts):
        runtime_objects: List[str] = list(dict.fromkeys(objects or ()))
        for record in tuple(plan_records or ()):
            for obj in tuple(record.get("objects") or ()):
                if obj not in runtime_objects:
                    runtime_objects.append(obj)
        for fact in tuple(seed_facts or ()):
            parsed = self._parse_positive_hddl_fact(fact)
            if parsed is None:
                continue
            _, args = parsed
            for arg in args:
                if arg not in runtime_objects:
                    runtime_objects.append(arg)
        return tuple(runtime_objects)

    def _stage5_augmented_seed_facts_from_plan_records(self, seed_facts, plan_records):
        facts: List[str] = []
        seen_facts: Set[str] = set()
        for fact in tuple(seed_facts or ()):
            rendered = str(fact).strip()
            if not rendered or rendered in seen_facts:
                continue
            seen_facts.add(rendered)
            facts.append(rendered)

        if not plan_records:
            return tuple(facts)

        action_schemas = self._stage6_action_schemas()
        if not action_schemas:
            return tuple(facts)

        runner = JasonRunner()
        predicate_name_map = getattr(self, "predicate_name_map", None)
        if predicate_name_map is None:
            predicate_name_map = runner._runtime_predicate_name_map(
                action_schemas=action_schemas,
                predicate_names=tuple(getattr(self, "predicate_type_map", {}).keys()),
            )
        replay_cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Dict[str, Any]] = {}
        for record in tuple(plan_records or ()):
            plan = record.get("plan") if isinstance(record, dict) else None
            action_path = tuple(
                JasonRunner._runtime_call(step.task_name, step.args)
                for step in getattr(plan, "steps", ()) or ()
            )
            initial_facts = tuple(
                str(fact).strip()
                for fact in (record.get("initial_facts") or ())
                if str(fact).strip()
            )
            if not action_path or not initial_facts:
                continue

            replay_key = (initial_facts, action_path)
            replay = replay_cache.get(replay_key)
            if replay is None:
                replay = runner._replay_action_path_against_schemas(
                    action_path=action_path,
                    action_schemas=action_schemas,
                    seed_facts=initial_facts,
                )
                replay_cache[replay_key] = replay
            if replay.get("passed") is not True:
                continue

            final_facts = runner._runtime_world_to_hddl_facts(
                replay.get("world_facts") or (),
                predicate_name_map=predicate_name_map,
            )
            for fact in final_facts:
                rendered = str(fact).strip()
                if not rendered or rendered in seen_facts:
                    continue
                seen_facts.add(rendered)
                facts.append(rendered)

        return tuple(facts)

    def _stage5_lowering_object_types(
        self,
        objects,
        plan_records,
        method_library,
        seed_facts,
    ):
        object_types: Dict[str, str] = {}
        for record in tuple(plan_records or ()):
            for obj, type_name in dict(record.get("object_types") or {}).items():
                object_types.setdefault(str(obj), str(type_name))
        inferred_object_types = self._stage6_object_types(
            objects,
            method_library,
            seed_facts,
            problem_object_types=object_types,
        )
        for obj, type_name in inferred_object_types.items():
            object_types.setdefault(str(obj), str(type_name))
        return object_types

    @staticmethod
    def _stage6_should_try_runtime_before_guided(
        method_library,
        *,
        ordered_query_sequence: bool = False,
    ) -> bool:
        if ordered_query_sequence:
            return False
        if method_library is None:
            return False
        compound_names = {
            str(getattr(task, "name", "") or "").strip()
            for task in getattr(method_library, "compound_tasks", ()) or ()
        }
        for task in getattr(method_library, "compound_tasks", ()) or ():
            task_name = str(getattr(task, "name", "") or "").strip()
            source_name = str(getattr(task, "source_name", "") or "").strip()
            if not task_name or not source_name or source_name == task_name:
                continue
            for method in method_library.methods_for_task(task_name):
                subtasks = tuple(getattr(method, "subtasks", ()) or ())
                if len(subtasks) != 1:
                    continue
                child = subtasks[0]
                if getattr(child, "kind", "") != "compound":
                    continue
                if str(getattr(child, "task_name", "") or "").strip() in compound_names:
                    return True
        return False

    def _stage6_jason_validation(
        self,
        ltl_spec,
        method_library,
        plan_records,
        stage4_data=None,
        asl_code="",
        query_context: Optional[QueryExecutionContext] = None,
    ):
        """Stage 6: run generated AgentSpeak with Jason (RunLocalMAS)."""
        print("\n[STAGE 6] Jason Runtime Validation")
        print("-"*80)
        stage_start = time.perf_counter()

        try:
            stage6_dir = Path(__file__).parent / "stage6_jason_validation"
            runner = JasonRunner(
                stage6_dir=stage6_dir,
                timeout_seconds=int(self.config.stage6_jason_timeout),
            )
            preparation_start = time.perf_counter()
            if query_context is None:
                query_context = self._build_query_execution_context(ltl_spec, method_library)
            seed_facts, seed_transition = self._stage6_runtime_seed_facts(
                plan_records,
                query_context.target_literals,
            )
            stage6_objects = self._stage6_runtime_objects(ltl_spec.objects, seed_facts)
            stage6_object_types = self._stage6_object_types(
                stage6_objects,
                method_library,
                seed_facts,
                problem_object_types=self.problem.object_types if self.problem is not None else None,
            )
            action_schemas = self._stage6_action_schemas()
            lowering_start = time.perf_counter()
            runtime_ready_asl_code = ASLMethodLowering().compile_method_plans(
                asl_code,
                seed_facts=seed_facts,
                runtime_objects=stage6_objects,
                object_types=stage6_object_types,
                type_parent_map=self.type_parent_map,
                method_library=method_library,
            )
            runtime_ready_asl_code = self._stage6_prune_stale_lowered_method_chunks(
                runtime_ready_asl_code,
            )
            lowering_seconds = time.perf_counter() - lowering_start
            preparation_seconds = time.perf_counter() - preparation_start
            ordered_query_sequence = self._query_task_sequence_is_ordered(ltl_spec)
            protected_target_literals = (
                self._stage6_protected_target_literals(
                    ltl_spec=ltl_spec,
                    method_library=method_library,
                    action_schemas=action_schemas,
                    query_context=query_context,
                )
                if ordered_query_sequence
                else ()
            )
            validation_start = time.perf_counter()
            result = runner.validate(
                agentspeak_code=runtime_ready_asl_code,
                target_literals=query_context.target_literals,
                protected_target_literals=protected_target_literals,
                method_library=method_library,
                action_schemas=action_schemas,
                seed_facts=seed_facts,
                runtime_objects=stage6_objects,
                object_types=stage6_object_types,
                type_parent_map=self.type_parent_map,
                domain_name=self.domain.name,
                problem_file=self.problem_file,
                output_dir=self.output_dir,
                completion_mode="target_literals",
                ordered_query_sequence=ordered_query_sequence,
            )
            validation_seconds = time.perf_counter() - validation_start
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
                "target_literal_count": len(query_context.target_literals),
                "protected_target_literal_count": len(protected_target_literals),
                "seed_fact_count": len(seed_facts),
                "seed_transition": seed_transition,
                "executed_action_count": len(result.action_path),
                "action_path_artifact": result.artifacts.get("action_path"),
                "runtime_object_count": len(stage6_objects),
                "resolved_object_types": stage6_object_types,
                "action_schema_count": len(action_schemas),
                "environment_adapter": result.environment_adapter,
                "failure_class": result.failure_class,
                "consistency_checks": result.consistency_checks,
            }
            stage6_breakdown = {
                "preparation_seconds": preparation_seconds,
                "lowering_seconds": lowering_seconds,
                "runtime_validation_seconds": validation_seconds,
            }
            stage6_breakdown.update(
                self._timing_breakdown_without_total(result.timing_profile),
            )
            self._record_stage_timing(
                "stage6",
                stage_start,
                breakdown=stage6_breakdown,
                metadata={},
            )
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
                    "failure_class": metadata.get("failure_class"),
                    "consistency_checks": metadata.get("consistency_checks"),
                })
            self._record_stage_timing(
                "stage6",
                stage_start,
                breakdown=self._timing_breakdown_without_total(metadata.get("timing_profile")),
                metadata={
                    "failure_class": metadata.get("failure_class"),
                },
            )
            self.logger.log_stage6_jason_validation(
                metadata if metadata else None,
                "Failed",
                error=str(e),
                metadata=summary,
            )
            print(f"✗ Stage 6 Failed: {e}")
            return None
        except Exception as e:
            self._record_stage_timing("stage6", stage_start)
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

    @staticmethod
    def _stage6_prune_stale_lowered_method_chunks(agentspeak_code: str) -> str:
        start_marker = "/* HTN Method Plans */"
        end_marker = "/* DFA Transition Wrappers */"
        start_index = agentspeak_code.find(start_marker)
        end_index = agentspeak_code.find(end_marker)
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return agentspeak_code

        prefix = agentspeak_code[:start_index]
        section = agentspeak_code[start_index:end_index]
        suffix = agentspeak_code[end_index:]
        section_lines = section.splitlines()
        if not section_lines:
            return agentspeak_code

        header = section_lines[0]
        chunks: List[List[str]] = []
        current_chunk: List[str] = []
        for line in section_lines[1:]:
            if not line.strip():
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                continue
            current_chunk.append(line)
        if current_chunk:
            chunks.append(current_chunk)

        preserved_chunks = [
            "\n".join(chunk)
            for chunk in chunks
            if not LTL_BDI_Pipeline._stage6_lowered_chunk_introduces_late_ground_constants(chunk)
        ]
        rewritten_section = "\n\n".join([header, *preserved_chunks]).rstrip() + "\n\n"
        return f"{prefix}{rewritten_section}{suffix}"

    @staticmethod
    def _stage6_lowered_chunk_introduces_late_ground_constants(chunk: Sequence[str]) -> bool:
        if not chunk:
            return False
        head_line = str(chunk[0]).strip()
        head_match = re.match(r"^\+![^(]+\((.*)\)\s*:", head_line)
        if head_match is None:
            return False
        head_args = {
            str(arg).strip()
            for arg in ASLMethodLowering._split_asl_arguments(head_match.group(1))
            if str(arg).strip()
        }
        introduced_constants: Set[str] = set()
        substantive_goal_index = 0
        for raw_line in chunk[1:]:
            line = str(raw_line).strip().rstrip(";.")
            if not line.startswith("!"):
                continue
            call_match = re.match(r"^!([a-z][a-z0-9_]*)\((.*)\)$", line)
            if call_match is None:
                continue
            call_args = tuple(
                str(arg).strip()
                for arg in ASLMethodLowering._split_asl_arguments(call_match.group(2))
                if str(arg).strip()
            )
            extra_constants = {
                arg
                for arg in call_args
                if arg not in head_args and not LTL_BDI_Pipeline._stage6_is_asl_variable_symbol(arg)
            }
            if substantive_goal_index > 0 and any(
                constant not in introduced_constants
                for constant in extra_constants
            ):
                return True
            introduced_constants.update(extra_constants)
            substantive_goal_index += 1
        return False

    @staticmethod
    def _stage6_is_asl_variable_symbol(token: str) -> bool:
        rendered = str(token or "").strip()
        return bool(rendered) and (rendered[0].isupper() or rendered[0] == "_")

    def _stage7_official_verification(self, ltl_spec, method_library, stage6_data):
        """Stage 7: verify the generated hierarchical plan with the official IPC verifier."""
        print("\n[STAGE 7] Official IPC HTN Plan Verification")
        print("-"*80)
        stage_start = time.perf_counter()

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
            self._record_stage_timing("stage7", stage_start)
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
            self._record_stage_timing("stage7", stage_start)
            print(f"✗ Stage 7 Failed: {error}")
            return None

        stage6_artifacts = stage6_data.get("artifacts") or {}
        planning_mode = str(stage6_artifacts.get("planning_mode") or "")
        if planning_mode == "official_problem_root":
            verification_domain_file = Path(self.domain_file).resolve()
            domain_build_seconds = 0.0
        else:
            domain_build_start = time.perf_counter()
            verification_domain_file = self._stage7_build_verification_domain(method_library)
            domain_build_seconds = time.perf_counter() - domain_build_start
        guided_plan_text = stage6_artifacts.get("guided_hierarchical_plan_text")
        verifier_start = time.perf_counter()
        if guided_plan_text:
            guided_plan_text = self._stage7_rewrite_guided_plan_source_names(
                guided_plan_text,
                method_library,
            )
            verifier_result = verifier.verify_plan_text(
                domain_file=verification_domain_file,
                problem_file=self.problem_file,
                plan_text=guided_plan_text,
                output_dir=self.output_dir,
                plan_kind="hierarchical",
                build_warning=None,
            )
        else:
            verifier_result = verifier.verify_plan(
                domain_file=verification_domain_file,
                problem_file=self.problem_file,
                action_path=stage6_artifacts.get("action_path") or [],
                method_library=method_library,
                method_trace=stage6_artifacts.get("method_trace") or [],
                output_dir=self.output_dir,
            )
        verifier_seconds = time.perf_counter() - verifier_start
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
            "verification_domain_file": str(verification_domain_file),
            "verification_problem_file": str(Path(self.problem_file).resolve()),
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
            self._record_stage_timing(
                "stage7",
                stage_start,
                breakdown={
                    "verification_domain_build_seconds": domain_build_seconds,
                    "official_verifier_seconds": verifier_seconds,
                },
                metadata={
                    "plan_kind": verifier_result.plan_kind,
                    "verification_result": verifier_result.verification_result,
                },
            )
            print(f"✗ Stage 7 Failed: {error}")
            return None

        self.logger.log_stage7_official_verification(
            artifacts,
            "Success",
            metadata=summary,
        )
        self._record_stage_timing(
            "stage7",
            stage_start,
            breakdown={
                "verification_domain_build_seconds": domain_build_seconds,
                "official_verifier_seconds": verifier_seconds,
            },
            metadata={
                "plan_kind": verifier_result.plan_kind,
                "verification_result": verifier_result.verification_result,
            },
        )
        print("✓ Official IPC verification complete")
        print(f"  Plan kind: {verifier_result.plan_kind}")
        print(f"  Verification result: {verifier_result.verification_result}")
        print(f"  Verifier output: {artifacts.get('output_file')}")

        return {
            "summary": summary,
            "artifacts": artifacts,
        }

    @staticmethod
    def _stage7_rewrite_guided_plan_source_names(plan_text, method_library):
        task_name_map = {
            task.name: str(task.source_name).strip()
            for task in getattr(method_library, "compound_tasks", ())
            if str(getattr(task, "source_name", "") or "").strip()
        }
        if not task_name_map:
            return plan_text

        trailing_newline = str(plan_text).endswith("\n")
        rewritten_lines = []
        for raw_line in str(plan_text).splitlines():
            stripped = raw_line.strip()
            if "->" not in stripped:
                rewritten_lines.append(raw_line)
                continue
            before, _, after = stripped.partition("->")
            head_tokens = before.split()
            if len(head_tokens) < 2 or not head_tokens[0].isdigit():
                rewritten_lines.append(raw_line)
                continue
            source_name = task_name_map.get(head_tokens[1])
            if not source_name:
                rewritten_lines.append(raw_line)
                continue
            head_tokens[1] = source_name
            leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            rewritten_lines.append(
                f"{leading}{' '.join(head_tokens)} -> {after.strip()}".rstrip(),
            )

        rewritten = "\n".join(rewritten_lines)
        if trailing_newline:
            rewritten += "\n"
        return rewritten

    def _stage7_build_verification_domain(self, method_library):
        planner = PANDAPlanner()
        verification_domain_hddl = planner._build_domain_hddl(
            self.domain,
            method_library,
            self.domain.name,
            export_source_names=True,
        )
        verification_domain_path = self.output_dir / "ipc_verification_domain.hddl"
        verification_domain_path.write_text(verification_domain_hddl)
        return verification_domain_path

    def _stage6_runtime_seed_facts(self, plan_records, target_literals):
        if self.problem is not None:
            return self._stage6_problem_seed_facts()
        return self._stage6_seed_facts(plan_records, target_literals)

    def _stage6_query_task_network(
        self,
        ltl_spec,
        method_library=None,
        query_context: Optional[QueryExecutionContext] = None,
    ):
        if query_context is not None and query_context.query_task_network:
            return tuple(query_context.query_task_network)
        query_text = getattr(ltl_spec, "source_instruction", "") or ""
        query_task_anchors = self._extract_query_task_anchors(query_text)
        literal_signatures = tuple(
            str(signature).strip()
            for signature in (getattr(ltl_spec, "query_task_literal_signatures", ()) or ())
            if str(signature).strip()
        )
        query_object_inventory = tuple(getattr(ltl_spec, "query_object_inventory", ()) or ())
        if not query_object_inventory:
            query_object_inventory = self._extract_query_object_inventory(query_text)

        task_network = []
        variable_assignments: Dict[str, str] = {}
        for anchor in query_task_anchors:
            source_task_name = str(anchor.get("task_name", "")).strip()
            raw_task_args = tuple(str(arg).strip() for arg in (anchor.get("args") or ()))
            if not source_task_name or any(not arg for arg in raw_task_args):
                return ()
            task_name = query_root_alias_task_name(len(task_network) + 1, source_task_name)
            task_args = self._stage6_ground_query_task_arguments(
                task_name=task_name,
                task_args=raw_task_args,
                literal_signature=(
                    literal_signatures[len(task_network)]
                    if len(task_network) < len(literal_signatures)
                    else None
                ),
                method_library=method_library,
                query_object_inventory=query_object_inventory,
                variable_assignments=variable_assignments,
            )
            if any(not arg for arg in task_args):
                return ()
            if any(self._is_query_variable_symbol(arg) for arg in task_args):
                return ()
            task_network.append((task_name, task_args))
        return tuple(task_network)

    def _stage6_ground_query_task_arguments(
        self,
        *,
        task_name: str,
        task_args: Sequence[str],
        literal_signature: Optional[str],
        method_library,
        query_object_inventory: Sequence[Dict[str, Any]],
        variable_assignments: Dict[str, str],
    ) -> Tuple[str, ...]:
        grounded_args = tuple(str(arg).strip() for arg in (task_args or ()))
        if not any(self._is_query_variable_symbol(arg) for arg in grounded_args):
            return grounded_args

        literal = self._stage6_parse_literal_signature(literal_signature or "")
        inferred_args = self._stage6_infer_task_args_from_literal(
            task_name=task_name,
            task_args=grounded_args,
            literal=literal,
            method_library=method_library,
        )
        if inferred_args is not None:
            grounded_args = self._stage6_bind_query_variables(
                task_args=grounded_args,
                candidate_args=inferred_args,
                variable_assignments=variable_assignments,
            )

        if any(self._is_query_variable_symbol(arg) for arg in grounded_args):
            grounded_args = self._ground_query_task_arguments(
                task_name=task_name,
                task_args=grounded_args,
                query_object_inventory=query_object_inventory,
                variable_assignments=variable_assignments,
            )
        return grounded_args

    def _stage6_infer_task_args_from_literal(
        self,
        *,
        task_name: str,
        task_args: Sequence[str],
        literal: Optional[HTNLiteral],
        method_library,
    ) -> Optional[Tuple[str, ...]]:
        if literal is None:
            return None

        literal_args = tuple(str(arg).strip() for arg in (literal.args or ()))
        if not literal_args:
            return None
        if len(literal_args) == len(task_args):
            return literal_args

        task_types = tuple(self._task_type_signature(task_name, method_library))
        predicate_types = tuple(self.predicate_type_map.get(literal.predicate, ()))
        if len(task_types) != len(task_args) or len(predicate_types) != len(literal_args):
            return None
        return self._stage6_project_args_by_type(
            literal_args,
            predicate_types,
            task_types,
        )

    def _stage6_bind_query_variables(
        self,
        *,
        task_args: Sequence[str],
        candidate_args: Sequence[str],
        variable_assignments: Dict[str, str],
    ) -> Tuple[str, ...]:
        if len(task_args) != len(candidate_args):
            return tuple(str(arg).strip() for arg in (task_args or ()))

        grounded: List[str] = []
        for task_arg, candidate_arg in zip(task_args, candidate_args):
            raw_task_arg = str(task_arg).strip()
            raw_candidate_arg = str(candidate_arg).strip()
            if not self._is_query_variable_symbol(raw_task_arg):
                grounded.append(raw_task_arg)
                continue
            if not raw_candidate_arg:
                grounded.append(raw_task_arg)
                continue
            existing = variable_assignments.get(raw_task_arg)
            if existing is not None:
                grounded.append(existing if existing == raw_candidate_arg else raw_task_arg)
                continue
            variable_assignments[raw_task_arg] = raw_candidate_arg
            grounded.append(raw_candidate_arg)
        return tuple(grounded)

    def _stage6_query_goal_facts(
        self,
        ltl_spec,
        task_network,
        method_library,
        action_schemas,
    ):
        if not task_network:
            return ()

        action_schema_lookup = self._stage6_action_schema_lookup(action_schemas)
        guaranteed_effect_cache: Dict[str, Tuple[HTNLiteral, ...]] = {}
        possible_negative_effect_cache: Dict[str, Tuple[HTNLiteral, ...]] = {}
        self._stage6_populate_task_effect_caches(
            method_library=method_library,
            action_schema_lookup=action_schema_lookup,
            guaranteed_effect_cache=guaranteed_effect_cache,
            possible_negative_effect_cache=possible_negative_effect_cache,
        )
        projected_world: Dict[Tuple[str, Tuple[str, ...]], HTNLiteral] = {}
        headline_literals = self._stage6_query_task_headline_literals(
            ltl_spec,
            task_network,
            method_library,
        )

        for index, (task_name, task_args) in enumerate(task_network):
            headline_literal = headline_literals[index] if index < len(headline_literals) else None
            for literal in self._stage6_ground_task_effects(
                task_name,
                task_args,
                method_library,
                action_schema_lookup,
                guaranteed_effect_cache,
            ):
                if literal.is_equality:
                    continue
                effect_key = self._stage6_effect_key(literal)
                if literal.is_positive and headline_literal is None:
                    projected_world[effect_key] = literal

            for literal in self._stage6_ground_task_negative_effects(
                task_name,
                task_args,
                method_library,
                action_schema_lookup,
                possible_negative_effect_cache,
            ):
                if literal.is_equality:
                    continue
                projected_world.pop(self._stage6_effect_key(literal), None)

            if headline_literal is not None:
                projected_world[self._stage6_effect_key(headline_literal)] = headline_literal

        if not projected_world:
            return ()

        return tuple(
            self._render_problem_fact(literal)
            for literal in sorted(
                (
                    literal
                    for literal in projected_world.values()
                    if literal.is_positive
                ),
                key=lambda item: (item.predicate, tuple(item.args)),
            )
        )

    def _stage6_protected_target_literals(
        self,
        *,
        ltl_spec,
        method_library,
        action_schemas,
        query_context: Optional[QueryExecutionContext] = None,
    ):
        target_literals = (
            tuple(query_context.target_literals)
            if query_context is not None
            else tuple(getattr(method_library, "target_literals", ()) or ())
        )
        if not target_literals:
            return ()

        task_network = self._stage6_query_task_network(
            ltl_spec,
            method_library,
            query_context=query_context,
        )
        if not task_network:
            return ()

        goal_signatures: Set[str] = set()
        for fact in self._stage6_query_goal_facts(
            ltl_spec=ltl_spec,
            task_network=task_network,
            method_library=method_library,
            action_schemas=action_schemas,
        ):
            parsed = self._parse_positive_hddl_fact(fact)
            if parsed is None:
                continue
            predicate, args = parsed
            goal_signatures.add(
                HTNLiteral(
                    predicate=predicate,
                    args=tuple(args),
                    is_positive=True,
                ).to_signature(),
            )

        if not goal_signatures:
            return ()

        return tuple(
            literal
            for literal in target_literals
            if (
                literal.is_positive
                and not literal.is_equality
                and literal.to_signature() in goal_signatures
            )
        )

    @staticmethod
    def _stage6_effect_key(literal: HTNLiteral) -> Tuple[str, Tuple[str, ...]]:
        return literal.predicate, tuple(literal.args)

    def _stage6_action_schema_lookup(self, action_schemas):
        lookup: Dict[str, Dict[str, Any]] = {}
        for schema in action_schemas or ():
            if not isinstance(schema, dict):
                continue
            for key in (
                str(schema.get("functor") or "").strip(),
                str(schema.get("source_name") or "").strip(),
            ):
                if not key:
                    continue
                lookup[key] = schema
                lookup[self._sanitize_name(key)] = schema
        return lookup

    def _stage6_populate_task_effect_caches(
        self,
        *,
        method_library,
        action_schema_lookup,
        guaranteed_effect_cache: Dict[str, Tuple[HTNLiteral, ...]],
        possible_negative_effect_cache: Dict[str, Tuple[HTNLiteral, ...]],
    ) -> None:
        compound_tasks = tuple(getattr(method_library, "compound_tasks", ()) or ())
        if not compound_tasks:
            return

        task_names = [
            str(getattr(task, "name", "") or "").strip()
            for task in compound_tasks
            if str(getattr(task, "name", "") or "").strip()
        ]
        if not task_names:
            return

        callers_by_child: Dict[str, Set[str]] = defaultdict(set)
        for method in tuple(getattr(method_library, "methods", ()) or ()):
            parent_name = str(getattr(method, "task_name", "") or "").strip()
            if not parent_name:
                continue
            for step in tuple(getattr(method, "subtasks", ()) or ()):
                if str(getattr(step, "kind", "") or "").strip() != "compound":
                    continue
                child_name = str(getattr(step, "task_name", "") or "").strip()
                if child_name:
                    callers_by_child[child_name].add(parent_name)

        worklist = deque(task_names)
        queued = set(task_names)
        while worklist:
            task_name = worklist.popleft()
            queued.discard(task_name)

            guaranteed = self._stage6_task_guaranteed_effects_from_snapshot(
                task_name,
                method_library,
                action_schema_lookup,
                guaranteed_effect_cache,
            )
            negatives = self._stage6_task_possible_negative_effects_from_snapshot(
                task_name,
                method_library,
                action_schema_lookup,
                guaranteed_effect_cache,
            )

            guaranteed_changed = guaranteed_effect_cache.get(task_name) != guaranteed
            negative_changed = possible_negative_effect_cache.get(task_name) != negatives
            if not guaranteed_changed and not negative_changed:
                continue

            guaranteed_effect_cache[task_name] = guaranteed
            possible_negative_effect_cache[task_name] = negatives
            for caller_name in sorted(callers_by_child.get(task_name, ())):
                if caller_name in queued:
                    continue
                worklist.append(caller_name)
                queued.add(caller_name)

    def _stage6_task_guaranteed_effects_from_snapshot(
        self,
        task_name,
        method_library,
        action_schema_lookup,
        guaranteed_effect_cache,
    ) -> Tuple[HTNLiteral, ...]:
        methods = list(method_library.methods_for_task(task_name))
        if not methods:
            return ()

        per_method_effects: List[Dict[Tuple[str, Tuple[str, ...], bool], HTNLiteral]] = []
        for method in methods:
            effect_map: Dict[Tuple[str, Tuple[str, ...], bool], HTNLiteral] = {}
            for literal in self._stage6_method_net_effects_from_snapshot(
                method,
                method_library,
                action_schema_lookup,
                guaranteed_effect_cache,
            ):
                effect_map[(literal.predicate, tuple(literal.args), literal.is_positive)] = literal
            per_method_effects.append(effect_map)

        if not per_method_effects:
            return ()

        common_keys = set(per_method_effects[0].keys())
        for effect_map in per_method_effects[1:]:
            common_keys &= set(effect_map.keys())

        return tuple(
            per_method_effects[0][key]
            for key in sorted(common_keys, key=lambda item: (item[0], item[1], item[2]))
        )

    def _stage6_task_possible_negative_effects_from_snapshot(
        self,
        task_name,
        method_library,
        action_schema_lookup,
        guaranteed_effect_cache,
    ) -> Tuple[HTNLiteral, ...]:
        methods = list(method_library.methods_for_task(task_name))
        if not methods:
            return ()

        negative_effects: Dict[Tuple[str, Tuple[str, ...], bool], HTNLiteral] = {}
        for method in methods:
            for literal in self._stage6_method_net_effects_from_snapshot(
                method,
                method_library,
                action_schema_lookup,
                guaranteed_effect_cache,
            ):
                if literal.is_positive:
                    continue
                negative_effects[(literal.predicate, tuple(literal.args), literal.is_positive)] = (
                    literal
                )

        return tuple(
            negative_effects[key]
            for key in sorted(negative_effects, key=lambda item: (item[0], item[1], item[2]))
        )

    def _stage6_method_net_effects_from_snapshot(
        self,
        method,
        method_library,
        action_schema_lookup,
        guaranteed_effect_cache,
    ) -> Tuple[HTNLiteral, ...]:
        task_schema = method_library.task_for_name(method.task_name)
        if task_schema is None:
            return ()

        task_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
        binding_args = self._method_task_binding_args(
            method,
            method_library,
            signature=task_parameters,
        )
        task_bindings = {
            binding_arg: task_parameter
            for binding_arg, task_parameter in zip(binding_args, task_parameters)
            if binding_arg
            and (
                binding_arg in getattr(method, "parameters", ())
                or binding_arg.startswith("?")
                or self._is_variable_symbol(binding_arg)
            )
        }

        net_effects: Dict[Tuple[str, Tuple[str, ...]], HTNLiteral] = {}
        for step in self._stage6_ordered_method_steps(method):
            for literal in self._stage6_step_effects_from_snapshot(
                step,
                method_library,
                action_schema_lookup,
                guaranteed_effect_cache,
            ):
                if literal.is_equality:
                    continue
                lifted_args: List[str] = []
                for arg in literal.args:
                    if arg in task_bindings:
                        lifted_args.append(task_bindings[arg])
                        continue
                    if (
                        arg in getattr(method, "parameters", ())
                        or arg.startswith("?")
                        or self._is_variable_symbol(arg)
                    ):
                        lifted_args = []
                        break
                    lifted_args.append(arg)
                if not lifted_args and literal.args:
                    continue
                lifted_literal = HTNLiteral(
                    predicate=literal.predicate,
                    args=tuple(lifted_args),
                    is_positive=literal.is_positive,
                    source_symbol=literal.source_symbol,
                    negation_mode=literal.negation_mode,
                )
                net_effects[self._stage6_effect_key(lifted_literal)] = lifted_literal

        return tuple(net_effects.values())

    def _stage6_step_effects_from_snapshot(
        self,
        step,
        method_library,
        action_schema_lookup,
        guaranteed_effect_cache,
    ) -> Tuple[HTNLiteral, ...]:
        effects: List[HTNLiteral] = list(getattr(step, "effects", ()) or ())
        step_kind = getattr(step, "kind", None)

        if step_kind == "primitive":
            action_schema = self._stage6_resolve_action_schema(step, action_schema_lookup)
            if action_schema is not None:
                effects.extend(
                    self._stage6_materialise_action_literals(
                        action_schema.get("effects") or (),
                        action_schema.get("parameters") or (),
                        getattr(step, "args", ()) or (),
                    ),
                )
        elif step_kind == "compound":
            task_schema = method_library.task_for_name(step.task_name)
            child_parameters = tuple(getattr(task_schema, "parameters", ()) or ()) if task_schema else ()
            child_bindings = {
                parameter: arg
                for parameter, arg in zip(child_parameters, getattr(step, "args", ()) or ())
            }
            for literal in tuple(guaranteed_effect_cache.get(str(step.task_name).strip(), ()) or ()):
                effects.append(
                    HTNLiteral(
                        predicate=literal.predicate,
                        args=tuple(child_bindings.get(arg, arg) for arg in literal.args),
                        is_positive=literal.is_positive,
                        source_symbol=literal.source_symbol,
                        negation_mode=literal.negation_mode,
                    ),
                )

        net_effects: Dict[Tuple[str, Tuple[str, ...]], HTNLiteral] = {}
        for literal in effects:
            if literal.is_equality:
                continue
            net_effects[self._stage6_effect_key(literal)] = literal
        return tuple(net_effects.values())

    def _stage6_resolve_effect_task_name(self, task_name, method_library) -> str:
        resolved_task_name = str(task_name or "").strip()
        if not resolved_task_name:
            return ""
        if method_library.task_for_name(resolved_task_name) is not None:
            return resolved_task_name

        alias_match = re.match(r"^query_root_\d+_(.+)$", resolved_task_name)
        if alias_match is None:
            return resolved_task_name

        alias_suffix = str(alias_match.group(1) or "").strip()
        if not alias_suffix:
            return resolved_task_name

        source_task_name_map = self._method_library_source_task_name_map(method_library)
        direct_match = source_task_name_map.get(alias_suffix)
        if direct_match:
            return direct_match

        for task in tuple(getattr(method_library, "compound_tasks", ()) or ()):
            internal_name = str(getattr(task, "name", "") or "").strip()
            source_name = str(getattr(task, "source_name", "") or "").strip()
            candidates = [
                sanitize_identifier(internal_name),
                sanitize_identifier(source_name) if source_name else "",
            ]
            if alias_suffix in {candidate for candidate in candidates if candidate}:
                return internal_name or resolved_task_name
        return resolved_task_name

    def _stage6_ground_task_effects(
        self,
        task_name,
        task_args,
        method_library,
        action_schema_lookup,
        guaranteed_effect_cache,
    ):
        task_name = self._stage6_resolve_effect_task_name(task_name, method_library)
        task_schema = method_library.task_for_name(task_name)
        if task_schema is None:
            return ()

        parameter_bindings = {
            parameter: arg
            for parameter, arg in zip(tuple(task_schema.parameters or ()), tuple(task_args or ()))
        }
        grounded: List[HTNLiteral] = []
        for literal in self._stage6_task_guaranteed_effects(
            task_name,
            method_library,
            action_schema_lookup,
            guaranteed_effect_cache,
        ):
            grounded.append(
                HTNLiteral(
                    predicate=literal.predicate,
                    args=tuple(parameter_bindings.get(arg, arg) for arg in literal.args),
                    is_positive=literal.is_positive,
                    source_symbol=literal.source_symbol,
                    negation_mode=literal.negation_mode,
                ),
            )
        return tuple(grounded)

    def _stage6_ground_task_negative_effects(
        self,
        task_name,
        task_args,
        method_library,
        action_schema_lookup,
        possible_negative_effect_cache,
    ):
        task_name = self._stage6_resolve_effect_task_name(task_name, method_library)
        task_schema = method_library.task_for_name(task_name)
        if task_schema is None:
            return ()

        parameter_bindings = {
            parameter: arg
            for parameter, arg in zip(tuple(task_schema.parameters or ()), tuple(task_args or ()))
        }
        grounded: List[HTNLiteral] = []
        for literal in self._stage6_task_possible_negative_effects(
            task_name,
            method_library,
            action_schema_lookup,
            possible_negative_effect_cache,
        ):
            grounded.append(
                HTNLiteral(
                    predicate=literal.predicate,
                    args=tuple(parameter_bindings.get(arg, arg) for arg in literal.args),
                    is_positive=literal.is_positive,
                    source_symbol=literal.source_symbol,
                    negation_mode=literal.negation_mode,
                ),
            )
        return tuple(grounded)

    def _stage6_query_task_headline_literals(
        self,
        ltl_spec,
        task_network,
        method_library,
    ):
        signatures = list(getattr(ltl_spec, "query_task_literal_signatures", ()) or ())
        if len(signatures) == len(task_network):
            literals = []
            for signature in signatures:
                literal = self._stage6_parse_literal_signature(signature)
                literals.append(literal)
            return tuple(literals)

        headlines: List[Optional[HTNLiteral]] = []
        for task_name, task_args in task_network:
            task_schema = method_library.task_for_name(task_name)
            if task_schema is None:
                headlines.append(None)
                continue
            task_types = tuple(self._task_type_signature(task_name, method_library))
            candidate_literal = None
            for predicate_name in (getattr(task_schema, "source_predicates", ()) or ()):
                predicate_signature = tuple(self.predicate_type_map.get(predicate_name, ()))
                projected_args = self._stage6_project_effect_args(
                    task_args,
                    task_types,
                    predicate_signature,
                )
                if projected_args is None:
                    continue
                candidate_literal = HTNLiteral(
                    predicate=predicate_name,
                    args=projected_args,
                    is_positive=True,
                    source_symbol=None,
                )
                break
            headlines.append(candidate_literal)
        return tuple(headlines)

    @staticmethod
    def _stage6_parse_literal_signature(signature: str) -> Optional[HTNLiteral]:
        token = str(signature or "").strip()
        if not token:
            return None
        is_positive = not token.startswith("!")
        if not is_positive:
            token = token[1:].strip()
        predicate, has_args, args_text = token.partition("(")
        predicate = predicate.strip()
        if not predicate:
            return None
        args: Tuple[str, ...] = ()
        if has_args:
            args = tuple(
                str(arg).strip()
                for arg in args_text.rstrip(")").split(",")
                if str(arg).strip()
            )
        return HTNLiteral(
            predicate=predicate,
            args=args,
            is_positive=is_positive,
            source_symbol=None,
        )

    @staticmethod
    def _stage6_literal_holds_in_seed_facts(
        literal: Optional[HTNLiteral],
        seed_facts: Sequence[str],
    ) -> bool:
        if literal is None:
            return False
        if literal.is_equality:
            if len(literal.args) != 2:
                return False
            equal = literal.args[0] == literal.args[1]
            return equal if literal.is_positive else not equal

        known_positive_facts = {
            parsed
            for parsed in (
                LTL_BDI_Pipeline._parse_positive_hddl_fact(fact)
                for fact in (seed_facts or ())
            )
            if parsed is not None
        }
        fact_signature = (literal.predicate, tuple(literal.args))
        if literal.is_positive:
            return fact_signature in known_positive_facts
        return fact_signature not in known_positive_facts

    @staticmethod
    def _stage6_plan_record_queues_by_literal(
        plan_records,
    ) -> Dict[str, List[Dict[str, Any]]]:
        queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in tuple(plan_records or ()):
            literal = record.get("target_literal")
            if literal is None:
                continue
            signature_getter = getattr(literal, "to_signature", None)
            if not callable(signature_getter):
                continue
            signature = str(signature_getter() or "").strip()
            if not signature:
                continue
            queues[signature].append(record)
        return dict(queues)

    @staticmethod
    def _stage6_pop_plan_record_for_literal(
        plan_record_queues: Dict[str, List[Dict[str, Any]]],
        literal: Optional[HTNLiteral],
    ) -> Optional[Dict[str, Any]]:
        if literal is None:
            return None
        queued = plan_record_queues.get(literal.to_signature())
        if not queued:
            return None
        return queued.pop(0)

    @staticmethod
    def _stage6_plan_record_queues_by_transition_name(
        plan_records,
    ) -> Dict[str, List[Dict[str, Any]]]:
        queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in tuple(plan_records or ()):
            transition_name = str(record.get("transition_name") or "").strip()
            if not transition_name:
                continue
            queues[transition_name].append(record)
        return dict(queues)

    @staticmethod
    def _stage6_pop_plan_record_for_transition_name(
        plan_record_queues: Dict[str, List[Dict[str, Any]]],
        transition_name: str,
    ) -> Optional[Dict[str, Any]]:
        name = str(transition_name or "").strip()
        if not name:
            return None
        queued = plan_record_queues.get(name)
        if not queued:
            return None
        return queued.pop(0)

    @staticmethod
    def _stage6_project_effect_args(
        task_args: Sequence[str],
        task_types: Sequence[str],
        predicate_types: Sequence[str],
    ) -> Optional[Tuple[str, ...]]:
        if not predicate_types:
            return tuple(task_args)
        if len(task_args) != len(task_types):
            if len(task_args) == len(predicate_types):
                return tuple(task_args)
            return None
        if len(task_args) == len(predicate_types):
            projected = LTL_BDI_Pipeline._stage6_project_args_by_type(
                task_args,
                task_types,
                predicate_types,
            )
            return projected or tuple(task_args)
        return LTL_BDI_Pipeline._stage6_project_args_by_type(
            task_args,
            task_types,
            predicate_types,
        )

    @staticmethod
    def _stage6_project_args_by_type(
        task_args: Sequence[str],
        task_types: Sequence[str],
        predicate_types: Sequence[str],
    ) -> Optional[Tuple[str, ...]]:
        if len(task_types) != len(task_args):
            return None

        used_indexes: Set[int] = set()
        projected: List[str] = []
        for required_type in predicate_types:
            candidates = [
                index
                for index, task_type in enumerate(task_types)
                if index not in used_indexes and task_type == required_type
            ]
            if len(candidates) != 1:
                return None
            chosen_index = candidates[0]
            used_indexes.add(chosen_index)
            projected.append(str(task_args[chosen_index]))
        return tuple(projected)

    def _stage6_task_guaranteed_effects(
        self,
        task_name,
        method_library,
        action_schema_lookup,
        guaranteed_effect_cache,
        stack: Tuple[str, ...] = (),
    ):
        if task_name in guaranteed_effect_cache:
            return guaranteed_effect_cache[task_name]
        if task_name in stack:
            return ()

        methods = list(method_library.methods_for_task(task_name))
        if not methods:
            guaranteed_effect_cache[task_name] = ()
            return ()

        per_method_effects: List[Dict[Tuple[str, Tuple[str, ...], bool], HTNLiteral]] = []
        for method in methods:
            effect_map: Dict[Tuple[str, Tuple[str, ...], bool], HTNLiteral] = {}
            for literal in self._stage6_method_net_effects(
                method,
                method_library,
                action_schema_lookup,
                guaranteed_effect_cache,
                stack + (task_name,),
            ):
                effect_map[(literal.predicate, tuple(literal.args), literal.is_positive)] = literal
            per_method_effects.append(effect_map)

        common_keys = set(per_method_effects[0].keys())
        for effect_map in per_method_effects[1:]:
            common_keys &= set(effect_map.keys())

        guaranteed = tuple(
            per_method_effects[0][key]
            for key in sorted(common_keys, key=lambda item: (item[0], item[1], item[2]))
        )
        guaranteed_effect_cache[task_name] = guaranteed
        return guaranteed

    def _stage6_task_possible_negative_effects(
        self,
        task_name,
        method_library,
        action_schema_lookup,
        possible_negative_effect_cache,
        stack: Tuple[str, ...] = (),
    ):
        if task_name in possible_negative_effect_cache:
            return possible_negative_effect_cache[task_name]
        if task_name in stack:
            return ()

        methods = list(method_library.methods_for_task(task_name))
        if not methods:
            possible_negative_effect_cache[task_name] = ()
            return ()

        negative_effects: Dict[Tuple[str, Tuple[str, ...], bool], HTNLiteral] = {}
        for method in methods:
            for literal in self._stage6_method_net_effects(
                method,
                method_library,
                action_schema_lookup,
                {},
                stack + (task_name,),
            ):
                if literal.is_positive:
                    continue
                negative_effects[(literal.predicate, tuple(literal.args), literal.is_positive)] = literal

        negatives = tuple(
            negative_effects[key]
            for key in sorted(negative_effects, key=lambda item: (item[0], item[1], item[2]))
        )
        possible_negative_effect_cache[task_name] = negatives
        return negatives

    def _stage6_method_net_effects(
        self,
        method,
        method_library,
        action_schema_lookup,
        guaranteed_effect_cache,
        stack: Tuple[str, ...] = (),
    ):
        task_schema = method_library.task_for_name(method.task_name)
        if task_schema is None:
            return ()

        task_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
        binding_args = self._method_task_binding_args(
            method,
            method_library,
            signature=task_parameters,
        )
        task_bindings = {
            binding_arg: task_parameter
            for binding_arg, task_parameter in zip(binding_args, task_parameters)
            if binding_arg
            and (
                binding_arg in getattr(method, "parameters", ())
                or binding_arg.startswith("?")
                or self._is_variable_symbol(binding_arg)
            )
        }

        net_effects: Dict[Tuple[str, Tuple[str, ...]], HTNLiteral] = {}
        for step in self._stage6_ordered_method_steps(method):
            for literal in self._stage6_step_effects(
                step,
                method_library,
                action_schema_lookup,
                guaranteed_effect_cache,
                stack,
            ):
                if literal.is_equality:
                    continue
                lifted_args: List[str] = []
                for arg in literal.args:
                    if arg in task_bindings:
                        lifted_args.append(task_bindings[arg])
                        continue
                    if (
                        arg in getattr(method, "parameters", ())
                        or arg.startswith("?")
                        or self._is_variable_symbol(arg)
                    ):
                        lifted_args = []
                        break
                    lifted_args.append(arg)
                if not lifted_args and literal.args:
                    continue
                lifted_literal = HTNLiteral(
                    predicate=literal.predicate,
                    args=tuple(lifted_args),
                    is_positive=literal.is_positive,
                    source_symbol=literal.source_symbol,
                    negation_mode=literal.negation_mode,
                )
                net_effects[self._stage6_effect_key(lifted_literal)] = lifted_literal

        return tuple(net_effects.values())

    def _stage6_step_effects(
        self,
        step,
        method_library,
        action_schema_lookup,
        guaranteed_effect_cache,
        stack: Tuple[str, ...] = (),
    ):
        effects: List[HTNLiteral] = list(getattr(step, "effects", ()) or ())
        step_kind = getattr(step, "kind", None)

        if step_kind == "primitive":
            action_schema = self._stage6_resolve_action_schema(step, action_schema_lookup)
            if action_schema is not None:
                effects.extend(
                    self._stage6_materialise_action_literals(
                        action_schema.get("effects") or (),
                        action_schema.get("parameters") or (),
                        getattr(step, "args", ()) or (),
                    ),
                )
        elif step_kind == "compound":
            task_schema = method_library.task_for_name(step.task_name)
            child_parameters = tuple(getattr(task_schema, "parameters", ()) or ()) if task_schema else ()
            child_bindings = {
                parameter: arg
                for parameter, arg in zip(child_parameters, getattr(step, "args", ()) or ())
            }
            for literal in self._stage6_task_guaranteed_effects(
                step.task_name,
                method_library,
                action_schema_lookup,
                guaranteed_effect_cache,
                stack,
            ):
                effects.append(
                    HTNLiteral(
                        predicate=literal.predicate,
                        args=tuple(child_bindings.get(arg, arg) for arg in literal.args),
                        is_positive=literal.is_positive,
                        source_symbol=literal.source_symbol,
                        negation_mode=literal.negation_mode,
                    ),
                )

        net_effects: Dict[Tuple[str, Tuple[str, ...]], HTNLiteral] = {}
        for literal in effects:
            if literal.is_equality:
                continue
            net_effects[self._stage6_effect_key(literal)] = literal
        return tuple(net_effects.values())

    def _stage6_resolve_action_schema(self, step, action_schema_lookup):
        for candidate in (
            str(getattr(step, "action_name", "") or "").strip(),
            str(getattr(step, "task_name", "") or "").strip(),
            self._sanitize_name(str(getattr(step, "task_name", "") or "").strip()),
        ):
            if candidate and candidate in action_schema_lookup:
                return action_schema_lookup[candidate]
        return None

    @staticmethod
    def _stage6_materialise_action_literals(patterns, schema_parameters, step_args):
        bindings = {
            str(parameter): str(arg)
            for parameter, arg in zip(tuple(schema_parameters or ()), tuple(step_args or ()))
        }
        materialised: List[HTNLiteral] = []
        for pattern in patterns or ():
            predicate = str(pattern.get("predicate") or "").strip()
            if not predicate:
                continue
            args = tuple(bindings.get(str(arg), str(arg)) for arg in (pattern.get("args") or ()))
            materialised.append(
                HTNLiteral(
                    predicate=predicate,
                    args=args,
                    is_positive=bool(pattern.get("is_positive", True)),
                    source_symbol=None,
                ),
            )
        return tuple(materialised)

    @staticmethod
    def _stage6_ordered_method_steps(method) -> List[Any]:
        if len(method.subtasks) <= 1 or not method.ordering:
            return list(method.subtasks)

        step_lookup = {
            step.step_id: step
            for step in method.subtasks
        }
        dependents: Dict[str, List[str]] = {
            step.step_id: []
            for step in method.subtasks
        }
        in_degree: Dict[str, int] = {
            step.step_id: 0
            for step in method.subtasks
        }

        for before, after in method.ordering:
            if before not in step_lookup or after not in step_lookup:
                return list(method.subtasks)
            dependents[before].append(after)
            in_degree[after] += 1

        ordered_steps: List[Any] = []
        ready = [
            step.step_id
            for step in method.subtasks
            if in_degree[step.step_id] == 0
        ]
        while ready:
            current_id = ready.pop(0)
            ordered_steps.append(step_lookup[current_id])
            for next_id in dependents[current_id]:
                in_degree[next_id] -= 1
                if in_degree[next_id] == 0:
                    ready.append(next_id)

        if len(ordered_steps) != len(method.subtasks):
            return list(method.subtasks)
        return ordered_steps

    def _stage6_symbolic_method_guided_execution(
        self,
        *,
        task_network: Sequence[Tuple[str, Sequence[str]]],
        method_library,
        runner: JasonRunner,
        action_schemas: Sequence[Dict[str, Any]],
        seed_facts: Sequence[str],
        predicate_name_map: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        if method_library is None or not hasattr(method_library, "methods_for_task"):
            return None
        if not task_network:
            return None

        schema_lookup: Dict[str, Dict[str, Any]] = {}
        for schema in action_schemas:
            functor = str(schema.get("functor", "")).strip()
            source_name = str(schema.get("source_name", "")).strip()
            if functor:
                schema_lookup.setdefault(functor, schema)
            if source_name:
                schema_lookup.setdefault(source_name, schema)
                schema_lookup.setdefault(self._sanitize_name(source_name), schema)
        if not schema_lookup:
            return None

        world: Set[str] = {
            atom
            for atom in (runner._hddl_fact_to_atom(fact) for fact in seed_facts)
            if atom is not None
        }
        action_path: List[str] = []
        method_trace: List[Dict[str, Any]] = []
        expansion_count = 0
        expansion_limit = max(256, len(task_network) * 128)
        transition_payload_by_task = {
            str(payload.get("name", "")).strip(): payload
            for payload in (
                (getattr(self, "_latest_transition_prompt_analysis", {}) or {}).get(
                    "transition_tasks",
                    (),
                )
                or ()
            )
            if isinstance(payload, dict) and str(payload.get("name", "")).strip()
        }

        def parse_atom(atom: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
            text = str(atom or "").strip()
            match = re.fullmatch(r"([A-Za-z0-9_]+)(?:\((.*)\))?", text)
            if match is None:
                return None
            args_text = (match.group(2) or "").strip()
            if not args_text:
                return match.group(1), ()
            return (
                match.group(1),
                tuple(part.strip() for part in args_text.split(",") if part.strip()),
            )

        def world_index() -> Dict[str, List[Tuple[str, ...]]]:
            index: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
            for atom in world:
                parsed = parse_atom(atom)
                if parsed is None:
                    continue
                predicate, args = parsed
                index[predicate].append(args)
            return index

        def is_variable(token: str, known_variables: Sequence[str]) -> bool:
            value = str(token or "").strip()
            return (
                value in set(known_variables)
                or value.startswith("?")
                or self._is_variable_symbol(value)
            )

        def resolve_token(token: str, bindings: Dict[str, str]) -> str:
            value = runner._canonical_runtime_token(token)
            if value in bindings:
                return bindings[value]
            if value.startswith("?") and value[1:] in bindings:
                return bindings[value[1:]]
            return value

        def unify_args(
            pattern_args: Sequence[str],
            grounded_args: Sequence[str],
            bindings: Dict[str, str],
            known_variables: Sequence[str],
        ) -> Optional[Dict[str, str]]:
            if len(pattern_args) != len(grounded_args):
                return None
            next_bindings = dict(bindings)
            for pattern_arg, grounded_arg in zip(pattern_args, grounded_args):
                pattern = runner._canonical_runtime_token(pattern_arg)
                value = runner._canonical_runtime_token(grounded_arg)
                if is_variable(pattern, known_variables):
                    existing = next_bindings.get(pattern)
                    if existing is not None and existing != value:
                        return None
                    next_bindings[pattern] = value
                    if pattern.startswith("?"):
                        next_bindings.setdefault(pattern[1:], value)
                    continue
                if pattern != value:
                    return None
            return next_bindings

        def ground_args(
            args: Sequence[str],
            bindings: Dict[str, str],
            known_variables: Sequence[str],
        ) -> Optional[Tuple[str, ...]]:
            grounded: List[str] = []
            for arg in args:
                value = runner._canonical_runtime_token(arg)
                if is_variable(value, known_variables) and value not in bindings:
                    return None
                grounded.append(resolve_token(value, bindings))
            return tuple(grounded)

        def method_task_pattern(method) -> Tuple[str, ...]:
            explicit_task_args = tuple(getattr(method, "task_args", ()) or ())
            if explicit_task_args:
                return explicit_task_args
            task_schema = method_library.task_for_name(method.task_name)
            if task_schema is not None:
                declared_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
                return tuple(method.parameters[: len(declared_parameters)])
            declared_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
            return tuple(getattr(method, "parameters", ()) or ())

        def literal_binding_options(
            literal: HTNLiteral,
            bindings: Dict[str, str],
            known_variables: Sequence[str],
        ) -> List[Dict[str, str]]:
            if literal.predicate == "=" and len(literal.args) == 2:
                left, right = literal.args
                left_value = resolve_token(left, bindings)
                right_value = resolve_token(right, bindings)
                left_variable = is_variable(left, known_variables)
                right_variable = is_variable(right, known_variables)
                if left_variable and left_value == left and right_value != right:
                    next_bindings = dict(bindings)
                    next_bindings[left] = right_value
                    return [next_bindings] if literal.is_positive else []
                if right_variable and right_value == right and left_value != left:
                    next_bindings = dict(bindings)
                    next_bindings[right] = left_value
                    return [next_bindings] if literal.is_positive else []
                return [bindings] if (left_value == right_value) == literal.is_positive else []

            index = world_index()
            predicate = self._sanitize_name(literal.predicate)
            candidate_args = [
                args
                for args in index.get(predicate, ())
                if len(args) == len(literal.args)
            ]
            if not literal.args:
                atom = runner._ground_runtime_pattern(literal.predicate, (), bindings)
                holds = atom in world
                return [bindings] if holds == literal.is_positive else []

            if literal.is_positive:
                matches: List[Dict[str, str]] = []
                for args in candidate_args:
                    resolved = unify_args(literal.args, args, bindings, known_variables)
                    if resolved is not None:
                        matches.append(resolved)
                return matches

            for args in candidate_args:
                if unify_args(literal.args, args, bindings, known_variables) is not None:
                    return []
            return [bindings]

        def context_binding_options(method, seed_bindings: Dict[str, str]) -> List[Dict[str, str]]:
            bindings_list = [dict(seed_bindings)]
            known_variables = tuple(getattr(method, "parameters", ()) or ())
            for literal in tuple(getattr(method, "context", ()) or ()):
                next_bindings: List[Dict[str, str]] = []
                for candidate in bindings_list:
                    next_bindings.extend(
                        literal_binding_options(literal, candidate, known_variables),
                    )
                if not next_bindings:
                    return []
                bindings_list = next_bindings[:256]
            return bindings_list

        def compiler_transition_context_bindings(
            method,
            seed_bindings: Dict[str, str],
        ) -> Optional[Dict[str, str]]:
            payload = transition_payload_by_task.get(str(getattr(method, "task_name", "") or ""))
            if not payload:
                return dict(seed_bindings)
            known_variables = tuple(getattr(method, "parameters", ()) or ())
            next_bindings = dict(seed_bindings)
            symbolic_signatures = tuple(payload.get("retained_prefix_literals") or ())
            grounded_signatures = tuple(payload.get("retained_prefix_grounded_literals") or ())
            for symbolic_signature, grounded_signature in zip(
                symbolic_signatures,
                grounded_signatures,
            ):
                symbolic_literal = parse_atom(str(symbolic_signature).strip())
                grounded_literal = parse_atom(str(grounded_signature).strip())
                if symbolic_literal is None or grounded_literal is None:
                    continue
                symbolic_predicate, symbolic_args = symbolic_literal
                grounded_predicate, grounded_args = grounded_literal
                if (
                    symbolic_predicate != grounded_predicate
                    or len(symbolic_args) != len(grounded_args)
                ):
                    continue
                for symbolic_arg, grounded_arg in zip(symbolic_args, grounded_args):
                    symbol = runner._canonical_runtime_token(symbolic_arg)
                    value = runner._canonical_runtime_token(grounded_arg)
                    if not is_variable(symbol, known_variables):
                        continue
                    existing = next_bindings.get(symbol)
                    if existing is not None and existing != value:
                        return None
                    next_bindings[symbol] = value
                    if symbol.startswith("?"):
                        next_bindings.setdefault(symbol[1:], value)
            return next_bindings

        def apply_primitive(action_name: str, action_args: Sequence[str]) -> bool:
            schema = schema_lookup.get(action_name) or schema_lookup.get(
                self._sanitize_name(action_name),
            )
            if schema is None:
                return False
            parameters = [str(parameter) for parameter in (schema.get("parameters") or ())]
            if len(parameters) != len(action_args):
                return False
            bindings: Dict[str, str] = {}
            for parameter, value in zip(parameters, action_args):
                token = runner._canonical_runtime_token(parameter)
                bindings[token] = str(value)
                if token.startswith("?"):
                    bindings[token[1:]] = str(value)

            clauses = list(schema.get("precondition_clauses") or [])
            if not clauses:
                clauses = [list(schema.get("preconditions") or [])]
            if not any(
                runner._replay_precondition_clause_holds(clause, bindings, world)
                for clause in clauses
            ):
                return False

            for effect in schema.get("effects") or []:
                predicate = str(effect.get("predicate", "")).strip()
                if not predicate or predicate == "=":
                    continue
                grounded = runner._ground_runtime_pattern(
                    predicate,
                    effect.get("args") or (),
                    bindings,
                )
                if effect.get("is_positive", True):
                    world.add(grounded)
                else:
                    world.discard(grounded)
            action_path.append(runner._runtime_call(action_name, action_args))
            return True

        def execute_compound(
            task_name: str,
            task_args: Sequence[str],
            stack: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
        ) -> bool:
            nonlocal expansion_count
            task_key = (str(task_name), tuple(str(arg) for arg in task_args))
            if task_key in stack:
                return False
            expansion_count += 1
            if expansion_count > expansion_limit:
                return False
            methods = list(method_library.methods_for_task(str(task_name)))
            if not methods:
                return False
            for method in methods:
                seed_bindings = unify_args(
                    method_task_pattern(method),
                    task_args,
                    {},
                    tuple(getattr(method, "parameters", ()) or ()),
                )
                if seed_bindings is None:
                    continue
                seed_bindings = compiler_transition_context_bindings(method, seed_bindings)
                if seed_bindings is None:
                    continue
                for bindings in context_binding_options(method, seed_bindings):
                    world_snapshot = set(world)
                    action_snapshot = list(action_path)
                    trace_snapshot = list(method_trace)
                    method_trace.append(
                        {
                            "method_name": method.method_name,
                            "task_args": [str(arg) for arg in task_args],
                        },
                    )
                    branch_ok = True
                    for step in self._stage6_ordered_method_steps(method):
                        step_args = ground_args(
                            tuple(getattr(step, "args", ()) or ()),
                            bindings,
                            tuple(getattr(method, "parameters", ()) or ()),
                        )
                        if step_args is None:
                            branch_ok = False
                            break
                        if getattr(step, "kind", "") == "primitive":
                            action_name = str(
                                getattr(step, "action_name", None)
                                or getattr(step, "task_name", "")
                            ).strip()
                            if not apply_primitive(action_name, step_args):
                                branch_ok = False
                                break
                        elif not execute_compound(
                            str(getattr(step, "task_name", "")).strip(),
                            step_args,
                            stack + (task_key,),
                        ):
                            branch_ok = False
                            break
                    if branch_ok:
                        return True
                    world.clear()
                    world.update(world_snapshot)
                    action_path[:] = action_snapshot
                    method_trace[:] = trace_snapshot
            return False

        for task_name, task_args in task_network:
            if not execute_compound(str(task_name), tuple(task_args or ())):
                return None

        replay = runner._replay_action_path_against_schemas(
            action_path=action_path,
            action_schemas=action_schemas,
            seed_facts=seed_facts,
        )
        if replay.get("passed") is not True:
            return None

        return {
            "source": "method_library_symbolic_execution",
            "task_network": [
                {"task_name": str(task_name), "args": list(task_args)}
                for task_name, task_args in task_network
            ],
            "action_path": action_path,
            "method_trace": method_trace,
            "post_guided_seed_facts": tuple(
                runner._runtime_world_to_hddl_facts(
                    replay.get("world_facts") or (),
                    predicate_name_map=predicate_name_map,
                ),
            ),
        }

    @staticmethod

    @staticmethod

    @staticmethod
    def _stage6_reorder_unordered_guided_task_network(
        *,
        task_network: Sequence[Tuple[str, Sequence[str]]],
        literal_signatures: Sequence[str],
        preferred_target_ids: Sequence[str],
    ) -> Tuple[Tuple[Tuple[str, Tuple[str, ...]], ...], Tuple[str, ...], Tuple[str, ...]]:
        indexed_entries = [
            (
                f"t{index}",
                (
                    str(task_name),
                    tuple(str(arg) for arg in (task_args or ())),
                ),
                str(literal_signatures[index - 1]).strip()
                if index - 1 < len(literal_signatures)
                else "",
            )
            for index, (task_name, task_args) in enumerate(task_network, start=1)
        ]
        if not indexed_entries:
            return (), (), ()

        preferred_order = [
            str(target_id).strip()
            for target_id in preferred_target_ids
            if str(target_id).strip()
        ]
        if not preferred_order:
            return (
                tuple(entry for _, entry, _ in indexed_entries),
                tuple(signature for _, _, signature in indexed_entries),
                tuple(target_id for target_id, _, _ in indexed_entries),
            )

        by_id = {
            target_id: (task_entry, signature)
            for target_id, task_entry, signature in indexed_entries
        }
        ordered_ids = [target_id for target_id in preferred_order if target_id in by_id]
        ordered_ids.extend(
            target_id
            for target_id, _, _ in indexed_entries
            if target_id not in ordered_ids
        )
        return (
            tuple(by_id[target_id][0] for target_id in ordered_ids),
            tuple(by_id[target_id][1] for target_id in ordered_ids),
            tuple(ordered_ids),
        )

    def _stage6_resolve_symbolic_literal_bindings(
        self,
        *,
        symbolic_requirements: Sequence[HTNLiteral],
        concrete_history: Sequence[HTNLiteral],
        seed_bindings: Dict[str, str],
    ) -> Optional[Dict[str, str]]:
        if not symbolic_requirements:
            return dict(seed_bindings)

        def match_requirement(
            requirement: HTNLiteral,
            candidate: HTNLiteral,
            current_bindings: Dict[str, str],
        ) -> Optional[Dict[str, str]]:
            if (
                requirement.predicate != candidate.predicate
                or requirement.is_positive != candidate.is_positive
                or len(requirement.args) != len(candidate.args)
            ):
                return None
            next_bindings = dict(current_bindings)
            for symbolic_arg, concrete_arg in zip(requirement.args, candidate.args):
                symbol = str(symbolic_arg).strip()
                value = str(concrete_arg).strip()
                if not symbol or not value:
                    return None
                bound_value = next_bindings.get(symbol)
                if bound_value is None:
                    next_bindings[symbol] = value
                elif bound_value != value:
                    return None
            return next_bindings

        if len(symbolic_requirements) == len(concrete_history):
            current_bindings = dict(seed_bindings)
            for requirement, candidate in zip(symbolic_requirements, concrete_history):
                matched_bindings = match_requirement(
                    requirement,
                    candidate,
                    current_bindings,
                )
                if matched_bindings is None:
                    break
                current_bindings = matched_bindings
            else:
                return current_bindings

        if len(symbolic_requirements) > 128:
            current_bindings = dict(seed_bindings)
            used_candidates: Set[int] = set()
            for requirement in symbolic_requirements:
                for candidate_index, candidate in enumerate(concrete_history):
                    if candidate_index in used_candidates:
                        continue
                    matched_bindings = match_requirement(
                        requirement,
                        candidate,
                        current_bindings,
                    )
                    if matched_bindings is None:
                        continue
                    used_candidates.add(candidate_index)
                    current_bindings = matched_bindings
                    break
                else:
                    return None
            return current_bindings

        def backtrack(
            requirement_index: int,
            current_bindings: Dict[str, str],
            used_candidates: Set[int],
        ) -> Optional[Dict[str, str]]:
            if requirement_index >= len(symbolic_requirements):
                return current_bindings
            requirement = symbolic_requirements[requirement_index]
            for candidate_index, candidate in enumerate(concrete_history):
                if candidate_index in used_candidates:
                    continue
                matched_bindings = match_requirement(
                    requirement,
                    candidate,
                    current_bindings,
                )
                if matched_bindings is None:
                    continue
                resolved = backtrack(
                    requirement_index + 1,
                    matched_bindings,
                    used_candidates | {candidate_index},
                )
                if resolved is not None:
                    return resolved
            return None

        return backtrack(0, dict(seed_bindings), set())

    def _stage6_transition_spec_literal_signature(self, transition_spec: Dict[str, Any]) -> str:
        target_literal = transition_spec.get("target_literal")
        if isinstance(target_literal, dict):
            predicate = str(target_literal.get("predicate") or "").strip()
            if predicate:
                args = [
                    str(arg).strip()
                    for arg in (target_literal.get("args") or ())
                    if str(arg).strip()
                ]
                return (
                    f"{predicate}({', '.join(args)})"
                    if args
                    else predicate
                )
        return str(transition_spec.get("label") or "").strip()

    def _stage6_transition_target_id_map(
        self,
        *,
        transition_specs: Sequence[Dict[str, Any]],
        target_literals: Sequence[HTNLiteral],
        plan_records: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        if not transition_specs or not target_literals:
            return {}

        target_signatures = {
            literal.to_signature()
            for literal in target_literals
        }
        transition_target_ids: Dict[str, str] = {}
        target_id_by_signature: Dict[str, str] = {}

        def _append_targets_from_records(records: Sequence[Dict[str, Any]]) -> None:
            for record in records:
                if not isinstance(record, dict):
                    continue
                transition_name = str(record.get("transition_name") or "").strip()
                label = str(record.get("label") or "").strip()
                if not transition_name or not label:
                    continue
                if label not in target_signatures:
                    continue
                if transition_name in transition_target_ids:
                    continue
                target_id = target_id_by_signature.get(label)
                if target_id is None:
                    target_id = f"t{len(target_id_by_signature) + 1}"
                    target_id_by_signature[label] = target_id
                transition_target_ids[transition_name] = target_id

        if plan_records:
            _append_targets_from_records(plan_records)
        if transition_target_ids:
            return transition_target_ids

        for transition_spec in transition_specs:
            if not isinstance(transition_spec, dict):
                continue
            transition_name = str(transition_spec.get("transition_name") or "").strip()
            if not transition_name:
                continue
            signature = self._stage6_transition_spec_literal_signature(transition_spec)
            if not signature or signature not in target_signatures:
                continue
            if transition_name in transition_target_ids:
                continue
            target_id = target_id_by_signature.get(signature)
            if target_id is None:
                target_id = f"t{len(target_id_by_signature) + 1}"
                target_id_by_signature[signature] = target_id
            transition_target_ids[transition_name] = target_id
        return transition_target_ids

    def _stage6_query_root_bridge_method_trace_entry(
        self,
        *,
        query_root_task_name: str,
        query_root_task_args: Sequence[str],
        transition_name: str,
        method_library,
    ) -> Optional[Dict[str, Any]]:
        if (
            method_library is None
            or not hasattr(method_library, "methods_for_task")
            or not query_root_task_name
            or not transition_name
        ):
            return None
        for method in method_library.methods_for_task(query_root_task_name):
            if len(method.subtasks) != 1:
                continue
            step = method.subtasks[0]
            if step.kind != "compound" or step.task_name != transition_name:
                continue
            return {
                "method_name": method.method_name,
                "task_args": [str(arg).strip() for arg in (query_root_task_args or ())],
            }
        return None

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod
    def _stage5_query_typed_objects(ltl_spec):
        inventory = tuple(getattr(ltl_spec, "query_object_inventory", ()) or ())
        if not inventory:
            return None

        object_types: Dict[str, str] = {}
        for entry in inventory:
            type_name = str(entry.get("type") or "").strip()
            if not type_name:
                continue
            for obj in entry.get("objects") or ():
                object_name = str(obj).strip()
                if object_name:
                    object_types[object_name] = type_name

        typed_objects = tuple(
            (obj, object_types[obj])
            for obj in getattr(ltl_spec, "objects", ()) or ()
            if obj in object_types
        )
        return typed_objects or None

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
        available_objects = list(dict.fromkeys(objects or ()))
        for fact in seed_facts:
            parsed = self._parse_positive_hddl_fact(fact)
            if parsed is None:
                continue
            _predicate, args = parsed
            for arg in args:
                if arg not in available_objects:
                    available_objects.append(arg)
        resolved: Dict[str, str] = {}
        if problem_object_types is not None:
            missing_objects: List[str] = []
            for obj in available_objects:
                if obj not in problem_object_types:
                    missing_objects.append(obj)
                    continue
                type_name = problem_object_types[obj]
                if type_name not in self.domain_type_names:
                    raise TypeResolutionError(
                        f"Stage 6 object typing: problem object '{obj}' has unknown type '{type_name}'.",
                    )
                resolved[obj] = type_name
            if not missing_objects:
                return resolved

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

        for obj in sorted(required_objects):
            if obj not in available_objects:
                available_objects.append(obj)
        for obj in available_objects:
            if obj not in required_objects and problem_object_types is None:
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
