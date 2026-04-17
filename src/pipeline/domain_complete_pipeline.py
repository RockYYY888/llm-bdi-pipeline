"""
Domain-complete Hierarchical Task Network pipeline with Temporally Extended Goal support.
"""

import copy
import io
import json
import multiprocessing
import os
import queue
import re
import signal
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Any, FrozenSet, List, Optional, Sequence, Set, Tuple

_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir in sys.path:
    sys.path.remove(_src_dir)
sys.path.insert(0, _src_dir)

from utils.config import get_config
from query_execution.goal_grounding.grounder import NLToLTLfGenerator
from query_execution.goal_grounding.formulas import LTLFormula, LogicalOperator, TemporalOperator
from domain_build.method_synthesis.synthesizer import HTNMethodSynthesizer
from domain_build.method_synthesis.schema import (
    HTNLiteral,
    HTNMethod,
    HTNMethodLibrary,
    HTNTask,
    HTNTargetTaskBinding,
    _parse_signature_literal,
)
from domain_build.method_synthesis.naming import query_root_alias_task_name, sanitize_identifier
from planning.problem_structure import ProblemStructure, ProblemStructureAnalyzer
from planning.official_benchmark import (
    OFFICIAL_BACKEND_SELECTION_RULE,
    OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS,
)
from planning.panda_portfolio import PANDAPlanner
from planning.representations import (
    PlanningRepresentationBuilder,
    PlanningRepresentation,
    RepresentationBuildResult,
)
from planning.backends import (
    PlanningBackendTask,
    backend_by_name,
    default_official_backends,
    expand_backend_tasks_for_representations,
)
from pipeline.artifacts import (
    DomainLibraryArtifact,
    GoalRequest,
    GoalRequestNode,
    PlanningRequest,
    load_domain_library_artifact,
    persist_domain_library_artifact,
    query_bound_method_library,
)
from utils.hddl_condition_parser import HDDLConditionParser
from verification.official_plan_verifier import IPCPlanVerifier
from pipeline.execution_logger import ExecutionLogger


class _NullExecutionLogger:
    """Minimal logger for isolated backend worker attempts."""

    current_record = None

    def record_step_timing(self, *args, **kwargs) -> None:
        return None

    def log_plan_solve(self, *args, **kwargs) -> None:
        return None

    def log_official_verification(self, *args, **kwargs) -> None:
        return None


OFFICIAL_PARALLEL_SELECTION_RULE = OFFICIAL_BACKEND_SELECTION_RULE
BACKEND_RESULT_MESSAGE = "backend_attempt"


def _official_problem_root_planning_task_worker(
    result_queue,
    *,
    domain_file: str,
    problem_file: str,
    output_dir: str,
    task_payload: Dict[str, Any],
    planning_timeout_seconds: float,
) -> None:
    plan_solve_seconds = 0.0
    plan_verification_seconds = 0.0
    total_start = time.perf_counter()
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    planning_task = PlanningBackendTask.from_dict(dict(task_payload))
    try:
        if hasattr(os, "setsid"):
            os.setsid()
        pipeline = DomainCompletePipeline(
            domain_file=domain_file,
            problem_file=problem_file,
        )
        pipeline.logger = _NullExecutionLogger()
        pipeline.output_dir = Path(output_dir).resolve()
        pipeline.output_dir.mkdir(parents=True, exist_ok=True)

        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            plan_solve_start = time.perf_counter()
            plan_solve_data = pipeline._solve_problem_root_backend_task(
                planning_task=planning_task,
                timeout_seconds=planning_timeout_seconds,
            )
            plan_solve_seconds = time.perf_counter() - plan_solve_start

            plan_verification_start = time.perf_counter()
            plan_verification_data = pipeline._verify_plan_officially(
                None,
                None,
                plan_solve_data,
            )
            plan_verification_seconds = time.perf_counter() - plan_verification_start

        plan_solve_summary = dict((plan_solve_data or {}).get("summary") or {})
        plan_verification_summary = dict((plan_verification_data or {}).get("summary") or {})
        result_queue.put(
            {
                "message_type": BACKEND_RESULT_MESSAGE,
                "backend_name": planning_task.backend_name,
                "task_id": planning_task.task_id,
                "representation_id": planning_task.representation.representation_id,
                "output_dir": str(Path(output_dir).resolve()),
                "plan_solve_data": plan_solve_data,
                "plan_verification_data": plan_verification_data,
                "plan_solve_seconds": plan_solve_seconds,
                "plan_verification_seconds": plan_verification_seconds,
                "total_seconds": time.perf_counter() - total_start,
                "success": (
                    plan_solve_summary.get("status") == "success"
                    and plan_verification_summary.get("status") == "success"
                ),
                "selected_bucket": (
                    plan_verification_summary.get("selected_bucket")
                    or plan_verification_summary.get("failure_bucket")
                    or plan_solve_summary.get("failure_bucket")
                    or "unknown_failure"
                ),
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "message_type": BACKEND_RESULT_MESSAGE,
                "backend_name": planning_task.backend_name,
                "task_id": planning_task.task_id,
                "representation_id": planning_task.representation.representation_id,
                "output_dir": str(Path(output_dir).resolve()),
                "plan_solve_data": {
                    "summary": {
                        "backend": planning_task.backend_name,
                        "status": "failed",
                        "planning_mode": "official_problem_root",
                        "failure_bucket": "worker_exception",
                        "representation_id": planning_task.representation.representation_id,
                    },
                    "artifacts": {
                        "backend": planning_task.backend_name,
                        "status": "failed",
                        "planning_mode": "official_problem_root",
                        "failure_bucket": "worker_exception",
                        "planning_representation": planning_task.representation.to_dict(),
                    },
                },
                "plan_verification_data": {
                    "summary": {
                        "backend": "pandaPIparser",
                        "status": "failed",
                        "selection_rule": OFFICIAL_PARALLEL_SELECTION_RULE,
                        "failure_bucket": "worker_exception",
                    },
                    "artifacts": {
                        "tool_available": None,
                        "verification_result": False,
                        "selected_bucket": "worker_exception",
                    },
                },
                "plan_solve_seconds": plan_solve_seconds,
                "plan_verification_seconds": plan_verification_seconds,
                "total_seconds": time.perf_counter() - total_start,
                "success": False,
                "selected_bucket": "worker_exception",
                "stdout": captured_stdout.getvalue(),
                "stderr": (
                    captured_stderr.getvalue()
                    + f"\nworker_exception: {exc}\n"
                ),
            }
        )


class TypeResolutionError(RuntimeError):
    """Raised when object/variable type inference is ambiguous or inconsistent."""

class DomainCompletePipeline:
    """
    Domain-complete Hierarchical Task Network pipeline with offline library build.

    Offline:
    method synthesis -> domain gate

    Online:
    goal grounding -> plan solve -> official verification
    """

    DOMAIN_GATE_COMPACT_TASK_ARG_THRESHOLD = 32
    DOMAIN_GATE_MAX_VALIDATION_COMPOUND_STEPS = 8
    OFFICIAL_PROBLEM_ROOT_PLANNING_TIMEOUT_SECONDS = (
        OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS
    )

    def __init__(self, domain_file: str, problem_file: str | None = None):
        """
        Initialize pipeline

        Args:
            domain_file: Path to HDDL domain file.
            problem_file: Optional path to HDDL problem file used for runtime initialisation.
        """
        self.config = get_config()

        # Resolve the repository root so generated artifacts do not leak back into src/.
        project_root = Path(__file__).resolve().parents[2]
        self.project_root = project_root
        self.logger = ExecutionLogger(logs_dir=str(project_root / "artifacts" / "runs"))

        if not domain_file:
            raise ValueError(
                "domain_file is required. Pass an explicit HDDL domain path to DomainCompletePipeline.",
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
        self._problem_structure_analyzer = ProblemStructureAnalyzer()

    def _record_step_timing(
        self,
        stage_name: str,
        stage_start: float,
        *,
        breakdown: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger.record_step_timing(
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

    def run_query(self, nl_instruction: str) -> Dict[str, Any]:
        """
        Run the end-to-end convenience path:
        build a domain library, then execute one query against it.
        """
        self.logger.start_pipeline(
            nl_instruction,
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

        print("=" * 80)
        print("DOMAIN-COMPLETE HTN PIPELINE")
        print("=" * 80)
        print(f"\n\"{nl_instruction}\"")
        print(f"Output directory: {self.output_dir}")
        print("\n" + "-" * 80)

        method_library, method_synthesis_data = self._synthesise_domain_methods()
        if not method_library:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {
                "success": False,
                "step": "method_synthesis",
                "error": "Domain HTN synthesis failed",
            }

        domain_gate_data = self._validate_domain_library(method_library)
        if domain_gate_data is None:
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return {
                "success": False,
                "step": "domain_gate",
                "error": "Domain gate failed",
            }

        artifact = DomainLibraryArtifact(
            domain_name=self.domain.name,
            method_library=method_library,
            method_synthesis_metadata=method_synthesis_data,
            domain_gate=domain_gate_data,
        )
        artifact_paths = self._persist_domain_library_artifact(artifact)
        query_result = self._execute_query_with_loaded_library(
            nl_instruction,
            artifact,
        )
        if not query_result.get("success", False):
            log_filepath = self.logger.end_pipeline(success=False)
            print(f"\nExecution log saved to: {log_filepath}")
            return query_result

        print("\n" + "=" * 80)
        print("QUERY EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        log_filepath = self.logger.end_pipeline(success=True)
        print(f"\nExecution log saved to: {log_filepath}")

        return {
            "success": True,
            "domain_build": {
                "method_library": method_library.to_dict(),
                "method_synthesis_metadata": method_synthesis_data,
                "domain_gate": domain_gate_data,
                "artifact_paths": artifact_paths,
            },
            **query_result,
        }

    def build_domain_library(
        self,
        *,
        output_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build and persist one domain-complete library artifact."""

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

        method_library, method_synthesis_data = self._synthesise_domain_methods()
        if not method_library:
            log_filepath = self.logger.end_pipeline(success=False)
            return {
                "success": False,
                "step": "method_synthesis",
                "error": "Domain HTN synthesis failed",
                "log_path": str(log_filepath),
            }

        domain_gate_data = self._validate_domain_library(method_library)
        if domain_gate_data is None:
            log_filepath = self.logger.end_pipeline(success=False)
            return {
                "success": False,
                "step": "domain_gate",
                "error": "Domain gate failed",
                "log_path": str(log_filepath),
            }

        artifact = DomainLibraryArtifact(
            domain_name=self.domain.name,
            method_library=method_library,
            method_synthesis_metadata=method_synthesis_data,
            domain_gate=domain_gate_data,
        )
        artifact_paths = self._persist_domain_library_artifact(
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
    ) -> Dict[str, Any]:
        """Execute one query against a cached domain-complete library artifact."""

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
            load_domain_library_artifact(library_artifact),
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

    def _persist_domain_library_artifact(
        self,
        artifact: DomainLibraryArtifact,
        *,
        output_root: Optional[str] = None,
    ) -> Dict[str, str]:
        return persist_domain_library_artifact(
            artifact_root=self._stable_domain_artifact_root(output_root),
            artifact=artifact,
        )

    def _execute_query_with_loaded_library(
        self,
        nl_instruction: str,
        artifact: DomainLibraryArtifact,
    ) -> Dict[str, Any]:
        """Run the new query-time path with a prebuilt domain-complete method library."""

        domain_library = artifact.method_library
        planning_request = self._ground_query_goal(
            nl_instruction,
            domain_library,
        )
        if planning_request is None:
            return {
                "success": False,
                "step": "goal_grounding",
                "error": "Goal grounding failed",
                "method_library": domain_library.to_dict(),
            }

        plan_solve_data = self._solve_query_request(
            domain_library,
            planning_request,
        )
        if plan_solve_data is None:
            return {
                "success": False,
                "step": "plan_solve",
                "error": "Hierarchical Task Network planning failed",
                "method_library": domain_library.to_dict(),
                "planning_request_context": planning_request.to_dict(),
            }

        plan_verification_data = self._verify_plan_officially(
            None,
            domain_library,
            plan_solve_data,
        )
        if plan_verification_data is None:
            return {
                "success": False,
                "step": "plan_verification",
                "error": "Official IPC verification failed",
                "method_library": domain_library.to_dict(),
                "planning_request_context": planning_request.to_dict(),
                "plan_solve": plan_solve_data,
            }

        return {
            "success": True,
            "method_library": domain_library.to_dict(),
            "goal_request": planning_request.goal_request.to_dict(),
            "planning_request_context": planning_request.to_dict(),
            "plans": [],
            "plan_solve": plan_solve_data,
            "plan_verification": plan_verification_data,
        }

    def _ground_query_goal(
        self,
        nl_instruction: str,
        method_library,
    ) -> Optional[PlanningRequest]:
        """Ground a natural-language query into a planner request."""
        print("\n[GOAL GROUNDING]")
        print("-" * 80)
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
            teg_nodes: List[GoalRequestNode] = []

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
                    GoalRequestNode(
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
            goal_request = GoalRequest(
                query_text=query_text,
                nodes=tuple(teg_nodes),
                precedence_edges=precedence_edges,
                query_object_inventory=tuple(query_object_inventory),
                typed_objects=dict(merged_object_types),
                diagnostics=tuple(diagnostics),
            )
            planning_request = self._build_planning_request_context(goal_request)

            self.logger.log_goal_grounding_success(
                {
                    "query_text": planning_request.query_text,
                    "goal_request": goal_request.to_dict(),
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
            self._record_step_timing(
                "goal_grounding",
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
            self.logger.log_goal_grounding_error(str(exc))
            self._record_step_timing("goal_grounding", stage_start)
            print(f"✗ Goal grounding failed: {exc}")
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
        teg_nodes: Sequence[GoalRequestNode],
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
        goal_request: GoalRequest,
    ) -> PlanningRequest:
        task_network = tuple(
            (node.task_name, tuple(node.args))
            for node in goal_request.nodes
        )
        problem_objects = tuple(
            str(obj).strip()
            for obj in (self.problem.objects if self.problem is not None else ())
            if str(obj).strip()
        )
        if not problem_objects:
            problem_objects = tuple(
                object_name
                for entry in goal_request.query_object_inventory
                for object_name in entry.get("objects", ())
                if str(object_name).strip()
            )
        task_network_ordered = (
            len(task_network) <= 1
            and not goal_request.precedence_edges
        )
        return PlanningRequest(
            query_text=goal_request.query_text,
            goal_request=goal_request,
            problem_objects=problem_objects,
            typed_objects=dict(goal_request.typed_objects),
            task_network=task_network,
            task_network_ordered=task_network_ordered,
            ordering_edges=tuple(goal_request.precedence_edges),
            diagnostics=tuple(goal_request.diagnostics),
        )

    def _solve_query_request(
        self,
        method_library,
        planning_request: PlanningRequest,
    ) -> Optional[Dict[str, Any]]:
        """Solve the grounded request with PANDA."""
        print("\n[PLAN SOLVE]")
        print("-" * 80)
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
            if self._should_sequence_goal_request(planning_request):
                sequential_result = self._solve_ordered_goal_sequence(
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
                    timeout_seconds=float(self.config.planning_timeout),
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

            action_path_file = self.output_dir / "plan_solve_action_path.txt"
            action_path_file.write_text(
                "".join(f"{step}\n" for step in action_path),
            )
            method_trace_file = self.output_dir / "plan_solve_method_trace.json"
            method_trace_file.write_text(json.dumps(method_trace, indent=2))
            combined_plan_file = None
            if combined_plan_text:
                combined_plan_file = self.output_dir / "plan_solve_hierarchical_plan.txt"
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
            self.logger.log_plan_solve(
                artifacts,
                "Success",
                metadata=summary,
            )
            self._record_step_timing(
                "plan_solve",
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
            self.logger.log_plan_solve(
                failure_artifacts,
                "Failed",
                error=str(exc),
                metadata={
                    "backend": "pandaPI",
                    "status": "failed",
                },
            )
            self._record_step_timing("plan_solve", stage_start)
            print(f"✗ Plan solve failed: {exc}")
            return None

    def _official_problem_root_structure_analysis(self) -> ProblemStructure:
        if self.problem is None:
            raise ValueError("Official problem-root analysis requires a loaded problem file.")
        return self._problem_structure_analyzer.analyze(
            domain=self.domain,
            problem=self.problem,
        )

    def _problem_root_task_network_ordering_edges(self) -> Tuple[Tuple[str, str], ...]:
        if self.problem is None:
            return ()
        return self._problem_structure_analyzer.root_ordering_edges(self.problem)

    def _problem_root_task_network_is_totally_ordered(self) -> bool:
        if self.problem is None:
            return False
        return self._official_problem_root_structure_analysis().root_task_network.is_total_order

    def _domain_methods_are_totally_ordered(self) -> bool:
        return all(
            network.is_total_order
            for network in self._official_problem_root_structure_analysis().method_task_networks
        )

    def _instance_is_totally_ordered(self) -> bool:
        return self._official_problem_root_structure_analysis().is_total_order

    def _problem_root_task_network_requires_linearizer(self) -> bool:
        return self._official_problem_root_structure_analysis().requires_linearization

    def _build_problem_representations(
        self,
        timeout_seconds: Optional[float] = None,
    ) -> RepresentationBuildResult:
        representation_root = (
            Path(self.output_dir).resolve() / "representations"
            if self.output_dir is not None
            else self.project_root / "artifacts" / "runs" / "tmp_representations"
        )
        builder = PlanningRepresentationBuilder(workspace=representation_root)
        return builder.build(
            domain=self.domain,
            problem=self.problem,
            domain_file=self.domain_file,
            problem_file=self.problem_file,
            timeout_seconds=timeout_seconds,
        )

    def _official_problem_root_planning_timeout_seconds(
        self,
        timeout_seconds: Optional[float] = None,
    ) -> float:
        if timeout_seconds is not None:
            return max(float(timeout_seconds), 1.0)
        return float(self.OFFICIAL_PROBLEM_ROOT_PLANNING_TIMEOUT_SECONDS)

    def _solve_problem_root_backend_task(
        self,
        *,
        planning_task: PlanningBackendTask,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Solve one compiled official representation with one backend."""
        print("\n[PLAN SOLVE]")
        print("-" * 80)
        stage_start = time.perf_counter()
        effective_timeout_seconds = self._official_problem_root_planning_timeout_seconds(
            timeout_seconds,
        )
        backend = backend_by_name(
            planning_task.backend_name,
            workspace=str(self.output_dir),
        )

        try:
            if self.problem is None:
                raise ValueError("Problem root planning requires a loaded problem file.")
            if not self.problem.htn_tasks:
                raise ValueError("Problem file contains no root HTN tasks.")
            if not backend.toolchain_available():
                raise ValueError(
                    f"Planning backend '{planning_task.backend_name}' is unavailable on PATH.",
                )

            problem_objects = tuple(self.problem.objects or ())
            if not problem_objects:
                raise ValueError("Problem file contains no declared objects.")
            task_network = tuple(
                (str(task.task_name), tuple(str(arg) for arg in (task.args or ())))
                for task in (self.problem.htn_tasks or ())
            )
            ordering_edges = self._problem_root_task_network_ordering_edges()
            task_network_ordered = self._problem_root_task_network_is_totally_ordered()
            primary_task_name, primary_task_args = task_network[0]
            plan = backend.solve(
                domain=self.domain,
                task_name=str(primary_task_name),
                representation=planning_task.representation,
                task_args=tuple(primary_task_args),
                timeout_seconds=effective_timeout_seconds,
            )
            action_path = [
                (
                    f"{step.action_name}({', '.join(step.args)})"
                    if step.args
                    else str(step.action_name)
                )
                for step in plan.steps
            ]
            method_trace = PANDAPlanner(workspace=str(self.output_dir)).extract_method_trace(
                plan.actual_plan,
            )
            timing_profile = dict(plan.timing_profile or {})
            work_dir = Path(str(plan.work_dir)).resolve() if plan.work_dir else None

            action_path_file = self.output_dir / "plan_solve_action_path.txt"
            action_path_file.write_text("".join(f"{step}\n" for step in action_path))
            method_trace_file = self.output_dir / "plan_solve_method_trace.json"
            method_trace_file.write_text(json.dumps(method_trace, indent=2))
            combined_plan_file = self.output_dir / "plan_solve_hierarchical_plan.txt"
            combined_plan_file.write_text(str(plan.actual_plan or ""))

            artifacts = {
                "backend": planning_task.backend_name,
                "status": "success",
                "planning_mode": "official_problem_root",
                "engine_mode": plan.engine_mode,
                "solver_id": plan.solver_id,
                "planning_representation": planning_task.representation.to_dict(),
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
                "guided_hierarchical_plan_source": backend.plan_source_label,
                "timing_profile": timing_profile,
                "solver_candidates": list(plan.solver_candidates or ()),
                "artifacts": {
                    "domain_hddl": planning_task.representation.domain_file,
                    "problem_hddl": planning_task.representation.problem_file,
                    "parsed_problem": str(work_dir / "problem.psas") if work_dir else None,
                    "grounded_problem": str(work_dir / "problem.psas.grounded") if work_dir else None,
                    "raw_plan": str(work_dir / "plan.original") if work_dir else None,
                    "actual_plan": str(combined_plan_file),
                    "action_path": str(action_path_file),
                    "method_trace": str(method_trace_file),
                },
            }
            summary = {
                "backend": planning_task.backend_name,
                "status": "success",
                "planning_mode": "official_problem_root",
                "engine_mode": plan.engine_mode,
                "solver_id": plan.solver_id,
                "representation_id": planning_task.representation.representation_id,
                "representation_source": planning_task.representation.representation_source,
                "representation_ordering_kind": planning_task.representation.ordering_kind,
                "task_count": len(task_network),
                "precedence_edge_count": len(ordering_edges),
                "step_count": len(action_path),
                "solver_candidate_count": len(plan.solver_candidates or ()),
            }
            self.logger.log_plan_solve(
                artifacts,
                "Success",
                metadata=summary,
            )
            self._record_step_timing(
                "plan_solve",
                stage_start,
                breakdown=self._timing_breakdown_without_total(timing_profile),
                metadata={
                    "task_count": len(task_network),
                    "step_count": len(action_path),
                    "planning_mode": "official_problem_root",
                    "solver_id": plan.solver_id,
                    "backend": planning_task.backend_name,
                    "representation_id": planning_task.representation.representation_id,
                },
            )
            print(f"✓ Planner returned {len(action_path)} primitive steps")
            print(f"  Problem root task count: {len(task_network)}")
            return {"summary": summary, "artifacts": artifacts}
        except Exception as exc:
            failure_metadata = dict(getattr(exc, "metadata", {}) or {})
            failure_artifacts = {
                "backend": planning_task.backend_name,
                "status": "failed",
                "planning_mode": "official_problem_root",
                "planning_representation": planning_task.representation.to_dict(),
                "task_network": [
                    {
                        "task_name": str(task.task_name),
                        "args": list(task.args or ()),
                    }
                    for task in (self.problem.htn_tasks or ())
                ] if self.problem is not None else [],
                "task_network_ordered": self._problem_root_task_network_is_totally_ordered(),
                "ordering_edges": [
                    {"before": before, "after": after}
                    for before, after in self._problem_root_task_network_ordering_edges()
                ] if self.problem is not None else [],
                "step_count": 0,
                "failure_bucket": "no_plan_from_solver",
                "solver_candidates": list(failure_metadata.get("engine_attempts") or ()),
                "failure_metadata": failure_metadata,
            }
            summary = {
                "backend": planning_task.backend_name,
                "status": "failed",
                "planning_mode": "official_problem_root",
                "engine_mode": failure_metadata.get("engine_mode"),
                "failure_bucket": "no_plan_from_solver",
                "representation_id": planning_task.representation.representation_id,
                "representation_source": planning_task.representation.representation_source,
                "representation_ordering_kind": planning_task.representation.ordering_kind,
                "solver_candidate_count": len(failure_metadata.get("engine_attempts") or ()),
            }
            self.logger.log_plan_solve(
                failure_artifacts,
                "Failed",
                error=str(exc),
                metadata=summary,
            )
            self._record_step_timing("plan_solve", stage_start)
            print(f"✗ Plan solve failed: {exc}")
            return {"summary": summary, "artifacts": failure_artifacts}

    @staticmethod
    def _official_problem_root_failure_rank(attempt: Dict[str, Any]) -> Tuple[int, float]:
        if attempt.get("success"):
            return (0, float(attempt.get("total_seconds") or 0.0))
        bucket = str(attempt.get("selected_bucket") or "unknown_failure")
        priority = {
            "hierarchical_plan_verified": 0,
            "primitive_plan_valid_but_hierarchical_rejected": 1,
            "primitive_plan_invalid": 2,
            "no_plan_from_solver": 3,
            "worker_exception": 4,
        }.get(bucket, 5)
        return (priority, float(attempt.get("total_seconds") or 0.0))

    def _select_official_problem_root_backend_attempt(
        self,
        attempts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        successful = [
            attempt
            for attempt in attempts
            if bool(attempt.get("success"))
        ]
        if successful:
            return min(
                successful,
                key=lambda attempt: float(attempt.get("total_seconds") or 0.0),
            )
        if not attempts:
            raise ValueError("No official problem-root backend attempts were provided.")
        return min(attempts, key=self._official_problem_root_failure_rank)

    @staticmethod
    def _rewrite_artifact_root_paths(value: Any, source_root: Path, target_root: Path) -> Any:
        if isinstance(value, dict):
            return {
                key: DomainCompletePipeline._rewrite_artifact_root_paths(
                    item,
                    source_root,
                    target_root,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                DomainCompletePipeline._rewrite_artifact_root_paths(
                    item,
                    source_root,
                    target_root,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                DomainCompletePipeline._rewrite_artifact_root_paths(
                    item,
                    source_root,
                    target_root,
                )
                for item in value
            )
        if not isinstance(value, str):
            return value
        candidate = Path(value)
        if not candidate.is_absolute():
            return value
        try:
            relative = candidate.resolve().relative_to(source_root.resolve())
        except Exception:
            return value
        rewritten = (target_root / relative).resolve()
        if rewritten.exists():
            return str(rewritten)
        return value

    def _merge_official_backend_output_dir(self, source_root: Path) -> None:
        target_root = Path(self.output_dir or "").resolve()
        target_root.mkdir(parents=True, exist_ok=True)
        if not source_root.exists():
            return
        for child in source_root.iterdir():
            # Keep the backend_race subtree as the canonical location for bulky planner
            # intermediates. The root log directory only needs the selected plan-solve and
            # official-verification
            # artifacts that downstream code reads directly.
            if child.is_dir():
                continue
            if not (
                child.name.startswith("plan_solve_")
                or child.name.startswith("plan_verification_")
                or child.suffix.lower() in {".json", ".txt"}
            ):
                continue
            destination = target_root / child.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)

    def _official_problem_root_planning_tasks(
        self,
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[PlanningBackendTask, ...]:
        build_result = self._build_problem_representations(
            timeout_seconds=timeout_seconds,
        )
        return expand_backend_tasks_for_representations(
            representations=build_result.representations,
            backends=default_official_backends(workspace=str(self.output_dir)),
        )

    def _run_official_problem_root_backend_race(self) -> Dict[str, Any]:
        if self.output_dir is None:
            raise ValueError("Official problem-root backend race requires an output directory.")

        planning_timeout_seconds = self._official_problem_root_planning_timeout_seconds()
        representation_build_start = time.perf_counter()
        planning_tasks = self._official_problem_root_planning_tasks(
            timeout_seconds=planning_timeout_seconds,
        )
        representation_build_seconds = time.perf_counter() - representation_build_start
        backend_root = Path(self.output_dir) / "backend_race"
        backend_root.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        processes: Dict[str, multiprocessing.Process] = {}
        attempts: List[Dict[str, Any]] = []
        race_start = time.perf_counter()
        selected_attempt: Optional[Dict[str, Any]] = None
        pending_tasks: Set[str] = set()

        def launch_task(planning_task: PlanningBackendTask) -> None:
            remaining_timeout = max(
                planning_timeout_seconds - (time.perf_counter() - race_start),
                1.0,
            )
            attempt_output_dir = backend_root / planning_task.task_id
            attempt_output_dir.mkdir(parents=True, exist_ok=True)
            process = context.Process(
                target=_official_problem_root_planning_task_worker,
                kwargs={
                    "result_queue": result_queue,
                    "domain_file": str(Path(self.domain_file).resolve()),
                    "problem_file": str(Path(self.problem_file).resolve()),
                    "output_dir": str(attempt_output_dir.resolve()),
                    "task_payload": planning_task.to_dict(),
                    "planning_timeout_seconds": remaining_timeout,
                },
            )
            process.start()
            processes[planning_task.task_id] = process
            pending_tasks.add(planning_task.task_id)

        for planning_task in planning_tasks:
            launch_task(planning_task)

        deadline = race_start + planning_timeout_seconds + 5.0
        while pending_tasks:
            try:
                wait_seconds = max(min(deadline - time.perf_counter(), 5.0), 0.1)
                message = result_queue.get(timeout=wait_seconds)
            except queue.Empty:
                if time.perf_counter() >= deadline:
                    break
                if not any(process.is_alive() for process in processes.values()):
                    break
                continue

            attempt = dict(message)
            attempts.append(attempt)
            task_id = str(attempt.get("task_id") or "")
            pending_tasks.discard(task_id)
            if bool(attempt.get("success")):
                selected_attempt = dict(attempt)
                break

        if selected_attempt is not None:
            for task_id, process in processes.items():
                if task_id == selected_attempt.get("task_id"):
                    continue
                self._terminate_official_backend_process(process)
        for process in processes.values():
            process.join(timeout=1.0)
            if process.is_alive():
                self._terminate_official_backend_process(process)
                process.join(timeout=1.0)

        if selected_attempt is None:
            selected_attempt = self._select_official_problem_root_backend_attempt(attempts)

        return {
            "planning_tasks": [task.to_dict() for task in planning_tasks],
            "attempts": attempts,
            "selected_attempt": selected_attempt,
            "representation_build_seconds": representation_build_seconds,
            "race_wallclock_seconds": time.perf_counter() - race_start,
        }

    @staticmethod
    def _terminate_official_backend_process(process: multiprocessing.Process) -> None:
        if not process.is_alive():
            return
        try:
            if os.name == "posix":
                process_group_id = os.getpgid(process.pid)
                os.killpg(process_group_id, signal.SIGTERM)
            else:
                process.terminate()
        except Exception:
            process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            try:
                if os.name == "posix":
                    process_group_id = os.getpgid(process.pid)
                    os.killpg(process_group_id, signal.SIGKILL)
                else:
                    process.kill()
            except Exception:
                process.kill()

    def _execute_official_problem_root_parallel_solver_race(
        self,
        method_library: Optional[HTNMethodLibrary] = None,
    ) -> Dict[str, Any]:
        print("\n[PLAN SOLVE]")
        print("-" * 80)
        race_result = self._run_official_problem_root_backend_race()
        planning_tasks = list(race_result.get("planning_tasks") or ())
        task_labels = [
            (
                f"{task.get('backend_name')}@"
                f"{((task.get('representation') or {}).get('representation_id') or 'unknown')}"
            )
            for task in planning_tasks
        ]
        print(f"• Running parallel official planning tasks: {', '.join(task_labels)}")
        attempts = list(race_result.get("attempts") or ())
        selected_attempt = dict(race_result.get("selected_attempt") or {})
        selected_backend = str(selected_attempt.get("backend_name") or "unknown")
        selected_representation = str(selected_attempt.get("representation_id") or "unknown")
        selected_output_dir = Path(str(selected_attempt.get("output_dir") or self.output_dir)).resolve()
        self._merge_official_backend_output_dir(selected_output_dir)

        plan_solve_data = copy.deepcopy(selected_attempt.get("plan_solve_data") or {})
        plan_verification_data = copy.deepcopy(selected_attempt.get("plan_verification_data") or {})
        plan_solve_data = self._rewrite_artifact_root_paths(
            plan_solve_data,
            selected_output_dir,
            Path(self.output_dir).resolve(),
        )
        plan_verification_data = self._rewrite_artifact_root_paths(
            plan_verification_data,
            selected_output_dir,
            Path(self.output_dir).resolve(),
        )

        attempt_summaries = [
            {
                "backend_name": str(attempt.get("backend_name") or "unknown"),
                "task_id": str(attempt.get("task_id") or "unknown"),
                "representation_id": str(attempt.get("representation_id") or "unknown"),
                "success": bool(attempt.get("success")),
                "selected_bucket": attempt.get("selected_bucket"),
                "plan_solve_status": ((attempt.get("plan_solve_data") or {}).get("summary") or {}).get("status"),
                "plan_verification_status": ((attempt.get("plan_verification_data") or {}).get("summary") or {}).get("status"),
                "total_seconds": attempt.get("total_seconds"),
                "stdout": attempt.get("stdout"),
                "stderr": attempt.get("stderr"),
            }
            for attempt in attempts
        ]

        plan_solve_summary = dict((plan_solve_data.get("summary") or {}))
        plan_solve_artifacts = dict((plan_solve_data.get("artifacts") or {}))
        plan_verification_summary = dict((plan_verification_data.get("summary") or {}))
        plan_verification_artifacts = dict((plan_verification_data.get("artifacts") or {}))
        plan_solve_summary.update(
            {
                "solver_race_strategy": OFFICIAL_PARALLEL_SELECTION_RULE,
                "selected_backend": selected_backend,
                "selected_representation": selected_representation,
                "backend_attempt_count": len(attempt_summaries),
            }
        )
        plan_solve_artifacts.update(
            {
                "solver_race_strategy": OFFICIAL_PARALLEL_SELECTION_RULE,
                "selected_backend": selected_backend,
                "selected_representation": selected_representation,
                "backend_attempts": attempt_summaries,
            }
        )
        plan_verification_summary.update(
            {
                "selection_rule": OFFICIAL_PARALLEL_SELECTION_RULE,
                "selected_backend": selected_backend,
                "selected_representation": selected_representation,
                "backend_attempt_count": len(attempt_summaries),
            }
        )
        plan_verification_artifacts.update(
            {
                "selection_rule": OFFICIAL_PARALLEL_SELECTION_RULE,
                "selected_backend": selected_backend,
                "selected_representation": selected_representation,
                "backend_attempts": attempt_summaries,
            }
        )
        plan_solve_data = {"summary": plan_solve_summary, "artifacts": plan_solve_artifacts}
        plan_verification_data = {"summary": plan_verification_summary, "artifacts": plan_verification_artifacts}

        plan_solve_status_label = "Success" if plan_solve_summary.get("status") == "success" else "Failed"
        self.logger.log_plan_solve(
            plan_solve_artifacts,
            plan_solve_status_label,
            error=None if plan_solve_status_label == "Success" else "parallel backend race failed",
            metadata=plan_solve_summary,
        )
        self.logger.record_step_timing(
            "plan_solve",
            float(selected_attempt.get("plan_solve_seconds") or 0.0)
            + float(race_result.get("representation_build_seconds") or 0.0),
            metadata={
                "selected_backend": selected_backend,
                "selected_representation": selected_representation,
                "representation_build_seconds": round(
                    float(race_result.get("representation_build_seconds") or 0.0),
                    6,
                ),
                "race_wallclock_seconds": round(float(race_result.get("race_wallclock_seconds") or 0.0), 6),
                "backend_attempt_count": len(attempt_summaries),
            },
        )
        if plan_solve_summary.get("status") == "success":
            print(
                f"✓ Planner returned via backend: {selected_backend} "
                f"on {selected_representation}"
            )
        else:
            print(
                "✗ Plan solve failed across all planning tasks "
                f"(selected failure: {selected_backend} on {selected_representation})"
            )

        print("\n[OFFICIAL VERIFICATION]")
        print("-" * 80)
        plan_verification_status_label = "Success" if plan_verification_summary.get("status") == "success" else "Failed"
        self.logger.log_official_verification(
            plan_verification_artifacts,
            plan_verification_status_label,
            error=(
                None
                if plan_verification_status_label == "Success"
                else "all parallel official backends failed verification"
            ),
            metadata=plan_verification_summary,
        )
        self.logger.record_step_timing(
            "plan_verification",
            float(selected_attempt.get("plan_verification_seconds") or 0.0),
            metadata={
                "selected_backend": selected_backend,
                "selected_representation": selected_representation,
                "race_wallclock_seconds": round(float(race_result.get("race_wallclock_seconds") or 0.0), 6),
                "backend_attempt_count": len(attempt_summaries),
            },
        )
        if plan_verification_summary.get("status") == "success":
            print("✓ Official IPC verification complete")
            print(f"  Selected backend: {selected_backend}")
            print(f"  Selected representation: {selected_representation}")
            print(f"  Verification result: {plan_verification_artifacts.get('verification_result')}")
        else:
            print("✗ Official verification failed: all planning tasks failed")
            print(f"  Selected failure backend: {selected_backend}")
            print(f"  Selected failure representation: {selected_representation}")

        return {
            "plan_solve": plan_solve_data,
            "plan_verification": plan_verification_data,
        }

    def _should_sequence_goal_request(
        self,
        planning_request: PlanningRequest,
    ) -> bool:
        nodes = tuple(planning_request.goal_request.nodes or ())
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

    def _solve_ordered_goal_sequence(
        self,
        *,
        planner: PANDAPlanner,
        method_library: HTNMethodLibrary,
        planning_request: PlanningRequest,
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
                timeout_seconds=float(self.config.planning_timeout),
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
                    "Sequential plan replay failed after "
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

        hierarchical_plan_text = self._render_supported_hierarchical_plan(
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

    def _render_supported_hierarchical_plan(
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
        return tuple(schemas)

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
        Prefer grounded query objects over free-form goal-grounding placeholder tokens.

        Task-grounded benchmark queries already enumerate the relevant object
        inventory in natural language. When the goal-grounding response echoes schema-like
        placeholders such as `ROVER` inside `objects`, those placeholders should
        not leak into plan solve. Rebuild the semantic object list from the
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

    def _synthesise_domain_methods(self):
        """Synthesise one domain-complete HTN method library."""
        print("\n[METHOD SYNTHESIS]")
        print("-"*80)
        stage_start = time.perf_counter()

        try:
            synthesizer = HTNMethodSynthesizer(
                api_key=self.config.openai_api_key,
                model=self.config.method_synthesis_model,
                base_url=self.config.openai_base_url,
                timeout=float(self.config.method_synthesis_timeout),
                max_tokens=int(self.config.method_synthesis_max_tokens),
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
            self._record_step_timing(
                "method_synthesis",
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
            self.logger.log_method_synthesis(
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
            self._record_step_timing("method_synthesis", stage_start)
            self.logger.log_method_synthesis(
                None,
                "Failed",
                error=str(e),
                model=getattr(e, "model", None),
                llm_prompt=getattr(e, "llm_prompt", None),
                llm_response=getattr(e, "llm_response", None),
                metadata=getattr(e, "metadata", None),
            )
            print(f"✗ Method synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _validate_domain_library(self, method_library):
        """Run the domain gate once per declared compound task."""
        print("\n[DOMAIN GATE]")
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
                    "Domain gate missing declared compound tasks: "
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
                    "Domain gate found undefined compound children: "
                    f"{undefined_child_names}",
                )

            gate_cases = self._validate_domain_library_cases(method_library)
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
                    self._compact_domain_gate_library_for_task(
                        method_library,
                        task_name,
                        task_args,
                        target_literal=None,
                        compact_arg_threshold=self.DOMAIN_GATE_COMPACT_TASK_ARG_THRESHOLD,
                        max_compound_steps=self.DOMAIN_GATE_MAX_VALIDATION_COMPOUND_STEPS,
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
                        "plan": self._plan_artifact_summary(plan),
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
            self._record_step_timing(
                "domain_gate",
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
            self.logger.log_domain_gate(
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
            self._record_step_timing("domain_gate", stage_start)
            self.logger.log_domain_gate(
                None,
                "Failed",
                error=str(e),
                metadata={"gate_type": "domain_complete"},
            )
            print(f"✗ Domain gate failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _validate_domain_library_cases(self, method_library) -> Tuple[Dict[str, Any], ...]:
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






    @staticmethod
    def _compact_domain_gate_library_for_task(
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
        compacted_library = DomainCompletePipeline._prune_method_library_for_task(
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
    def _prune_method_library_for_task(method_library, root_task_name: str):
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

    def _plan_artifact_summary(self, plan) -> Dict[str, Any]:
        """Return a compact, path-based domain-gate plan summary for persisted logs."""
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
                    artifacts[artifact_name] = self._relative_artifact_path(artifact_path)

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
                self._relative_artifact_path(work_dir)
                if work_dir is not None
                else None
            ),
            "artifacts": artifacts,
            "timing_profile": dict(getattr(plan, "timing_profile", {}) or {}),
        }


    def _relative_artifact_path(self, artifact_path: str | Path) -> str:
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
                    scope=f"Domain-gate task '{task_name}' object typing",
                )
                continue
            object_types[obj] = self._resolve_symbol_type(
                symbol=obj,
                candidate_types=type_candidates.get(obj, set()),
                scope=f"Domain-gate task '{task_name}' object typing",
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
                "Missing resolved object types for domain-gate problem export: "
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
                scope=f"Method-synthesis method '{method.method_name}' variable typing",
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
                    f"Method-synthesis binding references missing target literal "
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
                    f"Method-synthesis binding references missing target literal "
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
                        "Method-synthesis target-task binding typing "
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
                scope=f"Domain-gate method '{method.method_name}' object typing",
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
                    scope=f"Domain-gate method '{method.method_name}' object typing",
                )
                object_types[bound_object] = actual_type
            if not self._is_subtype(actual_type, expected_type):
                raise TypeResolutionError(
                    f"Domain-gate method '{method.method_name}' binds parameter '{parameter}' "
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
                        f"Domain-gate method '{method.method_name}' cannot type variable "
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
                        f"Domain-gate child witness typing cannot resolve variable '{token}' "
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
                f"Domain-gate task '{task_name}' has no type-compatible method branch "
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





    def _verify_plan_officially(self, ltl_spec, method_library, plan_solve_data):
        """Verify the generated hierarchical plan with the official IPC verifier."""
        print("\n[OFFICIAL VERIFICATION]")
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
            self.logger.log_official_verification(
                artifacts,
                "Skipped",
                metadata=summary,
            )
            self._record_step_timing("plan_verification", stage_start)
            print("• Skipped: no problem file was provided")
            return {
                "summary": summary,
                "artifacts": artifacts,
            }

        verifier = IPCPlanVerifier()
        if not verifier.tool_available():
            error = "pandaPIparser is not available on PATH for official IPC verification"
            self.logger.log_official_verification(
                None,
                "Failed",
                error=error,
                metadata={
                    "backend": "pandaPIparser",
                    "status": "failed",
                },
            )
            self._record_step_timing("plan_verification", stage_start)
            print(f"✗ Official verification failed: {error}")
            return None

        plan_solve_artifacts = plan_solve_data.get("artifacts") or {}
        planning_mode = str(plan_solve_artifacts.get("planning_mode") or "")
        if planning_mode == "official_problem_root":
            return self._verify_problem_root_solver_race(
                verifier=verifier,
                plan_solve_data=plan_solve_data,
                plan_solve_artifacts=plan_solve_artifacts,
                stage_start=stage_start,
            )

        if planning_mode != "official_problem_root":
            domain_build_start = time.perf_counter()
            verification_domain_file = self._build_verification_domain(method_library)
            domain_build_seconds = time.perf_counter() - domain_build_start
        guided_plan_text = plan_solve_artifacts.get("guided_hierarchical_plan_text")
        verifier_start = time.perf_counter()
        if guided_plan_text:
            guided_plan_text = self._rewrite_guided_plan_source_names(
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
                action_path=plan_solve_artifacts.get("action_path") or [],
                method_library=method_library,
                method_trace=plan_solve_artifacts.get("method_trace") or [],
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
            self.logger.log_official_verification(
                artifacts,
                "Failed",
                error=error,
                metadata=summary,
            )
            self._record_step_timing(
                "plan_verification",
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
            print(f"✗ Official verification failed: {error}")
            return None

        self.logger.log_official_verification(
            artifacts,
            "Success",
            metadata=summary,
        )
        self._record_step_timing(
            "plan_verification",
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

    def _verify_problem_root_solver_race(
        self,
        *,
        verifier: IPCPlanVerifier,
        plan_solve_data: Dict[str, Any],
        plan_solve_artifacts: Dict[str, Any],
        stage_start: float,
    ) -> Dict[str, Any]:
        verification_domain_file = Path(self.domain_file).resolve()
        plan_solve_summary = dict(plan_solve_data.get("summary") or {})
        solver_candidates = list(plan_solve_artifacts.get("solver_candidates") or ())

        if plan_solve_summary.get("status") != "success" or not solver_candidates:
            summary = {
                "backend": "pandaPIparser",
                "status": "failed",
                "selection_rule": "first_hierarchical_verification_success",
                "failure_bucket": "no_plan_from_solver",
                "solver_candidate_count": len(solver_candidates),
                "verification_domain_file": str(verification_domain_file),
                "verification_problem_file": str(Path(self.problem_file).resolve()),
            }
            artifacts = {
                "tool_available": verifier.tool_available(),
                "plan_kind": None,
                "verification_result": False,
                "primitive_plan_executable": None,
                "reached_goal_state": None,
                "selected_solver_id": None,
                "selected_bucket": "no_plan_from_solver",
                "selection_rule": "first_hierarchical_verification_success",
                "solver_candidates": solver_candidates,
            }
            self.logger.log_official_verification(
                artifacts,
                "Failed",
                error="Plan solve produced no executable solver candidate for the official baseline",
                metadata=summary,
            )
            self._record_step_timing("plan_verification", stage_start, metadata={"failure_bucket": "no_plan_from_solver"})
            print("✗ Official verification failed: no executable solver candidate was available")
            return {"summary": summary, "artifacts": artifacts}

        selected_success: Optional[Dict[str, Any]] = None
        selected_primitive_valid: Optional[Dict[str, Any]] = None
        verified_candidates: List[Dict[str, Any]] = []

        for candidate in solver_candidates:
            solver_id = str(candidate.get("solver_id") or candidate.get("mode") or "unknown").strip()
            solver_mode = str(candidate.get("mode") or "").strip() or None
            candidate_status = str(candidate.get("status") or "").strip() or "unknown"
            if candidate_status != "success":
                verified_candidates.append(
                    {
                        "solver_id": solver_id,
                        "mode": solver_mode,
                        "status": candidate_status,
                        "bucket": "no_plan_from_solver",
                        "command": list(candidate.get("command") or ()),
                        "reason": candidate.get("reason"),
                        "step_count": candidate.get("step_count"),
                    }
                )
                continue

            candidate_output_dir = self.output_dir / "solver_portfolio" / sanitize_identifier(solver_id)
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            action_path = list(candidate.get("action_path") or ())
            primitive_result = verifier.verify_primitive_plan(
                domain_file=verification_domain_file,
                problem_file=self.problem_file,
                action_path=action_path,
                output_dir=candidate_output_dir,
                plan_filename="primitive_plan.txt",
                output_filename="primitive_verifier.txt",
                json_filename="primitive_verification.json",
            )

            candidate_actual_plan_text = ""
            candidate_actual_plan_path = candidate.get("actual_plan_path")
            if candidate_actual_plan_path:
                actual_plan_path = Path(str(candidate_actual_plan_path))
                if actual_plan_path.exists():
                    candidate_actual_plan_text = actual_plan_path.read_text()
            hierarchical_result = verifier.verify_plan_text(
                domain_file=verification_domain_file,
                problem_file=self.problem_file,
                plan_text=candidate_actual_plan_text,
                output_dir=candidate_output_dir,
                plan_kind="hierarchical",
                build_warning=None,
                plan_filename="hierarchical_plan.txt",
                output_filename="hierarchical_verifier.txt",
                json_filename="hierarchical_verification.json",
            )

            primitive_valid = (
                primitive_result.tool_available is True
                and primitive_result.primitive_plan_executable is True
                and primitive_result.verification_result is True
                and primitive_result.reached_goal_state is True
            )
            hierarchical_valid = (
                hierarchical_result.tool_available is True
                and hierarchical_result.plan_kind == "hierarchical"
                and hierarchical_result.verification_result is True
            )
            if hierarchical_valid:
                bucket = "hierarchical_plan_verified"
            elif primitive_valid:
                bucket = "primitive_plan_valid_but_hierarchical_rejected"
            else:
                bucket = "primitive_plan_invalid"

            record = {
                "solver_id": solver_id,
                "mode": solver_mode,
                "status": candidate_status,
                "bucket": bucket,
                "command": list(candidate.get("command") or ()),
                "step_count": candidate.get("step_count"),
                "action_path": action_path,
                "raw_plan_path": candidate.get("raw_plan_path"),
                "actual_plan_path": candidate_actual_plan_path,
                "verification_output_dir": str(candidate_output_dir),
                "primitive_verification": primitive_result.to_dict(),
                "hierarchical_verification": hierarchical_result.to_dict(),
            }
            verified_candidates.append(record)
            if bucket == "hierarchical_plan_verified" and selected_success is None:
                selected_success = record
            if bucket == "primitive_plan_valid_but_hierarchical_rejected" and selected_primitive_valid is None:
                selected_primitive_valid = record

        selected_record = (
            selected_success
            or selected_primitive_valid
            or (verified_candidates[0] if verified_candidates else None)
        )
        selected_bucket = (
            str(selected_record.get("bucket"))
            if isinstance(selected_record, dict)
            else "no_plan_from_solver"
        )
        selected_solver_id = (
            str(selected_record.get("solver_id"))
            if isinstance(selected_record, dict) and selected_record.get("solver_id") is not None
            else None
        )

        if selected_success is not None:
            selected_hierarchical = dict(selected_success.get("hierarchical_verification") or {})
            selected_primitive = dict(selected_success.get("primitive_verification") or {})
            composite_artifacts = {
                **selected_hierarchical,
                "primitive_plan_executable": selected_primitive.get("primitive_plan_executable"),
                "reached_goal_state": selected_primitive.get("reached_goal_state"),
            }
        elif selected_primitive_valid is not None:
            selected_hierarchical = dict(selected_primitive_valid.get("hierarchical_verification") or {})
            selected_primitive = dict(selected_primitive_valid.get("primitive_verification") or {})
            composite_artifacts = {
                **selected_hierarchical,
                "tool_available": selected_hierarchical.get("tool_available", selected_primitive.get("tool_available")),
                "primitive_plan_executable": selected_primitive.get("primitive_plan_executable"),
                "reached_goal_state": selected_primitive.get("reached_goal_state"),
                "verification_result": selected_hierarchical.get("verification_result"),
            }
        else:
            composite_artifacts = {
                "tool_available": verifier.tool_available(),
                "plan_kind": None,
                "verification_result": False,
                "primitive_plan_executable": None,
                "reached_goal_state": None,
            }

        selected_verification_dir = None
        selected_json_path = None
        if isinstance(selected_record, dict):
            selected_verification_dir = selected_record.get("verification_output_dir")
        if selected_verification_dir:
            selected_verification_dir = str(selected_verification_dir)
            selected_json_path = str(
                Path(selected_verification_dir)
                / (
                    "hierarchical_verification.json"
                    if selected_record and selected_record.get("status") == "success"
                    else "primitive_verification.json"
                )
            )
        official_plan_file = self.output_dir / "ipc_official_plan.txt"
        official_output_file = self.output_dir / "ipc_official_verifier.txt"
        official_json_file = self.output_dir / "ipc_official_verification.json"
        source_plan_file = composite_artifacts.get("plan_file")
        source_output_file = composite_artifacts.get("output_file")
        if source_plan_file and Path(str(source_plan_file)).exists():
            shutil.copyfile(Path(str(source_plan_file)), official_plan_file)
            composite_artifacts["plan_file"] = str(official_plan_file)
        if source_output_file and Path(str(source_output_file)).exists():
            shutil.copyfile(Path(str(source_output_file)), official_output_file)
            composite_artifacts["output_file"] = str(official_output_file)
        if selected_json_path and Path(str(selected_json_path)).exists():
            shutil.copyfile(Path(str(selected_json_path)), official_json_file)
            composite_artifacts["json_file"] = str(official_json_file)

        artifacts = {
            **composite_artifacts,
            "selected_solver_id": selected_solver_id,
            "selected_bucket": selected_bucket,
            "selection_rule": "first_hierarchical_verification_success",
            "solver_candidates": verified_candidates,
        }
        summary = {
            "backend": "pandaPIparser",
            "status": "success" if selected_success is not None else "failed",
            "selection_rule": "first_hierarchical_verification_success",
            "failure_bucket": None if selected_success is not None else selected_bucket,
            "selected_solver_id": selected_solver_id,
            "selected_bucket": selected_bucket,
            "solver_candidate_count": len(verified_candidates),
            "verified_success_count": sum(
                1
                for candidate in verified_candidates
                if candidate.get("bucket") == "hierarchical_plan_verified"
            ),
            "primitive_valid_failure_count": sum(
                1
                for candidate in verified_candidates
                if candidate.get("bucket") == "primitive_plan_valid_but_hierarchical_rejected"
            ),
            "verification_domain_file": str(verification_domain_file),
            "verification_problem_file": str(Path(self.problem_file).resolve()),
        }

        status_label = "Success" if selected_success is not None else "Failed"
        error = None
        if selected_success is None:
            error = (
                "Official IPC verifier found no hierarchically verified solver candidate; "
                f"best bucket={selected_bucket}"
            )
        self.logger.log_official_verification(
            artifacts,
            status_label,
            error=error,
            metadata=summary,
        )
        self._record_step_timing(
            "plan_verification",
            stage_start,
            metadata={
                "selected_solver_id": selected_solver_id,
                "selected_bucket": selected_bucket,
                "solver_candidate_count": len(verified_candidates),
            },
        )
        if selected_success is None:
            print(
                "✗ Official verification failed: no solver candidate passed hierarchical verification "
                f"(best bucket: {selected_bucket})"
            )
        else:
            print("✓ Official IPC verification complete")
            print(f"  Selected solver: {selected_solver_id}")
            print(f"  Verification result: {artifacts.get('verification_result')}")
        return {"summary": summary, "artifacts": artifacts}

    @staticmethod
    def _rewrite_guided_plan_source_names(plan_text, method_library):
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

    def _build_verification_domain(self, method_library):
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
