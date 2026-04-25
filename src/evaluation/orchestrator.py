"""
Evaluation orchestrator for task grounding, Jason execution, and verification evidence.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from method_library.synthesis.naming import sanitize_identifier
from method_library.synthesis.schema import HTNMethodLibrary
from evaluation.agentspeak import AgentSpeakRenderer
from evaluation.failure_signature import build_failure_signature
from evaluation.domain_selection import (
	EvaluationDomainContext,
	normalize_evaluation_domain_source,
)
from evaluation.goal_grounding.grounder import (
	GoalGroundingProviderUnavailable,
	NLToLTLfGenerator,
)
from evaluation.jason_runtime import JasonRunner
from evaluation.jason_runtime.runner import JasonValidationError
from evaluation.runtime_context import (
	action_type_map_for_domain,
	build_type_parent_map_for_domain,
	planner_action_schemas_for_domain,
	predicate_type_map_for_domain,
	render_problem_fact,
	resolve_evaluation_domain_context,
	task_type_map_for_domain,
	typed_object_entries,
	validate_problem_domain_compatibility,
)
from evaluation.official_verification import (
	render_supported_hierarchical_plan,
	verify_jason_hierarchical_plan,
)
from evaluation.artifacts import (
	JasonExecutionResult,
	TemporalGroundingResult,
)
from execution_logging.execution_logger import ExecutionLogger
from plan_library.artifacts import PlanLibraryArtifactBundle, load_plan_library_artifact_bundle
from plan_library.models import PlanLibrary
from utils.config import get_config
from utils.hddl_parser import HDDLParser


class PlanLibraryEvaluationOrchestrator:
	"""Evaluate one natural-language instruction or stored temporal specification."""

	def __init__(
		self,
		*,
		domain_file: str,
		problem_file: str | None = None,
		evaluation_domain_source: str | None = None,
		runtime_backend: str | None = None,
	) -> None:
		self.config = get_config()
		self.project_root = Path(__file__).resolve().parents[2]
		self.evaluation_tmp_root = self.project_root / "tmp" / "evaluation"
		self.logger = ExecutionLogger(logs_dir=str(self.evaluation_tmp_root))

		if not domain_file:
			raise ValueError("domain_file is required for plan-library evaluation.")

		self.domain_file = str(Path(domain_file).expanduser().resolve())
		self.problem_file = (
			str(Path(problem_file).expanduser().resolve())
			if problem_file is not None
			else None
		)
		self.evaluation_domain_source = normalize_evaluation_domain_source(
			evaluation_domain_source or self.config.evaluation_domain_source,
		)
		self.runtime_backend = str(runtime_backend or "jason").strip().lower()
		if self.runtime_backend != "jason":
			raise ValueError(f"Unsupported runtime backend '{self.runtime_backend}'.")

		self.domain = HDDLParser.parse_domain(self.domain_file)
		self.problem = HDDLParser.parse_problem(self.problem_file) if self.problem_file else None
		self.type_parent_map = build_type_parent_map_for_domain(self.domain)
		self.domain_type_names = set(self.type_parent_map.keys())
		self.predicate_type_map = predicate_type_map_for_domain(
			self.domain,
			self.domain_type_names,
		)
		self.action_type_map = action_type_map_for_domain(self.domain, self.domain_type_names)
		self.task_type_map = task_type_map_for_domain(self.domain, self.domain_type_names)
		validate_problem_domain_compatibility(
			problem=self.problem,
			domain_type_names=self.domain_type_names,
			type_parent_map=self.type_parent_map,
			predicate_type_map=self.predicate_type_map,
			task_type_map=self.task_type_map,
		)

		self.output_dir: Optional[Path] = None
		self._last_goal_grounding_failure_class = ""
		self._last_goal_grounding_error = ""

	def run_query(self, nl_instruction: str) -> Dict[str, Any]:
		"""
		Execute one query against the cached domain library artifact.

		This evaluation path does not synthesize methods. If the cache is absent,
		callers must run plan-library generation first or pass `library_artifact`
		explicitly.
		"""

		self._start_query_run(nl_instruction, mode="plan_library_evaluation")
		try:
			artifact = self._load_default_plan_library_artifact()
			result = self._execute_query_with_loaded_library(nl_instruction, artifact)
		except Exception as exc:
			result = {
				"success": False,
				"step": "domain_library",
				"error": str(exc),
			}
		log_filepath = self.logger.end_pipeline(success=result.get("success", False))
		result["log_path"] = str(log_filepath)
		return result

	def execute_query_with_library(
		self,
		nl_query: str,
		*,
		library_artifact: Any,
		execution_mode: str = "plan_library_evaluation",
	) -> Dict[str, Any]:
		"""Execute one query against a supplied cached or in-memory method library."""

		self._start_query_run(nl_query, mode=execution_mode)
		result = self._execute_query_with_loaded_library(
			nl_query,
			load_plan_library_artifact_bundle(library_artifact),
		)
		log_filepath = self.logger.end_pipeline(success=result.get("success", False))
		result["log_path"] = str(log_filepath)
		return result

	def execute_grounded_query_with_library(
		self,
		nl_query: str,
		*,
		library_artifact: Any,
		grounding_result: TemporalGroundingResult,
		execution_mode: str = "plan_library_evaluation",
	) -> Dict[str, Any]:
		"""Execute one query using a precomputed temporal specification and subgoal grounding."""

		self._start_query_run(nl_query, mode=execution_mode)
		result = self._execute_query_with_loaded_library_and_grounding(
			nl_query,
			load_plan_library_artifact_bundle(library_artifact),
			grounding_result=grounding_result,
		)
		log_filepath = self.logger.end_pipeline(success=result.get("success", False))
		result["log_path"] = str(log_filepath)
		return result

	def _start_query_run(self, nl_instruction: str, *, mode: str = "plan_library_evaluation") -> None:
		self.logger.start_pipeline(
			nl_instruction,
			mode=mode,
			domain_file=self.domain_file,
			problem_file=self.problem_file,
			domain_name=self.domain.name,
			problem_name=self.problem.name if self.problem is not None else None,
			output_dir=str(self.evaluation_tmp_root),
		)
		self.output_dir = Path(self.logger.current_log_dir).resolve()
		if self.logger.current_record is not None:
			self.logger.current_record.output_dir = str(self.output_dir)
			self.logger._save_current_state()

	def _load_default_plan_library_artifact(self) -> PlanLibraryArtifactBundle:
		artifact_root = (
			self.project_root
			/ "artifacts"
			/ "plan_library"
			/ sanitize_identifier(self.domain.name)
		)
		if not (artifact_root / "plan_library.json").exists():
			raise FileNotFoundError(
				"Cached plan-library artifact not found. "
				"Pass --library-artifact or run plan-library generation first: "
				f"{artifact_root}",
			)
		return load_plan_library_artifact_bundle(artifact_root)

	def _execute_query_with_loaded_library(
		self,
		nl_instruction: str,
		artifact: PlanLibraryArtifactBundle,
	) -> Dict[str, Any]:
		"""Run the Jason query path with a prebuilt plan-library bundle."""

		return self._execute_query_with_loaded_library_and_grounding(
			nl_instruction,
			artifact,
			grounding_result=None,
		)

	def _execute_query_with_loaded_library_and_grounding(
		self,
		nl_instruction: str,
		artifact: PlanLibraryArtifactBundle,
		*,
		grounding_result: TemporalGroundingResult | None,
	) -> Dict[str, Any]:
		"""Run the Jason query path with either live or precomputed temporal grounding."""

		domain_library = artifact.method_library
		plan_library = artifact.plan_library
		evaluation_domain = self._resolve_evaluation_domain_context(artifact)
		if self.logger.current_record is not None:
			self.logger.current_record.domain_file = evaluation_domain.domain_file
			self.logger.current_record.domain_name = evaluation_domain.domain.name
			self.logger._save_current_state()

		grounding_prompt = None
		grounding_response = None
		if grounding_result is None:
			grounding_result, grounding_prompt, grounding_response = self._ground_query_temporally(
				nl_instruction,
				domain_library,
				evaluation_domain=evaluation_domain,
			)
			if grounding_result is None:
				failure_class = self._last_goal_grounding_failure_class or "goal_grounding_failed"
				return {
					"success": False,
					"step": "goal_grounding",
					"error": self._last_goal_grounding_error or "Temporal goal grounding failed",
					"failure_class": failure_class,
					"method_library": domain_library.to_dict(),
				}
		else:
			self._accept_precomputed_grounding_result(
				grounding_result,
				evaluation_domain=evaluation_domain,
			)

		verification_problem_file, verification_mode = self._determine_verification_problem()
		if verification_problem_file is None:
			return {
				"success": False,
				"step": "plan_verification",
				"error": "Could not determine a verification problem file",
				"method_library": domain_library.to_dict(),
				"goal_grounding": grounding_result.to_dict(),
			}

		agentspeak_render = self._render_agentspeak_program(
			grounding_result=grounding_result,
			method_library=domain_library,
			plan_library=plan_library,
			evaluation_domain=evaluation_domain,
		)
		if agentspeak_render is None:
			return {
				"success": False,
				"step": "agentspeak_rendering",
				"error": "AgentSpeak rendering failed",
				"method_library": domain_library.to_dict(),
				"goal_grounding": grounding_result.to_dict(),
			}

		runtime_result = self._execute_query_with_runtime_backend(
			grounding_result=grounding_result,
			method_library=domain_library,
			plan_library=plan_library,
			agentspeak_code=agentspeak_render["agentspeak_code"],
			agentspeak_artifacts=agentspeak_render["artifacts"],
			verification_problem_file=verification_problem_file,
			verification_mode=verification_mode,
			evaluation_domain=evaluation_domain,
		)
		if runtime_result is None:
			return {
				"success": False,
				"step": "runtime_execution",
				"error": f"{self.runtime_backend} runtime execution failed",
				"method_library": domain_library.to_dict(),
				"goal_grounding": grounding_result.to_dict(),
			}

		plan_solve_data = self._build_plan_solve_data(
			grounding_result=grounding_result,
			jason_result=runtime_result,
			evaluation_domain=evaluation_domain,
			verification_mode=verification_mode,
		)
		plan_verification_data = self._verify_plan_officially(
			method_library=domain_library,
			plan_solve_data=plan_solve_data,
			ltlf_formula=grounding_result.ltlf_formula,
			evaluation_domain=evaluation_domain,
		)
		if plan_verification_data is None:
			return {
				"success": False,
				"step": "plan_verification",
				"error": "Official IPC verification failed",
				"method_library": domain_library.to_dict(),
				"goal_grounding": grounding_result.to_dict(),
				"plan_solve": plan_solve_data,
			}

		return {
			"success": True,
			"method_library": domain_library.to_dict(),
			"goal_grounding": grounding_result.to_dict(),
			"plan_solve": plan_solve_data,
			"plan_verification": plan_verification_data,
			"llm_prompt": grounding_prompt,
			"llm_response": grounding_response,
		}

	def _accept_precomputed_grounding_result(
		self,
		grounding_result: TemporalGroundingResult,
		*,
		evaluation_domain: EvaluationDomainContext,
	) -> None:
		print("\n[GOAL GROUNDING]")
		print("-" * 80)
		stage_start = time.perf_counter()
		if not str(grounding_result.ltlf_formula or "").strip():
			raise ValueError("Precomputed temporal grounding requires a non-empty LTLf formula.")
		if not tuple(grounding_result.subgoals or ()):
			raise ValueError("Precomputed temporal grounding requires at least one grounded subgoal.")
		self.logger.log_goal_grounding_success(
			grounding_result.to_log_dict(),
			used_llm=False,
			model=None,
			llm_prompt=None,
			llm_response=None,
			metadata={
				"evaluation_domain_source": evaluation_domain.source,
				"grounding_mode": "precomputed_temporal_specification",
				"task_event_count": len(grounding_result.subgoals),
			},
		)
		self._record_failure_signature(ltlf_formula=grounding_result.ltlf_formula)
		self._record_step_timing(
			"goal_grounding",
			stage_start,
			metadata={
				"task_event_count": len(grounding_result.subgoals),
				"evaluation_domain_source": evaluation_domain.source,
				"grounding_mode": "precomputed_temporal_specification",
			},
		)
		print("✓ Loaded stored temporal specification")
		print(f"  LTLf: {self._preview_log_text(grounding_result.ltlf_formula)}")

	def _resolve_evaluation_domain_context(
		self,
		artifact: Optional[PlanLibraryArtifactBundle] = None,
		*,
		source: Optional[str] = None,
	) -> EvaluationDomainContext:
		return resolve_evaluation_domain_context(
			source_domain_file=self.domain_file,
			source_domain=self.domain,
			artifact_bundle=artifact,
			evaluation_domain_source=source or self.evaluation_domain_source,
			output_dir=self.output_dir,
			project_root=self.project_root,
		)

	def _ground_query_temporally(
		self,
		nl_instruction: str,
		method_library: HTNMethodLibrary,
		*,
		evaluation_domain: EvaluationDomainContext,
	) -> Tuple[Optional[TemporalGroundingResult], Optional[Dict[str, str]], Optional[str]]:
		print("\n[GOAL GROUNDING]")
		print("-" * 80)
		stage_start = time.perf_counter()
		generator: Optional[NLToLTLfGenerator] = None
		self._last_goal_grounding_failure_class = ""
		self._last_goal_grounding_error = ""

		try:
			generator = NLToLTLfGenerator(
				api_key=self.config.ltlf_generation_api_key,
				model=self.config.ltlf_generation_model,
				base_url=self.config.ltlf_generation_base_url,
				domain_file=evaluation_domain.domain_file,
				request_timeout=float(self.config.ltlf_generation_timeout),
				response_max_tokens=int(self.config.ltlf_generation_max_tokens),
				session_id=self.config.ltlf_generation_session_id,
			)
			typed_objects = (
				{
					str(name).strip(): str(type_name).strip()
					for name, type_name in dict(getattr(self.problem, "object_types", {}) or {}).items()
					if str(name).strip() and str(type_name).strip()
				}
				if self.problem is not None
				else {}
			)
			grounding_result, llm_prompt, llm_response = generator.generate(
				nl_instruction,
				method_library=method_library,
				typed_objects=typed_objects,
				task_type_map=evaluation_domain.task_type_map,
				type_parent_map=evaluation_domain.type_parent_map,
			)
			self.logger.log_goal_grounding_success(
				grounding_result.to_log_dict(),
				used_llm=True,
				model=self.config.ltlf_generation_model,
				llm_prompt=llm_prompt,
				llm_response=llm_response,
			)
			self._record_failure_signature(ltlf_formula=grounding_result.ltlf_formula)
			self._record_step_timing(
				"goal_grounding",
				stage_start,
				metadata={
					"task_event_count": len(grounding_result.subgoals),
					"evaluation_domain_source": evaluation_domain.source,
				},
			)
			print(f"✓ Grounded task events: {len(grounding_result.subgoals)}")
			print(f"  LTLf: {self._preview_log_text(grounding_result.ltlf_formula)}")
			return grounding_result, llm_prompt, llm_response
		except Exception as exc:
			failure_class = (
				"goal_grounding_provider_unavailable"
				if isinstance(exc, GoalGroundingProviderUnavailable)
				else "goal_grounding_failed"
			)
			self._last_goal_grounding_failure_class = failure_class
			self._last_goal_grounding_error = str(exc)
			generation_metadata = dict(getattr(generator, "last_generation_metadata", {}) or {})
			llm_prompt = generation_metadata.get("last_prompt")
			self.logger.log_goal_grounding_error(
				str(exc),
				model=self.config.ltlf_generation_model,
				llm_prompt=llm_prompt if isinstance(llm_prompt, dict) else None,
				llm_response=str(generation_metadata.get("last_response") or "") or None,
				metadata={
					"evaluation_domain_source": evaluation_domain.source,
					"failure_class": failure_class,
					"attempt_mode": generation_metadata.get("attempt_mode"),
					"attempt_count": generation_metadata.get("attempt_count"),
					"attempt_errors": generation_metadata.get("attempt_errors"),
				},
			)
			self._record_failure_signature()
			self._record_step_timing("goal_grounding", stage_start)
			print(f"✗ Goal grounding failed: {exc}")
			return None, None, None

	def _render_agentspeak_program(
		self,
		*,
		grounding_result: TemporalGroundingResult,
		method_library: HTNMethodLibrary,
		plan_library: PlanLibrary,
		evaluation_domain: EvaluationDomainContext,
	) -> Optional[Dict[str, Any]]:
		print("\n[AGENTSPEAK RENDERING]")
		print("-" * 80)
		stage_start = time.perf_counter()

		try:
			output_dir = self._require_output_dir()
			renderer = AgentSpeakRenderer()
			runtime_objects = tuple(
				str(object_name).strip()
				for object_name in (
					self.problem.objects
					if self.problem is not None
					else grounding_result.typed_objects.keys()
				)
				if str(object_name).strip()
			)
			typed_objects = typed_object_entries(runtime_objects, grounding_result.typed_objects)
			agentspeak_code = renderer.generate(
				domain=evaluation_domain.domain,
				objects=runtime_objects,
				method_library=method_library,
				plan_library=plan_library,
				plan_records=(),
				typed_objects=typed_objects,
				subgoals=[subgoal.to_dict() for subgoal in grounding_result.subgoals],
			)
			asl_path = output_dir / "query_runtime.asl"
			asl_path.write_text(agentspeak_code)
			artifacts = {
				"asl_file": str(asl_path),
				"plan_library_kind": "S",
				"runtime_rendering_role": "ungrounded_plan_library_rendering",
				"task_event_count": len(grounding_result.subgoals),
			}
			self.logger.log_agentspeak_rendering(
				artifacts,
				"Success",
				metadata={"task_event_count": len(grounding_result.subgoals)},
			)
			self._record_step_timing(
				"agentspeak_rendering",
				stage_start,
				metadata={
					"task_event_count": len(grounding_result.subgoals),
					"evaluation_domain_source": evaluation_domain.source,
				},
			)
			print(f"✓ AgentSpeak file: {asl_path}")
			return {
				"agentspeak_code": agentspeak_code,
				"artifacts": artifacts,
			}
		except Exception as exc:
			self.logger.log_agentspeak_rendering(None, "Failed", error=str(exc))
			self._record_step_timing("agentspeak_rendering", stage_start)
			print(f"✗ AgentSpeak rendering failed: {exc}")
			return None

	def _execute_query_with_runtime_backend(
		self,
		*,
		grounding_result: TemporalGroundingResult,
		method_library: HTNMethodLibrary,
		plan_library: PlanLibrary,
		agentspeak_code: str,
		agentspeak_artifacts: Dict[str, Any],
		verification_problem_file: str | Path,
		verification_mode: str,
		evaluation_domain: EvaluationDomainContext,
	) -> Optional[JasonExecutionResult]:
		return self._execute_query_with_jason(
			grounding_result=grounding_result,
			method_library=method_library,
			plan_library=plan_library,
			agentspeak_code=agentspeak_code,
			agentspeak_artifacts=agentspeak_artifacts,
			verification_problem_file=verification_problem_file,
			verification_mode=verification_mode,
			evaluation_domain=evaluation_domain,
		)

	def _execute_query_with_jason(
		self,
		*,
		grounding_result: TemporalGroundingResult,
		method_library: HTNMethodLibrary,
		plan_library: PlanLibrary,
		agentspeak_code: str,
		agentspeak_artifacts: Dict[str, Any],
		verification_problem_file: str | Path,
		verification_mode: str,
		evaluation_domain: EvaluationDomainContext,
	) -> Optional[JasonExecutionResult]:
		print("\n[RUNTIME EXECUTION]")
		print("-" * 80)
		stage_start = time.perf_counter()

		try:
			output_dir = self._require_output_dir()
			runner = JasonRunner(
				timeout_seconds=self._jason_runtime_timeout_seconds(
					subgoal_count=len(grounding_result.subgoals),
				),
			)
			action_schemas = planner_action_schemas_for_domain(evaluation_domain.domain)
			seed_facts = (
				tuple(render_problem_fact(fact) for fact in (self.problem.init_facts or ()))
				if self.problem is not None
				else ()
			)
			goal_facts = (
				tuple(render_problem_fact(fact) for fact in (self.problem.goal_facts or ()))
				if self.problem is not None
				else ()
			)
			runtime_objects = tuple(
				str(object_name).strip()
				for object_name in (
					self.problem.objects
					if self.problem is not None
					else grounding_result.typed_objects.keys()
				)
				if str(object_name).strip()
			)
			validation = runner.validate(
				agentspeak_code=agentspeak_code,
				method_library=method_library,
				plan_library=plan_library,
				action_schemas=action_schemas,
				seed_facts=seed_facts,
				runtime_objects=runtime_objects,
				object_types=dict(grounding_result.typed_objects),
				type_parent_map=dict(evaluation_domain.type_parent_map),
				query_goals=tuple(subgoal.to_dict() for subgoal in grounding_result.subgoals),
				goal_facts=goal_facts,
				domain_name=evaluation_domain.domain.name,
				problem_file=str(Path(verification_problem_file).resolve()),
				output_dir=output_dir,
			)
			if validation.status != "success":
				raise ValueError(validation.stderr or validation.stdout or "Jason runtime returned failure")

			goal_repair_pass_count = int(
				(dict(validation.artifacts).get("goal_repair_pass_count") or 0),
			)
			hierarchical_plan_text = None
			if goal_repair_pass_count <= 1:
				hierarchical_plan_text = render_supported_hierarchical_plan(
					action_path=validation.action_path,
					method_library=method_library,
					method_trace=validation.method_trace,
					problem_file=str(Path(verification_problem_file).resolve()),
					domain_file=evaluation_domain.domain_file,
				)
			result = JasonExecutionResult(
				query_text=grounding_result.query_text,
				ltlf_formula=grounding_result.ltlf_formula,
				action_path=tuple(validation.action_path),
				method_trace=tuple(dict(item) for item in validation.method_trace),
				hierarchical_plan_text=hierarchical_plan_text,
				verification_problem_file=str(Path(verification_problem_file).resolve()),
				verification_mode=verification_mode,
				failed_goals=tuple(validation.failed_goals),
				failure_class=validation.failure_class,
				consistency_checks=dict(validation.consistency_checks),
				artifacts={
					**agentspeak_artifacts,
					**dict(validation.artifacts),
					"runtime_backend": "jason",
					"stdout_path": dict(validation.artifacts).get("stdout"),
					"stderr_path": dict(validation.artifacts).get("stderr"),
				},
				timing_profile=dict(validation.timing_profile),
				diagnostics=tuple(str(item) for item in validation.failed_goals),
			)
			self.logger.log_runtime_execution(
				result.to_log_dict(),
				"Success",
				backend="RunLocalMAS",
				metadata={
					"step_count": len(result.action_path),
					"method_trace_count": len(result.method_trace),
					"verification_mode": verification_mode,
					"evaluation_domain_source": evaluation_domain.source,
					"runtime_execution_mode": "jason_runtime",
				},
			)
			self._record_failure_signature(
				ltlf_formula=grounding_result.ltlf_formula,
				jason_failure_class=result.failure_class,
				failed_goals=result.failed_goals,
			)
			self._record_step_timing(
				"runtime_execution",
				stage_start,
				breakdown=self._timing_breakdown_without_total(result.timing_profile),
				metadata={
					"step_count": len(result.action_path),
					"method_trace_count": len(result.method_trace),
					"verification_mode": verification_mode,
					"evaluation_domain_source": evaluation_domain.source,
					"runtime_execution_mode": "jason_runtime",
				},
			)
			print(f"✓ Jason action steps: {len(result.action_path)}")
			return result
		except JasonValidationError as exc:
			validation_metadata = dict(getattr(exc, "metadata", {}) or {})
			self.logger.log_runtime_execution(
				validation_metadata or None,
				"Failed",
				error=str(exc),
				backend="RunLocalMAS",
				metadata={
					"verification_mode": verification_mode,
					"evaluation_domain_source": evaluation_domain.source,
				},
			)
			self._record_failure_signature(
				ltlf_formula=grounding_result.ltlf_formula,
				jason_failure_class=validation_metadata.get("failure_class"),
				failed_goals=tuple(validation_metadata.get("failed_goals") or ()),
			)
			self._record_step_timing("runtime_execution", stage_start)
			print(f"✗ Runtime execution failed: {exc}")
			return None
		except Exception as exc:
			self.logger.log_runtime_execution(
				None,
				"Failed",
				error=str(exc),
				backend="RunLocalMAS",
				metadata={
					"verification_mode": verification_mode,
					"evaluation_domain_source": evaluation_domain.source,
				},
			)
			self._record_failure_signature(ltlf_formula=grounding_result.ltlf_formula)
			self._record_step_timing("runtime_execution", stage_start)
			print(f"✗ Runtime execution failed: {exc}")
			return None

	def _verify_plan_officially(
		self,
		*,
		method_library: HTNMethodLibrary,
		plan_solve_data: Dict[str, Any],
		ltlf_formula: str = "",
		evaluation_domain: EvaluationDomainContext,
	) -> Optional[Dict[str, Any]]:
		print("\n[OFFICIAL VERIFICATION]")
		print("-" * 80)
		stage_start = time.perf_counter()

		outcome = verify_jason_hierarchical_plan(
			method_library=method_library,
			plan_solve_data=plan_solve_data,
			evaluation_domain=evaluation_domain,
			problem_file=self.problem_file,
			output_dir=self._require_output_dir(),
		)
		if outcome.data is not None and (outcome.data.get("summary") or {}).get("status") == "skipped":
			self._record_failure_signature(ltlf_formula=ltlf_formula)
			self.logger.log_official_verification(
				outcome.data.get("artifacts"),
				"Skipped",
				metadata=outcome.data.get("summary"),
			)
			self._record_step_timing("plan_verification", stage_start)
			print("• Skipped: no problem file was provided")
			return outcome.data

		if not outcome.success:
			verifier_missing_goal_facts = tuple(
				str(fact).strip()
				for fact in (((outcome.data or {}).get("summary") or {}).get("missing_goal_facts") or ())
				if str(fact).strip()
			)
			self._record_failure_signature(
				ltlf_formula=ltlf_formula,
				verifier_missing_goal_facts=verifier_missing_goal_facts,
			)
			self.logger.log_official_verification(
				(outcome.data or {}).get("artifacts"),
				"Failed",
				error=outcome.error,
				metadata=(outcome.data or {}).get("summary")
				or {"backend": "pandaPIparser", "status": "failed"},
			)
			self._record_step_timing(
				"plan_verification",
				stage_start,
				breakdown=outcome.timing_breakdown,
				metadata={
					"plan_kind": ((outcome.data or {}).get("summary") or {}).get("plan_kind"),
					"verification_result": (
						((outcome.data or {}).get("summary") or {}).get("verification_result")
					),
				},
			)
			print(f"✗ Official verification failed: {outcome.error}")
			return None

		self.logger.log_official_verification(
			outcome.data.get("artifacts") if outcome.data else None,
			"Success",
			metadata=outcome.data.get("summary") if outcome.data else None,
		)
		self._record_failure_signature(
			ltlf_formula=ltlf_formula,
		)
		self._record_step_timing(
			"plan_verification",
			stage_start,
			breakdown=outcome.timing_breakdown,
			metadata={
				"plan_kind": ((outcome.data or {}).get("summary") or {}).get("plan_kind"),
				"verification_result": (
					((outcome.data or {}).get("summary") or {}).get("verification_result")
				),
			},
		)
		artifacts = (outcome.data or {}).get("artifacts") or {}
		summary = (outcome.data or {}).get("summary") or {}
		print("✓ Official IPC verification complete")
		print(f"  Plan kind: {summary.get('plan_kind')}")
		print(f"  Verification result: {summary.get('verification_result')}")
		print(f"  Verifier output: {artifacts.get('output_file')}")
		return outcome.data

	def _determine_verification_problem(self) -> Tuple[Optional[str], str]:
		if self.problem is None or not self.problem_file:
			return self.problem_file, "original_problem"
		return str(Path(self.problem_file).resolve()), "original_problem"

	def _build_plan_solve_data(
		self,
		*,
		grounding_result: TemporalGroundingResult,
		jason_result: JasonExecutionResult,
		evaluation_domain: EvaluationDomainContext,
		verification_mode: str,
	) -> Dict[str, Any]:
		runtime_backend = str(
			(jason_result.artifacts or {}).get("runtime_backend")
			or self.runtime_backend
			or "jason",
		).strip()
		planning_mode = "jason_runtime"
		return {
			"summary": {
				"backend": runtime_backend,
				"status": "success",
				"planning_mode": planning_mode,
				"verification_mode": verification_mode,
				"evaluation_domain_source": evaluation_domain.source,
				"evaluation_domain_file": evaluation_domain.domain_file,
				"task_count": len(grounding_result.subgoals),
				"step_count": len(jason_result.action_path),
				"failure_class": jason_result.failure_class,
			},
			"artifacts": {
				"backend": runtime_backend,
				"status": "success",
				"planning_mode": planning_mode,
				"verification_mode": verification_mode,
				"evaluation_domain_source": evaluation_domain.source,
				"evaluation_domain_file": evaluation_domain.domain_file,
				"ltlf_formula": grounding_result.ltlf_formula,
				"subgoals": [subgoal.to_dict() for subgoal in grounding_result.subgoals],
				"execution_goal_sequence": [
					{
						"task_name": subgoal.task_name,
						"args": list(subgoal.args),
					}
					for subgoal in grounding_result.subgoals
				],
				"action_path": list(jason_result.action_path),
				"method_trace": list(jason_result.method_trace),
				"hierarchical_plan_text": jason_result.hierarchical_plan_text,
				"hierarchical_plan_source": planning_mode,
				"verification_problem_file": jason_result.verification_problem_file,
				"verification_domain_file": None,
				"failed_goals": list(jason_result.failed_goals),
				"failure_class": jason_result.failure_class,
				"consistency_checks": dict(jason_result.consistency_checks),
				"timing_profile": dict(jason_result.timing_profile),
				"artifacts": dict(jason_result.artifacts),
			},
		}

	def _record_failure_signature(
		self,
		*,
		ltlf_formula: str | None = None,
		jason_failure_class: str | None = None,
		failed_goals: Tuple[str, ...] | list[str] = (),
		verifier_missing_goal_facts: Tuple[str, ...] | list[str] = (),
	) -> None:
		self.logger.record_failure_signature(
			build_failure_signature(
				ltlf_formula=ltlf_formula,
				jason_failure_class=jason_failure_class,
				failed_goals=tuple(failed_goals),
				verifier_missing_goal_facts=tuple(verifier_missing_goal_facts),
			),
		)

	@staticmethod
	def _preview_log_text(value: str, *, limit: int = 400) -> str:
		text = str(value or "")
		if len(text) <= limit:
			return text
		return f"{text[:limit]}... [truncated, chars={len(text)}]"

	def _record_step_timing(
		self,
		step_name: str,
		step_start: float,
		*,
		breakdown: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self.logger.record_step_timing(
			step_name,
			time.perf_counter() - step_start,
			breakdown=breakdown,
			metadata=metadata,
		)

	def _require_output_dir(self) -> Path:
		if self.output_dir is None:
			raise ValueError("Plan-library evaluation requires an active output directory.")
		return Path(self.output_dir).resolve()

	@staticmethod
	def _timing_breakdown_without_total(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
		if not profile:
			return {}
		return {
			str(key): value
			for key, value in profile.items()
			if key != "total_seconds" and value is not None
		}

	@staticmethod
	def _jason_runtime_timeout_seconds(*, subgoal_count: int) -> int:
		return 1800
