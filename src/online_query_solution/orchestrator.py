"""
Jason-based online query-solution orchestrator.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from offline_method_generation.method_synthesis.naming import sanitize_identifier
from offline_method_generation.method_synthesis.schema import HTNMethodLibrary
from online_query_solution.agentspeak import (
	AgentSpeakRenderer,
	build_agentspeak_transition_specs,
)
from online_query_solution.domain_selection import (
	OnlineDomainContext,
	normalize_online_domain_source,
)
from online_query_solution.goal_grounding.grounder import NLToLTLfGenerator
from online_query_solution.jason_runtime import JasonRunner
from online_query_solution.runtime_context import (
	action_type_map_for_domain,
	build_type_parent_map_for_domain,
	method_library_source_task_name_map,
	planner_action_schemas_for_domain,
	predicate_type_map_for_domain,
	render_problem_fact,
	resolve_online_domain_context,
	task_event_grounding_map,
	task_type_map_for_domain,
	typed_object_entries,
	validate_problem_domain_compatibility,
)
from online_query_solution.temporal_compilation import build_dfa_from_ltlf
from online_query_solution.official_verification import (
	render_supported_hierarchical_plan,
	verify_jason_hierarchical_plan,
)
from pipeline.artifacts import (
	DFACompilationResult,
	DomainLibraryArtifact,
	JasonExecutionResult,
	TemporalGroundingResult,
	load_domain_library_artifact,
)
from pipeline.execution_logger import ExecutionLogger
from utils.config import get_config
from utils.hddl_parser import HDDLParser


class OnlineQuerySolutionOrchestrator:
	"""Pure online runtime: natural language -> LTLf -> DFA -> AgentSpeak -> Jason."""

	def __init__(
		self,
		*,
		domain_file: str,
		problem_file: str | None = None,
		online_domain_source: str | None = None,
	) -> None:
		self.config = get_config()
		self.project_root = Path(__file__).resolve().parents[2]
		self.logger = ExecutionLogger(logs_dir=str(self.project_root / "artifacts" / "runs"))

		if not domain_file:
			raise ValueError("domain_file is required for online query execution.")

		self.domain_file = str(Path(domain_file).expanduser().resolve())
		self.problem_file = (
			str(Path(problem_file).expanduser().resolve())
			if problem_file is not None
			else None
		)
		self.online_domain_source = normalize_online_domain_source(
			online_domain_source or self.config.online_domain_source,
		)

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

	def run_query(self, nl_instruction: str) -> Dict[str, Any]:
		"""
		Execute one query against the cached domain library artifact.

		This online path does not synthesize methods. If the cache is absent, callers
		must run the offline build first or pass `library_artifact` explicitly.
		"""

		self._start_query_run(nl_instruction)
		try:
			artifact = self._load_default_domain_library_artifact()
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
	) -> Dict[str, Any]:
		"""Execute one query against a supplied cached or in-memory method library."""

		self._start_query_run(nl_query)
		result = self._execute_query_with_loaded_library(
			nl_query,
			load_domain_library_artifact(library_artifact),
		)
		log_filepath = self.logger.end_pipeline(success=result.get("success", False))
		result["log_path"] = str(log_filepath)
		return result

	def _start_query_run(self, nl_instruction: str) -> None:
		self.logger.start_pipeline(
			nl_instruction,
			mode="online_query_solution",
			domain_file=self.domain_file,
			problem_file=self.problem_file,
			domain_name=self.domain.name,
			problem_name=self.problem.name if self.problem is not None else None,
			output_dir="artifacts/runs",
		)
		self.output_dir = Path(self.logger.current_log_dir).resolve()
		if self.logger.current_record is not None:
			self.logger.current_record.output_dir = str(self.output_dir)
			self.logger._save_current_state()

	def _load_default_domain_library_artifact(self) -> DomainLibraryArtifact:
		artifact_root = (
			self.project_root
			/ "artifacts"
			/ "domain_builds"
			/ sanitize_identifier(self.domain.name)
		)
		if not (artifact_root / "method_library.json").exists():
			raise FileNotFoundError(
				"Cached method library artifact not found. "
				"Pass --library-artifact or run offline method generation first: "
				f"{artifact_root}",
			)
		return load_domain_library_artifact(artifact_root)

	def _execute_query_with_loaded_library(
		self,
		nl_instruction: str,
		artifact: DomainLibraryArtifact,
	) -> Dict[str, Any]:
		"""Run the online Jason query path with a prebuilt method library."""

		domain_library = artifact.method_library
		online_domain = self._resolve_online_domain_context(artifact)
		if self.logger.current_record is not None:
			self.logger.current_record.domain_file = online_domain.domain_file
			self.logger.current_record.domain_name = online_domain.domain.name
			self.logger._save_current_state()

		grounding_result, grounding_prompt, grounding_response = self._ground_query_temporally(
			nl_instruction,
			domain_library,
			online_domain=online_domain,
		)
		if grounding_result is None:
			return {
				"success": False,
				"step": "goal_grounding",
				"error": "Temporal goal grounding failed",
				"method_library": domain_library.to_dict(),
			}

		dfa_result = self._compile_temporal_goal(grounding_result)
		if dfa_result is None:
			return {
				"success": False,
				"step": "temporal_compilation",
				"error": "LTLf to DFA compilation failed",
				"method_library": domain_library.to_dict(),
				"goal_grounding": grounding_result.to_dict(),
			}

		verification_problem_file, verification_mode = self._determine_verification_problem()
		if verification_problem_file is None:
			return {
				"success": False,
				"step": "plan_verification",
				"error": "Could not determine a verification problem file",
				"method_library": domain_library.to_dict(),
				"goal_grounding": grounding_result.to_dict(),
				"temporal_compilation": dfa_result.to_dict(),
			}

		agentspeak_render = self._render_agentspeak_program(
			grounding_result=grounding_result,
			dfa_result=dfa_result,
			method_library=domain_library,
			online_domain=online_domain,
		)
		if agentspeak_render is None:
			return {
				"success": False,
				"step": "agentspeak_rendering",
				"error": "AgentSpeak rendering failed",
				"method_library": domain_library.to_dict(),
				"goal_grounding": grounding_result.to_dict(),
				"temporal_compilation": dfa_result.to_dict(),
			}

		jason_result = self._execute_query_with_jason(
			grounding_result=grounding_result,
			dfa_result=dfa_result,
			method_library=domain_library,
			agentspeak_code=agentspeak_render["agentspeak_code"],
			agentspeak_artifacts=agentspeak_render["artifacts"],
			verification_problem_file=verification_problem_file,
			verification_mode=verification_mode,
			online_domain=online_domain,
		)
		if jason_result is None:
			return {
				"success": False,
				"step": "runtime_execution",
				"error": "Jason runtime execution failed",
				"method_library": domain_library.to_dict(),
				"goal_grounding": grounding_result.to_dict(),
				"temporal_compilation": dfa_result.to_dict(),
			}

		plan_solve_data = self._build_plan_solve_data(
			grounding_result=grounding_result,
			dfa_result=dfa_result,
			jason_result=jason_result,
			online_domain=online_domain,
			verification_mode=verification_mode,
		)
		plan_verification_data = self._verify_plan_officially(
			method_library=domain_library,
			plan_solve_data=plan_solve_data,
			online_domain=online_domain,
		)
		if plan_verification_data is None:
			return {
				"success": False,
				"step": "plan_verification",
				"error": "Official IPC verification failed",
				"method_library": domain_library.to_dict(),
				"goal_grounding": grounding_result.to_dict(),
				"temporal_compilation": dfa_result.to_dict(),
				"plan_solve": plan_solve_data,
			}

		return {
			"success": True,
			"method_library": domain_library.to_dict(),
			"goal_grounding": grounding_result.to_dict(),
			"temporal_compilation": dfa_result.to_dict(),
			"plan_solve": plan_solve_data,
			"plan_verification": plan_verification_data,
			"llm_prompt": grounding_prompt,
			"llm_response": grounding_response,
		}

	def _resolve_online_domain_context(
		self,
		artifact: Optional[DomainLibraryArtifact] = None,
		*,
		source: Optional[str] = None,
	) -> OnlineDomainContext:
		return resolve_online_domain_context(
			source_domain_file=self.domain_file,
			source_domain=self.domain,
			artifact=artifact,
			online_domain_source=source or self.online_domain_source,
			output_dir=self.output_dir,
			project_root=self.project_root,
		)

	def _ground_query_temporally(
		self,
		nl_instruction: str,
		method_library: HTNMethodLibrary,
		*,
		online_domain: OnlineDomainContext,
	) -> Tuple[Optional[TemporalGroundingResult], Optional[Dict[str, str]], Optional[str]]:
		print("\n[GOAL GROUNDING]")
		print("-" * 80)
		stage_start = time.perf_counter()
		generator: Optional[NLToLTLfGenerator] = None

		try:
			generator = NLToLTLfGenerator(
				api_key=self.config.openai_api_key,
				model=self.config.goal_grounding_model,
				base_url=self.config.openai_base_url,
				domain_file=online_domain.domain_file,
				request_timeout=float(self.config.openai_timeout),
				response_max_tokens=int(self.config.goal_grounding_max_tokens),
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
				task_type_map=online_domain.task_type_map,
				type_parent_map=online_domain.type_parent_map,
			)
			self.logger.log_goal_grounding_success(
				grounding_result.to_dict(),
				used_llm=True,
				model=self.config.goal_grounding_model,
				llm_prompt=llm_prompt,
				llm_response=llm_response,
			)
			self._record_step_timing(
				"goal_grounding",
				stage_start,
				metadata={
					"task_event_count": len(grounding_result.subgoals),
					"online_domain_source": online_domain.source,
				},
			)
			print(f"✓ Grounded task events: {len(grounding_result.subgoals)}")
			print(f"  LTLf: {grounding_result.ltlf_formula}")
			return grounding_result, llm_prompt, llm_response
		except Exception as exc:
			generation_metadata = dict(getattr(generator, "last_generation_metadata", {}) or {})
			llm_prompt = generation_metadata.get("last_prompt")
			self.logger.log_goal_grounding_error(
				str(exc),
				model=self.config.goal_grounding_model,
				llm_prompt=llm_prompt if isinstance(llm_prompt, dict) else None,
				llm_response=str(generation_metadata.get("last_response") or "") or None,
				metadata={
					"online_domain_source": online_domain.source,
					"attempt_mode": generation_metadata.get("attempt_mode"),
					"attempt_count": generation_metadata.get("attempt_count"),
					"attempt_errors": generation_metadata.get("attempt_errors"),
				},
			)
			self._record_step_timing("goal_grounding", stage_start)
			print(f"✗ Goal grounding failed: {exc}")
			return None, None, None

	def _compile_temporal_goal(
		self,
		grounding_result: TemporalGroundingResult,
	) -> Optional[DFACompilationResult]:
		print("\n[TEMPORAL COMPILATION]")
		print("-" * 80)
		stage_start = time.perf_counter()

		try:
			output_dir = self._require_output_dir()
			dfa_payload = build_dfa_from_ltlf(grounding_result)
			dfa_dot = str(dfa_payload.get("dfa_dot") or "")
			dfa_dot_path = output_dir / "query_dfa.dot"
			dfa_dot_path.write_text(dfa_dot)
			grounding_map = task_event_grounding_map(grounding_result)
			transition_specs = build_agentspeak_transition_specs(
				dfa_result=dfa_payload,
				grounding_map=grounding_map,
				ordered_query_sequence=False,
			)
			result = DFACompilationResult(
				query_text=grounding_result.query_text,
				ltlf_formula=grounding_result.ltlf_formula,
				alphabet=tuple(
					str(item).strip()
					for item in (dfa_payload.get("alphabet") or ())
					if str(item).strip()
				),
				transition_specs=tuple(dict(spec) for spec in transition_specs),
				dfa_dot=dfa_dot,
				construction=str(dfa_payload.get("construction") or "generic_ltlf2dfa"),
				num_states=int(dfa_payload.get("num_states") or 0) or None,
				num_transitions=int(dfa_payload.get("num_transitions") or 0) or None,
				ordered_subgoal_sequence=False,
				subgoals=tuple(grounding_result.subgoals),
				timing_profile=dict(dfa_payload.get("timing_profile") or {}),
			)
			self.logger.log_temporal_compilation(
				{
					**result.to_dict(),
					"dfa_dot_path": str(dfa_dot_path),
				},
				"Success",
				metadata={
					"num_states": result.num_states,
					"num_transitions": result.num_transitions,
					"task_event_count": len(grounding_result.subgoals),
				},
			)
			self._record_step_timing(
				"temporal_compilation",
				stage_start,
				breakdown=self._timing_breakdown_without_total(result.timing_profile),
				metadata={
					"num_states": result.num_states,
					"num_transitions": result.num_transitions,
					"task_event_count": len(grounding_result.subgoals),
				},
			)
			print(f"✓ DFA states: {result.num_states or 0}")
			print(f"  Task-event alphabet size: {len(result.alphabet)}")
			return result
		except Exception as exc:
			self.logger.log_temporal_compilation(None, "Failed", error=str(exc))
			self._record_step_timing("temporal_compilation", stage_start)
			print(f"✗ Temporal compilation failed: {exc}")
			return None

	def _render_agentspeak_program(
		self,
		*,
		grounding_result: TemporalGroundingResult,
		dfa_result: DFACompilationResult,
		method_library: HTNMethodLibrary,
		online_domain: OnlineDomainContext,
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
				domain=online_domain.domain,
				objects=runtime_objects,
				method_library=method_library,
				plan_records=(),
				typed_objects=typed_objects,
				ordered_query_sequence=False,
				prompt_analysis=None,
				transition_specs=dfa_result.transition_specs,
				subgoals=[subgoal.to_dict() for subgoal in grounding_result.subgoals],
				subgoal_task_name_map=method_library_source_task_name_map(method_library),
			)
			asl_path = output_dir / "query_runtime.asl"
			asl_path.write_text(agentspeak_code)
			artifacts = {
				"asl_file": str(asl_path),
				"ordered_subgoal_sequence": dfa_result.ordered_subgoal_sequence,
				"transition_spec_count": len(dfa_result.transition_specs),
			}
			self.logger.log_agentspeak_rendering(
				artifacts,
				"Success",
				metadata={"transition_spec_count": len(dfa_result.transition_specs)},
			)
			self._record_step_timing(
				"agentspeak_rendering",
				stage_start,
				metadata={
					"transition_spec_count": len(dfa_result.transition_specs),
					"online_domain_source": online_domain.source,
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

	def _execute_query_with_jason(
		self,
		*,
		grounding_result: TemporalGroundingResult,
		dfa_result: DFACompilationResult,
		method_library: HTNMethodLibrary,
		agentspeak_code: str,
		agentspeak_artifacts: Dict[str, Any],
		verification_problem_file: str | Path,
		verification_mode: str,
		online_domain: OnlineDomainContext,
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
			action_schemas = planner_action_schemas_for_domain(online_domain.domain)
			seed_facts = (
				tuple(render_problem_fact(fact) for fact in (self.problem.init_facts or ()))
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
				action_schemas=action_schemas,
				seed_facts=seed_facts,
				runtime_objects=runtime_objects,
				object_types=dict(grounding_result.typed_objects),
				type_parent_map=dict(online_domain.type_parent_map),
				domain_name=online_domain.domain.name,
				problem_file=str(Path(verification_problem_file).resolve()),
				output_dir=output_dir,
			)
			if validation.status != "success":
				raise ValueError(validation.stderr or validation.stdout or "Jason runtime returned failure")

			hierarchical_plan_text = render_supported_hierarchical_plan(
				action_path=validation.action_path,
				method_library=method_library,
				method_trace=validation.method_trace,
				problem_file=str(Path(verification_problem_file).resolve()),
				domain_file=online_domain.domain_file,
			)
			result = JasonExecutionResult(
				query_text=grounding_result.query_text,
				ltlf_formula=grounding_result.ltlf_formula,
				action_path=tuple(validation.action_path),
				method_trace=tuple(dict(item) for item in validation.method_trace),
				hierarchical_plan_text=hierarchical_plan_text,
				verification_problem_file=str(Path(verification_problem_file).resolve()),
				verification_mode=verification_mode,
				artifacts={
					**agentspeak_artifacts,
					**dict(validation.artifacts),
					"stdout_path": dict(validation.artifacts).get("stdout"),
					"stderr_path": dict(validation.artifacts).get("stderr"),
				},
				timing_profile=dict(validation.timing_profile),
				diagnostics=tuple(str(item) for item in validation.failed_goals),
			)
			self.logger.log_runtime_execution(
				result.to_dict(),
				"Success",
				backend="RunLocalMAS",
				metadata={
					"step_count": len(result.action_path),
					"method_trace_count": len(result.method_trace),
					"verification_mode": verification_mode,
					"online_domain_source": online_domain.source,
					"runtime_execution_mode": "jason_runtime",
				},
			)
			self._record_step_timing(
				"runtime_execution",
				stage_start,
				breakdown=self._timing_breakdown_without_total(result.timing_profile),
				metadata={
					"step_count": len(result.action_path),
					"method_trace_count": len(result.method_trace),
					"verification_mode": verification_mode,
					"online_domain_source": online_domain.source,
					"runtime_execution_mode": "jason_runtime",
				},
			)
			print(f"✓ Jason action steps: {len(result.action_path)}")
			return result
		except Exception as exc:
			self.logger.log_runtime_execution(
				None,
				"Failed",
				error=str(exc),
				backend="RunLocalMAS",
				metadata={
					"verification_mode": verification_mode,
					"online_domain_source": online_domain.source,
				},
			)
			self._record_step_timing("runtime_execution", stage_start)
			print(f"✗ Runtime execution failed: {exc}")
			return None

	def _verify_plan_officially(
		self,
		*,
		method_library: HTNMethodLibrary,
		plan_solve_data: Dict[str, Any],
		online_domain: OnlineDomainContext,
	) -> Optional[Dict[str, Any]]:
		print("\n[OFFICIAL VERIFICATION]")
		print("-" * 80)
		stage_start = time.perf_counter()

		outcome = verify_jason_hierarchical_plan(
			method_library=method_library,
			plan_solve_data=plan_solve_data,
			online_domain=online_domain,
			problem_file=self.problem_file,
			output_dir=self._require_output_dir(),
		)
		if outcome.data is not None and (outcome.data.get("summary") or {}).get("status") == "skipped":
			self.logger.log_official_verification(
				outcome.data.get("artifacts"),
				"Skipped",
				metadata=outcome.data.get("summary"),
			)
			self._record_step_timing("plan_verification", stage_start)
			print("• Skipped: no problem file was provided")
			return outcome.data

		if not outcome.success:
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
		dfa_result: DFACompilationResult,
		jason_result: JasonExecutionResult,
		online_domain: OnlineDomainContext,
		verification_mode: str,
	) -> Dict[str, Any]:
		return {
			"summary": {
				"backend": "jason",
				"status": "success",
				"planning_mode": "jason_runtime",
				"verification_mode": verification_mode,
				"online_domain_source": online_domain.source,
				"online_domain_file": online_domain.domain_file,
				"task_count": len(grounding_result.subgoals),
				"step_count": len(jason_result.action_path),
			},
			"artifacts": {
				"backend": "jason",
				"status": "success",
				"planning_mode": "jason_runtime",
				"verification_mode": verification_mode,
				"online_domain_source": online_domain.source,
				"online_domain_file": online_domain.domain_file,
				"ltlf_formula": grounding_result.ltlf_formula,
				"subgoals": [subgoal.to_dict() for subgoal in grounding_result.subgoals],
				"ordered_subgoal_sequence": dfa_result.ordered_subgoal_sequence,
				"action_path": list(jason_result.action_path),
				"method_trace": list(jason_result.method_trace),
				"hierarchical_plan_text": jason_result.hierarchical_plan_text,
				"hierarchical_plan_source": "jason_runtime",
				"verification_problem_file": jason_result.verification_problem_file,
				"verification_domain_file": None,
				"timing_profile": dict(jason_result.timing_profile),
				"artifacts": dict(jason_result.artifacts),
			},
		}

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
			raise ValueError("Online query execution requires an active output directory.")
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
		if subgoal_count >= 1000:
			return 480
		if subgoal_count >= 800:
			return 360
		if subgoal_count >= 600:
			return 240
		if subgoal_count >= 400:
			return 180
		return 120
