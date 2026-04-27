"""
Incremental Jason-runtime evaluation and plan-library construction.

This module keeps the generated HTN method library M as the editable source of
truth and materialises the AgentSpeak(L) library S from M after every accepted
patch. The existing one-shot generation and evaluation paths are intentionally
left untouched.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

from domain_model import infer_query_domain, load_query_sequence_records
from evaluation.pipeline import PlanLibraryEvaluationPipeline
from method_library.synthesis.naming import sanitize_identifier
from method_library.synthesis.schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
)
from method_library.synthesis.synthesizer import HTNMethodSynthesizer
from plan_library.artifacts import (
	PlanLibraryArtifactBundle,
	load_plan_library_artifact_bundle,
	persist_plan_library_artifact_bundle,
)
from plan_library.models import (
	TranslationCoverage,
)
from plan_library.rendering import render_plan_library_asl
from plan_library.set_semantics import deduplicate_plan_library
from plan_library.translation import build_plan_library
from plan_library.validation import build_library_validation_record
from temporal_specification import (
	TemporalSpecificationRecord,
	extract_formula_atoms_in_order,
	parse_task_event_predicate_name,
)
from utils.config import Config, get_config
from utils.hddl_condition_parser import HDDLConditionParser
from utils.hddl_parser import HDDLParser


@dataclass(frozen=True)
class MethodLibraryMergeResult:
	"""Result of merging an incremental HTN method patch into M."""

	method_library: HTNMethodLibrary
	added_methods: int
	duplicate_methods: int
	renamed_methods: int
	merged_source_instruction_ids: int

	def to_dict(self) -> Dict[str, Any]:
		return {
			"added_methods": self.added_methods,
			"duplicate_methods": self.duplicate_methods,
			"renamed_methods": self.renamed_methods,
			"merged_source_instruction_ids": self.merged_source_instruction_ids,
			"method_count": len(self.method_library.methods),
		}


@dataclass(frozen=True)
class MethodPatchResult:
	"""Patch returned by an API or manual provider."""

	method_library: HTNMethodLibrary
	metadata: Dict[str, Any]


class MethodPatchProvider(Protocol):
	"""Provider interface for incremental method patches."""

	def request_patch(
		self,
		*,
		domain: Any,
		current_library: HTNMethodLibrary,
		temporal_specification: TemporalSpecificationRecord,
		evaluation_result: Dict[str, Any],
		output_dir: Path,
	) -> Optional[MethodPatchResult]:
		"""Return a patch for one failed query, or None when no patch is available."""


class NoOpMethodPatchProvider:
	"""Evaluation-only provider that never mutates M."""

	def request_patch(
		self,
		*,
		domain: Any,
		current_library: HTNMethodLibrary,
		temporal_specification: TemporalSpecificationRecord,
		evaluation_result: Dict[str, Any],
		output_dir: Path,
	) -> Optional[MethodPatchResult]:
		_ = (domain, current_library, temporal_specification, evaluation_result, output_dir)
		return None


class ManualMethodPatchProvider:
	"""
	Write a compact patch prompt and apply a matching response file if present.

	This supports offline review without changing the automatic API path: the
	prompt is written to ``<manual_dir>/<query_id>.prompt.txt`` and the response
	is read from ``<manual_dir>/<query_id>.response.json`` or ``.txt``.
	"""

	def __init__(
		self,
		*,
		manual_dir: str | Path,
		config: Config | None = None,
	) -> None:
		self.manual_dir = Path(manual_dir).expanduser().resolve()
		self.config = config or get_config()

	def request_patch(
		self,
		*,
		domain: Any,
		current_library: HTNMethodLibrary,
		temporal_specification: TemporalSpecificationRecord,
		evaluation_result: Dict[str, Any],
		output_dir: Path,
	) -> Optional[MethodPatchResult]:
		_ = output_dir
		self.manual_dir.mkdir(parents=True, exist_ok=True)
		query_id = _safe_query_id(temporal_specification.instruction_id)
		prompt = build_incremental_patch_prompt(
			domain=domain,
			current_library=current_library,
			temporal_specification=temporal_specification,
			evaluation_result=evaluation_result,
		)
		prompt_path = self.manual_dir / f"{query_id}.prompt.txt"
		prompt_path.write_text(
			f"SYSTEM:\n{prompt['system']}\n\nUSER:\n{prompt['user']}",
			encoding="utf-8",
		)
		response_path = _first_existing_path(
			self.manual_dir / f"{query_id}.response.json",
			self.manual_dir / f"{query_id}.response.txt",
		)
		if response_path is None:
			return None
		response_text = response_path.read_text(encoding="utf-8")
		patch_library = parse_method_patch_response(
			response_text=response_text,
			domain=domain,
			current_library=current_library,
			temporal_specification=temporal_specification,
			config=self.config,
		)
		return MethodPatchResult(
			method_library=patch_library,
			metadata={
				"provider": "manual",
				"prompt_path": str(prompt_path),
				"response_path": str(response_path),
			},
		)


class OpenAIMethodPatchProvider:
	"""OpenAI-compatible provider for incremental method patches."""

	def __init__(self, *, config: Config | None = None) -> None:
		self.config = config or get_config()
		self.synthesizer = HTNMethodSynthesizer(
			api_key=self.config.method_synthesis_api_key,
			model=self.config.method_synthesis_model,
			base_url=self.config.method_synthesis_base_url,
			timeout=float(self.config.method_synthesis_timeout),
			max_tokens=int(self.config.method_synthesis_max_tokens),
			session_id=self.config.method_synthesis_session_id,
		)

	def request_patch(
		self,
		*,
		domain: Any,
		current_library: HTNMethodLibrary,
		temporal_specification: TemporalSpecificationRecord,
		evaluation_result: Dict[str, Any],
		output_dir: Path,
	) -> Optional[MethodPatchResult]:
		if self.synthesizer.client is None:
			raise ValueError("METHOD_SYNTHESIS_API_KEY is required for API patch generation.")
		output_dir.mkdir(parents=True, exist_ok=True)
		prompt = build_incremental_patch_prompt(
			domain=domain,
			current_library=current_library,
			temporal_specification=temporal_specification,
			evaluation_result=evaluation_result,
		)
		prompt_path = output_dir / "patch_prompt.json"
		prompt_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")
		request_start = time.perf_counter()
		response_text, finish_reason, transport_metadata = self.synthesizer._call_llm(
			prompt,
			max_tokens=int(self.config.method_synthesis_max_tokens),
		)
		response_path = output_dir / "patch_response.json"
		response_path.write_text(response_text, encoding="utf-8")
		patch_library = parse_method_patch_response(
			response_text=response_text,
			domain=domain,
			current_library=current_library,
			temporal_specification=temporal_specification,
			config=self.config,
		)
		return MethodPatchResult(
			method_library=patch_library,
			metadata={
				"provider": "openai_compatible_api",
				"model": self.config.method_synthesis_model,
				"base_url": self.config.method_synthesis_base_url,
				"session_id": self.config.method_synthesis_session_id,
				"finish_reason": finish_reason,
				"seconds": round(time.perf_counter() - request_start, 6),
				"prompt_path": str(prompt_path),
				"response_path": str(response_path),
				"transport": dict(transport_metadata),
			},
		)


def empty_method_library_for_domain(domain: Any) -> HTNMethodLibrary:
	"""Create an empty incremental M scaffold over the declared domain vocabulary."""

	parser = HDDLConditionParser()
	compound_tasks = [
		HTNTask(
			name=sanitize_identifier(str(getattr(task, "name", "") or "").strip()),
			parameters=tuple(
				_parameter_name(parameter)
				for parameter in (getattr(task, "parameters", ()) or ())
			),
			is_primitive=False,
			source_predicates=tuple(
				str(predicate_name).strip()
				for predicate_name in (getattr(task, "source_predicates", ()) or ())
				if str(predicate_name).strip()
			),
			source_name=str(getattr(task, "name", "") or "").strip() or None,
		)
		for task in getattr(domain, "tasks", ()) or ()
		if str(getattr(task, "name", "") or "").strip()
	]
	primitive_tasks = [
		HTNTask(
			name=sanitize_identifier(str(action.name)),
			parameters=tuple(
				f"X{index + 1}"
				for index, _parameter in enumerate(parser.parse_action(action).parameters)
			),
			is_primitive=True,
			source_predicates=tuple(
				sorted(
					{
						literal.predicate
						for literal in parser.parse_action(action).positive_effects
					}
				),
			),
			source_name=str(action.name),
		)
		for action in getattr(domain, "actions", ()) or ()
	]
	return HTNMethodLibrary(
		compound_tasks=compound_tasks,
		primitive_tasks=primitive_tasks,
		methods=[],
		target_literals=[],
		target_task_bindings=[],
	)


def merge_method_libraries(
	existing_library: HTNMethodLibrary,
	patch_library: HTNMethodLibrary,
) -> MethodLibraryMergeResult:
	"""Merge one patch into M while enforcing method-set semantics."""

	compound_tasks = _merge_tasks(existing_library.compound_tasks, patch_library.compound_tasks)
	primitive_tasks = _merge_tasks(existing_library.primitive_tasks, patch_library.primitive_tasks)
	methods: list[HTNMethod] = []
	method_index_by_fingerprint: Dict[str, int] = {}
	used_method_names: set[str] = set()
	added_methods = 0
	duplicate_methods = 0
	renamed_methods = 0
	merged_source_instruction_ids = 0

	tagged_methods = [
		*(("existing", method) for method in existing_library.methods),
		*(("patch", method) for method in patch_library.methods),
	]
	for source, method in tagged_methods:
		fingerprint = method_fingerprint(method)
		existing_index = method_index_by_fingerprint.get(fingerprint)
		if existing_index is not None:
			duplicate_methods += 1
			merged_method = _merge_method_source_instruction_ids(
				methods[existing_index],
				method,
			)
			if merged_method.source_instruction_ids != methods[existing_index].source_instruction_ids:
				merged_source_instruction_ids += 1
			methods[existing_index] = merged_method
			continue

		method_name = str(method.method_name or "").strip() or f"method_{len(methods) + 1}"
		unique_method_name = _unique_name(method_name, used_method_names)
		if unique_method_name != method_name:
			renamed_methods += 1
			method = replace(method, method_name=unique_method_name)
		used_method_names.add(unique_method_name)
		method_index_by_fingerprint[fingerprint] = len(methods)
		methods.append(method)
		if source == "patch":
			added_methods += 1

	return MethodLibraryMergeResult(
		method_library=HTNMethodLibrary(
			compound_tasks=compound_tasks,
			primitive_tasks=primitive_tasks,
			methods=methods,
			target_literals=list(existing_library.target_literals),
			target_task_bindings=list(existing_library.target_task_bindings),
		),
		added_methods=added_methods,
		duplicate_methods=duplicate_methods,
		renamed_methods=renamed_methods,
		merged_source_instruction_ids=merged_source_instruction_ids,
	)


def materialize_incremental_bundle(
	*,
	domain: Any,
	method_library: HTNMethodLibrary,
	query_sequence: Sequence[Any],
	temporal_specifications: Sequence[TemporalSpecificationRecord],
	artifact_root: str | Path,
	method_synthesis_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Translate M into set-normalised S and persist a plan-library bundle."""

	plan_library, translation_coverage = build_plan_library(
		domain=domain,
		method_library=method_library,
	)
	set_result = deduplicate_plan_library(plan_library)
	if set_result.removed_duplicate_plans:
		translation_coverage = TranslationCoverage(
			domain_name=translation_coverage.domain_name,
			methods_considered=translation_coverage.methods_considered,
			plans_generated=len(set_result.plan_library.plans),
			accepted_translation=translation_coverage.accepted_translation,
			unsupported_buckets=dict(translation_coverage.unsupported_buckets),
			unsupported_methods=tuple(translation_coverage.unsupported_methods),
		)
	library_validation = build_library_validation_record(
		domain_name=str(getattr(domain, "name", "") or ""),
		domain=domain,
		method_library=method_library,
		plan_library=set_result.plan_library,
		translation_coverage=translation_coverage,
		method_validation=None,
	)
	root = Path(artifact_root).expanduser().resolve()
	metadata = dict(method_synthesis_metadata or {})
	metadata["incremental_library"] = True
	metadata["plan_set_normalisation"] = set_result.to_dict()
	bundle = PlanLibraryArtifactBundle(
		domain_name=str(getattr(domain, "name", "") or ""),
		query_sequence=tuple(query_sequence),
		temporal_specifications=tuple(temporal_specifications),
		method_library=method_library,
		plan_library=set_result.plan_library,
		translation_coverage=translation_coverage,
		library_validation=library_validation,
		method_synthesis_metadata=metadata,
		artifact_root=str(root),
		masked_domain_file=None,
		plan_library_asl_file=str(root / "plan_library.asl"),
	)
	artifact_paths = persist_plan_library_artifact_bundle(
		artifact_root=root,
		artifact=bundle,
		masked_domain_text=None,
		plan_library_asl_text=render_plan_library_asl(set_result.plan_library),
	)
	return {
		"bundle": bundle,
		"artifact_paths": artifact_paths,
		"set_normalisation": set_result.to_dict(),
		"library_validation": library_validation.to_dict(),
		"translation_coverage": translation_coverage.to_dict(),
	}


def run_incremental_jason_library_evaluation(
	*,
	domain_file: str | Path,
	output_root: str | Path,
	query_dataset: str | Path | None = None,
	query_domain: str | None = None,
	query_ids: Sequence[str] | None = None,
	seed_artifact: str | Path | None = None,
	patch_provider: Optional[MethodPatchProvider] = None,
	max_patch_attempts: int = 1,
	resume: bool = False,
) -> Dict[str, Any]:
	"""
	Incrementally construct M/S while using Jason runtime evaluation as the gate.

	The function never calls the one-shot generation path. If no seed artifact is
	provided, it starts from an empty domain-vocabulary scaffold and only grows M
	after failed Jason/verifier checks produce query-specific counterexamples.
	"""

	resolved_domain_file = Path(domain_file).expanduser().resolve()
	domain = HDDLParser.parse_domain(str(resolved_domain_file))
	query_sequence, temporal_specifications = load_query_sequence_records(
		domain_file=resolved_domain_file,
		dataset_path=query_dataset,
		query_domain=query_domain,
		query_ids=query_ids,
	)
	domain_key = infer_query_domain(
		domain_file=resolved_domain_file,
		explicit_domain=query_domain,
	)
	root = Path(output_root).expanduser().resolve()
	root.mkdir(parents=True, exist_ok=True)
	current_artifact_root = root / "current_library"
	coverage_path = root / "coverage_matrix.json"
	patch_history_path = root / "patch_history.json"
	summary_path = root / "incremental_summary.json"
	provider = patch_provider or NoOpMethodPatchProvider()
	patch_history: list[Dict[str, Any]] = _load_json_list(patch_history_path) if resume else []
	coverage_rows: list[Dict[str, Any]] = _load_json_list(coverage_path) if resume else []
	if not patch_history_path.exists():
		patch_history_path.write_text("[]", encoding="utf-8")
	if not coverage_path.exists():
		coverage_path.write_text("[]", encoding="utf-8")
	completed_successes = {
		str(row.get("query_id") or "")
		for row in coverage_rows
		if bool(row.get("success")) and str(row.get("query_id") or "").strip()
	}

	if seed_artifact is not None:
		current_bundle = load_plan_library_artifact_bundle(seed_artifact)
		current_library = current_bundle.method_library
	elif _is_complete_plan_library_artifact(current_artifact_root):
		current_bundle = load_plan_library_artifact_bundle(current_artifact_root)
		current_library = current_bundle.method_library
	else:
		current_library = empty_method_library_for_domain(domain)

	materialize_incremental_bundle(
		domain=domain,
		method_library=current_library,
		query_sequence=query_sequence,
		temporal_specifications=temporal_specifications,
		artifact_root=current_artifact_root,
		method_synthesis_metadata={"construction_mode": "incremental_jason_runtime"},
	)
	pipeline = PlanLibraryEvaluationPipeline(domain_file=str(resolved_domain_file))
	query_records_by_id = {
		str(record.instruction_id): record
		for record in temporal_specifications
	}

	for temporal_specification in temporal_specifications:
		query_id = str(temporal_specification.instruction_id)
		if resume and query_id in completed_successes:
			continue
		query_attempts: list[Dict[str, Any]] = []
		final_result: Dict[str, Any] = {}
		for attempt_index in range(max(int(max_patch_attempts), 0) + 1):
			evaluation_result = pipeline.evaluate_benchmark_case(
				library_artifact=current_artifact_root,
				query_id=query_id,
				query_dataset=str(query_dataset) if query_dataset is not None else None,
				query_domain=domain_key,
			)
			attempt_record = _compact_evaluation_attempt(
				attempt_index=attempt_index,
				evaluation_result=evaluation_result,
			)
			query_attempts.append(attempt_record)
			final_result = evaluation_result
			if bool(evaluation_result.get("success")):
				break
			if attempt_index >= max(int(max_patch_attempts), 0):
				break
			patch_dir = root / "patches" / _safe_query_id(query_id) / f"attempt_{attempt_index + 1}"
			patch_result = provider.request_patch(
				domain=domain,
				current_library=current_library,
				temporal_specification=temporal_specification,
				evaluation_result=evaluation_result,
				output_dir=patch_dir,
			)
			if patch_result is None:
				attempt_record["patch_status"] = "not_available"
				break
			merge_result = merge_method_libraries(
				current_library,
				patch_result.method_library,
			)
			current_library = merge_result.method_library
			materialize_result = materialize_incremental_bundle(
				domain=domain,
				method_library=current_library,
				query_sequence=query_sequence,
				temporal_specifications=temporal_specifications,
				artifact_root=current_artifact_root,
				method_synthesis_metadata={
					"construction_mode": "incremental_jason_runtime",
					"last_patch_query_id": query_id,
				},
			)
			patch_history_entry = {
				"query_id": query_id,
				"attempt": attempt_index + 1,
				"merge": merge_result.to_dict(),
				"provider": dict(patch_result.metadata),
				"set_normalisation": materialize_result["set_normalisation"],
			}
			patch_history.append(patch_history_entry)
			patch_history_path.write_text(json.dumps(patch_history, indent=2), encoding="utf-8")

		coverage_rows = [
			row
			for row in coverage_rows
			if str(row.get("query_id") or "") != query_id
		]
		coverage_rows.append(
			{
				"query_id": query_id,
				"problem_file": str(query_records_by_id[query_id].problem_file or ""),
				"success": bool(final_result.get("success")),
				"final_step": str(final_result.get("step") or ""),
				"attempts": query_attempts,
				"covered_by_existing_library": (
					bool(query_attempts)
					and bool(query_attempts[0].get("success"))
				),
			},
		)
		coverage_path.write_text(json.dumps(coverage_rows, indent=2), encoding="utf-8")

	final_materialization = materialize_incremental_bundle(
		domain=domain,
		method_library=current_library,
		query_sequence=query_sequence,
		temporal_specifications=temporal_specifications,
		artifact_root=current_artifact_root,
		method_synthesis_metadata={"construction_mode": "incremental_jason_runtime"},
	)
	success_count = sum(1 for row in coverage_rows if bool(row.get("success")))
	summary = {
		"success": success_count == len(temporal_specifications),
		"domain_key": domain_key,
		"domain_file": str(resolved_domain_file),
		"query_count": len(temporal_specifications),
		"covered_query_count": success_count,
		"remaining_query_count": len(temporal_specifications) - success_count,
		"artifact_root": str(current_artifact_root),
		"coverage_matrix": str(coverage_path),
		"patch_history": str(patch_history_path),
		"method_count": len(current_library.methods),
		"plan_count": len(final_materialization["bundle"].plan_library.plans),
		"library_validation": final_materialization["library_validation"],
		"translation_coverage": final_materialization["translation_coverage"],
	}
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	return summary


def build_incremental_patch_prompt(
	*,
	domain: Any,
	current_library: HTNMethodLibrary,
	temporal_specification: TemporalSpecificationRecord,
	evaluation_result: Dict[str, Any],
) -> Dict[str, str]:
	"""Build a compact prompt for a single-query method patch."""

	referenced_tasks = _referenced_task_names(temporal_specification)
	existing_methods = [
		method.to_dict()
		for method in current_library.methods
		if not referenced_tasks or method.task_name in referenced_tasks
	]
	if not existing_methods:
		existing_methods = [
			method.to_dict()
			for method in current_library.methods[:12]
		]
	system_prompt = (
		"Generate an incremental HTN method-library patch for Jason-runtime "
		"evaluation. Return exactly one JSON object with key methods. Do not "
		"return markdown, explanations, or natural-language prose."
	)
	user_payload = {
		"task": (
			"Add only reusable HTN methods needed to improve the failed query. "
			"Do not restate duplicate methods already present in existing_method_fingerprints. "
			"Do not copy benchmark object constants from the LTLf formula into method schemas."
		),
		"domain_contract": {
			"compound_tasks": _domain_compound_task_signatures(domain),
			"primitive_actions": _domain_action_summaries(domain),
			"predicates": _domain_predicate_signatures(domain),
		},
		"temporal_specification": {
			"instruction_id": temporal_specification.instruction_id,
			"ltlf_formula": temporal_specification.ltlf_formula,
			"referenced_task_names": sorted(referenced_tasks),
		},
		"failure_evidence": _compact_failure_evidence(evaluation_result),
		"current_library": {
			"method_count": len(current_library.methods),
			"methods_by_task": _methods_by_task_count(current_library),
			"relevant_methods": existing_methods,
			"existing_method_fingerprints": [
				method_fingerprint(method)
				for method in current_library.methods
			],
		},
		"output_contract": {
			"top_level": ["methods"],
			"method_fields": [
				"method_name",
				"task_name",
				"parameters",
				"task_args",
				"context",
				"subtasks",
				"ordering",
				"source_instruction_ids",
			],
			"subtask_fields": ["step_id", "task_name", "args", "kind"],
			"rules": [
				"method.task_name and compound subtasks must be declared compound tasks",
				"primitive subtasks must be declared primitive actions",
				"context literals may use only predicates or equality",
				"ordering edges must reference local step_id values",
				"source_instruction_ids must include the supplied instruction_id",
				"variables must be schema variables, not concrete benchmark objects",
			],
		},
	}
	return {
		"system": system_prompt,
		"user": json.dumps(user_payload, indent=2, ensure_ascii=False),
	}


def parse_method_patch_response(
	*,
	response_text: str,
	domain: Any,
	current_library: HTNMethodLibrary,
	temporal_specification: TemporalSpecificationRecord,
	config: Config | None = None,
) -> HTNMethodLibrary:
	"""Parse and normalise one method patch response."""

	active_config = config or get_config()
	synthesizer = HTNMethodSynthesizer(
		api_key=None,
		model=active_config.method_synthesis_model,
		base_url=active_config.method_synthesis_base_url,
		timeout=float(active_config.method_synthesis_timeout),
		max_tokens=int(active_config.method_synthesis_max_tokens),
		session_id=active_config.method_synthesis_session_id,
	)
	parsed = synthesizer._parse_llm_library(
		response_text,
		ast_compiler_defaults=synthesizer._build_domain_complete_ast_compiler_defaults(domain),
	)
	seed_library = HTNMethodLibrary(
		compound_tasks=(
			list(current_library.compound_tasks)
			or list(empty_method_library_for_domain(domain).compound_tasks)
		),
		primitive_tasks=(
			list(current_library.primitive_tasks)
			or synthesizer._build_primitive_tasks(domain)
		),
		methods=list(parsed.methods),
		target_literals=[],
		target_task_bindings=[],
	)
	normalised = synthesizer._normalise_llm_library(seed_library, domain)
	normalised = synthesizer._attach_source_instruction_ids(
		normalised,
		temporal_specifications=(temporal_specification.to_dict(),),
	)
	return HTNMethodLibrary(
		compound_tasks=list(seed_library.compound_tasks),
		primitive_tasks=list(seed_library.primitive_tasks),
		methods=list(normalised.methods),
		target_literals=[],
		target_task_bindings=[],
	)


def method_fingerprint(method: HTNMethod) -> str:
	"""Return a stable semantic fingerprint for a method, excluding its name."""

	step_id_map = {
		str(step.step_id): f"s{index + 1}"
		for index, step in enumerate(method.subtasks)
	}
	payload = {
		"task_name": method.task_name,
		"task_args": list(method.task_args),
		"context": _sorted_json_fingerprints(
			_literal_fingerprint(literal)
			for literal in method.context
		),
		"subtasks": [
			_step_fingerprint(step)
			for step in method.subtasks
		],
		"ordering": sorted(
			[
				step_id_map.get(str(before), str(before)),
				step_id_map.get(str(after), str(after)),
			]
			for before, after in method.ordering
		),
	}
	return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _merge_tasks(existing_tasks: Sequence[HTNTask], patch_tasks: Sequence[HTNTask]) -> list[HTNTask]:
	merged: list[HTNTask] = []
	seen: set[str] = set()
	for task in [*list(existing_tasks), *list(patch_tasks)]:
		task_name = str(task.name or "").strip()
		if not task_name or task_name in seen:
			continue
		seen.add(task_name)
		merged.append(task)
	return merged


def _merge_method_source_instruction_ids(base: HTNMethod, patch: HTNMethod) -> HTNMethod:
	source_ids = tuple(
		dict.fromkeys(
			[
				*list(base.source_instruction_ids),
				*list(patch.source_instruction_ids),
			],
		)
	)
	return replace(base, source_instruction_ids=source_ids)


def _unique_name(name: str, used_names: set[str]) -> str:
	base = sanitize_identifier(name) or "item"
	if base not in used_names:
		return base
	index = 2
	while f"{base}__{index}" in used_names:
		index += 1
	return f"{base}__{index}"


def _literal_fingerprint(literal: HTNLiteral) -> Dict[str, Any]:
	return {
		"predicate": literal.predicate,
		"args": list(literal.args),
		"is_positive": literal.is_positive,
	}


def _sorted_json_fingerprints(values: Any) -> list[Dict[str, Any]]:
	return sorted(
		list(values),
		key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
	)


def _step_fingerprint(step: HTNMethodStep) -> Dict[str, Any]:
	return {
		"task_name": step.task_name,
		"args": list(step.args),
		"kind": step.kind,
		"action_name": step.action_name,
	}


def _parameter_name(parameter: Any) -> str:
	text = str(parameter or "").strip()
	if ":" in text:
		text = text.split(":", 1)[0].strip()
	if " - " in text:
		text = text.split(" - ", 1)[0].strip()
	if text.startswith("?"):
		text = text[1:].strip()
	return sanitize_identifier(text).upper() or "ARG"


def _safe_query_id(query_id: Any) -> str:
	return sanitize_identifier(str(query_id or "").strip()) or "query"


def _first_existing_path(*paths: Path) -> Optional[Path]:
	for path in paths:
		if path.exists() and path.is_file():
			return path
	return None


def _is_complete_plan_library_artifact(path: Path) -> bool:
	required_files = (
		"query_sequence.json",
		"temporal_specifications.json",
		"method_library.json",
		"plan_library.json",
		"translation_coverage.json",
		"library_validation.json",
		"plan_library.asl",
	)
	return path.exists() and all((path / filename).exists() for filename in required_files)


def _load_json_list(path: Path) -> list[Dict[str, Any]]:
	if not path.exists():
		return []
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		return []
	if not isinstance(payload, list):
		return []
	return [dict(item) for item in payload if isinstance(item, dict)]


def _compact_evaluation_attempt(
	*,
	attempt_index: int,
	evaluation_result: Dict[str, Any],
) -> Dict[str, Any]:
	return {
		"attempt": attempt_index,
		"success": bool(evaluation_result.get("success")),
		"step": str(evaluation_result.get("step") or ""),
		"failure_class": str(evaluation_result.get("failure_class") or ""),
		"error": str(evaluation_result.get("error") or ""),
		"log_path": str(evaluation_result.get("log_path") or ""),
		"evaluation_report_path": str(evaluation_result.get("evaluation_report_path") or ""),
	}


def _compact_failure_evidence(evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
	plan_verification = dict(evaluation_result.get("plan_verification") or {})
	plan_solve = dict(evaluation_result.get("plan_solve") or {})
	return {
		"success": bool(evaluation_result.get("success")),
		"failed_step": str(evaluation_result.get("step") or ""),
		"failure_class": str(evaluation_result.get("failure_class") or ""),
		"error": str(evaluation_result.get("error") or ""),
		"failed_goals": list(
			((plan_solve.get("artifacts") or {}).get("failed_goals") or ())
		),
		"verifier_missing_goal_facts": list(
			((plan_verification.get("artifacts") or {}).get("missing_goal_facts") or ())
		),
	}


def _referenced_task_names(temporal_specification: TemporalSpecificationRecord) -> set[str]:
	task_names: set[str] = set()
	for atom_expression in extract_formula_atoms_in_order(temporal_specification.ltlf_formula):
		raw_task_name = str(atom_expression).split("(", 1)[0].strip()
		_exact_event_name, base_event_name, _ = parse_task_event_predicate_name(raw_task_name)
		if base_event_name:
			task_names.add(sanitize_identifier(base_event_name))
	return task_names


def _methods_by_task_count(method_library: HTNMethodLibrary) -> Dict[str, int]:
	counts: Dict[str, int] = {}
	for method in method_library.methods:
		counts[method.task_name] = counts.get(method.task_name, 0) + 1
	return dict(sorted(counts.items()))


def _domain_compound_task_signatures(domain: Any) -> list[str]:
	signatures: list[str] = []
	for task in getattr(domain, "tasks", ()) or ():
		task_name = str(getattr(task, "name", "") or "").strip()
		if not task_name:
			continue
		signatures.append(
			f"{task_name}({', '.join(str(parameter) for parameter in task.parameters)})",
		)
	return signatures


def _domain_predicate_signatures(domain: Any) -> list[str]:
	signatures: list[str] = []
	for predicate in getattr(domain, "predicates", ()) or ():
		name = str(getattr(predicate, "name", "") or "").strip()
		if not name:
			continue
		signatures.append(
			f"{name}({', '.join(str(parameter) for parameter in predicate.parameters)})",
		)
	return signatures


def _domain_action_summaries(domain: Any) -> list[Dict[str, Any]]:
	parser = HDDLConditionParser()
	summaries: list[Dict[str, Any]] = []
	for action in getattr(domain, "actions", ()) or ():
		parsed = parser.parse_action(action)
		summaries.append(
			{
				"name": parsed.name,
				"parameters": list(parsed.parameters),
				"preconditions": [
					_hddl_literal_signature(literal)
					for literal in parsed.preconditions
					if literal.predicate != "="
				],
				"effects": [
					_hddl_literal_signature(literal)
					for literal in parsed.effects
					if literal.predicate != "="
				],
			},
		)
	return summaries


def _hddl_literal_signature(literal: Any) -> str:
	base = str(getattr(literal, "predicate", "") or "").strip()
	args = tuple(str(arg).strip() for arg in (getattr(literal, "args", ()) or ()))
	if args:
		base = f"{base}({', '.join(args)})"
	if bool(getattr(literal, "is_positive", True)):
		return base
	return f"!{base}"
