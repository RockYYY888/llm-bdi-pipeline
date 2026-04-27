"""
Plan-library evaluation pipeline for artifact reuse and benchmark evidence collection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from domain_model import infer_query_domain, load_query_sequence_records
from evaluation.artifacts import GroundedSubgoal, TemporalGroundingResult
from evaluation.orchestrator import PlanLibraryEvaluationOrchestrator
from plan_library import PlanLibraryArtifactBundle, load_plan_library_artifact_bundle
from temporal_specification import (
	TemporalSpecificationRecord,
	extract_formula_atoms_in_order,
	parse_task_event_predicate_name,
	validate_temporal_specification_record,
)
from utils.hddl_parser import HDDLParser
from utils.symbol_normalizer import SymbolNormalizer


class PlanLibraryEvaluationPipeline:
	"""Evaluate a generated plan-library bundle on stored benchmark cases or ad hoc instructions."""

	def __init__(self, *, domain_file: str) -> None:
		self.project_root = Path(__file__).resolve().parents[2]
		self.domain_file = str(Path(domain_file).expanduser().resolve())
		self.domain = HDDLParser.parse_domain(self.domain_file)

	def evaluate_benchmark_case(
		self,
		*,
		library_artifact: str | Path | Dict[str, Any] | PlanLibraryArtifactBundle,
		query_id: str,
		query_dataset: str | None = None,
		query_domain: str | None = None,
	) -> Dict[str, Any]:
		bundle = load_plan_library_artifact_bundle(library_artifact)
		query_domain_key = infer_query_domain(
			domain_file=self.domain_file,
			explicit_domain=query_domain,
		)
		_, temporal_specifications = load_query_sequence_records(
			domain_file=self.domain_file,
			dataset_path=query_dataset,
			query_domain=query_domain_key,
		)
		record = next(
			(
				spec
				for spec in temporal_specifications
				if spec.instruction_id == str(query_id).strip()
			),
			None,
		)
		if record is None:
			raise ValueError(f'Unknown query id "{query_id}" for domain "{query_domain_key}".')
		if not str(record.problem_file or "").strip():
			raise ValueError(f'Benchmark query "{query_id}" is missing a bound problem file.')
		problem_file = self._domain_problem_path(record.problem_file)
		result = self._evaluate_temporal_specification(
			bundle=bundle,
			temporal_specification=record,
			problem_file=str(problem_file),
		)
		report_path = self._persist_evaluation_report(
			result=result,
			bundle=bundle,
			evaluation_mode="stored_benchmark_case",
			query_id=record.instruction_id,
			problem_file=str(problem_file),
		)
		result["evaluation_report_path"] = str(report_path)
		return result

	def evaluate_instruction(
		self,
		*,
		library_artifact: str | Path | Dict[str, Any] | PlanLibraryArtifactBundle,
		instruction: str,
		problem_file: str,
		ltlf_formula: str | None = None,
	) -> Dict[str, Any]:
		bundle = load_plan_library_artifact_bundle(library_artifact)
		resolved_problem_file = str(Path(problem_file).expanduser().resolve())
		if str(ltlf_formula or "").strip():
			record = validate_temporal_specification_record(
				TemporalSpecificationRecord(
					instruction_id="ad_hoc",
					source_text=str(instruction).strip(),
					ltlf_formula=str(ltlf_formula).strip(),
					referenced_events=(),
					diagnostics=(),
					problem_file=str(Path(resolved_problem_file).name),
				),
				domain=self.domain,
			)
			result = self._evaluate_temporal_specification(
				bundle=bundle,
				temporal_specification=record,
				problem_file=resolved_problem_file,
			)
			report_path = self._persist_evaluation_report(
				result=result,
				bundle=bundle,
				evaluation_mode="ad_hoc_temporal_specification",
				query_id=record.instruction_id,
				problem_file=resolved_problem_file,
			)
			result["evaluation_report_path"] = str(report_path)
			return result

		orchestrator = PlanLibraryEvaluationOrchestrator(
			domain_file=self.domain_file,
			problem_file=resolved_problem_file,
			evaluation_domain_source="benchmark",
		)
		result = orchestrator.execute_query_with_library(
			str(instruction).strip(),
			library_artifact=bundle,
			execution_mode="plan_library_evaluation",
		)
		report_path = self._persist_evaluation_report(
			result=result,
			bundle=bundle,
			evaluation_mode="ad_hoc_live_grounding",
			query_id="ad_hoc",
			problem_file=resolved_problem_file,
		)
		result["evaluation_report_path"] = str(report_path)
		return result

	def _evaluate_temporal_specification(
		self,
		*,
		bundle: PlanLibraryArtifactBundle,
		temporal_specification: TemporalSpecificationRecord,
		problem_file: str,
	) -> Dict[str, Any]:
		orchestrator = PlanLibraryEvaluationOrchestrator(
			domain_file=self.domain_file,
			problem_file=str(Path(problem_file).expanduser().resolve()),
			evaluation_domain_source="benchmark",
		)
		grounding_result = _temporal_specification_to_grounding_result(
			temporal_specification=temporal_specification,
			method_library=bundle.method_library,
			problem=orchestrator.problem,
			task_type_map=orchestrator.task_type_map,
		)
		return orchestrator.execute_grounded_query_with_library(
			nl_query=temporal_specification.source_text,
			library_artifact=bundle,
			grounding_result=grounding_result,
			execution_mode="plan_library_evaluation",
		)

	def _persist_evaluation_report(
		self,
		*,
		result: Dict[str, Any],
		bundle: PlanLibraryArtifactBundle,
		evaluation_mode: str,
		query_id: str,
		problem_file: str,
	) -> Path:
		log_path_text = str(result.get("log_path") or "").strip()
		if log_path_text:
			report_root = Path(log_path_text).expanduser().resolve().parent
		else:
			report_root = (
				self.project_root
				/ "tmp"
				/ "evaluation"
				/ str(bundle.domain_name or "unknown").strip()
			)
		report_root.mkdir(parents=True, exist_ok=True)
		report_path = report_root / "evaluation_report.json"
		report_path.write_text(
			json.dumps(
				{
					"evaluation_mode": evaluation_mode,
					"domain_name": bundle.domain_name,
					"library_artifact_root": bundle.artifact_root,
					"query_id": query_id,
					"problem_file": problem_file,
					"result": result,
				},
				indent=2,
				default=str,
			),
			encoding="utf-8",
		)
		return report_path

	def _domain_problem_path(self, problem_file: str) -> Path:
		domain_dir = Path(self.domain_file).resolve().parent
		problems_dir = domain_dir / "problems"
		candidate = (problems_dir / str(problem_file).strip()).resolve()
		if candidate.exists():
			return candidate
		candidate = (domain_dir / str(problem_file).strip()).resolve()
		if candidate.exists():
			return candidate
		raise FileNotFoundError(f"Could not resolve problem file {problem_file} for {self.domain_file}.")


def _temporal_specification_to_grounding_result(
	*,
	temporal_specification: TemporalSpecificationRecord,
	method_library,
	problem: Any,
	task_type_map: Dict[str, tuple[str, ...]],
) -> TemporalGroundingResult:
	symbol_normalizer = SymbolNormalizer()
	typed_objects = {
		str(name).strip(): str(type_name).strip()
		for name, type_name in dict(getattr(problem, "object_types", {}) or {}).items()
		if str(name).strip() and str(type_name).strip()
	}
	task_name_map = {}
	for task in [*list(method_library.compound_tasks), *list(method_library.primitive_tasks)]:
		task_name = str(getattr(task, "name", "") or "").strip()
		source_name = str(getattr(task, "source_name", "") or "").strip()
		if task_name:
			task_name_map[task_name] = task_name
		if source_name:
			task_name_map[source_name] = task_name or source_name

	subgoals = []
	seen_ids: set[str] = set()
	for atom_expression in extract_formula_atoms_in_order(temporal_specification.ltlf_formula):
		raw_task_name, raw_args = symbol_normalizer.parse_predicate_string(atom_expression)
		_exact_event_name, base_event_name, _ = parse_task_event_predicate_name(raw_task_name)
		task_name = (
			task_name_map.get(raw_task_name)
			or task_name_map.get(base_event_name)
			or base_event_name
		)
		args = tuple(str(arg).strip() for arg in raw_args if str(arg).strip())
		subgoal_id = symbol_normalizer.create_propositional_symbol(raw_task_name, list(args))
		if subgoal_id in seen_ids:
			continue
		seen_ids.add(subgoal_id)
		subgoals.append(
			GroundedSubgoal(
				subgoal_id=subgoal_id,
				task_name=task_name,
				args=args,
				argument_types=tuple(task_type_map.get(task_name, ())),
			),
		)

	return TemporalGroundingResult(
		query_text=temporal_specification.source_text,
		ltlf_formula=temporal_specification.ltlf_formula,
		subgoals=tuple(subgoals),
		typed_objects=typed_objects,
		query_object_inventory=(),
		diagnostics=tuple(temporal_specification.diagnostics),
	)
