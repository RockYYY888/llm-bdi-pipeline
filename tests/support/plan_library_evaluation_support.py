from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
	sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from execution_logging.execution_logger import ExecutionLogger
from evaluation.orchestrator import PlanLibraryEvaluationOrchestrator
from evaluation.pipeline import _temporal_specification_to_grounding_result
from evaluation.failure_signature import build_failure_signature
from domain_model import load_query_sequence_records
from plan_library import (
	PlanLibraryArtifactBundle,
	build_library_validation_record,
	build_plan_library,
	deduplicate_plan_library,
	load_plan_library_artifact_bundle,
	persist_plan_library_artifact_bundle,
	render_plan_library_asl,
)
from plan_library.models import TranslationCoverage
from utils.benchmark_query_dataset import load_problem_query_cases
from utils.hddl_parser import HDDLParser

from tests.support.plan_library_generation_support import (
	DOMAIN_FILES,
	GENERATED_LOGS_DIR,
	GENERATED_MASKED_DOMAIN_BUILDS_DIR,
	apply_generated_runtime_defaults,
	build_official_method_library,
	query_id_sort_key,
	run_generated_domain_build,
)


BENCHMARK_EVALUATION_RESULTS_DIR = PROJECT_ROOT / "tests" / "generated" / "plan_library_evaluation"
BENCHMARK_EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OFFICIAL_LIBRARY_ARTIFACTS_DIR = PROJECT_ROOT / "tests" / "generated" / "plan_library_evaluation_official_libraries"
OFFICIAL_LIBRARY_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_EVALUATION_DOMAIN_SOURCE = "benchmark"
BENCHMARK_EVALUATION_LIBRARY_SOURCE = "benchmark"
BENCHMARK_EVALUATION_RUNTIME_BACKEND = "jason"
GOAL_GROUNDING_PROVIDER_UNAVAILABLE_BUCKET = "goal_grounding_provider_unavailable"
OFFICIAL_LIBRARY_TRANSLATION_VERSION = "official_method_direct_query_runtime_v17"
OFFICIAL_LIBRARY_ARTIFACT_CACHE: Dict[str, Path] = {}
GENERATED_LIBRARY_ARTIFACT_REQUIRED_FILES = (
	"artifact_metadata.json",
	"query_sequence.json",
	"temporal_specifications.json",
	"method_library.json",
	"generated_domain.hddl",
	"plan_library.json",
	"plan_library.asl",
	"translation_coverage.json",
	"library_validation.json",
	"method_synthesis_metadata.json",
	"masked_domain.hddl",
)
COMPACT_RESULT_KEYS = frozenset({"success", "step", "error", "failure_class", "log_path"})
EXECUTION_SUMMARY_KEYS = (
	"timestamp",
	"success",
	"status",
	"step",
	"mode",
	"run_origin",
	"domain_name",
	"problem_name",
	"domain_file",
	"problem_file",
	"output_dir",
	"execution_time_seconds",
	"timings",
	"ltlf_formula",
	"ltlf_atom_count",
	"ltlf_operator_counts",
	"jason_failure_class",
	"failed_goals",
	"verifier_missing_goal_facts",
	"failure_signature",
)
EXECUTION_STEP_KEYS = (
	"goal_grounding",
	"agentspeak_rendering",
	"runtime_execution",
	"plan_verification",
)
REPORT_TEXT_FIELD_LIMIT_CHARS = 2_000
REPORT_SEQUENCE_FIELD_LIMIT = 20
LONG_TEXT_KEYS = frozenset(
	{
		"instruction",
		"natural_language",
		"query_text",
		"ltlf_formula",
	}
)


def apply_evaluation_runtime_defaults(
	env: Optional[MutableMapping[str, str]] = None,
) -> MutableMapping[str, str]:
	target_env = apply_generated_runtime_defaults(env)
	target_env["EVALUATION_DOMAIN_SOURCE"] = BENCHMARK_EVALUATION_DOMAIN_SOURCE
	return target_env


def _extract_reported_evaluation_domain_source(execution: Dict[str, Any]) -> str:
	candidate_paths = (
		("goal_grounding", "metadata", "evaluation_domain_source"),
		("agentspeak_rendering", "metadata", "evaluation_domain_source"),
		("runtime_execution", "metadata", "evaluation_domain_source"),
		("plan_verification", "metadata", "evaluation_domain_source"),
	)
	for step_name, metadata_key, source_key in candidate_paths:
		step_payload = execution.get(step_name)
		if not isinstance(step_payload, dict):
			continue
		metadata = step_payload.get(metadata_key)
		if not isinstance(metadata, dict):
			continue
		source = str(metadata.get(source_key) or "").strip().lower()
		if source:
			return source
	return BENCHMARK_EVALUATION_DOMAIN_SOURCE


def _classify_evaluation_failure(result: Dict[str, Any], execution: Dict[str, Any]) -> str:
	if bool(result.get("success")):
		summary = dict(
			result.get("plan_verification_summary")
			or ((result.get("plan_verification") or {}).get("summary") or {}),
		)
		if (
			str(summary.get("plan_kind") or "") == "primitive_only"
			and summary.get("primitive_plan_executable") is True
			and summary.get("runtime_goal_reached") is True
		):
			return "runtime_goal_verified"
		return "hierarchical_plan_verified"

	step = str(result.get("step") or "").strip()
	if step == "goal_grounding":
		if _is_goal_grounding_provider_unavailable(result, execution):
			return GOAL_GROUNDING_PROVIDER_UNAVAILABLE_BUCKET
		return "goal_grounding_failed"
	if step == "agentspeak_rendering":
		return "agentspeak_rendering_failed"
	if step == "runtime_execution":
		return "runtime_execution_failed"
	if step == "plan_verification":
		verification_status = str(
			((result.get("plan_verification") or {}).get("summary") or {}).get("status") or "",
		).strip()
		if verification_status == "failed":
			return "hierarchical_rejection_failed"
		return "plan_verification_failed"

	verification_payload = dict(execution.get("plan_verification") or {})
	if verification_payload.get("status") == "failed":
		return "hierarchical_rejection_failed"
	return "unknown_failure"


def _is_goal_grounding_provider_unavailable(
	result: Dict[str, Any],
	execution: Dict[str, Any],
) -> bool:
	if str(result.get("failure_class") or "").strip() == "goal_grounding_provider_unavailable":
		return True
	goal_grounding_payload = dict(execution.get("goal_grounding") or {})
	metadata = dict(goal_grounding_payload.get("metadata") or {})
	if str(metadata.get("failure_class") or "").strip() == "goal_grounding_provider_unavailable":
		return True
	error_text = str(goal_grounding_payload.get("error") or result.get("error") or "").lower()
	if not error_text:
		return False
	transport_fragments = (
		"provider did not return usable completion text",
		"exceeded the configured wall-clock timeout before a response chunk was created",
		"exceeded the configured first-chunk deadline before any streaming content arrived",
		"did not include any choices",
		"did not include a message payload",
		"did not contain any textual completion content",
		"returned empty textual completion content",
	)
	return any(fragment in error_text for fragment in transport_fragments)


def _extract_failure_signature(
	execution: Dict[str, Any],
	report_result: Dict[str, Any],
) -> Dict[str, Any]:
	ltlf_formula = str(execution.get("ltlf_formula") or "").strip() or None
	if not ltlf_formula:
		for step_name in ("goal_grounding", "runtime_execution"):
			step_payload = execution.get(step_name)
			if not isinstance(step_payload, dict):
				continue
			artifacts = dict(step_payload.get("artifacts") or {})
			ltlf_formula = str(artifacts.get("ltlf_formula") or "").strip() or None
			if ltlf_formula:
				break

	jason_failure_class = str(execution.get("jason_failure_class") or "").strip() or None
	if not jason_failure_class:
		runtime_payload = dict(execution.get("runtime_execution") or {})
		artifacts = dict(runtime_payload.get("artifacts") or {})
		jason_failure_class = str(artifacts.get("failure_class") or "").strip() or None

	failed_goals = tuple(
		str(goal).strip()
		for goal in (
			execution.get("failed_goals")
			or ((execution.get("runtime_execution") or {}).get("artifacts") or {}).get("failed_goals")
			or ()
		)
		if str(goal).strip()
	)
	verifier_missing_goal_facts: tuple[str, ...] = ()
	if not bool(report_result.get("success")):
		verifier_missing_goal_facts = tuple(
			str(fact).strip()
			for fact in (
				execution.get("verifier_missing_goal_facts")
				or ((execution.get("plan_verification") or {}).get("metadata") or {}).get("missing_goal_facts")
				or ((execution.get("plan_verification") or {}).get("artifacts") or {}).get("missing_goal_facts")
				or ()
			)
			if str(fact).strip()
		)
	signature = build_failure_signature(
		ltlf_formula=ltlf_formula,
		jason_failure_class=jason_failure_class,
		failed_goals=failed_goals,
		verifier_missing_goal_facts=verifier_missing_goal_facts,
	)
	stored_signature = dict(execution.get("failure_signature") or {})
	for key in (
		"ltlf_atom_count",
		"ltlf_operator_counts",
		"ltlf_formula_chars",
		"ltlf_formula_sha256",
		"ltlf_formula_truncated",
	):
		if key in stored_signature:
			signature[key] = stored_signature[key]
		elif key in execution:
			signature[key] = execution[key]
	return signature


def _compact_result_payload(result_payload: Dict[str, Any]) -> Dict[str, Any]:
	compact = {
		key: result_payload[key]
		for key in COMPACT_RESULT_KEYS
		if key in result_payload
	}
	plan_verification = result_payload.get("plan_verification")
	if isinstance(plan_verification, dict):
		summary = plan_verification.get("summary")
		if isinstance(summary, dict):
			compact["plan_verification_summary"] = dict(summary)
	return compact


def _compact_execution_summary(execution: Dict[str, Any]) -> Dict[str, Any]:
	summary: Dict[str, Any] = {
		key: execution[key]
		for key in EXECUTION_SUMMARY_KEYS
		if key in execution
	}
	for step_name in EXECUTION_STEP_KEYS:
		step_payload = execution.get(step_name)
		if not isinstance(step_payload, dict):
			continue
		step_summary = {
			key: step_payload[key]
			for key in ("status", "backend", "error", "metadata", "artifacts", "llm")
			if key in step_payload
		}
		if step_summary:
			summary[step_name] = step_summary
	return _compact_nested_long_text(summary)


def _compact_text_field(value: str) -> Dict[str, Any]:
	text = str(value or "")
	if len(text) <= REPORT_TEXT_FIELD_LIMIT_CHARS:
		return {"value": text, "truncated": False}
	return {
		"value": text[:REPORT_TEXT_FIELD_LIMIT_CHARS],
		"truncated": True,
		"chars": len(text),
		"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
	}


def _set_compact_text(
	payload: Dict[str, Any],
	key: str,
	value: Any,
) -> None:
	if value is None:
		payload[key] = None
		return
	compact = _compact_text_field(str(value))
	payload[key] = compact["value"]
	if compact["truncated"]:
		payload[f"{key}_truncated"] = True
		payload[f"{key}_chars"] = compact["chars"]
		payload[f"{key}_sha256"] = compact["sha256"]


def _compact_failure_signature(signature: Dict[str, Any]) -> Dict[str, Any]:
	compact = dict(signature)
	if "ltlf_formula" in compact:
		value = compact.pop("ltlf_formula")
		_set_compact_text(compact, "ltlf_formula", value)
	for key in ("failed_goals", "verifier_missing_goal_facts"):
		value = compact.get(key)
		if not isinstance(value, list) or len(value) <= REPORT_SEQUENCE_FIELD_LIMIT:
			continue
		compact[key] = value[:REPORT_SEQUENCE_FIELD_LIMIT]
		compact[f"{key}_truncated"] = True
		compact[f"{key}_count"] = len(value)
	return compact


def _compact_nested_long_text(value: Any, *, key: str = "") -> Any:
	if isinstance(value, dict):
		return {
			item_key: _compact_nested_long_text(item_value, key=str(item_key))
			for item_key, item_value in value.items()
		}
	if isinstance(value, list):
		return [_compact_nested_long_text(item) for item in value]
	if isinstance(value, str) and key in LONG_TEXT_KEYS:
		return _compact_text_field(value)["value"]
	return value


def _execution_path_for_log_dir(log_dir: Any) -> str | None:
	if not log_dir:
		return None
	return str((Path(str(log_dir)).resolve() / "execution.json"))


def _verification_mode_from_execution_summary(execution_summary: Dict[str, Any]) -> str:
	runtime_payload = execution_summary.get("runtime_execution")
	if not isinstance(runtime_payload, dict):
		return ""
	metadata = runtime_payload.get("metadata")
	if not isinstance(metadata, dict):
		return ""
	return str(metadata.get("verification_mode") or "")


def _missing_generated_library_artifact_files(artifact_root: Path) -> tuple[str, ...]:
	root = Path(artifact_root)
	return tuple(
		file_name
		for file_name in GENERATED_LIBRARY_ARTIFACT_REQUIRED_FILES
		if not (root / file_name).is_file()
	)


def _generated_library_artifact_is_complete(artifact_root: Path) -> bool:
	return not _missing_generated_library_artifact_files(Path(artifact_root))


def ensure_generated_library_artifact(domain_key: str) -> Path:
	artifact_root = GENERATED_MASKED_DOMAIN_BUILDS_DIR / domain_key
	if _generated_library_artifact_is_complete(artifact_root):
		return artifact_root.resolve()
	build_report = run_generated_domain_build(domain_key)
	if not build_report.get("success"):
		raise RuntimeError(f"Method-library generation failed for {domain_key}: {build_report}")
	rebuild_root = Path(build_report["artifact_root"]).resolve()
	missing_files = _missing_generated_library_artifact_files(rebuild_root)
	if missing_files:
		raise RuntimeError(
			"Generated plan-library artifact is incomplete after rebuild for "
			f"{domain_key}: missing {', '.join(missing_files)}",
		)
	return rebuild_root


def ensure_official_library_artifact(domain_key: str) -> Path:
	cached = OFFICIAL_LIBRARY_ARTIFACT_CACHE.get(domain_key)
	if cached is not None and _official_library_artifact_is_current(cached):
		return cached

	artifact_root = OFFICIAL_LIBRARY_ARTIFACTS_DIR / domain_key
	if _official_library_artifact_is_current(artifact_root):
		OFFICIAL_LIBRARY_ARTIFACT_CACHE[domain_key] = artifact_root
		return artifact_root

	domain_file = DOMAIN_FILES[domain_key]
	domain = HDDLParser.parse_domain(domain_file)
	method_library = build_official_method_library(domain_file)
	plan_library, translation_coverage = build_plan_library(
		domain=domain,
		method_library=method_library,
	)
	set_result = deduplicate_plan_library(plan_library)
	plan_library = set_result.plan_library
	if set_result.removed_duplicate_plans:
		translation_coverage = TranslationCoverage(
			domain_name=translation_coverage.domain_name,
			methods_considered=translation_coverage.methods_considered,
			plans_generated=len(plan_library.plans),
			accepted_translation=translation_coverage.accepted_translation,
			unsupported_buckets=dict(translation_coverage.unsupported_buckets),
			unsupported_methods=tuple(translation_coverage.unsupported_methods),
		)
	library_validation = build_library_validation_record(
		domain_name=str(getattr(domain, "name", "") or domain_key),
		domain=domain,
		method_library=method_library,
		plan_library=plan_library,
		translation_coverage=translation_coverage,
		method_validation={
			"layers": {
				"signature_conformance": {"passed": True, "warnings": []},
				"typed_structural_soundness": {"passed": True, "warnings": []},
				"decomposition_admissibility": {"passed": True, "warnings": []},
				"materialized_parseability": {"passed": True, "warnings": []},
			},
		},
	)
	artifact = PlanLibraryArtifactBundle(
		domain_name=str(getattr(domain, "name", "") or domain_key),
		query_sequence=(),
		temporal_specifications=(),
		method_library=method_library,
		plan_library=plan_library,
		translation_coverage=translation_coverage,
		library_validation=library_validation,
		method_synthesis_metadata={
			"source": "official_unmasked_hddl",
			"domain_file": domain_file,
			"domain_key": domain_key,
			"translation_version": OFFICIAL_LIBRARY_TRANSLATION_VERSION,
			"plan_set_normalisation": set_result.to_dict(),
		},
		artifact_root=str(artifact_root),
		plan_library_asl_file=str(artifact_root / "plan_library.asl"),
	)
	persist_plan_library_artifact_bundle(
		artifact_root=artifact_root,
		artifact=artifact,
		plan_library_asl_text=render_plan_library_asl(plan_library),
	)
	OFFICIAL_LIBRARY_ARTIFACT_CACHE[domain_key] = artifact_root
	return artifact_root


def _official_library_artifact_is_current(artifact_root: Path) -> bool:
	metadata_path = artifact_root / "method_synthesis_metadata.json"
	if not (artifact_root / "plan_library.json").exists() or not metadata_path.exists():
		return False
	try:
		metadata = json.loads(metadata_path.read_text())
	except json.JSONDecodeError:
		return False
	return (
		str(metadata.get("translation_version") or "").strip()
		== OFFICIAL_LIBRARY_TRANSLATION_VERSION
	)


def resolve_plan_library_input(
	domain_key: str,
	*,
	library_source: str,
) -> str | Any:
	source = str(library_source or BENCHMARK_EVALUATION_LIBRARY_SOURCE).strip().lower()
	if source in {"benchmark", "official"}:
		return str(ensure_official_library_artifact(domain_key))
	if source == "generated":
		return str(ensure_generated_library_artifact(domain_key))
	raise ValueError(f"Unsupported evaluation benchmark library source '{library_source}'.")


def load_domain_query_cases(domain_key: str) -> Dict[str, Dict[str, Any]]:
	return load_problem_query_cases(
		PROJECT_ROOT / "src" / "domains" / domain_key / "problems",
		limit=10**9,
	)


def load_domain_temporal_specifications(
	domain_key: str,
	*,
	query_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
	_, temporal_specifications = load_query_sequence_records(
		domain_file=DOMAIN_FILES[domain_key],
		query_domain=domain_key,
		query_ids=query_ids,
	)
	return {
		record.instruction_id: record
		for record in temporal_specifications
	}


def run_plan_library_evaluation_case(
	domain_key: str,
	query_id: str,
	*,
	library_source: str = BENCHMARK_EVALUATION_LIBRARY_SOURCE,
	runtime_backend: str = BENCHMARK_EVALUATION_RUNTIME_BACKEND,
	logs_root: str | Path | None = None,
) -> Dict[str, Any]:
	apply_evaluation_runtime_defaults()
	query_cases = load_domain_query_cases(domain_key)
	case = query_cases[query_id]
	temporal_specifications = load_domain_temporal_specifications(
		domain_key,
		query_ids=(query_id,),
	)
	temporal_specification = temporal_specifications[query_id]
	if str(temporal_specification.problem_file or "").strip():
		expected_problem_name = Path(str(temporal_specification.problem_file)).name
		actual_problem_name = Path(str(case["problem_file"])).name
		if expected_problem_name != actual_problem_name:
			raise AssertionError(
				"Benchmark query case and stored temporal specification disagree on "
				f"problem_file for {domain_key}/{query_id}: "
				f"{actual_problem_name} != {expected_problem_name}",
			)
	case = {
		**case,
		"instruction": temporal_specification.source_text or case["instruction"],
		"ltlf_formula": temporal_specification.ltlf_formula,
	}
	domain_file = DOMAIN_FILES[domain_key]
	method_library_input = resolve_plan_library_input(
		domain_key,
		library_source=library_source,
	)
	library_artifact = load_plan_library_artifact_bundle(method_library_input)

	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=domain_file,
		problem_file=str(case["problem_file"]),
		evaluation_domain_source=BENCHMARK_EVALUATION_DOMAIN_SOURCE,
		runtime_backend=runtime_backend,
	)
	resolved_logs_root = (
		Path(logs_root).resolve()
		if logs_root is not None
		else GENERATED_LOGS_DIR
	)
	orchestrator.logger = ExecutionLogger(logs_dir=str(resolved_logs_root), run_origin="tests")
	grounding_result = _temporal_specification_to_grounding_result(
		temporal_specification=temporal_specification,
		method_library=library_artifact.method_library,
		problem=orchestrator.problem,
		task_type_map=orchestrator.task_type_map,
	)
	result = orchestrator.execute_grounded_query_with_library(
		case["instruction"],
		library_artifact=library_artifact,
		grounding_result=grounding_result,
	)
	log_path = result.get("log_path")
	log_dir = Path(str(log_path)).resolve().parent if log_path else None
	execution = (
		json.loads((log_dir / "execution.json").read_text())
		if log_dir is not None and (log_dir / "execution.json").exists()
		else {}
	)
	reported_evaluation_domain_source = _extract_reported_evaluation_domain_source(execution)
	if reported_evaluation_domain_source != BENCHMARK_EVALUATION_DOMAIN_SOURCE:
		raise AssertionError(
			"Plan-library evaluation benchmark must run against the benchmark domain source. "
			f"Observed: {reported_evaluation_domain_source}",
		)
	outcome_bucket = _classify_evaluation_failure(result, execution)
	failure_signature = _extract_failure_signature(execution, result)

	return {
		"query_id": query_id,
		"case": case,
		"problem_file": str(case["problem_file"]),
		"library_source": library_source,
		"runtime_backend": runtime_backend,
		"success": bool(result.get("success")),
		"result": result,
		"outcome_bucket": outcome_bucket,
		"log_dir": log_dir,
		"execution": execution,
		"failure_signature": failure_signature,
		"evaluation_domain_source": reported_evaluation_domain_source,
	}


def _domain_output_root(
	output_root: str | Path | None,
	domain_key: str,
) -> Path:
	if output_root is None:
		return (BENCHMARK_EVALUATION_RESULTS_DIR / domain_key).resolve()
	return (Path(output_root).resolve() / domain_key).resolve()


def _query_results_root(domain_output_root: Path) -> Path:
	return domain_output_root / "query_results"


def _query_result_path(domain_output_root: Path, query_id: str) -> Path:
	return _query_results_root(domain_output_root) / f"{query_id}.json"


def _domain_state_path(domain_output_root: Path) -> Path:
	return domain_output_root / "state.json"


def _domain_summary_path(domain_output_root: Path) -> Path:
	return domain_output_root / "summary.json"


def _top_level_domain_summary_path(
	output_root: str | Path | None,
	domain_key: str,
) -> Path:
	if output_root is None:
		return _domain_summary_path(_domain_output_root(output_root, domain_key))
	return Path(output_root).resolve() / f"{domain_key}.summary.json"


def _serialize_query_report(report: Dict[str, Any]) -> Dict[str, Any]:
	problem_file = str(
		report.get("problem_file")
		or ((report.get("case") or {}).get("problem_file") or ""),
	)
	instruction = str(
		report.get("instruction")
		or ((report.get("case") or {}).get("instruction") or ""),
	).strip()
	log_dir = report.get("log_dir")
	result_payload = dict(report.get("result") or {})
	execution = dict(report.get("execution") or {})
	execution_summary = (
		dict(report.get("execution_summary") or {})
		or _compact_execution_summary(execution)
	)
	verification_mode = (
		str(report.get("verification_mode") or "").strip()
		or _verification_mode_from_execution_summary(execution_summary)
	)
	failure_signature = _compact_failure_signature(
		dict(report.get("failure_signature") or {}),
	)
	payload = {
		"run_id": report.get("run_id"),
		"domain_key": str(report.get("domain_key") or ""),
		"query_id": str(report.get("query_id") or ""),
		"problem_file": problem_file,
		"library_source": str(report.get("library_source") or ""),
		"runtime_backend": str(
			report.get("runtime_backend")
			or BENCHMARK_EVALUATION_RUNTIME_BACKEND,
		),
		"success": bool(report.get("success")),
		"result": _compact_result_payload(result_payload),
		"outcome_bucket": str(report.get("outcome_bucket") or "unknown_failure"),
		"goal_grounding_failure_class": str(
			report.get("goal_grounding_failure_class")
			or result_payload.get("failure_class")
			or "",
		),
		"log_dir": str(log_dir) if log_dir else None,
		"execution_path": (
			str(report.get("execution_path") or "").strip()
			or _execution_path_for_log_dir(log_dir)
		),
		"execution_summary": execution_summary,
		"verification_mode": verification_mode,
		"failure_signature": failure_signature,
		"ltlf_formula": failure_signature.get("ltlf_formula"),
		"ltlf_atom_count": failure_signature.get("ltlf_atom_count"),
		"ltlf_operator_counts": dict(
			(failure_signature.get("ltlf_operator_counts") or {}),
		),
		"jason_failure_class": failure_signature.get("jason_failure_class"),
		"failed_goals": list((failure_signature.get("failed_goals") or [])),
		"verifier_missing_goal_facts": list(
			(failure_signature.get("verifier_missing_goal_facts") or []),
		),
		"evaluation_domain_source": str(report.get("evaluation_domain_source") or ""),
	}
	_set_compact_text(payload, "instruction", instruction)
	if failure_signature.get("ltlf_formula_truncated"):
		payload["ltlf_formula_truncated"] = True
		payload["ltlf_formula_chars"] = failure_signature.get("ltlf_formula_chars")
		payload["ltlf_formula_sha256"] = failure_signature.get("ltlf_formula_sha256")
	return payload


def _load_query_report_checkpoint(
	query_result_path: Path,
	*,
	domain_key: str,
	query_id: str,
	library_source: str,
	runtime_backend: str,
) -> Optional[Dict[str, Any]]:
	if not query_result_path.exists():
		return None
	try:
		payload = json.loads(query_result_path.read_text())
	except json.JSONDecodeError:
		return None
	if str(payload.get("domain_key") or "") != domain_key:
		return None
	if str(payload.get("query_id") or "") != query_id:
		return None
	if str(payload.get("library_source") or "") != library_source:
		return None
	if str(payload.get("runtime_backend") or BENCHMARK_EVALUATION_RUNTIME_BACKEND) != runtime_backend:
		return None
	if (
		str(payload.get("evaluation_domain_source") or "").strip().lower()
		!= BENCHMARK_EVALUATION_DOMAIN_SOURCE
	):
		return None
	return _serialize_query_report(payload)


def _count_query_outcomes(query_reports: Sequence[Dict[str, Any]]) -> Dict[str, int]:
	counts = {
		"hierarchical_plan_verified": 0,
		"runtime_goal_verified": 0,
		"goal_grounding_failed": 0,
		GOAL_GROUNDING_PROVIDER_UNAVAILABLE_BUCKET: 0,
		"agentspeak_rendering_failed": 0,
		"runtime_execution_failed": 0,
		"plan_verification_failed": 0,
		"hierarchical_rejection_failed": 0,
		"unknown_failure": 0,
	}
	for report in query_reports:
		bucket = str(report.get("outcome_bucket") or "unknown_failure")
		counts[bucket] = counts.get(bucket, 0) + 1
	return counts


def _build_domain_summary(
	*,
	domain_key: str,
	run_id: str | None,
	output_root: str | Path | None,
	domain_output_root: Path,
	selected_query_ids: Sequence[str],
	query_reports: Sequence[Dict[str, Any]],
	library_source: str,
	runtime_backend: str,
	library_artifact_ref: str | None,
	resume: bool,
	resumed_query_ids: Sequence[str],
) -> Dict[str, Any]:
	counts = _count_query_outcomes(query_reports)
	completed_query_ids = [str(report.get("query_id") or "") for report in query_reports]
	completed_query_id_set = set(completed_query_ids)
	remaining_query_ids = [
		query_id
		for query_id in selected_query_ids
		if query_id not in completed_query_id_set
	]
	bdi_runtime_successes = (
		counts.get("hierarchical_plan_verified", 0)
		+ counts.get("runtime_goal_verified", 0)
	)
	summary = {
		"run_id": run_id,
		"domain_key": domain_key,
		"evaluation_domain_source": BENCHMARK_EVALUATION_DOMAIN_SOURCE,
		"library_source": library_source,
		"runtime_backend": runtime_backend,
		"domain_file": DOMAIN_FILES[domain_key],
		"library_artifact_root": library_artifact_ref,
		"output_root": str(Path(output_root).resolve()) if output_root is not None else None,
		"domain_output_root": str(domain_output_root),
		"query_results_root": str(_query_results_root(domain_output_root)),
		"logs_root": str((domain_output_root / "logs").resolve()),
		"total_queries": len(selected_query_ids),
		"selected_query_ids": list(selected_query_ids),
		"completed_query_ids": completed_query_ids,
		"remaining_query_ids": remaining_query_ids,
		"resumed_query_ids": list(resumed_query_ids),
		"resume_enabled": bool(resume),
		"complete": not remaining_query_ids,
		"verified_successes": bdi_runtime_successes,
		"bdi_runtime_successes": bdi_runtime_successes,
		"hierarchical_compatibility_successes": counts.get("hierarchical_plan_verified", 0),
		"hierarchical_verified_successes": counts.get("hierarchical_plan_verified", 0),
		"runtime_goal_verified_successes": counts.get("runtime_goal_verified", 0),
		"goal_grounding_failures": counts.get("goal_grounding_failed", 0),
		"goal_grounding_provider_failures": counts.get(
			GOAL_GROUNDING_PROVIDER_UNAVAILABLE_BUCKET,
			0,
		),
		"agentspeak_rendering_failures": counts.get("agentspeak_rendering_failed", 0),
		"runtime_execution_failures": counts.get("runtime_execution_failed", 0),
		"plan_verification_failures": counts.get("plan_verification_failed", 0),
		"hierarchical_rejection_failures": counts.get("hierarchical_rejection_failed", 0),
		"unknown_failures": counts.get("unknown_failure", 0),
		"query_results": [
			{
				"query_id": str(report.get("query_id") or ""),
				"run_id": report.get("run_id"),
				"problem_file": str(report.get("problem_file") or ""),
				"log_dir": str(report.get("log_dir") or ""),
				"query_result_path": str(
					_query_result_path(
						domain_output_root,
						str(report.get("query_id") or ""),
					).resolve(),
				),
				"success": bool(report.get("success")),
				"outcome_bucket": str(report.get("outcome_bucket") or ""),
				"goal_grounding_failure_class": str(
					report.get("goal_grounding_failure_class")
					or ((report.get("result") or {}).get("failure_class") or ""),
				),
				"step": str((report.get("result") or {}).get("step") or ""),
				"verification_mode": str(
					report.get("verification_mode")
					or _verification_mode_from_execution_summary(
						dict(report.get("execution_summary") or {}),
					),
				),
				"execution_path": str(report.get("execution_path") or ""),
				"ltlf_formula": (report.get("failure_signature") or {}).get("ltlf_formula"),
				"ltlf_atom_count": (report.get("failure_signature") or {}).get("ltlf_atom_count"),
				"ltlf_operator_counts": dict(
					((report.get("failure_signature") or {}).get("ltlf_operator_counts") or {}),
				),
				"jason_failure_class": (report.get("failure_signature") or {}).get("jason_failure_class"),
				"failed_goals": list(((report.get("failure_signature") or {}).get("failed_goals") or [])),
				"verifier_missing_goal_facts": list(
					((report.get("failure_signature") or {}).get("verifier_missing_goal_facts") or []),
				),
				"evaluation_domain_source": str(report.get("evaluation_domain_source") or ""),
				"library_source": str(report.get("library_source") or ""),
				"runtime_backend": str(report.get("runtime_backend") or ""),
			}
			for report in query_reports
		],
	}
	return summary


def _write_domain_summary_artifacts(
	*,
	output_root: str | Path | None,
	domain_key: str,
	domain_output_root: Path,
	summary: Dict[str, Any],
) -> None:
	domain_output_root.mkdir(parents=True, exist_ok=True)
	_query_results_root(domain_output_root).mkdir(parents=True, exist_ok=True)
	(domain_output_root / "logs").mkdir(parents=True, exist_ok=True)
	_domain_state_path(domain_output_root).write_text(json.dumps(summary, indent=2))
	_domain_summary_path(domain_output_root).write_text(json.dumps(summary, indent=2))
	top_level_summary_path = _top_level_domain_summary_path(output_root, domain_key)
	if top_level_summary_path != _domain_summary_path(domain_output_root):
		top_level_summary_path.write_text(json.dumps(summary, indent=2))


def run_plan_library_evaluation_benchmark_for_domain(
	domain_key: str,
	*,
	query_ids: Optional[Sequence[str]] = None,
	library_source: str = BENCHMARK_EVALUATION_LIBRARY_SOURCE,
	runtime_backend: str = BENCHMARK_EVALUATION_RUNTIME_BACKEND,
	output_root: str | Path | None = None,
	run_id: str | None = None,
	resume: bool = False,
) -> Dict[str, Any]:
	normalized_library_source = str(library_source or BENCHMARK_EVALUATION_LIBRARY_SOURCE).strip().lower()
	normalized_runtime_backend = str(runtime_backend or BENCHMARK_EVALUATION_RUNTIME_BACKEND).strip().lower()
	if normalized_runtime_backend != "jason":
		raise ValueError(f"Unsupported runtime backend '{runtime_backend}'.")
	if normalized_library_source == "generated":
		library_artifact_ref: str | None = str(ensure_generated_library_artifact(domain_key))
	elif normalized_library_source in {"benchmark", "official"}:
		library_artifact_ref = str(ensure_official_library_artifact(domain_key))
	else:
		library_artifact_ref = None
	domain_output_root = _domain_output_root(output_root, domain_key)
	domain_output_root.mkdir(parents=True, exist_ok=True)
	_query_results_root(domain_output_root).mkdir(parents=True, exist_ok=True)
	(domain_output_root / "logs").mkdir(parents=True, exist_ok=True)
	query_cases = load_domain_query_cases(domain_key)
	selected_query_ids = (
		tuple(sorted(query_ids, key=query_id_sort_key))
		if query_ids
		else tuple(sorted(query_cases, key=query_id_sort_key))
	)
	missing_query_ids = [query_id for query_id in selected_query_ids if query_id not in query_cases]
	if missing_query_ids:
		raise KeyError(
			f"Unknown query ids for domain '{domain_key}': {', '.join(missing_query_ids)}",
		)
	query_reports_by_id: Dict[str, Dict[str, Any]] = {}
	resumed_query_ids: list[str] = []
	if resume:
		for query_id in selected_query_ids:
			cached_report = _load_query_report_checkpoint(
				_query_result_path(domain_output_root, query_id),
				domain_key=domain_key,
				query_id=query_id,
				library_source=normalized_library_source,
				runtime_backend=normalized_runtime_backend,
			)
			if cached_report is None:
				continue
			query_reports_by_id[query_id] = cached_report
			resumed_query_ids.append(query_id)

	for query_id in selected_query_ids:
		if query_id in query_reports_by_id:
			continue
		query_report = _serialize_query_report(
			{
				**run_plan_library_evaluation_case(
					domain_key,
					query_id,
					library_source=normalized_library_source,
					runtime_backend=normalized_runtime_backend,
					logs_root=domain_output_root / "logs",
				),
				"run_id": run_id,
				"domain_key": domain_key,
			},
		)
		query_reports_by_id[query_id] = query_report
		_query_result_path(domain_output_root, query_id).write_text(
			json.dumps(query_report, indent=2),
		)
		partial_reports = [
			query_reports_by_id[current_query_id]
			for current_query_id in selected_query_ids
			if current_query_id in query_reports_by_id
		]
		partial_summary = _build_domain_summary(
			domain_key=domain_key,
			run_id=run_id,
			output_root=output_root,
			domain_output_root=domain_output_root,
			selected_query_ids=selected_query_ids,
			query_reports=partial_reports,
			library_source=normalized_library_source,
			runtime_backend=normalized_runtime_backend,
			library_artifact_ref=library_artifact_ref,
			resume=resume,
			resumed_query_ids=resumed_query_ids,
		)
		_write_domain_summary_artifacts(
			output_root=output_root,
			domain_key=domain_key,
			domain_output_root=domain_output_root,
			summary=partial_summary,
		)

	query_reports = [
		query_reports_by_id[query_id]
		for query_id in selected_query_ids
		if query_id in query_reports_by_id
	]
	summary = _build_domain_summary(
		domain_key=domain_key,
		run_id=run_id,
		output_root=output_root,
		domain_output_root=domain_output_root,
		selected_query_ids=selected_query_ids,
		query_reports=query_reports,
		library_source=normalized_library_source,
		runtime_backend=normalized_runtime_backend,
		library_artifact_ref=library_artifact_ref,
		resume=resume,
		resumed_query_ids=resumed_query_ids,
	)
	_write_domain_summary_artifacts(
		output_root=output_root,
		domain_key=domain_key,
		domain_output_root=domain_output_root,
		summary=summary,
	)
	return summary


__all__ = [
	"BENCHMARK_EVALUATION_RESULTS_DIR",
	"BENCHMARK_EVALUATION_DOMAIN_SOURCE",
	"BENCHMARK_EVALUATION_LIBRARY_SOURCE",
	"apply_evaluation_runtime_defaults",
	"ensure_generated_library_artifact",
	"load_domain_query_cases",
	"run_plan_library_evaluation_case",
	"run_plan_library_evaluation_benchmark_for_domain",
]
