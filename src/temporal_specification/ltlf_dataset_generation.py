"""
Generate stored LTLf benchmark specifications from natural-language query records.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from domain_model import infer_query_domain
from evaluation.goal_grounding.grounder import NLToLTLfGenerator
from evaluation.runtime_context import (
	build_type_parent_map_for_domain,
	task_type_map_for_domain,
)
from temporal_specification.models import TemporalSpecificationRecord
from temporal_specification.validation import validate_temporal_specification_record
from utils.benchmark_query_dataset import DEFAULT_BENCHMARK_QUERY_DATASET_PATH
from utils.config import Config, get_config
from utils.hddl_parser import HDDLParser


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LTLF_DATASET_PATH = PROJECT_ROOT / "src" / "benchmark_data" / "queries_LTLf.json"
DEFAULT_QUERY_DOMAIN_FILES: Dict[str, Path] = {
	"blocksworld": PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl",
	"marsrover": PROJECT_ROOT / "src" / "domains" / "marsrover" / "domain.hddl",
	"satellite": PROJECT_ROOT / "src" / "domains" / "satellite" / "domain.hddl",
	"transport": PROJECT_ROOT / "src" / "domains" / "transport" / "domain.hddl",
}

GeneratorFactory = Callable[..., NLToLTLfGenerator]


def generate_ltlf_dataset(
	*,
	source_query_dataset: str | Path | None = None,
	output_dataset: str | Path | None = None,
	domain_file: str | Path | None = None,
	query_domain: str | None = None,
	query_ids: Sequence[str] | None = None,
	regenerate_existing: bool = False,
	config: Config | None = None,
	generator_factory: GeneratorFactory = NLToLTLfGenerator,
) -> Dict[str, Any]:
	"""
	Generate a stored LTLf dataset from natural-language benchmark query records.

	The source dataset is expected to use the existing benchmark schema:
	``domains[domain_key].cases[query_id]`` with ``instruction`` and
	``problem_file`` fields. Existing ``ltlf_formula`` values are reused unless
	``regenerate_existing`` is true.
	"""

	active_config = config or get_config()
	source_path = Path(source_query_dataset or DEFAULT_BENCHMARK_QUERY_DATASET_PATH).expanduser().resolve()
	output_path = Path(output_dataset or DEFAULT_LTLF_DATASET_PATH).expanduser().resolve()
	source_payload = _load_query_dataset(source_path)
	selected_query_ids = _normalise_query_ids(query_ids)
	selected_domain_files = _selected_domain_files(
		source_payload=source_payload,
		domain_file=domain_file,
		query_domain=query_domain,
	)

	output_payload: Dict[str, Any] = {
		"version": 1,
		"dataset_kind": "stored_benchmark_ltlf_queries",
		"query_protocol_document": source_payload.get("query_protocol_document"),
		"source_query_dataset": str(source_path),
		"ltlf_generator": {
			"implementation": "evaluation.goal_grounding.NLToLTLfGenerator",
			"model": active_config.ltlf_generation_model,
			"base_url": active_config.ltlf_generation_base_url,
			"session_id": active_config.ltlf_generation_session_id,
		},
		"domains": {},
	}
	summary: Dict[str, Any] = {
		"success": True,
		"source_query_dataset": str(source_path),
		"output_dataset": str(output_path),
		"domains": {},
		"total_cases": 0,
		"total_generated": 0,
		"total_reused": 0,
	}

	for domain_key, resolved_domain_file in selected_domain_files:
		domain_cases = _domain_cases(source_payload, domain_key)
		if not domain_cases:
			raise ValueError(f'No source query cases found for domain "{domain_key}".')
		case_items = _selected_case_items(
			domain_key=domain_key,
			domain_cases=domain_cases,
			query_ids=selected_query_ids,
		)
		domain_summary, rendered_cases = _generate_domain_cases(
			domain_key=domain_key,
			domain_file=resolved_domain_file,
			case_items=case_items,
			regenerate_existing=regenerate_existing,
			config=active_config,
			generator_factory=generator_factory,
		)
		output_payload["domains"][domain_key] = {"cases": rendered_cases}
		summary["domains"][domain_key] = domain_summary
		summary["total_cases"] += domain_summary["case_count"]
		summary["total_generated"] += domain_summary["generated"]
		summary["total_reused"] += domain_summary["reused"]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n",
		encoding="utf-8",
	)
	return summary


def _load_query_dataset(source_path: Path) -> Dict[str, Any]:
	if not source_path.exists():
		raise FileNotFoundError(f"Missing source query dataset: {source_path}")
	payload = json.loads(source_path.read_text(encoding="utf-8"))
	if not isinstance(payload, dict):
		raise ValueError("Source query dataset must be a JSON object.")
	if not isinstance(payload.get("domains"), dict):
		raise ValueError('Source query dataset must contain a "domains" object.')
	return payload


def _selected_domain_files(
	*,
	source_payload: Dict[str, Any],
	domain_file: str | Path | None,
	query_domain: str | None,
) -> Tuple[Tuple[str, Path], ...]:
	if domain_file is not None:
		resolved_domain_file = Path(domain_file).expanduser().resolve()
		if not resolved_domain_file.exists():
			raise FileNotFoundError(f"Domain file does not exist: {resolved_domain_file}")
		domain_key = infer_query_domain(
			domain_file=resolved_domain_file,
			explicit_domain=query_domain,
		)
		return ((domain_key, resolved_domain_file),)

	if str(query_domain or "").strip():
		domain_key = infer_query_domain(
			domain_file=DEFAULT_QUERY_DOMAIN_FILES["blocksworld"],
			explicit_domain=query_domain,
		)
		return ((domain_key, _default_domain_file(domain_key)),)

	selected = []
	for domain_key in sorted((source_payload.get("domains") or {}).keys()):
		if domain_key not in DEFAULT_QUERY_DOMAIN_FILES:
			continue
		selected.append((domain_key, _default_domain_file(domain_key)))
	return tuple(selected)


def _default_domain_file(domain_key: str) -> Path:
	domain_file = DEFAULT_QUERY_DOMAIN_FILES.get(str(domain_key or "").strip())
	if domain_file is None:
		raise ValueError(f'No default domain file is registered for query domain "{domain_key}".')
	if not domain_file.exists():
		raise FileNotFoundError(f"Default domain file does not exist: {domain_file}")
	return domain_file.resolve()


def _domain_cases(source_payload: Dict[str, Any], domain_key: str) -> Dict[str, Dict[str, Any]]:
	return dict(((source_payload.get("domains") or {}).get(domain_key) or {}).get("cases") or {})


def _normalise_query_ids(query_ids: Sequence[str] | None) -> Tuple[str, ...]:
	if not query_ids:
		return ()
	seen: set[str] = set()
	normalised: list[str] = []
	for query_id in query_ids:
		query_id_text = str(query_id or "").strip()
		if not query_id_text or query_id_text in seen:
			continue
		seen.add(query_id_text)
		normalised.append(query_id_text)
	return tuple(normalised)


def _selected_case_items(
	*,
	domain_key: str,
	domain_cases: Dict[str, Dict[str, Any]],
	query_ids: Sequence[str],
) -> Tuple[Tuple[str, Dict[str, Any]], ...]:
	if query_ids:
		missing_query_ids = [
			query_id
			for query_id in query_ids
			if query_id not in domain_cases
		]
		if missing_query_ids:
			raise ValueError(
				f'Unknown query ids for domain "{domain_key}": {", ".join(missing_query_ids)}',
			)
		return tuple((query_id, dict(domain_cases[query_id])) for query_id in query_ids)
	return tuple(
		(query_id, dict(payload))
		for query_id, payload in sorted(domain_cases.items(), key=lambda item: _query_id_sort_key(item[0]))
	)


def _query_id_sort_key(query_id: str) -> tuple[int, str]:
	query_text = str(query_id or "").strip()
	prefix, separator, suffix = query_text.partition("_")
	if prefix == "query" and separator and suffix.isdigit():
		return (int(suffix), query_text)
	return (10**9, query_text)


def _generate_domain_cases(
	*,
	domain_key: str,
	domain_file: Path,
	case_items: Sequence[Tuple[str, Dict[str, Any]]],
	regenerate_existing: bool,
	config: Config,
	generator_factory: GeneratorFactory,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
	domain = HDDLParser.parse_domain(str(domain_file))
	type_parent_map = build_type_parent_map_for_domain(domain)
	domain_type_names = set(type_parent_map.keys())
	task_type_map = task_type_map_for_domain(domain, domain_type_names)
	generator = generator_factory(
		api_key=config.ltlf_generation_api_key,
		model=config.ltlf_generation_model,
		base_url=config.ltlf_generation_base_url,
		domain_file=str(domain_file),
		request_timeout=float(config.ltlf_generation_timeout),
		response_max_tokens=int(config.ltlf_generation_max_tokens),
		session_id=config.ltlf_generation_session_id,
	)
	rendered_cases: Dict[str, Dict[str, Any]] = {}
	case_summaries = []
	generated_count = 0
	reused_count = 0
	for query_id, payload in case_items:
		case_start = time.perf_counter()
		rendered_payload = dict(payload)
		instruction = str(
			payload.get("instruction")
			or payload.get("source_text")
			or payload.get("query_text")
			or ""
		).strip()
		if not instruction:
			raise ValueError(f'Query "{query_id}" in domain "{domain_key}" is missing instruction text.')
		problem_file = str(payload.get("problem_file") or "").strip()
		if not problem_file:
			raise ValueError(f'Query "{query_id}" in domain "{domain_key}" is missing problem_file.')

		existing_formula = str(payload.get("ltlf_formula") or "").strip()
		if existing_formula and not regenerate_existing:
			ltlf_formula = existing_formula
			reused_count += 1
			status = "reused"
		else:
			problem_path = _resolve_problem_file(domain_file=domain_file, problem_file=problem_file)
			problem = HDDLParser.parse_problem(str(problem_path))
			grounding_result, _llm_prompt, _llm_response = generator.generate(
				instruction,
				method_library=None,
				typed_objects={
					str(name).strip(): str(type_name).strip()
					for name, type_name in dict(problem.object_types or {}).items()
					if str(name).strip() and str(type_name).strip()
				},
				task_type_map=task_type_map,
				type_parent_map=type_parent_map,
			)
			ltlf_formula = grounding_result.ltlf_formula
			generated_count += 1
			status = "generated"

		validated_record = validate_temporal_specification_record(
			TemporalSpecificationRecord(
				instruction_id=str(query_id).strip(),
				source_text=instruction,
				ltlf_formula=ltlf_formula,
				referenced_events=(),
				diagnostics=(),
				problem_file=problem_file,
			),
			domain=domain,
		)
		rendered_payload["instruction"] = instruction
		rendered_payload["problem_file"] = problem_file
		rendered_payload["ltlf_formula"] = validated_record.ltlf_formula
		rendered_cases[str(query_id)] = rendered_payload
		case_summaries.append(
			{
				"query_id": str(query_id),
				"status": status,
				"ltlf_formula": validated_record.ltlf_formula,
				"referenced_event_count": len(validated_record.referenced_events),
				"seconds": round(time.perf_counter() - case_start, 6),
			},
		)

	return (
		{
			"domain_file": str(domain_file),
			"case_count": len(case_items),
			"generated": generated_count,
			"reused": reused_count,
			"cases": case_summaries,
		},
		rendered_cases,
	)


def _resolve_problem_file(*, domain_file: Path, problem_file: str) -> Path:
	problem_text = str(problem_file or "").strip()
	candidates = (
		(domain_file.parent / "problems" / problem_text).resolve(),
		(domain_file.parent / problem_text).resolve(),
		Path(problem_text).expanduser().resolve(),
	)
	for candidate in candidates:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(f"Could not resolve problem file {problem_file} for {domain_file}.")
