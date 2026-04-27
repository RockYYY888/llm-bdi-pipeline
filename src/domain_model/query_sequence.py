"""
Query-sequence loading for the plan-library generation workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from utils.hddl_parser import HDDLParser

from temporal_specification import (
	QueryInstructionRecord,
	TemporalSpecificationRecord,
	validate_temporal_specification_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPORAL_SPEC_DATASET_PATH = PROJECT_ROOT / "src" / "benchmark_data" / "queries_LTLf.json"
_DOMAIN_NAME_ALIASES = {
	"blocks": "blocksworld",
	"blocksworld": "blocksworld",
	"rover": "marsrover",
	"marsrover": "marsrover",
	"satellite": "satellite",
	"satellite2": "satellite",
	"transport": "transport",
}


def _query_id_sort_key(query_id: str) -> tuple[int, str]:
	query_text = str(query_id or "").strip()
	prefix, separator, suffix = query_text.partition("_")
	if prefix == "query" and separator and suffix.isdigit():
		return (int(suffix), query_text)
	return (10**9, query_text)


def infer_query_domain(
	*,
	domain_file: str | Path,
	explicit_domain: str | None = None,
) -> str:
	"""Infer the query-dataset domain key for one HDDL domain file."""

	if str(explicit_domain or "").strip():
		domain_key = _DOMAIN_NAME_ALIASES.get(str(explicit_domain).strip().lower())
		if domain_key:
			return domain_key
		raise ValueError(f'Unsupported query-domain key "{explicit_domain}".')

	domain_path = Path(domain_file).expanduser().resolve()
	parent_key = _DOMAIN_NAME_ALIASES.get(domain_path.parent.name.strip().lower())
	if parent_key:
		return parent_key

	parsed_domain = HDDLParser.parse_domain(str(domain_path))
	domain_name_key = _DOMAIN_NAME_ALIASES.get(str(parsed_domain.name or "").strip().lower())
	if domain_name_key:
		return domain_name_key

	raise ValueError(f"Could not infer query domain from {domain_file}.")


def load_temporal_specification_dataset(
	dataset_path: str | Path | None = None,
) -> Dict[str, Any]:
	"""Load the stored benchmark temporal-specification dataset."""

	target_path = (
		Path(dataset_path).expanduser().resolve()
		if dataset_path is not None
		else DEFAULT_TEMPORAL_SPEC_DATASET_PATH.resolve()
	)
	if not target_path.exists():
		raise FileNotFoundError(f"Missing temporal-specification dataset: {target_path}")
	return json.loads(target_path.read_text(encoding="utf-8"))


def load_query_sequence_records(
	*,
	domain_file: str | Path,
	dataset_path: str | Path | None = None,
	query_domain: str | None = None,
	query_ids: Sequence[str] | None = None,
) -> Tuple[Tuple[QueryInstructionRecord, ...], Tuple[TemporalSpecificationRecord, ...]]:
	"""Load the default query sequence and validated temporal specifications for one domain."""

	domain_path = Path(domain_file).expanduser().resolve()
	domain = HDDLParser.parse_domain(str(domain_path))
	domain_key = infer_query_domain(domain_file=domain_path, explicit_domain=query_domain)
	dataset = load_temporal_specification_dataset(dataset_path)
	domain_cases = (
		dict(((dataset.get("domains") or {}).get(domain_key) or {}).get("cases") or {})
	)
	if not domain_cases:
		raise ValueError(f'No temporal specification cases found for domain "{domain_key}".')

	selected_query_ids = _normalise_selected_query_ids(query_ids)
	if selected_query_ids:
		missing_query_ids = [
			query_id
			for query_id in selected_query_ids
			if query_id not in domain_cases
		]
		if missing_query_ids:
			raise ValueError(
				f'Unknown query ids for domain "{domain_key}": {", ".join(missing_query_ids)}',
			)
		case_items = tuple((query_id, domain_cases[query_id]) for query_id in selected_query_ids)
	else:
		case_items = tuple(sorted(domain_cases.items(), key=lambda item: _query_id_sort_key(item[0])))

	query_sequence = []
	temporal_specifications = []
	for query_id, payload in case_items:
		instruction_record = QueryInstructionRecord(
			instruction_id=str(query_id).strip(),
			source_text=str(payload.get("instruction") or payload.get("source_text") or "").strip(),
			problem_file=(
				str(payload.get("problem_file")).strip()
				if payload.get("problem_file") is not None
				else None
			),
		)
		query_sequence.append(instruction_record)
		temporal_specifications.append(
			validate_temporal_specification_record(
				TemporalSpecificationRecord(
					instruction_id=instruction_record.instruction_id,
					source_text=instruction_record.source_text,
					ltlf_formula=str(payload.get("ltlf_formula") or "").strip(),
					referenced_events=(),
					diagnostics=(),
					problem_file=instruction_record.problem_file,
				),
				domain=domain,
			),
		)
	return tuple(query_sequence), tuple(temporal_specifications)


def _normalise_selected_query_ids(query_ids: Sequence[str] | None) -> Tuple[str, ...]:
	if not query_ids:
		return ()
	seen: set[str] = set()
	selected_query_ids: list[str] = []
	for query_id in query_ids:
		query_id_text = str(query_id or "").strip()
		if not query_id_text or query_id_text in seen:
			continue
		seen.add(query_id_text)
		selected_query_ids.append(query_id_text)
	return tuple(selected_query_ids)
