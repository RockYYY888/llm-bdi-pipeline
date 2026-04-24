from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from domain_model import load_query_sequence_records
from temporal_specification import (
	TemporalSpecificationRecord,
	extract_formula_atoms_in_order,
	referenced_events_from_formula,
	validate_temporal_specification_record,
)
from tests.support.plan_library_generation_support import DOMAIN_FILES
from utils.hddl_parser import HDDLParser


def test_query_sequence_loader_filters_domain_cases_in_numeric_order() -> None:
	query_sequence, temporal_specifications = load_query_sequence_records(
		domain_file=DOMAIN_FILES["blocksworld"],
	)

	assert len(query_sequence) == len(temporal_specifications)
	assert query_sequence[0].instruction_id == "query_1"
	assert query_sequence[1].instruction_id == "query_2"
	assert all(record.instruction_id.startswith("query_") for record in query_sequence[:3])
	assert all("do_put_on" in record.ltlf_formula for record in temporal_specifications[:3])
	assert all(str(record.problem_file).endswith(".hddl") for record in temporal_specifications[:3])


def test_query_sequence_loader_filters_explicit_query_ids_in_requested_order() -> None:
	query_sequence, temporal_specifications = load_query_sequence_records(
		domain_file=DOMAIN_FILES["blocksworld"],
		query_ids=("query_3", "query_1", "query_3"),
	)

	assert [record.instruction_id for record in query_sequence] == ["query_3", "query_1"]
	assert [record.instruction_id for record in temporal_specifications] == ["query_3", "query_1"]


def test_query_sequence_loader_rejects_unknown_explicit_query_ids() -> None:
	with pytest.raises(ValueError, match='Unknown query ids for domain "blocksworld": query_missing'):
		load_query_sequence_records(
			domain_file=DOMAIN_FILES["blocksworld"],
			query_ids=("query_missing",),
		)


def test_extract_formula_atoms_and_referenced_events_preserve_source_order() -> None:
	formula = "F(do_put_on(b1,b2)) & X(do_put_on__e2(b3,b1))"

	atoms = extract_formula_atoms_in_order(formula)
	referenced_events = referenced_events_from_formula(formula)

	assert atoms == ("do_put_on(b1,b2)", "do_put_on__e2(b3,b1)")
	assert referenced_events[0].event == "do_put_on"
	assert referenced_events[0].arguments == ("b1", "b2")
	assert referenced_events[1].event == "do_put_on__e2"
	assert referenced_events[1].arguments == ("b3", "b1")


def test_temporal_spec_validation_preserves_repeated_event_identity() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	record = TemporalSpecificationRecord(
		instruction_id="query_repeat",
		source_text="Repeat the same stacking task twice.",
		ltlf_formula="do_put_on__e1(b1,b2) & X(do_put_on__e2(b1,b2))",
		referenced_events=(),
		diagnostics=(),
	)

	validated = validate_temporal_specification_record(record, domain=domain)

	assert tuple(event.event for event in validated.referenced_events) == (
		"do_put_on__e1",
		"do_put_on__e2",
	)
	assert any(
		"Repeated event identity preserved" in diagnostic
		for diagnostic in validated.diagnostics
	)


def test_temporal_spec_validation_rejects_unknown_events() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	record = TemporalSpecificationRecord(
		instruction_id="query_invalid",
		source_text="Use a task that does not exist in the domain.",
		ltlf_formula="unknown_task(b1)",
		referenced_events=(),
		diagnostics=(),
	)

	with pytest.raises(ValueError, match='references unknown event "unknown_task"'):
		validate_temporal_specification_record(record, domain=domain)
