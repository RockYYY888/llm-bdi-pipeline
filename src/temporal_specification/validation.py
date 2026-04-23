"""
Validation helpers for temporal specifications and LTLf event extraction.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.symbol_normalizer import SymbolNormalizer

from .models import ReferencedEvent, TemporalSpecificationRecord


_FORMULA_OPERATOR_SEQUENCE_PATTERN = re.compile(r"^(?:WX|[FGXURW])+$")
_TASK_EVENT_SUFFIX_PATTERN = re.compile(
	r"^(?P<base>[a-z_][a-z0-9_]*?)__(?:e|event)(?P<index>[1-9][0-9]*)$",
)
_RESERVED_FORMULA_TOKENS = {
	"true",
	"false",
	"F",
	"G",
	"X",
	"WX",
	"U",
	"R",
	"last",
}


def extract_formula_atoms_in_order(ltlf_formula: str) -> Tuple[str, ...]:
	"""Extract task-event atoms from one LTLf formula in source order."""

	ordered_atoms: List[str] = []
	invalid_tokens: set[str] = set()
	text = str(ltlf_formula or "")
	index = 0
	while index < len(text):
		if not (text[index].isalpha() or text[index] == "_"):
			index += 1
			continue
		start = index
		index += 1
		while index < len(text) and (text[index].isalnum() or text[index] == "_"):
			index += 1
		token = text[start:index]
		if token in _RESERVED_FORMULA_TOKENS or _FORMULA_OPERATOR_SEQUENCE_PATTERN.fullmatch(token):
			continue
		if index < len(text) and text[index] == "(":
			if token in _RESERVED_FORMULA_TOKENS or _FORMULA_OPERATOR_SEQUENCE_PATTERN.fullmatch(token):
				index += 1
				continue
			depth = 1
			index += 1
			while index < len(text) and depth > 0:
				if text[index] == "(":
					depth += 1
				elif text[index] == ")":
					depth -= 1
				index += 1
			token = text[start:index]
		if token.startswith("subgoal_") or token.startswith("query_step_"):
			invalid_tokens.add(token)
			continue
		ordered_atoms.append(token)
	if invalid_tokens:
		raise ValueError(
			"LTLf formula may not use placeholder atoms such as subgoal_* or query_step_*. "
			"Found: " + ", ".join(sorted(invalid_tokens)),
		)
	return tuple(ordered_atoms)


def parse_task_event_predicate_name(raw_predicate_name: str) -> Tuple[str, str, bool]:
	"""Split an event predicate into exact name, base task name, and suffix flag."""

	predicate_name = str(raw_predicate_name or "").strip()
	match = _TASK_EVENT_SUFFIX_PATTERN.fullmatch(predicate_name)
	if match is None:
		return predicate_name, predicate_name, False
	return predicate_name, str(match.group("base") or "").strip(), True


def referenced_events_from_formula(ltlf_formula: str) -> Tuple[ReferencedEvent, ...]:
	"""Build referenced-event records directly from the atoms in an LTLf formula."""

	normalizer = SymbolNormalizer()
	events: List[ReferencedEvent] = []
	for atom_expression in extract_formula_atoms_in_order(ltlf_formula):
		predicate_name, arguments = normalizer.parse_predicate_string(atom_expression)
		exact_event_name, _base_event_name, _ = parse_task_event_predicate_name(predicate_name)
		events.append(
			ReferencedEvent(
				event=exact_event_name,
				arguments=tuple(str(argument).strip() for argument in arguments if str(argument).strip()),
			),
		)
	return tuple(events)


def build_domain_event_name_map(domain: Any) -> Dict[str, str]:
	"""Build a raw-name lookup for task and action events declared by the domain."""

	name_map: Dict[str, str] = {}
	for task in getattr(domain, "tasks", ()) or ():
		task_name = str(getattr(task, "name", "") or "").strip()
		if task_name:
			name_map[task_name] = task_name
	for action in getattr(domain, "actions", ()) or ():
		action_name = str(getattr(action, "name", "") or "").strip()
		if action_name:
			name_map[action_name] = action_name
	return name_map


def validate_temporal_specification_record(
	record: TemporalSpecificationRecord,
	*,
	domain: Any,
) -> TemporalSpecificationRecord:
	"""Validate event references and enrich a temporal specification with extracted events."""

	if not record.instruction_id:
		raise ValueError("Temporal specification requires a non-empty instruction_id.")
	if not record.source_text:
		raise ValueError(f'Temporal specification "{record.instruction_id}" requires source_text.')
	if not record.ltlf_formula:
		raise ValueError(f'Temporal specification "{record.instruction_id}" requires ltlf_formula.')

	name_map = build_domain_event_name_map(domain)
	diagnostics: List[str] = [
		str(message).strip()
		for message in record.diagnostics
		if str(message).strip()
	]
	referenced_events = list(record.referenced_events or referenced_events_from_formula(record.ltlf_formula))
	if not referenced_events:
		raise ValueError(
			f'Temporal specification "{record.instruction_id}" does not reference any task events.',
		)

	validated_events: List[ReferencedEvent] = []
	for event in referenced_events:
		event_name = str(event.event or "").strip()
		if not event_name:
			continue
		_exact_name, base_event_name, has_explicit_suffix = parse_task_event_predicate_name(event_name)
		if base_event_name not in name_map:
			raise ValueError(
				f'Temporal specification "{record.instruction_id}" references unknown event '
				f'"{base_event_name}".',
			)
		if has_explicit_suffix:
			diagnostics.append(f"Repeated event identity preserved for {event_name}.")
		validated_events.append(
			ReferencedEvent(
				event=event_name,
				arguments=tuple(str(argument).strip() for argument in event.arguments if str(argument).strip()),
			),
		)

	return TemporalSpecificationRecord(
		instruction_id=record.instruction_id,
		source_text=record.source_text,
		ltlf_formula=record.ltlf_formula,
		referenced_events=tuple(validated_events),
		diagnostics=tuple(dict.fromkeys(diagnostics)),
		problem_file=record.problem_file,
	)


def normalise_temporal_specification_payloads(
	payloads: Sequence[Dict[str, Any]],
	*,
	domain: Any,
) -> Tuple[TemporalSpecificationRecord, ...]:
	"""Convert raw payloads into validated temporal-specification records."""

	return tuple(
		validate_temporal_specification_record(
			TemporalSpecificationRecord.from_dict(payload),
			domain=domain,
		)
		for payload in payloads
	)
