"""
Canonical ordered LTLf helpers for benchmark-style task-only queries.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Dict, Iterable, List, Literal, Mapping, Sequence, Tuple


BenchmarkOrderedFormulaStyle = Literal[
	"adjacent_until_strict_precedence",
	"anchored_next_chain",
	"eventual_next_chain",
]

ADJACENT_UNTIL_STRICT_PRECEDENCE: BenchmarkOrderedFormulaStyle = (
	"adjacent_until_strict_precedence"
)
ANCHORED_NEXT_CHAIN: BenchmarkOrderedFormulaStyle = "anchored_next_chain"
EVENTUAL_NEXT_CHAIN: BenchmarkOrderedFormulaStyle = "eventual_next_chain"
CANONICAL_BENCHMARK_ORDERED_FORMULA_STYLE: BenchmarkOrderedFormulaStyle = (
	ANCHORED_NEXT_CHAIN
)


def apply_task_event_occurrence_suffixes(task_calls: Sequence[str]) -> Tuple[str, ...]:
	"""Suffix repeated grounded task calls with stable ``__eN`` event identities."""

	total_counts = Counter(str(task_call).strip() for task_call in task_calls if str(task_call).strip())
	seen_counts: Counter[str] = Counter()
	suffixed_calls: List[str] = []
	for task_call in task_calls:
		normalized_task_call = str(task_call).strip()
		if not normalized_task_call:
			continue
		seen_counts[normalized_task_call] += 1
		if total_counts[normalized_task_call] <= 1:
			suffixed_calls.append(normalized_task_call)
			continue
		open_paren_index = normalized_task_call.find("(")
		if open_paren_index == -1:
			suffixed_calls.append(
				f"{normalized_task_call}__e{seen_counts[normalized_task_call]}",
			)
			continue
		suffixed_calls.append(
			"".join(
				[
					normalized_task_call[:open_paren_index],
					f"__e{seen_counts[normalized_task_call]}",
					normalized_task_call[open_paren_index:],
				],
			),
		)
	return tuple(suffixed_calls)


def build_unordered_eventuality_formula(task_atoms: Sequence[str]) -> str:
	"""Render an unordered conjunction of eventualities."""

	atoms = tuple(str(atom).strip() for atom in task_atoms if str(atom).strip())
	if not atoms:
		return ""
	return " & ".join(f"F({atom})" for atom in atoms)


def build_ordered_benchmark_formula(
	task_atoms: Sequence[str],
	*,
	style: BenchmarkOrderedFormulaStyle = CANONICAL_BENCHMARK_ORDERED_FORMULA_STYLE,
) -> str:
	"""Render a canonical ordered benchmark LTLf formula for one task-event sequence."""

	atoms = tuple(str(atom).strip() for atom in task_atoms if str(atom).strip())
	if not atoms:
		return ""
	if len(atoms) == 1:
		return f"F({atoms[0]})"
	if style == ADJACENT_UNTIL_STRICT_PRECEDENCE:
		return _build_adjacent_until_formula(atoms)
	if style == ANCHORED_NEXT_CHAIN:
		return _build_anchored_next_formula(atoms)
	if style == EVENTUAL_NEXT_CHAIN:
		return _build_eventual_next_formula(atoms)
	raise ValueError(f"Unsupported benchmark ordered formula style '{style}'.")


def select_canonical_benchmark_ordered_formula_style(
	candidate_metrics: Mapping[str, Mapping[str, object]],
) -> BenchmarkOrderedFormulaStyle:
	"""Choose the canonical ordered-form style using the agreed bakeoff policy."""

	normalized_candidates: List[Tuple[str, Dict[str, object]]] = []
	for style_name, metrics in candidate_metrics.items():
		style_token = str(style_name).strip()
		if not style_token:
			continue
		normalized_candidates.append((style_token, dict(metrics)))
	if not normalized_candidates:
		raise ValueError("No benchmark ordered-form candidates were provided.")

	complete_candidates = [
		(style_name, metrics)
		for style_name, metrics in normalized_candidates
		if int(metrics.get("compiled_case_count") or 0)
		>= int(metrics.get("total_case_count") or 0)
		and int(metrics.get("total_case_count") or 0) > 0
	]
	if not complete_candidates:
		raise ValueError("No candidate compiled every benchmark bakeoff case.")

	def metric_median(metrics: Mapping[str, object], key: str) -> float:
		value = metrics.get(key)
		if isinstance(value, (int, float)):
			return float(value)
		series = [
			float(item)
			for item in (metrics.get(f"{key}_series") or ())
			if isinstance(item, (int, float))
		]
		if not series:
			return float("inf")
		return float(statistics.median(series))

	ranked_candidates = sorted(
		complete_candidates,
		key=lambda item: (
			metric_median(item[1], "median_num_states"),
			metric_median(item[1], "median_num_transitions"),
			metric_median(item[1], "median_convert_seconds"),
			metric_median(item[1], "median_formula_length"),
			str(item[0]),
		),
	)
	chosen_style = str(ranked_candidates[0][0]).strip()
	if chosen_style not in {
		ADJACENT_UNTIL_STRICT_PRECEDENCE,
		ANCHORED_NEXT_CHAIN,
		EVENTUAL_NEXT_CHAIN,
	}:
		raise ValueError(f"Unsupported selected benchmark ordered-form style '{chosen_style}'.")
	return chosen_style  # type: ignore[return-value]


def ordered_formula_style_prompt_guidance(
	*,
	style: BenchmarkOrderedFormulaStyle = CANONICAL_BENCHMARK_ORDERED_FORMULA_STYLE,
) -> Tuple[str, ...]:
	"""Return prompt rules that lock benchmark ordered queries to one formula family."""

	if style == ADJACENT_UNTIL_STRICT_PRECEDENCE:
		return (
			"- For explicit ordered benchmark task lists, use adjacent strict precedence constraints only.",
			"- For 'A then B', use F(A) & F(B) & (!B U (A & !B)).",
			"- For 'A then B then C', add one adjacent strict-precedence clause for A before B and one for B before C.",
			"- Do not use deeply nested eventuality chains such as F(A & F(B & F(C))).",
		)
	if style == ANCHORED_NEXT_CHAIN:
		return (
			"- For explicit ordered benchmark task lists, encode the listed task-event sequence as one anchored Next chain.",
			"- For 'A then B', use A & X(B).",
			"- For 'A then B then C', use A & X(B & X(C)).",
			"- Do not wrap the ordered task-event chain in an outer F(...).",
			"- Do not use deep nested eventuality chains such as F(A & F(B & F(C))).",
			"- Use F(atom) only for single-task or unordered eventual obligations.",
		)
	return (
		"- For explicit ordered benchmark task lists, encode the listed task-event sequence as one eventual Next chain.",
		"- For 'A then B', use F(A & X(B)).",
		"- For 'A then B then C', use F(A & X(B & X(C))).",
		"- Do not use deep nested eventuality chains such as F(A & F(B & F(C))).",
	)


def _build_adjacent_until_formula(task_atoms: Sequence[str]) -> str:
	atoms = tuple(task_atoms)
	parts = [f"F({atom})" for atom in atoms]
	parts.extend(
		f"(!{atoms[index + 1]} U ({atoms[index]} & !{atoms[index + 1]}))"
		for index in range(len(atoms) - 1)
	)
	return " & ".join(parts)


def _build_anchored_next_formula(task_atoms: Sequence[str]) -> str:
	inner = str(task_atoms[-1]).strip()
	for atom in reversed(task_atoms[:-1]):
		inner = f"{str(atom).strip()} & X({inner})"
	return inner


def _build_eventual_next_formula(task_atoms: Sequence[str]) -> str:
	return f"F({_build_anchored_next_formula(task_atoms)})"
