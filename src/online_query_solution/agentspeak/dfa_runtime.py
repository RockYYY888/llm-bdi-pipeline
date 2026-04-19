"""
Raw DFA helpers for AgentSpeak rendering.

This module keeps DFA interpretation on the AgentSpeak rendering side so
offline method synthesis remains focused on domain-task methods.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from online_query_solution.goal_grounding.grounding_map import GroundingMap


def build_agentspeak_transition_specs(
	*,
	dfa_result: Dict[str, Any],
	grounding_map: GroundingMap | None,
	ordered_query_sequence: bool = False,
) -> List[Dict[str, Any]]:
	"""Build renderer-facing transition specs directly from the raw ltlf2dfa graph."""

	del ordered_query_sequence
	graph = _parse_dfa_graph(str(dfa_result.get("dfa_dot") or ""))
	expanded_edges = _expand_disjunctive_edges(graph.get("edges", ()))
	state_aliases = _build_state_aliases(graph, expanded_edges)
	accepting_states = sorted(
		state_aliases[state]
		for state in graph.get("accepting", set())
		if state in state_aliases
	)
	initial_state = graph.get("init_state")
	initial_alias = state_aliases.get(initial_state) if initial_state else None

	transition_specs: List[Dict[str, Any]] = []
	seen_edges: set[Tuple[str, str, str]] = set()
	for source, target, raw_label in expanded_edges:
		edge_key = (str(source), str(target), str(raw_label))
		if edge_key in seen_edges:
			continue
		seen_edges.add(edge_key)
		label = str(raw_label or "").strip()
		if not label or label == "false":
			continue
		positive_task_events, negative_task_events = _task_event_polarity(
			label,
			grounding_map,
		)
		task_event_symbol = (
			positive_task_events[0]
			if len(positive_task_events) == 1
			else None
		)
		task_name = ""
		task_args: Tuple[str, ...] = ()
		if task_event_symbol and grounding_map is not None:
			grounded_atom = grounding_map.get_atom(task_event_symbol)
			if grounded_atom is not None:
				task_name = str(grounded_atom.predicate or "").strip()
				task_args = tuple(str(arg).strip() for arg in grounded_atom.args if str(arg).strip())
		transition_specs.append(
			{
				"source_state": state_aliases[source],
				"target_state": state_aliases[target],
				"raw_source_state": source,
				"raw_target_state": target,
				"raw_label": label,
				"positive_task_event_symbols": list(positive_task_events),
				"negative_task_event_symbols": list(negative_task_events),
				"task_event_symbol": task_event_symbol,
				"task_name": task_name,
				"task_args": list(task_args),
				"is_epsilon_transition": label == "true" and source != target,
				"initial_state": initial_alias,
				"accepting_states": list(accepting_states),
			},
		)
	transition_specs.sort(key=_transition_spec_sort_key)
	for index, spec in enumerate(transition_specs, start=1):
		spec["transition_name"] = f"dfa_t{index}"
	return transition_specs


def _parse_dfa_graph(dfa_dot: str) -> Dict[str, Any]:
	accepting: set[str] = set()
	edges: List[Tuple[str, str, str]] = []
	init_state: Optional[str] = None

	for line in dfa_dot.splitlines():
		stripped = line.strip()
		if not stripped:
			continue

		multi_accepting = re.match(
			r'node\s*\[\s*shape\s*=\s*doublecircle\s*\];\s*([^;]+);',
			stripped,
		)
		if multi_accepting:
			accepting.update(re.findall(r"[A-Za-z0-9_]+", multi_accepting.group(1)))
			continue

		single_accepting = re.match(
			r'([A-Za-z0-9_]+)\s*\[\s*shape\s*=\s*doublecircle\s*\];',
			stripped,
		)
		if single_accepting:
			accepting.add(single_accepting.group(1))
			continue

		init_match = re.match(r"init\s*->\s*([A-Za-z0-9_]+)\s*;", stripped)
		if init_match:
			init_state = init_match.group(1)
			continue

		edge_match = re.match(
			r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*\[label="([^"]+)"\];',
			stripped,
		)
		if edge_match:
			source, target, label = edge_match.groups()
			edges.append((source, target, label))

	return {
		"accepting": accepting,
		"edges": edges,
		"init_state": init_state,
	}

def _expand_disjunctive_edges(
	edges: Iterable[Tuple[str, str, str]],
) -> List[Tuple[str, str, str]]:
	expanded_edges: List[Tuple[str, str, str]] = []
	for source, target, raw_label in edges:
		disjuncts = _split_top_level_disjunctions(str(raw_label or "").strip())
		if not disjuncts:
			expanded_edges.append((source, target, str(raw_label or "").strip()))
			continue
		for disjunct in disjuncts:
			expanded_edges.append((source, target, disjunct))
	return expanded_edges


def _split_top_level_disjunctions(raw_label: str) -> List[str]:
	label = str(raw_label or "").strip()
	if not label:
		return []
	parts: List[str] = []
	current: List[str] = []
	depth = 0
	index = 0
	while index < len(label):
		character = label[index]
		if character == "(":
			depth += 1
		elif character == ")":
			depth = max(0, depth - 1)
		elif character == "|" and depth == 0:
			part = "".join(current).strip()
			if part:
				parts.append(_strip_outer_parentheses(part))
			current = []
			index += 1
			continue
		current.append(character)
		index += 1
	part = "".join(current).strip()
	if part:
		parts.append(_strip_outer_parentheses(part))
	return parts or [label]


def _strip_outer_parentheses(text: str) -> str:
	candidate = str(text or "").strip()
	while candidate.startswith("(") and candidate.endswith(")"):
		depth = 0
		wraps_entire_expression = True
		for index, character in enumerate(candidate):
			if character == "(":
				depth += 1
			elif character == ")":
				depth -= 1
				if depth == 0 and index != len(candidate) - 1:
					wraps_entire_expression = False
					break
		if not wraps_entire_expression:
			break
		candidate = candidate[1:-1].strip()
	return candidate


def _task_event_polarity(
	raw_label: str,
	grounding_map: GroundingMap | None,
) -> Tuple[List[str], List[str]]:
	if grounding_map is None:
		return [], []
	positive_symbols: List[str] = []
	negative_symbols: List[str] = []
	negated = False
	for token in _tokenise_raw_label(raw_label):
		if token == "~":
			negated = True
			continue
		if token in {"&", "|", "(", ")"}:
			continue
		if grounding_map.get_atom(token) is None:
			negated = False
			continue
		if negated:
			if token not in negative_symbols:
				negative_symbols.append(token)
		else:
			if token not in positive_symbols:
				positive_symbols.append(token)
		negated = False
	return positive_symbols, negative_symbols


def _transition_spec_sort_key(spec: Dict[str, Any]) -> Tuple[int, int, int, str]:
	return (
		_numeric_state_sort_key(spec.get("raw_source_state")),
		0 if str(spec.get("task_event_symbol") or "").strip() else 1,
		_numeric_state_sort_key(spec.get("raw_target_state")),
		str(spec.get("raw_label") or ""),
	)


def _numeric_state_sort_key(value: Any) -> int:
	token = str(value or "").strip()
	return int(token) if token.isdigit() else 10**9


def _build_state_aliases(
	graph: Dict[str, Any],
	preferred_edges: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> Dict[str, str]:
	ordered_states: List[str] = []
	seen_states: set[str] = set()

	def add(state: Optional[str]) -> None:
		if state and state not in seen_states:
			seen_states.add(state)
			ordered_states.append(state)

	add(graph.get("init_state"))
	for source, target, _ in preferred_edges or ():
		add(source)
		add(target)
	for source, target, _ in graph.get("edges", []):
		add(source)
		add(target)
	for state in sorted(graph.get("accepting", set())):
		add(state)

	return {
		state: f"q{index}"
		for index, state in enumerate(ordered_states, start=1)
	}

def _tokenise_raw_label(raw_label: str) -> List[str]:
	pattern = re.compile(r"\s*([()&|~]|[A-Za-z0-9_]+)\s*")
	tokens = [
		match.group(1)
		for match in pattern.finditer(str(raw_label or ""))
		if match.group(1)
	]
	if not tokens:
		raise ValueError(f"Unsupported raw DFA label: {raw_label!r}")
	return tokens
