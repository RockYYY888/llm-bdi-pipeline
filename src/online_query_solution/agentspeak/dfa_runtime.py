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

	symbolic_monitor = dfa_result.get("symbolic_subgoal_monitor")
	if isinstance(symbolic_monitor, dict) and not ordered_query_sequence:
		return _build_symbolic_subgoal_transition_specs(symbolic_monitor)

	graph = _parse_dfa_graph(str(dfa_result.get("dfa_dot") or ""))
	subgoal_mode = _graph_uses_subgoal_symbols(graph)
	if subgoal_mode and not ordered_query_sequence:
		return _build_compressed_subgoal_transition_specs(
			graph,
			grounding_map=grounding_map,
		)
	selected_edges = _select_relevant_edges(
		graph,
		unit_progress_only=subgoal_mode,
	)
	state_aliases = _build_state_aliases(graph, selected_edges)
	accepting_states = sorted(
		state_aliases[state]
		for state in graph.get("accepting", set())
		if state in state_aliases
	)
	initial_state = graph.get("init_state")
	initial_alias = state_aliases.get(initial_state) if initial_state else None

	transition_specs: List[Dict[str, Any]] = []
	seen_edges: set[Tuple[str, str, str]] = set()
	for source, target, raw_label in selected_edges:
		edge_key = (str(source), str(target), str(raw_label))
		if edge_key in seen_edges:
			continue
		seen_edges.add(edge_key)
		label = str(raw_label or "").strip()
		if not label or label == "false":
			continue
		positive_subgoals = _positive_subgoal_indices(label)
		subgoal_index = (
			positive_subgoals[0]
			if subgoal_mode and len(positive_subgoals) == 1
			else None
		)
		transition_specs.append(
			{
				"source_state": state_aliases[source],
				"target_state": state_aliases[target],
				"raw_source_state": source,
				"raw_target_state": target,
				"raw_label": label,
				"guard_context": (
					"true"
					if subgoal_index is not None
					else _render_raw_label_as_agentspeak_context(
						label,
						grounding_map,
					)
				),
				"subgoal_index": subgoal_index,
				"initial_state": initial_alias,
				"accepting_states": list(accepting_states),
			},
		)
	transition_specs.sort(key=_transition_spec_sort_key)
	for index, spec in enumerate(transition_specs, start=1):
		spec["transition_name"] = f"dfa_t{index}"
	return transition_specs


def _build_compressed_subgoal_transition_specs(
	graph: Dict[str, Any],
	*,
	grounding_map: GroundingMap | None,
) -> List[Dict[str, Any]]:
	del grounding_map
	accepting = set(graph.get("accepting", set()))
	initial_state = graph.get("init_state")
	ordered_states = _build_state_aliases(graph)
	initial_alias = ordered_states.get(initial_state) if initial_state else None
	accepting_aliases = sorted(
		ordered_states[state]
		for state in accepting
		if state in ordered_states
	)

	specs: List[Dict[str, Any]] = []
	for subgoal_index in _sorted_subgoal_indices(graph.get("edges", ())):
		specs.append(
			{
				"source_state": initial_alias or "q1",
				"target_state": initial_alias or "q1",
				"raw_source_state": initial_state or "init",
				"raw_target_state": initial_state or "init",
				"raw_label": f"subgoal_{subgoal_index}",
				"guard_context": "true",
				"subgoal_index": subgoal_index,
				"initial_state": initial_alias,
				"accepting_states": list(accepting_aliases),
				"stateless_subgoal": True,
			},
		)

	specs.sort(key=_transition_spec_sort_key)
	for index, spec in enumerate(specs, start=1):
		spec["transition_name"] = f"dfa_t{index}"
	return specs


def _build_symbolic_subgoal_transition_specs(
	symbolic_monitor: Dict[str, Any],
) -> List[Dict[str, Any]]:
	subgoal_indices = [
		int(index)
		for index in list(symbolic_monitor.get("subgoal_indices") or ())
	]
	initial_state = str(symbolic_monitor.get("initial_state") or "q0").strip() or "q0"
	accepting_states = [
		str(state).strip()
		for state in list(symbolic_monitor.get("accepting_states") or ())
		if str(state).strip()
	]

	specs: List[Dict[str, Any]] = []
	for subgoal_index in sorted(subgoal_indices):
		specs.append(
			{
				"source_state": initial_state,
				"target_state": initial_state,
				"raw_source_state": initial_state,
				"raw_target_state": initial_state,
				"raw_label": f"subgoal_{subgoal_index}",
				"guard_context": "true",
				"subgoal_index": subgoal_index,
				"initial_state": initial_state,
				"accepting_states": list(accepting_states),
				"stateless_subgoal": True,
			},
		)

	specs.sort(key=_transition_spec_sort_key)
	for index, spec in enumerate(specs, start=1):
		spec["transition_name"] = f"dfa_t{index}"
	return specs


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


def _select_relevant_edges(
	graph: Dict[str, Any],
	*,
	unit_progress_only: bool = False,
) -> List[Tuple[str, str, str]]:
	edges = list(graph.get("edges", []))
	accepting = set(graph.get("accepting", set()))
	if not edges:
		return []

	progress_edges = _extract_progressing_edges(
		edges,
		accepting,
		unit_progress_only=unit_progress_only,
	)
	if progress_edges:
		return progress_edges

	accepting_loops = _extract_accepting_loop_edges(edges, accepting)
	if accepting_loops:
		return accepting_loops

	return edges


def _extract_progressing_edges(
	edges: Sequence[Tuple[str, str, str]],
	accepting: set[str],
	*,
	unit_progress_only: bool = False,
) -> List[Tuple[str, str, str]]:
	if not edges or not accepting:
		return []

	distances = _distance_to_accepting(edges, accepting)
	progress_edges: List[Tuple[str, str, str]] = []
	for source, target, label in edges:
		source_distance = distances.get(source)
		target_distance = distances.get(target)
		if source_distance is None or target_distance is None:
			continue
		if source == target:
			continue
		if unit_progress_only:
			if len(_positive_subgoal_indices(str(label or ""))) == 1:
				progress_edges.append((source, target, label))
			continue
		if target_distance <= source_distance:
			progress_edges.append((source, target, label))
	return progress_edges


def _graph_uses_subgoal_symbols(graph: Dict[str, Any]) -> bool:
	for _source, _target, label in graph.get("edges", ()):
		if _positive_subgoal_indices(str(label or "")):
			return True
	return False


def _positive_subgoal_indices(raw_label: str) -> List[int]:
	indices: List[int] = []
	negated = False
	for token in _tokenise_raw_label(raw_label):
		if token == "~":
			negated = True
			continue
		if token in {"&", "|", "(", ")"}:
			continue
		match = re.fullmatch(r"subgoal_(\d+)", token)
		if match and not negated:
			indices.append(int(match.group(1)))
		negated = False
	return indices


def _sorted_subgoal_indices(
	edges: Iterable[Tuple[str, str, str]],
) -> List[int]:
	indices = {
		index
		for _source, _target, label in edges
		for index in _positive_subgoal_indices(str(label or ""))
	}
	return sorted(indices)


def _extract_accepting_loop_edges(
	edges: Sequence[Tuple[str, str, str]],
	accepting: set[str],
) -> List[Tuple[str, str, str]]:
	if not edges or not accepting:
		return []

	distances = _distance_to_accepting(edges, accepting)
	return [
		(source, target, label)
		for source, target, label in edges
		if distances.get(source) == 0 and distances.get(target) == 0
	]


def _distance_to_accepting(
	edges: Sequence[Tuple[str, str, str]],
	accepting: set[str],
) -> Dict[str, int]:
	reverse_graph: Dict[str, List[str]] = {}
	for source, target, _ in edges:
		reverse_graph.setdefault(target, []).append(source)

	distances: Dict[str, int] = {state: 0 for state in accepting}
	queue: deque[str] = deque(accepting)
	while queue:
		state = queue.popleft()
		for predecessor in reverse_graph.get(state, []):
			if predecessor in distances:
				continue
			distances[predecessor] = distances[state] + 1
			queue.append(predecessor)
	return distances


def _transition_spec_sort_key(spec: Dict[str, Any]) -> Tuple[int, int, int, str]:
	subgoal_index = spec.get("subgoal_index")
	return (
		_numeric_state_sort_key(spec.get("raw_source_state")),
		int(subgoal_index) if isinstance(subgoal_index, int) else 10**9,
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


def _render_raw_label_as_agentspeak_context(
	raw_label: str,
	grounding_map: GroundingMap | None,
) -> str:
	label = str(raw_label or "").strip()
	if not label:
		return "__hddl_unsat_condition__"

	output: List[str] = []
	for token in _tokenise_raw_label(label):
		if token == "~":
			output.append("not")
			continue
		if token in {"&", "|", "(", ")"}:
			output.append(token)
			continue
		if token == "true":
			output.append("true")
			continue
		if token == "false":
			output.append("__hddl_unsat_condition__")
			continue
		output.append(_render_grounded_atom(token, grounding_map))
	return " ".join(output).strip() or "__hddl_unsat_condition__"


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


def _render_grounded_atom(symbol: str, grounding_map: GroundingMap | None) -> str:
	token = str(symbol or "").strip()
	if not token:
		return "__hddl_unsat_condition__"

	if grounding_map is not None:
		atom = grounding_map.get_atom(token)
		if atom is not None:
			return _render_call(str(atom.predicate), tuple(str(arg) for arg in atom.args))

	return _render_call(token, ())


def _render_call(name: str, args: Sequence[str]) -> str:
	functor = str(name or "").strip().replace("-", "_")
	if not args:
		return functor
	rendered_args = ", ".join(_render_term(str(arg)) for arg in args if str(arg).strip())
	return f"{functor}({rendered_args})"


def _render_term(value: str) -> str:
	text = str(value or "").strip()
	if re.fullmatch(r"[a-z][a-zA-Z0-9_]*", text):
		return text
	escaped = text.replace("\\", "\\\\").replace('"', '\\"')
	return f'"{escaped}"'
