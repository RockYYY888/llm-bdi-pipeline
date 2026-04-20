"""
Raw DFA helpers for AgentSpeak rendering.

This module keeps DFA interpretation on the AgentSpeak rendering side so
offline method synthesis remains focused on domain-task methods.
"""

from __future__ import annotations

import re
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
	guarded_transitions = _normalise_guarded_transitions(
		dfa_result.get("guarded_transitions") or (),
	)
	if guarded_transitions:
		return _build_transition_specs_from_guarded_transitions(
			guarded_transitions=guarded_transitions,
			dfa_result=dfa_result,
			grounding_map=grounding_map,
		)

	graph = _parse_dfa_graph(str(dfa_result.get("dfa_dot") or ""))
	return _build_transition_specs_from_dot_graph(
		graph=graph,
		dfa_result=dfa_result,
		grounding_map=grounding_map,
	)


def _build_transition_specs_from_guarded_transitions(
	*,
	guarded_transitions: Sequence[Dict[str, Any]],
	dfa_result: Dict[str, Any],
	grounding_map: GroundingMap | None,
) -> List[Dict[str, Any]]:
	free_variables = tuple(
		str(item).strip()
		for item in (dfa_result.get("free_variables") or ())
		if str(item).strip()
	)
	alphabet_symbols = _ordered_runtime_symbols(
		dfa_result,
		grounding_map,
		preferred_symbols=free_variables,
	)
	graph = {
		"accepting": {
			str(state).strip()
			for state in (dfa_result.get("accepting_states") or ())
			if str(state).strip()
		},
		"edges": [
			(
				str(record.get("source_state") or "").strip(),
				str(record.get("target_state") or "").strip(),
				str(record.get("raw_label") or "").strip(),
			)
			for record in guarded_transitions
		],
		"init_state": str(dfa_result.get("initial_state") or "").strip() or None,
	}
	state_aliases = _build_state_aliases(graph, graph.get("edges", ()))
	accepting_states = sorted(
		state_aliases[state]
		for state in graph.get("accepting", set())
		if state in state_aliases
	)
	initial_state = graph.get("init_state")
	initial_alias = state_aliases.get(initial_state) if initial_state else None

	transition_specs: List[Dict[str, Any]] = []
	seen_specs: set[Tuple[str, str, str, bool]] = set()
	for record in guarded_transitions:
		source = str(record.get("source_state") or "").strip()
		target = str(record.get("target_state") or "").strip()
		if source not in state_aliases or target not in state_aliases:
			continue
		label = str(record.get("raw_label") or "").strip() or "true"
		guards = tuple(
			str(guard).strip()
			for guard in (record.get("guards") or ())
			if str(guard).strip()
		)
		if not guards:
			continue
		candidate_symbols, epsilon_feasible = _materialize_transition_choices_from_guards(
			guards=guards,
			free_variables=free_variables or alphabet_symbols,
			alphabet_symbols=alphabet_symbols,
			allow_epsilon=source != target,
		)
		_append_materialized_transition_specs(
			transition_specs=transition_specs,
			seen_specs=seen_specs,
			source=source,
			target=target,
			raw_label=label,
			candidate_symbols=candidate_symbols,
			epsilon_feasible=epsilon_feasible,
			state_aliases=state_aliases,
			initial_alias=initial_alias,
			accepting_states=accepting_states,
			grounding_map=grounding_map,
		)
	return _finalize_transition_specs(transition_specs)


def _build_transition_specs_from_dot_graph(
	*,
	graph: Dict[str, Any],
	dfa_result: Dict[str, Any],
	grounding_map: GroundingMap | None,
) -> List[Dict[str, Any]]:
	expanded_edges = _expand_disjunctive_edges(graph.get("edges", ()))
	state_aliases = _build_state_aliases(graph, expanded_edges)
	accepting_states = sorted(
		state_aliases[state]
		for state in graph.get("accepting", set())
		if state in state_aliases
	)
	initial_state = graph.get("init_state")
	initial_alias = state_aliases.get(initial_state) if initial_state else None
	alphabet_symbols = _ordered_runtime_symbols(dfa_result, grounding_map)

	transition_specs: List[Dict[str, Any]] = []
	seen_specs: set[Tuple[str, str, str, bool]] = set()
	for source, target, raw_label in expanded_edges:
		label = str(raw_label or "").strip()
		if not label or label == "false":
			continue
		candidate_symbols, epsilon_feasible = _materialize_transition_choices_from_label(
			label=label,
			alphabet_symbols=alphabet_symbols,
			allow_epsilon=str(source) != str(target),
		)
		_append_materialized_transition_specs(
			transition_specs=transition_specs,
			seen_specs=seen_specs,
			source=str(source),
			target=str(target),
			raw_label=label,
			candidate_symbols=candidate_symbols,
			epsilon_feasible=epsilon_feasible,
			state_aliases=state_aliases,
			initial_alias=initial_alias,
			accepting_states=accepting_states,
			grounding_map=grounding_map,
		)
	return _finalize_transition_specs(transition_specs)


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


def _normalise_guarded_transitions(
	records: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	normalised: List[Dict[str, Any]] = []
	for record in records:
		if not isinstance(record, dict):
			continue
		source_state = str(record.get("source_state") or "").strip()
		target_state = str(record.get("target_state") or "").strip()
		guards = tuple(
			str(guard).strip()
			for guard in (record.get("guards") or ())
			if str(guard).strip()
		)
		if not source_state or not target_state or not guards:
			continue
		normalised.append(
			{
				"source_state": source_state,
				"target_state": target_state,
				"guards": guards,
				"raw_label": str(record.get("raw_label") or "").strip(),
			},
		)
	return normalised


def _ordered_runtime_symbols(
	dfa_result: Dict[str, Any],
	grounding_map: GroundingMap | None,
	*,
	preferred_symbols: Sequence[str] = (),
) -> Tuple[str, ...]:
	ordered_symbols: List[str] = []
	seen_symbols: set[str] = set()

	def add(symbol: str) -> None:
		token = str(symbol or "").strip()
		if not token or token in seen_symbols:
			return
		seen_symbols.add(token)
		ordered_symbols.append(token)

	for symbol in preferred_symbols:
		add(symbol)
	for symbol in dfa_result.get("alphabet") or ():
		add(str(symbol))
	if grounding_map is not None:
		for symbol in grounding_map.atoms.keys():
			add(symbol)
	return tuple(ordered_symbols)


def _materialize_transition_choices_from_guards(
	*,
	guards: Sequence[str],
	free_variables: Sequence[str],
	alphabet_symbols: Sequence[str],
	allow_epsilon: bool,
) -> Tuple[Tuple[str, ...], bool]:
	candidate_symbols: set[str] = set()
	epsilon_feasible = False
	ordered_symbols = tuple(
		symbol
		for symbol in free_variables
		if symbol in set(alphabet_symbols)
	) or tuple(alphabet_symbols)

	for guard in guards:
		for symbol in ordered_symbols:
			if _guard_bits_match_runtime_choice(
				guard=guard,
				free_variables=free_variables,
				selected_symbol=symbol,
			):
				candidate_symbols.add(symbol)
		if allow_epsilon and _guard_bits_match_runtime_choice(
			guard=guard,
			free_variables=free_variables,
			selected_symbol=None,
		):
			epsilon_feasible = True

	return (
		tuple(symbol for symbol in ordered_symbols if symbol in candidate_symbols),
		epsilon_feasible,
	)


def _guard_bits_match_runtime_choice(
	*,
	guard: str,
	free_variables: Sequence[str],
	selected_symbol: Optional[str],
) -> bool:
	for index, value in enumerate(str(guard).strip()):
		if value == "X":
			continue
		if index >= len(free_variables):
			return False
		symbol = str(free_variables[index]).strip()
		assigned = selected_symbol == symbol
		if value == "1" and not assigned:
			return False
		if value == "0" and assigned:
			return False
	return True


def _materialize_transition_choices_from_label(
	*,
	label: str,
	alphabet_symbols: Sequence[str],
	allow_epsilon: bool,
) -> Tuple[Tuple[str, ...], bool]:
	if not alphabet_symbols:
		return (), label == "true" and allow_epsilon
	candidate_symbols = tuple(
		symbol
		for symbol in alphabet_symbols
		if _label_matches_runtime_choice(label, alphabet_symbols, selected_symbol=symbol)
	)
	epsilon_feasible = (
		allow_epsilon
		and _label_matches_runtime_choice(label, alphabet_symbols, selected_symbol=None)
	)
	return candidate_symbols, epsilon_feasible


def _append_materialized_transition_specs(
	*,
	transition_specs: List[Dict[str, Any]],
	seen_specs: set[Tuple[str, str, str, bool]],
	source: str,
	target: str,
	raw_label: str,
	candidate_symbols: Sequence[str],
	epsilon_feasible: bool,
	state_aliases: Dict[str, str],
	initial_alias: Optional[str],
	accepting_states: Sequence[str],
	grounding_map: GroundingMap | None,
) -> None:
	for symbol in candidate_symbols:
		grounded_atom = grounding_map.get_atom(symbol) if grounding_map is not None else None
		if grounding_map is not None and grounded_atom is None:
			continue
		task_name = str(getattr(grounded_atom, "predicate", "") or "").strip()
		task_args = tuple(
			str(arg).strip()
			for arg in getattr(grounded_atom, "args", ())
			if str(arg).strip()
		)
		spec_key = (source, target, symbol, False)
		if spec_key in seen_specs:
			continue
		seen_specs.add(spec_key)
		transition_specs.append(
			{
				"source_state": state_aliases[source],
				"target_state": state_aliases[target],
				"raw_source_state": source,
				"raw_target_state": target,
				"raw_label": raw_label,
				"positive_task_event_symbols": [symbol],
				"negative_task_event_symbols": [],
				"task_event_symbol": symbol,
				"task_name": task_name,
				"task_args": list(task_args),
				"is_epsilon_transition": False,
				"initial_state": initial_alias,
				"accepting_states": list(accepting_states),
			},
		)

	if not epsilon_feasible:
		return
	spec_key = (source, target, "", True)
	if spec_key in seen_specs:
		return
	seen_specs.add(spec_key)
	transition_specs.append(
		{
			"source_state": state_aliases[source],
			"target_state": state_aliases[target],
			"raw_source_state": source,
			"raw_target_state": target,
			"raw_label": raw_label,
			"positive_task_event_symbols": [],
			"negative_task_event_symbols": [],
			"task_event_symbol": None,
			"task_name": "",
			"task_args": [],
			"is_epsilon_transition": True,
			"initial_state": initial_alias,
			"accepting_states": list(accepting_states),
		},
	)


def _finalize_transition_specs(transition_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	transition_specs.sort(key=_transition_spec_sort_key)
	for index, spec in enumerate(transition_specs, start=1):
		spec["transition_name"] = f"dfa_t{index}"
	return transition_specs


def _label_matches_runtime_choice(
	label: str,
	alphabet_symbols: Sequence[str],
	*,
	selected_symbol: Optional[str],
) -> bool:
	tokens = _tokenise_raw_label(label)
	expression, next_index = _parse_label_expression(tokens, 0)
	if next_index != len(tokens):
		raise ValueError(f"Unsupported raw DFA label: {label!r}")

	valuation = {
		symbol: symbol == selected_symbol
		for symbol in alphabet_symbols
	}
	return bool(_evaluate_label_expression(expression, valuation))


def _parse_label_expression(
	tokens: Sequence[str],
	index: int,
) -> Tuple[Any, int]:
	return _parse_label_or(tokens, index)


def _parse_label_or(tokens: Sequence[str], index: int) -> Tuple[Any, int]:
	left, index = _parse_label_and(tokens, index)
	while index < len(tokens) and tokens[index] == "|":
		right, index = _parse_label_and(tokens, index + 1)
		left = ("or", left, right)
	return left, index


def _parse_label_and(tokens: Sequence[str], index: int) -> Tuple[Any, int]:
	left, index = _parse_label_not(tokens, index)
	while index < len(tokens) and tokens[index] == "&":
		right, index = _parse_label_not(tokens, index + 1)
		left = ("and", left, right)
	return left, index


def _parse_label_not(tokens: Sequence[str], index: int) -> Tuple[Any, int]:
	if index >= len(tokens):
		raise ValueError("Unexpected end of raw DFA label.")
	token = tokens[index]
	if token in {"~", "!"}:
		operand, next_index = _parse_label_not(tokens, index + 1)
		return ("not", operand), next_index
	if token == "(":
		expression, next_index = _parse_label_expression(tokens, index + 1)
		if next_index >= len(tokens) or tokens[next_index] != ")":
			raise ValueError("Unbalanced parentheses in raw DFA label.")
		return expression, next_index + 1
	if token in {"true", "false"} or re.fullmatch(r"[A-Za-z0-9_]+", token):
		return token, index + 1
	raise ValueError(f"Unsupported token in raw DFA label: {token!r}")


def _evaluate_label_expression(expression: Any, valuation: Dict[str, bool]) -> bool:
	if isinstance(expression, tuple):
		operator = expression[0]
		if operator == "not":
			return not _evaluate_label_expression(expression[1], valuation)
		if operator == "and":
			return _evaluate_label_expression(expression[1], valuation) and _evaluate_label_expression(
				expression[2],
				valuation,
			)
		if operator == "or":
			return _evaluate_label_expression(expression[1], valuation) or _evaluate_label_expression(
				expression[2],
				valuation,
			)
		raise ValueError(f"Unsupported raw label operator: {operator!r}")
	token = str(expression).strip()
	if token == "true":
		return True
	if token == "false":
		return False
	return bool(valuation.get(token, False))


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
	pattern = re.compile(r"\s*([()&|~!]|[A-Za-z0-9_]+)\s*")
	tokens = [
		match.group(1)
		for match in pattern.finditer(str(raw_label or ""))
		if match.group(1)
	]
	if not tokens:
		raise ValueError(f"Unsupported raw DFA label: {raw_label!r}")
	return tokens
