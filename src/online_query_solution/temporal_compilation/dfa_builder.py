"""
Deterministic automaton compilation for online temporal queries.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ltlf_to_dfa import LTLfToDFA


class DFABuilder:
	"""Compile one validated LTLf formula into a renderer-facing DFA payload."""

	def __init__(self) -> None:
		self.converter = LTLfToDFA()

	def build(self, grounding_result: Any) -> Dict[str, Any]:
		total_start = time.perf_counter()
		formula_str = self._normalise_formula_string(self._formula_string(grounding_result))
		if not formula_str:
			raise ValueError("Temporal grounding result contains no LTLf formula.")

		symbolic_fragment = self._build_symbolic_fragment_payload(formula_str)
		if symbolic_fragment is not None:
			timing_profile = {
				"convert_seconds": 0.0,
				"total_seconds": time.perf_counter() - total_start,
			}
			return {
				**symbolic_fragment,
				"formula": formula_str,
				"timing_profile": timing_profile,
			}

		convert_start = time.perf_counter()
		dfa_dot, metadata = self.converter.convert(formula_str)
		convert_seconds = time.perf_counter() - convert_start

		return {
			"formula": formula_str,
			"dfa_dot": dfa_dot,
			"dfa_path": "dfa.dot",
			"construction": metadata.get("construction") or "generic_ltlf2dfa",
			"num_states": int(metadata.get("num_states") or self._count_states(dfa_dot)),
			"num_transitions": int(
				metadata.get("num_transitions") or self._count_transitions(dfa_dot)
			),
			"alphabet": list(metadata.get("alphabet") or ()),
			"timing_profile": {
				"convert_seconds": convert_seconds,
				"total_seconds": time.perf_counter() - total_start,
			},
		}

	@staticmethod
	def _formula_string(grounding_result: Any) -> str:
		if isinstance(grounding_result, str):
			return grounding_result.strip()
		if hasattr(grounding_result, "ltlf_formula"):
			return str(getattr(grounding_result, "ltlf_formula") or "").strip()
		if hasattr(grounding_result, "combined_formula_string"):
			return str(grounding_result.combined_formula_string()).strip()
		formulas = list(getattr(grounding_result, "formulas", ()) or ())
		if not formulas:
			return ""
		if len(formulas) == 1:
			return str(formulas[0].to_string()).strip()
		return " & ".join(f"({formula.to_string()})" for formula in formulas)

	@staticmethod
	def _normalise_formula_string(formula_str: str) -> str:
		text = re.sub(r"\s+", " ", str(formula_str or "").strip())
		if not text:
			return ""
		characters: list[str] = []
		open_parentheses = 0
		for character in text:
			if character == "(":
				open_parentheses += 1
				characters.append(character)
				continue
			if character == ")":
				if open_parentheses <= 0:
					continue
				open_parentheses -= 1
				characters.append(character)
				continue
			characters.append(character)
		if open_parentheses > 0:
			characters.extend(")" for _ in range(open_parentheses))
		return "".join(characters).strip()

	@classmethod
	def _build_symbolic_fragment_payload(
		cls,
		formula_str: str,
	) -> Optional[Dict[str, Any]]:
		ordered_subgoal_ids = cls._parse_total_ordered_subgoal_sequence(formula_str)
		if ordered_subgoal_ids:
			return cls._build_ordered_subgoal_sequence_payload(ordered_subgoal_ids)

		unordered_subgoal_ids = cls._parse_unordered_eventual_conjunction(formula_str)
		if unordered_subgoal_ids:
			return {
				"dfa_dot": "",
				"construction": "symbolic_unordered_subgoal_set",
				"num_states": 1,
				"num_transitions": len(unordered_subgoal_ids),
				"alphabet": list(unordered_subgoal_ids),
				"symbolic_subgoal_monitor": {
					"subgoal_indices": [
						int(subgoal_id.split("_", maxsplit=1)[1])
						for subgoal_id in unordered_subgoal_ids
					],
					"initial_state": "q0",
					"accepting_states": ["q0"],
				},
				"ordered_subgoal_sequence": False,
			}
		return None

	@staticmethod
	def _tokenize_formula(formula_str: str) -> List[str]:
		return re.findall(r"subgoal_\d+|F|[()&]", str(formula_str or ""))

	@classmethod
	def _parse_total_ordered_subgoal_sequence(cls, formula_str: str) -> Tuple[str, ...]:
		tokens = cls._tokenize_formula(formula_str)
		if not tokens:
			return ()
		subgoal_ids: List[str] = []
		position = 0
		while position < len(tokens):
			if tokens[position] != "F":
				return ()
			position += 1
			if position >= len(tokens) or tokens[position] != "(":
				return ()
			position += 1
			if position >= len(tokens) or not tokens[position].startswith("subgoal_"):
				return ()
			subgoal_ids.append(tokens[position])
			position += 1
			if position < len(tokens) and tokens[position] == ")":
				position += 1
				break
			if position >= len(tokens) or tokens[position] != "&":
				return ()
			position += 1

		if not subgoal_ids:
			return ()
		remaining_tokens = tokens[position:]
		if remaining_tokens != [")"] * (len(subgoal_ids) - 1):
			return ()
		return tuple(subgoal_ids)

	@classmethod
	def _parse_unordered_eventual_conjunction(cls, formula_str: str) -> Tuple[str, ...]:
		tokens = cls._tokenize_formula(formula_str)
		if not tokens:
			return ()
		ordered_subgoals: List[str] = []
		position = 0
		while position < len(tokens):
			if tokens[position] != "F":
				return ()
			position += 1
			if position >= len(tokens) or tokens[position] != "(":
				return ()
			position += 1
			if position >= len(tokens) or not tokens[position].startswith("subgoal_"):
				return ()
			ordered_subgoals.append(tokens[position])
			position += 1
			if position >= len(tokens) or tokens[position] != ")":
				return ()
			position += 1
			if position == len(tokens):
				break
			if tokens[position] != "&":
				return ()
			position += 1
		return tuple(ordered_subgoals)

	@staticmethod
	def _build_ordered_subgoal_sequence_payload(
		ordered_subgoal_ids: Sequence[str],
	) -> Dict[str, Any]:
		lines = [
			"digraph MONA_DFA {",
			' rankdir = LR;',
			" init [shape=plaintext,label=\"\"];",
		]
		state_count = len(ordered_subgoal_ids) + 1
		for state_index in range(1, state_count + 1):
			if state_index == state_count:
				lines.append(f" {state_index} [shape=doublecircle];")
			else:
				lines.append(f" {state_index} [shape=circle];")
		lines.append(" init -> 1;")
		for state_index, subgoal_id in enumerate(ordered_subgoal_ids, start=1):
			lines.append(f' {state_index} -> {state_index + 1} [label="{subgoal_id}"];')
		lines.append("}")
		return {
			"dfa_dot": "\n".join(lines),
			"construction": "symbolic_ordered_subgoal_sequence",
			"num_states": state_count,
			"num_transitions": len(ordered_subgoal_ids),
			"alphabet": list(ordered_subgoal_ids),
			"ordered_subgoal_sequence": True,
		}

	@staticmethod
	def _count_states(dfa_dot: str) -> int:
		states = set()
		for line in str(dfa_dot or "").splitlines():
			grouped_match = re.search(r"node\\s+\\[.*?];\\s*([^;]+);", line)
			if grouped_match:
				tokens = re.findall(r"[A-Za-z0-9_]+", grouped_match.group(1))
				states.update(token for token in tokens if token != "init")
				continue
			single_match = re.search(r"([A-Za-z0-9_]+)\\s*\\[\\s*shape\\s*=\\s*", line)
			if single_match:
				token = single_match.group(1)
				if token != "init":
					states.add(token)
		return len(states)

	@staticmethod
	def _count_transitions(dfa_dot: str) -> int:
		transition_count = 0
		for line in str(dfa_dot or "").splitlines():
			if "->" not in line:
				continue
			total_edges = line.count("->")
			init_edges = len(re.findall(r"init\\s*->", line))
			transition_count += max(0, total_edges - init_edges)
		return transition_count


def build_dfa_from_ltlf(grounding_result: Any) -> Dict[str, Any]:
	"""Compatibility wrapper for code paths that expect a module-level builder function."""
	return DFABuilder().build(grounding_result)
