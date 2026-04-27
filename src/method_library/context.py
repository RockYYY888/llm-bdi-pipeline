"""
Context for method-library synthesis.

This module intentionally contains only the state and helpers required by the
domain-only generation path.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from method_library.synthesis.naming import sanitize_identifier
from method_library.synthesis.schema import HTNLiteral
from execution_logging.execution_logger import ExecutionLogger
from utils.config import Config, get_config
from utils.hddl_parser import HDDLParser


class TypeResolutionError(RuntimeError):
	"""Raised when method-library type inference is ambiguous or inconsistent."""


class MethodLibrarySynthesisContext:
	"""Minimal context object required by method-library synthesis."""

	def __init__(self, *, domain_file: str) -> None:
		if not domain_file:
			raise ValueError("domain_file is required for method-library synthesis.")

		self.config: Config = get_config()
		self.project_root = Path(__file__).resolve().parents[2]
		self.logger = ExecutionLogger(logs_dir=str(self.project_root / "artifacts" / "runs"))
		self.domain_file = str(domain_file)
		self.domain = HDDLParser.parse_domain(self.domain_file)
		self.output_dir: Optional[Path] = None
		self.type_parent_map = self._build_type_parent_map()
		self.domain_type_names = set(self.type_parent_map.keys())
		self.predicate_type_map = self._predicate_type_map()
		self.action_type_map = self._action_type_map()
		self.task_type_map = self._task_type_map()
		self._subtype_check_cache: Dict[Tuple[str, str], bool] = {}
		self._resolved_symbol_type_cache: Dict[frozenset[str], str] = {}
		self._dynamic_task_signature_cache: Dict[Tuple[int, str], Tuple[str, ...]] = {}
		self._method_variable_type_hint_cache: Dict[Tuple[int, int], Dict[str, str]] = {}

	def _record_step_timing(
		self,
		stage_name: str,
		stage_start: float,
		*,
		breakdown: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self.logger.record_step_timing(
			stage_name,
			time.perf_counter() - stage_start,
			breakdown=breakdown,
			metadata=metadata,
		)

	@staticmethod
	def _emit_domain_gate_progress(message: str) -> None:
		if not str(os.getenv("DOMAIN_GATE_PROGRESS", "")).strip():
			return
		sys.stderr.write(f"[DOMAIN GATE PROGRESS] {message}\n")
		sys.stderr.flush()

	@staticmethod
	def _sanitize_name(name: str) -> str:
		return sanitize_identifier(name)

	@staticmethod
	def _is_variable_symbol(symbol: str) -> bool:
		return bool(symbol) and str(symbol)[0].isupper()

	@staticmethod
	def _parameter_type(parameter: str) -> str:
		text = str(parameter or "").strip()
		if ":" in text:
			type_name = text.split(":", 1)[1].strip()
			return type_name or "object"
		if "-" in text:
			type_name = text.split("-", 1)[1].strip()
			return type_name or "object"
		return "object"

	@staticmethod
	def _merge_type_candidates(
		target: Dict[str, Set[str]],
		incoming: Dict[str, Set[str]],
	) -> None:
		for symbol, type_names in incoming.items():
			if not symbol:
				continue
			target.setdefault(symbol, set()).update(item for item in type_names if item)

	@staticmethod
	def _add_type_candidate(
		candidates: Dict[str, Set[str]],
		symbol: str,
		type_name: Optional[str],
	) -> None:
		if not symbol or not type_name:
			return
		candidates.setdefault(symbol, set()).add(type_name)

	def _require_known_type(self, type_name: str, source: str) -> str:
		if type_name in self.domain_type_names:
			return type_name
		raise TypeResolutionError(
			f"{source} references unknown type '{type_name}'. "
			f"Known types: {sorted(self.domain_type_names)}",
		)

	def _build_type_parent_map(self) -> Dict[str, Optional[str]]:
		tokens = [
			token.strip()
			for token in (getattr(self.domain, "types", []) or [])
			if token and token.strip()
		]
		if not tokens:
			return {"object": None}

		parent_map: Dict[str, Optional[str]] = {}
		pending_children: list[str] = []
		index = 0
		while index < len(tokens):
			token = tokens[index]
			if token == "-":
				if not pending_children or index + 1 >= len(tokens):
					raise ValueError("Malformed HDDL :types declaration.")
				parent_type = tokens[index + 1]
				for child_type in pending_children:
					previous = parent_map.get(child_type)
					if previous is not None and previous != parent_type:
						raise ValueError(
							f"Type '{child_type}' has conflicting parents "
							f"('{previous}' vs '{parent_type}').",
						)
					parent_map[child_type] = parent_type
				pending_children = []
				index += 2
				continue
			pending_children.append(token)
			index += 1

		for child_type in pending_children:
			parent_map.setdefault(child_type, "object")
		parent_map["object"] = None

		changed = True
		while changed:
			changed = False
			for parent_type in list(parent_map.values()):
				if parent_type is None or parent_type in parent_map:
					continue
				parent_map[parent_type] = "object" if parent_type != "object" else None
				changed = True

		for type_name in list(parent_map.keys()):
			if type_name == "object":
				parent_map[type_name] = None
				continue
			if parent_map[type_name] == type_name:
				raise ValueError(f"Type '{type_name}' cannot inherit from itself.")
			seen = {type_name}
			cursor = parent_map[type_name]
			while cursor is not None:
				if cursor in seen:
					raise ValueError(f"Cyclic type hierarchy detected at '{type_name}'.")
				seen.add(cursor)
				cursor = parent_map.get(cursor)
		return parent_map

	def _predicate_type_map(self) -> Dict[str, Tuple[str, ...]]:
		return {
			predicate.name: tuple(
				self._require_known_type(
					self._parameter_type(parameter),
					f"Predicate '{predicate.name}'",
				)
				for parameter in predicate.parameters
			)
			for predicate in getattr(self.domain, "predicates", [])
		}

	def _action_type_map(self) -> Dict[str, Tuple[str, ...]]:
		mapping: Dict[str, Tuple[str, ...]] = {}
		for action in getattr(self.domain, "actions", []):
			type_signature = tuple(
				self._require_known_type(
					self._parameter_type(parameter),
					f"Action '{action.name}'",
				)
				for parameter in action.parameters
			)
			mapping[action.name] = type_signature
			mapping[self._sanitize_name(action.name)] = type_signature
		return mapping

	def _task_type_map(self) -> Dict[str, Tuple[str, ...]]:
		mapping: Dict[str, Tuple[str, ...]] = {}
		for task in getattr(self.domain, "tasks", []):
			type_signature = tuple(
				self._require_known_type(
					self._parameter_type(parameter),
					f"Task '{task.name}'",
				)
				for parameter in task.parameters
			)
			mapping[task.name] = type_signature
			mapping[self._sanitize_name(task.name)] = type_signature
		return mapping

	def _is_subtype(self, candidate_type: str, expected_type: str) -> bool:
		cache_key = (candidate_type, expected_type)
		if cache_key in self._subtype_check_cache:
			return self._subtype_check_cache[cache_key]

		if candidate_type == expected_type:
			self._subtype_check_cache[cache_key] = True
			return True
		if candidate_type not in self.type_parent_map or expected_type not in self.type_parent_map:
			self._subtype_check_cache[cache_key] = False
			return False

		cursor = self.type_parent_map.get(candidate_type)
		visited = {candidate_type}
		while cursor is not None and cursor not in visited:
			if cursor == expected_type:
				self._subtype_check_cache[cache_key] = True
				return True
			visited.add(cursor)
			cursor = self.type_parent_map.get(cursor)
		self._subtype_check_cache[cache_key] = False
		return False

	def _resolve_symbol_type(
		self,
		*,
		symbol: str,
		candidate_types: Set[str],
		scope: str,
	) -> str:
		if not candidate_types:
			raise TypeResolutionError(f"{scope}: symbol '{symbol}' has no type evidence.")

		unknown_types = sorted(
			type_name
			for type_name in candidate_types
			if type_name not in self.domain_type_names
		)
		if unknown_types:
			raise TypeResolutionError(
				f"{scope}: symbol '{symbol}' references unknown types {unknown_types}.",
			)

		candidate_key = frozenset(candidate_types)
		if candidate_key in self._resolved_symbol_type_cache:
			return self._resolved_symbol_type_cache[candidate_key]

		feasible = sorted(
			type_name
			for type_name in self.domain_type_names
			if all(self._is_subtype(type_name, required) for required in candidate_types)
		)
		if not feasible:
			raise TypeResolutionError(
				f"{scope}: symbol '{symbol}' has conflicting type constraints "
				f"{sorted(candidate_types)}.",
			)

		most_general = sorted(
			type_name
			for type_name in feasible
			if not any(
				other != type_name and self._is_subtype(type_name, other)
				for other in feasible
			)
		)
		if len(most_general) != 1:
			raise TypeResolutionError(
				f"{scope}: symbol '{symbol}' is ambiguous under constraints "
				f"{sorted(candidate_types)}; candidate schema types={most_general}.",
			)
		self._resolved_symbol_type_cache[candidate_key] = most_general[0]
		return most_general[0]

	def _task_type_signature(self, task_name: str, method_library=None) -> Tuple[str, ...]:
		signature = self.task_type_map.get(task_name)
		if signature is not None:
			return signature
		sanitized_signature = self.task_type_map.get(self._sanitize_name(task_name))
		if sanitized_signature is not None:
			return sanitized_signature
		if method_library is None:
			return ()

		cache_key = (id(method_library), task_name)
		if cache_key in self._dynamic_task_signature_cache:
			return self._dynamic_task_signature_cache[cache_key]

		task_schema = method_library.task_for_name(task_name)
		if task_schema is None:
			self._dynamic_task_signature_cache[cache_key] = ()
			return ()
		predicate_name = ""
		if len(getattr(task_schema, "source_predicates", ()) or ()) == 1:
			predicate_name = str(task_schema.source_predicates[0]).strip()
		elif getattr(task_schema, "headline_literal", None) is not None:
			predicate_name = str(task_schema.headline_literal.predicate).strip()
		if not predicate_name:
			self._dynamic_task_signature_cache[cache_key] = ()
			return ()
		predicate_signature = self.predicate_type_map.get(predicate_name, ())
		if not predicate_signature:
			self._dynamic_task_signature_cache[cache_key] = ()
			return ()
		if len(predicate_signature) != len(task_schema.parameters):
			raise TypeResolutionError(
				f"Task '{task_name}' source predicate '{predicate_name}' arity mismatch: "
				f"task has {len(task_schema.parameters)} args, predicate has "
				f"{len(predicate_signature)}.",
			)
		self._dynamic_task_signature_cache[cache_key] = predicate_signature
		return predicate_signature

	def _collect_argument_signature_constraints(
		self,
		*,
		candidates: Dict[str, Set[str]],
		args: Sequence[str],
		signature: Sequence[str],
		scope: str,
	) -> None:
		if not signature:
			return
		if len(args) != len(signature):
			raise TypeResolutionError(
				f"{scope}: arity mismatch (args={len(args)}, signature={len(signature)}).",
			)
		for index, arg in enumerate(args):
			self._add_type_candidate(candidates, arg, signature[index])

	def _literal_type_candidates(self, literal: HTNLiteral) -> Dict[str, Set[str]]:
		if literal.is_equality:
			return {}
		predicate_types = self.predicate_type_map.get(literal.predicate)
		if predicate_types is None:
			raise TypeResolutionError(
				f"Unknown predicate '{literal.predicate}' in literal '{literal.to_signature()}'.",
			)
		candidates: Dict[str, Set[str]] = defaultdict(set)
		self._collect_argument_signature_constraints(
			candidates=candidates,
			args=literal.args,
			signature=predicate_types,
			scope=f"Literal '{literal.to_signature()}' typing",
		)
		return candidates

	def _method_task_binding_args(
		self,
		method,
		method_library,
		*,
		signature: Sequence[str] = (),
	) -> Tuple[str, ...]:
		explicit_task_args = tuple(getattr(method, "task_args", ()) or ())
		if explicit_task_args:
			if signature and len(explicit_task_args) != len(signature):
				raise TypeResolutionError(
					f"Method '{method.method_name}' task-argument arity mismatch: "
					f"task_args={len(explicit_task_args)}, signature={len(signature)}.",
				)
			return explicit_task_args

		task_schema = method_library.task_for_name(method.task_name)
		if task_schema is not None and task_schema.parameters:
			task_binding_args: list[str] = []
			for index, task_parameter in enumerate(task_schema.parameters):
				if task_parameter in method.parameters:
					task_binding_args.append(task_parameter)
				elif index < len(method.parameters):
					task_binding_args.append(method.parameters[index])
				else:
					raise TypeResolutionError(
						f"Method '{method.method_name}' is missing parameter mapping for "
						f"task argument '{task_parameter}'.",
					)
			return tuple(task_binding_args)

		if signature:
			return tuple(method.parameters[:len(signature)])
		return tuple(method.parameters)

	def _method_variable_type_hints(
		self,
		method,
		method_library,
	) -> Dict[str, str]:
		cache_key = (id(method_library), id(method))
		if cache_key in self._method_variable_type_hint_cache:
			return dict(self._method_variable_type_hint_cache[cache_key])

		candidates: Dict[str, Set[str]] = defaultdict(set)
		task_signature = self._task_type_signature(method.task_name, method_library)
		task_binding_args = list(
			self._method_task_binding_args(
				method,
				method_library,
				signature=task_signature,
			),
		)
		self._collect_argument_signature_constraints(
			candidates=candidates,
			args=tuple(task_binding_args),
			signature=task_signature,
			scope=f"Method '{method.method_name}' task parameter typing",
		)
		schematic_symbols = set(method.parameters) | set(task_binding_args)

		def collect_literal(literal: Optional[HTNLiteral]) -> None:
			if literal is None or literal.is_equality:
				return
			self._merge_type_candidates(candidates, self._literal_type_candidates(literal))

		for literal in method.context:
			collect_literal(literal)

		for step in method.subtasks:
			collect_literal(step.literal)
			for literal in step.preconditions:
				collect_literal(literal)
			for literal in step.effects:
				collect_literal(literal)
			if step.kind == "compound":
				step_signature = self._task_type_signature(step.task_name, method_library)
				if not step_signature:
					continue
				self._collect_argument_signature_constraints(
					candidates=candidates,
					args=step.args,
					signature=step_signature,
					scope=(
						f"Method '{method.method_name}' compound step "
						f"'{step.step_id}:{step.task_name}' typing"
					),
				)
				continue
			if step.kind != "primitive":
				continue
			action_types = self.action_type_map.get(step.action_name or "")
			if action_types is None:
				action_types = self.action_type_map.get(step.task_name)
			if action_types is None and step.action_name:
				action_types = self.action_type_map.get(self._sanitize_name(step.action_name))
			if action_types is None:
				raise TypeResolutionError(
					f"Method '{method.method_name}' references primitive step "
					f"'{step.step_id}:{step.task_name}' without known action signature.",
				)
			self._collect_argument_signature_constraints(
				candidates=candidates,
				args=step.args,
				signature=action_types,
				scope=(
					f"Method '{method.method_name}' primitive step "
					f"'{step.step_id}:{step.task_name}' typing"
				),
			)

		variable_symbols: Set[str] = set(schematic_symbols)
		for literal in method.context:
			variable_symbols.update(
				arg
				for arg in literal.args
				if arg in schematic_symbols or self._is_variable_symbol(arg)
			)
		for step in method.subtasks:
			variable_symbols.update(
				arg
				for arg in step.args
				if arg in schematic_symbols or self._is_variable_symbol(arg)
			)
			if step.literal:
				variable_symbols.update(
					arg
					for arg in step.literal.args
					if arg in schematic_symbols or self._is_variable_symbol(arg)
				)
			for literal in (*step.preconditions, *step.effects):
				variable_symbols.update(
					arg
					for arg in literal.args
					if arg in schematic_symbols or self._is_variable_symbol(arg)
				)

		resolved = {
			symbol: self._resolve_symbol_type(
				symbol=symbol,
				candidate_types=candidates.get(symbol, set()),
				scope=f"Method-synthesis method '{method.method_name}' variable typing",
			)
			for symbol in sorted(variable_symbols)
		}
		self._method_variable_type_hint_cache[cache_key] = dict(resolved)
		return resolved

	def _validate_method_library_typing(self, method_library) -> None:
		for method in method_library.methods:
			self._method_variable_type_hints(method, method_library)
