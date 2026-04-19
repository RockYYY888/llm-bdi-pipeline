"""Response salvage and abstract syntax tree compilation for method synthesis."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


class MethodSynthesisAstCompilationMixin:
	@staticmethod
	def _apply_ast_compiler_defaults(
		payload: Dict[str, Any],
		*,
		ast_compiler_defaults: Dict[str, Any],
	) -> Dict[str, Any]:
		if "tasks" not in payload or "compound_tasks" in payload or "methods" in payload:
			return payload
		tasks_payload = payload.get("tasks", [])
		if not isinstance(tasks_payload, list):
			return payload
		task_defaults = dict(ast_compiler_defaults.get("task_defaults") or {})
		default_target_task_bindings = list(
			ast_compiler_defaults.get("target_task_bindings") or (),
		)
		normalised_payload = dict(payload)
		if default_target_task_bindings:
			normalised_payload["target_task_bindings"] = default_target_task_bindings
		elif normalised_payload.get("target_task_bindings") in (None, []):
			normalised_payload["target_task_bindings"] = default_target_task_bindings
		if ast_compiler_defaults.get("primitive_aliases"):
			normalised_payload["primitive_aliases"] = list(
				ast_compiler_defaults.get("primitive_aliases") or (),
			)
		if ast_compiler_defaults.get("call_arities"):
			normalised_payload["call_arities"] = dict(
				ast_compiler_defaults.get("call_arities") or {},
			)
		strict_hddl_ast = bool(ast_compiler_defaults.get("strict_hddl_ast"))
		normalised_tasks: List[Any] = []
		for task_entry in tasks_payload:
			if not isinstance(task_entry, dict):
				normalised_tasks.append(task_entry)
				continue
			task_name = str(task_entry.get("name", "")).strip()
			default_entry = task_defaults.get(task_name)
			if not default_entry:
				normalised_tasks.append(task_entry)
				continue
			raw_task_parameters = [
				str(value).strip()
				for value in (task_entry.get("parameters") or ())
				if str(value).strip()
			]
			merged_entry = dict(default_entry)
			if strict_hddl_ast:
				merged_entry.update(task_entry)
				merged_entry = MethodSynthesisAstCompilationMixin._normalise_query_root_bridge_layout(
					merged_entry,
					raw_task_parameters=tuple(raw_task_parameters),
					default_task_parameters=tuple(default_entry.get("parameters") or ()),
				)
				merged_entry = MethodSynthesisAstCompilationMixin._migrate_ast_task_precondition_shorthand(
					merged_entry,
				)
				merged_entry = MethodSynthesisAstCompilationMixin._migrate_ast_branch_parameter_shorthand(
					merged_entry,
					raw_task_parameters=raw_task_parameters,
					default_task_parameters=tuple(default_entry.get("parameters") or ()),
				)
			else:
				task_entry = MethodSynthesisAstCompilationMixin._migrate_ast_ordered_subtasks_branch_array_shorthand(
					task_entry,
				)
				merged_entry.update(task_entry)
				merged_entry = MethodSynthesisAstCompilationMixin._migrate_ast_branch_parameter_shorthand(
					merged_entry,
					raw_task_parameters=raw_task_parameters,
					default_task_parameters=tuple(default_entry.get("parameters") or ()),
				)
				merged_entry = MethodSynthesisAstCompilationMixin._migrate_ast_branch_field_shorthand(
					merged_entry,
				)
			for fixed_key in ("name", "parameters", "parameter_types", "headline", "source_name"):
				if fixed_key in default_entry:
					merged_entry[fixed_key] = default_entry[fixed_key]
			normalised_tasks.append(merged_entry)
		normalised_payload["tasks"] = normalised_tasks
		return normalised_payload

	@staticmethod
	def _normalise_query_root_bridge_layout(
		task_entry: Dict[str, Any],
		*,
		raw_task_parameters: Sequence[str],
		default_task_parameters: Sequence[str],
	) -> Dict[str, Any]:
		"""
		Keep strict AST inputs in a branch-oriented layout.

		The older query-conditioned path used an internal root-bridge wrapper, but
		the current semantic pipeline already models executable structure directly
		through constructive branches. For strict AST defaults, the only safe
		normalisation we need here is to preserve an existing branch layout and to
		leave task-level shorthand for the later migration helpers.
		"""
		_ = raw_task_parameters
		_ = default_task_parameters
		return dict(task_entry)

	@staticmethod
	def _migrate_ast_ordered_subtasks_branch_array_shorthand(
		task_entry: Dict[str, Any],
	) -> Dict[str, Any]:
		if task_entry.get("constructive") not in (None, [], {}):
			return task_entry
		raw_ordered_subtasks = task_entry.get("ordered_subtasks")
		if not isinstance(raw_ordered_subtasks, (list, tuple)) or not raw_ordered_subtasks:
			return task_entry
		if not all(isinstance(item, dict) for item in raw_ordered_subtasks):
			return task_entry
		branch_keys = {
			"label",
			"parameters",
			"task_args",
			"precondition",
			"context",
			"ordered_subtasks",
			"ordering",
			"orderings",
			"ordering_edges",
			"subtasks",
			"steps",
			"support_before",
			"producer",
			"produce",
			"followup",
			"followups",
		}
		if not all(any(key in item for key in branch_keys) for item in raw_ordered_subtasks):
			return task_entry
		migrated_entry = dict(task_entry)
		migrated_entry["constructive"] = list(raw_ordered_subtasks)
		migrated_entry.pop("ordered_subtasks", None)
		return migrated_entry

	@staticmethod
	def _migrate_ast_branch_parameter_shorthand(
		task_entry: Dict[str, Any],
		*,
		raw_task_parameters: Sequence[str],
		default_task_parameters: Sequence[str],
	) -> Dict[str, Any]:
		"""
		Preserve branch-local AUX witness parameters before fixed task defaults apply.

		Transition-native contracts intentionally keep task headers equal to the
		canonical headline arity. Some providers still place extra branch witness
		parameters on the task object instead of on the constructive branch. When
		that happens, migrate the longer parameter list onto any constructive
		branch that omitted an explicit parameters field.
		"""
		normalised_raw_parameters = tuple(
			str(value).strip()
			for value in raw_task_parameters
			if str(value).strip()
		)
		normalised_default_parameters = tuple(
			str(value).strip()
			for value in default_task_parameters
			if str(value).strip()
		)
		if (
			not normalised_raw_parameters
			or normalised_raw_parameters == normalised_default_parameters
		):
			return task_entry

		migrated_entry = dict(task_entry)
		constructive_payload = task_entry.get("constructive")
		if constructive_payload not in (None, [], {}):
			migrated_entry["constructive"] = (
				MethodSynthesisAstCompilationMixin._inject_missing_ast_branch_parameters(
					constructive_payload,
					branch_parameters=normalised_raw_parameters,
				)
			)
			return migrated_entry

		branch_payload = MethodSynthesisAstCompilationMixin._extract_ast_task_level_branch_shorthand(
			task_entry,
		)
		if branch_payload is None:
			return migrated_entry

		migrated_branch = dict(branch_payload)
		migrated_branch.setdefault("parameters", list(normalised_raw_parameters))
		migrated_entry["constructive"] = [migrated_branch]
		for key in (
			"label",
			"task_args",
			"precondition",
			"context",
			"ordered_subtasks",
			"ordering",
			"orderings",
			"ordering_edges",
			"subtasks",
			"steps",
			"support_before",
			"producer",
			"produce",
			"followup",
			"followups",
		):
			migrated_entry.pop(key, None)
		return migrated_entry

	@staticmethod
	def _migrate_ast_task_precondition_shorthand(
		task_entry: Dict[str, Any],
	) -> Dict[str, Any]:
		"""
		Promote shared task-level preconditions onto constructive branches.

		Strict transition-native AST allows a task object to carry a shared
		precondition, but downstream method matching is branch-based. When a
		provider places the contract precondition on the task object instead of
		inside each constructive branch, inherit that context into every branch
		that omitted its own precondition.
		"""
		constructive_payload = task_entry.get("constructive")
		if constructive_payload in (None, [], {}):
			return task_entry

		task_precondition = task_entry.get("precondition")
		if task_precondition in (None, [], {}):
			return task_entry

		migrated_entry = dict(task_entry)
		migrated_entry["constructive"] = (
			MethodSynthesisAstCompilationMixin._inject_missing_ast_branch_fields(
				constructive_payload,
				inherited_fields={"precondition": task_precondition},
			)
		)
		migrated_entry.pop("precondition", None)
		return migrated_entry

	@staticmethod
	def _migrate_ast_branch_field_shorthand(
		task_entry: Dict[str, Any],
	) -> Dict[str, Any]:
		"""
		Promote task-level branch fields onto constructive branches when present.

		Some providers keep branch-scoped metadata such as precondition or
		support_before on the task object even when they already emit a
		constructive branch array. That shape is semantically unambiguous for a
		single task definition, so normalise it before validation rather than
		rejecting a mechanically recoverable AST variant.
		"""
		constructive_payload = task_entry.get("constructive")
		if constructive_payload in (None, [], {}):
			return task_entry

		inherited_fields = {
			key: task_entry[key]
			for key in (
				"label",
				"task_args",
				"precondition",
				"context",
				"ordered_subtasks",
				"ordering",
				"orderings",
				"ordering_edges",
				"subtasks",
				"steps",
				"support_before",
				"producer",
				"produce",
				"followup",
				"followups",
			)
			if key in task_entry
		}
		if not inherited_fields:
			return task_entry

		migrated_entry = dict(task_entry)
		migrated_entry["constructive"] = (
			MethodSynthesisAstCompilationMixin._inject_missing_ast_branch_fields(
				constructive_payload,
				inherited_fields=inherited_fields,
			)
		)
		for key in inherited_fields:
			migrated_entry.pop(key, None)
		return migrated_entry

	@staticmethod
	def _inject_missing_ast_branch_parameters(
		constructive_payload: Any,
		*,
		branch_parameters: Sequence[str],
	) -> Any:
		if isinstance(constructive_payload, list):
			return [
				MethodSynthesisAstCompilationMixin._inject_missing_ast_branch_parameters(
					item,
					branch_parameters=branch_parameters,
				)
				for item in constructive_payload
			]
		if isinstance(constructive_payload, tuple):
			return tuple(
				MethodSynthesisAstCompilationMixin._inject_missing_ast_branch_parameters(
					item,
					branch_parameters=branch_parameters,
				)
				for item in constructive_payload
			)
		if not isinstance(constructive_payload, dict):
			return constructive_payload

		for wrapper_key in ("branch", "branches"):
			if wrapper_key in constructive_payload:
				migrated_wrapper = dict(constructive_payload)
				migrated_wrapper[wrapper_key] = (
					MethodSynthesisAstCompilationMixin._inject_missing_ast_branch_parameters(
						constructive_payload.get(wrapper_key),
						branch_parameters=branch_parameters,
					)
				)
				return migrated_wrapper

		if "parameters" in constructive_payload:
			return constructive_payload

		migrated_branch = dict(constructive_payload)
		migrated_branch["parameters"] = list(branch_parameters)
		return migrated_branch

	@staticmethod
	def _inject_missing_ast_branch_fields(
		constructive_payload: Any,
		*,
		inherited_fields: Dict[str, Any],
	) -> Any:
		if isinstance(constructive_payload, list):
			return [
				MethodSynthesisAstCompilationMixin._inject_missing_ast_branch_fields(
					item,
					inherited_fields=inherited_fields,
				)
				for item in constructive_payload
			]
		if isinstance(constructive_payload, tuple):
			return tuple(
				MethodSynthesisAstCompilationMixin._inject_missing_ast_branch_fields(
					item,
					inherited_fields=inherited_fields,
				)
				for item in constructive_payload
			)
		if not isinstance(constructive_payload, dict):
			return constructive_payload

		for wrapper_key in ("branch", "branches"):
			if wrapper_key in constructive_payload:
				migrated_wrapper = dict(constructive_payload)
				migrated_wrapper[wrapper_key] = (
					MethodSynthesisAstCompilationMixin._inject_missing_ast_branch_fields(
						constructive_payload.get(wrapper_key),
						inherited_fields=inherited_fields,
					)
				)
				return migrated_wrapper

		migrated_branch = dict(constructive_payload)
		for key, value in inherited_fields.items():
			migrated_branch.setdefault(key, value)
		return migrated_branch

	@staticmethod
	def _extract_ast_task_level_branch_shorthand(
		task_entry: Dict[str, Any],
	) -> Dict[str, Any] | None:
		branch_keys = (
			"label",
			"task_args",
			"precondition",
			"context",
			"ordered_subtasks",
			"ordering",
			"orderings",
			"ordering_edges",
			"subtasks",
			"steps",
			"support_before",
			"producer",
			"produce",
			"followup",
			"followups",
		)
		if task_entry.get("constructive") not in (None, [], {}):
			return None
		if not any(key in task_entry for key in branch_keys):
			return None
		return {
			key: task_entry[key]
			for key in branch_keys
			if key in task_entry
		}

	@staticmethod
	def _appears_truncated_json(text: str) -> bool:
		open_curly = text.count("{")
		close_curly = text.count("}")
		open_square = text.count("[")
		close_square = text.count("]")
		if open_curly > close_curly:
			return True
		if open_square > close_square:
			return True
		return False

	@staticmethod
	def _salvage_missing_object_closer_at_tail(
		result_text: str,
	) -> dict | list | None:
		if result_text.count("[") != result_text.count("]"):
			return None
		if result_text.count("{") != result_text.count("}") + 1:
			return None
		candidate_indexes = [
			index
			for index, character in enumerate(result_text)
			if character == "]"
		]
		for insert_index in reversed(candidate_indexes):
			candidate = result_text[:insert_index] + "}" + result_text[insert_index:]
			try:
				return json.loads(candidate)
			except json.JSONDecodeError:
				continue
		return None

	@staticmethod
	def _extract_json_object_candidate(result_text: str) -> str | None:
		start_index = result_text.find("{")
		end_index = result_text.rfind("}")
		if start_index == -1 or end_index == -1 or end_index <= start_index:
			return None
		candidate = result_text[start_index:end_index + 1].strip()
		return candidate or None

	@staticmethod
	def _salvage_missing_object_closer(
		result_text: str,
		error: json.JSONDecodeError,
	) -> dict | list | None:
		position = int(getattr(error, "pos", -1))
		if position < 0 or position >= len(result_text):
			return None
		if result_text[position] != "]":
			return None
		if result_text.count("[") != result_text.count("]"):
			return None
		if result_text.count("{") != result_text.count("}") + 1:
			return None
		candidate = result_text[:position] + "}" + result_text[position:]
		try:
			return json.loads(candidate)
		except json.JSONDecodeError:
			return None

	@staticmethod
	def _decode_leading_json_object(result_text: str) -> dict | None:
		stripped = result_text.lstrip()
		if not stripped.startswith("{"):
			return None
		try:
			decoder = json.JSONDecoder()
			parsed, _ = decoder.raw_decode(stripped)
		except json.JSONDecodeError:
			return None
		return parsed if isinstance(parsed, dict) else None

	@classmethod
	def _salvage_ast_payload(cls, result_text: str) -> dict | None:
		target_task_bindings_array = cls._extract_balanced_array_for_key(
			result_text,
			"target_task_bindings",
		)
		if target_task_bindings_array is None:
			return None
		try:
			target_task_bindings = json.loads(target_task_bindings_array)
		except json.JSONDecodeError:
			return None
		if not isinstance(target_task_bindings, list):
			return None

		task_object_texts = cls._extract_named_task_object_fragments(result_text)
		if not task_object_texts:
			return None

		tasks: List[Dict[str, Any]] = []
		for task_text in task_object_texts:
			try:
				task_payload = json.loads(task_text)
			except json.JSONDecodeError:
				continue
			if not isinstance(task_payload, dict):
				continue
			if not str(task_payload.get("name", "")).strip():
				continue
			tasks.append(task_payload)

		if not tasks:
			return None
		return {
			"target_task_bindings": target_task_bindings,
			"tasks": tasks,
		}

	@classmethod
	def _extract_balanced_array_for_key(
		cls,
		result_text: str,
		key: str,
	) -> str | None:
		match = re.search(rf'"{re.escape(key)}"\s*:', result_text)
		if match is None:
			return None
		index = match.end()
		while index < len(result_text) and result_text[index].isspace():
			index += 1
		if index >= len(result_text) or result_text[index] != "[":
			return None
		end_index = cls._find_matching_delimiter(
			result_text,
			start_index=index,
			open_char="[",
			close_char="]",
		)
		if end_index is None:
			return None
		return result_text[index:end_index + 1]

	@classmethod
	def _extract_named_task_object_fragments(
		cls,
		result_text: str,
	) -> List[str]:
		tasks_match = re.search(r'"tasks"\s*:', result_text)
		if tasks_match is None:
			return []
		search_index = tasks_match.end()
		fragments: List[str] = []
		while True:
			match = re.search(r'\{\s*"name"\s*:', result_text[search_index:])
			if match is None:
				break
			start_index = search_index + match.start()
			end_index = cls._find_matching_delimiter(
				result_text,
				start_index=start_index,
				open_char="{",
				close_char="}",
			)
			if end_index is None:
				break
			fragments.append(result_text[start_index:end_index + 1])
			search_index = end_index + 1
		return fragments

	@staticmethod
	def _find_matching_delimiter(
		result_text: str,
		*,
		start_index: int,
		open_char: str,
		close_char: str,
	) -> int | None:
		depth = 0
		in_string = False
		escape_next = False
		for index in range(start_index, len(result_text)):
			character = result_text[index]
			if in_string:
				if escape_next:
					escape_next = False
				elif character == "\\":
					escape_next = True
				elif character == '"':
					in_string = False
				continue
			if character == '"':
				in_string = True
				continue
			if character == open_char:
				depth += 1
				continue
			if character == close_char:
				depth -= 1
				if depth == 0:
					return index
		return None
