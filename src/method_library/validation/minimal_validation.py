"""
Minimal structural validation for domain-complete method synthesis.

This module intentionally avoids speculative semantic repair rules. It keeps only
the checks that protect parser stability, structural well-formedness, and
obviously degenerate decomposition structures.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from method_library.synthesis.schema import HTNLiteral, HTNMethodLibrary
from method_library.synthesis.naming import sanitize_identifier


def validate_domain_complete_coverage(domain: Any, library: HTNMethodLibrary) -> None:
	declared_tasks = {
		_canonical_name(getattr(task, "name", ""))
		for task in getattr(domain, "tasks", [])
		if str(getattr(task, "name", "")).strip()
	}
	library_tasks = {
		_canonical_name(task.name)
		for task in library.compound_tasks
		if str(task.name).strip()
	}
	missing_tasks = sorted(declared_tasks - library_tasks)
	extra_tasks = sorted(library_tasks - declared_tasks)
	if missing_tasks:
		raise ValueError(
			f"Generated library omitted declared compound tasks: {', '.join(missing_tasks)}"
		)
	if extra_tasks:
		raise ValueError(
			f"Generated library introduced undeclared compound tasks: {', '.join(extra_tasks)}"
		)


def validate_minimal_library(
	library: HTNMethodLibrary,
	domain: Any,
) -> None:
	validate_signature_conformance(domain, library)
	validate_typed_structural_soundness(domain, library)
	validate_decomposition_admissibility(domain, library)


def _canonical_name(value: Any) -> str:
	return sanitize_identifier(str(value or "").strip())


def _name_aliases(value: Any) -> Set[str]:
	raw_name = str(value or "").strip()
	if not raw_name:
		return set()
	return {raw_name, _canonical_name(raw_name)}


def _compound_task_lookup(library: HTNMethodLibrary) -> Dict[str, Any]:
	task_lookup: Dict[str, Any] = {}
	canonical_names: Set[str] = set()
	for task in library.compound_tasks:
		raw_name = str(getattr(task, "name", "")).strip()
		if not raw_name:
			continue
		canonical_name = _canonical_name(raw_name)
		if canonical_name in canonical_names:
			raise ValueError(f"Duplicate compound task declaration '{raw_name}'.")
		canonical_names.add(canonical_name)
		for task_name in _name_aliases(raw_name):
			task_lookup[task_name] = task
	return task_lookup


def validate_signature_conformance(domain: Any, library: HTNMethodLibrary) -> None:
	validate_domain_complete_coverage(domain, library)
	task_lookup = _compound_task_lookup(library)
	primitive_names = {
		name
		for action in getattr(domain, "actions", [])
		for name in _name_aliases(getattr(action, "name", ""))
		if str(getattr(action, "name", "")).strip()
	}
	predicate_arities: Dict[str, int] = {}
	for predicate in getattr(domain, "predicates", []):
		if not str(getattr(predicate, "name", "")).strip():
			continue
		arity = len(getattr(predicate, "parameters", ()) or ())
		for predicate_name in _name_aliases(getattr(predicate, "name", "")):
			predicate_arities[predicate_name] = arity
	seen_method_names: set[str] = set()
	for method in library.methods:
		method_name = str(method.method_name).strip()
		if not method_name:
			raise ValueError("Every method must have a non-empty method_name.")
		if method_name in seen_method_names:
			raise ValueError(f"Duplicate method identifier '{method_name}'.")
		seen_method_names.add(method_name)
		if method.task_name not in task_lookup:
			raise ValueError(
				f"Method '{method_name}' targets unknown compound task '{method.task_name}'."
			)
		task = task_lookup[method.task_name]
		if method.task_args and len(method.task_args) != len(task.parameters):
			raise ValueError(
				f"Method '{method_name}' binds {len(method.task_args)} task args for "
				f"task '{method.task_name}' with arity {len(task.parameters)}."
			)
		_validate_literal_collection(
			method.context,
			predicate_arities=predicate_arities,
			context_label=f"method '{method_name}' context",
		)
		step_ids: set[str] = set()
		for step in method.subtasks:
			step_id = str(step.step_id).strip()
			if not step_id:
				raise ValueError(f"Method '{method_name}' contains a step without step_id.")
			if step_id in step_ids:
				raise ValueError(
					f"Method '{method_name}' contains duplicate step_id '{step_id}'."
				)
			step_ids.add(step_id)
			step_name = str(step.task_name).strip()
			if not step_name:
				raise ValueError(
					f"Method '{method_name}' step '{step_id}' is missing its task_name."
				)
			if step.kind == "primitive":
				if step_name not in primitive_names:
					raise ValueError(
						f"Method '{method_name}' step '{step_id}' references unknown primitive "
						f"'{step_name}'."
					)
			elif step.kind == "compound":
				if step_name not in task_lookup:
					raise ValueError(
						f"Method '{method_name}' step '{step_id}' references unknown compound "
						f"'{step_name}'."
					)
				if (
					_canonical_name(step_name) == _canonical_name(method.task_name)
					and tuple(step.args) == tuple(method.task_args)
				):
					raise ValueError(
						f"Method '{method_name}' contains immediate same-argument recursion via "
						f"step '{step_id}'."
					)
			_validate_literal_collection(
				step.preconditions,
				predicate_arities=predicate_arities,
				context_label=f"method '{method_name}' step '{step_id}' preconditions",
			)
			_validate_literal_collection(
				step.effects,
				predicate_arities=predicate_arities,
				context_label=f"method '{method_name}' step '{step_id}' effects",
			)
			if step.literal is not None:
				_validate_literal_collection(
					(step.literal,),
					predicate_arities=predicate_arities,
					context_label=f"method '{method_name}' step '{step_id}' literal",
				)


def validate_typed_structural_soundness(
	domain: Any,
	library: HTNMethodLibrary,
) -> List[str]:
	warnings: List[str] = []
	task_lookup = _compound_task_lookup(library)
	for method in library.methods:
		method_name = str(method.method_name).strip()
		if method.task_name not in task_lookup:
			raise ValueError(
				f"Method '{method_name}' targets unknown compound task '{method.task_name}'."
			)
		task = task_lookup[method.task_name]
		if method.task_args and len(method.task_args) != len(task.parameters):
			raise ValueError(
				f"Method '{method_name}' binds {len(method.task_args)} task args for "
				f"task '{method.task_name}' with arity {len(task.parameters)}."
			)
		_validate_ordering_graph(method)
	warnings.extend(_validate_variable_type_constraints(domain, library))
	return warnings


def validate_decomposition_admissibility(
	domain: Any,
	library: HTNMethodLibrary,
) -> List[str]:
	_ = domain
	warnings: List[str] = []
	task_lookup = _compound_task_lookup(library)
	declared_task_names: Dict[str, str] = {}
	for task in library.compound_tasks:
		display_task_name = str(getattr(task, "name", "")).strip()
		if display_task_name:
			declared_task_names[_canonical_name(display_task_name)] = display_task_name
	methods_by_task: Dict[str, List[Any]] = defaultdict(list)
	for method in library.methods:
		methods_by_task[_canonical_name(method.task_name)].append(method)
	for canonical_task_name, display_task_name in sorted(declared_task_names.items()):
		if not methods_by_task.get(canonical_task_name):
			raise ValueError(
				f"Declared compound task '{display_task_name}' has no generated methods."
			)
	for method in library.methods:
		method_name = str(method.method_name).strip()
		for step in method.subtasks:
			if getattr(step, "kind", "") != "compound":
				continue
			step_name = str(getattr(step, "task_name", "")).strip()
			if step_name not in task_lookup:
				raise ValueError(
					f"Method '{method_name}' step '{step.step_id}' references unknown compound "
					f"'{step_name}'."
				)
			if (
				_canonical_name(step_name) == _canonical_name(method.task_name)
				and tuple(step.args) == tuple(method.task_args)
			):
				raise ValueError(
					f"Method '{method_name}' contains immediate same-argument recursion via "
					f"step '{step.step_id}'."
				)
		if method.subtasks:
			continue
		if not method.context:
			raise ValueError(
				f"Method '{method_name}' has empty subtasks without an already-satisfied guard."
			)
			warnings.append(
				"Guarded empty method accepted by structural guard-only policy; "
				f"method='{method_name}'."
			)
		return warnings


def _validate_ordering_graph(method: Any) -> None:
	step_ids = {str(step.step_id).strip() for step in method.subtasks}
	adjacency: Dict[str, List[str]] = {step_id: [] for step_id in step_ids}
	for before_step_id, after_step_id in method.ordering:
		if before_step_id not in step_ids or after_step_id not in step_ids:
			raise ValueError(
				f"Method '{method.method_name}' ordering references missing step ids "
				f"'{before_step_id}' -> '{after_step_id}'."
			)
		if before_step_id == after_step_id:
			raise ValueError(
				f"Method '{method.method_name}' contains self-ordering edge on "
				f"'{before_step_id}'."
			)
		adjacency.setdefault(before_step_id, []).append(after_step_id)

	visiting: Set[str] = set()
	visited: Set[str] = set()

	def visit(step_id: str) -> None:
		if step_id in visited:
			return
		if step_id in visiting:
			raise ValueError(
				f"Method '{method.method_name}' ordering graph contains a cycle."
			)
		visiting.add(step_id)
		for child_step_id in adjacency.get(step_id, ()):
			visit(child_step_id)
		visiting.remove(step_id)
		visited.add(step_id)

	for step_id in sorted(step_ids):
		visit(step_id)


def _build_type_parent_map(domain: Any) -> Dict[str, Optional[str]]:
	tokens = [
		str(token).strip()
		for token in (getattr(domain, "types", ()) or ())
		if str(token).strip()
	]
	if not tokens:
		return {"object": None}
	parent_map: Dict[str, Optional[str]] = {}
	pending_children: List[str] = []
	index = 0
	while index < len(tokens):
		token = tokens[index]
		if token == "-":
			if not pending_children or index + 1 >= len(tokens):
				raise ValueError("Malformed HDDL :types declaration.")
			parent_type = tokens[index + 1]
			for child_type in pending_children:
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
			parent_map[parent_type] = "object"
			changed = True
	return parent_map


def _parameter_symbol(parameter: Any) -> str:
	text = str(parameter or "").strip()
	if ":" in text:
		text = text.split(":", 1)[0].strip()
	elif "-" in text:
		text = text.split("-", 1)[0].strip()
	return text


def _parameter_type(parameter: Any) -> str:
	text = str(parameter or "").strip()
	if ":" in text:
		return text.split(":", 1)[1].strip() or "object"
	if "-" in text:
		return text.split("-", 1)[1].strip() or "object"
	return "object"


def _is_variable_symbol(symbol: Any) -> bool:
	text = _parameter_symbol(symbol)
	return bool(text) and (text.startswith("?") or text[0].isupper())


def _symbol_key(symbol: Any) -> str:
	return _parameter_symbol(symbol)


class _DisjointSet:
	def __init__(self) -> None:
		self.parent: Dict[str, str] = {}

	def find(self, item: str) -> str:
		self.parent.setdefault(item, item)
		if self.parent[item] != item:
			self.parent[item] = self.find(self.parent[item])
		return self.parent[item]

	def union(self, left: str, right: str) -> None:
		left_root = self.find(left)
		right_root = self.find(right)
		if left_root != right_root:
			self.parent[right_root] = left_root


def _is_subtype(
	type_name: str,
	expected_type: str,
	type_parent_map: Dict[str, Optional[str]],
) -> bool:
	if type_name == expected_type:
		return True
	cursor = type_parent_map.get(type_name)
	visited = {type_name}
	while cursor is not None and cursor not in visited:
		if cursor == expected_type:
			return True
		visited.add(cursor)
		cursor = type_parent_map.get(cursor)
	return False


def _has_common_subtype(
	type_names: Set[str],
	type_parent_map: Dict[str, Optional[str]],
) -> bool:
	if not type_names:
		return True
	for candidate_type in type_parent_map:
		if all(_is_subtype(candidate_type, required_type, type_parent_map) for required_type in type_names):
			return True
	return False


def _validate_variable_type_constraints(
	domain: Any,
	library: HTNMethodLibrary,
) -> List[str]:
	type_parent_map = _build_type_parent_map(domain)
	domain_type_names = set(type_parent_map)
	predicate_types = {
		str(getattr(predicate, "name", "")).strip(): tuple(
			_parameter_type(parameter)
			for parameter in (getattr(predicate, "parameters", ()) or ())
		)
		for predicate in getattr(domain, "predicates", ())
		if str(getattr(predicate, "name", "")).strip()
	}
	action_types: Dict[str, Tuple[str, ...]] = {}
	for action in getattr(domain, "actions", ()):
		action_name = str(getattr(action, "name", "")).strip()
		if not action_name:
			continue
		signature = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(action, "parameters", ()) or ())
		)
		action_types[action_name] = signature
		action_types[sanitize_identifier(action_name)] = signature
	task_types: Dict[str, Tuple[str, ...]] = {}
	for task in getattr(domain, "tasks", ()):
		task_name = str(getattr(task, "name", "")).strip()
		if not task_name:
			continue
		signature = tuple(
			_parameter_type(parameter)
			for parameter in (getattr(task, "parameters", ()) or ())
		)
		task_types[task_name] = signature
		task_types[sanitize_identifier(task_name)] = signature
	for task in library.compound_tasks:
		task_name = str(getattr(task, "name", "")).strip()
		if task_name and task_name not in task_types:
			task_types[task_name] = tuple(
				_parameter_type(parameter)
				for parameter in (getattr(task, "parameters", ()) or ())
			)

	warnings: List[str] = []
	for method in library.methods:
		method_name = str(method.method_name).strip()
		constraints: Dict[str, Set[str]] = defaultdict(set)
		variables: Set[str] = set()
		disjoint_set = _DisjointSet()

		def add_candidate(symbol: Any, type_name: str) -> None:
			key = _symbol_key(symbol)
			if not key or not _is_variable_symbol(key):
				return
			variables.add(key)
			disjoint_set.find(key)
			if type_name:
				constraints[key].add(type_name)

		def collect_args(args: Sequence[Any], signature: Sequence[str], scope: str) -> None:
			if not signature:
				for arg in args:
					if _is_variable_symbol(arg):
						variables.add(_symbol_key(arg))
						disjoint_set.find(_symbol_key(arg))
				return
			if len(args) != len(signature):
				raise ValueError(
					f"{scope}: arity mismatch (args={len(args)}, signature={len(signature)})."
				)
			for arg, type_name in zip(args, signature):
				add_candidate(arg, type_name)

		def collect_literal(literal: HTNLiteral, scope: str) -> None:
			if literal.is_equality:
				for arg in literal.args:
					if _is_variable_symbol(arg):
						variables.add(_symbol_key(arg))
						disjoint_set.find(_symbol_key(arg))
				if len(literal.args) == 2 and all(_is_variable_symbol(arg) for arg in literal.args):
					disjoint_set.union(_symbol_key(literal.args[0]), _symbol_key(literal.args[1]))
				return
			signature = predicate_types.get(literal.predicate)
			if signature is None:
				raise ValueError(
					f"{scope}: unknown predicate '{literal.predicate}'."
				)
			collect_args(literal.args, signature, scope)

		for parameter in method.parameters:
			add_candidate(parameter, _parameter_type(parameter) if (":" in str(parameter) or "-" in str(parameter)) else "")
		task_signature = task_types.get(method.task_name, ())
		collect_args(method.task_args, task_signature, f"Method '{method_name}' task typing")
		for literal in method.context:
			collect_literal(literal, f"Method '{method_name}' context typing")
		for step in method.subtasks:
			if step.kind == "primitive":
				signature = action_types.get(step.action_name or "") or action_types.get(step.task_name) or action_types.get(sanitize_identifier(step.task_name))
				if signature is None:
					raise ValueError(
						f"Method '{method_name}' step '{step.step_id}' references unknown primitive "
						f"'{step.task_name}'."
					)
				collect_args(step.args, signature, f"Method '{method_name}' primitive step typing")
			elif step.kind == "compound":
				signature = task_types.get(step.task_name) or task_types.get(sanitize_identifier(step.task_name)) or ()
				collect_args(step.args, signature, f"Method '{method_name}' compound step typing")
			for literal in (*step.preconditions, *step.effects):
				collect_literal(literal, f"Method '{method_name}' step '{step.step_id}' literal typing")
			if step.literal is not None:
				collect_literal(step.literal, f"Method '{method_name}' step '{step.step_id}' literal typing")

		group_constraints: Dict[str, Set[str]] = defaultdict(set)
		group_members: Dict[str, Set[str]] = defaultdict(set)
		for variable in variables:
			root = disjoint_set.find(variable)
			group_members[root].add(variable)
			group_constraints[root].update(constraints.get(variable, set()))
		for root, type_names in group_constraints.items():
			unknown_types = sorted(type_name for type_name in type_names if type_name not in domain_type_names)
			if unknown_types:
				raise ValueError(
					f"Method '{method_name}' variable typing references unknown types "
					f"{unknown_types}."
				)
			if type_names and not _has_common_subtype(type_names, type_parent_map):
				members = sorted(group_members.get(root, {root}))
				raise ValueError(
					f"Method '{method_name}' variable typing conflict for {members}: "
					f"{sorted(type_names)}."
				)
		for root, members in sorted(group_members.items()):
			if not group_constraints.get(root):
				warnings.append(
					f"Method '{method_name}' has unresolved variable type(s): "
					f"{sorted(members)}."
				)
	return warnings


def _validate_literal_collection(
	literals: Iterable[HTNLiteral],
	*,
	predicate_arities: Dict[str, int],
	context_label: str,
) -> None:
	for literal in literals:
		predicate = str(literal.predicate).strip()
		if not predicate:
			raise ValueError(f"{context_label} contains a literal without predicate name.")
		if predicate == "=":
			if len(literal.args) != 2:
				raise ValueError(
					f"{context_label} contains equality literal with arity {len(literal.args)}."
				)
			continue
		expected_arity = predicate_arities.get(predicate)
		if expected_arity is None:
			raise ValueError(
				f"{context_label} contains unknown predicate '{predicate}'."
			)
		if len(literal.args) != expected_arity:
			raise ValueError(
				f"{context_label} contains predicate '{predicate}' with arity "
				f"{len(literal.args)} but expected {expected_arity}."
			)
