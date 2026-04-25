"""Library normalization and lightweight postprocessing for method synthesis."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from method_library.synthesis.schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
)
from method_library.synthesis.naming import sanitize_identifier
from .errors import HTNSynthesisError


class MethodSynthesisLibraryPostprocessMixin:
	@staticmethod
	def _drop_trivial_boolean_literals(
		literals: Sequence[HTNLiteral],
	) -> Tuple[HTNLiteral, ...]:
		return tuple(
			literal
			for literal in literals
			if str(getattr(literal, "predicate", "")).strip().lower() != "true"
		)

	def _normalise_llm_library(
		self,
		library: HTNMethodLibrary,
		domain: Any,
		*,
		prompt_analysis: Optional[Dict[str, Any]] = None,
	) -> HTNMethodLibrary:
		_ = prompt_analysis
		action_schemas = self._action_schema_map(domain)
		domain_predicate_names = {
			str(getattr(predicate, "name", "") or "").strip()
			for predicate in getattr(domain, "predicates", [])
			if str(getattr(predicate, "name", "") or "").strip()
		}
		raw_task_to_alias, alias_to_raw_task = self._declared_task_alias_maps(domain)
		domain_task_defaults = {
			self._sanitize_name(str(getattr(task, "name", "")).strip()): tuple(
				self._sanitize_name(str(parameter).split("-", 1)[0].strip())
				for parameter in (getattr(task, "parameters", ()) or ())
			)
			for task in getattr(domain, "tasks", [])
			if str(getattr(task, "name", "")).strip()
		}
		domain_source_predicates = {
			self._sanitize_name(str(getattr(task, "name", "")).strip()): tuple(
				str(predicate_name).strip()
				for predicate_name in (getattr(task, "source_predicates", ()) or ())
				if str(predicate_name).strip()
			)
			for task in getattr(domain, "tasks", [])
			if str(getattr(task, "name", "")).strip()
		}
		compound_tasks: list[HTNTask] = []
		compound_alias_lookup: Dict[str, str] = {}
		for task in library.compound_tasks:
			normalised_task_name, source_task_name = self._normalise_declared_task_identifier(
				task.name,
				raw_to_alias=raw_task_to_alias,
				alias_to_raw=alias_to_raw_task,
			)
			source_name = task.source_name or source_task_name
			canonical_parameters = domain_task_defaults.get(normalised_task_name, task.parameters)
			compound_alias_lookup[task.name] = normalised_task_name
			compound_alias_lookup[normalised_task_name] = normalised_task_name
			if source_name:
				compound_alias_lookup[source_name] = normalised_task_name
			compound_tasks.append(
				HTNTask(
					name=normalised_task_name,
					parameters=tuple(canonical_parameters),
					is_primitive=False,
					source_predicates=domain_source_predicates.get(normalised_task_name, task.source_predicates),
					headline_literal=task.headline_literal,
					source_name=source_name,
				)
			)
		compound_task_names = {task.name for task in compound_tasks}
		task_lookup = {task.name: task for task in compound_tasks}
		primitive_task_names = {
			self._sanitize_name(str(getattr(action, "name", "")).strip())
			for action in getattr(domain, "actions", [])
			if str(getattr(action, "name", "")).strip()
		}
		methods: list[HTNMethod] = []
		for method in library.methods:
			normalised_method_task_name = compound_alias_lookup.get(
				method.task_name,
				self._normalise_declared_task_identifier(
					method.task_name,
					raw_to_alias=raw_task_to_alias,
					alias_to_raw=alias_to_raw_task,
				)[0],
			)
			normalised_steps: list[HTNMethodStep] = []
			for step in method.subtasks:
				action_name = step.action_name
				task_name = compound_alias_lookup.get(
					step.task_name,
					self._normalise_declared_task_identifier(
						step.task_name,
						raw_to_alias=raw_task_to_alias,
						alias_to_raw=alias_to_raw_task,
					)[0],
				)
				kind = self._coerce_step_kind(
					HTNMethodStep(
						step_id=step.step_id,
						task_name=task_name,
						args=step.args,
						kind=step.kind,
						action_name=action_name,
						literal=step.literal,
						preconditions=step.preconditions,
						effects=step.effects,
					),
					primitive_task_names=primitive_task_names,
					compound_task_names=compound_task_names,
				)
				preconditions = step.preconditions
				effects = step.effects
				if kind == "primitive":
					action_schema = self._resolve_action_schema(step, action_schemas)
					if action_schema is not None:
						action_name = action_schema.name
						task_name = self._sanitize_name(action_schema.name)
						preconditions = self._materialise_action_literals(
							action_schema.preconditions,
							action_schema.parameters,
							step.args,
						)
						effects = self._materialise_action_literals(
							action_schema.effects,
							action_schema.parameters,
							step.args,
						)
				normalised_steps.append(
					HTNMethodStep(
						step_id=step.step_id,
						task_name=task_name,
						args=step.args,
						kind=kind,
						action_name=action_name,
						literal=None if kind == "primitive" else step.literal,
						preconditions=preconditions,
						effects=effects,
					)
				)
			task_schema = task_lookup.get(normalised_method_task_name)
			default_task_args = tuple(task_schema.parameters) if task_schema is not None else ()
			task_args = self._normalise_method_task_args(
				method.task_args,
				default_task_args=default_task_args,
			)
			normalised_parameters = self._normalise_method_parameters(
				method.parameters,
				method.context,
				tuple(normalised_steps),
			)
			normalised_parameters = self._retain_used_method_parameters(
				normalised_parameters,
				task_args=task_args,
				context=method.context,
				steps=tuple(normalised_steps),
			)
			normalised_ordering = self._normalise_method_ordering(
				method.ordering,
				steps=tuple(normalised_steps),
			)
			normalised_method_name = self._normalise_method_identifier(
				method.method_name,
				original_task_name=method.task_name,
				normalised_task_name=normalised_method_task_name,
			)
			methods.append(
				HTNMethod(
					method_name=normalised_method_name,
					task_name=normalised_method_task_name,
					parameters=normalised_parameters,
					task_args=task_args,
					context=self._drop_trivial_boolean_literals(method.context),
					subtasks=tuple(normalised_steps),
					ordering=normalised_ordering,
					origin=method.origin,
					source_method_name=(
						method.source_method_name
						or (method.method_name if normalised_method_name != method.method_name else None)
					),
					source_instruction_ids=tuple(method.source_instruction_ids),
				)
			)
			methods[-1] = self._normalise_method_negation_literals(
				methods[-1],
				domain_predicate_names=domain_predicate_names,
			)
		methods = self._deduplicate_method_identifiers(methods)
		return HTNMethodLibrary(
			compound_tasks=compound_tasks,
			primitive_tasks=list(library.primitive_tasks),
			methods=methods,
			target_literals=[],
			target_task_bindings=[],
		)

	def _normalise_method_negation_literals(
		self,
		method: HTNMethod,
		*,
		domain_predicate_names: set[str],
	) -> HTNMethod:
		return HTNMethod(
			method_name=method.method_name,
			task_name=method.task_name,
			parameters=method.parameters,
			task_args=method.task_args,
			context=tuple(
				self._normalise_literal_negation_form(
					literal,
					domain_predicate_names=domain_predicate_names,
				)
				for literal in method.context
			),
			subtasks=tuple(
				HTNMethodStep(
					step_id=step.step_id,
					task_name=step.task_name,
					args=step.args,
					kind=step.kind,
					action_name=step.action_name,
					literal=(
						self._normalise_literal_negation_form(
							step.literal,
							domain_predicate_names=domain_predicate_names,
						)
						if step.literal is not None
						else None
					),
					preconditions=tuple(
						self._normalise_literal_negation_form(
							literal,
							domain_predicate_names=domain_predicate_names,
						)
						for literal in step.preconditions
					),
					effects=tuple(
						self._normalise_literal_negation_form(
							literal,
							domain_predicate_names=domain_predicate_names,
						)
						for literal in step.effects
					),
				)
				for step in method.subtasks
			),
			ordering=method.ordering,
			origin=method.origin,
			source_method_name=method.source_method_name,
			source_instruction_ids=tuple(method.source_instruction_ids),
		)

	@staticmethod
	def _normalise_literal_negation_form(
		literal: HTNLiteral,
		*,
		domain_predicate_names: set[str],
	) -> HTNLiteral:
		predicate_text = str(getattr(literal, "predicate", "") or "").strip()
		if not predicate_text or not literal.is_positive:
			return literal
		lowered_predicate = predicate_text.lower()
		if not lowered_predicate.startswith("not"):
			return literal
		for candidate in sorted(domain_predicate_names, key=len, reverse=True):
			candidate_text = str(candidate or "").strip()
			if not candidate_text:
				continue
			lowered_candidate = candidate_text.lower()
			if lowered_predicate == f"not{lowered_candidate}":
				return HTNLiteral(
					predicate=candidate_text,
					args=literal.args,
					is_positive=False,
					source_symbol=literal.source_symbol,
					negation_mode=literal.negation_mode,
				)
		return literal

	def _coerce_step_kind(
		self,
		step: HTNMethodStep,
		*,
		primitive_task_names: set[str],
		compound_task_names: set[str],
	) -> str:
		"""Repair obvious `subtask.kind` mismatches from the LLM output."""
		if step.task_name in compound_task_names:
			return "compound"
		if step.task_name in primitive_task_names:
			return "primitive"
		if step.action_name:
			alias = self._sanitize_name(step.action_name)
			if alias in primitive_task_names and alias not in compound_task_names:
				return "primitive"
		return step.kind

	def _normalise_method_parameters(
		self,
		parameters: Tuple[str, ...],
		context: Tuple[HTNLiteral, ...],
		steps: Tuple[HTNMethodStep, ...],
	) -> Tuple[str, ...]:
		ordered_parameters = [
			self._strip_variable_type_annotation(parameter)
			for parameter in parameters
		]
		seen = set(ordered_parameters)

		def consider(symbol: str) -> None:
			symbol = self._strip_variable_type_annotation(symbol)
			if not self._looks_like_variable(symbol):
				return
			if symbol in seen:
				return
			seen.add(symbol)
			ordered_parameters.append(symbol)

		for literal in context:
			for arg in literal.args:
				consider(arg)

		for step in steps:
			for arg in step.args:
				consider(arg)
			if step.literal is not None:
				for arg in step.literal.args:
					consider(arg)
			for literal in (*step.preconditions, *step.effects):
				for arg in literal.args:
					consider(arg)

		return tuple(ordered_parameters)

	def _normalise_method_task_args(
		self,
		task_args: Sequence[str],
		*,
		default_task_args: Sequence[str],
	) -> Tuple[str, ...]:
		defaults = tuple(str(arg).strip() for arg in default_task_args if str(arg).strip())
		if not task_args:
			return defaults
		normalised = tuple(
			self._strip_variable_type_annotation(str(arg))
			for arg in task_args
			if str(arg).strip()
		)
		if not defaults:
			return normalised
		if len(normalised) == len(defaults):
			return normalised
		if len(normalised) > len(defaults):
			return normalised[:len(defaults)]
		return normalised + defaults[len(normalised):]

	@staticmethod
	def _normalise_method_ordering(
		ordering: Tuple[Tuple[str, str], ...],
		*,
		steps: Tuple[HTNMethodStep, ...],
	) -> Tuple[Tuple[str, str], ...]:
		known_step_ids = {
			str(step.step_id).strip()
			for step in steps
			if str(step.step_id).strip()
		}
		normalised_edges: List[Tuple[str, str]] = []
		seen_edges: set[Tuple[str, str]] = set()
		for before_step_id, after_step_id in ordering:
			before = str(before_step_id).strip()
			after = str(after_step_id).strip()
			if not before or not after:
				continue
			if before == after:
				continue
			if before not in known_step_ids or after not in known_step_ids:
				continue
			edge = (before, after)
			if edge in seen_edges:
				continue
			seen_edges.add(edge)
			normalised_edges.append(edge)
		return tuple(normalised_edges)

	def _retain_used_method_parameters(
		self,
		parameters: Tuple[str, ...],
		*,
		task_args: Sequence[str],
		context: Tuple[HTNLiteral, ...],
		steps: Tuple[HTNMethodStep, ...],
	) -> Tuple[str, ...]:
		required_symbols = {
			str(symbol)
			for symbol in task_args
			if self._looks_like_variable(str(symbol))
		}
		for literal in context:
			required_symbols.update(
				arg
				for arg in literal.args
				if self._looks_like_variable(arg)
			)
		for step in steps:
			required_symbols.update(
				arg
				for arg in step.args
				if self._looks_like_variable(arg)
			)
			if step.literal is not None:
				required_symbols.update(
					arg
					for arg in step.literal.args
					if self._looks_like_variable(arg)
				)
			for literal in (*step.preconditions, *step.effects):
				required_symbols.update(
					arg
					for arg in literal.args
					if self._looks_like_variable(arg)
				)
		ordered_parameters: List[str] = []
		seen: set[str] = set()
		for symbol in task_args:
			candidate = str(symbol)
			if not self._looks_like_variable(candidate):
				continue
			if candidate in seen:
				continue
			seen.add(candidate)
			ordered_parameters.append(candidate)
		for parameter in parameters:
			candidate_parameter = self._strip_variable_type_annotation(parameter)
			if candidate_parameter not in required_symbols:
				continue
			if candidate_parameter in seen:
				continue
			seen.add(candidate_parameter)
			ordered_parameters.append(candidate_parameter)
		pruned_parameters = tuple(ordered_parameters)
		return pruned_parameters or tuple(parameters)

	def _build_synthesis_error(
		self,
		metadata: Dict[str, Any],
		failure_stage: str,
		failure_reason: str,
	) -> HTNSynthesisError:
		error_metadata = dict(metadata)
		error_metadata["failure_stage"] = failure_stage
		error_metadata["failure_reason"] = failure_reason
		error_metadata["failure_class"] = self._classify_failure(
			failure_stage,
			failure_reason,
		)
		return HTNSynthesisError(
			f"HTN synthesis failed during {failure_stage}: {failure_reason}",
			model=error_metadata.get("model"),
			llm_prompt=error_metadata.get("llm_prompt"),
			llm_response=error_metadata.get("llm_response"),
			metadata=error_metadata,
		)

	@staticmethod
	def _classify_failure(
		failure_stage: str,
		failure_reason: str,
	) -> str:
		reason = str(failure_reason or "").lower()
		if failure_stage == "preflight":
			return "llm_not_configured"
		if failure_stage == "llm_call":
			return "llm_call_failed"
		if failure_stage == "response_parse":
			return "llm_response_parse_failed"
		if "explicit ordering edges" in reason:
			return "missing_ordering_edges"
		if "uses auxiliary parameter" in reason:
			return "unconstrained_auxiliary_parameter"
		if "shared dynamic prerequisites" in reason:
			return "missing_shared_child_prerequisites"
		if "dynamic preconditions" in reason:
			return "missing_primitive_dynamic_preconditions"
		if "grounded query object" in reason:
			return "grounded_query_object_leakage"
		if failure_stage == "library_validation":
			return "library_validation_failed"
		return f"{failure_stage}_failed"

	def _action_schema_map(self, domain: Any) -> Dict[str, Any]:
		action_schemas: Dict[str, Any] = {}
		for action in domain.actions:
			parsed_action = self.parser.parse_action(action)
			action_schemas[action.name] = parsed_action
			action_schemas[self._sanitize_name(action.name)] = parsed_action
		return action_schemas

	def _action_type_map(self, domain: Any) -> Dict[str, Tuple[str, ...]]:
		action_types: Dict[str, Tuple[str, ...]] = {}
		for action in domain.actions:
			type_signature = tuple(
				self._parameter_type(parameter)
				for parameter in action.parameters
			)
			action_types[action.name] = type_signature
			action_types[self._sanitize_name(action.name)] = type_signature
		return action_types

	def _task_type_map(self, domain: Any) -> Dict[str, Tuple[str, ...]]:
		task_types: Dict[str, Tuple[str, ...]] = {}
		for task in getattr(domain, "tasks", []):
			type_signature = tuple(
				self._parameter_type(parameter)
				for parameter in task.parameters
			)
			task_types[task.name] = type_signature
			task_types[self._sanitize_name(task.name)] = type_signature
		return task_types

	def _resolve_action_schema(self, step: HTNMethodStep, action_schemas: Dict[str, Any]) -> Any:
		if step.action_name and step.action_name in action_schemas:
			return action_schemas[step.action_name]
		return action_schemas.get(self._sanitize_name(step.task_name))

	def _materialise_action_literals(
		self,
		patterns: Tuple[Any, ...],
		schema_parameters: Tuple[str, ...],
		step_args: Tuple[str, ...],
	) -> Tuple[HTNLiteral, ...]:
		bindings = {
			parameter: arg
			for parameter, arg in zip(schema_parameters, step_args)
		}
		return tuple(
			HTNLiteral(
				predicate=pattern.predicate,
				args=tuple(bindings.get(arg, arg) for arg in pattern.args),
				is_positive=pattern.is_positive,
				source_symbol=None,
			)
			for pattern in patterns
		)

	@staticmethod
	def _sanitize_name(name: str) -> str:
		return sanitize_identifier(name)

	def _declared_task_alias_maps(
		self,
		domain: Any,
	) -> Tuple[Dict[str, str], Dict[str, str]]:
		raw_to_alias: Dict[str, str] = {}
		alias_to_raw: Dict[str, str] = {}
		for task in getattr(domain, "tasks", []):
			raw_name = str(getattr(task, "name", "") or "").strip()
			if not raw_name:
				continue
			alias_name = self._sanitize_name(raw_name)
			raw_to_alias[raw_name] = alias_name
			alias_to_raw[alias_name] = raw_name
		return raw_to_alias, alias_to_raw

	def _normalise_declared_task_identifier(
		self,
		task_name: str,
		*,
		raw_to_alias: Dict[str, str],
		alias_to_raw: Dict[str, str],
	) -> Tuple[str, Optional[str]]:
		candidate = str(task_name or "").strip()
		if not candidate:
			return "", None
		if candidate in raw_to_alias:
			return raw_to_alias[candidate], candidate
		if candidate in alias_to_raw:
			return candidate, alias_to_raw[candidate]
		sanitized_candidate = self._sanitize_name(candidate)
		if sanitized_candidate in alias_to_raw:
			return sanitized_candidate, alias_to_raw[sanitized_candidate]
		return candidate, None

	def _normalise_method_identifier(
		self,
		method_name: str,
		*,
		original_task_name: str,
		normalised_task_name: str,
	) -> str:
		candidate = str(method_name or "").strip()
		if not candidate:
			return candidate

		strategy_suffix = self._extract_method_strategy_suffix(
			candidate,
			original_task_name=original_task_name,
			normalised_task_name=normalised_task_name,
		)
		if strategy_suffix is None:
			return candidate

		normalised_suffix = self._normalise_method_strategy_suffix(strategy_suffix)
		if not normalised_suffix:
			return candidate
		return f"m_{normalised_task_name}_{normalised_suffix}"

	def _extract_method_strategy_suffix(
		self,
		method_name: str,
		*,
		original_task_name: str,
		normalised_task_name: str,
	) -> Optional[str]:
		prefix_candidates = [
			f"m_{normalised_task_name}_",
			f"m_{original_task_name}_",
			f"m_{self._sanitize_name(original_task_name)}_",
		]
		for prefix in prefix_candidates:
			if method_name.startswith(prefix):
				return method_name[len(prefix):]
		return None

	@staticmethod
	def _normalise_method_strategy_suffix(strategy_suffix: str) -> str:
		normalised = re.sub(r"[^A-Za-z0-9]+", "_", str(strategy_suffix).strip())
		normalised = re.sub(r"_+", "_", normalised).strip("_").lower()
		if not normalised:
			return "branch"
		if not normalised[0].isalpha():
			normalised = f"branch_{normalised}"
		return normalised

	@staticmethod
	def _deduplicate_method_identifiers(
		methods: Sequence[HTNMethod],
	) -> List[HTNMethod]:
		seen: Dict[str, int] = {}
		unique_methods: List[HTNMethod] = []
		for method in methods:
			base_name = method.method_name
			count = seen.get(base_name, 0)
			seen[base_name] = count + 1
			if count == 0:
				unique_methods.append(method)
				continue
			unique_methods.append(
				HTNMethod(
					method_name=f"{base_name}_{count + 1}",
					task_name=method.task_name,
					parameters=method.parameters,
					task_args=method.task_args,
					context=method.context,
					subtasks=method.subtasks,
					ordering=method.ordering,
					origin=method.origin,
					source_method_name=method.source_method_name or method.method_name,
					source_instruction_ids=tuple(method.source_instruction_ids),
				),
			)
		return unique_methods

	@staticmethod
	def _parameter_type(parameter: str) -> str:
		if "-" not in parameter:
			return "OBJECT"
		return parameter.split("-", 1)[1].strip().upper()

	@staticmethod
	def _build_domain_type_parent_map(domain: Any) -> Dict[str, Optional[str]]:
		tokens = [
			str(token).strip().upper()
			for token in (getattr(domain, "types", []) or [])
			if str(token).strip()
		]
		if not tokens:
			return {"OBJECT": None}

		parent_map: Dict[str, Optional[str]] = {}
		pending_children: List[str] = []
		index = 0
		while index < len(tokens):
			token = tokens[index]
			if token == "-":
				if not pending_children or index + 1 >= len(tokens):
					raise ValueError("Malformed HDDL :types declaration (dangling '-').")
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
			parent_map.setdefault(child_type, "OBJECT")

		parent_map["OBJECT"] = None
		changed = True
		while changed:
			changed = False
			for parent_type in list(parent_map.values()):
				if parent_type is None or parent_type in parent_map:
					continue
				parent_map[parent_type] = "OBJECT" if parent_type != "OBJECT" else None
				changed = True

		for type_name in list(parent_map.keys()):
			if type_name == "OBJECT":
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

	@staticmethod
	def _ordered_method_steps(self, method: HTNMethod) -> List[HTNMethodStep]:
		if len(method.subtasks) <= 1 or not method.ordering:
			return list(method.subtasks)

		step_lookup = {
			step.step_id: step
			for step in method.subtasks
		}
		dependents: Dict[str, List[str]] = {
			step.step_id: []
			for step in method.subtasks
		}
		in_degree: Dict[str, int] = {
			step.step_id: 0
			for step in method.subtasks
		}

		for before, after in method.ordering:
			if before not in step_lookup or after not in step_lookup:
				return list(method.subtasks)
			dependents[before].append(after)
			in_degree[after] += 1

		ordered_steps: List[HTNMethodStep] = []
		ready = [
			step.step_id
			for step in method.subtasks
			if in_degree[step.step_id] == 0
		]

		while ready:
			current_id = ready.pop(0)
			ordered_steps.append(step_lookup[current_id])
			for next_id in dependents[current_id]:
				in_degree[next_id] -= 1
				if in_degree[next_id] == 0:
					ready.append(next_id)

		if len(ordered_steps) != len(method.subtasks):
			return list(method.subtasks)

		return ordered_steps

	def _default_method_task_args(
		method: HTNMethod,
		task_schema: Optional[HTNTask],
	) -> Tuple[str, ...]:
		if method.task_args:
			return tuple(method.task_args)
		if task_schema is None:
			return tuple(method.parameters)
		declared_parameters = tuple(getattr(task_schema, "parameters", ()) or ())
		if not declared_parameters:
			return ()
		leading_parameters = tuple(method.parameters[: len(declared_parameters)])
		if len(leading_parameters) == len(declared_parameters):
			return leading_parameters
		return declared_parameters

	def _method_parameter_bindings(
		self,
		method: HTNMethod,
		task_schema: Optional[HTNTask],
		parameter_bindings: Optional[Dict[str, str]] = None,
	) -> Dict[str, str]:
		bindings = dict(parameter_bindings or {})
		if task_schema is None:
			return bindings
		for method_arg, task_parameter in zip(
			self._default_method_task_args(method, task_schema),
			task_schema.parameters,
		):
			bindings[method_arg] = bindings.get(task_parameter, task_parameter)
		return bindings

	@classmethod
	def _method_constructively_supports_literal(
		self,
		method: HTNMethod,
		expected_literal: HTNLiteral,
		methods_by_task: Dict[str, List[HTNMethod]],
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
		parameter_bindings: Optional[Dict[str, str]] = None,
		visiting: Optional[set[Tuple[str, str]]] = None,
	) -> bool:
		expected_signature = expected_literal.to_signature()
		task_schema = task_lookup.get(method.task_name)
		parameter_bindings = self._method_parameter_bindings(
			method,
			task_schema,
			parameter_bindings,
		)
		visiting = set() if visiting is None else visiting
		visit_key = (method.method_name, expected_signature)
		if visit_key in visiting:
			return False
		visiting.add(visit_key)

		for step in method.subtasks:
			candidate_literals: List[HTNLiteral] = []
			if step.literal is not None:
				candidate_literals.append(step.literal)
			candidate_literals.extend(
				self._step_effect_literals(
					step,
					task_lookup,
					action_schemas,
					predicate_arities,
				),
			)
			if any(
				self._literal_matches_expected_signature(
					literal,
					expected_literal,
					parameter_bindings,
				)
				for literal in candidate_literals
			):
				visiting.remove(visit_key)
				return True

			if step.kind == "compound":
				child_task = task_lookup.get(step.task_name)
				rebound_step_args = tuple(
					parameter_bindings.get(arg, arg)
					for arg in step.args
				)
				child_bindings = {
					parameter: arg
					for parameter, arg in zip(
						child_task.parameters if child_task is not None else (),
						rebound_step_args,
					)
				}
				for child_method in methods_by_task.get(step.task_name, []):
					if self._method_constructively_supports_literal(
						child_method,
						expected_literal,
						methods_by_task,
						task_lookup,
						action_schemas,
						predicate_arities,
						parameter_bindings=child_bindings,
						visiting=visiting,
					):
						visiting.remove(visit_key)
						return True

		visiting.remove(visit_key)
		return False

	def _step_effect_literals(
		self,
		step: HTNMethodStep,
		task_lookup: Dict[str, HTNTask],
		action_schemas: Dict[str, Any],
		predicate_arities: Dict[str, int],
	) -> Tuple[HTNLiteral, ...]:
		if step.kind == "primitive":
			effects = list(step.effects)
			seen_signatures = {
				self._literal_signature(literal)
				for literal in effects
			}
			action_schema = self._resolve_action_schema(step, action_schemas)
			if action_schema is None:
				return tuple(effects)
			for literal in self._materialise_action_literals(
				action_schema.effects,
				action_schema.parameters,
				step.args,
			):
				signature = self._literal_signature(literal)
				if signature in seen_signatures:
					continue
				seen_signatures.add(signature)
				effects.append(literal)
			return tuple(effects)
		if step.effects:
			return tuple(step.effects)

		task_schema = task_lookup.get(step.task_name)
		headline_literal = self._materialise_task_headline_literal(
			task_schema,
			bound_args=step.args,
			predicate_arities=predicate_arities,
		)
		if headline_literal is None:
			return ()
		return (headline_literal,)

	@staticmethod
	def _literal_signature(literal: HTNLiteral) -> str:
		return literal.to_signature()

	@classmethod
	def _literal_matches_expected_signature(
		cls,
		candidate: HTNLiteral,
		expected: HTNLiteral,
		parameter_bindings: Dict[str, str],
	) -> bool:
		if candidate.predicate != expected.predicate:
			return False
		if candidate.is_positive != expected.is_positive:
			return False
		if len(candidate.args) != len(expected.args):
			return False

		for raw_candidate_arg, expected_arg in zip(candidate.args, expected.args):
			is_bound_from_parent = raw_candidate_arg in parameter_bindings
			candidate_arg = parameter_bindings.get(raw_candidate_arg, raw_candidate_arg)
			if is_bound_from_parent:
				if candidate_arg != expected_arg:
					return False
				continue
			if candidate_arg != expected_arg:
				return False

		return True

	@staticmethod
	def _looks_like_variable(symbol: str) -> bool:
		text = str(symbol or "").strip()
		return bool(text) and (text.startswith("?") or text[0].isupper())

	@staticmethod
	def _strip_variable_type_annotation(symbol: str) -> str:
		text = str(symbol).strip()
		if not text:
			return text
		if text.startswith("?") and ":" in text:
			return text.split(":", 1)[0].strip()
		return text

	@staticmethod
	def _domain_schema_hint() -> str:
		return (
			'{"compound_tasks":[{"name":"TASK","parameters":["?x:type"]}],"methods":'
			'[{"method_name":"m_task_already_satisfied","task_name":"TASK","parameters":["?x:type"],'
			'"task_args":["?x"],"context":["goal(?x)"],"subtasks":[],"ordering":[]},'
			'{"method_name":"m_task_constructive","task_name":"TASK","parameters":["?x:type"],'
			'"task_args":["?x"],"context":["pre(?x)"],'
			'"subtasks":[{"step_id":"s1","task_name":"child","args":["?x"],"kind":"primitive"}],'
			'"ordering":[]}]}'			
		)

	@staticmethod
	def _strip_code_fences(text: str) -> str:
		if not text.startswith("```"):
			return text
		first_newline = text.find("\n")
		closing_fence = text.rfind("```")
		if first_newline == -1 or closing_fence <= first_newline:
			return text
		return text[first_newline + 1:closing_fence].strip()
