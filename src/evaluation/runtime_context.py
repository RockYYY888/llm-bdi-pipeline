"""
Runtime domain context helpers for plan-library evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from domain_model.materialization import (
	write_generated_domain_file,
	write_masked_domain_file,
)
from method_library.synthesis.naming import sanitize_identifier
from method_library.synthesis.schema import HTNMethodLibrary
from evaluation.domain_selection import (
	EVALUATION_DOMAIN_SOURCE_GENERATED,
	EvaluationDomainContext,
	normalize_evaluation_domain_source,
)
from evaluation.artifacts import TemporalGroundingResult
from evaluation.goal_grounding.grounding_map import GroundingMap
from plan_library.artifacts import PlanLibraryArtifactBundle
from utils.hddl_condition_parser import HDDLConditionParser
from utils.hddl_parser import HDDLParser


class EvaluationTypeResolutionError(RuntimeError):
	"""Raised when evaluation-time type inference is ambiguous or inconsistent."""


def build_type_parent_map_for_domain(domain: Any) -> Dict[str, Optional[str]]:
	"""Build a child-to-parent type map from a parsed HDDL domain."""

	tokens = [
		token.strip()
		for token in (getattr(domain, "types", []) or [])
		if token and token.strip()
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


def sanitize_name(name: str) -> str:
	"""Return the AgentSpeak-safe surface form used by the runtime."""

	return str(name).replace("-", "_")


def parameter_type(parameter: str) -> str:
	"""Extract the declared type from a compact HDDL parameter token."""

	if "-" not in parameter:
		return "object"
	type_name = parameter.split("-", 1)[1].strip()
	return type_name or "object"


def require_known_type(type_name: str, source: str, domain_type_names: Set[str]) -> str:
	"""Validate that a type name is declared in the current evaluation domain."""

	if type_name in domain_type_names:
		return type_name
	raise EvaluationTypeResolutionError(
		f"{source} references unknown type '{type_name}'. "
		f"Known types: {sorted(domain_type_names)}",
	)


def predicate_type_map_for_domain(
	domain: Any,
	domain_type_names: Set[str],
) -> Dict[str, Tuple[str, ...]]:
	"""Build predicate argument type signatures for a parsed domain."""

	mapping: Dict[str, Tuple[str, ...]] = {}
	for predicate in getattr(domain, "predicates", []):
		mapping[predicate.name] = tuple(
			require_known_type(
				parameter_type(parameter),
				f"Predicate '{predicate.name}'",
				domain_type_names,
			)
			for parameter in predicate.parameters
		)
	return mapping


def action_type_map_for_domain(
	domain: Any,
	domain_type_names: Set[str],
) -> Dict[str, Tuple[str, ...]]:
	"""Build action argument type signatures for a parsed domain."""

	mapping: Dict[str, Tuple[str, ...]] = {}
	for action in getattr(domain, "actions", []):
		type_signature = tuple(
			require_known_type(
				parameter_type(parameter),
				f"Action '{action.name}'",
				domain_type_names,
			)
			for parameter in action.parameters
		)
		mapping[action.name] = type_signature
		mapping[sanitize_name(action.name)] = type_signature
	return mapping


def task_type_map_for_domain(
	domain: Any,
	domain_type_names: Set[str],
) -> Dict[str, Tuple[str, ...]]:
	"""Build compound-task argument type signatures for a parsed domain."""

	mapping: Dict[str, Tuple[str, ...]] = {}
	for task in getattr(domain, "tasks", []):
		type_signature = tuple(
			require_known_type(
				parameter_type(parameter),
				f"Task '{task.name}'",
				domain_type_names,
			)
			for parameter in task.parameters
		)
		mapping[task.name] = type_signature
		mapping[sanitize_name(task.name)] = type_signature
	return mapping


def is_subtype(
	candidate_type: str,
	expected_type: str,
	type_parent_map: Dict[str, Optional[str]],
) -> bool:
	"""Return whether candidate_type equals or inherits from expected_type."""

	if candidate_type == expected_type:
		return True
	if candidate_type not in type_parent_map or expected_type not in type_parent_map:
		return False
	cursor = type_parent_map.get(candidate_type)
	visited = {candidate_type}
	while cursor is not None and cursor not in visited:
		if cursor == expected_type:
			return True
		visited.add(cursor)
		cursor = type_parent_map.get(cursor)
	return False


def validate_problem_arguments_against_signature(
	*,
	args: Sequence[str],
	signature: Sequence[str],
	object_types: Dict[str, str],
	domain_type_names: Set[str],
	type_parent_map: Dict[str, Optional[str]],
	scope: str,
) -> None:
	"""Validate one grounded problem tuple against its declared type signature."""

	if len(args) != len(signature):
		raise ValueError(
			f"{scope}: arity mismatch (args={len(args)}, signature={len(signature)}).",
		)
	for arg, expected_type in zip(args, signature):
		actual_type = object_types.get(arg)
		if actual_type is None:
			continue
		if actual_type not in domain_type_names:
			raise ValueError(f"{scope}: object '{arg}' uses unknown type '{actual_type}'.")
		if not is_subtype(actual_type, expected_type, type_parent_map):
			raise ValueError(
				f"{scope}: object '{arg}' has type '{actual_type}', expected "
				f"'{expected_type}'.",
			)


def validate_problem_domain_compatibility(
	*,
	problem: Any,
	domain_type_names: Set[str],
	type_parent_map: Dict[str, Optional[str]],
	predicate_type_map: Dict[str, Tuple[str, ...]],
	task_type_map: Dict[str, Tuple[str, ...]],
) -> None:
	"""Validate that the loaded problem can run against the evaluation domain."""

	if problem is None:
		return

	unknown_problem_types = sorted(
		{
			type_name
			for type_name in problem.object_types.values()
			if type_name not in domain_type_names
		},
	)
	if unknown_problem_types:
		raise ValueError(
			"problem_file references object types missing from domain_file: "
			f"{unknown_problem_types}",
		)

	for task in problem.htn_tasks:
		if task.task_name not in task_type_map:
			raise ValueError(
				"problem_file HTN task is not declared in domain_file: "
				f"{task.task_name}",
			)
		validate_problem_arguments_against_signature(
			args=task.args,
			signature=task_type_map[task.task_name],
			object_types=problem.object_types,
			domain_type_names=domain_type_names,
			type_parent_map=type_parent_map,
			scope=f"problem HTN task '{task.to_signature()}'",
		)

	for fact in (*problem.init_facts, *problem.goal_facts):
		if fact.predicate not in predicate_type_map:
			raise ValueError(
				"problem_file predicate is not declared in domain_file: "
				f"{fact.predicate}",
			)
		validate_problem_arguments_against_signature(
			args=fact.args,
			signature=predicate_type_map[fact.predicate],
			object_types=problem.object_types,
			domain_type_names=domain_type_names,
			type_parent_map=type_parent_map,
			scope=f"problem fact '{fact.to_signature()}'",
		)


def typed_object_entries(
	object_pool: Sequence[str],
	object_types: Dict[str, str],
) -> Tuple[Tuple[str, str], ...]:
	"""Return typed HDDL object entries, rejecting untyped runtime objects."""

	missing = [obj for obj in object_pool if obj not in object_types]
	if missing:
		raise EvaluationTypeResolutionError(
			"Missing resolved object types for evaluation problem export: "
			+ ", ".join(sorted(missing)),
		)
	return tuple((obj, object_types[obj]) for obj in object_pool)


def task_event_grounding_map(grounding_result: TemporalGroundingResult) -> GroundingMap:
	"""Build a task-event map from a validated LTLf grounding result."""

	grounding_map = GroundingMap()
	for task_event in grounding_result.subgoals:
		grounding_map.add_atom(
			symbol=str(task_event.subgoal_id),
			predicate=str(task_event.task_name),
			args=[str(arg) for arg in task_event.args],
		)
	return grounding_map


def method_library_source_task_name_map(
	method_library: HTNMethodLibrary,
) -> Dict[str, str]:
	"""Map source task names to internal method-library task names."""

	mapping: Dict[str, str] = {}
	for task in tuple(getattr(method_library, "compound_tasks", ()) or ()):
		internal_name = str(getattr(task, "name", "") or "").strip()
		source_name = str(getattr(task, "source_name", "") or "").strip()
		if internal_name:
			mapping.setdefault(internal_name, internal_name)
		if source_name:
			mapping.setdefault(source_name, internal_name or source_name)
	return mapping


def source_task_name_for_task(task_name: str, method_library: HTNMethodLibrary) -> str:
	"""Return the original benchmark task name for a method-library task."""

	task = method_library.task_for_name(str(task_name).strip())
	if task is None:
		return str(task_name).strip()
	return str(getattr(task, "source_name", None) or getattr(task, "name", "") or "").strip()


def render_problem_fact(fact: Any) -> str:
	"""Render one parsed HDDL fact in runtime seed-belief form."""

	inner = fact.predicate
	if fact.args:
		inner = f"{inner} {' '.join(fact.args)}"
	return f"({inner})" if fact.is_positive else f"(not ({inner}))"


def planner_action_schemas_for_domain(domain: Any) -> Tuple[Dict[str, Any], ...]:
	"""Parse HDDL action schemas into the Jason environment adapter format."""

	parser = HDDLConditionParser()
	schemas = []
	for action in getattr(domain, "actions", []):
		parsed = parser.parse_action(action)
		schemas.append(
			{
				"functor": sanitize_name(action.name),
				"source_name": action.name,
				"parameters": list(parsed.parameters),
				"preconditions": [
					{
						"predicate": literal.predicate,
						"args": list(literal.args),
						"is_positive": literal.is_positive,
					}
					for literal in parsed.preconditions
				],
				"precondition_clauses": [
					[
						{
							"predicate": literal.predicate,
							"args": list(literal.args),
							"is_positive": literal.is_positive,
						}
						for literal in clause
					]
					for clause in parsed.precondition_clauses
				],
				"effects": [
					{
						"predicate": literal.predicate,
						"args": list(literal.args),
						"is_positive": literal.is_positive,
					}
					for literal in parsed.effects
				],
			},
		)
	return tuple(schemas)


def resolve_evaluation_domain_context(
	*,
	source_domain_file: str,
	source_domain: Any,
	artifact_bundle: Optional[PlanLibraryArtifactBundle],
	evaluation_domain_source: str,
	output_dir: Optional[Path],
	project_root: Path,
) -> EvaluationDomainContext:
	"""Resolve the HDDL domain used by evaluation-time execution."""

	source = normalize_evaluation_domain_source(evaluation_domain_source)
	domain_file = str(Path(source_domain_file).resolve())
	domain = source_domain

	if source == EVALUATION_DOMAIN_SOURCE_GENERATED:
		generated_domain_path = None
		if output_dir is not None:
			generated_domain_path = (
				Path(output_dir).resolve()
				/ "evaluation_domain_artifact"
				/ "generated_domain.hddl"
			)
		elif (
			generated_domain_path is None
			and artifact_bundle is not None
			and str(getattr(artifact_bundle, "artifact_root", "") or "").strip()
		):
			artifact_domain_path = (
				Path(str(artifact_bundle.artifact_root)).expanduser().resolve()
				/ "generated_domain.hddl"
			)
			generated_domain_path = artifact_domain_path
		if generated_domain_path is None or not generated_domain_path.exists():
			generated_domain_path = materialize_generated_evaluation_domain_artifact(
				source_domain_file=source_domain_file,
				source_domain=source_domain,
				artifact_bundle=artifact_bundle,
				output_dir=output_dir,
				project_root=project_root,
			)
		domain_file = str(generated_domain_path)
		domain = HDDLParser.parse_domain(domain_file)

	type_parent_map = build_type_parent_map_for_domain(domain)
	domain_type_names = set(type_parent_map.keys())
	return EvaluationDomainContext(
		source=source,
		domain_file=domain_file,
		domain=domain,
		type_parent_map=type_parent_map,
		predicate_type_map=predicate_type_map_for_domain(domain, domain_type_names),
		action_type_map=action_type_map_for_domain(domain, domain_type_names),
		task_type_map=task_type_map_for_domain(domain, domain_type_names),
	)


def materialize_generated_evaluation_domain_artifact(
	*,
	source_domain_file: str,
	source_domain: Any,
	artifact_bundle: Optional[PlanLibraryArtifactBundle],
	output_dir: Optional[Path],
	project_root: Path,
) -> Path:
	"""Materialize generated_domain.hddl for evaluation_domain_source=generated."""

	if artifact_bundle is None:
		raise ValueError(
			"EVALUATION_DOMAIN_SOURCE=generated requires a plan-library artifact bundle in order "
			"to materialize generated_domain.hddl.",
		)

	artifact_root = (
		Path(output_dir).resolve() / "evaluation_domain_artifact"
		if output_dir is not None
		else (
			Path(str(artifact_bundle.artifact_root)).expanduser().resolve()
			if getattr(artifact_bundle, "artifact_root", None)
			else (
				project_root
				/ "tmp"
				/ "evaluation"
				/ sanitize_identifier(source_domain.name)
				/ "evaluation_domain_artifact"
			)
		)
	)
	artifact_root.mkdir(parents=True, exist_ok=True)

	masked_domain_path = artifact_root / "masked_domain.hddl"
	if masked_domain_path.exists():
		masked_domain_text = masked_domain_path.read_text()
	else:
		masked_inputs = write_masked_domain_file(
			official_domain_file=source_domain_file,
			output_path=masked_domain_path,
		)
		masked_domain_text = str(masked_inputs["masked_domain_text"])

	generated_outputs = write_generated_domain_file(
		masked_domain_text=masked_domain_text,
		domain=source_domain,
		method_library=artifact_bundle.method_library,
		output_path=artifact_root / "generated_domain.hddl",
	)
	return Path(str(generated_outputs["generated_domain_file"])).resolve()
