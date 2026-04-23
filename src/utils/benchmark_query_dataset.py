from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from utils.hddl_parser import HDDLParser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_QUERY_DATASET_PATH = (
	PROJECT_ROOT / "src" / "benchmark_data" / "benchmark_queries.json"
)
DEFAULT_BENCHMARK_QUERY_DOMAIN_PROBLEM_DIRS: Dict[str, Path] = {
	"blocksworld": PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems",
	"marsrover": PROJECT_ROOT / "src" / "domains" / "marsrover" / "problems",
	"satellite": PROJECT_ROOT / "src" / "domains" / "satellite" / "problems",
	"transport": PROJECT_ROOT / "src" / "domains" / "transport" / "problems",
}
DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS: Dict[str, str] = {
	"blocksworld": "p*.hddl",
	"marsrover": "p*.hddl",
	"satellite": "*.hddl",
	"transport": "p*.hddl",
}


def serialise_nl_list(items: List[str]) -> str:
	if not items:
		return ""
	if len(items) == 1:
		return items[0]
	if len(items) == 2:
		return f"{items[0]} and {items[1]}"
	return f"{', '.join(items[:-1])}, and {items[-1]}"


def serialise_task_clause_sequence(task_clauses: List[str], *, ordered: bool) -> str:
	if not ordered or len(task_clauses) <= 1:
		return serialise_nl_list(task_clauses)
	return ", then ".join(task_clauses)


def task_invocation_to_query_clause(task_name: str, args: List[str]) -> str:
	if not args:
		return f"{task_name}()"
	return f"{task_name}({', '.join(args)})"


def typed_object_phrase(problem: Any) -> str:
	return typed_object_phrase_for_objects(problem, list(problem.objects or ()))


def _is_problem_variable_symbol(symbol: str) -> bool:
	token = str(symbol or "").strip()
	return bool(token) and token.startswith("?")


def query_referenced_problem_objects(problem: Any) -> List[str]:
	"""
	Return the minimal object inventory justified by the root task network.

	If every root-task argument is already grounded, the query only needs the
	objects referenced in those task invocations. When any root task still uses a
	variable, fall back to the full problem inventory so the stored benchmark
	query still exposes the candidate grounded objects required for resolution.
	"""
	referenced_objects: List[str] = []
	seen = set()
	variable_present = False
	for invocation in list(problem.htn_tasks or ()):
		for raw_arg in list(getattr(invocation, "args", ()) or ()):
			arg = str(raw_arg or "").strip()
			if not arg:
				continue
			if _is_problem_variable_symbol(arg):
				variable_present = True
				continue
			if arg in seen:
				continue
			seen.add(arg)
			referenced_objects.append(arg)

	if variable_present or not referenced_objects:
		return list(problem.objects or ())
	return referenced_objects


def typed_object_phrase_for_objects(problem: Any, objects: List[str]) -> str:
	if not objects:
		return "Using the task arguments"

	grouped_objects: Dict[str, List[str]] = {}
	type_order: List[str] = []
	for obj in objects:
		object_type = problem.object_types.get(obj) or "object"
		if object_type not in grouped_objects:
			grouped_objects[object_type] = []
			type_order.append(object_type)
		grouped_objects[object_type].append(obj)

	if len(type_order) == 1 and type_order[0] != "object":
		object_type = type_order[0]
		type_phrase = object_type if object_type.endswith("s") else f"{object_type}s"
		return f"Using {type_phrase} {serialise_nl_list(grouped_objects[object_type])}"

	group_phrases: List[str] = []
	for object_type in type_order:
		members = grouped_objects[object_type]
		if object_type == "object":
			group_phrases.append(serialise_nl_list(members))
			continue
		type_phrase = object_type if len(members) == 1 else (
			object_type if object_type.endswith("s") else f"{object_type}s"
		)
		group_phrases.append(f"{type_phrase} {serialise_nl_list(members)}")
	return f"Using {serialise_nl_list(group_phrases)}"


def build_stored_case_from_problem(problem_path: Path) -> Dict[str, Any] | None:
	"""
	Build one explicit benchmark-query record for benchmark dataset maintenance.

	This helper is not part of the runtime loading path. Runtime benchmark query
	resolution must consume the stored JSON directly rather than reconstructing
	cases from problem files on the fly.
	"""

	problem = HDDLParser.parse_problem(str(problem_path))
	task_clauses = [
		task_invocation_to_query_clause(invocation.task_name, invocation.args)
		for invocation in problem.htn_tasks
	]
	if not task_clauses:
		return None
	query_objects = query_referenced_problem_objects(problem)

	return {
		"instruction": (
			f"{typed_object_phrase_for_objects(problem, query_objects)}, complete the tasks "
			f"{serialise_task_clause_sequence(task_clauses, ordered=problem.htn_ordered)}."
		),
		"problem_file": problem_path.name,
	}


def build_benchmark_query_dataset() -> Dict[str, Any]:
	dataset: Dict[str, Any] = {
		"version": 5,
		"dataset_kind": "stored_benchmark_queries",
		"query_protocol_document": "docs/query_protocol.md",
		"generator": "canonical_root_task_query_v3",
		"domains": {},
	}
	for domain_key, problem_dir in DEFAULT_BENCHMARK_QUERY_DOMAIN_PROBLEM_DIRS.items():
		pattern = DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS[domain_key]
		cases: Dict[str, Dict[str, str]] = {}
		for index, problem_path in enumerate(sorted(problem_dir.glob(pattern)), start=1):
			case = build_stored_case_from_problem(problem_path)
			if case is None:
				continue
			cases[f"query_{index}"] = {
				"instruction": str(case["instruction"]),
				"problem_file": str(case["problem_file"]),
			}
		dataset["domains"][domain_key] = {"cases": cases}
	return dataset


def write_benchmark_query_dataset(
	dataset_path: Path | None = None,
) -> Path:
	target_path = dataset_path or DEFAULT_BENCHMARK_QUERY_DATASET_PATH
	target_path.parent.mkdir(parents=True, exist_ok=True)
	target_path.write_text(
		json.dumps(build_benchmark_query_dataset(), indent=2) + "\n",
		encoding="utf-8",
	)
	return target_path


def load_benchmark_query_dataset(
	dataset_path: Path | None = None,
) -> Dict[str, Any]:
	target_path = dataset_path or DEFAULT_BENCHMARK_QUERY_DATASET_PATH
	if not target_path.exists():
		raise FileNotFoundError(
			f"Missing benchmark query dataset: {target_path}.",
		)
	return json.loads(target_path.read_text(encoding="utf-8"))


def _infer_domain_key_from_problem_dir(problem_dir: Path) -> str:
	resolved_problem_dir = problem_dir.resolve()
	for domain_key, default_dir in DEFAULT_BENCHMARK_QUERY_DOMAIN_PROBLEM_DIRS.items():
		if resolved_problem_dir == default_dir.resolve():
			return domain_key
	raise ValueError(f"Unrecognised benchmark problem directory: {problem_dir}")


def load_problem_query_cases(
	problem_dir: Path,
	*,
	limit: int = 3,
	pattern: str = "p*.hddl",
	dataset_path: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
	del pattern
	domain_key = _infer_domain_key_from_problem_dir(problem_dir)
	dataset = load_benchmark_query_dataset(dataset_path)
	domain_record = dataset["domains"].get(domain_key, {})
	case_items = list((domain_record.get("cases") or {}).items())
	if limit > 0:
		case_items = case_items[:limit]

	normalised_cases: Dict[str, Dict[str, Any]] = {}
	for query_id, stored_case in case_items:
		normalised_cases[query_id] = _normalise_stored_query_case(
			query_id=query_id,
			stored_case=stored_case,
			problem_dir=problem_dir,
		)
	return normalised_cases


def _normalise_stored_query_case(
	*,
	query_id: str,
	stored_case: Any,
	problem_dir: Path,
) -> Dict[str, Any]:
	if not isinstance(stored_case, dict):
		raise ValueError(
			f"Benchmark query '{query_id}' must be an object record with "
			'"instruction" and "problem_file".',
		)

	instruction = str(stored_case.get("instruction") or "").strip()
	if not instruction:
		raise ValueError(f'Benchmark query "{query_id}" is missing a non-empty "instruction".')

	raw_problem_file = str(stored_case.get("problem_file") or "").strip()
	if not raw_problem_file:
		raise ValueError(f'Benchmark query "{query_id}" is missing a non-empty "problem_file".')

	problem_reference = Path(raw_problem_file)
	if problem_reference.is_absolute():
		raise ValueError(
			f'Benchmark query "{query_id}" must use a domain-local relative "problem_file", '
			f"got absolute path '{raw_problem_file}'.",
		)

	resolved_problem_dir = problem_dir.resolve()
	resolved_problem_file = (resolved_problem_dir / problem_reference).resolve()
	try:
		resolved_problem_file.relative_to(resolved_problem_dir)
	except ValueError as exc:
		raise ValueError(
			f'Benchmark query "{query_id}" uses illegal "problem_file" path '
			f"'{raw_problem_file}'.",
		) from exc

	if not resolved_problem_file.exists() or not resolved_problem_file.is_file():
		raise FileNotFoundError(
			f'Benchmark query "{query_id}" references missing problem file '
			f"'{raw_problem_file}'.",
		)

	return {
		"instruction": instruction,
		"problem_file": str(resolved_problem_file),
	}
