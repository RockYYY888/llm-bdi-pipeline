from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from utils.hddl_parser import HDDLParser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_QUERY_MANIFEST_PATH = (
	PROJECT_ROOT / "src" / "benchmark_data" / "official_problem_queries.json"
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
	variable, fall back to the full problem inventory so the natural-language
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


def build_case_from_problem(problem_path: Path) -> Dict[str, Any] | None:
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
		"required_task_clauses": task_clauses,
		"problem_file": str(problem_path.resolve()),
		"minimum_action_count": 1,
		"description": f"Auto-generated from {problem_path.name} ({problem.name})",
	}


def build_benchmark_query_manifest() -> Dict[str, Any]:
	manifest: Dict[str, Any] = {
		"version": 2,
		"protocol_document": "docs/nl_instruction_template.md",
		"generator": "canonical_root_task_query_v2",
		"domains": {},
	}
	for domain_key, problem_dir in DEFAULT_BENCHMARK_QUERY_DOMAIN_PROBLEM_DIRS.items():
		pattern = DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS[domain_key]
		cases: Dict[str, Dict[str, Any]] = {}
		for index, problem_path in enumerate(sorted(problem_dir.glob(pattern)), start=1):
			case = build_case_from_problem(problem_path)
			if case is None:
				continue
			stored_case = dict(case)
			stored_case["problem_file"] = problem_path.relative_to(PROJECT_ROOT).as_posix()
			cases[f"query_{index}"] = stored_case
		manifest["domains"][domain_key] = {
			"problem_dir": problem_dir.relative_to(PROJECT_ROOT).as_posix(),
			"problem_pattern": pattern,
			"cases": cases,
		}
	return manifest


def write_benchmark_query_manifest(
	manifest_path: Path | None = None,
) -> Path:
	target_path = manifest_path or DEFAULT_BENCHMARK_QUERY_MANIFEST_PATH
	target_path.parent.mkdir(parents=True, exist_ok=True)
	target_path.write_text(
		json.dumps(build_benchmark_query_manifest(), indent=2) + "\n",
		encoding="utf-8",
	)
	return target_path


def load_benchmark_query_manifest(
	manifest_path: Path | None = None,
) -> Dict[str, Any]:
	target_path = manifest_path or DEFAULT_BENCHMARK_QUERY_MANIFEST_PATH
	if not target_path.exists():
		raise FileNotFoundError(
			f"Missing benchmark query manifest: {target_path}. "
			"Regenerate it with utils.benchmark_query_manifest.write_benchmark_query_manifest().",
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
	manifest_path: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
	del pattern  # The manifest already fixes the canonical pattern per domain.
	domain_key = _infer_domain_key_from_problem_dir(problem_dir)
	manifest = load_benchmark_query_manifest(manifest_path)
	domain_record = manifest["domains"].get(domain_key, {})
	case_items = list((domain_record.get("cases") or {}).items())
	if limit > 0:
		case_items = case_items[:limit]

	normalised_cases: Dict[str, Dict[str, Any]] = {}
	for query_id, stored_case in case_items:
		case = dict(stored_case)
		case["problem_file"] = str((PROJECT_ROOT / case["problem_file"]).resolve())
		normalised_cases[query_id] = case
	return normalised_cases
