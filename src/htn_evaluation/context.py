"""
Evaluation-only context for planner-based HTN benchmark execution.

This module owns the state and helpers required by the Hierarchical Task
Network evaluation track. It must not import or instantiate broader pipeline
orchestration.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from execution_logging.execution_logger import ExecutionLogger
from planning.official_benchmark import (
	OFFICIAL_BENCHMARK_CPU_COUNT,
	OFFICIAL_BENCHMARK_MEMORY_LIMIT_MIB,
	OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS,
)
from planning.problem_structure import ProblemStructure, ProblemStructureAnalyzer
from planning.representations import PlanningRepresentationBuilder, RepresentationBuildResult
from utils.config import Config, get_config
from utils.hddl_parser import HDDLParser


class HTNEvaluationContext:
	"""Minimal runtime context for planner-based HTN evaluation."""

	OFFICIAL_PROBLEM_ROOT_PLANNING_TIMEOUT_SECONDS = OFFICIAL_BENCHMARK_PLANNING_TIMEOUT_SECONDS
	OFFICIAL_PROBLEM_ROOT_MEMORY_LIMIT_MIB = OFFICIAL_BENCHMARK_MEMORY_LIMIT_MIB
	OFFICIAL_PROBLEM_ROOT_CPU_COUNT = OFFICIAL_BENCHMARK_CPU_COUNT

	def __init__(
		self,
		*,
		domain_file: str,
		problem_file: str,
	) -> None:
		if not domain_file:
			raise ValueError("domain_file is required for HTN evaluation.")
		if not problem_file:
			raise ValueError("problem_file is required for HTN evaluation.")

		self.config: Config = get_config()
		self.project_root = Path(__file__).resolve().parents[2]
		self.logger = ExecutionLogger(logs_dir=str(self.project_root / "artifacts" / "runs"))
		self.domain_file = str(domain_file)
		self.problem_file = str(problem_file)
		self.domain = HDDLParser.parse_domain(self.domain_file)
		self.problem = HDDLParser.parse_problem(self.problem_file)
		self.output_dir: Optional[Path] = None
		self._problem_structure_analyzer = ProblemStructureAnalyzer()
		self.type_parent_map = self._build_type_parent_map()
		self.domain_type_names = set(self.type_parent_map.keys())
		self.predicate_type_map = self._predicate_type_map()
		self.task_type_map = self._task_type_map()
		self._subtype_check_cache: Dict[Tuple[str, str], bool] = {}
		self._validate_problem_domain_compatibility()

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
	def _timing_breakdown_without_total(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
		if not profile:
			return {}
		return {
			str(key): value
			for key, value in profile.items()
			if key != "total_seconds" and value is not None
		}

	def _official_problem_root_structure_analysis(self) -> ProblemStructure:
		return self._problem_structure_analyzer.analyze(
			domain=self.domain,
			problem=self.problem,
		)

	def _problem_root_task_network_ordering_edges(self) -> Tuple[Tuple[str, str], ...]:
		return self._problem_structure_analyzer.root_ordering_edges(self.problem)

	def _problem_root_task_network_is_totally_ordered(self) -> bool:
		return self._official_problem_root_structure_analysis().root_task_network.is_total_order

	def _problem_root_task_network_requires_linearizer(self) -> bool:
		return self._official_problem_root_structure_analysis().requires_linearization

	def _build_problem_representations(
		self,
		timeout_seconds: Optional[float] = None,
	) -> RepresentationBuildResult:
		representation_root = (
			Path(self.output_dir).resolve() / "representations"
			if self.output_dir is not None
			else self.project_root / "artifacts" / "runs" / "tmp_representations"
		)
		builder = PlanningRepresentationBuilder(workspace=representation_root)
		return builder.build(
			domain=self.domain,
			problem=self.problem,
			domain_file=self.domain_file,
			problem_file=self.problem_file,
			timeout_seconds=timeout_seconds,
		)

	def _official_problem_root_planning_timeout_seconds(
		self,
		timeout_seconds: Optional[float] = None,
	) -> float:
		if timeout_seconds is not None:
			return max(float(timeout_seconds), 1.0)
		return float(self.OFFICIAL_PROBLEM_ROOT_PLANNING_TIMEOUT_SECONDS)

	def _official_problem_root_resource_profile(self) -> Dict[str, Any]:
		return {
			"planning_timeout_seconds": float(
				self.OFFICIAL_PROBLEM_ROOT_PLANNING_TIMEOUT_SECONDS,
			),
			"memory_limit_mib": int(self.OFFICIAL_PROBLEM_ROOT_MEMORY_LIMIT_MIB),
			"cpu_count": int(self.OFFICIAL_PROBLEM_ROOT_CPU_COUNT),
		}

	@staticmethod
	def _rewrite_artifact_root_paths(value: Any, source_root: Path, target_root: Path) -> Any:
		if isinstance(value, dict):
			return {
				key: HTNEvaluationContext._rewrite_artifact_root_paths(
					item,
					source_root,
					target_root,
				)
				for key, item in value.items()
			}
		if isinstance(value, list):
			return [
				HTNEvaluationContext._rewrite_artifact_root_paths(
					item,
					source_root,
					target_root,
				)
				for item in value
			]
		if isinstance(value, tuple):
			return tuple(
				HTNEvaluationContext._rewrite_artifact_root_paths(
					item,
					source_root,
					target_root,
				)
				for item in value
			)
		if not isinstance(value, str):
			return value
		candidate = Path(value)
		if not candidate.is_absolute():
			return value
		try:
			relative = candidate.resolve().relative_to(source_root.resolve())
		except Exception:
			return value
		rewritten = (target_root / relative).resolve()
		if rewritten.exists():
			return str(rewritten)
		return value

	def _merge_primary_planner_output_dir(self, source_root: Path) -> None:
		target_root = Path(self.output_dir or "").resolve()
		target_root.mkdir(parents=True, exist_ok=True)
		if not source_root.exists():
			return
		try:
			source_children = list(source_root.iterdir())
		except PermissionError:
			return
		for child in source_children:
			if child.is_dir():
				continue
			if not (
				child.name.startswith("plan_solve_")
				or child.name.startswith("plan_verification_")
				or child.suffix.lower() in {".json", ".txt"}
			):
				continue
			destination = target_root / child.name
			destination.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(child, destination)

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
				self._parameter_type(parameter)
				for parameter in predicate.parameters
			)
			for predicate in getattr(self.domain, "predicates", [])
		}

	def _task_type_map(self) -> Dict[str, Tuple[str, ...]]:
		return {
			task.name: tuple(
				self._parameter_type(parameter)
				for parameter in task.parameters
			)
			for task in getattr(self.domain, "tasks", [])
		}

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

	def _validate_problem_arguments_against_signature(
		self,
		*,
		args: Sequence[str],
		signature: Sequence[str],
		object_types: Dict[str, str],
		scope: str,
	) -> None:
		if len(args) != len(signature):
			raise ValueError(
				f"{scope}: arity mismatch (args={len(args)}, signature={len(signature)}).",
			)
		for arg, expected_type in zip(args, signature):
			actual_type = object_types.get(arg)
			if actual_type is None:
				continue
			if actual_type not in self.domain_type_names:
				raise ValueError(
					f"{scope}: object '{arg}' uses unknown type '{actual_type}'.",
				)
			if not self._is_subtype(actual_type, expected_type):
				raise ValueError(
					f"{scope}: object '{arg}' has type '{actual_type}', expected "
					f"'{expected_type}'.",
				)

	def _validate_problem_domain_compatibility(self) -> None:
		domain_task_names = set(self.task_type_map.keys())
		domain_predicate_names = set(self.predicate_type_map.keys())

		unknown_problem_types = sorted(
			{
				type_name
				for type_name in self.problem.object_types.values()
				if type_name not in self.domain_type_names
			},
		)
		if unknown_problem_types:
			raise ValueError(
				"problem_file references object types missing from domain_file: "
				f"{unknown_problem_types}",
			)

		for task in self.problem.htn_tasks:
			if task.task_name not in domain_task_names:
				raise ValueError(
					"problem_file HTN task is not declared in domain_file: "
					f"{task.task_name}",
				)
			self._validate_problem_arguments_against_signature(
				args=task.args,
				signature=self.task_type_map[task.task_name],
				object_types=self.problem.object_types,
				scope=f"problem HTN task '{task.to_signature()}'",
			)

		for fact in (*self.problem.init_facts, *self.problem.goal_facts):
			if fact.predicate not in domain_predicate_names:
				raise ValueError(
					"problem_file predicate is not declared in domain_file: "
					f"{fact.predicate}",
				)
			self._validate_problem_arguments_against_signature(
				args=fact.args,
				signature=self.predicate_type_map[fact.predicate],
				object_types=self.problem.object_types,
				scope=f"problem fact '{fact.to_signature()}'",
			)
