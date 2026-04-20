"""
Offline domain-gate orchestration.

The domain gate is a lightweight domain-only preflight. It checks that the
generated method library is structurally complete and syntactically legal, but
it does not solve benchmark problem instances.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from offline_method_generation.context import OfflineSynthesisContext


class OfflineDomainGateValidator:
	"""Run one legality-only preflight per declared compound task."""

	def __init__(self, pipeline_context: OfflineSynthesisContext) -> None:
		self.context = pipeline_context

	def validate(self, method_library):
		print("\n[DOMAIN GATE]")
		print("-"*80)
		stage_start = time.perf_counter()

		try:
			declared_compound_names = {
				str(getattr(task, "name", "")).strip()
				for task in getattr(self.context.domain, "tasks", ())
				if str(getattr(task, "name", "")).strip()
			}
			library_compound_names = {
				str(getattr(task, "name", "")).strip()
				for task in getattr(method_library, "compound_tasks", ())
				if str(getattr(task, "name", "")).strip()
			}
			library_task_name_aliases = self.method_library_source_task_name_map(method_library)
			normalized_library_names = {
				*library_compound_names,
				*library_task_name_aliases.keys(),
				*library_task_name_aliases.values(),
				*(self.context._sanitize_name(name) for name in library_compound_names),
				*(self.context._sanitize_name(name) for name in library_task_name_aliases.keys()),
				*(self.context._sanitize_name(name) for name in library_task_name_aliases.values()),
			}
			missing_tasks = sorted(
				task_name
				for task_name in declared_compound_names
				if task_name not in normalized_library_names
				and self.context._sanitize_name(task_name) not in normalized_library_names
			)
			if missing_tasks:
				raise ValueError(
					"Domain gate missing declared compound tasks: "
					f"{missing_tasks}",
				)

			referenced_child_names = sorted(
				{
					str(getattr(step, "task_name", "")).strip()
					for method in getattr(method_library, "methods", ()) or ()
					for step in (getattr(method, "subtasks", ()) or ())
					if getattr(step, "kind", "") == "compound"
					and str(getattr(step, "task_name", "")).strip()
				}
			)
			undefined_child_names = sorted(
				child_name
				for child_name in referenced_child_names
				if child_name not in normalized_library_names
				and self.context._sanitize_name(child_name) not in normalized_library_names
			)
			if undefined_child_names:
				raise ValueError(
					"Domain gate found undefined compound children: "
					f"{undefined_child_names}",
				)

			gate_cases = self.build_cases(method_library)
			gate_results: List[Dict[str, Any]] = []
			object_scope_seconds = 0.0

			for case in gate_cases:
				task_name = case["task_name"]
				task_args = case["task_args"]
				object_types = case["object_types"]
				object_pool = case["object_pool"]
				self.context._emit_domain_gate_progress(
					f"start task={task_name} arg_count={len(task_args)}",
				)
				object_scope_start = time.perf_counter()
				case_object_pool = list(object_pool)
				case_object_types = dict(object_types)
				object_scope_seconds += time.perf_counter() - object_scope_start
				self.context._emit_domain_gate_progress(
					f"finish task={task_name}",
				)
				gate_results.append(
					{
						"task_name": task_name,
						"task_args": list(task_args),
						"object_types": dict(case_object_types),
						"object_pool": list(case_object_pool),
						"parameter_count": len(task_args),
						"validation_mode": "legality_only",
					}
				)

			summary = {
				"gate_type": "domain_complete",
				"gate_profile": "legality_only",
				"declared_compound_task_count": len(declared_compound_names),
				"validated_task_count": len(gate_results),
				"validated_tasks": [record["task_name"] for record in gate_results],
				"undefined_child_task_count": 0,
				"missing_declared_task_count": 0,
				"query_specific_runtime_records": 0,
				"task_validations": gate_results,
			}
			self.context._record_step_timing(
				"domain_gate",
				stage_start,
				breakdown={
					"object_scope_seconds": object_scope_seconds,
				},
				metadata={
					"gate_type": "domain_complete",
					"gate_profile": "legality_only",
					"validated_task_count": len(gate_results),
				},
			)
			self.context.logger.log_domain_gate(
				summary,
				"Success",
				metadata={
					"gate_type": "domain_complete",
					"gate_profile": "legality_only",
					"validated_task_count": len(gate_results),
				},
			)

			print("✓ Domain gate complete")
			print(f"  Declared compound tasks: {len(declared_compound_names)}")
			print(f"  Validated tasks: {len(gate_results)}")
			return summary
		except Exception as exc:
			self.context._record_step_timing("domain_gate", stage_start)
			self.context.logger.log_domain_gate(
				None,
				"Failed",
				error=str(exc),
				metadata={"gate_type": "domain_complete"},
			)
			print(f"✗ Domain gate failed: {exc}")
			import traceback
			traceback.print_exc()
			return None

	def build_cases(self, method_library) -> Tuple[Dict[str, Any], ...]:
		candidate_root_types = sorted(
			type_name
			for type_name in self.context.domain_type_names
			if self.context.type_parent_map.get(type_name) is None
		)
		default_type_name = (
			"object"
			if "object" in self.context.domain_type_names
			else (candidate_root_types[0] if candidate_root_types else "")
		)
		cases: List[Dict[str, Any]] = []
		for task in getattr(method_library, "compound_tasks", ()) or ():
			task_name = str(getattr(task, "name", "") or "").strip()
			if not task_name:
				continue
			type_signature = tuple(
				type_name or default_type_name
				for type_name in self.context._task_type_signature(task_name, method_library)
			)
			task_args: List[str] = []
			object_pool: List[str] = []
			object_types: Dict[str, str] = {}
			parameters = tuple(getattr(task, "parameters", ()) or ())
			for index, _parameter in enumerate(parameters, start=1):
				object_name = f"gate_{self.context._sanitize_name(task_name)}_{index}"
				type_name = (
					type_signature[index - 1]
					if index - 1 < len(type_signature)
					else default_type_name
				)
				task_args.append(object_name)
				object_pool.append(object_name)
				if type_name:
					object_types[object_name] = type_name
			cases.append(
				{
					"task_name": task_name,
					"task_args": tuple(task_args),
					"type_signature": type_signature,
					"object_pool": tuple(object_pool),
					"object_types": object_types,
				}
			)
		return tuple(cases)

	@staticmethod
	def method_library_source_task_name_map(method_library) -> Dict[str, str]:
		mapping: Dict[str, str] = {}
		for task in tuple(getattr(method_library, "compound_tasks", ()) or ()):
			internal_name = str(getattr(task, "name", "") or "").strip()
			source_name = str(getattr(task, "source_name", "") or "").strip()
			if internal_name:
				mapping.setdefault(internal_name, internal_name)
			if source_name:
				mapping.setdefault(source_name, internal_name or source_name)
		return mapping


DomainGateSummary = Dict[str, Any]
