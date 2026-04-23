"""
Method-library validation orchestration.

This validation layer is a lightweight domain-only preflight. It checks that the
generated method library is structurally complete and syntactically legal, but
it does not solve benchmark problem instances.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from method_library.context import MethodLibrarySynthesisContext
from method_library.validation.minimal_validation import (
	validate_decomposition_admissibility,
	validate_signature_conformance,
	validate_typed_structural_soundness,
)
from utils.hddl_parser import HDDLParser


class MethodLibraryValidator:
	"""Run one structural-admissibility preflight per declared compound task."""

	def __init__(self, pipeline_context: MethodLibrarySynthesisContext) -> None:
		self.context = pipeline_context

	def validate(self, method_library, *, generated_domain_file: str | None = None):
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

			layer_results = self._build_layer_results(
				method_library,
				generated_domain_file=generated_domain_file,
			)
			gate_passed = all(bool(layer.get("passed")) for layer in layer_results.values())
			gate_cases = self.build_cases(method_library) if gate_passed else ()
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
						"validation_mode": "structural_admissibility",
					}
				)

			summary = {
				"gate_type": "domain_complete",
				"gate_profile": "structural_admissibility",
				"signature_conformance": layer_results["signature_conformance"],
				"typed_structural_soundness": layer_results["typed_structural_soundness"],
				"decomposition_admissibility": layer_results["decomposition_admissibility"],
				"materialized_parseability": layer_results["materialized_parseability"],
				"layers": layer_results,
				"declared_compound_task_count": len(declared_compound_names),
				"validated_task_count": len(gate_results),
				"validated_tasks": [record["task_name"] for record in gate_results],
				"undefined_child_task_count": len(undefined_child_names),
				"missing_declared_task_count": len(missing_tasks),
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
					"gate_profile": "structural_admissibility",
					"validated_task_count": len(gate_results),
				},
			)
			status = "Success" if gate_passed else "Failed"
			self.context.logger.log_domain_gate(
				summary,
				status,
				error=None if gate_passed else self._first_failure_reason(layer_results),
				metadata={
					"gate_type": "domain_complete",
					"gate_profile": "structural_admissibility",
					"validated_task_count": len(gate_results),
				},
			)

			if not gate_passed:
				print(f"✗ Domain gate failed: {self._first_failure_reason(layer_results)}")
				return None
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

	def _build_layer_results(
		self,
		method_library,
		*,
		generated_domain_file: str | None,
	) -> Dict[str, Dict[str, Any]]:
		return {
			"signature_conformance": self._run_layer(
				checked_count=(
					len(getattr(method_library, "compound_tasks", ()) or ())
					+ len(getattr(method_library, "methods", ()) or ())
				),
				check=lambda: validate_signature_conformance(self.context.domain, method_library),
			),
			"typed_structural_soundness": self._run_layer(
				checked_count=len(getattr(method_library, "methods", ()) or ()),
				check=lambda: validate_typed_structural_soundness(
					self.context.domain,
					method_library,
				),
			),
			"decomposition_admissibility": self._run_layer(
				checked_count=(
					len(getattr(method_library, "compound_tasks", ()) or ())
					+ len(getattr(method_library, "methods", ()) or ())
				),
				check=lambda: validate_decomposition_admissibility(
					self.context.domain,
					method_library,
				),
			),
			"materialized_parseability": self._materialized_parseability_layer(
				generated_domain_file,
			),
		}

	@staticmethod
	def _run_layer(*, checked_count: int, check) -> Dict[str, Any]:
		try:
			warnings = check() or []
			return {
				"passed": True,
				"checked_count": checked_count,
				"failure_reason": None,
				"warnings": list(warnings),
			}
		except Exception as exc:
			return {
				"passed": False,
				"checked_count": checked_count,
				"failure_reason": str(exc),
				"warnings": [],
			}

	@staticmethod
	def _materialized_parseability_layer(generated_domain_file: str | None) -> Dict[str, Any]:
		if not str(generated_domain_file or "").strip():
			return {
				"passed": True,
				"checked_count": 0,
				"failure_reason": None,
				"warnings": ["No materialized generated_domain.hddl path was provided."],
			}
		try:
			domain_path = Path(str(generated_domain_file)).expanduser().resolve()
			HDDLParser.parse_domain(str(domain_path))
			return {
				"passed": True,
				"checked_count": 1,
				"failure_reason": None,
				"warnings": [],
			}
		except Exception as exc:
			return {
				"passed": False,
				"checked_count": 1,
				"failure_reason": str(exc),
				"warnings": [],
			}

	@staticmethod
	def _first_failure_reason(layer_results: Dict[str, Dict[str, Any]]) -> str:
		for layer_name, layer_result in layer_results.items():
			if not bool(layer_result.get("passed")):
				return f"{layer_name}: {layer_result.get('failure_reason')}"
		return "unknown gate failure"

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
