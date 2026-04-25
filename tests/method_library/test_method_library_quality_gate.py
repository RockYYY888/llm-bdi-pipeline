from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from method_library.validation.minimal_validation import (
	validate_decomposition_admissibility,
	validate_signature_conformance,
	validate_typed_structural_soundness,
)
from method_library.synthesis.schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
)
from method_library.validation.validator import MethodLibraryValidator


def _domain(
	*,
	tasks=None,
	actions=None,
	predicates=None,
	types=None,
):
	return SimpleNamespace(
		name="quality-gate-test",
		requirements=[],
		types=list(types or ["vehicle", "package", "-", "object"]),
		predicates=list(
			predicates
			or [
				SimpleNamespace(name="at", parameters=["?v - vehicle"]),
				SimpleNamespace(name="stored", parameters=["?p - package"]),
				SimpleNamespace(name="linked", parameters=["?v - vehicle", "?p - package"]),
			],
		),
		tasks=list(
			tasks
			or [
				SimpleNamespace(name="deliver", parameters=["?p - package"]),
			],
		),
		actions=list(
			actions
			or [
				SimpleNamespace(name="load", parameters=["?p - package"]),
				SimpleNamespace(name="move", parameters=["?v - vehicle"]),
			],
		),
		methods=[],
	)


def _compound_task(name: str = "deliver", parameters=("?p:package",)) -> HTNTask:
	return HTNTask(name=name, parameters=tuple(parameters), is_primitive=False)


def _method(
	*,
	method_name: str = "m_deliver_already_satisfied",
	task_name: str = "deliver",
	parameters=("?p:package",),
	task_args=("?p",),
	context=(HTNLiteral("stored", ("?p",)),),
	subtasks=(),
	ordering=(),
) -> HTNMethod:
	return HTNMethod(
		method_name=method_name,
		task_name=task_name,
		parameters=tuple(parameters),
		task_args=tuple(task_args),
		context=tuple(context),
		subtasks=tuple(subtasks),
		ordering=tuple(ordering),
	)


def _library(*, tasks=None, methods=None) -> HTNMethodLibrary:
	return HTNMethodLibrary(
		compound_tasks=list(tasks or [_compound_task()]),
		primitive_tasks=[],
		methods=list(methods or [_method()]),
		target_literals=[],
		target_task_bindings=[],
	)


class _CapturingLogger:
	def __init__(self) -> None:
		self.summary = None
		self.status = None
		self.error = None

	def log_domain_gate(self, summary, status, error=None, metadata=None) -> None:
		_ = metadata
		self.summary = summary
		self.status = status
		self.error = error


class _FakeGateContext:
	def __init__(self, domain) -> None:
		self.domain = domain
		self.output_dir = PROJECT_ROOT / "tests" / "generated" / "quality_gate"
		self.domain_type_names = {"object", "vehicle", "package"}
		self.type_parent_map = {"object": None, "vehicle": "object", "package": "object"}
		self.logger = _CapturingLogger()

	@staticmethod
	def _sanitize_name(value: str) -> str:
		return str(value).strip().replace("-", "_")

	@staticmethod
	def _emit_domain_gate_progress(message: str) -> None:
		_ = message

	@staticmethod
	def _record_step_timing(
		step: str,
		stage_start: float,
		breakdown=None,
		metadata=None,
	) -> None:
		_ = (step, stage_start, breakdown, metadata)

	def _task_type_signature(self, task_name, method_library):
		_ = method_library
		for task in getattr(self.domain, "tasks", ()):
			if getattr(task, "name", None) == task_name:
				return tuple(self._parse_parameter_type(parameter) for parameter in task.parameters)
		return ()

	@staticmethod
	def _parse_parameter_type(parameter: str) -> str:
		text = str(parameter or "").strip()
		if ":" in text:
			return text.split(":", 1)[1].strip()
		if "-" in text:
			return text.split("-", 1)[1].strip()
		return "object"


def _run_gate(method_library: HTNMethodLibrary, *, generated_domain_file=None, domain=None):
	context = _FakeGateContext(domain or _domain())
	result = MethodLibraryValidator(context).validate(
		method_library,
		generated_domain_file=generated_domain_file,
	)
	return result, context.logger


def test_domain_quality_gate_emits_stable_four_layer_report_schema() -> None:
	summary, logger = _run_gate(_library())

	assert logger.status == "Success"
	assert summary is not None
	assert summary["gate_profile"] == "structural_admissibility"
	for layer_name in (
		"signature_conformance",
		"typed_structural_soundness",
		"decomposition_admissibility",
		"materialized_parseability",
	):
		layer = summary[layer_name]
		assert set(layer) == {"passed", "checked_count", "failure_reason", "warnings"}
		assert isinstance(layer["passed"], bool)
		assert isinstance(layer["checked_count"], int)
		assert layer["failure_reason"] is None or isinstance(layer["failure_reason"], str)
		assert isinstance(layer["warnings"], list)


def test_signature_conformance_rejects_undeclared_compound_child() -> None:
	library = _library(
		methods=[
			_method(
				method_name="m_deliver_expand",
				subtasks=(
					HTNMethodStep("s1", "missing_task", ("?p",), "compound"),
				),
			),
		],
	)

	with pytest.raises(ValueError, match="unknown compound"):
		validate_signature_conformance(_domain(), library)


def test_signature_conformance_rejects_undeclared_primitive_action() -> None:
	library = _library(
		methods=[
			_method(
				method_name="m_deliver_expand",
				subtasks=(
					HTNMethodStep("s1", "missing_action", ("?p",), "primitive"),
				),
			),
		],
	)

	with pytest.raises(ValueError, match="unknown primitive"):
		validate_signature_conformance(_domain(), library)


def test_signature_conformance_rejects_undeclared_predicate() -> None:
	library = _library(
		methods=[
			_method(context=(HTNLiteral("missing_predicate", ("?p",)),)),
		],
	)

	with pytest.raises(ValueError, match="unknown predicate"):
		validate_signature_conformance(_domain(), library)


def test_decomposition_admissibility_rejects_missing_method_for_declared_task() -> None:
	domain = _domain(
		tasks=[
			SimpleNamespace(name="deliver", parameters=["?p - package"]),
			SimpleNamespace(name="get_to", parameters=["?v - vehicle"]),
		],
	)
	library = _library(
		tasks=[
			_compound_task("deliver", ("?p:package",)),
			_compound_task("get_to", ("?v:vehicle",)),
		],
		methods=[_method()],
	)

	with pytest.raises(ValueError, match="has no generated methods"):
		validate_decomposition_admissibility(domain, library)


def test_typed_structural_soundness_rejects_explicit_type_conflict() -> None:
	library = _library(
		methods=[
			_method(
				method_name="m_deliver_conflict",
				context=(HTNLiteral("at", ("?p",)),),
			),
		],
	)

	with pytest.raises(ValueError, match="variable typing conflict"):
		validate_typed_structural_soundness(_domain(), library)


def test_typed_structural_soundness_rejects_equality_merged_type_conflict() -> None:
	library = _library(
		methods=[
			_method(
				method_name="m_deliver_equality_conflict",
				parameters=("?p:package", "?v:vehicle"),
				context=(
					HTNLiteral("=", ("?p", "?v")),
					HTNLiteral("at", ("?v",)),
				),
			),
		],
	)

	with pytest.raises(ValueError, match="variable typing conflict"):
		validate_typed_structural_soundness(_domain(), library)


def test_typed_structural_soundness_warns_for_unresolved_variable_type() -> None:
	library = _library(
		methods=[
			_method(
				method_name="m_deliver_expand",
				parameters=("?p:package", "?unused"),
				context=(),
				subtasks=(HTNMethodStep("s1", "load", ("?p",), "primitive"),),
			),
		],
	)

	warnings = validate_typed_structural_soundness(_domain(), library)

	assert any("unresolved variable type" in warning and "?unused" in warning for warning in warnings)


def test_typed_structural_soundness_rejects_cyclic_ordering_graph() -> None:
	steps = (
		HTNMethodStep("s1", "load", ("?p",), "primitive"),
		HTNMethodStep("s2", "load", ("?p",), "primitive"),
	)
	library = _library(
		methods=[
			_method(
				method_name="m_deliver_cycle",
				context=(),
				subtasks=steps,
				ordering=(("s1", "s2"), ("s2", "s1")),
			),
		],
	)

	with pytest.raises(ValueError, match="ordering graph contains a cycle"):
		validate_typed_structural_soundness(_domain(), library)


def test_typed_structural_soundness_accepts_acyclic_partial_order_graph() -> None:
	steps = (
		HTNMethodStep("s1", "load", ("?p",), "primitive"),
		HTNMethodStep("s2", "load", ("?p",), "primitive"),
		HTNMethodStep("s3", "load", ("?p",), "primitive"),
	)
	library = _library(
		methods=[
			_method(
				method_name="m_deliver_partial_order",
				context=(),
				subtasks=steps,
				ordering=(("s1", "s3"), ("s2", "s3")),
			),
		],
	)

	assert validate_typed_structural_soundness(_domain(), library) == []


def test_decomposition_admissibility_rejects_unguarded_empty_method() -> None:
	library = _library(methods=[_method(context=())])

	with pytest.raises(ValueError, match="empty subtasks without an already-satisfied guard"):
		validate_decomposition_admissibility(_domain(), library)


def test_decomposition_admissibility_accepts_guarded_empty_method_structurally() -> None:
	warnings = validate_decomposition_admissibility(_domain(), _library())

	assert warnings == []


def test_domain_quality_gate_rejects_unreparseable_materialized_domain_file(tmp_path: Path) -> None:
	missing_domain_file = tmp_path / "generated_domain.hddl"

	result, logger = _run_gate(_library(), generated_domain_file=str(missing_domain_file))

	assert result is None
	assert logger.status == "Failed"
	assert logger.summary["materialized_parseability"]["passed"] is False
	assert "No such file" in logger.summary["materialized_parseability"]["failure_reason"]


def test_method_library_validation_does_not_import_retired_or_evaluation_modules() -> None:
	validate_parameters = set(inspect.signature(MethodLibraryValidator.validate).parameters)
	assert validate_parameters == {"self", "method_library", "generated_domain_file"}

	source_path = PROJECT_ROOT / "src" / "method_library" / "validation" / "validator.py"
	tree = ast.parse(source_path.read_text())
	imported_modules = []
	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			imported_modules.extend(alias.name for alias in node.names)
		elif isinstance(node, ast.ImportFrom) and node.module:
			imported_modules.append(node.module)

	for module_name in imported_modules:
		assert not module_name.startswith("query_solution_runtime")
		assert not module_name.startswith("htn_evaluation")
		assert not module_name.startswith("pipeline.domain_complete_pipeline")
