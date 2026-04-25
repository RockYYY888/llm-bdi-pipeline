from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from method_library import HTNLiteral, HTNMethod, HTNMethodLibrary, HTNMethodStep, HTNTask
from plan_library import (
	AgentSpeakBodyStep,
	AgentSpeakPlan,
	AgentSpeakTrigger,
	PlanLibrary,
	build_plan_library,
)
from plan_library.validation import build_library_validation_record


def _domain():
	return SimpleNamespace(
		name="courier",
		types=("object", "package", "location"),
		predicates=(
			SimpleNamespace(name="loaded", parameters=("?pkg:package",)),
		),
		tasks=(
			SimpleNamespace(name="deliver", parameters=("?pkg:package", "?loc:location")),
		),
		actions=(
			SimpleNamespace(name="load", parameters=("?pkg:package",)),
			SimpleNamespace(name="move", parameters=("?loc:location",)),
			SimpleNamespace(name="drop", parameters=("?pkg:package", "?loc:location")),
		),
	)


def _method_library_with_auxiliary_step_semantics() -> HTNMethodLibrary:
	return HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="deliver", parameters=("?pkg", "?loc"), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="load", parameters=("?pkg",), is_primitive=True),
			HTNTask(name="move", parameters=("?loc",), is_primitive=True),
			HTNTask(name="drop", parameters=("?pkg", "?loc"), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m_deliver_branching",
				task_name="deliver",
				parameters=("?pkg", "?loc"),
				task_args=("?pkg", "?loc"),
				context=(HTNLiteral(predicate="loaded", args=("?pkg",)),),
				subtasks=(
					HTNMethodStep(
						"a",
						"load",
						("?pkg",),
						"primitive",
						action_name="load",
						preconditions=(HTNLiteral(predicate="loaded", args=("?pkg",)),),
						effects=(HTNLiteral(predicate="loaded", args=("?pkg",)),),
					),
					HTNMethodStep("b", "move", ("?loc",), "primitive", action_name="move"),
					HTNMethodStep("c", "drop", ("?pkg", "?loc"), "primitive", action_name="drop"),
				),
				ordering=(("a", "b"), ("a", "c")),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)


def _all_pass_method_validation() -> dict[str, object]:
	return {
		"layers": {
			"signature_conformance": {"passed": True, "warnings": []},
			"typed_structural_soundness": {"passed": True, "warnings": []},
			"decomposition_admissibility": {"passed": True, "warnings": []},
			"materialized_parseability": {"passed": True, "warnings": []},
		},
	}


def test_library_validation_record_checks_plan_library_structure_directly() -> None:
	method_library = _method_library_with_auxiliary_step_semantics()
	plan_library, translation_coverage = build_plan_library(
		domain=_domain(),
		method_library=method_library,
	)

	record = build_library_validation_record(
		domain_name="courier",
		domain=_domain(),
		method_library=method_library,
		plan_library=plan_library,
		translation_coverage=translation_coverage,
		method_validation=_all_pass_method_validation(),
	)

	assert record.passed is True
	assert record.checked_layers == {
		"signature_conformance": True,
		"typed_structure": True,
		"body_symbol_validity": True,
		"groundability_precheck": True,
	}
	assert any("expanded into sequential plan variants" in warning for warning in record.warnings)
	assert any("remain auxiliary in the method library" in warning for warning in record.warnings)


def test_library_validation_record_rejects_invalid_plan_library_symbols() -> None:
	method_library = _method_library_with_auxiliary_step_semantics()
	plan_library = PlanLibrary(
		domain_name="courier",
		plans=(
			AgentSpeakPlan(
				plan_name="invalid_plan",
				trigger=AgentSpeakTrigger(
					event_type="achievement_goal",
					symbol="deliver",
					arguments=("PKG:package", "LOC:location"),
				),
				context=("unknown_predicate(PKG)",),
				body=(
					AgentSpeakBodyStep(kind="action", symbol="missing_action", arguments=("PKG",)),
				),
			),
		),
	)

	record = build_library_validation_record(
		domain_name="courier",
		domain=_domain(),
		method_library=method_library,
		plan_library=plan_library,
		translation_coverage=build_plan_library(
			domain=_domain(),
			method_library=method_library,
		)[1],
		method_validation=_all_pass_method_validation(),
	)

	assert record.passed is False
	assert record.checked_layers["signature_conformance"] is False
	assert record.checked_layers["body_symbol_validity"] is False
	assert record.checked_layers["groundability_precheck"] is False
	assert record.failure_reason is not None


def test_library_validation_record_rejects_jason_functor_collisions() -> None:
	domain = SimpleNamespace(
		name="collision",
		types=("object", "block"),
		predicates=(),
		tasks=(
			SimpleNamespace(name="move-block", parameters=("?block:block",)),
		),
		actions=(
			SimpleNamespace(name="pick-up", parameters=("?block:block",)),
			SimpleNamespace(name="pick_up", parameters=("?block:block",)),
		),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="move-block", parameters=("?block",), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="pick-up", parameters=("?block",), is_primitive=True),
			HTNTask(name="pick_up", parameters=("?block",), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m_move_block",
				task_name="move-block",
				parameters=("?block",),
				task_args=("?block",),
				subtasks=(
					HTNMethodStep("s1", "pick-up", ("?block",), "primitive"),
					HTNMethodStep("s2", "pick_up", ("?block",), "primitive"),
				),
				ordering=(("s1", "s2"),),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)
	plan_library, translation_coverage = build_plan_library(
		domain=domain,
		method_library=method_library,
	)

	record = build_library_validation_record(
		domain_name="collision",
		domain=domain,
		method_library=method_library,
		plan_library=plan_library,
		translation_coverage=translation_coverage,
		method_validation=_all_pass_method_validation(),
	)

	assert record.passed is False
	assert record.checked_layers["signature_conformance"] is False
	assert record.checked_layers["body_symbol_validity"] is False
	assert any("Jason functor collision" in warning for warning in record.warnings)


def test_library_validation_record_rejects_body_variables_without_context_binding() -> None:
	method_library = _method_library_with_auxiliary_step_semantics()
	plan_library = PlanLibrary(
		domain_name="courier",
		plans=(
			AgentSpeakPlan(
				plan_name="unbound_local_variable",
				trigger=AgentSpeakTrigger(
					event_type="achievement_goal",
					symbol="deliver",
					arguments=("PKG:package", "LOC:location"),
				),
				context=("object_type(PKG, package)", "object_type(LOC, location)"),
				body=(
					AgentSpeakBodyStep(kind="action", symbol="move", arguments=("MID",)),
					AgentSpeakBodyStep(kind="action", symbol="drop", arguments=("PKG", "LOC")),
				),
			),
		),
	)

	record = build_library_validation_record(
		domain_name="courier",
		domain=_domain(),
		method_library=method_library,
		plan_library=plan_library,
		translation_coverage=build_plan_library(
			domain=_domain(),
			method_library=method_library,
		)[1],
		method_validation=_all_pass_method_validation(),
	)

	assert record.passed is False
	assert record.checked_layers["groundability_precheck"] is False
	assert any("uses unbound variable 'MID'" in warning for warning in record.warnings)


def test_library_validation_does_not_treat_object_type_as_binding_literal() -> None:
	method_library = _method_library_with_auxiliary_step_semantics()
	plan_library = PlanLibrary(
		domain_name="courier",
		plans=(
			AgentSpeakPlan(
				plan_name="type_guard_only_binding",
				trigger=AgentSpeakTrigger(
					event_type="achievement_goal",
					symbol="deliver",
					arguments=("PKG:package", "LOC:location"),
				),
				context=(
					"object_type(PKG, package)",
					"object_type(LOC, location)",
					"object_type(MID, location)",
				),
				body=(
					AgentSpeakBodyStep(kind="action", symbol="move", arguments=("MID",)),
					AgentSpeakBodyStep(kind="action", symbol="drop", arguments=("PKG", "LOC")),
				),
			),
		),
	)

	record = build_library_validation_record(
		domain_name="courier",
		domain=_domain(),
		method_library=method_library,
		plan_library=plan_library,
		translation_coverage=build_plan_library(
			domain=_domain(),
			method_library=method_library,
		)[1],
		method_validation=_all_pass_method_validation(),
	)

	assert record.passed is False
	assert record.checked_layers["groundability_precheck"] is False
	assert any("uses unbound variable 'MID'" in warning for warning in record.warnings)


def test_library_validation_accepts_action_precondition_bound_variables() -> None:
	domain = SimpleNamespace(
		name="routing",
		types=("object", "vehicle", "location"),
		predicates=(
			SimpleNamespace(name="at", parameters=("?vehicle:vehicle", "?loc:location")),
		),
		tasks=(
			SimpleNamespace(name="dispatch", parameters=("?to:location",)),
		),
		actions=(
			SimpleNamespace(
				name="move",
				parameters=("?vehicle:vehicle", "?from:location", "?to:location"),
				preconditions="(at ?vehicle ?from)",
				effects="(and (not (at ?vehicle ?from)) (at ?vehicle ?to))",
			),
		),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="dispatch", parameters=("?to",), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="move", parameters=("?vehicle", "?from", "?to"), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m-dispatch",
				task_name="dispatch",
				parameters=("?vehicle", "?from", "?to"),
				task_args=("?to",),
				subtasks=(
					HTNMethodStep(
						"s1",
						"move",
						("?vehicle", "?from", "?to"),
						"primitive",
						action_name="move",
					),
				),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)
	plan_library = PlanLibrary(
		domain_name="routing",
		plans=(
			AgentSpeakPlan(
				plan_name="action_precondition_binding",
				trigger=AgentSpeakTrigger(
					event_type="achievement_goal",
					symbol="dispatch",
					arguments=("TO:location",),
				),
				context=(
					"object_type(TO, location)",
					"object_type(VEHICLE, vehicle)",
					"object_type(FROM, location)",
				),
				body=(
					AgentSpeakBodyStep(
						kind="action",
						symbol="move",
						arguments=("VEHICLE", "FROM", "TO"),
					),
				),
			),
		),
	)

	record = build_library_validation_record(
		domain_name="routing",
		domain=domain,
		method_library=method_library,
		plan_library=plan_library,
		translation_coverage=build_plan_library(
			domain=domain,
			method_library=method_library,
		)[1],
		method_validation=_all_pass_method_validation(),
	)

	assert record.passed is True
	assert record.checked_layers["groundability_precheck"] is True
	assert not any("uses unbound variable" in warning for warning in record.warnings)
