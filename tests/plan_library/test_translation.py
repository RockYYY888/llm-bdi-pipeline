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
from plan_library import build_plan_library, render_plan_library_asl
from plan_library.models import PlanLibrary
from tests.support.plan_library_generation_support import (
	DOMAIN_FILES,
	build_official_method_library,
)
from utils.hddl_parser import HDDLParser


def _sample_domain():
	return SimpleNamespace(
		name="courier",
		tasks=(
			SimpleNamespace(
				name="deliver",
				parameters=("?pkg:package", "?loc:location"),
			),
		),
		predicates=(
			SimpleNamespace(name="loaded", parameters=("?pkg:package",)),
		),
		actions=(
			SimpleNamespace(name="load", parameters=("?pkg:package",)),
			SimpleNamespace(name="move", parameters=("?loc:location",)),
			SimpleNamespace(name="drop", parameters=("?pkg:package", "?loc:location")),
		),
	)


def _sample_method_library() -> HTNMethodLibrary:
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
				method_name="m_deliver_serial",
				task_name="deliver",
				parameters=("?pkg", "?loc"),
				task_args=("?pkg", "?loc"),
				context=(
					HTNLiteral(predicate="loaded", args=("?pkg",)),
				),
				subtasks=(
					HTNMethodStep("s1", "load", ("?pkg",), "primitive", action_name="load"),
					HTNMethodStep("s2", "move", ("?loc",), "primitive", action_name="move"),
					HTNMethodStep("s3", "drop", ("?pkg", "?loc"), "primitive", action_name="drop"),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				source_instruction_ids=("query_7",),
			),
			HTNMethod(
				method_name="m_deliver_branching",
				task_name="deliver",
				parameters=("?pkg", "?loc"),
				task_args=("?pkg", "?loc"),
				subtasks=(
					HTNMethodStep("a", "load", ("?pkg",), "primitive", action_name="load"),
					HTNMethodStep("b", "move", ("?loc",), "primitive", action_name="move"),
					HTNMethodStep("c", "drop", ("?pkg", "?loc"), "primitive", action_name="drop"),
				),
				ordering=(("a", "b"), ("a", "c")),
				source_instruction_ids=("query_9",),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)


def test_translation_reports_accepted_and_rejected_methods() -> None:
	plan_library, coverage = build_plan_library(
		domain=_sample_domain(),
		method_library=_sample_method_library(),
	)

	assert len(plan_library.plans) == 3
	assert plan_library.plans[0].plan_name == "m_deliver_serial"
	assert plan_library.plans[0].source_instruction_ids == ("query_7",)
	assert plan_library.plans[0].trigger.arguments == ("PKG:package", "LOC:location")
	assert plan_library.plans[0].context == (
		"object_type(PKG, package)",
		"object_type(LOC, location)",
		"loaded(PKG)",
	)
	assert tuple(step.kind for step in plan_library.plans[0].body) == (
		"action",
		"action",
		"action",
	)
	assert tuple(step.arguments for step in plan_library.plans[0].body) == (
		("PKG",),
		("LOC",),
		("PKG", "LOC"),
	)
	certificate = plan_library.plans[0].binding_certificate
	assert {
		"variable": "PKG",
		"source": "trigger-bound",
		"position": 0,
		"type": "package",
	} in certificate
	assert {
		"variable": "PKG",
		"source": "context-bound",
		"origin": "method_context",
		"literal": "loaded(PKG)",
	} in certificate
	assert all(
		entry.get("binding_status") != "unbound_at_use"
		for entry in certificate
	)
	assert coverage.methods_considered == 2
	assert coverage.accepted_translation == 2
	assert coverage.plans_generated == 3
	assert coverage.unsupported_buckets == {}
	assert coverage.unsupported_methods == ()
	assert PlanLibrary.from_dict(plan_library.to_dict()).plans[0].binding_certificate == certificate

	assert plan_library.plans[1].plan_name == "m_deliver_branching__variant_1"
	assert tuple(step.symbol for step in plan_library.plans[1].body) == (
		"load",
		"move",
		"drop",
	)
	assert plan_library.plans[2].plan_name == "m_deliver_branching__variant_2"
	assert tuple(step.symbol for step in plan_library.plans[2].body) == (
		"load",
		"drop",
		"move",
	)


def test_translation_orders_same_trigger_plans_by_lifted_structure() -> None:
	domain = SimpleNamespace(
		name="routing",
		tasks=(
			SimpleNamespace(name="travel", parameters=("?vehicle:vehicle", "?to:location")),
		),
		predicates=(
			SimpleNamespace(name="at", parameters=("?vehicle:vehicle", "?location:location")),
			SimpleNamespace(name="road", parameters=("?from:location", "?to:location")),
		),
		actions=(
			SimpleNamespace(name="drive", parameters=("?vehicle:vehicle", "?from:location", "?to:location")),
		),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="travel", parameters=("?vehicle", "?to"), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="drive", parameters=("?vehicle", "?from", "?to"), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m-recursive",
				task_name="travel",
				parameters=("?vehicle", "?from", "?mid", "?to"),
				task_args=("?vehicle", "?to"),
				context=(
					HTNLiteral(predicate="at", args=("?vehicle", "?from")),
					HTNLiteral(predicate="road", args=("?from", "?mid")),
					HTNLiteral(predicate="road", args=("?mid", "?to")),
				),
				subtasks=(
					HTNMethodStep("s1", "travel", ("?vehicle", "?mid"), "compound"),
					HTNMethodStep("s2", "drive", ("?vehicle", "?mid", "?to"), "primitive", action_name="drive"),
				),
				ordering=(("s1", "s2"),),
			),
			HTNMethod(
				method_name="m-direct",
				task_name="travel",
				parameters=("?vehicle", "?from", "?to"),
				task_args=("?vehicle", "?to"),
				context=(
					HTNLiteral(predicate="at", args=("?vehicle", "?from")),
					HTNLiteral(predicate="road", args=("?from", "?to")),
				),
				subtasks=(
					HTNMethodStep("s1", "drive", ("?vehicle", "?from", "?to"), "primitive", action_name="drive"),
				),
			),
			HTNMethod(
				method_name="m-already-there",
				task_name="travel",
				parameters=("?vehicle", "?to"),
				task_args=("?vehicle", "?to"),
				context=(
					HTNLiteral(predicate="at", args=("?vehicle", "?to")),
				),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	plan_library, _coverage = build_plan_library(domain=domain, method_library=method_library)

	assert [plan.plan_name for plan in plan_library.plans] == [
		"m-already-there",
		"m-direct",
		"m-recursive",
	]


def test_rendering_emits_structured_agentspeak_library() -> None:
	plan_library, _coverage = build_plan_library(
		domain=_sample_domain(),
		method_library=_sample_method_library(),
	)

	rendered = render_plan_library_asl(plan_library)

	assert "source_instruction_ids=query_7" in rendered
	assert (
		"+!deliver(PKG, LOC) : object_type(PKG, package) & "
		"object_type(LOC, location) & loaded(PKG) <-"
	) in rendered
	assert "plan=m_deliver_branching__variant_1" in rendered
	assert "plan=m_deliver_branching__variant_2" in rendered
	assert "\tload(PKG);" in rendered
	assert "\tmove(LOC);" in rendered
	assert "\tdrop(PKG, LOC)." in rendered


def test_rendering_emits_jason_safe_functors_for_hddl_symbols() -> None:
	domain = SimpleNamespace(
		name="hyphenated",
		tasks=(
			SimpleNamespace(name="do-put-on", parameters=("?block:block",)),
		),
		predicates=(
			SimpleNamespace(name="clear-top", parameters=("?block:block",)),
		),
		actions=(
			SimpleNamespace(name="pick-up", parameters=("?block:block",)),
		),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="do_put_on",
				parameters=("?block",),
				is_primitive=False,
				source_name="do-put-on",
			),
		],
		primitive_tasks=[
			HTNTask(name="pick-up", parameters=("?block",), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m_do_put_on",
				task_name="do_put_on",
				parameters=("?block",),
				task_args=("?block",),
				context=(HTNLiteral(predicate="clear-top", args=("?block",)),),
				subtasks=(
					HTNMethodStep("s1", "pick-up", ("?block",), "primitive"),
				),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	plan_library, _coverage = build_plan_library(domain=domain, method_library=method_library)
	rendered = render_plan_library_asl(plan_library)

	assert plan_library.plans[0].trigger.arguments == ("BLOCK:block",)
	assert "+!do_put_on(BLOCK) : object_type(BLOCK, block) & clear_top(BLOCK) <-" in rendered
	assert "\tpick_up(BLOCK)." in rendered
	assert "do-put-on" not in rendered
	assert "pick-up" not in rendered
	assert "clear-top" not in rendered


def test_translation_canonicalises_local_method_variables() -> None:
	domain = SimpleNamespace(
		name="courier",
		tasks=(SimpleNamespace(name="deliver_via", parameters=("?pkg:package", "?loc:location")),),
		predicates=(),
		actions=(
			SimpleNamespace(name="load", parameters=("?pkg:package",)),
			SimpleNamespace(name="move", parameters=("?mid:location",)),
			SimpleNamespace(name="drop", parameters=("?pkg:package", "?loc:location")),
		),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="deliver_via", parameters=("?pkg", "?loc"), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="load", parameters=("?pkg",), is_primitive=True),
			HTNTask(name="move", parameters=("?mid",), is_primitive=True),
			HTNTask(name="drop", parameters=("?pkg", "?loc"), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m_deliver_via_midpoint",
				task_name="deliver_via",
				parameters=("?pkg", "?loc", "?mid"),
				task_args=("?pkg", "?loc"),
				subtasks=(
					HTNMethodStep("s1", "load", ("?pkg",), "primitive", action_name="load"),
					HTNMethodStep("s2", "move", ("?mid",), "primitive", action_name="move"),
					HTNMethodStep("s3", "drop", ("?pkg", "?loc"), "primitive", action_name="drop"),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	plan_library, _coverage = build_plan_library(domain=domain, method_library=method_library)

	assert plan_library.plans[0].trigger.arguments == ("PKG:package", "LOC:location")
	assert tuple(step.arguments for step in plan_library.plans[0].body) == (
		("PKG",),
		("MID",),
		("PKG", "LOC"),
	)
	assert "object_type(MID, location)" in plan_library.plans[0].context


def test_translation_lifts_local_witness_action_preconditions_into_context() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	method_library = build_official_method_library(DOMAIN_FILES["blocksworld"])

	plan_library, _coverage = build_plan_library(
		domain=domain,
		method_library=method_library,
	)

	plans_by_name = {plan.plan_name: plan for plan in plan_library.plans}
	assert "on(X, Y)" in plans_by_name["m2_do_on_table"].context
	assert "on(X, Z)" in plans_by_name["m5_do_move"].context
	assert "on(Y, X)" in plans_by_name["m7_do_clear"].context
	assert "clear(Y)" not in plans_by_name["m7_do_clear"].context
	assert "holding(Y)" not in plans_by_name["m7_do_clear"].context


def test_translation_lifts_conservative_multi_method_compound_summary() -> None:
	domain = SimpleNamespace(
		name="routing",
		tasks=(
			SimpleNamespace(name="deliver", parameters=("?pkg:package", "?to:location")),
			SimpleNamespace(name="move_abs", parameters=("?from:location", "?to:location")),
		),
		predicates=(
			SimpleNamespace(name="road", parameters=("?from:location", "?to:location")),
			SimpleNamespace(name="fuelled", parameters=("?from:location",)),
		),
		actions=(),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="deliver", parameters=("?pkg", "?to"), is_primitive=False),
			HTNTask(name="move_abs", parameters=("?from", "?to"), is_primitive=False),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m-deliver",
				task_name="deliver",
				parameters=("?pkg", "?from", "?to"),
				task_args=("?pkg", "?to"),
				subtasks=(
					HTNMethodStep("s1", "move_abs", ("?from", "?to"), "compound"),
				),
			),
			HTNMethod(
				method_name="m-move-direct",
				task_name="move_abs",
				parameters=("?from", "?to"),
				task_args=("?from", "?to"),
				context=(
					HTNLiteral(predicate="road", args=("?from", "?to")),
				),
			),
			HTNMethod(
				method_name="m-move-fuelled",
				task_name="move_abs",
				parameters=("?from", "?to"),
				task_args=("?from", "?to"),
				context=(
					HTNLiteral(predicate="road", args=("?from", "?to")),
					HTNLiteral(predicate="fuelled", args=("?from",)),
				),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	plan_library, _coverage = build_plan_library(domain=domain, method_library=method_library)

	plans_by_name = {plan.plan_name: plan for plan in plan_library.plans}
	parent_context = plans_by_name["m-deliver"].context
	assert "road(FROM, TO)" in parent_context
	assert "fuelled(FROM)" not in parent_context
	assert "object_type(FROM, location)" in parent_context


def test_translation_does_not_lift_preconditions_delegated_to_prior_compound_steps() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["marsrover"])
	method_library = build_official_method_library(DOMAIN_FILES["marsrover"])

	plan_library, _coverage = build_plan_library(
		domain=domain,
		method_library=method_library,
	)

	plans_by_name = {plan.plan_name: plan for plan in plan_library.plans}
	context = plans_by_name["m-get_rock_data"].context
	assert "equipped_for_rock_analysis(ROVER)" in context
	assert "store_of(S, ROVER)" in context
	assert "at_rock_sample(WAYPOINT)" in context
	assert all(not literal.startswith("empty(") for literal in context)
	assert all(not literal.startswith("at(") for literal in context)
	soil_context = plans_by_name["m-get_soil_data"].context
	assert "at_soil_sample(WAYPOINT)" in soil_context
	assert all(not literal.startswith("empty(") for literal in soil_context)
	assert all(not literal.startswith("at(") for literal in soil_context)
	send_soil_context = plans_by_name["m-send_soil_data"].context
	assert "have_soil_analysis(ROVER, WAYPOINT)" in send_soil_context
	assert "channel_free(L)" in send_soil_context
	image_context = plans_by_name["m-get_image_data"].context
	assert "calibration_target(CAMERA, OBJECTIVE)" not in image_context
	assert all(not literal.startswith("have_image(") for literal in image_context)
	assert all(not literal.startswith("at(") for literal in image_context)


def test_translation_orders_positive_context_literals_before_negation_for_jason_binding() -> None:
	domain = SimpleNamespace(
		name="rover",
		tasks=(
			SimpleNamespace(
				name="navigate_abs",
				parameters=("?rover:rover", "?to:waypoint"),
			),
		),
		predicates=(
			SimpleNamespace(name="at", parameters=("?rover:rover", "?wp:waypoint")),
			SimpleNamespace(
				name="can_traverse",
				parameters=("?rover:rover", "?from:waypoint", "?to:waypoint"),
			),
			SimpleNamespace(name="visited", parameters=("?wp:waypoint",)),
		),
		actions=(
			SimpleNamespace(
				name="navigate",
				parameters=("?rover:rover", "?from:waypoint", "?to:waypoint"),
			),
		),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="navigate_abs", parameters=("?rover", "?to"), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="navigate", parameters=("?rover", "?from", "?to"), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m_navigate_abs_recursive",
				task_name="navigate_abs",
				parameters=("?rover", "?from", "?to", "?mid"),
				task_args=("?rover", "?to"),
				context=(
					HTNLiteral(predicate="at", args=("?rover", "?to"), is_positive=False),
					HTNLiteral(
						predicate="can_traverse",
						args=("?rover", "?from", "?to"),
						is_positive=False,
					),
					HTNLiteral(predicate="can_traverse", args=("?rover", "?from", "?mid")),
					HTNLiteral(predicate="visited", args=("?mid",), is_positive=False),
					HTNLiteral(predicate="at", args=("?rover", "?from")),
					HTNLiteral(predicate="can_traverse", args=("?rover", "?mid", "?to")),
				),
				subtasks=(
					HTNMethodStep(
						"s1",
						"navigate",
						("?rover", "?from", "?mid"),
						"primitive",
						action_name="navigate",
					),
					HTNMethodStep(
						"s2",
						"navigate",
						("?rover", "?mid", "?to"),
						"primitive",
						action_name="navigate",
					),
				),
				ordering=(("s1", "s2"),),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	plan_library, _coverage = build_plan_library(domain=domain, method_library=method_library)

	assert plan_library.plans[0].context == (
		"object_type(ROVER, rover)",
		"object_type(FROM, waypoint)",
		"object_type(TO, waypoint)",
		"object_type(MID, waypoint)",
		"can_traverse(ROVER, FROM, MID)",
		"at(ROVER, FROM)",
		"can_traverse(ROVER, MID, TO)",
		"!at(ROVER, TO)",
		"!can_traverse(ROVER, FROM, TO)",
		"!visited(MID)",
	)


def test_translation_adds_type_guards_for_context_bound_method_variables() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["transport"])
	method_library = build_official_method_library(DOMAIN_FILES["transport"])

	plan_library, _coverage = build_plan_library(
		domain=domain,
		method_library=method_library,
	)

	plans_by_name = {plan.plan_name: plan for plan in plan_library.plans}
	assert plans_by_name["m-deliver"].context == (
		"object_type(P, package)",
		"object_type(L1, location)",
		"object_type(L2, location)",
		"object_type(V, vehicle)",
		"at(P, L1)",
	)
	assert "at(P, L)" in plans_by_name["m-load"].context
	assert "at(V, L)" in plans_by_name["m-load"].context
	assert "object_type(V, vehicle)" in plans_by_name["m-drive-to"].context
	assert "object_type(L1, location)" in plans_by_name["m-drive-to"].context
	certificate = plans_by_name["m-deliver"].binding_certificate
	assert {
		"variable": "L1",
		"source": "witness-literal-bound",
		"origin": "safe_precondition_lift",
		"literal": "at(P, L1)",
	} in certificate
	assert {
		"variable": "V",
		"source": "type-domain-bound",
		"type": "vehicle",
		"literal": "object_type(V, vehicle)",
	} in certificate
	assert {
		"variable": "V",
		"source": "subgoal-bound",
		"role": "output_variable_from_subgoal",
		"step_index": 0,
		"argument_index": 0,
		"step_kind": "subgoal",
		"step_symbol": "get-to",
		"binding_status": "subgoal_output_bound",
	} in certificate
	assert not any(
		entry.get("variable") == "V"
		and entry.get("step_index") in {1, 2, 3}
		and entry.get("binding_status") == "unbound_at_use"
		for entry in certificate
	)


def test_translation_does_not_treat_type_guards_as_semantic_bindings() -> None:
	domain = SimpleNamespace(
		name="routing",
		tasks=(
			SimpleNamespace(name="dispatch", parameters=("?to:location",)),
		),
		predicates=(),
		actions=(
			SimpleNamespace(name="move", parameters=("?vehicle:vehicle", "?to:location")),
		),
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="dispatch", parameters=("?to",), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="move", parameters=("?vehicle", "?to"), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m-dispatch",
				task_name="dispatch",
				parameters=("?vehicle", "?to"),
				task_args=("?to",),
				subtasks=(
					HTNMethodStep(
						"s1",
						"move",
						("?vehicle", "?to"),
						"primitive",
						action_name="move",
					),
				),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	plan_library, _coverage = build_plan_library(domain=domain, method_library=method_library)

	certificate = plan_library.plans[0].binding_certificate
	assert {
		"variable": "VEHICLE",
		"source": "type-domain-bound",
		"type": "vehicle",
		"literal": "object_type(VEHICLE, vehicle)",
	} in certificate
	assert {
		"variable": "VEHICLE",
		"source": "action-bound",
		"role": "input_variable_already_bound",
		"step_index": 0,
		"argument_index": 0,
		"step_kind": "action",
		"step_symbol": "move",
		"binding_status": "unbound_at_use",
	} in certificate


def test_translation_does_not_lift_preconditions_achieved_by_prior_compound_steps() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["satellite"])
	method_library = build_official_method_library(DOMAIN_FILES["satellite"])

	plan_library, _coverage = build_plan_library(
		domain=domain,
		method_library=method_library,
	)

	plans_by_name = {plan.plan_name: plan for plan in plan_library.plans}
	method0_context = plans_by_name["method0"].context
	method2_context = plans_by_name["method2"].context
	assert all(not literal.startswith("calibrated(") for literal in method0_context)
	assert all(not literal.startswith("power_on(") for literal in method0_context)
	assert all(not literal.startswith("calibrated(") for literal in method2_context)
	assert all(not literal.startswith("power_on(") for literal in method2_context)
	assert all(not literal.startswith("pointing(") for literal in method0_context)
	assert all(not literal.startswith("pointing(") for literal in method2_context)
	assert "supports(MDOATT_TI_I, MDOATT_TI_M)" in method0_context
	assert "supports(MDOAT_TI_I, MDOAT_TI_M)" in method2_context
	assert {
		"variable": "MDOATT_T_D_PREV",
		"source": "action-bound",
		"role": "output_variable_from_action_precondition",
		"step_index": 1,
		"argument_index": 2,
		"step_kind": "action",
		"step_symbol": "turn_to",
		"binding_status": "action_precondition_bound",
		"binding_source": "positive_action_precondition",
		"binding_literals": ("pointing(MDOATT_T_S,MDOATT_T_D_PREV)",),
	} in plans_by_name["method0"].binding_certificate
	assert not any(
		entry.get("variable") == "MDOATT_T_D_PREV"
		and entry.get("step_symbol") == "turn_to"
		and entry.get("binding_status") == "unbound_at_use"
		for entry in plans_by_name["method0"].binding_certificate
	)
