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
	assert plan_library.plans[0].context == ("loaded(PKG)",)
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
	assert coverage.methods_considered == 2
	assert coverage.accepted_translation == 2
	assert coverage.plans_generated == 3
	assert coverage.unsupported_buckets == {}
	assert coverage.unsupported_methods == ()

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


def test_rendering_emits_structured_agentspeak_library() -> None:
	plan_library, _coverage = build_plan_library(
		domain=_sample_domain(),
		method_library=_sample_method_library(),
	)

	rendered = render_plan_library_asl(plan_library)

	assert "source_instruction_ids=query_7" in rendered
	assert "+!deliver(PKG, LOC) : loaded(PKG) <-" in rendered
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
	assert "+!do_put_on(BLOCK) : clear_top(BLOCK) <-" in rendered
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
