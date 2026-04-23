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

from method_library import HTNMethod, HTNMethodLibrary, HTNMethodStep, HTNTask
from plan_library import build_plan_library, render_plan_library_asl


def _sample_domain():
	return SimpleNamespace(
		name="courier",
		tasks=(
			SimpleNamespace(
				name="deliver",
				parameters=("?pkg:package", "?loc:location"),
			),
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
	assert tuple(step.kind for step in plan_library.plans[0].body) == (
		"action",
		"action",
		"action",
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
	assert "+!deliver(PKG, LOC) : true <-" in rendered
	assert "plan=m_deliver_branching__variant_1" in rendered
	assert "plan=m_deliver_branching__variant_2" in rendered
	assert "\tload(?pkg);" in rendered
	assert "\tmove(?loc);" in rendered
	assert "\tdrop(?pkg, ?loc)." in rendered
