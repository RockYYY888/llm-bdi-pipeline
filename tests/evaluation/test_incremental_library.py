from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from evaluation.incremental_library import (
	build_incremental_patch_prompt,
	deduplicate_plan_library,
	empty_method_library_for_domain,
	materialize_incremental_bundle,
	merge_method_libraries,
	parse_method_patch_response,
)
from method_library import HTNLiteral, HTNMethod, HTNMethodLibrary, HTNMethodStep
from plan_library.models import (
	AgentSpeakBodyStep,
	AgentSpeakPlan,
	AgentSpeakTrigger,
	PlanLibrary,
)
from temporal_specification.models import QueryInstructionRecord, TemporalSpecificationRecord
from tests.support.plan_library_generation_support import DOMAIN_FILES
from utils.hddl_parser import HDDLParser


def _do_move_method(method_name: str, source_instruction_ids: tuple[str, ...]) -> HTNMethod:
	return HTNMethod(
		method_name=method_name,
		task_name="do_move",
		parameters=("?x", "?y"),
		task_args=("?x", "?y"),
		subtasks=(
			HTNMethodStep("first", "pick_up", ("?x",), "primitive", action_name="pick-up"),
			HTNMethodStep("second", "stack", ("?x", "?y"), "primitive", action_name="stack"),
		),
		ordering=(("first", "second"),),
		source_instruction_ids=source_instruction_ids,
	)


def test_merge_method_libraries_deduplicates_semantic_methods_and_merges_sources() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	scaffold = empty_method_library_for_domain(domain)
	existing = HTNMethodLibrary(
		compound_tasks=scaffold.compound_tasks,
		primitive_tasks=scaffold.primitive_tasks,
		methods=[_do_move_method("m_do_move_a", ("query_1",))],
	)
	patch = HTNMethodLibrary(
		compound_tasks=scaffold.compound_tasks,
		primitive_tasks=scaffold.primitive_tasks,
		methods=[_do_move_method("m_do_move_duplicate_name", ("query_2",))],
	)

	result = merge_method_libraries(existing, patch)

	assert len(result.method_library.methods) == 1
	assert result.duplicate_methods == 1
	assert result.method_library.methods[0].source_instruction_ids == ("query_1", "query_2")


def test_merge_method_libraries_deduplicates_methods_with_context_literals() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	scaffold = empty_method_library_for_domain(domain)
	base_method = _do_move_method("m_do_move_a", ("query_1",))
	contextual_method = HTNMethod(
		method_name=base_method.method_name,
		task_name=base_method.task_name,
		parameters=base_method.parameters,
		task_args=base_method.task_args,
		context=(
			HTNLiteral("handempty"),
			HTNLiteral("clear", ("?y",)),
		),
		subtasks=base_method.subtasks,
		ordering=base_method.ordering,
		source_instruction_ids=base_method.source_instruction_ids,
	)
	existing = HTNMethodLibrary(
		compound_tasks=scaffold.compound_tasks,
		primitive_tasks=scaffold.primitive_tasks,
		methods=[contextual_method],
	)
	patch = HTNMethodLibrary(
		compound_tasks=scaffold.compound_tasks,
		primitive_tasks=scaffold.primitive_tasks,
		methods=[
			HTNMethod(
				method_name="renamed_duplicate",
				task_name=contextual_method.task_name,
				parameters=contextual_method.parameters,
				task_args=contextual_method.task_args,
				context=tuple(reversed(contextual_method.context)),
				subtasks=contextual_method.subtasks,
				ordering=contextual_method.ordering,
				source_instruction_ids=("query_2",),
			),
		],
	)

	result = merge_method_libraries(existing, patch)

	assert len(result.method_library.methods) == 1
	assert result.duplicate_methods == 1
	assert result.method_library.methods[0].source_instruction_ids == ("query_1", "query_2")


def test_deduplicate_plan_library_preserves_s_as_a_set() -> None:
	plan_a = AgentSpeakPlan(
		plan_name="p_a",
		trigger=AgentSpeakTrigger(
			event_type="achievement_goal",
			symbol="do_move",
			arguments=("X:block", "Y:block"),
		),
		context=("clear(X)",),
		body=(
			AgentSpeakBodyStep("action", "pick-up", ("X",)),
			AgentSpeakBodyStep("action", "stack", ("X", "Y")),
		),
		source_instruction_ids=("query_1",),
	)
	plan_b = AgentSpeakPlan(
		plan_name="p_b",
		trigger=plan_a.trigger,
		context=plan_a.context,
		body=plan_a.body,
		source_instruction_ids=("query_2",),
	)

	result = deduplicate_plan_library(PlanLibrary(domain_name="BLOCKS", plans=(plan_a, plan_b)))

	assert len(result.plan_library.plans) == 1
	assert result.removed_duplicate_plans == 1
	assert result.plan_library.plans[0].source_instruction_ids == ("query_1", "query_2")


def test_materialize_incremental_bundle_writes_set_normalised_plan_artifact(
	tmp_path: Path,
) -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	scaffold = empty_method_library_for_domain(domain)
	method_library = HTNMethodLibrary(
		compound_tasks=scaffold.compound_tasks,
		primitive_tasks=scaffold.primitive_tasks,
		methods=[
			_do_move_method("m_do_move_a", ("query_1",)),
			_do_move_method("m_do_move_b", ("query_2",)),
		],
	)
	query_sequence = (
		QueryInstructionRecord(
			instruction_id="query_1",
			source_text="debug-only",
			problem_file="p01.hddl",
		),
	)
	temporal_specifications = (
		TemporalSpecificationRecord(
			instruction_id="query_1",
			source_text="debug-only",
			ltlf_formula="F(do_move(b1, b2))",
			referenced_events=(),
			problem_file="p01.hddl",
		),
	)

	result = materialize_incremental_bundle(
		domain=domain,
		method_library=method_library,
		query_sequence=query_sequence,
		temporal_specifications=temporal_specifications,
		artifact_root=tmp_path / "library",
	)

	plan_library_payload = json.loads(Path(result["artifact_paths"]["plan_library"]).read_text())
	assert len(plan_library_payload["plans"]) == 1
	assert result["set_normalisation"]["removed_duplicate_plans"] == 1
	assert Path(result["artifact_paths"]["plan_library_asl"]).exists()


def test_parse_method_patch_response_normalises_methods_only_payload() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	scaffold = empty_method_library_for_domain(domain)
	response_text = json.dumps(
		{
			"methods": [
				{
					"method_name": "m_patch_do_move",
					"task_name": "do_move",
					"parameters": ["?x", "?y"],
					"task_args": ["?x", "?y"],
					"context": [],
					"subtasks": [
						{
							"step_id": "s1",
							"task_name": "pick-up",
							"args": ["?x"],
							"kind": "primitive",
						},
						{
							"step_id": "s2",
							"task_name": "stack",
							"args": ["?x", "?y"],
							"kind": "primitive",
						},
					],
					"ordering": [["s1", "s2"]],
					"source_instruction_ids": ["query_1"],
				},
			],
		},
	)
	temporal_specification = TemporalSpecificationRecord(
		instruction_id="query_1",
		source_text="not sent to the patch model",
		ltlf_formula="F(do_move(b1, b2))",
		referenced_events=(),
		problem_file="p01.hddl",
	)

	patch = parse_method_patch_response(
		response_text=response_text,
		domain=domain,
		current_library=scaffold,
		temporal_specification=temporal_specification,
	)

	assert len(patch.methods) == 1
	assert patch.methods[0].task_name == "do_move"
	assert patch.methods[0].subtasks[0].task_name == "pick_up"
	assert patch.methods[0].subtasks[0].action_name == "pick-up"


def test_incremental_patch_prompt_uses_ltlf_not_natural_language_text() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	scaffold = empty_method_library_for_domain(domain)
	temporal_specification = TemporalSpecificationRecord(
		instruction_id="query_1",
		source_text="this natural language should not be sent",
		ltlf_formula="F(do_move(b1, b2))",
		referenced_events=(),
		problem_file="p01.hddl",
	)

	prompt = build_incremental_patch_prompt(
		domain=domain,
		current_library=scaffold,
		temporal_specification=temporal_specification,
		evaluation_result={"success": False, "step": "runtime_execution"},
	)

	assert "F(do_move(b1, b2))" in prompt["user"]
	assert "this natural language should not be sent" not in prompt["user"]
	assert "primitive_actions" in prompt["user"]
