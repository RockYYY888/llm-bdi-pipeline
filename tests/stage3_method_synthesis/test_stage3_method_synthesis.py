"""
Focused tests for Stage 3 HTN method synthesis.
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_tests_dir = str(Path(__file__).parent.parent)
if _tests_dir not in sys.path:
	sys.path.insert(0, _tests_dir)
_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import (
	LTLFormula,
	LTLSpecification,
	TemporalOperator,
)
from stage3_method_synthesis.htn_method_synthesis import (
	HTNMethodSynthesizer,
	HTNSynthesisError,
	LLMStreamingResponseError,
)
from stage3_method_synthesis.htn_prompts import (
	_candidate_support_task_names,
	_declared_task_schema_map,
	_candidate_headline_predicates_for_task,
	_required_helper_specs_for_query_targets,
	_render_producer_mode_options_for_predicate,
	_same_arity_caller_shared_requirements,
	_same_arity_packaging_candidates_for_query_task,
	build_prompt_analysis_payload,
	build_htn_system_prompt,
	build_htn_user_prompt,
	_render_signature_with_mapping,
)
from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
	HTNTargetTaskBinding,
)
from utils.config import get_config
from utils.hddl_parser import HDDLParser

OFFICIAL_BLOCKSWORLD_DOMAIN_FILE = (
	Path(__file__).parent.parent.parent / "src" / "domains" / "blocksworld" / "domain.hddl"
)

def _domain():
	return HDDLParser.parse_domain(str(OFFICIAL_BLOCKSWORLD_DOMAIN_FILE))

def _marsrover_domain():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "src"
		/ "domains"
		/ "marsrover"
		/ "domain.hddl"
	)
	return HDDLParser.parse_domain(str(domain_path))

def _satellite_domain():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "src"
		/ "domains"
		/ "satellite"
		/ "domain.hddl"
	)
	return HDDLParser.parse_domain(str(domain_path))

def _transport_domain():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "src"
		/ "domains"
		/ "transport"
		/ "domain.hddl"
	)
	return HDDLParser.parse_domain(str(domain_path))

def _live_stage3_kwargs() -> dict:
	config = get_config()
	if not config.validate():
		pytest.skip("Stage 3 live tests require a valid OPENAI_API_KEY")

	return {
		"api_key": config.openai_api_key,
		"model": config.openai_stage3_model,
		"base_url": config.openai_base_url,
		"timeout": float(config.openai_stage3_timeout),
	}

def _atomic_formula(predicate: str, args: list[str]) -> LTLFormula:
	return LTLFormula(
		operator=None,
		predicate={predicate: args},
		sub_formulas=[],
		logical_op=None,
	)

def _eventually_on_spec() -> LTLSpecification:
	spec = LTLSpecification()
	spec.objects = ["a", "b"]
	spec.grounding_map = GroundingMap()
	spec.grounding_map.add_atom("on_a_b", "on", ["a", "b"])
	spec.formulas = [
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[_atomic_formula("on", ["a", "b"])],
			logical_op=None,
		),
	]
	return spec

def _extract_prompt_section(text: str, header: str, next_header: str | None = None) -> str:
	start = text.index(header) + len(header)
	if next_header is None:
		return text[start:].strip()
	end = text.index(next_header, start)
	return text[start:end].strip()

def test_extract_target_literals_reads_stage1_signatures_only():
	synthesizer = HTNMethodSynthesizer()

	literals = synthesizer.extract_target_literals(
		ordered_literal_signatures=[
			"clear(a)",
			"on(a, b)",
			"on(b, c)",
		],
	)

	assert [literal.to_signature() for literal in literals] == [
		"clear(a)",
		"on(a, b)",
		"on(b, c)",
	]

def test_extract_target_literals_deduplicates_repeated_signatures():
	synthesizer = HTNMethodSynthesizer()

	literals = synthesizer.extract_target_literals(
		ordered_literal_signatures=[
			"on(b10, b6)",
			"on(b5, b10)",
			"on(b10, b6)",
		],
	)

	assert [literal.to_signature() for literal in literals] == [
		"on(b10, b6)",
		"on(b5, b10)",
	]

def test_synthesize_requires_live_llm():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "src"
		/ "domains"
		/ "blocksworld"
		/ "domain.hddl"
	)
	domain = HDDLParser.parse_domain(str(domain_path))

	with pytest.raises(HTNSynthesisError) as exc_info:
		HTNMethodSynthesizer().synthesize(
			domain=domain,
			ordered_literal_signatures=["on(b4, b2)", "on(b1, b4)", "on(b3, b1)"],
		)

	assert "requires a configured OPENAI_API_KEY" in str(exc_info.value)

def test_extract_target_literals_supports_negative_signatures():
	synthesizer = HTNMethodSynthesizer()

	literals = synthesizer.extract_target_literals(
		ordered_literal_signatures=["!clear(a)"],
	)

	assert [literal.to_signature() for literal in literals] == ["!clear(a)"]

def test_method_synthesizer_uses_live_llm_output():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer(**_live_stage3_kwargs())

	try:
		library, metadata = synthesizer.synthesize(
			domain=domain,
			ordered_literal_signatures=["on(a, b)"],
		)
	except HTNSynthesisError as exc:
		metadata = exc.metadata
		assert metadata["llm_prompt"] is not None
		assert metadata["llm_response"]
		assert metadata["target_literals"] == ["on(a, b)"]
		assert metadata["failure_stage"] in {"response_parse", "library_validation"}
		return

	assert metadata["used_llm"] is True
	assert metadata["llm_prompt"] is not None
	assert metadata["llm_response"]
	assert metadata["target_literals"] == ["on(a, b)"]
	assert metadata["compound_tasks"] >= 1
	assert metadata["methods"] >= 1
	assert library.compound_tasks
	assert library.methods
	assert library.target_task_bindings

	primitive_task_names = {task.name for task in library.primitive_tasks}
	assert primitive_task_names == {
		"nop",
		"pick_up",
		"put_down",
		"stack",
		"unstack",
	}

	compound_task_names = {task.name for task in library.compound_tasks}
	assert all(method.task_name in compound_task_names for method in library.methods)
	assert all(
		not task_name.startswith(("achieve_", "maintain_not_"))
		for task_name in compound_task_names
	)

	bound_task = library.task_name_for_literal(library.target_literals[0])
	assert bound_task in compound_task_names

def test_method_synthesizer_requires_target_task_binding_for_each_target_literal():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
	)

	with pytest.raises(ValueError, match="missing a target_task_binding"):
		synthesizer._validate_library(library, domain)

def test_validate_library_no_longer_requires_obsolete_helper_contract_tasks():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	prompt_analysis = build_prompt_analysis_payload(
		domain,
		target_literals=["on(a, b)"],
		query_task_anchors=(
			{"task_name": "do_put_on", "args": ["a", "b"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_put_on", ("ARG1", "ARG2"), False, ("on",)),
			HTNTask("do_move", ("ARG1", "ARG2"), False, ("on",)),
			HTNTask("do_clear", ("ARG1",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_put_on_noop",
				task_name="do_put_on",
				parameters=("ARG1", "ARG2"),
				context=(HTNLiteral("on", ("ARG1", "ARG2"), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_move_constructive",
				task_name="do_move",
				parameters=("ARG1", "ARG2"),
				context=(
					HTNLiteral("holding", ("ARG1",), True, None),
					HTNLiteral("clear", ("ARG2",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="stack",
						args=("ARG1", "ARG2"),
						kind="primitive",
						action_name="stack",
						literal=None,
						preconditions=(),
						effects=(),
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_noop",
				task_name="do_clear",
				parameters=("ARG1",),
				context=(HTNLiteral("clear", ("ARG1",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="on(a, b)", task_name="do_put_on"),
		],
	)

	synthesizer._validate_library(
		library,
		domain,
		prompt_analysis=prompt_analysis,
	)

def test_validate_library_accepts_direct_parent_support_when_no_required_helper_exists():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	prompt_analysis = build_prompt_analysis_payload(
		domain,
		target_literals=["on(a, b)"],
		query_task_anchors=(
			{"task_name": "do_put_on", "args": ["a", "b"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_put_on", ("ARG1", "ARG2"), False, ("on",)),
			HTNTask("do_move", ("ARG1", "ARG2"), False, ("on",)),
			HTNTask("do_clear", ("ARG1",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_put_on_noop",
				task_name="do_put_on",
				parameters=("ARG1", "ARG2"),
				context=(HTNLiteral("on", ("ARG1", "ARG2"), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_put_on_direct",
				task_name="do_put_on",
				parameters=("ARG1", "ARG2"),
				context=(
					HTNLiteral("ontable", ("ARG1",), True, None),
					HTNLiteral("handempty", tuple(), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_clear",
						args=("ARG1",),
						kind="compound",
						action_name=None,
						literal=None,
						preconditions=(),
						effects=(),
					),
					HTNMethodStep(
						step_id="s2",
						task_name="do_clear",
						args=("ARG2",),
						kind="compound",
						action_name=None,
						literal=None,
						preconditions=(),
						effects=(),
					),
					HTNMethodStep(
						step_id="s3",
						task_name="pick_up",
						args=("ARG1",),
						kind="primitive",
						action_name="pick-up",
						literal=None,
						preconditions=(),
						effects=(),
					),
					HTNMethodStep(
						step_id="s4",
						task_name="do_move",
						args=("ARG1", "ARG2"),
						kind="compound",
						action_name=None,
						literal=None,
						preconditions=(),
						effects=(),
					),
				),
				ordering=(("s1", "s2"), ("s2", "s3"), ("s3", "s4")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_move_constructive",
				task_name="do_move",
				parameters=("ARG1", "ARG2"),
				context=(
					HTNLiteral("holding", ("ARG1",), True, None),
					HTNLiteral("clear", ("ARG2",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="stack",
						args=("ARG1", "ARG2"),
						kind="primitive",
						action_name="stack",
						literal=None,
						preconditions=(),
						effects=(),
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_noop",
				task_name="do_clear",
				parameters=("ARG1",),
				context=(HTNLiteral("clear", ("ARG1",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="on(a, b)", task_name="do_put_on"),
		],
	)

	synthesizer._validate_library(
		library,
		domain,
		prompt_analysis=prompt_analysis,
	)

def test_validate_library_accepts_required_helper_with_parent_local_parameter_names():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	prompt_analysis = build_prompt_analysis_payload(
		domain,
		target_literals=["on(a, b)"],
		query_task_anchors=(
			{"task_name": "do_put_on", "args": ["a", "b"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_put_on", ("X", "Y"), False, ("on",)),
			HTNTask("do_move", ("X", "Y"), False, ("on",)),
			HTNTask("do_clear", ("X",), False, ("clear",)),
			HTNTask("do_holding", ("X",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_put_on_noop",
				task_name="do_put_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("on", ("X", "Y"), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_put_on_constructive",
				task_name="do_put_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("clear", ("Y",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_holding",
						args=("X",),
						kind="compound",
						action_name=None,
						literal=None,
						preconditions=(),
						effects=(),
					),
					HTNMethodStep(
						step_id="s2",
						task_name="do_move",
						args=("X", "Y"),
						kind="compound",
						action_name=None,
						literal=None,
						preconditions=(),
						effects=(),
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_move_constructive",
				task_name="do_move",
				parameters=("X", "Y"),
				context=(
					HTNLiteral("holding", ("X",), True, None),
					HTNLiteral("clear", ("Y",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="stack",
						args=("X", "Y"),
						kind="primitive",
						action_name="stack",
						literal=None,
						preconditions=(),
						effects=(),
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_noop",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("clear", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_holding_noop",
				task_name="do_holding",
				parameters=("X",),
				context=(HTNLiteral("holding", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding(target_literal="on(a, b)", task_name="do_put_on"),
		],
	)

	synthesizer._validate_library(
		library,
		domain,
		prompt_analysis=prompt_analysis,
	)

def test_normalise_library_repairs_subtask_kind_by_declared_task_sets():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("communicate_rock_data", ("P",), False, ("communicated_rock_data",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_communicate_rock_data_direct",
				task_name="communicate_rock_data",
				parameters=("P",),
				context=(HTNLiteral("have_rock_analysis", ("ROVER", "P"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="communicate_rock_data",
						args=("ROVER", "L", "P", "X", "Y"),
						kind="primitive",
						action_name="communicate_rock_data",
						literal=HTNLiteral("communicated_rock_data", ("P",), True, None),
						preconditions=(),
						effects=(),
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("communicated_rock_data", ("waypoint2",), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding("communicated_rock_data(waypoint2)", "communicate_rock_data"),
		],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	[method] = normalised.methods
	[step] = method.subtasks
	assert step.kind == "compound"
	assert step.task_name == "communicate_rock_data"

def test_validate_library_rejects_constant_symbol_type_conflicts():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("communicate_image", ("OBJECTIVE", "MODE"), False, ("communicated_image_data",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_communicate_image_noop",
				task_name="communicate_image",
				parameters=("OBJECTIVE", "MODE"),
				context=(
					HTNLiteral("communicated_image_data", ("objective0", "low_res"), True, None),
					HTNLiteral("at_lander", ("objective0", "waypoint0"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("communicated_image_data", ("objective0", "low_res"), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding(
				"communicated_image_data(objective0, low_res)",
				"communicate_image",
			),
		],
	)

	with pytest.raises(ValueError, match="objective0"):
		synthesizer._validate_library(library, domain)

def test_validate_library_rejects_task_source_predicate_arity_mismatch():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("navigate_rover", ("ROVER", "FROM_WP", "TO_WP"), False, ("at",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_navigate_rover_direct",
				task_name="navigate_rover",
				parameters=("ROVER", "FROM_WP", "TO_WP"),
				context=(
					HTNLiteral("at", ("ROVER", "FROM_WP"), True, None),
					HTNLiteral("can_traverse", ("ROVER", "FROM_WP", "TO_WP"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("at", ("rover0", "waypoint5"), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("at(rover0, waypoint5)", "navigate_rover")],
	)

	with pytest.raises(ValueError, match="arity mismatch"):
		synthesizer._validate_library(library, domain)

def test_stage3_prompts_make_binding_and_naming_rules_explicit():
	synthesizer = HTNMethodSynthesizer()
	domain = _marsrover_domain()
	analysis = synthesizer._analyse_domain_actions(domain)
	derived_analysis = build_prompt_analysis_payload(
		domain,
		target_literals=[
			"communicated_soil_data(waypoint2)",
			"communicated_image_data(objective1, high_res)",
		],
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
			{"task_name": "get_image_data", "args": ["objective1", "high_res"]},
		),
		action_analysis=analysis,
	)
	system_prompt = build_htn_system_prompt()
	user_prompt = build_htn_user_prompt(
		domain,
		["communicated_soil_data(waypoint2)", "communicated_image_data(objective1, high_res)"],
		'{"target_task_bindings": [], "compound_tasks": [], "methods": []}',
		query_text=(
			"Using rover rover0, waypoint waypoint2, mode high_res, and objective objective1, "
			"complete the tasks get_soil_data(waypoint2) and get_image_data(objective1, high_res)."
		),
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
			{"task_name": "get_image_data", "args": ["objective1", "high_res"]},
		),
		semantic_objects=("waypoint2", "objective1", "high_res"),
		query_object_inventory=(
			{"type": "rover", "label": "rover", "objects": ["rover0"]},
			{"type": "waypoint", "label": "waypoint", "objects": ["waypoint2"]},
			{"type": "mode", "label": "mode", "objects": ["high_res"]},
			{"type": "objective", "label": "objective", "objects": ["objective1"]},
			{"type": "camera", "label": "camera", "objects": ["camera0"]},
		),
		query_objects=("rover0", "waypoint2", "high_res", "objective1", "camera0"),
		action_analysis=analysis,
		derived_analysis=derived_analysis,
	)
	assert "GLOBAL RULES:" in system_prompt
	assert "ordering must be explicit pairwise edges" in system_prompt
	assert "query inventory is authoritative for top-level grounding only" in system_prompt
	assert "Never emit a chain edge like [[\"s1\",\"s2\",\"s3\"]]" in system_prompt
	assert "never invent type predicates such as block(X) or rover(R)" in system_prompt
	assert "Do not infer new packaging candidates or caller-shared envelopes." in system_prompt
	assert "top-level keys target_task_bindings and tasks" in system_prompt
	assert "Each tasks entry defines one compound task once, with fields name, parameters, noop, constructive." in system_prompt
	assert "source_predicates is optional compiler metadata and may be omitted." in system_prompt
	assert "Every compound child you call must also appear as another task entry in tasks." in system_prompt
	assert "materialize it as its own task entry with branches in the same JSON." in system_prompt
	assert "Never invent aggregate/root wrapper tasks that merely sequence the ordered query tasks" in system_prompt
	assert "the constructive branch must call that packaging child" in system_prompt
	assert "the parent boundary is exactly the listed caller-shared set" in system_prompt
	assert "do not hide unmet dynamic prerequisites in branch context" in system_prompt
	assert "use those listed options or declared support tasks instead of inventing a fresh helper" in system_prompt
	assert "any dynamic prerequisite linking AUX1 to a task argument or shared resource is mandatory" in system_prompt
	assert "If that need binds AUX_* to ARG*" in system_prompt
	assert "Support tasks should return in a reusable stable state." in system_prompt

	assert derived_analysis["query_task_contracts"]
	assert derived_analysis["support_task_contracts"]
	assert derived_analysis["task_headline_candidates"]["send_soil_data"]

	assert "<query_task_contracts>" in user_prompt
	assert "<support_task_contracts>" in user_prompt
	assert "<domain_summary>" in user_prompt
	assert "<instructions>" in user_prompt
	assert "<output_schema>" in user_prompt
	assert "<query_summary>" in user_prompt
	assert "ordered_binding #1: communicated_soil_data(waypoint2) -> get_soil_data(waypoint2)" in user_prompt
	assert "ordered_binding #2: communicated_image_data(objective1, high_res) -> get_image_data(objective1, high_res)" in user_prompt
	assert "query_type_inventory:" in user_prompt
	assert "- rover: 1 object(s)" in user_prompt
	assert "- camera: 1 object(s)" in user_prompt
	assert "grounding_rules:" in user_prompt
	assert "Ordered target bindings below are the authoritative grounded binding source for Stage 3." in user_prompt
	assert "Do not copy grounded object names into methods; methods must stay schematic." in user_prompt
	assert "<query_task_contract name=\"get_soil_data\">" in user_prompt
	assert "<support_task_contract name=\"send_soil_data\">" in user_prompt
	assert "AUX_STORE1" in user_prompt
	assert (
		'send_soil_data(?rover, ?waypoint): exact producer slots '
		'[{"producer":"sample_soil(ARG1, AUX_STORE1, ARG2)"}].'
	) in user_prompt
	assert "valid internal-support sibling for the false-at(ARG1, ARG2) case" in user_prompt
	assert (
		"send_soil_data(?rover, ?waypoint) targets have_soil_analysis(?rover, ?waypoint); "
		"templates: sample_soil(?rover, AUX_STORE1, ?waypoint)"
	) in user_prompt
	assert "Use ARG1..ARGn for task-signature roles and AUX_* for extra roles." in user_prompt
	assert "Do not invent aggregate/root wrappers such as do_world, do_all, goal_root, or __top" in user_prompt
	assert "Type names are not predicates." in user_prompt
	assert "Do not copy grounded constants from the original sentence into task definitions." in user_prompt
	assert "ACTION [needs ...] or [extra needs ...]" in user_prompt
	assert "use earlier subtasks instead of leaving unmet dynamic prerequisites in constructive context" in user_prompt
	assert "If a contract line already gives precondition/context, producer, and followup but no support_before, do not invent extra support_before steps" in user_prompt
	assert "inferred_task_headline_candidates:" not in user_prompt
	assert "likely headline predicates" not in user_prompt

def test_stage3_prompt_makes_child_shared_support_requirements_explicit_for_query_tasks():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["on(a, b)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text="Using blocks a and b, complete the tasks do_put_on(a, b).",
		query_task_anchors=(
			{"task_name": "do_put_on", "args": ["a", "b"]},
		),
		query_objects=("a", "b"),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert "<query_task_contract name=\"do_put_on\">" in user_prompt
	assert "<support_task_contract name=\"do_clear\">" in user_prompt
	assert "<support_task_contract name=\"do_move\">" in user_prompt
	assert "Support caller-shared prerequisites clear(?x), clear(?y), handempty before the child call" in user_prompt
	assert "do_put_on(?x, ?y): support holding(?x) before stack(?x, ?y)." not in user_prompt
	assert "do_put_on(?x, ?y): exact same-arity packaging contract for on(?x, ?y) is do_move(?x, ?y)." in user_prompt
	assert "do_put_on(?x, ?y): required noop branch precondition/context on(ARG1, ARG2). Include noop explicitly in the task entry; tasks missing noop are invalid." in user_prompt
	assert "do_put_on(?x, ?y): once you choose do_move(?x, ?y) as same-arity packaging for on(?x, ?y), use a parent skeleton" in user_prompt
	assert "required helper task do_holding(ARG1)" not in user_prompt
	assert "producer do_holding(ARG1); followup do_move(ARG1, ARG2)." not in user_prompt
	assert "Do not place pick_up(ARG1) or unstack(ARG1, AUX_BLOCK1) directly in the parent branch." not in user_prompt
	assert "in any compact support_before/producer/followup branch where holding(ARG1) is not already in branch precondition/context, the producer slot itself must establish it" not in user_prompt
	assert "valid constructive sibling for pick_up(ARG1): precondition/context clear(ARG1), ontable(ARG1), handempty, clear(ARG2); producer pick_up(ARG1); followup stack(ARG1, ARG2)." in user_prompt
	assert "valid support-then-produce sibling for the false-clear(ARG1) case with pick_up(ARG1): precondition/context ontable(ARG1), handempty, clear(ARG2); support_before do_clear(ARG1); producer pick_up(ARG1); followup stack(ARG1, ARG2)." in user_prompt
	assert "valid support-then-produce sibling for the false-clear(ARG1) case with unstack(ARG1, AUX_BLOCK1)" not in user_prompt
	assert "non-leading support/base role" not in user_prompt
	assert "do_move(?x, ?y): exact same-arity packaging child for on(?x, ?y) when called by do_put_on(?x, ?y)." in user_prompt
	assert "Parent-side caller-shared prerequisites: clear(?x), clear(?y), handempty." in user_prompt
	assert 'do_move(?x, ?y): exact producer slots' not in user_prompt
	assert "do_move(?x, ?y): with parent-side caller-shared prerequisites holding(?x), keep the remaining shared final-producer prerequisites clear(?y) explicit in the constructive branch precondition/context at child entry unless earlier child subtasks establish them." not in user_prompt
	assert "Inside this task, support holding(?x) before the final producer" in user_prompt
	assert "do_move(ARG1, ARG2): stack(ARG1, ARG2) requires clear(ARG2)." not in user_prompt
	assert "do_move(ARG1, ARG2): valid internal-support sibling for the false-clear(ARG2)" not in user_prompt
	assert "valid support-then-produce sibling for the false-clear(ARG1) case with pick_up(ARG1): precondition/context ontable(ARG1), handempty, clear(ARG2); support_before do_clear(ARG1); producer pick_up(ARG1); followup stack(ARG1, ARG2)." in user_prompt
	assert 'AST slot shape for that constructive sibling: {"precondition":["clear(ARG1)", "ontable(ARG1)", "handempty", "clear(ARG2)"],"producer":"pick_up(ARG1)","followup":"stack(ARG1, ARG2)"}.' in user_prompt
	assert 'AST slot shape for that support-then-produce sibling: {"precondition":["ontable(ARG1)", "handempty", "clear(ARG2)"],"support_before":["do_clear(ARG1)"],"producer":"pick_up(ARG1)","followup":"stack(ARG1, ARG2)"}.' in user_prompt
	assert "split already-supported and support-then-produce siblings." in user_prompt
	assert "do_clear(?x) targets clear(?x);" not in user_prompt
	assert "do_clear(?x): required stable/noop branch precondition/context clear(ARG1)." in user_prompt
	assert "put_down(?x) [needs holding(?x)]" not in user_prompt
	assert "stack(?x, AUX_BLOCK1) [needs holding(?x); extra needs clear(AUX_BLOCK1)]" not in user_prompt
	assert "do_clear(?x): valid constructive sibling for put_down(ARG1): precondition/context holding(ARG1); producer put_down(ARG1)." in user_prompt
	assert "do_clear(?x): valid constructive sibling for stack(ARG1, AUX_BLOCK1): precondition/context holding(ARG1); clear(AUX_BLOCK1); producer stack(ARG1, AUX_BLOCK1)." in user_prompt
	assert "valid constructive sibling for put_down(ARG1): precondition/context holding(ARG1); producer put_down(ARG1)." in user_prompt
	assert 'AST slot shape for that recursive sibling: {"precondition":["holding(?x)"],"support_before":["do_clear(AUX_BLOCK1)"],"producer":"stack(?x, AUX_BLOCK1)"}.' not in user_prompt
	assert 'AST slot shape for that recursive sibling: {"precondition":["on(AUX_BLOCK1, ?x)", "handempty"],"support_before":["do_clear(AUX_BLOCK1)"],"producer":"unstack(AUX_BLOCK1, ?x)","followup":"put_down(AUX_BLOCK1)"}.' in user_prompt
	assert 'AST slot shape for the already-supported recursive sibling: {"precondition":["on(AUX_BLOCK1, ?x)", "handempty", "clear(AUX_BLOCK1)"],"producer":"unstack(AUX_BLOCK1, ?x)","followup":"put_down(AUX_BLOCK1)"}.' in user_prompt
	assert "support on(AUX_BLOCK1, ?x) via do_put_on(AUX_BLOCK1, ?x) before unstack(AUX_BLOCK1, ?x)" not in user_prompt
	assert "valid constructive sibling for unstack(AUX_BLOCK1, ARG1): precondition/context on(AUX_BLOCK1, ARG1); clear(AUX_BLOCK1); handempty; producer unstack(AUX_BLOCK1, ARG1)." not in user_prompt
	assert "if a constructive sibling adds do_clear(AUX_BLOCK1) before unstack(AUX_BLOCK1, ?x), keep that mode's other unmet needs on(AUX_BLOCK1, ?x), handempty in the same sibling precondition/context" not in user_prompt
	assert 'AST slot shape for that recursive sibling: {"precondition":["on(AUX_BLOCK1, ?x)", "handempty"],"support_before":["do_clear(AUX_BLOCK1)"],"producer":"unstack(AUX_BLOCK1, ?x)","followup":"put_down(AUX_BLOCK1)"}.' in user_prompt
	assert "if support_before do_clear(AUX_BLOCK1) handles false-clear(AUX_BLOCK1) before unstack(AUX_BLOCK1, ?x), do not also keep clear(AUX_BLOCK1) in that same sibling precondition/context; split already-supported and recursive-support siblings." in user_prompt
	assert "Binding literals among those needs still constrain the AUX role and cannot be dropped." not in user_prompt
	assert "valid recursive slot sibling for unstack(AUX_BLOCK1, ?x): precondition/context on(AUX_BLOCK1, ?x); handempty; support_before do_clear(AUX_BLOCK1); producer unstack(AUX_BLOCK1, ?x)." not in user_prompt
	assert "do_clear(?x): cleanup followup after unstack(AUX_BLOCK1, ?x). If this branch should return with handempty restored, use followup put_down(AUX_BLOCK1) before returning." in user_prompt
	assert "AUX_BLOCK1" in user_prompt
	assert "do_clear(?x): caller-shared dynamic prerequisites" not in user_prompt
	assert "Prefer ordered_subtasks for total orders" in user_prompt
	assert "ordering uses pairwise edges only" in user_prompt
	assert "If a line lists ACTION [needs ...] or [extra needs ...]" in user_prompt
	assert "If support_before TASK(...) handles false-P, do not also keep P in that sibling precondition/context; split already-P and false-P siblings." in user_prompt
	assert "parent supports only the listed caller-shared prerequisites, then calls that child" in user_prompt
	assert "treat the listed caller-shared set as the complete parent boundary" in user_prompt
	assert "Declaring AUX_* is not enough; constrain it before first use." in user_prompt
	assert "Keep template argument positions exact." in user_prompt
	assert "For supportable AUX_* needs, use earlier subtasks instead of leaving unmet dynamic prerequisites in constructive context." in user_prompt
	assert "If a listed need binds ARG* to AUX_* for the chosen mode" in user_prompt
	assert "copy that slot shape directly into the constructive branch" in user_prompt
	assert "copy one producer invocation verbatim into producer and still keep the listed branch obligations" in user_prompt
	assert "support-task contract names a cleanup followup after the producer" in user_prompt

def test_same_arity_packaging_contract_uses_refined_parent_envelope():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	action_analysis = synthesizer._analyse_domain_actions(domain)
	task_schemas = _declared_task_schema_map(domain)
	candidates = [
		candidate
		for candidate in _same_arity_packaging_candidates_for_query_task(
			domain,
			"do_put_on",
			"on",
			("ARG1", "ARG2"),
			task_schemas,
			action_analysis,
		)
		if candidate.get("candidate") == "do_move"
	]

	assert len(candidates) == 1
	assert _same_arity_caller_shared_requirements(
		domain,
		"on",
		("ARG1", "ARG2"),
		action_analysis,
		candidates[0],
	) == ("clear(ARG1)", "clear(ARG2)", "handempty")

def test_prune_unreachable_task_structures_preserves_required_contract_tasks():
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("root_task", ("ARG1",), False, ("goal_predicate",)),
			HTNTask("required_helper", ("ARG1",), False, ("helper_predicate",)),
			HTNTask("unused_task", ("ARG1",), False, ("unused_predicate",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_root_noop",
				task_name="root_task",
				parameters=("ARG1",),
				task_args=("ARG1",),
				context=(HTNLiteral("goal_predicate", ("ARG1",), True),),
				subtasks=(),
				ordering=(),
			),
			HTNMethod(
				method_name="m_required_helper_noop",
				task_name="required_helper",
				parameters=("ARG1",),
				task_args=("ARG1",),
				context=(HTNLiteral("helper_predicate", ("ARG1",), True),),
				subtasks=(),
				ordering=(),
			),
			HTNMethod(
				method_name="m_unused_noop",
				task_name="unused_task",
				parameters=("ARG1",),
				task_args=("ARG1",),
				context=(HTNLiteral("unused_predicate", ("ARG1",), True),),
				subtasks=(),
				ordering=(),
			),
		],
		target_literals=[HTNLiteral("goal_predicate", ("obj",), True)],
		target_task_bindings=[HTNTargetTaskBinding("goal_predicate(obj)", "root_task")],
	)

	pruned_library, pruned_count = HTNMethodSynthesizer._prune_unreachable_task_structures(
		library,
		prompt_analysis={
			"support_task_contracts": [
				{"task_name": "required_helper"},
			],
		},
	)

	assert {task.name for task in pruned_library.compound_tasks} == {
		"root_task",
		"required_helper",
	}
	assert {method.task_name for method in pruned_library.methods} == {
		"root_task",
		"required_helper",
	}
	assert pruned_count == 2

def test_required_helper_specs_skip_internal_packaging_support_when_parent_envelope_is_refined():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	action_analysis = synthesizer._analyse_domain_actions(domain)

	helper_specs = _required_helper_specs_for_query_targets(
		domain,
		["on(a, b)"],
		({"task_name": "do_put_on", "args": ["a", "b"]},),
		action_analysis,
	)

	assert helper_specs == []

def test_stage3_validation_rejects_compound_child_calls_with_no_reachable_child_method():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_move", ("BLOCK", "BLOCK"), False, ("on",)),
			HTNTask("do_clear", ("BLOCK",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_move_noop",
				task_name="do_move",
				parameters=("X", "Y"),
				context=(HTNLiteral("on", ("X", "Y"), True, None),),
			),
			HTNMethod(
				method_name="m_do_move_constructive",
				task_name="do_move",
				parameters=("X", "Y"),
				context=(
					HTNLiteral("holding", ("X",), True, None),
					HTNLiteral("clear", ("Y",), False, None),
					HTNLiteral("handempty", (), False, None),
				),
				subtasks=(
					HTNMethodStep("s1", "do_clear", ("Y",), "compound"),
					HTNMethodStep("s2", "stack", ("X", "Y"), "primitive", action_name="stack"),
				),
				ordering=(("s1", "s2"),),
			),
			HTNMethod(
				method_name="m_do_clear_noop",
				task_name="do_clear",
				parameters=("ARG1",),
				context=(HTNLiteral("clear", ("ARG1",), True, None),),
			),
			HTNMethod(
				method_name="m_do_clear_constructive_2",
				task_name="do_clear",
				parameters=("ARG1", "AUX1"),
				context=(
					HTNLiteral("on", ("AUX1", "ARG1"), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep("s1", "do_clear", ("AUX1",), "compound"),
					HTNMethodStep("s2", "unstack", ("AUX1", "ARG1"), "primitive", action_name="unstack"),
					HTNMethodStep("s3", "put_down", ("AUX1",), "primitive", action_name="put-down"),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
			),
		],
		target_literals=[HTNLiteral("on", ("b1", "b2"), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("on(b1, b2)", "do_move")],
	)

	with pytest.raises(ValueError, match="no child method for 'do_clear' is compatible"):
		synthesizer._validate_library(library, domain)

def test_render_signature_with_mapping_does_not_cascade_replacements():
	assert _render_signature_with_mapping(
		"on(?x, ?y)",
		{"?y": "?x", "?x": "BLOCK1"},
	) == "on(BLOCK1, ?x)"

def test_stage3_prompt_uses_declared_source_names_for_hyphenated_task_anchors():
	domain = _marsrover_domain()
	user_prompt = build_htn_user_prompt(
		domain,
		["empty(s0)"],
		'{"target_task_bindings": [], "compound_tasks": [], "methods": []}',
		query_text="Using store s0 and rover rover0, complete the tasks empty-store(s0, rover0).",
		query_task_anchors=(
			{
				"task_name": "empty_store",
				"source_name": "empty-store",
				"args": ["s0", "rover0"],
			},
		),
		query_objects=("s0", "rover0"),
		action_analysis=HTNMethodSynthesizer()._analyse_domain_actions(domain),
	)

	assert "ordered_binding #1: empty(s0) -> empty-store(s0, rover0)" in user_prompt
	assert "ordered_binding #1: empty(s0) -> empty-store(s0, rover0)" in user_prompt
	assert "<query_task_contract name=\"empty-store\">" in user_prompt
	assert "empty_store(s0, rover0)" not in user_prompt

def test_stage3_prompt_stays_compact_for_multi_goal_blocksworld_case():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	targets = [
		"on(b3, b5)",
		"on(b6, b3)",
		"on(b1, b6)",
		"on(b2, b1)",
		"on(b4, b2)",
		"on(b7, b4)",
	]
	system_prompt = build_htn_system_prompt()
	user_prompt = build_htn_user_prompt(
		domain,
		targets,
		HTNMethodSynthesizer._schema_hint(),
		query_text=(
			"Using blocks b1, b2, b3, b4, b5, b6, and b7, complete the tasks "
			"do_put_on(b3, b5), do_put_on(b6, b3), do_put_on(b1, b6), "
			"do_put_on(b2, b1), do_put_on(b4, b2), and do_put_on(b7, b4)."
		),
		query_task_anchors=tuple(
			{"task_name": "do_put_on", "args": list(args)}
			for args in (
				("b3", "b5"),
				("b6", "b3"),
				("b1", "b6"),
				("b2", "b1"),
				("b4", "b2"),
				("b7", "b4"),
			)
		),
		action_analysis=HTNMethodSynthesizer()._analyse_domain_actions(domain),
	)

	assert len(system_prompt) + len(user_prompt) < 30100
	assert "Pattern: if final producer PLACE(ARG1, ARG2) needs READY(ARG2)" in system_prompt
	assert "If a listed template is ACTION(AUX1, ARG1), do not swap it to ACTION(ARG1, AUX1)" in system_prompt
	assert user_prompt.count("<query_task_contract name=\"do_put_on\">") == 1
	assert user_prompt.count("ordered_binding #") == 1
	assert "<required_tasks>" in user_prompt
	assert "- do_put_on(X, Y)" in user_prompt
	assert "- do_move(X, Y)" in user_prompt
	assert "- do_clear(X)" in user_prompt
	assert "<support_task_contract name=\"do_clear\">" in user_prompt
	assert "<support_task_contract name=\"do_move\">" in user_prompt
	assert "<support_task_contract name=\"do_on_table\">" in user_prompt
	assert "Use ARG1..ARGn for task-signature roles and AUX_* for extra roles." in user_prompt
	assert "Prefer ordered_subtasks for total orders" in user_prompt
	assert "Support caller-shared prerequisites clear(?x), clear(?y), handempty before the child call" in user_prompt
	assert "do_move(ARG1, ARG2) expects caller-shared clear(ARG2). Before the child call, establish clear(ARG2)" in user_prompt
	assert "required helper task do_holding(ARG1) before do_move(ARG1, ARG2)." not in user_prompt
	assert "do_move(ARG1, ARG2) expects caller-shared holding(ARG1). Before the child call, establish holding(ARG1)" not in user_prompt
	assert "valid support-then-produce sibling for the false-clear(ARG1) case with pick_up(ARG1): precondition/context ontable(ARG1), handempty, clear(ARG2); support_before do_clear(ARG1); producer pick_up(ARG1); followup stack(ARG1, ARG2)." in user_prompt
	assert "ARG2 acts as a non-leading support/base role in the repeated query skeleton." in user_prompt
	assert "do_on_table(ARG2) can stabilize ARG2 beyond the headline producer's direct prerequisites" in user_prompt
	assert "keep that stabilizer as a real sibling step before same-arity packaging child do_move(ARG1, ARG2)" in user_prompt
	assert "when used as a role stabilizer, required stable/noop branch precondition/context clear(ARG1) instead of requiring ontable(ARG1)" in user_prompt
	assert "if used as a role stabilizer, its constructive branch must internally close ontable(ARG2)" in user_prompt
	assert "valid constructive sibling for put_down(ARG1): precondition/context holding(ARG1); producer put_down(ARG1)." in user_prompt
	assert "support_before do_clear(?x); do_clear(?y); do_on_table(?y); producer do_move(?x, ?y)" in user_prompt
	assert "exact same-arity packaging child for on(?x, ?y) when called by do_put_on(?x, ?y)" in user_prompt
	assert "Parent-side caller-shared prerequisites: clear(?x), clear(?y), handempty. This set is exhaustive at child entry." in user_prompt
	assert "do_move(?x, ?y): if used as same-arity packaging for on(?x, ?y), its own constructive branch must internally close the headline effect via stack(?x, ?y) [needs holding(?x), clear(?y)]." in user_prompt
	assert "do_move(?x, ?y): when used as same-arity packaging for on(?x, ?y), every constructive sibling should keep clear(?x), clear(?y), handempty explicit in branch context" not in user_prompt
	assert "do_move(?x, ?y): with parent-side caller-shared prerequisites holding(?x)" not in user_prompt
	assert "must establish clear(ARG2) via do_clear(ARG2) before stack(ARG1, ARG2) instead of leaving it in branch context" not in user_prompt
	assert "Declaring AUX_* is not enough; constrain it before first use." in user_prompt
	assert "Keep template argument positions exact." in user_prompt
	assert "inferred_task_headline_candidates:" not in user_prompt

def test_prompt_analysis_payload_keeps_role_stabilizer_and_packaging_child_contracts_live():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	synthesizer = HTNMethodSynthesizer()
	analysis = synthesizer._analyse_domain_actions(domain)
	prompt_analysis = build_prompt_analysis_payload(
		domain,
		target_literals=[
			"on(b3, b5)",
			"on(b6, b3)",
			"on(b1, b6)",
			"on(b2, b1)",
			"on(b4, b2)",
			"on(b7, b4)",
		],
		query_task_anchors=tuple(
			{"task_name": "do_put_on", "args": list(args)}
			for args in (
				("b3", "b5"),
				("b6", "b3"),
				("b1", "b6"),
				("b2", "b1"),
				("b4", "b2"),
				("b7", "b4"),
			)
		),
		action_analysis=analysis,
	)

	assert prompt_analysis["query_task_contracts"]
	query_contract = prompt_analysis["query_task_contracts"][0]
	assert any(
		"do_on_table(ARG2) can stabilize ARG2 beyond the headline producer's direct prerequisites"
		in line
		for line in query_contract["contract_lines"]
	)
	assert any(
		"keep that stabilizer as a real sibling step before same-arity packaging child do_move(ARG1, ARG2)"
		in line
		for line in query_contract["contract_lines"]
	)
	assert any(
		"first support child shared prerequisites clear(?x) via do_clear(?x); clear(?y) via do_clear(?y); handempty explicitly in parent context"
		in line
		for line in query_contract["contract_lines"]
	)

	support_contracts = {
		payload["display_name"]: payload["contract_lines"]
		for payload in prompt_analysis["support_task_contracts"]
	}
	assert "do_on_table" in support_contracts
	assert "do_move" in support_contracts
	assert any(
		"when used as a role stabilizer, required stable/noop branch precondition/context clear(ARG1) instead of requiring ontable(ARG1)"
		for line in support_contracts["do_on_table"]
	)
	assert all(
		"required stable/noop branch precondition/context ontable(ARG1)" not in line
		for line in support_contracts["do_on_table"]
	)
	assert any(
		"if used as a role stabilizer, its constructive branch must internally close ontable(ARG2)"
		in line
		for line in support_contracts["do_on_table"]
	)
	assert any(
		"valid constructive sibling for put_down(ARG1): precondition/context holding(ARG1); producer put_down(ARG1)."
		in line
		for line in support_contracts["do_clear"]
	)
	assert any(
		"its own constructive branch must internally close the headline effect via stack(?x, ?y) [needs holding(?x), clear(?y)]"
		in line
		for line in support_contracts["do_move"]
	)

def test_stage3_prompt_stays_compact_for_marsrover_benchmark_case():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	system_prompt = build_htn_system_prompt()
	user_prompt = build_htn_user_prompt(
		domain,
		[
			"communicated_soil_data(waypoint2)",
			"communicated_rock_data(waypoint3)",
			"communicated_image_data(objective1, high_res)",
		],
		HTNMethodSynthesizer._schema_hint(),
		query_text=(
			"Using lander general, modes colour, high_res, and low_res, rover rover0, "
			"store rover0store, waypoints waypoint0, waypoint1, waypoint2, and waypoint3, "
			"camera camera0, and objectives objective0 and objective1, complete the tasks "
			"get_soil_data(waypoint2), get_rock_data(waypoint3), and "
			"get_image_data(objective1, high_res)."
		),
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
			{"task_name": "get_rock_data", "args": ["waypoint3"]},
			{"task_name": "get_image_data", "args": ["objective1", "high_res"]},
		),
		query_object_inventory=(
			{"type": "lander", "objects": ["general"]},
			{"type": "mode", "objects": ["colour", "high_res", "low_res"]},
			{"type": "rover", "objects": ["rover0"]},
			{"type": "store", "objects": ["rover0store"]},
			{"type": "waypoint", "objects": ["waypoint0", "waypoint1", "waypoint2", "waypoint3"]},
			{"type": "camera", "objects": ["camera0"]},
			{"type": "objective", "objects": ["objective0", "objective1"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert len(system_prompt) + len(user_prompt) < 30100
	assert "<query_task_contract name=\"get_soil_data\">" in user_prompt
	assert "<required_tasks>" in user_prompt
	assert "<support_task_contract name=\"send_soil_data\">" in user_prompt
	assert (
		'send_soil_data(?rover, ?waypoint): exact producer slots '
		'[{"producer":"sample_soil(ARG1, AUX_STORE1, ARG2)"}].'
	) in user_prompt
	assert "valid internal-support sibling for the false-at(ARG1, ARG2) case" in user_prompt
	assert "sample_soil(" in user_prompt
	assert "[extra needs" in user_prompt
	assert "empty(AUX_STORE1)" in user_prompt
	assert "take_image(" in user_prompt
	assert "calibrated(AUX_CAMERA1, AUX_ROVER1)" in user_prompt
	assert "at(AUX_ROVER1, AUX_WAYPOINT1)" in user_prompt

def test_stage3_prompt_filters_same_arity_packaging_by_parameter_types():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["communicated_image_data(objective1, high_res)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text=(
			"Using lander general, mode high_res, rover rover0, waypoint waypoint2, "
			"camera camera0, and objective objective1, complete the tasks "
			"get_image_data(objective1, high_res)."
		),
		query_task_anchors=(
			{"task_name": "get_image_data", "args": ["objective1", "high_res"]},
		),
		query_object_inventory=(
			{"type": "lander", "objects": ["general"]},
			{"type": "mode", "objects": ["high_res"]},
			{"type": "rover", "objects": ["rover0"]},
			{"type": "waypoint", "objects": ["waypoint2"]},
			{"type": "camera", "objects": ["camera0"]},
			{"type": "objective", "objects": ["objective1"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert "prefer same-arity declared tasks calibrate_abs(" not in user_prompt
	assert "prefer same-arity declared tasks navigate_abs(" not in user_prompt
	assert "when chosen by get_image_data(?objective, ?mode) as same-arity packaging" not in user_prompt

def test_same_arity_packaging_candidates_require_real_headline_match():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	task_schemas = _declared_task_schema_map(domain)

	candidates = _same_arity_packaging_candidates_for_query_task(
		domain,
		"get_soil_data",
		"communicated_soil_data",
		("WAYPOINT",),
		task_schemas,
		synthesizer._analyse_domain_actions(domain),
	)

	assert candidates == []

def test_rover_support_task_headline_inference_prefers_discriminative_semantic_tokens():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	action_analysis = synthesizer._analyse_domain_actions(domain)

	assert _candidate_headline_predicates_for_task(
		"send_soil_data",
		2,
		action_analysis,
	)[0] == "have_soil_analysis"
	assert _candidate_headline_predicates_for_task(
		"send_rock_data",
		2,
		action_analysis,
	)[0] == "have_rock_analysis"
	assert _candidate_headline_predicates_for_task(
		"get_soil_data",
		1,
		action_analysis,
	)[0] == "communicated_soil_data"

def test_rover_support_task_contracts_render_effect_aligned_headline_roles():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["communicated_image_data(objective0, colour)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text=(
			"Using objective objective0 and mode colour, complete the tasks "
			"get_image_data(objective0, colour)."
		),
		query_task_anchors=(
			{"task_name": "get_image_data", "args": ["objective0", "colour"]},
		),
		query_object_inventory=(
			{"type": "objective", "objects": ["objective0"]},
			{"type": "mode", "objects": ["colour"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert (
		"calibrate_abs(?rover, ?camera): required stable/noop branch "
		"precondition/context calibrated(ARG2, ARG1)."
	) in user_prompt
	assert (
		"calibrate_abs(?rover, ?camera) supports calibrated(?camera, ?rover); "
		"templates: calibrate(?rover, ?camera"
	) in user_prompt

def test_rover_support_task_internal_support_siblings_include_child_shared_requirements():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["communicated_image_data(objective0, colour)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text=(
			"Using objective objective0 and mode colour, complete the tasks "
			"get_image_data(objective0, colour)."
		),
		query_task_anchors=(
			{"task_name": "get_image_data", "args": ["objective0", "colour"]},
		),
		query_object_inventory=(
			{"type": "objective", "objects": ["objective0"]},
			{"type": "mode", "objects": ["colour"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert (
		"calibrate_abs(?rover, ?camera): valid internal-support sibling for the false-"
		"at(ARG1, AUX_WAYPOINT1) case: precondition/context "
	) in user_prompt
	assert "available(ARG1); support_before navigate_abs(ARG1, AUX_WAYPOINT1);" in user_prompt
	assert (
		'AST slot shape for that internal-support sibling: {"precondition":['
	) in user_prompt
	assert '"available(ARG1)"],"support_before":["navigate_abs(ARG1, AUX_WAYPOINT1)"],"producer":"calibrate(' in user_prompt

def test_rover_support_task_contracts_expose_full_caller_shared_envelope():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["communicated_soil_data(waypoint2)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text=(
			"Using waypoint waypoint2, complete the tasks get_soil_data(waypoint2)."
		),
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
		),
		query_object_inventory=(
			{"type": "waypoint", "objects": ["waypoint2"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert (
		'send_soil_data(?rover, ?waypoint): exact producer slots '
		'[{"producer":"sample_soil(ARG1, AUX_STORE1, ARG2)"}].'
	) in user_prompt
	assert "valid internal-support sibling for the false-at(ARG1, ARG2) case" in user_prompt
	assert "available(ARG1); support_before navigate_abs(ARG1, ARG2);" in user_prompt

def test_rover_support_task_invocations_align_with_task_signature_order():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["communicated_image_data(objective0, colour)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text=(
			"Using objective objective0 and mode colour, complete the tasks "
			"get_image_data(objective0, colour)."
		),
		query_task_anchors=(
			{"task_name": "get_image_data", "args": ["objective0", "colour"]},
		),
		query_object_inventory=(
			{"type": "objective", "objects": ["objective0"]},
			{"type": "mode", "objects": ["colour"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert "support_before calibrate_abs(ARG1, AUX_CAMERA1);" in user_prompt
	assert "support_before calibrate_abs(AUX_CAMERA1, ARG1);" not in user_prompt
	assert (
		"support_before calibrate_abs(ARG1, AUX_CAMERA1); navigate_abs(ARG1, AUX_WAYPOINT1); "
		"producer take_image(ARG1, AUX_WAYPOINT1, ARG2, AUX_CAMERA1, ARG3)."
	) in user_prompt
	assert (
		'"support_before":["calibrate_abs(ARG1, AUX_CAMERA1)", "navigate_abs(ARG1, AUX_WAYPOINT1)"],'
	) in user_prompt
	assert (
		"valid internal-support sibling for the false-at(ARG1, AUX_WAYPOINT1) case: "
		"precondition/context calibrated(AUX_CAMERA1, ARG1); on_board(AUX_CAMERA1, ARG1); "
		"equipped_for_imaging(ARG1); supports(AUX_CAMERA1, ARG3); visible_from(ARG2, AUX_WAYPOINT1); "
		"available(ARG1); support_before navigate_abs(ARG1, AUX_WAYPOINT1); "
		"producer take_image(ARG1, AUX_WAYPOINT1, ARG2, AUX_CAMERA1, ARG3)."
	) in user_prompt

def test_rover_query_task_contract_materialises_declared_support_sibling_slots():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["communicated_soil_data(waypoint2)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text=(
			"Using waypoint waypoint2, complete the tasks get_soil_data(waypoint2)."
		),
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
		),
		query_object_inventory=(
			{"type": "waypoint", "objects": ["waypoint2"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert (
		"get_soil_data(ARG1): valid support-then-produce sibling for the "
		"false-have_soil_analysis(AUX_ROVER1, ARG1) case with send_soil_data(AUX_ROVER1, ARG1)"
	) not in user_prompt
	assert (
		"get_soil_data(ARG1): valid support-then-produce sibling for the "
		"false-at(AUX_ROVER1, AUX_WAYPOINT1) case with navigate_abs(AUX_ROVER1, AUX_WAYPOINT1)"
	) not in user_prompt
	assert (
		"get_soil_data(ARG1): complete constructive sibling for "
		"communicate_soil_data(AUX_ROVER1, AUX_LANDER1, ARG1, AUX_WAYPOINT1, AUX_WAYPOINT2): "
		"precondition/context available(AUX_ROVER1), at(AUX_ROVER1, ARG1), "
		"at_soil_sample(ARG1), channel_free(AUX_LANDER1), "
		"at_lander(AUX_LANDER1, AUX_WAYPOINT2), visible(AUX_WAYPOINT1, AUX_WAYPOINT2); support_before "
		"navigate_abs(AUX_ROVER1, AUX_WAYPOINT1); send_soil_data(AUX_ROVER1, ARG1); "
		"producer communicate_soil_data(AUX_ROVER1, AUX_LANDER1, ARG1, AUX_WAYPOINT1, AUX_WAYPOINT2)."
	) in user_prompt
	assert (
		'get_soil_data(ARG1): AST slot shape for that complete constructive sibling: '
		'{"precondition":["available(AUX_ROVER1)", "at(AUX_ROVER1, ARG1)", '
		'"at_soil_sample(ARG1)", "channel_free(AUX_LANDER1)", '
		'"at_lander(AUX_LANDER1, AUX_WAYPOINT2)", "visible(AUX_WAYPOINT1, AUX_WAYPOINT2)"],'
		'"support_before":["navigate_abs(AUX_ROVER1, AUX_WAYPOINT1)", '
		'"send_soil_data(AUX_ROVER1, ARG1)"],"producer":"communicate_soil_data('
		'AUX_ROVER1, AUX_LANDER1, ARG1, AUX_WAYPOINT1, AUX_WAYPOINT2)"}.'
	) in user_prompt

def test_support_task_candidates_do_not_overmatch_generic_data_tokens():
	domain = _marsrover_domain()
	assert _candidate_support_task_names(
		domain,
		"channel_free",
		("AUX_LANDER1",),
		("communicate_soil_data", "communicate_rock_data", "communicate_image_data"),
	) == []

def test_stage3_prompt_does_not_invent_cross_task_packaging_for_rover_data_queries():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["communicated_soil_data(waypoint2)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text=(
			"Using rover rover0, lander general, store rover0store, and waypoint waypoint2, "
			"complete the tasks get_soil_data(waypoint2)."
		),
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
		),
		query_object_inventory=(
			{"type": "rover", "objects": ["rover0"]},
			{"type": "lander", "objects": ["general"]},
			{"type": "store", "objects": ["rover0store"]},
			{"type": "waypoint", "objects": ["waypoint2"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert "exact same-arity packaging contract for communicated_soil_data(?waypoint)" not in user_prompt
	assert (
		"get_rock_data(?waypoint): if used as same-arity packaging for "
		"communicated_soil_data(?waypoint)"
	) not in user_prompt

def test_stage3_prompt_forbids_grounded_constants_and_type_predicates_in_methods():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	system_prompt = build_htn_system_prompt()
	user_prompt = build_htn_user_prompt(
		domain,
		["communicated_soil_data(waypoint2)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text=(
			"Using lander general, rover rover0, store rover0store, and waypoint waypoint2, "
			"complete the tasks get_soil_data(waypoint2)."
		),
		query_task_anchors=(
			{"task_name": "get_soil_data", "args": ["waypoint2"]},
		),
		query_object_inventory=(
			{"type": "lander", "objects": ["general"]},
			{"type": "rover", "objects": ["rover0"]},
			{"type": "store", "objects": ["rover0store"]},
			{"type": "waypoint", "objects": ["waypoint2"]},
		),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert "Grounded query objects may appear only in target_task_bindings and ordered bindings." in system_prompt
	assert "Never invent type predicates unless the domain declares them." in system_prompt
	assert "Do not copy grounded object names into methods; methods must stay schematic." in user_prompt
	assert "Use the ordered query bindings below as the canonical decomposition." in user_prompt
	assert "do not invent aggregate/root wrappers such as do_world, do_all, goal_root, or __top." in user_prompt.lower()

def test_common_child_constructive_requirements_ignore_extra_role_blockers():
	synthesizer = HTNMethodSynthesizer()
	domain = _domain()
	action_schemas = synthesizer._action_schema_map(domain)
	predicate_arities = {
		predicate.name: len(predicate.parameters)
		for predicate in domain.predicates
	}
	dynamic_predicates = set(
		synthesizer._analyse_domain_actions(domain)["dynamic_predicates"]
	)
	task_lookup = {
		"do_on_table": HTNTask("do_on_table", ("X",), False, ("ontable",)),
	}
	step = HTNMethodStep(
		step_id="s1",
		task_name="do_on_table",
		args=("Y",),
		kind="compound",
	)
	child_methods = [
		HTNMethod(
			method_name="m_do_on_table_noop",
			task_name="do_on_table",
			parameters=("X",),
			context=(HTNLiteral("ontable", ("X",), True, None),),
			subtasks=(),
			ordering=(),
			origin="llm",
		),
		HTNMethod(
			method_name="m_do_on_table_constructive",
			task_name="do_on_table",
			parameters=("X", "Z"),
			context=(
				HTNLiteral("on", ("X", "Z"), True, None),
				HTNLiteral("clear", ("X",), True, None),
				HTNLiteral("handempty", (), True, None),
			),
			subtasks=(
				HTNMethodStep(
					step_id="s1",
					task_name="do_clear",
					args=("Z",),
					kind="compound",
				),
				HTNMethodStep(
					step_id="s2",
					task_name="unstack",
					args=("X", "Z"),
					kind="primitive",
					action_name="unstack",
				),
				HTNMethodStep(
					step_id="s3",
					task_name="put_down",
					args=("X",),
					kind="primitive",
					action_name="put_down",
				),
			),
			ordering=(("s1", "s2"), ("s2", "s3")),
			origin="llm",
		),
	]

	requirements = synthesizer._common_child_constructive_requirements(
		step,
		child_methods,
		task_lookup,
		action_schemas,
		predicate_arities,
		dynamic_predicates=dynamic_predicates,
	)

	assert "clear(Y)" in requirements
	assert "handempty" in requirements
	assert "on(Y, Z)" not in requirements

def test_common_child_constructive_requirements_do_not_alias_unbound_child_locals():
	synthesizer = HTNMethodSynthesizer()
	domain = _domain()
	action_schemas = synthesizer._action_schema_map(domain)
	predicate_arities = {
		predicate.name: len(predicate.parameters)
		for predicate in domain.predicates
	}
	dynamic_predicates = set(
		synthesizer._analyse_domain_actions(domain)["dynamic_predicates"]
	)
	task_lookup = {
		"do_clear": HTNTask("do_clear", ("X",), False, ("clear",)),
	}
	step = HTNMethodStep(
		step_id="s1",
		task_name="do_clear",
		args=("AUX_BLOCK1",),
		kind="compound",
	)
	child_methods = [
		HTNMethod(
			method_name="m_do_clear_noop",
			task_name="do_clear",
			parameters=("ARG1",),
			context=(HTNLiteral("clear", ("ARG1",), True, None),),
			subtasks=(),
			ordering=(),
			origin="llm",
		),
		HTNMethod(
			method_name="m_do_clear_constructive",
			task_name="do_clear",
			parameters=("ARG1", "AUX_BLOCK1"),
			context=(
				HTNLiteral("on", ("AUX_BLOCK1", "ARG1"), True, None),
				HTNLiteral("clear", ("AUX_BLOCK1",), True, None),
				HTNLiteral("handempty", (), True, None),
			),
			subtasks=(
				HTNMethodStep(
					step_id="s1",
					task_name="do_clear",
					args=("AUX_BLOCK1",),
					kind="compound",
				),
				HTNMethodStep(
					step_id="s2",
					task_name="unstack",
					args=("AUX_BLOCK1", "ARG1"),
					kind="primitive",
					action_name="unstack",
				),
			),
			ordering=(("s1", "s2"),),
			origin="llm",
		),
	]

	requirements = synthesizer._common_child_constructive_requirements(
		step,
		child_methods,
		task_lookup,
		action_schemas,
		predicate_arities,
		dynamic_predicates=dynamic_predicates,
	)

	assert "on(AUX_BLOCK1, AUX_BLOCK1)" not in requirements
	assert "clear(AUX_BLOCK1)" not in requirements
	assert "handempty" in requirements

def test_common_compatible_child_dynamic_requirements_ignore_internal_branch_split():
	synthesizer = HTNMethodSynthesizer()
	domain = _marsrover_domain()
	action_schemas = synthesizer._action_schema_map(domain)
	predicate_arities = {
		predicate.name: len(predicate.parameters)
		for predicate in domain.predicates
	}
	dynamic_predicates = set(
		synthesizer._analyse_domain_actions(domain)["dynamic_predicates"]
	)
	task_lookup = {
		"helper_empty": HTNTask(
			"helper_empty",
			("ARG1",),
			False,
			("empty",),
			headline_literal=HTNLiteral("empty", ("ARG1",), True, None),
		),
	}
	step = HTNMethodStep(
		step_id="s1",
		task_name="helper_empty",
		args=("AUX_STORE1",),
		kind="compound",
	)
	child_methods = [
		HTNMethod(
			method_name="m_helper_empty_noop",
			task_name="helper_empty",
			parameters=("ARG1",),
			context=(HTNLiteral("empty", ("ARG1",), True, None),),
			subtasks=(),
			ordering=(),
			origin="llm",
		),
		HTNMethod(
			method_name="m_helper_empty_constructive",
			task_name="helper_empty",
			parameters=("ARG1", "AUX_ROVER1"),
			context=(HTNLiteral("full", ("ARG1",), True, None),),
			subtasks=(
				HTNMethodStep(
					step_id="s1",
					task_name="drop",
					args=("AUX_ROVER1", "ARG1"),
					kind="primitive",
					action_name="drop",
				),
			),
			ordering=(),
			origin="llm",
		),
	]

	requirements = synthesizer._common_compatible_child_dynamic_requirements(
		step,
		child_methods,
		task_lookup,
		action_schemas,
		predicate_arities,
		dynamic_predicates=dynamic_predicates,
		available_positive=set(),
		known_negative=set(),
	)

	assert requirements == ()

def test_constructive_validator_rejects_compound_prep_that_does_not_feed_later_requirements():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_on_table", ("X",), False, ("ontable",)),
			HTNTask("do_clear", ("X",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_on_table_noop",
				task_name="do_on_table",
				parameters=("X",),
				context=(
					HTNLiteral("ontable", ("X",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_on_table_constructive",
				task_name="do_on_table",
				parameters=("X", "Z"),
				context=(
					HTNLiteral("on", ("X", "Z"), True, None),
					HTNLiteral("clear", ("X",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_clear",
						args=("Z",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="unstack",
						args=("X", "Z"),
						kind="primitive",
						action_name="unstack",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="put_down",
						args=("X",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_noop",
				task_name="do_clear",
				parameters=("X",),
				context=(
					HTNLiteral("clear", ("X",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_unstack",
				task_name="do_clear",
				parameters=("X", "SUPPORT"),
				context=(
					HTNLiteral("on", ("SUPPORT", "X"), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_clear",
						args=("SUPPORT",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="unstack",
						args=("SUPPORT", "X"),
						kind="primitive",
						action_name="unstack",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="put_down",
						args=("SUPPORT",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_putdown",
				task_name="do_clear",
				parameters=("X",),
				context=(
					HTNLiteral("holding", ("X",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_down",
						args=("X",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(
		ValueError,
		match=r"does not supply any unresolved later dynamic requirement",
	):
		synthesizer._validate_library(library, domain)

def test_ast_header_only_duplicate_task_entry_does_not_become_empty_constructive_branch():
	library = HTNMethodLibrary.from_dict(
		{
			"target_task_bindings": [],
			"primitive_aliases": ["put_down"],
			"call_arities": {
				"helper_holding": 1,
				"put_down": 1,
			},
			"tasks": [
				{
					"name": "helper_handempty",
					"parameters": [],
					"headline": "handempty",
				},
				{
					"name": "helper_handempty",
					"parameters": ["AUX_BLOCK1"],
					"headline": "handempty",
					"support_before": ["helper_holding(AUX_BLOCK1)"],
					"producer": "put_down(AUX_BLOCK1)",
				},
				{
					"name": "helper_holding",
					"parameters": ["ARG1"],
					"headline": "holding(ARG1)",
					"noop": "holding(ARG1)",
				},
			],
		},
	)

	constructive_methods = [
		method
		for method in library.methods
		if method.task_name == "helper_handempty" and method.subtasks
	]
	assert len(constructive_methods) == 1
	assert [step.task_name for step in constructive_methods[0].subtasks] == [
		"helper_holding",
		"put_down",
	]
	assert all(
		method.subtasks or method.context
		for method in library.methods
		if method.task_name == "helper_handempty"
	)

def test_stage3_user_prompt_carries_branchy_action_schemas_into_domain_summary():
	domain = type(
		"DomainStub",
		(),
		{
			"name": "branch_domain",
			"types": ["object"],
			"predicates": [],
			"actions": [
				type(
					"ActionStub",
					(),
					{
						"name": "probe",
						"parameters": ["?x - object"],
						"preconditions": "(or (clear ?x) (holding ?x))",
						"effects": "(and (checked ?x))",
					},
				)(),
				type(
					"ActionStub",
					(),
					{
						"name": "seal_if_clear",
						"parameters": ["?x - object"],
						"preconditions": "(imply (clear ?x) (holding ?x))",
						"effects": "(and (sealed ?x))",
					},
				)(),
			],
		},
	)()

	user_prompt = build_htn_user_prompt(
		domain,
		["checked(a)"],
		'{"target_task_bindings": [], "compound_tasks": [], "methods": []}',
	)

	assert "<domain_summary>" in user_prompt
	assert "relevant_primitive_actions:" in user_prompt
	assert "- probe" in user_prompt
	assert "- seal_if_clear" not in user_prompt
	assert "needs clear(?x) -> holding(?x)" not in user_prompt

def test_action_analysis_includes_producer_effect_patterns():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(domain)

	assert "producer_patterns_by_predicate" in analysis
	clear_patterns = analysis["producer_patterns_by_predicate"]["clear"]
	assert any(
		pattern["effect_signature"] == "clear(?y)"
		and pattern["action_name"] == "unstack"
		and "on(?x, ?y)" in pattern["dynamic_precondition_signatures"]
		for pattern in clear_patterns
	)

def test_validate_library_requires_query_anchor_tasks_and_bindings():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	analysis = synthesizer._analyse_domain_actions(domain)
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("get_soil_data", ("WAYPOINT",), False, ("communicated_soil_data",)),
			HTNTask("soil_report_ready", ("WAYPOINT",), False, ("communicated_soil_data",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_get_soil_data_ready",
				task_name="get_soil_data",
				parameters=("WAYPOINT",),
				context=(HTNLiteral("communicated_soil_data", ("WAYPOINT",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_soil_report_ready_ready",
				task_name="soil_report_ready",
				parameters=("WAYPOINT",),
				context=(HTNLiteral("communicated_soil_data", ("WAYPOINT",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("communicated_soil_data", ("waypoint2",), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding("communicated_soil_data(waypoint2)", "soil_report_ready"),
		],
	)

	with pytest.raises(ValueError, match="must use the ordered query task anchor 'get_soil_data'"):
		synthesizer._validate_library(
			library,
			domain,
			query_task_anchors=(
				{"task_name": "get_soil_data", "args": ["waypoint2"]},
			),
			static_predicates=tuple(analysis["static_predicates"]),
		)

def test_validate_library_rejects_fresh_helper_for_static_predicate():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	analysis = synthesizer._analyse_domain_actions(domain)
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("remember_equipment", ("ROVER",), False, ("equipped_for_soil_analysis",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_remember_equipment_ready",
				task_name="remember_equipment",
				parameters=("ROVER",),
				context=(HTNLiteral("equipped_for_soil_analysis", ("ROVER",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="cannot headline static predicates"):
		synthesizer._validate_library(
			library,
			domain,
			static_predicates=tuple(analysis["static_predicates"]),
		)

def test_validate_library_rejects_primitive_step_literal_not_in_action_effects():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_clear", ("X",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_clear_bad_pickup",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("ontable", ("X",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("X",),
						kind="primitive",
						action_name="pick-up",
						literal=HTNLiteral("clear", ("X",), True, None),
					),
				),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="does not make that positive effect true"):
		synthesizer._validate_library(library, domain)

def test_validate_library_requires_compound_task_to_support_its_headline_predicate():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_on_table", ("X",), False, ("ontable",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_on_table_bad_noop",
				task_name="do_on_table",
				parameters=("X",),
				context=(HTNLiteral("clear", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="headlines 'ontable\\(X\\)'"):
		synthesizer._validate_library(library, domain)

def test_validate_library_rejects_constructive_branch_that_still_requires_headline_literal():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_on_table", ("X",), False, ("ontable",)),
			HTNTask("do_clear", ("X",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_clear_already",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("clear", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_on_table_bad_pickup",
				task_name="do_on_table",
				parameters=("X",),
				context=(
					HTNLiteral("ontable", ("X",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_clear",
						args=("X",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up",
						args=("X",),
						kind="primitive",
						action_name="pick-up",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="put_down",
						args=("X",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="headline literal is currently false"):
		synthesizer._validate_library(library, domain)

def test_validate_library_requires_dynamic_preconditions_to_be_supported_before_primitive_step():
	domain = HDDLParser.parse_domain("src/domains/blocksworld/domain.hddl")
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_clear", ("X",), False, ("clear",)),
			HTNTask("do_on_table", ("X",), False, ("ontable",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_on_table_already",
				task_name="do_on_table",
				parameters=("X",),
				context=(HTNLiteral("ontable", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_bad_preconditions",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("clear", ("X",), False, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_on_table",
						args=("X",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up",
						args=("X",),
						kind="primitive",
						action_name="pick-up",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="without first supporting its dynamic preconditions"):
		synthesizer._validate_library(library, domain)

def test_validate_library_rejects_compound_steps_that_skip_shared_child_dynamic_support():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_put_on", ("X", "Y"), False, ("on",)),
			HTNTask("do_clear", ("X",), False, ("clear",)),
			HTNTask("hold_block", ("X",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_put_on_constructive",
				task_name="do_put_on",
				parameters=("X", "Y"),
				context=(),
				subtasks=(
					HTNMethodStep("s1", "hold_block", ("X",), "compound"),
					HTNMethodStep("s2", "do_clear", ("Y",), "compound"),
					HTNMethodStep(
						"s3",
						"stack",
						("X", "Y"),
						"primitive",
						action_name="stack",
					),
				),
				ordering=(("s1", "s3"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_already",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("clear", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_already",
				task_name="hold_block",
				parameters=("X",),
				context=(HTNLiteral("holding", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("X",),
				context=(
					HTNLiteral("clear", ("X",), True, None),
					HTNLiteral("ontable", ("X",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						"s1",
						"pick_up",
						("X",),
						"primitive",
						action_name="pick-up",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("X", "AUX"),
				context=(
					HTNLiteral("on", ("X", "AUX"), True, None),
					HTNLiteral("clear", ("X",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						"s1",
						"unstack",
						("X", "AUX"),
						"primitive",
						action_name="unstack",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="shared dynamic prerequisites"):
		synthesizer._validate_library(library, domain)

def test_validate_library_rejects_grounded_query_object_leakage():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("get_image_data", ("OBJECTIVE", "MODE"), False, ("communicated_image_data",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_get_image_data_constructive",
				task_name="get_image_data",
				parameters=("OBJECTIVE", "MODE"),
				context=(
					HTNLiteral("on_board", ("camera0", "ROVER"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="take_image",
						args=("ROVER", "WAYPOINT", "OBJECTIVE", "camera0", "MODE"),
						kind="primitive",
						action_name="take_image",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("communicated_image_data", ("objective1", "high_res"), True, None)],
		target_task_bindings=[
			HTNTargetTaskBinding("communicated_image_data(objective1, high_res)", "get_image_data"),
		],
	)

	with pytest.raises(ValueError, match="grounded query object 'camera0'"):
		synthesizer._validate_library(
			library,
			domain,
			query_objects=("camera0", "objective1", "high_res"),
		)

def test_method_validation_rejects_multi_step_methods_without_explicit_ordering():
	domain = _marsrover_domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("send_rock_data", ("WAYPOINT",), False, ("communicated_rock_data",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_send_rock_data_constructive",
				task_name="send_rock_data",
				parameters=("WAYPOINT", "ROVER", "STORE", "LANDER", "CHANNEL"),
				context=(
					HTNLiteral("at_rock_sample", ("WAYPOINT",), True, None),
					HTNLiteral("available", ("ROVER",), True, None),
					HTNLiteral("at_lander", ("LANDER", "WAYPOINT"), True, None),
					HTNLiteral("channel_free", ("CHANNEL",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="empty_store",
						args=("STORE", "ROVER"),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="sample_rock",
						args=("ROVER", "STORE", "WAYPOINT"),
						kind="primitive",
						action_name="sample_rock",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="explicit ordering edges"):
		synthesizer._validate_library(library, domain)

def test_method_synthesizer_rejects_llm_identifiers_that_need_silent_sanitising():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place-on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place-on_noop",
				task_name="place-on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)
	assert normalised.compound_tasks[0].name == "place-on"

	with pytest.raises(ValueError, match="Invalid task identifier 'place-on'"):
		synthesizer._validate_library(normalised, domain)

def test_normalise_llm_library_canonicalises_method_strategy_suffixes():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_B1c_Xt",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)
	assert normalised.methods[0].method_name == "m_place_on_b1c_xt"
	assert normalised.methods[0].source_method_name == "m_place_on_B1c_Xt"
	synthesizer._validate_library(normalised, domain)

def test_normalise_llm_library_rewrites_primitive_action_name_to_source_hddl_name():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B",), False, ("holding",)),
			HTNTask("clear_top", ("B",), False, ("clear",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("B",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("B",),
						kind="compound",
						action_name=None,
						literal=None,
						preconditions=(),
						effects=(),
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up_from_table",
						args=("B",),
						kind="primitive",
						action_name="pick_up_from_table",
						literal=None,
						preconditions=(),
						effects=(),
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)
	primitive_step = normalised.methods[0].subtasks[1]

	assert primitive_step.task_name == "pick_up_from_table"
	assert primitive_step.action_name == "pick-up-from-table"

def test_normalise_llm_library_promotes_used_local_variables_into_method_parameters():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("cover_top", ("BLOCK",), False, ("clear",)),
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_cover_top_acquire",
				task_name="cover_top",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK"),
						kind="primitive",
						action_name="put_on_block",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	assert normalised.methods[0].parameters == ("BLOCK", "BLOCK1")

def test_method_validation_rejects_deprecated_task_prefixes():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("achieve_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_achieve_on_stack",
				task_name="achieve_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="Legacy task name 'achieve_on' is not allowed"):
		synthesizer._validate_library(library, domain)

def test_method_validation_enforces_method_name_contract():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="place_on_stack",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("on(a, b)", "place_on"),
		],
	)

	with pytest.raises(ValueError, match="must follow the exact naming pattern"):
		synthesizer._validate_library(library, domain)

def test_negative_target_requires_constructive_method():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("keep_not_clear", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_keep_not_clear_noop",
				task_name="keep_not_clear",
				parameters=("B",),
				context=(
					HTNLiteral("clear", ("B",), False, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("clear", ("a",), False, "clear_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("!clear(a)", "keep_not_clear"),
		],
	)

	with pytest.raises(ValueError, match="has no constructive non-zero-subtask method"):
		synthesizer._validate_library(library, domain)

def test_negative_target_binding_must_match_negative_semantics():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="put-on-block",
						literal=HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), False, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("!on(a, b)", "place_on"),
		],
	)

	with pytest.raises(ValueError, match="none of that task's methods exposes an already-satisfied context"):
		synthesizer._validate_library(library, domain)

def test_synthesize_preserves_negative_query_target_signatures(monkeypatch):
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	synthesizer.client = object()

	llm_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("remove_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_remove_on_noop",
				task_name="remove_on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), False, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_remove_on_pickup",
				task_name="remove_on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), True, None),
					HTNLiteral("clear", ("B1",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="unstack",
						args=("B1", "B2"),
						kind="primitive",
						action_name="unstack",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[
			HTNTargetTaskBinding("!on(a, b)", "remove_on"),
		],
	)

	def _fake_request_complete_llm_library(
		prompt,
		domain_obj,
		metadata,
		*,
		prompt_analysis=None,
		ast_compiler_defaults=None,
		max_tokens=None,
	):
		del ast_compiler_defaults, max_tokens
		return llm_library, '{"ok": true}', "stop"

	monkeypatch.setattr(
		synthesizer,
		"_request_complete_llm_library",
		_fake_request_complete_llm_library,
	)

	library, metadata = synthesizer.synthesize(
		domain=domain,
		query_text="Keep on(a,b) explicitly false.",
		ordered_literal_signatures=["!on(a, b)"],
	)

	assert metadata["target_literals"] == ["!on(a, b)"]
	assert [literal.to_signature() for literal in library.target_literals] == ["!on(a, b)"]

def test_target_binding_normalisation_accepts_object_form_literal():
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[],
		primitive_tasks=[],
		methods=[],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, None),
		],
		target_task_bindings=[
			HTNTargetTaskBinding(
				{
					"predicate": "on",
					"args": ["a", "b"],
					"is_positive": True,
				},
				"place_on",
			),
		],
	)

	normalised = synthesizer._normalise_target_binding_signatures(library)

	assert normalised.target_task_bindings[0].target_literal == "on(a, b)"

def test_method_validation_rejects_unbound_free_variables_in_subtasks():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("clear_top", ("B",), False, ("clear",)),
			HTNTask("hold_block", ("B",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_clear_top_recursive",
				task_name="clear_top",
				parameters=("B",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("TOP",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_down",
						args=("TOP",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="uses unbound variable 'TOP'"):
		synthesizer._validate_library(library, domain)

def test_method_validation_allows_local_variables_when_bound_in_context():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("clear_top", ("B",), False, ("clear",)),
			HTNTask("hold_block", ("B",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_clear_top_recursive",
				task_name="clear_top",
				parameters=("B",),
				context=(
					HTNLiteral("on", ("TOP", "B"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("TOP",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_down",
						args=("TOP",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)

def test_method_validation_allows_local_variables_when_bound_in_preconditions():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B",), False, ("holding",)),
			HTNTask("clear_top", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("B1",),
				context=(
					HTNLiteral("holding", ("B1",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("B",),
				context=(
					HTNLiteral("on", ("B", "SUPPORT"), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("B",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="unstack",
						args=("B", "SUPPORT"),
						kind="primitive",
						action_name="unstack",
						preconditions=(
							HTNLiteral("on", ("B", "SUPPORT"), True, None),
						),
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)

def test_method_validation_allows_auxiliary_method_parameters_constrained_in_context():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("clear_top", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_clear_top_stack_elsewhere",
				task_name="clear_top",
				parameters=("B", "SUPPORT"),
				context=(
					HTNLiteral("holding", ("B",), True, None),
					HTNLiteral("clear", ("SUPPORT",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="stack",
						args=("B", "SUPPORT"),
						kind="primitive",
						action_name="stack",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)

def test_method_validation_rejects_auxiliary_method_parameters_used_before_constraint():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("clear_top", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_clear_top_stack_elsewhere",
				task_name="clear_top",
				parameters=("B", "SUPPORT"),
				context=(
					HTNLiteral("holding", ("B",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="stack",
						args=("B", "SUPPORT"),
						kind="primitive",
						action_name="stack",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="uses auxiliary parameter 'SUPPORT'"):
		synthesizer._validate_library(library, domain)

def test_method_validation_allows_auxiliary_parameters_first_used_in_compound_step():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("helper_handempty", (), False, ("handempty",)),
			HTNTask("helper_holding", ("ARG1",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_helper_handempty_noop",
				task_name="helper_handempty",
				parameters=(),
				context=(HTNLiteral("handempty", (), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_helper_handempty_constructive",
				task_name="helper_handempty",
				parameters=("AUX_BLOCK1",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="helper_holding",
						args=("AUX_BLOCK1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_down",
						args=("AUX_BLOCK1",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_helper_holding_noop",
				task_name="helper_holding",
				parameters=("ARG1",),
				context=(HTNLiteral("holding", ("ARG1",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)

def test_method_validation_rejects_constructive_branch_that_does_not_support_headline_literal():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_clear", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_clear_already",
				task_name="do_clear",
				parameters=("B",),
				context=(
					HTNLiteral("clear", ("B",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_bad_put_support_down",
				task_name="do_clear",
				parameters=("B", "SUPPORT"),
				context=(
					HTNLiteral("holding", ("SUPPORT",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_down",
						args=("SUPPORT",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(
		ValueError,
		match=r"do not make 'clear\(B\)' true via real subtask effects",
	):
		synthesizer._validate_library(library, domain)

def test_method_validation_accepts_renamed_task_parameters_without_explicit_task_args():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_clear", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_clear_already",
				task_name="do_clear",
				parameters=("TARGET",),
				context=(
					HTNLiteral("clear", ("TARGET",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_constructive",
				task_name="do_clear",
				parameters=("TARGET",),
				context=(
					HTNLiteral("holding", ("TARGET",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_down",
						args=("TARGET",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)

def test_method_validation_rejects_extra_role_support_left_only_in_context():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_clear", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_clear_already",
				task_name="do_clear",
				parameters=("B",),
				context=(
					HTNLiteral("clear", ("B",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_bad_unstack_context_only",
				task_name="do_clear",
				parameters=("B", "SUPPORT"),
				context=(
					HTNLiteral("on", ("SUPPORT", "B"), True, None),
					HTNLiteral("clear", ("SUPPORT",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="unstack",
						args=("SUPPORT", "B"),
						kind="primitive",
						action_name="unstack",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(
		ValueError,
		match=r"leaves extra-role dynamic prerequisite 'clear\(SUPPORT\)' only as context",
	):
		synthesizer._validate_library(library, domain)

def test_method_validation_allows_consumed_mode_selector_to_stay_in_context():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_clear", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_clear_already",
				task_name="do_clear",
				parameters=("B",),
				context=(
					HTNLiteral("clear", ("B",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_putdown",
				task_name="do_clear",
				parameters=("B",),
				context=(
					HTNLiteral("holding", ("B",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s0",
						task_name="put_down",
						args=("B",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_unstack_recursive",
				task_name="do_clear",
				parameters=("B", "SUPPORT"),
				context=(
					HTNLiteral("on", ("SUPPORT", "B"), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_clear",
						args=("SUPPORT",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="unstack",
						args=("SUPPORT", "B"),
						kind="primitive",
						action_name="unstack",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="put_down",
						args=("SUPPORT",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)

def test_method_validation_rejects_recursive_support_literal_left_in_same_context():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_clear", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_clear_already",
				task_name="do_clear",
				parameters=("B",),
				context=(
					HTNLiteral("clear", ("B",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_recursive_bad_context",
				task_name="do_clear",
				parameters=("B", "SUPPORT"),
				context=(
					HTNLiteral("on", ("SUPPORT", "B"), True, None),
					HTNLiteral("clear", ("SUPPORT",), True, None),
					HTNLiteral("handempty", (), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_clear",
						args=("SUPPORT",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="unstack",
						args=("SUPPORT", "B"),
						kind="primitive",
						action_name="unstack",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="put_down",
						args=("SUPPORT",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(
		ValueError,
		match=r"keeps supportable dynamic prerequisite 'clear\(SUPPORT\)' in context",
	):
		synthesizer._validate_library(library, domain)

def test_stage3_prompt_promotes_same_arity_final_producer_prerequisites_to_child_entry():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["on(a, b)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text="Using blocks a and b, complete the tasks do_put_on(a, b).",
		query_task_anchors=(
			{"task_name": "do_put_on", "args": ["a", "b"]},
		),
		query_objects=("a", "b"),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert "support_before do_clear(ARG2); producer do_holding(ARG1); followup do_move(ARG1, ARG2)." not in user_prompt
	assert "support_before do_clear(ARG1); do_clear(ARG2); producer pick_up(ARG1); followup do_move(ARG1, ARG2)." not in user_prompt
	assert "if you use do_clear(ARG2) to support clear(ARG2)" not in user_prompt
	assert "non-leading support/base role" not in user_prompt
	assert "Parent-side caller-shared prerequisites: clear(?x), clear(?y), handempty." in user_prompt
	assert "do_move(?x, ?y): with parent-side caller-shared prerequisites holding(?x)" not in user_prompt

def test_primitive_alias_cannot_use_non_primitive_subtask_kind():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B1",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("B1",),
				context=(
					HTNLiteral("holding", ("B1",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("B1", "B2"),
						kind="compound",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("holding", ("a",), True, "holding_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("holding(a)", "hold_block"),
		],
	)

	with pytest.raises(ValueError, match="Primitive aliases must use kind='primitive'"):
		synthesizer._validate_library(library, domain)

def test_sibling_constructive_methods_must_have_distinguishable_contexts():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
			HTNTask("clear_top", ("BLOCK",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("holding", ("BLOCK",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("BLOCK",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up_from_table",
						args=("BLOCK",),
						kind="primitive",
						action_name="pick-up-from-table",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_table_again",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("BLOCK",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up_from_table",
						args=("BLOCK",),
						kind="primitive",
						action_name="pick-up-from-table",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("holding", ("a",), True, "holding_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("holding(a)", "hold_block"),
		],
	)

	with pytest.raises(ValueError, match="semantically duplicate|not semantically distinguishable"):
		synthesizer._validate_library(library, domain)

def test_redundant_constructive_siblings_are_pruned_before_validation():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
			HTNTask("hold_block", ("BLOCK1",), False, ("holding",)),
			HTNTask("clear_top", ("BLOCK2",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_stack_primary",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("handempty", (), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="clear_top",
						args=("BLOCK2",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="stack",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="stack",
					),
				),
				ordering=(("s1", "s3"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_stack_backup",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("handempty", (), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="clear_top",
						args=("BLOCK2",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s3",
						task_name="stack",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="stack",
					),
				),
				ordering=(("s1", "s3"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("BLOCK1",),
				context=(
					HTNLiteral("holding", ("BLOCK1",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("BLOCK2",),
				context=(
					HTNLiteral("clear", ("BLOCK2",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("on(a, b)", "place_on"),
		],
	)

	pruned_library, pruned_count = synthesizer._prune_redundant_constructive_siblings(
		library,
		domain,
	)

	assert pruned_count == 1
	assert {method.method_name for method in pruned_library.methods} == {
		"m_place_on_noop",
		"m_place_on_stack_primary",
		"m_hold_block_noop",
		"m_clear_top_noop",
	}
	synthesizer._validate_library(pruned_library, domain)

def test_unreachable_wrapper_tasks_are_pruned_before_validation():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_put_on", ("X", "Y"), False, ("on",)),
			HTNTask("do_world", (), False, ()),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_put_on_noop",
				task_name="do_put_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("on", ("X", "Y"), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_put_on_stack",
				task_name="do_put_on",
				parameters=("X", "Y"),
				context=(
					HTNLiteral("holding", ("X",), True, None),
					HTNLiteral("clear", ("Y",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="stack",
						args=("X", "Y"),
						kind="primitive",
						action_name="stack",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_world_sequential",
				task_name="do_world",
				parameters=(),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="do_put_on",
						args=("X", "Y"),
						kind="compound",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), True, "on_a_b")],
		target_task_bindings=[HTNTargetTaskBinding("on(a, b)", "do_put_on")],
	)

	pruned_library, pruned_count = synthesizer._prune_unreachable_task_structures(library)

	assert pruned_count == 2
	assert [task.name for task in pruned_library.compound_tasks] == ["do_put_on"]
	assert {method.method_name for method in pruned_library.methods} == {
		"m_do_put_on_noop",
		"m_do_put_on_stack",
	}
	synthesizer._validate_library(pruned_library, domain)

def test_direct_self_recursive_siblings_are_preserved_when_contexts_are_distinct():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
			HTNTask("clear_top", ("BLOCK",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("ontable", ("BLOCK",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up_from_table",
						args=("BLOCK",),
						kind="primitive",
						action_name="pick-up-from-table",
					),
				),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_clear_first",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("clear", ("BLOCK",), False, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("BLOCK",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="hold_block",
						args=("BLOCK",),
						kind="compound",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("clear", ("BLOCK",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
	)

	pruned_library, pruned_count = synthesizer._prune_redundant_constructive_siblings(
		library,
		domain,
	)

	assert pruned_count == 0
	assert {method.method_name for method in pruned_library.methods} == {
		"m_hold_block_from_table",
		"m_hold_block_clear_first",
		"m_clear_top_noop",
	}

def test_pruning_removes_more_specific_constructive_sibling_when_simpler_one_dominates():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("do_clear", ("X",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_do_clear_already",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("clear", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_putdown",
				task_name="do_clear",
				parameters=("X",),
				context=(HTNLiteral("holding", ("X",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_down",
						args=("X",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_do_clear_stack",
				task_name="do_clear",
				parameters=("X", "SUPPORT"),
				context=(
					HTNLiteral("holding", ("X",), True, None),
					HTNLiteral("clear", ("SUPPORT",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="stack",
						args=("X", "SUPPORT"),
						kind="primitive",
						action_name="stack",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
	)

	pruned_library, pruned_count = synthesizer._prune_redundant_constructive_siblings(
		library,
		domain,
	)

	assert pruned_count == 1
	assert {method.method_name for method in pruned_library.methods} == {
		"m_do_clear_already",
		"m_do_clear_putdown",
	}

def test_single_empty_context_fallback_constructive_sibling_is_preserved():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("X", "Y"), False, ("on",)),
			HTNTask("hold_block", ("X",), False, ("holding",)),
			HTNTask("clear_top", ("Y",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_already",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("on", ("X", "Y"), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_missing_holding",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("clear", ("Y",), True, None),),
				subtasks=(
					HTNMethodStep("s1", "hold_block", ("X",), "compound"),
					HTNMethodStep("s2", "stack", ("X", "Y"), "primitive", "stack"),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_missing_clear",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(HTNLiteral("holding", ("X",), True, None),),
				subtasks=(
					HTNMethodStep("s1", "clear_top", ("Y",), "compound"),
					HTNMethodStep("s2", "stack", ("X", "Y"), "primitive", "stack"),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_missing_both",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(),
				subtasks=(
					HTNMethodStep("s1", "clear_top", ("Y",), "compound"),
					HTNMethodStep("s2", "hold_block", ("X",), "compound"),
					HTNMethodStep("s3", "stack", ("X", "Y"), "primitive", "stack"),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("X",),
				context=(HTNLiteral("holding", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("Y",),
				context=(HTNLiteral("clear", ("Y",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), True, "on_a_b")],
		target_task_bindings=[HTNTargetTaskBinding("on(a, b)", "place_on")],
	)

	pruned_library, pruned_count = synthesizer._prune_redundant_constructive_siblings(
		library,
		domain,
	)

	assert pruned_count == 2
	assert {method.method_name for method in pruned_library.methods} >= {
		"m_place_on_missing_both",
	}
	synthesizer._validate_library(pruned_library, domain)

def test_multiple_empty_context_fallback_constructive_siblings_are_rejected():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("X", "Y"), False, ("on",)),
			HTNTask("clear_top", ("Y",), False, ("clear",)),
			HTNTask("hold_block", ("X",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_missing_both",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(),
				subtasks=(
					HTNMethodStep("s1", "clear_top", ("Y",), "compound"),
					HTNMethodStep("s2", "hold_block", ("X",), "compound"),
					HTNMethodStep("s3", "stack", ("X", "Y"), "primitive", "stack"),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_fallback_2",
				task_name="place_on",
				parameters=("X", "Y"),
				context=(),
				subtasks=(
					HTNMethodStep("s1", "hold_block", ("X",), "compound"),
					HTNMethodStep("s2", "clear_top", ("Y",), "compound"),
					HTNMethodStep("s3", "stack", ("X", "Y"), "primitive", "stack"),
				),
				ordering=(("s1", "s2"), ("s2", "s3")),
				origin="llm",
			),
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("Y",),
				context=(HTNLiteral("clear", ("Y",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("X",),
				context=(HTNLiteral("holding", ("X",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="empty-context fallback branches"):
		synthesizer._validate_library(library, domain)

def test_declared_hyphenated_task_names_are_normalised_internally_but_keep_source_names():
	synthesizer = HTNMethodSynthesizer()
	domain = _marsrover_domain()

	anchors = synthesizer._normalise_query_task_anchors(
		(
			{"task_name": "empty-store", "args": ["s0", "rover0"]},
			{"task_name": "navigate_abs", "args": ["rover0", "waypoint1"]},
		),
		domain,
	)
	assert anchors == (
		{
			"task_name": "empty_store",
			"source_name": "empty-store",
			"args": ["s0", "rover0"],
		},
		{
			"task_name": "navigate_abs",
			"source_name": "navigate_abs",
			"args": ["rover0", "waypoint1"],
		},
	)

	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("empty-store", ("STORE", "ROVER"), False, ("empty",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_empty-store_ready",
				task_name="empty-store",
				parameters=("STORE", "ROVER"),
				context=(HTNLiteral("empty", ("STORE",), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("empty", ("s0",), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("empty(s0)", "empty-store")],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	assert [task.name for task in normalised.compound_tasks] == ["empty_store"]
	assert [task.source_name for task in normalised.compound_tasks] == ["empty-store"]
	assert [method.task_name for method in normalised.methods] == ["empty_store"]
	assert [method.method_name for method in normalised.methods] == ["m_empty_store_ready"]
	assert [binding.task_name for binding in normalised.target_task_bindings] == ["empty_store"]

def test_method_validation_rejects_conflicting_variable_types():
	domain = SimpleNamespace(
		name="typed_domain",
		actions=[
			SimpleNamespace(
				name="place",
				parameters=["?item - block", "?slot - location"],
				preconditions="(and)",
				effects="(and (stored ?item))",
			),
		],
		predicates=[
			SimpleNamespace(
				name="stored",
				parameters=["?item - block"],
				to_signature=lambda: "stored(?item - block)",
			),
		],
		requirements=[],
		types=["block", "location"],
	)
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("misbind", ("BLOCK",), False, ("stored",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_misbind_noop",
				task_name="misbind",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("stored", ("BLOCK",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_misbind_conflict",
				task_name="misbind",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="place",
						args=("BLOCK", "BLOCK"),
						kind="primitive",
						action_name="place",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("stored", ("a",), True, "stored_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("stored(a)", "misbind"),
		],
	)

	with pytest.raises(ValueError, match="conflicting inferred types"):
		synthesizer._validate_library(library, domain)

def test_method_type_validation_prefers_declared_task_signature_over_source_predicate_order():
	synthesizer = HTNMethodSynthesizer()
	task_lookup = {
		"calibrate_abs": HTNTask(
			"calibrate_abs",
			("ROVER", "CAMERA"),
			False,
			("calibrated",),
		),
	}
	method = HTNMethod(
		method_name="m_calibrate_abs_constructive",
		task_name="calibrate_abs",
		parameters=("ROVER", "CAMERA"),
		subtasks=(
			HTNMethodStep(
				step_id="s1",
				task_name="calibrate",
				args=("ROVER", "CAMERA"),
				kind="primitive",
				action_name="calibrate",
			),
		),
		origin="llm",
	)

	synthesizer._validate_method_variable_types(
		method,
		task_lookup,
		action_types={"calibrate": ("ROVER", "CAMERA")},
		task_types={"calibrate_abs": ("ROVER", "CAMERA")},
		predicate_types={"calibrated": ("CAMERA", "ROVER")},
	)

def test_method_type_validation_accepts_subtype_compatible_argument_reuse():
	domain = _satellite_domain()
	synthesizer = HTNMethodSynthesizer()
	task_lookup = {
		"dfa_step_q1_q1_t1_have_image": HTNTask(
			"dfa_step_q1_q1_t1_have_image",
			("ARG1", "ARG2"),
			False,
			("have_image",),
			headline_literal=HTNLiteral("have_image", ("ARG1", "ARG2"), True),
		),
		"helper_power_on": HTNTask("helper_power_on", ("ARG1",), False, ("power_on",)),
		"helper_calibrated": HTNTask("helper_calibrated", ("ARG1",), False, ("calibrated",)),
		"helper_pointing": HTNTask("helper_pointing", ("ARG1", "ARG2"), False, ("pointing",)),
	}
	method = HTNMethod(
		method_name="m_dfa_step_q1_q1_t1_have_image_constructive",
		task_name="dfa_step_q1_q1_t1_have_image",
		parameters=("ARG1", "ARG2", "AUX_SATELLITE1", "AUX_INSTRUMENT1"),
		context=(
			HTNLiteral("on_board", ("AUX_INSTRUMENT1", "AUX_SATELLITE1"), True),
			HTNLiteral("supports", ("AUX_INSTRUMENT1", "ARG2"), True),
		),
		subtasks=(
			HTNMethodStep("s1", "helper_power_on", ("AUX_INSTRUMENT1",), "compound"),
			HTNMethodStep("s2", "helper_calibrated", ("AUX_INSTRUMENT1",), "compound"),
			HTNMethodStep("s3", "helper_pointing", ("AUX_SATELLITE1", "ARG1"), "compound"),
			HTNMethodStep(
				"s4",
				"take_image",
				("AUX_SATELLITE1", "ARG1", "AUX_INSTRUMENT1", "ARG2"),
				"primitive",
				action_name="take_image",
			),
		),
		ordering=(("s1", "s2"), ("s2", "s3"), ("s3", "s4")),
		origin="llm",
	)

	synthesizer._validate_method_variable_types(
		method,
		task_lookup,
		synthesizer._action_type_map(domain),
		{
			"dfa_step_q1_q1_t1_have_image": ("IMAGE_DIRECTION", "MODE"),
			"helper_power_on": ("INSTRUMENT",),
			"helper_calibrated": ("INSTRUMENT",),
			"helper_pointing": ("SATELLITE", "DIRECTION"),
		},
		{
			predicate.name: tuple(
				synthesizer._parameter_type(parameter)
				for parameter in predicate.parameters
			)
			for predicate in domain.predicates
		},
		type_parent_map=synthesizer._build_domain_type_parent_map(domain),
	)

def test_parse_llm_library_ast_task_preserves_explicit_headline_literal():
	synthesizer = HTNMethodSynthesizer()

	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [],
				"tasks": [
					{
						"name": "calibrate_abs",
						"parameters": ["ROVER", "CAMERA"],
						"headline": "calibrated(CAMERA, ROVER)",
						"source_predicates": ["calibrated"],
						"noop": {"precondition": ["calibrated(CAMERA, ROVER)"]},
						"constructive": [
							{
								"precondition": ["at(ROVER, AUX_WAYPOINT1)"],
								"producer": "calibrate(ROVER, CAMERA, AUX_OBJECTIVE1, AUX_WAYPOINT1)",
							},
						],
					},
				],
			},
		)
	)

	assert library.compound_tasks[0].headline_literal is not None
	assert (
		library.compound_tasks[0].headline_literal.to_signature()
		== "calibrated(CAMERA, ROVER)"
	)

def test_validate_library_infers_reordered_task_headline_from_noop_context():
	synthesizer = HTNMethodSynthesizer()
	domain = _marsrover_domain()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("calibrate_abs", ("ARG1", "ARG2"), False, ("calibrated",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_calibrate_abs_noop",
				task_name="calibrate_abs",
				parameters=("ARG1", "ARG2"),
				context=(HTNLiteral("calibrated", ("ARG2", "ARG1"), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_calibrate_abs_constructive",
				task_name="calibrate_abs",
				parameters=("ARG1", "ARG2", "AUX_OBJECTIVE1", "AUX_WAYPOINT1"),
				context=(HTNLiteral("at", ("ARG1", "AUX_WAYPOINT1"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="calibrate",
						args=("ARG1", "ARG2", "AUX_OBJECTIVE1", "AUX_WAYPOINT1"),
						kind="primitive",
						action_name="calibrate",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)
	task = normalised.task_for_name("calibrate_abs")
	assert task is not None
	assert task.headline_literal is not None
	assert task.headline_literal.to_signature() == "calibrated(ARG2, ARG1)"
	synthesizer._validate_library(normalised, domain)

def test_method_validation_allows_mixed_arg_aux_mode_selector_in_context():
	synthesizer = HTNMethodSynthesizer()
	domain = _marsrover_domain()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				"calibrate_abs",
				("ARG1", "ARG2"),
				False,
				("calibrated",),
				headline_literal=HTNLiteral("calibrated", ("ARG2", "ARG1"), True, None),
			),
			HTNTask("navigate_abs", ("ARG1", "ARG2"), False, ("at",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_calibrate_abs_noop",
				task_name="calibrate_abs",
				parameters=("ARG1", "ARG2"),
				context=(HTNLiteral("calibrated", ("ARG2", "ARG1"), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_calibrate_abs_constructive",
				task_name="calibrate_abs",
				parameters=("ARG1", "ARG2", "AUX_OBJECTIVE1", "AUX_WAYPOINT1"),
				context=(
					HTNLiteral("at", ("ARG1", "AUX_WAYPOINT1"), True, None),
					HTNLiteral("equipped_for_imaging", ("ARG1",), True, None),
					HTNLiteral("calibration_target", ("ARG2", "AUX_OBJECTIVE1"), True, None),
					HTNLiteral("visible_from", ("AUX_OBJECTIVE1", "AUX_WAYPOINT1"), True, None),
					HTNLiteral("on_board", ("ARG2", "ARG1"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="calibrate",
						args=("ARG1", "ARG2", "AUX_OBJECTIVE1", "AUX_WAYPOINT1"),
						kind="primitive",
						action_name="calibrate",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_navigate_abs_noop",
				task_name="navigate_abs",
				parameters=("ARG1", "ARG2"),
				context=(HTNLiteral("at", ("ARG1", "ARG2"), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_navigate_abs_constructive",
				task_name="navigate_abs",
				parameters=("ARG1", "ARG2", "AUX_WAYPOINT1"),
				context=(
					HTNLiteral("available", ("ARG1",), True, None),
					HTNLiteral("at", ("ARG1", "AUX_WAYPOINT1"), True, None),
					HTNLiteral("can_traverse", ("ARG1", "AUX_WAYPOINT1", "ARG2"), True, None),
					HTNLiteral("visible", ("AUX_WAYPOINT1", "ARG2"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="navigate",
						args=("ARG1", "AUX_WAYPOINT1", "ARG2"),
						kind="primitive",
						action_name="navigate",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)

def test_method_validation_accepts_supported_equality_constraints():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("keep_apart", ("BLOCK1", "BLOCK2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_keep_apart_distinct",
				task_name="keep_apart",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("=", ("BLOCK1", "BLOCK2"), False, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)

def test_parse_llm_library_rejects_truncated_json_with_clear_error():
	synthesizer = HTNMethodSynthesizer()

	with pytest.raises(ValueError, match="appears truncated"):
		synthesizer._parse_llm_library(
			'{"target_task_bindings": [], "compound_tasks": [',
		)

def test_parse_llm_library_rejects_non_pairwise_ordering_edges():
	synthesizer = HTNMethodSynthesizer()

	with pytest.raises(ValueError, match="length-2 arrays"):
		synthesizer._parse_llm_library(
			json.dumps(
				{
					"target_task_bindings": [],
					"compound_tasks": [
						{
							"name": "do_put_on",
							"parameters": ["BLOCK1", "BLOCK2"],
							"goal_predicates": ["on"],
							"is_top_level": True,
						},
					],
					"methods": [
						{
							"method_name": "m_do_put_on_constructive",
							"task_name": "do_put_on",
							"parameters": ["BLOCK1", "BLOCK2"],
							"context": [],
							"subtasks": [
								{"step_id": "s1", "task_name": "pick_up", "args": ["BLOCK1"]},
								{"step_id": "s2", "task_name": "stack", "args": ["BLOCK1", "BLOCK2"]},
								{"step_id": "s3", "task_name": "nop", "args": []},
							],
							"ordering": [["s1", "s2", "s3"]],
						},
					],
				},
			),
		)

def test_parse_llm_library_accepts_orderings_alias():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [
					{"target_literal": "on(a, b)", "task_name": "place_on"},
				],
				"compound_tasks": [
					{
						"name": "place_on",
						"parameters": ["A", "B"],
						"is_primitive": False,
						"source_predicates": ["on"],
					},
				],
				"methods": [
					{
						"method_name": "m_place_on_constructive",
						"task_name": "place_on",
						"parameters": ["A", "B"],
						"context": [
							{"predicate": "clear", "args": ["B"], "is_positive": True},
						],
						"subtasks": [
							{
								"step_id": "s1",
								"task_name": "pick_up_from_table",
								"args": ["A"],
								"kind": "primitive",
								"action_name": "pick_up",
							},
							{
								"step_id": "s2",
								"task_name": "stack",
								"args": ["A", "B"],
								"kind": "primitive",
								"action_name": "stack",
							},
						],
						"orderings": [["s1", "s2"]],
					},
				],
			},
		),
	)

	assert library.methods[0].ordering == (("s1", "s2"),)

def test_parse_llm_library_accepts_ordering_edges_alias():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [
					{"target_literal": "on(a, b)", "task_name": "place_on"},
				],
				"compound_tasks": [
					{
						"name": "place_on",
						"parameters": ["A", "B"],
						"is_primitive": False,
						"source_predicates": ["on"],
					},
				],
				"methods": [
					{
						"method_name": "m_place_on_constructive",
						"task_name": "place_on",
						"parameters": ["A", "B"],
						"context": [
							{"predicate": "clear", "args": ["B"], "is_positive": True},
						],
						"subtasks": [
							{
								"step_id": "s1",
								"task_name": "pick_up_from_table",
								"args": ["A"],
								"kind": "primitive",
								"action_name": "pick_up",
							},
							{
								"step_id": "s2",
								"task_name": "stack",
								"args": ["A", "B"],
								"kind": "primitive",
								"action_name": "stack",
							},
						],
						"ordering_edges": [["s1", "s2"]],
					},
				],
			},
		),
	)

	assert library.methods[0].ordering == (("s1", "s2"),)

def test_parse_llm_library_accepts_ast_style_task_branches():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [
					{"target_literal": "on(a, b)", "task_name": "place_on"},
				],
				"tasks": [
					{
						"name": "place_on",
						"parameters": ["A", "B"],
						"source_predicates": ["on"],
						"noop": {
							"context": [
								{"predicate": "on", "args": ["A", "B"], "is_positive": True},
							],
						},
						"constructive": [
							{
								"label": "stack_mode",
								"context": [
									{"predicate": "holding", "args": ["A"], "is_positive": True},
								],
								"steps": [
									{
										"id": "s1",
										"kind": "primitive",
										"call": "stack",
										"args": ["A", "B"],
									},
								],
							},
						],
					},
				],
			},
		),
	)

	assert [task.name for task in library.compound_tasks] == ["place_on"]
	assert [method.method_name for method in library.methods] == [
		"m_place_on_noop",
		"m_place_on_stack_mode",
	]
	assert library.methods[0].context[0].to_signature() == "on(A, B)"
	assert library.methods[1].subtasks[0].task_name == "stack"
	assert library.methods[1].subtasks[0].action_name == "stack"

def test_parse_llm_library_ast_defaults_override_model_target_task_bindings():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [
					{"name": "do_put_on", "parameters": ["b3", "b5"]},
				],
				"tasks": [
					{
						"name": "query_root_1_do_put_on",
						"ordered_subtasks": ["dfa_step_q1_q2_on_b3_b5(ARG1, ARG2)"],
					},
				],
			},
		),
		ast_compiler_defaults={
			"target_task_bindings": [
				{
					"target_literal": "on(b3, b5)",
					"task_name": "query_root_1_do_put_on",
				},
			],
			"task_defaults": {
				"query_root_1_do_put_on": {
					"name": "query_root_1_do_put_on",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"noop": "on(ARG1, ARG2)",
				},
				"dfa_step_q1_q2_on_b3_b5": {
					"name": "dfa_step_q1_q2_on_b3_b5",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"noop": "on(ARG1, ARG2)",
				},
			},
			"call_arities": {
				"query_root_1_do_put_on": 2,
				"dfa_step_q1_q2_on_b3_b5": 2,
			},
		},
	)

	assert [binding.target_literal for binding in library.target_task_bindings] == ["on(b3, b5)"]
	assert [binding.task_name for binding in library.target_task_bindings] == ["query_root_1_do_put_on"]

def test_parse_llm_library_ast_defaults_accepts_branch_array_inside_task_level_ordered_subtasks():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [],
				"tasks": [
					{
						"name": "dfa_step_q1_q2_on_b3_b5",
						"ordered_subtasks": [
							{
								"support_before": [
									"helper_clear(ARG2)",
									"helper_holding(ARG1)",
								],
								"producer": "stack(ARG1, ARG2)",
							},
						],
					},
				],
			},
		),
		ast_compiler_defaults={
			"task_defaults": {
				"dfa_step_q1_q2_on_b3_b5": {
					"name": "dfa_step_q1_q2_on_b3_b5",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"noop": "on(ARG1, ARG2)",
				},
				"helper_clear": {
					"name": "helper_clear",
					"parameters": ["ARG1"],
					"headline": "clear(ARG1)",
					"noop": "clear(ARG1)",
				},
				"helper_holding": {
					"name": "helper_holding",
					"parameters": ["ARG1"],
					"headline": "holding(ARG1)",
					"noop": "holding(ARG1)",
				},
			},
			"call_arities": {
				"dfa_step_q1_q2_on_b3_b5": 2,
				"helper_clear": 1,
				"helper_holding": 1,
				"stack": 2,
			},
		},
	)

	constructive_methods = [
		method
		for method in library.methods
		if method.task_name == "dfa_step_q1_q2_on_b3_b5" and method.method_name.endswith("constructive")
	]
	assert len(constructive_methods) == 1
	assert [step.task_name for step in constructive_methods[0].subtasks] == [
		"helper_clear",
		"helper_holding",
		"stack",
	]

def test_parse_llm_library_accepts_constructive_branch_wrapper():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [],
				"tasks": [
					{
						"name": "helper_holding",
						"parameters": ["ARG1"],
						"headline": "holding(ARG1)",
						"constructive": [
							{
								"branch": [
									{
										"parameters": ["ARG1"],
										"support_before": [
											"helper_clear(ARG1)",
											"helper_handempty",
										],
										"producer": "pick_up(ARG1)",
									},
								],
							},
						],
					},
				],
				"primitive_aliases": ["pick_up"],
				"call_arities": {
					"helper_holding": 1,
					"helper_clear": 1,
					"helper_handempty": 0,
					"pick_up": 1,
				},
			},
		),
	)

	constructive_method = next(
		method
		for method in library.methods
		if method.task_name == "helper_holding" and method.subtasks
	)
	assert [step.task_name for step in constructive_method.subtasks] == [
		"helper_clear",
		"helper_handempty",
		"pick_up",
	]
	assert [step.kind for step in constructive_method.subtasks] == [
		"compound",
		"compound",
		"primitive",
	]

def test_parse_llm_library_normalises_ast_style_parameter_aliases():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [
					{"target_literal": "on(b4, b2)", "task_name": "do_put_on"},
				],
				"tasks": [
					{
						"name": "do_put_on",
						"parameters": ["x", "y"],
						"source_predicates": ["on"],
						"noop": {
							"context": [
								{"predicate": "on", "args": ["?x", "?y"], "is_positive": True},
							],
						},
						"constructive": [
							{
								"context": [
									{"predicate": "holding", "args": ["?x"], "is_positive": True},
								],
								"steps": [
									{
										"id": "s1",
										"kind": "compound",
										"call": "do_clear",
										"args": ["?y"],
									},
								],
							},
						],
					},
				],
			},
		),
	)

	assert library.compound_tasks[0].parameters == ("X", "Y")
	assert library.methods[0].parameters == ("X", "Y")
	assert library.methods[0].context[0].to_signature() == "on(X, Y)"
	assert library.methods[1].context[0].to_signature() == "holding(X)"
	assert library.methods[1].subtasks[0].args == ("Y",)

def test_parse_llm_library_merges_duplicate_ast_task_entries():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [
					{"target_literal": "clear(b1)", "task_name": "do_clear"},
				],
				"tasks": [
					{
						"name": "do_clear",
						"parameters": ["ARG1"],
						"source_predicates": ["clear"],
						"noop": {
							"context": [
								{"predicate": "clear", "args": ["ARG1"], "is_positive": True},
							],
						},
						"constructive": [
							{
								"context": [
									{"predicate": "holding", "args": ["ARG1"], "is_positive": True},
								],
								"steps": [
									{
										"id": "s1",
										"kind": "primitive",
										"call": "put_down",
										"args": ["ARG1"],
									},
								],
							},
						],
					},
					{
						"name": "do_clear",
						"parameters": ["ARG1", "AUX1"],
						"source_predicates": ["clear"],
						"noop": {
							"context": [
								{"predicate": "clear", "args": ["ARG1"], "is_positive": True},
							],
						},
						"constructive": [
							{
								"parameters": ["ARG1", "AUX1"],
								"context": [
									{"predicate": "holding", "args": ["ARG1"], "is_positive": True},
								],
								"steps": [
									{
										"id": "s1",
										"kind": "compound",
										"call": "do_clear",
										"args": ["AUX1"],
									},
									{
										"id": "s2",
										"kind": "primitive",
										"call": "stack",
										"args": ["ARG1", "AUX1"],
									},
								],
								"ordering": [["s1", "s2"]],
							},
						],
					},
				],
			},
		),
	)

	assert [task.name for task in library.compound_tasks] == ["do_clear"]
	assert library.compound_tasks[0].parameters == ("ARG1",)
	assert [method.method_name for method in library.methods] == [
		"m_do_clear_noop",
		"m_do_clear_constructive",
		"m_do_clear_constructive_2",
	]
	assert library.methods[2].parameters == ("ARG1", "AUX1")
	assert library.methods[2].subtasks[0].task_name == "do_clear"
	assert library.methods[2].subtasks[1].task_name == "stack"

def test_parse_llm_library_accepts_hddl_grammar_style_branch_aliases():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [
					{"target_literal": "on(a, b)", "task_name": "do_move"},
				],
				"tasks": [
					{
						"name": "do_move",
						"parameters": ["ARG1", "ARG2"],
						"source_predicates": ["on"],
						"noop": {
							"precondition": [
								{"predicate": "on", "args": ["ARG1", "ARG2"], "is_positive": True},
							],
						},
						"constructive": [
							{
								"precondition": [
									{"predicate": "holding", "args": ["ARG1"], "is_positive": True},
								],
								"ordered_subtasks": [
									{
										"id": "s1",
										"kind": "compound",
										"call": "do_clear",
										"args": ["ARG2"],
									},
									{
										"id": "s2",
										"kind": "primitive",
										"call": "stack",
										"args": ["ARG1", "ARG2"],
									},
								],
							},
						],
					},
				],
			},
		),
	)

	assert library.methods[0].context[0].to_signature() == "on(ARG1, ARG2)"
	assert library.methods[1].context[0].to_signature() == "holding(ARG1)"
	assert [step.task_name for step in library.methods[1].subtasks] == ["do_clear", "stack"]
	assert library.methods[1].ordering == (("s1", "s2"),)

def test_parse_llm_library_accepts_hddl_style_literal_and_subtask_shorthand():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [
					{"target_literal": "on(a, b)", "task_name": "do_move"},
				],
				"tasks": [
					{
						"name": "do_move",
						"parameters": ["ARG1", "ARG2"],
						"source_predicates": ["on"],
						"noop": {
							"precondition": ["on(ARG1, ARG2)"],
						},
						"constructive": [
							{
								"precondition": ["holding(ARG1)"],
								"ordered_subtasks": [
									"do_clear(ARG2)",
									"stack(ARG1, ARG2)",
								],
							},
						],
					},
				],
			},
		),
	)

	assert library.methods[0].context[0].to_signature() == "on(ARG1, ARG2)"
	assert library.methods[1].context[0].to_signature() == "holding(ARG1)"
	assert [step.step_id for step in library.methods[1].subtasks] == ["s1", "s2"]
	assert [step.task_name for step in library.methods[1].subtasks] == ["do_clear", "stack"]
	assert library.methods[1].ordering == (("s1", "s2"),)

def test_normalise_llm_library_inferrs_missing_source_predicates_from_contracts():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	analysis = synthesizer._analyse_domain_actions(domain)
	prompt_analysis = build_prompt_analysis_payload(
		domain,
		target_literals=("on(a, b)",),
		query_task_anchors=(
			{"task_name": "do_put_on", "args": ["a", "b"]},
		),
		action_analysis=analysis,
	)
	parsed_library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [
					{"target_literal": "on(a, b)", "task_name": "do_put_on"},
				],
				"tasks": [
					{
						"name": "do_put_on",
						"parameters": ["ARG1", "ARG2"],
						"noop": {"precondition": ["on(ARG1, ARG2)"]},
						"constructive": [
							{"ordered_subtasks": ["do_move(ARG1, ARG2)"]},
						],
					},
					{
						"name": "do_move",
						"parameters": ["ARG1", "ARG2"],
						"noop": {"precondition": ["on(ARG1, ARG2)"]},
						"constructive": [
							{"precondition": ["holding(ARG1)", "clear(ARG2)"], "ordered_subtasks": ["stack(ARG1, ARG2)"]},
						],
					},
					{
						"name": "do_clear",
						"parameters": ["ARG1"],
						"noop": {"precondition": ["clear(ARG1)"]},
						"constructive": [
							{"precondition": ["holding(ARG1)"], "ordered_subtasks": ["put_down(ARG1)"]},
						],
					},
				],
			},
		),
	)

	normalised_library = synthesizer._normalise_llm_library(
		parsed_library,
		domain,
		prompt_analysis=prompt_analysis,
	)
	task_lookup = {
		task.name: task
		for task in normalised_library.compound_tasks
	}

	assert task_lookup["do_put_on"].source_predicates == ("on",)
	assert task_lookup["do_move"].source_predicates == ("on",)
	assert task_lookup["do_clear"].source_predicates == ("clear",)

def test_ast_payload_accepts_support_before_producer_followup_shortcut():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	parsed_library = synthesizer._parse_llm_library(
		json.dumps(
			{
				"target_task_bindings": [
					{"target_literal": "on(a, b)", "task_name": "do_put_on"},
				],
				"tasks": [
					{
						"name": "do_put_on",
						"parameters": ["ARG1", "ARG2"],
						"noop": {"precondition": ["on(ARG1, ARG2)"]},
						"constructive": [
							{
								"precondition": ["ontable(ARG1)", "handempty"],
								"support_before": ["do_clear(ARG1)"],
								"producer": "pick_up(ARG1)",
								"followup": "do_move(ARG1, ARG2)",
							},
						],
					},
				],
			},
		),
	)

	methods = [method for method in parsed_library.methods if method.task_name == "do_put_on"]
	assert len(methods) == 2
	constructive = next(method for method in methods if method.method_name.endswith("constructive"))
	assert [step.task_name for step in constructive.subtasks] == ["do_clear", "pick_up", "do_move"]
	assert constructive.ordering == (("s1", "s2"), ("s2", "s3"))

def test_ast_payload_accepts_bare_branch_level_producer_with_task_argument_passthrough():
	library = HTNMethodLibrary.from_dict(
		{
			"target_task_bindings": [
				{"target_literal": "on(a, b)", "task_name": "query_root_1_do_put_on"},
			],
			"tasks": [
				{
					"name": "query_root_1_do_put_on",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"source_name": "do_put_on",
					"constructive": [
						{"producer": "dfa_step_q1_q2_on_a_b"},
					],
				},
				{
					"name": "dfa_step_q1_q2_on_a_b",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"constructive": [
						{"ordered_subtasks": ["stack(ARG1, ARG2)"]},
					],
				},
			],
		},
	)

	query_root_method = next(
		method
		for method in library.methods
		if method.task_name == "query_root_1_do_put_on"
		and method.method_name.endswith("constructive")
	)
	assert [step.task_name for step in query_root_method.subtasks] == ["dfa_step_q1_q2_on_a_b"]
	assert [step.args for step in query_root_method.subtasks] == [("ARG1", "ARG2")]

def test_ast_payload_accepts_query_root_compact_branches_wrapper():
	library = HTNMethodLibrary.from_dict(
		{
			"target_task_bindings": [
				{"target_literal": "on(a, b)", "task_name": "query_root_1_do_put_on"},
			],
			"tasks": [
				{
					"name": "query_root_1_do_put_on",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"source_name": "do_put_on",
					"constructive": {
						"branches": [
							"dfa_step_q1_q2_on_a_b(ARG1, ARG2)",
							"dfa_step_q3_q4_on_a_b(ARG1, ARG2)",
						],
					},
				},
				{
					"name": "dfa_step_q1_q2_on_a_b",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"constructive": [
						{"ordered_subtasks": ["stack(ARG1, ARG2)"]},
					],
				},
				{
					"name": "dfa_step_q3_q4_on_a_b",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"constructive": [
						{"ordered_subtasks": ["stack(ARG1, ARG2)"]},
					],
				},
			],
		},
	)

	query_root_methods = [
		method
		for method in library.methods
		if method.task_name == "query_root_1_do_put_on"
		and method.subtasks
	]
	assert len(query_root_methods) == 2
	assert {
		tuple(step.task_name for step in method.subtasks)
		for method in query_root_methods
	} == {
		("dfa_step_q1_q2_on_a_b",),
		("dfa_step_q3_q4_on_a_b",),
	}

def test_ast_payload_accepts_task_level_single_method_shorthand():
	library = HTNMethodLibrary.from_dict(
		{
			"tasks": [
				{
					"name": "query_root_1_do_put_on",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"source_name": "do_put_on",
					"ordered_subtasks": [
						{"call": "dfa_step_q1_q2_on_a_b", "args": ["ARG1", "ARG2"]},
					],
				},
				{
					"name": "dfa_step_q1_q2_on_a_b",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"ordered_subtasks": [
						{"call": "stack", "args": ["ARG1", "ARG2"]},
					],
				},
			],
		},
	)

	query_root_method = next(
		method
		for method in library.methods
		if method.task_name == "query_root_1_do_put_on"
	)
	assert [step.task_name for step in query_root_method.subtasks] == ["dfa_step_q1_q2_on_a_b"]
	assert [step.args for step in query_root_method.subtasks] == [("ARG1", "ARG2")]

def test_ast_payload_inferrs_missing_task_parameters_from_headline():
	library = HTNMethodLibrary.from_dict(
		{
			"tasks": [
				{
					"name": "helper_full",
					"headline": "full(ARG1)",
					"constructive": [
						{
							"parameters": ["ARG1"],
							"producer": "unvisit(ARG1)",
						},
					],
				},
			],
		},
	)

	task = library.task_for_name("helper_full")
	assert task is not None
	assert task.parameters == ("ARG1",)
	assert task.headline_literal is not None
	assert task.headline_literal.to_signature() == "full(ARG1)"

def test_method_library_lookup_indexes_refresh_after_list_mutation():
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("task_a", ("ARG1",), False, ("done_a",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_task_a",
				task_name="task_a",
				parameters=("ARG1",),
				subtasks=(),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	assert library.task_for_name("task_a") is not None
	assert [method.method_name for method in library.methods_for_task("task_a")] == [
		"m_task_a",
	]

	library.compound_tasks.append(
		HTNTask("task_b", ("ARG1",), False, ("done_b",)),
	)
	library.methods.append(
		HTNMethod(
			method_name="m_task_b",
			task_name="task_b",
			parameters=("ARG1",),
			subtasks=(),
		),
	)

	assert library.task_for_name("task_b") is not None
	assert [method.method_name for method in library.methods_for_task("task_b")] == [
		"m_task_b",
	]

def test_ast_payload_accepts_ordered_subtask_objects_with_producer_and_parameters():
	library = HTNMethodLibrary.from_dict(
		{
			"tasks": [
				{
					"name": "dfa_step_q1_q2_on_a_b",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"constructive": [
						{
							"precondition": ["clear(ARG2)", "clear(ARG1)", "ontable(ARG1)", "handempty"],
							"ordered_subtasks": [
								{"producer": "pick_up", "parameters": ["ARG1"]},
								{"producer": "stack", "parameters": ["ARG1", "ARG2"]},
							],
						},
					],
				},
			],
		},
	)

	method = next(
		method
		for method in library.methods
		if method.task_name == "dfa_step_q1_q2_on_a_b"
		and method.method_name.endswith("constructive")
	)
	assert [step.task_name for step in method.subtasks] == ["pick_up", "stack"]
	assert [step.args for step in method.subtasks] == [("ARG1",), ("ARG1", "ARG2")]

def test_render_producer_mode_options_prunes_dominated_modes():
	action_analysis = HTNMethodSynthesizer()._analyse_domain_actions(_domain())

	clear_modes = _render_producer_mode_options_for_predicate(
		"clear",
		("ARG1",),
		action_analysis,
	)
	handempty_modes = _render_producer_mode_options_for_predicate(
		"handempty",
		(),
		action_analysis,
	)

	assert [mode_call for mode_call, _ in clear_modes] == [
		"put_down(ARG1)",
		"unstack(AUX_BLOCK1, ARG1)",
	]
	assert [mode_call for mode_call, _ in handempty_modes] == [
		"put_down(AUX_BLOCK1)",
	]

def test_render_producer_mode_options_rejects_identity_preserving_dynamic_modes():
	action_analysis = HTNMethodSynthesizer()._analyse_domain_actions(_marsrover_domain())

	available_modes = _render_producer_mode_options_for_predicate(
		"available",
		("ARG1",),
		action_analysis,
		limit=10,
	)
	channel_free_modes = _render_producer_mode_options_for_predicate(
		"channel_free",
		("ARG1",),
		action_analysis,
		limit=10,
	)

	assert available_modes == ()
	assert channel_free_modes == ()

def test_ast_payload_accepts_compact_target_task_binding_invocation_strings():
	library = HTNMethodLibrary.from_dict(
		{
			"target_task_bindings": [
				{"target_literal": "on(a, b)", "task_name": "do_put_on(a, b)"},
			],
			"tasks": [
				{
					"name": "do_put_on",
					"parameters": ["ARG1", "ARG2"],
					"noop": {"precondition": ["on(ARG1, ARG2)"]},
					"constructive": [
						{"ordered_subtasks": ["do_move(ARG1, ARG2)"]},
					],
				},
				{
					"name": "do_move",
					"parameters": ["ARG1", "ARG2"],
					"noop": {"precondition": ["on(ARG1, ARG2)"]},
					"constructive": [
						{
							"precondition": ["holding(ARG1)", "clear(ARG2)"],
							"ordered_subtasks": ["stack(ARG1, ARG2)"],
						},
					],
				},
			],
		},
	)

	assert [binding.task_name for binding in library.target_task_bindings] == ["do_put_on"]

def test_ast_payload_accepts_compact_noop_string_and_task_level_producer():
	library = HTNMethodLibrary.from_dict(
		{
			"target_task_bindings": [
				{"target_literal": "on(a, b)", "task_name": "query_root_1_do_put_on"},
			],
			"tasks": [
				{
					"name": "query_root_1_do_put_on",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"source_name": "do_put_on",
					"noop": "on(ARG1, ARG2)",
					"producer": "dfa_step_q1_q2_on_a_b(ARG1, ARG2)",
				},
			],
		},
	)

	task = library.compound_tasks[0]
	assert task.name == "query_root_1_do_put_on"
	assert task.headline_literal.to_signature() == "on(ARG1, ARG2)"
	assert task.source_name == "do_put_on"
	noop_method = next(
		method
		for method in library.methods
		if method.task_name == "query_root_1_do_put_on"
		and not method.subtasks
	)
	assert [literal.to_signature() for literal in noop_method.context] == ["on(ARG1, ARG2)"]
	method = next(
		method
		for method in library.methods
		if method.task_name == "query_root_1_do_put_on"
		and method.subtasks
	)
	assert [step.task_name for step in method.subtasks] == ["dfa_step_q1_q2_on_a_b"]
	assert [step.args for step in method.subtasks] == [("ARG1", "ARG2")]

def test_ast_payload_accepts_compact_constructive_subtask_list_and_string_precondition():
	library = HTNMethodLibrary.from_dict(
		{
			"tasks": [
				{
					"name": "helper_holding",
					"parameters": ["ARG1"],
					"headline": "holding(ARG1)",
					"noop": "holding(ARG1)",
					"constructive": [
						"helper_clear(ARG1)",
						"helper_ontable(ARG1)",
						"helper_handempty",
						"pick_up(ARG1)",
					],
				},
				{
					"name": "helper_handempty",
					"parameters": [],
					"headline": "handempty",
					"noop": "handempty",
					"constructive": [
						{
							"parameters": ["AUX1"],
							"precondition": "holding(AUX1)",
							"producer": "put_down(AUX1)",
						},
					],
				},
			],
		},
	)

	holding_method = next(
		method
		for method in library.methods
		if method.task_name == "helper_holding"
		and method.subtasks
	)
	assert [step.task_name for step in holding_method.subtasks] == [
		"helper_clear",
		"helper_ontable",
		"helper_handempty",
		"pick_up",
	]
	aux_method = next(
		method
		for method in library.methods
		if method.task_name == "helper_handempty"
		and method.subtasks
	)
	assert aux_method.parameters == ("AUX1",)
	assert [literal.to_signature() for literal in aux_method.context] == ["holding(AUX1)"]

def test_ast_payload_splits_conjunctive_string_preconditions_and_normalises_constructive_nop():
	library = HTNMethodLibrary.from_dict(
		{
			"tasks": [
				{
					"name": "query_root_1_do_put_on",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"source_name": "do_put_on",
					"constructive": [
						{
							"precondition": "on(ARG1, ARG2)",
							"producer": "nop",
						},
						{
							"producer": "dfa_step_q1_q2_on_a_b(ARG1, ARG2)",
						},
					],
				},
				{
					"name": "helper_holding",
					"parameters": ["ARG1"],
					"headline": "holding(ARG1)",
					"constructive": [
						{
							"precondition": "clear(ARG1) & ontable(ARG1) & handempty",
							"producer": "pick_up(ARG1)",
						},
					],
				},
			],
		},
	)

	query_root_methods = [
		method
		for method in library.methods
		if method.task_name == "query_root_1_do_put_on"
	]
	assert len(query_root_methods) == 2
	noop_method = next(method for method in query_root_methods if not method.subtasks)
	assert [literal.to_signature() for literal in noop_method.context] == ["on(ARG1, ARG2)"]
	holding_method = next(
		method
		for method in library.methods
		if method.task_name == "helper_holding" and method.subtasks
	)
	assert [literal.to_signature() for literal in holding_method.context] == [
		"clear(ARG1)",
		"ontable(ARG1)",
		"handempty",
	]

def test_ast_payload_accepts_constructive_branch_list_shorthand():
	library = HTNMethodLibrary.from_dict(
		{
			"tasks": [
				{
					"name": "helper_holding",
					"parameters": ["ARG1"],
					"headline": "holding(ARG1)",
					"constructive": [
						["pick_up(ARG1)"],
						["unstack(ARG1, AUX1)"],
					],
				},
			],
		},
	)

	methods = [
		method
		for method in library.methods
		if method.task_name == "helper_holding"
	]
	assert len(methods) == 2
	assert [step.task_name for step in methods[0].subtasks] == ["pick_up"]
	assert [step.task_name for step in methods[1].subtasks] == ["unstack"]

def test_ast_payload_defaults_missing_invocation_args_to_task_parameters():
	library = HTNMethodLibrary.from_dict(
		{
			"tasks": [
				{
					"name": "query_root_1_do_put_on",
					"parameters": ["ARG1", "ARG2"],
					"headline": "on(ARG1, ARG2)",
					"source_name": "do_put_on",
					"constructive": ["dfa_step_q1_q2_on_a_b"],
				},
			],
		},
	)

	method = next(
		method
		for method in library.methods
		if method.task_name == "query_root_1_do_put_on" and method.subtasks
	)
	assert [step.args for step in method.subtasks] == [("ARG1", "ARG2")]

def test_stage3_prompt_forbids_leaving_supportable_extra_needs_in_constructive_context():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	user_prompt = build_htn_user_prompt(
		domain,
		["on(a, b)"],
		HTNMethodSynthesizer._schema_hint(),
		query_text="Using blocks a and b, complete the tasks do_put_on(a, b).",
		query_task_anchors=(
			{"task_name": "do_put_on", "args": ["a", "b"]},
		),
		query_objects=("a", "b"),
		action_analysis=synthesizer._analyse_domain_actions(domain),
	)

	assert (
		"If a line lists ACTION [needs ...] or [extra needs ...], satisfy those needs before ACTION."
		in user_prompt
	)
	assert (
		"For supportable AUX_* needs, use earlier subtasks instead of leaving unmet dynamic prerequisites in constructive context."
		in user_prompt
	)

def test_validate_library_rejects_semantically_duplicate_methods():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("holding", ("BLOCK1",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_duplicate_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("holding", ("BLOCK1",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="semantically duplicate"):
		synthesizer._validate_library(library, domain)

def test_request_complete_llm_library_fails_on_truncated_json():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	prompt = {"system": "system", "user": "user"}
	metadata = {"llm_attempts": 0}

	def fake_call_llm(
		prompt_payload: dict,
		*,
		max_tokens: int | None = None,
	):
		return ('{"compound_tasks": [', "length")

	synthesizer._call_llm = fake_call_llm  # type: ignore[method-assign]

	with pytest.raises(HTNSynthesisError, match="truncated before completion"):
		synthesizer._request_complete_llm_library(
			prompt,
			domain,
			metadata,
		)

	assert metadata["llm_attempts"] == 1
	assert len(metadata["llm_attempt_durations_seconds"]) == 1
	assert metadata["llm_response_time_seconds"] >= 0

def test_request_complete_llm_library_preserves_partial_streaming_response_on_failure():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	prompt = {"system": "system", "user": "user"}
	metadata = {"llm_attempts": 0}

	def fake_call_llm(
		prompt_payload: dict,
		*,
		max_tokens: int | None = None,
	):
		raise LLMStreamingResponseError(
			"LLM response did not contain usable textual JSON content. finish_reason='length'",
			partial_text='{"target_task_bindings":[],"tasks":[{"name":"dfa_step_q1_q2"',
			finish_reason="length",
		)

	synthesizer._call_llm = fake_call_llm  # type: ignore[method-assign]

	with pytest.raises(HTNSynthesisError, match="did not contain usable textual JSON content"):
		synthesizer._request_complete_llm_library(
			prompt,
			domain,
			metadata,
		)

	assert metadata["llm_response"].startswith('{"target_task_bindings":[],"tasks"')
	assert metadata["llm_finish_reason"] == "length"

def test_validate_library_rejects_nop_inside_constructive_branch():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				"helper_handempty",
				(),
				False,
				("handempty",),
				headline_literal=HTNLiteral("handempty", (), True, None),
			),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_helper_handempty_noop",
				task_name="helper_handempty",
				parameters=(),
				context=(HTNLiteral("handempty", (), True, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_helper_handempty_constructive",
				task_name="helper_handempty",
				parameters=(),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="nop",
						args=(),
						kind="primitive",
						action_name="nop",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="uses nop in a constructive branch"):
		synthesizer._validate_library(
			library,
			domain,
			query_task_anchors=(),
			prompt_analysis={},
			query_objects=(),
			static_predicates=(),
		)

def test_default_method_task_args_keeps_zero_arity_tasks_argument_free():
	synthesizer = HTNMethodSynthesizer()
	task = HTNTask(
		"helper_handempty",
		(),
		False,
		("handempty",),
		headline_literal=HTNLiteral("handempty", (), True, None),
	)
	method = HTNMethod(
		method_name="m_helper_handempty_constructive",
		task_name="helper_handempty",
		parameters=("AUX_BLOCK1",),
		context=(),
		subtasks=(),
		ordering=(),
		origin="llm",
	)

	assert synthesizer._default_method_task_args(method, task) == ()

class _FakeStage3Completions:
	def __init__(self, scripted_results):
		self.scripted_results = list(scripted_results)
		self.calls = []

	def create(self, **kwargs):
		self.calls.append(kwargs)
		next_result = self.scripted_results.pop(0)
		if isinstance(next_result, Exception):
			raise next_result
		return next_result

def _stage3_response(
	*,
	content=None,
	parsed=None,
	finish_reason="stop",
):
	message = SimpleNamespace(content=content, parsed=parsed)
	return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])

def test_stage3_create_chat_completion_uses_single_non_streaming_text_path():
	synthesizer = HTNMethodSynthesizer()
	completions = _FakeStage3Completions([
		_stage3_response(content='{"target_task_bindings":[],"compound_tasks":[],"methods":[]}'),
	])
	synthesizer.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

	synthesizer._create_chat_completion({"system": "system", "user": "user"}, max_tokens=321)

	assert completions.calls[0]["max_tokens"] == 321
	assert completions.calls[0]["stream"] is False
	assert "response_format" not in completions.calls[0]
	assert "extra_body" not in completions.calls[0]

def test_stage3_create_chat_completion_routes_openrouter_to_model_provider():
	synthesizer = HTNMethodSynthesizer(
		api_key="sk-test",
		model="minimax/minimax-m2",
		base_url="https://openrouter.ai/api/v1",
	)
	completions = _FakeStage3Completions([
		_stage3_response(content='{"target_task_bindings":[],"tasks":[]}'),
	])
	synthesizer.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

	synthesizer._create_chat_completion({"system": "system", "user": "user"}, max_tokens=111)

	assert completions.calls[0]["stream"] is False
	assert completions.calls[0]["extra_body"] == {
		"provider": {
			"only": ["minimax"],
			"allow_fallbacks": False,
		},
	}

def test_stage3_estimates_compact_one_shot_token_budget_from_task_defaults():
	estimated = HTNMethodSynthesizer._estimate_stage3_response_token_budget(
		prompt_analysis={},
		ast_compiler_defaults={
			"task_defaults": {f"task_{index}": {} for index in range(10)},
		},
		default_max_tokens=48000,
	)

	assert estimated == 12000

def test_stage3_caps_minimax_one_shot_token_budget_to_provider_safe_ceiling():
	synthesizer = HTNMethodSynthesizer(model="minimax/minimax-m2")

	assert synthesizer._apply_stage3_provider_token_ceiling(13120) == 8192
	assert synthesizer._apply_stage3_provider_token_ceiling(4096) == 4096

def test_stage3_preserves_non_minimax_one_shot_token_budget():
	synthesizer = HTNMethodSynthesizer(model="openai/gpt-4.1")

	assert synthesizer._apply_stage3_provider_token_ceiling(13120) == 13120

def test_stage3_extract_response_text_falls_back_to_response_model_dump():
	synthesizer = HTNMethodSynthesizer()
	message = SimpleNamespace(content=None, parsed=None)

	class _Response:
		choices = [SimpleNamespace(message=message, finish_reason="stop")]

		@staticmethod
		def model_dump():
			return {
				"choices": [
					{
						"finish_reason": "stop",
						"message": {
							"content": '{"target_task_bindings":[],"tasks":[]}',
						},
					},
				],
			}

	assert synthesizer._extract_response_text(_Response()) == (
		'{"target_task_bindings":[],"tasks":[]}'
	)

def test_stage3_extract_response_text_ignores_empty_message_envelope_dump():
	synthesizer = HTNMethodSynthesizer()
	message = SimpleNamespace(content=None, parsed=None)

	class _Response:
		choices = [SimpleNamespace(message=message, finish_reason="stop")]

		@staticmethod
		def model_dump():
			return {
				"choices": [
					{
						"finish_reason": "stop",
						"message": {
							"content": None,
							"refusal": None,
							"role": "assistant",
							"annotations": None,
							"audio": None,
							"function_call": None,
							"tool_calls": None,
							"reasoning": None,
						},
					},
				],
			}

	with pytest.raises(RuntimeError, match="usable textual JSON content"):
		synthesizer._extract_response_text(_Response())

def test_stage3_consume_streaming_response_returns_first_complete_json_payload():
	synthesizer = HTNMethodSynthesizer()
	stream = [
		SimpleNamespace(
			choices=[
				SimpleNamespace(
					delta=SimpleNamespace(content='preface {"target_task_bindings":[]'),
					finish_reason=None,
				),
			],
		),
		SimpleNamespace(
			choices=[
				SimpleNamespace(
					delta=SimpleNamespace(content=',"tasks":[]} trailing'),
					finish_reason="stop",
				),
			],
		),
	]

	response_text, finish_reason = synthesizer._consume_streaming_llm_response(stream)

	assert response_text == '{"target_task_bindings":[],"tasks":[]}'
	assert finish_reason == "stop"

def test_stage3_call_llm_uses_direct_single_request_path():
	synthesizer = HTNMethodSynthesizer(timeout=0.05)
	captured = {"calls": 0}

	def fake_create_chat_completion(prompt_payload, *, max_tokens=None):
		captured["calls"] += 1
		return _stage3_response(content='{"target_task_bindings":[],"tasks":[]}')

	synthesizer._create_chat_completion = fake_create_chat_completion  # type: ignore[method-assign]

	response_text, finish_reason = synthesizer._call_llm({"system": "system", "user": "user"})

	assert captured["calls"] == 1
	assert response_text == '{"target_task_bindings":[],"tasks":[]}'
	assert finish_reason == "stop"

def test_parse_llm_library_accepts_leading_json_object_with_trailing_junk():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		'{"target_task_bindings":[],"compound_tasks":[],"methods":[]} trailing duplicated text',
	)

	assert isinstance(library, HTNMethodLibrary)

def test_parse_llm_library_accepts_singleton_json_array_wrapper():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		'[{"target_task_bindings":[],"tasks":[]}]',
	)

	assert isinstance(library, HTNMethodLibrary)

def test_parse_llm_library_salvages_single_missing_object_closer_at_tail():
	synthesizer = HTNMethodSynthesizer()
	library = synthesizer._parse_llm_library(
		'{"target_task_bindings":[],"tasks":[{"name":"helper_full","parameters":["ARG1"],'
		'"headline":"full(ARG1)","constructive":[{"producer":"sample_rock(ARG1)"},'
		'{"producer":"sample_soil(ARG1)"]}]}',
	)

	assert isinstance(library, HTNMethodLibrary)
	assert [task.name for task in library.compound_tasks] == ["helper_full"]
	assert len(library.methods) == 2

def test_negative_target_binding_rejects_helper_call_with_hidden_support_role():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("remove_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_remove_on_noop",
				task_name="remove_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("on", ("BLOCK1", "BLOCK2"), False, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_remove_on_from_block",
				task_name="remove_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("BLOCK", "SUPPORT"),
				context=(
					HTNLiteral("on", ("BLOCK", "SUPPORT"), True, None),
					HTNLiteral("clear", ("BLOCK",), True, None),
					HTNLiteral("handempty", (), True, None),
					HTNLiteral("=", ("BLOCK", "SUPPORT"), False, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="unstack",
						args=("BLOCK", "SUPPORT"),
						kind="primitive",
						action_name="unstack",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), False, "on_a_b")],
		target_task_bindings=[HTNTargetTaskBinding("!on(a, b)", "remove_on")],
	)

	with pytest.raises(ValueError, match="none of its constructive methods makes '!on\\(BLOCK1, BLOCK2\\)' true"):
		synthesizer._validate_library(library, domain)
