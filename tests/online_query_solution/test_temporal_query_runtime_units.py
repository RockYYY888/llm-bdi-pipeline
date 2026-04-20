from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from offline_method_generation.method_synthesis.schema import HTNMethodLibrary, HTNTask
from offline_method_generation.artifacts import DomainLibraryArtifact
from online_query_solution.artifacts import DFACompilationResult, GroundedSubgoal
from online_query_solution.agentspeak import (
	AgentSpeakRenderer,
	build_agentspeak_transition_specs,
)
from online_query_solution.goal_grounding.grounder import NLToLTLfGenerator
from online_query_solution.goal_grounding.grounding_map import GroundingMap
from online_query_solution.jason_runtime.environment_adapter import EnvironmentAdapterResult
from online_query_solution.jason_runtime.runner import JasonRunner, JasonValidationResult
from online_query_solution.official_verification import resolve_verification_domain_file
from online_query_solution.orchestrator import OnlineQuerySolutionOrchestrator
from online_query_solution.temporal_compilation import dfa_builder
from online_query_solution.temporal_compilation import ltlf_to_dfa as ltlf_to_dfa_module
from pipeline.artifacts import TemporalGroundingResult
from utils.hddl_parser import HDDLParser
from verification.official_plan_verifier import IPCPrimitivePlanVerificationResult
from online_query_solution import official_verification as online_official_verification_module
from online_query_solution import orchestrator as online_orchestrator_module


def _sample_method_library() -> HTNMethodLibrary:
	return HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="stack",
				parameters=("x", "y"),
				is_primitive=False,
				source_name="stack",
			),
			HTNTask(
				name="do_put_on",
				parameters=("x", "y"),
				is_primitive=False,
				source_name="do_put_on",
			),
		],
		primitive_tasks=[],
		methods=[],
	)


def test_goal_grounding_validator_accepts_grounded_task_event_formula() -> None:
	generator = NLToLTLfGenerator()
	result = generator._validate_payload(
		query_text="stack block b on a",
		payload={
			"ltlf_formula": "F(stack(b, a))",
		},
		method_library=_sample_method_library(),
		typed_objects={"a": "block", "b": "block"},
		task_type_map={"stack": ("block", "block")},
		type_parent_map={"block": "object", "object": None},
	)

	assert result.ltlf_formula == "F(stack(b, a))"
	assert len(result.subgoals) == 1
	assert result.subgoals[0].task_name == "stack"
	assert result.subgoals[0].args == ("b", "a")
	assert result.subgoals[0].subgoal_id == "stack_b_a"


def test_goal_grounding_validator_accepts_occurrence_tagged_repeated_task_events() -> None:
	generator = NLToLTLfGenerator()
	result = generator._validate_payload(
		query_text="repeat stack",
		payload={
			"ltlf_formula": "F(stack__e1(b, a) & F(stack__e2(b, a)))",
		},
		method_library=_sample_method_library(),
		typed_objects={"a": "block", "b": "block"},
		task_type_map={"stack": ("block", "block")},
		type_parent_map={"block": "object", "object": None},
	)

	assert [subgoal.task_name for subgoal in result.subgoals] == ["stack", "stack"]
	assert [subgoal.subgoal_id for subgoal in result.subgoals] == [
		"stack__e1_b_a",
		"stack__e2_b_a",
	]


def test_goal_grounding_validator_allows_repeated_formula_references_to_same_event_atom() -> None:
	generator = NLToLTLfGenerator()

	result = generator._validate_payload(
		query_text="strictly order two grounded task events",
		payload={
			"ltlf_formula": "F(stack(b, a)) & F(stack(c, b)) & "
			"(!stack(c, b) U (stack(b, a) & !stack(c, b) & X F(stack(c, b))))",
		},
		method_library=_sample_method_library(),
		typed_objects={"a": "block", "b": "block", "c": "block"},
		task_type_map={"stack": ("block", "block")},
		type_parent_map={"block": "object", "object": None},
	)

	assert [subgoal.subgoal_id for subgoal in result.subgoals] == [
		"stack_b_a",
		"stack_c_b",
	]


def test_goal_grounding_validator_rejects_extra_semantic_keys() -> None:
	generator = NLToLTLfGenerator()

	with pytest.raises(ValueError, match="only the key ltlf_formula"):
		generator._validate_payload(
			query_text="stack block b on a",
			payload={
				"ltlf_formula": "F(stack(b, a))",
				"subgoals": [
					{"id": "subgoal_1", "task_name": "stack", "args": ["b", "a"]},
				],
			},
			method_library=_sample_method_library(),
			typed_objects={"a": "block", "b": "block"},
			task_type_map={"stack": ("block", "block")},
			type_parent_map={"block": "object", "object": None},
		)


def test_goal_grounding_validator_rejects_placeholder_formula_atoms() -> None:
	generator = NLToLTLfGenerator()

	with pytest.raises(ValueError, match="placeholder atoms"):
		generator._validate_payload(
			query_text="stack block b on a",
			payload={
				"ltlf_formula": "F(subgoal_2)",
			},
			method_library=_sample_method_library(),
			typed_objects={"a": "block", "b": "block"},
			task_type_map={"stack": ("block", "block")},
			type_parent_map={"block": "object", "object": None},
		)


def test_goal_grounding_validator_rejects_lifted_arguments() -> None:
	generator = NLToLTLfGenerator()

	with pytest.raises(ValueError, match="lifted variables"):
		generator._validate_payload(
			query_text="stack something on a",
			payload={
				"ltlf_formula": "F(stack(?x, a))",
			},
			method_library=_sample_method_library(),
			typed_objects={"a": "block", "b": "block"},
			task_type_map={"stack": ("block", "block")},
			type_parent_map={"block": "object", "object": None},
		)


def test_goal_grounding_formula_atom_extractor_accepts_operator_sequences() -> None:
	atoms = NLToLTLfGenerator._extract_formula_atoms("XF(stack(b, a))")

	assert atoms == {"stack(b, a)"}


def test_goal_grounding_validator_repairs_formula_parentheses_before_validation() -> None:
	generator = NLToLTLfGenerator()

	result = generator._validate_payload(
		query_text="ordered stack",
		payload={
			"ltlf_formula": "F(stack(b, a) & F(stack(c, b))))",
		},
		method_library=_sample_method_library(),
		typed_objects={"a": "block", "b": "block", "c": "block"},
		task_type_map={"stack": ("block", "block")},
		type_parent_map={"block": "object", "object": None},
	)

	assert result.ltlf_formula == "F(stack(b, a) & F(stack(c, b)))"


def test_goal_grounding_validator_accepts_large_grounded_formula_without_fragment_gate() -> None:
	generator = NLToLTLfGenerator()
	typed_objects = {f"b{index}": "block" for index in range(1, 103)}
	result = generator._validate_payload(
		query_text="large ordered stack",
		payload={
			"ltlf_formula": " & ".join(
				f"F(stack(b{index + 1}, b{index}))"
				for index in range(1, 101)
			),
		},
		method_library=_sample_method_library(),
		typed_objects=typed_objects,
		task_type_map={"stack": ("block", "block")},
		type_parent_map={"block": "object", "object": None},
	)

	assert len(result.subgoals) == 100


def test_goal_grounding_prompt_requires_explicit_order_preservation() -> None:
	domain = HDDLParser.parse_domain(
		str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
	)
	generator = NLToLTLfGenerator(domain_file=str(
		PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"
	))
	generator.domain = domain

	system_prompt, _ = generator._build_prompts(
		query_text="First do_put_on(b4, b2), then do_put_on(b1, b4).",
		method_library=_sample_method_library(),
		typed_objects={"b1": "block", "b2": "block", "b4": "block"},
		task_type_map={"do_put_on": ("block", "block")},
	)

	assert "Preserve temporal meaning exactly." in system_prompt
	assert "Do not collapse an explicitly ordered task list" in system_prompt
	assert "treat that inventory as context only" in system_prompt
	assert "Avoid deeply nested eventuality chains" in system_prompt
	assert "shallow adjacent strict-precedence encoding" in system_prompt
	assert "(!B U (A & !B))" in system_prompt
	assert "do_put_on__e1(b8, b9)" in system_prompt
	assert "final JSON answer must appear in the completion response content itself" in system_prompt
	assert "MUST NOT leave the final answer only in hidden reasoning content" in system_prompt
	assert "MUST NOT return an empty completion response" in system_prompt
	assert "goal_facts" not in system_prompt


def test_goal_grounding_requires_llm_for_benchmark_style_query() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)

	with pytest.raises(RuntimeError, match="No API key configured"):
		generator.generate(
			"Using blocks b4, b2, and b1, complete the tasks do_put_on(b4, b2), then do_put_on(b1, b4).",
			method_library=_sample_method_library(),
			typed_objects={"b1": "block", "b2": "block", "b4": "block"},
			task_type_map={"do_put_on": ("block", "block")},
			type_parent_map={"block": "object", "object": None},
		)


def test_goal_grounding_benchmark_style_query_still_uses_llm_prompting() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)
	generator.client = object()

	seen_messages: list[list[dict[str, str]]] = []
	response_text = (
		'{"ltlf_formula":"F(do_put_on(b4, b2)) & F(do_put_on(b1, b4))"}'
	)

	def fake_create(self, messages, **_kwargs):
		seen_messages.append(list(messages))
		return response_text

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(NLToLTLfGenerator, "_create_chat_completion", fake_create)
	monkeypatch.setattr(NLToLTLfGenerator, "_extract_response_text", lambda self, response: str(response))
	try:
		result, llm_prompt, llm_response = generator.generate(
			"Using blocks b4, b2, and b1, complete the tasks do_put_on(b4, b2), do_put_on(b1, b4).",
			method_library=_sample_method_library(),
			typed_objects={"b1": "block", "b2": "block", "b4": "block"},
			task_type_map={"do_put_on": ("block", "block")},
			type_parent_map={"block": "object", "object": None},
		)
	finally:
		monkeypatch.undo()

	assert result.ltlf_formula == "F(do_put_on(b4, b2)) & F(do_put_on(b1, b4))"
	assert [subgoal.task_name for subgoal in result.subgoals] == ["do_put_on", "do_put_on"]
	assert [subgoal.args for subgoal in result.subgoals] == [("b4", "b2"), ("b1", "b4")]
	assert llm_prompt["user"].endswith(
		'"Using blocks b4, b2, and b1, complete the tasks do_put_on(b4, b2), do_put_on(b1, b4)."'
	)
	assert llm_response == response_text
	assert len(seen_messages) == 1


def test_goal_grounding_prompt_attempts_do_not_special_case_benchmark_queries() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)

	attempts = generator._build_prompt_attempts(
		query_text="Using blocks b4, b2, and b1, complete the tasks do_put_on(b4, b2), do_put_on(b1, b4).",
		method_library=_sample_method_library(),
		typed_objects={"b1": "block", "b2": "block", "b4": "block"},
		task_type_map={"do_put_on": ("block", "block")},
	)

	assert tuple(attempt["mode"] for attempt in attempts) == ("few_shot_strict",)


def test_goal_grounding_prompt_ignores_setup_inventory_and_requires_strict_json() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)

	system_prompt, _ = generator._build_prompts(
		query_text=(
			"Using blocks b4, b2, and b1, complete the tasks "
			"do_put_on(b4, b2), then do_put_on(b1, b4)."
		),
		method_library=_sample_method_library(),
		typed_objects={"b1": "block", "b2": "block", "b4": "block"},
		task_type_map={"do_put_on": ("block", "block")},
	)

	assert "treat that inventory as context only" in system_prompt
	assert "preserve the repeated order and count" in system_prompt
	assert "Output must be minified JSON" in system_prompt
	assert "do_put_on__e2(b8, b9)" in system_prompt


def test_goal_grounding_generate_does_not_retry_after_nontransport_response_extraction_error() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)
	generator.client = object()
	call_count = {"create": 0}

	def fake_create(self, messages, **_kwargs):
		call_count["create"] += 1
		return {"raw": "response"}

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(NLToLTLfGenerator, "_create_chat_completion", fake_create)
	monkeypatch.setattr(
		NLToLTLfGenerator,
		"_extract_response_text",
		lambda self, response: (_ for _ in ()).throw(
			RuntimeError("Synthetic extraction failure."),
		),
	)
	try:
		with pytest.raises(RuntimeError, match="Synthetic extraction failure"):
			generator.generate(
				"Using blocks b1 and b2, complete the tasks do_put_on(b1, b2).",
				method_library=_sample_method_library(),
				typed_objects={"b1": "block", "b2": "block"},
				task_type_map={"do_put_on": ("block", "block")},
				type_parent_map={"block": "object", "object": None},
			)
	finally:
		monkeypatch.undo()

	assert call_count["create"] == 1


def test_goal_grounding_retries_after_empty_response_extraction_and_then_succeeds() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)
	generator.client = object()
	call_count = {"create": 0, "extract": 0}

	def fake_create(self, messages, **_kwargs):
		del messages
		call_count["create"] += 1
		return {"raw": "response"}

	def fake_extract(self, response):
		del response
		call_count["extract"] += 1
		if call_count["extract"] < 3:
			raise RuntimeError("LLM response did not contain usable textual JSON content.")
		return '{"ltlf_formula":"F(do_put_on(b1, b2))"}'

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(NLToLTLfGenerator, "_create_chat_completion", fake_create)
	monkeypatch.setattr(NLToLTLfGenerator, "_extract_response_text", fake_extract)
	try:
		result, _, _ = generator.generate(
			"Using blocks b1 and b2, complete the tasks do_put_on(b1, b2).",
			method_library=_sample_method_library(),
			typed_objects={"b1": "block", "b2": "block"},
			task_type_map={"do_put_on": ("block", "block")},
			type_parent_map={"block": "object", "object": None},
		)
	finally:
		monkeypatch.undo()

	assert result.ltlf_formula == "F(do_put_on(b1, b2))"
	assert call_count["create"] == 3
	assert call_count["extract"] == 3
	assert generator.last_generation_metadata["attempt_count"] == 3
	assert len(generator.last_generation_metadata["attempt_errors"]) == 2


def test_goal_grounding_retries_timeout_before_first_chunk_and_then_succeeds() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)
	generator.client = object()
	call_count = {"create": 0}

	def fake_create(self, messages, **_kwargs):
		del messages
		call_count["create"] += 1
		if call_count["create"] < 3:
			raise TimeoutError(
				"Goal-grounding LLM request exceeded the configured wall-clock timeout "
				"before a response chunk was created.",
			)
		return '{"ltlf_formula":"F(do_put_on(b1, b2))"}'

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(NLToLTLfGenerator, "_create_chat_completion", fake_create)
	monkeypatch.setattr(NLToLTLfGenerator, "_extract_response_text", lambda self, response: str(response))
	try:
		result, _, _ = generator.generate(
			"Using blocks b1 and b2, complete the tasks do_put_on(b1, b2).",
			method_library=_sample_method_library(),
			typed_objects={"b1": "block", "b2": "block"},
			task_type_map={"do_put_on": ("block", "block")},
			type_parent_map={"block": "object", "object": None},
		)
	finally:
		monkeypatch.undo()

	assert result.ltlf_formula == "F(do_put_on(b1, b2))"
	assert call_count["create"] == 3
	assert generator.last_generation_metadata["attempt_count"] == 3
	assert len(generator.last_generation_metadata["attempt_errors"]) == 2


def test_goal_grounding_stops_after_three_transport_retries() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)
	generator.client = object()
	call_count = {"create": 0}

	def fake_create(self, messages, **_kwargs):
		del messages
		call_count["create"] += 1
		raise TimeoutError(
			"Goal-grounding LLM request exceeded the configured wall-clock timeout "
			"before a response chunk was created.",
		)

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(NLToLTLfGenerator, "_create_chat_completion", fake_create)
	try:
		with pytest.raises(TimeoutError, match="before a response chunk was created"):
			generator.generate(
				"Using blocks b1 and b2, complete the tasks do_put_on(b1, b2).",
				method_library=_sample_method_library(),
				typed_objects={"b1": "block", "b2": "block"},
				task_type_map={"do_put_on": ("block", "block")},
				type_parent_map={"block": "object", "object": None},
			)
	finally:
		monkeypatch.undo()

	assert call_count["create"] == 4
	assert generator.last_generation_metadata["attempt_count"] == 4
	assert len(generator.last_generation_metadata["attempt_errors"]) == 4


def test_dfa_builder_wraps_converter_output(monkeypatch: pytest.MonkeyPatch) -> None:
	class FakeConverter:
		def convert(self, formula: str):
			assert formula == "G(do_put_on(b4, b2))"
			return (
				'digraph G { init -> q0; q0 [shape=circle]; q1 [shape=doublecircle]; q0 -> q1 [label="do_put_on_b4_b2"]; }',
				{
					"construction": "fake_converter",
					"num_states": 2,
					"num_transitions": 1,
					"alphabet": ["do_put_on_b4_b2"],
				},
			)

	monkeypatch.setattr(dfa_builder, "LTLfToDFA", lambda: FakeConverter())

	result = dfa_builder.build_dfa_from_ltlf("G(do_put_on(b4, b2))")

	assert result["construction"] == "fake_converter"
	assert result["num_states"] == 2
	assert result["num_transitions"] == 1
	assert result["alphabet"] == ["do_put_on_b4_b2"]
	assert "do_put_on_b4_b2" in result["dfa_dot"]


def test_dfa_builder_always_uses_ltlf2dfa_for_ordered_sequence(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	seen_formulas: list[str] = []

	class FakeConverter:
		def convert(self, formula: str):
			seen_formulas.append(formula)
			return (
				'digraph G { init -> q0; q0 [shape=circle]; q1 [shape=doublecircle]; q0 -> q1 [label="stack_b_a"]; q1 -> q2 [label="stack_c_b"]; }',
				{
					"construction": "generic_ltlf2dfa",
					"num_states": 3,
					"num_transitions": 2,
					"alphabet": ["stack_b_a", "stack_c_b"],
				},
			)

	monkeypatch.setattr(dfa_builder, "LTLfToDFA", lambda: FakeConverter())

	result = dfa_builder.build_dfa_from_ltlf(
		"F(stack(b, a) & F(stack(c, b)))",
	)

	assert seen_formulas == ["F(stack(b, a) & F(stack(c, b)))"]
	assert result["construction"] == "generic_ltlf2dfa"
	assert result["alphabet"] == ["stack_b_a", "stack_c_b"]


def test_dfa_builder_always_uses_ltlf2dfa_for_large_ordered_sequence(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	captured: list[str] = []

	class FakeConverter:
		def convert(self, formula: str):
			captured.append(formula)
			return ("digraph G { init -> q0; }", {"construction": "generic_ltlf2dfa"})

	monkeypatch.setattr(dfa_builder, "LTLfToDFA", lambda: FakeConverter())

	formula = "F(stack(b2, b1))"
	for index in range(3, 51):
		formula = f"F(stack(b{index}, b{index - 1}) & {formula})"

	result = dfa_builder.build_dfa_from_ltlf(formula)

	assert len(captured) == 1
	assert captured[0] == formula
	assert result["construction"] == "generic_ltlf2dfa"


def test_agentspeak_renderer_emits_subgoal_runtime_without_query_step() -> None:
	domain = HDDLParser.parse_domain(str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"))
	renderer = AgentSpeakRenderer()
	asl = renderer.generate(
		domain=domain,
		objects=("a", "b"),
		method_library=_sample_method_library(),
		plan_records=(),
		typed_objects=(("a", "block"), ("b", "block")),
		ordered_query_sequence=False,
		transition_specs=(
			{
				"transition_name": "dfa_t1",
				"source_state": "q0",
				"target_state": "q0",
				"raw_source_state": "q0",
				"raw_target_state": "q0",
				"raw_label": "subgoal_1",
				"guard_context": "true",
				"subgoal_index": 1,
				"initial_state": "q0",
				"accepting_states": ("qf",),
				"stateless_subgoal": True,
			},
		),
		subgoals=[{"id": "subgoal_1", "task_name": "stack", "args": ["b", "a"]}],
		subgoal_task_name_map={"stack": "stack"},
	)

	assert "query_step" not in asl
	assert "subgoal_cursor(1)." not in asl
	assert 'dfa_edge_label(dfa_t1, "subgoal_1").' in asl


def test_build_agentspeak_transition_specs_materialises_guard_groups() -> None:
	grounding_map = GroundingMap()
	grounding_map.add_atom("task_a", "stack", ["b", "a"])
	grounding_map.add_atom("task_b", "do_put_on", ["b", "a"])

	specs = build_agentspeak_transition_specs(
		dfa_result={
			"guarded_transitions": [
				{
					"source_state": "1",
					"target_state": "2",
					"guards": ["0X"],
					"raw_label": "guard_group_9",
				},
			],
			"free_variables": ["task_a", "task_b"],
			"alphabet": ["task_a", "task_b"],
			"initial_state": "1",
			"accepting_states": ["2"],
		},
		grounding_map=grounding_map,
	)

	assert [spec["task_event_symbol"] for spec in specs] == ["task_b", None]
	assert specs[0]["task_name"] == "do_put_on"
	assert specs[0]["task_args"] == ["b", "a"]
	assert specs[1]["is_epsilon_transition"] is True
	assert all(spec["raw_label"] == "guard_group_9" for spec in specs)


def test_build_agentspeak_transition_specs_supports_boolean_raw_labels() -> None:
	grounding_map = GroundingMap()
	grounding_map.add_atom("task_a", "stack", ["b", "a"])
	grounding_map.add_atom("task_b", "do_put_on", ["b", "a"])

	specs = build_agentspeak_transition_specs(
		dfa_result={
			"dfa_dot": (
				"digraph MONA_DFA {\n"
				" node [shape = doublecircle]; 2;\n"
				" node [shape = circle]; 1;\n"
				' init [shape = plaintext, label = ""];\n'
				" init -> 1;\n"
				' 1 -> 2 [label="(~task_a & task_b)"];\n'
				"}"
			),
			"alphabet": ["task_a", "task_b"],
		},
		grounding_map=grounding_map,
	)

	assert len(specs) == 1
	assert specs[0]["task_event_symbol"] == "task_b"
	assert specs[0]["task_name"] == "do_put_on"
	assert specs[0]["task_args"] == ["b", "a"]


def test_ltlf_to_dfa_invokes_mona_with_session_and_memory_limit(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	converter = ltlf_to_dfa_module.LTLfToDFA()
	monkeypatch.setenv("ONLINE_MONA_MEMORY_LIMIT_MIB", "4321")
	monkeypatch.setattr(converter, "_render_mona_program", lambda _formula: "ws1s;")
	monkeypatch.setattr(converter, "_resolve_mona_runtime", lambda: ("mona", {}))

	captured: dict[str, object] = {}

	class FakeProcess:
		def __init__(self, *args, **kwargs) -> None:
			captured["args"] = args
			captured["kwargs"] = kwargs
			self.pid = 1234
			self.returncode = 0
			stdout_handle = kwargs["stdout"]
			stdout_handle.write("digraph G {}")
			stdout_handle.flush()

		def communicate(self, timeout=None):
			captured["timeout"] = timeout
			return ("", "")

	monkeypatch.setattr(ltlf_to_dfa_module.subprocess, "Popen", FakeProcess)

	result = converter._invoke_mona_directly(object())

	assert result == "digraph G {}"
	assert captured["timeout"] == converter.MONA_TIMEOUT_SECONDS
	assert captured["kwargs"]["start_new_session"] is True
	assert callable(captured["kwargs"]["preexec_fn"])


def test_ltlf_to_dfa_kills_mona_process_group_on_timeout(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	converter = ltlf_to_dfa_module.LTLfToDFA()
	monkeypatch.setattr(converter, "_render_mona_program", lambda _formula: "ws1s;")
	monkeypatch.setattr(converter, "_resolve_mona_runtime", lambda: ("mona", {}))

	killed: dict[str, object] = {}

	class FakeProcess:
		def __init__(self, *args, **kwargs) -> None:
			self.pid = 4321
			self.returncode = None

		def communicate(self, timeout=None):
			raise ltlf_to_dfa_module.subprocess.TimeoutExpired("mona", timeout)

		def poll(self):
			return None

		def kill(self) -> None:
			killed["kill_called"] = True

		def wait(self, timeout=None) -> None:
			killed["wait_timeout"] = timeout

	monkeypatch.setattr(ltlf_to_dfa_module.subprocess, "Popen", FakeProcess)
	monkeypatch.setattr(
		ltlf_to_dfa_module.os,
		"killpg",
		lambda pid, sig: killed.update({"pid": pid, "signal": sig}),
	)

	result = converter._invoke_mona_directly(object())

	assert result is False
	assert killed["pid"] == 4321
	assert killed["signal"] == ltlf_to_dfa_module.signal.SIGKILL


def test_online_orchestrator_prefers_original_problem_for_verification() -> None:
	goal_free_problem = PROJECT_ROOT / "tests" / "generated" / "goal_free_p01.hddl"
	goal_free_problem.write_text(
		(
			PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"
		).read_text().replace(
			"\t(:goal (and\n(on b1 b4)\n(on b3 b1)\n\t))\n",
			"\t(:goal (and))\n",
		),
	)
	orchestrator = OnlineQuerySolutionOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(goal_free_problem),
	)
	problem_file, mode = orchestrator._determine_verification_problem()

	assert mode == "original_problem"
	assert problem_file == str(goal_free_problem.resolve())


def test_online_orchestrator_uses_original_problem_when_problem_has_goal_facts() -> None:
	orchestrator = OnlineQuerySolutionOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p02.hddl"),
	)
	problem_file, mode = orchestrator._determine_verification_problem()

	assert mode == "original_problem"
	assert Path(problem_file).resolve() == (
		PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p02.hddl"
	).resolve()


def test_online_domain_source_defaults_to_benchmark_domain() -> None:
	orchestrator = OnlineQuerySolutionOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)

	context = orchestrator._resolve_online_domain_context()

	assert context.source == "benchmark"
	assert context.domain_file == str(
		(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl").resolve()
	)
	assert context.domain.name == orchestrator.domain.name


def test_benchmark_online_path_uses_benchmark_domain_for_verification() -> None:
	orchestrator = OnlineQuerySolutionOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	online_domain = orchestrator._resolve_online_domain_context()

	verification_domain_file, domain_build_seconds = resolve_verification_domain_file(
		method_library=_sample_method_library(),
		online_domain=online_domain,
		output_dir=str(PROJECT_ROOT / "tests" / "generated"),
	)

	assert verification_domain_file == Path(online_domain.domain_file).resolve()
	assert domain_build_seconds == 0.0


def test_online_domain_source_can_switch_to_generated_domain(tmp_path: Path) -> None:
	generated_domain_path = tmp_path / "generated_domain.hddl"
	generated_domain_path.write_text(
		(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl").read_text()
	)
	orchestrator = OnlineQuerySolutionOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
		online_domain_source="generated",
	)
	artifact = DomainLibraryArtifact(
		domain_name="blocks",
		method_library=_sample_method_library(),
		method_synthesis_metadata={},
		domain_gate={},
		source_domain_kind="masked_official",
		artifact_root=str(tmp_path),
		generated_domain_file=str(generated_domain_path),
	)

	context = orchestrator._resolve_online_domain_context(artifact)

	assert context.source == "generated"
	assert context.domain_file == str(generated_domain_path.resolve())
	assert context.domain.name == orchestrator.domain.name


def test_online_domain_source_materializes_generated_domain_for_legacy_artifact(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / "legacy_artifact"
	artifact_root.mkdir()
	orchestrator = OnlineQuerySolutionOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
		online_domain_source="generated",
	)
	artifact = DomainLibraryArtifact(
		domain_name="blocks",
		method_library=_sample_method_library(),
		method_synthesis_metadata={},
		domain_gate={},
		source_domain_kind="masked_official",
		artifact_root=str(artifact_root),
	)

	context = orchestrator._resolve_online_domain_context(artifact)

	assert context.source == "generated"
	assert Path(context.domain_file).exists()
	assert Path(artifact_root / "masked_domain.hddl").exists()
	assert Path(artifact_root / "generated_domain.hddl").exists()


def test_online_orchestrator_uses_fixed_jason_runtime_timeout_budget() -> None:
	assert OnlineQuerySolutionOrchestrator._jason_runtime_timeout_seconds(subgoal_count=10) == 1800
	assert OnlineQuerySolutionOrchestrator._jason_runtime_timeout_seconds(subgoal_count=450) == 1800
	assert OnlineQuerySolutionOrchestrator._jason_runtime_timeout_seconds(subgoal_count=1000) == 1800


def test_jason_runner_execute_entry_perceives_initial_world() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */
dfa_state(q1).
accepting_state(q2).

/* Primitive Action Plans */

/* HTN Method Plans */

/* DFA Transition Wrappers */
+!dfa_t1 : dfa_state(q1) <-
	-dfa_state(q1);
	+dfa_state(q2).

/* DFA Control Plans */
+!run_dfa : dfa_state(q2) & accepting_state(q2) <-
	true.

+!run_dfa : dfa_state(q1) <-
	!dfa_t1;
	!run_dfa.
""".strip(),
		method_library=_sample_method_library(),
		seed_facts=("(clear b1)", "(handempty)"),
		runtime_objects=("b1",),
		object_types={"b1": "block"},
		type_parent_map={"block": "object", "object": None},
	)

	assert '.print("execute start")' in runtime_program
	assert ".perceive" in runtime_program
	assert "clear(b1)." in runtime_program
	assert runtime_program.index('.print("execute start")') < runtime_program.index(".perceive")


def test_jason_runner_accepting_state_uses_raw_dfa_execution_entry() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */
dfa_state(q0).
accepting_state(q0).
dfa_edge_label(dfa_t1, "subgoal_1").
dfa_edge_label(dfa_t2, "subgoal_2").

/* Primitive Action Plans */

/* HTN Method Plans */
+!task_a : true <-
	true.

+!task_b : true <-
	true.

/* DFA Transition Wrappers */
+!dfa_t1 : true <-
	!task_a.

+!dfa_t2 : true <-
	!task_b.

/* DFA Control Plans */
+!run_dfa : dfa_state(q0) & accepting_state(q0) <-
	true.
""".strip(),
		method_library=_sample_method_library(),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
	)

	execute_section = runtime_program.split("+!execute : true <-", maxsplit=1)[1]

	assert "?accepting_state(FINAL_STATE)" in runtime_program
	assert "!run_dfa" in execute_section


def test_jason_runner_accepting_state_execution_is_direct() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */
dfa_state(q0).
accepting_state(q0).

/* Primitive Action Plans */

/* HTN Method Plans */

/* DFA Transition Wrappers */

/* DFA Control Plans */
+!run_dfa : true <-
	true.
""".strip(),
		method_library=_sample_method_library(),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
	)

	execute_section = runtime_program.split("+!execute : true <-", maxsplit=1)[1]

	assert "!run_dfa" in execute_section
	assert '.print("execute success")' in execute_section
	assert execute_section.count("!run_dfa") == 1


def test_jason_runner_validate_passes_raw_agentspeak_program_to_runtime_builder(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	runner = JasonRunner(runtime_dir=tmp_path)
	captured: dict[str, str] = {}
	jason_jar = tmp_path / "jason.jar"
	log_conf = tmp_path / "logging.properties"
	jason_jar.write_text("")
	log_conf.write_text("")

	def fake_build_runner_asl(agentspeak_code: str, *args, **kwargs) -> str:
		del args, kwargs
		captured["agentspeak_code"] = agentspeak_code
		return """
/* Initial Beliefs */
dfa_state(q0).
accepting_state(q0).

/* Primitive Action Plans */

/* HTN Method Plans */

/* DFA Transition Wrappers */

/* DFA Control Plans */
+!run_dfa : true <-
	true.
""".strip()

	def fake_compile_environment_java(**kwargs) -> None:
		env_java_path = Path(str(kwargs["env_java_path"]))
		output_path = Path(str(kwargs["output_path"]))
		env_java_path.write_text("class JasonPipelineEnvironment {}")
		(output_path / f"{runner.environment_class_name}.class").write_text("")

	class FakeCompletedProcess:
		returncode = 0
		stdout = "runtime env ready\nexecute success\n"
		stderr = ""

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("java", 17))
	monkeypatch.setattr(runner, "_select_javac_binary", lambda java_bin: "javac")
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jason_jar)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(runner, "_build_runner_asl", fake_build_runner_asl)
	monkeypatch.setattr(runner, "_compile_environment_java", fake_compile_environment_java)
	monkeypatch.setattr(runner.environment_adapter, "validate", lambda *, stdout, stderr: EnvironmentAdapterResult(
		success=True,
		adapter_name="fake",
		mode="test",
		details={},
	))
	monkeypatch.setattr(runner, "_extract_action_path", lambda stdout: [])
	monkeypatch.setattr(runner, "_extract_method_trace", lambda output: [])
	monkeypatch.setattr(runner, "_run_consistency_checks", lambda **kwargs: {})
	monkeypatch.setattr("online_query_solution.jason_runtime.runner.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

	runner.validate(
		agentspeak_code="""
/* Initial Beliefs */
dfa_state(q0).
accepting_state(q0).

/* Primitive Action Plans */

/* HTN Method Plans */
+!navigate_abs(rover1, waypoint5) : true <-
	.print("runtime trace method flat ", "m-navigate_abs-1");
	true.

+!navigate_abs(rover1, waypoint5) : true <-
	.print("runtime trace method flat ", "m-navigate_abs-3");
	true.

/* DFA Transition Wrappers */

/* DFA Control Plans */
+!run_dfa : true <-
	true.
""".strip(),
		method_library=_sample_method_library(),
		action_schemas=[{"name": "noop"}],
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		domain_name="blocks",
		problem_file=None,
		output_dir=tmp_path,
	)

	assert captured["agentspeak_code"].index("m-navigate_abs-1") < captured["agentspeak_code"].index(
		"m-navigate_abs-3"
	)


def test_jason_runner_validate_downgrades_consistency_check_failures_to_diagnostics(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	runner = JasonRunner(runtime_dir=tmp_path)
	jason_jar = tmp_path / "jason.jar"
	log_conf = tmp_path / "logging.properties"
	jason_jar.write_text("")
	log_conf.write_text("")

	def fake_compile_environment_java(**kwargs) -> None:
		env_java_path = Path(str(kwargs["env_java_path"]))
		output_path = Path(str(kwargs["output_path"]))
		env_java_path.write_text("class JasonPipelineEnvironment {}")
		(output_path / f"{runner.environment_class_name}.class").write_text("")

	class FakeCompletedProcess:
		returncode = 0
		stdout = "runtime env ready\nexecute success\n"
		stderr = ""

	monkeypatch.setattr(runner, "_select_java_binary", lambda: ("java", 17))
	monkeypatch.setattr(runner, "_select_javac_binary", lambda java_bin: "javac")
	monkeypatch.setattr(runner, "_ensure_jason_jar", lambda java_bin: jason_jar)
	monkeypatch.setattr(runner, "_resolve_log_config", lambda: log_conf)
	monkeypatch.setattr(
		runner,
		"_compile_environment_java",
		fake_compile_environment_java,
	)
	monkeypatch.setattr(
		runner.environment_adapter,
		"validate",
		lambda *, stdout, stderr: EnvironmentAdapterResult(
			success=True,
			adapter_name="fake",
			mode="test",
			details={},
		),
	)
	monkeypatch.setattr(runner, "_extract_action_path", lambda stdout: [])
	monkeypatch.setattr(runner, "_extract_method_trace", lambda output: [])
	monkeypatch.setattr(
		runner,
		"_run_consistency_checks",
		lambda **kwargs: (_ for _ in ()).throw(RuntimeError("diagnostic boom")),
	)
	monkeypatch.setattr(
		"online_query_solution.jason_runtime.runner.subprocess.run",
		lambda *args, **kwargs: FakeCompletedProcess(),
	)

	result = runner.validate(
		agentspeak_code="""
/* Initial Beliefs */
dfa_state(q0).
accepting_state(q0).

/* Primitive Action Plans */

/* HTN Method Plans */

/* DFA Transition Wrappers */

/* DFA Control Plans */
+!run_dfa : true <-
	true.
""".strip(),
		method_library=_sample_method_library(),
		action_schemas=[{"name": "noop"}],
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		domain_name="blocks",
		problem_file=None,
		output_dir=tmp_path,
	)

	assert result.status == "success"
	assert result.consistency_checks["diagnostics_only"] is True
	assert result.consistency_checks["failure_class"] == "consistency_diagnostics_exception"
	assert result.consistency_checks["message"] == "diagnostic boom"


def test_goal_grounding_raises_immediately_after_invalid_json_without_retry() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)
	generator.client = object()

	seen_messages: list[list[dict[str, str]]] = []

	def fake_create(self, messages, **_kwargs):
		seen_messages.append(list(messages))
		return "{\n  \"ltlf_formula\": \"unterminated"

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(NLToLTLfGenerator, "_create_chat_completion", fake_create)
	monkeypatch.setattr(NLToLTLfGenerator, "_extract_response_text", lambda self, response: str(response))
	try:
		with pytest.raises(Exception):
			generator.generate(
				"First put block b4 on block b2, then put block b1 on block b4.",
				method_library=_sample_method_library(),
				typed_objects={"b1": "block", "b2": "block", "b4": "block"},
				task_type_map={"do_put_on": ("block", "block")},
				type_parent_map={"block": "object", "object": None},
			)
	finally:
		monkeypatch.undo()

	assert len(seen_messages) == 1


def test_goal_grounding_raises_immediately_after_validation_error_without_retry() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)
	generator.client = object()

	seen_messages: list[list[dict[str, str]]] = []

	def fake_create(self, messages, **_kwargs):
		seen_messages.append(list(messages))
		return '{"ltlf_formula":"F(do_put_on(b4, b2))","diagnostics":[]}'

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(NLToLTLfGenerator, "_create_chat_completion", fake_create)
	monkeypatch.setattr(NLToLTLfGenerator, "_extract_response_text", lambda self, response: str(response))
	try:
		with pytest.raises(ValueError, match="only the key ltlf_formula"):
			generator.generate(
				"First put block b4 on block b2, then put block b1 on block b4.",
				method_library=_sample_method_library(),
				typed_objects={"b1": "block", "b2": "block", "b4": "block"},
				task_type_map={"do_put_on": ("block", "block")},
				type_parent_map={"block": "object", "object": None},
			)
	finally:
		monkeypatch.undo()

	assert len(seen_messages) == 1


def test_goal_grounding_prompts_treat_complete_the_tasks_lists_as_ordered_by_default() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)

	system_prompt, _ = generator._build_prompts(
		query_text="complete the tasks do_a(x), do_b(y)",
		method_library=_sample_method_library(),
		typed_objects={"a": "block", "b": "block"},
		task_type_map={"do_put_on": ("block", "block")},
	)

	assert "ordered by default" in system_prompt
	assert "Few-shot examples:" in system_prompt
	assert "Supported unary operators: F, G, X, WX" in system_prompt
	assert "Supported binary operators: U, R" in system_prompt
	assert "last" in system_prompt
	assert "Do not use unsupported past-time operators" in system_prompt
	assert "Avoid deeply nested eventuality chains" in system_prompt
	assert "F(A & F(B & F(C)))" in system_prompt
	assert "(!B U (A & !B))" in system_prompt
	assert "X F(do_put_on(b1, b4))" not in system_prompt
	assert "F(do_put_on(b4, b2) & F(do_put_on(b1, b4)" not in system_prompt


def test_goal_grounding_prompt_attempts_keep_single_strict_mode_for_huge_queries() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)

	huge_query = "Using blocks " + ", ".join(f"b{i}" for i in range(1, 4000))
	attempts = generator._build_prompt_attempts(
		query_text=huge_query,
		method_library=_sample_method_library(),
		typed_objects={f"b{i}": "block" for i in range(1, 20)},
		task_type_map={"do_put_on": ("block", "block")},
	)

	assert [attempt["mode"] for attempt in attempts] == ["few_shot_strict"]
	assert "Available grounded problem objects:" in attempts[0]["system"]
	assert attempts[0]["request_timeout"] >= 120.0


def test_goal_grounding_response_budget_scales_with_explicit_task_list_length() -> None:
	query_text = (
		"Using blocks b1, b2, b3, b4, b5, b6, b7, and b8, complete the tasks "
		"do_put_on(b1, b2), then do_put_on(b2, b3), then do_put_on(b3, b4), "
		"then do_put_on(b4, b5), then do_put_on(b5, b6), then do_put_on(b6, b7), "
		"then do_put_on(b7, b8), then do_put_on(b8, b1), then do_put_on(b1, b2), "
		"then do_put_on(b2, b3), then do_put_on(b3, b4), then do_put_on(b4, b5)."
	)

	assert NLToLTLfGenerator._suggest_response_max_tokens(query_text) == 32000


def test_goal_grounding_request_timeout_scales_for_long_explicit_task_sequences() -> None:
	query_text = (
		"Using blocks b1, b5, b7, b13, b11, b3, b12, b2, b4, b20, b21, b10, b16, "
		"b8, b9, b19, b18, b15, and b17, complete the tasks "
		"do_put_on(b1, b5), then do_put_on(b7, b1), then do_put_on(b13, b7), "
		"then do_put_on(b1, b5), then do_put_on(b7, b1), then do_put_on(b13, b7), "
		"then do_put_on(b11, b13), then do_put_on(b3, b11), then do_put_on(b12, b2), "
		"then do_put_on(b4, b12), then do_put_on(b20, b4), then do_put_on(b21, b20), "
		"then do_put_on(b10, b16), then do_put_on(b8, b10), then do_put_on(b9, b8), "
		"then do_put_on(b19, b9), then do_put_on(b18, b15), then do_put_on(b17, b18)."
	)

	assert NLToLTLfGenerator._suggest_request_timeout(query_text) == 240.0


def test_goal_grounding_minimax_request_profile_uses_dynamic_reasoning_budget() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(
		domain_file=domain_file,
		model="minimax/minimax-m2.7",
		base_url="https://openrouter.ai/api/v1",
	)
	messages = [
		{"role": "system", "content": "Return strict minified JSON only."},
		{"role": "user", "content": "Ground this query into one LTLf formula."},
	]

	profile = generator._goal_grounding_request_profile(messages=messages)

	expected_prompt_estimate = generator._estimate_goal_grounding_prompt_token_budget(messages)
	expected_margin = math.ceil(196_608 * 0.10)
	expected_headroom = max(
		196_608 - expected_margin - expected_prompt_estimate - 5_805 - 2_048,
		0,
	)
	expected_reasoning_budget = int(expected_headroom * 0.70)

	assert profile["name"] == "minimax_stream_single_pass"
	assert profile["stream_response"] is True
	assert profile["first_chunk_timeout_seconds"] == 300.0
	assert profile["context_window_tokens"] == 196_608
	assert profile["prompt_token_estimate"] == expected_prompt_estimate
	assert profile["answer_token_reserve"] == 5_805
	assert profile["context_margin_tokens"] == expected_margin
	assert profile["transport_overhead_tokens"] == 2_048
	assert profile["reasoning_headroom_tokens"] == expected_headroom
	assert profile["reasoning_headroom_ratio"] == 0.70
	assert profile["reasoning_max_tokens"] == expected_reasoning_budget
	assert profile["session_id"] == "online-ltlf-generation"


def test_goal_grounding_chat_completion_uses_openrouter_streaming_without_completion_cap() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	captured_request: dict[str, object] = {}

	class FakeGenerator(NLToLTLfGenerator):
		def _create_raw_openrouter_stream_response(
			self,
			request_kwargs,
			*,
			request_timeout_seconds=None,
		):
			captured_request["request_kwargs"] = dict(request_kwargs)
			captured_request["request_timeout_seconds"] = request_timeout_seconds
			return {"ok": True}

	generator = FakeGenerator(
		domain_file=domain_file,
		model="minimax/minimax-m2.7",
		base_url="https://openrouter.ai/api/v1",
		api_key="sk-test",
	)

	response = generator._create_chat_completion(
		[{"role": "system", "content": "Return JSON."}],
		response_max_tokens=321,
		request_timeout=123.0,
	)

	assert response == {"ok": True}
	request_kwargs = captured_request["request_kwargs"]
	assert captured_request["request_timeout_seconds"] == 123.0
	assert request_kwargs["timeout"] == 123.0
	assert request_kwargs["stream"] is True
	assert "max_tokens" not in request_kwargs
	assert "response_format" not in request_kwargs
	expected_profile = generator._goal_grounding_request_profile(
		messages=[{"role": "system", "content": "Return JSON."}],
	)
	assert request_kwargs["extra_body"] == {
		"provider": {"only": ["minimax"], "allow_fallbacks": False},
		"session_id": "online-ltlf-generation",
		"reasoning": {
			"max_tokens": expected_profile["reasoning_max_tokens"],
			"exclude": True,
		},
	}


def test_goal_grounding_chat_completion_keeps_json_object_request_off_openrouter_streaming_path() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(
		domain_file=domain_file,
		model="other/model",
		base_url="https://api.example.com/v1",
	)

	class FakeCompletions:
		def __init__(self) -> None:
			self.calls: list[dict[str, object]] = []

		def create(self, **kwargs):
			self.calls.append(dict(kwargs))
			return {"ok": True}

	fake_completions = FakeCompletions()

	class FakeChat:
		def __init__(self, completions) -> None:
			self.completions = completions

	class FakeClient:
		def __init__(self, completions) -> None:
			self.chat = FakeChat(completions)

	generator.client = FakeClient(fake_completions)

	response = generator._create_chat_completion(
		[{"role": "system", "content": "Return JSON."}],
		response_max_tokens=321,
		request_timeout=123.0,
	)

	assert response == {"ok": True}
	assert len(fake_completions.calls) == 1
	assert fake_completions.calls[0]["timeout"] == 123.0
	assert fake_completions.calls[0]["max_tokens"] == 321
	assert fake_completions.calls[0]["stream"] is False
	assert fake_completions.calls[0]["response_format"] == {"type": "json_object"}


def test_goal_grounding_streaming_rejects_length_finish_even_when_json_is_complete() -> None:
	class FakeDelta:
		def __init__(self, content):
			self.content = content

	class FakeChoice:
		def __init__(self, content, finish_reason=None):
			self.delta = FakeDelta(content)
			self.finish_reason = finish_reason

	class FakeChunk:
		def __init__(self, chunk_id, content, finish_reason=None):
			self.id = chunk_id
			self.choices = [FakeChoice(content, finish_reason=finish_reason)]

	class FakeStream:
		def __iter__(self):
			yield FakeChunk("req_goal_123", '{"ltlf_formula":"F(do_put_on(b4, b2))')
			yield FakeChunk("req_goal_123", '"}', finish_reason="length")

		def close(self):
			return None

	generator = NLToLTLfGenerator()

	with pytest.raises(RuntimeError, match="finish_reason=length") as exc_info:
		generator._consume_streaming_llm_response(
			FakeStream(),
			total_timeout_seconds=10.0,
		)

	transport_metadata = getattr(exc_info.value, "transport_metadata", {})
	assert transport_metadata["llm_request_id"] == "req_goal_123"
	assert transport_metadata["llm_response_mode"] == "streaming"
	assert transport_metadata["llm_first_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_complete_json_seconds"] >= 0.0
	assert transport_metadata["llm_finish_reason"] == "length"


def test_goal_grounding_streaming_enforces_first_chunk_deadline() -> None:
	class BlockingStream:
		def __iter__(self):
			return self

		def __next__(self):
			time.sleep(0.05)
			raise AssertionError("stream iteration should have timed out before yielding")

		def close(self):
			return None

	generator = NLToLTLfGenerator()

	with pytest.raises(TimeoutError, match="first-chunk deadline") as exc_info:
		generator._consume_streaming_llm_response(
			BlockingStream(),
			transport_metadata={"llm_first_chunk_timeout_seconds": 0.01},
			total_timeout_seconds=0.1,
		)

	transport_metadata = getattr(exc_info.value, "transport_metadata", {})
	assert transport_metadata["llm_response_mode"] == "streaming"
	assert transport_metadata["llm_first_chunk_timeout_seconds"] == 0.01
	assert transport_metadata.get("llm_first_chunk_seconds") is None


def test_execute_query_with_jason_returns_none_when_runtime_fails(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	orchestrator = OnlineQuerySolutionOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	orchestrator.output_dir = tmp_path
	online_domain = orchestrator._resolve_online_domain_context()
	grounding_result = TemporalGroundingResult(
		query_text="stack b on a",
		ltlf_formula="F(subgoal_1)",
		subgoals=(GroundedSubgoal("subgoal_1", "do_put_on", ("b", "a")),),
		typed_objects={"a": "block", "b": "block"},
		query_object_inventory=(),
		diagnostics=(),
	)
	dfa_result = DFACompilationResult(
		query_text="stack b on a",
		ltlf_formula="F(subgoal_1)",
		alphabet=("subgoal_1",),
		transition_specs=(),
		ordered_subgoal_sequence=True,
		subgoals=(GroundedSubgoal("subgoal_1", "do_put_on", ("b", "a")),),
	)
	validation_calls: list[dict[str, object]] = []

	class FakeRunner:
		def validate(self, **kwargs):
			validation_calls.append(dict(kwargs))
			return JasonValidationResult(
				status="failed",
				backend="RunLocalMAS",
				java_path=None,
				java_version=None,
				javac_path=None,
				jason_jar=None,
				exit_code=0,
				timed_out=False,
				stdout="execute failed",
				stderr="runtime failed",
				action_path=[],
				method_trace=[],
				failed_goals=["goal"],
				environment_adapter={},
				failure_class="runtime_failure",
				consistency_checks={},
				artifacts={},
				timing_profile={},
			)

	monkeypatch.setattr(online_orchestrator_module, "JasonRunner", lambda **_kwargs: FakeRunner())
	monkeypatch.setattr(
		online_orchestrator_module,
		"planner_action_schemas_for_domain",
		lambda _domain: [{"action_name": "turn_to"}],
	)
	monkeypatch.setattr(
		online_orchestrator_module,
		"render_supported_hierarchical_plan",
		lambda **_kwargs: "guided plan",
	)

	result = orchestrator._execute_query_with_jason(
		grounding_result=grounding_result,
		dfa_result=dfa_result,
		method_library=_sample_method_library(),
		agentspeak_code="!execute.",
		agentspeak_artifacts={},
		verification_problem_file=str(
			PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"
		),
		verification_mode="original_problem",
		online_domain=online_domain,
	)

	assert result is None
	assert len(validation_calls) == 1


def test_execute_query_with_jason_uses_reconstructed_hierarchical_plan_text(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	orchestrator = OnlineQuerySolutionOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	orchestrator.output_dir = tmp_path
	online_domain = orchestrator._resolve_online_domain_context()
	grounding_result = TemporalGroundingResult(
		query_text="stack b on a",
		ltlf_formula="F(subgoal_1)",
		subgoals=(GroundedSubgoal("subgoal_1", "do_put_on", ("b", "a")),),
		typed_objects={"a": "block", "b": "block"},
		query_object_inventory=(),
		diagnostics=(),
	)
	dfa_result = DFACompilationResult(
		query_text="stack b on a",
		ltlf_formula="F(subgoal_1)",
		alphabet=("subgoal_1",),
		transition_specs=(),
		ordered_subgoal_sequence=True,
		subgoals=(GroundedSubgoal("subgoal_1", "do_put_on", ("b", "a")),),
	)

	class FakeRunner:
		def validate(self, **_kwargs):
			return JasonValidationResult(
				status="success",
				backend="RunLocalMAS",
				java_path=None,
				java_version=None,
				javac_path=None,
				jason_jar=None,
				exit_code=0,
				timed_out=False,
				stdout="execute success",
				stderr="",
				action_path=["turn_to(a,b,c)"],
				method_trace=[{"method_name": "runtime_method", "task_args": ["a", "b"]}],
				failed_goals=[],
				environment_adapter={},
				failure_class=None,
				consistency_checks={},
				artifacts={},
				timing_profile={},
			)

	monkeypatch.setattr(online_orchestrator_module, "JasonRunner", lambda **_kwargs: FakeRunner())
	monkeypatch.setattr(
		online_orchestrator_module,
		"planner_action_schemas_for_domain",
		lambda _domain: [{"action_name": "turn_to"}],
	)
	monkeypatch.setattr(
		online_orchestrator_module,
		"render_supported_hierarchical_plan",
		lambda **_kwargs: "reconstructed plan",
	)

	result = orchestrator._execute_query_with_jason(
		grounding_result=grounding_result,
		dfa_result=dfa_result,
		method_library=_sample_method_library(),
		agentspeak_code="!execute.",
		agentspeak_artifacts={},
		verification_problem_file=str(
			PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"
		),
		verification_mode="original_problem",
		online_domain=online_domain,
	)

	assert result is not None
	assert result.hierarchical_plan_text == "reconstructed plan"


def test_verify_plan_officially_restores_terminal_newline_for_hierarchical_plan_text(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	orchestrator = OnlineQuerySolutionOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
		online_domain_source="benchmark",
	)
	orchestrator.output_dir = tmp_path
	online_domain = orchestrator._resolve_online_domain_context(source="benchmark")

	class FakeVerifier:
		def tool_available(self) -> bool:
			return True

		def verify_plan_text(self, **kwargs):
			assert str(kwargs["plan_text"]).endswith("\n")
			return IPCPrimitivePlanVerificationResult(
				tool_available=True,
				command=["fake"],
				plan_file=str(tmp_path / "plan.txt"),
				output_file=str(tmp_path / "verifier.txt"),
				stdout="Plan verification result: true",
				stderr="",
				primitive_plan_only=False,
				primitive_plan_executable=True,
				verification_result=True,
				reached_goal_state=True,
				plan_kind="hierarchical",
				build_warning=None,
				error=None,
			)

		def verify_plan(self, **kwargs):
			raise AssertionError(
				"verify_plan should not be called when hierarchical plan text is present"
			)

	monkeypatch.setattr(
		online_official_verification_module,
		"IPCPlanVerifier",
		lambda: FakeVerifier(),
	)

	plan_verification = orchestrator._verify_plan_officially(
		method_library=_sample_method_library(),
		plan_solve_data={
			"summary": {
				"backend": "jason",
				"status": "success",
			},
			"artifacts": {
				"planning_mode": "jason_runtime",
				"verification_problem_file": str(
					PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"
				),
				"verification_mode": "original_problem",
				"hierarchical_plan_text": "==>\nroot",
				"action_path": [],
				"method_trace": [],
			},
		},
		online_domain=online_domain,
	)

	assert plan_verification is not None
	assert plan_verification["summary"]["status"] == "success"
