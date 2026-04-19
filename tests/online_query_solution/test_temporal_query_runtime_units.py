from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from offline_method_generation.method_synthesis.schema import HTNMethodLibrary, HTNTask
from offline_method_generation.artifacts import DomainLibraryArtifact
from online_query_solution.artifacts import DFACompilationResult, GroundedSubgoal
from pipeline.domain_complete_pipeline import DomainCompletePipeline
from online_query_solution.agentspeak import AgentSpeakRenderer
from online_query_solution.goal_grounding.grounder import NLToLTLfGenerator
from online_query_solution.jason_runtime.environment_adapter import EnvironmentAdapterResult
from online_query_solution.jason_runtime.runner import JasonRunner, JasonValidationResult
from online_query_solution.temporal_compilation import dfa_builder
from pipeline import domain_complete_pipeline as domain_complete_pipeline_module
from utils.hddl_parser import HDDLParser
from verification.official_plan_verifier import IPCPrimitivePlanVerificationResult
from tests.support.ground_truth_baseline_support import DOMAIN_FILES, build_official_method_library


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


def test_goal_grounding_validator_rejects_repeated_raw_task_calls_without_event_identity() -> None:
	generator = NLToLTLfGenerator()

	with pytest.raises(ValueError, match="explicit grounded task-event identities"):
		generator._validate_payload(
			query_text="repeat stack",
			payload={
				"ltlf_formula": "F(stack(b, a) & F(stack(b, a)))",
			},
			method_library=_sample_method_library(),
			typed_objects={"a": "block", "b": "block"},
			task_type_map={"stack": ("block", "block")},
			type_parent_map={"block": "object", "object": None},
		)


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
	assert "F(do_put_on(b4, b2) & F(do_put_on(b1, b4) & F(do_put_on(b3, b1))))" in system_prompt
	assert "do_put_on__e1(b8, b9)" in system_prompt


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


def test_goal_grounding_generate_does_not_retry_after_invalid_response_extraction() -> None:
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
			RuntimeError("LLM response did not contain usable textual JSON content."),
		),
	)
	try:
		with pytest.raises(RuntimeError, match="usable textual JSON content"):
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


def test_pipeline_prefers_original_problem_when_grounded_subgoals_match_root() -> None:
	goal_free_problem = PROJECT_ROOT / "tests" / "generated" / "goal_free_p01.hddl"
	goal_free_problem.write_text(
		(
			PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"
		).read_text().replace(
			"\t(:goal (and\n(on b1 b4)\n(on b3 b1)\n\t))\n",
			"\t(:goal (and))\n",
		),
	)
	pipeline = DomainCompletePipeline(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(goal_free_problem),
	)
	method_library = _sample_method_library()
	online_domain = pipeline._resolve_online_domain_context()
	dfa_result = DFACompilationResult(
		query_text="q",
		ltlf_formula="F(subgoal_1) & F(subgoal_2) & F(subgoal_3)",
		alphabet=("subgoal_1", "subgoal_2", "subgoal_3"),
		transition_specs=(),
		ordered_subgoal_sequence=True,
		subgoals=(
			GroundedSubgoal("subgoal_1", "do_put_on", ("b4", "b2")),
			GroundedSubgoal("subgoal_2", "do_put_on", ("b1", "b4")),
			GroundedSubgoal("subgoal_3", "do_put_on", ("b3", "b1")),
		),
	)

	problem_file, mode = pipeline._determine_verification_problem(
		dfa_result=dfa_result,
		method_library=method_library,
		online_domain=online_domain,
	)

	assert mode == "original_problem"
	assert problem_file == str(goal_free_problem.resolve())


def test_pipeline_uses_original_problem_even_when_problem_has_goal_facts(
	tmp_path: Path,
) -> None:
	pipeline = DomainCompletePipeline(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p02.hddl"),
	)
	pipeline.output_dir = tmp_path
	method_library = _sample_method_library()
	online_domain = pipeline._resolve_online_domain_context()
	dfa_result = DFACompilationResult(
		query_text="q",
		ltlf_formula="F(subgoal_1 & F(subgoal_2 & F(subgoal_3 & F(subgoal_4 & F(subgoal_5 & F(subgoal_6))))))",
		alphabet=("subgoal_1", "subgoal_2", "subgoal_3", "subgoal_4", "subgoal_5", "subgoal_6"),
		transition_specs=(),
		ordered_subgoal_sequence=True,
		subgoals=(
			GroundedSubgoal("subgoal_1", "do_put_on", ("b3", "b5")),
			GroundedSubgoal("subgoal_2", "do_put_on", ("b6", "b3")),
			GroundedSubgoal("subgoal_3", "do_put_on", ("b1", "b6")),
			GroundedSubgoal("subgoal_4", "do_put_on", ("b2", "b1")),
			GroundedSubgoal("subgoal_5", "do_put_on", ("b4", "b2")),
			GroundedSubgoal("subgoal_6", "do_put_on", ("b7", "b4")),
		),
	)

	problem_file, mode = pipeline._determine_verification_problem(
		dfa_result=dfa_result,
		method_library=method_library,
		online_domain=online_domain,
	)

	assert mode == "original_problem"
	assert Path(problem_file).resolve() == (
		PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p02.hddl"
	).resolve()


def test_query_specific_verification_problem_uses_empty_goal_block(tmp_path: Path) -> None:
	pipeline = DomainCompletePipeline(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	pipeline.output_dir = tmp_path
	method_library = _sample_method_library()
	online_domain = pipeline._resolve_online_domain_context()
	dfa_result = DFACompilationResult(
		query_text="q",
		ltlf_formula="F(subgoal_1)",
		alphabet=("subgoal_1",),
		transition_specs=(),
		ordered_subgoal_sequence=True,
		subgoals=(
			GroundedSubgoal("subgoal_1", "do_on_table", ("b2",)),
		),
	)

	problem_path = pipeline._build_query_verification_problem(
		dfa_result,
		method_library,
		online_domain=online_domain,
	)
	problem_text = problem_path.read_text()

	assert "(:goal (and))" in problem_text
	assert "(on b4 b2)" not in problem_text


def test_online_domain_source_defaults_to_benchmark_domain() -> None:
	pipeline = DomainCompletePipeline(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)

	context = pipeline._resolve_online_domain_context()

	assert context.source == "benchmark"
	assert context.domain_file == str(
		(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl").resolve()
	)
	assert context.domain.name == pipeline.domain.name


def test_benchmark_online_path_uses_benchmark_domain_for_verification() -> None:
	pipeline = DomainCompletePipeline(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	online_domain = pipeline._resolve_online_domain_context()

	verification_domain_file, domain_build_seconds = pipeline._resolve_verification_domain_file(
		method_library=_sample_method_library(),
		online_domain=online_domain,
	)

	assert verification_domain_file == Path(online_domain.domain_file).resolve()
	assert domain_build_seconds == 0.0


def test_online_domain_source_can_switch_to_generated_domain(tmp_path: Path) -> None:
	generated_domain_path = tmp_path / "generated_domain.hddl"
	generated_domain_path.write_text(
		(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl").read_text()
	)
	pipeline = DomainCompletePipeline(
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

	context = pipeline._resolve_online_domain_context(artifact)

	assert context.source == "generated"
	assert context.domain_file == str(generated_domain_path.resolve())
	assert context.domain.name == pipeline.domain.name


def test_online_domain_source_materializes_generated_domain_for_legacy_artifact(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / "legacy_artifact"
	artifact_root.mkdir()
	pipeline = DomainCompletePipeline(
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

	context = pipeline._resolve_online_domain_context(artifact)

	assert context.source == "generated"
	assert Path(context.domain_file).exists()
	assert Path(artifact_root / "masked_domain.hddl").exists()
	assert Path(artifact_root / "generated_domain.hddl").exists()


def test_pipeline_detects_total_ordered_subgoal_chain() -> None:
	transition_specs = (
		{
			"source_state": "q1",
			"target_state": "q2",
			"subgoal_index": 1,
			"initial_state": "q1",
			"accepting_states": ("q4",),
		},
		{
			"source_state": "q2",
			"target_state": "q3",
			"subgoal_index": 2,
			"initial_state": "q1",
			"accepting_states": ("q4",),
		},
		{
			"source_state": "q3",
			"target_state": "q4",
			"subgoal_index": 3,
			"initial_state": "q1",
			"accepting_states": ("q4",),
		},
	)

	assert DomainCompletePipeline._is_total_ordered_subgoal_chain(
		transition_specs,
		expected_subgoal_count=3,
	)


def test_pipeline_rejects_unordered_subgoal_chain_detection() -> None:
	transition_specs = (
		{
			"source_state": "q1",
			"target_state": "q2",
			"subgoal_index": 1,
			"initial_state": "q1",
			"accepting_states": ("q4",),
		},
		{
			"source_state": "q1",
			"target_state": "q3",
			"subgoal_index": 2,
			"initial_state": "q1",
			"accepting_states": ("q4",),
		},
		{
			"source_state": "q2",
			"target_state": "q4",
			"subgoal_index": 2,
			"initial_state": "q1",
			"accepting_states": ("q4",),
		},
		{
			"source_state": "q3",
			"target_state": "q4",
			"subgoal_index": 1,
			"initial_state": "q1",
			"accepting_states": ("q4",),
		},
	)

	assert not DomainCompletePipeline._is_total_ordered_subgoal_chain(
		transition_specs,
		expected_subgoal_count=2,
	)


def test_pipeline_scales_jason_runtime_timeout_with_subgoal_count() -> None:
	assert DomainCompletePipeline._jason_runtime_timeout_seconds(subgoal_count=10) == 120
	assert DomainCompletePipeline._jason_runtime_timeout_seconds(subgoal_count=450) == 180
	assert DomainCompletePipeline._jason_runtime_timeout_seconds(subgoal_count=650) == 240
	assert DomainCompletePipeline._jason_runtime_timeout_seconds(subgoal_count=850) == 360
	assert DomainCompletePipeline._jason_runtime_timeout_seconds(subgoal_count=1000) == 480


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
		target_literals=(),
		method_library=_sample_method_library(),
		seed_facts=("(clear b1)", "(handempty)"),
		runtime_objects=("b1",),
		object_types={"b1": "block"},
		type_parent_map={"block": "object", "object": None},
		completion_mode="accepting_state",
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
		target_literals=(),
		method_library=_sample_method_library(),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		completion_mode="accepting_state",
	)

	execute_section = runtime_program.split("+!execute : true <-", maxsplit=1)[1]

	assert "!verify_targets" not in execute_section
	assert "?accepting_state(FINAL_STATE)" in runtime_program
	assert "!run_dfa" in execute_section


def test_jason_runner_accepting_state_execution_does_not_use_guided_replay() -> None:
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
		target_literals=(),
		method_library=_sample_method_library(),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		completion_mode="accepting_state",
	)

	execute_section = runtime_program.split("+!execute : true <-", maxsplit=1)[1]

	assert "!run_dfa" in execute_section
	assert '.print("execute success")' in execute_section
	assert "!guided_replay_1" not in execute_section


def test_jason_runner_guided_replay_quotes_uppercase_constants() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_guided_runner_asl(
		"""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */

/* DFA Transition Wrappers */
""".strip(),
		method_library=_sample_method_library(),
		ordered_query_sequence=False,
		guided_action_path=("turn_to(satellite1, GroundStation1, Phenomenon7)",),
		guided_method_trace=(),
	)

	assert '!turn_to(satellite1, "GroundStation1", "Phenomenon7")' in runtime_program


def test_pipeline_infers_unordered_execution_guidance_from_problem_root_planner(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	pipeline = DomainCompletePipeline(
		domain_file=DOMAIN_FILES["marsrover"],
		problem_file=str(
			PROJECT_ROOT / "src" / "domains" / "marsrover" / "problems" / "pfile06.hddl"
		),
		online_domain_source="benchmark",
	)
	pipeline.output_dir = tmp_path
	method_library = build_official_method_library(DOMAIN_FILES["marsrover"])
	online_domain = pipeline._resolve_online_domain_context(source="benchmark")
	subgoals = tuple(
		GroundedSubgoal(
			f"subgoal_{index}",
			str(task.task_name),
			tuple(str(arg) for arg in tuple(task.args or ())),
		)
		for index, task in enumerate(tuple(pipeline.problem.htn_tasks or ()), start=1)
	)
	dfa_result = DFACompilationResult(
		query_text="unordered rover query",
		ltlf_formula=" & ".join(
			f"F(subgoal_{index})"
			for index in range(1, len(subgoals) + 1)
		),
		alphabet=tuple(subgoal.subgoal_id for subgoal in subgoals),
		transition_specs=(),
		ordered_subgoal_sequence=False,
		subgoals=subgoals,
	)

	class FakePlan:
		actual_plan = "fake"
		action_path = ("sample_soil(rover1, rover1store, waypoint5)", "communicate_soil_data(...)")

	class FakePlanner:
		def __init__(self, workspace: str) -> None:
			self.workspace = workspace

		def toolchain_available(self) -> bool:
			return True

		def plan_hddl_files(self, **_: object) -> FakePlan:
			return FakePlan()

		def extract_method_trace(self, plan_text: str):
			assert plan_text == "fake"
			return [
				{"method_name": "m-get_rock_data", "task_args": ["waypoint0"]},
				{"method_name": "m-get_soil_data", "task_args": ["waypoint5"]},
			]

	monkeypatch.setattr("pipeline.domain_complete_pipeline.PANDAPlanner", FakePlanner)

	guidance = pipeline._infer_unordered_execution_guidance(
		dfa_result=dfa_result,
		method_library=method_library,
		online_domain=online_domain,
	)

	assert guidance["preferred_unordered_target_ids"][:2] == ("subgoal_5", "subgoal_1")
	assert guidance["preferred_method_trace"][:2] == (
		{"method_name": "m-get_rock_data", "task_args": ("waypoint0",)},
		{"method_name": "m-get_soil_data", "task_args": ("waypoint5",)},
	)
	assert guidance["preferred_action_path"] == (
		"sample_soil(rover1, rover1store, waypoint5)",
		"communicate_soil_data(...)",
	)
	assert guidance["preferred_hierarchical_plan_text"] == "fake"


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
	monkeypatch.setattr(runner, "_is_successful_run", lambda **kwargs: True)
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
		target_literals=(),
		protected_target_literals=(),
		method_library=_sample_method_library(),
		action_schemas=[{"name": "noop"}],
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		domain_name="blocks",
		problem_file=None,
		output_dir=tmp_path,
		completion_mode="accepting_state",
	)

	assert captured["agentspeak_code"].index("m-navigate_abs-1") < captured["agentspeak_code"].index(
		"m-navigate_abs-3"
	)


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


def test_goal_grounding_chat_completion_uses_single_json_object_request() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(
		domain_file=domain_file,
		model="minimax/minimax-m2.7",
		base_url="https://openrouter.ai/api/v1",
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
	assert fake_completions.calls[0]["response_format"] == {"type": "json_object"}
	assert fake_completions.calls[0]["extra_body"] == {
		"provider": {"only": ["minimax"], "allow_fallbacks": False},
		"reasoning": {"max_tokens": 1, "exclude": True},
	}


def test_execute_query_with_jason_returns_none_when_runtime_fails(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	pipeline = DomainCompletePipeline(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	pipeline.output_dir = tmp_path
	online_domain = pipeline._resolve_online_domain_context()
	grounding_result = domain_complete_pipeline_module.TemporalGroundingResult(
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

	monkeypatch.setattr(domain_complete_pipeline_module, "JasonRunner", lambda **_kwargs: FakeRunner())
	monkeypatch.setattr(
		pipeline,
		"_planner_action_schemas_for_domain",
		lambda _domain: [{"action_name": "turn_to"}],
	)
	monkeypatch.setattr(
		pipeline,
		"_render_supported_hierarchical_plan",
		lambda **_kwargs: "guided plan",
	)

	result = pipeline._execute_query_with_jason(
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
	pipeline = DomainCompletePipeline(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	pipeline.output_dir = tmp_path
	online_domain = pipeline._resolve_online_domain_context()
	grounding_result = domain_complete_pipeline_module.TemporalGroundingResult(
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

	monkeypatch.setattr(domain_complete_pipeline_module, "JasonRunner", lambda **_kwargs: FakeRunner())
	monkeypatch.setattr(
		pipeline,
		"_planner_action_schemas_for_domain",
		lambda _domain: [{"action_name": "turn_to"}],
	)
	monkeypatch.setattr(
		pipeline,
		"_render_supported_hierarchical_plan",
		lambda **_kwargs: "reconstructed plan",
	)

	result = pipeline._execute_query_with_jason(
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
	assert result.guided_hierarchical_plan_text == "reconstructed plan"


def test_verify_plan_officially_restores_terminal_newline_for_guided_plan_text(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	pipeline = DomainCompletePipeline(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
		online_domain_source="benchmark",
	)
	pipeline.output_dir = tmp_path
	online_domain = pipeline._resolve_online_domain_context(source="benchmark")

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
			raise AssertionError("verify_plan should not be called when guided plan text is present")

	monkeypatch.setattr(domain_complete_pipeline_module, "IPCPlanVerifier", lambda: FakeVerifier())

	plan_verification = pipeline._verify_plan_officially(
		None,
		_sample_method_library(),
		{
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
				"guided_hierarchical_plan_text": "==>\nroot",
				"action_path": [],
				"method_trace": [],
			},
		},
		online_domain=online_domain,
	)

	assert plan_verification is not None
	assert plan_verification["summary"]["status"] == "success"
