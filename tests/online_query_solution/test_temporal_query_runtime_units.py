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
from online_query_solution.jason_runtime.runner import JasonRunner
from online_query_solution.temporal_compilation import dfa_builder
from utils.hddl_parser import HDDLParser
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


def test_goal_grounding_validator_accepts_grounded_subgoals() -> None:
	generator = NLToLTLfGenerator()
	result = generator._validate_payload(
		query_text="stack block b on a",
		payload={
			"ltlf_formula": "F(subgoal_1)",
			"subgoals": [
				{"id": "subgoal_1", "task_name": "stack", "args": ["b", "a"]},
			],
		},
		method_library=_sample_method_library(),
		typed_objects={"a": "block", "b": "block"},
		task_type_map={"stack": ("block", "block")},
		type_parent_map={"block": "object", "object": None},
	)

	assert result.ltlf_formula == "F(subgoal_1)"
	assert len(result.subgoals) == 1
	assert result.subgoals[0].task_name == "stack"
	assert result.subgoals[0].args == ("b", "a")


def test_goal_grounding_validator_rejects_undeclared_formula_atoms() -> None:
	generator = NLToLTLfGenerator()

	with pytest.raises(ValueError, match="undeclared subgoals"):
		generator._validate_payload(
			query_text="stack block b on a",
			payload={
				"ltlf_formula": "F(subgoal_2)",
				"subgoals": [
					{"id": "subgoal_1", "task_name": "stack", "args": ["b", "a"]},
				],
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
				"ltlf_formula": "F(subgoal_1)",
				"subgoals": [
					{"id": "subgoal_1", "task_name": "stack", "args": ["?x", "a"]},
				],
			},
			method_library=_sample_method_library(),
			typed_objects={"a": "block", "b": "block"},
			task_type_map={"stack": ("block", "block")},
			type_parent_map={"block": "object", "object": None},
		)


def test_goal_grounding_formula_atom_extractor_accepts_operator_sequences() -> None:
	atoms = NLToLTLfGenerator._extract_formula_atoms("XF(subgoal_1)")

	assert atoms == {"subgoal_1"}


def test_goal_grounding_validator_repairs_formula_parentheses_before_validation() -> None:
	generator = NLToLTLfGenerator()

	result = generator._validate_payload(
		query_text="ordered stack",
		payload={
			"ltlf_formula": "F(subgoal_1 & F(subgoal_2)))",
			"subgoals": [
				{"id": "subgoal_1", "task_name": "stack", "args": ["b", "a"]},
				{"id": "subgoal_2", "task_name": "stack", "args": ["c", "b"]},
			],
		},
		method_library=_sample_method_library(),
		typed_objects={"a": "block", "b": "block", "c": "block"},
		task_type_map={"stack": ("block", "block")},
		type_parent_map={"block": "object", "object": None},
	)

	assert result.ltlf_formula == "F(subgoal_1 & F(subgoal_2))"


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
	assert "F(subgoal_1 & F(subgoal_2 & F(subgoal_3)))" in system_prompt


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
		'{"ltlf_formula":"F(subgoal_1) & F(subgoal_2)",'
		'"subgoals":['
		'{"id":"subgoal_1","task_name":"do_put_on","args":["b4","b2"]},'
		'{"id":"subgoal_2","task_name":"do_put_on","args":["b1","b4"]}'
		'],'
		'"diagnostics":[]}'
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

	assert result.ltlf_formula == "F(subgoal_1) & F(subgoal_2)"
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

	assert tuple(attempt["mode"] for attempt in attempts) == ("standard", "compact")


def test_dfa_builder_wraps_converter_output(monkeypatch: pytest.MonkeyPatch) -> None:
	class FakeConverter:
		def convert(self, formula: str):
			assert formula == "G(subgoal_1)"
			return (
				'digraph G { init -> q0; q0 [shape=circle]; q1 [shape=doublecircle]; q0 -> q1 [label="subgoal_1"]; }',
				{
					"construction": "fake_converter",
					"num_states": 2,
					"num_transitions": 1,
					"alphabet": ["subgoal_1"],
				},
			)

	monkeypatch.setattr(dfa_builder, "LTLfToDFA", lambda: FakeConverter())

	result = dfa_builder.build_dfa_from_ltlf("G(subgoal_1)")

	assert result["construction"] == "fake_converter"
	assert result["num_states"] == 2
	assert result["num_transitions"] == 1
	assert result["alphabet"] == ["subgoal_1"]
	assert "subgoal_1" in result["dfa_dot"]


def test_dfa_builder_symbolically_compiles_total_ordered_subgoal_sequence(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	class FakeConverter:
		def convert(self, formula: str):
			raise AssertionError(f"Generic converter should not run for {formula!r}")

	monkeypatch.setattr(dfa_builder, "LTLfToDFA", lambda: FakeConverter())

	result = dfa_builder.build_dfa_from_ltlf(
		"F(subgoal_1 & F(subgoal_2 & F(subgoal_3))))",
	)

	assert result["construction"] == "symbolic_ordered_subgoal_sequence"
	assert result["num_states"] == 4
	assert result["num_transitions"] == 3
	assert result["alphabet"] == ["subgoal_1", "subgoal_2", "subgoal_3"]
	assert '1 -> 2 [label="subgoal_1"]' in result["dfa_dot"]
	assert '3 -> 4 [label="subgoal_3"]' in result["dfa_dot"]


def test_dfa_builder_symbolically_compiles_very_large_ordered_subgoal_sequence() -> None:
	formula = "F(subgoal_1039)"
	for subgoal_index in range(1038, 0, -1):
		formula = f"F(subgoal_{subgoal_index} & {formula})"

	result = dfa_builder.build_dfa_from_ltlf(formula)

	assert result["construction"] == "symbolic_ordered_subgoal_sequence"
	assert result["num_states"] == 1040
	assert result["num_transitions"] == 1039
	assert result["alphabet"][0] == "subgoal_1"
	assert result["alphabet"][-1] == "subgoal_1039"


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
	assert "subgoal_cursor(1)." in asl
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


def test_pipeline_prefers_query_specific_problem_when_original_problem_has_goal_facts(
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

	assert mode == "query_specific_problem"
	assert Path(problem_file).resolve() != (
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
		ordered_query_sequence=True,
	)

	assert '.print("execute start")' in runtime_program
	assert ".perceive" in runtime_program
	assert "clear(b1)." in runtime_program
	assert runtime_program.index('.print("execute start")') < runtime_program.index(".perceive")


def test_jason_runner_unordered_accepting_state_executes_until_all_subgoals_seen() -> None:
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
		ordered_query_sequence=False,
	)

	assert "!verify_targets" in runtime_program
	assert "target_seen(subgoal_1)" in runtime_program
	assert "target_seen(subgoal_2)" in runtime_program
	assert "+!mark_target_subgoal_1 : true <-" in runtime_program
	assert "+!mark_target_subgoal_2 : true <-" in runtime_program
	assert "?accepting_state(FINAL_STATE)" not in runtime_program


def test_jason_runner_accepting_state_guided_replay_uses_guided_execution_entry() -> None:
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
		ordered_query_sequence=False,
		guided_action_path=("navigate(rover1, waypoint4, waypoint5)",),
		guided_method_trace=(
			{"method_name": "m-navigate_abs-3", "task_args": ["rover1", "waypoint5"]},
		),
	)

	execute_section = runtime_program.split("+!execute : true <-", maxsplit=1)[1]

	assert "!guided_replay_1" in execute_section
	assert '.print("execute success")' in execute_section
	assert "!run_dfa" not in execute_section


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


def test_jason_runner_validate_prioritises_preferred_method_guidance_before_build(
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
		ordered_query_sequence=False,
		preferred_method_trace=(
			{"method_name": "m-navigate_abs-3", "task_args": ["rover1", "waypoint5"]},
		),
	)

	assert captured["agentspeak_code"].index("m-navigate_abs-3") < captured["agentspeak_code"].index(
		"m-navigate_abs-1"
	)


def test_goal_grounding_retries_with_compact_prompt_after_invalid_json() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)
	generator.client = object()

	seen_messages: list[list[dict[str, str]]] = []
	responses = iter(
		[
			"{\n  \"ltlf_formula\": \"unterminated",
			'{"ltlf_formula":"F(subgoal_1 & F(subgoal_2))","subgoals":[{"id":"subgoal_1","task_name":"do_put_on","args":["b4","b2"]},{"id":"subgoal_2","task_name":"do_put_on","args":["b1","b4"]}],"diagnostics":[]}',
		],
	)

	def fake_create(self, messages, **_kwargs):
		seen_messages.append(list(messages))
		return next(responses)

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(NLToLTLfGenerator, "_create_chat_completion", fake_create)
	monkeypatch.setattr(NLToLTLfGenerator, "_extract_response_text", lambda self, response: str(response))
	try:
		result, _, response_text = generator.generate(
			"First put block b4 on block b2, then put block b1 on block b4.",
			method_library=_sample_method_library(),
			typed_objects={"b1": "block", "b2": "block", "b4": "block"},
			task_type_map={"do_put_on": ("block", "block")},
			type_parent_map={"block": "object", "object": None},
		)
	finally:
		monkeypatch.undo()

	assert result.ltlf_formula == "F(subgoal_1 & F(subgoal_2))"
	assert response_text.startswith('{"ltlf_formula"')
	assert len(seen_messages) == 2
	assert "Return minified JSON only." in seen_messages[1][0]["content"]
