from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from method_library.synthesis.schema import HTNMethod, HTNMethodLibrary, HTNMethodStep, HTNTask
from evaluation.artifacts import GroundedSubgoal, TemporalGroundingResult
from evaluation.agentspeak import AgentSpeakRenderer
from evaluation.goal_grounding.canonical_ordered_formula import (
	ANCHORED_NEXT_CHAIN,
	CANONICAL_BENCHMARK_ORDERED_FORMULA_STYLE,
	select_canonical_benchmark_ordered_formula_style,
)
from evaluation.goal_grounding.grounder import (
	GoalGroundingEmptyResponseError,
	GoalGroundingProviderUnavailable,
	NLToLTLfGenerator,
)
from evaluation.jason_runtime.environment_adapter import EnvironmentAdapterResult
from evaluation.jason_runtime.runner import JasonRunner, JasonValidationResult
from evaluation.failure_signature import infer_missing_goal_facts
from evaluation.official_verification import resolve_verification_domain_file
from evaluation.orchestrator import PlanLibraryEvaluationOrchestrator
from plan_library import (
	AgentSpeakBodyStep,
	AgentSpeakPlan,
	AgentSpeakTrigger,
	LibraryValidationRecord,
	PlanLibrary,
	PlanLibraryArtifactBundle,
	TranslationCoverage,
)
from utils.hddl_parser import HDDLParser
from verification.official_plan_verifier import IPCPlanVerifier, IPCPrimitivePlanVerificationResult
from evaluation import official_verification as evaluation_official_verification_module
from evaluation import orchestrator as evaluation_orchestrator_module


def _artifact_bundle(
	*,
	method_library: HTNMethodLibrary,
	plan_library: PlanLibrary | None = None,
	artifact_root: str | None = None,
	domain_name: str = "blocks",
) -> PlanLibraryArtifactBundle:
	return PlanLibraryArtifactBundle(
		domain_name=domain_name,
		query_sequence=(),
		temporal_specifications=(),
		method_library=method_library,
		plan_library=plan_library or PlanLibrary(domain_name=domain_name, plans=()),
		translation_coverage=TranslationCoverage(
			domain_name=domain_name,
			methods_considered=len(tuple(method_library.methods or ())),
			plans_generated=0,
			accepted_translation=0,
		),
		library_validation=LibraryValidationRecord(
			library_id=domain_name,
			passed=True,
			method_count=len(tuple(method_library.methods or ())),
			plan_count=0,
			checked_layers={},
		),
		method_synthesis_metadata={},
		artifact_root=artifact_root,
	)


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


def _sample_plan_library() -> PlanLibrary:
	return PlanLibrary(
		domain_name="blocks",
		plans=(
			AgentSpeakPlan(
				plan_name="m_do_put_on_serial",
				trigger=AgentSpeakTrigger(
					event_type="achievement_goal",
					symbol="do_put_on",
					arguments=("X:block", "Y:block"),
				),
				context=("clear(X)", "X != Y"),
				body=(
					AgentSpeakBodyStep(kind="action", symbol="pick_up", arguments=("X",)),
					AgentSpeakBodyStep(kind="subgoal", symbol="stack", arguments=("X", "Y")),
				),
				source_instruction_ids=("query_1",),
			),
		),
	)


def _rover_method_library() -> HTNMethodLibrary:
	return HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="get_soil_data", parameters=("waypoint",), is_primitive=False),
			HTNTask(name="get_rock_data", parameters=("waypoint",), is_primitive=False),
			HTNTask(
				name="get_image_data",
				parameters=("objective", "mode"),
				is_primitive=False,
			),
		],
		primitive_tasks=[],
		methods=[],
	)


def _transport_method_library() -> HTNMethodLibrary:
	return HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="deliver", parameters=("package", "location"), is_primitive=False),
			HTNTask(name="get_to", parameters=("vehicle", "location"), is_primitive=False),
			HTNTask(name="load", parameters=("vehicle", "location", "package"), is_primitive=False),
		],
		primitive_tasks=[],
		methods=[],
	)


def _satellite_method_library() -> HTNMethodLibrary:
	return HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="do_observation", parameters=("direction", "mode"), is_primitive=False),
			HTNTask(
				name="activate_instrument",
				parameters=("satellite", "instrument"),
				is_primitive=False,
			),
			HTNTask(name="auto_calibrate", parameters=("satellite", "instrument"), is_primitive=False),
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


def test_goal_grounding_validator_rejects_formula_with_json_braces() -> None:
	generator = NLToLTLfGenerator()

	with pytest.raises(ValueError, match="JSON braces"):
		generator._validate_payload(
			query_text="stack block b on a",
			payload={
				"ltlf_formula": "F(stack(b, a))}",
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
	assert "anchored Next chain" in system_prompt
	assert "A & X(B)" in system_prompt
	assert "do not wrap the ordered task-event chain in an outer f(...)" in system_prompt.lower()
	assert "__e1(" in system_prompt
	assert "final JSON answer must appear in the completion response content itself" in system_prompt
	assert "MUST NOT leave the final answer only in hidden reasoning content" in system_prompt
	assert "MUST NOT return an empty completion response" in system_prompt
	assert "goal_facts" not in system_prompt


def test_goal_grounding_prompt_compacts_large_object_inventory_without_changing_validation() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)
	typed_objects = {f"b{index}": "block" for index in range(1, 301)}
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="do_clear", parameters=("x",), is_primitive=False),
			HTNTask(name="do_on_table", parameters=("x",), is_primitive=False),
		],
		primitive_tasks=[],
		methods=[],
	)

	system_prompt, _ = generator._build_prompts(
		query_text=(
			"Using blocks b18 and b24, complete the tasks "
			"do_clear(b18), then do_on_table(b18), then do_clear(b24)."
		),
		method_library=method_library,
		typed_objects=typed_objects,
		task_type_map={
			"do_clear": ("block",),
			"do_on_table": ("block",),
		},
	)
	result = generator._validate_payload(
		query_text="do_clear(b300)",
		payload={"ltlf_formula": "F(do_clear(b300))"},
		method_library=method_library,
		typed_objects=typed_objects,
		task_type_map={"do_clear": ("block",)},
		type_parent_map={"block": "object", "object": None},
	)

	assert "Grounded problem objects relevant to this query:" in system_prompt
	assert "b18" in system_prompt
	assert "b24" in system_prompt
	assert "b300" not in system_prompt
	assert "omitted_objects" in system_prompt
	assert result.subgoals[0].args == ("b300",)


def test_goal_grounding_prompt_uses_rover_local_examples_without_do_prefix_bias() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "marsrover" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)

	system_prompt, _ = generator._build_prompts(
		query_text="Using rover waypoint objects, complete the tasks get_soil_data(waypoint0), then get_rock_data(waypoint1).",
		method_library=_rover_method_library(),
		typed_objects={
			"waypoint0": "waypoint",
			"waypoint1": "waypoint",
			"objective0": "objective",
			"mode0": "mode",
		},
		task_type_map={
			"get_soil_data": ("waypoint",),
			"get_rock_data": ("waypoint",),
			"get_image_data": ("objective", "mode"),
		},
	)

	assert "get_soil_data(" in system_prompt
	assert "get_rock_data(" in system_prompt
	assert "get_image_data(" in system_prompt
	assert "do_get_soil_data(" not in system_prompt
	assert "do_put_on(" not in system_prompt


def test_goal_grounding_prompt_uses_transport_local_examples_without_do_deliver_bias() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "transport" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)

	system_prompt, _ = generator._build_prompts(
		query_text="Using transport objects, complete the tasks deliver(package0, location0).",
		method_library=_transport_method_library(),
		typed_objects={
			"truck0": "vehicle",
			"location0": "location",
			"package0": "package",
		},
		task_type_map={
			"deliver": ("package", "location"),
			"get_to": ("vehicle", "location"),
			"load": ("vehicle", "location", "package"),
		},
	)

	assert "deliver(" in system_prompt
	assert "get_to(" in system_prompt
	assert "do_deliver(" not in system_prompt
	assert "do_put_on(" not in system_prompt


def test_goal_grounding_prompt_uses_grounded_satellite_examples_without_lifted_atoms() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "satellite" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)

	system_prompt, _ = generator._build_prompts(
		query_text="Using satellite objects, complete the tasks do_observation(direction0, mode0).",
		method_library=_satellite_method_library(),
		typed_objects={
			"direction0": "image_direction",
			"mode0": "mode",
			"satellite0": "satellite",
			"instrument0": "instrument",
		},
		task_type_map={
			"do_observation": ("image_direction", "mode"),
			"activate_instrument": ("satellite", "instrument"),
			"auto_calibrate": ("satellite", "instrument"),
		},
	)

	assert "do_observation(direction0, mode0)" in system_prompt
	assert "activate_instrument(satellite0, instrument0)" in system_prompt
	assert "auto_calibrate(satellite0, instrument0)" in system_prompt
	assert "do_observation(?" not in system_prompt


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
	assert "__e2(" in system_prompt


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
			raise GoalGroundingEmptyResponseError(
				"LLM response did not contain any textual completion content.",
			)
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


def test_goal_grounding_marks_provider_unavailable_after_transport_retries() -> None:
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
		with pytest.raises(GoalGroundingProviderUnavailable, match="did not return usable"):
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


def test_goal_grounding_does_not_retry_malformed_text_response() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(domain_file=domain_file)
	generator.client = object()
	call_count = {"create": 0}

	def fake_create(self, messages, **_kwargs):
		del messages
		call_count["create"] += 1
		return "this is not json"

	monkeypatch = pytest.MonkeyPatch()
	monkeypatch.setattr(NLToLTLfGenerator, "_create_chat_completion", fake_create)
	monkeypatch.setattr(NLToLTLfGenerator, "_extract_response_text", lambda self, response: str(response))
	try:
		with pytest.raises(Exception):
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
	assert len(generator.last_generation_metadata["attempt_errors"]) == 1
	assert generator.last_generation_metadata["attempt_errors"][0]["retryable"] == "false"


def test_canonical_ordered_formula_style_selection_prefers_small_complete_candidate() -> None:
	chosen_style = select_canonical_benchmark_ordered_formula_style(
		{
			"adjacent_until_strict_precedence": {
				"compiled_case_count": 0,
				"total_case_count": 7,
				"median_num_states": float("inf"),
				"median_num_transitions": float("inf"),
				"median_convert_seconds": float("inf"),
				"median_formula_length": 1000.0,
			},
			"anchored_next_chain": {
				"compiled_case_count": 7,
				"total_case_count": 7,
				"median_num_states": 26.0,
				"median_num_transitions": 50.0,
				"median_convert_seconds": 0.04,
				"median_formula_length": 420.0,
			},
			"eventual_next_chain": {
				"compiled_case_count": 7,
				"total_case_count": 7,
				"median_num_states": 35.0,
				"median_num_transitions": 72.0,
				"median_convert_seconds": 0.18,
				"median_formula_length": 424.0,
			},
		},
	)

	assert CANONICAL_BENCHMARK_ORDERED_FORMULA_STYLE == ANCHORED_NEXT_CHAIN
	assert chosen_style == ANCHORED_NEXT_CHAIN


def test_jason_runner_rejects_unbound_runtime_variables_in_method_bodies() -> None:
	runner = JasonRunner()

	unsafe_chunk = [
		"+!get_to(V, DEST) : road(SRC, DEST) <-",
		'\t.print("runtime trace method flat ", "m-drive-to-via");',
		"\t!get_to(V, MID);",
		"\t!drive(V, MID, DEST).",
	]
	safe_chunk = [
		"+!get_to(V, DEST) : road(MID, DEST) & at(V, MID) <-",
		'\t.print("runtime trace method flat ", "m-drive-to-via");',
		"\t!get_to(V, MID);",
		"\t!drive(V, MID, DEST).",
	]
	primitive_wrapper_binding_chunk = [
		"+!collect_data(TARGET, MODE) : resource_ready(DEVICE, HOST) & supports(DEVICE, MODE) <-",
		'\t.print("runtime trace method flat ", "method0");',
		"\t!activate_resource(HOST, DEVICE);",
		"\t!align_resource(HOST, TARGET, OLD_TARGET);",
		"\t!capture_data(HOST, TARGET, DEVICE, MODE).",
	]

	assert runner._chunk_runtime_variables_are_safe(unsafe_chunk) is False
	assert runner._chunk_runtime_variables_are_safe(safe_chunk) is True
	assert runner._chunk_runtime_variables_are_safe(primitive_wrapper_binding_chunk) is True


def test_jason_runner_defers_runtime_only_type_guards_for_method_local_variables() -> None:
	runner = JasonRunner()
	agentspeak_code = """
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
+!observe(DIR, MODE) : object_type(PREV, direction) & object_type(DIR, image_direction) & object_type(INST, instrument) & object_type(SAT, satellite) & on_board(INST, SAT) <-
	!turn_to(SAT, DIR, PREV);
	!capture(SAT, DIR, INST, MODE).

+!already_observe(DIR) : object_type(DIR, image_direction) <-
	true.

/* Failure Handlers */
""".strip()

	rewritten = runner._defer_type_only_local_context_guards(agentspeak_code)
	observe_head = rewritten.split("+!observe", maxsplit=1)[1].split("<-", maxsplit=1)[0]
	already_head = rewritten.split("+!already_observe", maxsplit=1)[1].split("<-", maxsplit=1)[0]

	assert "object_type(PREV, direction)" not in observe_head
	assert "object_type(DIR, image_direction)" in observe_head
	assert "object_type(INST, instrument)" in observe_head
	assert "object_type(SAT, satellite)" in observe_head
	assert "on_board(INST, SAT)" in observe_head
	assert "object_type(DIR, image_direction)" in already_head


def test_jason_runtime_ordering_contains_no_domain_named_heuristics() -> None:
	source = (SRC_ROOT / "evaluation" / "jason_runtime" / "runner.py").read_text()
	for forbidden_token in (
		"take_image",
		"turn_to",
		"activate_instrument",
		"calibration_target",
		"via_like_method",
	):
		assert forbidden_token not in source


def test_jason_runner_orders_already_satisfied_then_direct_then_recursive_chunks() -> None:
	runner = JasonRunner()
	chunks = [
		"\n".join(
			[
				"+!get_to(V, DEST) : at(V, DEST) <-",
				'\t.print("runtime trace method flat ", "m-i-am-there");',
				"\ttrue.",
			],
		),
		"\n".join(
			[
				"+!get_to(V, DEST) : at(V, SRC) & road(SRC, DEST) <-",
				'\t.print("runtime trace method flat ", "m-drive-to");',
				"\t!drive(V, SRC, DEST).",
			],
		),
		"\n".join(
			[
				"+!get_to(V, DEST) : road(MID, DEST) & at(V, MID) <-",
				'\t.print("runtime trace method flat ", "m-drive-to-via");',
				"\t!get_to(V, MID);",
				"\t!drive(V, MID, DEST).",
			],
		),
	]

	ordered_chunks = runner._order_runtime_method_plan_chunks(
		chunks,
		fact_index={
			("at", 2): (("truck_0", "loc_0"),),
			("road", 2): (("loc_0", "loc_1"), ("loc_1", "loc_2")),
		},
	)

	assert '"m-i-am-there"' in ordered_chunks[0]
	assert '"m-drive-to"' in ordered_chunks[1]
	assert '"m-drive-to-via"' in ordered_chunks[2]


def test_jason_runner_prefers_full_nonrecursive_decomposition_over_shorter_prefix() -> None:
	runner = JasonRunner()
	chunks = [
		"\n".join(
			[
				"+!achieve(TARGET) : ready(RESOURCE) <-",
				'\t.print("runtime trace method flat ", "short-prefix");',
				"\t!prepare(RESOURCE);",
				"\t!finish(TARGET).",
			],
		),
		"\n".join(
			[
				"+!achieve(TARGET) : ready(RESOURCE) <-",
				'\t.print("runtime trace method flat ", "full-decomposition");',
				"\t!prepare(RESOURCE);",
				"\t!align(RESOURCE, TARGET);",
				"\t!finish(TARGET).",
			],
		),
	]

	ordered_chunks = runner._order_runtime_method_plan_chunks(
		chunks,
		fact_index={("ready", 1): (("resource_0",),)},
	)

	assert '"full-decomposition"' in ordered_chunks[0]
	assert '"short-prefix"' in ordered_chunks[1]


def test_jason_runner_keeps_recursive_transport_via_after_current_context_via() -> None:
	runner = JasonRunner()
	chunks = [
		"\n".join(
			[
				"+!get_to(V, DEST) : road(MID, DEST) <-",
				'\t.print("runtime trace method flat ", "m-drive-to-via");',
				"\t!get_to(V, MID);",
				"\t!drive(V, MID, DEST).",
			],
		),
		"\n".join(
			[
				"+!get_to(V, DEST) : road(MID, DEST) & at(V, MID) <-",
				'\t.print("runtime trace method flat ", "m-drive-to-via");',
				"\t!get_to(V, MID);",
				"\t!drive(V, MID, DEST).",
			],
		),
	]

	ordered_chunks = runner._order_runtime_method_plan_chunks(
		chunks,
		fact_index={
			("at", 2): (("truck_0", "loc_0"),),
			("road", 2): (("loc_0", "loc_1"),),
		},
	)

	assert len(ordered_chunks) == 2
	assert "at(V, MID)" in ordered_chunks[0]
	assert "road(MID, DEST) <-" in ordered_chunks[1]


def test_jason_runner_orders_safe_chunks_without_dropping_lifted_fallbacks() -> None:
	runner = JasonRunner()
	unsafe_lifted_fallback = "\n".join(
		[
			"+!achieve(TARGET) : object_type(WITNESS, witness) <-",
			'\t.print("runtime trace method flat ", "unsafe-lifted");',
			"\t-+runtime_current_call(method, achieve, runtime_args(TARGET), "
			"runtime_binding(WITNESS));",
			"\t!step(TARGET, WITNESS).",
		],
	)
	safe_chunk = "\n".join(
		[
			"+!achieve(TARGET) : ready(TARGET) <-",
			'\t.print("runtime trace method flat ", "safe");',
			"\t!finish(TARGET).",
		],
	)

	ordered_chunks = runner._order_runtime_method_plan_chunks(
		[unsafe_lifted_fallback, safe_chunk],
		fact_index={("ready", 1): (("goal0",),)},
	)

	assert len(ordered_chunks) == 2
	assert "safe" in ordered_chunks[0]
	assert "unsafe-lifted" in ordered_chunks[1]


def test_jason_runner_local_witness_specialisation_keeps_canonical_fallback() -> None:
	runner = JasonRunner()
	chunk = [
		"+!observe(DIR, MODE) : object_type(PREV, direction) & "
		"object_type(SAT, satellite) & object_type(INST, instrument) & "
		"object_type(MODE, mode) & on_board(INST, SAT) & supports(INST, MODE) <-",
		'\t.print("runtime trace method flat ", "observe-with-turn");',
		"\t-+runtime_current_call(method, observe, runtime_args(DIR, MODE), "
		"runtime_binding(PREV, SAT, INST));",
		"\t!activate_instrument(SAT, INST);",
		"\t!turn_to(SAT, DIR, PREV);",
		"\t!take_image(SAT, DIR, INST, MODE).",
	]
	fact_index, type_domains = runner._runtime_fact_index_for_local_witness_grounding(
		seed_facts=("(on_board instrument0 satellite0)", "(supports instrument0 mode0)"),
		runtime_objects=("instrument0", "satellite0", "mode0", "old_direction"),
		object_types={
			"instrument0": "instrument",
			"satellite0": "satellite",
			"mode0": "mode",
			"old_direction": "direction",
		},
		type_parent_map={
			"instrument": "object",
			"satellite": "object",
			"mode": "object",
			"direction": "object",
			"object": None,
		},
	)

	specialised_chunks = runner._specialise_method_chunk_local_witnesses(
		chunk,
		fact_index=fact_index,
		type_domains=type_domains,
		max_candidates_per_clause=64,
	)

	original = "\n".join(chunk)
	assert original in specialised_chunks
	assert any(
		"old_direction" in specialised_chunk and specialised_chunk != original
		for specialised_chunk in specialised_chunks
	)


def test_jason_runner_inserts_no_ancestor_guard_for_self_recursive_methods() -> None:
	runner = JasonRunner()
	chunks = [
		"\n".join(
			[
				"+!get_to(V, DEST) : road(MID, DEST) <-",
				'\t.print("runtime trace method flat ", "m-drive-to-via");',
				"\t!get_to(V, MID);",
				"\t!drive(V, MID, DEST).",
			],
		),
	]

	guarded_chunks = runner._promote_body_no_ancestor_guards_to_context(
		runner._insert_self_recursive_no_ancestor_guards(chunks),
	)

	assert len(guarded_chunks) == 1
	assert (
		"+!get_to(V, DEST) : road(MID, DEST) & "
		"pipeline.no_ancestor_goal(get_to, V, MID) <-"
	) in guarded_chunks[0]
	assert "\t!get_to(V, MID);" in guarded_chunks[0]


def test_jason_runner_does_not_apply_domain_named_action_setup_ordering() -> None:
	runner = JasonRunner()
	chunks = [
		"\n".join(
			[
				"+!do_observation(DIR, MODE) : supports(I, MODE) & on_board(I, SAT) & power_avail(SAT) <-",
				'\t.print("runtime trace method flat ", "method_without_turn");',
				"\t!activate_instrument(SAT, I);",
				"\t!take_image(SAT, DIR, I, MODE).",
			],
		),
		"\n".join(
			[
				"+!do_observation(DIR, MODE) : supports(I, MODE) & on_board(I, SAT) & power_avail(SAT) & pointing(SAT, OLD_DIR) <-",
				'\t.print("runtime trace method flat ", "method_with_turn");',
				"\t!activate_instrument(SAT, I);",
				"\t!turn_to(SAT, DIR, OLD_DIR);",
				"\t!take_image(SAT, DIR, I, MODE).",
			],
		),
		"\n".join(
			[
				"+!do_observation(DIR, MODE) : supports(I, MODE) & on_board(I, SAT) & calibrated(I) & power_on(I) & pointing(SAT, DIR) <-",
				'\t.print("runtime trace method flat ", "method_already_pointing");',
				"\t!take_image(SAT, DIR, I, MODE).",
			],
		),
	]

	ordered_chunks = runner._order_runtime_method_plan_chunks(
		chunks,
		fact_index={},
	)

	assert "method_already_pointing" in ordered_chunks[0]
	assert {ordered_chunks[1], ordered_chunks[2]} == {
		chunks[0],
		chunks[1],
	}


def test_jason_runner_ignores_domain_named_fact_indexes_during_chunk_ordering() -> None:
	runner = JasonRunner()
	chunks = [
		"\n".join(
			[
				"+!do_observation(DIR, MODE) : supports(instrument0, MODE) & on_board(instrument0, satellite0) & power_avail(satellite0) <-",
				'\t.print("runtime trace method flat ", "method_bad_source");',
				"\t!activate_instrument(satellite0, instrument0);",
				"\t!turn_to(satellite0, DIR, old_direction);",
				"\t!take_image(satellite0, DIR, instrument0, MODE).",
			],
		),
		"\n".join(
			[
				"+!do_observation(DIR, MODE) : supports(instrument0, MODE) & on_board(instrument0, satellite0) & power_avail(satellite0) <-",
				'\t.print("runtime trace method flat ", "method_calibration_source");',
				"\t!activate_instrument(satellite0, instrument0);",
				"\t!turn_to(satellite0, DIR, calib_direction0);",
				"\t!take_image(satellite0, DIR, instrument0, MODE).",
			],
		),
	]

	ordered_chunks = runner._order_runtime_method_plan_chunks(
		chunks,
		fact_index={
			("calibration_target", 2): (("instrument0", "calib_direction0"),),
		},
	)

	assert "method_bad_source" in ordered_chunks[0]
	assert "method_calibration_source" in ordered_chunks[1]


def test_jason_runner_build_asl_applies_runtime_method_lowering() -> None:
	runner = JasonRunner()
	agentspeak_code = "\n".join(
		[
			"/* Initial Beliefs */",
			"at(truck, loc0).",
			"road(loc0, loc1).",
			"",
			"/* Primitive Action Plans */",
			"",
			"/* HTN Method Plans */",
			"+!get_to(V, DEST) : at(V, SRC) & road(SRC, DEST) <-",
			'\t.print("runtime trace method flat ", "m-drive-to");',
			"\t!drive(V, SRC, DEST).",
			"",
			"+!get_to(V, DEST) : road(MID, DEST) <-",
			'\t.print("runtime trace method flat ", "m-drive-to-via-unsafe");',
			"\t!get_to(V, MID);",
			"\t!drive(V, MID, DEST).",
			"",
			"/* Failure Handlers */",
		],
	)

	runner_asl = runner._build_runner_asl(
		agentspeak_code,
		seed_facts=("at(truck, loc0)", "road(loc0, loc1)"),
		runtime_objects=("truck", "loc0", "loc1"),
		object_types={"truck": "vehicle", "loc0": "location", "loc1": "location"},
		type_parent_map={"vehicle": "object", "location": "object", "object": None},
	)

	assert "m-drive-to" in runner_asl
	assert runner_asl.index("m-drive-to") < runner_asl.index("m-drive-to-via-unsafe")


def test_jason_runner_rewrites_method_primitive_actions_to_wrapper_goals() -> None:
	runner = JasonRunner()
	agentspeak_code = "\n".join(
		[
			"/* Initial Beliefs */",
			"",
			"/* Primitive Action Plans */",
			"+!unstack(BLOCK1, BLOCK2) : true <-",
			"\tunstack(BLOCK1, BLOCK2).",
			"",
			"+!put_down(BLOCK) : true <-",
			"\tput_down(BLOCK).",
			"",
			"/* HTN Method Plans */",
			"+!do_on_table(X) : on(X, Y) <-",
			"\tunstack(X, Y);",
			"\tput_down(X).",
			"",
			"/* Failure Handlers */",
		],
	)

	runner_asl = runner._build_runner_asl(
		agentspeak_code,
		action_schemas=(
			{"functor": "unstack", "source_name": "unstack", "parameters": ["?x", "?y"]},
			{"functor": "put_down", "source_name": "put-down", "parameters": ["?x"]},
		),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
	)
	method_section = runner_asl.split("/* HTN Method Plans */", maxsplit=1)[1].split(
		"/* Failure Handlers */",
		maxsplit=1,
	)[0]
	primitive_section = runner_asl.split("/* Primitive Action Plans */", maxsplit=1)[1].split(
		"/* HTN Method Plans */",
		maxsplit=1,
	)[0]

	assert "\t!unstack(X, Y);" in method_section
	assert "\t!put_down(X)." in method_section
	assert "\tunstack(BLOCK1, BLOCK2);" in primitive_section
	assert "\t!unstack(BLOCK1, BLOCK2);" not in primitive_section


def test_jason_runner_instruments_before_runtime_specialisation() -> None:
	runner = JasonRunner()
	agentspeak_code = "\n".join(
		[
			"/* Initial Beliefs */",
			"on(b1, b2).",
			"",
			"/* Primitive Action Plans */",
			"",
			"/* HTN Method Plans */",
			"+!do_on_table(X) : on(X, Y) <-",
			"\tunstack(X, Y).",
			"",
			"/* Failure Handlers */",
		],
	)
	plan_library = PlanLibrary(
		domain_name="blocks",
		plans=(
			AgentSpeakPlan(
				plan_name="m2_do_on_table",
				trigger=AgentSpeakTrigger(
					event_type="achievement_goal",
					symbol="do_on_table",
					arguments=("X:block",),
				),
				context=("on(X, Y)",),
				body=(AgentSpeakBodyStep(kind="action", symbol="unstack", arguments=("X", "Y")),),
			),
		),
	)

	runner_asl = runner._build_runner_asl(
		agentspeak_code,
		plan_library=plan_library,
		action_schemas=(
			{"functor": "unstack", "source_name": "unstack", "parameters": ["?x", "?y"]},
		),
		seed_facts=("(on b1 b2)",),
		runtime_objects=("b1", "b2"),
		object_types={"b1": "block", "b2": "block"},
		type_parent_map={"block": "object", "object": None},
	)

	assert '"runtime trace method flat "' in runner_asl
	assert '"m2_do_on_table"' in runner_asl
	assert "+!do_on_table(b1)" in runner_asl


def test_jason_runner_instruments_plan_library_variants_with_source_method_name() -> None:
	runner = JasonRunner()
	agentspeak_code = "\n".join(
		[
			"/* Initial Beliefs */",
			"",
			"/* Primitive Action Plans */",
			"",
			"/* HTN Method Plans */",
			"+!deliver(PKG, LOC) : true <-",
			"\tload(PKG);",
			"\tmove(LOC);",
			"\tdrop(PKG, LOC).",
			"",
			"+!deliver(PKG, LOC) : true <-",
			"\tload(PKG);",
			"\tdrop(PKG, LOC);",
			"\tmove(LOC).",
			"",
			"/* Failure Handlers */",
		],
	)
	plan_library = PlanLibrary(
		domain_name="courier",
		plans=(
			AgentSpeakPlan(
				plan_name="m_deliver_branching__variant_1",
				trigger=AgentSpeakTrigger(
					event_type="achievement_goal",
					symbol="deliver",
					arguments=("PKG:package", "LOC:location"),
				),
				context=(),
				body=(),
			),
			AgentSpeakPlan(
				plan_name="m_deliver_branching__variant_2",
				trigger=AgentSpeakTrigger(
					event_type="achievement_goal",
					symbol="deliver",
					arguments=("PKG:package", "LOC:location"),
				),
				context=(),
				body=(),
			),
		),
	)

	instrumented = runner._instrument_method_plans(
		agentspeak_code,
		None,
		plan_library=plan_library,
	)

	assert '"m_deliver_branching"' in instrumented
	assert '.print("runtime trace method flat ", "m_deliver_branching"' in instrumented
	assert (
		'blocked_runtime_method("m_deliver_branching__variant_1", deliver, PKG, LOC, '
		"runtime_binding)"
	) in instrumented
	assert (
		'blocked_runtime_method("m_deliver_branching__variant_2", deliver, PKG, LOC, '
		"runtime_binding)"
	) in instrumented


def test_jason_runner_renders_failure_handlers_from_plan_library_and_action_schemas() -> None:
	runner = JasonRunner()

	lines = runner._render_failure_handlers(
		None,
		plan_library=_sample_plan_library(),
		action_schemas=(
			{
				"source_name": "pick_up",
				"functor": "pick_up",
				"parameters": ["?x"],
			},
		),
	)
	rendered = "\n".join(lines)

	assert "-!do_put_on(X, Y) : true <-" in rendered
	assert "-!stack(X, Y) : true <-" in rendered
	assert "-!pick_up(X) : true <-" in rendered


def test_jason_runner_delegates_dynamic_effects_to_environment() -> None:
	runner = JasonRunner()
	agentspeak_code = "\n".join(
		[
			"/* Initial Beliefs */",
			"",
			"/* Primitive Action Plans */",
			"+!update_position(OBJ, TO, FROM) : at(OBJ, FROM) <-",
			"\tupdate_position(OBJ, TO, FROM);",
			"\t+at(OBJ, TO);",
			"\t-at(OBJ, FROM).",
			"",
			"/* HTN Method Plans */",
			"",
			"/* Failure Handlers */",
		],
	)

	rewritten = runner._rewrite_primitive_wrappers_for_environment(agentspeak_code)

	assert "+at(OBJ, TO)" not in rewritten
	assert "-at(OBJ, FROM)" not in rewritten
	assert "update_position(OBJ, TO, FROM);" in rewritten
	assert ".perceive." in rewritten
	assert runner._ordered_runtime_effects(
		(
			{"predicate": "at", "args": ("obj0", "loc0"), "is_positive": True},
			{"predicate": "at", "args": ("obj0", "loc0"), "is_positive": False},
		),
	)[-1]["is_positive"] is True


def test_agentspeak_renderer_emits_plan_library_without_temporal_runtime() -> None:
	domain = HDDLParser.parse_domain(str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"))
	renderer = AgentSpeakRenderer()
	asl = renderer.generate(
		domain=domain,
		objects=("a", "b"),
		method_library=_sample_method_library(),
		plan_records=(),
		typed_objects=(("a", "block"), ("b", "block")),
		subgoals=[{"id": "subgoal_1", "task_name": "stack", "args": ["b", "a"]}],
	)

	assert "query_step" not in asl
	assert "subgoal_cursor(1)." not in asl
	assert "HTN Method Plans" in asl


def test_agentspeak_renderer_treats_empty_hddl_precondition_as_true() -> None:
	domain = HDDLParser.parse_domain(str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"))
	renderer = AgentSpeakRenderer()
	asl = renderer.generate(
		domain=domain,
		objects=("a",),
		method_library=_sample_method_library(),
		plan_records=(),
		typed_objects=(("a", "block"),),
		subgoals=(),
	)

	assert "+!nop : true <-" in asl
	assert "__hddl_unsat_condition__" not in asl


def test_agentspeak_renderer_renders_structured_plan_library_as_method_plans() -> None:
	domain = HDDLParser.parse_domain(str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"))
	renderer = AgentSpeakRenderer()
	asl = renderer.generate(
		domain=domain,
		objects=("a", "b"),
		method_library=_sample_method_library(),
		plan_library=_sample_plan_library(),
		plan_records=(),
		typed_objects=(("a", "block"), ("b", "block")),
		subgoals=(),
	)

	assert "+!do_put_on(X, Y) : clear(X) & X \\== Y <-" in asl
	assert "\tpick_up(X);" in asl
	assert "\t!stack(X, Y)." in asl


def test_agentspeak_renderer_normalises_structured_object_type_type_atoms() -> None:
	domain = HDDLParser.parse_domain(str(PROJECT_ROOT / "src" / "domains" / "transport" / "domain.hddl"))
	renderer = AgentSpeakRenderer()
	plan_library = PlanLibrary(
		domain_name="transport",
		plans=(
			AgentSpeakPlan(
				plan_name="m_load_capacity_guard",
				trigger=AgentSpeakTrigger(
					event_type="achievement_goal",
					symbol="load",
					arguments=("V:vehicle", "L:location", "P:package"),
				),
				context=("object_type(S, capacity-number)",),
				body=(),
			),
		),
	)

	asl = renderer.generate(
		domain=domain,
		objects=("truck-0", "capacity-0"),
		method_library=_sample_method_library(),
		plan_library=plan_library,
		plan_records=(),
		typed_objects=(("truck-0", "vehicle"), ("capacity-0", "capacity-number")),
		subgoals=(),
	)

	assert "object_type(S, capacity_number)" in asl
	assert 'object_type(S, "capacity-number")' not in asl

def test_evaluation_orchestrator_prefers_original_problem_for_verification() -> None:
	goal_free_problem = PROJECT_ROOT / "tests" / "generated" / "goal_free_p01.hddl"
	goal_free_problem.write_text(
		(
			PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"
		).read_text().replace(
			"\t(:goal (and\n(on b1 b4)\n(on b3 b1)\n\t))\n",
			"\t(:goal (and))\n",
		),
	)
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(goal_free_problem),
	)
	problem_file, mode = orchestrator._determine_verification_problem()

	assert mode == "original_problem"
	assert problem_file == str(goal_free_problem.resolve())


def test_evaluation_orchestrator_uses_original_problem_when_problem_has_goal_facts() -> None:
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p02.hddl"),
	)
	problem_file, mode = orchestrator._determine_verification_problem()

	assert mode == "original_problem"
	assert Path(problem_file).resolve() == (
		PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p02.hddl"
	).resolve()


def test_evaluation_domain_source_defaults_to_benchmark_domain() -> None:
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)

	context = orchestrator._resolve_evaluation_domain_context()

	assert context.source == "benchmark"
	assert context.domain_file == str(
		(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl").resolve()
	)
	assert context.domain.name == orchestrator.domain.name


def test_evaluation_orchestrator_defaults_runtime_logs_to_tmp_root() -> None:
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)

	assert orchestrator.evaluation_tmp_root == (PROJECT_ROOT / "tmp" / "evaluation").resolve()
	assert orchestrator.logger.logs_dir == orchestrator.evaluation_tmp_root


def test_benchmark_evaluation_path_uses_benchmark_domain_for_verification() -> None:
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	evaluation_domain = orchestrator._resolve_evaluation_domain_context()

	verification_domain_file, domain_build_seconds = resolve_verification_domain_file(
		method_library=_sample_method_library(),
		evaluation_domain=evaluation_domain,
		output_dir=str(PROJECT_ROOT / "tests" / "generated"),
	)

	assert verification_domain_file == Path(evaluation_domain.domain_file).resolve()
	assert domain_build_seconds == 0.0


def test_evaluation_domain_source_can_switch_to_generated_domain(tmp_path: Path) -> None:
	generated_domain_path = tmp_path / "generated_domain.hddl"
	generated_domain_path.write_text(
		(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl").read_text()
	)
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
		evaluation_domain_source="generated",
	)
	artifact = _artifact_bundle(
		method_library=_sample_method_library(),
		artifact_root=str(tmp_path),
	)

	context = orchestrator._resolve_evaluation_domain_context(artifact)

	assert context.source == "generated"
	assert context.domain_file == str(generated_domain_path.resolve())
	assert context.domain.name == orchestrator.domain.name


def test_evaluation_domain_source_materializes_generated_domain_from_bundle(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / "legacy_artifact"
	artifact_root.mkdir()
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
		evaluation_domain_source="generated",
	)
	artifact = _artifact_bundle(
		method_library=_sample_method_library(),
		artifact_root=str(artifact_root),
	)

	context = orchestrator._resolve_evaluation_domain_context(artifact)

	assert context.source == "generated"
	assert Path(context.domain_file).exists()
	assert Path(artifact_root / "masked_domain.hddl").exists()
	assert Path(artifact_root / "generated_domain.hddl").exists()


def test_generated_evaluation_domain_uses_runtime_tmp_root_and_source_action_names(
	tmp_path: Path,
) -> None:
	artifact_root = tmp_path / "artifact_bundle"
	runtime_output_dir = tmp_path / "runtime_output"
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
		evaluation_domain_source="generated",
	)
	orchestrator.output_dir = runtime_output_dir
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="do_put_on", parameters=("?x", "?y"), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="pick_up", parameters=("?x",), is_primitive=True),
			HTNTask(name="stack", parameters=("?x", "?y"), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m_do_put_on_serial",
				task_name="do_put_on",
				parameters=("?x", "?y"),
				task_args=("?x", "?y"),
				subtasks=(
					HTNMethodStep("s1", "pick_up", ("?x",), "primitive", action_name="pick_up"),
					HTNMethodStep("s2", "stack", ("?x", "?y"), "primitive", action_name="stack"),
				),
				ordering=(("s1", "s2"),),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)
	artifact = _artifact_bundle(
		method_library=method_library,
		artifact_root=str(artifact_root),
	)

	context = orchestrator._resolve_evaluation_domain_context(artifact)
	generated_domain_path = Path(context.domain_file).resolve()
	generated_domain_text = generated_domain_path.read_text(encoding="utf-8")

	assert generated_domain_path == (
		runtime_output_dir / "evaluation_domain_artifact" / "generated_domain.hddl"
	).resolve()
	assert not (artifact_root / "generated_domain.hddl").exists()
	assert "(s1 (pick-up ?x))" in generated_domain_text
	assert "(s1 (pick_up ?x))" not in generated_domain_text


def test_evaluation_orchestrator_uses_fixed_jason_runtime_timeout_budget() -> None:
	assert PlanLibraryEvaluationOrchestrator._jason_runtime_timeout_seconds(subgoal_count=10) == 1800
	assert PlanLibraryEvaluationOrchestrator._jason_runtime_timeout_seconds(subgoal_count=450) == 1800
	assert PlanLibraryEvaluationOrchestrator._jason_runtime_timeout_seconds(subgoal_count=1000) == 1800


def test_jason_runner_execute_entry_perceives_initial_world() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */
clear(b1).

/* Primitive Action Plans */

/* HTN Method Plans */
+!do_put_on(X, Y) : clear(X) <-
	true.
""".strip(),
		method_library=_sample_method_library(),
		seed_facts=("(clear b1)", "(handempty)"),
		runtime_objects=("b1",),
		object_types={"b1": "block"},
		type_parent_map={"block": "object", "object": None},
		query_goals=({"task_name": "do_put_on", "args": ["b1", "b2"]},),
	)

	assert '.print("execute start")' in runtime_program
	assert ".perceive" in runtime_program
	assert "clear(b1)." in runtime_program
	execute_section = runtime_program.split("+!execute : true <-", maxsplit=1)[1].split(
		"\n\n",
		maxsplit=1,
	)[0]
	assert execute_section.index('.print("execute start")') < execute_section.index(".perceive")
	assert execute_section.index(".perceive") < execute_section.index("!runtime_execute_from_1")
	assert "!do_put_on(b1, b2);" in runtime_program


def test_jason_runner_execution_entry_runs_query_goals_directly() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
+!task_a : true <-
	true.

+!task_b : true <-
	true.
""".strip(),
		method_library=_sample_method_library(),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		query_goals=(
			{"task_name": "task_a", "args": []},
			{"task_name": "task_b", "args": []},
		),
	)

	execute_section = runtime_program.split("+!execute : true <-", maxsplit=1)[1].split(
		"\n\n",
		maxsplit=1,
	)[0]

	assert "!runtime_execute_from_1;" in execute_section
	assert "!finish_or_retry_0;" in execute_section
	assert "!task_a;" in runtime_program
	assert "!task_b;" in runtime_program


def test_jason_runner_repairs_final_goal_state_with_whole_query_passes() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
+!task_a(X) : true <-
	true.
""".strip(),
		method_library=_sample_method_library(),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		query_goals=({"task_name": "task_a", "args": ["a"]},),
		goal_facts=("(done a)",),
	)

	execute_section = runtime_program.split("+!execute : true <-", maxsplit=1)[1].split(
		"\n\n",
		maxsplit=1,
	)[0]

	assert "!finish_or_retry_0;" in execute_section
	assert "!runtime_execute_from_1;" not in execute_section
	assert "+!finish_or_retry_0 : done(a) & not runtime_pass_failed <-" in runtime_program
	assert "+!finish_or_retry_0 : true <-" in runtime_program
	finish_fallback = runtime_program.split("+!finish_or_retry_0 : true <-", maxsplit=1)[
		1
	].split("\n\n", maxsplit=1)[0]
	assert '.print("runtime query pass ", 1);' in finish_fallback
	assert "!runtime_execute_from_1;" in finish_fallback
	assert "!finish_or_retry_1." in finish_fallback
	assert '.print("execute failed")' not in finish_fallback
	assert "+!execute_query_pass_1 : true <-" not in runtime_program
	assert "runtime query pass" in runtime_program
	assert "!runtime_query_goal_1;" in runtime_program
	assert "+!runtime_query_goal_1 : done(a) <-" in runtime_program
	assert "+!runtime_query_goal_1 : runtime_query_goal_completed(1) <-" not in runtime_program
	assert "+!runtime_query_goal_1 : runtime_pass_failed <-" in runtime_program
	assert "-!runtime_query_goal_1 : true <-" in runtime_program
	assert "+!runtime_mark_query_goal_1 : not runtime_pass_failed <-" in runtime_program
	assert "runtime_snapshot(runtime_query_checkpoint(1));" in runtime_program
	assert "runtime_set_active_query_goal(1);" in runtime_program
	assert "runtime_clear_active_query_goal(1);" in runtime_program
	assert "runtime_commit(runtime_query_checkpoint(1));" not in runtime_program
	assert "runtime_restore(runtime_query_checkpoint(1));" in runtime_program
	assert ".perceive;" in runtime_program
	assert "!task_a(a);" in runtime_program
	assert "runtime_pass_failed" in runtime_program
	assert "+!finish_or_retry_3 : true <-" in runtime_program


def test_jason_runner_uses_matching_goal_fact_as_query_completion_context() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
+!deliver(P, L) : true <-
	true.
""".strip(),
		method_library=HTNMethodLibrary(
			compound_tasks=[HTNTask(name="deliver", parameters=("?p", "?l"), is_primitive=False)],
			primitive_tasks=[],
			methods=[],
		),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		query_goals=({"task_name": "deliver", "args": ["package-0", "city-loc-1"]},),
		goal_facts=("(at package-0 city-loc-1)",),
	)

	assert '+!runtime_query_goal_1 : at("package-0","city-loc-1") <-' in runtime_program
	assert "+!runtime_query_goal_1 : runtime_query_goal_completed(1) <-" not in runtime_program


def test_jason_runner_derives_query_completion_context_from_lifted_action_effects() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */
+!drop(V, L, P) : true <-
	drop(V, L, P).

/* HTN Method Plans */
+!deliver(P, L) : true <-
	!drop(V, L, P).
""".strip(),
		method_library=HTNMethodLibrary(
			compound_tasks=[HTNTask(name="deliver", parameters=("?p", "?l"), is_primitive=False)],
			primitive_tasks=[HTNTask(name="drop", parameters=("?v", "?l", "?p"), is_primitive=True)],
			methods=[
				HTNMethod(
					method_name="m-deliver",
					task_name="deliver",
					parameters=("?p", "?l", "?v"),
					task_args=("?p", "?l"),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="drop",
							args=("?v", "?l", "?p"),
							kind="primitive",
							action_name="drop",
						),
					),
				),
			],
		),
		action_schemas=[
			{
				"functor": "drop",
				"source_name": "drop",
				"parameters": ["?v", "?l", "?p"],
				"preconditions": [],
				"effects": [
					{"predicate": "in", "args": ["?p", "?v"], "is_positive": False},
					{"predicate": "at", "args": ["?p", "?l"], "is_positive": True},
				],
			},
		],
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		query_goals=({"task_name": "deliver", "args": ["package-0", "city-loc-1"]},),
		goal_facts=(),
	)

	assert '+!runtime_query_goal_1 : at("package-0", "city-loc-1") <-' in runtime_program
	assert "+!runtime_query_goal_1 : runtime_query_goal_completed(1) <-" not in runtime_program


def test_jason_runner_does_not_treat_consumed_preconditions_as_completion_context() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */
+!consume(A, T) : true <-
	consume(A, T).

/* HTN Method Plans */
+!collect(T) : true <-
	!consume(A, T).
""".strip(),
		method_library=HTNMethodLibrary(
			compound_tasks=[HTNTask(name="collect", parameters=("?target",), is_primitive=False)],
			primitive_tasks=[
				HTNTask(name="consume", parameters=("?agent", "?target"), is_primitive=True),
			],
			methods=[
				HTNMethod(
					method_name="m-collect",
					task_name="collect",
					parameters=("?target", "?agent"),
					task_args=("?target",),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="consume",
							args=("?agent", "?target"),
							kind="primitive",
							action_name="consume",
						),
					),
				),
			],
		),
		action_schemas=[
			{
				"functor": "consume",
				"source_name": "consume",
				"parameters": ["?agent", "?target"],
				"preconditions": [
					{"predicate": "available", "args": ["?target"], "is_positive": True},
				],
				"effects": [
					{"predicate": "done_by", "args": ["?agent"], "is_positive": True},
					{"predicate": "available", "args": ["?target"], "is_positive": False},
				],
			},
		],
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		query_goals=({"task_name": "collect", "args": ["sample-0"]},),
		goal_facts=(),
	)

	assert '+!runtime_query_goal_1 : available("sample-0") <-' not in runtime_program
	assert "+!runtime_query_goal_1 : runtime_query_goal_completed(1) <-" in runtime_program


def test_jason_runner_repair_mode_blocks_failed_child_goal_choices(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv("JASON_RUNTIME_FAILURE_REPAIR", "1")
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
+!parent(X) : ready(X) <-
	!child(X).

+!child(X) : option(X) <-
	true.
""".strip(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask(name="parent", parameters=("x",), is_primitive=False),
				HTNTask(name="child", parameters=("x",), is_primitive=False),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m-parent",
					task_name="parent",
					parameters=("?x",),
					task_args=("?x",),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="child",
							args=("?x",),
							kind="compound",
						),
					),
				),
			],
		),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		query_goals=({"task_name": "parent", "args": ["a"]},),
		goal_facts=("(done a)",),
	)

	assert "blocked_runtime_goal" not in runtime_program
	assert "+blocked_runtime_method(METHOD, child, X, BINDING)" in runtime_program
	child_failure_handler = runtime_program.split("-!child(X)", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
	assert ".fail" not in child_failure_handler
	assert "!child(X)" in child_failure_handler
	assert "+runtime_pass_failed" not in child_failure_handler


def test_jason_runner_retries_runtime_failures_without_problem_goal_context() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
+!parent(X) : ready(X) <-
	!child(X).
""".strip(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask(name="parent", parameters=("x",), is_primitive=False),
				HTNTask(name="child", parameters=("x",), is_primitive=False),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m-parent",
					task_name="parent",
					parameters=("?x",),
					task_args=("?x",),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="child",
							args=("?x",),
							kind="compound",
						),
					),
				),
			],
		),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		query_goals=({"task_name": "parent", "args": ["a"]},),
		goal_facts=(),
	)

	parent_handlers = runtime_program.split("-!parent(X)")
	parent_choice_handler = parent_handlers[1].split("\n\n", maxsplit=1)[0]
	parent_failure_handler = parent_handlers[2].split("\n\n", maxsplit=1)[0]
	child_handlers = runtime_program.split("-!child(X)")
	child_choice_handler = child_handlers[1].split("\n\n", maxsplit=1)[0]
	child_active_handler = child_handlers[2].split("\n\n", maxsplit=1)[0]
	child_parent_caller_handler = child_handlers[3].split("\n\n", maxsplit=1)[0]
	child_legacy_parent_caller_handler = child_handlers[4].split("\n\n", maxsplit=1)[0]
	child_query_fallback_handler = child_handlers[5].split("\n\n", maxsplit=1)[0]
	child_fallback_handler = child_handlers[6].split("\n\n", maxsplit=1)[0]
	assert "!finish_or_retry_0" in runtime_program
	assert "+!finish_or_retry_0 : not runtime_pass_failed <-" in runtime_program
	assert "+!runtime_query_goal_1 : runtime_query_goal_completed(1) <-" in runtime_program
	assert "+!execute_query_pass_1 : true <-" not in runtime_program
	assert "runtime query pass" not in runtime_program
	assert "-!runtime_query_goal_1 : true <-" in runtime_program
	assert "blocked_runtime_goal" not in runtime_program
	assert "runtime_push_method_choice(" in runtime_program
	assert "runtime_pop_method_choice(" in runtime_program
	assert "+blocked_runtime_method(METHOD, child, X, BINDING)" in runtime_program
	assert "runtime_latest_method_choice_point(CHOICE, parent, runtime_args(X), SNAPSHOT)" in (
		parent_choice_handler
	)
	assert "!parent(X)" in parent_choice_handler
	assert "+runtime_pass_failed" not in parent_failure_handler
	assert "!parent(X)" in parent_failure_handler
	assert "runtime_restore(runtime_method_snapshot(METHOD, parent, X, BINDING))" in parent_failure_handler
	assert "runtime_latest_method_choice_point(CHOICE, child, runtime_args(X), SNAPSHOT)" in (
		child_choice_handler
	)
	assert "!child(X)" in child_choice_handler
	assert "!child(X)" in child_active_handler
	assert "+runtime_pass_failed" not in child_active_handler
	assert "runtime_current_call(METHOD, parent, runtime_args(PARENT_X), BINDING, SNAPSHOT" in (
		child_parent_caller_handler
	)
	assert "child, runtime_args(X))" in child_parent_caller_handler
	assert "runtime_restore(SNAPSHOT)" in child_parent_caller_handler
	assert "+blocked_runtime_choice(runtime_method_choice(METHOD, parent, runtime_args(PARENT_X), BINDING))" in (
		child_parent_caller_handler
	)
	assert "+blocked_runtime_method(METHOD, parent, PARENT_X, BINDING)" in (
		child_parent_caller_handler
	)
	assert "!parent(PARENT_X)" in child_parent_caller_handler
	assert "runtime_commit(SNAPSHOT)" in child_parent_caller_handler
	assert ".succeed_goal(parent(PARENT_X))." in child_parent_caller_handler
	assert "runtime_current_method(METHOD, child, X, BINDING)" not in child_parent_caller_handler
	assert "runtime_current_method(METHOD, parent, PARENT_X, BINDING)" in (
		child_legacy_parent_caller_handler
	)
	assert "runtime_active_query_goal(1)" in child_query_fallback_handler
	assert ".fail_goal(runtime_execute_from_1)." in child_query_fallback_handler
	assert "if (not runtime_reported_failure(fail_goal(child, X)))" in child_fallback_handler
	assert runtime_program.count("-!child(X)") == 6
	assert ".fail." in child_fallback_handler


def test_jason_runner_records_recursive_caller_frames_with_separate_arguments() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
+!get_to(V, TARGET) : edge(VIA, TARGET) <-
	!get_to(V, VIA);
	drive(V, VIA, TARGET).
""".strip(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask(name="get_to", parameters=("v", "target"), is_primitive=False),
			],
			primitive_tasks=[
				HTNTask(name="drive", parameters=("v", "from", "to"), is_primitive=True),
			],
			methods=[
				HTNMethod(
					method_name="m-get-to-via",
					task_name="get_to",
					parameters=("?v", "?via", "?target"),
					task_args=("?v", "?target"),
					subtasks=(
						HTNMethodStep(
							step_id="s1",
							task_name="get_to",
							args=("?v", "?via"),
							kind="compound",
						),
						HTNMethodStep(
							step_id="s2",
							task_name="drive",
							action_name="drive",
							args=("?v", "?via", "?target"),
							kind="primitive",
						),
					),
				),
			],
		),
		action_schemas=(
			{
				"source_name": "drive",
				"functor": "drive",
				"parameters": ["?v", "?from", "?to"],
			},
		),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		query_goals=({"task_name": "get_to", "args": ["truck", "loc3"]},),
		goal_facts=(),
	)

	assert (
		'runtime_current_call("m-get-to-via", get_to, runtime_args(V, TARGET), '
		'runtime_binding(VIA), runtime_method_snapshot("m-get-to-via", get_to, V, TARGET, '
		"runtime_binding(VIA)), get_to, runtime_args(V, VIA))"
	) in runtime_program
	assert (
		"-!get_to(V, TARGET) : runtime_current_call(METHOD, get_to, "
		"runtime_args(PARENT_V, PARENT_TARGET), BINDING, SNAPSHOT, get_to, "
		"runtime_args(V, TARGET))"
	) in runtime_program
	assert "!get_to(PARENT_V, PARENT_TARGET)" in runtime_program
	assert "runtime_method_choice(METHOD, get_to, runtime_args(PARENT_V, PARENT_TARGET), BINDING)" in (
		runtime_program
	)


def test_jason_runner_adds_query_level_chronological_backtracking() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
+!first(X) : ready(X) <-
	true.

+!second(X) : ready(X) <-
	true.
""".strip(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask(name="first", parameters=("x",), is_primitive=False),
				HTNTask(name="second", parameters=("x",), is_primitive=False),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m-first",
					task_name="first",
					parameters=("?x",),
					task_args=("?x",),
				),
				HTNMethod(
					method_name="m-second",
					task_name="second",
					parameters=("?x",),
					task_args=("?x",),
				),
			],
		),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		query_goals=(
			{"task_name": "first", "args": ["a"]},
			{"task_name": "second", "args": ["b"]},
		),
		goal_facts=(),
	)

	assert "+!runtime_execute_from_1 : true <-" in runtime_program
	assert "+!runtime_execute_from_2 : true <-" in runtime_program
	assert "-!runtime_execute_from_1 : pipeline.choose_runtime_choice(1, 1, CHOICE)" in (
		runtime_program
	)
	assert "-!runtime_execute_from_1 : runtime_last_query_choice(1, CHOICE)" in runtime_program
	assert "-!runtime_execute_from_1 : runtime_query_choice(1, CHOICE)" in runtime_program
	assert "-!runtime_execute_from_2 : pipeline.choose_runtime_choice(2, 2, CHOICE)" in (
		runtime_program
	)
	assert "-!runtime_execute_from_2 : runtime_last_query_choice(2, CHOICE)" in runtime_program
	assert "-!runtime_execute_from_2 : runtime_query_choice(2, CHOICE)" in runtime_program
	assert "-!runtime_execute_from_2 : pipeline.choose_runtime_choice(2, 1, CHOICE)" in (
		runtime_program
	)
	assert "-!runtime_execute_from_2 : runtime_last_query_choice(1, CHOICE)" in runtime_program
	assert "-!runtime_execute_from_2 : runtime_query_choice(1, CHOICE)" in runtime_program
	assert "+!runtime_backtrack_from_2 : pipeline.choose_runtime_choice(2, 2, CHOICE)" in (
		runtime_program
	)
	assert "+!runtime_backtrack_from_2 : pipeline.choose_runtime_choice(2, 1, CHOICE)" in (
		runtime_program
	)
	assert "+!runtime_backtrack_from_2 : runtime_last_query_choice(2, CHOICE)" in runtime_program
	assert "+!runtime_backtrack_from_2 : runtime_query_choice(1, CHOICE)" in runtime_program
	assert "+!runtime_backtrack_from_2 : true <-" in runtime_program
	assert "runtime_restore(runtime_query_checkpoint(1));" in runtime_program
	assert "+runtime_backtracked_choice(1, CHOICE);" in runtime_program
	assert "+blocked_runtime_choice(CHOICE);" in runtime_program
	assert "!runtime_clear_local_repair_state;" in runtime_program
	assert ".abolish(blocked_runtime_method(_, _, _))" in runtime_program
	assert "!runtime_clear_query_progress_from_1;" in runtime_program
	assert "!runtime_execute_from_1;" in runtime_program
	assert "!finish_or_retry_0;" in runtime_program
	assert ".stopMAS." in runtime_program


def test_jason_runner_records_domain_agnostic_runtime_method_choices() -> None:
	runner = JasonRunner()
	runtime_program = runner._build_runner_asl(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
+!move_abs(R, T) : at(R, F) <-
	true.
""".strip(),
		method_library=HTNMethodLibrary(
			compound_tasks=[
				HTNTask(name="move_abs", parameters=("r", "t"), is_primitive=False),
			],
			primitive_tasks=[],
			methods=[
				HTNMethod(
					method_name="m-move",
					task_name="move_abs",
					parameters=("?r", "?f", "?t"),
					task_args=("?r", "?t"),
				),
			],
		),
		seed_facts=(),
		runtime_objects=(),
		object_types={},
		type_parent_map={},
		query_goals=({"task_name": "move_abs", "args": ["r1", "l2"]},),
		goal_facts=(),
	)

	choice = 'runtime_method_choice("m-move", move_abs, runtime_args(R, T), runtime_binding(F))'
	assert f"not blocked_runtime_choice({choice})" in runtime_program
	assert f"runtime_record_query_choice({choice});" in runtime_program
	assert "+!runtime_record_query_choice" not in runtime_program
	assert "runtime_query_choice_frame(1, _, _)" in runtime_program
	assert "runtime_last_query_choice_frame(1, _, _)" in runtime_program
	assert "runtime_last_query_choice(1, CHOICE)" in runtime_program


def test_jason_runner_extracts_only_committed_snapshot_actions() -> None:
	runner = JasonRunner()
	stdout = "\n".join(
		[
			"runtime env action success move(a,b)",
			"runtime env snapshot 1",
			"runtime env action success move(b,c)",
			"runtime env restore 1",
			"runtime env snapshot 1",
			"runtime env action success move(b,d)",
			"runtime env commit 1",
			"runtime env action success move(d,e)",
		],
	)

	assert runner._extract_action_path(stdout) == [
		"move(a,b)",
		"move(b,d)",
		"move(d,e)",
	]


def test_jason_runner_bounds_repetitive_runtime_artifacts() -> None:
	runner = JasonRunner()
	runner.runtime_output_artifact_limit_chars = 12
	runner.method_trace_record_limit = 2

	bounded_output, output_truncated = runner._bounded_runtime_output_artifact("0123456789abcdef")
	bounded_trace, original_trace_count, trace_truncated = runner._cap_method_trace_records(
		[
			{"method_name": "m1", "task_args": []},
			{"method_name": "m2", "task_args": []},
			{"method_name": "m3", "task_args": []},
		],
	)

	assert output_truncated is True
	assert "original_chars=16" in bounded_output
	assert bounded_output.endswith("456789abcdef")
	assert original_trace_count == 3
	assert trace_truncated is True
	assert [item["method_name"] for item in bounded_trace] == ["m1", "m2"]


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

/* Primitive Action Plans */

/* HTN Method Plans */
+!navigate_abs(rover1, waypoint5) : true <-
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
	monkeypatch.setattr("evaluation.jason_runtime.runner.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

	result = runner.validate(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
+!navigate_abs(rover1, waypoint5) : true <-
	.print("runtime trace method flat ", "m-navigate_abs-1");
	true.

+!navigate_abs(rover1, waypoint5) : true <-
	.print("runtime trace method flat ", "m-navigate_abs-3");
	true.
""".strip(),
		method_library=_sample_method_library(),
			action_schemas=[{"name": "idle_action"}],
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
	projection_path = tmp_path / "runtime_grounding_projection.asl"
	assert projection_path.exists()
	projection_text = projection_path.read_text()
	assert "/* HTN Method Plans */" in projection_text
	assert "+!navigate_abs(rover1, waypoint5)" in projection_text
	assert "/* Failure Handlers */" not in projection_text
	assert result.artifacts["source_plan_library_kind"] == "S"
	assert result.artifacts["runtime_projection_kind"] == "S_{I,g}"
	assert result.artifacts["runtime_grounding_projection"] == str(projection_path)


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
		"evaluation.jason_runtime.runner.subprocess.run",
		lambda *args, **kwargs: FakeCompletedProcess(),
	)

	result = runner.validate(
		agentspeak_code="""
/* Initial Beliefs */

/* Primitive Action Plans */

/* HTN Method Plans */
	+!idle_goal : true <-
		true.
""".strip(),
			method_library=_sample_method_library(),
			action_schemas=[{"name": "idle_action"}],
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


def test_method_trace_reconstruction_accepts_hyphenated_runtime_action_names(
	tmp_path: Path,
) -> None:
	problem_file = tmp_path / "blocksworld_simple_put_on.hddl"
	problem_file.write_text(
		"""(define (problem BW-simple-put-on)
(:domain BLOCKS)
(:objects b2 b4 - block)
(:htn :parameters () :ordered-subtasks (and
(task1 (do_put_on b4 b2))
))
(:init
(handempty)
(ontable b2)
(ontable b4)
(clear b2)
(clear b4)
)
\t(:goal (and
(on b4 b2)
\t))
)
""",
		encoding="utf-8",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="do_put_on", parameters=("?x", "?y"), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="pick_up", parameters=("?x",), is_primitive=True),
			HTNTask(name="stack", parameters=("?x", "?y"), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m_do_put_on_serial",
				task_name="do_put_on",
				parameters=("?x", "?y"),
				task_args=("?x", "?y"),
				subtasks=(
					HTNMethodStep("s1", "pick_up", ("?x",), "primitive", action_name="pick_up"),
					HTNMethodStep("s2", "stack", ("?x", "?y"), "primitive", action_name="stack"),
				),
				ordering=(("s1", "s2"),),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)
	runner = JasonRunner()

	reconstruction = runner._check_method_trace_reconstruction(
		action_path=("pick-up(b4)", "stack(b4,b2)"),
		method_trace=({"method_name": "m_do_put_on_serial", "task_args": ("b4", "b2")},),
		method_library=method_library,
		problem_file=problem_file,
	)
	rendered_plan = IPCPlanVerifier()._render_supported_hierarchical_plan(
		domain_file=problem_file,
		problem_file=problem_file,
		action_path=("pick-up(b4)", "stack(b4,b2)"),
		method_library=method_library,
		method_trace=({"method_name": "m_do_put_on_serial", "task_args": ("b4", "b2")},),
	)

	assert reconstruction == {
		"passed": True,
		"failure_class": None,
		"message": None,
	}
	assert rendered_plan is not None
	assert "pick-up b4" in rendered_plan
	assert "stack b4 b2" in rendered_plan


def test_hierarchical_plan_exporter_rejects_runtime_repair_suffix(tmp_path: Path) -> None:
	problem_file = tmp_path / "blocksworld_repeated_runtime_roots.hddl"
	problem_file.write_text(
		"""(define (problem BW-repair-root)
(:domain BLOCKS)
(:objects b2 b4 - block)
(:htn :parameters () :ordered-subtasks (and
(task1 (do_put_on b4 b2))
))
(:init
(handempty)
(ontable b2)
(ontable b4)
(clear b2)
(clear b4)
)
\t(:goal (and
(on b4 b2)
\t))
)
""",
		encoding="utf-8",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="do_put_on", parameters=("?x", "?y"), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="pick_up", parameters=("?x",), is_primitive=True),
			HTNTask(name="stack", parameters=("?x", "?y"), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m_do_put_on_serial",
				task_name="do_put_on",
				parameters=("?x", "?y"),
				task_args=("?x", "?y"),
				subtasks=(
					HTNMethodStep("s1", "pick_up", ("?x",), "primitive", action_name="pick_up"),
					HTNMethodStep("s2", "stack", ("?x", "?y"), "primitive", action_name="stack"),
				),
				ordering=(("s1", "s2"),),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	rendered_plan = IPCPlanVerifier()._render_supported_hierarchical_plan(
		domain_file=problem_file,
		problem_file=problem_file,
		action_path=(
			"pick-up(b4)",
			"stack(b4,b2)",
			"pick-up(b4)",
			"stack(b4,b2)",
		),
		method_library=method_library,
		method_trace=(
			{"method_name": "m_do_put_on_serial", "task_args": ("b4", "b2")},
			{"method_name": "m_do_put_on_serial", "task_args": ("b4", "b2")},
		),
	)

	assert rendered_plan is None


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
	assert "deep nested eventuality chains" in system_prompt
	assert "F(A & F(B & F(C)))" in system_prompt
	assert "A & X(B)" in system_prompt
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
	assert "Grounded problem objects relevant to this query:" in attempts[0]["system"]
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


def test_goal_grounding_openai_compatible_request_profile_uses_configured_json_budget() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(
		domain_file=domain_file,
		model="deepseek-v4-pro",
		base_url="https://api.deepseek.com",
		response_max_tokens=12000,
	)
	messages = [
		{"role": "system", "content": "Return strict minified JSON only."},
		{"role": "user", "content": "Ground this query into one LTLf formula."},
	]

	profile = generator._goal_grounding_request_profile(messages=messages)

	assert profile["name"] == "openai_compatible_json_chat"
	assert profile["stream_response"] is False
	assert profile["first_chunk_timeout_seconds"] == 0.0
	assert profile["completion_max_tokens"] == 12000
	assert profile["max_tokens_policy"] == "configured_ltlf_generation_max_tokens"
	assert profile["thinking_type"] == "enabled"
	assert profile["reasoning_effort"] == "high"


def test_goal_grounding_chat_completion_uses_openai_compatible_json_request() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	captured_kwargs: dict[str, object] = {}

	class FakeCompletions:
		def create(self, **kwargs):
			captured_kwargs.update(kwargs)
			return {"ok": True}

	class FakeChat:
		def __init__(self):
			self.completions = FakeCompletions()

	class FakeClient:
		def __init__(self):
			self.chat = FakeChat()

	generator = NLToLTLfGenerator(
		domain_file=domain_file,
		model="deepseek-v4-pro",
		base_url="https://api.deepseek.com",
		api_key="sk-test",
	)
	generator.client = FakeClient()

	messages = [{"role": "system", "content": "Return JSON."}]
	response = generator._create_chat_completion(
		messages,
		response_max_tokens=321,
		request_timeout=123.0,
	)

	assert response == {"ok": True}
	assert captured_kwargs["timeout"] == 123.0
	assert captured_kwargs["stream"] is False
	assert captured_kwargs["max_tokens"] == 321
	assert captured_kwargs["response_format"] == {"type": "json_object"}
	assert captured_kwargs["reasoning_effort"] == "high"
	assert captured_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_goal_grounding_chat_completion_keeps_json_object_request_on_openai_compatible_path() -> None:
	domain_file = str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl")
	generator = NLToLTLfGenerator(
		domain_file=domain_file,
		model="deepseek-v4-pro",
		base_url="https://api.deepseek.com",
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
	assert fake_completions.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}


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
	assert transport_metadata["llm_first_stream_chunk_seconds"] >= 0.0
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


def test_goal_grounding_streaming_ignores_reasoning_payload_without_storing_it() -> None:
	class FakeDelta:
		def __init__(self, reasoning=None):
			self.content = None
			self.reasoning = reasoning

	class FakeChoice:
		def __init__(self, reasoning=None):
			self.delta = FakeDelta(reasoning=reasoning)
			self.finish_reason = None
			self.reasoning = reasoning

	class FakeChunk:
		def __init__(self, chunk_id, reasoning=None):
			self.id = chunk_id
			self.choices = [FakeChoice(reasoning=reasoning)]

	class ReasoningThenBlockingStream:
		def __init__(self):
			self.index = 0

		def __iter__(self):
			return self

		def __next__(self):
			self.index += 1
			if self.index == 1:
				return FakeChunk("req_goal_reasoning", reasoning="thinking")
			time.sleep(0.05)
			raise AssertionError("stream iteration should have timed out by total deadline")

		def close(self):
			return None

	generator = NLToLTLfGenerator()

	with pytest.raises(TimeoutError, match="configured timeout") as exc_info:
		generator._consume_streaming_llm_response(
			ReasoningThenBlockingStream(),
			transport_metadata={"llm_first_chunk_timeout_seconds": 0.01},
			total_timeout_seconds=0.02,
		)

	transport_metadata = getattr(exc_info.value, "transport_metadata", {})
	assert transport_metadata["llm_request_id"] == "req_goal_reasoning"
	assert transport_metadata["llm_first_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_first_stream_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_first_reasoning_chunk_seconds"] >= 0.0
	assert transport_metadata["llm_reasoning_chunks_ignored"] == 2
	assert "llm_reasoning_preview" not in transport_metadata
	assert "llm_reasoning_characters" not in transport_metadata
	assert "llm_first_content_chunk_seconds" not in transport_metadata


def test_missing_goal_inference_normalises_hddl_and_runtime_fact_formats() -> None:
	missing = infer_missing_goal_facts(
		problem_file=PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl",
		world_facts=("on(b1,b4)", "on(b3,b1)"),
	)

	assert missing == ()


def test_execute_query_with_jason_returns_none_when_runtime_fails(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	orchestrator.output_dir = tmp_path
	evaluation_domain = orchestrator._resolve_evaluation_domain_context()
	grounding_result = TemporalGroundingResult(
		query_text="stack b on a",
		ltlf_formula="F(subgoal_1)",
		subgoals=(GroundedSubgoal("subgoal_1", "do_put_on", ("b", "a")),),
		typed_objects={"a": "block", "b": "block"},
		query_object_inventory=(),
		diagnostics=(),
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

	monkeypatch.setattr(evaluation_orchestrator_module, "JasonRunner", lambda **_kwargs: FakeRunner())
	monkeypatch.setattr(
		evaluation_orchestrator_module,
		"planner_action_schemas_for_domain",
		lambda _domain: [{"action_name": "turn_to"}],
	)
	monkeypatch.setattr(
		evaluation_orchestrator_module,
		"render_supported_hierarchical_plan",
		lambda **_kwargs: "guided plan",
	)

	result = orchestrator._execute_query_with_jason(
		grounding_result=grounding_result,
		method_library=_sample_method_library(),
		plan_library=_sample_plan_library(),
		agentspeak_code="!execute.",
		agentspeak_artifacts={},
		verification_problem_file=str(
			PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"
		),
		verification_mode="original_problem",
		evaluation_domain=evaluation_domain,
	)

	assert result is None
	assert len(validation_calls) == 1
	assert validation_calls[0]["plan_library"] == _sample_plan_library()
	assert validation_calls[0]["query_goals"] == (grounding_result.subgoals[0].to_dict(),)
	assert validation_calls[0]["goal_facts"]


def test_execute_query_with_jason_uses_reconstructed_hierarchical_plan_text(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	orchestrator.output_dir = tmp_path
	evaluation_domain = orchestrator._resolve_evaluation_domain_context()
	grounding_result = TemporalGroundingResult(
		query_text="stack b on a",
		ltlf_formula="F(subgoal_1)",
		subgoals=(GroundedSubgoal("subgoal_1", "do_put_on", ("b", "a")),),
		typed_objects={"a": "block", "b": "block"},
		query_object_inventory=(),
		diagnostics=(),
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

	monkeypatch.setattr(evaluation_orchestrator_module, "JasonRunner", lambda **_kwargs: FakeRunner())
	monkeypatch.setattr(
		evaluation_orchestrator_module,
		"planner_action_schemas_for_domain",
		lambda _domain: [{"action_name": "turn_to"}],
	)
	monkeypatch.setattr(
		evaluation_orchestrator_module,
		"render_supported_hierarchical_plan",
		lambda **_kwargs: "reconstructed plan",
	)

	result = orchestrator._execute_query_with_jason(
		grounding_result=grounding_result,
		method_library=_sample_method_library(),
		plan_library=_sample_plan_library(),
		agentspeak_code="!execute.",
		agentspeak_artifacts={},
		verification_problem_file=str(
			PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"
		),
		verification_mode="original_problem",
		evaluation_domain=evaluation_domain,
	)

	assert result is not None
	assert result.hierarchical_plan_text == "reconstructed plan"


def test_execute_query_with_jason_uses_primitive_verification_after_goal_repair(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	orchestrator.output_dir = tmp_path
	evaluation_domain = orchestrator._resolve_evaluation_domain_context()
	grounding_result = TemporalGroundingResult(
		query_text="stack b on a",
		ltlf_formula="F(subgoal_1)",
		subgoals=(GroundedSubgoal("subgoal_1", "do_put_on", ("b", "a")),),
		typed_objects={"a": "block", "b": "block"},
		query_object_inventory=(),
		diagnostics=(),
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
				artifacts={"goal_repair_pass_count": 2},
				timing_profile={},
			)

	monkeypatch.setattr(evaluation_orchestrator_module, "JasonRunner", lambda **_kwargs: FakeRunner())
	monkeypatch.setattr(
		evaluation_orchestrator_module,
		"planner_action_schemas_for_domain",
		lambda _domain: [{"action_name": "turn_to"}],
	)
	monkeypatch.setattr(
		evaluation_orchestrator_module,
		"render_supported_hierarchical_plan",
		lambda **_kwargs: (_ for _ in ()).throw(AssertionError("hierarchical export skipped")),
	)

	result = orchestrator._execute_query_with_jason(
		grounding_result=grounding_result,
		method_library=_sample_method_library(),
		plan_library=_sample_plan_library(),
		agentspeak_code="!execute.",
		agentspeak_artifacts={},
		verification_problem_file=str(
			PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"
		),
		verification_mode="original_problem",
		evaluation_domain=evaluation_domain,
	)

	assert result is not None
	assert result.hierarchical_plan_text is None


def test_verify_plan_officially_restores_terminal_newline_for_hierarchical_plan_text(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
		evaluation_domain_source="benchmark",
	)
	orchestrator.output_dir = tmp_path
	evaluation_domain = orchestrator._resolve_evaluation_domain_context(source="benchmark")

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
		evaluation_official_verification_module,
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
		evaluation_domain=evaluation_domain,
	)

	assert plan_verification is not None
	assert plan_verification["summary"]["status"] == "success"


def test_verify_plan_officially_accepts_runtime_repair_primitive_goal_reach(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
		evaluation_domain_source="benchmark",
	)
	orchestrator.output_dir = tmp_path
	evaluation_domain = orchestrator._resolve_evaluation_domain_context(source="benchmark")

	class FakeVerifier:
		def tool_available(self) -> bool:
			return True

		def verify_primitive_plan(self, **_kwargs):
			return IPCPrimitivePlanVerificationResult(
				tool_available=True,
				command=["fake"],
				plan_file=str(tmp_path / "plan.txt"),
				output_file=str(tmp_path / "verifier.txt"),
				stdout="Primitive plan alone executable: true\nPlan verification result: false",
				stderr="",
				primitive_plan_only=True,
				primitive_plan_executable=True,
				verification_result=False,
				reached_goal_state=True,
				plan_kind="primitive_only",
				build_warning=None,
				error="verifier exited with code 1",
			)

	monkeypatch.setattr(
		evaluation_official_verification_module,
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
				"action_path": ["pick-up(b1)", "stack(b1,b4)"],
				"method_trace": [],
				"consistency_checks": {
					"action_path_schema_replay": {
						"world_facts": ["on(b1,b4)", "on(b3,b1)"],
					},
				},
			},
		},
		evaluation_domain=evaluation_domain,
	)

	assert plan_verification is not None
	assert plan_verification["summary"]["status"] == "success"
	assert plan_verification["summary"]["plan_kind"] == "primitive_only"
	assert plan_verification["summary"]["runtime_goal_reached"] is True


def test_verify_plan_officially_accepts_primitive_goal_reach_after_hierarchical_rejection(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	orchestrator = PlanLibraryEvaluationOrchestrator(
		domain_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
		problem_file=str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
		evaluation_domain_source="benchmark",
	)
	orchestrator.output_dir = tmp_path
	evaluation_domain = orchestrator._resolve_evaluation_domain_context(source="benchmark")

	class FakeVerifier:
		def tool_available(self) -> bool:
			return True

		def verify_plan_text(self, **kwargs):
			assert str(kwargs["plan_text"]).endswith("\n")
			return IPCPrimitivePlanVerificationResult(
				tool_available=True,
				command=["fake"],
				plan_file=str(tmp_path / "hierarchical_plan.txt"),
				output_file=str(tmp_path / "hierarchical_verifier.txt"),
				stdout="Plan verification result: false",
				stderr="",
				primitive_plan_only=False,
				primitive_plan_executable=None,
				verification_result=False,
				reached_goal_state=False,
				plan_kind="hierarchical",
				build_warning=None,
				error="verifier exited with code 1",
			)

		def verify_primitive_plan(self, **kwargs):
			assert kwargs["plan_filename"] == "ipc_official_primitive_plan.txt"
			return IPCPrimitivePlanVerificationResult(
				tool_available=True,
				command=["fake"],
				plan_file=str(tmp_path / "primitive_plan.txt"),
				output_file=str(tmp_path / "primitive_verifier.txt"),
				stdout="Primitive plan alone executable: true\nPlan verification result: false",
				stderr="",
				primitive_plan_only=True,
				primitive_plan_executable=True,
				verification_result=False,
				reached_goal_state=True,
				plan_kind="primitive_only",
				build_warning=None,
				error="verifier exited with code 1",
			)

		def verify_plan(self, **_kwargs):
			raise AssertionError("verify_plan should not be called")

	monkeypatch.setattr(
		evaluation_official_verification_module,
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
				"action_path": ["pick-up(b1)", "stack(b1,b4)"],
				"method_trace": [],
				"consistency_checks": {
					"action_path_schema_replay": {
						"world_facts": ["on(b1,b4)", "on(b3,b1)"],
					},
				},
			},
		},
		evaluation_domain=evaluation_domain,
	)

	assert plan_verification is not None
	assert plan_verification["summary"]["status"] == "success"
	assert plan_verification["summary"]["plan_kind"] == "primitive_only"
	assert plan_verification["summary"]["primitive_runtime_fallback_used"] is True
	assert plan_verification["summary"]["hierarchical_verification_result"] is False
	assert plan_verification["artifacts"]["hierarchical_verification"]["plan_kind"] == "hierarchical"


def test_official_plan_verifier_result_dict_omits_full_process_output() -> None:
	result = IPCPrimitivePlanVerificationResult(
		tool_available=True,
		command=["pandaPIparser", "-v"],
		plan_file="plan.txt",
		output_file="verifier.txt",
		stdout="x" * 10_000,
		stderr="",
		primitive_plan_only=True,
		primitive_plan_executable=True,
		verification_result=False,
		reached_goal_state=True,
		plan_kind="primitive_only",
		build_warning=None,
		error=None,
	)

	payload = result.to_dict()

	assert "stdout" not in payload
	assert payload["stdout_chars"] == 10_000
	assert "full text in output_file" in str(payload["stdout_preview"])
